"""
Diagnostic: compare FDFD epsilon grid vs MPB epsilon grid for monolayer honeycomb.

This script:
1. Runs MPB to get epsilon statistics and eigenvalues for honeycomb monolayer
2. Builds the equivalent FDFD epsilon grid
3. Compares statistics (mean, harmonic mean, fill fraction) and eigenvalues

This is the cheapest, most definitive test for geometry convention errors.
"""

import numpy as np
import subprocess
import os
import re
import tempfile

# ────────────────────────────────────────────
# Parameters (honeycomb TM: Si rods in air)
# ────────────────────────────────────────────
EPS_BG = 1.0       # background: air
EPS_HOLE = 11.56   # rods: silicon
R_OVER_A = 0.2     # radius / lattice constant
RESOLUTION = 64    # grid resolution per lattice constant


def write_mpb_ctl(ctl_path: str, resolution: int = RESOLUTION):
    """Write MPB control file for monolayer honeycomb."""
    ctl = f"""; Honeycomb monolayer: Si rods in air, TM polarization
; Lattice: triangular with two-atom basis

(set! geometry-lattice
  (make lattice
    (size 1 1 no-size)
    (basis1 1 0)
    (basis2 0.5 (/ (sqrt 3) 2))))

(set! default-material (make dielectric (epsilon {EPS_BG})))

(set! geometry
  (list
    (make cylinder
      (center 0 0 0)
      (radius {R_OVER_A})
      (height infinity)
      (material (make dielectric (epsilon {EPS_HOLE}))))
    (make cylinder
      (center (/ 1 3) (/ 1 3) 0)
      (radius {R_OVER_A})
      (height infinity)
      (material (make dielectric (epsilon {EPS_HOLE}))))))

(set! resolution {resolution})
(set! num-bands 6)

; K-point: solve AND output epsilon
(set! k-points (list (vector3 (/ 1 3) (/ 1 3) 0)))

; Output epsilon
(run-tm output-epsilon)
"""
    with open(ctl_path, 'w') as f:
        f.write(ctl)
    return ctl_path


def run_mpb_and_extract_eps(work_dir: str) -> tuple:
    """Run MPB to get epsilon statistics and eigenvalues.
    
    Returns (mpb_stats_dict, freqs_list, stdout_str).
    mpb_stats has keys: min, max, mean, harm_mean, pct_above_1, fill_pct
    """
    ctl_path = os.path.join(work_dir, 'honeycomb.ctl')
    write_mpb_ctl(ctl_path)

    result = subprocess.run(
        ['mpb', 'honeycomb.ctl'],
        cwd=work_dir,
        capture_output=True, text=True, timeout=120,
    )
    print("MPB stdout (last 20 lines):")
    for line in result.stdout.strip().split('\n')[-20:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"MPB stderr: {result.stderr}")
        raise RuntimeError("MPB failed")

    # Parse epsilon statistics from output
    # Format: "epsilon: 1-11.56, mean 4.06459, harm. mean 1.38908, 32.0068% > 1, 29.0208% "fill""
    mpb_stats = {}
    for line in result.stdout.split('\n'):
        m = re.search(
            r'epsilon:\s+([\d.]+)-([\d.]+),\s+mean\s+([\d.]+),\s+harm\.\s+mean\s+([\d.]+),\s+([\d.]+)%\s+>\s+1,\s+([\d.]+)%\s+"fill"',
            line
        )
        if m:
            mpb_stats = {
                'min': float(m.group(1)),
                'max': float(m.group(2)),
                'mean': float(m.group(3)),
                'harm_mean': float(m.group(4)),
                'pct_above_1': float(m.group(5)),
                'fill_pct': float(m.group(6)),
            }

    # Parse eigenvalues from output (skip header line, take data line)
    freqs = []
    for line in result.stdout.split('\n'):
        if 'tmfreqs:' in line and 'band' not in line:
            parts = line.split(',')
            freqs = [float(x.strip()) for x in parts[6:] if x.strip()]
            break

    return mpb_stats, freqs, result.stdout


def build_fdfd_eps_monolayer(resolution: int = RESOLUTION) -> tuple:
    """Build FDFD epsilon grid for a single honeycomb unit cell.

    This bypasses the supercell code and builds the grid directly,
    matching MPB's convention:
    - Lattice vectors: a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    - Sublattice: (0, 0) and (1/3, 1/3) in fractional coords
    - Grid: fractional coords s1, s2 in [0, 1)
    """
    a = 1.0
    r = R_OVER_A * a

    # Lattice vectors (columns of B)
    B = a * np.array([[1.0, 0.5],
                      [0.0, np.sqrt(3) / 2]])
    B_inv = np.linalg.inv(B)

    # Sublattice positions in fractional coords
    sublattice = np.array([[0.0, 0.0],
                           [1.0 / 3, 1.0 / 3]])

    Nx = Ny = resolution
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    # Cartesian coordinates
    X = S1 * B[0, 0] + S2 * B[0, 1]
    Y = S1 * B[1, 0] + S2 * B[1, 1]

    eps_grid = np.full((Nx, Ny), EPS_BG, dtype=np.float64)

    XY = np.stack([X, Y], axis=0)

    for sub_frac in sublattice:
        offset = B @ sub_frac  # Cartesian offset
        shifted = XY - offset[:, None, None]
        # To fractional
        frac = np.einsum('ij,jkl->ikl', B_inv, shifted)
        f_near = frac - np.round(frac)
        # Back to Cartesian
        disp = np.einsum('ij,jkl->ikl', B, f_near)
        dist_sq = disp[0]**2 + disp[1]**2
        eps_grid[dist_sq < r**2] = EPS_HOLE

    info = {
        'B_super': B,
        'L1': B[:, 0],
        'L2': B[:, 1],
    }
    return eps_grid, info


def compare_epsilon_stats(eps_fdfd, mpb_stats):
    """Compare FDFD epsilon statistics against MPB epsilon statistics."""
    print("\n" + "=" * 60)
    print("EPSILON STATISTICS COMPARISON")
    print("=" * 60)

    # FDFD statistics
    fdfd_mean = eps_fdfd.mean()
    fdfd_harm_mean = 1.0 / (1.0 / eps_fdfd).mean()
    fdfd_pct_above_1 = (eps_fdfd > 1.0 + 1e-6).mean() * 100
    # Analytic filling fraction for honeycomb (2 rods per cell)
    cell_area = np.sqrt(3) / 2  # for a=1 hex cell
    rod_area = 2 * np.pi * R_OVER_A**2
    fill_analytic = rod_area / cell_area * 100

    print(f"\n{'Statistic':>20}  {'FDFD':>12}  {'MPB':>12}  {'Analytic':>12}")
    print(f"{'─' * 20}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    print(f"{'min':>20}  {eps_fdfd.min():>12.4f}  {mpb_stats.get('min', 0):>12.4f}")
    print(f"{'max':>20}  {eps_fdfd.max():>12.4f}  {mpb_stats.get('max', 0):>12.4f}")
    print(f"{'mean':>20}  {fdfd_mean:>12.4f}  {mpb_stats.get('mean', 0):>12.4f}")
    print(f"{'harmonic mean':>20}  {fdfd_harm_mean:>12.4f}  {mpb_stats.get('harm_mean', 0):>12.4f}")
    print(f"{'% > 1':>20}  {fdfd_pct_above_1:>12.4f}  {mpb_stats.get('pct_above_1', 0):>12.4f}")
    print(f"{'fill %':>20}  {fdfd_pct_above_1:>12.4f}  {mpb_stats.get('fill_pct', 0):>12.4f}  {fill_analytic:>12.4f}")

    # Check if they match
    if mpb_stats:
        mean_diff = abs(fdfd_mean - mpb_stats['mean']) / mpb_stats['mean']
        fill_diff = abs(fdfd_pct_above_1 - mpb_stats['pct_above_1'])
        print(f"\n  Mean ε relative difference: {mean_diff:.4%}")
        print(f"  Fill fraction absolute diff: {fill_diff:.2f}%")
        if mean_diff < 0.01 and fill_diff < 2:
            print("  ✓ Statistics MATCH (geometry is likely correct)")
        else:
            print("  ✗ Statistics MISMATCH (geometry convention error!)")


def compute_k_point_cartesian():
    """Compute Cartesian K-point for the honeycomb BZ."""
    a = 1.0
    B = a * np.array([[1.0, 0.5],
                      [0.0, np.sqrt(3) / 2]])
    # Reciprocal lattice: b = 2π (B^{-T})
    B_inv_T = np.linalg.inv(B).T
    b1 = 2 * np.pi * B_inv_T[:, 0]
    b2 = 2 * np.pi * B_inv_T[:, 1]
    # K = (1/3) b1 + (1/3) b2
    K = (1.0/3) * b1 + (1.0/3) * b2
    print(f"\nReciprocal vectors:")
    print(f"  b1 = ({b1[0]:.6f}, {b1[1]:.6f})")
    print(f"  b2 = ({b2[0]:.6f}, {b2[1]:.6f})")
    print(f"  K = (1/3, 1/3) → Cartesian ({K[0]:.6f}, {K[1]:.6f})")
    print(f"  |K| = {np.linalg.norm(K):.6f}")
    return K


def run_fdfd_at_K(eps_grid, info, K_cart):
    """Run FDFD at the K-point and report frequencies."""
    from .fdfd_solver import build_fdfd_operator
    from scipy.sparse.linalg import eigsh

    L_op = build_fdfd_operator(eps_grid, info, q_vec=K_cart)
    eigenvalues, _ = eigsh(L_op, k=10, sigma=0.01, which='LM',
                           maxiter=10000, tol=1e-10)
    eigenvalues = np.sort(eigenvalues)
    omega = np.sqrt(np.maximum(eigenvalues, 0))
    freqs = omega / (2 * np.pi)
    return eigenvalues, freqs


def main():
    print("=" * 60)
    print("DIAGNOSTIC: FDFD vs MPB Epsilon Comparison")
    print("Honeycomb monolayer, TM polarization")
    print(f"eps_bg={EPS_BG}, eps_hole={EPS_HOLE}, r/a={R_OVER_A}")
    print(f"Resolution: {RESOLUTION}")
    print("=" * 60)

    # Step 1: Build FDFD epsilon ─────────────────────
    print("\n── Building FDFD epsilon grid ──")
    eps_fdfd, info = build_fdfd_eps_monolayer(RESOLUTION)
    print(f"  Shape: {eps_fdfd.shape}")
    print(f"  Range: [{eps_fdfd.min():.4f}, {eps_fdfd.max():.4f}]")
    print(f"  Rod fraction: {(eps_fdfd > 5).mean():.4f}")

    # Step 2: Run MPB ─────────────────────────────────
    print("\n── Running MPB ──")
    work_dir = tempfile.mkdtemp(prefix='mpb_diag_')
    print(f"  Working directory: {work_dir}")
    mpb_stats, mpb_freqs, mpb_output = run_mpb_and_extract_eps(work_dir)

    if mpb_freqs:
        print(f"\n  MPB K-point frequencies: {mpb_freqs}")

    # Step 3: Compare statistics ──────────────────────
    compare_epsilon_stats(eps_fdfd, mpb_stats)

    # Step 4: FDFD eigenvalues at K ───────────────────
    print("\n── FDFD eigenvalues at K ──")
    K_cart = compute_k_point_cartesian()
    eigenvalues, fdfd_freqs = run_fdfd_at_K(eps_fdfd, info, K_cart)

    print(f"\nFDFD K-point frequencies (f = √λ/2π):")
    for i, f in enumerate(fdfd_freqs[:6]):
        print(f"  band {i}: f = {f:.6f}")

    if mpb_freqs:
        print(f"\nComparison (first 6 bands):")
        print(f"  {'band':>4}  {'FDFD':>10}  {'MPB':>10}  {'ratio':>8}")
        for i in range(min(6, len(mpb_freqs))):
            f_fdfd = fdfd_freqs[i] if i < len(fdfd_freqs) else float('nan')
            f_mpb = mpb_freqs[i]
            ratio = f_fdfd / f_mpb if f_mpb > 0 else float('nan')
            print(f"  {i:>4}  {f_fdfd:>10.6f}  {f_mpb:>10.6f}  {ratio:>8.4f}")

    # Step 5: Also test at K'=(2/3,1/3) which was used before ────
    print("\n── FDFD eigenvalues at K'=(2/3,1/3) ──")
    B = info['B_super']
    B_inv_T = np.linalg.inv(B).T
    b1 = 2 * np.pi * B_inv_T[:, 0]
    b2 = 2 * np.pi * B_inv_T[:, 1]
    Kprime = (2.0/3) * b1 + (1.0/3) * b2
    print(f"  K' Cartesian: ({Kprime[0]:.6f}, {Kprime[1]:.6f})")

    eigenvalues_Kp, fdfd_freqs_Kp = run_fdfd_at_K(eps_fdfd, info, Kprime)
    print(f"\nFDFD K'-point frequencies (f = √λ/2π):")
    for i, f in enumerate(fdfd_freqs_Kp[:6]):
        print(f"  band {i}: f = {f:.6f}")

    # Run MPB at K' too for comparison
    work_dir2 = tempfile.mkdtemp(prefix='mpb_diag_Kp_')
    ctl_Kp = f"""; Honeycomb monolayer at K'=(2/3, 1/3)
(set! geometry-lattice
  (make lattice (size 1 1 no-size)
    (basis1 1 0) (basis2 0.5 (/ (sqrt 3) 2))))
(set! default-material (make dielectric (epsilon {EPS_BG})))
(set! geometry (list
    (make cylinder (center 0 0 0) (radius {R_OVER_A}) (height infinity)
      (material (make dielectric (epsilon {EPS_HOLE}))))
    (make cylinder (center (/ 1 3) (/ 1 3) 0) (radius {R_OVER_A}) (height infinity)
      (material (make dielectric (epsilon {EPS_HOLE}))))))
(set! resolution {RESOLUTION})
(set! num-bands 6)
(set! k-points (list (vector3 (/ 2 3) (/ 1 3) 0)))
(run-tm)
"""
    ctl_path = os.path.join(work_dir2, 'honeycomb_Kp.ctl')
    with open(ctl_path, 'w') as f:
        f.write(ctl_Kp)

    result = subprocess.run(['mpb', 'honeycomb_Kp.ctl'], cwd=work_dir2,
                            capture_output=True, text=True, timeout=120)
    mpb_freqs_Kp = []
    for line in result.stdout.split('\n'):
        if 'tmfreqs:' in line and 'band' not in line:
            parts = line.split(',')
            mpb_freqs_Kp = [float(x.strip()) for x in parts[6:] if x.strip()]
            break

    if mpb_freqs_Kp:
        print(f"\nComparison at K' (first 6 bands):")
        print(f"  {'band':>4}  {'FDFD':>10}  {'MPB':>10}  {'ratio':>8}")
        for i in range(min(6, len(mpb_freqs_Kp))):
            f_fdfd = fdfd_freqs_Kp[i] if i < len(fdfd_freqs_Kp) else float('nan')
            f_mpb = mpb_freqs_Kp[i]
            ratio = f_fdfd / f_mpb if f_mpb > 0 else float('nan')
            print(f"  {i:>4}  {f_fdfd:>10.6f}  {f_mpb:>10.6f}  {ratio:>8.4f}")


if __name__ == '__main__':
    main()
