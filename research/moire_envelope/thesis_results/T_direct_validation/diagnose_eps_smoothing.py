"""
Phase D1: Epsilon Smoothing Diagnostic

Compares binary (staircase) vs subpixel-smoothed epsilon on:
  1. Monolayer honeycomb unit cell — against MPB reference eigenvalues
  2. Moiré (30,29) supercell at res=40 — against EA eigenvalues

Key questions answered:
  - How many boundary pixels exist? What fraction of the grid?
  - How much does smoothing change eigenvalues on a monolayer?
  - Does smoothed epsilon reduce the EA-FDFD residual on the supercell?
"""

import numpy as np
import sys
import os
import time
import subprocess
import tempfile
import re
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from T_direct_validation.subpixel_smoothing import (
    build_smoothed_eps_monolayer,
    build_smoothed_eps_supercell,
)
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.commensurate_utils import commensurate_twist_angle

from scipy.sparse.linalg import eigsh


# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
EPS_BG = 1.0
EPS_ROD = 11.56
R_OVER_A = 0.2

MONO_RESOLUTION = 64     # monolayer grid resolution
SUPERCELL_RES = 40        # supercell grid resolution per unit cell
N_SUB = 16                # sub-grid resolution for smoothing

M, N_MN = 30, 29          # commensurate indices
N_FDFD_MODES = 100        # modes to compute


# ════════════════════════════════════════════════════════════════
# Part 1: Monolayer comparison
# ════════════════════════════════════════════════════════════════
def run_monolayer_comparison():
    """Compare binary vs smoothed epsilon on monolayer, against MPB."""
    print("=" * 70)
    print("PART 1: MONOLAYER EPSILON COMPARISON")
    print(f"Resolution={MONO_RESOLUTION}, n_sub={N_SUB}")
    print("=" * 70)

    # Build binary and smoothed epsilon
    t0 = time.time()
    eps_binary, eps_smoothed, info = build_smoothed_eps_monolayer(
        resolution=MONO_RESOLUTION, a=1.0, r_over_a=R_OVER_A,
        eps_rod=EPS_ROD, eps_bg=EPS_BG, n_sub=N_SUB,
    )
    t_build = time.time() - t0

    print(f"\nBuild time: {t_build:.2f}s")
    print(f"Grid shape: {eps_binary.shape}")
    print(f"Boundary pixel candidates: {info['n_boundary_candidates']}")
    print(f"Actually smoothed: {info['n_smoothed']}")
    print(f"Smoothed fraction: {info['n_smoothed'] / eps_binary.size:.4%}")
    print(f"Pixel diagonal: {info['pixel_diag']:.6f}")

    # Statistics comparison
    print(f"\n{'─' * 60}")
    print(f"{'Statistic':>25}  {'Binary':>12}  {'Smoothed':>12}  {'Diff':>12}")
    print(f"{'─' * 60}")

    for label, arr in [('Binary', eps_binary), ('Smoothed', eps_smoothed)]:
        pass  # just to define scope

    stats = {}
    for label, arr in [('binary', eps_binary), ('smoothed', eps_smoothed)]:
        s = {
            'mean': arr.mean(),
            'harm_mean': 1.0 / (1.0 / arr).mean(),
            'fill_pct': (arr > 0.5 * (EPS_BG + EPS_ROD)).mean() * 100,
            'n_unique': len(np.unique(arr)),
        }
        stats[label] = s

    # Analytic fill fraction
    cell_area = np.sqrt(3) / 2
    rod_area = 2 * np.pi * R_OVER_A**2
    fill_analytic = rod_area / cell_area * 100

    b, s = stats['binary'], stats['smoothed']
    print(f"{'mean ε':>25}  {b['mean']:>12.6f}  {s['mean']:>12.6f}  {s['mean']-b['mean']:>+12.6f}")
    print(f"{'harmonic mean ε':>25}  {b['harm_mean']:>12.6f}  {s['harm_mean']:>12.6f}  {s['harm_mean']-b['harm_mean']:>+12.6f}")
    print(f"{'fill fraction (%)':>25}  {b['fill_pct']:>12.4f}  {s['fill_pct']:>12.4f}  {'N/A':>12}")
    print(f"{'analytic fill (%)':>25}  {fill_analytic:>12.4f}")
    print(f"{'unique values':>25}  {b['n_unique']:>12d}  {s['n_unique']:>12d}")

    # Smoothed epsilon distribution at boundary pixels
    diff = eps_smoothed - eps_binary
    changed_mask = np.abs(diff) > 1e-10
    if changed_mask.sum() > 0:
        changed_vals = eps_smoothed[changed_mask]
        print(f"\n{'─' * 60}")
        print(f"Smoothed boundary pixels: {changed_mask.sum()}")
        print(f"  ε range: [{changed_vals.min():.4f}, {changed_vals.max():.4f}]")
        print(f"  ε mean:  {changed_vals.mean():.4f}")
        print(f"  |Δε| range: [{np.abs(diff[changed_mask]).min():.4f}, {np.abs(diff[changed_mask]).max():.4f}]")
        print(f"  |Δε| mean:  {np.abs(diff[changed_mask]).mean():.4f}")

    # ── Run MPB for reference eigenvalues ──
    print(f"\n{'─' * 60}")
    print("Running MPB for reference eigenvalues...")
    mpb_freqs = run_mpb_monolayer()

    # ── FDFD eigenvalues at K-point ──
    print(f"\n{'─' * 60}")
    print("Running FDFD at K-point (binary and smoothed)...")

    B = info['B_super']
    B_inv_T = np.linalg.inv(B).T
    b1 = 2 * np.pi * B_inv_T[:, 0]
    b2 = 2 * np.pi * B_inv_T[:, 1]
    K_cart = (1.0 / 3) * b1 + (1.0 / 3) * b2

    fdfd_info = {
        'B_super': B,
        'L1': B[:, 0],
        'L2': B[:, 1],
    }

    freqs_binary = solve_fdfd_monolayer(eps_binary, fdfd_info, K_cart, n_modes=10)
    freqs_smoothed = solve_fdfd_monolayer(eps_smoothed, fdfd_info, K_cart, n_modes=10)

    print(f"\n{'─' * 60}")
    print(f"{'band':>4}  {'MPB':>10}  {'FDFD-bin':>10}  {'FDFD-smo':>10}  {'Δ(bin)':>10}  {'Δ(smo)':>10}")
    print(f"{'─' * 60}")

    n_compare = min(6, len(mpb_freqs), len(freqs_binary), len(freqs_smoothed))
    for i in range(n_compare):
        f_mpb = mpb_freqs[i]
        f_bin = freqs_binary[i]
        f_smo = freqs_smoothed[i]
        d_bin = f_bin - f_mpb
        d_smo = f_smo - f_mpb
        print(f"{i:>4}  {f_mpb:>10.6f}  {f_bin:>10.6f}  {f_smo:>10.6f}  {d_bin:>+10.6f}  {d_smo:>+10.6f}")

    if n_compare >= 2:
        err_bin = np.array([freqs_binary[i] - mpb_freqs[i] for i in range(n_compare)])
        err_smo = np.array([freqs_smoothed[i] - mpb_freqs[i] for i in range(n_compare)])
        print(f"\n  RMS error (binary):   {np.sqrt(np.mean(err_bin**2)):.6e}")
        print(f"  RMS error (smoothed): {np.sqrt(np.mean(err_smo**2)):.6e}")
        improvement = np.sqrt(np.mean(err_bin**2)) / np.sqrt(np.mean(err_smo**2))
        print(f"  Improvement factor:   {improvement:.2f}×")

    return eps_binary, eps_smoothed, info


def run_mpb_monolayer() -> list:
    """Run MPB to get monolayer eigenvalues at K."""
    ctl = f"""; Honeycomb monolayer: Si rods in air, TM polarization
(set! geometry-lattice
  (make lattice (size 1 1 no-size)
    (basis1 1 0) (basis2 0.5 (/ (sqrt 3) 2))))
(set! default-material (make dielectric (epsilon {EPS_BG})))
(set! geometry (list
    (make cylinder (center 0 0 0) (radius {R_OVER_A}) (height infinity)
      (material (make dielectric (epsilon {EPS_ROD}))))
    (make cylinder (center (/ 1 3) (/ 1 3) 0) (radius {R_OVER_A}) (height infinity)
      (material (make dielectric (epsilon {EPS_ROD}))))))
(set! resolution {MONO_RESOLUTION})
(set! num-bands 10)
(set! k-points (list (vector3 (/ 1 3) (/ 1 3) 0)))
(run-tm)
"""
    work_dir = tempfile.mkdtemp(prefix='mpb_d1_')
    ctl_path = os.path.join(work_dir, 'mono.ctl')
    with open(ctl_path, 'w') as f:
        f.write(ctl)

    result = subprocess.run(
        ['mpb', 'mono.ctl'], cwd=work_dir,
        capture_output=True, text=True, timeout=120,
    )

    freqs = []
    for line in result.stdout.split('\n'):
        if 'tmfreqs:' in line and 'band' not in line.lower():
            parts = line.split(',')
            freqs = [float(x.strip()) for x in parts[6:] if x.strip()]
            break

    if freqs:
        print(f"  MPB returned {len(freqs)} frequencies")
    else:
        print(f"  WARNING: No frequencies parsed from MPB output")
        # Try alternate parsing
        for line in result.stdout.split('\n')[-30:]:
            print(f"    {line}")

    return freqs


def solve_fdfd_monolayer(eps_grid, info, q_vec, n_modes=10):
    """Solve FDFD eigenproblem and return sorted frequencies."""
    L_op = build_fdfd_operator(eps_grid, info, q_vec=q_vec, polarization='tm')
    eigenvalues, _ = eigsh(L_op, k=n_modes, sigma=0.01, which='LM',
                           maxiter=10000, tol=1e-10)
    eigenvalues = np.sort(eigenvalues)
    freqs = np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    return freqs


# ════════════════════════════════════════════════════════════════
# Part 2: Supercell comparison
# ════════════════════════════════════════════════════════════════
def run_supercell_comparison():
    """Compare binary vs smoothed epsilon on moiré supercell."""
    print("\n\n" + "=" * 70)
    print("PART 2: MOIRÉ SUPERCELL EPSILON COMPARISON")
    print(f"(m,n)=({M},{N_MN}), res={SUPERCELL_RES}, n_sub={N_SUB}")
    print("=" * 70)

    N_cells = M * M + M * N_MN + N_MN * N_MN
    Nx = int(round(np.sqrt(N_cells) * SUPERCELL_RES))
    theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N_MN))
    print(f"θ = {theta_deg:.4f}°, N_cells = {N_cells}, Nx = {Nx}, DOF = {Nx*Nx:,}")

    # Build binary epsilon
    print("\n── Building binary epsilon ──")
    t0 = time.time()
    eps_binary, sc_info = build_supercell_eps(
        'honeycomb', m=M, n=N_MN, a=1.0,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=Nx, Ny=Nx,
    )
    t_binary = time.time() - t0
    print(f"  Build time: {t_binary:.1f}s")

    # Build smoothed epsilon
    print("\n── Building smoothed epsilon ──")
    t0 = time.time()
    eps_smoothed, smooth_info = build_smoothed_eps_supercell(
        eps_binary, sc_info, n_sub=N_SUB,
        eps_rod=EPS_ROD, eps_bg=EPS_BG,
    )
    t_smooth = time.time() - t0
    print(f"  Build time: {t_smooth:.1f}s")
    print(f"  Boundary candidates: {smooth_info['n_boundary_candidates']}")
    print(f"  Actually smoothed: {smooth_info['n_smoothed']}")
    print(f"  Smoothed fraction: {smooth_info['n_smoothed'] / eps_binary.size:.4%}")

    # Statistics
    diff = eps_smoothed - eps_binary
    changed = np.abs(diff) > 1e-10
    print(f"\n  Pixels changed: {changed.sum()}")
    print(f"  Mean |Δε|: {np.abs(diff[changed]).mean():.4f}" if changed.any() else "")
    print(f"  Binary mean ε: {eps_binary.mean():.6f}")
    print(f"  Smoothed mean ε: {eps_smoothed.mean():.6f}")

    # ── Load envelope reference ──
    env_freqs = load_envelope_reference()
    if env_freqs is None:
        print("\n  WARNING: Could not load envelope reference. Skipping FDFD solve.")
        return

    env_center = 0.5 * (env_freqs.min() + env_freqs.max())
    env_bw = env_freqs.max() - env_freqs.min()
    sigma_target = (2 * np.pi * env_center) ** 2

    print(f"\n  Envelope: {len(env_freqs)} modes, center={env_center:.6f}, BW={env_bw:.6f}")

    # ── FDFD solve: binary ──
    print(f"\n── FDFD solve (binary ε) ──")
    fdfd_freqs_binary = solve_fdfd_supercell(eps_binary, sc_info, sigma_target)

    # ── FDFD solve: smoothed ──
    print(f"\n── FDFD solve (smoothed ε) ──")
    fdfd_freqs_smoothed = solve_fdfd_supercell(eps_smoothed, sc_info, sigma_target)

    # ── Compare against envelope ──
    print(f"\n{'─' * 70}")
    print("EA-FDFD RESIDUAL COMPARISON")
    print(f"{'─' * 70}")

    from scipy.optimize import linear_sum_assignment

    for label, fdfd_freqs in [('Binary', fdfd_freqs_binary), ('Smoothed', fdfd_freqs_smoothed)]:
        # Window: modes within envelope frequency range
        mask = (fdfd_freqs >= env_freqs.min() - 0.002) & (fdfd_freqs <= env_freqs.max() + 0.002)
        fdfd_window = fdfd_freqs[mask]

        n_match = min(len(env_freqs), len(fdfd_window))
        if n_match < 10:
            print(f"  {label}: only {n_match} modes in window — skipping")
            continue

        # Hungarian matching
        cost = np.abs(env_freqs[:n_match, None] - fdfd_window[None, :n_match])
        row_idx, col_idx = linear_sum_assignment(cost)
        residuals = np.abs(env_freqs[row_idx] - fdfd_window[col_idx])

        mean_res = residuals.mean()
        max_res = residuals.max()
        bw_ratio = (fdfd_window[col_idx].max() - fdfd_window[col_idx].min()) / env_bw

        print(f"\n  {label} ε:")
        print(f"    Matched modes: {len(row_idx)}")
        print(f"    Mean |Δω|: {mean_res:.6e}")
        print(f"    Max  |Δω|: {max_res:.6e}")
        print(f"    Mean |Δω|/BW: {mean_res/env_bw:.4%}")
        print(f"    BW ratio (FDFD/EA): {bw_ratio:.4f}")


def load_envelope_reference():
    """Load EA eigenvalues from existing sweep results."""
    sweep_paths = [
        '/home/renlephy/msl/research/moire_envelope/runsV3/'
        'thesis_honeycomb_K_b1_20260307_171424/'
        'eta_sweep_20260310_191610/sweep_results.json',
    ]
    for path in sweep_paths:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # Find the (30,29) entry (θ ≈ 1.12°)
            for entry in data:
                if abs(entry.get('theta_deg', 0) - 1.12) < 0.1:
                    omega_ref = entry['omega_ref']
                    evals = np.array(entry['eigenvalues'])
                    freqs = np.sort(omega_ref + evals)
                    return freqs
            # If no theta_deg field, take the first entry
            entry = data[0]
            omega_ref = entry['omega_ref']
            evals = np.array(entry['eigenvalues'])
            freqs = np.sort(omega_ref + evals)
            return freqs
    return None


def solve_fdfd_supercell(eps_grid, info, sigma_target):
    """Solve FDFD eigenproblem on supercell with CHOLMOD shift-invert."""
    import scipy.sparse as sp

    t0 = time.time()
    L = build_fdfd_operator(eps_grid, info, q_vec=np.zeros(2), polarization='tm')
    t_assemble = time.time() - t0
    N_dof = L.shape[0]
    print(f"  Operator: {N_dof:,} DOF, nnz={L.nnz:,}, assembly={t_assemble:.1f}s")

    # Shift-invert with CHOLMOD
    L_shifted = (L - sigma_target * sp.eye(N_dof, format='csc')).tocsc()

    from sksparse.cholmod import cholesky
    from scipy.sparse.linalg import LinearOperator

    t0 = time.time()
    factor = cholesky(L_shifted, beta=0, mode='simplicial')
    t_factor = time.time() - t0
    print(f"  CHOLMOD factorization: {t_factor:.1f}s")

    del L_shifted

    OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L.dtype)

    t0 = time.time()
    evals, _ = eigsh(L, k=N_FDFD_MODES, sigma=sigma_target, which='LM',
                     OPinv=OPinv, maxiter=10000, tol=1e-8)
    t_solve = time.time() - t0
    print(f"  Eigensolver: {t_solve:.1f}s")

    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    return np.sort(freqs)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase D1: Epsilon Smoothing Diagnostic                     ║")
    print("║  Binary vs Subpixel-Smoothed ε                              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Part 1: Monolayer (cheap — seconds)
    run_monolayer_comparison()

    # Part 2: Supercell (expensive — ~15 min per FDFD solve)
    run_supercell_comparison()

    print("\n" + "=" * 70)
    print("Phase D1 complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
