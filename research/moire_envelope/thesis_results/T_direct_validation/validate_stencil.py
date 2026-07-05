#!/usr/bin/env python3
"""
Stencil Expansion Validation
=============================
Validates the expanded k-stencil (7×7, dk=0.06, 6th-order FD) against
the old stencil (5×5, dk=0.01, 4th-order FD).

Three validation panels:
  1. Stencil Coverage Visualization — ω(kx,ky) surface with stencil grid
     and moiré BZ excursion circles at various angles
  2. Interpolation Accuracy — polynomial fit vs direct MPB at test K-points
  3. Numerical Differences vs Angle — old vs new stencil at θ=1°,2°,5°,10°

Usage:
    python validate_stencil.py              # run all
    python validate_stencil.py --plot-only  # just plot from saved data
"""

import sys, os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# ═══════════════════════════════════════════════════════════════
#  Physical parameters (same as square_supercell_3way.py)
# ═══════════════════════════════════════════════════════════════

A = 1.0
R_OVER_A = 0.2
EPS_ROD = 11.56
EPS_BG = 1.0
TARGET_BAND = 3   # 0-indexed, TM band 3 at M
K0 = [0.5, 0.5]   # M-point
N_BANDS = 8
MPB_RES = 128

OUTDIR = SCRIPT_DIR / "square_3way"


def compute_single_stencil(dk, fd_order, k0=K0):
    """Compute a single k-stencil at δ=(0,0) for comparison."""
    from phase1_mpb_v3 import compute_bands_at_k_stencil, create_mpb_geometry, create_mpb_solver

    geometry, lattice, bg_eps = create_mpb_geometry(
        'square', R_OVER_A, EPS_BG, eps_hole=EPS_ROD, delta_frac=np.array([0.0, 0.0])
    )
    ms = create_mpb_solver(geometry, lattice, bg_eps, N_BANDS, MPB_RES, 'TM')

    result = compute_bands_at_k_stencil(
        ms, k0, dk, list(range(N_BANDS)), 'TM', fd_order
    )
    return result


# ═══════════════════════════════════════════════════════════════
#  Panel 1: Stencil Coverage Visualization
# ═══════════════════════════════════════════════════════════════

def plot_stencil_coverage(stencil_new, stencil_old):
    """Plot ω(kx,ky) surface with stencil points and excursion circles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    band = TARGET_BAND  # band index within all_bands

    for ax_idx, (stencil, dk, label, offsets) in enumerate([
        (stencil_old, 0.01, '5×5, dk=0.01 (old)', [-2,-1,0,1,2]),
        (stencil_new, 0.06, '7×7, dk=0.06 (new)', [-3,-2,-1,0,1,2,3]),
    ]):
        ax = axes[ax_idx]
        omega = stencil['omega_stencil'][band]  # (n_stencil, n_stencil)
        n = len(offsets)

        # Stencil grid in k-space
        kx_grid = np.array(offsets) * dk + K0[0]
        ky_grid = np.array(offsets) * dk + K0[1]
        KX, KY = np.meshgrid(kx_grid, ky_grid, indexing='ij')

        # Plot omega as pcolormesh
        im = ax.pcolormesh(KX, KY, omega, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, label=r'$\omega\, a/2\pi c$', shrink=0.8)

        # Overlay stencil points
        ax.scatter(KX.ravel(), KY.ravel(), c='white', s=15, edgecolor='black',
                   linewidth=0.5, zorder=5)

        # Mark center
        ax.plot(K0[0], K0[1], 'r+', ms=12, mew=2, zorder=6)

        # Excursion circles for various twist angles
        angles_deg = [1, 2, 5, 10]
        colors = ['cyan', 'lime', 'orange', 'red']
        for theta_deg, color in zip(angles_deg, colors):
            radius = 2 * np.sin(np.radians(theta_deg / 2))
            circle = plt.Circle((K0[0], K0[1]), radius, fill=False,
                                color=color, lw=1.5, ls='--', zorder=4)
            ax.add_patch(circle)
            ax.annotate(f'{theta_deg}°', xy=(K0[0] + radius * 0.7, K0[1] + radius * 0.7),
                        color=color, fontsize=8, fontweight='bold')

        # Stencil boundary
        half_extent = max(abs(offsets[-1]), abs(offsets[0])) * dk
        rect = plt.Rectangle((K0[0] - half_extent, K0[1] - half_extent),
                              2 * half_extent, 2 * half_extent,
                              fill=False, edgecolor='red', lw=2, ls='-', zorder=3)
        ax.add_patch(rect)

        ax.set_xlabel(r'$k_x\, (2\pi/a)$')
        ax.set_ylabel(r'$k_y\, (2\pi/a)$')
        ax.set_title(f'Stencil: {label}')
        ax.set_aspect('equal')

        # Set consistent view
        view = 0.22
        ax.set_xlim(K0[0] - view, K0[0] + view)
        ax.set_ylim(K0[1] - view, K0[1] + view)

    fig.suptitle(f'Stencil Coverage: TM Band {TARGET_BAND} at M-point\n'
                 'Circles show moiré BZ excursion at various twist angles',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig_stencil_coverage.png', dpi=200)
    print(f"  Saved stencil coverage plot")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  Panel 2: Interpolation Accuracy
# ═══════════════════════════════════════════════════════════════

def validate_interpolation(stencil_new):
    """Compare polynomial interpolation vs direct MPB at test K-points."""
    from phasesV3.stencil_interpolation import fit_quadratic_2d, fit_quartic_2d
    from phasesV3.stencil_interpolation import evaluate_quadratic, evaluate_quartic
    from phase1_mpb_v3 import compute_bands_at_k_stencil, create_mpb_geometry, create_mpb_solver

    dk = 0.06
    offsets = [-3, -2, -1, 0, 1, 2, 3]
    band = TARGET_BAND

    # Fit polynomials to the stencil
    omega_stencil = stencil_new['omega_stencil'][band]
    coeffs_q2, rms_q2 = fit_quadratic_2d(omega_stencil, offsets, dk)
    coeffs_q4, rms_q4 = fit_quartic_2d(omega_stencil, offsets, dk)

    print(f"  Polynomial fit RMS — quadratic: {rms_q2:.2e}, quartic: {rms_q4:.2e}")

    # Test K-points (within stencil patch)
    test_Ks = [
        (0.00, 0.00),
        (0.05, 0.00),
        (0.00, 0.05),
        (0.05, 0.05),
        (-0.05, 0.03),
        (0.10, 0.00),
        (0.10, 0.10),
        (-0.10, -0.10),
        (0.15, 0.00),
    ]

    # Direct MPB at each test K-point
    geometry, lattice, bg_eps = create_mpb_geometry(
        'square', R_OVER_A, EPS_BG, eps_hole=EPS_ROD, delta_frac=np.array([0.0, 0.0])
    )
    ms = create_mpb_solver(geometry, lattice, bg_eps, N_BANDS, MPB_RES, 'TM')

    results = {'K': [], 'mpb': [], 'quad': [], 'quartic': []}

    for Kx, Ky in test_Ks:
        # Direct MPB at k0 + K
        import meep as mp
        ms.k_points = [mp.Vector3(K0[0] + Kx, K0[1] + Ky, 0)]
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            mp.verbosity(0)
            ms.run_tm()
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(devnull_fd)
            os.close(old_stdout)
            os.close(old_stderr)

        omega_mpb = float(np.array(ms.all_freqs[0])[band])

        # Polynomial predictions
        omega_q2, _, _ = evaluate_quadratic(coeffs_q2, Kx, Ky)
        omega_q4, _, _ = evaluate_quartic(coeffs_q4, Kx, Ky)

        results['K'].append((Kx, Ky))
        results['mpb'].append(omega_mpb)
        results['quad'].append(float(omega_q2))
        results['quartic'].append(float(omega_q4))

        print(f"    K=({Kx:+.3f}, {Ky:+.3f}): MPB={omega_mpb:.6f}, "
              f"quad={omega_q2:.6f} (Δ={abs(omega_q2-omega_mpb):.2e}), "
              f"quart={omega_q4:.6f} (Δ={abs(omega_q4-omega_mpb):.2e})")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    K_labels = [f'({k[0]:+.2f},{k[1]:+.2f})' for k in results['K']]
    K_dist = [np.sqrt(k[0]**2 + k[1]**2) for k in results['K']]
    idx = np.argsort(K_dist)

    mpb_arr = np.array(results['mpb'])
    quad_arr = np.array(results['quad'])
    quart_arr = np.array(results['quartic'])

    # Panel 1: absolute error
    ax = axes[0]
    x = np.arange(len(idx))
    ax.semilogy(x, np.abs(quad_arr[idx] - mpb_arr[idx]), 'bo-', ms=6, label='Quadratic')
    ax.semilogy(x, np.abs(quart_arr[idx] - mpb_arr[idx]), 'rs-', ms=6, label='Quartic')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{K_dist[i]:.3f}' for i in idx], rotation=45, fontsize=8)
    ax.set_xlabel('|K| from center')
    ax.set_ylabel(r'$|\omega_{fit} - \omega_{MPB}|$')
    ax.set_title('Interpolation Error vs |K|')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: relative error (%)
    ax = axes[1]
    rel_q2 = np.abs(quad_arr[idx] - mpb_arr[idx]) / mpb_arr[idx] * 100
    rel_q4 = np.abs(quart_arr[idx] - mpb_arr[idx]) / mpb_arr[idx] * 100
    ax.semilogy(x, rel_q2, 'bo-', ms=6, label='Quadratic')
    ax.semilogy(x, rel_q4, 'rs-', ms=6, label='Quartic')
    ax.axhline(0.1, color='gray', ls='--', lw=0.5, label='0.1%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{K_dist[i]:.3f}' for i in idx], rotation=45, fontsize=8)
    ax.set_xlabel('|K| from center')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Relative Interpolation Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Stencil Interpolation Accuracy: TM Band {TARGET_BAND} at M\n'
                 f'dk=0.06, 7×7 stencil, δ=(0,0)', fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig_interpolation_accuracy.png', dpi=200)
    print(f"  Saved interpolation accuracy plot")
    plt.close()

    return results


# ═══════════════════════════════════════════════════════════════
#  Panel 3: Old vs New Stencil — Numerical Differences vs Angle
# ═══════════════════════════════════════════════════════════════

def compare_stencils_vs_angle():
    """Compare old (5×5, dk=0.01) vs new (7×7, dk=0.06) stencil at various δ shifts."""
    from phasesV3.stencil_interpolation import fit_quadratic_2d, evaluate_quadratic
    from phase1_mpb_v3 import compute_bands_at_k_stencil, create_mpb_geometry, create_mpb_solver

    band = TARGET_BAND

    # Compute stencils at several δ shifts for both old and new
    delta_fracs = [
        np.array([0.0, 0.0]),
        np.array([0.25, 0.0]),
        np.array([0.0, 0.25]),
        np.array([0.25, 0.25]),
        np.array([0.5, 0.0]),
    ]

    configs = [
        ('old', 0.01, 4, [-2,-1,0,1,2]),
        ('new', 0.06, 6, [-3,-2,-1,0,1,2,3]),
    ]

    # Angles to test excursion radii
    angles_deg = [1, 2, 5, 10]

    results = {
        'angles': angles_deg,
        'deltas': [d.tolist() for d in delta_fracs],
    }

    for label, dk, fd_order, offsets in configs:
        omega_at_K = []
        vg_at_K = []
        Minv_at_K = []

        for delta_frac in delta_fracs:
            geometry, lattice, bg_eps = create_mpb_geometry(
                'square', R_OVER_A, EPS_BG, eps_hole=EPS_ROD, delta_frac=delta_frac
            )
            ms = create_mpb_solver(geometry, lattice, bg_eps, N_BANDS, MPB_RES, 'TM')

            result = compute_bands_at_k_stencil(
                ms, K0, dk, list(range(N_BANDS)), 'TM', fd_order
            )

            # Fit polynomial
            omega_stencil = result['omega_stencil'][band]
            coeffs, rms = fit_quadratic_2d(omega_stencil, offsets, dk)

            # Evaluate at K-shifts corresponding to each angle
            omega_angle = []
            vg_angle = []
            Minv_angle = []
            for theta_deg in angles_deg:
                K_excursion = 2 * np.sin(np.radians(theta_deg / 2))
                Kx = K_excursion  # just use x-direction for comparison
                Ky = 0.0
                omega_K, vg_K, M_K = evaluate_quadratic(coeffs, Kx, Ky)
                omega_angle.append(float(omega_K))
                vg_angle.append(vg_K.tolist())
                Minv_angle.append(M_K.tolist())

            omega_at_K.append(omega_angle)
            vg_at_K.append(vg_angle)
            Minv_at_K.append(Minv_angle)

        results[f'{label}_omega'] = omega_at_K
        results[f'{label}_vg'] = vg_at_K
        results[f'{label}_Minv'] = Minv_at_K

    # Compute differences
    old_omega = np.array(results['old_omega'])   # (n_delta, n_angles)
    new_omega = np.array(results['new_omega'])
    diff_omega = np.abs(new_omega - old_omega)
    rel_diff = diff_omega / np.abs(old_omega) * 100

    # Also compute direct MPB at excursion points for ground truth
    print("  Computing direct MPB ground truth at excursion K-points...")
    mpb_omega = np.zeros((len(delta_fracs), len(angles_deg)))
    for di, delta_frac in enumerate(delta_fracs):
        geometry, lattice, bg_eps = create_mpb_geometry(
            'square', R_OVER_A, EPS_BG, eps_hole=EPS_ROD, delta_frac=delta_frac
        )
        ms = create_mpb_solver(geometry, lattice, bg_eps, N_BANDS, MPB_RES, 'TM')

        for ai, theta_deg in enumerate(angles_deg):
            import meep as mp
            K_excursion = 2 * np.sin(np.radians(theta_deg / 2))
            ms.k_points = [mp.Vector3(K0[0] + K_excursion, K0[1], 0)]
            mp.verbosity(0)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                ms.run_tm()
            finally:
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(devnull_fd)
                os.close(old_stdout)
                os.close(old_stderr)
            mpb_omega[di, ai] = float(np.array(ms.all_freqs[0])[band])

    # Error relative to MPB ground truth
    old_err = np.abs(old_omega - mpb_omega) / mpb_omega * 100
    new_err = np.abs(new_omega - mpb_omega) / mpb_omega * 100

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(angles_deg))
    delta_labels = [f'δ=({d[0]:.2f},{d[1]:.2f})' for d in delta_fracs]

    # Panel 1: Old stencil error vs angle
    ax = axes[0]
    for di in range(len(delta_fracs)):
        ax.semilogy(x, old_err[di], 'o-', ms=5, label=delta_labels[di])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}°' for a in angles_deg])
    ax.set_xlabel('Twist angle')
    ax.set_ylabel('Relative error vs MPB (%)')
    ax.set_title('Old stencil (5×5, dk=0.01)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: New stencil error vs angle
    ax = axes[1]
    for di in range(len(delta_fracs)):
        ax.semilogy(x, new_err[di], 's-', ms=5, label=delta_labels[di])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}°' for a in angles_deg])
    ax.set_xlabel('Twist angle')
    ax.set_ylabel('Relative error vs MPB (%)')
    ax.set_title('New stencil (7×7, dk=0.06)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Improvement ratio
    ax = axes[2]
    improvement = old_err / np.maximum(new_err, 1e-15)
    for di in range(len(delta_fracs)):
        ax.semilogy(x, improvement[di], 'd-', ms=5, label=delta_labels[di])
    ax.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}°' for a in angles_deg])
    ax.set_xlabel('Twist angle')
    ax.set_ylabel('Error ratio: old / new')
    ax.set_title('Improvement Factor')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Stencil Comparison vs Twist Angle — TM Band {TARGET_BAND} at M\n'
                 f'Quadratic fit error at K_excursion = 2 sin(θ/2) along kx',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig_stencil_comparison_vs_angle.png', dpi=200)
    print(f"  Saved stencil comparison plot")
    plt.close()

    # Print summary table
    print(f"\n  {'='*70}")
    print(f"  Stencil Error Summary (relative to direct MPB, %)")
    print(f"  {'='*70}")
    print(f"  {'Angle':>8s}", end='')
    for d in delta_labels:
        print(f"  {d:>22s}", end='')
    print()
    for ai, theta_deg in enumerate(angles_deg):
        print(f"  {theta_deg:>6d}°  ", end='')
        for di in range(len(delta_fracs)):
            o = old_err[di, ai]
            n = new_err[di, ai]
            print(f"  old:{o:.4f}% new:{n:.4f}%", end='')
        print()

    return results, old_err, new_err


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--skip-interpolation', action='store_true',
                        help='Skip interpolation accuracy test (fast)')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip angle comparison (slower)')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("  Stencil Expansion Validation")
    print(f"  Square lattice, TM band {TARGET_BAND} at M, MPB res={MPB_RES}")
    print("=" * 70)

    # ── 1. Compute both stencils at δ=(0,0) ──
    print("\n1. Computing stencils at δ=(0,0)...")
    stencil_old = compute_single_stencil(dk=0.01, fd_order=4)
    stencil_new = compute_single_stencil(dk=0.06, fd_order=6)

    omega0_old = stencil_old['omega0'][TARGET_BAND]
    omega0_new = stencil_new['omega0'][TARGET_BAND]
    print(f"  ω₀ old={omega0_old:.6f}, new={omega0_new:.6f}, "
          f"diff={abs(omega0_old-omega0_new):.2e}")

    print(f"  vg old={stencil_old['vg'][TARGET_BAND]}, "
          f"new={stencil_new['vg'][TARGET_BAND]}")
    print(f"  M_inv old diag={stencil_old['M_inv'][TARGET_BAND].diagonal()}, "
          f"new diag={stencil_new['M_inv'][TARGET_BAND].diagonal()}")

    # ── 2. Stencil coverage plot ──
    print("\n2. Plotting stencil coverage...")
    plot_stencil_coverage(stencil_new, stencil_old)

    # ── 3. Interpolation accuracy ──
    if not args.skip_interpolation:
        print("\n3. Validating interpolation accuracy...")
        interp_results = validate_interpolation(stencil_new)
    else:
        print("\n3. Skipping interpolation accuracy")

    # ── 4. Angle comparison ──
    if not args.skip_comparison:
        print("\n4. Comparing stencils vs twist angle...")
        comp_results, old_err, new_err = compare_stencils_vs_angle()
    else:
        print("\n4. Skipping angle comparison")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.0f}s ({dt/60:.1f}min)")


if __name__ == '__main__':
    main()
