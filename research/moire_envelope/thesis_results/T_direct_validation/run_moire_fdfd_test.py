#!/usr/bin/env python3
"""
Moiré FDFD Test: Single moiré cell FDFD vs EA comparison.

Steps:
1. Build ε(x,y) on a single moiré supercell (with & without subpixel smoothing)
2. Plot the epsilon map for visual verification
3. Solve FDFD eigenproblem on the single moiré cell
4. Compare with EA eigenvalues (from Phase A single-band test)

This eliminates the N_moire ambiguity: both EA and FDFD use exactly
one moiré period with periodic BCs at Γ.
"""

import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add phasesV3 to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / 'phasesV3'))
sys.path.insert(0, str(SCRIPT_DIR))

from T_direct_validation.supercell_geometry import build_moire_supercell_eps, build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle

# ═══════════════════════════════════════════════════════════════
#  Physical constants (same as overnight script)
# ═══════════════════════════════════════════════════════════════
A = 1.0
R_OVER_A = 0.2
EPS_BG = 1.0
EPS_ROD = 11.56
OMEGA0 = 0.68457

# Test case: 10° angle
M_IDX, N_IDX = 11, 1
THETA_RAD = commensurate_twist_angle('square', M_IDX, N_IDX)
THETA_DEG = math.degrees(THETA_RAD)
L_MOIRE = A / (2.0 * math.sin(THETA_RAD / 2.0))

# Output directory
OUTPUT_DIR = SCRIPT_DIR / 'moire_fdfd_test'
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_epsilon_comparison():
    """
    Plot epsilon maps for:
    1. Single moiré cell WITHOUT subpixel smoothing
    2. Single moiré cell WITH subpixel smoothing
    3. Commensurate supercell (for reference, only a portion)
    """
    print('='*60)
    print('  Step 1: Epsilon Maps')
    print('='*60)

    # Moiré cell resolution (pixels per moiré cell side)
    res_moire = 256

    # 1. Single moiré cell, no smoothing
    t0 = time.time()
    eps_no_smooth, info_no = build_moire_supercell_eps(
        'square', THETA_RAD, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=res_moire, Ny=res_moire,
        subpixel_smoothing=False,
    )
    t1 = time.time()
    print(f'  No smoothing: {eps_no_smooth.shape}, range=[{eps_no_smooth.min():.2f}, {eps_no_smooth.max():.2f}], t={t1-t0:.1f}s')

    # 2. Single moiré cell, with smoothing (8×8 subpixel)
    t0 = time.time()
    eps_smooth, info_sm = build_moire_supercell_eps(
        'square', THETA_RAD, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=res_moire, Ny=res_moire,
        subpixel_smoothing=True, smoothing_Nsub=8,
    )
    t1 = time.time()
    print(f'  With smoothing: {eps_smooth.shape}, range=[{eps_smooth.min():.2f}, {eps_smooth.max():.2f}], t={t1-t0:.1f}s')

    # 3. Commensurate supercell (full, for comparison)
    N_cells = M_IDX**2 + N_IDX**2
    l_super = math.sqrt(N_cells)
    res_commensurate = 32  # lower res for visualization
    nx_com = int(round(l_super * res_commensurate))
    eps_com, info_com = build_supercell_eps(
        'square', M_IDX, N_IDX, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=nx_com, Ny=nx_com,
    )
    print(f'  Commensurate: {eps_com.shape}, N_cells={N_cells}, range=[{eps_com.min():.2f}, {eps_com.max():.2f}]')
    print(f'  L_moire={L_MOIRE:.4f}a, theta={THETA_DEG:.2f}°')
    print(f'  Commensurate L = sqrt({N_cells})a = {l_super:.4f}a')
    print(f'  Moiré cells in commensurate supercell: {N_cells * (2*math.sin(THETA_RAD/2))**2:.1f}')

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: No smoothing
    ax = axes[0]
    L = info_no['L_moire']
    extent = [0, L, 0, L]
    im = ax.imshow(eps_no_smooth.T, origin='lower', cmap='RdBu_r',
                   extent=extent, aspect='equal', vmin=EPS_BG, vmax=EPS_ROD)
    ax.set_title(f'Single moiré cell (no smoothing)\n{res_moire}×{res_moire} px')
    ax.set_xlabel('x (a)')
    ax.set_ylabel('y (a)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='ε')
    # Mark unit cells
    for i in range(int(L/A) + 1):
        ax.axhline(i*A, color='gray', linewidth=0.3, alpha=0.5)
        ax.axvline(i*A, color='gray', linewidth=0.3, alpha=0.5)

    # Panel 2: With smoothing
    ax = axes[1]
    im = ax.imshow(eps_smooth.T, origin='lower', cmap='RdBu_r',
                   extent=extent, aspect='equal', vmin=EPS_BG, vmax=EPS_ROD)
    ax.set_title(f'Single moiré cell (8×8 subpixel)\n{res_moire}×{res_moire} px')
    ax.set_xlabel('x (a)')
    ax.set_ylabel('y (a)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='ε')
    for i in range(int(L/A) + 1):
        ax.axhline(i*A, color='gray', linewidth=0.3, alpha=0.5)
        ax.axvline(i*A, color='gray', linewidth=0.3, alpha=0.5)

    # Panel 3: Commensurate (zoom to one moiré cell region)
    ax = axes[2]
    L_com = l_super * A
    extent_com = [0, L_com, 0, L_com]
    im = ax.imshow(eps_com.T, origin='lower', cmap='RdBu_r',
                   extent=extent_com, aspect='equal', vmin=EPS_BG, vmax=EPS_ROD)
    ax.set_title(f'Commensurate supercell (N={N_cells})\n{nx_com}×{nx_com} px')
    ax.set_xlabel('x (a)')
    ax.set_ylabel('y (a)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='ε')
    # Show moiré cell boundary
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], 'g--', linewidth=2, label='1 moiré cell')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'θ = {THETA_DEG:.2f}° ({M_IDX},{N_IDX}), L_moiré = {L_MOIRE:.2f}a, '
                 f'ε_rod = {EPS_ROD}, r/a = {R_OVER_A}', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'epsilon_maps.png', dpi=200)
    plt.close(fig)
    print(f'  Saved epsilon_maps.png')

    # ── Zoom plot: pixel-level smoothing comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Zoom into one rod region (center-ish)
    center_idx = res_moire // 2
    zoom = 30  # pixels on each side
    slc = slice(center_idx - zoom, center_idx + zoom)

    for ax, eps, title in [
        (axes[0], eps_no_smooth, 'No smoothing'),
        (axes[1], eps_smooth, 'Subpixel smoothed (8×8)'),
    ]:
        sub = eps[slc, slc]
        im = ax.imshow(sub.T, origin='lower', cmap='RdBu_r',
                       vmin=EPS_BG, vmax=EPS_ROD, interpolation='none')
        ax.set_title(title)
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Zoom: Rod boundary smoothing comparison', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'epsilon_smoothing_zoom.png', dpi=200)
    plt.close(fig)
    print(f'  Saved epsilon_smoothing_zoom.png')

    return eps_smooth, info_sm, eps_no_smooth, info_no


def run_fdfd_moire(eps_grid, info, n_modes=80, label=''):
    """Run FDFD eigensolve on a single moiré cell."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, eigsh

    Nx = info['Nx']
    dof = Nx * Nx
    sigma = (2.0 * np.pi * OMEGA0) ** 2

    print(f'\n  FDFD solve ({label}):')
    print(f'    Grid: {Nx}×{Nx} = {dof:,} DOF')
    print(f'    sigma={sigma:.4f}, n_modes={n_modes}')

    t0 = time.time()
    operator = build_fdfd_operator(eps_grid, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    print(f'    Operator: nnz={operator.nnz:,}, dtype={operator.dtype}, t={t_op:.1f}s')

    # Shift-invert
    shifted = operator - sigma * sp.eye(dof, format='csc', dtype=operator.dtype)

    try:
        from sksparse.cholmod import cholesky
        t0 = time.time()
        factor = cholesky(shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        print(f'    CHOLMOD factorization: t={t_factor:.1f}s')

        op_inv = LinearOperator(operator.shape, matvec=lambda vec: factor(vec), dtype=operator.dtype)
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=sigma, which='LM',
                         OPinv=op_inv, maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0
    except ImportError:
        print('    CHOLMOD unavailable, using scipy fallback')
        t_factor = 0.0
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=sigma, which='LM',
                         maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0

    del operator, shifted
    gc.collect()

    evals = np.sort(evals)
    freqs = np.sqrt(np.maximum(evals, 0.0)) / (2.0 * np.pi)
    print(f'    Eigensolve: t={t_solve:.1f}s')
    print(f'    Freq range: [{freqs.min():.6f}, {freqs.max():.6f}]')
    print(f'    Bandwidth: {freqs.max() - freqs.min():.6e}')

    return freqs, evals


def run_comparison():
    """Full comparison: plot epsilon, run FDFD, compare with EA."""
    print('='*60)
    print(f'  Moiré FDFD Test: θ={THETA_DEG:.2f}°')
    print(f'  L_moire = {L_MOIRE:.4f}a')
    print('='*60)

    # ── Step 1: Plot epsilon maps ──
    eps_smooth, info_sm, eps_no_smooth, info_no = plot_epsilon_comparison()

    # ── Step 2: FDFD at multiple resolutions ──
    print('\n' + '='*60)
    print('  Step 2: FDFD Eigensolves')
    print('='*60)

    n_modes = 80
    results = {}

    for res_name, res_px in [('low', 128), ('med', 256), ('high', 384)]:
        for smooth in [False, True]:
            label = f'{res_name}_{("smooth" if smooth else "nosmooth")}'
            print(f'\n--- {label}: {res_px}×{res_px}, smooth={smooth} ---')

            eps, info = build_moire_supercell_eps(
                'square', THETA_RAD, a=A,
                r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
                Nx=res_px, Ny=res_px,
                subpixel_smoothing=smooth, smoothing_Nsub=8,
            )

            freqs, evals = run_fdfd_moire(eps, info, n_modes=n_modes, label=label)
            results[label] = {
                'freqs': freqs, 'evals': evals,
                'Nx': res_px, 'smooth': smooth,
            }

            del eps
            gc.collect()

    # ── Step 3: Load EA reference ──
    print('\n' + '='*60)
    print('  Step 3: Load EA Reference')
    print('='*60)

    ea_path = SCRIPT_DIR / 'phase_a_diagnostic' / 'run_20260313_110637' / 'single_band' / 'ea_freqs.npz'
    if ea_path.exists():
        ea_data = np.load(ea_path)
        ea_freqs = np.sort(ea_data['freqs'])
        print(f'  EA single-band: {len(ea_freqs)} modes')
        print(f'  EA freq range: [{ea_freqs.min():.6f}, {ea_freqs.max():.6f}]')
        print(f'  EA BW: {ea_freqs.max() - ea_freqs.min():.6f}')
    else:
        print(f'  WARNING: EA data not found at {ea_path}')
        ea_freqs = None

    # Also load old commensurate FDFD for comparison
    old_fdfd_path = SCRIPT_DIR / 'overnight_validation' / 'run_20260313_004032' / '10deg' / 'fdfd_res64_k80.npz'
    if old_fdfd_path.exists():
        old_data = np.load(old_fdfd_path)
        old_fdfd_freqs = np.sort(old_data['freqs'])
        print(f'  Old FDFD (commensurate N=122): {len(old_fdfd_freqs)} modes')
        print(f'  Old FDFD range: [{old_fdfd_freqs.min():.6f}, {old_fdfd_freqs.max():.6f}]')
        print(f'  Old FDFD BW: {old_fdfd_freqs.max() - old_fdfd_freqs.min():.6f}')
    else:
        old_fdfd_freqs = None

    # ── Step 4: Comparison plot ──
    print('\n' + '='*60)
    print('  Step 4: Comparison')
    print('='*60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: frequency spectra
    ax = axes[0]
    y_offset = 0
    labels_used = []

    if ea_freqs is not None:
        ax.eventplot([ea_freqs], lineoffsets=y_offset, linelengths=0.8,
                     colors='blue', linewidths=0.5)
        labels_used.append(('EA (single-band, Γ)', 'blue', y_offset))
        y_offset += 1

    if old_fdfd_freqs is not None:
        ax.eventplot([old_fdfd_freqs], lineoffsets=y_offset, linelengths=0.8,
                     colors='gray', linewidths=0.5)
        labels_used.append(('FDFD commensurate (N=122, 4 moiré)', 'gray', y_offset))
        y_offset += 1

    colors = ['green', 'darkgreen', 'orange', 'darkorange', 'red', 'darkred']
    for i, (label, data) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.eventplot([data['freqs']], lineoffsets=y_offset, linelengths=0.8,
                     colors=c, linewidths=0.5)
        labels_used.append((f'FDFD moiré {label} ({data["Nx"]}px)', c, y_offset))
        y_offset += 1

    ax.set_xlabel('Frequency ω (c/a)')
    ax.set_yticks([l[2] for l in labels_used])
    ax.set_yticklabels([l[0] for l in labels_used], fontsize=9)
    ax.set_title(f'Eigenvalue spectra comparison — θ={THETA_DEG:.2f}°')
    ax.axvline(OMEGA0, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'ω₀={OMEGA0}')

    # Bottom: bandwidth convergence
    ax = axes[1]
    res_list = []
    bw_smooth = []
    bw_nosmooth = []
    for label, data in results.items():
        bw = data['freqs'].max() - data['freqs'].min()
        if data['smooth']:
            bw_smooth.append((data['Nx'], bw))
        else:
            bw_nosmooth.append((data['Nx'], bw))

    if bw_nosmooth:
        xs, ys = zip(*sorted(bw_nosmooth))
        ax.plot(xs, ys, 'o-', color='red', label='No smoothing')
    if bw_smooth:
        xs, ys = zip(*sorted(bw_smooth))
        ax.plot(xs, ys, 's-', color='green', label='Subpixel smoothed (8×8)')

    if ea_freqs is not None:
        ea_bw = ea_freqs.max() - ea_freqs.min()
        ax.axhline(ea_bw, color='blue', linestyle='--', label=f'EA BW = {ea_bw:.6f}')
    if old_fdfd_freqs is not None:
        old_bw = old_fdfd_freqs.max() - old_fdfd_freqs.min()
        ax.axhline(old_bw, color='gray', linestyle=':', label=f'Old FDFD BW = {old_bw:.6f}')

    ax.set_xlabel('FDFD resolution (pixels/moiré cell)')
    ax.set_ylabel('Bandwidth (c/a)')
    ax.set_title('Bandwidth convergence: FDFD resolution & smoothing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'moire_fdfd_comparison.png', dpi=200)
    plt.close(fig)
    print(f'  Saved moire_fdfd_comparison.png')

    # ── Summary table ──
    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    print(f'  {"Config":<30s} {"BW":>10s} {"Ratio vs EA":>12s}')
    print(f'  {"-"*52}')

    ea_bw = (ea_freqs.max() - ea_freqs.min()) if ea_freqs is not None else None

    if ea_freqs is not None:
        print(f'  {"EA single-band (Γ)":<30s} {ea_bw:>10.6f} {"1.000":>12s}')
    if old_fdfd_freqs is not None:
        old_bw = old_fdfd_freqs.max() - old_fdfd_freqs.min()
        ratio = old_bw / ea_bw if ea_bw else float('nan')
        print(f'  {"FDFD commensurate (N=122)":<30s} {old_bw:>10.6f} {ratio:>12.4f}')

    for label, data in results.items():
        bw = data['freqs'].max() - data['freqs'].min()
        ratio = bw / ea_bw if ea_bw else float('nan')
        print(f'  {label:<30s} {bw:>10.6f} {ratio:>12.4f}')

    # Save results
    np.savez(
        OUTPUT_DIR / 'moire_fdfd_results.npz',
        **{f'fdfd_{k}_freqs': v['freqs'] for k, v in results.items()},
        **{f'fdfd_{k}_Nx': v['Nx'] for k, v in results.items()},
        ea_freqs=ea_freqs if ea_freqs is not None else np.array([]),
        old_fdfd_freqs=old_fdfd_freqs if old_fdfd_freqs is not None else np.array([]),
        theta_deg=THETA_DEG, L_moire=L_MOIRE, omega0=OMEGA0,
    )
    print(f'\n  Saved moire_fdfd_results.npz')


if __name__ == '__main__':
    run_comparison()
