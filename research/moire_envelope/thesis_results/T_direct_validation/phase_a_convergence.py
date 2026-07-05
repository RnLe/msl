#!/usr/bin/env python3
"""
Phase A: FDFD convergence verification on (4,3) supercell vs MPB ground truth.

Geometry: twisted bilayer honeycomb PhC, (m,n) = (4,3), θ ≈ 9.43°
          N_cells = 37, 148 rods, r/a = 0.2, ε_rod = 11.56

Conventions: GEOMETRIC_CONVENTIONS.md
  - Coincidence supercell: C1 = n·a1 + m·a2, C2 = -m·a1 + (m+n)·a2     (§4)
  - MPB: size=(1,1), raw basis, radius = r_phys / |C1|, roll(N//2,N//2)  (§5)
  - FDFD grid: s = i/N, i=0..N-1, corner convention                      (§1)

Verifies:
  1. FDFD eigenvalues converge to MPB as resolution → ∞
  2. Convergence order (2nd for binary, higher for smoothed)
  3. Absolute error floor
"""
import numpy as np
import sys
import os
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import (
    build_monolayer_basis,
    get_sublattice_positions,
    rotation_matrix_2d,
)
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from T_direct_validation.subpixel_smoothing import build_smoothed_eps_supercell
from scipy.sparse.linalg import eigsh

out_dir = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
M, N_MN = 4, 3
EPS_BG = 1.0
EPS_ROD = 11.56
R_OVER_A = 0.2
a = 1.0

N_BANDS = 20
N_SUB = 16  # subpixel smoothing sub-grid

FDFD_RESOLUTIONS = [16, 32, 64, 128]
MPB_RESOLUTIONS = [32, 64]  # two MPB runs to verify internal convergence

# ════════════════════════════════════════════════════════════════
# Derived quantities
# ════════════════════════════════════════════════════════════════
theta_rad = commensurate_twist_angle('honeycomb', M, N_MN)
theta_deg = math.degrees(theta_rad)
N_cells = M**2 + M * N_MN + N_MN**2  # 37
EXPECTED_RODS = 4 * N_cells  # 148


# ════════════════════════════════════════════════════════════════
# Geometry: coincidence supercell (GEOMETRIC_CONVENTIONS.md §4)
# ════════════════════════════════════════════════════════════════
def build_coincidence_supercell(m, n, a_val=1.0):
    """C1 = n·a1 + m·a2,  C2 = -m·a1 + (m+n)·a2."""
    B_mono = build_monolayer_basis('honeycomb', a_val)
    a1, a2 = B_mono[:, 0], B_mono[:, 1]
    C1 = n * a1 + m * a2
    C2 = -m * a1 + (m + n) * a2
    return np.column_stack([C1, C2])


def enumerate_rods(m, n, a_val=1.0):
    """
    Enumerate all rod positions in the coincidence supercell.
    Returns list of fractional coords (f1, f2) in [0, 1).
    """
    B_mono = build_monolayer_basis('honeycomb', a_val)
    a1, a2 = B_mono[:, 0], B_mono[:, 1]
    B_coinc = build_coincidence_supercell(m, n, a_val)
    B_coinc_inv = np.linalg.inv(B_coinc)
    R = rotation_matrix_2d(commensurate_twist_angle('honeycomb', m, n))
    sublattice = get_sublattice_positions('honeycomb')
    N_c = m**2 + m * n + n**2

    rod_positions = []
    for B_layer in [B_mono, np.column_stack([R @ a1, R @ a2])]:
        for sub_frac in sublattice:
            sub_cart = B_layer @ sub_frac
            seen = set()
            N_scan = int(math.sqrt(N_c)) + 5
            for i1 in range(-N_scan, N_scan + 1):
                for i2 in range(-N_scan, N_scan + 1):
                    pos_cart = i1 * B_layer[:, 0] + i2 * B_layer[:, 1] + sub_cart
                    f = B_coinc_inv @ pos_cart
                    f_wrapped = f - np.floor(f)
                    key = (round(f_wrapped[0], 8), round(f_wrapped[1], 8))
                    if key[0] > 1.0 - 1e-6 or key[1] > 1.0 - 1e-6:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    rod_positions.append(f_wrapped)
    return rod_positions


def build_binary_eps(m, n, Nx, a_val=1.0):
    """Build binary ε on coincidence supercell grid (FDFD corner convention §1)."""
    B_mono = build_monolayer_basis('honeycomb', a_val)
    a1, a2 = B_mono[:, 0], B_mono[:, 1]
    B_coinc = build_coincidence_supercell(m, n, a_val)
    C1, C2 = B_coinc[:, 0], B_coinc[:, 1]
    R = rotation_matrix_2d(commensurate_twist_angle('honeycomb', m, n))
    sublattice = get_sublattice_positions('honeycomb')

    s = np.arange(Nx) / Nx  # corner convention: s = i/N ∈ [0,1)
    S1, S2 = np.meshgrid(s, s, indexing='ij')
    X = S1 * C1[0] + S2 * C2[0]
    Y = S1 * C1[1] + S2 * C2[1]
    XY = np.stack([X, Y], axis=0)

    eps = np.full((Nx, Nx), EPS_BG, dtype=np.float64)
    r_rod = R_OVER_A * a_val

    for B_layer in [B_mono, np.column_stack([R @ a1, R @ a2])]:
        B_inv = np.linalg.inv(B_layer)
        for sub_pos in sublattice:
            offset = B_layer @ sub_pos
            shifted = XY - offset[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_layer, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps[dist_sq < r_rod**2] = EPS_ROD

    return eps


def build_sc_info(m, n, Nx, a_val=1.0):
    """Build the info dict expected by build_fdfd_operator and build_smoothed_eps_supercell."""
    B_coinc = build_coincidence_supercell(m, n, a_val)
    B_mono = build_monolayer_basis('honeycomb', a_val)

    return {
        'lattice_type': 'honeycomb',
        'm': m, 'n': n, 'a': a_val,
        'theta_deg': math.degrees(commensurate_twist_angle('honeycomb', m, n)),
        'theta_rad': commensurate_twist_angle('honeycomb', m, n),
        'r_over_a': R_OVER_A,
        'eps_rod': EPS_ROD, 'eps_bg': EPS_BG,
        'N_cells': m**2 + m * n + n**2,
        'Nx': Nx, 'Ny': Nx,
        'B_super': B_coinc,
        'B_mono': B_mono,
        'L1': B_coinc[:, 0], 'L2': B_coinc[:, 1],
    }


# ════════════════════════════════════════════════════════════════
# MPB eigensolver (GEOMETRIC_CONVENTIONS.md §5)
# ════════════════════════════════════════════════════════════════
def run_mpb(m, n, res_per_ucell, n_bands):
    """
    MPB TM eigenfrequencies at Γ on the coincidence supercell.

    Convention §5: size=(1,1), raw basis, radius = r_phys / |C1|,
    fractional rod centers in [0,1), mpb_res = round(res * |C1|).
    """
    import meep as mp
    from meep import mpb

    B_coinc = build_coincidence_supercell(m, n, a)
    C1, C2 = B_coinc[:, 0], B_coinc[:, 1]
    C1_len = np.linalg.norm(C1)

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(C1[0], C1[1], 0),
        basis2=mp.Vector3(C2[0], C2[1], 0),
    )

    mpb_res = int(round(res_per_ucell * C1_len))
    rod_positions = enumerate_rods(m, n, a)
    n_rods = len(rod_positions)

    print(f"  MPB: {n_rods} rods (expected {EXPECTED_RODS}), mpb_res={mpb_res}")
    assert n_rods == EXPECTED_RODS, f"Rod count mismatch: {n_rods} != {EXPECTED_RODS}"

    geometry = [
        mp.Cylinder(
            radius=R_OVER_A * a / C1_len,   # §5.2
            center=mp.Vector3(pos[0], pos[1], 0),  # §5.3: fractional [0,1)
            material=mp.Medium(epsilon=EPS_ROD),
        )
        for pos in rod_positions
    ]

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=n_bands,
        resolution=mpb_res,
        k_points=[mp.Vector3(0, 0, 0)],
    )

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

    freqs = np.array(ms.all_freqs[0])  # single k-point

    # MPB frequencies are in c/|C1| units (lattice constant = |C1| when
    # basis vectors have non-unit magnitude). Convert to c/a units:
    freqs_ca = freqs / C1_len
    return np.sort(freqs_ca)


# ════════════════════════════════════════════════════════════════
# FDFD eigensolver wrapper
# ════════════════════════════════════════════════════════════════
def run_fdfd(m, n, res, n_modes, eps_type='binary'):
    """FDFD TM eigenvalues at Γ for the (m,n) coincidence supercell."""
    N_c = m**2 + m * n + n**2
    Nx = int(round(math.sqrt(N_c) * res))

    eps_bin = build_binary_eps(m, n, Nx, a)
    sc_info = build_sc_info(m, n, Nx, a)

    if eps_type == 'smoothed':
        eps_grid, smooth_diag = build_smoothed_eps_supercell(
            eps_bin, sc_info, n_sub=N_SUB,
            eps_rod=EPS_ROD, eps_bg=EPS_BG,
        )
        n_smoothed = smooth_diag['n_smoothed']
    else:
        eps_grid = eps_bin
        n_smoothed = 0

    # Sanity checks
    fill = np.mean(eps_grid > 0.5 * (EPS_BG + EPS_ROD))
    print(f"    {eps_type:8s}: Nx={Nx}, DOF={Nx*Nx:,}, fill={fill:.4f}, "
          f"<ε>={eps_grid.mean():.4f}, smoothed_px={n_smoothed}")

    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, sc_info, q_vec=np.zeros(2), polarization='tm')
    t_build = time.time() - t0

    t0 = time.time()
    evals, _ = eigsh(L_op, k=n_modes, sigma=0.01, which='LM',
                     maxiter=10000, tol=1e-12)
    t_solve = time.time() - t0

    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

    print(f"             build={t_build:.1f}s, solve={t_solve:.1f}s, "
          f"ω=[{freqs[0]:.6f}, {freqs[-1]:.6f}]")

    return np.sort(freqs)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    B_coinc = build_coincidence_supercell(M, N_MN, a)
    C1 = B_coinc[:, 0]
    C1_len = np.linalg.norm(C1)

    print(f"{'='*70}")
    print(f"PHASE A: FDFD CONVERGENCE VERIFICATION")
    print(f"(m,n) = ({M},{N_MN}), θ = {theta_deg:.4f}°, N_cells = {N_cells}")
    print(f"C1 = n·a1 + m·a2 = {C1}  |C1| = {C1_len:.4f}")
    print(f"Expected rods: {EXPECTED_RODS}")
    print(f"FDFD resolutions: {FDFD_RESOLUTIONS}")
    print(f"MPB resolutions: {MPB_RESOLUTIONS}")
    print(f"{'='*70}\n")

    # ── Step 1: MPB ground truth ──
    print(f"{'─'*50}")
    print("Step 1: MPB ground truth eigenvalues")
    print(f"{'─'*50}")
    t0 = time.time()

    mpb_by_res = {}
    for mpb_res in MPB_RESOLUTIONS:
        mpb_by_res[mpb_res] = run_mpb(M, N_MN, mpb_res, N_BANDS)
    t_mpb = time.time() - t0
    print(f"\n  MPB total time: {t_mpb:.1f}s")

    # Internal convergence check
    r_lo, r_hi = MPB_RESOLUTIONS[0], MPB_RESOLUTIONS[-1]
    mpb_drift = np.abs(mpb_by_res[r_hi] - mpb_by_res[r_lo])
    print(f"\n  MPB convergence (res={r_lo} → {r_hi}):")
    print(f"    max |Δω| = {np.max(mpb_drift):.2e}")
    print(f"    mean|Δω| = {np.mean(mpb_drift):.2e}")

    mpb_freqs = mpb_by_res[r_hi]
    print(f"\n  MPB eigenfrequencies (res={r_hi}, in c/a units):")
    for i, f in enumerate(mpb_freqs):
        print(f"    band {i+1:2d}: ω = {f:.8f}")

    # ── Step 1b: Epsilon sanity check (FDFD vs MPB geometry) ──
    print(f"\n{'─'*50}")
    print("Step 1b: Epsilon geometry sanity check")
    print(f"{'─'*50}")
    # Build FDFD eps at lowest resolution and compare statistics
    Nx_check = int(round(math.sqrt(N_cells) * FDFD_RESOLUTIONS[0]))
    eps_check = build_binary_eps(M, N_MN, Nx_check, a)
    fill_fdfd = np.mean(eps_check > 0.5 * (EPS_BG + EPS_ROD))
    mean_eps_fdfd = eps_check.mean()
    print(f"  FDFD binary (Nx={Nx_check}): fill={fill_fdfd:.4f}, <ε>={mean_eps_fdfd:.4f}")
    # Expected fill for bilayer honeycomb: ~0.496 (from eps_forensics)
    assert 0.40 < fill_fdfd < 0.60, f"Suspicious fill fraction: {fill_fdfd:.4f}"
    print(f"  Fill fraction within expected range [0.40, 0.60]: OK")

    # ── Step 2: FDFD at multiple resolutions ──
    print(f"\n{'─'*50}")
    print("Step 2: FDFD eigenvalues at multiple resolutions")
    print(f"{'─'*50}")

    fdfd_results = {}  # res -> {'binary': freqs, 'smoothed': freqs, 'Nx': Nx}
    for res in FDFD_RESOLUTIONS:
        Nx = int(round(math.sqrt(N_cells) * res))
        print(f"\n  res={res} (Nx={Nx}):")
        fb = run_fdfd(M, N_MN, res, N_BANDS, 'binary')
        fs = run_fdfd(M, N_MN, res, N_BANDS, 'smoothed')
        fdfd_results[res] = {'binary': fb, 'smoothed': fs, 'Nx': Nx}

    # ── Step 3: Convergence analysis ──
    print(f"\n\n{'='*70}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nMPB ground truth (res={r_hi}): ω = [{mpb_freqs[0]:.8f}, {mpb_freqs[-1]:.8f}]")

    errors = {}  # (label, res) -> per-band absolute errors

    for label in ['binary', 'smoothed']:
        print(f"\n--- {label.upper()} ε ---")
        print(f"{'res':>5s} {'Nx':>5s} {'mean|Δω|':>12s} {'max|Δω|':>12s} {'rate':>6s}")
        print(f"{'─'*45}")
        prev_mean = None

        for res in FDFD_RESOLUTIONS:
            fdfd_freqs = fdfd_results[res][label]
            Nx = fdfd_results[res]['Nx']
            err = np.abs(fdfd_freqs - mpb_freqs)
            errors[(label, res)] = err
            mean_err = np.mean(err)
            max_err = np.max(err)

            if prev_mean is not None and prev_mean > 0 and mean_err > 0:
                rate = np.log2(prev_mean / mean_err)
                print(f"{res:5d} {Nx:5d} {mean_err:12.2e} {max_err:12.2e} {rate:6.2f}")
            else:
                print(f"{res:5d} {Nx:5d} {mean_err:12.2e} {max_err:12.2e}    ---")
            prev_mean = mean_err

    # Per-band details at highest resolution
    best_res = FDFD_RESOLUTIONS[-1]
    print(f"\n--- Per-band error at res={best_res} ---")
    print(f"{'band':>5s} {'ω_MPB':>10s} {'|Δ|_bin':>12s} {'|Δ|_smo':>12s}")
    print(f"{'─'*45}")
    for i in range(N_BANDS):
        eb = errors[('binary', best_res)][i]
        es = errors[('smoothed', best_res)][i]
        print(f"{i+1:5d} {mpb_freqs[i]:10.6f} {eb:12.2e} {es:12.2e}")

    # ── Step 4: Convergence rate fit ──
    print(f"\n--- Convergence rate fit (log-log) ---")
    res_arr = np.array(FDFD_RESOLUTIONS, dtype=float)
    for label in ['binary', 'smoothed']:
        mean_errs = np.array([np.mean(errors[(label, r)]) for r in FDFD_RESOLUTIONS])
        valid = mean_errs > 0
        if valid.sum() >= 2:
            coeffs = np.polyfit(np.log(res_arr[valid]), np.log(mean_errs[valid]), 1)
            rate = -coeffs[0]
            print(f"  {label:8s}: p ≈ {rate:.2f}  (error ∝ res^{{-{rate:.1f}}})")

    # ── Step 5: Plots ──
    print(f"\n{'─'*50}")
    print("Step 5: Generating convergence plot")
    print(f"{'─'*50}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    COL_BIN = '#2563EB'
    COL_SMO = '#DC2626'
    COL_MPB = '#16A34A'

    # (a) Mean error vs resolution (log-log)
    ax = axes[0]
    mean_err_b = [np.mean(errors[('binary', r)]) for r in FDFD_RESOLUTIONS]
    mean_err_s = [np.mean(errors[('smoothed', r)]) for r in FDFD_RESOLUTIONS]
    ax.loglog(FDFD_RESOLUTIONS, mean_err_b, 's-', color=COL_BIN, ms=8, lw=2, label='Binary')
    ax.loglog(FDFD_RESOLUTIONS, mean_err_s, 'o-', color=COL_SMO, ms=8, lw=2, label='Smoothed')

    # Reference slopes
    h = 1.0 / res_arr
    ref_val = mean_err_b[0]
    ax.loglog(res_arr, ref_val * (h / h[0])**2, '--', color='gray', alpha=0.5, lw=1.5, label='O(h²)')
    ax.loglog(res_arr, ref_val * (h / h[0])**4, ':', color='gray', alpha=0.5, lw=1.5, label='O(h⁴)')
    ax.set_xlabel('Resolution (pixels/a)', fontsize=11)
    ax.set_ylabel('Mean |ω_FDFD − ω_MPB|', fontsize=11)
    ax.set_title('(a) Convergence: mean error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Per-band error at first and last resolution
    ax = axes[1]
    for res_idx, res in enumerate(FDFD_RESOLUTIONS):
        alpha = 0.3 + 0.7 * (res_idx / max(len(FDFD_RESOLUTIONS) - 1, 1))
        x = np.arange(1, N_BANDS + 1)
        show_label = (res_idx == 0 or res_idx == len(FDFD_RESOLUTIONS) - 1)
        ax.semilogy(x, errors[('binary', res)], 's-', ms=4 if show_label else 3,
                    lw=0.8, color=COL_BIN, alpha=alpha,
                    label=f'bin res={res}' if show_label else None)
        ax.semilogy(x, errors[('smoothed', res)], 'o-', ms=4 if show_label else 3,
                    lw=0.8, color=COL_SMO, alpha=alpha,
                    label=f'smo res={res}' if show_label else None)
    ax.set_xlabel('Band index', fontsize=11)
    ax.set_ylabel('|ω_FDFD − ω_MPB|', fontsize=11)
    ax.set_title('(b) Per-band error', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Spectrum comparison at highest resolution
    ax = axes[2]
    x = np.arange(1, N_BANDS + 1)
    ax.plot(x, mpb_freqs, 'x', ms=10, color=COL_MPB, label='MPB (ground truth)', zorder=3)
    ax.plot(x, fdfd_results[best_res]['binary'], 's', ms=6, color=COL_BIN,
            alpha=0.7, label=f'FDFD binary (res={best_res})')
    ax.plot(x, fdfd_results[best_res]['smoothed'], 'o', ms=6, color=COL_SMO,
            alpha=0.7, label=f'FDFD smoothed (res={best_res})')
    ax.set_xlabel('Band index', fontsize=11)
    ax.set_ylabel('Frequency ω (c/a)', fontsize=11)
    ax.set_title(f'(c) Eigenfrequencies at res={best_res}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'FDFD Convergence vs MPB  |  ({M},{N_MN}) supercell  |  θ = {theta_deg:.2f}°',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    fname = os.path.join(out_dir, 'phase_a_convergence.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved: {fname}")

    # Save data
    np.savez(
        os.path.join(out_dir, 'phase_a_convergence_data.npz'),
        mpb_freqs=mpb_freqs,
        mpb_freqs_lo=mpb_by_res[r_lo],
        fdfd_resolutions=np.array(FDFD_RESOLUTIONS),
        **{f'fdfd_{label}_res{res}': fdfd_results[res][label]
           for label in ['binary', 'smoothed'] for res in FDFD_RESOLUTIONS},
        **{f'err_{label}_res{res}': errors[(label, res)]
           for label in ['binary', 'smoothed'] for res in FDFD_RESOLUTIONS},
        m=M, n_mn=N_MN, N_cells=N_cells, theta_deg=theta_deg,
        mpb_res_hi=r_hi, n_bands=N_BANDS,
    )
    print(f"Saved: phase_a_convergence_data.npz")


if __name__ == '__main__':
    main()
