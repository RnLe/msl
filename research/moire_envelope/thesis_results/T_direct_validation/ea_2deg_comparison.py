#!/usr/bin/env python3
"""
EA vs FDFD comparison — Square (57,1), θ ≈ 2.01°
==================================================
Small-angle validation of the envelope approximation.
Higher resolution: 64×64 registry, MPB res=64, FDFD res=32/cell.

Phases (sequential):
  1. Phase 1 — registry sweep: 64×64, MPB res=64, 7×7 stencil, 10 bands
  2. FDFD — build supercell epsilon, CHOLMOD shift-invert, 50 modes
  3. EA — trace-regularized single-band (band 3), sweep mt values
  4. Plot and analysis

Usage:
    python ea_2deg_comparison.py                 # full run
    python ea_2deg_comparison.py --skip-fdfd     # skip FDFD (load from file)
    python ea_2deg_comparison.py --skip-phase1   # skip registry (load)
    python ea_2deg_comparison.py --plot-only      # just re-plot
    python ea_2deg_comparison.py --timing-only    # mini-batch timing estimates
"""

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Set threading BEFORE any numerical imports.
# MPB internal threading creates lock contention.
# ═══════════════════════════════════════════════════════════════
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import sys, argparse, time, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT / "phasesV3"))
sys.path.insert(0, str(MOIRE_ROOT))
sys.path.insert(0, str(SCRIPT_DIR.parent))

# ═══════════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════════

A         = 1.0
R_OVER_A  = 0.2
EPS_ROD   = 11.56
EPS_BG    = 1.0

M_IDX, N_IDX = 57, 1
N_CELLS   = M_IDX**2 + N_IDX**2   # = 3250
THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
THETA_DEG = np.degrees(THETA_RAD)

L1 = np.array([M_IDX, N_IDX], dtype=float) * A
L2 = np.array([-N_IDX, M_IDX], dtype=float) * A
L_SUPER = np.sqrt(L1 @ L1)   # = sqrt(3250) ≈ 57.009
B_SUPER = np.column_stack([L1, L2])
B_MONO  = np.eye(2) * A

OMEGA0       = 0.68457   # TM band 3 at M
SIGMA_FDFD   = (2 * np.pi * OMEGA0)**2
TARGET_BAND  = 3
N_MODES      = 50

# ═══════════════════════════════════════════════════════════════
#  Resolution parameters
# ═══════════════════════════════════════════════════════════════

# Phase 1: Registry sweep
REGISTRY_NR     = 64     # 64×64 registry
MPB_RES_PHASE1  = 64     # MPB resolution per registry point
FD_ORDER        = 6      # → 7×7 stencil
DK              = 0.06
N_ALL_BANDS     = 10     # bands 0-9 computed
N_WORKERS       = 16

# FDFD
RES_PER_CELL_FDFD = 32   # pixels per unit cell → NX = 1824
NX_FDFD = int(round(L_SUPER * RES_PER_CELL_FDFD))
DOF_FDFD = NX_FDFD ** 2

# EA
NS_EA = 128              # moiré envelope grid (upscaled from 64×64 registry)
dR_EA = L_SUPER / NS_EA
eta_EA = A / L_SUPER

OUTDIR = SCRIPT_DIR / "square_2deg"


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Registry sweep
# ═══════════════════════════════════════════════════════════════

def run_phase1(n_points=None):
    """
    Sweep 64×64 registry with MPB at res=64.
    If n_points is set, only compute that many (for timing).
    """
    from multiprocessing import Pool
    from phase1_mpb_v3 import _compute_single_registry_point

    NR = REGISTRY_NR
    step = 1.0 / NR
    params = {
        'lattice_type': 'square',
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_ROD,
        'k0': [0.5, 0.5],
        'dk': DK,
        'all_bands': list(range(N_ALL_BANDS)),
        'polarization': 'TM',
        'fd_order': FD_ORDER,
        'resolution': MPB_RES_PHASE1,
        'max_band': N_ALL_BANDS,
        'export_bloch_fields': False,
    }

    work = []
    for ix in range(NR):
        for iy in range(NR):
            delta_frac = np.array([ix * step, iy * step])
            work.append((ix, iy, delta_frac, params))

    if n_points is not None:
        work = work[:n_points]

    n_stencil = FD_ORDER + 1
    omega0_reg = np.full((NR, NR, N_ALL_BANDS), np.nan)
    vg_reg = np.full((NR, NR, N_ALL_BANDS, 2), np.nan)
    Minv_reg = np.full((NR, NR, N_ALL_BANDS, 2, 2), np.nan)

    checkpoint_file = OUTDIR / 'ea_registry_2deg.npz'
    checkpoint_interval = max(1, len(work) // 20)  # 5%

    t0 = time.time()
    done = 0
    with Pool(processes=N_WORKERS) as pool:
        for ix, iy, result in pool.imap_unordered(
                _compute_single_registry_point, work, chunksize=4):
            omega0_reg[ix, iy] = result['omega0']
            vg_reg[ix, iy] = result['vg']
            Minv_reg[ix, iy] = result['M_inv']
            done += 1
            if done % checkpoint_interval == 0 or done == len(work):
                pct = 100 * done / len(work)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (len(work) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(work)} ({pct:.0f}%) "
                      f"rate={rate:.1f} pts/s, ETA={eta_s:.0f}s", flush=True)
                if n_points is None:
                    np.savez(checkpoint_file,
                             omega0=omega0_reg, vg=vg_reg, M_inv=Minv_reg)

    dt = time.time() - t0
    print(f"  Phase 1 done: {done} points in {dt:.1f}s ({done/dt:.1f} pts/s)")

    if n_points is None:
        np.savez(checkpoint_file,
                 omega0=omega0_reg, vg=vg_reg, M_inv=Minv_reg)
        print(f"  Saved {checkpoint_file}")

    return omega0_reg, vg_reg, Minv_reg, dt


# ═══════════════════════════════════════════════════════════════
#  FDFD supercell
# ═══════════════════════════════════════════════════════════════

def build_epsilon():
    """Build bilayer supercell epsilon grid."""
    from T_direct_validation.supercell_geometry import build_supercell_eps

    t0 = time.time()
    eps, info = build_supercell_eps(
        'square', M_IDX, N_IDX, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=NX_FDFD, Ny=NX_FDFD,
    )
    dt = time.time() - t0
    rod_frac = (eps > (EPS_BG + 0.1)).mean()
    print(f"  ε grid: {eps.shape}, build={dt:.1f}s, "
          f"rod frac={rod_frac:.4f} (expect ≈{2*np.pi*R_OVER_A**2:.4f})")
    return eps, info


def run_fdfd(eps_grid, info, timing_only=False):
    """FDFD eigensolver with CHOLMOD shift-invert near ω₀²."""
    import scipy.sparse as sp
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    from scipy.sparse.linalg import eigsh

    q_vec = np.zeros(2)
    print(f"  FDFD: NX={NX_FDFD}, DOF={DOF_FDFD:,}, "
          f"σ={SIGMA_FDFD:.4f}, {N_MODES} modes")

    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, info, q_vec=q_vec, polarization='tm')
    t_build = time.time() - t0
    print(f"  Operator built: nnz={L_op.nnz:,}, time={t_build:.1f}s")

    if timing_only:
        return None, None, t_build

    try:
        from sksparse.cholmod import cholesky
        t0 = time.time()
        shifted = L_op - SIGMA_FDFD * sp.eye(L_op.shape[0],
                                               format='csc', dtype=L_op.dtype)
        factor = cholesky(shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        print(f"  CHOLMOD factorization: {t_factor:.1f}s")

        from scipy.sparse.linalg import LinearOperator
        OPinv = LinearOperator(L_op.shape,
                               matvec=lambda x: factor(x),
                               dtype=L_op.dtype)
        t0 = time.time()
        evals, evecs = eigsh(L_op, k=N_MODES, sigma=SIGMA_FDFD, which='LM',
                             OPinv=OPinv, maxiter=10000, tol=1e-10)
        t_eig = time.time() - t0
        print(f"  eigsh: {t_eig:.1f}s")
        total_t = t_build + t_factor + t_eig
    except ImportError:
        print("  CHOLMOD not available, using scipy LU")
        t0 = time.time()
        evals, evecs = eigsh(L_op, k=N_MODES, sigma=SIGMA_FDFD, which='LM',
                             maxiter=10000, tol=1e-10)
        total_t = time.time() - t0 + t_build

    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    print(f"  ω range: [{freqs[0]:.6f}, {freqs[-1]:.6f}]")

    return freqs, evals, total_t


# ═══════════════════════════════════════════════════════════════
#  EA solve — single-band with trace regularization
# ═══════════════════════════════════════════════════════════════

def compute_delta_grid():
    """Map moiré grid to registry fractional δ-coordinates."""
    R_mat = np.array([[np.cos(THETA_RAD), -np.sin(THETA_RAD)],
                       [np.sin(THETA_RAD),  np.cos(THETA_RAD)]])
    Ns = NS_EA
    s1 = np.arange(Ns) / Ns
    s2 = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    disp = ((R_mat - np.eye(2)) @ pos.T).T
    return disp - np.floor(disp)   # (Ns², 2)


def interp_periodic(data_2d, pts, NR):
    """Interpolate periodic 2D field from registry grid."""
    reg_ax = np.linspace(0, 1, NR, endpoint=False)
    ext = np.concatenate([reg_ax, [1.0]])
    padded = np.concatenate([data_2d, data_2d[:1, :]], axis=0)
    padded = np.concatenate([padded, padded[:, :1]], axis=1)
    f = RegularGridInterpolator((ext, ext), padded,
                                method='linear', bounds_error=False,
                                fill_value=None)
    return f(pts)


def clamp_trace(M_inv, max_trace):
    """Scale M⁻¹ at each point so |Tr| ≤ max_trace."""
    M_out = M_inv.copy()
    tr = M_out[:, :, 0, 0, 0, 0] + M_out[:, :, 0, 0, 1, 1]
    mask = np.abs(tr) > max_trace
    if np.any(mask):
        scale = max_trace / np.abs(tr[mask])
        M_out[mask, 0, 0, :, :] *= scale[:, None, None]
        n = np.count_nonzero(mask)
        total = M_inv.shape[0] * M_inv.shape[1]
        print(f"    Trace-clamped {n}/{total} ({100*n/total:.1f}%) "
              f"to |Tr|≤{max_trace}")
    return M_out


def run_ea(omega0_reg, vg_reg, Minv_reg, max_trace=2.0,
           include_kinetic=True, include_drift=True):
    """Single-band EA solve with trace-clamped M⁻¹."""
    from phase3_mpb_v3 import (assemble_multiband_hamiltonian,
                                solve_multiband_envelope)

    NR = omega0_reg.shape[0]
    Ns = NS_EA
    pts = compute_delta_grid()
    b = TARGET_BAND

    V_m = interp_periodic(omega0_reg[:, :, b], pts, NR).reshape(Ns, Ns) - OMEGA0
    vgx = interp_periodic(vg_reg[:, :, b, 0], pts, NR).reshape(Ns, Ns)
    vgy = interp_periodic(vg_reg[:, :, b, 1], pts, NR).reshape(Ns, Ns)
    Mxx = interp_periodic(Minv_reg[:, :, b, 0, 0], pts, NR).reshape(Ns, Ns)
    Mxy = interp_periodic(Minv_reg[:, :, b, 0, 1], pts, NR).reshape(Ns, Ns)
    Myy = interp_periodic(Minv_reg[:, :, b, 1, 1], pts, NR).reshape(Ns, Ns)

    Lambda = V_m.reshape(Ns, Ns, 1, 1)
    v_drift = np.zeros((Ns, Ns, 1, 1, 2))
    v_drift[:, :, 0, 0, 0] = vgx
    v_drift[:, :, 0, 0, 1] = vgy
    M_inv = np.zeros((Ns, Ns, 1, 1, 2, 2))
    M_inv[:, :, 0, 0, 0, 0] = Mxx
    M_inv[:, :, 0, 0, 0, 1] = Mxy
    M_inv[:, :, 0, 0, 1, 0] = Mxy
    M_inv[:, :, 0, 0, 1, 1] = Myy
    A_berry = np.zeros((Ns, Ns, 1, 1, 2))
    Phi_BH = np.zeros((Ns, Ns, 1, 1))

    if max_trace is not None and max_trace > 0 and include_kinetic:
        M_inv = clamp_trace(M_inv, max_trace)
    if not include_kinetic:
        M_inv[:] = 0

    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        eta_EA, Ns, Ns, 1, dR_EA, dR_EA, B_SUPER,
        include_drift=include_drift, include_kinetic=include_kinetic,
        include_born_huang=False, order=4)

    evals, evecs = solve_multiband_envelope(H, N_MODES, sigma=0.0)
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    freqs = np.sort(OMEGA0 + evals)
    return freqs, V_m


# ═══════════════════════════════════════════════════════════════
#  Timing estimates
# ═══════════════════════════════════════════════════════════════

def run_timing_estimates():
    """Run mini-batches to estimate wall time for each phase."""
    print("\n" + "=" * 70)
    print("  TIMING ESTIMATES")
    print("=" * 70)

    # Phase 1: time 16 registry points
    print("\n  Phase 1: Registry sweep (16 points at 64×64 res=64)...")
    _, _, _, dt1 = run_phase1(n_points=16)
    rate1 = 16 / dt1
    total_pts = REGISTRY_NR ** 2
    est_phase1 = total_pts / rate1
    print(f"    Rate: {rate1:.1f} pts/s → estimated {total_pts} pts: "
          f"{est_phase1:.0f}s ({est_phase1/60:.1f}min)")

    # FDFD: build operator only
    print("\n  FDFD: Building operator (no solve)...")
    eps, info = build_epsilon()
    _, _, dt_build = run_fdfd(eps, info, timing_only=True)
    # CHOLMOD factor ≈ 3-5× build time for 3.3M DOF
    est_fdfd = dt_build * 10  # rough
    print(f"    Build: {dt_build:.1f}s → estimated total: ~{est_fdfd:.0f}s "
          f"({est_fdfd/60:.1f}min)")
    del eps; gc.collect()

    # EA: always fast
    est_ea = 15  # 128×128 envelope solve
    print(f"\n  EA solve: ~{est_ea}s (128×128, 5 configs)")

    total = est_phase1 + est_fdfd + est_ea
    print(f"\n  {'─'*50}")
    print(f"  TOTAL ESTIMATED: {total:.0f}s ({total/60:.1f}min)")
    print(f"    Phase 1 registry: ~{est_phase1:.0f}s ({est_phase1/60:.1f}min)")
    print(f"    FDFD solve:       ~{est_fdfd:.0f}s ({est_fdfd/60:.1f}min)")
    print(f"    EA sweep:         ~{est_ea}s")
    print(f"  {'─'*50}")

    return est_phase1, est_fdfd, est_ea


# ═══════════════════════════════════════════════════════════════
#  Plot & analysis
# ═══════════════════════════════════════════════════════════════

def plot_and_analyze(freqs_fdfd, results, V_m):
    """Generate 4-panel comparison plot and print summary."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Level diagram — best config
    ax = axes[0, 0]
    best_name = 'mt=2.0'
    freqs_best = results[best_name]
    win = 0.04
    for label, freqs, color, x in [
            ('FDFD', freqs_fdfd, '#d62728', 0.25),
            (f'EA ({best_name})', freqs_best, '#2ca02c', 0.75)]:
        mask = np.abs(freqs - OMEGA0) < win
        f = freqs[mask]
        ax.hlines(f, x - 0.12, x + 0.12, color=color, lw=1.0)
        ax.text(x, OMEGA0 + win * 0.9, label, ha='center',
                fontsize=10, color=color, fontweight='bold')
    ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel(r'$\omega\, (a / 2\pi c)$')
    ax.set_title('Eigenvalue Level Diagram')
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(OMEGA0 - win, OMEGA0 + win)

    # Panel 2: Sorted eigenvalue errors
    ax = axes[0, 1]
    for name, color, ls in [
            ('V-only', '#9467bd', '--'),
            ('Full (raw)', '#ff7f0e', ':'),
            ('mt=3.0', '#d62728', '-.'),
            ('mt=2.0', '#2ca02c', '-'),
            ('mt=1.0', '#1f77b4', '-.')]:
        if name in results:
            diff = (results[name] - freqs_fdfd) * 1000
            ax.plot(range(N_MODES), diff, color=color, ls=ls, lw=1.2,
                    label=name)
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel('Sorted eigenvalue index')
    ax.set_ylabel(r'$(\omega_{\rm EA} - \omega_{\rm FDFD}) \times 10^3$')
    ax.set_title('EA − FDFD Eigenvalue Errors')
    ax.legend(fontsize=8)

    # Panel 3: Moiré potential
    ax = axes[1, 0]
    im = ax.imshow(V_m.T * 1000, origin='lower', cmap='coolwarm',
                    extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label=r'$V_3(R) \times 10^3$')
    ax.set_title(f'Band {TARGET_BAND} Potential V(R)')
    ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$')

    # Panel 4: RMS & bandwidth vs regularization
    ax = axes[1, 1]
    sweep_names = ['Full (raw)', 'mt=5.0', 'mt=3.0', 'mt=2.0', 'mt=1.0', 'mt=0.5']
    sweep_names = [n for n in sweep_names if n in results]
    mt_labels = [n.replace('Full (raw)', 'raw') for n in sweep_names]
    bws = [(results[n][-1] - results[n][0]) * 1000 for n in sweep_names]
    rmss = [np.sqrt(np.mean((results[n] - freqs_fdfd)**2)) * 1000
            for n in sweep_names]
    x = range(len(mt_labels))
    ax.plot(x, bws, 'bs-', label='EA bandwidth', ms=6)
    ax.plot(x, rmss, 'ro-', label='RMS error', ms=6)
    bw_fdfd = (freqs_fdfd[-1] - freqs_fdfd[0]) * 1000
    ax.axhline(bw_fdfd, color='b', ls='--', lw=1,
               label=f'FDFD bw={bw_fdfd:.1f}')
    ax.set_xticks(x)
    ax.set_xticklabels(mt_labels)
    ax.set_xlabel(r'$M^{-1}_{\rm max\,trace}$ regularization')
    ax.set_ylabel(r'$\times 10^{-3}$')
    ax.set_title('Accuracy vs Regularization')
    ax.legend(fontsize=8)

    fig.suptitle(
        f'EA vs FDFD — Square ({M_IDX},{N_IDX}): '
        f'θ={THETA_DEG:.2f}°, N={N_CELLS}, '
        f'band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUTDIR / 'fig_ea_vs_fdfd_2deg.png'
    fig.savefig(out, dpi=200)
    print(f"\n  Saved {out}")
    plt.close()

    # Summary table
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — {N_MODES} sorted eigenvalues")
    print(f"{'='*70}")
    bw_fdfd = (freqs_fdfd[-1] - freqs_fdfd[0]) * 1000
    print(f"\n  FDFD: [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}], "
          f"bw={bw_fdfd:.1f}×10⁻³")

    for name in ['V-only', 'Full (raw)', 'mt=5.0', 'mt=3.0',
                  'mt=2.0', 'mt=1.0', 'mt=0.5']:
        if name not in results:
            continue
        f = results[name]
        bw = (f[-1] - f[0]) * 1000
        diff = f - freqs_fdfd
        rms = np.sqrt(np.mean(diff**2)) * 1000
        mx = np.max(np.abs(diff)) * 1000
        print(f"  {name:>12s}: bw={bw:6.1f}  RMS={rms:5.2f}  max={mx:5.2f}")

    # Detailed comparison for best
    best = 'mt=2.0'
    if best in results:
        f_e = results[best]
        diff = f_e - freqs_fdfd
        print(f"\n  Detailed: {best}")
        print(f"  {'idx':>3s}  {'FDFD':>10s}  {'EA':>10s}  {'Δω×10³':>8s}")
        for i in range(N_MODES):
            print(f"  {i:3d}  {freqs_fdfd[i]:.6f}  {f_e[i]:.6f}  "
                  f"{diff[i]*1000:+.3f}")

    # Monolayer M_inv diagnostic
    tr_mono = Minv_reg_global[0, 0, TARGET_BAND, 0, 0] + \
              Minv_reg_global[0, 0, TARGET_BAND, 1, 1]
    print(f"\n  Monolayer Tr(M⁻¹) at δ=0: {tr_mono:.4f}")

    # Save results
    save_dict = {'freqs_fdfd': freqs_fdfd, 'V_moire': V_m}
    for name, f in results.items():
        key = name.replace(' ', '_').replace('(', '').replace(')', '')
        save_dict[f'freqs_ea_{key}'] = f
    np.savez(OUTDIR / 'ea_2deg_results.npz', **save_dict)
    print(f"  Saved results to {OUTDIR / 'ea_2deg_results.npz'}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

# Global for diagnostics
Minv_reg_global = None


def main():
    global Minv_reg_global

    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fdfd', action='store_true')
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print(f"  EA vs FDFD — Square ({M_IDX},{N_IDX}), θ={THETA_DEG:.2f}°")
    print(f"  N_cells={N_CELLS}, L_super={L_SUPER:.3f}")
    print(f"  Target: TM band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}")
    print(f"  Phase 1: {REGISTRY_NR}×{REGISTRY_NR} registry, "
          f"MPB res={MPB_RES_PHASE1}, {N_WORKERS} workers")
    print(f"  FDFD: {RES_PER_CELL_FDFD} px/cell → NX={NX_FDFD}, "
          f"DOF={DOF_FDFD:,}")
    print(f"  EA: {NS_EA}×{NS_EA} envelope grid, {N_MODES} modes")
    print("=" * 70)

    # ── Timing only ──
    if args.timing_only:
        run_timing_estimates()
        return

    # ── Plot only ──
    if args.plot_only:
        f_res = OUTDIR / 'ea_2deg_results.npz'
        if not f_res.exists():
            print("No results file found.")
            return
        d = np.load(f_res)
        freqs_fdfd = d['freqs_fdfd']
        results = {}
        for key in d.files:
            if key.startswith('freqs_ea_'):
                name = key[len('freqs_ea_'):].replace('_', ' ')
                # Reconstruct name
                name = name.replace('mt ', 'mt=').replace('Full raw', 'Full (raw)')
                name = name.replace('V only', 'V-only')
                results[name] = d[key]
        V_m = d.get('V_moire')
        # Load registry for diagnostics
        reg_file = OUTDIR / 'ea_registry_2deg.npz'
        if reg_file.exists():
            rd = np.load(reg_file)
            Minv_reg_global = rd['M_inv']
        else:
            Minv_reg_global = np.zeros((1, 1, N_ALL_BANDS, 2, 2))
        plot_and_analyze(freqs_fdfd, results, V_m)
        return

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Registry sweep
    # ═══════════════════════════════════════════════════════════
    reg_file = OUTDIR / 'ea_registry_2deg.npz'
    if args.skip_phase1 and reg_file.exists():
        print("\n1. Loading saved registry data...")
        d = np.load(reg_file)
        omega0_reg = d['omega0']
        vg_reg = d['vg']
        Minv_reg = d['M_inv']
        n_valid = np.isfinite(omega0_reg[:, :, 0]).sum()
        print(f"   {n_valid}/{REGISTRY_NR**2} valid points")
    else:
        print("\n1. Phase 1: Registry sweep...")
        omega0_reg, vg_reg, Minv_reg, dt1 = run_phase1()
    Minv_reg_global = Minv_reg

    # ═══════════════════════════════════════════════════════════
    # FDFD supercell
    # ═══════════════════════════════════════════════════════════
    fdfd_file = OUTDIR / 'fdfd_supercell_2deg.npz'
    if args.skip_fdfd and fdfd_file.exists():
        print("\n2. Loading saved FDFD data...")
        d = np.load(fdfd_file)
        freqs_fdfd = d['freqs']
    else:
        print("\n2. FDFD supercell solve...")
        eps, info = build_epsilon()

        # Save epsilon plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(eps.T, origin='lower', cmap='RdBu_r', extent=[0, 1, 0, 1])
        ax.set_title(f'ε(s₁,s₂) — ({M_IDX},{N_IDX}), θ={THETA_DEG:.2f}°')
        fig.tight_layout()
        fig.savefig(OUTDIR / 'eps_supercell_2deg.png', dpi=100)
        plt.close()

        freqs_fdfd, evals_fdfd, dt_fdfd = run_fdfd(eps, info)
        np.savez(fdfd_file, freqs=freqs_fdfd, evals=evals_fdfd,
                 wall_time=dt_fdfd)
        print(f"  Saved {fdfd_file}")
        del eps; gc.collect()

    freqs_fdfd = np.sort(freqs_fdfd)
    print(f"  FDFD: {len(freqs_fdfd)} modes, "
          f"[{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}], "
          f"bw={(freqs_fdfd[-1]-freqs_fdfd[0])*1000:.1f}×10⁻³")

    # ═══════════════════════════════════════════════════════════
    # EA sweep
    # ═══════════════════════════════════════════════════════════
    print("\n3. EA regularization sweep...")
    configs = [
        ('V-only',       None,  False, False),
        ('Full (raw)',   None,  True,  True),
        ('mt=5.0',       5.0,   True,  True),
        ('mt=3.0',       3.0,   True,  True),
        ('mt=2.0',       2.0,   True,  True),
        ('mt=1.0',       1.0,   True,  True),
        ('mt=0.5',       0.5,   True,  True),
    ]

    results = {}
    V_m_saved = None
    for name, mt, inc_drift, inc_kinetic in configs:
        print(f"\n  --- {name} ---")
        freqs, V_m = run_ea(omega0_reg, vg_reg, Minv_reg,
                            max_trace=mt,
                            include_kinetic=inc_kinetic,
                            include_drift=inc_drift)
        results[name] = freqs
        if V_m_saved is None:
            V_m_saved = V_m

        bw = (freqs[-1] - freqs[0]) * 1000
        diff = freqs - freqs_fdfd
        rms = np.sqrt(np.mean(diff**2)) * 1000
        print(f"    bw={bw:.1f}×10⁻³, RMS={rms:.2f}×10⁻³")

    # ═══════════════════════════════════════════════════════════
    # Plot and analysis
    # ═══════════════════════════════════════════════════════════
    print("\n4. Generating plots and analysis...")
    plot_and_analyze(freqs_fdfd, results, V_m_saved)

    dt_total = time.time() - t_total
    print(f"\nTotal time: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == '__main__':
    main()
