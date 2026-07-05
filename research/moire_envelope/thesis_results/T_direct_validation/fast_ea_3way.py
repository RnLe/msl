#!/usr/bin/env python3
"""
Fast EA vs FDFD comparison for the (11,1) square supercell (θ≈10.39°).

Phase 1: 32×32 registry sweep at MPB resolution 32 (quick), dk=0.06, fd_order=6
         → saves checkpoint every 5% of registry points
Phase 2: (skipped — no Bloch fields for Berry/BH, single-band diagonal EA)
Phase 3: Interpolate registry → 128×128 moiré grid, assemble H, diag
Compare: Load FDFD data from previous run, plot EA vs FDFD

Usage:
    python fast_ea_3way.py                # full run
    python fast_ea_3way.py --resume       # resume from checkpoint
    python fast_ea_3way.py --skip-phase1  # skip phase1 (load saved registry)
    python fast_ea_3way.py --plot-only    # just plot from saved data
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

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(THESIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# ═══════════════════════════════════════════════════════════════
#  Physical parameters (identical to square_supercell_3way.py)
# ═══════════════════════════════════════════════════════════════

A         = 1.0
R_OVER_A  = 0.2
EPS_ROD   = 11.56
EPS_BG    = 1.0
M_IDX, N_IDX = 11, 1
N_CELLS   = M_IDX**2 + N_IDX**2   # 122
THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
THETA_DEG = np.degrees(THETA_RAD)

L1 = np.array([M_IDX, N_IDX], dtype=float) * A
L2 = np.array([-N_IDX, M_IDX], dtype=float) * A
L_SUPER = np.sqrt(L1 @ L1)
B_SUPER = np.column_stack([L1, L2])
B_MONO  = np.eye(2) * A

OMEGA0    = 0.68457
TARGET_BAND = 3
N_BANDS_MPB = 8
N_MODES     = 50

# Fast settings
MPB_RES      = 32       # resolution per monolayer cell (fast!)
REGISTRY_NR  = 32       # registry grid
NS_EA        = 128      # moiré spatial grid (upscaled from 32×32 registry)
DK           = 0.06
FD_ORDER     = 6
N_STENCIL    = 7
N_WORKERS    = 16

OUTDIR = SCRIPT_DIR / "square_3way"

# ═══════════════════════════════════════════════════════════════
#  Phase 1: Resumable registry sweep
# ═══════════════════════════════════════════════════════════════

def run_phase1_registry(resume=False):
    """32×32 registry sweep at MPB res=32. Saves checkpoint every 5%."""
    from multiprocessing import Pool
    from phase1_mpb_v3 import _compute_single_registry_point

    checkpoint_file = OUTDIR / 'ea_registry_fast_checkpoint.npz'
    final_file = OUTDIR / 'ea_registry_fast.npz'

    NR = REGISTRY_NR
    total = NR * NR
    checkpoint_interval = max(1, total // 20)  # every 5%

    # Initialise or resume arrays
    omega0_reg = np.full((NR, NR, N_BANDS_MPB), np.nan)
    vg_reg = np.full((NR, NR, N_BANDS_MPB, 2), np.nan)
    Minv_reg = np.full((NR, NR, N_BANDS_MPB, 2, 2), np.nan)
    stencil_omega_reg = np.full((NR, NR, N_BANDS_MPB, N_STENCIL, N_STENCIL), np.nan)
    done_mask = np.zeros((NR, NR), dtype=bool)

    if resume and checkpoint_file.exists():
        d = np.load(checkpoint_file)
        omega0_reg = d['omega0']
        vg_reg = d['vg']
        Minv_reg = d['M_inv']
        stencil_omega_reg = d['stencil_omega']
        done_mask = np.isfinite(omega0_reg[:, :, 0])
        n_done = done_mask.sum()
        print(f"  Resuming from checkpoint: {n_done}/{total} points done")
    else:
        n_done = 0
        print(f"  Starting fresh Phase 1: {NR}×{NR} registry, "
              f"MPB res={MPB_RES}, dk={DK}, fd_order={FD_ORDER}")

    params = {
        'lattice_type': 'square',
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_ROD,
        'k0': [0.5, 0.5],
        'dk': DK,
        'all_bands': list(range(N_BANDS_MPB)),
        'polarization': 'TM',
        'fd_order': FD_ORDER,
        'resolution': MPB_RES,
        'max_band': N_BANDS_MPB,
        'export_bloch_fields': False,
    }

    step = 1.0 / NR
    work = []
    for ix in range(NR):
        for iy in range(NR):
            if not done_mask[ix, iy]:
                delta_frac = np.array([ix * step, iy * step])
                work.append((ix, iy, delta_frac, params))

    if not work:
        print("  All registry points already computed!")
        return omega0_reg, vg_reg, Minv_reg, stencil_omega_reg

    print(f"  {len(work)} points remaining. Using {N_WORKERS} workers.")

    t0 = time.time()
    completed_since_save = 0

    def save_checkpoint():
        np.savez(checkpoint_file,
                 omega0=omega0_reg, vg=vg_reg, M_inv=Minv_reg,
                 stencil_omega=stencil_omega_reg)

    with Pool(processes=N_WORKERS) as pool:
        for ix, iy, result in pool.imap_unordered(
                _compute_single_registry_point, work, chunksize=4):
            omega0_reg[ix, iy] = result['omega0']
            vg_reg[ix, iy] = result['vg']
            Minv_reg[ix, iy] = result['M_inv']
            stencil_omega_reg[ix, iy] = result['omega_stencil']

            n_done += 1
            completed_since_save += 1

            if completed_since_save >= checkpoint_interval:
                save_checkpoint()
                elapsed = time.time() - t0
                pct = n_done / total * 100
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"    [{pct:5.1f}%] {n_done}/{total} done "
                      f"({rate:.1f} pts/s, checkpoint saved)", flush=True)
                completed_since_save = 0

    dt = time.time() - t0
    print(f"  Phase 1 complete: {total} pts in {dt:.0f}s ({dt/60:.1f}min)")

    # Save final result
    np.savez(final_file,
             omega0=omega0_reg, vg=vg_reg, M_inv=Minv_reg,
             stencil_omega=stencil_omega_reg)
    print(f"  Saved: {final_file}")

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return omega0_reg, vg_reg, Minv_reg, stencil_omega_reg


# ═══════════════════════════════════════════════════════════════
#  Phase 3: EA solve (single-band, upscaled to 128×128)
# ═══════════════════════════════════════════════════════════════

def run_ea_solve(omega0_reg, Minv_reg):
    """Interpolate 32×32 registry to 128×128 moiré grid, assemble H, solve."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from scipy.interpolate import RegularGridInterpolator

    Ns = NS_EA
    NR = REGISTRY_NR
    print(f"\n  EA Phase 3: {NR}×{NR} registry → {Ns}×{Ns} moiré grid, "
          f"band {TARGET_BAND}")

    # ── 1. Moiré grid → registry shifts ──
    R_mat = np.array([
        [np.cos(THETA_RAD), -np.sin(THETA_RAD)],
        [np.sin(THETA_RAD),  np.cos(THETA_RAD)],
    ])
    B_mono_inv = np.linalg.inv(B_MONO)

    s1 = np.arange(Ns) / Ns
    s2 = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]
    pos = np.stack([X, Y], axis=-1).reshape(-1, 2)

    disp = ((R_mat - np.eye(2)) @ pos.T).T
    delta_frac = (B_mono_inv @ disp.T).T
    delta_frac = delta_frac - np.floor(delta_frac)
    delta_frac = delta_frac.reshape(Ns, Ns, 2)

    # ── 2. Interpolate registry → moiré ──
    reg_ax = np.linspace(0, 1, NR, endpoint=False)
    reg_ax_ext = np.concatenate([reg_ax, [1.0]])

    def pad_periodic(arr):
        p = np.concatenate([arr, arr[:1, :]], axis=0)
        return np.concatenate([p, p[:, :1]], axis=1)

    omega0_b = omega0_reg[:, :, TARGET_BAND]
    Minv_b = Minv_reg[:, :, TARGET_BAND, :, :]

    omega_moire = RegularGridInterpolator(
        (reg_ax_ext, reg_ax_ext), pad_periodic(omega0_b),
        method='linear', bounds_error=False, fill_value=None
    )(delta_frac.reshape(-1, 2)).reshape(Ns, Ns)

    omega_ref = np.mean(omega_moire)
    V_moire = omega_moire - omega_ref

    Minv_moire = np.zeros((Ns, Ns, 2, 2))
    for i in range(2):
        for j in range(2):
            Minv_moire[:, :, i, j] = RegularGridInterpolator(
                (reg_ax_ext, reg_ax_ext), pad_periodic(Minv_b[:, :, i, j]),
                method='linear', bounds_error=False, fill_value=None
            )(delta_frac.reshape(-1, 2)).reshape(Ns, Ns)

    print(f"  ω_ref = {omega_ref:.6f}")
    print(f"  V range: [{V_moire.min():.6f}, {V_moire.max():.6f}]")
    tr = Minv_moire[:, :, 0, 0] + Minv_moire[:, :, 1, 1]
    print(f"  Tr(M⁻¹) range: [{tr.min():.4f}, {tr.max():.4f}]")

    # ── 3. Build Hamiltonian ──
    dR = L_SUPER / Ns

    from phase3_mpb_v3 import (build_periodic_laplacian_matrix as build_lap,
                                build_periodic_derivative_matrix as build_deriv)

    L1_mat = build_lap(Ns, dR, order=4)
    L2_mat = build_lap(Ns, dR, order=4)
    D1_mat = build_deriv(Ns, dR, order=4)
    D2_mat = build_deriv(Ns, dR, order=4)

    I_Ns = sp.eye(Ns, format='csr')
    L1_2d = sp.kron(L1_mat, I_Ns, format='csr')
    L2_2d = sp.kron(I_Ns, L2_mat, format='csr')
    D1_2d = sp.kron(D1_mat, I_Ns, format='csr')
    D2_2d = sp.kron(I_Ns, D2_mat, format='csr')

    M11 = sp.diags(Minv_moire[:, :, 0, 0].ravel(), format='csr')
    M22 = sp.diags(Minv_moire[:, :, 1, 1].ravel(), format='csr')
    M12 = sp.diags(Minv_moire[:, :, 0, 1].ravel(), format='csr')

    prefactor = 0.5 / (2 * np.pi)**2
    K_op = -prefactor * (M11 @ L1_2d + M22 @ L2_2d + 2 * M12 @ (D1_2d @ D2_2d))
    K_op = 0.5 * (K_op + K_op.conj().T)

    V_op = sp.diags(V_moire.ravel(), format='csr')
    H = V_op + K_op

    # ── 4. Diag ──
    sigma_ea = V_moire.min()
    print(f"  H size: {H.shape[0]}×{H.shape[0]}, nnz={H.nnz:,}, "
          f"σ={sigma_ea:.6f}")

    t0 = time.time()
    evals_ea, evecs_ea = eigsh(H, k=N_MODES, sigma=sigma_ea, which='LM',
                                maxiter=10000, tol=1e-10)
    dt = time.time() - t0
    idx = np.argsort(evals_ea)
    evals_ea = evals_ea[idx]

    freqs_ea = omega_ref + evals_ea
    print(f"  eigsh: {dt:.1f}s, {N_MODES} modes")
    print(f"  ω range: [{freqs_ea[0]:.6f}, {freqs_ea[-1]:.6f}]")

    return freqs_ea, evals_ea, omega_ref, V_moire


# ═══════════════════════════════════════════════════════════════
#  Comparison plot
# ═══════════════════════════════════════════════════════════════

def plot_comparison(freqs_fdfd, freqs_ea, omega_ref_ea, V_moire):
    """EA vs FDFD eigenvalue comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    window = 0.06

    # ── Panel 1: Level diagram ──
    ax = axes[0]
    for label, freqs, color, xpos in [
            ('FDFD', freqs_fdfd, 'red', 0.35),
            ('EA', freqs_ea, 'green', 0.65)]:
        if freqs is None:
            continue
        mask = np.abs(freqs - OMEGA0) < window
        f = freqs[mask]
        ax.hlines(f, xpos - 0.1, xpos + 0.1, color=color, lw=0.8)
        ax.text(xpos, OMEGA0 + window * 0.9, label, ha='center',
                fontsize=11, color=color, fontweight='bold')

    ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5,
               label=f'ω₀={OMEGA0:.5f}')
    ax.set_ylabel(r'$\omega\, a / 2\pi c$')
    ax.set_title('Eigenvalue Level Diagram')
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(OMEGA0 - window, OMEGA0 + window)

    # ── Panel 2: Sorted eigenvalue comparison ──
    ax = axes[1]
    if freqs_fdfd is not None and freqs_ea is not None:
        mask_fdfd = np.abs(freqs_fdfd - OMEGA0) < window
        mask_ea   = np.abs(freqs_ea - OMEGA0) < window
        f_fdfd = np.sort(freqs_fdfd[mask_fdfd])
        f_ea   = np.sort(freqs_ea[mask_ea])
        n_compare = min(len(f_fdfd), len(f_ea))

        if n_compare > 0:
            diff = (f_ea[:n_compare] - f_fdfd[:n_compare]) * 1000
            ax.plot(range(n_compare), diff, 'go-', ms=3, label='EA − FDFD')
            ax.axhline(0, color='gray', ls='--', lw=0.5)
            rms = np.sqrt(np.mean(diff**2))
            ax.text(0.95, 0.95, f'RMS = {rms:.3f}×10⁻³',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(boxstyle='round', fc='lightyellow'))

    ax.set_xlabel('Eigenvalue index (near ω₀)')
    ax.set_ylabel(r'$\Delta\omega \times 10^3$')
    ax.set_title('EA − FDFD Differences')
    ax.legend()

    # ── Panel 3: Moiré potential ──
    ax = axes[2]
    if V_moire is not None:
        im = ax.imshow(V_moire.T * 1000, origin='lower', cmap='coolwarm',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, label=r'$V \times 10^3$')
        ax.set_title(r'Moiré potential $V(\mathbf{R})$')
        ax.set_xlabel('s₁'); ax.set_ylabel('s₂')

    fig.suptitle(f'EA vs FDFD — Square ({M_IDX},{N_IDX}), '
                 f'θ={THETA_DEG:.2f}°, N={N_CELLS}, '
                 f'TM band {TARGET_BAND} at M\n'
                 f'Phase 1: res={MPB_RES}, dk={DK}, '
                 f'fd_order={FD_ORDER}, registry={REGISTRY_NR}²',
                 fontsize=11)
    plt.tight_layout()
    out = OUTDIR / 'fig_fast_ea_vs_fdfd.png'
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume Phase 1 from checkpoint')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 (load saved registry)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Just plot from saved data')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  Fast EA vs FDFD Comparison")
    print(f"  Square ({M_IDX},{N_IDX}): θ={THETA_DEG:.2f}°, N={N_CELLS}")
    print(f"  Phase 1: MPB res={MPB_RES}, dk={DK}, fd_order={FD_ORDER}")
    print(f"  Registry: {REGISTRY_NR}², moiré grid: {NS_EA}²")
    print(f"  Target: TM band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}")
    print("=" * 70)

    # ── Load FDFD reference ──
    fdfd_file = OUTDIR / 'fdfd_supercell.npz'
    if fdfd_file.exists():
        freqs_fdfd = np.load(fdfd_file)['freqs']
        print(f"\n  FDFD reference loaded: {len(freqs_fdfd)} modes, "
              f"range [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]")
    else:
        print("\n  WARNING: No FDFD data found — will plot EA only")
        freqs_fdfd = None

    if args.plot_only:
        ea_file = OUTDIR / 'ea_fast_results.npz'
        if ea_file.exists():
            d = np.load(ea_file)
            plot_comparison(freqs_fdfd, d['freqs_ea'], d['omega_ref'],
                            d['V_moire'])
        else:
            print(f"  No EA data: {ea_file}")
        return

    # ── Phase 1 ──
    final_file = OUTDIR / 'ea_registry_fast.npz'
    if args.skip_phase1 and final_file.exists():
        print("\n  Loading saved registry...")
        d = np.load(final_file)
        omega0_reg = d['omega0']
        vg_reg = d['vg']
        Minv_reg = d['M_inv']
        stencil_omega_reg = d['stencil_omega']
        print(f"  Loaded {REGISTRY_NR}² registry from {final_file}")
    else:
        print("\n  Phase 1: Registry sweep...")
        omega0_reg, vg_reg, Minv_reg, stencil_omega_reg = \
            run_phase1_registry(resume=args.resume)

    # Quick sanity: band 3 at center
    cx, cy = REGISTRY_NR // 2, REGISTRY_NR // 2
    print(f"  ω₃ at center δ=(0.5,0.5): {omega0_reg[cx, cy, TARGET_BAND]:.6f}"
          f"  (ref: {OMEGA0:.6f})")

    # ── Phase 3: EA solve ──
    freqs_ea, evals_ea, omega_ref, V_moire = run_ea_solve(omega0_reg, Minv_reg)

    # Save results
    np.savez(OUTDIR / 'ea_fast_results.npz',
             freqs_ea=freqs_ea, evals_ea=evals_ea,
             omega_ref=omega_ref, V_moire=V_moire)

    # ── Comparison ──
    print("\n  Generating comparison plot...")
    plot_comparison(freqs_fdfd, freqs_ea, omega_ref, V_moire)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    w = 0.03
    for label, freqs in [('FDFD', freqs_fdfd), ('EA', freqs_ea)]:
        if freqs is None:
            continue
        mask = np.abs(freqs - OMEGA0) < w
        f = np.sort(freqs[mask])
        print(f"\n  {label}: {len(f)} eigenvalues within ±{w} of ω₀")
        for i, ff in enumerate(f[:15]):
            print(f"    {i:3d}: ω = {ff:.6f}  (Δ = {ff-OMEGA0:+.6f})")

    if freqs_fdfd is not None:
        mask_fdfd = np.abs(freqs_fdfd - OMEGA0) < w
        mask_ea = np.abs(freqs_ea - OMEGA0) < w
        f_fdfd = np.sort(freqs_fdfd[mask_fdfd])
        f_ea = np.sort(freqs_ea[mask_ea])
        n_cmp = min(len(f_fdfd), len(f_ea))
        if n_cmp > 0:
            diff = f_ea[:n_cmp] - f_fdfd[:n_cmp]
            print(f"\n  EA − FDFD (first {n_cmp} matched eigenvalues):")
            print(f"    Mean Δω = {np.mean(diff):+.6f}")
            print(f"    RMS  Δω = {np.sqrt(np.mean(diff**2)):.6f}")
            print(f"    Max |Δω|= {np.max(np.abs(diff)):.6f}")

    dt_total = time.time() - t_total
    print(f"\nTotal time: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == '__main__':
    main()
