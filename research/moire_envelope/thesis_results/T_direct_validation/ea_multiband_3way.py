#!/usr/bin/env python3
"""
Multi-band EA vs FDFD comparison for square lattice bilayer supercell.
=====================================================================
θ ≈ 10.39° (m,n)=(11,1), N_cells=122.

This script does a proper multi-band envelope approximation:
  - Subspace bands: [3, 4, 5, 6] (bands that cross the ω₀ window across δ)
  - All bands (incl. BH buffer): [0..9] (±3 bands beyond subspace)
  - Registry: 32×32 with 5% checkpointing
  - MPB resolution: 32 (fast mode) or configurable
  - Moiré grid: 128×128 (upscaled from registry via interpolation)

Compares EA eigenvalues against existing FDFD data.

Usage:
    python ea_multiband_3way.py                   # full run
    python ea_multiband_3way.py --res 64           # higher MPB resolution
    python ea_multiband_3way.py --skip-phase1      # skip Phase 1 (load from file)
    python ea_multiband_3way.py --plot-only         # just plot from saved data
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
# thesis_results/T_direct_validation → moire_envelope
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT))
sys.path.insert(0, str(MOIRE_ROOT / "phasesV3"))

# ═══════════════════════════════════════════════════════════════
#  Physical parameters (same as square_supercell_3way.py)
# ═══════════════════════════════════════════════════════════════

A         = 1.0
R_OVER_A  = 0.2
EPS_ROD   = 11.56
EPS_BG    = 1.0
M_IDX, N_IDX = 11, 1
N_CELLS   = M_IDX**2 + N_IDX**2  # 122
THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
THETA_DEG = np.degrees(THETA_RAD)

L1 = np.array([M_IDX, N_IDX], dtype=float) * A
L2 = np.array([-N_IDX, M_IDX], dtype=float) * A
L_SUPER = np.sqrt(L1 @ L1)
B_SUPER = np.column_stack([L1, L2])
B_MONO  = np.eye(2) * A

OMEGA0    = 0.68457
TARGET_BAND = 3   # 0-indexed monolayer band

# Multi-band config
# Bands 3-6 cross the ω₀±0.10 window across the δ sweep.
# Add ±3 bands as BH buffer → all_bands = [0..9].
SUBSPACE_BANDS = [3, 4, 5, 6]
ALL_BANDS      = list(range(10))
N_SUBSPACE     = len(SUBSPACE_BANDS)
N_ALL          = len(ALL_BANDS)

# Grid sizes
REGISTRY_NR = 32
NS_EA       = 128
N_MODES     = 50

# Output
OUTDIR = SCRIPT_DIR / "square_3way"

# ═══════════════════════════════════════════════════════════════
#  Phase 1: Registry sweep with checkpointing
# ═══════════════════════════════════════════════════════════════

def run_phase1(mpb_resolution=32, n_workers=16):
    """Multi-band Phase 1 registry sweep with 5% checkpoints."""
    from multiprocessing import Pool
    from phase1_mpb_v3 import _compute_single_registry_point

    NR = REGISTRY_NR
    n_stencil = 7
    checkpoint_file = OUTDIR / 'ea_multiband_registry.npz'
    
    print(f"\n  Phase 1: {NR}×{NR} registry, {N_ALL} bands, "
          f"MPB res={mpb_resolution}, {n_workers} workers")
    print(f"  Subspace: {SUBSPACE_BANDS}, All: {ALL_BANDS}")

    params = {
        'lattice_type': 'square',
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_ROD,
        'k0': [0.5, 0.5],
        'dk': 0.06,
        'all_bands': ALL_BANDS,
        'polarization': 'TM',
        'fd_order': 6,
        'resolution': mpb_resolution,
        'max_band': max(ALL_BANDS) + 1,
        'export_bloch_fields': False,
    }

    # Initialize or resume from checkpoint
    omega0_reg = np.full((NR, NR, N_ALL), np.nan)
    vg_reg     = np.full((NR, NR, N_ALL, 2), np.nan)
    Minv_reg   = np.full((NR, NR, N_ALL, 2, 2), np.nan)
    stencil_reg = np.full((NR, NR, N_ALL, n_stencil, n_stencil), np.nan)
    done_mask  = np.zeros((NR, NR), dtype=bool)

    if checkpoint_file.exists():
        d = np.load(checkpoint_file)
        if d['omega0'].shape == omega0_reg.shape:
            omega0_reg = d['omega0']
            vg_reg = d['vg']
            Minv_reg = d['M_inv']
            stencil_reg = d['stencil_omega']
            done_mask = np.isfinite(omega0_reg[:, :, 0])
            n_done = done_mask.sum()
            print(f"  Resumed from checkpoint: {n_done}/{NR**2} done "
                  f"({100*n_done/NR**2:.0f}%)")
        else:
            print(f"  Checkpoint has different shape, starting fresh")

    # Build work items for remaining points
    step = 1.0 / NR
    work = []
    for ix in range(NR):
        for iy in range(NR):
            if not done_mask[ix, iy]:
                delta_frac = np.array([ix * step, iy * step])
                work.append((ix, iy, delta_frac, params))

    if not work:
        print("  All registry points already computed!")
        return omega0_reg, vg_reg, Minv_reg, stencil_reg

    total = NR * NR
    n_done_initial = done_mask.sum()
    checkpoint_interval = max(1, int(0.05 * total))  # 5%
    
    print(f"  Computing {len(work)} remaining points...")
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        for ix, iy, result in pool.imap_unordered(
                _compute_single_registry_point, work, chunksize=4):
            omega0_reg[ix, iy] = result['omega0']
            vg_reg[ix, iy] = result['vg']
            Minv_reg[ix, iy] = result['M_inv']
            stencil_reg[ix, iy] = result['omega_stencil']

            n_done = np.isfinite(omega0_reg[:, :, 0]).sum()
            n_new = n_done - n_done_initial

            if n_new % checkpoint_interval == 0 and n_new > 0:
                pct = 100 * n_done / total
                elapsed = time.time() - t0
                rate = n_new / elapsed if elapsed > 0 else 0
                eta_s = (len(work) - n_new) / rate if rate > 0 else 0
                print(f"    {n_done}/{total} ({pct:.0f}%) - "
                      f"{rate:.1f} pts/s - ETA {eta_s:.0f}s", flush=True)
                np.savez(checkpoint_file,
                         omega0=omega0_reg, vg=vg_reg,
                         M_inv=Minv_reg, stencil_omega=stencil_reg)

    # Final save
    np.savez(checkpoint_file,
             omega0=omega0_reg, vg=vg_reg,
             M_inv=Minv_reg, stencil_omega=stencil_reg)

    dt = time.time() - t0
    print(f"  Phase 1 done in {dt:.0f}s ({dt/60:.1f}min)")
    return omega0_reg, vg_reg, Minv_reg, stencil_reg


# ═══════════════════════════════════════════════════════════════
#  Phase 2+3: Multi-band Hamiltonian assembly and solve
# ═══════════════════════════════════════════════════════════════

def compute_delta_frac_grid(Ns):
    """Map moiré grid points to registry shifts δ(R)."""
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

    pos_flat = np.stack([X.ravel(), Y.ravel()], axis=-1)
    disp = ((R_mat - np.eye(2)) @ pos_flat.T).T
    delta_frac = (B_mono_inv @ disp.T).T
    delta_frac = delta_frac - np.floor(delta_frac)
    return delta_frac.reshape(Ns, Ns, 2)


def interpolate_registry_to_moire(registry_data, delta_frac_grid):
    """Interpolate registry grid data to moiré grid via δ mapping.
    
    Args:
        registry_data: (NR, NR, ...) array on periodic registry grid
        delta_frac_grid: (Ns, Ns, 2) fractional delta coordinates
        
    Returns:
        (Ns, Ns, ...) interpolated data
    """
    from scipy.interpolate import RegularGridInterpolator

    NR = registry_data.shape[0]
    Ns = delta_frac_grid.shape[0]
    extra_shape = registry_data.shape[2:]
    
    reg_ax = np.linspace(0, 1, NR, endpoint=False)
    reg_ax_ext = np.concatenate([reg_ax, [1.0]])

    def pad_periodic(arr_2d):
        padded = np.concatenate([arr_2d, arr_2d[:1, :]], axis=0)
        padded = np.concatenate([padded, padded[:, :1]], axis=1)
        return padded

    pts = delta_frac_grid.reshape(-1, 2)

    if len(extra_shape) == 0:
        padded = pad_periodic(registry_data)
        interp = RegularGridInterpolator(
            (reg_ax_ext, reg_ax_ext), padded,
            method='linear', bounds_error=False, fill_value=None)
        return interp(pts).reshape(Ns, Ns)

    # Flatten extra dims, interpolate each, reshape
    flat = registry_data.reshape(NR, NR, -1)
    n_extra = flat.shape[2]
    result = np.zeros((Ns * Ns, n_extra))
    
    for k in range(n_extra):
        padded = pad_periodic(flat[:, :, k])
        interp = RegularGridInterpolator(
            (reg_ax_ext, reg_ax_ext), padded,
            method='linear', bounds_error=False, fill_value=None)
        result[:, k] = interp(pts)
    
    return result.reshape(Ns, Ns, *extra_shape)


def run_multiband_ea(omega0_reg, vg_reg, Minv_reg, M_inv_max_trace=None):
    """Multi-band envelope Hamiltonian assembly and diagonalization.
    
    Args:
        M_inv_max_trace: if set, clamp |Tr(M_inv)| per grid point per band.
            Regularizes hot spots where band crossings inflate curvature.
            Recommended: ~2.0 (matches monolayer Tr=2.43).
    """
    from phase3_mpb_v3 import (
        assemble_multiband_hamiltonian,
        solve_multiband_envelope,
    )

    Ns = NS_EA
    Nb = N_SUBSPACE
    dR = L_SUPER / Ns

    print(f"\n  Multi-band EA: Ns={Ns}, N_subspace={Nb}, "
          f"bands={SUBSPACE_BANDS}")

    # 1. Map to moiré grid 
    delta_frac = compute_delta_frac_grid(Ns)

    # 2. Extract subspace band data from registry
    sub_idx = np.array(SUBSPACE_BANDS)
    omega0_sub = omega0_reg[:, :, sub_idx]   # (NR, NR, Nb)
    vg_sub     = vg_reg[:, :, sub_idx, :]    # (NR, NR, Nb, 2)
    Minv_sub   = Minv_reg[:, :, sub_idx, :, :]  # (NR, NR, Nb, 2, 2)

    # 3. Interpolate to moiré grid
    omega_moire = interpolate_registry_to_moire(omega0_sub, delta_frac)
    vg_moire    = interpolate_registry_to_moire(vg_sub, delta_frac)
    Minv_moire  = interpolate_registry_to_moire(Minv_sub, delta_frac)

    omega_ref = OMEGA0
    target_in_sub = SUBSPACE_BANDS.index(TARGET_BAND)
    
    print(f"  ω_ref = {omega_ref:.6f} (= ω₀)")
    for n in range(Nb):
        v = omega_moire[:, :, n]
        print(f"  Band {SUBSPACE_BANDS[n]}: ω ∈ [{v.min():.4f}, {v.max():.4f}], "
              f"V ∈ [{(v-omega_ref).min():.4f}, {(v-omega_ref).max():.4f}]")

    # 4. Build multi-band matrices (diagonal in bands, no inter-band coupling)
    Lambda = np.zeros((Ns, Ns, Nb, Nb))
    v_drift = np.zeros((Ns, Ns, Nb, Nb, 2))
    M_inv = np.zeros((Ns, Ns, Nb, Nb, 2, 2))
    for n in range(Nb):
        Lambda[:, :, n, n] = omega_moire[:, :, n] - omega_ref
        v_drift[:, :, n, n, :] = vg_moire[:, :, n, :]
        M_inv[:, :, n, n, :, :] = Minv_moire[:, :, n, :, :]

    A_berry = np.zeros((Ns, Ns, Nb, Nb, 2))
    Phi_BH = np.zeros((Ns, Ns, Nb, Nb))
    eta = A / L_SUPER

    print(f"  η = {eta:.6f}, dR = {dR:.6f}")
    if M_inv_max_trace is not None:
        print(f"  M_inv_max_trace = {M_inv_max_trace}")

    # 5. Assemble Hamiltonian
    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_SUPER,
        include_drift=True, include_kinetic=True,
        include_born_huang=False,
        order=4, include_offdiag_A=False,
        M_inv_max_trace=M_inv_max_trace,
    )

    # 6. Solve: sigma=0 targets modes near ω₀
    sigma = 0.0
    print(f"  σ = {sigma:.6f}")

    evals, evecs = solve_multiband_envelope(H, N_MODES, sigma=sigma)
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]

    freqs_ea = omega_ref + evals

    # Band character analysis: project eigenvectors onto band blocks
    band_weights = np.zeros((N_MODES, Nb))
    N_s = Ns * Ns
    for mode in range(N_MODES):
        v = evecs[:, mode]
        for n in range(Nb):
            band_block = v[n::Nb]  # every Nb-th element starting at n
            band_weights[mode, n] = np.sum(np.abs(band_block)**2)

    # Print character for first 10 modes
    print(f"\n  Band character (first 10 modes near ω₀):")
    print(f"  {'mode':>4s}  {'ω':>8s}  {'ε':>8s}  " + 
          "  ".join(f"b{SUBSPACE_BANDS[n]}" for n in range(Nb)))
    for i in range(min(10, N_MODES)):
        w_str = "  ".join(f"{band_weights[i,n]:.2f}" for n in range(Nb))
        print(f"  {i:4d}  {freqs_ea[i]:.5f}  {evals[i]:+.5f}  {w_str}")

    print(f"\n  EA frequencies range: [{freqs_ea.min():.6f}, {freqs_ea.max():.6f}]")

    return freqs_ea, evals, omega_ref, Lambda, band_weights


# ═══════════════════════════════════════════════════════════════
#  Comparison Plot
# ═══════════════════════════════════════════════════════════════

def plot_comparison(freqs_fdfd, freqs_ea, omega_ref, Lambda=None):
    """Side-by-side comparison of FDFD vs multi-band EA."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    window = 0.06

    # Panel 1: Level diagram
    ax = axes[0]
    for label, freqs, color, x in [
            ('FDFD', freqs_fdfd, 'red', 0.3),
            ('EA', freqs_ea, 'green', 0.7)]:
        if freqs is None:
            continue
        mask = np.abs(freqs - OMEGA0) < window
        f = freqs[mask]
        ax.hlines(f, x - 0.12, x + 0.12, color=color, lw=0.8)
        ax.text(x, OMEGA0 + window * 0.9, label, ha='center',
                fontsize=11, color=color, fontweight='bold')
    ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5)
    ax.set_ylabel(r'$\omega\, a / 2\pi c$')
    ax.set_title('Eigenvalue Level Diagram')
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(OMEGA0 - window, OMEGA0 + window)

    # Panel 2: Sorted eigenvalue differences
    ax = axes[1]
    if freqs_fdfd is not None and freqs_ea is not None:
        mask_f = np.abs(freqs_fdfd - OMEGA0) < window
        mask_e = np.abs(freqs_ea - OMEGA0) < window
        f_f = np.sort(freqs_fdfd[mask_f])
        f_e = np.sort(freqs_ea[mask_e])
        n = min(len(f_f), len(f_e))
        if n > 0:
            diff = (f_e[:n] - f_f[:n]) * 1000
            ax.plot(range(n), diff, 'go-', ms=3, label='EA − FDFD')
            ax.axhline(0, color='gray', ls='--', lw=0.5)
            rms = np.sqrt(np.mean(diff**2))
            ax.set_title(f'EA − FDFD (RMS = {rms:.2f} × 10⁻³)')
    ax.set_xlabel('Eigenvalue index (near ω₀)')
    ax.set_ylabel(r'$\Delta\omega \times 10^3$')
    ax.legend()

    # Panel 3: Moiré potential landscape
    ax = axes[2]
    if Lambda is not None:
        Nb = Lambda.shape[2]
        target_sub = SUBSPACE_BANDS.index(TARGET_BAND)
        V = Lambda[:, :, target_sub, target_sub] * 1000
        im = ax.imshow(V.T, origin='lower', cmap='coolwarm',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, label=r'$V \times 10^3$')
        ax.set_title(f'V(R) band {TARGET_BAND} ({N_SUBSPACE}-band subspace)')
        ax.set_xlabel('s₁'); ax.set_ylabel('s₂')

    fig.suptitle(f'Multi-band EA vs FDFD — Square ({M_IDX},{N_IDX}): '
                 f'θ={THETA_DEG:.2f}°, subspace={SUBSPACE_BANDS}',
                 fontsize=12)
    plt.tight_layout()
    out = OUTDIR / 'fig_multiband_ea_vs_fdfd.png'
    fig.savefig(out, dpi=200)
    print(f"  Saved {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=int, default=32,
                        help='MPB resolution per unit cell')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--mt', type=float, default=2.0,
                        help='M_inv_max_trace regularization (0=off)')
    args = parser.parse_args()
    mt = args.mt if args.mt > 0 else None

    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Multi-band EA vs FDFD Comparison")
    print(f"  Square ({M_IDX},{N_IDX}): θ={THETA_DEG:.2f}°, N={N_CELLS}")
    print(f"  Target: TM band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}")
    print(f"  Subspace: {SUBSPACE_BANDS} ({N_SUBSPACE} bands)")
    print(f"  All bands (incl BH): {ALL_BANDS} ({N_ALL} bands)")
    print(f"  Registry: {REGISTRY_NR}×{REGISTRY_NR}, "
          f"Moiré grid: {NS_EA}×{NS_EA}")
    print(f"  MPB resolution: {args.res}")
    print("=" * 70)

    # Load FDFD reference
    fdfd_file = OUTDIR / 'fdfd_supercell.npz'
    if fdfd_file.exists():
        freqs_fdfd = np.load(fdfd_file)['freqs']
        print(f"\n  FDFD reference: {len(freqs_fdfd)} modes, "
              f"[{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]")
    else:
        print("\n  WARNING: No FDFD data found!")
        freqs_fdfd = None

    if args.plot_only:
        ea_file = OUTDIR / 'ea_multiband_results.npz'
        if ea_file.exists():
            d = np.load(ea_file)
            plot_comparison(freqs_fdfd, d['freqs_ea'], 
                          float(d['omega_ref']), d.get('Lambda'))
        else:
            print(f"  No EA data: {ea_file}")
        return

    # Phase 1
    reg_file = OUTDIR / 'ea_multiband_registry.npz'
    if args.skip_phase1 and reg_file.exists():
        print(f"\n  Loading Phase 1 from {reg_file.name}...")
        d = np.load(reg_file)
        omega0_reg = d['omega0']
        vg_reg = d['vg']
        Minv_reg = d['M_inv']
        stencil_reg = d['stencil_omega']
        n_done = np.isfinite(omega0_reg[:, :, 0]).sum()
        print(f"  Loaded: {n_done}/{REGISTRY_NR**2} points, "
              f"{omega0_reg.shape[2]} bands")
    else:
        omega0_reg, vg_reg, Minv_reg, stencil_reg = run_phase1(
            mpb_resolution=args.res, n_workers=args.workers)

    # Multi-band EA solve
    freqs_ea, evals_ea, omega_ref, Lambda, band_weights = run_multiband_ea(
        omega0_reg, vg_reg, Minv_reg, M_inv_max_trace=mt)

    # Save results
    np.savez(OUTDIR / 'ea_multiband_results.npz',
             freqs_ea=freqs_ea, evals_ea=evals_ea,
             omega_ref=omega_ref, Lambda=Lambda,
             band_weights=band_weights,
             subspace_bands=SUBSPACE_BANDS,
             all_bands=ALL_BANDS,
             M_inv_max_trace=mt if mt else 0.0)

    # Plot
    plot_comparison(freqs_fdfd, freqs_ea, omega_ref, Lambda)

    # Summary
    if freqs_fdfd is not None:
        w = 0.03
        mask_f = np.abs(freqs_fdfd - OMEGA0) < w
        mask_e = np.abs(freqs_ea - OMEGA0) < w
        f_f = np.sort(freqs_fdfd[mask_f])
        f_e = np.sort(freqs_ea[mask_e])
        n = min(len(f_f), len(f_e))

        print(f"\n{'='*70}")
        print(f"  SUMMARY: eigenvalues within ±{w} of ω₀={OMEGA0:.5f}")
        print(f"{'='*70}")
        print(f"  FDFD: {len(f_f)} modes")
        print(f"  EA:   {len(f_e)} modes")
        if n > 0:
            diff = f_e[:n] - f_f[:n]
            print(f"  Matched: {n} pairs")
            print(f"  RMS diff: {np.sqrt(np.mean(diff**2))*1000:.3f} × 10⁻³")
            print(f"  Max diff: {np.max(np.abs(diff))*1000:.3f} × 10⁻³")
            print(f"\n  {'idx':>3s}  {'FDFD':>10s}  {'EA':>10s}  {'Δω×10³':>8s}")
            for i in range(min(n, 20)):
                print(f"  {i:3d}  {f_f[i]:.6f}  {f_e[i]:.6f}  "
                      f"{(f_e[i]-f_f[i])*1000:+.3f}")


if __name__ == '__main__':
    main()
