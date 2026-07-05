#!/usr/bin/env python3
"""
R05 Compute: Scaling Laws & Tunability
========================================
Collate data across the full eta sweep (8 twist angles θ=0.5° to 8.0°)
to extract scaling laws, tunability, and operating regime information.

Quantities computed:
  1. Bandwidth (BW_50, BW_20, total) vs θ
  2. Mode spacing δω vs θ
  3. Localization metrics (IPR, spread) vs θ
  4. Kinetic-to-potential ratio vs θ (controls flat-band regime)
  5. Power-law fits BW ~ η^α, IPR ~ η^β, etc.
  6. Band mixing vs θ

Output: R05_data.json, R05_scaling.npz
"""

import sys, json
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
OUTDIR   = Path(__file__).resolve().parent
BASE_RUN = OUTDIR.parent / "runsV3" / "phase0_mpb_v3_20260206_152443"
SWEEP    = BASE_RUN / "eta_sweep_20260206_173808"

sys.path.insert(0, str(OUTDIR.parent / "phasesV3"))

import h5py


def power_law_fit(x, y, label=""):
    """Log-log OLS: y = A * x^alpha. Returns (alpha, A, R²)."""
    mask = (np.asarray(x) > 0) & (np.asarray(y) > 0)
    lx = np.log(np.asarray(x)[mask])
    ly = np.log(np.asarray(y)[mask])
    if len(lx) < 3:
        return np.nan, np.nan, np.nan
    p = np.polyfit(lx, ly, 1)
    ly_fit = np.polyval(p, lx)
    ss_res = np.sum((ly - ly_fit)**2)
    ss_tot = np.sum((ly - ly.mean())**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0
    return p[0], np.exp(p[1]), R2


def main():
    print("="*70)
    print("R05 Compute: Scaling Laws & Tunability")
    print("="*70)

    # ── Load sweep summary ────────────────────────────────────────────────
    with open(SWEEP / "sweep_results.json") as f:
        sweep = json.load(f)
    sweep.sort(key=lambda e: e['theta_deg'])

    thetas  = np.array([e['theta_deg']    for e in sweep])
    etas    = np.array([e['eta']           for e in sweep])
    L_moire = np.array([e['moire_length'] for e in sweep])
    BW_50   = np.array([e['bandwidth_50'] for e in sweep])
    gap_01  = np.array([e['gap_01']       for e in sweep])
    omega_ref_arr = np.array([e['omega_ref'] for e in sweep])

    n_angles = len(thetas)
    N_BANDS  = 5
    TARGET   = 2  # target band index in subspace

    print(f"Angles: {thetas}")
    print(f"η range: [{etas.min():.5f}, {etas.max():.5f}]")

    # ── Per-angle eigenvalue analysis ─────────────────────────────────────
    BW_20 = np.zeros(n_angles)
    BW_total = np.zeros(n_angles)
    mode_spacing_mean = np.zeros(n_angles)
    mode_spacing_first = np.zeros(n_angles)
    n_modes_arr = np.zeros(n_angles, dtype=int)

    for i, entry in enumerate(sweep):
        evals = np.array(entry['eigenvalues'])
        n_modes_arr[i] = len(evals)
        BW_total[i] = evals[-1] - evals[0]
        BW_20[i] = evals[min(19, len(evals)-1)] - evals[0]
        diffs = np.diff(evals)
        mode_spacing_mean[i] = diffs.mean() if len(diffs) > 0 else 0
        mode_spacing_first[i] = diffs[0] if len(diffs) > 0 else 0

    # ── Per-angle localization analysis from Phase 3 mode stats ───────────
    ipr_ground = np.zeros(n_angles)
    ipr_mean_10 = np.zeros(n_angles)
    spread_ground = np.zeros(n_angles)
    spread_mean_10 = np.zeros(n_angles)
    dom_weight_ground = np.zeros(n_angles)
    max_mixing = np.zeros(n_angles)

    for i, entry in enumerate(sweep):
        theta_str = f"{entry['theta_deg']:.3f}"
        stats_file = SWEEP / f"theta_{theta_str}" / "candidate_0000" / "phase3_mode_stats.json"
        if not stats_file.exists():
            print(f"  WARNING: {stats_file} not found, skipping")
            continue
        with open(stats_file) as f:
            stats = json.load(f)
        if len(stats) == 0:
            continue

        ipr_ground[i] = stats[0].get('ipr', 0)
        spread_ground[i] = stats[0].get('spread', 0)
        dom_weight_ground[i] = stats[0].get('dominant_band_weight', 0)
        max_mixing[i] = entry['max_mixing']

        n10 = min(10, len(stats))
        ipr_mean_10[i] = np.mean([s.get('ipr', 0) for s in stats[:n10]])
        spread_mean_10[i] = np.mean([s.get('spread', 0) for s in stats[:n10]])

    # ── Per-angle Phase 2 Hamiltonian parameter analysis ──────────────────
    # Lambda (potential) is identical in normalized coords, but we extract
    # the PHYSICAL potential depth V_depth * (2π/L_m)² and kinetic scale
    V_diag_depth = np.zeros((n_angles, N_BANDS))
    M_trace_mean = np.zeros((n_angles, N_BANDS))
    KE_scale     = np.zeros((n_angles, N_BANDS))  # Kinetic energy estimate
    VK_ratio     = np.zeros((n_angles, N_BANDS))   # V/KE ratio

    for i, entry in enumerate(sweep):
        theta_str = f"{entry['theta_deg']:.3f}"
        p2_file = SWEEP / f"theta_{theta_str}" / "candidate_0000" / "phase2_multiband_data.h5"
        if not p2_file.exists():
            print(f"  WARNING: {p2_file} not found")
            continue
        with h5py.File(p2_file, 'r') as hf:
            Lambda = hf['Lambda'][:]     # (Ns, Ns, N_BANDS, N_BANDS)
            M_inv  = hf['M_inv'][:]       # (Ns, Ns, N_BANDS, N_BANDS, 2, 2)
            eta_h5 = float(hf.attrs.get('eta', entry['eta']))
            Ns = Lambda.shape[0]
            B_m = np.array(hf.attrs.get('B_moire', [[1,0],[0,1]]))

        dR = L_moire[i] / Ns  # physical grid spacing

        for n in range(N_BANDS):
            V = Lambda[:, :, n, n].real
            V_diag_depth[i, n] = V.max() - V.min()

            Minv_trace = (M_inv[:, :, n, n, 0, 0] + M_inv[:, :, n, n, 1, 1]).real
            M_trace_mean[i, n] = np.mean(np.abs(Minv_trace))

            # Kinetic energy scale = |Tr(M⁻¹)_mean| * (pi/L_m)²
            KE_scale[i, n] = M_trace_mean[i, n] * (np.pi / L_moire[i])**2
            # Ratio determines flat-band vs dispersive regime
            VK_ratio[i, n] = V_diag_depth[i, n] / KE_scale[i, n] if KE_scale[i, n] > 0 else np.inf

    # ──Fit power laws ────────────────────────────────────────────────────
    print("\n--- Power-law fits ---")
    fits = {}

    # BW_50 ~ eta^alpha
    alpha, A, R2 = power_law_fit(etas, BW_50, "BW_50")
    fits['BW_50'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  BW_50  ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # BW_20 ~ eta^alpha
    alpha, A, R2 = power_law_fit(etas, BW_20, "BW_20")
    fits['BW_20'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  BW_20  ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # Mode spacing ~ eta^alpha
    alpha, A, R2 = power_law_fit(etas, mode_spacing_mean, "δω_mean")
    fits['mode_spacing_mean'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  δω_mean ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # IPR (ground) ~ eta^beta
    alpha, A, R2 = power_law_fit(etas, ipr_ground, "IPR_ground")
    fits['IPR_ground'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  IPR_ground ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # Spread ~ eta^beta
    alpha, A, R2 = power_law_fit(etas, spread_ground, "spread_ground")
    fits['spread_ground'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  spread_ground ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # V/KE ratio target band
    alpha, A, R2 = power_law_fit(etas, VK_ratio[:, TARGET], "V/KE_target")
    fits['VK_ratio_target'] = {'alpha': alpha, 'A': A, 'R2': R2}
    print(f"  V/KE_target  ~ η^{alpha:.3f}  (A={A:.4e}, R²={R2:.4f})")

    # Small-angle regime fits (θ ≤ 3° → first 6 points)
    small_mask = thetas <= 3.0
    if small_mask.sum() >= 3:
        print("\n--- Small-angle regime (θ ≤ 3°) ---")
        alpha, A, R2 = power_law_fit(etas[small_mask], BW_50[small_mask])
        fits['BW_50_small'] = {'alpha': alpha, 'A': A, 'R2': R2}
        print(f"  BW_50  ~ η^{alpha:.3f}  (R²={R2:.4f})")

        alpha, A, R2 = power_law_fit(etas[small_mask], BW_20[small_mask])
        fits['BW_20_small'] = {'alpha': alpha, 'A': A, 'R2': R2}
        print(f"  BW_20  ~ η^{alpha:.3f}  (R²={R2:.4f})")

    # ── Band mixing analysis ─────────────────────────────────────────────
    # From sweep_results: band_compositions for first 50 modes at each angle
    band_purity = np.zeros((n_angles, N_BANDS))  # avg dom_weight for modes dominated by each band
    band_counts = np.zeros((n_angles, N_BANDS))   # how many modes dominated by each band

    for i, entry in enumerate(sweep):
        comps = entry.get('band_compositions', [])
        for mode_comp in comps:
            dom = mode_comp.get('dominant', -1)
            if 0 <= dom < N_BANDS:
                band_purity[i, dom] += mode_comp.get('max_weight', 0)
                band_counts[i, dom] += 1
        for n in range(N_BANDS):
            if band_counts[i, n] > 0:
                band_purity[i, n] /= band_counts[i, n]

    # ── Save results ──────────────────────────────────────────────────────
    np.savez(OUTDIR / "R05_scaling.npz",
             thetas=thetas,
             etas=etas,
             L_moire=L_moire,
             BW_50=BW_50,
             BW_20=BW_20,
             BW_total=BW_total,
             gap_01=gap_01,
             mode_spacing_mean=mode_spacing_mean,
             mode_spacing_first=mode_spacing_first,
             ipr_ground=ipr_ground,
             ipr_mean_10=ipr_mean_10,
             spread_ground=spread_ground,
             spread_mean_10=spread_mean_10,
             dom_weight_ground=dom_weight_ground,
             max_mixing=max_mixing,
             V_diag_depth=V_diag_depth,
             M_trace_mean=M_trace_mean,
             KE_scale=KE_scale,
             VK_ratio=VK_ratio,
             band_purity=band_purity,
             band_counts=band_counts)
    print(f"\nSaved {OUTDIR / 'R05_scaling.npz'}")

    meta = {
        'n_angles': int(n_angles),
        'N_BANDS': N_BANDS,
        'TARGET_BAND': TARGET,
        'thetas': thetas.tolist(),
        'etas': etas.tolist(),
        'L_moire': L_moire.tolist(),
        'power_law_fits': {k: {kk: (float(vv) if np.isfinite(vv) else None)
                                for kk, vv in v.items()}
                           for k, v in fits.items()},
        'band_purity_at_1.1deg': band_purity[list(thetas).index(1.1)].tolist()
            if 1.1 in thetas else None,
        'VK_ratio_target': VK_ratio[:, TARGET].tolist(),
    }
    with open(OUTDIR / "R05_data.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {OUTDIR / 'R05_data.json'}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'θ (°)':>7} {'η':>9} {'L_m':>7} {'BW50':>12} {'BW20':>12} "
          f"{'δω_mean':>12} {'IPR_gnd':>9} {'V/KE_tgt':>9} {'mixing':>7}")
    print("-"*90)
    for i in range(n_angles):
        print(f"{thetas[i]:7.1f} {etas[i]:9.5f} {L_moire[i]:7.1f} "
              f"{BW_50[i]:12.6e} {BW_20[i]:12.6e} "
              f"{mode_spacing_mean[i]:12.6e} {ipr_ground[i]:9.5f} "
              f"{VK_ratio[i,TARGET]:9.2f} {max_mixing[i]:7.4f}")
    print("="*90)

    print("\nR05 compute complete.")


if __name__ == '__main__':
    main()
