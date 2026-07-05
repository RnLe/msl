#!/usr/bin/env python3
"""A2: remote-band ladder — EA (Blaze V4) vs FDFD on the golden benchmark.

For each Phase-1 run (n_remote ∈ {0,4,8,16}) solve the 2-band envelope
problem at θ = 1.1213° (the (30,29) commensurate angle) targeting the
*bottom* of the spectrum, and Hungarian-match the 50 lowest envelope modes
against the FDFD res-40 reference — the exact protocol of
thesis_results/T_direct_validation/plot_definitive_1deg.py.

Outputs: a2_ladder_results.json + printed convergence table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lib.phase2_blaze_v4 import (  # noqa: E402
    load_phase1_h5,
    compute_moire_metadata,
    transform_mass_tensor,
    transform_velocity,
    born_huang_metric_factor,
    assemble_hamiltonian,
    solve_envelope,
    eigenvalue_to_frequency,
)
from lib.solve_driver import extract_multi_band  # noqa: E402

THETA_DEG = 1.1213111153817366
N_MODES = 50
BANDS = [0, 1]  # Dirac pair of the golden crystal
FDFD_REF = Path(
    "/home/renlephy/msl/research/moire_envelope/thesis_results/"
    "T_direct_validation/fdfd_dirac_m30_n29_res40_v2.npz"
)


def solve_bottom(band_data: dict, theta_deg: float, Ns: int, n_modes: int,
                 include_bh: bool = True) -> dict:
    """Multi-band envelope solve targeting the spectrum bottom."""
    Nb = band_data["Nb"]
    moire = compute_moire_metadata("honeycomb", 1.0, theta_deg)
    B_moire = moire["B_moire"]
    B_inv = np.linalg.inv(B_moire)

    eig = band_data["eigenvalues"]
    lambda_ref = float(np.mean(eig))

    Lambda = np.zeros((Ns, Ns, Nb, Nb), dtype=complex)
    for n in range(Nb):
        Lambda[..., n, n] = eig[..., n] - lambda_ref

    v1, v2 = transform_velocity(band_data["velocity_x"], band_data["velocity_y"], B_inv)
    M11, M12, M21, M22 = transform_mass_tensor(
        band_data["mass_xx"], band_data["mass_xy"],
        band_data["mass_yx"], band_data["mass_yy"], B_inv)

    bh_factor = born_huang_metric_factor(B_moire)
    bh = (band_data["born_huang"] * bh_factor
          if (include_bh and band_data["born_huang"] is not None) else None)
    sc = (band_data["slow_coefficient"] * bh_factor
          if band_data["slow_coefficient"] is not None else None)

    H = assemble_hamiltonian(
        Lambda, v1, v2, M11, M12, M22,
        band_data["berry_x"], band_data["berry_y"], bh, sc,
        Ns, Nb,
        include_drift=True, include_kinetic=True,
        include_born_huang=include_bh, include_slow_coeff=sc is not None,
        fd_order=4, k_s=(0.0, 0.0),
    )

    # Target the spectrum bottom: just below min of the lower-band potential.
    lam00 = Lambda[..., 0, 0].real
    sigma = float(lam00.min() - 0.05 * (lam00.max() - lam00.min()) - 1e-6)

    eigenvals, _ = solve_envelope(H, n_modes, sigma)
    eigenvals = np.sort(eigenvals.real)
    freqs = np.array([eigenvalue_to_frequency(l, lambda_ref) for l in eigenvals])
    return {"freqs": freqs, "lambda_ref": lambda_ref, "sigma": sigma}


def hungarian_match(env_freqs: np.ndarray, fdfd_all: np.ndarray) -> dict:
    """Match protocol of plot_definitive_1deg.py."""
    env_bw = env_freqs.max() - env_freqs.min()
    spacing = float(np.mean(np.diff(np.sort(env_freqs))))
    lo, hi = env_freqs.min() - 2 * spacing, env_freqs.max() + 2 * spacing
    fdfd_win = fdfd_all[(fdfd_all >= lo) & (fdfd_all <= hi)]
    if len(fdfd_win) < len(env_freqs):
        fdfd_win = fdfd_all  # degenerate fallback

    cost = np.abs(env_freqs[:, None] - fdfd_win[None, :])
    rows, cols = linear_sum_assignment(cost)
    d = cost[rows, cols]
    return {
        "n_matched": int(len(rows)),
        "mean_abs_dw": float(d.mean()),
        "max_abs_dw": float(d.max()),
        "mean_rel_bw": float(d.mean() / env_bw) if env_bw > 0 else np.nan,
        "within_1_spacing": int((d <= spacing).sum()),
        "within_2_spacing": int((d <= 2 * spacing).sum()),
        "env_bw": float(env_bw),
        "fdfd_in_window": int(len(fdfd_win)),
        "bw_ratio": float(env_bw / (fdfd_win[:len(env_freqs)].max()
                                    - fdfd_win[:len(env_freqs)].min())),
    }


def main() -> None:
    fdfd_all = np.sort(np.load(FDFD_REF)["freqs"])
    results = {}
    for nrem in [0, 4, 8, 16]:
        p1_path = HERE / f"phase1_nrem{nrem}" / "honeycomb_tm_golden_tm_phase1.npz"
        if not p1_path.exists():
            print(f"n_remote={nrem}: phase1 output missing, skipping")
            continue
        p1 = load_phase1_h5(p1_path)
        Ns = p1["n_reg"]  # 1:1 registry -> moire grid
        band_data = extract_multi_band(p1, BANDS, Ns)

        sol = solve_bottom(band_data, THETA_DEG, Ns, N_MODES)
        m = hungarian_match(sol["freqs"], fdfd_all)
        m["freq_min"] = float(sol["freqs"].min())
        m["freq_max"] = float(sol["freqs"].max())
        m["freqs"] = sol["freqs"].tolist()
        results[nrem] = m
        print(f"n_remote={nrem:2d}: mean|dw|={m['mean_abs_dw']*1e6:7.1f}e-6 "
              f"({100*m['mean_rel_bw']:.2f}% of BW)  max={m['max_abs_dw']*1e6:7.1f}e-6  "
              f"win1sp={m['within_1_spacing']}/{N_MODES}  "
              f"freq=[{m['freq_min']:.4f},{m['freq_max']:.4f}]  BWratio={m['bw_ratio']:.4f}")

    # Eigenvalue drift between consecutive rungs (EA-internal convergence)
    rungs = sorted(results)
    print("\nEA-internal convergence (freq drift between rungs):")
    for a, b in zip(rungs, rungs[1:]):
        fa, fb = np.array(results[a]["freqs"]), np.array(results[b]["freqs"])
        n = min(len(fa), len(fb))
        drift = np.abs(fa[:n] - fb[:n])
        print(f"  n_remote {a:2d} -> {b:2d}: mean drift {drift.mean()*1e6:7.1f}e-6, "
              f"max {drift.max()*1e6:7.1f}e-6")

    out = HERE / "a2_ladder_results.json"
    with open(out, "w") as f:
        json.dump({"theta_deg": THETA_DEG, "n_modes": N_MODES,
                   "fdfd_ref": str(FDFD_REF), "results": results}, f, indent=1)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
