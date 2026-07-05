#!/usr/bin/env python3
"""B1+B2: repaired magic-angle sweep for the golden honeycomb Dirac crystal.

B1 — bandwidth vs θ with the COMPLETE Hamiltonian (Dirac pair + Löwdin mass
     from n_remote=16, Born–Huang active, registry 64), fixing the gaps of
     the thesis sweep (which used 0 remote bands and zeroed Born–Huang).
B2 — Dirac velocity renormalization v*(θ): slopes of the two central
     minibands along Γ_s→K_s near the moiré Dirac point. The BM magic-angle
     criterion is v* → 0 — a far sharper observable than RMS bandwidth.

Phase 1 input is angle-independent: reuses A_triple_match/phase1_nrem16.
Checkpointed per (θ, k_s); safe to re-run.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

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

P1_PATH = HERE.parent / "A_triple_match/phase1_nrem16/honeycomb_tm_golden_tm_phase1.npz"
CKPT_DIR = HERE / "ckpt"
BANDS = [0, 1]
N_MODES = 12          # minibands around the Dirac region
OMEGA_D = 0.2744      # monolayer Dirac frequency (verified)

# θ grid: dense below 1.5°, coarser above
THETAS = sorted(set(
    list(np.round(np.linspace(0.3, 1.5, 13), 3)) +
    list(np.round(np.linspace(1.75, 3.0, 6), 3))
))

# k_s path Γ_s -> K_s in fractional moiré-BZ coords (K_s = (1/3, 2/3) frac,
# same corner convention as the monolayer — moiré lattice is hexagonal too).
KPATH = [(0.0, 0.0), (1/9, 2/9), (2/9, 4/9), (1/3, 2/3)]
K_LABELS = ["G", "p1", "p2", "K"]
DK_FRAC = 0.02        # FD step around K_s for slopes (fractional)


def build_H(band_data, theta_deg, Ns, k_s):
    Nb = band_data["Nb"]
    moire = compute_moire_metadata("honeycomb", 1.0, theta_deg)
    B_inv = np.linalg.inv(moire["B_moire"])
    eig = band_data["eigenvalues"]
    lambda_ref = float(np.mean(eig))
    Lambda = np.zeros((Ns, Ns, Nb, Nb), dtype=complex)
    for n in range(Nb):
        Lambda[..., n, n] = eig[..., n] - lambda_ref
    v1, v2 = transform_velocity(band_data["velocity_x"], band_data["velocity_y"], B_inv)
    M11, M12, M21, M22 = transform_mass_tensor(
        band_data["mass_xx"], band_data["mass_xy"],
        band_data["mass_yx"], band_data["mass_yy"], B_inv)
    bh_factor = born_huang_metric_factor(moire["B_moire"])
    bh = band_data["born_huang"] * bh_factor if band_data["born_huang"] is not None else None
    sc = band_data["slow_coefficient"] * bh_factor if band_data["slow_coefficient"] is not None else None
    H = assemble_hamiltonian(
        Lambda, v1, v2, M11, M12, M22,
        band_data["berry_x"], band_data["berry_y"], bh, sc,
        Ns, Nb, include_drift=True, include_kinetic=True,
        include_born_huang=True, include_slow_coeff=sc is not None,
        fd_order=4, k_s=k_s)
    return H, Lambda, lambda_ref


def solve_at(band_data, theta_deg, Ns, k_s, lam_target):
    """Solve N_MODES envelope modes nearest the Dirac region (sigma at λ_D)."""
    H, Lambda, lambda_ref = build_H(band_data, theta_deg, Ns, k_s)
    sigma = lam_target - lambda_ref
    evals, _ = solve_envelope(H, N_MODES, sigma)
    evals = np.sort(evals.real)
    freqs = np.array([eigenvalue_to_frequency(l, lambda_ref) for l in evals])
    return freqs


def ck_key(theta, k_s):
    return f"t{theta:.3f}_k{k_s[0]:.4f}_{k_s[1]:.4f}.npz"


def solve_cached(band_data, theta, Ns, k_s, lam_target):
    CKPT_DIR.mkdir(exist_ok=True)
    p = CKPT_DIR / ck_key(theta, k_s)
    if p.exists():
        return np.load(p)["freqs"]
    t0 = time.time()
    freqs = solve_at(band_data, theta, Ns, tuple(k_s), lam_target)
    np.savez(p, freqs=freqs, theta=theta, k_s=k_s)
    print(f"    solved θ={theta:.3f} k_s=({k_s[0]:.3f},{k_s[1]:.3f}) "
          f"in {time.time()-t0:.1f}s  f∈[{freqs.min():.4f},{freqs.max():.4f}]")
    return freqs


def main():
    p1 = load_phase1_h5(P1_PATH)
    Ns = p1["n_reg"]
    band_data = extract_multi_band(p1, BANDS, Ns)
    lam_target = (2 * np.pi * OMEGA_D) ** 2

    # Monolayer Dirac velocity |v_01| at the registry with minimal local gap
    eig = p1["eigenvalues"]
    gap = np.abs(eig[..., 1] - eig[..., 0])
    i, j = np.unravel_index(np.argmin(gap), gap.shape)
    v01 = abs(p1["velocity_x"][i, j, 0, 1]) + 1j * 0
    vD_lambda = float(abs(p1["velocity_x"][i, j, 0, 1]) ** 2
                      + abs(p1["velocity_y"][i, j, 0, 1]) ** 2) ** 0.5
    print(f"Registry min-gap at ({i},{j}), gap={gap[i, j]:.2e}, "
          f"|v_D| (λ per k-unit) = {vD_lambda:.4f}")

    results = []
    for theta in THETAS:
        # B1: minibands along the k-path -> per-mode bandwidth
        F = np.stack([solve_cached(band_data, theta, Ns, k, lam_target) for k in KPATH])
        per_mode_bw = F.max(axis=0) - F.min(axis=0)      # (N_MODES,)

        # B2: slopes at K_s via central FD along the path direction
        kK = np.array(KPATH[-1])
        dvec = kK / np.linalg.norm(kK)
        kp = tuple(kK + DK_FRAC * dvec)
        km = tuple(kK - DK_FRAC * dvec)
        Fp = solve_cached(band_data, theta, Ns, kp, lam_target)
        Fm = solve_cached(band_data, theta, Ns, km, lam_target)
        FK = F[-1]
        # central pair = two modes closest to ω_D at K_s
        order = np.argsort(np.abs(FK - OMEGA_D))
        c0, c1 = sorted(order[:2])
        # frac k-step -> physical: |Δk| = DK_FRAC * |B_moire·dvec_frac| — report
        # slopes in frequency per fractional step; ratio v*/v_D uses same units
        slope = (np.abs(Fp - Fm) / (2 * DK_FRAC))
        vstar = float(0.5 * (slope[c0] + slope[c1]))
        gapK = float(FK[c1] - FK[c0])

        eta = compute_moire_metadata("honeycomb", 1.0, theta)["eta"]
        rec = {
            "theta": float(theta), "eta": float(eta),
            "bw_central_pair": float(0.5 * (per_mode_bw[c0] + per_mode_bw[c1])),
            "bw_rms": float(np.sqrt((per_mode_bw ** 2).mean())),
            "vstar_freq_per_frack": vstar,
            "gap_at_Ks": gapK,
            "central_modes": [int(c0), int(c1)],
        }
        results.append(rec)
        print(f"θ={theta:5.3f}  η={eta:.5f}  BW_pair={rec['bw_central_pair']:.3e} "
              f"BW_rms={rec['bw_rms']:.3e}  v*={vstar:.4e}  gap(K_s)={gapK:.3e}")

    out = HERE / "b1b2_results.json"
    with open(out, "w") as f:
        json.dump({"phase1": str(P1_PATH), "Ns": int(Ns), "n_modes": N_MODES,
                   "kpath": KPATH, "dk_frac": DK_FRAC,
                   "vD_lambda_units": vD_lambda, "results": results}, f, indent=1)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
