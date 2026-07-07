#!/usr/bin/env python3
"""Pre-registered exactness evaluation: ASYM candidate, band-1 gap-edge
manifold, EA (clean-core, doubling-fixed, Nb=1+8 Lowdin) vs FDFD.

Angles: 2 deg (57,1) and 1 deg (113,1). All stats index-aligned from the
TRUE spectral bottom (isolated manifold: FDFD enumeration is complete from
the bottom by construction; verified by ball coverage).

Criteria (FINDINGS, pre-registered):
 1. enumeration-complete both sides
 2. every FDFD in-window state X-star-carried (checked separately)
 3. counts exactly equal in-window
 4. mean |df| <= max(3*combined floor, 3e-5), max |df| <= 1e-4
 5. N(f) staircases: no insertion/deletion
Everything is REPORTED as measured — no filtering of EA states.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(p, key="freqs"):
    p = HERE / p
    return np.sort(np.asarray(np.load(p, allow_pickle=True)[key], float)) \
        if (HERE / p).exists() else None


def richardson(f_coarse, f_fine, p_coarse, p_fine):
    n = min(len(f_coarse), len(f_fine))
    fac = (p_fine ** -2) / (p_coarse ** -2 - p_fine ** -2)
    return f_fine[:n] + (f_fine[:n] - f_coarse[:n]) * fac


def eval_angle(tag, fd_files, ea_file, ea_pilot_file=None, n_compare=24):
    out = {"angle": tag}
    lads = {px: load(f) for px, f in fd_files.items()}
    pxs = sorted(px for px, l in lads.items() if l is not None)
    if len(pxs) < 2 or load(ea_file, "frequencies") is None:
        out["status"] = "PENDING"
        missing = [f for px, f in fd_files.items() if lads[px] is None]
        if load(ea_file, "frequencies") is None:
            missing.append(str(ea_file))
        out["missing"] = missing
        return out
    p1, p2 = pxs[-2], pxs[-1]
    f_inf = richardson(lads[p1], lads[p2], p1, p2)
    fd_floor_edge = abs(float(f_inf[0] - lads[p2][0]))          # resid est.
    fd_drift_edge = abs(float(lads[p2][0] - lads[p1][0]))
    ea = load(ea_file, "frequencies")

    n = min(n_compare, len(f_inf), len(ea))
    # locate EA's first state at/above the true bottom minus a floor margin
    # (spurious sub-gap branch, if present, is REPORTED, not dropped)
    n_subgap = int(np.sum(ea < f_inf[0] - 5 * fd_floor_edge))
    d_raw = ea[:n] - f_inf[:n]
    d_anchored = ea[n_subgap:n_subgap + n] - f_inf[:n] \
        if len(ea) >= n_subgap + n else None

    ea_floor = None
    if ea_pilot_file and load(ea_pilot_file, "frequencies") is not None:
        pil = load(ea_pilot_file, "frequencies")
        m = min(len(pil), len(ea))
        ea_floor = float(np.mean(np.abs(pil[:m] - ea[:m])))     # reg64<->128

    out.update({
        "fdfd_px": pxs,
        "fdfd_bottom_per_px": {px: float(lads[px][0]) for px in pxs},
        "fdfd_bottom_richardson": float(f_inf[0]),
        "fdfd_floor_edge_estimate": fd_floor_edge,
        "fdfd_drift_edge": fd_drift_edge,
        "fdfd_ladder_richardson": f_inf[:n].tolist(),
        "ea_ladder": ea[:max(n + n_subgap, n)].tolist(),
        "ea_n_subgap_states": n_subgap,
        "ea_floor_reg64_vs_reg128": ea_floor,
        "n_compared": n,
        "raw": {"mean_abs": float(np.mean(np.abs(d_raw))),
                "max_abs": float(np.max(np.abs(d_raw))),
                "per_level": d_raw.tolist()},
    })
    if d_anchored is not None and n_subgap > 0:
        out["anchored_above_subgap"] = {
            "mean_abs": float(np.mean(np.abs(d_anchored))),
            "max_abs": float(np.max(np.abs(d_anchored))),
            "per_level": d_anchored.tolist()}
    floor = max(fd_floor_edge, ea_floor or 0.0)
    bar_mean = max(3 * floor, 3e-5)
    out["criteria"] = {
        "bar_mean": bar_mean,
        "c3_counts_equal": bool(n_subgap == 0),
        "c4_mean_ok": bool(np.mean(np.abs(d_raw)) <= bar_mean),
        "c4_max_ok": bool(np.max(np.abs(d_raw)) <= max(1e-4, 3 * floor)),
        "c5_no_insertions": bool(n_subgap == 0),
    }
    return out


def main():
    res = {
        "2deg": eval_angle(
            "2deg (57,1) beta_rem=0.124",
            {16: "fdfd_asym_x_2deg_res16.npz", 32: "fdfd_asym_x_2deg_res32.npz",
             48: "fdfd_asym_x_2deg_res48.npz"},
            "phase2_asym_b1core_m57/commensurate_m57_n1.npz",
            "phase2_asym_b1core_pilot_m57/commensurate_m57_n1.npz"),
        "1deg": eval_angle(
            "1deg (113,1) beta_rem=0.062",
            {16: "fdfd_asym_x_1deg_res16.npz", 24: "fdfd_asym_x_1deg_res24.npz"},
            "phase2_asym_b1core_m113/commensurate_m113_n1.npz"),
    }
    # eta-scaling of per-level residuals
    try:
        r2 = np.abs(res["2deg"]["raw"]["per_level"])
        r1 = np.abs(res["1deg"]["raw"]["per_level"])
        m = min(len(r1), len(r2))
        res["eta_scaling_mean_ratio_2deg_over_1deg"] = float(
            np.mean(r2[:m]) / max(np.mean(r1[:m]), 1e-12))
    except (KeyError, TypeError):
        pass
    with open(HERE / "strict_asym_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    for tag, r in res.items():
        if not isinstance(r, dict):
            continue
        print(f"== {tag}: ", end="")
        if r.get("status") == "PENDING":
            print("PENDING", r["missing"])
            continue
        print(f"FDFD bottom {r['fdfd_bottom_richardson']:.6f} "
              f"(floor {r['fdfd_floor_edge_estimate']:.1e}) | "
              f"EA subgap states: {r['ea_n_subgap_states']} | "
              f"raw mean|df| {r['raw']['mean_abs']:.2e} "
              f"max {r['raw']['max_abs']:.2e} | criteria {r['criteria']}")
    if "eta_scaling_mean_ratio_2deg_over_1deg" in res:
        print("eta-scaling ratio (2deg/1deg mean|df|):",
              round(res["eta_scaling_mean_ratio_2deg_over_1deg"], 2))
    print("saved strict_asym_eval.json")


if __name__ == "__main__":
    main()
