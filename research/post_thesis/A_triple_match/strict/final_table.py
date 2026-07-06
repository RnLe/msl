#!/usr/bin/env python3
"""Final strict comparison table: EA (per Nb rung, 4-lane, valley-doubled)
vs FDFD reference lanes, per angle. Window = FDFD reference span clipped to
EA coverage; counts reported; index-aligned residuals; no matching.

Writes strict_final_table.json and prints a markdown table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FD = Path("/home/renlephy/msl/research/studies/fdfd_convergence/data_x_tm")

ANGLES = [
    ("m29", "4deg", 32, 8),    # (case tag, fdfd label, ref res, drift res)
    ("m57", "2deg", 32, 16),
    ("m114", "1deg", 16, 8),
]

# EA sources per rung: {rung label: dir with commensurate_m{m}_n1.npz}
RUNGS = {
    "Nb2+6rem": HERE / "phase2_x_reg128_kfold25",       # m57, m114 (25/lane)
    "Nb2+6rem_m29": HERE / "phase2_x_2r6_reg128_m29",   # m29 (10/lane)
    "Nb4": HERE / "phase2_x_4r0_reg128",
    "Nb6": HERE / "phase2_x_6r0_reg128",
    "Nb8": HERE / "phase2_x_8r0_reg128",
}


def load(path):
    return np.sort(np.load(path, allow_pickle=True)["frequencies"]) \
        if path.exists() else None


def main() -> None:
    out = {}
    rows = []
    for tag, lab, res_ref, res_lo in ANGLES:
        m = tag[1:]
        fd_ref = np.sort(np.load(FD / f"fdfd_tm_x_{lab}_res{res_ref}_fEActr.npz")["freqs"])
        fd_lo = np.sort(np.load(FD / f"fdfd_tm_x_{lab}_res{res_lo}_fEActr.npz")["freqs"])
        n0 = min(len(fd_ref), len(fd_lo))
        drift = float(np.abs(fd_ref[:n0] - fd_lo[:n0]).mean())

        for rung, d in RUNGS.items():
            ea = load(d / f"commensurate_m{m}_n1.npz")
            if ea is None:
                continue
            ea2 = np.sort(np.repeat(ea, 2))  # valley doubling (FDFD-exact pairs)
            lo = max(ea2.min(), fd_ref.min()) - 1e-9
            hi = min(ea2.max(), fd_ref.max()) + 1e-9
            eaw = ea2[(ea2 >= lo) & (ea2 <= hi)]
            fdw = fd_ref[(fd_ref >= lo) & (fd_ref <= hi)]
            n = min(len(eaw), len(fdw))
            if n == 0:
                continue
            diff = np.abs(eaw[:n] - fdw[:n])
            rel = diff / fdw[:n]
            row = {
                "angle": lab, "rung": rung,
                "n_ea": int(len(eaw)), "n_fdfd": int(len(fdw)),
                "count_equal": bool(len(eaw) == len(fdw)),
                "mean_abs": float(diff.mean()), "max_abs": float(diff.max()),
                "mean_rel_pct": float(100 * rel.mean()),
                "fdfd_drift_mean": drift,
                "below_fdfd_drift": bool(diff.mean() < drift),
            }
            rows.append(row)
            out.setdefault(lab, []).append(row)

    print("| θ | EA rung | nEA/nFDFD | mean|Δf| | max | mean rel | FDFD drift | EA<drift |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['angle']} | {r['rung']} | {r['n_ea']}/{r['n_fdfd']}"
              f"{'' if r['count_equal'] else ' ⚠'} | {r['mean_abs']:.2e} "
              f"| {r['max_abs']:.2e} | {r['mean_rel_pct']:.4f}% "
              f"| {r['fdfd_drift_mean']:.2e} | {'✓' if r['below_fdfd_drift'] else '—'} |")

    with open(HERE / "strict_final_table.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved strict_final_table.json")


if __name__ == "__main__":
    main()
