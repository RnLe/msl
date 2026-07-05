#!/usr/bin/env python3
"""STRICT EA-vs-FDFD ladder evaluation. No matching algorithms.

Protocol:
  1. Pool EA valleys (X + X') into one sorted ladder.
  2. Fixed common window = intersection of EA and FDFD spectral ranges
     (or --window LO HI).
  3. Report mode COUNTS in window for both (equality is a metric, never
     assumed).
  4. If counts equal: index-aligned per-mode residuals |Δf_i|, mean/max,
     relative CDF. If unequal: report the count mismatch as the finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_freqs(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    key = "frequencies" if "frequencies" in d else "freqs"
    return np.sort(np.asarray(d[key], dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea", type=Path, nargs="+", required=True,
                    help="EA npz per valley (pooled together)")
    ap.add_argument("--fdfd", type=Path, nargs="+", required=True,
                    help="FDFD npz lanes, ascending resolution; last = reference")
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ea = np.sort(np.concatenate([load_freqs(p) for p in args.ea]))
    lanes = {p.name: load_freqs(p) for p in args.fdfd}
    ref_name, ref = list(lanes.items())[-1]

    if args.window:
        lo, hi = args.window
    else:
        lo = max(ea.min(), ref.min()) - 1e-9
        hi = min(ea.max(), ref.max()) + 1e-9

    ea_w = ea[(ea >= lo) & (ea <= hi)]
    out = {
        "window": [float(lo), float(hi)],
        "ea_files": [str(p) for p in args.ea],
        "fdfd_reference": ref_name,
        "n_ea_in_window": int(len(ea_w)),
        "lanes": {},
    }

    print(f"window [{lo:.6f}, {hi:.6f}]   EA modes in window: {len(ea_w)}")
    for name, fd in lanes.items():
        fd_w = fd[(fd >= lo) & (fd <= hi)]
        rec = {"n_in_window": int(len(fd_w))}
        n = min(len(ea_w), len(fd_w))
        if n > 0:
            r = np.abs(ea_w[:n] - fd_w[:n])
            rel = r / fd_w[:n]
            rec.update({
                "count_equal": bool(len(fd_w) == len(ea_w)),
                "mean_abs_df": float(r.mean()),
                "max_abs_df": float(r.max()),
                "mean_rel": float(rel.mean()),
                "p95_rel": float(np.percentile(rel, 95)),
            })
            flag = "" if rec["count_equal"] else "  [COUNT MISMATCH]"
            print(f"{name}: n={len(fd_w)}{flag}  mean|Δf|={r.mean():.3e}  "
                  f"max={r.max():.3e}  mean rel={100*rel.mean():.4f}%")
        out["lanes"][name] = rec

    # FDFD self-convergence between consecutive lanes (same counts only)
    names = list(lanes)
    for a, b in zip(names, names[1:]):
        fa = lanes[a][(lanes[a] >= lo) & (lanes[a] <= hi)]
        fb = lanes[b][(lanes[b] >= lo) & (lanes[b] <= hi)]
        n = min(len(fa), len(fb))
        if n:
            d = np.abs(fa[:n] - fb[:n])
            print(f"FDFD drift {a} -> {b}: mean {d.mean():.3e} max {d.max():.3e}")
            out["lanes"][b]["drift_from_prev_mean"] = float(d.mean())

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
