#!/usr/bin/env python3
"""Parameterized clone of blaze2d run_commensurate_phase2.py (Mar 22 final).

Exact-TM Phase 2 at commensurate angles, 2x2 moire tiling -> 4 k-points,
pooled ladder per angle. Identical config/flow to the thesis runner; only
paths/cases/N_modes are CLI-configurable so we can run per-valley (X, X')
and per-Nb phase-1 inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from lib import phase2_blaze_v4 as p2  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=str, nargs="+", default=["57,1", "114,1"],
                    help="commensurate m,n pairs")
    ap.add_argument("--n-modes", type=int, default=10)
    ap.add_argument("--target-freq", type=float, default=0.241)
    ap.add_argument("--fd-order", type=int, default=4)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_fh = open(args.out / "strict_sweep.log", "w")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    p2._log_fn = log

    p1 = p2.load_phase1_h5(args.phase1)
    eigs = p1["eigenvalues"]
    lambda_ref = float(np.mean(eigs[..., 0]))
    sigma = (2 * math.pi * args.target_freq) ** 2 - lambda_ref
    log(f"phase1={args.phase1}  n_reg={p1['n_reg']}  Nb={p1['n_retained']} "
        f"rem={p1['n_remote']}  lambda_ref={lambda_ref:.6f}  sigma={sigma:.6f}")

    for case in args.cases:
        cm, cn = (int(x) for x in case.split(","))
        theta_deg = p2.commensurate_angle(cm, cn)
        t0 = time.time()
        case_dir = args.out / f"m{cm}_n{cn}"
        case_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "commensurate_mn": (cm, cn),
            "n_modes": args.n_modes,
            "target_band": 0,
            "fd_order": args.fd_order,
            "tm_operator_model": "exact",
            "sigma": sigma,
        }
        summary = p2.process_case(args.phase1, config, case_dir)
        modes_json = case_dir / f"{args.phase1.stem.replace('_phase1', '')}_modes.json"
        with open(modes_json) as f:
            modes = json.load(f)
        freqs = np.array([m["frequency"] for m in modes])
        np.savez(
            args.out / f"commensurate_m{cm}_n{cn}.npz",
            theta_deg=theta_deg, commensurate_m=cm, commensurate_n=cn,
            eta=summary["eta"], lambda_ref=lambda_ref,
            n_modes=args.n_modes, n_k_points=summary["n_k_points"],
            frequencies=freqs,
            k_labels=np.array([m.get("k_label", "k") for m in modes]),
        )
        log(f"({cm},{cn}) theta={theta_deg:.4f}  n_eig={len(freqs)} "
            f"f=[{freqs.min():.6f},{freqs.max():.6f}]  t={time.time()-t0:.1f}s")

    log_fh.close()


if __name__ == "__main__":
    main()
