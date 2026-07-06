#!/usr/bin/env python3
"""V2 null test: frozen registry ⇒ zero moiré physics.

Takes a real Phase-1 npz, replaces every registry point's data with the
values at one fixed registry (default: index 0,0), writes a frozen npz,
runs the exact-TM commensurate solve, and checks:

  PASS criteria
  1. Miniband bandwidth collapses (BW << the real run's BW; ideally ~solver tol)
  2. Eigenvalues cluster at the frozen local band value(s) folded by the
     envelope-momentum offsets (levels approach λ_n(δ0) as θ→0)

This validates assembly, tiling, k-folding and mode counting with the moiré
modulation switched off.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", type=Path, required=True)
    ap.add_argument("--index", type=int, nargs=2, default=[0, 0])
    ap.add_argument("--case", type=str, default="114,1")
    ap.add_argument("--n-modes", type=int, default=10)
    args = ap.parse_args()

    d = dict(np.load(args.phase1, allow_pickle=True))
    n_pts = d["eigenvalues"].shape[0]
    n_reg = int(round(np.sqrt(n_pts)))
    flat_idx = args.index[0] * n_reg + args.index[1]

    frozen = {}
    for k, v in d.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == n_pts:
            frozen[k] = np.broadcast_to(arr[flat_idx], arr.shape).copy()
        else:
            frozen[k] = arr
    out_dir = HERE / "v2_frozen"
    out_dir.mkdir(exist_ok=True)
    frozen_npz = out_dir / args.phase1.name
    np.savez(frozen_npz, **frozen)
    lam0 = np.sort(np.asarray(d["eigenvalues"])[flat_idx].real)
    print(f"frozen at registry index {tuple(args.index)}  "
          f"local lambdas: {np.round(lam0, 4)}")

    subprocess.run([sys.executable, str(HERE / "strict_commensurate.py"),
                    "--phase1", str(frozen_npz),
                    "--out", str(out_dir / "phase2"),
                    "--cases", args.case,
                    "--n-modes", str(args.n_modes)], check=True)

    m, n = args.case.split(",")
    r = np.load(out_dir / "phase2" / f"commensurate_m{m}_n{n}.npz")
    f = np.sort(r["frequencies"])
    f0 = np.sqrt(lam0[0]) / (2 * np.pi)
    print(f"frozen-run: {len(f)} modes, f=[{f.min():.6f},{f.max():.6f}], "
          f"BW={f.max()-f.min():.2e}")
    print(f"local band-0 frequency at frozen registry: {f0:.6f}  "
          f"(lowest envelope mode offset: {f.min()-f0:+.2e})")


if __name__ == "__main__":
    main()
