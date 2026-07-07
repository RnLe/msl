#!/usr/bin/env python3
"""Assemble a phase-1 npz from blaze checkpoint rows, working around the
deterministic 'stack smashing detected' crash in EAExtractor when
load_checkpoint_row is called ~64 times in one process: rows are loaded in
small-chunk SUBPROCESSES (verified safe), converted to blocks, and stacked
here.

Usage: assemble_checkpoints.py <checkpoint_dir> <out_npz> <n_ret> <n_rem>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

CHUNK = 4
DROP_SUB = ("hermitized_eigenvectors", "_r_derivatives_", "_r_second_derivatives_")

WORKER = r'''
import sys, numpy as np
sys.path.insert(0, {libdir!r})
from phase1_blaze_v4 import EAExtractor, raw_to_blocks
DROP_SUB = {drop!r}
out = {{}}
count = 0
for rf in {rows!r}:
    for raw in EAExtractor.load_checkpoint_row(rf):
        if hasattr(raw, 'pop'):
            raw.pop("eigenvectors", None); raw.pop("grid_dims", None)
        block = raw_to_blocks(raw, {n_ret}, {n_rem})
        block.pop("eigenvectors_raw", None); block.pop("grid_dims", None)
        for k, v in block.items():
            if any(s in k for s in DROP_SUB):
                continue
            out.setdefault(k, []).append(v)
        count += 1
np.savez({out_npz!r}, **{{k: np.asarray(v) for k, v in out.items()
                          if isinstance(v[0], (np.ndarray, int, float, complex, np.generic))}})
print("worker ok", count)
'''


def main():
    ckpt_dir, out_npz, n_ret, n_rem = (Path(sys.argv[1]), Path(sys.argv[2]),
                                       int(sys.argv[3]), int(sys.argv[4]))
    libdir = str(Path(__file__).resolve().parents[2] / "lib")
    meta = json.loads((ckpt_dir / "sweep_meta.json").read_text())
    total = meta["total_points"]
    rows = sorted(str(p) for p in ckpt_dir.glob("row_*.json")
                  if not p.name.endswith(".tmp"))
    print(f"{len(rows)} rows, {total} points expected")

    parts = []
    with tempfile.TemporaryDirectory() as td:
        for ci in range(0, len(rows), CHUNK):
            chunk = rows[ci:ci + CHUNK]
            part = f"{td}/part_{ci:04d}.npz"
            code = WORKER.format(libdir=libdir, drop=DROP_SUB, rows=chunk,
                                 n_ret=n_ret, n_rem=n_rem, out_npz=part)
            r = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"chunk {ci} FAILED rc={r.returncode}\n{r.stderr[-500:]}")
                sys.exit(1)
            parts.append(part)
            print(f"  chunk {ci // CHUNK + 1}/{(len(rows) + CHUNK - 1) // CHUNK} ok")

        stacked = {}
        for part in parts:
            d = np.load(part, allow_pickle=False)
            for k in d.files:
                stacked.setdefault(k, []).append(d[k])
        arrays = {k: np.concatenate(v, axis=0) for k, v in stacked.items()}

    n = arrays["eigenvalues"].shape[0]
    assert n == total, f"stacked {n} != expected {total}"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    mb = out_npz.stat().st_size / 1e6
    print(f"saved {out_npz} ({mb:.1f} MB), {n} points, "
          f"keys: {sorted(arrays)[:8]}... ({len(arrays)} total)")


if __name__ == "__main__":
    main()
