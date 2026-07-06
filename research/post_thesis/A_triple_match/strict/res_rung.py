#!/usr/bin/env python3
"""Resolution rung: March reg-128 exact phase-1 archive at Ns=128, m57, Γ-lane."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from lib import phase2_blaze_v4 as p2  # noqa: E402

p1_path = Path("/home/renlephy/msl/research/post_thesis/A_triple_match/strict/"
               "phase1_x_reg128_slim/square_x_tm_phase1.npz")
p1 = p2.load_phase1_h5(p1_path)
print("n_reg:", p1["n_reg"], "Nb:", p1["n_retained"], "rem:", p1["n_remote"])
lam_ref = float(np.mean(p1["eigenvalues"][..., 0]))
sigma = (2 * math.pi * 0.241) ** 2 - lam_ref
out = HERE / "phase2_x_reg128_ns128"
out.mkdir(exist_ok=True)
cfg = {"commensurate_mn": (57, 1), "n_modes": 10, "target_band": 0,
       "fd_order": 4, "tm_operator_model": "exact", "sigma": sigma,
       "k_s": (0.0, 0.0), "Ns": 128}
t0 = time.time()
p2.process_case(p1_path, cfg, out / "m57_G")
print("elapsed", round(time.time() - t0, 1))
modes = json.load(open(out / "m57_G" / "square_x_tm_modes.json"))
f = np.sort([m["frequency"] for m in modes])
np.savez(out / "m57_G_freqs.npz", frequencies=f)
print("reg128/Ns128 m57 G:", np.round(f, 6))
