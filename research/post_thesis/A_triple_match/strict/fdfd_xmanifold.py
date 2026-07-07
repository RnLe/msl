#!/usr/bin/env python3
"""Extract the X-band-manifold FDFD ladder: solve the ASYM twisted supercell,
classify each eigenstate by X-star Fourier weight, keep only w_X+w_X' > THR
(the genuine band-1-at-X moire states, excluding band-0/background leakage),
and save the filtered ladder for the strict EA comparison.

Usage: fdfd_xmanifold.py <m> <px> <sigma_omega> <nmodes> <out.npz>
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, HERE)
from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

m = int(sys.argv[1]); px = int(sys.argv[2]); sigw = float(sys.argv[3])
nmodes = int(sys.argv[4]); out = sys.argv[5]
n = 1; THR = 0.5; R_CUT = 0.45
N = px * round((m * m + n * n) ** 0.5)
eps, info = build_bilayer_eps_asym(m, n, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "centered")
Q = np.array([np.pi, 0.0])
L = build_fdfd_operator(eps, info, Q, "tm")
sig = (2 * np.pi * sigw) ** 2
nd = L.shape[0]
t0 = time.time()
fac = cholesky((L - sig * sp.eye(nd, format="csc")).tocsc(), beta=0, mode="simplicial")
op = LinearOperator((nd, nd), matvec=fac, dtype=L.dtype)
vals, vecs = eigsh(L, k=nmodes, sigma=sig, which="LM", OPinv=op, maxiter=20000, tol=1e-9)
f = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
o = np.argsort(f)
B = np.asarray(info["B_super"], float); Gr = 2 * np.pi * np.linalg.inv(B).T
g1 = np.fft.fftfreq(N) * N; G1, G2 = np.meshgrid(g1, g1, indexing="ij")
qx = G1 * Gr[0, 0] + G2 * Gr[0, 1]; qy = G1 * Gr[1, 0] + G2 * Gr[1, 1]
qxr = (qx + np.pi) % (2 * np.pi) - np.pi; qyr = (qy + np.pi) % (2 * np.pi) - np.pi
dX = np.hypot(np.pi - np.abs(qxr), qyr); dXp = np.hypot(qxr, np.pi - np.abs(qyr))
fX, wX_all = [], []
for i in o:
    P = np.abs(np.fft.fft2(vecs[:, i].reshape(N, N))) ** 2; P /= P.sum()
    w = float(P[dX < R_CUT].sum() + P[dXp < R_CUT].sum())
    wX_all.append((float(f[i]), w))
    if w > THR:
        fX.append(float(f[i]))
fX = np.sort(fX)
np.savez(out, freqs_xmanifold=fX, all_freqs=np.sort(f),
         wX=np.array(wX_all), m=m, px=px, sigma_omega=sigw, thr=THR)
print(f"m{m} px{px}: {len(f)} total, {len(fX)} X-manifold (w>{THR}), "
      f"t={time.time()-t0:.0f}s  X-ladder [{fX.min():.6f},{fX.max():.6f}]")
print("X-manifold first 16:", " ".join(f"{x:.6f}" for x in fX[:16]))
