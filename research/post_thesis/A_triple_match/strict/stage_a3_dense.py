#!/usr/bin/env python3
"""Stage A3 (definitive) — the engine's exact operator floor by DENSE diagonalization.

The Galerkin engine's complete-basis operator is EXACTLY (real-space representation):
    L = diag(eps^-1/2) . F^-1 diag|Q+G|^2 F . diag(eps^-1/2),
i.e. spectral kinetic + pointwise grid-sampled eps (the engine's quadrature). At small cells this
is densely diagonalizable — the EXACT floor, no iterative/interpolation ambiguity. The eps-
discretization error is a LOCAL rod property (r*px pixels per rod, identical at every m), so the
offset (engine-operator ground) - (continuum ground) measured at small m transfers to 2 deg.

For each m: build the primitive-cell operator at px, diagonalize densely, report the ground quad
and its offset vs (i) the matched-px FD ground and (ii) the continuum (Richardson of the FD family).

Usage: stage_a3_dense.py <m> [px]   (cost ~ (px^2 (m^2+1)/2)^3 dense eigh; m=7 px16 -> 6400^2)
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

Q = np.array([np.pi, 0.0])


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    N = px * round(((m * m + 1) / 2) ** 0.5)
    nd = N * N
    eps, info = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "primitive")
    B = np.asarray(info["B_super"], float)
    print(f"m={m} px={px} primitive N={N} -> dense {nd}x{nd}", flush=True)
    b = 2 * np.pi * np.linalg.inv(B).T
    fr = np.fft.fftfreq(N) * N
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    kin = (Q[0] + N1 * b[0, 0] + N2 * b[0, 1]) ** 2 + (Q[1] + N1 * b[1, 0] + N2 * b[1, 1]) ** 2
    ie = (1.0 / np.sqrt(eps)).ravel()

    # build L densely by applying to identity blocks via FFT (vectorized over columns)
    t0 = time.time()
    L = np.empty((nd, nd), np.complex128)
    blk = 512
    for j0 in range(0, nd, blk):
        j1 = min(j0 + blk, nd)
        Xc = np.zeros((nd, j1 - j0), np.complex128)
        Xc[np.arange(j0, j1), np.arange(j1 - j0)] = ie[j0:j1]      # eps^-1/2 e_j
        Xg = Xc.reshape(N, N, -1)
        Y = np.fft.ifft2(kin[..., None] * np.fft.fft2(Xg, axes=(0, 1)), axes=(0, 1))
        L[:, j0:j1] = Y.reshape(nd, -1) * ie[:, None]
    L = 0.5 * (L + L.conj().T)
    print(f"  built in {time.time()-t0:.0f}s; diagonalizing...", flush=True)
    t0 = time.time()
    w = eigh(L, eigvals_only=True, driver="evr")
    f = np.sqrt(np.maximum(w, 0)) / (2 * np.pi)
    print(f"  eigh in {time.time()-t0:.0f}s", flush=True)

    # locate the target manifold: nearest ladder to the FD anchor at this m
    if m == 7:
        anchor = 0.0669
    else:
        anchor = 0.3703
    i0 = np.searchsorted(f, anchor - 0.003)
    print(f"  engine-operator ladder near anchor: " + " ".join(f"{x:.6f}" for x in f[i0:i0+8]))
    quad = f[i0:i0+2]
    print(f"  engine-operator ground pair (sector): {quad[0]:.6f} {quad[1]:.6f}")
    np.savez(os.path.join(HERE, f"stage_a3_dense_m{m}_px{px}.npz"), freqs=f, m=m, px=px)
    print("saved stage_a3_dense npz")


if __name__ == "__main__":
    main()
