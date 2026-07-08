#!/usr/bin/env python3
"""Registry-ADAPTED (multi-reference) Galerkin continuum model.

The single-reference Galerkin (galerkin_moire.py) under-populates the small-
angle manifold because the moiré-manifold states have registry-varying Bloch
character that one reference frame cannot span. Here the trial space is
enriched with local Bloch frames from a GRID of reference registries s_k:

    basis = { e^{i p·r} u_n(r; p; s_k) : p = X+½(j₁g₁+j₂g₂), n∈bands, s_k∈grid }

As the registry grid → the full torus this becomes complete (→ FDFD). It is the
momentum-space analogue of the thesis EA's local Bloch basis u_n(r;R), but
carrying the EXACT local dispersion (full Bloch functions, all orders in q) —
so it is not O(η²)-truncated and does not over-bind.

Same generalized eigenproblem H c = λ S c as galerkin_moire; canonical
orthogonalization handles the (now larger) redundancy. Memory-lean: complex64,
chunked matmul, preallocated. Compares to the FDFD X-manifold.
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galerkin_moire as gm            # reuse validated machinery
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = gm.X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=12)
    ap.add_argument("--gcut", type=int, default=3)
    ap.add_argument("--nbands", type=int, default=2)
    ap.add_argument("--band-lo", type=int, default=0)
    ap.add_argument("--nref", type=int, default=2, help="K for a K×K registry grid")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--s-tol", type=float, default=1e-6)
    ap.add_argument("--window", type=float, nargs=2, default=[0.365, 0.385])
    ap.add_argument("--fdfd", default=os.path.join(HERE, "fdfd_xman_2deg.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "galerkin_multiref.npz"))
    args = ap.parse_args()

    g, B_super, theta = gm.moire_g(args.m)
    Ngrid = args.px * round(float(np.sqrt(args.m**2 + 1)))
    J = 2 * args.gcut + 1
    js = np.arange(-J, J + 1)
    momenta = np.array([X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1]
                        for j1 in js for j2 in js])
    bands = list(range(args.band_lo, args.band_lo + args.nbands))
    nb_solve = args.band_lo + args.nbands

    # registry grid (K×K over [0,1)²)
    K = args.nref
    sgrid = [((i + 0.5) / K, (j + 0.5) / K) for i in range(K) for j in range(K)]
    print(f"m={args.m} px={args.px} Ngrid={Ngrid} gcut={args.gcut} "
          f"bands={bands} n_ref={len(sgrid)} momenta={len(momenta)}", flush=True)

    Xc, Yc = gm._supercell_coords(B_super, Ngrid)
    M1, M2 = gm.build_nudft_matrices(args.res, Xc, Yc)
    npix = Ngrid * Ngrid
    n_per_ref = len(momenta) * len(bands)
    Nb = len(sgrid) * n_per_ref
    Psi = np.empty((Nb, npix), np.complex64)
    phaseX = np.exp(-1j * (X[0] * Xc + X[1] * Yc))
    Wh = np.empty((Nb, npix), np.complex64)
    a = 0
    for sk in sgrid:
        u, _, _ = gm.extract_reference_bloch(sk, momenta, nb_solve, args.res)
        coeffs = np.fft.fft2(u, axes=(2, 3))
        for ik in range(len(momenta)):
            for b in bands:
                E = gm.build_field_fourier(coeffs[ik, b], momenta[ik], Xc, Yc, M1, M2)
                Psi[a] = E.ravel().astype(np.complex64)
                Wh[a] = np.fft.fft2(E * phaseX).ravel().astype(np.complex64)
                a += 1
        print(f"  ref {sk} done ({a}/{Nb})", flush=True)

    eps_bl, _ = build_bilayer_eps_asym(args.m, 1, 0.20, 0.10, 8.9, 8.9, 1.0,
                                       Ngrid, Ngrid, 8, "centered")
    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]
    Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kinf = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel().astype(np.float32)
    epsv = eps_bl.ravel().astype(np.float32)
    dA = abs(np.linalg.det(B_super)) / npix
    norm = dA / npix

    H = np.zeros((Nb, Nb), np.complex128)
    S = np.zeros((Nb, Nb), np.complex128)
    CH = 64
    for i0 in range(0, Nb, CH):
        i1 = min(i0 + CH, Nb)
        S[i0:i1] = ((Psi[i0:i1].conj() * epsv[None, :]) @ Psi.T).astype(np.complex128) * dA
        H[i0:i1] = ((Wh[i0:i1].conj() * kinf[None, :]) @ Wh.T).astype(np.complex128) * norm
    del Psi, Wh
    H = 0.5 * (H + H.conj().T); S = 0.5 * (S + S.conj().T)

    sval, svec = eigh(S)
    keep = sval > args.s_tol * sval.max()
    Vp = svec[:, keep] / np.sqrt(sval[keep])
    Hp = Vp.conj().T @ H @ Vp
    Hp = 0.5 * (Hp + Hp.conj().T)
    w = eigh(Hp, eigvals_only=True)
    fvals = np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))

    lo, hi = args.window
    win = fvals[(fvals >= lo) & (fvals <= hi)]
    fd = np.sort(np.load(args.fdfd)["freqs_xmanifold"])
    np.savez(args.out, freqs=fvals, m=args.m, nref=len(sgrid), n_basis=Nb,
             n_kept=int(keep.sum()), gcut=args.gcut, nbands=args.nbands)
    print(f"S-rank {int(keep.sum())}/{Nb} | in-window [{lo},{hi}]: "
          f"Galerkin {len(win)} vs FDFD X-manifold {len(fd)}", flush=True)
    print("  Galerkin window:", " ".join(f"{x:.5f}" for x in win[:16]), flush=True)
    print("  FDFD X-manifold:", " ".join(f"{x:.5f}" for x in fd[:8]), flush=True)
    if len(win) >= 4 and len(fd) >= 4:
        n = min(len(win), len(fd))
        off = win[0] - fd[0]
        print(f"  edge Δ {off:+.2e} | de-trended mean "
              f"{np.abs((win[:n]-off)-fd[:n]).mean():.2e}", flush=True)


if __name__ == "__main__":
    main()
