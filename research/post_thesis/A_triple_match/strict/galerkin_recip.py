#!/usr/bin/env python3
"""Reciprocal-space (plane-wave) Galerkin moiré continuum model — the
memory-robust, accurate engine for the registry-adapted exact continuum model.

Each reference-Bloch basis function is stored as SPARSE supercell plane-wave
coefficients (~res² nonzeros, not the full Ngrid² real-space field), so a
large multi-reference basis fits in <1 GB even at px16. The ε_bl coupling is
computed by per-basis FFT-convolution (one transient grid at a time). Same
generalized eigenproblem H c = λ S c as galerkin_moire.py, to which it is
numerically identical (validated), but scalable to registry-adapted bases.

Basis: work with the supercell-periodic amplitude w = e^{-iX·r}E of each
E_{n,p,s_k}=e^{ip·r}u_n(r;p;s_k). Then
    ĉ^b(Gs) = c_n(g;p;s_k)   at   Gs = (p−X) + g   (supercell reciprocal),
    H_αβ = Σ_Gs ĉ^α*(Gs) |X+Gs|² ĉ^β(Gs)             (kinetic, diagonal),
    S_αβ = ⟨w_α|ε_bl|w_β⟩ = ĉ^α* · FFT(ε_bl · IFFT(ĉ^β)).
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galerkin_moire as gm
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = gm.X


def basis_coeffs(u_kb, p, B_super, Ngrid):
    """Sparse supercell-reciprocal coeffs of w=e^{-iX·r}E for one basis fn.
    Returns (flat_indices, values) on the Ngrid×Ngrid FFT grid."""
    res = u_kb.shape[0]
    c = np.fft.fft2(u_kb) / (res * res)                 # monolayer coeffs c_n(g)
    gi = np.fft.fftfreq(res) * res                       # integer g indices
    G1, G2 = np.meshgrid(gi, gi, indexing="ij")
    gx = 2 * np.pi * G1
    gy = 2 * np.pi * G2                                  # monolayer recip cartesian
    Qx = (p[0] - X[0]) + gx                              # (p-X)+g cartesian
    Qy = (p[1] - X[1]) + gy
    # supercell reciprocal integer index: n = B_super^T Q / (2π)
    n1 = np.rint((B_super[0, 0] * Qx + B_super[1, 0] * Qy) / (2 * np.pi)).astype(int)
    n2 = np.rint((B_super[0, 1] * Qx + B_super[1, 1] * Qy) / (2 * np.pi)).astype(int)
    i1 = n1 % Ngrid
    i2 = n2 % Ngrid
    flat = (i1 * Ngrid + i2).ravel()
    vals = c.ravel()
    # collapse duplicate indices (different g mapping to same supercell G)
    order = np.argsort(flat)
    flat_s = flat[order]; vals_s = vals[order]
    uniq, start = np.unique(flat_s, return_index=True)
    summed = np.add.reduceat(vals_s, start)
    return uniq, summed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--gcut", type=int, default=4)
    ap.add_argument("--nbands", type=int, default=2)
    ap.add_argument("--band-lo", type=int, default=0)
    ap.add_argument("--nref", type=int, default=1, help="K for K×K registry grid")
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.23046875, 0.10546875])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--s-tol", type=float, default=1e-6)
    ap.add_argument("--window", type=float, nargs=2, default=[0.365, 0.385])
    ap.add_argument("--fdfd", default=os.path.join(HERE, "fdfd_xman_2deg.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "galerkin_recip.npz"))
    args = ap.parse_args()

    g, B_super, theta = gm.moire_g(args.m)
    Ngrid = args.px * round(float(np.sqrt(args.m**2 + 1)))
    npix = Ngrid * Ngrid
    J = 2 * args.gcut + 1
    js = np.arange(-J, J + 1)
    momenta = np.array([X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1]
                        for j1 in js for j2 in js])
    bands = list(range(args.band_lo, args.band_lo + args.nbands))
    nb_solve = args.band_lo + args.nbands
    if args.nref <= 1:
        sgrid = [tuple(args.sbar)]
    else:
        K = args.nref
        sgrid = [((i + 0.5) / K, (j + 0.5) / K) for i in range(K) for j in range(K)]
    print(f"m={args.m} px={args.px} Ngrid={Ngrid} gcut={args.gcut} bands={bands} "
          f"n_ref={len(sgrid)} momenta={len(momenta)}", flush=True)

    # build sparse coeff matrix C (nPW × Nb)
    rows, cols, data = [], [], []
    b = 0
    for sk in sgrid:
        u, _, _ = gm.extract_reference_bloch(sk, momenta, nb_solve, args.res)
        for ik in range(len(momenta)):
            for bd in bands:
                idx, val = basis_coeffs(u[ik, bd], momenta[ik], B_super, Ngrid)
                rows.append(idx); cols.append(np.full(idx.size, b)); data.append(val)
                b += 1
        print(f"  ref {sk} done ({b} basis)", flush=True)
    Nb = b
    C = sp.csc_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(npix, Nb), dtype=np.complex128)
    print(f"  sparse C: {C.nnz} nnz, {C.nnz/Nb:.0f}/basis", flush=True)

    # kinetic |X+Gs|² on the FFT grid
    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]
    Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()
    dA = abs(np.linalg.det(B_super))                  # cell area; Parseval below

    eps_bl, _ = build_bilayer_eps_asym(args.m, 1, 0.20, 0.10, 8.9, 8.9, 1.0,
                                       Ngrid, Ngrid, 8, "centered")

    # H = C† diag(kin) C  · dA   (Parseval: Σ_G â*b̂ · area = ⟨a|b⟩ for unit-norm FFT)
    Ck = C.multiply(kin[:, None])
    H = (C.conj().T @ Ck).toarray() * dA
    # S via per-basis FFT-convolution:  S[:,β] = C† FFT(ε_bl · IFFT(ĉ^β)) · dA
    # (one sparse column -> transient grid at a time; peak memory ~ one grid)
    S = np.zeros((Nb, Nb), np.complex128)
    Ccsr = C.tocsr()
    Ch = C.conj().T.tocsr()                            # for fast C† @ vec
    for bcol in range(Nb):
        col = np.asarray(C.getcol(bcol).todense()).reshape(Ngrid, Ngrid)
        w = np.fft.ifft2(col) * npix                   # w_β(r)
        Meps = (np.fft.fft2(eps_bl * w).ravel() / npix)
        S[:, bcol] = (Ch @ Meps) * dA
    H = 0.5 * (H + H.conj().T); S = 0.5 * (S + S.conj().T)

    sval, svec = eigh(S)
    keep = sval > args.s_tol * sval.max()
    Vp = svec[:, keep] / np.sqrt(sval[keep])
    Hp = 0.5 * (Vp.conj().T @ H @ Vp + (Vp.conj().T @ H @ Vp).conj().T)
    w = eigh(Hp, eigvals_only=True)
    fvals = np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))
    lo, hi = args.window
    win = fvals[(fvals >= lo) & (fvals <= hi)]
    fd = np.sort(np.load(args.fdfd)["freqs_xmanifold"]) if os.path.exists(args.fdfd) else np.array([])
    np.savez(args.out, freqs=fvals, m=args.m, nref=len(sgrid), n_basis=Nb,
             n_kept=int(keep.sum()))
    print(f"S-rank {int(keep.sum())}/{Nb} | window [{lo},{hi}]: Galerkin {len(win)}"
          + (f" vs FDFD X-manifold {len(fd)}" if fd.size else ""), flush=True)
    print("  Galerkin window:", " ".join(f"{x:.5f}" for x in win[:16]), flush=True)
    if fd.size:
        print("  FDFD X-manifold:", " ".join(f"{x:.5f}" for x in fd[:8]), flush=True)


if __name__ == "__main__":
    main()
