#!/usr/bin/env python3
"""Stage 0b — little-group character analysis of the FDFD ground 4-fold at 2 deg.

WHY. The two-valley Galerkin recovers the 24-state count but not the 1.7e-10 four-fold
degeneracy. Is the split (a) a numerical basis-symmetry defect (fixable by a symmetry-adapted
basis) or (b) residual missing physics? Stage 0c showed the space group is CHIRAL (no mirrors)
with C4 about BOTH the origin and the cell center tau=(1/2,1/2)L -> a candidate nonsymmorphic
{C4|tau} roto-translation. Both X=(pi,0) and X'=(0,pi) fold to the SAME supercell M-point
(B_super^T X/2pi = (28.5,-0.5) == (1/2,1/2) mod 1). A symmorphic chiral C4 protects at most a
2-fold (the T-glued E pair), so an exact 4-fold at M must be either accidental or protected by
the nonsymmorphic generator. This script LABELS it by computing the representation of each
symmetry operation in the degenerate subspace and reading off characters/eigenphases.

METHOD. FDFD (TM) returns x = sqrt(eps)*u (u = cell-periodic part; Bloch phase Q in the operator).
Physical field E = e^{iQ.r} x / sqrt(eps); the eps-weighted overlap sum_r eps E_a* E_b = x_a*.x_b
is orthonormal. For a space-group op g (grid index map g^{-1}, so (g.E)(r)=E(g^{-1}r)), the rep
matrix in the 4-fold is D(g)_{ab} = sum_r eps(r) conj(E_a(r)) E_b(g^{-1}r); character chi(g)=Tr D;
eigenphases of D give the C4 eigenvalues (1, i, -1, -i) labelling the irreps.
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

M, PX, SIGW = 57, 16, 0.367
Q = np.array([np.pi, 0.0])


def gmap(N, Mfrac, t):
    """Index arrays (ip,jp) = g^{-1}(I,J) for g(s)=Mfrac.s + t (t in index units)."""
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Minv = np.linalg.inv(Mfrac)
    di, dj = I - t[0], J - t[1]
    ip = np.rint(Minv[0, 0] * di + Minv[0, 1] * dj).astype(int) % N
    jp = np.rint(Minv[1, 0] * di + Minv[1, 1] * dj).astype(int) % N
    return ip, jp


C4 = np.array([[0, -1], [1, 0]])
C2 = np.array([[-1, 0], [0, -1]])


def main():
    N = PX * round((M * M + 1) ** 0.5)
    print(f"m={M} px={PX} N={N} ({N*N} DOF)  building eps (Nsub=8) + FDFD operator...", flush=True)
    eps, info = build_bilayer_eps_asym(M, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "centered")
    L = build_fdfd_operator(eps, info, Q, "tm")
    sig = (2 * np.pi * SIGW) ** 2
    t0 = time.time()
    fac = cholesky((L - sig * sp.eye(L.shape[0], format="csc")).tocsc(), beta=0, mode="simplicial")
    op = LinearOperator(L.shape, matvec=fac, dtype=L.dtype)
    vals, vecs = eigsh(L, k=8, sigma=sig, which="LM", OPinv=op, maxiter=20000, tol=1e-11)
    o = np.argsort(vals)
    vals, vecs = vals[o], vecs[:, o]
    f = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
    print(f"  solved in {time.time()-t0:.0f}s. lowest 8 freqs:")
    print("   ", " ".join(f"{x:.6f}" for x in f))
    split4 = f[3] - f[0]
    print(f"  ground 4-fold split = {split4:.2e}   (gap to 5th = {f[4]-f[3]:.2e})")

    # --- physical fields of the ground 4-fold, eps-weighted-orthonormal
    B = np.asarray(info["B_super"], float)
    s1 = np.arange(N) / N
    S1, S2 = np.meshgrid(s1, s1, indexing="ij")
    x = S1 * B[0, 0] + S2 * B[0, 1]
    y = S1 * B[1, 0] + S2 * B[1, 1]
    ph = np.exp(1j * (Q[0] * x + Q[1] * y))
    sqeps = np.sqrt(eps)
    ndeg = 4
    E = [ph * vecs[:, i].reshape(N, N) / sqeps for i in range(ndeg)]
    # sanity: eps-weighted Gram should be identity
    G = np.array([[np.sum(eps * np.conj(E[a]) * E[b]) for b in range(ndeg)] for a in range(ndeg)])
    print(f"  eps-weighted Gram deviation from I: {np.abs(G-np.eye(ndeg)).max():.2e}")

    # --- valley content (Fourier weight on the X vs X' stars) for context
    Gr = 2 * np.pi * np.linalg.inv(B).T
    g1 = np.fft.fftfreq(N) * N
    G1, G2 = np.meshgrid(g1, g1, indexing="ij")
    qx = G1 * Gr[0, 0] + G2 * Gr[0, 1]
    qy = G1 * Gr[1, 0] + G2 * Gr[1, 1]
    qxr = (qx + np.pi) % (2 * np.pi) - np.pi
    qyr = (qy + np.pi) % (2 * np.pi) - np.pi
    dX = np.hypot(np.pi - np.abs(qxr), qyr)
    dXp = np.hypot(qxr, np.pi - np.abs(qyr))
    print("  per-state valley weight (wX, wX'):")
    for i in range(ndeg):
        P = np.abs(np.fft.fft2(vecs[:, i].reshape(N, N))) ** 2
        P /= P.sum()
        print(f"    state {i}: f={f[i]:.6f}  wX={P[dX<0.45].sum():.3f}  wX'={P[dXp<0.45].sum():.3f}")

    # --- representation matrices + characters of each symmetry op
    h, qtr = N // 2, N // 4
    ops = [
        ("E",            None,           None),
        ("C4 @ origin",  C4, (0, 0)),
        ("C2 @ origin",  C2, (0, 0)),
        ("C4^3 @ origin", C4.T, (0, 0)),      # inverse of C4
        ("C4 @ center",  C4, (h, h)),
        ("C2 @ center",  C2, (h, h)),
        ("C4 @ quarter", C4, (qtr, qtr)),     # expected NOT a symmetry (control)
    ]
    print("\n  little-group representation in the ground 4-fold:")
    print(f"    {'operation':<16} {'|chi|':>7} {'chi':>22}  {'unitary?':>9}  eigenphases(deg)")
    for name, Mf, t in ops:
        if Mf is None:
            D = G  # identity
        else:
            ip, jp = gmap(N, Mf, t)
            D = np.array([[np.sum(eps * np.conj(E[a]) * E[b][ip, jp])
                           for b in range(ndeg)] for a in range(ndeg)])
        chi = np.trace(D)
        unit = np.abs(D.conj().T @ D - np.eye(ndeg)).max()
        evph = np.angle(np.linalg.eigvals(D)) * 180 / np.pi
        evph = np.sort(np.round(evph, 1))
        print(f"    {name:<16} {abs(chi):>7.3f} {chi.real:>+9.3f}{chi.imag:>+8.3f}i  "
              f"{unit:>9.1e}  {evph}")

    np.savez(os.path.join(HERE, "stage0b_characters.npz"),
             freqs=f, split4=split4, ground_fields=np.array([vecs[:, i] for i in range(ndeg)]),
             eps=eps, N=N)
    print("\n  saved stage0b_characters.npz (freqs + 4 ground eigenvectors + eps)")


if __name__ == "__main__":
    main()
