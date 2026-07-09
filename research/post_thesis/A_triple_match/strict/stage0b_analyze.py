#!/usr/bin/env python3
"""Stage 0b (analysis) — characters of the ground 4-fold from the saved FDFD fields.

FIX vs the naive version: the symmetry operator that commutes with the FDFD operator L(Q) is NOT
the bare geometric rotation of the full field. A space-group op {R|v} sends a Bloch state at Q to
one at RQ; for C4, RQ = C4.(pi,0) = (0,pi) = X', which is EQUIVALENT to X only up to the supercell
reciprocal vector G0 = RQ - Q. Correct operator on the PERIODIC part u (u = x/sqrt(eps), truly
periodic so the mod-N grid wrap is exact):
    (S_g u)(r) = e^{-i c} e^{-i G0 . r} * u( g^{-1} r ),   G0 = R_cart Q - Q,
with a constant phase c = Q . R^{-1} v for a roto-translation about a shifted center. Then
D(g)_{ab} = <u_a | S_g u_b>_eps (eps-weighted), chi(g) = Tr D. Unitarity of D self-checks the gauge.
u is Lowdin-orthonormalized first (ARPACK returns the correct degenerate subspace but a
0.09-non-orthonormal internal basis across the 1.7e-10 cluster).
"""
import os
import numpy as np
from scipy.linalg import sqrtm

HERE = os.path.dirname(os.path.abspath(__file__))
Q = np.array([np.pi, 0.0])
B = np.array([[57.0, -1.0], [1.0, 57.0]])          # L1=(57,1), L2=(-1,57)
C4c = np.array([[0.0, -1.0], [1.0, 0.0]])          # Cartesian 90 deg CCW
C2c = np.array([[-1.0, 0.0], [0.0, -1.0]])
C4f = np.array([[0, -1], [1, 0]])                  # fractional (L1,L2 basis) C4
C2f = np.array([[-1, 0], [0, -1]])


def gmap(N, Mfrac, t):
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Minv = np.linalg.inv(Mfrac)
    di, dj = I - t[0], J - t[1]
    ip = np.rint(Minv[0, 0] * di + Minv[0, 1] * dj).astype(int) % N
    jp = np.rint(Minv[1, 0] * di + Minv[1, 1] * dj).astype(int) % N
    return ip, jp


def main():
    d = np.load(os.path.join(HERE, "stage0b_characters.npz"))
    N = int(d["N"]); eps = d["eps"]; gf = d["ground_fields"]
    nd = gf.shape[0]
    s1 = np.arange(N) / N
    S1, S2 = np.meshgrid(s1, s1, indexing="ij")
    x = S1 * B[0, 0] + S2 * B[0, 1]
    y = S1 * B[1, 0] + S2 * B[1, 1]
    # eigenvector x = sqrt(eps)*E_full (Bloch phase is in the stencil); periodic part u = e^{-iQr} E
    phQ = np.exp(-1j * (Q[0] * x + Q[1] * y))
    u = [phQ * gf[i].reshape(N, N) / np.sqrt(eps) for i in range(nd)]   # periodic parts

    def ipx(A, Bx):
        return np.sum(eps * np.conj(A) * Bx)

    # Lowdin orthonormalize in eps-metric
    G = np.array([[ipx(u[a], u[b]) for b in range(nd)] for a in range(nd)])
    Ginvsq = np.linalg.inv(sqrtm(G))
    uo = [sum(u[a] * Ginvsq[a, b] for a in range(nd)) for b in range(nd)]

    def rep(Rcart, Mfrac, t, const_phase=0.0):
        G0 = Rcart @ Q - Q                                   # reciprocal folding vector
        # (S_g E)(r) = e^{i(RQ).r} (rot u)(r); overlap with e^{iQr}u_a gives phase e^{+iG0.r}
        gauge = np.exp(1j * (G0[0] * x + G0[1] * y) - 1j * const_phase)
        ip, jp = gmap(N, Mfrac, t)
        Su = [gauge * uo[b][ip, jp] for b in range(nd)]
        return np.array([[ipx(uo[a], Su[b]) for b in range(nd)] for a in range(nd)])

    h, qtr = N // 2, N // 4
    # constant phase c = Q . R^{-1} v for a rotation about fractional center tau (v=(I-R)tau)
    def cphase(Rcart, tau_frac):
        tau = B @ np.array(tau_frac)                          # cartesian center
        v = (np.eye(2) - Rcart) @ tau
        return float(Q @ (np.linalg.inv(Rcart) @ v))

    ops = [
        ("E",            np.eye(2), np.eye(2, dtype=int), (0, 0), 0.0),
        ("C4 @ origin",  C4c, C4f, (0, 0), 0.0),
        ("C2 @ origin",  C2c, C2f, (0, 0), 0.0),
        ("C4^3 @ origin", C4c.T, -C4f, (0, 0), 0.0),
        ("C4 @ center",  C4c, C4f, (h, h), cphase(C4c, (0.5, 0.5))),
        ("C2 @ center",  C2c, C2f, (h, h), cphase(C2c, (0.5, 0.5))),
        ("C4^3 @ center", C4c.T, -C4f, (h, h), cphase(C4c.T, (0.5, 0.5))),
        ("C4 @ quarter", C4c, C4f, (qtr, qtr), cphase(C4c, (0.25, 0.25))),
    ]
    print(f"{'operation':<15} {'|chi|':>6} {'chi':>19}  {'nonunit':>8}  eigenphases(deg)")
    for name, Rc, Mf, t, cp in ops:
        D = rep(Rc, Mf, t, cp)
        chi = np.trace(D)
        nonu = np.abs(D.conj().T @ D - np.eye(nd)).max()
        evph = np.sort(np.round(np.angle(np.linalg.eigvals(D)) * 180 / np.pi, 1))
        print(f"{name:<15} {abs(chi):>6.3f} {chi.real:>+8.3f}{chi.imag:>+8.3f}i  {nonu:>8.1e}  {evph}")


if __name__ == "__main__":
    main()
