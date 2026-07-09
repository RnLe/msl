#!/usr/bin/env python3
"""Stage 1 prep — fine structure of the ground 4-fold vs angle (the C4-irrep resolution).

Stage 0b (2 deg) showed the ground 4-fold is the regular rep of C4 = A(+1) + B(-1) + 1E(i) + 2E(-i),
with T gluing 1E,2E into a RIGOROUS 2-fold and the A-B degeneracy EMERGENT (theta-suppressed).
Prediction: at a larger angle (where the emergent split is resolved) the four levels should organize
as a 2+1+1 pattern -- a machine-degenerate {1E,2E} pair (T-protected) plus two split singlets A,B.
This script solves FDFD at (m,1), assigns each ground state a C4 eigenvalue, and prints the
eigenvalue split BY C4-IRREP, confirming which part is rigorous (E doublet) vs emergent (A-B).

Usage: stage1_finestructure.py <m> [px]
"""
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.linalg import sqrtm
from sksparse.cholmod import cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, HERE)
from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

Q = np.array([np.pi, 0.0])
C4c = np.array([[0.0, -1.0], [1.0, 0.0]])
C4f = np.array([[0, -1], [1, 0]])


def main():
    m = int(sys.argv[1]); px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    N = px * round((m * m + 1) ** 0.5)
    B = np.array([[float(m), -1.0], [1.0, float(m)]])
    # shift-invert target near the band-1-at-X manifold (STRONGLY angle-dependent:
    # ~0.067 at 16deg (m=7), ~0.370 at 2deg (m=57)); pass explicitly as argv[3].
    sigw = float(sys.argv[3]) if len(sys.argv) > 3 else (0.367 if m >= 30 else 0.067)
    print(f"m={m} px={px} N={N} ({N*N} DOF)  solving FDFD...", flush=True)
    eps, info = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "centered")
    L = build_fdfd_operator(eps, info, Q, "tm")
    sig = (2 * np.pi * sigw) ** 2
    t0 = time.time()
    fac = cholesky((L - sig * sp.eye(L.shape[0], format="csc")).tocsc(), beta=0, mode="simplicial")
    vals, vecs = eigsh(L, k=8, sigma=sig, which="LM",
                       OPinv=LinearOperator(L.shape, matvec=fac, dtype=L.dtype),
                       maxiter=20000, tol=1e-11)
    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]
    f = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
    print(f"  {time.time()-t0:.0f}s. lowest 8 f: " + " ".join(f"{x:.7f}" for x in f))
    # X-star weight of the ground 4-fold (confirm it is the band-1-at-X manifold)
    Gr = 2 * np.pi * np.linalg.inv(B).T
    g1 = np.fft.fftfreq(N) * N; G1, G2 = np.meshgrid(g1, g1, indexing="ij")
    qxr = ((G1 * Gr[0, 0] + G2 * Gr[0, 1]) + np.pi) % (2 * np.pi) - np.pi
    qyr = ((G1 * Gr[1, 0] + G2 * Gr[1, 1]) + np.pi) % (2 * np.pi) - np.pi
    dX = np.hypot(np.pi - np.abs(qxr), qyr); dXp = np.hypot(qxr, np.pi - np.abs(qyr))
    ws = []
    for i in range(4):
        P = np.abs(np.fft.fft2(vecs[:, i].reshape(N, N))) ** 2; P /= P.sum()
        ws.append((P[dX < 0.45].sum(), P[dXp < 0.45].sum()))
    print("  ground-4 (wX,wX'): " + " ".join(f"({a:.2f},{b:.2f})" for a, b in ws))

    # periodic parts of the ground 4-fold, Lowdin-orthonormalized in eps-metric
    s = np.arange(N) / N
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    x = S1 * B[0, 0] + S2 * B[0, 1]; y = S1 * B[1, 0] + S2 * B[1, 1]
    phQ = np.exp(-1j * (Q[0] * x + Q[1] * y))
    u = [phQ * vecs[:, i].reshape(N, N) / np.sqrt(eps) for i in range(4)]
    ipx = lambda A, Bx: np.sum(eps * np.conj(A) * Bx)
    G = np.array([[ipx(u[a], u[b]) for b in range(4)] for a in range(4)])
    Gi = np.linalg.inv(sqrtm(G))
    uo = [sum(u[a] * Gi[a, b] for a in range(4)) for b in range(4)]

    # C4 rep in the 4-fold -> eigenvalues label the irreps; then project H (diagonal in this basis)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ip = J % N; jp = (-I) % N                       # C4^{-1}
    G0 = C4c @ Q - Q
    gauge = np.exp(1j * (G0[0] * x + G0[1] * y))
    D = np.array([[ipx(uo[a], gauge * uo[b][ip, jp]) for b in range(4)] for a in range(4)])
    w, V = np.linalg.eig(D)                          # C4 eigenvalues + eigenvectors
    ph = np.angle(w) * 180 / np.pi
    # H (frequency) is diagonal in the FDFD eigenbasis with entries f; in the C4-eigenbasis the
    # energy of C4-eigenvector V[:,k] is sum_i |V[i,k]|^2 f[i]
    fC4 = np.array([np.real(np.sum(np.abs(V[:, k])**2 * f[:4])) for k in range(4)])
    order = np.argsort(ph)
    print("\n  C4-irrep resolution of the ground 4-fold:")
    print(f"    {'C4 eigenphase':>14} {'irrep':>6} {'energy f':>12}")
    lab = {0: "A", 180: "B", 90: "¹E", -90: "²E", -180: "B"}
    for k in order:
        p = int(round(ph[k]))
        print(f"    {ph[k]:>+13.1f}° {lab.get(p,'?'):>6} {fC4[k]:>12.7f}")
    fA = fC4[np.argmin(np.abs(ph - 0))]
    fB = fC4[np.argmin(np.abs(np.abs(ph) - 180))]
    eE = fC4[np.abs(np.abs(ph) - 90) < 5]
    print(f"\n  4-fold total split (max-min f)      = {f[:4].max()-f[:4].min():.2e}")
    if len(eE) == 2:
        print(f"  RIGOROUS 1E-2E split (should be ~0) = {abs(eE[0]-eE[1]):.2e}")
    print(f"  EMERGENT A-B split                  = {abs(fA-fB):.2e}")


if __name__ == "__main__":
    main()
