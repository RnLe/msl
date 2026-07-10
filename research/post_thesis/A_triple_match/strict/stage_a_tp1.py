#!/usr/bin/env python3
"""Stage A1 — the hidden symmetry identified & proven: the primitive half-cell translation T_P1.

CLAIM (closes §14's open question). For the (m,1) commensurate twist with odd m, the vector
    P1 = (L1+L2)/2 = ((m-1)/2, (m+1)/2)
is an integer lattice vector of BOTH layers:
    layer 1: integer square lattice -> trivially (odd m makes both components integers);
    layer 2: B2^{-1} P1 = R(-theta) P1 = ((m+1)/2, (m-1)/2)   [exact algebra, verified below]
       using cos(theta)=(m^2-1)/(m^2+1), sin(theta)=2m/(m^2+1):
       cos*(m-1)/2 + sin*(m+1)/2 = (m+1)[(m-1)^2+2m] / (2(m^2+1)) = (m+1)/2.  QED
So the centered (m,1) cell is a 2x SUPERCELL of the true primitive crystal (supercell_asym's own
cell='primitive', P1/P2, area (m^2+1)/2), and translation T_P1 is an EXACT symmetry of eps_bl —
one the campaign's centered-cell treatment had hidden.

CONSEQUENCE at the FDFD momentum Q=X: conjugation gives T_P1 C4 T_P1^{-1} = {C4 | P1 - C4 P1}
= {C4 | L1} (P1 - C4 P1 = P1 - P2 = L1), and the supercell translation L1 acts on Bloch states
at Q as the scalar e^{-iQ.L1} = e^{-i pi m} = -1 (odd m). Hence on every Q_X eigenspace
    T_P1 C4 = - C4 T_P1,      T_P1^2 = T_{L1+L2} = e^{-iQ.(L1+L2)} = +1.
The algebra {C4, T_P1 | C4^4=1, T^2=1, TC4=-C4T} has only 2-dimensional irreps on this space:
EVERY level at Q_X is an exact doublet pairing C4-eigenvalues {lambda, -lambda} — i.e. {A,B} and
{1E,2E}. This single rigorous symmetry explains BOTH exact 2-folds of §13 (time reversal is
consistent but redundant for 1E2E, and irrelevant for A=B), and the observed all-doublet m=7
spectra. The 4-fold = two T_P1-doublets fused by the EMERGENT valley degeneracy (theta->0).

MODEL COROLLARY (corrects §14): every Galerkin basis function e^{ip.r}u_n(r) is an exact T_P1
eigenvector (u_n is monolayer-periodic and P1 is an integer monolayer vector), and on the
supercell plane-wave coeffs T_P1 is DIAGONAL with parity phase (-1)^{n1+n2} (since G.P1 =
(n1+n2)pi and X.P1 = 28pi). The C4 permutation P: n -> nG0 + C4.n flips this parity
(parity(nG0) = parity((1-m)/2 + (1+m)/2) = parity(1) = odd), realizing TP = -PT exactly at the
coeff level. Therefore the projected A and B blocks are unitarily equivalent (iso-spectral) in
exact arithmetic — §14's measured A-B split (5.9e-6) is an s_tol TRUNCATION artifact (rank_A=268
vs rank_B=270 flip modes at the cut), not physics convergence. Falsifiable test: stage_a2_sector.py.

This script persists the numerical proofs (previously print-only, session-verified).
"""
import os
import sys
from fractions import Fraction

import numpy as np
from scipy.linalg import sqrtm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

Q = np.array([np.pi, 0.0])


def exact_layer2_integrality(m):
    """R(-theta) P1 with exact rational arithmetic; returns the (exact) result and integrality."""
    c = Fraction(m * m - 1, m * m + 1)
    s = Fraction(2 * m, m * m + 1)
    p1 = (Fraction(m - 1, 2), Fraction(m + 1, 2))
    r = (c * p1[0] + s * p1[1], -s * p1[0] + c * p1[1])
    return r, (r[0].denominator == 1 and r[1].denominator == 1)


def main():
    out = {}
    print("== A1.1  P1 integer in BOTH layers (exact rational algebra) ==")
    for m in [7, 57, 113]:
        r, ok = exact_layer2_integrality(m)
        print(f"  m={m}: P1=(({m-1})/2,({m+1})/2)  B2^-1 P1 = ({r[0]},{r[1]})  integer={ok}")
        assert ok
    out["layer2_integer_m"] = np.array([7, 57, 113])

    print("\n== A1.2  eps_bl invariance under T_P1 (grid roll by (N/2,N/2)) ==")
    for m, px in [(7, 16), (57, 16)]:
        N = px * round((m * m + 1) ** 0.5)
        h = N // 2
        eps, _ = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 8, "centered")
        dev = np.abs(eps - np.roll(np.roll(eps, h, axis=0), h, axis=1)).max() / eps.max()
        print(f"  m={m} N={N} (Nsub=8): max|eps - T_P1 eps|/max = {dev:.2e}")
        out[f"eps_dev_m{m}"] = dev
        assert dev < 1e-14

    print("\n== A1.3  T_P1 representation on the saved 2deg FDFD ground 4-fold ==")
    d = np.load(os.path.join(HERE, "stage0b_characters.npz"))
    N = int(d["N"]); eps = d["eps"]; gf = d["ground_fields"]
    B = np.array([[57.0, -1.0], [1.0, 57.0]])
    s = np.arange(N) / N
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    x = S1 * B[0, 0] + S2 * B[0, 1]; y = S1 * B[1, 0] + S2 * B[1, 1]
    u = [np.exp(-1j * (Q[0] * x + Q[1] * y)) * gf[i].reshape(N, N) / np.sqrt(eps) for i in range(4)]
    ipx = lambda A_, B_: np.sum(eps * np.conj(A_) * B_)
    G = np.array([[ipx(u[a], u[b]) for b in range(4)] for a in range(4)])
    Gi = np.linalg.inv(sqrtm(G))
    uo = [sum(u[a] * Gi[a, b] for a in range(4)) for b in range(4)]
    h = N // 2
    T = lambda f: np.roll(np.roll(f, h, axis=0), h, axis=1)  # e^{-iQ.P1}=e^{-i28pi}=+1
    DT = np.array([[ipx(uo[a], T(uo[b])) for b in range(4)] for a in range(4)])
    I4 = np.eye(4)
    nonunit = np.abs(DT.conj().T @ DT - I4).max()
    t2dev = np.abs(DT @ DT - I4).max()
    # C4 rep (validated gauge: (S_g u)(r) = e^{+iG0.r} u(g^{-1} r), G0 = RQ - Q)
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ip, jp = J % N, (-I) % N
    G0 = np.array([0.0, np.pi]) - Q
    gauge = np.exp(1j * (G0[0] * x + G0[1] * y))
    D4 = np.array([[ipx(uo[a], gauge * uo[b][ip, jp]) for b in range(4)] for a in range(4)])
    anti = np.abs(DT @ D4 + D4 @ DT).max()
    print(f"  D(T_P1): nonunitarity={nonunit:.1e}  |T^2 - 1|={t2dev:.1e}")
    print(f"  |{{T_P1, C4}}| = {anti:.1e}   (anticommutation T C4 = -C4 T)")
    w, V = np.linalg.eig(D4)
    ph = np.angle(w) * 180 / np.pi
    lab = {0: "A", 180: "B", -180: "B", 90: "1E", -90: "2E"}
    names = [lab.get(int(round(p)), "?") for p in ph]
    psi = [sum(V[a, k] * uo[a] for a in range(4)) for k in range(4)]
    amp = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            amp[a, b] = abs(ipx(psi[a], T(psi[b])))
    print("  T_P1 mapping on C4 sectors:")
    for a in range(4):
        b = int(np.argmax(amp[a]))
        print(f"    T({names[a]}) -> {names[b]}   |amp|={amp[a, b]:.6f}")
    out.update(DT_nonunit=nonunit, T2_dev=t2dev, anticomm=anti,
               sector_names=np.array(names), T_map_amp=amp)
    np.savez(os.path.join(HERE, "stage_a_tp1.npz"), **out)
    print("\nsaved stage_a_tp1.npz")


if __name__ == "__main__":
    main()
