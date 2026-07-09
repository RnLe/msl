#!/usr/bin/env python3
"""Stage 0c — the space group of the moire supercell eps_bl, and the exact C4 permutation.

Two jobs:
 (1) Confirm the TRUE even-grid C4 about the origin is the roll-corrected permutation
     c4_o(A) = A[:, (-arange(N)) % N].T  (NOT np.rot90, which is a half-pixel off the center
     and is NOT a symmetry -> would cap any degeneracy fix at O(1/N)).
 (2) Enumerate which candidate operations are actual symmetries of eps_bl. This LABELS the
     space group and therefore what can protect the ground 4-fold at the supercell M-point.
     The twist is CHIRAL (layer2 = R(theta) layer1, r1 != r2), so mirrors and layer-exchange
     are expected to be BROKEN -> the point group is at most C4 (order 4), whose irreps are 1D
     (A,B) + a time-reversal-glued E doublet (max 2-fold). If a 4-fold is degenerate to 1e-10,
     it must then be protected by a NONSYMMORPHIC generator (a C4/C2 about a shifted center =
     rotation composed with a fractional supercell translation) -- tested here as C4/C2 about
     the cell center and quarter points.

Grid convention (supercell_asym.build_bilayer_eps_asym): index [i,j] <-> fractional (i/N, j/N);
real position = s1*L1 + s2*L2 with L1=(m,n), L2=(-n,m). R90*L1=L2, R90*L2=-L1, so a real-space
C4 about the origin is the fractional map (s1,s2)->(-s2,s1).
"""
import numpy as np
from supercell_asym import build_bilayer_eps_asym


def apply_op(A, Mfrac, t):
    """(g A)[I] = A[g^{-1}(I)], g(s) = Mfrac.s + t (t in index units), lattice-periodic."""
    N = A.shape[0]
    I, J = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Minv = np.linalg.inv(Mfrac)
    di = I - t[0]
    dj = J - t[1]
    ip = np.rint(Minv[0, 0] * di + Minv[0, 1] * dj).astype(int) % N
    jp = np.rint(Minv[1, 0] * di + Minv[1, 1] * dj).astype(int) % N
    return A[ip, jp]


C4 = np.array([[0, -1], [1, 0]])       # (s1,s2)->(-s2,s1) = real C4 about origin
C2 = np.array([[-1, 0], [0, -1]])
MIR_A = np.array([[-1, 0], [0, 1]])    # fractional reflection s1->-s1
MIR_D = np.array([[0, 1], [1, 0]])     # fractional reflection swap (diagonal)


def rel(A, B):
    return float(np.abs(A - B).max() / np.abs(A).max())


def analyze(m, px):
    N = px * round((m * m + 1) ** 0.5)
    eps, info = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, N, N, 1, "centered")
    h = N // 2
    q = N // 4
    print(f"\n=== m={m} (theta={info['theta_deg']:.2f} deg), N={N} ===")
    # (1) C4 permutation: roll-corrected vs np.rot90
    c4_roll = eps[:, (-np.arange(N)) % N].T
    print(f"  roll-corrected c4_o  : max|eps - c4(eps)|/max = {rel(eps, c4_roll):.2e}"
          f"   {'<-- SYMMETRY' if rel(eps, c4_roll) < 1e-12 else ''}")
    print(f"  np.rot90 (half-pixel): max|eps - rot90|/max   = {rel(eps, np.rot90(eps)):.2e}"
          f"   {'(NOT a symmetry)' if rel(eps, np.rot90(eps)) > 1e-6 else ''}")
    # (2) space-group enumeration
    ops = [
        ("C4 @ origin",        C4, (0, 0)),
        ("C4 @ center",        C4, (h, h)),
        ("C2 @ origin",        C2, (0, 0)),
        ("C2 @ center",        C2, (h, h)),
        ("C2 @ quarter",       C2, (q, q)),
        ("C4 @ quarter",       C4, (q, q)),
        ("mirror s1->-s1",     MIR_A, (0, 0)),
        ("mirror diag swap",   MIR_D, (0, 0)),
    ]
    print("  space-group candidates (max|eps - op(eps)|/max):")
    for name, M, t in ops:
        r = rel(eps, apply_op(eps, M, t))
        tag = "  <== SYMMETRY" if r < 1e-12 else ""
        print(f"    {name:<18} shift={str(t):<10} : {r:.2e}{tag}")


if __name__ == "__main__":
    analyze(7, 16)     # tiny cell: fast sanity on the permutation + op logic
    analyze(57, 16)    # the 2deg target
