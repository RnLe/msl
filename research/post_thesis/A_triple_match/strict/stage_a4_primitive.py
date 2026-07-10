#!/usr/bin/env python3
"""Stage A4 — primitive-cell FDFD pipeline: 2x cheaper ground truth + exact sector resolution.

§15: the centered (m,1) cell is a 2x supercell of the true primitive crystal (P1=(L1+L2)/2,
P2=(L2-L1)/2, area (m²+1)/2). The centered-cell spectrum at Q_X therefore superposes TWO
primitive-Bloch sectors — the primitive momenta q+ = X and q- = X + b1c (b1c = a centered-cell
reciprocal with ODD index sum, not a primitive reciprocal) — which are exactly the T_P1 = +/-1
sectors. Predictions tested here:
  P1. primitive-cell FDFD at q+ and q- each reproduce HALF the centered X-manifold: the union
      matches the centered-cell ladder (fdfd_asym_x_2deg_res16 / fdfd_m7_px24) to solver tol;
  P2. the ground quadruplet appears as ONE level in each sector per doublet: each sector shows
      {E1, E2} (the two T_P1-doublets contribute one T=+1 and one T=-1 combination each);
  P3. the centered-cell doublets = one state per sector at the same energy => the primitive
      solves have no systematic 2-folds (generic singlets).
Memory: DOF halves (m=57 px16: 640^2=410k vs 912^2=832k) -> px32/48 Richardson becomes cheap.

Usage: stage_a4_primitive.py <m> [px] [sigma] [nmodes]
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


def solve(eps, info, q, sig_w, k):
    L = build_fdfd_operator(eps, info, q, "tm")
    sig = (2 * np.pi * sig_w) ** 2
    fac = cholesky((L - sig * sp.eye(L.shape[0], format="csc")).tocsc(), beta=0, mode="simplicial")
    vals, vecs = eigsh(L, k=k, sigma=sig, which="LM",
                       OPinv=LinearOperator(L.shape, matvec=fac, dtype=L.dtype),
                       maxiter=20000, tol=1e-10)
    o = np.argsort(vals)
    return np.sqrt(np.maximum(vals[o], 0)) / (2 * np.pi), vecs[:, o]


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 57
    px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sig_w = float(sys.argv[3]) if len(sys.argv) > 3 else (0.367 if m >= 30 else 0.067)
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    # primitive cell: P1=((m-1)/2,(m+1)/2), |P1|=sqrt((m^2+1)/2)
    Np = px * round(((m * m + 1) / 2) ** 0.5)
    eps, info = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, Np, Np, 8, "primitive")
    print(f"m={m} px={px} PRIMITIVE cell: N={Np} ({Np*Np} DOF, centered would be "
          f"{(px*round((m*m+1)**0.5))**2})", flush=True)
    X = np.array([np.pi, 0.0])
    # centered-cell reciprocal b1c (odd index sum -> the sector-shift momentum)
    Bc = np.array([[float(m), -1.0], [1.0, float(m)]])
    bc = 2 * np.pi * np.linalg.inv(Bc).T          # columns b1c, b2c
    qs = {"T=+1 (q=X)": X, "T=-1 (q=X+b1c)": X + bc[:, 0]}
    t0 = time.time()
    ladders = {}
    for tag, q in qs.items():
        f, _ = solve(eps, info, q, sig_w, k)
        ladders[tag] = f
        print(f"  [{tag}] lowest 12: " + " ".join(f"{x:.6f}" for x in f[:12]), flush=True)
    print(f"  ({time.time()-t0:.0f}s)")

    fp, fm = ladders["T=+1 (q=X)"], ladders["T=-1 (q=X+b1c)"]
    union = np.sort(np.concatenate([fp, fm]))
    # centered-cell reference
    ref_file = ("fdfd_asym_x_2deg_res16.npz" if m == 57 else None)
    if m == 57:
        ref = np.sort(np.load(os.path.join(HERE, ref_file))["freqs"])
    elif m == 7:
        ref = np.sort(np.load(os.path.join(HERE, "fdfd_m7_px24.npy")))
    else:
        ref = None
    if ref is not None:
        n = min(len(union), len(ref))
        dev = np.abs(union[:n] - ref[:n]).max()
        print(f"\nP1  union(q+,q-) vs centered-cell ladder: max|Δf| over {n} = {dev:.2e}")
    # P2/P3: sector structure of the ground quad
    print(f"P2  ground quad as sectors: q+ lowest2 = {fp[0]:.8f}, {fp[1]:.8f}; "
          f"q- lowest2 = {fm[0]:.8f}, {fm[1]:.8f}")
    print(f"    cross-sector match (doublet check): |fp0-fm0|={abs(fp[0]-fm[0]):.2e}, "
          f"|fp1-fm1|={abs(fp[1]-fm[1]):.2e}")
    gaps_p = np.diff(fp[:10]); gaps_m = np.diff(fm[:10])
    print(f"P3  smallest nn-gap within a single sector: q+ {gaps_p.min():.2e}, q- {gaps_m.min():.2e} "
          f"(generic singlets expected)")
    np.savez(os.path.join(HERE, f"stage_a4_prim_m{m}_px{px}.npz"),
             f_plus=fp, f_minus=fm, m=m, px=px, sigma=sig_w)
    print("saved stage_a4_prim npz")


if __name__ == "__main__":
    main()
