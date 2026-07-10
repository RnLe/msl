#!/usr/bin/env python3
"""Stage 1 — C4-irrep-PROJECTED Galerkin: restore the rigorous degeneracy by construction.

stage1_c4basis showed that a merely C4-closed basis does not auto-restore the 2-fold (the
canonical orthogonalization + generalized eigensolve do not enforce the symmetry, and the
MPB-gauge X block is only approximately C4-closed). The rigorous fix is to SYMMETRY-ADAPT:

  Take the X-carrier block as the seed (its C4 orbit spans X and X'). Build the exact C4
  operator on the sparse supercell plane-wave coeffs as the index permutation
  perm: n -> nG0 + C4.n  (nG0 = ((1-m)/2,(1+m)/2), C4.(n1,n2)=(-n2,n1); value unchanged),
  which realizes C4 EXACTLY on the grid. Project each seed onto the four C4 irreps
    v_chi(b) = (1/4) sum_{k=0..3} conj(chi^k) P^k C[:,b],   chi in {A:1, B:-1, 1E:i, 2E:-i},
  assemble H,S within each irrep block, and solve. The 1E and 2E blocks are time-reversal
  images (chi conjugate), so their spectra are IDENTICAL -> the {1E,2E} 2-fold is EXACT by
  construction, independent of energy convergence. A,B are the C2=+1 sector.

Metric: max |f(1E)_k - f(2E)_k| over the block (rigorous 2-fold, must be ~machine eps), and the
per-irrep ground energies (their spread = the emergent inter-C2-sector split, a convergence/theta
quantity). Compare to the un-projected (plain two-valley) ground gap.

Usage: stage1_c4proj.py <m> [px] [gcut] [nbands] [sx sy]
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galerkin_moire as gm
from galerkin_recip import basis_coeffs
from supercell_asym import build_bilayer_eps_asym

X = gm.X
IRREPS = {"A": 1.0, "B": -1.0, "1E": 1j, "2E": -1j}


def solve_block(C, npix, Ngrid, B_super, kin, eps_bl, s_tol=1e-6):
    Nb = C.shape[1]
    if Nb == 0:
        return np.array([]), 0
    dA = abs(np.linalg.det(B_super))
    Csr = C.tocsr(); Ch = C.conj().T.tocsr()
    S = np.zeros((Nb, Nb), np.complex128)
    for b in range(Nb):
        col = np.asarray(C.getcol(b).todense()).reshape(Ngrid, Ngrid)
        w = np.fft.ifft2(col) * npix
        S[:, b] = (Ch @ (np.fft.fft2(eps_bl * w).ravel() / npix)) * dA
    S = 0.5 * (S + S.conj().T)
    smax = eigh(S, subset_by_index=[Nb - 1, Nb - 1], eigvals_only=True, driver="evr")[0]
    if smax <= 0:
        return np.array([]), 0
    sval, svec = eigh(S, subset_by_value=(s_tol * smax, np.inf), driver="evr")
    Vp = svec / np.sqrt(sval)
    Hp = Vp.conj().T @ (dA * (Ch @ (kin[:, None] * (Csr @ Vp))))
    Hp = 0.5 * (Hp + Hp.conj().T)
    w = eigh(Hp, eigvals_only=True, driver="evr")
    return np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi)), int(sval.size)


def main():
    m = int(sys.argv[1]); px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    gcut = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    nbands = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    sbar = np.array([float(sys.argv[5]), float(sys.argv[6])]) if len(sys.argv) > 6 else np.array([0.0, 0.0])
    g, B_super, theta = gm.moire_g(m)
    Ngrid = px * round(float(np.sqrt(m * m + 1))); npix = Ngrid * Ngrid
    J = 2 * gcut + 1; js = np.arange(-J, J + 1)
    bands = list(range(nbands))
    momX = np.array([X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1] for j1 in js for j2 in js])
    nG0 = np.array([(1 - m) // 2, (1 + m) // 2])
    print(f"m={m} px={px} Ngrid={Ngrid} gcut={gcut} nbands={nbands} sbar={sbar} "
          f"seeds/carrier={len(momX)*nbands} nG0={tuple(nG0)}", flush=True)

    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid; N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]; Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()
    eps_bl, _ = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, Ngrid, Ngrid, 8, "centered")

    # flat C4 permutation on the Ngrid×Ngrid supercell-reciprocal grid: n -> nG0 + C4.n
    allflat = np.arange(npix)
    i1, i2 = allflat // Ngrid, allflat % Ngrid
    perm = ((nG0[0] - i2) % Ngrid) * Ngrid + ((nG0[1] + i1) % Ngrid)   # perm[flat] = C4-image flat

    # seed X-block sparse coeffs (list of (idx,val) per seed column)
    uX, _, _ = gm.extract_reference_bloch(sbar, momX, nbands, 64, r1=0.20, r2=0.10)
    seeds = []
    for ik in range(len(momX)):
        for bd in bands:
            seeds.append(basis_coeffs(uX[ik, bd], momX[ik], B_super, Ngrid))
    print(f"  {len(seeds)} seeds extracted; assembling 4 C4-orbit copies...", flush=True)

    # precompute the 4 rotated index sets per seed (P^k idx), values unchanged
    rot_idx = []   # rot_idx[seed] = [idx_k for k=0..3]
    for (idx, val) in seeds:
        ks = [idx]
        for _ in range(3):
            ks.append(perm[ks[-1]])
        rot_idx.append(ks)

    # per-irrep projected basis: v_chi(b) = sum_k conj(chi^k) * (P^k C[:,b])  (same values, shifted idx)
    results = {}
    for name, chi in IRREPS.items():
        rows, cols, data, col = [], [], [], 0
        chik = [np.conj(chi ** k) for k in range(4)]
        for b, (idx, val) in enumerate(seeds):
            # accumulate the four shifted copies on a dict keyed by flat index
            acc = {}
            for k in range(4):
                ck = chik[k]
                ik_idx = rot_idx[b][k]
                for fi, vv in zip(ik_idx, val):
                    acc[fi] = acc.get(fi, 0.0 + 0.0j) + ck * vv
            fis = np.fromiter(acc.keys(), dtype=np.int64)
            vs = np.fromiter(acc.values(), dtype=np.complex128) * 0.25
            nz = np.abs(vs) > 1e-14
            if nz.sum() == 0:
                continue
            rows.append(fis[nz]); cols.append(np.full(nz.sum(), col)); data.append(vs[nz]); col += 1
        if col == 0:
            results[name] = (np.array([]), 0); continue
        C = sp.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(npix, col), dtype=np.complex128)
        f, nk = solve_block(C, npix, Ngrid, B_super, kin, eps_bl)
        results[name] = (f, nk)
        print(f"  [{name}] block Nb={col} rank={nk} lowest6: "
              + " ".join(f"{x:.7f}" for x in f[:6]), flush=True)

    # rigorous 2-fold check: 1E vs 2E spectra must be identical
    f1, f2 = results["1E"][0], results["2E"][0]
    fA, fB = results["A"][0], results["B"][0]
    n = min(len(f1), len(f2))
    print("\n  RIGOROUS {1E,2E} 2-fold: max|f(1E)-f(2E)| over %d levels = %.2e" %
          (n, np.abs(f1[:n] - f2[:n]).max() if n else np.nan))
    grounds = {k: (v[0][0] if len(v[0]) else np.nan) for k, v in results.items()}
    print("  per-irrep ground f:", {k: f"{v:.7f}" for k, v in grounds.items()})
    gs = np.array([grounds[k] for k in ["A", "B", "1E", "2E"]])
    print(f"  emergent inter-sector split (max-min of irrep grounds) = {np.nanmax(gs)-np.nanmin(gs):.2e}")
    np.savez(os.path.join(HERE, f"stage1_c4proj_m{m}.npz"),
             **{f"f_{k}": v[0] for k, v in results.items()}, m=m, gcut=gcut)


if __name__ == "__main__":
    main()
