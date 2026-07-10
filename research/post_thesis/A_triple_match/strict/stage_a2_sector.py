#!/usr/bin/env python3
"""Stage A2 — falsifiable test of the T_P1 iso-spectrality claim (A≡B in the model).

CLAIM (stage_a_tp1): the C4-irrep-projected A and B blocks are unitarily equivalent via T_P1
(exact at the coeff level), so their generalized spectra are IDENTICAL in exact arithmetic; the
A-B split measured in §14 (2.7e-5 .. 5.9e-6) is an artifact of the s_tol truncation picking
different ranks (268 vs 270) near the quasi-continuous bottom of the S spectrum — NOT physics.

TESTS (m=7, gcut3, nbands2 — the exact §14 configuration):
  T1. eig(S_A) vs eig(S_B): the overlap spectra must agree level-by-level to ~roundoff.
  T2. rank-MATCHED canonical orthogonalization (keep the K largest S-modes for BOTH blocks, K at
      a spectral gap): the A-B eigenvalue split must collapse from ~1e-5 to ~1e-10.
  T3. control: reproduce the §14 mismatched-rank splits at s_tol=1e-6 (the artifact).

FALSIFIER: if the rank-matched A-B split does NOT collapse, the T_P1 story is wrong.
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


def build_projected_block(chi, seeds, rot_idx, npix):
    rows, cols, data, col = [], [], [], 0
    chik = [np.conj(chi ** k) for k in range(4)]
    for b, (idx, val) in enumerate(seeds):
        acc = {}
        for k in range(4):
            ck = chik[k]
            for fi, vv in zip(rot_idx[b][k], val):
                acc[fi] = acc.get(fi, 0.0 + 0.0j) + ck * vv
        fis = np.fromiter(acc.keys(), dtype=np.int64)
        vs = np.fromiter(acc.values(), dtype=np.complex128) * 0.25
        nz = np.abs(vs) > 1e-14
        if nz.sum() == 0:
            continue
        rows.append(fis[nz]); cols.append(np.full(nz.sum(), col)); data.append(vs[nz]); col += 1
    return sp.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(npix, col), dtype=np.complex128)


def assemble_S(C, npix, Ngrid, dA, eps_bl):
    Nb = C.shape[1]
    Ch = C.conj().T.tocsr()
    S = np.zeros((Nb, Nb), np.complex128)
    for b in range(Nb):
        col = np.asarray(C.getcol(b).todense()).reshape(Ngrid, Ngrid)
        w = np.fft.ifft2(col) * npix
        S[:, b] = (Ch @ (np.fft.fft2(eps_bl * w).ravel() / npix)) * dA
    return 0.5 * (S + S.conj().T)


def solve_kept(C, S, kin, dA, keep_idx):
    sval, svec = np.linalg.eigh(S)
    sv, sV = sval[keep_idx], svec[:, keep_idx]
    Vp = sV / np.sqrt(sv)
    Csr = C.tocsr(); Ch = C.conj().T.tocsr()
    Hp = Vp.conj().T @ (dA * (Ch @ (kin[:, None] * (Csr @ Vp))))
    Hp = 0.5 * (Hp + Hp.conj().T)
    w = eigh(Hp, eigvals_only=True, driver="evr")
    return np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))


def main():
    m, px, gcut, nbands = 7, 16, 3, 2
    if len(sys.argv) > 1:
        m = int(sys.argv[1])
    if len(sys.argv) > 2:
        gcut = int(sys.argv[2])
    g, B_super, _ = gm.moire_g(m)
    Ngrid = px * round(float(np.sqrt(m * m + 1))); npix = Ngrid * Ngrid
    J = 2 * gcut + 1; js = np.arange(-J, J + 1)
    bands = list(range(nbands))
    momX = np.array([X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1] for j1 in js for j2 in js])
    nG0 = np.array([(1 - m) // 2, (1 + m) // 2])
    dA = abs(np.linalg.det(B_super))
    print(f"m={m} gcut={gcut} nbands={nbands} Ngrid={Ngrid}", flush=True)

    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    kin = ((X[0] + N1 * b_sup[0, 0] + N2 * b_sup[0, 1]) ** 2 +
           (X[1] + N1 * b_sup[1, 0] + N2 * b_sup[1, 1]) ** 2).ravel()
    eps_bl, _ = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, Ngrid, Ngrid, 8, "centered")

    allflat = np.arange(npix)
    i1, i2 = allflat // Ngrid, allflat % Ngrid
    perm = ((nG0[0] - i2) % Ngrid) * Ngrid + ((nG0[1] + i1) % Ngrid)

    uX, _, _ = gm.extract_reference_bloch((0.0, 0.0), momX, nbands, 64, r1=0.20, r2=0.10)
    seeds, rot_idx = [], []
    for ik in range(len(momX)):
        for bd in bands:
            s_ = basis_coeffs(uX[ik, bd], momX[ik], B_super, Ngrid)
            seeds.append(s_)
            ks = [s_[0]]
            for _ in range(3):
                ks.append(perm[ks[-1]])
            rot_idx.append(ks)
    print(f"  {len(seeds)} seeds", flush=True)

    CA = build_projected_block(1.0, seeds, rot_idx, npix)
    CB = build_projected_block(-1.0, seeds, rot_idx, npix)
    SA = assemble_S(CA, npix, Ngrid, dA, eps_bl)
    SB = assemble_S(CB, npix, Ngrid, dA, eps_bl)
    print(f"  blocks: A Nb={CA.shape[1]}, B Nb={CB.shape[1]}", flush=True)

    # T1: overlap spectra identical?
    sA = np.linalg.eigvalsh(SA); sB = np.linalg.eigvalsh(SB)
    n = min(sA.size, sB.size)
    reldev = np.abs(sA[-n:] - sB[-n:]) / np.abs(sA[-1])
    print(f"\nT1  eig(S_A) vs eig(S_B): max rel dev = {reldev.max():.2e} "
          f"(over all {n}); top-500 modes: {reldev[-500:].max():.2e}")

    # T3 control: the §14 s_tol=1e-6 cut (mismatched ranks -> the artifact)
    for tag, S_, C_ in [("A", SA, CA), ("B", SB, CB)]:
        pass
    stol = 1e-6
    kA = np.where(sA > stol * sA[-1])[0]
    kB = np.where(sB > stol * sB[-1])[0]
    fA_cut = solve_kept(CA, SA, kin, dA, kA)
    fB_cut = solve_kept(CB, SB, kin, dA, kB)
    print(f"\nT3  s_tol=1e-6 (the §14 treatment): rank_A={kA.size} rank_B={kB.size}")
    print(f"    ground A={fA_cut[0]:.8f}  B={fB_cut[0]:.8f}  A-B split = {abs(fA_cut[0]-fB_cut[0]):.2e}")

    # T2: rank-MATCHED truncation (same K for both, K away from the cut cluster)
    for K in [min(kA.size, kB.size) - 5, 250, 200]:
        if K <= 0 or K > n:
            continue
        idxA = np.arange(sA.size - K, sA.size)
        idxB = np.arange(sB.size - K, sB.size)
        fA_m = solve_kept(CA, SA, kin, dA, idxA)
        fB_m = solve_kept(CB, SB, kin, dA, idxB)
        nn = min(8, fA_m.size, fB_m.size)
        mx = np.abs(fA_m[:nn] - fB_m[:nn]).max()
        print(f"T2  rank-matched K={K}: ground A={fA_m[0]:.10f} B={fB_m[0]:.10f}  "
              f"max|A-B| over lowest {nn} = {mx:.2e}")

    np.savez(os.path.join(HERE, f"stage_a2_sector_m{m}.npz"),
             sA=sA, sB=sB, fA_cut=fA_cut[:50], fB_cut=fB_cut[:50])
    print("\nsaved stage_a2_sector npz")


if __name__ == "__main__":
    main()
