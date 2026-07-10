#!/usr/bin/env python3
"""Stage A2b — the corrected (integer-momentum) engine: admissibility, sectors, and A≡B.

AUDIT FINDING (follows the FAILED stage_a2_sector falsifier). The admissible trial momenta for
the supercell Q_X Bloch problem are p = X + (j1 b1 + j2 b2) with INTEGER j (b = supercell
reciprocals, = moire_g's columns). The engine's historical grid uses HALF-integer steps
(0.5*j*g): for any odd j the supercell index of every coefficient is exactly half-integer and
basis_coeffs' np.rint snaps the .5-ties with the ties-to-even rule — silently ALIASING ~3/4 of
the basis into corrupted, mixed-T_P1-parity, near-duplicate vectors. Variationally harmless
(§11/§12 upper bounds stand) but it (a) broke the exact T_P1-eigenvector property of the seeds
(the stage_a2_sector A/B failure), and (b) is the prime suspect for the §12 conditioning wall.

Physics of the corrected grid: eps_bl is T_P1-invariant => its Fourier support lies on the
PRIMITIVE reciprocal lattice = the even-(n1+n2) sublattice of b_sup. Hence S,H never couple
integer momenta of different (j1+j2) parity: the T_P1 = +/-1 sectors decouple EXACTLY, realizing
the sector resolution for free. Each seed has uniform support parity (j1+j2 mod 2) — an exact
T_P1 eigenvector — so the C4-projected A and B blocks are exactly unitarily equivalent.

PREDICTIONS TESTED HERE (m=7, both proven-anchor comparisons available):
  P1. every seed has uniform T-parity (assert);
  P2. A-B ground split collapses to ~1e-10 (vs 2.7e-5 on the old grid);
  P3. conditioning: clean-rank fraction of S far higher than the old half-step engine;
  P4. spectra still converge toward the FDFD anchor (variational sanity).

Usage: stage_a2_integer.py [m] [J] [nbands] [px]
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


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    Jmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    nbands = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    px = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    g, B_super, _ = gm.moire_g(m)
    Ngrid = px * round(float(np.sqrt(m * m + 1))); npix = Ngrid * Ngrid
    js = np.arange(-Jmax, Jmax + 1)
    bands = list(range(nbands))
    # INTEGER momentum grid (admissible at Q_X): p = X + j1 b1 + j2 b2
    jj = [(j1, j2) for j1 in js for j2 in js]
    momX = np.array([X + j1 * g[:, 0] + j2 * g[:, 1] for j1, j2 in jj])
    nG0 = np.array([(1 - m) // 2, (1 + m) // 2])
    dA = abs(np.linalg.det(B_super))
    print(f"m={m} J={Jmax} nbands={nbands} Ngrid={Ngrid} momenta={len(momX)} (INTEGER grid)",
          flush=True)

    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    kin = ((X[0] + N1 * b_sup[0, 0] + N2 * b_sup[0, 1]) ** 2 +
           (X[1] + N1 * b_sup[1, 0] + N2 * b_sup[1, 1]) ** 2).ravel()
    eps_bl, _ = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, Ngrid, Ngrid, 8, "centered")

    allflat = np.arange(npix)
    i1f, i2f = allflat // Ngrid, allflat % Ngrid
    perm = ((nG0[0] - i2f) % Ngrid) * Ngrid + ((nG0[1] + i1f) % Ngrid)
    parity_of_flat = ((i1f + i2f) % 2)

    uX, _, _ = gm.extract_reference_bloch((0.0, 0.0), momX, nbands, 64, r1=0.20, r2=0.10)
    seeds, rot_idx, seed_parity = [], [], []
    for ik in range(len(momX)):
        for bd in bands:
            idx, val = basis_coeffs(uX[ik, bd], momX[ik], B_super, Ngrid)
            # P1 check: uniform T_P1 parity of the support
            pars = np.unique(parity_of_flat[idx])
            assert pars.size == 1, f"seed ik={ik} bd={bd}: MIXED parity {pars}"
            seeds.append((idx, val)); seed_parity.append(int(pars[0]))
            ks = [idx]
            for _ in range(3):
                ks.append(perm[ks[-1]])
            rot_idx.append(ks)
    seed_parity = np.array(seed_parity)
    print(f"P1  all {len(seeds)} seeds have UNIFORM T_P1 parity "
          f"(sector counts: even={np.sum(seed_parity==0)}, odd={np.sum(seed_parity==1)})", flush=True)

    def solve_block(C, s_tol=1e-6):
        Nb = C.shape[1]
        Ch = C.conj().T.tocsr(); Csr = C.tocsr()
        S = np.zeros((Nb, Nb), np.complex128)
        for b in range(Nb):
            col = np.asarray(C.getcol(b).todense()).reshape(Ngrid, Ngrid)
            w = np.fft.ifft2(col) * npix
            S[:, b] = (Ch @ (np.fft.fft2(eps_bl * w).ravel() / npix)) * dA
        S = 0.5 * (S + S.conj().T)
        sval = np.linalg.eigvalsh(S)
        smax = sval[-1]
        svv, svec = eigh(S, subset_by_value=(s_tol * smax, np.inf), driver="evr")
        Vp = svec / np.sqrt(svv)
        Hp = Vp.conj().T @ (dA * (Ch @ (kin[:, None] * (Csr @ Vp))))
        Hp = 0.5 * (Hp + Hp.conj().T)
        w = eigh(Hp, eigvals_only=True, driver="evr")
        return np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi)), int(svv.size), Nb, sval

    results = {}
    for name, chi in IRREPS.items():
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
        C = sp.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(npix, col), dtype=np.complex128)
        f, nk, Nb, sval = solve_block(C)
        results[name] = (f, nk, Nb, sval)
        print(f"  [{name}] Nb={Nb} clean-rank={nk} ({nk/Nb:.0%})  lowest4: "
              + " ".join(f"{x:.8f}" for x in f[:4]), flush=True)

    fA, fB = results["A"][0], results["B"][0]
    f1, f2 = results["1E"][0], results["2E"][0]
    n = min(len(fA), len(fB)); n2 = min(len(f1), len(f2))
    print(f"\nP2  A-B: ground split = {abs(fA[0]-fB[0]):.2e}; "
          f"max over lowest 8 = {np.abs(fA[:8]-fB[:8]).max():.2e}   (old grid: 2.7e-5)")
    print(f"    1E-2E: max over lowest 8 = {np.abs(f1[:8]-f2[:8]).max():.2e}")
    rk = [(k, results[k][1], results[k][2]) for k in IRREPS]
    print(f"P3  clean-rank fractions: " + ", ".join(f"{k}:{nk}/{Nb}" for k, nk, Nb in rk)
          + "   (old half-step engine at comparable size: ~60%)")
    fd = np.sort(np.load(os.path.join(HERE, "fdfd_m7_px24.npy"))) if m == 7 else None
    if fd is not None:
        allf = np.sort(np.concatenate([results[k][0] for k in IRREPS]))
        print(f"P4  vs FDFD anchor: ground {allf[0]:.7f} vs {fd[0]:.7f} "
              f"(Δ={allf[0]-fd[0]:+.2e}); lowest-4 model: "
              + " ".join(f"{x:.6f}" for x in allf[:4]))
    np.savez(os.path.join(HERE, f"stage_a2_integer_m{m}.npz"),
             **{f"f_{k}": results[k][0] for k in IRREPS},
             ranks=np.array([[results[k][1], results[k][2]] for k in IRREPS]),
             m=m, J=Jmax, nbands=nbands)
    print("saved stage_a2_integer npz")


if __name__ == "__main__":
    main()
