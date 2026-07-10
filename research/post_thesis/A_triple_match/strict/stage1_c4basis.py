#!/usr/bin/env python3
"""Stage 1 — does a C4-CLOSED two-valley basis restore the rigorous degeneracy?

The two-valley Galerkin currently extracts the X and X' carrier blocks INDEPENDENTLY from MPB
(arbitrary per-k gauge), so the trial span is only approximately C4-invariant. Stage 0b showed the
FDFD ground 4-fold = regular rep of C4, containing a RIGOROUS T-protected {1E,2E} 2-fold. A basis
that is EXACTLY C4+T-closed must reproduce that 2-fold to machine precision, INDEPENDENT of energy
convergence -- a clean, decisive symmetry test.

This builds the X' block two ways and compares the ground spectrum:
  (indep)  X' from an independent MPB extraction at carrier X'=(0,pi)   [= current engine]
  (c4sym)  X' as the exact C4-image of the X block: on the sparse supercell plane-wave coeffs,
           C4 maps index n -> nG0 + C4.n  (value unchanged), nG0 = B^T(X'-X)/2pi = ((1-m)/2,(1+m)/2),
           C4.(n1,n2)=(-n2,n1). Derivation: M_C4 = B^T C4 B^{-T} = C4 (integer); the G0=X'-X shift
           realizes the Bloch gauge as an index translation.
Metric: the smallest nearest-neighbour gap among the lowest ground states -> an EXACT 2-fold
(<1e-10) signals restored rigorous degeneracy. Reported for both bases.

Usage: stage1_c4basis.py <m> [px] [gcut] [nbands] [sx sy]
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
Xp = np.array([0.0, np.pi])


def build_block(u, momenta, bands, B_super, Ngrid):
    """sparse (npix, nk*nbands) coeffs for a carrier block, columns loc-ordered."""
    rows, cols, data, loc = [], [], [], 0
    for ik in range(len(momenta)):
        for bd in bands:
            idx, val = basis_coeffs(u[ik, bd], momenta[ik], B_super, Ngrid)
            rows.append(idx); cols.append(np.full(idx.size, loc)); data.append(val); loc += 1
    return (np.concatenate(rows), np.concatenate(cols), np.concatenate(data), loc)


def c4_remap_flat(flat, Ngrid, nG0):
    """C4 on supercell plane-wave indices: n -> nG0 + C4.n, C4.(n1,n2)=(-n2,n1)."""
    i1, i2 = flat // Ngrid, flat % Ngrid
    j1 = (nG0[0] - i2) % Ngrid
    j2 = (nG0[1] + i1) % Ngrid
    return j1 * Ngrid + j2


def solve(C, npix, Ngrid, B_super, kin, eps_bl, s_tol=1e-6):
    Nb = C.shape[1]
    dA = abs(np.linalg.det(B_super))
    Csr = C.tocsr(); Ch = C.conj().T.tocsr()
    S = np.zeros((Nb, Nb), np.complex128)
    for b in range(Nb):
        col = np.asarray(C.getcol(b).todense()).reshape(Ngrid, Ngrid)
        w = np.fft.ifft2(col) * npix
        S[:, b] = (Ch @ (np.fft.fft2(eps_bl * w).ravel() / npix)) * dA
    S = 0.5 * (S + S.conj().T)
    smax = eigh(S, subset_by_index=[Nb - 1, Nb - 1], eigvals_only=True, driver="evr")[0]
    sval, svec = eigh(S, subset_by_value=(s_tol * smax, np.inf), driver="evr")
    Vp = svec / np.sqrt(sval)

    def H_op(V):
        return dA * (Ch @ (kin[:, None] * (Csr @ V)))
    Hp = Vp.conj().T @ H_op(Vp); Hp = 0.5 * (Hp + Hp.conj().T)
    w = eigh(Hp, eigvals_only=True, driver="evr")
    f = np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))
    return f, int(sval.size)


def main():
    m = int(sys.argv[1]); px = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    gcut = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    nbands = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    sbar = np.array([float(sys.argv[5]), float(sys.argv[6])]) if len(sys.argv) > 6 else np.array([0.0, 0.0])
    g, B_super, theta = gm.moire_g(m)
    Ngrid = px * round(float(np.sqrt(m * m + 1))); npix = Ngrid * Ngrid
    J = 2 * gcut + 1; js = np.arange(-J, J + 1)
    bands = list(range(nbands)); nb_solve = nbands
    momX = np.array([X + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1] for j1 in js for j2 in js])
    momXp = np.array([Xp + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1] for j1 in js for j2 in js])
    nG0 = np.array([(1 - m) // 2, (1 + m) // 2])
    print(f"m={m} px={px} Ngrid={Ngrid} gcut={gcut} nbands={nbands} sbar={sbar} "
          f"momenta/carrier={len(momX)} nG0={tuple(nG0)}", flush=True)

    # kinetic + eps
    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid; N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]; Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()
    eps_bl, _ = build_bilayer_eps_asym(m, 1, 0.20, 0.10, 8.9, 8.9, 1.0, Ngrid, Ngrid, 8, "centered")

    # X block (shared)
    uX, _, _ = gm.extract_reference_bloch(sbar, momX, nb_solve, 64, r1=0.20, r2=0.10)
    rX, cX, dX_, nX = build_block(uX, momX, bands, B_super, Ngrid)
    print(f"  X block built ({nX} cols)", flush=True)

    # X' block, INDEPENDENT (registry C4-mapped: sbar -> C4.sbar for consistency)
    sC4 = np.array([(-sbar[1]) % 1.0, sbar[0] % 1.0])
    uXp, _, _ = gm.extract_reference_bloch(sC4, momXp, nb_solve, 64, r1=0.20, r2=0.10)
    rXi, cXi, dXi, nXi = build_block(uXp, momXp, bands, B_super, Ngrid)
    # X' block, C4-IMAGE of the X block
    rX4 = c4_remap_flat(rX, Ngrid, nG0)
    # self-check: the C4-image index SET must equal the independent X' index set
    setc4 = set(np.unique(rX4).tolist()); seti = set(np.unique(rXi).tolist())
    print(f"  C4-image vs independent X' populated-index overlap: "
          f"{len(setc4 & seti)}/{len(setc4)} (image) {len(seti)} (indep)", flush=True)

    results = {}
    for tag, (r_, c_, d_, n_) in [("indep", (rXi, cXi, dXi, nXi)), ("c4sym", (rX4, cX, dX_, nX))]:
        rows = np.concatenate([rX, r_]); cols = np.concatenate([cX, c_ + nX])
        data = np.concatenate([dX_, d_])
        C = sp.csc_matrix((data, (rows, cols)), shape=(npix, nX + n_), dtype=np.complex128)
        f, nkept = solve(C, npix, Ngrid, B_super, kin, eps_bl)
        results[tag] = (f, nkept)
        # smallest gaps in the low spectrum -> exact 2-folds
        lo = f[:12]
        gaps = np.diff(lo)
        print(f"\n[{tag}] Nb={C.shape[1]} S-rank={nkept}  lowest 8: "
              + " ".join(f"{x:.7f}" for x in f[:8]))
        print(f"[{tag}] nn-gaps(lowest 12): " + " ".join(f"{gp:.1e}" for gp in gaps))
        print(f"[{tag}] min gap = {gaps.min():.2e}  (<1e-9 => exact 2-fold restored)")

    np.savez(os.path.join(HERE, f"stage1_c4_m{m}.npz"),
             f_indep=results["indep"][0], f_c4sym=results["c4sym"][0], m=m, gcut=gcut)
    print("\nsaved stage1_c4_m%d.npz" % m)


if __name__ == "__main__":
    main()
