#!/usr/bin/env python3
"""Reciprocal-space (plane-wave) Galerkin moiré continuum model — the
memory-robust, accurate engine for the registry-adapted exact continuum model.

Each reference-Bloch basis function is stored as SPARSE supercell plane-wave
coefficients (~res² nonzeros, not the full Ngrid² real-space field), so a
large multi-reference basis fits in <1 GB even at px16. The ε_bl coupling is
computed by per-basis FFT-convolution (one transient grid at a time). Same
generalized eigenproblem H c = λ S c as galerkin_moire.py, to which it is
numerically identical (validated), but scalable to registry-adapted bases.

Basis: work with the supercell-periodic amplitude w = e^{-iX·r}E of each
E_{n,p,s_k}=e^{ip·r}u_n(r;p;s_k). Then
    ĉ^b(Gs) = c_n(g;p;s_k)   at   Gs = (p−X) + g   (supercell reciprocal),
    H_αβ = Σ_Gs ĉ^α*(Gs) |X+Gs|² ĉ^β(Gs)             (kinetic, diagonal),
    S_αβ = ⟨w_α|ε_bl|w_β⟩ = ĉ^α* · FFT(ε_bl · IFFT(ĉ^β)).
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galerkin_moire as gm
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = gm.X


def basis_coeffs(u_kb, p, B_super, Ngrid):
    """Sparse supercell-reciprocal coeffs of w=e^{-iX·r}E for one basis fn.
    Returns (flat_indices, values) on the Ngrid×Ngrid FFT grid."""
    res = u_kb.shape[0]
    c = np.fft.fft2(u_kb) / (res * res)                 # monolayer coeffs c_n(g)
    gi = np.fft.fftfreq(res) * res                       # integer g indices
    G1, G2 = np.meshgrid(gi, gi, indexing="ij")
    gx = 2 * np.pi * G1
    gy = 2 * np.pi * G2                                  # monolayer recip cartesian
    Qx = (p[0] - X[0]) + gx                              # (p-X)+g cartesian
    Qy = (p[1] - X[1]) + gy
    # supercell reciprocal integer index: n = B_super^T Q / (2π)
    n1 = np.rint((B_super[0, 0] * Qx + B_super[1, 0] * Qy) / (2 * np.pi)).astype(int)
    n2 = np.rint((B_super[0, 1] * Qx + B_super[1, 1] * Qy) / (2 * np.pi)).astype(int)
    i1 = n1 % Ngrid
    i2 = n2 % Ngrid
    flat = (i1 * Ngrid + i2).ravel()
    vals = c.ravel()
    # collapse duplicate indices (different g mapping to same supercell G)
    order = np.argsort(flat)
    flat_s = flat[order]; vals_s = vals[order]
    uniq, start = np.unique(flat_s, return_index=True)
    summed = np.add.reduceat(vals_s, start)
    return uniq, summed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--gcut", type=int, default=4)
    ap.add_argument("--nbands", type=int, default=2)
    ap.add_argument("--band-lo", type=int, default=0)
    ap.add_argument("--r1", type=float, default=0.20, help="layer-1 rod radius")
    ap.add_argument("--r2", type=float, default=0.10, help="layer-2 rod radius (weak knob)")
    ap.add_argument("--nref", type=int, default=1, help="K for K×K registry grid")
    ap.add_argument("--two-valley", action="store_true",
                    help="add an X'=(0,π) carrier patch to the basis (the missing "
                         "second valley; X'-X is a supercell-G vector so everything "
                         "downstream — kinetic |X+G|², basis_coeffs, ε-coupling — is "
                         "valley-agnostic and unchanged)")
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.23046875, 0.10546875])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--s-tol", type=float, default=1e-6)
    ap.add_argument("--window", type=float, nargs=2, default=[0.365, 0.385])
    ap.add_argument("--fdfd", default=os.path.join(HERE, "fdfd_xman_2deg.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "galerkin_recip.npz"))
    args = ap.parse_args()

    g, B_super, theta = gm.moire_g(args.m)
    Ngrid = args.px * round(float(np.sqrt(args.m**2 + 1)))
    npix = Ngrid * Ngrid
    J = 2 * args.gcut + 1
    js = np.arange(-J, J + 1)
    # carriers: X=(π,0) always; add X'=(0,π) when two-valley. Both fold to Q_X
    # (X'-X=(-π,π) is a supercell reciprocal vector for odd m), so a single
    # eigh(H,S) at Q_X spans both valleys and can host the X⊕X' 4-fold cluster.
    carriers = [X] + ([np.array([0.0, np.pi])] if args.two_valley else [])
    momenta = np.array([c + 0.5 * j1 * g[:, 0] + 0.5 * j2 * g[:, 1]
                        for c in carriers for j1 in js for j2 in js])
    bands = list(range(args.band_lo, args.band_lo + args.nbands))
    nb_solve = args.band_lo + args.nbands
    if args.nref <= 1:
        sgrid = [tuple(args.sbar)]
    else:
        # grid ON the high-symmetry registries (includes 0, ½ — the valley /
        # well-bottom registries where the manifold ground state concentrates)
        K = args.nref
        sgrid = [(i / K, j / K) for i in range(K) for j in range(K)]
    print(f"m={args.m} px={args.px} Ngrid={Ngrid} gcut={args.gcut} bands={bands} "
          f"n_ref={len(sgrid)} momenta={len(momenta)}", flush=True)

    # build sparse coeff matrix C (nPW × Nb) with per-reference CHECKPOINTS
    ckdir = args.out.replace(".npz", "_ck")
    os.makedirs(ckdir, exist_ok=True)
    n_per_ref = len(momenta) * len(bands)
    rows, cols, data = [], [], []
    for ri, sk in enumerate(sgrid):
        ckf = os.path.join(ckdir, f"ref{ri}.npz")
        if os.path.isfile(ckf):
            d = np.load(ckf)
            rows.append(d["r"]); cols.append(d["c"] + ri * n_per_ref); data.append(d["v"])
            print(f"  ref {ri} {sk}: checkpoint", flush=True)
            continue
        u, _, _ = gm.extract_reference_bloch(sk, momenta, nb_solve, args.res,
                                             r1=args.r1, r2=args.r2)
        rr, cc, vv, loc = [], [], [], 0
        for ik in range(len(momenta)):
            for bd in bands:
                idx, val = basis_coeffs(u[ik, bd], momenta[ik], B_super, Ngrid)
                rr.append(idx); cc.append(np.full(idx.size, loc)); vv.append(val)
                loc += 1
        rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
        np.savez(ckf, r=rr, c=cc, v=vv)
        rows.append(rr); cols.append(cc + ri * n_per_ref); data.append(vv)
        print(f"  ref {ri} {sk} extracted+saved", flush=True)
    Nb = len(sgrid) * n_per_ref
    C = sp.csc_matrix((np.concatenate(data),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(npix, Nb), dtype=np.complex128)
    print(f"  sparse C: {C.nnz} nnz, {C.nnz/Nb:.0f}/basis", flush=True)

    # kinetic |X+Gs|² on the FFT grid
    b_sup = 2 * np.pi * np.linalg.inv(B_super).T
    fr = np.fft.fftfreq(Ngrid) * Ngrid
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * b_sup[0, 0] + N2 * b_sup[0, 1]
    Gy = N1 * b_sup[1, 0] + N2 * b_sup[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()
    dA = abs(np.linalg.det(B_super))                  # cell area; Parseval below

    eps_bl, _ = build_bilayer_eps_asym(args.m, 1, args.r1, args.r2, 8.9, 8.9, 1.0,
                                       Ngrid, Ngrid, 8, "centered")

    # S CHECKPOINT (the FFT-convolution assembly is the long phase). H is kept
    # MATRIX-FREE from the sparse C (H = dA·C†·diag(kin)·C), never densified —
    # forming dense H (Nb²) + the zheevd O(Nb²) workspace was the OOM driver.
    Csr = C.tocsr(); Ch = C.conj().T.tocsr()

    def H_op(V):                            # V:(Nb,k) -> dA·C†·(kin⊙(C·V)), blocked
        out = np.empty((V.shape[0], V.shape[1]), np.complex128)
        for j0 in range(0, V.shape[1], 64):
            j1 = min(j0 + 64, V.shape[1])
            CV = Csr @ V[:, j0:j1]          # (npix, blk)
            out[:, j0:j1] = dA * (Ch @ (kin[:, None] * CV))
        return out

    hsf = args.out.replace(".npz", "_HS.npz")
    if os.path.isfile(hsf):
        S = np.load(hsf)["S"]               # (also accepts legacy H,S checkpoints)
        print(f"  S checkpoint loaded ({S.shape})", flush=True)
    else:
        S = np.zeros((Nb, Nb), np.complex128)
        for bcol in range(Nb):
            col = np.asarray(C.getcol(bcol).todense()).reshape(Ngrid, Ngrid)
            w = np.fft.ifft2(col) * npix
            Meps = (np.fft.fft2(eps_bl * w).ravel() / npix)
            S[:, bcol] = (Ch @ Meps) * dA
            if bcol % 1000 == 0:
                print(f"    S col {bcol}/{Nb}", flush=True)
        S = 0.5 * (S + S.conj().T)
        np.savez(hsf, S=S)
        print("  S assembled+saved", flush=True)

    # canonical orthogonalization via evr (O(Nb) workspace, returns only the kept
    # subspace) + matrix-free Hp — replaces the zheevd full eigh(S)+dense H·Vp.
    smax = eigh(S, subset_by_index=[Nb - 1, Nb - 1], eigvals_only=True,
                driver="evr")[0]
    sval, svec = eigh(S, subset_by_value=(args.s_tol * smax, np.inf), driver="evr")
    Vp = svec / np.sqrt(sval)               # (Nb, n_kept)
    Hp = Vp.conj().T @ H_op(Vp)             # matrix-free H, dense only in n_kept²
    Hp = 0.5 * (Hp + Hp.conj().T)
    w, y = eigh(Hp, driver="evr")           # eigenvalues + eigenvectors (kept subspace)
    n_kept = int(sval.size)                 # number of retained (well-conditioned) modes
    fvals = np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi)   # (unsorted, aligned with y)
    # BAND-1 WEIGHT per eigenstate (to filter the band-1 manifold from band-0
    # active-band pollution: band_lo=0 makes band 0 active, producing spurious
    # sub-manifold states). Basis col c has band = bands[c % len(bands)]; the
    # band-1 rows are those with bands[...]==1. Compute only for near-window
    # states (memory-lean: Vp@y is Nb×n_kept, so slice y first).
    lo, hi = args.window
    band_of_col = np.array([bands[c % len(bands)] for c in range(Nb)])
    is_b1 = (band_of_col == 1)
    near = np.where((fvals >= lo - 0.01) & (fvals <= hi + 0.01))[0]
    b1w = np.full(fvals.size, np.nan)
    if near.size:
        cvec = Vp @ y[:, near]              # (Nb, n_near) coeffs of near-window states
        p = np.abs(cvec) ** 2
        b1w[near] = p[is_b1].sum(0) / p.sum(0)
    order = np.argsort(fvals)
    fvals = fvals[order]; b1w = b1w[order]
    win_mask = (fvals >= lo) & (fvals <= hi)
    win = fvals[win_mask]
    winb1 = b1w[win_mask]
    # band-1 manifold = in-window states dominated by band 1 (matches FDFD's
    # w_X X-manifold selection, computed here from the band character instead)
    man = win[np.nan_to_num(winb1) > 0.5]
    fd = np.sort(np.load(args.fdfd)["freqs_xmanifold"]) if os.path.exists(args.fdfd) else np.array([])
    np.savez(args.out, freqs=fvals, band1_weight=b1w, m=args.m, nref=len(sgrid),
             n_basis=Nb, n_kept=n_kept, two_valley=args.two_valley, gcut=args.gcut,
             nbands=args.nbands)
    if man.size:
        print(f"  band-1 manifold (b1w>0.5): {man.size} states, bottom {man[0]:.6f}"
              + (f"  Δ vs FDFD {man[0]-fd[0]:+.2e}" if fd.size else ""), flush=True)
    print(f"S-rank {n_kept}/{Nb} | window [{lo},{hi}]: Galerkin {len(win)}"
          + (f" vs FDFD X-manifold {len(fd)}" if fd.size else ""), flush=True)
    print("  Galerkin window:", " ".join(f"{x:.5f}" for x in win[:16]), flush=True)
    if fd.size:
        print("  FDFD X-manifold:", " ".join(f"{x:.5f}" for x in fd[:8]), flush=True)


if __name__ == "__main__":
    main()
