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

    # H,S CHECKPOINT (the assembly+eigh is the long, crash-prone phase)
    hsf = args.out.replace(".npz", "_HS.npz")
    if os.path.isfile(hsf):
        d = np.load(hsf); H, S = d["H"], d["S"]
        print(f"  H,S checkpoint loaded ({H.shape})", flush=True)
    else:
        # H = C† diag(kin) C · dA  (Parseval)
        Ck = C.multiply(kin[:, None])
        H = (C.conj().T @ Ck).toarray() * dA
        # S via per-basis FFT-convolution: S[:,β]=C† FFT(ε_bl·IFFT(ĉ^β))·dA
        S = np.zeros((Nb, Nb), np.complex128)
        Ch = C.conj().T.tocsr()
        for bcol in range(Nb):
            col = np.asarray(C.getcol(bcol).todense()).reshape(Ngrid, Ngrid)
            w = np.fft.ifft2(col) * npix
            Meps = (np.fft.fft2(eps_bl * w).ravel() / npix)
            S[:, bcol] = (Ch @ Meps) * dA
            if bcol % 1000 == 0:
                print(f"    S col {bcol}/{Nb}", flush=True)
        H = 0.5 * (H + H.conj().T); S = 0.5 * (S + S.conj().T)
        np.savez(hsf, H=H, S=S)
        print("  H,S assembled+saved", flush=True)

    sval, svec = eigh(S)
    keep = sval > args.s_tol * sval.max()
    Vp = svec[:, keep] / np.sqrt(sval[keep])
    Hp = 0.5 * (Vp.conj().T @ H @ Vp + (Vp.conj().T @ H @ Vp).conj().T)
    w = eigh(Hp, eigvals_only=True)
    fvals = np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi))
    lo, hi = args.window
    win = fvals[(fvals >= lo) & (fvals <= hi)]
    fd = np.sort(np.load(args.fdfd)["freqs_xmanifold"]) if os.path.exists(args.fdfd) else np.array([])
    np.savez(args.out, freqs=fvals, m=args.m, nref=len(sgrid), n_basis=Nb,
             n_kept=int(keep.sum()), two_valley=args.two_valley, gcut=args.gcut,
             nbands=args.nbands)
    print(f"S-rank {int(keep.sum())}/{Nb} | window [{lo},{hi}]: Galerkin {len(win)}"
          + (f" vs FDFD X-manifold {len(fd)}" if fd.size else ""), flush=True)
    print("  Galerkin window:", " ".join(f"{x:.5f}" for x in win[:16]), flush=True)
    if fd.size:
        print("  FDFD X-manifold:", " ".join(f"{x:.5f}" for x in fd[:8]), flush=True)


if __name__ == "__main__":
    main()
