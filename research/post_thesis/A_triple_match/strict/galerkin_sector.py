#!/usr/bin/env python3
"""Stage B — the corrected Galerkin engine: single-valley, primitive cell, integer momenta.

§15 collapses the design: (i) the admissible momenta are integer reciprocal shifts (no half-step
aliasing); (ii) in the PRIMITIVE frame (cell='primitive', area (m²+1)/2) the two valleys are
different Bloch momenta q± (T_P1 sectors), C4-related and EXACTLY decoupled (ε̂ has zero
odd-parity support) — so solving ONE valley at the primitive cell captures everything: the other
valley's ladder is identical by C4, and the centered-cell spectrum is the two-fold union.
This corrects §11's narrative: ε_bl never coupled X↔X′ (that Fourier component is exactly zero);
the "two-valley completion" added the decoupled partner states, and the 4-fold is C4 × emergent.

Engine: trial functions E = e^{ip·r}u_n(r;p;s_k), p = X + j₁bp₁ + j₂bp₂ (INTEGER j, primitive
reciprocals bp), registry frames s_k on a K×K grid; per-frame MPB extraction checkpointed.
H = dA·C†·diag|X+G|²·C matrix-free; S per-column FFT-convolution with ε_prim; canonical
orthogonalization via evr + s_tol. Optional C2-block split (C2 is in the little group of
primitive-X; halves each solve, labels states) — default on.

Comparison: the operator-consistent floor is stage_a3_spectral (cell=primitive, matched px);
the FDFD sector ladder is stage_a4_prim (f_plus). Manifold = 12 states per sector at 2°.

Usage: galerkin_sector.py --m 57 --px 16 --J 6 --nbands 2 --nref 3 --out gsec_2deg.npz
"""
import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import galerkin_moire as gm
from supercell_asym import build_bilayer_eps_asym

X = gm.X


def basis_coeffs_prim(u_kb, p, B_prim, N):
    """Sparse primitive-cell plane-wave coeffs of w = e^{-iX·r}E. All indices exactly integer."""
    res = u_kb.shape[0]
    c = np.fft.fft2(u_kb) / (res * res)
    gi = np.fft.fftfreq(res) * res
    G1, G2 = np.meshgrid(gi, gi, indexing="ij")
    Qx = (p[0] - X[0]) + 2 * np.pi * G1
    Qy = (p[1] - X[1]) + 2 * np.pi * G2
    n1f = (B_prim[0, 0] * Qx + B_prim[1, 0] * Qy) / (2 * np.pi)
    n2f = (B_prim[0, 1] * Qx + B_prim[1, 1] * Qy) / (2 * np.pi)
    n1 = np.rint(n1f).astype(int)
    n2 = np.rint(n2f).astype(int)
    # integrality guard: the §15 aliasing defect must be impossible here
    tie = max(np.abs(n1f - n1).max(), np.abs(n2f - n2).max())
    assert tie < 1e-6, f"non-integer supercell index (tie {tie:.3e}) — inadmissible momentum"
    flat = ((n1 % N) * N + (n2 % N)).ravel()
    vals = c.ravel()
    order = np.argsort(flat)
    flat_s, vals_s = flat[order], vals[order]
    uniq, start = np.unique(flat_s, return_index=True)
    return uniq, np.add.reduceat(vals_s, start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--J", type=int, default=6, help="integer momentum window: j in [-J..J]^2")
    ap.add_argument("--nbands", type=int, default=2)
    ap.add_argument("--band-lo", type=int, default=0)
    ap.add_argument("--nref", type=int, default=1)
    ap.add_argument("--sbar", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--s-tol", type=float, default=1e-6)
    ap.add_argument("--r1", type=float, default=0.20)
    ap.add_argument("--r2", type=float, default=0.10)
    ap.add_argument("--c2-split", type=int, default=1, help="block-solve by C2 parity")
    ap.add_argument("--window", type=float, nargs=2, default=[0.3661, 0.3785])
    ap.add_argument("--floor", type=float, default=None,
                    help="operator-consistent floor (stage_a3 primitive) for Δ reporting")
    ap.add_argument("--out", default="gsec.npz")
    args = ap.parse_args()

    m = args.m
    # primitive cell geometry
    N = args.px * round(((m * m + 1) / 2) ** 0.5)
    npix = N * N
    eps, info = build_bilayer_eps_asym(m, 1, args.r1, args.r2, 8.9, 8.9, 1.0, N, N, 8, "primitive")
    B_prim = np.asarray(info["B_super"], float)
    bp = 2 * np.pi * np.linalg.inv(B_prim).T          # primitive reciprocals (columns)
    js = np.arange(-args.J, args.J + 1)
    jj = [(j1, j2) for j1 in js for j2 in js]
    momenta = np.array([X + j1 * bp[:, 0] + j2 * bp[:, 1] for j1, j2 in jj])
    bands = list(range(args.band_lo, args.band_lo + args.nbands))
    nb_solve = args.band_lo + args.nbands
    if args.nref <= 1:
        sgrid = [tuple(args.sbar)]
    else:
        K = args.nref
        sgrid = [(i / K, j / K) for i in range(K) for j in range(K)]
    print(f"m={m} px={args.px} PRIMITIVE N={N} J={args.J} bands={bands} nref={len(sgrid)} "
          f"momenta={len(momenta)} Nb={len(sgrid)*len(momenta)*len(bands)}", flush=True)

    # sparse C with per-reference checkpoints
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
                idx, val = basis_coeffs_prim(u[ik, bd], momenta[ik], B_prim, N)
                rr.append(idx); cc.append(np.full(idx.size, loc)); vv.append(val)
                loc += 1
        rr = np.concatenate(rr); cc = np.concatenate(cc); vv = np.concatenate(vv)
        np.savez(ckf, r=rr, c=cc, v=vv)
        rows.append(rr); cols.append(cc + ri * n_per_ref); data.append(vv)
        print(f"  ref {ri} {sk} extracted+saved", flush=True)
    Nb = len(sgrid) * n_per_ref
    C = sp.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(npix, Nb), dtype=np.complex128)
    print(f"  sparse C: {C.nnz} nnz", flush=True)

    fr = np.fft.fftfreq(N) * N
    N1, N2 = np.meshgrid(fr, fr, indexing="ij")
    Gx = N1 * bp[0, 0] + N2 * bp[0, 1]
    Gy = N1 * bp[1, 0] + N2 * bp[1, 1]
    kin = ((X[0] + Gx) ** 2 + (X[1] + Gy) ** 2).ravel()
    dA = abs(np.linalg.det(B_prim))

    # C2 parity blocks: C2@origin is in the little group of primitive-X (−X ≡ X mod 2π·e1,
    # a primitive reciprocal). On coeffs: C2 index map n → nC2 − n with nC2 = B_primᵀ(C2X−X)/2π.
    blocks = {}
    if args.c2_split:
        nC2 = np.rint(B_prim.T @ (np.array([-np.pi, 0.0]) - X) / (2 * np.pi)).astype(int)
        allflat = np.arange(npix)
        i1f, i2f = allflat // N, allflat % N
        permC2 = ((nC2[0] - i1f) % N) * N + ((nC2[1] - i2f) % N)
        # project each column onto C2 = ±1
        Cc = C.tocsc()
        for tag, sgn in [("C2+", 1.0), ("C2-", -1.0)]:
            rows2, cols2, data2, col = [], [], [], 0
            for b in range(Nb):
                sl = slice(Cc.indptr[b], Cc.indptr[b + 1])
                idx, val = Cc.indices[sl], Cc.data[sl]
                acc = {}
                for fi, vv in zip(idx, val):
                    acc[fi] = acc.get(fi, 0.0 + 0.0j) + 0.5 * vv
                    fj = permC2[fi]
                    acc[fj] = acc.get(fj, 0.0 + 0.0j) + 0.5 * sgn * vv
                fis = np.fromiter(acc.keys(), dtype=np.int64)
                vs = np.fromiter(acc.values(), dtype=np.complex128)
                nz = np.abs(vs) > 1e-14
                if nz.sum() == 0:
                    continue
                rows2.append(fis[nz]); cols2.append(np.full(nz.sum(), col)); data2.append(vs[nz])
                col += 1
            blocks[tag] = sp.csc_matrix(
                (np.concatenate(data2), (np.concatenate(rows2), np.concatenate(cols2))),
                shape=(npix, col), dtype=np.complex128)
    else:
        blocks["all"] = C

    def solve_block(Cb):
        Nb_ = Cb.shape[1]
        Ch = Cb.conj().T.tocsr(); Csr = Cb.tocsr()
        S = np.zeros((Nb_, Nb_), np.complex128)
        for b in range(Nb_):
            col = np.asarray(Cb.getcol(b).todense()).reshape(N, N)
            w = np.fft.ifft2(col) * npix
            S[:, b] = (Ch @ (np.fft.fft2(eps * w).ravel() / npix)) * dA
            if Nb_ > 2000 and b % 1000 == 0:
                print(f"    S col {b}/{Nb_}", flush=True)
        S = 0.5 * (S + S.conj().T)
        smax = eigh(S, subset_by_index=[Nb_ - 1, Nb_ - 1], eigvals_only=True, driver="evr")[0]
        sval, svec = eigh(S, subset_by_value=(args.s_tol * smax, np.inf), driver="evr")
        Vp = svec / np.sqrt(sval)
        Hp = Vp.conj().T @ (dA * (Ch @ (kin[:, None] * (Csr @ Vp))))
        Hp = 0.5 * (Hp + Hp.conj().T)
        w = eigh(Hp, eigvals_only=True, driver="evr")
        return np.sort(np.sqrt(np.maximum(w.real, 0)) / (2 * np.pi)), int(sval.size), Nb_

    out = {}
    for tag, Cb in blocks.items():
        f, nk, Nb_ = solve_block(Cb)
        out[tag] = f
        lo, hi = args.window
        win = f[(f >= lo) & (f <= hi)]
        msg = f"  [{tag}] Nb={Nb_} rank={nk}  window[{lo},{hi}]: {win.size} states"
        if win.size:
            msg += f"  bottom {win[0]:.6f}"
            if args.floor is not None:
                msg += f"  Δfloor {win[0]-args.floor:+.2e}"
        print(msg, flush=True)
        print("    lowest8:", " ".join(f"{x:.6f}" for x in f[:8]), flush=True)
    np.savez(args.out, **{f"f_{k.replace('+','p').replace('-','m')}": v for k, v in out.items()},
             m=m, px=args.px, J=args.J, nbands=args.nbands, nref=len(sgrid))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
