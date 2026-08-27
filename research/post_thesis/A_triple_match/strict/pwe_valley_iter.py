#!/usr/bin/env python3
"""Stage B escalation - iterative valley-windowed PWE for windows beyond dense-eigh RAM.

Same pencil as pwe_valley.py (H = diag|k|^2, S = eps_hat Toeplitz on the window
k = X + g_mono + G_env), but S is applied MATRIX-FREE through the convolution it comes
from: scatter the window coefficients onto the N x N primitive reciprocal grid, multiply
by eps in real space, gather back (2 FFTs per apply - the window operator is a restriction
of multiplication-by-eps). The manifold is interior spectrum (the folded band-0 tower lies
below it), so the eigensolver is ARPACK shift-invert about a sigma inside the manifold
window, with (H - sigma S) x = b solved by Jacobi-preconditioned MINRES. eps_hat is real
(C2 about the origin => eps(-r) = eps(r)), so the whole pencil is real-symmetric.

Gates (run before any production window):
  --check-dense : build the dense S on the same window; verify the FFT matvec to machine
                  precision and the eigsh spectrum against dense eigh.
  Reproduce the dense {18,18,18,12,6} even-sector ladder (pwe_valley_m57_final.npz).

Usage:
  pwe_valley_iter.py --m 57 --px 16 --renv-shells 18,18,18,12,6 --sector even --k 16
  pwe_valley_iter.py --m 57 --px 16 --renv-shells 12,12,12,8 --check-dense
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.fft as sfft
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh, minres

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

X = np.array([np.pi, 0.0])


def build_window(m, px, renv_shells, renv, gmono, sector, r1, r2, eps2):
    N = px * round(((m * m + 1) / 2) ** 0.5)
    cache = os.path.join(HERE, f"eps_prim_m{m}_px{px}_r{r1:g}_{r2:g}_e{eps2:g}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        eps, B = d["eps"], d["B"]
    else:
        eps, info = build_bilayer_eps_asym(m, 1, r1, r2, 8.9, eps2, 1.0, N, N, 8,
                                           "primitive")
        B = np.asarray(info["B_super"], float)
        np.savez(cache, eps=eps, B=B)
    bp = 2 * np.pi * np.linalg.inv(B).T
    Bc = np.array([[float(m), -1.0], [1.0, float(m)]])
    bc = 2 * np.pi * np.linalg.inv(Bc).T
    bnorm = np.linalg.norm(bp[:, 0])
    par = 0 if sector == "even" else 1
    if renv_shells:
        shell_r = [float(x) for x in renv_shells.split(",")]
        gm_ = len(shell_r) - 1
    else:
        gm_ = gmono
        shell_r = [renv] * (gm_ + 1)
    monos = [(k1, k2) for k1 in range(-gm_, gm_ + 1) for k2 in range(-gm_, gm_ + 1)]

    def env_disk(r):
        Jm = int(np.ceil(r * bnorm / np.linalg.norm(bc[:, 0]))) + 2
        return [(o1, o2) for o1 in range(-Jm, Jm + 1) for o2 in range(-Jm, Jm + 1)
                if (o1 + o2) % 2 == par
                and np.linalg.norm(o1 * bc[:, 0] + o2 * bc[:, 1]) <= r * bnorm + 1e-12]

    disks = {r: env_disk(r) for r in set(shell_r)}
    idx_set = {}
    for (k1, k2) in monos:
        r = shell_r[max(abs(k1), abs(k2))]
        n1m = m * k1 + k2
        n2m = -k1 + m * k2
        for (o1, o2) in disks[r]:
            idx_set[(n1m + o1, n2m + o2)] = True
    nn = np.array(sorted(idx_set.keys()))
    assert np.all((nn[:, 0] + nn[:, 1]) % 2 == par), "window parity broken"
    return eps, N, B, bc, nn, par, shell_r, gm_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=57)
    ap.add_argument("--px", type=int, default=16)
    ap.add_argument("--renv", type=float, default=12.0)
    ap.add_argument("--gmono", type=int, default=3)
    ap.add_argument("--renv-shells", type=str, default=None)
    ap.add_argument("--sector", choices=["even", "odd"], default="even")
    ap.add_argument("--window", type=float, nargs=2, default=[0.3661, 0.3785])
    ap.add_argument("--floor", type=float, default=0.370907)
    ap.add_argument("--sigma-f", type=float, default=0.3722,
                    help="shift-invert target frequency (inside the manifold window)")
    ap.add_argument("--k", type=int, default=16, help="eigenpairs per C2 block")
    ap.add_argument("--tol", type=float, default=1e-9, help="ARPACK tolerance")
    ap.add_argument("--minres-rtol", type=float, default=1e-10)
    ap.add_argument("--minres-maxiter", type=int, default=20000)
    ap.add_argument("--r1", type=float, default=0.20)
    ap.add_argument("--r2", type=float, default=0.10)
    ap.add_argument("--eps2", type=float, default=8.9)
    ap.add_argument("--check-dense", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()

    eps, N, B, bc, nn, par, shell_r, gm_ = build_window(
        args.m, args.px, args.renv_shells, args.renv, args.gmono, args.sector,
        args.r1, args.r2, args.eps2)
    m = args.m
    Nb = len(nn)
    dA = abs(np.linalg.det(B))
    epshat = np.fft.fft2(eps) / (N * N)
    assert abs(epshat.imag).max() < 1e-12 * abs(epshat.real).max(), "eps_hat not real"
    cfg = (f"shells={shell_r}" if args.renv_shells else f"renv={args.renv}|b| gmono={gm_}")
    print(f"m={m} px={args.px} N={N}: PW window {cfg} sector={args.sector} -> Nb={Nb}",
          flush=True)

    # ---- C2-symmetrize the window by intersection (identical operator to pwe_valley.py)
    nC2 = np.array([-m, 1])
    lookup = {tuple(v): i for i, v in enumerate(nn)}
    partner = np.array([lookup.get((int(nC2[0] - a), int(nC2[1] - b)), -1) for a, b in nn])
    keep = partner >= 0
    print(f"  C2 closure: {keep.sum()}/{Nb} have partners in-window", flush=True)
    nn = nn[keep]
    old2new = -np.ones(Nb, int)
    old2new[np.where(keep)[0]] = np.arange(keep.sum())
    partner = old2new[partner[keep]]
    Nk = len(nn)
    kvec = X[None, :] + nn[:, 0:1] * bc[:, 0][None, :] + nn[:, 1:2] * bc[:, 1][None, :]
    kin = (kvec ** 2).sum(1)

    # ---- momentum -> primitive-FFT grid index (subtract the parity offset (par, 0))
    q1 = ((nn[:, 0] - par) + nn[:, 1]) // 2 % N
    q2 = (nn[:, 1] - (nn[:, 0] - par)) // 2 % N
    assert np.all(((nn[:, 0] - par) + nn[:, 1]) % 2 == 0)
    assert len(set(zip(q1.tolist(), q2.tolist()))) == Nk, "grid index collision"

    buf = np.zeros((N, N), complex)

    def apply_S_full(c):
        """(S c)_u = dA * sum_v eps_hat(k_u - k_v) c_v  via 2 FFTs."""
        buf[:] = 0.0
        buf[q1, q2] = c
        out = sfft.fft2(eps * sfft.ifft2(buf, workers=-1), workers=-1)
        return dA * out[q1, q2].real if np.isrealobj(c) else dA * out[q1, q2]

    # ---- C2 orbit basis (reps; fixed points only in the + block)
    reps = np.array([i for i in range(Nk) if i <= partner[i]])
    fixed = partner[reps] == reps
    prt = partner[reps]
    wgt = np.where(fixed, 1.0, 1.0 / np.sqrt(2))

    def block_ops(sgn):
        sel = np.ones(len(reps), bool) if sgn > 0 else ~fixed
        ra, pa, wv = reps[sel], prt[sel], wgt[sel]
        fx = fixed[sel]
        R = len(ra)
        kd = kin[ra] * dA

        def embed(x):
            c = np.zeros(Nk)
            c[ra] = wv * x
            c[pa[~fx]] += sgn * wv[~fx] * x[~fx]
            return c

        def project(c):
            y = wv * c[ra]
            y[~fx] += sgn * wv[~fx] * c[pa[~fx]]
            return y

        def Sx(x):
            return project(apply_S_full(embed(x)))

        return R, kd, embed, project, Sx

    sig_lam = (2 * np.pi * args.sigma_f) ** 2
    results = {}
    stats = {}
    for tag, sgn in [("C2+", 1.0), ("C2-", -1.0)]:
        R, kd, embed, project, Sx = block_ops(sgn)
        Sop = LinearOperator((R, R), matvec=Sx, dtype=float)
        Hop = LinearOperator((R, R), matvec=lambda x: kd * x, dtype=float)

        def Kx(x):
            return kd * x - sig_lam * Sx(x)

        Kop = LinearOperator((R, R), matvec=Kx, dtype=float)
        eps0 = epshat[0, 0].real
        dvec = kd - sig_lam * eps0 * dA
        dvec = np.where(np.abs(dvec) < 1e-2, 1e-2, np.abs(dvec))
        Mpre = LinearOperator((R, R), matvec=lambda x: x / dvec, dtype=float)
        nsolve = [0]
        nit = [0]

        def opinv(b):
            xs, info = minres(Kop, b, M=Mpre, rtol=args.minres_rtol,
                              maxiter=args.minres_maxiter)
            if info != 0:
                print(f"    [warn] minres info={info} at solve {nsolve[0]}", flush=True)
            nsolve[0] += 1
            return xs

        OPinv = LinearOperator((R, R), matvec=opinv, dtype=float)
        tb = time.time()
        w, V = eigsh(Hop, k=min(args.k, R - 1), M=Sop, sigma=sig_lam, OPinv=OPinv,
                     which="LM", mode="normal", tol=args.tol)
        order = np.argsort(w)
        w, V = w[order], V[:, order]
        f = np.sqrt(np.maximum(w, 0)) / (2 * np.pi)
        results[tag] = f
        stats[tag] = dict(R=R, solves=nsolve[0], secs=time.time() - tb)
        lo, hi = args.window
        win = f[(f >= lo) & (f <= hi)]
        msg = (f"  [{tag}] R={R}  {nsolve[0]} inner solves, {time.time()-tb:.0f}s"
               f"  window: {win.size} states")
        if win.size:
            msg += f"  bottom {win[0]:.6f}  Dfloor {win[0]-args.floor:+.2e}"
        print(msg, flush=True)
        print("    ladder:", " ".join(f"{x:.6f}" for x in f), flush=True)

        if args.check_dense:
            dn1 = nn[:, 0][:, None] - nn[:, 0][None, :]
            dn2 = nn[:, 1][:, None] - nn[:, 1][None, :]
            Sd = epshat[((dn1 + dn2) // 2) % N, ((dn2 - dn1) // 2) % N].real * dA
            Sd = 0.5 * (Sd + Sd.T)
            rng = np.random.default_rng(0)
            c = rng.standard_normal(Nk)
            err = np.linalg.norm(apply_S_full(c) - Sd @ c) / np.linalg.norm(Sd @ c)
            print(f"    [gate1a {tag}] FFT-vs-dense matvec rel err = {err:.2e}", flush=True)
            sel = np.ones(len(reps), bool) if sgn > 0 else ~fixed
            ra, pa, wv = reps[sel], prt[sel], wgt[sel]
            fx = fixed[sel]
            P = np.zeros((Nk, len(ra)))
            P[ra, np.arange(len(ra))] = wv
            P[pa[~fx], np.arange(len(ra))[~fx]] = sgn * wv[~fx]
            Hb = P.T @ (kin[:, None] * P) * dA
            Sb = P.T @ (Sd @ P)
            wd = eigh(0.5 * (Hb + Hb.T), 0.5 * (Sb + Sb.T), eigvals_only=True)
            fd = np.sqrt(np.maximum(wd, 0)) / (2 * np.pi)
            near = np.array([fd[np.argmin(abs(fd - x))] for x in f])
            print(f"    [gate1b {tag}] max |f_iter - f_dense| = {abs(f-near).max():.2e}",
                  flush=True)

    out = args.out or (f"pwe_iter_m{m}_" + (f"sh{'_'.join(f'{float(x):g}' for x in (args.renv_shells or '').split(','))}"
                       if args.renv_shells else f"r{args.renv:g}_g{gm_}") + f"_px{args.px}_{args.sector}.npz")
    np.savez(os.path.join(HERE, out),
             **{f"f_{k.replace('+', 'p').replace('-', 'm')}": v for k, v in results.items()},
             m=m, shells=str(shell_r), sector=args.sector, Nb=Nk, sigma_f=args.sigma_f)
    print(f"saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
