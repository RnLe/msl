#!/usr/bin/env python3
"""Wide sector ladders: 40+ states of the M2 tower in one window, three solvers.

The speed trick: the commensurate moire dielectric of finite-Fourier layers has an
EXACT 13-term supercell Fourier series (lifted.moire_coeffs), so the plane-wave mass
matrix S = eps_hat is SPARSE (13 bands). The TM pencil K u = lambda S u with
K = diag|k_s + G|^2 is then a sparse symmetric pencil:

  * shift-invert reaches interior window states directly (no Lanczos climb through the
    N_cells band-0 states below — that climb is what made the wide runs cost hours),
  * the LDL^T factorization of (K - lambda S) gives an EXACT eigenvalue count below
    lambda by Sylvester inertia, so the window census is certified, not assumed.

Legs: pwe (this sparse Galerkin pencil), fdfd (finite differences on the identical
analytic dielectric, two resolutions + h^2 extrapolation), ea (registry-adapted
single-band raw projection, unchanged from the hero engine).
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
POST_THESIS = os.path.abspath(os.path.join(HERE, "..", ".."))
THESIS_RESULTS = os.path.abspath(os.path.join(
    POST_THESIS, "..", "moire_envelope", "thesis_results"))
sys.path.insert(0, THESIS_RESULTS)
sys.path.insert(0, POST_THESIS)
sys.path.insert(0, HERE)

from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
import candidate_hexM as cand  # noqa: E402
import hero_engine as he  # noqa: E402

from sksparse.cholmod import cholesky  # noqa: E402

LATTICE = "hex"
ETA_REF = 2 * np.sin(lat.twist_angle(LATTICE, 5, 4) / 2)   # the (5,4) family anchor


def eta(m, n):
    return 2 * np.sin(lat.twist_angle(LATTICE, m, n) / 2)


def scaled_layers(m, n):
    """Uniform asymptotic family: a2 proportional to eta^2 keeps V/E_kin fixed, so the
    envelope theory stays in its asymptotic regime as the angle shrinks (section 20)."""
    a2 = cand.LAYER2_AMP * (eta(m, n) / ETA_REF) ** 2
    return (mat.cosine_layer(cand.LAYER1_AMPS),
            mat.cosine_layer({h: a2 for h in cand.STAR})), a2


# ---------------------------------------------------------------- sparse PWE leg

def pencil(m, n, gcut_mono, layers=None):
    """Sparse TM plane-wave pencil (K, S) at the folded M2 sector.

    Isotropic Cartesian cutoff |G| <= gcut_mono * |b_mono| (a disk, not the box the
    collocation engine uses — same continuum limit, ~30% fewer plane waves)."""
    A = lat.supercell_A(LATTICE, m, n)
    Bs = lf.supercell_basis(LATTICE, m, n)
    l1, l2 = layers if layers is not None else cand.layers()
    c_sc = lf.moire_coeffs(LATTICE, m, n, cand.EPS0, l1, l2)
    ks = lat.fold_sector(A, cand.CARRIER_FRAC)
    k_sc = lat.sector_to_cartesian(Bs, ks)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    B0 = lat.monolayer_basis(LATTICE)
    b0 = np.linalg.norm(2 * np.pi * np.linalg.inv(B0).T[:, 0])
    Gcut = gcut_mono * b0

    lim = int(np.ceil(np.linalg.norm(np.linalg.inv(Brec), 2)
                      * (Gcut + np.linalg.norm(k_sc)))) + 2
    ax = np.arange(-lim, lim + 1)
    N1, N2 = np.meshgrid(ax, ax, indexing="ij")
    n1, n2 = N1.reshape(-1), N2.reshape(-1)
    Gx = Brec[0, 0] * n1 + Brec[0, 1] * n2
    Gy = Brec[1, 0] * n1 + Brec[1, 1] * n2
    kin = (Gx + k_sc[0]) ** 2 + (Gy + k_sc[1]) ** 2
    keep = kin <= Gcut ** 2
    n1, n2, kin = n1[keep], n2[keep], kin[keep]
    npw = len(kin)

    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(n1, n2))}
    rows, cols, vals = [], [], []
    for (d1, d2), c in c_sc.items():
        assert abs(np.imag(c)) < 1e-12, c
        tgt1, tgt2 = n1 + d1, n2 + d2
        for i in range(npw):
            j = idx.get((int(tgt1[i]), int(tgt2[i])))
            if j is not None:
                rows.append(j)
                cols.append(i)
                vals.append(float(np.real(c)))
    S = sp.csc_matrix((vals, (rows, cols)), shape=(npw, npw))
    asym = abs(S - S.T).max()
    assert asym < 1e-12, asym
    K = sp.diags(kin, format="csc")
    return dict(K=K, S=S, npw=npw, kappa_s=ks, ncells=lat.n_cells(A), Bs=Bs)


def inertia(K, S, lam):
    """Number of pencil eigenvalues strictly below lam (Sylvester, LDL^T of K - lam S).
    S is positive definite, so congruence preserves the count exactly."""
    f = cholesky((K - lam * S).tocsc(), beta=0, mode="simplicial")
    return int(np.sum(f.D() < 0.0))


def pwe_window(P, lo, hi, k0=64, verbose=True):
    """All pencil eigenvalues in [lo, hi], count-certified by inertia."""
    K, S = P["K"], P["S"]
    n_lo, n_hi = inertia(K, S, lo), inertia(K, S, hi)
    want = n_hi - n_lo
    sigma = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    fac = cholesky((K - sigma * S).tocsc(), beta=0, mode="simplicial")
    OPinv = spla.LinearOperator(K.shape, matvec=fac, dtype=float)
    k = max(k0, want + 12)
    for _ in range(5):
        w = np.sort(spla.eigsh(K, k=min(k, P["npw"] - 2), M=S, sigma=sigma,
                               which="LM", OPinv=OPinv, tol=1e-11,
                               return_eigenvectors=False))
        got = w[(w >= lo) & (w <= hi)]
        if float(np.max(np.abs(w - sigma))) >= rad or k >= P["npw"] - 2:
            break
        k = int(1.6 * k) + 8
        if verbose:
            print(f"    grow k -> {k}", flush=True)
    if verbose:
        print(f"    inertia census [{lo:.4f},{hi:.4f}] = {want}, "
              f"eigsh returned {len(got)}", flush=True)
    return got, want


# ---------------------------------------------------------------- EA leg

def ea_window(m, n, lo, hi, gmax_mono=4, Ns=17, fine=192, layers=None):
    """Registry-adapted single-band raw projection (hero engine, unchanged)."""
    A = lat.supercell_A(LATTICE, m, n)
    A2i = lf.layer2_integer_matrix(LATTICE, m, n)
    W = np.asarray(A2i, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(LATTICE, m, n)
    l1, l2 = layers if layers is not None else cand.layers()

    def coeffs_fn(d):
        return mat.bilayer(cand.EPS0, l1, l2, delta=d)

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(Ns * Ns)]
    frames = he.adapted_frames(coeffs_fn, cand.CARRIER_FRAC, gmax_mono, deltas,
                               [cand.BAND], fine)
    H_P = he.lazy_project(coeffs_fn, cand.CARRIER_FRAC, gmax_mono, Ns, reg,
                          np.linalg.inv(Bs).T, frames, fine)
    w = np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))
    return w[(w >= lo) & (w <= hi)]


# ---------------------------------------------------------------- FDFD leg

def fdfd_window(m, n, lo, hi, res_list=(16, 24), layers=None, verbose=True):
    """Finite differences on the identical analytic dielectric; per-state h^2
    Richardson extrapolation across the resolutions."""
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    import fdfd_leg as fl
    l1, l2 = layers if layers is not None else cand.layers()
    A = lat.supercell_A(LATTICE, m, n)
    ncell = lat.n_cells(A)
    Bs = lf.supercell_basis(LATTICE, m, n)
    ks = lat.fold_sector(A, cand.CARRIER_FRAC)
    q_vec = lat.sector_to_cartesian(Bs, ks)
    sigma = 0.5 * (lo + hi)
    rad = 0.5 * (hi - lo)
    ladders, grids = [], []
    for res in res_list:
        Nx = Ny = round(res * np.sqrt(ncell))
        t0 = time.time()
        eps, _ = fl.eps_on_grid(m, n, Nx, Ny, Bs, l1, l2)
        L = build_fdfd_operator(eps, {"B_super": Bs}, q_vec, polarization="tm")
        fac = cholesky((L - sigma * sp.eye(L.shape[0], format="csc")).tocsc(),
                       beta=0, mode="simplicial")
        OPinv = spla.LinearOperator(L.shape, matvec=fac, dtype=L.dtype)
        k = 80
        for _ in range(5):
            w = np.sort(spla.eigsh(L, k=k, sigma=sigma, which="LM", OPinv=OPinv,
                                   tol=1e-10, return_eigenvectors=False))
            if float(np.max(np.abs(w - sigma))) >= rad:
                break
            k = int(1.6 * k) + 8
        lad = w[(w >= lo) & (w <= hi)]
        ladders.append(lad)
        grids.append(Nx)
        if verbose:
            print(f"    fdfd res{res}: {Nx}x{Ny} ({Nx*Ny:,} DOF) k={k} "
                  f"n_win={len(lad)}  {time.time()-t0:.0f}s", flush=True)
    L = min(len(x) for x in ladders)
    F = np.stack([x[:L] for x in ladders])
    h = 1.0 / np.array(grids, float)
    ext = F[-1] + (F[-1] - F[-2]) / ((h[-2] / h[-1]) ** 2 - 1.0)
    return ext, F, np.array(grids)


# ---------------------------------------------------------------- drivers

def probe(m, n, gcuts=(5.0, 6.0)):
    """Validate the sparse pencil against the known collocation/FDFD rungs and
    measure the true state density of the sector tower."""
    for g in gcuts:
        t0 = time.time()
        P = pencil(m, n, g)
        lo = cand.WINDOW[0] - 0.02
        w, cen = pwe_window(P, lo, lo + 0.20)
        print(f"({m},{n}) gcut={g} npw={P['npw']:,} N_cells={P['ncells']} "
              f"census={cen} ({time.time()-t0:.0f}s)")
        print("   ", np.array2string(w[:14], precision=7, separator=","))
        if len(w):
            print(f"    floor {w[0]:.7f}; states within +0.05/+0.10/+0.148: "
                  f"{int(np.sum(w <= w[0]+0.05))}/{int(np.sum(w <= w[0]+0.10))}/"
                  f"{int(np.sum(w <= w[0]+0.148))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run"])
    ap.add_argument("--m", type=int, default=9)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gcut", type=float, default=6.0)
    ap.add_argument("--span", type=float, default=0.148)
    ap.add_argument("--ns", type=int, default=17)
    ap.add_argument("--res", type=str, default="16,24")
    ap.add_argument("--no-fdfd", action="store_true")
    ap.add_argument("--scaled", action="store_true",
                    help="uniform asymptotic family, a2 proportional to eta^2")
    ap.add_argument("--out", type=str, default="ladder_wide.npz")
    args = ap.parse_args()

    if args.cmd == "probe":
        probe(args.m, args.n)
        return

    m, n = args.m, args.n
    t0 = time.time()
    lyr, a2 = (scaled_layers(m, n) if args.scaled else (None, cand.LAYER2_AMP))
    print(f"=== wide ladder ({m},{n})  eta={eta(m, n):.5f}  a2={a2:.5f}"
          f"{'  [scaled family]' if args.scaled else ''} ===", flush=True)
    P = pencil(m, n, args.gcut, layers=lyr)
    print(f"  pencil: npw={P['npw']:,} N_cells={P['ncells']} "
          f"nnz(S)={P['S'].nnz:,}", flush=True)
    lo = cand.WINDOW[0] - 0.02
    w_pw, census = pwe_window(P, lo, lo + args.span + 0.02)
    floor = float(w_pw[0])
    hi = floor + args.span
    w_pw = w_pw[w_pw <= hi]
    print(f"  pwe: floor {floor:.7f}, {len(w_pw)} states in "
          f"[{floor:.4f},{hi:.4f}]  ({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    w_ea = ea_window(m, n, lo, hi, Ns=args.ns, layers=lyr)
    print(f"  ea : {len(w_ea)} states  ({time.time()-t1:.0f}s)", flush=True)

    store = dict(pwe=w_pw, ea=w_ea, floor=floor, span=args.span,
                 census=census, mn=np.array([m, n]), ncells=P["ncells"],
                 npw=P["npw"], gcut=args.gcut, ns=args.ns, a2=a2,
                 scaled=bool(args.scaled))
    if not args.no_fdfd:
        t2 = time.time()
        res_list = tuple(int(x) for x in args.res.split(","))
        ext, F, grids = fdfd_window(m, n, lo, hi, res_list=res_list, layers=lyr)
        print(f"  fdfd: {len(ext)} states  ({time.time()-t2:.0f}s)", flush=True)
        store.update(fdfd=ext, fdfd_raw=F, fdfd_grids=grids)
    np.savez(os.path.join(HERE, args.out), **store)
    print(f"saved {args.out}  (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
