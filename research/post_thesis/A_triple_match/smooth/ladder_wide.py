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
    return dict(K=K, S=S, npw=npw, kappa_s=ks, ncells=lat.n_cells(A), Bs=Bs,
                n1=n1, n2=n2, k_sc=k_sc, Brec=Brec, A=A)


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


def ea_domain(m, n, Ns, harmonics, layers=None, gmax_mono=4, fine=192):
    """Domain-restricted registry-adapted EA: the trial space is built directly on
    the given envelope harmonics (momentum basis) instead of the full slow grid —
    the a-priori restriction, not a post-hoc filter. Returns all eigenvalues."""
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
                          np.linalg.inv(Bs).T, frames, fine,
                          slow_modes=harmonics)
    return np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))


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


def fdfd_big(m, n, lo, hi, res_list=(16, 24), layers=None, verbose=True):
    """FDFD-only certified reference for the large-angle run: per resolution the
    window census is certified by Sylvester inertia on the FDFD matrix itself
    (CHOLMOD LDL of L - lambda I), the extraction must reproduce it exactly, and
    eigenvectors are kept at the coarsest resolution for valley classification."""
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    import fdfd_leg as fl
    l1, l2 = layers if layers is not None else cand.layers()
    A = lat.supercell_A(LATTICE, m, n)
    ncell = lat.n_cells(A)
    Bs = lf.supercell_basis(LATTICE, m, n)
    ks = lat.fold_sector(A, cand.CARRIER_FRAC)
    q_vec = lat.sector_to_cartesian(Bs, ks)
    sigma = 0.5 * (lo + hi)
    out = dict(grids=[], ladders=[], census=[])
    for ir, res in enumerate(res_list):
        Nx = Ny = round(res * np.sqrt(ncell))
        t0 = time.time()
        eps, _ = fl.eps_on_grid(m, n, Nx, Ny, Bs, l1, l2)
        L = build_fdfd_operator(eps, {"B_super": Bs}, q_vec,
                                polarization="tm").tocsc()
        n_dof = L.shape[0]
        eye = sp.eye(n_dof, format="csc")

        def count_below(lam):
            f = cholesky((L - lam * eye), beta=0, mode="simplicial")
            return int(np.sum(f.D() < 0.0))

        cen = count_below(hi) - count_below(lo)
        fac = cholesky((L - sigma * eye), beta=0, mode="simplicial")
        OPinv = spla.LinearOperator(L.shape, matvec=fac, dtype=L.dtype)
        k = cen + 12
        w = V = None
        for _ in range(5):
            r = spla.eigsh(L, k=k, sigma=sigma, which="LM", OPinv=OPinv,
                           tol=1e-10, return_eigenvectors=(ir == 0))
            w, V = r if ir == 0 else (r, None)
            if float(np.max(np.abs(w - sigma))) >= 0.5 * (hi - lo):
                break
            k = int(1.6 * k) + 8
        keep = (w >= lo) & (w <= hi)
        order = np.argsort(w[keep])
        lad = w[keep][order]
        if len(lad) != cen:
            out.setdefault("warnings", []).append(
                f"res{res}: extracted {len(lad)} vs census {cen}")
        out["grids"].append(Nx)
        out["ladders"].append(lad)
        out["census"].append(cen)
        if ir == 0:
            out["V0"] = V[:, keep][:, order]
            out["shape0"] = (Nx, Ny)
        if verbose:
            print(f"    fdfd res{res}: {Nx}x{Ny} ({n_dof:,} DOF) census={cen} "
                  f"extracted={len(lad)}  {time.time()-t0:.0f}s", flush=True)
    L0 = min(len(x) for x in out["ladders"])
    F = np.stack([x[:L0] for x in out["ladders"]])
    h = 1.0 / np.array(out["grids"], float)
    ext = F[-1] + (F[-1] - F[-2]) / ((h[-2] / h[-1]) ** 2 - 1.0)
    unc = np.abs(ext - F[-1]) * (h[-1] / h[-2]) ** 2   # next-order scale
    out["extrap"] = ext
    out["unc"] = unc
    return out


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


def bigrun(m, n, Ns, res_list, buffer_e, out_name):
    """The domain-restricted ladder at a large angle: FDFD-only certified reference
    against the momentum-restricted EA, claims bounded by the a-priori ceiling."""
    import valley_diagnosis as vd
    t0 = time.time()
    lyr, a2 = scaled_layers(m, n)
    dom, dom_e, grid = vd.domain_harmonics(m, n)
    ce = grid["ceiling"]
    off = ce - dom_e[0]                       # a-priori ceiling above the floor
    sel = np.where((grid["basin"] == 1) & np.isfinite(grid["e"])
                   & (grid["e"] > ce) & (grid["e"] <= ce + buffer_e))[0]
    buf = np.stack([grid["n1"][sel], grid["n2"][sel]], 1) if len(sel) \
        else np.zeros((0, 2), int)
    hs = np.vstack([dom, buf])
    N = lat.n_cells(lat.supercell_A(LATTICE, m, n))
    print(f"=== bigrun ({m},{n})  N={N}  eta={eta(m, n):.5f}  a2={a2:.5f}\n"
          f"    domain {len(dom)} harmonics + buffer {len(buf)}  "
          f"ceiling offset +{off:.4f}", flush=True)

    # a-priori dispersion error of the fixed-frame model per domain harmonic
    Bs0 = lf.supercell_basis(LATTICE, m, n)
    Brec0 = 2 * np.pi * np.linalg.inv(np.asarray(Bs0, float)).T
    de = np.abs(vd.ea_symbol(dom @ Brec0.T) - dom_e)

    t1 = time.time()
    w_ea = ea_domain(m, n, Ns, hs, layers=lyr)
    ea_claim = w_ea[w_ea <= w_ea[0] + off]
    print(f"  ea (fixed frame): {len(w_ea)} trial states, {len(ea_claim)} "
          f"below the ceiling  ({time.time()-t1:.0f}s)", flush=True)

    t1 = time.time()
    import hierarchy_ladder as hl
    P = pencil(m, n, 4.0, layers=lyr)
    w_rz, _, _ = hl.lifted_ritz(P, m, n, [cand.BAND], buffer_e=buffer_e)
    rz_claim = w_rz[(w_rz >= w_ea[0] - 0.01) & (w_rz <= w_rz[0] + off)]
    print(f"  ritz (exact frames): {len(rz_claim)} claimed states "
          f"[pencil npw {P['npw']:,}]  ({time.time()-t1:.0f}s)", flush=True)
    del P

    lo = float(w_ea[0]) - 0.006
    hi = float(w_ea[0]) + off + 0.004
    fd = fdfd_big(m, n, lo, hi, res_list=res_list, layers=lyr)
    ext, unc = fd["extrap"], fd["unc"]
    fd_claim = ext[ext <= ext[0] + off]
    print(f"  fdfd: census {fd['census']}, {len(fd_claim)} claimed below the "
          f"ceiling", flush=True)

    # valley classification of the coarse-grid eigenvectors (cross-check only)
    Nx, Ny = fd["shape0"]
    Bs = lf.supercell_basis(LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    ks = lat.fold_sector(lat.supercell_A(LATTICE, m, n), cand.CARRIER_FRAC)
    q = lat.sector_to_cartesian(Bs, ks)
    f1 = np.rint(np.fft.fftfreq(Nx) * Nx)
    f2 = np.rint(np.fft.fftfreq(Ny) * Ny)
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    kx = q[0] + Brec[0, 0] * F1 + Brec[0, 1] * F2
    ky = q[1] + Brec[1, 0] * F1 + Brec[1, 1] * F2
    bas, _ = vd.basin_of(kx.reshape(-1), ky.reshape(-1))
    bas = bas.reshape(Nx, Ny)
    nst = fd["V0"].shape[1]
    wb = np.zeros((3, nst))
    for j in range(nst):
        W2 = np.abs(np.fft.fft2(fd["V0"][:, j].reshape(Nx, Ny))) ** 2
        W2 /= W2.sum()
        for i in range(3):
            wb[i, j] = W2[bas == i].sum()
    n_m2 = int(np.sum(wb[1, :len(fd_claim)] > 0.5))
    print(f"  classification: {n_m2}/{len(fd_claim)} claimed reference states "
          f"are M2-dominant", flush=True)

    conv = 8 * np.pi ** 2 * np.sqrt(ext[0]) / (2 * np.pi)
    kr = min(len(fd_claim), len(rz_claim))
    dvr = np.abs(rz_claim[:kr] - fd_claim[:kr]) / conv
    print(f"  counts: fdfd {len(fd_claim)} / ritz {len(rz_claim)} / ea "
          f"{len(ea_claim)}", flush=True)
    print(f"  ritz vs fdfd sorted-1:1 over {kr}: min {dvr.min():.1e} "
          f"med {np.median(dvr):.1e} max {dvr.max():.1e}", flush=True)
    ke = min(len(fd_claim), len(ea_claim))
    dve = np.abs(ea_claim[:ke] - fd_claim[:ke]) / conv
    ok = np.sort(de)[:ke] / conv <= 1e-5
    if ok.any():
        print(f"  ea vs fdfd on its model-domain rungs ({int(ok.sum())}): "
              f"max {dve[ok].max():.1e}", flush=True)
    np.savez(os.path.join(HERE, out_name),
             ea=w_ea, ea_claim=ea_claim, fdfd_extrap=ext, fdfd_unc=unc,
             fdfd_claim=fd_claim,
             fdfd_raw=np.stack([x[:min(len(y) for y in fd["ladders"])]
                                for x in fd["ladders"]]),
             grids=np.array(fd["grids"]), census=np.array(fd["census"]),
             wb=wb, dom=dom, dom_e=dom_e, de=de, ritz=rz_claim,
             buffer=buf, off=off, a2=a2, mn=np.array([m, n]), ns=Ns,
             warnings=np.array(fd.get("warnings", []), dtype=object))
    print(f"saved {out_name}  ({time.time()-t0:.0f}s total)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run", "bigrun"])
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
    ap.add_argument("--buffer", type=float, default=0.006)
    args = ap.parse_args()

    if args.cmd == "probe":
        probe(args.m, args.n)
        return
    if args.cmd == "bigrun":
        res_list = tuple(int(x) for x in args.res.split(","))
        bigrun(args.m, args.n, args.ns, res_list, args.buffer, args.out)
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
