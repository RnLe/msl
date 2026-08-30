#!/usr/bin/env python3
"""The model-accuracy hierarchy on the in-domain tower, one angle at a time.

Four models against the sparse-pencil reference, all sharing the identical analytic
dielectric:

  ea1    single-band registry-adapted raw projection (the production model)
  ea3    the same with bands 0-2 retained (band truncation relaxed, frame fixed)
  ritz1  lifted exact-Bloch Ritz, band 1: trial columns are the EXACT Bloch functions
         of the registry-averaged monolayer at each in-domain folded momentum, mapped
         into the supercell plane-wave basis by the exact integer index maps. This is
         the infinite-order k.p resummation at fixed band content — the frame error
         is zero by construction, what remains is pure band truncation.
  ritz3  the same with bands 0-2 per momentum (band truncation relaxed as well).

Differences isolate the two approximations the envelope theory stacks:
  |ea1 - ritz1|   = envelope/frame + slow-grid error at fixed band content,
  |ritz1 - ref|   = pure band-truncation error,
  |ritz3 - ref|   = its cure by more bands.

The Ritz trial space uses the a-priori domain plus an energy buffer (boundary
truncation of the envelope tails); only states below the ceiling are claimed.
"""
import argparse
import time

import numpy as np
import scipy.linalg as sla

import ladder_wide as lw
import valley_diagnosis as vd
from ladder_wide import HERE, LATTICE, cand, lat, lf  # noqa: F401
from lib_v5 import micro_pwe as mp
from lib_v5 import oracles as oc

GMAX_MONO = vd.GMAX_MONO


def lifted_ritz(P, m, n, band_ids, buffer_e=0.008):
    """Galerkin on exact registry-averaged Bloch functions at the in-domain (plus
    buffer) folded momenta. Returns (eigenvalues, n_claim, harmonics)."""
    Bs = lf.supercell_basis(LATTICE, m, n)
    Brec = P["Brec"]
    ceil_e = vd.ceiling()
    dom, dom_e, grid = vd.domain_harmonics(m, n)
    # buffer: M2-basin harmonics with energy in (ceiling, ceiling + buffer_e]
    sel = np.where((grid["basin"] == 1) & np.isfinite(grid["e"])
                   & (grid["e"] > ceil_e) & (grid["e"] <= ceil_e + buffer_e))[0]
    hs = np.vstack([dom, np.stack([grid["n1"][sel], grid["n2"][sel]], 1)])
    n_claim = len(dom)

    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(P["n1"], P["n2"]))}
    n0 = np.linalg.solve(Brec, vd.M_CART["M2"] - P["k_sc"])
    n0i = np.rint(n0).astype(int)
    assert np.max(np.abs(n0 - n0i)) < 1e-9, n0
    At = np.asarray(P["A"], int).T
    mono_ns = mp.pw_set(GMAX_MONO)

    cols = []
    drop_max = 0.0
    for (a, b) in hs:
        k_n = vd.M_CART["M2"] + Brec @ np.array([a, b], float)
        w, V, _, _ = mp.solve(vd.avg_coeffs(), k_n, B0=vd.B0, gmax=GMAX_MONO,
                              n_bands=max(band_ids) + 1)
        for bid in band_ids:
            col = np.zeros(P["npw"], complex)
            dropped = 0.0
            for (h1, h2), c in zip(mono_ns, V[:, bid]):
                tgt = (int(n0i[0] + a + At[0, 0] * h1 + At[0, 1] * h2),
                       int(n0i[1] + b + At[1, 0] * h1 + At[1, 1] * h2))
                j = idx.get(tgt)
                if j is None:
                    dropped += abs(c) ** 2
                else:
                    col[j] = c
            drop_max = max(drop_max, dropped)
            cols.append(col)
    assert drop_max < 1e-6, f"trial truncation {drop_max:.1e}"
    T = np.stack(cols, 1)
    w, _, _ = oc.ritz_pencil(P["K"], P["S"], T)
    return np.sort(w.real), n_claim, hs


def ea_model(m, n, lo, hi, Ns, band_ids, layers, fine=192):
    """Registry-adapted raw projection with the given retained bands."""
    A = lat.supercell_A(LATTICE, m, n)
    A2i = lf.layer2_integer_matrix(LATTICE, m, n)
    W = np.asarray(A2i, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(LATTICE, m, n)
    l1, l2 = layers

    def coeffs_fn(d):
        import lib_v5.materials as mat
        return mat.bilayer(cand.EPS0, l1, l2, delta=d)

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(Ns * Ns)]
    frames = vd.he.adapted_frames(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO,
                                  deltas, band_ids, fine)
    H_P = vd.he.lazy_project(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, Ns, reg,
                             np.linalg.inv(Bs).T, frames, fine)
    w = np.sort(sla.eigvalsh(0.5 * (H_P + H_P.conj().T)))
    return w[(w >= lo) & (w <= hi)]


def nearest_dev(ref, model):
    """|model - ref| per reference state, nearest available model level (greedy by
    closeness, one model level per reference state)."""
    from fig_ladder_wide import match
    pairs, _ = match(np.asarray(ref), np.asarray(model),
                     tol_frac=2.0, floor_frac=2.0)
    return np.array([np.nan if p is None else abs(model[p] - r)
                     for p, r in zip(pairs, ref)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=18)
    ap.add_argument("--n", type=int, default=17)
    ap.add_argument("--ns", type=int, default=21)
    ap.add_argument("--buffer", type=float, default=0.012)
    args = ap.parse_args()
    m, n = args.m, args.n
    lyr, a2 = lw.scaled_layers(m, n)
    tag = f"{m}_{n}s"
    t0 = time.time()

    d = np.load(f"{HERE}/diag_{tag}.npz")
    w_r = d["w_r"]
    r_in = d["r_in"]
    ref = w_r[r_in]
    re = d["r_e"][r_in]
    rk = d["r_dom_k"][r_in]
    kdist = vd.basin_of(rk[:, 0], rk[:, 1])[1]
    floor = vd.band1_avg(*vd.M_CART["M2"])
    print(f"=== hierarchy ({m},{n}) scaled  {len(ref)} in-domain reference "
          f"states ===", flush=True)

    P = lw.pencil(m, n, 4.0, layers=lyr)
    lo, hi = ref[0] - 0.006, ref[-1] + 0.02
    models = {}
    for name, bands, kind in (("ea1", [1], "ea"), ("ea3", [0, 1, 2], "ea"),
                              ("ritz1", [1], "ritz"),
                              ("ritz3", [0, 1, 2], "ritz")):
        t1 = time.time()
        if kind == "ea":
            w = ea_model(m, n, lo, hi, args.ns, bands, lyr)
        else:
            w, n_claim, _ = lifted_ritz(P, m, n, bands, buffer_e=args.buffer)
            w = w[(w >= lo) & (w <= hi)]
        models[name] = w
        print(f"  {name}: {len(w)} states in range  ({time.time()-t1:.0f}s)",
              flush=True)

    conv = 8 * np.pi ** 2 * np.sqrt(ref[0]) / (2 * np.pi)
    devs = {k: nearest_dev(ref, v) / conv for k, v in models.items()}
    print("\n   i   lam        E-floor   kdist    " +
          "   ".join(f"{k:>8s}" for k in devs))
    for i in range(len(ref)):
        row = "   ".join(f"{devs[k][i]:8.1e}" for k in devs)
        print(f"  {i:3d}  {ref[i]:.6f}  {re[i]-floor:+.4f}  "
              f"{kdist[i]:.3f}   {row}")
    for k, v in devs.items():
        ok = np.isfinite(v)
        print(f"  {k}: med {np.median(v[ok]):.1e}  max {np.nanmax(v):.1e}  "
              f"(f units)")
    np.savez(f"{HERE}/hier_{tag}.npz", ref=ref, re=re, kdist=kdist,
             floor=floor, conv=conv, a2=a2,
             **{f"w_{k}": v for k, v in models.items()},
             **{f"dev_{k}": v for k, v in devs.items()})
    print(f"saved hier_{tag}.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()


def fold_model(m, n, Ns, harmonics, band_ids_q, layers, zbar, fine=192,
               mode="lowdin", margin=0.03, window=None):
    """The remote-band downfold on the momentum-restricted product space: build the
    P (band 1) + Q (remote bands) trial in ONE lazy_project call (columns ordered
    (mode, band)), partition, and fold Q into P instead of diagonalizing P+Q —
    the thesis's eta^2 Lowdin dressing (fixed zbar) or the exact Feshbach
    (self-consistent z per state). The fold is invariant under any unitary mixing
    within Q, so remote-band relabeling across registries is harmless.

    A-priori pole guard: every Q column's raw surface E_r + (u_r^H R^2 u_r)|kappa|^2
    must stay `margin` away from the comparison window (monolayer-only check)."""
    bands = [cand.BAND] + list(band_ids_q)
    nb = len(bands)
    A = lat.supercell_A(LATTICE, m, n)
    A2i = lf.layer2_integer_matrix(LATTICE, m, n)
    W = np.asarray(A2i, float) - np.asarray(A, float)
    Bs = lf.supercell_basis(LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    l1, l2 = layers

    # pole guard from the isotropic raw surfaces of the remote bands
    if window is not None:
        import scipy.linalg as _sla
        from lib_v5 import raw_projection as rp
        h0, R, _, _ = rp.mono_hermitized(vd.avg_coeffs(), vd.M_CART["M2"],
                                         vd.B0, GMAX_MONO, 128)
        w0, V0 = _sla.eigh(h0)
        kaps2 = ((np.asarray(harmonics, float) @ Brec.T) ** 2).sum(1)
        for bq in band_ids_q:
            raw_b = float(np.real(V0[:, bq].conj() @ (R @ (R @ V0[:, bq]))))
            surf = w0[bq] + raw_b * kaps2
            dist = np.minimum(np.abs(surf - window[0]),
                              np.abs(surf - window[1]))
            inside = (surf > window[0]) & (surf < window[1])
            assert not inside.any() and dist.min() > margin, \
                (bq, float(dist.min()))

    def coeffs_fn(d):
        import lib_v5.materials as mat
        return mat.bilayer(cand.EPS0, l1, l2, delta=d)

    def reg(a, b):
        v = W @ np.array([a, b])
        return (float(v[0]), float(v[1]))

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [reg(S1.reshape(-1)[j], S2.reshape(-1)[j]) for j in range(Ns * Ns)]
    frames = vd.he.adapted_frames(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO,
                                  deltas, bands, fine)
    H = vd.he.lazy_project(coeffs_fn, cand.CARRIER_FRAC, GMAX_MONO, Ns, reg,
                           np.linalg.inv(Bs).T, frames, fine,
                           slow_modes=harmonics)
    H = 0.5 * (H + H.conj().T)
    nm = len(harmonics)
    iP = np.arange(nm) * nb
    iQ = np.setdiff1d(np.arange(nm * nb), iP)
    Hpp = H[np.ix_(iP, iP)]
    Hpq = H[np.ix_(iP, iQ)]
    Hqq = H[np.ix_(iQ, iQ)]

    def eff(z):
        M = Hpp + Hpq @ np.linalg.solve(
            z * np.eye(len(iQ)) - Hqq, Hpq.conj().T)
        return 0.5 * (M + M.conj().T)

    if mode == "lowdin":
        return np.sort(sla.eigvalsh(eff(zbar)))
    # feshbach: per-state fixed point z -> eigenvalue_i(eff(z))
    w = np.sort(sla.eigvalsh(eff(zbar)))
    out = []
    for i in range(len(w)):
        z = w[i]
        for _ in range(6):
            z_new = np.sort(sla.eigvalsh(eff(z)))[i]
            if abs(z_new - z) < 1e-13:
                z = z_new
                break
            z = z_new
        out.append(z)
    return np.array(out)
