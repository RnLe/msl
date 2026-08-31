#!/usr/bin/env python3
"""The thesis crystal, ported to the completed envelope machinery.

Material: the thesis's asymmetric square bilayer (layer-1 rods r=0.20, layer-2 rods
r=0.10, both eps 8.9 on background 1, TM, X carrier, band 1) represented by its
truncated Fourier series at |h|_inf <= H — every solver (monolayer PWE, supercell
PWE, FDFD, MPB, EA) consumes the IDENTICAL analytic coefficients, so the
comparison measures models, not rasterization. Rod coefficients are the exact disk
transform c_h = deps * f * 2 J1(|G| r)/(|G| r), f = pi r^2.

Carriers: X = (1/2, 0) and X' = (0, 1/2), band-1 minima, C4 partners. For odd m in
the (m, 1) square commensuration both fold into one supercell sector. The
registry-averaged material (layer-2 contributes only its mean) defines the exact
per-momentum frames; EA v2 hops run over ALL layer-2 harmonics (n -> n - W^T h).

Commands:
  material  — coefficient checks, min-eps, averaged band-1 landscape, the
              registry window of E1 at X, convergence of the monolayer knobs
  ea        — the EA v2 ladder at one angle (--m), window census and claims
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
from scipy.special import j1

sys.path.insert(0, "/home/renlephy/msl/research/post_thesis")
from lib_v5 import lattice as lat
from lib_v5 import lifted as lf
from lib_v5 import materials as mat
from lib_v5 import micro_pwe as mp

LATTICE = "square"
B0 = lat.monolayer_basis(LATTICE)
BREC0 = 2 * np.pi * np.linalg.inv(B0).T
EPS0 = 1.0
EPS_ROD = 8.9
R1, R2 = 0.20, 0.10
H = 10                       # Fourier truncation of the rod layers
GMAX = 10                    # monolayer plane-wave cutoff
FINE_REG = 8                 # registry grid for the window measurement
X_FRAC = {"X": (0.5, 0.0), "Xp": (0.0, 0.5)}
X_CART = {k: BREC0 @ np.array(v) for k, v in X_FRAC.items()}
BAND = 1

_avg = None
_ecache = {}


def rod_layer(r, deps=EPS_ROD - EPS0, h_max=H):
    """Fourier coefficients of a disk of radius r (unit square cell), through a
    Lanczos window: c_h = deps * f * 2 J1(|G| r)/(|G| r) * sinc(h1/(H+1)) *
    sinc(h2/(H+1)). The apodization removes the Gibbs undershoot of the bare
    truncation (which dips below zero on the supercell), so the material stays a
    positive analytic dielectric that every solver consumes identically."""
    f = np.pi * r * r
    out = {}
    for a in range(-h_max, h_max + 1):
        for b in range(-h_max, h_max + 1):
            win = np.sinc(a / (h_max + 1)) * np.sinc(b / (h_max + 1))
            if a == 0 and b == 0:
                out[(0, 0)] = deps * f
                continue
            x = 2 * np.pi * np.hypot(a, b) * r
            out[(a, b)] = deps * f * 2 * j1(x) / x * win
    return out


def layers():
    return rod_layer(R1), rod_layer(R2)


def coeffs(delta=(0.0, 0.0)):
    l1, l2 = layers()
    return mat.bilayer(EPS0, l1, l2, delta=delta)


def avg_coeffs():
    """Registry-averaged material: layer-2 keeps only its mean."""
    global _avg
    if _avg is None:
        l1, l2 = layers()
        _avg = mat.bilayer(EPS0, l1, {(0, 0): l2[(0, 0)]}, delta=(0.0, 0.0))
    return _avg


def band_avg(kx, ky, band=BAND, gmax=GMAX):
    f = np.linalg.inv(BREC0) @ np.array([kx, ky])
    key = (round(float(f[0]) % 1.0, 9), round(float(f[1]) % 1.0, 9), band, gmax)
    if key not in _ecache:
        k = BREC0 @ np.array([key[0], key[1]])
        A, Bm, _, _ = mp.tm_pencil(avg_coeffs(), k, B0, gmax)
        w = sla.eigh(A, Bm, eigvals_only=True, subset_by_index=[0, band])
        _ecache[key] = float(w[band])
    return _ecache[key]


def basin_of(kx, ky):
    """Nearest carrier (X or X') mod the monolayer lattice; vectorized."""
    kx = np.atleast_1d(np.asarray(kx, float))
    ky = np.atleast_1d(np.asarray(ky, float))
    inv = np.linalg.inv(BREC0)
    d = np.full((2, len(kx)), np.inf)
    for i, nm in enumerate(("X", "Xp")):
        fx = inv[0, 0] * (kx - X_CART[nm][0]) + inv[0, 1] * (ky - X_CART[nm][1])
        fy = inv[1, 0] * (kx - X_CART[nm][0]) + inv[1, 1] * (ky - X_CART[nm][1])
        fx -= np.rint(fx)
        fy -= np.rint(fy)
        d[i] = np.hypot(BREC0[0, 0] * fx + BREC0[0, 1] * fy,
                        BREC0[1, 0] * fx + BREC0[1, 1] * fy)
    return np.argmin(d, axis=0), np.min(d, axis=0)


def registry_window(nreg=FINE_REG, gmax=GMAX):
    """The manifold window: E1 at the X carrier across the registry torus, on the
    FULL bilayer (exact coefficients per registry)."""
    l1, l2 = layers()
    vals = []
    for i in range(nreg):
        for jj in range(nreg):
            c = mat.bilayer(EPS0, l1, l2, delta=(i / nreg, jj / nreg))
            w, _, _, _ = mp.solve(c, X_CART["X"], B0, gmax, n_bands=BAND + 1)
            vals.append(w[BAND])
    return float(np.min(vals)), float(np.max(vals))


def material_report():
    l1, l2 = layers()
    # coefficient check against a direct numerical transform of the disk
    ng = 512
    x = (np.arange(ng) + 0.5) / ng - 0.5
    Xg, Yg = np.meshgrid(x, x, indexing="ij")
    disk = ((Xg ** 2 + Yg ** 2) <= R1 ** 2).astype(float) * (EPS_ROD - EPS0)
    F = np.fft.fft2(disk) / ng ** 2
    errs = [abs(F[a % ng, b % ng] - l1[(a, b)]) for a in range(-4, 5)
            for b in range(-4, 5)]
    print(f"rod coefficients vs numeric disk transform: max "
          f"|diff| = {max(errs):.1e} (pixelization-limited)")
    # positivity of the truncated material at the worst registry
    e = mat.sample(coeffs((0.5, 0.5)), 512, 512)
    print(f"truncated-series eps range: [{e.min():.3f}, {e.max():.3f}]  "
          f"(H = {H})")
    assert e.min() > 0.05, e.min()
    # the averaged band-1 landscape: X vs the competition
    pts = {"X": X_CART["X"], "Xp": X_CART["Xp"],
           "Gamma": np.zeros(2), "M": BREC0 @ np.array([0.5, 0.5])}
    es = {k: band_avg(*v) for k, v in pts.items()}
    print("averaged band-1: " + "  ".join(f"{k} {v:.5f}" for k, v in es.items()))
    b0_floor = band_avg(*X_CART["X"], band=0)
    for g in (8, 10, 12):
        print(f"  E1(X) at gmax {g}: {band_avg(*X_CART['X'], gmax=g):.7f}")
    lo, hi = registry_window()
    print(f"registry window of E1(X): [{lo:.5f}, {hi:.5f}]  V = {hi - lo:.5f}")
    print(f"f-window: [{np.sqrt(lo) / (2 * np.pi):.5f}, "
          f"{np.sqrt(hi) / (2 * np.pi):.5f}]")
    # the a-priori ceiling: lowest band-1 structure above the X floor
    # (X and X' are degenerate; the saddle toward M/Gamma bounds the basins)
    ts = np.linspace(0, 1, 41)
    path = np.array([X_CART["X"] * (1 - t) + pts["M"] * t for t in ts])
    saddle = max(band_avg(*k) for k in path)
    print(f"X -> M band-1 saddle: {saddle:.5f} "
          f"(+{saddle - es['X']:.5f} above the X floor)")
    return es, (lo, hi), saddle


def capped_set(m, e_cap, buffer_e, gmax=GMAX):
    """Energy-capped, coset-deduplicated harmonic set around the X/X' carriers
    for the (m, 1) commensuration."""
    A = lat.supercell_A(LATTICE, m, 1)
    Bs = lf.supercell_basis(LATTICE, m, 1)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    lim = int(np.ceil(0.75 * np.linalg.norm(BREC0[:, 0])
                      / np.linalg.norm(Brec[:, 0]))) + 2
    ax = np.arange(-lim, lim + 1)
    N1, N2 = np.meshgrid(ax, ax, indexing="ij")
    n1, n2 = N1.reshape(-1), N2.reshape(-1)
    kx = X_CART["X"][0] + Brec[0, 0] * n1 + Brec[0, 1] * n2
    ky = X_CART["X"][1] + Brec[1, 0] * n1 + Brec[1, 1] * n2
    # coarse prefilter by basin distance, then energy on the survivors
    _, dist = basin_of(kx, ky)
    rough = dist <= 0.45 * np.linalg.norm(BREC0[:, 0])
    e = np.full(len(n1), np.inf)
    for i in np.where(rough)[0]:
        e[i] = band_avg(kx[i], ky[i], gmax=gmax)
    At = np.asarray(A, float).T
    fr = np.linalg.solve(At, np.stack([n1, n2]).astype(float)) % 1.0
    key = (np.round(fr[0], 6) * 1e6).astype(np.int64) * 10_000_019 \
        + (np.round(fr[1], 6) * 1e6).astype(np.int64)
    rad = n1 ** 2 + n2 ** 2
    best = {}
    for i in np.argsort(rad):
        if key[i] not in best:
            best[key[i]] = i
    rep = np.zeros(len(n1), bool)
    rep[list(best.values())] = True
    trial = rep & (e <= e_cap + buffer_e)
    claim = rep & (e <= e_cap)
    order = np.argsort(e[trial])
    return (np.stack([n1[trial], n2[trial]], 1)[order], int(claim.sum()))


def ea_solve(m, harmonics, gmax=GMAX):
    """EA v2 for the thesis bilayer at (m, 1): exact averaged-monolayer frames per
    harmonic, hops over ALL layer-2 harmonics (coset-aware)."""
    A = lat.supercell_A(LATTICE, m, 1)
    Bs = lf.supercell_basis(LATTICE, m, 1)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    A2i = lf.layer2_integer_matrix(LATTICE, m, 1)
    Wm = np.asarray(A2i, int) - np.asarray(A, int)
    _, l2 = layers()
    star2 = {h: c for h, c in mat._sym(l2).items() if h != (0, 0)}

    hs = [(int(a), int(b)) for a, b in harmonics]
    idx = {h: i for i, h in enumerate(hs)}
    N = len(hs)
    ns = mp.pw_set(gmax)
    pw_idx = {h: i for i, h in enumerate(ns)}
    At = np.asarray(A, int).T
    Ati = np.linalg.inv(At.astype(float))

    def coset_key(a, b):
        f = Ati @ np.array([a, b], float)
        return (round(float(f[0]) % 1.0, 6), round(float(f[1]) % 1.0, 6))

    ckey = {coset_key(a, b): (a, b) for (a, b) in hs}

    def resolve(a, b):
        j = idx.get((a, b))
        if j is not None:
            return j, (0, 0)
        rp = ckey.get(coset_key(a, b))
        if rp is None:
            return None, None
        g0 = Ati @ np.array([a - rp[0], b - rp[1]], float)
        g0i = np.rint(g0).astype(int)
        assert np.max(np.abs(g0 - g0i)) < 1e-9
        return idx[rp], (int(g0i[0]), int(g0i[1]))

    E = np.zeros(N)
    U = np.zeros((len(ns), N), complex)
    for i, (a, b) in enumerate(hs):
        k = X_CART["X"] + Brec @ np.array([a, b], float)
        w, V, _, _ = mp.solve(avg_coeffs(), k, B0, gmax, n_bands=BAND + 1)
        E[i] = w[BAND]
        U[:, i] = V[:, BAND]

    # vectorized hop assembly: for each layer-2 harmonic, the PW-shifted overlap
    ns_arr = np.array(ns)
    V2 = np.zeros((N, N), complex)
    for (h1, h2), c in star2.items():
        d = (int(Wm[0, 0] * h1 + Wm[1, 0] * h2),
             int(Wm[0, 1] * h1 + Wm[1, 1] * h2))
        for (a, b), i in idx.items():
            j, g0 = resolve(a - d[0], b - d[1])
            if j is None:
                continue
            sh1 = ns_arr[:, 0] - h1 + g0[0]
            sh2 = ns_arr[:, 1] - h2 + g0[1]
            ok = (np.abs(sh1) <= gmax) & (np.abs(sh2) <= gmax)
            src = (sh1[ok] + gmax) * (2 * gmax + 1) + (sh2[ok] + gmax)
            V2[i, j] += c * np.vdot(U[ok, i], U[src, j])
    V2 = 0.5 * (V2 + V2.conj().T)
    w = sla.eigh(np.diag(E), np.eye(N) + V2, eigvals_only=True)
    return np.sort(w), E


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["material", "ea"])
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--cap-off", type=float, default=None,
                    help="claim cap above the X floor (default: half the saddle)")
    ap.add_argument("--buffer", type=float, default=0.15)
    args = ap.parse_args()
    if args.cmd == "material":
        material_report()
    else:
        t0 = time.time()
        floor = band_avg(*X_CART["X"])
        es_m = band_avg(*(BREC0 @ np.array([0.5, 0.5])))
        cap = floor + (args.cap_off if args.cap_off is not None
                       else 0.5 * (es_m - floor))
        hs, n_claim = capped_set(args.m, cap, args.buffer)
        w, E = ea_solve(args.m, hs)
        claim = w[w <= cap]
        f = np.sqrt(claim) / (2 * np.pi)
        print(f"(m,1)=({args.m},1)  N={lat.n_cells(lat.supercell_A(LATTICE, args.m, 1))}  "
              f"theta={np.degrees(lat.twist_angle(LATTICE, args.m, 1)):.3f} deg")
        print(f"  trial {len(hs)} harmonics, census {n_claim}, claimed "
              f"{len(claim)}  ({time.time()-t0:.0f}s)")
        print("  f:", np.array2string(f[:24], precision=6, separator=","))


def box_set(m, n_max):
    """Trial rule for the strong-coupling regime: every envelope harmonic in the
    box |n|_inf <= n_max, deduplicated by momentum coset (n and n + A^T h are the
    same Bloch momentum). Pocket-bound envelopes have a fixed width in MOIRE
    harmonic units, so n_max is angle-independent — that is what makes the model
    cheaper than the supercell as the angle shrinks. Saturating at N_cells means
    the single-band basis is complete."""
    A = lat.supercell_A(LATTICE, m, 1)
    At = np.asarray(A, float).T
    ax = np.arange(-n_max, n_max + 1)
    N1, N2 = np.meshgrid(ax, ax, indexing="ij")
    n1, n2 = N1.reshape(-1), N2.reshape(-1)
    fr = np.linalg.solve(At, np.stack([n1, n2]).astype(float)) % 1.0
    key = (np.round(fr[0], 6) * 1e6).astype(np.int64) * 10_000_019 \
        + (np.round(fr[1], 6) * 1e6).astype(np.int64)
    rad = n1 ** 2 + n2 ** 2
    best = {}
    for i in np.argsort(rad):
        if key[i] not in best:
            best[key[i]] = i
    keep = np.array(sorted(best.values()))
    return np.stack([n1[keep], n2[keep]], 1)


def c4_map(m):
    """C4 on envelope-harmonic indices. With k = X + Brec_sc n, a rotation by
    pi/2 about the origin sends X -> X' and n -> M n + n0, both integer:
        M  = A^T C4 A^{-T},   n0 = A^T (X' - X) / 2pi.
    X and X' fold into the same sector, so the physical spectrum carries exact
    C4 quartets (the thesis's four-fold at X). A trial set that is not closed
    under this map cannot represent them and splits every quartet — sections 14
    and 15 found the same thing for the Galerkin basis."""
    A = np.asarray(lat.supercell_A(LATTICE, m, 1), float)
    C4 = np.array([[0.0, -1.0], [1.0, 0.0]])
    M = A.T @ C4 @ np.linalg.inv(A.T)
    n0 = A.T @ ((X_CART["Xp"] - X_CART["X"]) / (2 * np.pi))
    Mi, n0i = np.rint(M).astype(int), np.rint(n0).astype(int)
    assert np.max(np.abs(M - Mi)) < 1e-9 and np.max(np.abs(n0 - n0i)) < 1e-9
    return Mi, n0i


def c4_closed_set(m, n_max):
    """The box trial set symmetrized over the C4 orbit, then coset-deduplicated:
    the smallest C4-invariant trial space containing the box."""
    base = box_set(m, n_max)
    Mi, n0i = c4_map(m)
    pts = [tuple(v) for v in base]
    cur = np.asarray(base, int)
    for _ in range(3):
        cur = (cur @ Mi.T) + n0i[None, :]
        pts.extend(tuple(v) for v in cur)
    n1 = np.array([p[0] for p in pts])
    n2 = np.array([p[1] for p in pts])
    At = np.asarray(lat.supercell_A(LATTICE, m, 1), float).T
    fr = np.linalg.solve(At, np.stack([n1, n2]).astype(float)) % 1.0
    key = (np.round(fr[0], 6) * 1e6).astype(np.int64) * 10_000_019 \
        + (np.round(fr[1], 6) * 1e6).astype(np.int64)
    rad = n1 ** 2 + n2 ** 2
    best = {}
    for i in np.argsort(rad):
        if key[i] not in best:
            best[key[i]] = i
    keep = np.array(sorted(best.values()))
    return np.stack([n1[keep], n2[keep]], 1)


def ea_full(m, harmonics, Ns=21, gmax=GMAX, fine=256, dk_cell=0.025,
            verbose=True):
    """The completed envelope model on the thesis crystal: registry-adapted AND
    k-resummed single-band trial in the product space (hermitized family),

        column (n) : e^{2 pi i n . s} x u1(k_n; delta(s))  per slow point,

    with u1 the band-1 eigenvector of the hermitized monolayer operator at the
    mode's momentum and the LOCAL registry — the frame follows both arguments, so
    the slow derivative differentiates through registry and momentum content
    mechanically. Frames are cached on k-cells of size dk_cell*|b| (the basin is
    angle-independent, so the cache is bounded as the angle shrinks); columns are
    not orthogonal across modes, so the Galerkin uses the overlap metric.
    Returns the sorted spectrum."""
    import scipy.linalg as _sla
    from lib_v5 import raw_projection as rp
    from lib_v5 import oracles as oc
    A = lat.supercell_A(LATTICE, m, 1)
    Bs = lf.supercell_basis(LATTICE, m, 1)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    A2i = lf.layer2_integer_matrix(LATTICE, m, 1)
    Wm = np.asarray(A2i, float) - np.asarray(A, float)
    ns = mp.pw_set(gmax)
    npw = len(ns)
    t0 = time.time()

    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    deltas = [tuple((Wm @ np.array([S1.reshape(-1)[j], S2.reshape(-1)[j]]))
                    .tolist()) for j in range(Ns * Ns)]
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"rho_cache_m{m}_Ns{Ns}_g{gmax}_f{fine}_H{H}.npz")
    if os.path.exists(cache):
        rho_arr = np.load(cache)["rho"]
        rho_mats = [rho_arr.reshape(Ns * Ns, npw, npw)[j]
                    for j in range(Ns * Ns)]
    else:
        rho_mats = rp.rho_tables(coeffs, deltas, gmax, fine=fine)
        rho_arr = np.array(rho_mats).reshape(Ns, Ns, npw, npw)
        np.savez(cache, rho=rho_arr)
    if verbose:
        print(f"    rho tables ({Ns}x{Ns} registry, npw {npw})  "
              f"{time.time()-t0:.0f}s", flush=True)

    b0n = np.linalg.norm(BREC0[:, 0])
    kcell = {}
    modes = [(int(a), int(b)) for a, b in harmonics]
    ncol = len(modes)
    kmods = np.array([X_CART["X"] + Brec @ np.array(nm, float) for nm in modes])
    cells = [tuple(np.rint(k / (dk_cell * b0n)).astype(int)) for k in kmods]
    need = sorted(set(cells))
    t1 = time.time()
    Brec_pw = BREC0
    kG0 = np.array([Brec_pw @ np.array(h, float) for h in ns])
    cell_of = {c: i for i, c in enumerate(need)}
    import hashlib
    tag = hashlib.md5(np.asarray(need, np.int64).tobytes()).hexdigest()[:10]
    fcache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"frame_cache_m{m}_Ns{Ns}_g{gmax}_{tag}.npy")
    if os.path.exists(fcache):
        U = np.load(fcache)
        if verbose:
            print(f"    frames: cached ({len(need)} k-cells)", flush=True)
        U = _gauge_fix(U, verbose)
        return _assemble(m, Ns, gmax, npw, modes, cells, cell_of, U, rho_arr,
                         kG0, Bs, verbose, t1)
    U = np.zeros((Ns * Ns, len(need), npw), complex)
    # per registry: h(k) = H0 + kx Hx + ky Hy + |k|^2 S2 exactly, so each k-cell
    # costs one 441-dim subset eigh instead of two dense matmuls
    g2 = (kG0 ** 2).sum(1)
    for j in range(Ns * Ns):
        R = rho_mats[j]
        H0 = R @ (g2[:, None] * R)
        Hx = 2.0 * (R @ (kG0[:, 0][:, None] * R))
        Hy = 2.0 * (R @ (kG0[:, 1][:, None] * R))
        Sq = R @ R
        for ci, c in enumerate(need):
            kc = np.array(c, float) * dk_cell * b0n
            h = H0 + kc[0] * Hx + kc[1] * Hy + (kc @ kc) * Sq
            w, V = _sla.eigh(0.5 * (h + h.conj().T),
                             subset_by_index=[0, BAND])
            U[j, ci] = V[:, BAND]
    np.save(fcache, U)
    U = _gauge_fix(U, verbose)
    if verbose:
        print(f"    frames: {len(need)} k-cells x {Ns * Ns} registries  "
              f"{time.time()-t1:.0f}s", flush=True)
    return _assemble(m, Ns, gmax, npw, modes, cells, cell_of, U, rho_arr,
                     kG0, Bs, verbose, t1)


def _gauge_fix(U, verbose=True):
    """Smooth the U(1) gauge of the per-(registry, k-cell) frames.

    Each frame comes from an independent eigh, so its phase is arbitrary; left
    alone the trial column u1(k_n; delta(s)) e^{i n.s} is DISCONTINUOUS in the
    slow variable and the slow derivative manufactures spurious kinetic energy
    (the variational estimate then sits far above the truth). Align every
    registry frame to the reference registry of its own k-cell, exactly as
    hero_engine.adapted_frames does. An overall phase per column is harmless
    (a diagonal unitary on the basis), so only within-column smoothness matters."""
    nreg, ncell, npw = U.shape
    Ns = int(round(np.sqrt(nreg)))
    out = np.array(U).reshape(Ns, Ns, ncell, npw).copy()
    mins = []
    # parallel transport: align each registry frame to its already-aligned
    # neighbour (row 0 along j, then every row to the row above). Local
    # alignment survives a frame that rotates strongly across the torus, which
    # a single global reference cannot.
    for j in range(1, Ns):
        ov = np.einsum("cg,cg->c", out[0, j], np.conj(out[0, j - 1]))
        mag = np.abs(ov)
        mins.append(float(mag.min()))
        out[0, j] *= (np.conj(ov) / np.maximum(mag, 1e-30))[:, None]
    for i in range(1, Ns):
        for j in range(Ns):
            ov = np.einsum("cg,cg->c", out[i, j], np.conj(out[i - 1, j]))
            mag = np.abs(ov)
            mins.append(float(mag.min()))
            out[i, j] *= (np.conj(ov) / np.maximum(mag, 1e-30))[:, None]
    # seam check: the holonomy around the torus (a smooth global gauge exists
    # only if these stay near 1; report it rather than hide it)
    if verbose:
        print(f"    gauge transport: min neighbour |overlap| {min(mins):.3f}",
              flush=True)
    return out.reshape(nreg, ncell, npw)


def _assemble(m, Ns, gmax, npw, modes, cells, cell_of, U, rho_arr, kG0, Bs,
              verbose, t1):
    import scipy.linalg as _sla
    from lib_v5 import oracles as oc
    s = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    ncol = len(modes)
    t2 = time.time()
    X = np.zeros((Ns, Ns, npw, ncol), complex)
    for ci, nm in enumerate(modes):
        ph = np.exp(2j * np.pi * (nm[0] * S1 + nm[1] * S2)) / Ns
        Uc = U[:, cell_of[cells[ci]], :].reshape(Ns, Ns, npw)
        X[:, :, :, ci] = ph[:, :, None] * Uc
    ik = 2j * np.pi * np.fft.fftfreq(Ns) * Ns
    slow_to_cart = np.linalg.inv(Bs).T

    def dslow(T, axis):
        F = np.fft.fft(T, axis=axis)
        shape = [1, 1, 1, 1]
        shape[axis] = Ns
        F *= ik.reshape(shape)
        return np.fft.ifft(F, axis=axis)

    kGX = kG0 + X_CART["X"][None, :]

    def L_apply(T):
        Y = np.einsum("abij,abjc->abic", rho_arr, T)
        out = np.zeros_like(T)
        for i in (0, 1):
            Z = Y * (1j * kGX[:, i])[None, None, :, None] \
                + slow_to_cart[i, 0] * dslow(Y, 0) \
                + slow_to_cart[i, 1] * dslow(Y, 1)
            Z = Z * (1j * kGX[:, i])[None, None, :, None] \
                + slow_to_cart[i, 0] * dslow(Z, 0) \
                + slow_to_cart[i, 1] * dslow(Z, 1)
            out -= np.einsum("abij,abjc->abic", rho_arr, Z)
        return out

    LX = L_apply(X)
    Xf = X.reshape(-1, ncol)
    Hm = Xf.conj().T @ LX.reshape(-1, ncol)
    Sm = Xf.conj().T @ Xf
    Hm = 0.5 * (Hm + Hm.conj().T)
    Sm = 0.5 * (Sm + Sm.conj().T)
    ws = _sla.eigh(Sm, eigvals_only=True)
    Q = oc.b_orthonormalize(np.eye(ncol), Sm)
    Hq = Q.conj().T @ Hm @ Q
    w = np.sort(_sla.eigvalsh(0.5 * (Hq + Hq.conj().T)))
    if verbose:
        print(f"    assembled ncol={ncol} (rank {Q.shape[1]}, "
              f"min overlap eig {ws.min():.1e})  {time.time()-t2:.0f}s",
              flush=True)
    return w
