#!/usr/bin/env python3
"""EA v2 — the resummed-frame envelope model, built from the monolayer alone.

Trial: the exact band-1 Bloch function of the registry-averaged monolayer at each
in-domain folded momentum, t_n = u1(k_n) e^{i k_n r}, k_n = M2 + Brec_sc n. In the
TM generalized pencil A u = lambda B u (A = diag|k+G|^2, B = eps_hat) the Galerkin
blocks collapse to monolayer quantities:

  * distinct harmonics are distinct momentum cosets, so A is diagonal and, with
    B-orthonormal monolayer eigenvectors (micro_pwe.solve), A_nn = E_band1(k_n)
    and B_nn = 1 exactly;
  * the supercell harmonics of the dielectric split: eps0 + layer-1 act within a
    coset (already inside E_band1), the six layer-2 star harmonics hop one moire
    unit, n -> n - W^T h with W = A2 - A (the registry identity delta(s) = W s):
        B_{n, n-W^T h} = c_h  sum_g  u1*(g; k_n) u1(g - h; k_{n-W^T h}).

So the whole model is:   diag(E_band1(k_n)) c = lambda (I + V) c
with V the six-neighbor interlayer hop matrix — exact local band data plus
nearest-neighbor envelope hopping. No supercell object anywhere; cost is
N_domain monolayer solves and O(N_domain) overlap sums.

The gauge of u1(k_n) cancels: each column's phase appears once in V's row and
once conjugated in its column, and eigenvalues are invariant.
"""
import argparse
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "/home/renlephy/msl/research/post_thesis")
import ladder_wide as lw
import valley_diagnosis as vd
from lib_v5 import lattice as lat
from lib_v5 import lifted as lf
from lib_v5 import micro_pwe as mp

B0 = vd.B0
GMAX = vd.GMAX_MONO


def solve_domain(m, n, harmonics, layers=None, gmax=GMAX, band=None):
    """The EA v2 spectrum on the given envelope harmonics. Returns (eigenvalues,
    E_band1 per harmonic, the hop matrix norm) — everything monolayer-built."""
    band = vd.cand.BAND if band is None else band
    Bs = lf.supercell_basis(vd.LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    A2i = lf.layer2_integer_matrix(vd.LATTICE, m, n)
    Wm = (np.asarray(A2i, int) - np.asarray(lat.supercell_A(vd.LATTICE, m, n),
                                            int))
    l1, l2 = layers if layers is not None else vd.cand.layers()
    from lib_v5.materials import _sym
    star2 = dict(_sym(l2))                      # the six layer-2 harmonics

    hs = [(int(a), int(b)) for a, b in harmonics]
    idx = {h: i for i, h in enumerate(hs)}
    N = len(hs)
    ns = mp.pw_set(gmax)
    pw_idx = {h: i for i, h in enumerate(ns)}
    avg = vd.avg_coeffs()
    # coset-aware lookup: a hop target may be represented by a column at a
    # monolayer-lattice-shifted index (the trial keeps one representative per
    # coset); resolve it and carry the exact re-indexing u1(g; k+G0) = u1(g+g0; k)
    At = np.asarray(lat.supercell_A(vd.LATTICE, m, n), int).T
    Ati = np.linalg.inv(At.astype(float))

    def coset_key(a, b):
        f = Ati @ np.array([a, b], float)
        return (round(float(f[0]) % 1.0, 6), round(float(f[1]) % 1.0, 6))

    ckey = {coset_key(a, b): (a, b) for (a, b) in hs}

    def resolve(a, b):
        """(column index, integer pw shift g0) for the harmonic (a, b)."""
        j = idx.get((a, b))
        if j is not None:
            return j, (0, 0)
        rep = ckey.get(coset_key(a, b))
        if rep is None:
            return None, None
        g0 = Ati @ np.array([a - rep[0], b - rep[1]], float)
        g0i = np.rint(g0).astype(int)
        assert np.max(np.abs(g0 - g0i)) < 1e-9
        return idx[rep], (int(g0i[0]), int(g0i[1]))

    E = np.zeros(N)
    U = np.zeros((len(ns), N), complex)
    for i, (a, b) in enumerate(hs):
        k = vd.M_CART["M2"] + Brec @ np.array([a, b], float)
        w, V, _, _ = mp.solve(avg, k, B0, gmax, n_bands=band + 1)
        E[i] = w[band]
        U[:, i] = V[:, band]

    V2 = np.zeros((N, N), complex)
    for (h1, h2), c in star2.items():
        d = (int(Wm[0, 0] * h1 + Wm[1, 0] * h2),
             int(Wm[0, 1] * h1 + Wm[1, 1] * h2))     # W^T h
        for (a, b), i in idx.items():
            j, g0 = resolve(a - d[0], b - d[1])
            if j is None:
                continue
            s = 0.0 + 0.0j
            for (g1, g2), gi in pw_idx.items():
                gj = pw_idx.get((g1 - h1 + g0[0], g2 - h2 + g0[1]))
                if gj is not None:
                    s += np.conj(U[gi, i]) * U[gj, j]
            V2[i, j] += c * s
    V2 = 0.5 * (V2 + V2.conj().T)
    w = sla.eigh(np.diag(E), np.eye(N) + V2, eigvals_only=True)
    return np.sort(w), E, float(np.linalg.norm(V2))


def capped_harmonics(m, n, e_cap, buffer_e=0.008, lim=None):
    """Energy-capped harmonic set WITHOUT the basin restriction: every folded
    momentum whose averaged band-1 energy is at or below e_cap (+ buffer for the
    trial). The resummed model carries the exact u1(k) per harmonic, so it is
    valley-agnostic — the single-valley domain was a limitation of the FIXED
    frame only. Returns (claim harmonics, energies, trial harmonics)."""
    Bs = lf.supercell_basis(vd.LATTICE, m, n)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    b0 = np.linalg.norm(vd.BREC0[:, 0])
    if lim is None:
        lim = int(np.ceil(0.62 * b0 / np.linalg.norm(Brec[:, 0]))) + 2
    ax = np.arange(-lim, lim + 1)
    N1, N2 = np.meshgrid(ax, ax, indexing="ij")
    n1, n2 = N1.reshape(-1), N2.reshape(-1)
    kx = vd.M_CART["M2"][0] + Brec[0, 0] * n1 + Brec[0, 1] * n2
    ky = vd.M_CART["M2"][1] + Brec[1, 0] * n1 + Brec[1, 1] * n2
    e = np.array([vd.band1_avg(x, y) for x, y in zip(kx, ky)])
    # deduplicate by momentum coset: harmonics differing by A^T h are the SAME
    # Bloch state (k and k + G_mono); keep the lowest-|n| representative
    At = np.asarray(lat.supercell_A(vd.LATTICE, m, n), float).T
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
    hs = np.stack([n1[trial], n2[trial]], 1)[order]
    return (np.stack([n1[claim], n2[claim]], 1), np.sort(e[claim]), hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--n", type=int, default=31)
    ap.add_argument("--scaled", action="store_true")
    ap.add_argument("--buffer", type=float, default=0.006)
    args = ap.parse_args()
    m, n = args.m, args.n
    t0 = time.time()
    lyr = lw.scaled_layers(m, n)[0] if args.scaled else None
    dom, dom_e, grid = vd.domain_harmonics(m, n)
    ce = grid["ceiling"]
    off = ce - dom_e[0]
    sel = np.where((grid["basin"] == 1) & np.isfinite(grid["e"])
                   & (grid["e"] > ce) & (grid["e"] <= ce + args.buffer))[0]
    hs = np.vstack([dom, np.stack([grid["n1"][sel], grid["n2"][sel]], 1)])
    w, E, vn = solve_domain(m, n, hs, layers=lyr)
    claim = w[w <= w[0] + off]
    print(f"({m},{n}){' scaled' if args.scaled else ''}: {len(hs)} harmonics, "
          f"|V| = {vn:.2e}, {len(claim)} claimed  ({time.time()-t0:.0f}s)")
    print(np.array2string(claim, precision=7, separator=","))


if __name__ == "__main__":
    main()
