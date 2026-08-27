#!/usr/bin/env python3
"""Candidate scan for the smooth weak-bilayer validation family.

For the square finite-Fourier bilayer eps = eps0 + a1*(cos 2pi s1 + cos 2pi s2)
+ a2*(cos 2pi (s+delta)_1 + cos 2pi (s+delta)_2), find (eps0, a1, a2, band, carrier)
with: a registry-common isolated band (uniform gap below and above over the full
registry torus and BZ), and a controlled registry modulation of the carrier level
(the moire potential V). Acceptance targets: common gaps > 0, V > 0 (nonzero
modulation), V / gap moderate; the angle then sets V/E_kin ~ V / (eta^2 curvature).

Usage: scan_candidates.py [--gmax 4] [--nreg 6] [--nk 6]
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402

B0 = np.eye(2)


def band_ranges(eps0, a1, a2, gmax, nreg, nk, nbands):
    """min/max of each band over registry torus x BZ, plus the carrier-level
    registry spread at a set of candidate carriers."""
    l1 = mat.cosine_layer({(1, 0): a1, (0, 1): a1})
    l2 = mat.cosine_layer({(1, 0): a2, (0, 1): a2})
    deltas = [(i / nreg, j / nreg) for i in range(nreg) for j in range(nreg)]
    ks = [(i / nk, j / nk) for i in range(nk) for j in range(nk)]
    carriers = {"X": (0.5, 0.0), "M": (0.5, 0.5), "G2": (0.25, 0.1)}
    lo = np.full(nbands, np.inf)
    hi = np.full(nbands, -np.inf)
    carrier_levels = {c: {b: [] for b in range(nbands)} for c in carriers}
    for d in deltas:
        c = mat.bilayer(eps0, l1, l2, delta=d)
        for kf in ks:
            k_cart = 2 * np.pi * np.array(kf)
            w, _, _, _ = mp.solve(c, k_cart, B0, gmax, n_bands=nbands)
            lo = np.minimum(lo, w[:nbands])
            hi = np.maximum(hi, w[:nbands])
        for name, kf in carriers.items():
            k_cart = 2 * np.pi * np.array(kf)
            w, _, _, _ = mp.solve(c, k_cart, B0, gmax, n_bands=nbands)
            for b in range(nbands):
                carrier_levels[name][b].append(w[b])
    return lo, hi, carrier_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmax", type=int, default=4)
    ap.add_argument("--nreg", type=int, default=6)
    ap.add_argument("--nk", type=int, default=6)
    ap.add_argument("--nbands", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for eps0, a1, a2 in itertools.product([4.0, 6.0], [0.8, 1.2, 1.6],
                                          [0.15, 0.3]):
        lo, hi, cl = band_ranges(eps0, a1, a2, args.gmax, args.nreg, args.nk,
                                 args.nbands)
        for b in range(1, args.nbands - 1):
            for cname in cl:
                lv = np.array(cl[cname][b])
                V = lv.max() - lv.min()
                # window criterion: the manifold window [min lv, max lv] (band b at
                # the carrier, over the registry torus) must contain no OTHER band's
                # content anywhere in the BZ, and the carrier should be the band's
                # own dispersion floor (same-band in-window content is then the
                # near-carrier envelope physics the EA describes)
                below_clear = lv.min() - hi[b - 1]
                headroom = lo[b + 1] - lv.max()
                at_min = lv.min() - lo[b] < 1e-9
                rows.append((eps0, a1, a2, b, cname, below_clear, headroom, V,
                             float(lv.min()), at_min))
    rows.sort(key=lambda r: -min(r[5], r[6]))
    print(f"{'eps0':>5} {'a1':>4} {'a2':>4} {'b':>2} {'k0':>3} "
          f"{'below_clr':>10} {'headroom':>10} {'V(k0)':>9} {'lam_min':>9} {'@min':>4}")
    for r in rows[:25]:
        print(f"{r[0]:5.1f} {r[1]:4.1f} {r[2]:4.2f} {r[3]:2d} {r[4]:>3} "
              f"{r[5]:+10.4f} {r[6]:+10.4f} {r[7]:9.4f} {r[8]:9.4f} {'Y' if r[9] else 'n':>4}")


if __name__ == "__main__":
    main()
