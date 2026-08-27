#!/usr/bin/env python3
"""Frozen smooth-bilayer validation candidate: single-M-valley hex crystal.

Design (see FINDINGS, smooth campaign): TM, hexagonal lattice, finite-Fourier host with
a detuned three-star (breaks C3 so the three M points split; M is time-reversal
invariant, so the retained valley has NO symmetry or TR partner — a single-carrier,
single-valley manifold), weak layer-2 star as the moire potential. Every solver consumes
the identical 7-coefficient analytic dielectric; there is no rasterization anywhere.

Frozen numbers (verified by running this file; coarse grids, gmax=4):
  band-1 floor at M2 = (0, 1/2), single valley
  manifold window [1.6383, 1.6615], V = 0.0232
  registry-common below-clearance +0.148 (full 0-1 gap), band-2 headroom +0.611,
  next-M separation +0.011, min-eps bound 0.84
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402

LATTICE = "hex"
EPS0 = 9.0
STAR = [(1, 0), (0, 1), (-1, -1)]
LAYER1_AMPS = {(1, 0): 2.95, (0, 1): 2.25, (-1, -1): 2.60}
LAYER2_AMP = 0.12
BAND = 1
CARRIER_FRAC = (0.0, 0.5)          # M2; time-reversal invariant, unique band-1 floor
WINDOW = (1.6383, 1.6615)          # lambda = (2 pi f)^2 units


def layers():
    return (mat.cosine_layer(LAYER1_AMPS),
            mat.cosine_layer({h: LAYER2_AMP for h in STAR}))


def coeffs(delta=(0.0, 0.0)):
    l1, l2 = layers()
    return mat.bilayer(EPS0, l1, l2, delta=delta)


def carrier_cartesian():
    B0 = lat.monolayer_basis(LATTICE)
    return 2 * np.pi * np.linalg.inv(B0).T @ np.array(CARRIER_FRAC)


def verify(nreg=5, nk=6, gmax=4, verbose=True):
    B0 = lat.monolayer_basis(LATTICE)
    l1, l2 = layers()
    assert mat.min_bound(coeffs((0.5, 0.5))) > 0.5
    Ms = {"M1": (0.5, 0.0), "M2": CARRIER_FRAC, "M3": (0.5, 0.5)}
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    lvs = {nm: [] for nm in Ms}
    for i, j in itertools.product(range(nreg), repeat=2):
        c = mat.bilayer(EPS0, l1, l2, delta=(i / nreg, j / nreg))
        for nm, kf in Ms.items():
            k = 2 * np.pi * np.linalg.inv(B0).T @ np.array(kf)
            w, _, _, _ = mp.solve(c, k, B0, gmax, n_bands=3)
            lvs[nm].append(w[BAND])
        for ki, kj in itertools.product(range(nk), repeat=2):
            k = 2 * np.pi * np.linalg.inv(B0).T @ np.array([ki / nk, kj / nk])
            w, _, _, _ = mp.solve(c, k, B0, gmax, n_bands=3)
            lo = np.minimum(lo, w[:3])
            hi = np.maximum(hi, w[:3])
    m2 = np.array(lvs["M2"])
    below = m2.min() - hi[BAND - 1]
    head = lo[BAND + 1] - m2.max()
    sep = min(np.array(lvs[nm]).min() for nm in ("M1", "M3")) - m2.max()
    V = m2.max() - m2.min()
    if verbose:
        print(f"window [{m2.min():.4f},{m2.max():.4f}] V={V:.4f}")
        print(f"below {below:+.4f}  head {head:+.4f}  next-M sep {sep:+.4f}")
    assert below > 0.10, below
    assert head > 0.50, head
    assert sep > 0.005, sep
    assert abs(m2.min() - WINDOW[0]) < 0.01 and abs(m2.max() - WINDOW[1]) < 0.01
    return dict(window=(m2.min(), m2.max()), V=V, below=below, head=head, sep=sep)


def curvature(gmax=5, h=1e-3):
    """Band-1 dispersion curvature at the carrier (for the V/E_kin angle choice)."""
    c0 = coeffs((0.0, 0.0))
    B0 = lat.monolayer_basis(LATTICE)
    k0 = carrier_cartesian()
    out = {}
    for name, e in (("xx", np.array([1.0, 0.0])), ("yy", np.array([0.0, 1.0]))):
        wp, _, _, _ = mp.solve(c0, k0 + h * e, B0, gmax, n_bands=2)
        w0, _, _, _ = mp.solve(c0, k0, B0, gmax, n_bands=2)
        wm, _, _, _ = mp.solve(c0, k0 - h * e, B0, gmax, n_bands=2)
        out[name] = (wp[BAND] - 2 * w0[BAND] + wm[BAND]) / h ** 2
    return out


if __name__ == "__main__":
    stats = verify()
    curv = curvature()
    print("curvature d2lam/dk2:", {k: round(v, 3) for k, v in curv.items()})
    # V / E_kin at commensurate hex angles: E_kin ~ 0.5*curv*(eta*|b|)^2 scale
    B0 = lat.monolayer_basis(LATTICE)
    babs = np.linalg.norm(2 * np.pi * np.linalg.inv(B0).T[:, 0])
    for m, n in [(4, 3), (5, 4), (6, 5), (7, 6), (9, 8)]:
        th = lat.twist_angle(LATTICE, m, n)
        eta = 2 * np.sin(th / 2)
        ekin = 0.5 * min(abs(c) for c in curv.values()) * (eta * babs) ** 2
        print(f"(m,n)=({m},{n}) theta={np.degrees(th):.2f} deg  "
              f"V/E_kin ~ {stats['V'] / ekin:.1f}")
