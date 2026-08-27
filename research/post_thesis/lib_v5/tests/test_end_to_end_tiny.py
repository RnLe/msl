"""Tiny synthetic end-to-end regression: oblique basis, off-symmetry carrier,
band_lo=1, gauge invariance — the master pin for axes, units, band identity,
adjoints, and the product-space assembly, on top of test_raw_projection.py."""
import os
import sys
from fractions import Fraction

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402
from lib_v5 import raw_projection as rp  # noqa: E402

EPS0 = 4.0
L1H = mat.cosine_layer({(1, 0): 0.55, (0, 1): 0.35, (-1, -1): 0.45})
L2H = mat.cosine_layer({(1, 0): 0.1, (0, 1): 0.1, (-1, -1): 0.1})
GMAX = 3
K_GEN = (Fraction(2, 7), Fraction(1, 5))       # generic, no symmetry


def _coeffs_hex(d):
    return mat.bilayer(EPS0, L1H, L2H, delta=d)


def test_hex_frozen_complete_frame_exact():
    """Oblique-basis product-space assembly against the monolayer family at the
    shifted momenta (complete frame, constant registry) — machine-exact."""
    Ns = 5
    orders = rp.product_space_orders("hex", _coeffs_hex, K_GEN, GMAX, Ns,
                                     registry_of_R=lambda a, b: (0.31, 0.62))
    Bs = lat.monolayer_basis("hex") @ np.array([[3.0, 0.0], [0.0, 3.0]])
    Lfull, _, _ = rp.assemble(orders, 1.0, np.linalg.inv(Bs).T)
    assert np.linalg.norm(Lfull - Lfull.conj().T) < 1e-9 * np.linalg.norm(Lfull)
    w = np.sort(sla.eigvalsh(Lfull))
    B0 = lat.monolayer_basis("hex")
    Brec = 2 * np.pi * np.linalg.inv(B0).T
    k0 = Brec @ np.array([float(K_GEN[0]), float(K_GEN[1])])
    c = _coeffs_hex((0.31, 0.62))
    ref = []
    for n1 in range(Ns):
        for n2 in range(Ns):
            q = k0 + 2 * np.pi * np.linalg.inv(Bs).T @ np.array(
                [np.fft.fftfreq(Ns)[n1] * Ns, np.fft.fftfreq(Ns)[n2] * Ns])
            h, _, _, _ = rp.mono_hermitized(c, q, B0, GMAX)
            ref.extend(sla.eigvalsh(h).tolist())
    assert np.max(np.abs(w - np.sort(ref))) < 1e-8


def test_band_lo_1_raw_vs_direct_hex():
    """Retained band = 1 (NOT the lowest), generic carrier, hex commensuration:
    the raw projection tracks the direct lifted projection at the same scale as the
    square family (no band-identity or axis confusion anywhere in the chain)."""
    m, n = 5, 4
    Ns = 7
    band = 1
    lattice = "hex"
    A = lat.supercell_A(lattice, m, n)
    A2 = lf.layer2_integer_matrix(lattice, m, n)
    W = np.asarray(A2, float) - np.asarray(A, float)
    orders = rp.product_space_orders(
        lattice, _coeffs_hex, K_GEN, GMAX, Ns,
        registry_of_R=lambda a, b: tuple(W @ np.array([a, b])))
    Bsup = lf.supercell_basis(lattice, m, n)
    Lfull, _, _ = rp.assemble(orders, 1.0, np.linalg.inv(Bsup).T)
    U, _ = rp.frozen_frame(lattice, _coeffs_hex, (0.0, 0.0), K_GEN, GMAX, [band])
    H_P, herm = rp.project(Lfull, U, orders["nslow"])
    assert herm < 1e-9
    w_raw = np.sort(np.linalg.eigvalsh(0.5 * (H_P + H_P.conj().T)))

    c_sc = lf.moire_coeffs(lattice, m, n, EPS0, L1H, L2H)
    ks = lat.fold_sector(A, K_GEN)
    k_sc = lat.sector_to_cartesian(Bsup, ks)
    Af = np.abs(np.asarray(A, dtype=float))
    gmax_sc = int(GMAX * max(Af[0, 0] + Af[1, 0], Af[0, 1] + Af[1, 1])
                  + Ns // 2 + 6)
    C, _, ns_sc, _ = rp.mono_hermitized(c_sc, k_sc, Bsup, gmax_sc, fine=160)
    lookup = {tuple(v): i for i, v in enumerate(ns_sc)}
    Ai = np.asarray(A, dtype=object)
    n0f = lat.fold_sector(A, K_GEN)
    n0 = (int(Ai[0, 0] * K_GEN[0] + Ai[1, 0] * K_GEN[1] - n0f[0]),
          int(Ai[0, 1] * K_GEN[0] + Ai[1, 1] * K_GEN[1] - n0f[1]))
    env = [(e1, e2) for e1 in range(-(Ns // 2), Ns // 2 + 1)
           for e2 in range(-(Ns // 2), Ns // 2 + 1)]
    # operator consistency: restrict the supercell operator to the EXACT index set
    # the product space represents ({A^T h + n0 + e}); the comparison then measures
    # the lift/diagonal-restriction error alone, not the fast-window truncation
    # mismatch (measured contrast: unrestricted C gives 1.3e-3 here, and a 2-band
    # frame 3e-2, both dominated by rho-spread beyond gmax_mono)
    idx = []
    for e in env:
        for h in mp.pw_set(GMAX):
            key = (int(Ai[0, 0]) * h[0] + int(Ai[1, 0]) * h[1] + n0[0] + e[0],
                   int(Ai[0, 1]) * h[0] + int(Ai[1, 1]) * h[1] + n0[1] + e[1])
            idx.append(lookup[key])
    idx = sorted(set(idx))
    sub = {j: i for i, j in enumerate(idx)}
    Csub = C[np.ix_(idx, idx)]
    cols = []
    for e in env:
        col = np.zeros(len(idx), complex)
        for i_h, h in enumerate(mp.pw_set(GMAX)):
            key = (int(Ai[0, 0]) * h[0] + int(Ai[1, 0]) * h[1] + n0[0] + e[0],
                   int(Ai[0, 1]) * h[0] + int(Ai[1, 1]) * h[1] + n0[1] + e[1])
            col[sub[lookup[key]]] = U[i_h, 0]
        cols.append(col)
    T = np.array(cols).T
    HT = T.conj().T @ (Csub @ T)
    ST = T.conj().T @ T
    w_dir = np.sort(sla.eigh(0.5 * (HT + HT.conj().T), 0.5 * (ST + ST.conj().T),
                             eigvals_only=True))
    # At a GENERIC carrier the frozen single-band drift term is linear with v != 0,
    # and BOTH projected spectra develop an unphysical low tail that dives with the
    # envelope grid (measured: bottom 2.9 -> 0.98 for Ns 7 -> 15, far below
    # lambda_1 = 7.47 — the parabola-extension disease reproduced in miniature, and
    # why validation carriers must be dispersion extrema; the quantitative
    # raw-vs-direct scaling lives in test_raw_projection at a band minimum).
    lam1 = 7.4675
    assert w_raw[0] < lam1 - 3.0, w_raw[0]          # the disease is present
    assert w_dir[0] < lam1 - 3.0, w_dir[0]          # ... in both representations
    # loose set-consistency of the near-carrier content (index alignment is not
    # meaningful in a polluted window; tight matching is deferred to extremum
    # carriers where the window is clean)
    wr = w_raw[(w_raw > lam1 - 0.6) & (w_raw < lam1 + 1.2)]
    wd = w_dir[(w_dir > lam1 - 0.6) & (w_dir < lam1 + 1.2)]
    assert len(wr) >= 4 and len(wd) >= 4
    nn = max(np.abs(wd - w_raw[:, None]).min(axis=0).max(),
             np.abs(wr - w_dir[:, None]).min(axis=0).max())
    assert nn < 5e-2, nn


def test_two_band_gauge_invariance():
    """A random U(2) rotation of the retained frame leaves the raw-projection
    spectrum invariant to roundoff."""
    Ns = 5
    orders = rp.product_space_orders("square",
                                     lambda d: mat.bilayer(EPS0, L1H, L2H, delta=d),
                                     (Fraction(1, 3), Fraction(1, 7)), GMAX, Ns,
                                     registry_of_R=lambda a, b: (a, b))
    Lfull, _, _ = rp.assemble(orders, 1.0, np.linalg.inv(2.5 * np.eye(2)).T)
    U, _ = rp.frozen_frame("square",
                           lambda d: mat.bilayer(EPS0, L1H, L2H, delta=d),
                           (0.0, 0.0), (Fraction(1, 3), Fraction(1, 7)), GMAX, [0, 1])
    H0, _ = rp.project(Lfull, U, orders["nslow"])
    rng = np.random.default_rng(11)
    X = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Wg, _ = np.linalg.qr(X)
    H1, _ = rp.project(Lfull, U @ Wg, orders["nslow"])
    w0 = np.sort(np.linalg.eigvalsh(0.5 * (H0 + H0.conj().T)))
    w1 = np.sort(np.linalg.eigvalsh(0.5 * (H1 + H1.conj().T)))
    assert np.max(np.abs(w0 - w1)) < 1e-9 * max(np.max(np.abs(w0)), 1.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
