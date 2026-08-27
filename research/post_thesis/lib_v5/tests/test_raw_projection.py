"""Raw projected operator: exactness gates and the raw-vs-direct scaling measurement.

Conventions verified here:
  - odd slow grids make the spectral slow derivative exactly anti-symmetric, so all
    three lifted orders are Hermitian by construction (asserted, never repaired);
  - with the exact phase registry map delta(s) = (A2 - A) s, the finite-Fourier local
    proxy reproduces the twisted bilayer material identically, so the complete-frame
    product-space operator at frozen registry must match the monolayer dispersion at
    the shifted momenta to machine precision (the decisive assembly gate);
  - the raw frozen-frame projection is then compared against the direct lifted-basis
    projection of the exact supercell pencil across a commensurate angle family.
"""
import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402
from lib_v5 import raw_projection as rp  # noqa: E402
from lib_v5 import ritz_moire as rz  # noqa: E402

L1A = mat.cosine_layer({(1, 0): 0.5, (0, 1): 0.35})
L2A = mat.cosine_layer({(1, 0): 0.12, (0, 1): 0.12})
EPS0 = 4.0
GMAX = 3


def _coeffs_fn(d):
    return mat.bilayer(EPS0, L1A, L2A, delta=d)


def test_orders_hermitian_and_frozen_exact():
    """Constant registry, COMPLETE frame: the product-space spectrum must equal the
    union of monolayer spectra at the slow-grid shifted momenta (machine-exact)."""
    Ns = 5
    kappa0 = (0.0, 0.0)
    orders = rp.product_space_orders("square", _coeffs_fn, kappa0, GMAX, Ns,
                                     registry_of_R=lambda a, b: (0.2, 0.7))
    Bs = np.eye(2) * 3.0            # an arbitrary slow cell three cells wide
    Lfull, L1o, L2o = rp.assemble(orders, 1.0, np.linalg.inv(Bs).T)
    for M in (Lfull, L1o, L2o):
        assert np.linalg.norm(M - M.conj().T) < 1e-9 * max(np.linalg.norm(M), 1)
    w = np.sort(sla.eigvalsh(Lfull))
    B0 = lat.monolayer_basis("square")
    c = _coeffs_fn((0.2, 0.7))
    ref = []
    for n1 in range(Ns):
        for n2 in range(Ns):
            q = 2 * np.pi * np.linalg.inv(Bs).T @ np.array(
                [np.fft.fftfreq(Ns)[n1] * Ns, np.fft.fftfreq(Ns)[n2] * Ns])
            h, _, _, _ = rp.mono_hermitized(c, q, B0, GMAX)
            ref.extend(sla.eigvalsh(h).tolist())
    ref = np.sort(ref)
    assert np.max(np.abs(w - ref)) < 1e-8, np.max(np.abs(w - ref))


def _raw_vs_direct(m, Ns, band=0, n_env_compare=5):
    """Bottom-eigenvalue difference between the frozen-frame raw projection and the
    direct lifted projection of the exact supercell pencil, for square (m, 1)."""
    lattice = "square"
    A = lat.supercell_A(lattice, m, 1)
    A2 = lf.layer2_integer_matrix(lattice, m, 1)
    W = (np.asarray(A2, float) - np.asarray(A, float))

    def registry_of_R(s1, s2):
        v = W @ np.array([s1, s2])
        return (float(v[0]), float(v[1]))

    kappa0 = (0.0, 0.0)
    orders = rp.product_space_orders(lattice, _coeffs_fn, kappa0, GMAX, Ns,
                                     registry_of_R=registry_of_R)
    Bs = lf.supercell_basis(lattice, m, 1)
    Lfull, _, _ = rp.assemble(orders, 1.0, np.linalg.inv(Bs).T)
    U, _ = rp.frozen_frame(lattice, _coeffs_fn, (0.0, 0.0), kappa0, GMAX, [band])
    H_P, herm = rp.project(Lfull, U, orders["nslow"])
    assert herm < 1e-9, herm
    w_raw = np.sort(np.linalg.eigvalsh(0.5 * (H_P + H_P.conj().T)))

    # direct side: hermitized-collocation supercell operator (the SAME discretization
    # family), projected onto the lifted frozen-frame trial columns
    Bsup = lf.supercell_basis(lattice, m, 1)
    c_sc = lf.moire_coeffs(lattice, m, 1, EPS0, L1A, L2A)
    gmax_sc = GMAX * m + Ns // 2 + 4
    ks = lat.fold_sector(A, kappa0)
    k_sc = lat.sector_to_cartesian(Bsup, ks)
    C, Rsc, ns_sc, _ = rp.mono_hermitized(c_sc, k_sc, Bsup, gmax_sc, fine=128)
    lookup = {tuple(v): i for i, v in enumerate(ns_sc)}
    ns_m = mp.pw_set(GMAX)
    env = [(e1, e2) for e1 in range(-(Ns // 2), Ns // 2 + 1)
           for e2 in range(-(Ns // 2), Ns // 2 + 1)]
    Ai = np.asarray(A, dtype=object)
    cols = []
    for e in env:
        col = np.zeros(len(ns_sc), complex)
        for i_h, h in enumerate(ns_m):
            key = (int(Ai[0, 0]) * h[0] + int(Ai[1, 0]) * h[1] + e[0],
                   int(Ai[0, 1]) * h[0] + int(Ai[1, 1]) * h[1] + e[1])
            col[lookup[key]] = U[i_h, 0]
        cols.append(col)
    T = np.array(cols).T
    HT = T.conj().T @ (C @ T)
    ST = T.conj().T @ T
    w_dir = np.sort(sla.eigh(0.5 * (HT + HT.conj().T), 0.5 * (ST + ST.conj().T),
                             eigvals_only=True))
    nc = min(n_env_compare, len(w_dir), len(w_raw))
    return np.abs(w_raw[:nc] - w_dir[:nc]).max()


def test_raw_vs_direct_scaling():
    """The headline measurement: the raw-projection-vs-direct-lift deviation across a
    commensurate family. eta = 2 sin(atan(1/m)); the deviation must shrink with eta
    and the fitted order is reported (and pinned loosely, to catch regressions)."""
    # m=3 is excluded: its registry winding needs a denser slow grid than Ns=7
    # (aliasing measured at 4.6e-3); the resolved family is m >= 5
    ms = [5, 7, 9]
    devs = []
    for m in ms:
        devs.append(_raw_vs_direct(m, Ns=7))
    etas = [2 * np.sin(np.arctan2(1, m)) for m in ms]
    assert devs[0] > devs[1] > devs[2], devs
    p = np.polyfit(np.log(etas), np.log(devs), 1)[0]
    # measured at freeze time: deviations ~3e-7 -> 1e-7 -> 5e-8 in lambda
    # (~2e-8 in frequency); the fitted order alerts on regressions
    assert p > 1.5, (p, devs)
    assert devs[0] < 1e-5, devs
    print(f"raw-vs-direct deviations {['%.3e' % d for d in devs]}  "
          f"fitted eta-order {p:.2f}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
