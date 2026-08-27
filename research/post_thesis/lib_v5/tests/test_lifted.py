"""Lifted-bilayer exactness + the L2 zero-modulation fold-union test."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402
from lib_v5 import lifted as lf  # noqa: E402
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402

L1 = mat.cosine_layer({(1, 0): 0.5, (0, 1): 0.5})
L2 = mat.cosine_layer({(1, 0): 0.3, (0, 1): 0.3})
EPS0 = 4.0


def test_layer2_matrix_integer():
    for lattice, m, n in [("square", 3, 2), ("square", 7, 1), ("honeycomb", 4, 3),
                          ("honeycomb", 30, 29)]:
        A2 = lf.layer2_integer_matrix(lattice, m, n)
        det = int(A2[0, 0]) * int(A2[1, 1]) - int(A2[0, 1]) * int(A2[1, 0])
        assert abs(det) == lat.n_cells(lat.supercell_A(lattice, m, n))


def test_moire_coeffs_match_direct_eval():
    lattice, m, n = "square", 3, 2
    c = lf.moire_coeffs(lattice, m, n, EPS0, L1, L2)
    Bs = lf.supercell_basis(lattice, m, n)
    N = 32
    s = np.arange(N) / N
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    X = Bs[0, 0] * S1 + Bs[0, 1] * S2
    Y = Bs[1, 0] * S1 + Bs[1, 1] * S2
    e_direct = lf.direct_eval(lattice, m, n, EPS0, L1, L2, X, Y)
    e_four = np.zeros_like(e_direct)
    for (m1, m2), cc in c.items():
        e_four += cc * np.exp(2j * np.pi * (m1 * S1 + m2 * S2))
    assert np.max(np.abs(e_four - e_direct)) < 1e-10


def test_l2_zero_modulation_fold_union():
    """The L2 ladder rung: with layer2 = 0, the supercell spectrum at sector kappa_s is
    exactly the union over reciprocal cosets of the monolayer spectra."""
    lattice, m, n = "square", 2, 1
    A = lat.supercell_A(lattice, m, n)
    Ncell = lat.n_cells(A)                      # 5
    B0 = lat.monolayer_basis(lattice)
    Bs = lf.supercell_basis(lattice, m, n)
    c_mono = mat.bilayer(EPS0, L1, {})
    c_sc = lf.moire_coeffs(lattice, m, n, EPS0, L1, {})
    kappa_s = np.array([0.21, 0.34])
    k_cart = 2 * np.pi * np.linalg.inv(Bs).T @ kappa_s

    gmax_sc = 8
    w_sc, _, _, _ = mp.solve(c_sc, k_cart, Bs, gmax_sc)

    Af = np.asarray(A, dtype=float)
    folded = []
    for rep in lat.coset_representatives(A):
        kappa0 = np.linalg.solve(Af.T, kappa_s + np.array(rep, float))
        k0_cart = 2 * np.pi * np.linalg.inv(B0).T @ kappa0
        # matched basis truncation: monolayer PWs that fit inside the supercell window
        w_m, _, kGm, _ = mp.solve(c_mono, k0_cart, B0, gmax=4)
        folded.extend(w_m.tolist())
    folded = np.sort(folded)

    # compare the bottom of the spectra, where both truncations are converged
    nb = 3 * Ncell
    d = np.abs(np.sort(w_sc)[:nb] - folded[:nb])
    assert np.max(d) < 1e-8, d.max()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
