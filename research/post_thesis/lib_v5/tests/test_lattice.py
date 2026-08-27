"""Lattice/sector algebra regression tests (audit anchors T0.7, M6, M14)."""
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import lattice as lat  # noqa: E402


def test_hex_k_corner_geometry():
    B = lat.monolayer_basis("honeycomb")
    Brec = 2 * np.pi * np.linalg.inv(B).T
    babs = np.linalg.norm(Brec[:, 0])
    for f in lat.k_corner_frac("honeycomb"):
        k = Brec @ np.array([float(f[0]), float(f[1])])
        assert abs(np.linalg.norm(k) - babs / np.sqrt(3)) < 1e-12
    # the historical V4 value is NOT a corner
    k_bad = Brec @ np.array([1 / 3, 1 / 3])
    assert abs(np.linalg.norm(k_bad) - babs / np.sqrt(3)) > 0.1


def test_fold_sector_honeycomb_30_29():
    A = lat.supercell_A("honeycomb", 30, 29)
    ks = lat.fold_sector(A, (Fraction(2, 3), Fraction(1, 3)))
    assert ks == (Fraction(1, 3), Fraction(2, 3))
    # the compensating envelope sector brings it to Gamma
    ks0 = lat.fold_sector(A, (Fraction(2, 3), Fraction(1, 3)),
                          kappa_env=(Fraction(2, 3), Fraction(1, 3)))
    assert ks0 == (Fraction(0), Fraction(0))


def test_fold_sector_square_57_1():
    A = lat.supercell_A("square", 57, 1)
    ks = lat.fold_sector(A, (Fraction(1, 2), Fraction(0)))
    assert ks == (Fraction(1, 2), Fraction(1, 2))


def test_n_cells():
    assert lat.n_cells(lat.supercell_A("honeycomb", 30, 29)) == 2611
    assert lat.n_cells(lat.supercell_A("square", 57, 1)) == 3250
    assert lat.n_cells(lat.supercell_A("square", 7, 1)) == 50


def test_smith_normal_form_random():
    rng = np.random.default_rng(0)
    for _ in range(200):
        M = rng.integers(-9, 10, size=(2, 2))
        if M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0] == 0:
            continue
        D, U, V = lat.smith_normal_form([list(map(int, r)) for r in M])
        UMV = np.array(U, dtype=object) @ M.astype(object) @ np.array(V, dtype=object)
        assert np.all(UMV == D), (M, D, UMV)
        assert D[0, 1] == 0 and D[1, 0] == 0
        assert D[0, 0] >= 0 and D[1, 1] >= 0
        assert D[0, 0] == 0 or int(D[1, 1]) % int(D[0, 0]) == 0
        for X in (U, V):
            d = int(X[0][0] * X[1][1] - X[0][1] * X[1][0])
            assert abs(d) == 1


def _in_lattice(AT, v):
    # v in A^T Z^2  <=>  (A^T)^{-1} v integer
    det = AT[0][0] * AT[1][1] - AT[0][1] * AT[1][0]
    x = AT[1][1] * v[0] - AT[0][1] * v[1]
    y = -AT[1][0] * v[0] + AT[0][0] * v[1]
    return x % det == 0 and y % det == 0


def test_coset_representatives():
    for lattice, m, n in [("square", 7, 1), ("square", 3, 2), ("honeycomb", 4, 3)]:
        A = lat.supercell_A(lattice, m, n)
        reps = lat.coset_representatives(A)
        assert len(reps) == lat.n_cells(A)
        AT = [[int(A[0, 0]), int(A[1, 0])], [int(A[0, 1]), int(A[1, 1])]]
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d = (reps[i][0] - reps[j][0], reps[i][1] - reps[j][1])
                assert not _in_lattice(AT, d), (reps[i], reps[j])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
