"""Exact lattice, commensuration, and Bloch-sector algebra (integer/rational, no floats
in any identity that can be exact).

Conventions (serialized with every result; see manifest.py):
  - monolayer direct basis B0 = [a1 | a2] (columns), a = 1
  - supercell integer matrix A: B_s = B0 @ A (columns of A = supercell vectors in the
    primitive basis)
  - reciprocal-fractional coordinates kappa: k_cartesian = 2*pi * B^{-T} @ kappa
  - sector fold: a plane wave at primitive kappa0 with envelope sector kappa_env lives at
    the physical supercell sector  kappa_s = A^T kappa0 + kappa_env  (mod 1)

Fixes over the V3/V4 helpers: the hex K point is the actual BZ corner star (permutations
of (2/3,1/3)); the supercell/moire area ratio is |det A| computed, never asserted; coset
representatives come from the Smith normal form of A^T, valid for any commensuration.
"""
from fractions import Fraction

import numpy as np

SQ3 = np.sqrt(3.0)


def monolayer_basis(lattice):
    if lattice == "square":
        return np.array([[1.0, 0.0], [0.0, 1.0]])
    if lattice in ("hex", "triangular", "honeycomb"):
        return np.array([[1.0, 0.5], [0.0, SQ3 / 2.0]])
    raise ValueError(lattice)


def supercell_A(lattice, m, n):
    """Integer coincidence matrix (columns = supercell vectors in the primitive basis),
    matching the historical construction (commensurate_utils.build_supercell_vectors)."""
    if lattice == "square":
        return np.array([[m, -n], [n, m]], dtype=object)
    if lattice in ("hex", "triangular", "honeycomb"):
        return np.array([[n, -m], [m, n + m]], dtype=object)
    raise ValueError(lattice)


def n_cells(A):
    A = np.asarray(A, dtype=object)
    return abs(int(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]))


def twist_angle(lattice, m, n):
    if lattice == "square":
        return 2.0 * np.arctan2(n, m)
    N = m * m + m * n + n * n
    return float(np.arccos(np.clip((m * m + 4 * m * n + n * n) / (2.0 * N), -1, 1)))


def k_corner_frac(lattice):
    """The BZ corner star in reciprocal-fractional coordinates (exact)."""
    if lattice == "square":
        return [(Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 2), Fraction(-1, 2))]
    if lattice in ("hex", "triangular", "honeycomb"):
        K = (Fraction(2, 3), Fraction(1, 3))
        Kp = (Fraction(1, 3), Fraction(2, 3))
        return [K, Kp, (-K[0], -K[1]), (-Kp[0], -Kp[1]),
                (K[0] - 1, K[1]), (Kp[0], Kp[1] - 1)]
    raise ValueError(lattice)


def fold_sector(A, kappa0, kappa_env=(0, 0)):
    """kappa_s = A^T kappa0 + kappa_env (mod 1), exact. kappa* as Fraction pairs."""
    A = np.asarray(A, dtype=object)
    k0 = [Fraction(x) for x in kappa0]
    ke = [Fraction(x) for x in kappa_env]
    ks1 = A[0, 0] * k0[0] + A[1, 0] * k0[1] + ke[0]
    ks2 = A[0, 1] * k0[0] + A[1, 1] * k0[1] + ke[1]
    return (ks1 % 1, ks2 % 1)


def sector_to_cartesian(B_super, kappa_s):
    """Physical Bloch vector q = 2*pi * B_s^{-T} kappa_s."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(B_super, float)).T @ np.array(
        [float(kappa_s[0]), float(kappa_s[1])])


def smith_normal_form(M):
    """SNF of an integer 2x2 matrix: U M V = D with U, V unimodular, D = diag(d1, d2),
    d1 | d2. Exact integer algorithm."""
    M = [[int(M[0][0]), int(M[0][1])], [int(M[1][0]), int(M[1][1])]]
    U = [[1, 0], [0, 1]]
    V = [[1, 0], [0, 1]]

    def swap_rows(X, i, j):
        X[i], X[j] = X[j], X[i]

    def addmul_row(X, i, j, c):
        X[i][0] += c * X[j][0]
        X[i][1] += c * X[j][1]

    def swap_cols(X, i, j):
        for r in X:
            r[i], r[j] = r[j], r[i]

    def addmul_col(X, i, j, c):
        for r in X:
            r[i] += c * r[j]

    for _ in range(64):
        if M[0][1] == 0 and M[1][0] == 0:
            break
        # move smallest nonzero to (0,0)
        entries = [(abs(M[i][j]), i, j) for i in range(2) for j in range(2)
                   if M[i][j] != 0]
        _, i0, j0 = min(entries)
        if i0 == 1:
            swap_rows(M, 0, 1)
            swap_rows(U, 0, 1)
        if j0 == 1:
            swap_cols(M, 0, 1)
            swap_cols(V, 0, 1)
        # reduce
        if M[1][0] != 0:
            c = -(M[1][0] // M[0][0])
            addmul_row(M, 1, 0, c)
            addmul_row(U, 1, 0, c)
            continue
        if M[0][1] != 0:
            c = -(M[0][1] // M[0][0])
            addmul_col(M, 1, 0, c)
            addmul_col(V, 1, 0, c)
            continue
    if M[0][0] != 0 and M[1][1] % M[0][0] != 0:
        addmul_col(M, 0, 1, 1)
        addmul_col(V, 0, 1, 1)
        return smith_normal_form_from(M, U, V)
    return _snf_normalize(M, U, V)


def smith_normal_form_from(M, U, V):
    # continue elimination after the divisibility fix-up
    for _ in range(64):
        if M[0][1] == 0 and M[1][0] == 0:
            break
        if M[1][0] != 0 and M[0][0] != 0:
            c = -(M[1][0] // M[0][0])
            for k in range(2):
                M[1][k] += c * M[0][k]
                U[1][k] += c * U[0][k]
            if M[1][0] != 0:
                M[0], M[1] = M[1], M[0]
                U[0], U[1] = U[1], U[0]
            continue
        if M[0][1] != 0 and M[0][0] != 0:
            c = -(M[0][1] // M[0][0])
            for r in (M, V):
                r[0][1] += c * r[0][0]
                r[1][1] += c * r[1][0]
            if M[0][1] != 0:
                for r in (M, V):
                    r[0][0], r[0][1] = r[0][1], r[0][0]
                    r[1][0], r[1][1] = r[1][1], r[1][0]
            continue
    return _snf_normalize(M, U, V)


def _snf_normalize(M, U, V):
    # sign and divisibility normalization
    if M[0][0] < 0:
        M[0][0] *= -1
        U[0][0] *= -1
        U[0][1] *= -1
    if M[1][1] < 0:
        M[1][1] *= -1
        U[1][0] *= -1
        U[1][1] *= -1
    return (np.array(M, dtype=object), np.array(U, dtype=object),
            np.array(V, dtype=object))


def coset_representatives(A):
    """Representatives of Z^2 / A^T Z^2 (the reciprocal folding cosets), |det A| of them,
    via the Smith normal form of A^T."""
    A = np.asarray(A, dtype=object)
    AT = [[int(A[0, 0]), int(A[1, 0])], [int(A[0, 1]), int(A[1, 1])]]
    D, U, V = smith_normal_form([row[:] for row in AT])
    d1, d2 = int(D[0, 0]), int(D[1, 1])
    # lattice A^T Z^2 = U^{-1} D V^{-1} Z^2; cosets of D Z^2 are (i, j), i<d1, j<d2,
    # mapped back through U^{-1}
    Uinv = np.array([[U[1][1], -U[0][1]], [-U[1][0], U[0][0]]], dtype=object)
    detU = U[0][0] * U[1][1] - U[0][1] * U[1][0]
    assert abs(int(detU)) == 1
    if int(detU) == -1:
        Uinv = -Uinv
    reps = []
    for i in range(d1):
        for j in range(d2):
            v = Uinv @ np.array([i, j], dtype=object)
            reps.append((int(v[0]), int(v[1])))
    assert len(reps) == n_cells(A)
    return reps
