"""Exact commensurate moiré bilayer from finite-Fourier layers.

For a commensurate twist, both layers' reciprocal lattices are sublattices of the
supercell reciprocal lattice, so the moiré dielectric of finite-Fourier layers has an
EXACT finite Fourier series on the supercell lattice:

    layer-1 harmonic h  ->  supercell index  A^T h
    layer-2 harmonic h  ->  supercell index  A2^T h,   A2 = B0^{-1} R^{-1} B0 A

(A2 is integer exactly when the twist is commensurate; verified at build time).
This removes every sampling/rasterization error from the full moiré reference.
"""
import numpy as np

from . import lattice as lat


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def layer2_integer_matrix(lattice, m, n):
    """A2 with B_s = (R B0) A2; integer for commensurate (m, n)."""
    B0 = lat.monolayer_basis(lattice)
    A = np.asarray(lat.supercell_A(lattice, m, n), dtype=float)
    th = lat.twist_angle(lattice, m, n)
    A2 = np.linalg.inv(rotation(th) @ B0) @ (B0 @ A)
    A2i = np.rint(A2)
    assert np.max(np.abs(A2 - A2i)) < 1e-9, f"non-commensurate: {A2}"
    return A2i.astype(int)


def moire_coeffs(lattice, m, n, eps0, layer1, layer2):
    """Exact supercell Fourier coefficients {(m1, m2): c} of the twisted bilayer."""
    from .materials import _sym
    A = np.asarray(lat.supercell_A(lattice, m, n), dtype=float).astype(int)
    A2 = layer2_integer_matrix(lattice, m, n)
    out = {(0, 0): complex(eps0)}
    for (h1, h2), c in _sym(layer1).items():
        key = (A[0, 0] * h1 + A[1, 0] * h2, A[0, 1] * h1 + A[1, 1] * h2)
        out[key] = out.get(key, 0) + c
    for (h1, h2), c in _sym(layer2).items():
        key = (A2[0, 0] * h1 + A2[1, 0] * h2, A2[0, 1] * h1 + A2[1, 1] * h2)
        out[key] = out.get(key, 0) + c
    return out


def supercell_basis(lattice, m, n):
    B0 = lat.monolayer_basis(lattice)
    A = np.asarray(lat.supercell_A(lattice, m, n), dtype=float)
    return B0 @ A


def direct_eval(lattice, m, n, eps0, layer1, layer2, X, Y):
    """Direct real-space evaluation eps0 + layer1(x) + layer2(R_-theta-ish x) for
    verification. Layer 2 harmonics live on the rotated lattice: exp(2 pi i h . s2),
    s2 = (R B0)^{-1} x."""
    from .materials import _sym
    B0 = lat.monolayer_basis(lattice)
    th = lat.twist_angle(lattice, m, n)
    inv1 = np.linalg.inv(B0)
    inv2 = np.linalg.inv(rotation(th) @ B0)
    e = np.full(X.shape, complex(eps0))
    for (h1, h2), c in _sym(layer1).items():
        s1 = inv1[0, 0] * X + inv1[0, 1] * Y
        s2 = inv1[1, 0] * X + inv1[1, 1] * Y
        e = e + c * np.exp(2j * np.pi * (h1 * s1 + h2 * s2))
    for (h1, h2), c in _sym(layer2).items():
        s1 = inv2[0, 0] * X + inv2[0, 1] * Y
        s2 = inv2[1, 0] * X + inv2[1, 1] * Y
        e = e + c * np.exp(2j * np.pi * (h1 * s1 + h2 * s2))
    return e
