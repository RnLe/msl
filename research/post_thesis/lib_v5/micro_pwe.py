"""Dense plane-wave TM pencil for smooth finite-Fourier materials — the local-crystal
reference oracle.

TM generalized pencil in the plane-wave basis {G = 2 pi B^{-T} n, |n|_inf <= gmax}:
    A_nn' = |k + G_n|^2 delta_nn',   B_nn' = eps_hat(n - n'),
    A u = lambda B u,  lambda = (omega a / c)^2 = (2 pi f)^2.
Eigenvectors are B-orthonormalized (the epsilon metric). Hellmann-Feynman velocity:
    v^(i)_mn = <u_m| dA/dk_i |u_n> = <u_m| diag(2 (k+G)_i) |u_n>   (exact for the pencil).
All complex128, all exact coefficients — no grids, no sampling error.
"""
import numpy as np
import scipy.linalg as sla


def pw_set(gmax):
    return [(n1, n2) for n1 in range(-gmax, gmax + 1) for n2 in range(-gmax, gmax + 1)]


def tm_pencil(coeffs, k_cart, B0, gmax):
    ns = pw_set(gmax)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(B0, float)).T
    G = np.array([Brec @ np.array(n, float) for n in ns])
    kG = G + np.asarray(k_cart, float)[None, :]
    A = np.diag((kG ** 2).sum(1)).astype(complex)
    N = len(ns)
    Bm = np.zeros((N, N), complex)
    for i, ni in enumerate(ns):
        for j, nj in enumerate(ns):
            d = (ni[0] - nj[0], ni[1] - nj[1])
            Bm[i, j] = coeffs.get(d, 0.0)
    Bm = 0.5 * (Bm + Bm.conj().T)
    return A, Bm, kG, ns


def solve(coeffs, k_cart, B0, gmax, n_bands=None):
    A, Bm, kG, ns = tm_pencil(coeffs, k_cart, B0, gmax)
    w, V = sla.eigh(A, Bm)
    if n_bands:
        w, V = w[:n_bands], V[:, :n_bands]
    # B-orthonormality is guaranteed by eigh(A, B); enforce phase convention: largest
    # coefficient real positive (deterministic gauge for tests)
    for j in range(V.shape[1]):
        i0 = np.argmax(np.abs(V[:, j]))
        V[:, j] *= np.exp(-1j * np.angle(V[i0, j]))
    return w, V, kG, ns


def velocity_hf(V, kG, i):
    """Exact pencil velocity matrix <u_m| diag(2 (k+G)_i) |u_n> (plain product — the
    dA/dk operator is diagonal in the plane-wave basis)."""
    d = 2.0 * kG[:, i]
    return V.conj().T @ (d[:, None] * V)
