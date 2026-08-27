"""Oracle self-tests on random Hermitian pencils, plus injected-defect detection."""
import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import oracles as oc  # noqa: E402

rng = np.random.default_rng(7)
N = 40


def _pencil():
    X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    A = 0.5 * (X + X.conj().T)
    Y = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B = Y @ Y.conj().T / N + np.eye(N)
    return A, B


def test_ritz_full_basis_is_exact():
    A, B = _pencil()
    w_exact = sla.eigh(A, B, eigvals_only=True)
    w, _, _ = oc.ritz_pencil(A, B, np.eye(N, dtype=complex))
    assert np.max(np.abs(w - w_exact)) < 1e-10


def test_ritz_is_variational():
    A, B = _pencil()
    w_exact = sla.eigh(A, B, eigvals_only=True)
    T = rng.standard_normal((N, 12)) + 1j * rng.standard_normal((N, 12))
    w, _, _ = oc.ritz_pencil(A, B, T)
    # Cauchy interlacing: ritz value i >= exact value i
    assert np.all(w - w_exact[: len(w)] >= -1e-10)


def test_feshbach_reproduces_eigenvalues():
    A, B = _pencil()
    C = oc.whiten(A, B)
    w = sla.eigvalsh(C)
    lam = w[N // 2]
    P = np.arange(5)
    Ceff = oc.feshbach(C, P, lam)
    d = sla.eigvals(Ceff) - lam
    assert np.min(np.abs(d)) < 1e-8


def test_principal_angles_identity_and_disjoint():
    A, B = _pencil()
    w, V = sla.eigh(A, B)
    X = V[:, :5]
    s, recall, precision = oc.principal_angles(X, X.copy(), B)
    assert np.allclose(s, 1, atol=1e-10) and abs(recall - 1) < 1e-10
    Y = V[:, 5:10]  # B-orthogonal complement slice
    s2, r2, p2 = oc.principal_angles(X, Y, B)
    assert np.max(s2) < 1e-10 and r2 < 1e-10


def test_lifted_residual_certificate():
    A, B = _pencil()
    w, V = sla.eigh(A, B)
    assert oc.lifted_residual(A, B, V[:, 3], w[3]) < 1e-10
    y = V[:, 3] + 0.05 * V[:, 4]
    bound = oc.lifted_residual(A, B, y, w[3])
    assert bound > 1e-4  # a polluted vector cannot certify tightly


def test_inertia_count():
    A, B = _pencil()
    w = sla.eigh(A, B, eigvals_only=True)
    a, b = w[10] - 1e-9, w[20] - 1e-9
    assert oc.inertia_count(A, B, a, b) == 10


def test_detects_injected_defects():
    A, B = _pencil()
    w, V = sla.eigh(A, B)
    ref = V[:, :6]
    # injected count error: model cluster missing one state
    model = V[:, :5]
    _, recall, precision = oc.principal_angles(ref, model, B)
    assert recall < 0.9 and precision > 0.99
    # injected surplus: an extra unrelated state
    model2 = np.column_stack([V[:, :6], V[:, 30]])
    _, r2, p2 = oc.principal_angles(ref, model2, B)
    assert p2 < 0.9 and r2 > 0.99
    # injected eigenvalue shift caught by the certificate
    assert oc.lifted_residual(A, B, V[:, 2], w[2] + 1e-3) > 1e-4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
