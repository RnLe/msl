"""Exact oracles and comparison metrics for generalized pencils A x = lambda B x, B > 0.

These are the acceptance instruments of the v5 validation program:
  - ritz_pencil:      the best Galerkin spectrum in a given lifted trial space
  - feshbach:         the exact energy-dependent downfold (adjudicates Lowdin forms)
  - target_weight:    B-metric weight of a reference vector in the trial space
  - principal_angles: subspace identity between reference and model clusters
  - lifted_residual:  certificate that a model eigenpair approximates a true one
  - inertia_count:    certified number of pencil eigenvalues in an interval

No Hungarian matching anywhere. All complex128.
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def ritz_pencil(A, B, T):
    """Galerkin projection onto span(T): solve (T^H A T) c = lambda (T^H B T) c.
    Returns (eigenvalues, coefficient vectors, lifted vectors)."""
    T = np.asarray(T)
    AT = T.conj().T @ (A @ T)
    BT = T.conj().T @ (B @ T)
    AT = 0.5 * (AT + AT.conj().T)
    BT = 0.5 * (BT + BT.conj().T)
    w, c = sla.eigh(AT, BT)
    return w, c, T @ c


def feshbach(C, P_idx, z):
    """Exact effective operator on the P block of a WHITENED matrix C (= B^{-1/2} A
    B^{-1/2}): C_eff(z) = C_PP + C_PQ (z - C_QQ)^{-1} C_QP."""
    n = C.shape[0]
    P = np.asarray(P_idx)
    Q = np.setdiff1d(np.arange(n), P)
    CPP = C[np.ix_(P, P)]
    CPQ = C[np.ix_(P, Q)]
    CQP = C[np.ix_(Q, P)]
    CQQ = C[np.ix_(Q, Q)]
    return CPP + CPQ @ np.linalg.solve(z * np.eye(len(Q)) - CQQ, CQP)


def whiten(A, B):
    """C = B^{-1/2} A B^{-1/2} (dense; small systems only)."""
    w, V = sla.eigh(0.5 * (B + B.conj().T))
    assert w.min() > 0, "B not positive definite"
    Bmh = (V * (w ** -0.5)) @ V.conj().T
    C = Bmh @ A @ Bmh
    return 0.5 * (C + C.conj().T)


def b_orthonormalize(X, B):
    """B-orthonormal basis of span(X) (thin, rank-revealing)."""
    G = X.conj().T @ (B @ X)
    G = 0.5 * (G + G.conj().T)
    w, V = sla.eigh(G)
    keep = w > 1e-12 * w.max()
    return X @ (V[:, keep] / np.sqrt(w[keep]))


def target_weight(x, T, B):
    """B-metric weight of vector x (B-normalized) inside span(T)."""
    Tb = b_orthonormalize(T, B)
    x = x / np.sqrt(np.real(np.vdot(x, B @ x)))
    c = Tb.conj().T @ (B @ x)
    return float(np.real(np.vdot(c, c)))


def principal_angles(X, Y, B):
    """Principal-angle cosines between B-orthonormalized span(X) and span(Y),
    plus recall (coverage of X by Y) and precision (coverage of Y by X)."""
    Xb = b_orthonormalize(X, B)
    Yb = b_orthonormalize(Y, B)
    O = Xb.conj().T @ (B @ Yb)
    s = sla.svdvals(O)
    I = float(np.sum(s ** 2))
    return s, I / Xb.shape[1], I / Yb.shape[1]


def lifted_residual(A, B, y, lam):
    """Certified distance bound: dist(lam, spec(B^{-1/2}AB^{-1/2}))
    <= ||B^{-1/2} r|| / ||B^{1/2} y||, r = Ay - lam By.
    B may be sparse; solves B z = r iteratively if needed."""
    r = A @ y - lam * (B @ y)
    if sp.issparse(B):
        z, info = spla.cg(B, r, rtol=1e-12, maxiter=5000)
        assert info == 0
        num = np.sqrt(np.real(np.vdot(r, z)))
        den = np.sqrt(np.real(np.vdot(y, B @ y)))
    else:
        w, V = sla.eigh(0.5 * (B + B.conj().T))
        Bmh = (V * (w ** -0.5)) @ V.conj().T
        Bh = (V * (w ** 0.5)) @ V.conj().T
        num = np.linalg.norm(Bmh @ r)
        den = np.linalg.norm(Bh @ y)
    return num / den


def inertia_count(A, B, a, b):
    """Certified count of pencil eigenvalues in [a, b):
    N = n_-(A - b B) - n_-(A - a B) via LDL^H inertia (dense or sparse-LU signs)."""

    def n_neg(M):
        if sp.issparse(M):
            M = M.toarray()
        w = sla.eigvalsh(0.5 * (M + M.conj().T))
        return int(np.sum(w < 0))

    return n_neg(A - b * B) - n_neg(A - a * B)
