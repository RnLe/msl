"""Doubler-free envelope discretization on the periodic slow torus.

Kinetic form: diagonal terms as forward-difference flux  D+^H M_aa(mid) D+  (discrete
symbol (2-2cos)/ds^2: single zero at theta=0, checkerboard modes at ~4/ds^2), cross terms
as symmetrized centered differences (their Nyquist zeros are harmless because the diagonal
flux dominates there). Hermitian by construction for a Hermitian mass field; second-order
accurate; Bloch sector enters as a wrap phase. This replaces the V4 centered-first-
derivative-squared kinetic whose 2D symbol had four low-energy species (fermion doubling).

Layout: fields are (Ns1, Ns2, Nb, Nb) arrays over slow fractional coordinates [s1, s2]
(axis order is part of the archive contract, manifest.geometry.registry_axis_order).
Operators act on vectors ordered as (s1, s2, band) C-order flattened.
"""
import numpy as np
import scipy.sparse as sp


def _dplus_1d(N, ds, phase):
    """Forward difference with Bloch wrap: (f[i+1] - f[i])/ds, f[N] = phase*f[0]."""
    d = sp.lil_matrix((N, N), dtype=complex)
    for i in range(N):
        d[i, i] = -1.0 / ds
        d[i, (i + 1) % N] = (phase if i == N - 1 else 1.0) / ds
    return d.tocsr()


def _dcent_1d(N, ds, phase):
    """Centered difference with Bloch wrap: (f[i+1] - f[i-1])/(2 ds)."""
    d = sp.lil_matrix((N, N), dtype=complex)
    for i in range(N):
        d[i, (i + 1) % N] = (phase if i == N - 1 else 1.0) / (2 * ds)
        d[i, (i - 1) % N] = -(np.conj(phase) if i == 0 else 1.0) / (2 * ds)
    return d.tocsr()


def _block_diag_field(F, N1, N2, Nb):
    """Sparse block-diagonal operator from a (N1, N2, Nb, Nb) coefficient field."""
    n = N1 * N2
    if Nb == 1:
        return sp.diags(F.reshape(n).astype(complex))
    blocks = F.reshape(n, Nb, Nb).astype(complex)
    data = blocks.reshape(-1)
    rows = (np.repeat(np.arange(n) * Nb, Nb * Nb)
            + np.tile(np.repeat(np.arange(Nb), Nb), n))
    cols = (np.repeat(np.arange(n) * Nb, Nb * Nb)
            + np.tile(np.tile(np.arange(Nb), Nb), n))
    return sp.csr_matrix((data, (rows, cols)), shape=(n * Nb, n * Nb))


def _midpoint(F, axis):
    """Average a coefficient field onto the forward-difference midpoints along axis."""
    return 0.5 * (F + np.roll(F, -1, axis=axis))


def kinetic_operator(M11, M12, M21, M22, k_s=(0.0, 0.0), Nb=1):
    """Assemble  K = sum_ab D_a^H M_ab D_b  on the slow torus.

    M_ab: (N1, N2) scalar fields (Nb=1) or (N1, N2, Nb, Nb) block fields, in slow
    FRACTIONAL coordinates (any oblique metric is inside M_ab = B^{-1} M_cart B^{-T}).
    M21 must be the conjugate transpose field of M12 (checked). k_s: fractional Bloch
    sector (wrap phase e^{2 pi i k_a}).
    """
    M11 = np.asarray(M11)
    M12 = np.asarray(M12)
    M21 = np.asarray(M21)
    M22 = np.asarray(M22)
    if Nb == 1 and M11.ndim == 2:
        pass
    N1, N2 = M11.shape[:2]
    ds1, ds2 = 1.0 / N1, 1.0 / N2
    ph1 = np.exp(2j * np.pi * k_s[0])
    ph2 = np.exp(2j * np.pi * k_s[1])
    if Nb == 1:
        herm_defect = np.max(np.abs(M21 - np.conj(M12)))
    else:
        herm_defect = np.max(np.abs(M21 - np.conj(np.swapaxes(M12, -1, -2))))
    if herm_defect > 1e-12 * max(np.max(np.abs(M12)), 1e-300):
        raise ValueError(f"M21 != M12^H (defect {herm_defect:.2e}) — refuse to assemble")

    Ib = sp.eye(Nb, format="csr", dtype=complex)
    I1 = sp.eye(N1, format="csr", dtype=complex)
    I2 = sp.eye(N2, format="csr", dtype=complex)
    Dp1 = sp.kron(sp.kron(_dplus_1d(N1, ds1, ph1), I2), Ib, format="csr")
    Dp2 = sp.kron(sp.kron(I1, _dplus_1d(N2, ds2, ph2)), Ib, format="csr")
    Dc1 = sp.kron(sp.kron(_dcent_1d(N1, ds1, ph1), I2), Ib, format="csr")
    Dc2 = sp.kron(sp.kron(I1, _dcent_1d(N2, ds2, ph2)), Ib, format="csr")

    K = (Dp1.conj().T @ _block_diag_field(_midpoint(M11, 0), N1, N2, Nb) @ Dp1
         + Dp2.conj().T @ _block_diag_field(_midpoint(M22, 1), N1, N2, Nb) @ Dp2
         + Dc1.conj().T @ _block_diag_field(M12, N1, N2, Nb) @ Dc2
         + Dc2.conj().T @ _block_diag_field(M21, N1, N2, Nb) @ Dc1)
    return K.tocsr()


def constant_symbol(M11, M12, M21, M22, th1, th2, ds1, ds2):
    """Discrete symbol of the assembled kinetic for CONSTANT coefficients at grid
    momentum (th1, th2) (radians per grid step). Used by the full-BZ scan test."""
    f1 = (2 - 2 * np.cos(th1)) / ds1 ** 2
    f2 = (2 - 2 * np.cos(th2)) / ds2 ** 2
    c1 = np.sin(th1) / ds1
    c2 = np.sin(th2) / ds2
    return M11 * f1 + M22 * f2 + (M12 + M21) * c1 * c2


def scan_symbol_zeros(M11, M12, M21, M22, N1, N2, tol=1e-9):
    """Count zeros of the constant-coefficient discrete symbol over the FULL discrete
    Brillouin zone. A correct discretization has exactly one (theta = 0)."""
    ds1, ds2 = 1.0 / N1, 1.0 / N2
    th1 = 2 * np.pi * np.fft.fftfreq(N1)
    th2 = 2 * np.pi * np.fft.fftfreq(N2)
    T1, T2 = np.meshgrid(th1, th2, indexing="ij")
    S = constant_symbol(M11, M12, M21, M22, T1, T2, ds1, ds2)
    smax = np.max(np.abs(S))
    return int(np.sum(np.abs(S) < tol * smax)), S
