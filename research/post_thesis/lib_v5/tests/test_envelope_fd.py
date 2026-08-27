"""Doubler and symbol tests for the v5 envelope kinetic (audit claim M2)."""
import os
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import envelope_fd as efd  # noqa: E402

N = 16
M11c, M12c, M22c = 2.0, 0.3, 1.2


def _const_fields(m12=M12c):
    M11 = np.full((N, N), M11c)
    M22 = np.full((N, N), M22c)
    M12 = np.full((N, N), m12, dtype=complex)
    M21 = np.conj(M12)
    return M11, M12, M21, M22


def test_single_zero_no_doublers():
    nz, S = efd.scan_symbol_zeros(M11c, M12c, M12c, M22c, N, N)
    assert nz == 1
    # checkerboard/Nyquist modes carry large energy
    assert S[N // 2, 0] > 0.5 * 4 * M11c * N ** 2
    assert S[N // 2, N // 2] > 0.5 * 4 * (M11c + M22c) * N ** 2


def test_v4_centered_square_has_four_zeros():
    # the defective legacy symbol: (sin theta / ds)^2 products
    th = 2 * np.pi * np.fft.fftfreq(N)
    T1, T2 = np.meshgrid(th, th, indexing="ij")
    ds = 1.0 / N
    S_bad = (M11c * (np.sin(T1) / ds) ** 2 + M22c * (np.sin(T2) / ds) ** 2
             + 2 * M12c * (np.sin(T1) / ds) * (np.sin(T2) / ds))
    nz = np.sum(np.abs(S_bad) < 1e-9 * np.max(np.abs(S_bad)))
    assert nz == 4  # the fermion-doubling mechanism this module removes


def test_assembled_matches_symbol_on_plane_waves():
    K = efd.kinetic_operator(*_const_fields(), k_s=(0.0, 0.0), Nb=1).toarray()
    assert np.max(np.abs(K - K.conj().T)) < 1e-12
    s = np.arange(N) / N
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    ds = 1.0 / N
    for n1, n2 in [(0, 0), (1, 0), (0, 1), (3, 2), (N // 2, 0), (N // 2, N // 2),
                   (5, N - 3)]:
        pw = np.exp(2j * np.pi * (n1 * S1 + n2 * S2)).reshape(-1)
        th1, th2 = 2 * np.pi * n1 / N, 2 * np.pi * n2 / N
        pred = efd.constant_symbol(M11c, M12c, M12c, M22c,
                                   th1, th2, ds, ds)
        got = np.vdot(pw, K @ pw) / np.vdot(pw, pw)
        assert abs(got - pred) < 1e-9 * max(abs(pred), 1.0), (n1, n2, got, pred)


def test_second_order_accuracy():
    # smallest nonzero momentum: symbol -> continuum with O(ds^2) error
    errs = []
    for Ngrid in (16, 32, 64):
        ds = 1.0 / Ngrid
        th = 2 * np.pi / Ngrid
        cont = M11c * (2 * np.pi) ** 2
        disc = efd.constant_symbol(M11c, 0, 0, M22c, th, 0, ds, ds)
        errs.append(abs(disc - cont) / cont)
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert min(order) > 1.9, (errs, order)


def test_bloch_sector_shifts_symbol():
    ks = (0.21, -0.13)
    K = efd.kinetic_operator(*_const_fields(m12=0.0), k_s=ks, Nb=1).toarray()
    assert np.max(np.abs(K - K.conj().T)) < 1e-10
    w = np.linalg.eigvalsh(K)
    ds = 1.0 / N
    th = 2 * np.pi * (np.fft.fftfreq(N) * N + 0) / N
    pred = sorted(
        efd.constant_symbol(M11c, 0, 0, M22c,
                            2 * np.pi * (n1 + ks[0]) / N * 1, 0, ds, ds).real
        for n1 in range(N) for _ in (0,))
    # spot-check the bottom eigenvalue against the analytic shifted symbol minimum
    grid1 = 2 * np.pi * (np.arange(N) + ks[0]) / N / ds * 0  # placeholder
    mins = []
    for n1 in range(N):
        for n2 in range(N):
            t1 = 2 * np.pi * (n1 + ks[0]) / N
            t2 = 2 * np.pi * (n2 + ks[1]) / N
            mins.append(efd.constant_symbol(M11c, 0, 0, M22c, t1, t2, ds, ds).real)
    assert abs(w[0] - min(mins)) < 1e-8 * max(abs(min(mins)), 1.0)


def test_variable_coefficients_hermitian_and_positive():
    rng = np.random.default_rng(3)
    s = np.arange(N) / N
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    M11 = 2.0 + 0.5 * np.cos(2 * np.pi * S1)
    M22 = 1.5 + 0.4 * np.sin(2 * np.pi * S2)
    M12 = 0.2 * np.exp(2j * np.pi * (S1 - S2)) * 0.5
    M21 = np.conj(M12)
    K = efd.kinetic_operator(M11, M12, M21, M22, k_s=(0.1, 0.2), Nb=1).toarray()
    assert np.max(np.abs(K - K.conj().T)) < 1e-11
    w = np.linalg.eigvalsh(K)
    assert w[0] > -1e-9 * abs(w[-1])  # ellipticity preserved (M field PD here)


def test_m21_mismatch_refused():
    M11, M12, M21, M22 = _const_fields()
    try:
        efd.kinetic_operator(M11, M12, M21 + 0.1, M22)
        raise AssertionError("non-Hermitian mass field accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
