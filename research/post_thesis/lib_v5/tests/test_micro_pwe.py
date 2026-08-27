"""Material + dense micro-PWE oracle tests: homogeneous limit, HF velocity vs finite
differences at generic k, exact registry derivatives."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib_v5 import materials as mat  # noqa: E402
from lib_v5 import micro_pwe as mp  # noqa: E402

B0 = np.array([[1.0, 0.0], [0.0, 1.0]])
LAYER1 = mat.cosine_layer({(1, 0): 0.6, (0, 1): 0.6})
LAYER2 = mat.cosine_layer({(1, 0): 0.25, (0, 1): 0.25, (1, 1): 0.1})
EPS0 = 4.0
K_GEN = 2 * np.pi * np.array([0.137, 0.291])   # generic non-symmetry point


def test_material_positive_and_sampled():
    c = mat.bilayer(EPS0, LAYER1, LAYER2, delta=(0.23, 0.71))
    assert mat.min_bound(c) > 1.5
    e = mat.sample(c, 24, 24)
    assert e.min() > 1.0 and abs(e.mean() - EPS0) < 1e-12


def test_homogeneous_limit():
    c = {(0, 0): complex(EPS0)}
    w, V, kG, ns = mp.solve(c, K_GEN, B0, gmax=3)
    pred = np.sort((kG ** 2).sum(1) / EPS0)
    assert np.max(np.abs(np.sort(w) - pred)) < 1e-10


def test_registry_derivative_exact_vs_fd():
    d0 = (0.3, 0.4)
    h = 1e-6
    exact = mat.d_delta(LAYER2, d0, order=(1, 0))
    for key in [(1, 0), (1, 1), (-1, 0)]:
        cp = mat.bilayer(0, {}, LAYER2, delta=(d0[0] + h, d0[1])).get(key, 0)
        cm = mat.bilayer(0, {}, LAYER2, delta=(d0[0] - h, d0[1])).get(key, 0)
        fd = (cp - cm) / (2 * h)
        assert abs(fd - exact.get(key, 0)) < 1e-6 * max(abs(fd), 1.0)


def test_hf_velocity_vs_eigenvalue_fd():
    c = mat.bilayer(EPS0, LAYER1, LAYER2, delta=(0.23, 0.71))
    gmax = 4
    w0, V0, kG0, _ = mp.solve(c, K_GEN, B0, gmax)
    v = mp.velocity_hf(V0, kG0, 0)
    # diagonal HF = d lambda / d k_x at a generic (nondegenerate) k
    h = 1e-5
    wp, _, _, _ = mp.solve(c, K_GEN + np.array([h, 0]), B0, gmax)
    wm, _, _, _ = mp.solve(c, K_GEN - np.array([h, 0]), B0, gmax)
    fd = (wp - wm) / (2 * h)
    for band in range(5):
        gap = min(abs(w0[band] - w0[j]) for j in range(len(w0)) if j != band)
        if gap < 1e-3:
            continue
        assert abs(np.real(v[band, band]) - fd[band]) < 5e-5 * max(abs(fd[band]), 1.0), \
            (band, v[band, band], fd[band])


def test_pencil_hermitian_and_metric():
    c = mat.bilayer(EPS0, LAYER1, LAYER2, delta=(0.1, 0.9))
    A, Bm, kG, ns = mp.tm_pencil(c, K_GEN, B0, 3)
    assert np.max(np.abs(Bm - Bm.conj().T)) < 1e-14
    w, V, _, _ = mp.solve(c, K_GEN, B0, 3)
    G = V.conj().T @ Bm @ V
    assert np.max(np.abs(G - np.eye(G.shape[0]))) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
