"""Smooth finite-Fourier materials: exact coefficients, exact registry derivatives,
no rasterization, identical input for every solver (dense PWE, matrix-free PWE, FDFD
sampling, Blaze external map).

A material is a dict of Fourier coefficients on the monolayer reciprocal lattice,
{(h1, h2): c} with eps(r) = sum_h c_h exp(2 pi i h . s(r)), s = fractional coords,
c_{-h} = conj(c_h) (real field). The bilayer registry family is
    eps_loc(s; delta) = layer1(s) + layer2(s + delta)  (+ background in the (0,0) term),
so registry derivatives are exact: d/d delta_a -> 2 pi i h_a on layer-2 coefficients.
"""
import numpy as np


def _sym(coeffs):
    """Ensure c_{-h} = conj(c_h) closure."""
    out = dict(coeffs)
    for (h1, h2), c in list(coeffs.items()):
        out[(-h1, -h2)] = np.conj(c)
    return out


def cosine_layer(amplitudes):
    """Layer from {(h1,h2): real amplitude}: a*cos(2 pi h.s) => c_h = c_{-h} = a/2."""
    out = {}
    for (h1, h2), a in amplitudes.items():
        out[(h1, h2)] = out.get((h1, h2), 0) + a / 2.0
        out[(-h1, -h2)] = out.get((-h1, -h2), 0) + a / 2.0
    return out


def bilayer(eps0, layer1, layer2, delta=(0.0, 0.0)):
    """Combined coefficient dict at registry delta (fractional)."""
    out = {(0, 0): complex(eps0)}
    for h, c in _sym(layer1).items():
        out[h] = out.get(h, 0) + c
    for (h1, h2), c in _sym(layer2).items():
        ph = np.exp(2j * np.pi * (h1 * delta[0] + h2 * delta[1]))
        out[(h1, h2)] = out.get((h1, h2), 0) + c * ph
    return out


def d_delta(layer2, delta, order=(1, 0)):
    """Exact registry-derivative coefficients of the layer-2 part:
    d^{o1+o2} / d delta_1^{o1} d delta_2^{o2}."""
    out = {}
    for (h1, h2), c in _sym(layer2).items():
        ph = np.exp(2j * np.pi * (h1 * delta[0] + h2 * delta[1]))
        out[(h1, h2)] = c * ph * (2j * np.pi * h1) ** order[0] \
            * (2j * np.pi * h2) ** order[1]
    return out


def sample(coeffs, N1, N2):
    """Exact real-space samples on the fractional grid s = (i/N1, j/N2), [s1, s2] axes."""
    s1 = np.arange(N1) / N1
    s2 = np.arange(N2) / N2
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    e = np.zeros((N1, N2), complex)
    for (h1, h2), c in coeffs.items():
        e += c * np.exp(2j * np.pi * (h1 * S1 + h2 * S2))
    assert np.max(np.abs(e.imag)) < 1e-12 * max(np.max(np.abs(e.real)), 1e-300), \
        "coefficients not conjugate-symmetric"
    return e.real


def min_bound(coeffs):
    """Rigorous lower bound: eps(r) >= c_00 - sum_{h!=0} |c_h|."""
    return float(np.real(coeffs[(0, 0)])
                 - sum(abs(c) for h, c in coeffs.items() if h != (0, 0)))
