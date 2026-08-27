"""Lifted-basis Ritz oracle for smooth commensurate moiré bilayers.

The trial space of any envelope model (frozen-Bloch, registry frames, exact window) is
lifted EXACTLY into the full supercell plane-wave pencil built from lifted.moire_coeffs:
monolayer harmonic h of the layer-1 frame maps to supercell index A^T h; an envelope
harmonic e shifts it by e; the carrier offset n0 accounts for the folded sector,

    k0 = k_sc + 2 pi B_s^{-T} n0,   n0 = A^T kappa0 - fold(A, kappa0)   (exact integers).

ritz(A_full, B_full, T) is then the best possible spectrum in that trial space — the
model-independent upper oracle of the validation program (report section 9.1).
"""
from fractions import Fraction

import numpy as np

from . import lattice as lat
from . import lifted as lf
from . import micro_pwe as mp


def sector_pencil(lattice, m, n, eps0, layer1, layer2, kappa0, gmax_sc):
    """Full supercell TM pencil at the folded sector of carrier kappa0 (kappa_env = 0).
    Returns dict with A, B, index lookup, bases, k vectors, and the integer carrier
    offset n0."""
    A_int = lat.supercell_A(lattice, m, n)
    Bs = lf.supercell_basis(lattice, m, n)
    ks_frac = lat.fold_sector(A_int, kappa0)
    k_sc = lat.sector_to_cartesian(Bs, ks_frac)
    coeffs = lf.moire_coeffs(lattice, m, n, eps0, layer1, layer2)
    Afull, Bfull, kG, ns = mp.tm_pencil(coeffs, k_sc, Bs, gmax_sc)
    lookup = {tuple(nv): i for i, nv in enumerate(ns)}
    # exact integer carrier offset
    k0f = [Fraction(x) for x in kappa0]
    n0 = (int(A_int[0, 0] * k0f[0] + A_int[1, 0] * k0f[1] - ks_frac[0]),
          int(A_int[0, 1] * k0f[0] + A_int[1, 1] * k0f[1] - ks_frac[1]))
    return {
        "lattice": lattice, "m": m, "n": n, "A_int": A_int, "B_super": Bs,
        "kappa_s": ks_frac, "k_sc_cart": k_sc, "coeffs": coeffs,
        "A": Afull, "B": Bfull, "kG": kG, "ns": ns, "lookup": lookup, "n0": n0,
    }


def lift_bloch_columns(pen, mono_coeffs_at_delta, kappa0, gmax_mono, band_ids,
                       env_list, B0):
    """Trial columns: frozen-registry monolayer Bloch states (bands band_ids at carrier
    kappa0, registry inside mono_coeffs_at_delta) times envelope plane waves env_list.
    Columns lie exactly in the pencil's Bloch sector."""
    k0_cart = 2 * np.pi * np.linalg.inv(np.asarray(B0, float)).T @ np.array(
        [float(kappa0[0]), float(kappa0[1])])
    w, V, kGm, ns_m = mp.solve(mono_coeffs_at_delta, k0_cart, B0, gmax_mono)
    A_int = pen["A_int"]
    n0 = pen["n0"]
    cols = []
    labels = []
    for b in band_ids:
        for e in env_list:
            col = np.zeros(len(pen["ns"]), complex)
            ok = True
            for i_h, h in enumerate(ns_m):
                key = (int(A_int[0, 0]) * h[0] + int(A_int[1, 0]) * h[1]
                       + n0[0] + e[0],
                       int(A_int[0, 1]) * h[0] + int(A_int[1, 1]) * h[1]
                       + n0[1] + e[1])
                j = pen["lookup"].get(key)
                if j is None:
                    if abs(V[i_h, b]) > 1e-8:
                        ok = False  # window clipped a significant coefficient
                    continue
                col[j] = V[i_h, b]
            if ok and np.linalg.norm(col) > 0:
                cols.append(col)
                labels.append((b, e))
    return np.array(cols).T, labels, w
