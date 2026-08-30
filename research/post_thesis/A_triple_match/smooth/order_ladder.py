#!/usr/bin/env python3
"""The order ladder of the envelope approximation, measured.

mass  — symbol-level closure at the carrier: the fixed-frame (raw) surface is an
        EXACTLY isotropic parabola, h11(kappa) = E1 + (u1^H R^2 u1) |kappa|^2
        (the linear term vanishes at the time-reversal-invariant M2), so every bit
        of the true band's ~9:1 mass anisotropy must come from the second-order
        k.p sum over remote bands:

            curv(e) = 2 u1^H R^2 u1 + 2 sum_r |<u_r| e.C' |u1>|^2 / (E1 - E_r),
            C'(e) = R diag(2 (k0+G).e) R.

        With the full sum this is an identity of the hermitized operator family;
        truncating the sum at a few remote bands IS the Lowdin order-2 model.
        This command tabulates exact / raw / raw+{nearest bands} per direction and
        the per-shell error prediction the truncation implies.

fold  — the product-space downfold at one angle (lowdin2 / feshbach models):
        see hierarchy_ladder.py; this command runs the fold models against the
        stored diagnosis references and prints the order table.
"""
import argparse
import sys

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, "/home/renlephy/msl/research/post_thesis")
import valley_diagnosis as vd
from lib_v5 import raw_projection as rp

B0 = vd.B0
GMAX = vd.GMAX_MONO
FINE = 128


def hermitized_at(k):
    h, R, ns, kG = rp.mono_hermitized(vd.avg_coeffs(), k, B0, GMAX, FINE)
    return h, R, kG


def mass_closure(nb_list=(1, 2, 6, 12), h_fd=2e-3):
    k0 = vd.M_CART["M2"]
    C0, R, kG = hermitized_at(k0)
    w, V = sla.eigh(C0)
    b = vd.cand.BAND
    u1 = V[:, b]
    E1 = w[b]
    Bs = vd.lf.supercell_basis(vd.LATTICE, 32, 31)
    Brec = 2 * np.pi * np.linalg.inv(np.asarray(Bs, float)).T
    dirs = {"light": Brec[:, 0] / np.linalg.norm(Brec[:, 0]),
            "heavy": Brec[:, 1] / np.linalg.norm(Brec[:, 1])}

    raw = float(np.real(u1.conj() @ (R @ (R @ u1))))   # u1^H R^2 u1
    print(f"raw (isotropic) curvature coefficient  u1^H R^2 u1 = {raw:.6f}\n")
    print(f"{'direction':>9s} {'exact(FD)':>12s} {'raw':>10s} "
          + " ".join(f"{'+nb=' + str(nb):>10s}" for nb in nb_list)
          + f" {'full sum':>10s}")
    out = {}
    for name, e in dirs.items():
        # exact curvature: finite difference of the band-1 eigenvalue of C(k)
        Cp, _, _ = hermitized_at(k0 + h_fd * e)
        Cm, _, _ = hermitized_at(k0 - h_fd * e)
        wp = sla.eigh(Cp, eigvals_only=True)[b]
        wm = sla.eigh(Cm, eigvals_only=True)[b]
        exact = (wp - 2 * E1 + wm) / h_fd ** 2

        # velocity operator along e, with a finite-difference cross-check
        d = 2.0 * (kG @ e)
        Cpr = R @ (d[:, None] * R)
        fd_check = np.linalg.norm((Cp - Cm) / (2 * h_fd) - Cpr) \
            / np.linalg.norm(Cpr)
        assert fd_check < 5e-4, fd_check

        me = V.conj().T @ (Cpr @ u1)                 # <u_r| e.C' |u1>
        gaps = E1 - w
        gaps[b] = np.inf
        terms = np.abs(me) ** 2 / gaps
        # remote bands ordered by |contribution|
        order = np.argsort(-np.abs(terms))
        cur = {"exact": 2 * exact / 2, "raw": 2 * raw}
        row = []
        for nb in nb_list:
            row.append(2 * raw + 2 * float(np.sum(terms[order[:nb]])))
        full = 2 * raw + 2 * float(np.sum(terms))
        print(f"{name:>9s} {2 * exact / 2:12.6f} {2 * raw:10.6f} "
              + " ".join(f"{x:10.6f}" for x in row) + f" {full:10.6f}")
        out[name] = dict(exact=2 * exact / 2, raw=2 * raw, partial=row,
                         full=full, top_terms=terms[order[:4]],
                         top_bands=order[:4], fd_check=fd_check)
    print("\nclosure |full - exact| / exact: "
          + ", ".join(f"{k}: {abs(v['full'] - v['exact']) / abs(v['exact']):.1e}"
                      for k, v in out.items()))
    for k, v in out.items():
        print(f"{k}: dominant remote bands {list(v['top_bands'])} with "
              f"curvature contributions "
              + " ".join(f"{2 * t:+.4f}" for t in v["top_terms"]))
    # per-shell prediction of the order-2 (nb=2) model at (32,31)
    print("\nper-shell |order-2 model - true| prediction at (32,31), heavy "
          "direction (f units):")
    e = dirs["heavy"]
    step = np.linalg.norm(Brec[:, 1])
    conv = 8 * np.pi ** 2 * np.sqrt(E1) / (2 * np.pi)
    me = V.conj().T @ ((R @ ((2.0 * (kG @ e))[:, None] * R)) @ u1)
    gaps = E1 - w
    gaps[b] = np.inf
    terms = np.abs(me) ** 2 / gaps
    order = np.argsort(-np.abs(terms))
    curv2 = 2 * raw + 2 * float(np.sum(terms[order[:2]]))
    for j in (1, 2):
        kap = j * step
        true_e = sla.eigh(hermitized_at(k0 + kap * e)[0],
                          eigvals_only=True)[b]
        model = E1 + 0.5 * curv2 * kap ** 2
        print(f"  shell n=(0,{j}): |model - true| = "
              f"{abs(model - true_e) / conv:.1e}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mass"])
    args = ap.parse_args()
    mass_closure()
