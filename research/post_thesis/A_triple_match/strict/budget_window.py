#!/usr/bin/env python3
"""Section 16.3 excluded-weight budget for candidate PWE windows, from the measured
FDFD ground state (stage0b_characters.npz, centered cell, 2 deg).

For a candidate shell list {r_0..r_g} (envelope radius per |g|_inf monolayer shell, units
of |b_prim|), the window is the union over BOTH valleys' star systems of
{star + o : |o . bc| <= r_shell}; the first-order variational budget is
    Dlambda ~ sum_{k not in W} p_k (|k|^2 - lambda),   p_k = |u_hat(k)|^2 normalized,
    Df = Dlambda / (8 pi^2 f0).
Anchors (section 16.3 table): g<=2@12 -> 7.9e-3 | g<=3@12 -> 1.7e-3 |
{18,18,18,12,6} -> 1.2e-3 | {24,24,24,16,8} -> 7.1e-4. Measured/budget ratio ~0.65.

Usage: budget_window.py --shells 18,18,18,12,6 [--shells 24,24,24,16,8 ...]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = 57
X = np.array([np.pi, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shells", action="append", required=True,
                    help="comma list renv per |g|_inf shell; repeatable")
    args = ap.parse_args()

    d = np.load(os.path.join(HERE, "stage0b_characters.npz"))
    Nc = int(d["N"])
    gf = d["ground_fields"]
    epsc = d["eps"]
    f0 = float(np.asarray(d["freqs"]).ravel()[0])
    lam = (2 * np.pi * f0) ** 2
    Bc = np.array([[float(M), -1.0], [1.0, float(M)]])
    bc = 2 * np.pi * np.linalg.inv(Bc).T
    bnorm = np.linalg.norm(2 * np.pi * np.linalg.inv(
        np.array([[(M - 1) / 2., (M + 1) / 2.], [-(M + 1) / 2., (M - 1) / 2.]])).T[:, 0])
    # |b_prim| from the primitive cell basis P1=((m-1)/2,(m+1)/2), P2=C4 P1
    s = np.arange(Nc) / Nc
    S1, S2 = np.meshgrid(s, s, indexing="ij")
    xr = S1 * Bc[0, 0] + S2 * Bc[0, 1]
    yr = S1 * Bc[1, 0] + S2 * Bc[1, 1]
    phQ = np.exp(-1j * (X[0] * xr + X[1] * yr))

    # centered index grid (signed, fft order) and |k|^2
    idx = np.fft.fftfreq(Nc, 1.0 / Nc).astype(int)
    I1, I2 = np.meshgrid(idx, idx, indexing="ij")
    KX = X[0] + I1 * bc[0, 0] + I2 * bc[0, 1]
    KY = X[1] + I1 * bc[1, 0] + I2 * bc[1, 1]
    K2 = KX ** 2 + KY ** 2

    # average the 4 degenerate ground states' weights (valley-symmetric budget)
    P = np.zeros((Nc, Nc))
    for i in range(gf.shape[0]):
        u = phQ * gf[i].reshape(Nc, Nc) / np.sqrt(epsc)
        uh = np.fft.fft2(u) / (Nc * Nc)
        w = np.abs(uh) ** 2
        P += w / w.sum()
    P /= gf.shape[0]

    # the two valleys' monolayer-star systems in centered indices
    nG0 = np.array([(1 - M) // 2, (1 + M) // 2])   # X' - X in centered units

    def star(gm, which):
        base = np.array([M * gm[0] + gm[1], -gm[0] + M * gm[1]])
        if which == 1:
            base = nG0 + np.array([-base[1], base[0]])  # C4 image lands on the X' system
        return base

    for sh in args.shells:
        shell_r = [float(x) for x in sh.split(",")]
        gm_ = len(shell_r) - 1
        mask = np.zeros((Nc, Nc), bool)
        for k1 in range(-gm_, gm_ + 1):
            for k2 in range(-gm_, gm_ + 1):
                r = shell_r[max(abs(k1), abs(k2))]
                Jm = int(np.ceil(r * bnorm / np.linalg.norm(bc[:, 0]))) + 2
                for which in (0, 1):
                    b0 = star((k1, k2), which)
                    o1 = np.arange(-Jm, Jm + 1)
                    O1, O2 = np.meshgrid(o1, o1, indexing="ij")
                    rr = np.sqrt((O1 * bc[0, 0] + O2 * bc[0, 1]) ** 2
                                 + (O1 * bc[1, 0] + O2 * bc[1, 1]) ** 2)
                    sel = rr <= r * bnorm + 1e-12
                    mask[(b0[0] + O1[sel]) % Nc, (b0[1] + O2[sel]) % Nc] = True
        excl = ~mask
        wexc = P[excl].sum()
        dlam = (P[excl] * (K2[excl] - lam)).sum()
        df = dlam / (8 * np.pi ** 2 * f0)
        print(f"shells {shell_r}: excluded weight {wexc:.2e}  Df budget {df:+.2e}"
              f"  (x0.65 -> {0.65*df:+.2e})", flush=True)


if __name__ == "__main__":
    main()
