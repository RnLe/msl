#!/usr/bin/env python3
"""The hero figure: EA-vs-reference residual across the three model/scaling families
on the smooth single-valley candidate, with the certified reference band and the
triple-match landing point. Reads hero_family/hero_adapted/hero_scaled/fdfd_scaled npz.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def series(npz, family, key):
    d = np.load(os.path.join(HERE, npz), allow_pickle=True)
    etas, devs = [], []
    for m, n in family:
        try:
            eta = float(np.atleast_1d(d[f"eta_{m}_{n}"])[0])
            if key == "dev":
                v = float(np.atleast_1d(d[f"dev_{m}_{n}"])[0])
            else:
                r = np.atleast_1d(d[f"ref_{m}_{n}"])[0]
                e = np.atleast_1d(d[f"ea_{m}_{n}"])[0]
                v = abs(e - r)
            etas.append(eta)
            devs.append(v)
        except KeyError:
            pass
    return np.array(etas), np.array(devs)


def main():
    fam5 = [(4, 3), (5, 4), (6, 5), (7, 6), (9, 8)]
    fam4 = [(5, 4), (6, 5), (7, 6), (9, 8)]
    d_frozen = np.load(os.path.join(HERE, "hero_family.npz"), allow_pickle=True)
    e_fr, v_fr = [], []
    for m, n in fam5:
        e_fr.append(float(np.atleast_1d(d_frozen[f"eta_{m}_{n}"])[0]))
        r = np.atleast_1d(d_frozen[f"ref_{m}_{n}"])[0]
        a = np.atleast_1d(d_frozen[f"ea_{m}_{n}"])[0]
        v_fr.append(abs(a - r))
    e_ad, v_ad = series("hero_adapted.npz", fam5, "dev")
    e_sc, v_sc = series("hero_scaled.npz", fam4, "dev")

    f0 = 0.2043
    conv = 8 * np.pi ** 2 * f0

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(e_fr, np.array(v_fr) / conv, "s-", color="0.55",
              label="frozen frame, fixed $a_2$  ($\\eta^{-0.18}$)")
    ax.loglog(e_ad, v_ad / conv, "o-", color="C0",
              label="registry-adapted, fixed $a_2$  ($\\eta^{-0.41}$)")
    ax.loglog(e_sc, v_sc / conv, "D-", color="C2",
              label="registry-adapted, $a_2\\propto\\eta^2$  ($\\eta^{3.8}$)")
    p = np.polyfit(np.log(e_sc), np.log(v_sc / conv), 1)
    xs = np.linspace(min(e_sc) * 0.85, max(e_sc) * 1.15, 20)
    ax.loglog(xs, np.exp(np.polyval(p, np.log(xs))), "--", color="C2", lw=1)
    ax.axhline(2.7e-7 / conv, color="C3", lw=1, ls=":",
               label="reference certification (PWE vs FDFD)")
    ax.annotate("triple match:\nEA = FDFD = PWE\nto $8\\times10^{-7}$ in $f$",
                xy=(e_sc[-1], v_sc[-1] / conv), xytext=(0.075, 3e-7),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"$\eta = 2\sin(\theta/2)$")
    ax.set_ylabel(r"$|f_{\mathrm{EA}} - f_{\mathrm{ref}}|$")
    ax.set_title("Envelope approximation vs full Maxwell references\n"
                 "(smooth single-valley bilayer, manifold ground state)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_hero_scaling.{ext}"), dpi=180)
    print("saved fig_hero_scaling.{png,pdf}")


if __name__ == "__main__":
    main()
