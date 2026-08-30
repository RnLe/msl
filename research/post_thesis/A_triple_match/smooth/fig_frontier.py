#!/usr/bin/env python3
"""The coupling frontier: accuracy of the resummed envelope model vs interlayer
strength at fixed angle, against the a2^2 law of the next-order (registry-dressing)
correction. Full +0.148 window, valley-agnostic capped harmonic set, certified
references per point."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_MED = "#0072B2"
C_MAX = "#D55E00"


def main():
    d = np.load(os.path.join(HERE, "c2_sweep_1817_fixed.npz"))
    a2s = np.asarray(d["a2_list"], float)
    med = np.array([np.median(d[f"dev_{a2}"]) for a2 in a2s])
    mx = np.array([np.max(d[f"dev_{a2}"]) for a2 in a2s])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.loglog(a2s, med, "o-", color=C_MED, ms=6, label="median over the window")
    ax.loglog(a2s, mx, "s--", color=C_MAX, ms=6, mfc="none", mew=1.4,
              label="worst rung")
    ref = med[np.argmin(np.abs(a2s - 0.12))] if 0.12 in a2s else med[-1]
    a_ref = 0.12 if 0.12 in a2s else a2s[-1]
    xs = np.linspace(a2s[0] * 0.8, a2s[-1] * 1.25, 20)
    ax.loglog(xs, ref * (xs / a_ref) ** 2, "-", color="0.6", lw=1.0, zorder=0)
    ax.annotate(r"$\propto a_2^2$  (the registry-dressing order)",
                (xs[3], ref * (xs[3] / a_ref) ** 2 * 1.6), fontsize=8.5,
                color="0.4")
    ax.axhline(5e-8, color="0.5", lw=0.8, ls=":")
    ax.text(a2s[0] * 0.85, 6.2e-8, "reference floor", fontsize=7.5,
            color="0.4")
    for x, y in zip(a2s, med):
        ax.annotate(f"{y:.0e}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7.5, color="0.35")
    ax.set_xlabel(r"interlayer amplitude $a_2$", fontsize=10.5)
    ax.set_ylabel(r"$|f_{\mathrm{EA}} - f_{\mathrm{ref}}|$", fontsize=10.5)
    ax.set_title("The validity frontier of the resummed envelope model\n"
                 "(m,n)=(18,17), full window, all valleys, counts certified",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.15, which="both")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_frontier.{ext}"), dpi=170,
                    bbox_inches="tight")
    print("saved fig_frontier.{png,pdf}")


if __name__ == "__main__":
    main()
