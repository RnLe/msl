#!/usr/bin/env python3
"""The order ladder of the envelope approximation, per rung and as a staircase.

Left  — per-rung |f deviation| against the certified reference for each model
        generation at (32,31)-scaled: raw fixed frame, the eta^2 Lowdin fold with
        one and with three remote bands, the exact Feshbach downfold of the same
        content, and the fully resummed exact-frame model.
Right — the same as medians/maxima: what each order of the expansion buys.

Colors: Okabe-Ito categorical order, fixed per model across all campaign figures.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("raw", "fixed frame (raw)", "#009E73", "D"),
    ("lowdin2_q1", "+ Lowdin fold, 1 remote band", "#E69F00", "s"),
    ("lowdin2_q3", "+ Lowdin fold, 3 remote bands", "#56B4E9", "v"),
    ("feshbach_q3", "exact Feshbach, 3 remote bands", "#0072B2", "o"),
    ("ritz1", "resummed exact frames", "#CC79A7", "^"),
]


def main(tag="32_31s"):
    d = np.load(os.path.join(HERE, f"fold_{tag}.npz"))
    ref = d["ref"]
    conv = float(d["conv"])
    m, n = tag.split("_")[0], tag.split("_")[1].rstrip("s")

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.2, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1.0]))
    stats = []
    for key, lab, c, mk in MODELS:
        v = np.asarray(d[key], float)
        k = min(len(v), len(ref))
        dev = np.abs(v[:k] - ref[:k]) / conv
        axl.plot(np.arange(k), np.maximum(dev, 1e-11), mk, color=c, ms=5,
                 mfc="none" if key in ("feshbach_q3", "ritz1") else c,
                 mew=1.3, label=lab, alpha=0.9)
        stats.append((lab, c, np.median(dev), dev.max()))
    axl.set_yscale("log")
    axl.set_xlabel("rung index (all in-domain states, sorted)", fontsize=10)
    axl.set_ylabel(r"$|f_{\mathrm{model}} - f_{\mathrm{ref}}|$", fontsize=10.5)
    axl.axhline(5e-8, color="0.5", lw=0.8, ls=":")
    axl.text(0.2, 6e-8, "reference floor", fontsize=7.5, color="0.4")
    axl.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               frameon=False, ncol=2)
    axl.grid(alpha=0.12, which="both")
    axl.set_title(f"(m,n)=({m},{n}), scaled family: every in-domain rung,\n"
                  "five model generations", fontsize=10)

    xs = np.arange(len(stats))
    for i, (lab, c, med, mx) in enumerate(stats):
        axr.plot([i - 0.28, i + 0.28], [med, med], color=c, lw=3.2)
        axr.plot(i, mx, "v", color=c, ms=5, mfc="none", mew=1.3)
        axr.annotate(f"{med:.0e}", (i, med), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=7.5, color="0.3")
    axr.set_yscale("log")
    lo_all = min(x[2] for x in stats)
    axr.set_ylim(lo_all * 0.25, None)
    axr.set_xticks(xs, ["raw", "Lowdin\n1 band", "Lowdin\n3 bands",
                        "Feshbach\n3 bands", "resummed\nframes"], fontsize=8.5)
    axr.set_ylabel("median (bar), max (marker)", fontsize=10)
    axr.grid(alpha=0.12, axis="y", which="both")
    axr.set_title("what each order buys\n(the eta-square term the thesis "
                  "already contains\nrecovers 2-3 decades)", fontsize=9.5)

    fig.suptitle("Completing the envelope approximation order by order",
                 fontsize=11.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_order_ladder.{ext}"), dpi=170,
                    bbox_inches="tight")
    print("saved fig_order_ladder.{png,pdf}")


if __name__ == "__main__":
    main()
