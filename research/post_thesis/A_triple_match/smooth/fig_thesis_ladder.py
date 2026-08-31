#!/usr/bin/env python3
"""The thesis crystal across the angle ladder: grouped rung towers per angle,
every available solver side by side (MPB and FDFD as references at the anchor
angles, the completed envelope model everywhere — including the angles no
brute-force solver reaches).

Top    — the towers, grouped by angle, bottom NSHOW rungs each, in frequency.
Bottom — per-rung |f_EA - f_FDFD| at the reference angles, plus |f_MPB - f_FDFD|
         at the anchor angle (the reference-vs-reference floor).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD = "#D55E00"
C_MPB = "#56B4E9"
C_EA = "#009E73"
ANGLES = [15, 29, 57, 113, 229]
NSHOW = 14


def f_of(lam):
    return np.sqrt(np.asarray(lam, float)) / (2 * np.pi)


def load():
    out = {}
    for m in ANGLES:
        row = {}
        p = os.path.join(HERE, f"thesis_fdfd_{m}.npz")
        if os.path.exists(p):
            row["FDFD"] = np.load(p)["extrap"]
        for res in (24, 16):
            p = os.path.join(HERE, f"thesis_mpb_{m}_r{res}.npz")
            if os.path.exists(p):
                row["MPB"] = np.load(p)["lam"]
                break
        import glob
        # the registry-adapted resummed model (accurate floor); prefer the
        # common sweep setting, fall back to whatever exists for that angle
        pref = os.path.join(HERE, f"thesis_ea_{m}_n6_Ns21_g10_dk0.05.npz")
        cands = ([pref] if os.path.exists(pref) else
                 sorted(glob.glob(os.path.join(HERE,
                        f"thesis_ea_{m}_n*_g10.npz"))))
        if cands:
            row["EA"] = np.load(cands[-1])["w"]
        if row:
            out[m] = row
    return out


def main():
    data = load()
    order = [("MPB", C_MPB), ("FDFD", C_FD), ("EA", C_EA)]
    fig, (axt, axb) = plt.subplots(
        2, 1, figsize=(11.6, 9.2), height_ratios=[1.75, 1.0],
        gridspec_kw=dict(hspace=0.30))

    xg = 0.0
    xticks, xlabels = [], []
    for m in ANGLES:
        if m not in data:
            continue
        row = data[m]
        cols = [(nm, c) for nm, c in order if nm in row]
        th = np.degrees(2 * np.arctan2(1, m))
        N = m * m + 1
        # equal rung counts across a group: the comparison is only as deep as
        # the shallowest available solver there
        nshow = min([NSHOW] + [len(row[nm]) for nm, _ in cols])
        # normalize each group to its own floor (the reference floor when a
        # reference exists) so towers are comparable across four decades of cell
        # count; the absolute floor is annotated instead
        base_key = "FDFD" if "FDFD" in row else ("MPB" if "MPB" in row else "EA")
        f0 = f_of(np.sort(row[base_key])[0])
        for ci, (nm, c) in enumerate(cols):
            vals = f_of(np.sort(row[nm])[:nshow]) - f0
            for v in vals:
                axt.hlines(v, xg + ci - 0.34, xg + ci + 0.34, color=c,
                           lw=1.9, zorder=3)
        if "EA" in row and base_key != "EA":
            ri = [nm for nm, _ in cols].index(base_key)
            ei = [nm for nm, _ in cols].index("EA")
            rv = f_of(np.sort(row[base_key])[:nshow]) - f0
            ev = f_of(np.sort(row["EA"])[:nshow]) - f0
            for i in range(nshow):
                axt.plot([xg + ri + 0.34, xg + ei - 0.34], [rv[i], ev[i]],
                         color="0.80", lw=0.6, zorder=1)
        axt.annotate(f"$f_0$ = {f0:.5f}\n{nshow} rungs shown",
                     (xg + 0.5 * (len(cols) - 1), 1.0),
                     xycoords=("data", "axes fraction"),
                     textcoords="offset points", xytext=(0, -22),
                     ha="center", fontsize=7.5, color="0.35")
        xticks.append(xg + 0.5 * (len(cols) - 1))
        lab = f"({m},1)\n{th:.2f}" + r"$^\circ$" + f"\nN={N:,}"
        if "FDFD" not in row:
            lab += "\n(no reference:\nbeyond brute force)"
        xlabels.append(lab)
        xg += len(cols) + 1.15
    axt.set_xticks(xticks, xlabels, fontsize=8)
    axt.set_ylabel(r"$f - f_0$   (tower above each angle's own floor)",
                   fontsize=10.5)
    axt.set_title(
        "The thesis crystal angle by angle: the manifold tower in every "
        "available solver,\neach group referred to its own floor "
        "(absolute floors annotated)", fontsize=10)
    axt.grid(alpha=0.12, axis="y")
    handles = [plt.Line2D([], [], color=c, lw=2.4, label=nm)
               for nm, c in order]
    axt.legend(handles=handles, fontsize=9, loc="upper left", frameon=False)

    # ---- bottom: per-rung deviations at the reference angles
    for m in ANGLES:
        if m not in data or "FDFD" not in data[m]:
            continue
        row = data[m]
        fd = np.sort(row["FDFD"])
        conv = 8 * np.pi ** 2 * f_of(fd[0])
        th = np.degrees(2 * np.arctan2(1, m))
        if "EA" in row:
            ea = np.sort(row["EA"])
            k = min(len(ea), len(fd))
            dev = np.abs(ea[:k] - fd[:k]) / conv
            axb.plot([th] * k, np.maximum(dev, 1e-9), "D", color=C_EA, ms=5,
                     alpha=0.75)
        if "MPB" in row:
            mb = np.sort(row["MPB"])
            k = min(len(mb), len(fd))
            dev = np.abs(mb[:k] - fd[:k]) / conv
            axb.plot([th] * k, np.maximum(dev, 1e-9), "o", color=C_MPB, ms=5,
                     mfc="none", mew=1.3, alpha=0.9)
    axb.set_yscale("log")
    axb.invert_xaxis()
    axb.set_xlabel(r"twist angle  $\theta$  (deg)", fontsize=10.5)
    axb.set_ylabel(r"$|f - f_{\mathrm{FDFD}}|$", fontsize=10.5)
    hs = [plt.Line2D([], [], marker="D", color=C_EA, ls="", ms=5,
                     label="envelope model"),
          plt.Line2D([], [], marker="o", color=C_MPB, ls="", ms=5, mfc="none",
                     label="MPB (reference vs reference)")]
    axb.legend(handles=hs, fontsize=8.5, loc="best", frameon=False)
    axb.grid(alpha=0.15, which="both")
    axb.set_title("per-rung deviation against FDFD where a reference exists\n(angle decreasing to the right)", fontsize=9.5)

    fig.suptitle("Closing the loop: the completed envelope approximation on "
                 "the thesis crystal", fontsize=12, y=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_thesis_ladder.{ext}"), dpi=170,
                    bbox_inches="tight")
    print(f"saved fig_thesis_ladder.{{png,pdf}}  (angles: "
          f"{[m for m in ANGLES if m in data]})")


if __name__ == "__main__":
    main()
