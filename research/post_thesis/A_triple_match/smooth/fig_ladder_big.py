#!/usr/bin/env python3
"""The headline ladder: every state of the declared domain at the large angle,
FDFD reference (census inertia-certified on the FDFD matrix itself) against the
exact-frame single-band envelope model, with the fixed-frame model's claimable
subset alongside.

Left  — the three-column ladder, matched rungs joined.
Right — per-rung |f deviation| of both models against the FDFD extrapolation
        uncertainty band.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD = "#D55E00"
C_EA = "#009E73"
C_R1 = "#CC79A7"
TOL_F = 1e-5


def main(npz="ladder_big_5554.npz", out="fig_ladder_big"):
    d = np.load(os.path.join(HERE, npz), allow_pickle=True)
    m, n = (int(x) for x in d["mn"])
    fd = np.asarray(d["fdfd_claim"], float)
    unc = np.asarray(d["fdfd_unc"], float)[:len(fd)]
    rz = np.asarray(d["ritz"], float)
    ea = np.asarray(d["ea_claim"], float)
    de = np.asarray(d["de"], float)
    dom_e = np.asarray(d["dom_e"], float)
    conv = 8 * np.pi ** 2 * np.sqrt(fd[0]) / (2 * np.pi)
    off = float(d["off"])
    lo, hi = fd[0] - 0.002, fd[0] + off + 0.002
    # per-rung a-priori prediction: domain harmonics sorted by empty energy align
    # with the sorted reference ladder (moire shifts are far below the shell gaps)
    pred = de[np.argsort(dom_e)][:len(fd)] / conv
    ok_model = pred <= TOL_F
    n_ok = int(ok_model.sum())

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.2, 12.6), sharey=True,
        gridspec_kw=dict(width_ratios=[1.2, 1.0], wspace=0.06))

    cols = [("FDFD\nreference", fd, C_FD),
            ("envelope,\nfixed frame", ea, C_EA),
            ("envelope,\nexact frames", rz, C_R1)]
    near = np.array([ea[np.argmin(np.abs(ea - v))] for v in fd])
    claimed_ea = {float(near[i]) for i in range(len(fd)) if ok_model[i]}
    for i, (name, vals, c) in enumerate(cols):
        for j, v in enumerate(vals):
            solid = (i != 1) or (float(v) in claimed_ea)
            axl.hlines(v, i - 0.30, i + 0.30, color=c,
                       lw=1.9 if solid else 1.2,
                       ls="-" if solid else (0, (2.5, 1.7)),
                       alpha=1.0 if solid else 0.45, zorder=3)
    k_r = min(len(fd), len(rz))
    for i in range(k_r):
        axl.plot([0.30, 1.70], [fd[i], rz[i]], color="0.85", lw=0.5, zorder=1)
    for i in range(len(fd)):
        if ok_model[i]:
            axl.plot([0.30, 0.70], [fd[i], near[i]], color="0.85", lw=0.5,
                     zorder=1)
    dev_r = np.abs(rz[:k_r] - fd[:k_r]) / conv
    dev_e = np.abs(near - fd)[ok_model] / conv
    k_e = n_ok
    axl.set_xticks(range(3), [c[0] for c in cols], fontsize=10)
    axl.set_xlim(-0.6, 2.6)
    axl.set_ylabel(r"eigenvalue $\lambda = (2\pi f)^2$", fontsize=11)
    axl.set_title(
        f"(m,n)=({m},{n}), {int(d['census'][0])} states certified by inertia\n"
        f"exact frames: {k_r} rungs, max "
        rf"$|\Delta f| = {dev_r.max():.0e}$;  fixed frame: {k_e} claimable "
        "rungs\n(dashed = beyond its a-priori dispersion limit)",
        fontsize=9.5)
    axl.grid(alpha=0.12, axis="y")

    # ---- right: per-rung deviations vs the measured reference error
    # (FDFD h^2 extrapolation vs the certified plane-wave pencil on the 21
    # in-domain states at (32,31): 3.2e-8 .. 5.2e-8 in f)
    axr.axvspan(3.2e-8, 5.2e-8, color=C_FD, alpha=0.15, lw=0,
                label="measured reference error\n(FDFD extrap. vs certified "
                      "plane-wave)")
    axr.plot(np.maximum(dev_r, 1e-11), fd[:k_r], "^", color=C_R1, ms=5.5,
             mfc="none", mew=1.4, label="exact frames")
    axr.plot(np.maximum(dev_e, 1e-11), fd[ok_model], "D", color=C_EA, ms=5,
             label="fixed frame (claimable rungs)")
    axr.set_xscale("log")
    axr.set_xlim(8e-9, 3e-5)
    axr.axvline(1e-6, color="0.45", lw=0.8, ls=":")
    axr.text(1.25e-6, fd[0] + 0.0002, "$10^{-6}$", fontsize=8, color="0.4")
    axr.set_xlabel(r"$|f_{\mathrm{model}} - f_{\mathrm{FDFD}}|$", fontsize=10.5)
    axr.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.045),
               frameon=False)
    axr.grid(alpha=0.12, which="both")
    axr.set_ylim(lo, hi)

    fig.suptitle(
        "Every state of the declared single-valley domain, matched\n"
        "(domain and claims fixed a priori from the monolayer dispersion alone)",
        fontsize=11.5, y=0.985)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{out}.{ext}"), dpi=170,
                    bbox_inches="tight")
    print(f"saved {out}.{{png,pdf}}  (ritz {k_r} rungs max {dev_r.max():.1e}, "
          f"ea {k_e} rungs max {dev_e.max():.1e})")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
