#!/usr/bin/env python3
"""Ladder views of the triple match.

A: the manifold eigenvalue across the scaled family, three solvers overlaid, with the
   residual panel (log) showing the eta^3.8 law and the reference-certification band.
B: the landing point under the microscope — the three solvers' rungs on a
   frequency-deviation axis (units 1e-6), FDFD extrapolation band included.
C: the sector-resolved multi-rung ladder at (9,8), fixed material: the M2 envelope
   tower in all three solvers (the three M valleys fold to distinct supercell
   sectors, so this sector holds the M2 tower only).

Colors: Okabe-Ito CVD-safe triple (FDFD vermillion, PWE blue, EA green), validated.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD = "#D55E00"
C_PW = "#0072B2"
C_EA = "#009E73"
FAM = [(5, 4), (6, 5), (7, 6), (9, 8)]


def _f(lam):
    return np.sqrt(lam) / (2 * np.pi)


def load_scaled():
    hs = np.load(os.path.join(HERE, "hero_scaled.npz"), allow_pickle=True)
    fd_sources = [np.load(os.path.join(HERE, n), allow_pickle=True)
                  for n in ("fdfd_scaled.npz", "fdfd_scaled65.npz",
                            "fdfd_leg_ladders.npz")]
    rows = []
    for m, n in FAM:
        eta = float(np.atleast_1d(hs[f"eta_{m}_{n}"])[0])
        ref = float(np.atleast_1d(hs[f"ref_{m}_{n}"])[0])
        ea = float(np.atleast_1d(hs[f"ea_{m}_{n}"])[0])
        fd = unc = np.nan
        for src in fd_sources:
            if f"{m}_{n}_extrap" in src:
                fd = float(np.atleast_1d(src[f"{m}_{n}_extrap"])[0])
                unc = float(np.atleast_1d(src[f"{m}_{n}_unc"])[0])
                break
        rows.append(dict(m=m, n=n, eta=eta, ref=ref, ea=ea, fd=fd, unc=unc))
    return rows


def fig_a(rows):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.6, 6.2), height_ratios=[1.1, 1.0],
        gridspec_kw=dict(hspace=0.32))
    etas = [r["eta"] for r in rows]
    ax.plot(etas, [r["fd"] for r in rows], "s", color=C_FD, ms=8, mfc="none",
            mew=1.6, label="FDFD (extrapolated)")
    ax.plot(etas, [r["ref"] for r in rows], "o", color=C_PW, ms=5.5,
            label="plane-wave reference")
    ax.plot(etas, [r["ea"] for r in rows], "D", color=C_EA, ms=4,
            label="envelope approximation")
    ax.plot(etas, [r["ref"] for r in rows], "-", color=C_PW, lw=0.8, alpha=0.4)
    for r in rows:
        ax.annotate(f"({r['m']},{r['n']})", (r["eta"], r["ref"]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color="0.35")
    ax.set_xlabel(r"$\eta = 2\sin(\theta/2)$")
    ax.set_ylabel(r"manifold eigenvalue $\lambda$")
    ax.set_title("The manifold state in three solvers (scaled family, "
                 r"$a_2 \propto \eta^2$)" + "\nthe envelope rung converges onto "
                 "the coinciding references as the angle shrinks", fontsize=10)
    lo = min(min(r["ea"] for r in rows), min(r["ref"] for r in rows))
    hi = max(max(r["ea"] for r in rows), max(r["ref"] for r in rows))
    pad = (hi - lo) * 0.18
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    conv = [8 * np.pi ** 2 * _f(r["ref"]) for r in rows]
    d_er = [abs(r["ea"] - r["ref"]) / c for r, c in zip(rows, conv)]
    d_ef = [abs(r["ea"] - r["fd"]) / c for r, c in zip(rows, conv)]
    d_rf = [abs(r["ref"] - r["fd"]) / c for r, c in zip(rows, conv)]
    axr.loglog(etas, d_er, "D-", color=C_EA, ms=5, label="|EA $-$ plane-wave|")
    axr.loglog(etas, d_ef, "s--", color=C_FD, ms=5, mfc="none",
               label="|EA $-$ FDFD|")
    axr.loglog(etas, d_rf, "o:", color=C_PW, ms=4,
               label="|plane-wave $-$ FDFD| (certification)")
    p = np.polyfit(np.log(etas), np.log(d_er), 1)
    xs = np.linspace(min(etas) * 0.9, max(etas) * 1.1, 10)
    axr.loglog(xs, np.exp(np.polyval(p, np.log(xs))), "-", color="0.6", lw=0.8,
               zorder=0)
    axr.axhline(1e-6, color="0.3", lw=0.8, ls=":")
    axr.text(etas[0] * 0.94, 1.15e-6, r"$10^{-6}$", fontsize=8, color="0.3")
    axr.text(etas[1], d_er[1] * 1.5, rf"$\eta^{{{p[0]:.1f}}}$", fontsize=9,
             color="0.4")
    axr.set_xlabel(r"$\eta$")
    axr.set_ylabel(r"$|\Delta f|$ (frequency units)")
    axr.legend(fontsize=8, loc="upper left")
    axr.grid(alpha=0.2, which="both")
    fig.savefig(os.path.join(HERE, "fig_ladder_family.png"), dpi=180,
                bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig_ladder_family.pdf"),
                bbox_inches="tight")
    print("saved fig_ladder_family.{png,pdf}")


def fig_b(rows):
    r = rows[-1]
    conv = 8 * np.pi ** 2 * _f(r["ref"])
    y = {"FDFD": 0.0,
         "plane-wave\nreference": (r["ref"] - r["fd"]) / conv * 1e6,
         "envelope\napproximation": (r["ea"] - r["fd"]) / conv * 1e6}
    unc = r["unc"] / conv * 1e6
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    cols = [C_FD, C_PW, C_EA]
    offs = [(0, 10), (0, -16), (0, 10)]
    for i, (k, v) in enumerate(y.items()):
        ax.hlines(v, i - 0.28, i + 0.28, color=cols[i], lw=3, zorder=3)
        ax.annotate(f"{v:+.3f}" if i else "0 (anchor)", (i, v),
                    textcoords="offset points", xytext=offs[i], ha="center",
                    fontsize=9, color="0.25")
    ax.axhspan(-unc, +unc, color=C_FD, alpha=0.12, lw=0, zorder=1,
               label=f"FDFD extrapolation band ±{unc:.3f}")
    ax.set_xticks(range(3), list(y.keys()), fontsize=9)
    ax.set_ylabel(r"$f - f_{\mathrm{FDFD}}$   ($10^{-6}$ frequency units)")
    ax.set_title(f"The landing point under the microscope\n"
                 f"(m,n)=({r['m']},{r['n']}), "
                 r"$a_2 \propto \eta^2$ family", fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.2, axis="y")
    ax.set_xlim(-0.6, 2.6)
    ymin = min(y.values())
    ax.set_ylim(ymin * 1.18, 0.14)
    fig.savefig(os.path.join(HERE, "fig_ladder_landing.png"), dpi=180,
                bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig_ladder_landing.pdf"),
                bbox_inches="tight")
    print("saved fig_ladder_landing.{png,pdf}")


def _ladder_panel(ax, w_fd, w_pw, w_ea, title, match_tol=2e-4):
    """Three-column rung panel: matched rungs solid + connected, unmatched EA rungs
    dashed (honesty: the single-band adapted model's tower is next-order)."""
    conv = 8 * np.pi ** 2 * _f(w_fd[0])
    for i, (c, w) in enumerate([(C_FD, w_fd), (C_PW, w_pw)]):
        for v in w:
            ax.hlines(v, i - 0.26, i + 0.26, color=c, lw=2.6)
    matched_ea = np.zeros(len(w_ea), bool)
    for v in w_fd:
        j = int(np.argmin(np.abs(w_pw - v)))
        if abs(w_pw[j] - v) < match_tol:
            ax.plot([0.26, 0.74], [v, w_pw[j]], color="0.8", lw=0.7, zorder=0)
        k = int(np.argmin(np.abs(w_ea - v)))
        if abs(w_ea[k] - v) < match_tol:
            matched_ea[k] = True
            ax.plot([1.26, 1.74], [w_pw[j], w_ea[k]], color="0.8", lw=0.7,
                    zorder=0)
            ax.annotate(rf"$\Delta f\,{abs(w_ea[k]-v)/conv:.0e}$",
                        (2.34, w_ea[k]), fontsize=7.5, color="0.35", va="center")
    for k, v in enumerate(w_ea):
        ax.hlines(v, 2 - 0.26, 2 + 0.26, color=C_EA, lw=2.6,
                  ls="-" if matched_ea[k] else (0, (3, 2)),
                  alpha=1.0 if matched_ea[k] else 0.65)
    ax.set_xticks(range(3), ["FDFD", "plane-wave", "envelope"], fontsize=9)
    ax.set_xlim(-0.6, 3.4)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.15, axis="y")


def fig_c():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    made = 0
    # scaled ladder at (9,8): both references show three sector rungs, pairwise 3e-7
    s = np.load(os.path.join(HERE, "fdfd_scaled.npz"), allow_pickle=True)
    hs = np.load(os.path.join(HERE, "hero_scaled.npz"), allow_pickle=True)
    w_fd = np.atleast_1d(s["9_8_extrap_all"])
    w_pw = np.atleast_1d(hs["ref_9_8"])
    w_ea = np.atleast_1d(hs["ea_9_8"])
    _ladder_panel(axes[0], w_fd, w_pw, w_ea,
                  "(9,8), scaled family ($a_2 \\propto \\eta^2$)\n"
                  "references agree on all rungs to $3\\times10^{-7}$")
    axes[0].set_ylabel(r"eigenvalue $\lambda$")
    made += 1
    # fixed-material ladder at (9,8): the four-rung sector tower
    p_lad = os.path.join(HERE, "ladder98_unscaled.npz")
    if os.path.exists(p_lad):
        lad = np.load(p_lad)
        fdw = np.load(os.path.join(HERE, "ladder_fdfd_wide.npz"),
                      allow_pickle=True)
        _ladder_panel(axes[1], np.atleast_1d(fdw["9_8_wide_extrap"]),
                      np.atleast_1d(lad["ref"]), np.atleast_1d(lad["ea"]),
                      "(9,8), fixed material — four-rung sector tower\n"
                      "references agree on all rungs to $3\\times10^{-7}$",
                      match_tol=1.2e-3)
        made += 1
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "fixed-material panel pending", ha="center",
                     fontsize=9, color="0.5")
    fig.suptitle("Sector-resolved eigenvalue ladders (the three M valleys fold "
                 "to distinct sectors; this sector holds the M2 tower only)\n"
                 "dashed rungs: unmatched envelope tower states — next-order in "
                 "the single-band model", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_ladder_tower.png"), dpi=180,
                bbox_inches="tight")
    fig.savefig(os.path.join(HERE, "fig_ladder_tower.pdf"),
                bbox_inches="tight")
    print(f"saved fig_ladder_tower.{{png,pdf}} ({made} panels)")


if __name__ == "__main__":
    rows = load_scaled()
    fig_a(rows)
    fig_b(rows)
    fig_c()
