#!/usr/bin/env python3
"""The wide sector ladder: every state of the M2 tower in one window, side by side.

Left  — the ladder itself: one column per solver, one rung per eigenvalue, matched
        rungs joined across the columns.
Right — the same rungs on a shared eigenvalue axis, plotted as |f_EA - f_ref| so the
        per-state accuracy is readable rung by rung, against the reference
        certification band |f_PWE - f_FDFD|.

Colors: Okabe-Ito CVD-safe triple (FDFD vermillion, PWE blue, EA green).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD = "#D55E00"
C_PW = "#0072B2"
C_EA = "#009E73"
M3_OFFSET = 0.0355        # M3 monolayer floor above the M2 carrier (candidate_hexM)


def _f(lam):
    return np.sqrt(lam) / (2 * np.pi)


def match(ref, ea, tol_frac=0.45, floor_frac=0.15):
    """Globally greedy pairing reference <-> EA: closest pairs are assigned first, so a
    near-degenerate doublet cannot be spoiled by the order it is visited in. The
    tolerance is a fraction of the local reference gap, with a floor set by the mean
    level spacing (a degenerate pair has no local gap to scale by)."""
    ref = np.asarray(ref, float)
    ea = np.asarray(ea, float)
    if len(ref) < 2 or not len(ea):
        return [None] * len(ref), np.zeros(len(ea), bool)
    d = np.diff(ref)
    gaps = np.minimum(np.r_[d, d[-1]], np.r_[d[0], d])
    mean_sp = (ref[-1] - ref[0]) / max(len(ref) - 1, 1)
    tol = np.maximum(tol_frac * gaps, floor_frac * mean_sp)
    cand = []
    for i, v in enumerate(ref):
        for j in np.where(np.abs(ea - v) <= tol[i])[0]:
            cand.append((abs(ea[j] - v), i, int(j)))
    cand.sort()
    pairs = [None] * len(ref)
    taken = np.zeros(len(ea), bool)
    for _, i, j in cand:
        if pairs[i] is None and not taken[j]:
            pairs[i] = j
            taken[j] = True
    return pairs, taken


def main(npz="ladder_wide.npz", out="fig_ladder_wide"):
    d = np.load(os.path.join(HERE, npz))
    m, n = (int(x) for x in d["mn"])
    pw = np.asarray(d["pwe"], float)
    ea = np.asarray(d["ea"], float)
    fd = np.asarray(d["fdfd"], float) if "fdfd" in d.files else None
    lo, hi = pw[0] - 0.004, float(d["floor"]) + float(d["span"])
    ea = ea[(ea >= lo) & (ea <= hi)]
    if fd is not None:
        # the inertia census fixes the exact state count; the FD ladder is aligned
        # from the band-1 floor, so any tail beyond the census is resolution drift
        fd = fd[(fd >= lo) & (fd <= hi)][:len(pw)]

    cols = ([("FDFD", fd, C_FD)] if fd is not None else []) + \
           [("plane-wave", pw, C_PW), ("envelope", ea, C_EA)]
    ncol = len(cols)
    pairs, taken = match(pw, ea)

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(10.2, 12.4), sharey=True,
        gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.06))

    # ---- left: the ladder
    for i, (name, vals, c) in enumerate(cols):
        for v in vals:
            axl.hlines(v, i - 0.30, i + 0.30, color=c, lw=1.9,
                       zorder=3, alpha=0.95)
    i_pw = ncol - 2
    if fd is not None:
        pf, _ = match(pw, fd)
        for i, j in enumerate(pf):
            if j is not None:
                axl.plot([0.30, 0.70], [fd[j], pw[i]], color="0.78", lw=0.6,
                         zorder=1)
    for i, j in enumerate(pairs):
        if j is not None:
            axl.plot([i_pw + 0.30, i_pw + 0.70], [pw[i], ea[j]], color="0.78",
                     lw=0.6, zorder=1)
    for j in np.where(~taken)[0]:
        axl.hlines(ea[j], ncol - 1 - 0.26, ncol - 1 + 0.26, color=C_EA, lw=1.1,
                   ls=(0, (2.0, 2.0)), alpha=0.3, zorder=2)
    from matplotlib.lines import Line2D
    axl.legend(handles=[
        Line2D([], [], color=C_EA, lw=1.9, label="envelope rung matched to the "
               "references"),
        Line2D([], [], color=C_EA, lw=1.1, ls=(0, (2.0, 2.0)), alpha=0.5,
               label="extra envelope level (no reference partner)")],
        fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.045),
        frameon=False)
    axl.set_xticks(range(ncol), [c[0] for c in cols], fontsize=10)
    axl.set_xlim(-0.7, ncol - 0.3)
    axl.set_ylabel(r"eigenvalue $\lambda = (2\pi f)^2$", fontsize=11)
    axl.grid(alpha=0.13, axis="y")
    nmatch = int(taken.sum())
    axl.set_title(f"{len(pw)} sector states (count certified by inertia)\n"
                  f"{nmatch} matched rung{'s' if nmatch != 1 else ''}",
                  fontsize=10.5)
    # the single-valley ceiling: above the M3 floor this sector also carries
    # M3-basin states, which a one-valley envelope theory cannot represent
    m3 = float(d["floor"]) + M3_OFFSET
    if lo < m3 < hi:
        for a in (axl, axr):
            a.axhline(m3, color="0.35", lw=0.9, ls=(0, (6, 3)), zorder=2)
        axl.annotate("M3 valley floor — above this the sector also holds\n"
                     "M3-basin states, outside a single-valley envelope theory",
                     (-0.62, m3), textcoords="offset points", xytext=(0, 5),
                     fontsize=8, color="0.3", va="bottom")

    # ---- right: per-rung accuracy in frequency
    conv = 8 * np.pi ** 2 * _f(pw[0])
    allx = []
    if fd is not None:
        pf, _ = match(pw, fd)
        rr = [(pw[i], abs(pw[i] - fd[j]) / conv) for i, j in enumerate(pf)
              if j is not None]
        if rr:
            y0, x0 = np.array([r[0] for r in rr]), np.array([r[1] for r in rr])
            allx += list(x0)
            axr.plot(x0, y0, "o", color=C_PW, ms=4.5, mfc="none", mew=1.2,
                     label=r"$|f_{\mathrm{PWE}}-f_{\mathrm{FDFD}}|$  (reference "
                           "certification)")
    rows = [(pw[i], abs(ea[j] - pw[i]) / conv) for i, j in enumerate(pairs)
            if j is not None]
    xlo = max(min(allx + [r[1] for r in rows] + [1e-11]) / 4, 1e-12)
    if rows:
        y, x = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        allx += list(x)
        axr.hlines(y, xlo, x, color=C_EA, lw=0.8, alpha=0.45)
        axr.plot(x, y, "D", color=C_EA, ms=5,
                 label=r"$|f_{\mathrm{EA}}-f_{\mathrm{ref}}|$")
    axr.set_xscale("log")
    axr.set_xlim(xlo, max(allx + [1e-6]) * 6)
    axr.axvline(1e-5, color="0.45", lw=0.8, ls=":")
    axr.set_xlabel("deviation in frequency", fontsize=11)
    axr.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.045),
               frameon=False)
    axr.grid(alpha=0.13, which="both")
    axr.set_title("per-rung accuracy\n(dotted line: $10^{-5}$)", fontsize=10.5)
    axr.set_ylim(lo - 0.003, hi + 0.004)

    fig.suptitle(
        f"Sector-resolved eigenvalue ladder, (m,n)=({m},{n}), "
        f"{int(d['ncells'])} moire cells\n"
        "envelope approximation against two independent full-Maxwell solvers, "
        "state by state",
        fontsize=11.5, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{out}.{ext}"), dpi=170,
                    bbox_inches="tight")
    print(f"saved {out}.{{png,pdf}}  ({len(pw)} ref rungs, {nmatch} matched, "
          f"{len(ea)-nmatch} unmatched EA)")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
