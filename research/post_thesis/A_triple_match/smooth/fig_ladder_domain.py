#!/usr/bin/env python3
"""The domain-resolved ladder and the a-priori error closure.

Left  — three columns at one angle: the full-Maxwell reference (rungs muted where
        the state belongs to another valley basin or sits above the ceiling), the
        production fixed-frame envelope model on the restricted domain, and the
        exact-frame single-band model (lifted Bloch Ritz) on the same domain.
        Matched rungs joined; fixed-frame rungs beyond that model's dispersion
        limit drawn hollow.
Right — the closure: measured per-rung deviation of the fixed-frame model against
        the A-PRIORI dispersion error |h11 - E_true| at each rung's harmonic
        (computable from the monolayer alone, no reference data), plus the
        exact-frame model's residual per rung. The model errors are predictable
        before any reference is run.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_REF = "#0072B2"
C_EA = "#009E73"
C_R1 = "#CC79A7"
TOL_F = 1e-5          # fixed-frame model validity: a-priori dispersion error <= this


def main(mn=(32, 31)):
    m, n = mn
    tag = f"{m}_{n}s"
    d = np.load(os.path.join(HERE, f"diag_{tag}.npz"))
    ed = np.load(os.path.join(HERE, f"ea_dom_{tag}.npz"))
    h = np.load(os.path.join(HERE, f"hier_{tag}.npz"))
    w_r, r_in = d["w_r"], d["r_in"]
    ref = w_r[r_in]
    floor = float(ref[0])
    conv = 8 * np.pi ** 2 * np.sqrt(floor) / (2 * np.pi)
    ceil_lam = floor + float(d["ceiling"]) - float(d["dom_e"][0])
    w_ea = np.asarray(ed["w"], float)
    # per-reference-state a-priori prediction: each in-domain reference state's
    # dominant momentum -> the fixed-frame dispersion gap there
    import valley_diagnosis as vd
    rk = d["r_dom_k"][r_in]
    kap = rk - vd.M_CART["M2"][None, :]
    f = np.linalg.solve(vd.BREC0, kap.T)
    kap = (vd.BREC0 @ (f - np.rint(f))).T          # reduce mod monolayer lattice
    pred = np.abs(vd.ea_symbol(kap)
                  - np.array([vd.band1_avg(*(vd.M_CART["M2"] + k))
                              for k in kap])) / conv
    ritz = np.asarray(h["w_ritz1"], float)
    ritz = ritz[ritz <= ceil_lam + 0.002]
    lo = floor - 0.004
    hi = ceil_lam + 0.004

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.0, 8.6), sharey=True,
        gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.06))

    # ---- left: the three-column ladder
    for i, v in enumerate(w_r):
        if lo <= v <= hi:
            axl.hlines(v, -0.30, 0.30, color=C_REF if r_in[i] else "0.75",
                       lw=2.2 if r_in[i] else 1.1, zorder=3)
    ok_model = pred <= TOL_F
    # nearest-EA deviation per reference state (used for both panels)
    near = np.array([w_ea[np.argmin(np.abs(w_ea - v))] for v in ref])
    dev_e = np.abs(near - ref) / conv
    claimed_ea = {float(near[i]) for i in range(len(ref)) if ok_model[i]}
    for v in w_ea:
        if not (lo <= v <= hi):
            continue
        if float(v) in claimed_ea:
            axl.hlines(v, 0.70, 1.30, color=C_EA, lw=2.2, zorder=3)
        else:
            axl.hlines(v, 0.72, 1.28, color=C_EA, lw=1.5, zorder=3,
                       alpha=0.45, ls=(0, (2.5, 1.6)))
    for v in ritz:
        if lo <= v <= hi:
            axl.hlines(v, 1.70, 2.30, color=C_R1, lw=2.2, zorder=3)
    k = min(len(ref), len(ritz))
    for i in range(k):
        axl.plot([0.30, 1.70], [ref[i], ritz[i]], color="0.82", lw=0.6,
                 zorder=1)
    for i in range(len(ref)):
        if ok_model[i]:
            axl.plot([0.30, 0.70], [ref[i], near[i]], color="0.82", lw=0.6,
                     zorder=1)
    axl.axhline(ceil_lam, color="0.3", lw=1.0, ls=(0, (6, 3)))
    axl.annotate("domain ceiling (M3 valley floor) — no claims above",
                 (-0.55, ceil_lam), textcoords="offset points", xytext=(0, 4),
                 fontsize=8.5, color="0.3", va="bottom")
    axl.set_xticks([0, 1, 2], ["full-Maxwell\nreference",
                               "envelope,\nfixed frame",
                               "envelope,\nexact frames"], fontsize=9.5)
    axl.set_xlim(-0.6, 2.6)
    axl.set_ylabel(r"eigenvalue $\lambda = (2\pi f)^2$", fontsize=11)
    dev_r = np.abs(ritz[:k] - ref[:k]) / conv
    n_ok = int(ok_model.sum())
    axl.set_title(
        f"(m,n)=({m},{n}): all {len(ref)} in-domain states, one valley\n"
        f"exact frames: {k} rungs at "
        rf"$\leq {dev_r.max():.0e}$ in $f$;  fixed frame: {n_ok} rungs inside "
        "its a-priori\ndispersion limit (dashed = beyond it, not claimed)",
        fontsize=9.3)
    axl.grid(alpha=0.12, axis="y")

    # ---- right: measured vs a-priori predicted
    axr.plot(np.maximum(dev_e, 1e-11), ref, "D", color=C_EA, ms=5,
             label="fixed frame: measured (nearest level)")
    axr.plot(np.maximum(pred, 1e-11), ref, "o", color="0.4", ms=6,
             mfc="none", mew=1.2,
             label="fixed frame: a-priori dispersion error")
    axr.plot(np.maximum(dev_r, 1e-11), ref[:k], "^", color=C_R1, ms=5.5,
             mfc="none", mew=1.4, label="exact frames: measured")
    axr.set_xscale("log")
    axr.axvline(TOL_F, color="0.5", lw=0.8, ls=":")
    axr.text(TOL_F * 1.3, floor + 0.0005, "fixed-frame\nclaim limit",
             fontsize=7.5, color="0.4")
    axr.axhline(ceil_lam, color="0.3", lw=1.0, ls=(0, (6, 3)))
    axr.set_xlabel(r"$|f_{\mathrm{model}} - f_{\mathrm{ref}}|$  and its "
                   "a-priori prediction", fontsize=10)
    axr.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06),
               frameon=False)
    axr.grid(alpha=0.12, which="both")
    axr.set_ylim(lo, hi + 0.003)
    axr.set_title("the model error is predicted by the monolayer\n"
                  "dispersion gap — before any reference is run",
                  fontsize=9.3)

    fig.suptitle(
        "One valley, one band, every state accounted for: the a-priori domain "
        "and the two model frames",
        fontsize=11, y=0.98)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_ladder_domain.{ext}"), dpi=170,
                    bbox_inches="tight")
    print(f"saved fig_ladder_domain.{{png,pdf}}  (ritz {k} rungs max "
          f"{dev_r.max():.1e}; ea model-ok {n_ok})")


if __name__ == "__main__":
    main(tuple(int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (32, 31))
