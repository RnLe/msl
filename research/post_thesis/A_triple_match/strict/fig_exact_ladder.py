#!/usr/bin/env python3
"""Section 17 deliverable figure: the eigenvalue-exact PWE <-> FDFD comparison at 2 deg.

(a) Convergence + the sampling wall: px16 Dfloor(bottom) vs window budget with the
    delta + r*budget fit; px32 points show the wall collapsing 4x.
(b) The ladder: per-state px16/px32 Richardson-extrapolated valley-PWE (sh40 window)
    against the Richardson-extrapolated FDFD quadruplets, with residuals.
(c) 16 deg (m=7): the window solver is exact to 5e-6 with 259 plane waves.

Reads: pwe_iter_m57_sh{18,24,40...}_px{16,32}_even.npz, fdfd_ladder_2deg.npz,
       pwe_iter_m7_r3_g2_even.npz. Writes fig_exact_ladder.{png,pdf} + exact_ladder_data.npz
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 0.370907
LO, HI = 0.3661, 0.3790

SH18 = "sh18_18_18_12_6"
SH24 = "sh24_24_24_16_8"
SH40 = "sh40_40_40_32_16_8"
BUDGET = {SH18: 1.10e-3, SH24: 7.14e-4, SH40: 3.95e-4}   # budget_window.py


def load(name, px):
    return np.load(os.path.join(HERE, f"pwe_iter_m57_{name}_px{px}_even.npz"))


def win(f):
    return f[(f >= LO) & (f <= HI)]


def main():
    ref = np.load(os.path.join(HERE, "fdfd_ladder_2deg.npz"))
    fx, sg = ref["f_ext"], ref["sigma"]

    # ---- (a) bottoms vs budget
    b16 = {k: min(win(load(k, 16)["f_C2m"])[0], win(load(k, 16)["f_C2p"])[0])
           for k in (SH18, SH24, SH40) if os.path.exists(
               os.path.join(HERE, f"pwe_iter_m57_{k}_px16_even.npz"))}
    b32 = {}
    for k in (SH18, SH40):
        p = os.path.join(HERE, f"pwe_iter_m57_{k}_px32_even.npz")
        if os.path.exists(p):
            d = np.load(p)
            b32[k] = min(win(d["f_C2m"])[0], win(d["f_C2p"])[0])
    bud = np.array([BUDGET[k] for k in b16])
    dm = np.array([b16[k] - FLOOR for k in b16])
    A = np.vstack([np.ones_like(bud), bud]).T
    delta, r = np.linalg.lstsq(A, dm, rcond=None)[0]

    # ---- (b) final ladder: sh40 px16/px32 Richardson per state, both C2 blocks
    # Strict protocol: sorted ladders, index-aligned (no matching). Each C2 block holds
    # exactly one state per quadruplet (m=7 validated), so block index i <-> quadruplet i.
    d16, d32 = load(SH40, 16), load(SH40, 32)
    rows = []
    for blk in ("f_C2p", "f_C2m"):
        w16, w32 = np.sort(win(d16[blk])), np.sort(win(d32[blk]))
        n = min(len(w16), len(w32))
        fr = (4 * w32[:n] - w16[:n]) / 3
        for j, (v16, v32, v) in enumerate(zip(w16[:n], w32[:n], fr)):
            rows.append((blk, v16, v32, v, j, v - fx[j]))
    rows.sort(key=lambda t: t[3])

    print(f"wall fit: delta = {delta:+.2e}, r = {r:.2f}")
    print(f"{'blk':<6} {'px16':>9} {'px32':>9} {'Richardson':>10} {'ref':>9} {'resid':>10}")
    for blk, v16, v32, v, j, res in rows:
        print(f"{blk:<6} {v16:9.6f} {v32:9.6f} {v:10.6f} q{j}:{fx[j]:.6f} {res:>+10.2e}")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ax = axes[0]
    xs = np.linspace(0, max(bud) * 1.15, 50)
    ax.plot(xs * 1e3, (delta + r * xs) * 1e3, "-", color="0.6", lw=1,
            label=fr"$\delta + r\,B$:  $\delta$={delta*1e4:.1f}e-4, $r$={r:.2f}")
    ax.plot(bud * 1e3, dm * 1e3, "o", color="C0", label="px16 bottoms")
    for k in b32:
        ax.plot([BUDGET[k] * 1e3], [(b32[k] - FLOOR) * 1e3], "s", color="C3")
    ax.plot([], [], "s", color="C3", label="px32 bottoms")
    ax.set_xlabel("first-order window budget  ($10^{-3}$)")
    ax.set_ylabel(r"$\Delta f$ vs continuum floor  ($10^{-3}$)")
    ax.set_title("(a) window ladder + the sampling wall")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    for i, (fq, s) in enumerate(zip(fx, sg)):
        ax.axhspan(fq - s, fq + s, color="0.85", zorder=0)
        ax.axhline(fq, color="0.55", lw=0.8, zorder=1)
    for blk, mk in (("f_C2p", "o"), ("f_C2m", "D")):
        vals = [t[3] for t in rows if t[0] == blk]
        ax.plot([0.35 if blk == "f_C2p" else 0.65] * len(vals), vals, mk, ms=5,
                color="C0" if blk == "f_C2p" else "C2", label=blk[2:])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(LO, HI)
    ax.set_ylabel("frequency (c/a)")
    ax.set_title("(b) extrapolated PWE ladder on the FDFD quadruplets")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    m7 = np.load(os.path.join(HERE, "pwe_iter_m7_r3_g2_even.npz"))
    exact = {"C2+": 0.066924, "C2-": 0.067031}   # section 15.8 dense-exact operator
    got = {"C2+": m7["f_C2p"][0], "C2-": m7["f_C2m"][0]}
    ax.bar([0, 1], [abs(got["C2+"] - exact["C2+"]) * 1e6,
                    abs(got["C2-"] - exact["C2-"]) * 1e6], width=0.5, color="C0")
    ax.set_xticks([0, 1], ["C2+", "C2$-$"])
    ax.set_ylabel(r"|window $-$ dense-exact|  ($10^{-6}$)")
    ax.set_title("(c) 16$°$: exact with 259 PWs")
    fig.suptitle("Valley-windowed PWE vs FDFD at 2$°$ — the eigenvalue-exact comparison",
                 y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_exact_ladder.{ext}"), dpi=180,
                    bbox_inches="tight")
    np.savez(os.path.join(HERE, "exact_ladder_data.npz"),
             rows=np.array([(t[1], t[2], t[3], t[4], t[5]) for t in rows]),
             blocks=np.array([t[0] for t in rows]),
             fdfd_f=fx, fdfd_sigma=sg, delta=delta, ratio=r,
             budgets=bud, bottoms16=dm)
    print("saved fig_exact_ladder.{png,pdf} + exact_ladder_data.npz")


if __name__ == "__main__":
    main()
