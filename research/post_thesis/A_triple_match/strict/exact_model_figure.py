#!/usr/bin/env python3
"""Consolidated deliverable: the exact continuum model + regime map.

Panel A — the exact Galerkin continuum model converges to FDFD (variational),
          quantified on (7,1): mean|Δf| vs retained bands.
Panel B — the practical momentum-space model reproduces the FDFD X-manifold at
          2° (from momentum_hero_data.json): stripes + de-trended residual.
Panel C — the regime map: two hard walls (dissolution, two-scale) + the
          V/E_kin soft-cost axis; campaign data points placed.

Pure plotting from cached results — no solver calls.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD, C_EA, C_G = "#222222", "#d62728", "#1f77b4"

fig = plt.figure(figsize=(13.5, 5.0))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.985,
                      top=0.86, bottom=0.16)

# ---- Panel A: Galerkin convergence certificate (7,1)
axA = fig.add_subplot(gs[0, 0])
fd7 = np.load(os.path.join(HERE, "fdfd_m7_px24.npy"))
nbs, means, maxs = [], [], []
for nb in [2, 4, 6, 8]:
    p = os.path.join(HERE, f"galerkin_m7_nb{nb}.npz")
    if not os.path.exists(p):
        continue
    ea = np.sort(np.load(p)["freqs"])
    n = 12
    d = np.abs(ea[:n] - fd7[:n])
    nbs.append(nb); means.append(d.mean()); maxs.append(d.max())
axA.semilogy(nbs, means, "o-", color=C_G, lw=1.5, ms=7, label="mean |Δf| (bottom 12)")
axA.semilogy(nbs, maxs, "s--", color=C_G, lw=1, ms=5, alpha=0.6, label="max |Δf|")
axA.axhline(6e-4, color="0.5", ls=":", lw=1)
axA.text(2.1, 6.5e-4, "FDFD px-drift floor", fontsize=7, color="0.4")
axA.set_xlabel("retained bands  N_b")
axA.set_ylabel("|Δf| vs FDFD  [c/a]")
axA.set_title("A  Exact Galerkin model → FDFD\n(7,1) θ=16.3°, variational, "
              "monotone", fontsize=9.5)
axA.set_xticks([2, 4, 6, 8]); axA.legend(fontsize=7.5, loc="upper right")
axA.text(0.04, 0.06, "provably convergent:\nH c = λ S c on the\nreference-Bloch basis",
         transform=axA.transAxes, fontsize=7.5, va="bottom",
         bbox=dict(fc="white", ec="0.8", alpha=0.9))

# ---- Panel B: practical momentum model at 2° (from hero data)
axB = fig.add_subplot(gs[0, 1])
try:
    d = json.load(open(os.path.join(HERE, "momentum_hero_data.json")))
    cd = d["cases"]["2deg"]
    fd = np.array(cd["fdfd_xmanifold"]); ea = np.array(cd["ea_ladder"])
    off = cd["edge_offset"]
    for f in fd:
        axB.hlines(f, 0.66, 1.34, color=C_FD, lw=1.1)
    for f in ea - off:
        axB.hlines(f, 1.66, 2.34, color=C_EA, lw=1.1)
    axB.set_xticks([1, 2], ["FDFD\nX-manifold", "momentum\nmodel −offset"], fontsize=8)
    axB.set_xlim(0.4, 2.6)
    axB.set_ylabel("frequency  f a/c")
    axB.set_title("B  Practical continuum model, 2°\n(57,1): count 24/24, "
                  "×4 quadruplets", fontsize=9.5)
    axB.text(0.03, 0.97,
             f"edge offset {off:+.1e} (→0 as θ→0)\n"
             f"shape resid {cd['mean_abs_detrended']:.1e} (≈FDFD floor)",
             transform=axB.transAxes, va="top", fontsize=7.5,
             bbox=dict(fc="white", ec="0.8", alpha=0.9))
except Exception as e:
    axB.text(0.5, 0.5, f"(hero data unavailable)\n{e}", ha="center", fontsize=7)

# ---- Panel C: regime map
axC = fig.add_subplot(gs[0, 2])
# axes: x = V/E_kin (log), y = β = θ/γ (log)
axC.set_xscale("log"); axC.set_yscale("log")
axC.set_xlim(0.5, 500); axC.set_ylim(0.02, 3)
# two-scale wall (β≳1)
axC.axhspan(1.0, 3, color="#f2c8c8", alpha=0.6)
axC.text(1.5, 1.5, "two-scale wall  (β≳1)\nGalerkin→FDFD but needs full basis",
         fontsize=6.8, color="#7a2020")
# strong-modulation zone (needs registry adaptation)
axC.axvspan(30, 500, ymin=0.0, ymax=np.log(1/0.02)/np.log(3/0.02), color="#fde8c8", alpha=0.5)
axC.text(45, 0.03, "strong modulation:\nregistry-adapted\nbasis required",
         fontsize=6.8, color="#8a5a10")
# weak/efficient zone
axC.text(0.7, 0.03, "weak modulation:\nfew-band single-ref\nEXACT", fontsize=6.8,
         color="#1a5a1a")
# data points
def pt(x, y, label, mk, col):
    axC.plot(x, y, mk, color=col, ms=10, mec="k", mew=0.5)
    axC.annotate(label, (x, y), fontsize=7, xytext=(6, 5),
                 textcoords="offset points")
pt(86, 0.075, "asym 2° (this work)", "s", C_EA)
pt(86 * 4, 0.037, "asym 1°", "s", "#d68080")
pt(300, 2.2, "(7,1) 16°", "D", C_G)
pt(3, 0.15, "honeycomb-K\n(Λ₀₁≡0, weak)", "^", "#2ca02c")
# dissolution wall (annotation — orthogonal axis)
axC.text(0.6, 2.0, "dissolution wall (no common gap):\nthesis case — impossible "
         "at any N_b", fontsize=6.8, color="#404040",
         bbox=dict(fc="#eeeeee", ec="0.7"))
axC.set_xlabel("modulation strength  V/E_kin")
axC.set_ylabel("two-scale param  β = θ/γ")
axC.set_title("C  Regime map: where few-band\neigenvalue-exactness is possible",
              fontsize=9.5)

fig.suptitle("The exact photonic moiré continuum model, and the regimes of "
             "eigenvalue-exactness", fontsize=12.5, y=0.965)
fig.text(0.06, 0.015,
         "Exact model: Galerkin projection of the true supercell TM operator onto reference-Bloch states at the folded moiré momenta — "
         "H=⟨E|−∇²|E⟩, S=⟨E|ε_bl|E⟩; variational, converges to FDFD. | "
         "Square CML TM, asym bilayer r₁=0.20/r₂=0.10 ε=8.9. | "
         "Prerequisite: registry-common gap (isolation). Efficient few-band exactness needs β≲0.1 AND weak V/E_kin; strong V/E_kin needs registry-adapted frames.",
         fontsize=6.6, family="monospace", va="bottom")

fig.savefig(os.path.join(HERE, "fig_exact_model.pdf"))
fig.savefig(os.path.join(HERE, "fig_exact_model.png"), dpi=200)
print("saved fig_exact_model.{pdf,png}")
print("Galerkin (7,1) convergence:", list(zip(nbs, [f'{m:.1e}' for m in means])))
