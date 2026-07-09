#!/usr/bin/env python3
"""The corrected weak-coupling verdict (post adversarial-verification).

An adversarial audit of the momentum-space model found a TRANSPOSE BUG in the
Ṽ coupling (momentum_kp_moire.py:55 — strong sx-registry harmonic paired with
the wrong moiré-reciprocal axis). Fixing it, and Richardson-extrapolating the
FDFD ground truth (which is itself px16-under-resolved even at 1.6px rods),
collapses the previously-reported 2° ground residual +2.74e-3 to +1.8e-5:

A — RESIDUAL DECOMPOSITION: +2.74e-3 = transpose bug (+1.86e-3) + FDFD
    sub-pixel (−0.86e-3) + TRUE (+1.8e-5). The corrected model's ground ENERGY
    is essentially exact; 96% of the old residual was artifact.
B — STRUCTURAL non-exactness (the genuine limit): the single-band, X-only model
    cannot represent the X⊕X' manifold, so it BREAKS the symmetry-protected
    4-fold ground degeneracy (FDFD split 1.7e-10; model 1.17e-4 at 2°, 8.9e-7 at
    1°) and over-splits the miniband fine-structure (span 1.3×). The
    degeneracy-breaking → 0 as θ→0; the span over-split persists.
C — THE ERROR MADE VISIBLE: FDFD's lowest 8 states form a near-degenerate
    ground cluster (X⊕X' × C4, split ~0.2e-3); the single-valley model spreads
    them into an over-split ladder (~13× wider). This is the structural
    non-exactness in one picture.

(Methodological aside, in §10 not the figure: a naive r₂ rod-SIZE sweep is
unsafe because weak coupling ⟺ sub-pixel rods (rod = r₂·px); the FDFD ground is
px-under-resolved for r₂≲0.07 — Richardson px16→px32 moves it by ~1e-3.)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD, C_EA, C_G = "#222222", "#d62728", "#1f77b4"
FLOOR = 6e-4

fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.985, top=0.82, bottom=0.15)

# ---------- Panel A: residual decomposition (waterfall)
axA = fig.add_subplot(gs[0, 0])
parts = [("reported\n(buggy)", 2.74e-3, C_EA),
         ("−transpose\nbug", -1.86e-3, "#8a5a10"),
         ("−FDFD\nsub-pixel", -0.86e-3, C_G),
         ("TRUE\nresidual", None, "#2ca02c")]
cum = 0.0
xs = np.arange(len(parts))
for i, (lab, val, col) in enumerate(parts):
    if val is None:
        axA.bar(i, 1.76e-5 * 1e3, color=col, width=0.6)
        axA.text(i, 1.76e-5 * 1e3 + 0.05, "1.8e-5", ha="center", fontsize=7)
    elif i == 0:
        axA.bar(i, val * 1e3, color=col, width=0.6)
        axA.text(i, val * 1e3 + 0.05, "2.74e-3", ha="center", fontsize=7)
        cum = val
    else:
        axA.bar(i, val * 1e3, bottom=cum * 1e3, color=col, width=0.6, alpha=0.85)
        cum += val
axA.axhline(FLOOR * 1e3, color="0.5", ls=":", lw=1)
axA.text(2.3, FLOOR * 1e3 + 0.03, "FDFD floor", fontsize=6.5, color="0.4")
axA.set_xticks(xs, [p[0] for p in parts], fontsize=7.5)
axA.set_ylabel("2° ground residual  [×10⁻³ c/a]")
axA.set_title("A  96% of the reported residual was\nartifact — corrected ground E is EXACT",
              fontsize=9.5)
axA.set_ylim(0, 3.0)

# ---------- Panel B: structural non-exactness (degeneracy-breaking vs theta)
axB = fig.add_subplot(gs[0, 1])
thetas = [2.01, 1.01]
model_split = [1.17e-4, 8.89e-7]
fdfd_split = [1.7e-10, 2.1e-11]
axB.loglog(thetas, model_split, "o-", color=C_EA, lw=1.6, ms=9,
           label="model 4-fold split (breaks it)")
axB.loglog(thetas, fdfd_split, "s--", color=C_FD, lw=1.2, ms=7,
           label="FDFD (symmetry-exact)")
axB.axhspan(0, FLOOR, color="#eeeeee", alpha=0.9)
axB.axhline(FLOOR, color="0.5", ls=":", lw=1)
for t, m in zip(thetas, model_split):
    axB.annotate(f"{m:.1e}", (t, m), fontsize=7, xytext=(5, 5), textcoords="offset points")
axB.set_xlabel("twist angle θ  [deg]")
axB.set_ylabel("ground 4-fold degeneracy split  [c/a]")
axB.set_title("B  structural limit: single-valley model\nbreaks the X⊕X′ degeneracy (→0 as θ→0)",
              fontsize=9.5)
axB.legend(fontsize=7, loc="lower right")
axB.text(0.03, 0.03, "+ miniband over-split\nspan ≈1.3× (persists)",
         transform=axB.transAxes, fontsize=7, va="bottom",
         bbox=dict(fc="white", ec="0.7", alpha=0.9))

# ---------- Panel C: the fine-structure — model over-splits FDFD's near-
# degenerate ground cluster (the single-valley error made visible), 2°.
axC = fig.add_subplot(gs[0, 2])
fd = np.sort(np.load(os.path.join(HERE, "fdfd_xman_2deg.npz"))["freqs_xmanifold"])[:8]
ea = np.sort(np.load(os.path.join(HERE, "momentum_kp_m57.npz"))["pooled"])[:8]
off = ea[0] - fd[0]
ea = ea - off  # remove the (near-zero, corrected) rigid shift to compare structure
f0 = fd[0]
for f in fd:
    axC.hlines((f - f0) * 1e3, 0.6, 1.4, color=C_FD, lw=1.6)
for f in ea:
    axC.hlines((f - f0) * 1e3, 1.6, 2.4, color=C_EA, lw=1.6)
axC.set_xticks([1, 2], ["FDFD\nlowest 8", "model\n(−offset)"], fontsize=8)
axC.set_xlim(0.3, 2.7)
axC.set_ylabel("f − f₀  [×10⁻³ c/a]")
axC.set_title("C  the single-valley error, visible:\nmodel over-splits the ground cluster",
              fontsize=9.5)
axC.annotate("", xy=(0.5, (fd[7] - f0) * 1e3), xytext=(0.5, 0),
             arrowprops=dict(arrowstyle="<->", color=C_FD, lw=1))
axC.text(0.33, (fd[7] - f0) * 1e3 / 2, f"{(fd[7]-fd[0])*1e3:.2f}\n(≈degenerate)",
         fontsize=6.3, color=C_FD, va="center")
axC.annotate("", xy=(2.5, (ea[7] - f0) * 1e3), xytext=(2.5, 0),
             arrowprops=dict(arrowstyle="<->", color=C_EA, lw=1))
axC.text(2.52, (ea[7] - f0) * 1e3 / 2, f"{(ea[7]-ea[0])*1e3:.2f}\n({(ea[7]-ea[0])/(fd[7]-fd[0]):.0f}× wider)",
         fontsize=6.3, color=C_EA, va="center")

fig.suptitle("Corrected weak-coupling verdict (post adversarial-verification): the momentum "
             "model's ground ENERGY is exact, but the single-valley reduction is NOT "
             "eigenvalue-exact — it breaks the X⊕X′ symmetry", fontsize=10.5, y=0.965)
fig.savefig(os.path.join(HERE, "fig_weak_verdict.pdf"))
fig.savefig(os.path.join(HERE, "fig_weak_verdict.png"), dpi=200)
print("saved fig_weak_verdict.{pdf,png}")
pass
