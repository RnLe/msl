#!/usr/bin/env python3
"""Does the two-valley Galerkin converge to eigenvalue-exact EA↔FDFD?

A — (7,1)/16.3° valley-complete band ladder (galerkin_moire, gcut=4): the edge
    offset vs FDFD falls MONOTONICALLY to +3.4e-5 at 16 bands — the variational
    method demonstrably reaches the FDFD floor at a tractable cell.
B — 2° two-valley (nref=3): the in-window band-1 bottom vs the well-conditioned
    basis rank (varying gcut & the canonical-orthogonalization tol). At matched
    RANK, gcut=5 beats gcut=4 (▽ below the gcut=4 curve) → the plane-wave cutoff
    genuinely adds convergence. BUT the fixed-frame reciprocal basis grows
    ill-conditioned: the clean rank caps ~4900 (spurious sub-floor states appear
    beyond), so the best CLEAN bottom is +1.5e-3 — a conditioning floor, not a
    completeness wall. Efficient 2° exactness needs a better-conditioned (real-
    space registry-adapted) two-valley basis (§9's program + the valley).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD, C_2V, C_SP = "#222222", "#1f77b4", "#d62728"

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.15, wspace=0.28)

# ---- A: (7,1) band ladder
fd7 = np.sort(np.load(os.path.join(HERE, "fdfd_m7_px24.npy")))
nb7, edge7 = [], []
for nb in [2, 4, 6, 8, 10, 12, 16]:
    p = os.path.join(HERE, f"galerkin_m7_g4nb{nb}.npz")
    if not os.path.isfile(p):
        p = os.path.join(HERE, f"galerkin_m7_nb{nb}.npz")
    if os.path.isfile(p):
        g = np.sort(np.load(p)["freqs"]); nb7.append(nb); edge7.append(abs(g[0] - fd7[0]))
ax[0].semilogy(nb7, edge7, "o-", color=C_2V, lw=1.7, ms=8)
ax[0].axhspan(0, 6e-4, color="#eee", alpha=0.9)
ax[0].axhline(6e-4, color="0.5", ls=":", lw=1)
ax[0].text(2.2, 7e-4, "px16-scale floor", fontsize=7, color="0.4")
ax[0].annotate(f"+{edge7[-1]:.1e}\n(≈ exact)", (nb7[-1], edge7[-1]), fontsize=7.5,
               xytext=(-6, 14), textcoords="offset points", color=C_2V,
               arrowprops=dict(arrowstyle="->", color=C_2V, lw=0.8))
ax[0].set_xlabel("retained bands  N_b")
ax[0].set_ylabel("(7,1) edge |Δf| vs FDFD  [c/a]")
ax[0].set_title("A  (7,1) valley-complete ladder → FDFD floor\n"
                "(the method reaches eigenvalue-exactness)", fontsize=9.5)
ax[0].set_xticks(nb7)

# ---- B: 2° rank vs bottom (gcut & s_tol)
FLOOR = 0.370047
# (gcut, s_tol, rank, bottomΔ×1e3, clean)
pts = [(4, "1e-6", 4921, 1.49, True), (4, "1e-5", 3165, 4.50, True),
       (4, "1e-4", 1804, 8.57, True), (5, "1e-4", 2675, 2.03, True),
       (5, "1e-5", 4653, -2.13, False)]
g4 = [(r, d) for gc, t, r, d, c in pts if gc == 4 and c]
g5c = [(r, d) for gc, t, r, d, c in pts if gc == 5 and c]
g5s = [(r, d) for gc, t, r, d, c in pts if gc == 5 and not c]
g4 = sorted(g4)
ax[1].plot([r for r, d in g4], [d for r, d in g4], "o-", color=C_2V, lw=1.6, ms=8,
           label="gcut=4 (clean)")
ax[1].plot([r for r, d in g5c], [d for r, d in g5c], "D", color="#2ca02c", ms=11,
           mec="k", mew=0.5, label="gcut=5 (clean) — below gcut=4 at matched rank")
for r, d in g5s:
    ax[1].plot(r, d, "x", color=C_SP, ms=11, mew=2, label="gcut=5 SPURIOUS (sub-floor)")
ax[1].axhline(0, color=C_FD, lw=1.2); ax[1].text(1900, 0.2, "FDFD floor", fontsize=7, color=C_FD)
ax[1].axhline(1.49, color="0.6", ls="--", lw=1)
ax[1].text(3400, 1.65, "best clean +1.5e-3\n(conditioning floor)", fontsize=6.8, color="0.35")
ax[1].annotate("gcut helps per rank\n(convergence)", (2675, 2.03), fontsize=6.8,
               color="#2ca02c", xytext=(2750, 4.5),
               arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8))
ax[1].set_xlabel("well-conditioned basis rank  (via gcut & s_tol)")
ax[1].set_ylabel("2° band-1 bottom Δ vs FDFD  [×10⁻³ c/a]")
ax[1].set_title("B  2° two-valley: gcut converges per rank, but the\n"
                "fixed-frame basis is conditioning-limited (+1.5e-3)", fontsize=9.5)
ax[1].legend(fontsize=6.6, loc="upper right")
ax[1].set_ylim(-3.5, 9.5)

fig.suptitle("The two-valley Galerkin is variational-convergent — exact at (7,1); at 2° the "
             "valley lifts the plateau 4.5×, with a residual fixed-frame conditioning floor",
             fontsize=9.8, y=0.965)
fig.savefig(os.path.join(HERE, "fig_exactness_ladder.pdf"))
fig.savefig(os.path.join(HERE, "fig_exactness_ladder.png"), dpi=200)
print("saved fig_exactness_ladder.{pdf,png}")
print("(7,1) edge ladder:", [f"nb{n}:{e:.1e}" for n, e in zip(nb7, edge7)])
