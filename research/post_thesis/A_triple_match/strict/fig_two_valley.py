#!/usr/bin/env python3
"""The two-valley (X⊕X′) completion: the missing ingredient that lifts the §9
Galerkin plateau — a real path toward eigenvalue-exact EA↔FDFD.

A — the valley premise: the FDFD ground quadruplet is 2-at-X + 2-at-X′
    (valley_composition_2deg.npz), 4-fold degenerate to 1.7e-10. A single-X
    carrier model must miss half of it.
B — the plateau-lift + isolation: adding the X′ carrier takes the captured
    manifold 10→24 (=FDFD) and lifts the ground 0.37680→0.37154 (+6.8e-3→+1.5e-3).
    The cause is the VALLEY, not basis size: single-valley SATURATED going
    9→16 frames (+6.8→+6.5e-3), yet two-valley at 9 frames broke through.
C — the mechanism, isolated on the fast (7,1) cell: as gcut adds the X′-star
    shell (3→4), the ground 4-cluster split converges 2.09e-4 → 1.48e-4 toward
    FDFD's physical 1.25e-4.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C_FD, C_1V, C_2V = "#222222", "#1f77b4", "#d62728"
fd = np.sort(np.load(os.path.join(HERE, "fdfd_xman_2deg.npz"))["freqs_xmanifold"])

fig = plt.figure(figsize=(14, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.06, right=0.985, top=0.83, bottom=0.15)

# ---------- Panel A: valley premise (ground quadruplet = 2X + 2X')
axA = fig.add_subplot(gs[0, 0])
vc = np.load(os.path.join(HERE, "valley_composition_2deg.npz"))
wX, wXp = vc["wX"][:8], vc["wXp"][:8]
idx = np.arange(8)
axA.bar(idx - 0.2, wX, 0.4, color=C_FD, label="w_X  (peak (π,0))")
axA.bar(idx + 0.2, wXp, 0.4, color=C_2V, label="w_X′ (peak (0,π))")
axA.axvline(3.5, color="0.6", ls=":", lw=1)
axA.text(1.5, 0.72, "ground\nquadruplet", ha="center", fontsize=7.5, color="0.3")
axA.text(5.5, 0.72, "2nd\nquadruplet", ha="center", fontsize=7.5, color="0.3")
axA.set_xlabel("FDFD state index (from bottom)")
axA.set_ylabel("valley Fourier weight")
axA.set_title("A  the premise: ground quadruplet is\n2-at-X + 2-at-X′ (valley-mixed)",
              fontsize=9.5)
axA.legend(fontsize=7.5, loc="upper right"); axA.set_ylim(0, 0.9)

# ---------- Panel B: plateau-lift + isolation
axB = fig.add_subplot(gs[0, 1])
def bottom(p):
    g = np.sort(np.load(os.path.join(HERE, p))["freqs"]); w = g[(g >= 0.365) & (g <= 0.385)]
    d = np.load(os.path.join(HERE, p)); return (w[0] - fd[0]) * 1e3, len(w), int(d["n_kept"])
pts = [("1-valley\n9 fr", "grecip_mr3.npz", C_1V),
       ("1-valley\n16 fr", "grecip_mr4.npz", C_1V),
       ("2-VALLEY\n9 fr (b0)", "grecip_2deg_2v_mr3.npz", C_2V),
       ("2-VALLEY\n9 fr (b1)", "grecip_2deg_2v_mr3_b1.npz", C_2V)]
xs = np.arange(len(pts))
for i, (lab, p, c) in enumerate(pts):
    db, n, nk = bottom(p)
    axB.bar(i, db, 0.62, color=c, alpha=0.9 if "2-VALLEY" in lab else 0.5)
    axB.text(i, db + 0.1, f"{n}/24\nrank {nk//1000}k", ha="center", fontsize=6.6)
axB.axhline(0, color=C_FD, lw=1.2); axB.text(3.0, 0.15, "FDFD floor", fontsize=7, color=C_FD)
axB.set_xticks(xs, [p[0] for p in pts], fontsize=7.3)
axB.set_ylabel("ground Δ vs FDFD  [×10⁻³ c/a]")
axB.set_title("B  X′ lifts the §9 plateau 2.8–4.5× —\nit's the VALLEY, not basis size",
              fontsize=9.5)
axB.annotate("single-valley\nSATURATES\n(9→16 fr)", xy=(1, bottom(pts[1][1])[0]),
             xytext=(0.15, 4.3), fontsize=6.8, color=C_1V,
             arrowprops=dict(arrowstyle="->", color=C_1V, lw=0.8))
axB.text(2.5, 3.4, "count 10→24\n(=FDFD);\ndegeneracy\nnot yet\nrestored",
         fontsize=6.3, color="0.35", ha="center")
axB.set_ylim(0, 8.2)

# ---------- Panel C: (7,1) mechanism — X'-shell converges the ground split
axC = fig.add_subplot(gs[0, 2])
fd7 = np.sort(np.load(os.path.join(HERE, "fdfd_m7_px24.npy")))
fd7s = (fd7[:4].max() - fd7[:4].min()) * 1e4
gcuts, splits, means = [], [], []
for g, p in [(3, "galerkin_m7_nb8.npz"), (4, "galerkin_m7_g4nb8.npz"), (5, "galerkin_m7_g5nb8.npz")]:
    gg = np.sort(np.load(os.path.join(HERE, p))["freqs"])
    gcuts.append(g); splits.append((gg[:4].max() - gg[:4].min()) * 1e4)
axC.axhline(fd7s, color=C_FD, ls="--", lw=1.2, label=f"FDFD physical split {fd7s:.2f}")
axC.plot(gcuts, splits, "o-", color=C_2V, lw=1.6, ms=9)
axC.axvspan(3.5, 5.5, color="#fde8e8", alpha=0.5)
axC.text(4.5, max(splits) * 0.9, "X′-star\nshell in", ha="center", fontsize=7, color=C_2V)
axC.set_xlabel("plane-wave cutoff  gcut  (X′ enters at ≥4)")
axC.set_ylabel("(7,1) ground 4-cluster split  [×10⁻⁴]")
axC.set_title("C  mechanism (fast (7,1) cell): X′-shell\nconverges the split toward FDFD",
              fontsize=9.5)
axC.set_xticks([3, 4, 5]); axC.legend(fontsize=7.5, loc="upper right")

fig.suptitle("The two-valley (X⊕X′) completion corrects §9: the missing ingredient is a 2nd "
             "momentum carrier (X′), not real-space adaptation — it lifts the plateau 2.8–4.5× "
             "and recovers the count; full exactness needs X↔X′ coupling (§10.3)",
             fontsize=9.6, y=0.965)
fig.savefig(os.path.join(HERE, "fig_two_valley.pdf"))
fig.savefig(os.path.join(HERE, "fig_two_valley.png"), dpi=200)
print("saved fig_two_valley.{pdf,png}")
for lab, p, c in pts:
    db, n, nk = bottom(p); print(f"  {lab.replace(chr(10),' ')}: Δ+{db:.2f}e-3, {n}/24, rank {nk}")
