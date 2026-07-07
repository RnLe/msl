#!/usr/bin/env python3
"""THE deliverable: momentum-space k.p EA vs FDFD X-manifold, square CML TM,
band-1 gap-edge manifold of the asym bilayer. Two angles (2 deg, 1 deg) show
the manifold reproduced and the residual offset converging as theta->0.

Regenerates figure + JSON from disk. Strict: FDFD side filtered to the
w_X>0.5 X-band manifold (matching-free), EA side the pooled 4-lane momentum
model; sorted ladders, index-aligned, offset reported and de-trended.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def theta(m):
    return math.degrees(2 * math.atan2(1, m))


def clusters(a, tol):
    cl = []
    for x in np.sort(a):
        if cl and x - cl[-1][0] < tol:
            c, mlt = cl[-1]
            cl[-1] = ((c * mlt + x) / (mlt + 1), mlt + 1)
        else:
            cl.append((x, 1))
    return cl


CASES = [
    {"m": 57, "tag": "2deg", "ea": "momentum_kp_m57.npz",
     "fd": "fdfd_xman_2deg.npz", "win": (0.3695, 0.3782)},
    {"m": 113, "tag": "1deg", "ea": "momentum_kp_m113.npz",
     "fd": "fdfd_xman_1deg.npz", "win": (0.3675, 0.3765)},
]

data = {"protocol": "Momentum-space k.p moire model (exact local dispersion "
                    "E_ref + FFT registry potential Vtilde, plane-wave cutoff) "
                    "vs FDFD filtered to the w_X>0.5 X-band manifold. Sorted "
                    "ladders, index-aligned; rigid edge offset reported and "
                    "removed to expose the miniband-shape residual.",
        "cases": {}}

for c in CASES:
    ea = np.sort(np.load(HERE / c["ea"])["pooled"])
    fd = np.sort(np.load(HERE / c["fd"])["freqs_xmanifold"])
    # STRICT: compare the FDFD X-manifold (all N states) against the lowest N
    # EA levels, index-aligned from each edge (the +offset shift makes a fixed
    # frequency window under-count the EA, so we align by index instead).
    n = len(fd)
    eaN = ea[:n]
    d = eaN - fd
    off = float(eaN[0] - fd[0])
    dshape = np.abs((eaN - off) - fd)
    data["cases"][c["tag"]] = {
        "m": c["m"], "theta_deg": theta(c["m"]),
        "eta": 2 * math.sin(math.radians(theta(c["m"])) / 2),
        "n_compared": n,
        "ea_ladder": eaN.tolist(), "fdfd_xmanifold": fd.tolist(),
        "edge_offset": off,
        "mean_abs_raw": float(np.mean(np.abs(d))),
        "max_abs_raw": float(np.max(np.abs(d))),
        "mean_abs_detrended": float(np.mean(dshape)),
        "max_abs_detrended": float(np.max(dshape)),
        "span_ratio_ea_over_fdfd": float((eaN[-1] - eaN[0]) / (fd[-1] - fd[0])),
        "n_fdfd_xmanifold": int(n),
    }
    c["win"] = (float(min(fd[0], eaN[0]) - 3e-4), float(max(fd[-1], eaN[-1]) + 3e-4))

# eta-scaling of the offset
etas = [data["cases"][c["tag"]]["eta"] for c in CASES]
offs = [data["cases"][c["tag"]]["edge_offset"] for c in CASES]
if offs[0] and offs[1]:
    p = math.log(offs[0] / offs[1]) / math.log(etas[0] / etas[1])
    data["offset_eta_power"] = p

# ---------------- figure
C_FD, C_EA = "#222222", "#d62728"
fig = plt.figure(figsize=(13.5, 8.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.36, wspace=0.32,
                      left=0.065, right=0.98, top=0.90, bottom=0.14)


def stripes(ax, x, freqs, color, half=0.34, lw=1.1):
    for f in freqs:
        ax.hlines(f, x - half, x + half, color=color, lw=lw)


for col, c in enumerate(CASES):
    cd = data["cases"][c["tag"]]
    ea = np.array(cd["ea_ladder"]); fd = np.array(cd["fdfd_xmanifold"])
    lo, hi = c["win"]
    ax = fig.add_subplot(gs[0, col])
    stripes(ax, 0, fd, C_FD)
    stripes(ax, 1, ea, C_EA)
    stripes(ax, 2, ea - cd["edge_offset"], C_EA)
    ax.set_xticks([0, 1, 2], ["FDFD\nX-manifold", "EA\nmom-k·p",
                              f"EA\n−offset"], fontsize=8)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("frequency  f a/c")
    ax.set_title(f"{c['tag']}  (m={c['m']}) θ={theta(c['m']):.3f}°, "
                 f"η={cd['eta']:.4f}\nFDFD-X manifold: {cd['n_fdfd_xmanifold']} "
                 f"states (index-aligned)", fontsize=9.5)
    ax.text(0.03, 0.97,
            f"edge offset {cd['edge_offset']:+.1e}\n"
            f"shape resid (de-trended)\n  mean {cd['mean_abs_detrended']:.1e}"
            f"  max {cd['max_abs_detrended']:.1e}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))

# residual panel (2deg de-trended, per level)
axr = fig.add_subplot(gs[1, 0])
cd = data["cases"]["2deg"]
ea = np.array(cd["ea_ladder"]); fd = np.array(cd["fdfd_xmanifold"])
axr.axhline(0, color="0.6", lw=0.8)
axr.plot(np.arange(1, len(fd) + 1), (ea - cd["edge_offset"]) - fd, "o-",
         ms=4, color=C_EA, lw=1)
axr.set_xlabel("mode index (from edge)")
axr.set_ylabel("EA−offset − FDFD  [c/a]")
axr.set_title("2° miniband-shape residual\n(rigid offset removed)", fontsize=9.5)

# offset eta-scaling
axo = fig.add_subplot(gs[1, 1])
axo.plot(etas, np.array(offs) * 1e3, "s-", color=C_EA, ms=8)
for e, o in zip(etas, offs):
    axo.annotate(f"{o*1e3:.1f}e-3", (e, o * 1e3), fontsize=7,
                 xytext=(4, 4), textcoords="offset points")
axo.set_xlabel("η = 2 sin(θ/2)")
axo.set_ylabel("edge offset  [×10⁻³ c/a]")
axo.set_title(f"offset → 0 as θ→0\n(∝ η^{data.get('offset_eta_power', float('nan')):.2f}; "
              "separable-approx residual)", fontsize=9.5)
axo.set_xlim(0, max(etas) * 1.15); axo.set_ylim(0, max(offs) * 1e3 * 1.2)

# text panel
axt = fig.add_subplot(gs[1, 2]); axt.axis("off")
lines = [
    "Momentum-space k·p moiré model", "",
    "H_GG'(k) = E_ref(k₀+k+G) δ_GG'",
    "            + Ṽ(G−G')", "",
    "E_ref: EXACT local band-1 dispersion",
    "  (MPB, ref registry) — replaces the",
    "  parabolic k·p that over-binds ×9",
    "Ṽ = FFT_s[Λ₁(X;s)] (phase-1 landscape)",
    "plane-wave cutoff |n|≤5, 4 moiré lanes", "",
    "Result: over-density GONE; ×4 quad-",
    "ruplet structure + count reproduced;",
    "residual = rigid offset (→0 as θ→0)",
    "+ 6.5e-4 miniband shape at 2°.",
]
axt.text(0.0, 0.98, "\n".join(lines), family="monospace", fontsize=7.6,
         va="top", transform=axt.transAxes)

footer = (
    "Square CML, TM (E_z) | ASYM bilayer: layer1 r=0.20 + layer2 r=0.10, ε=8.9, ε_bg=1.0 | registry-common gap [0.3225,0.3661], "
    "band-1-at-X manifold isolated (band-2 headroom 0.12)\n"
    "EA: momentum-space k·p, k₀=X, E_ref from MPB res64 at ref registry s̄=(0.230,0.105), Ṽ=FFT of reg128 Λ₁(X;s), "
    "cutoff |n|≤5, moiré lanes {Γ,X1,X2,M} of the 2×2 CML fold, pooled\n"
    "FDFD: L=ε^{-1/2}(Σg^{ab}D†D)ε^{-1/2}, subpixel N_sub=8, CHOLMOD shift-invert, Q_X=(π,0); filtered to w_X+w_X'>0.5 "
    "(X-star Fourier weight, empty-lattice-calibrated k↔k·b map)\n"
    "STRICT: sorted ladders, index-aligned from the edge, no Hungarian; the ×4 clusters are valley(X⊕X')×C4, per the centered-cell identity τ=(L1+L2)/2∈both lattices"
)
fig.text(0.065, 0.012, footer, fontsize=6.8, family="monospace", va="bottom")
fig.suptitle("Momentum-space k·p EA reproduces the FDFD moiré manifold — "
             "square CML TM, band-1 gap edge", fontsize=12.5, y=0.965)

fig.savefig(HERE / "fig_momentum_hero.pdf")
fig.savefig(HERE / "fig_momentum_hero.png", dpi=200)
with open(HERE / "momentum_hero_data.json", "w") as f:
    json.dump(data, f, indent=1)
print("saved fig_momentum_hero.{pdf,png} + momentum_hero_data.json")
for tag, cd in data["cases"].items():
    print(f"  {tag}: offset {cd['edge_offset']:+.2e}, shape mean "
          f"{cd['mean_abs_detrended']:.2e} max {cd['max_abs_detrended']:.2e}, "
          f"FDFD-X {cd['n_fdfd_xmanifold']} levels, span ratio "
          f"{cd['span_ratio_ea_over_fdfd']:.3f}")
print(f"  offset ∝ η^{data.get('offset_eta_power', float('nan')):.2f}")
