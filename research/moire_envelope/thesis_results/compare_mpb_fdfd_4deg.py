#!/usr/bin/env python3
"""Combined MPB vs FDFD comparison at 4° — mode index plot + eigenvalue ladders."""

import os
import numpy as np
import matplotlib.pyplot as plt

outdir = os.path.dirname(os.path.abspath(__file__))

# ── Load data ───────────────────────────────────────────────────
mpb = np.load(os.path.join(outdir, "supercell_4deg_mpb_64.npz"))
fdfd = np.load(os.path.join(outdir, "supercell_4deg_fdfd_64.npz"))

freqs_mpb = mpb["freqs_mpb"]
freqs_fdfd = fdfd["freqs_fdfd"]
n_mpb = len(freqs_mpb)
n_fdfd = len(freqs_fdfd)

theta_deg = float(mpb["theta_deg"])
m_idx = int(mpb["m"])
n_idx = int(mpb["n"])
px_cell = int(mpb["px_per_cell"])
sigma = float(fdfd["sigma_omega"])

print(f"MPB:  {n_mpb} modes, ω ∈ [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]")
print(f"FDFD: {n_fdfd} modes, ω ∈ [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]")

# ── Colours ─────────────────────────────────────────────────────
C_MPB = "#1f77b4"
C_FDFD = "#d62728"

# ── Figure: 3-panel  [ladder MPB | ladder FDFD | ω vs mode] ────
fig, axes = plt.subplots(
    1, 3, figsize=(12, 6),
    gridspec_kw={"width_ratios": [1, 1, 3], "wspace": 0.05},
)
ax_lad_mpb, ax_lad_fdfd, ax_mode = axes

# Shared y-limits
y_lo = 0
y_hi = max(freqs_mpb.max(), freqs_fdfd.max()) * 1.05

# ── Left: MPB eigenvalue ladder ────────────────────────────────
for f in freqs_mpb:
    ax_lad_mpb.plot([0.15, 0.85], [f, f], "-", color=C_MPB, lw=0.8, alpha=0.7)
ax_lad_mpb.set_xlim(0, 1)
ax_lad_mpb.set_xticks([])
ax_lad_mpb.set_ylim(y_lo, y_hi)
ax_lad_mpb.set_ylabel("Frequency  ω  [a / 2πc]")
ax_lad_mpb.set_title("MPB", color=C_MPB, fontweight="bold")
ax_lad_mpb.grid(True, alpha=0.3, axis="y")

# ── Centre: FDFD eigenvalue ladder ─────────────────────────────
for f in freqs_fdfd:
    ax_lad_fdfd.plot([0.15, 0.85], [f, f], "-", color=C_FDFD, lw=0.8, alpha=0.7)
ax_lad_fdfd.set_xlim(0, 1)
ax_lad_fdfd.set_xticks([])
ax_lad_fdfd.set_ylim(y_lo, y_hi)
ax_lad_fdfd.set_yticklabels([])
ax_lad_fdfd.set_title("FDFD", color=C_FDFD, fontweight="bold")
ax_lad_fdfd.grid(True, alpha=0.3, axis="y")

# ── Right: ω vs mode index ─────────────────────────────────────
idx_mpb = np.arange(1, n_mpb + 1)
idx_fdfd = np.arange(1, n_fdfd + 1)
ax_mode.plot(idx_mpb, freqs_mpb, "o-", ms=3, lw=0.8, color=C_MPB, label="MPB")
ax_mode.plot(idx_fdfd, freqs_fdfd, "s-", ms=3, lw=0.8, color=C_FDFD, label="FDFD")
ax_mode.set_xlabel("Mode index")
ax_mode.set_ylabel("Frequency  ω  [a / 2πc]")
ax_mode.set_title("Sorted eigenvalues")
ax_mode.set_xlim(0, max(n_mpb, n_fdfd) + 1)
ax_mode.set_ylim(y_lo, y_hi)
ax_mode.legend(fontsize=10)
ax_mode.grid(True, alpha=0.3)

# ── Suptitle ────────────────────────────────────────────────────
fig.suptitle(
    f"MPB vs FDFD  —  ({m_idx},{n_idx}) supercell,  "
    f"θ = {theta_deg:.2f}°,  {px_cell} px/cell,  σ = {sigma}",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()

figfile = os.path.join(outdir, "compare_mpb_fdfd_4deg.png")
fig.savefig(figfile, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Plot → {figfile}")

# ── Residual statistics ─────────────────────────────────────────
n_common = min(n_mpb, n_fdfd)
delta = freqs_fdfd[:n_common] - freqs_mpb[:n_common]
rel = np.where(freqs_mpb[:n_common] > 1e-12,
               np.abs(delta) / freqs_mpb[:n_common], 0.0)
print(f"\nResiduals (first {n_common} modes):")
print(f"  max |Δω|          = {np.max(np.abs(delta)):.6e}")
print(f"  mean |Δω|         = {np.mean(np.abs(delta)):.6e}")
print(f"  max  |Δω/ω_mpb|   = {np.max(rel):.4e}")
print(f"  mean |Δω/ω_mpb|   = {np.mean(rel):.4e}")
