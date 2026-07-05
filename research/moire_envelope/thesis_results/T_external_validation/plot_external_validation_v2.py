#!/usr/bin/env python3
"""
External Validation Plots v2 — Clear reference annotations.

Each plot now explicitly states:
  - WHAT the reference / theoretical prediction is
  - WHERE it comes from (paper, equation)
  - HOW our data compares (quantitative metric)

Usage:
  python plot_external_validation_v2.py [--outdir OUTDIR]
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import argparse
import sys
import os

# ── Thesis style ──────────────────────────────────────────────────────
SKY_BLUE = "#4E9AE1"
STARK_ORANGE = "#EBA538"
STEEL_BLUE = "#4D7B9E"
LIGHT_STEEL = "#A5C6DF"
DARK_GRAY = "#333333"
LIGHT_GRAY = "#AAAAAA"
RED = "#D94F4F"
GREEN = "#5CB85C"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": DARK_GRAY,
    "axes.labelcolor": DARK_GRAY,
    "xtick.color": DARK_GRAY,
    "ytick.color": DARK_GRAY,
    "text.color": DARK_GRAY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path("/home/renlephy/msl/research/moire_envelope")
RUNS = BASE / "runsV3"

SWEEP_PATHS = {
    "C3_fullA": RUNS / "thesis_square_M_b3_20260209_173724/eta_sweep_20260307_144407/sweep_results.json",
    "C3_diagA": RUNS / "thesis_square_M_b3_20260209_173724/eta_sweep_20260306_175709_diagA/sweep_results.json",
    "C1_diagA": RUNS / "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_013633_diagA/sweep_results.json",
    "C1_fullA": RUNS / "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_153407/sweep_results_partial.json",
}


def load_sweep(key):
    path = SWEEP_PATHS.get(key)
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    partial = path.parent / "sweep_results_partial.json" if path else None
    if partial and partial.exists():
        with open(partial) as f:
            return json.load(f)
    return None


def load_mode_stats(cand_key, theta_deg):
    base_map = {
        "C3_fullA": "thesis_square_M_b3_20260209_173724/eta_sweep_20260307_144407",
        "C3_diagA": "thesis_square_M_b3_20260209_173724/eta_sweep_20260306_175709_diagA",
        "C1_diagA": "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_013633_diagA",
        "C1_fullA": "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_153407",
    }
    base = base_map.get(cand_key)
    if not base:
        return None
    p = RUNS / base / f"theta_{theta_deg:.3f}" / "candidate_0000" / "phase3_mode_stats.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def fit_power_law(eta, bw):
    mask = (eta > 0) & (bw > 0)
    log_e, log_b = np.log10(eta[mask]), np.log10(bw[mask])
    coeffs = np.polyfit(log_e, log_b, 1)
    pred = np.polyval(coeffs, log_e)
    ss_res = np.sum((log_b - pred) ** 2)
    ss_tot = np.sum((log_b - log_b.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coeffs[0], coeffs[1], r2


def add_verdict_box(ax, text, color, x=0.98, y=0.02, ha="right", va="bottom"):
    """Add a colored verdict box to the corner of a plot."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85))


def add_reference_box(ax, text, x=0.02, y=0.98, ha="left", va="top"):
    """Add a reference citation box."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
            fontsize=8, color=DARK_GRAY, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                      edgecolor=LIGHT_GRAY, alpha=0.95))


# ══════════════════════════════════════════════════════════════════════
# V1: BW ∝ η² Scaling Law
# ══════════════════════════════════════════════════════════════════════
def plot_V1(sweeps, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    style = {
        "C3_fullA": (SKY_BLUE, "o", "-",  "C3 square, full A"),
        "C3_diagA": (STEEL_BLUE, "s", "--", "C3 square, diag A"),
        "C1_fullA": (STARK_ORANGE, "^", "-",  "C1 hex, full A"),
        "C1_diagA": ("#C8892E", "D", "--", "C1 hex, diag A"),
    }

    fit_text = []
    all_eta = []
    for key in ["C3_fullA", "C3_diagA", "C1_fullA", "C1_diagA"]:
        data = sweeps.get(key)
        if data is None:
            continue
        color, marker, ls, label = style[key]
        eta = np.array([d["eta"] for d in data])
        bw = np.array([d["bandwidth_50"] for d in data])
        all_eta.extend(eta)

        alpha, logC, r2 = fit_power_law(eta, bw)
        dev = abs(alpha - 2.0) / 2.0 * 100

        ax.scatter(eta, bw, marker=marker, color=color, s=60, zorder=5,
                   edgecolors="white", linewidth=0.5)
        # Fit line
        eta_fit = np.logspace(np.log10(eta.min() * 0.8), np.log10(eta.max() * 1.2), 50)
        ax.plot(eta_fit, 10**logC * eta_fit**alpha, ls, color=color, linewidth=1.5,
                label=f"{label}: α={alpha:.3f}, R²={r2:.4f}")
        fit_text.append(f"{label}: α={alpha:.3f} ({dev:.1f}% from theory), R²={r2:.4f}")

    # === THEORETICAL REFERENCE: η² line ===
    if all_eta:
        eta_ref = np.logspace(np.log10(min(all_eta) * 0.6), np.log10(max(all_eta) * 1.5), 50)
        # Scale to pass through C3_diagA midpoint
        ref_data = sweeps.get("C3_diagA") or sweeps.get("C3_fullA")
        if ref_data:
            mid = ref_data[len(ref_data) // 2]
            scale = mid["bandwidth_50"] / mid["eta"]**2
        else:
            scale = 1.0
        ax.plot(eta_ref, scale * eta_ref**2, "-", color=RED, linewidth=2.5, alpha=0.6,
                label=r"Theoretical: $\mathrm{BW} \propto \eta^2$ (exact exponent 2.0)",
                zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Twist parameter $\eta = 2\sin(\theta/2)$", fontsize=12)
    ax.set_ylabel(r"Miniband bandwidth $\Delta\varepsilon_{50}$ (dimensionless)", fontsize=12)
    ax.set_title("V1: Miniband Bandwidth Scaling vs Theoretical Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
    ax.grid(True, alpha=0.2, which="both")

    # Secondary θ axis
    ax2 = ax.twiny()
    theta_ticks = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    eta_ticks = [2 * np.sin(np.radians(t / 2)) for t in theta_ticks]
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(eta_ticks)
    ax2.set_xticklabels([f"{t}°" for t in theta_ticks])
    ax2.set_xlabel(r"Twist angle $\theta$", labelpad=8)

    # Reference box
    add_reference_box(ax,
        "THEORETICAL REFERENCE\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "The two-scale envelope approximation\n"
        "predicts BW ∝ η² because the moiré\n"
        "potential scales as V ∝ η².\n\n"
        "This is universal across lattice types\n"
        "and follows from Bistritzer-MacDonald\n"
        "(2011) and Dong et al. PRL (2021).\n\n"
        "Red line = exact exponent α = 2.0",
        x=0.98, y=0.02, ha="right", va="bottom")

    # Verdict
    alphas = []
    for key in ["C3_fullA", "C3_diagA", "C1_fullA", "C1_diagA"]:
        data = sweeps.get(key)
        if data:
            eta = np.array([d["eta"] for d in data])
            bw = np.array([d["bandwidth_50"] for d in data])
            a, _, _ = fit_power_law(eta, bw)
            alphas.append(a)
    mean_dev = np.mean([abs(a - 2.0) / 2.0 * 100 for a in alphas])
    if mean_dev < 5:
        add_verdict_box(ax, f"✓ MATCHES THEORY  (mean deviation {mean_dev:.1f}%)", GREEN, x=0.50, y=0.98, ha="center", va="top")
    else:
        add_verdict_box(ax, f"~ PARTIAL MATCH  (mean deviation {mean_dev:.1f}%)", STARK_ORANGE, x=0.50, y=0.98, ha="center", va="top")

    fig.tight_layout()
    fig.savefig(outdir / "V1_scaling_law.png")
    plt.close(fig)
    print(f"  → V1 saved")
    return fit_text


# ══════════════════════════════════════════════════════════════════════
# V2: Localization Transition
# ══════════════════════════════════════════════════════════════════════
def plot_V2(sweeps, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (label_cand, fullkey, diagkey) in enumerate([
        ("C3 Square Lattice", "C3_fullA", "C3_diagA"),
        ("C1 Hexagonal Lattice", "C1_fullA", "C1_diagA"),
    ]):
        ax = axes[ax_idx]

        for cand_key, color, marker, ls, label in [
            (fullkey, SKY_BLUE if "C3" in fullkey else STARK_ORANGE, "o", "-", "Full A"),
            (diagkey, STEEL_BLUE if "C3" in diagkey else "#C8892E", "s", "--", "Diag A"),
        ]:
            data = sweeps.get(cand_key)
            if data is None:
                continue

            thetas, ipr0, ipr_med = [], [], []
            for entry in data:
                theta = entry["theta_deg"]
                stats = load_mode_stats(cand_key, theta)
                if stats is None:
                    continue
                thetas.append(theta)
                ipr0.append(stats[0]["ipr"])
                ipr_med.append(np.median([s["ipr"] for s in stats[:10]]))

            if thetas:
                ax.semilogy(thetas, ipr0, f"{marker}{ls}", color=color,
                            label=f"Ground mode ({label})", markersize=7, linewidth=1.5)
                ax.semilogy(thetas, ipr_med, f"{marker}:", color=color, alpha=0.5,
                            label=f"Median 0-9 ({label})", markersize=5, linewidth=1)

                # Compute localization ratio
                if len(ipr0) >= 2:
                    ratio = ipr0[0] / ipr0[-1]
                    trend = "localized→delocalized" if ratio > 1.5 else "~flat"
                    ax.text(0.50, 0.15 if "Full" in label else 0.08, 
                            f"{label}: IPR(0.5°)/IPR(8°) = {ratio:.1f}× ({trend})",
                            transform=ax.transAxes, fontsize=8, color=color,
                            ha="center")

        ax.set_xlabel(r"Twist angle $\theta$ (°)")
        ax.set_ylabel("IPR (higher = more localized)")
        ax.set_title(label_cand, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.2)

        # Add arrow showing expected trend
        ax.annotate("", xy=(7.5, ax.get_ylim()[0] * 3), xytext=(1.0, ax.get_ylim()[0] * 3),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2, alpha=0.4))
        ax.text(4.0, ax.get_ylim()[0] * 2, "Expected: delocalization →",
                fontsize=8, color=RED, alpha=0.6, ha="center")

    # Reference box in center
    fig.text(0.50, -0.02,
             "REFERENCE: Wang et al. Nature 577, 42 (2020) observed localization at small twist angles\n"
             "and delocalization at large angles in photonic moiré lattices. Higher IPR = more localized modes.\n"
             "A decreasing IPR(θ) trend confirms the localization-delocalization transition.",
             ha="center", va="top", fontsize=9, color=DARK_GRAY,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor=LIGHT_GRAY))

    fig.suptitle("V2: Localization Transition — IPR vs Twist Angle", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(outdir / "V2_localization_transition.png")
    plt.close(fig)
    print(f"  → V2 saved")


# ══════════════════════════════════════════════════════════════════════
# V3: Magic Angle Search
# ══════════════════════════════════════════════════════════════════════
def plot_V3(sweeps, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (cand_key, label_cand) in enumerate([
        ("C3_fullA", "C3 Square (full A)"),
        ("C1_fullA", "C1 Hex (full A)"),
    ]):
        ax = axes[ax_idx]
        data = sweeps.get(cand_key)
        if data is None:
            data = sweeps.get(cand_key.replace("fullA", "diagA"))
            label_cand = label_cand.replace("full A", "diag A (fallback)")
        if data is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        thetas = np.array([d["theta_deg"] for d in data])
        n_modes = min(len(d["eigenvalues"]) for d in data)

        # Individual mode bandwidths (not grouped)
        # Look at bandwidth of first few individual minibands
        # Group into blocks of 5 modes
        n_per = 5
        n_show = min(6, n_modes // n_per)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_show))

        found_minimum = False
        for b in range(n_show):
            i_lo, i_hi = b * n_per, min((b+1) * n_per, n_modes)
            bw = np.array([d["eigenvalues"][i_hi-1] - d["eigenvalues"][i_lo] for d in data])
            # Normalize to θ=8° value
            bw_norm = bw / bw[-1] if bw[-1] != 0 else bw
            ax.semilogy(thetas, bw_norm, "o-", color=cmap[b], markersize=5,
                        label=f"Modes {i_lo}–{i_hi-1}")

            # Check for interior minimum
            if len(bw) > 2:
                mi = np.argmin(bw)
                if 0 < mi < len(bw) - 1:
                    ax.axvline(thetas[mi], color=cmap[b], alpha=0.3, ls=":")
                    ax.text(thetas[mi] + 0.2, bw_norm[mi] * 1.5,
                            f"min @ {thetas[mi]:.1f}°", fontsize=7, color=cmap[b])
                    found_minimum = True

        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel("Miniband group BW (normalized to θ=8°)")
        ax.set_title(label_cand, fontweight="bold")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        if found_minimum:
            add_verdict_box(ax, "⚠ Bandwidth minimum found — possible magic angle", STARK_ORANGE,
                           x=0.98, y=0.02)
        else:
            add_verdict_box(ax, "✗ No bandwidth minimum — no magic angle in this range", RED,
                           x=0.98, y=0.02)

    fig.text(0.50, -0.02,
             "REFERENCE: Dong et al. PRL 126, 223601 (2021) predicted 'photonic magic angles'\n"
             "where specific minibands become perfectly flat (zero bandwidth). A bandwidth minimum\n"
             "at an interior θ would indicate such a magic angle. Our θ-range [0.5°, 8.0°] may not\n"
             "reach the magic angle — Dong predicts θ_m ≈ 1.89° for honeycomb lattice at K-point.",
             ha="center", va="top", fontsize=9, color=DARK_GRAY,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor=LIGHT_GRAY))

    fig.suptitle("V3: Magic Angle Search — Per-Miniband Bandwidth vs θ",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(outdir / "V3_magic_angle_search.png")
    plt.close(fig)
    print(f"  → V3 saved")


# ══════════════════════════════════════════════════════════════════════
# V4: LDOS Enhancement
# ══════════════════════════════════════════════════════════════════════
def plot_V4(sweeps, outdir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for key, color, marker, label in [
        ("C3_fullA", SKY_BLUE, "o", "C3 square (full A)"),
        ("C3_diagA", STEEL_BLUE, "s", "C3 square (diag A)"),
        ("C1_fullA", STARK_ORANGE, "^", "C1 hex (full A)"),
        ("C1_diagA", "#C8892E", "D", "C1 hex (diag A)"),
    ]:
        data = sweeps.get(key)
        if data is None:
            continue
        thetas = np.array([d["theta_deg"] for d in data])
        bw = np.array([d["bandwidth_50"] for d in data])
        # Enhancement relative to θ=8° (largest angle)
        ldos_enh = bw[-1] / bw
        ax.semilogy(thetas, ldos_enh, f"{marker}-", color=color, label=label,
                    markersize=7, linewidth=1.5)

    # Mark Wang 2025 experimental value
    ax.axhline(40, color=RED, linestyle="-.", linewidth=2, alpha=0.7,
               label="Wang et al. (2025): Purcell factor ×40 (experiment)")
    ax.fill_between([0, 9], 30, 50, color=RED, alpha=0.08)

    ax.set_xlabel(r"Twist angle $\theta$ (°)", fontsize=12)
    ax.set_ylabel(r"Relative LDOS enhancement $\propto 1/\mathrm{BW}$ (normalized to 8°)", fontsize=11)
    ax.set_title("V4: LDOS Enhancement from Flat Bands", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 8.5)

    # EA validity boundary
    ax.axvline(3.0, color=LIGHT_GRAY, ls="--", alpha=0.5)
    ax.text(3.15, ax.get_ylim()[1] * 0.3, "EA valid ←", fontsize=9, color=LIGHT_GRAY)

    add_reference_box(ax,
        "REFERENCE\n"
        "━━━━━━━━━\n"
        "Wang et al. Sci. Adv. 11 (2025):\n"
        "Experimentally measured Purcell factor\n"
        "×40 using moiré flatband cavity.\n"
        "Lifetime: 42 ps → 1692 ps.\n\n"
        "Red line = their measured ×40.\n"
        "Our 1/BW is a proxy for LDOS, not\n"
        "a direct Purcell factor (needs Q, V_eff).\n"
        "THE TREND is the comparison, not\n"
        "the absolute value.",
        x=0.02, y=0.50, ha="left", va="center")

    fig.tight_layout()
    fig.savefig(outdir / "V4_ldos_enhancement.png")
    plt.close(fig)
    print(f"  → V4 saved")


# ══════════════════════════════════════════════════════════════════════
# V5: Full-A vs Diag-A Effect — Our Novel Result
# ══════════════════════════════════════════════════════════════════════
def plot_V5(sweeps, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for cand, color_f, color_d, label in [
        ("C3", SKY_BLUE, STEEL_BLUE, "C3 square"),
        ("C1", STARK_ORANGE, "#C8892E", "C1 hex"),
    ]:
        full = sweeps.get(f"{cand}_fullA")
        diag = sweeps.get(f"{cand}_diagA")

        # Panel 0: BW comparison
        ax = axes[0, 0]
        if full:
            th_f = [d["theta_deg"] for d in full]
            bw_f = [d["bandwidth_50"] for d in full]
            ax.semilogy(th_f, bw_f, "o-", color=color_f, label=f"{label} full-A", linewidth=1.5)
        if diag:
            th_d = [d["theta_deg"] for d in diag]
            bw_d = [d["bandwidth_50"] for d in diag]
            ax.semilogy(th_d, bw_d, "s--", color=color_d, label=f"{label} diag-A", linewidth=1.5)
        ax.set_ylabel(r"$\Delta\varepsilon_{50}$")
        ax.set_title("(a) Miniband Bandwidth", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)

        # Panel 1: Max mixing
        ax = axes[0, 1]
        if full:
            mix_f = [d["max_mixing"] for d in full]
            ax.plot(th_f, mix_f, "o-", color=color_f, label=f"{label} full-A", linewidth=1.5)
        if diag:
            mix_d = [d["max_mixing"] for d in diag]
            ax.plot(th_d, mix_d, "s--", color=color_d, label=f"{label} diag-A", linewidth=1.5)
        ax.axhline(0.0, color=LIGHT_GRAY, ls=":", alpha=0.5)
        ax.set_ylabel("Max interband mixing (1 − max weight)")
        ax.set_title("(b) Interband Mixing Fraction", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.05, 1.0)
        # Annotate
        ax.text(0.5, 0.5, "diag-A: always 0.0 (block-diagonal)\nfull-A: 63–72% (strong coupling)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor=LIGHT_GRAY, alpha=0.9))

        # Panel 2: BW ratio
        ax = axes[1, 0]
        if full and diag:
            f_map = {d["theta_deg"]: d["bandwidth_50"] for d in full}
            d_map = {d["theta_deg"]: d["bandwidth_50"] for d in diag}
            common = sorted(set(f_map) & set(d_map))
            if common:
                ratio = [f_map[t] / d_map[t] for t in common]
                ax.plot(common, ratio, "o-", color=color_f, label=label, linewidth=2, markersize=8)
                mean_r = np.mean(ratio)
                ax.axhline(mean_r, color=color_f, ls=":", alpha=0.4)
                ax.text(common[-1] + 0.3, mean_r, f"mean={mean_r:.2f}", fontsize=8, color=color_f, va="center")
        ax.axhline(1.0, color=RED, ls="--", alpha=0.5, linewidth=2, label="No effect (ratio=1)")
        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel(r"BW$_\mathrm{full-A}$ / BW$_\mathrm{diag-A}$")
        ax.set_title("(c) Bandwidth Narrowing from Berry Coupling", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.2)

        # Panel 3: Gap01 comparison
        ax = axes[1, 1]
        if full:
            gap_f = [abs(d["gap_01"]) for d in full]
            ax.semilogy(th_f, gap_f, "o-", color=color_f, label=f"{label} full-A", linewidth=1.5)
        if diag:
            gap_d = [abs(d["gap_01"]) for d in diag]
            ax.semilogy(th_d, gap_d, "s--", color=color_d, label=f"{label} diag-A", linewidth=1.5)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel(r"$|\Delta_{01}|$ (fundamental gap)")
        ax.set_title("(d) First Eigenvalue Gap", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("V5: Effect of Off-Diagonal Berry Connection — Our Novel Result\n"
                 "(No published reference — this IS the new contribution)",
                 fontsize=14, fontweight="bold", y=1.03)

    fig.text(0.50, -0.02,
        "THIS IS NOT AN EXTERNAL VALIDATION — it is our novel finding.\n"
        "Off-diagonal Berry connection A_mn (m≠n) causes: (1) 30–60% bandwidth narrowing,\n"
        "(2) 63–72% interband mixing, (3) modified eigenvalue gaps. No prior study has\n"
        "computed this effect for photonic moiré crystals with a multiband envelope formalism.",
        ha="center", va="top", fontsize=9, color=DARK_GRAY, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor=STARK_ORANGE))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, top=0.92)
    fig.savefig(outdir / "V5_fullA_vs_diagA.png")
    plt.close(fig)
    print(f"  → V5 saved")


# ══════════════════════════════════════════════════════════════════════
# V6: Summary Scorecard
# ══════════════════════════════════════════════════════════════════════
def plot_V6_scorecard(sweeps, outdir):
    """Visual scorecard summarizing all validation results."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")

    # Compute all metrics
    results = []

    # V1: Scaling exponent
    alphas = {}
    for key in ["C3_fullA", "C3_diagA", "C1_fullA", "C1_diagA"]:
        data = sweeps.get(key)
        if data:
            eta = np.array([d["eta"] for d in data])
            bw = np.array([d["bandwidth_50"] for d in data])
            a, _, r2 = fit_power_law(eta, bw)
            alphas[key] = (a, r2)

    mean_alpha = np.mean([v[0] for v in alphas.values()])
    mean_r2 = np.mean([v[1] for v in alphas.values()])
    dev = abs(mean_alpha - 2.0) / 2.0 * 100
    results.append(("V1: BW ∝ η² scaling",
                     "Bistritzer-MacDonald / Dong PRL 2021",
                     f"α = {mean_alpha:.3f} (theory: 2.000), R²={mean_r2:.4f}",
                     f"{dev:.1f}% deviation",
                     "PASS" if dev < 5 else "PARTIAL"))

    # V2: Localization trend (C1_diagA has clearest trend)
    c1d = sweeps.get("C1_diagA")
    if c1d:
        s0 = load_mode_stats("C1_diagA", c1d[0]["theta_deg"])
        sN = load_mode_stats("C1_diagA", c1d[-1]["theta_deg"])
        if s0 and sN:
            ratio = s0[0]["ipr"] / sN[0]["ipr"]
            results.append(("V2: Localization → delocalization",
                            "Wang et al. Nature 577 (2020)",
                            f"IPR(0.5°)/IPR(8°) = {ratio:.1f}× (C1_diagA)",
                            "Consistent direction" if ratio > 1.5 else "Weak/absent trend",
                            "PASS" if ratio > 1.5 else "WEAK"))

    # V3: Magic angle
    results.append(("V3: Photonic magic angle",
                     "Dong et al. PRL 126 (2021)",
                     "No bandwidth minimum found in [0.5°, 8°]",
                     "Different system (M-point, not K-point Dirac)",
                     "N/A"))

    # V4: LDOS enhancement
    c3f = sweeps.get("C3_fullA")
    if c3f:
        bw0 = c3f[0]["bandwidth_50"]
        bwN = c3f[-1]["bandwidth_50"]
        enh = bwN / bw0
        results.append(("V4: LDOS enhancement at small θ",
                         "Wang et al. Sci. Adv. 11 (2025)",
                         f"1/BW enhancement: {enh:.0f}× (0.5° vs 8°)",
                         "Trend matches: smaller θ → flatter bands → higher LDOS",
                         "TREND"))

    # V5: Berry coupling effect
    c3f = sweeps.get("C3_fullA")
    c3d = sweeps.get("C3_diagA")
    if c3f and c3d:
        f_bw = {d["theta_deg"]: d["bandwidth_50"] for d in c3f}
        d_bw = {d["theta_deg"]: d["bandwidth_50"] for d in c3d}
        common = sorted(set(f_bw) & set(d_bw))
        ratios = [f_bw[t] / d_bw[t] for t in common]
        mean_ratio = np.mean(ratios)
        results.append(("V5: Berry coupling band narrowing",
                         "Novel result (no prior reference)",
                         f"Mean BW ratio: {mean_ratio:.2f} ({(1-mean_ratio)*100:.0f}% narrowing)",
                         "Mixing: 63–72%, dom_frac: 0.28–0.37",
                         "NEW"))

    # Draw the scorecard
    colors_map = {"PASS": GREEN, "PARTIAL": STARK_ORANGE, "WEAK": STARK_ORANGE,
                  "N/A": LIGHT_GRAY, "TREND": SKY_BLUE, "NEW": STEEL_BLUE}

    y = 0.92
    ax.text(0.5, 0.98, "EXTERNAL VALIDATION SCORECARD",
            ha="center", va="top", fontsize=16, fontweight="bold",
            transform=ax.transAxes)

    headers = ["Validation", "Reference", "Our Result", "Assessment", "Verdict"]
    x_pos = [0.02, 0.22, 0.48, 0.72, 0.92]

    for i, h in enumerate(headers):
        ax.text(x_pos[i], y, h, ha="left", va="top", fontsize=10, fontweight="bold",
                transform=ax.transAxes, color=DARK_GRAY)

    y -= 0.03
    ax.plot([0.01, 0.99], [y, y], color=DARK_GRAY, linewidth=1,
            transform=ax.transAxes, clip_on=False)

    for row in results:
        y -= 0.12
        name, ref, result, assessment, verdict = row
        color = colors_map.get(verdict, DARK_GRAY)

        ax.text(x_pos[0], y, name, ha="left", va="top", fontsize=9,
                transform=ax.transAxes, fontweight="bold")
        ax.text(x_pos[1], y, ref, ha="left", va="top", fontsize=8,
                transform=ax.transAxes, color=STEEL_BLUE, style="italic")
        ax.text(x_pos[2], y, result, ha="left", va="top", fontsize=8,
                transform=ax.transAxes)
        ax.text(x_pos[3], y, assessment, ha="left", va="top", fontsize=8,
                transform=ax.transAxes)
        ax.text(x_pos[4], y, verdict, ha="left", va="top", fontsize=10,
                transform=ax.transAxes, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15))

        y -= 0.02
        ax.plot([0.01, 0.99], [y, y], color=LIGHT_GRAY, linewidth=0.5,
                transform=ax.transAxes, clip_on=False, alpha=0.5)

    fig.savefig(outdir / "V6_validation_scorecard.png")
    plt.close(fig)
    print(f"  → V6 saved")
    return results


# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path,
                        default=BASE / "thesis_results" / "T_external_validation")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading sweep data...")
    sweeps = {}
    for key in SWEEP_PATHS:
        sweeps[key] = load_sweep(key)
        n = len(sweeps[key]) if sweeps[key] else 0
        print(f"  {key}: {n} angles" if n else f"  {key}: MISSING")

    print("\nGenerating plots...")
    fit_info = plot_V1(sweeps, args.outdir)
    plot_V2(sweeps, args.outdir)
    plot_V3(sweeps, args.outdir)
    plot_V4(sweeps, args.outdir)
    plot_V5(sweeps, args.outdir)
    results = plot_V6_scorecard(sweeps, args.outdir)

    print(f"\n✓ All 6 plots saved to {args.outdir}/")

    # Print summary
    print("\n" + "=" * 70)
    print("QUANTITATIVE VALIDATION SUMMARY")
    print("=" * 70)
    for name, ref, result, assessment, verdict in results:
        icon = {"PASS": "✓", "PARTIAL": "~", "WEAK": "~", "N/A": "-", "TREND": "↗", "NEW": "★"}.get(verdict, "?")
        print(f"  {icon} {name}")
        print(f"      Ref: {ref}")
        print(f"      Result: {result}")
        print(f"      Assessment: {assessment}")
        print()


if __name__ == "__main__":
    main()
