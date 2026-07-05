#!/usr/bin/env python3
"""
External Validation Plots — Compare our envelope approximation results
against published literature.

Plots generated:
  V1: BW ∝ η² scaling law (vs theoretical prediction)
  V2: IPR localization transition (vs Wang 2020)
  V3: Per-miniband bandwidth → magic angle search
  V4: LDOS enhancement estimate (connect to Wang 2025 Purcell)
  V5: Full-A vs Diag-A comparison across all angles

Usage:
  python plot_external_validation.py [--outdir OUTDIR]

Requires: sweep_results.json from η-sweep runs (full-A and diag-A).
Does NOT require the η-sweep to be running — reads saved results.

Author: Generated for thesis validation, 2026-03-07
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# ── Thesis style ──────────────────────────────────────────────────────
SKY_BLUE = "#4E9AE1"
STARK_ORANGE = "#EBA538"
STEEL_BLUE = "#4D7B9E"
LIGHT_STEEL = "#A5C6DF"
DARK_GRAY = "#333333"
LIGHT_GRAY = "#AAAAAA"

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
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ── Paths ─────────────────────────────────────────────────────────────
BASE = Path("/home/renlephy/msl/research/moire_envelope")
RUNS = BASE / "runsV3"

# Sweep result paths — update these if directory names change
SWEEP_PATHS = {
    "C3_fullA": RUNS / "thesis_square_M_b3_20260209_173724/eta_sweep_20260307_144407/sweep_results.json",
    "C3_diagA": RUNS / "thesis_square_M_b3_20260209_173724/eta_sweep_20260306_175709_diagA/sweep_results.json",
    "C1_diagA": RUNS / "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_013633_diagA/sweep_results.json",
    "C1_fullA": RUNS / "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_153407/sweep_results_partial.json",
}

# Per-angle mode stats (for IPR data)
def get_mode_stats_dir(cand_key: str, theta_deg: float) -> Path:
    """Return path to phase3_mode_stats.json for a given candidate+angle."""
    base_name = {
        "C3_fullA": "thesis_square_M_b3_20260209_173724/eta_sweep_20260307_144407",
        "C3_diagA": "thesis_square_M_b3_20260209_173724/eta_sweep_20260306_175709_diagA",
        "C1_diagA": "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_013633_diagA",
        "C1_fullA": "thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_153407",
    }[cand_key]
    sweep_dir = RUNS / base_name
    theta_dir = sweep_dir / f"theta_{theta_deg:.3f}" / "candidate_0000"
    return theta_dir / "phase3_mode_stats.json"


def load_sweep(key: str) -> list[dict] | None:
    """Load sweep results. Returns None if file doesn't exist."""
    path = SWEEP_PATHS.get(key)
    if path is None or not path.exists():
        # Try partial
        partial = path.parent / "sweep_results_partial.json" if path else None
        if partial and partial.exists():
            path = partial
        else:
            print(f"  ⚠  {key}: file not found at {path}")
            return None
    with open(path) as f:
        data = json.load(f)
    print(f"  ✓  {key}: {len(data)} angles loaded from {path.name}")
    return data


def load_mode_stats(cand_key: str, theta_deg: float) -> list[dict] | None:
    """Load per-mode statistics (IPR, spread, etc.) for a given angle."""
    path = get_mode_stats_dir(cand_key, theta_deg)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════
# V1: BW ∝ η² Scaling Law
# ══════════════════════════════════════════════════════════════════════
def plot_V1_scaling_law(sweeps: dict, outdir: Path):
    """
    Log-log plot of miniband bandwidth vs η.
    Overlays theoretical η² reference line.
    Compares full-A vs diag-A and C3 vs C1.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = {"C3_fullA": "o", "C3_diagA": "s", "C1_fullA": "^", "C1_diagA": "D"}
    colors = {"C3_fullA": SKY_BLUE, "C3_diagA": STEEL_BLUE,
              "C1_fullA": STARK_ORANGE, "C1_diagA": "#C8892E"}
    labels = {
        "C3_fullA": "C3 square (full A)",
        "C3_diagA": "C3 square (diag A)",
        "C1_fullA": "C1 hex (full A)",
        "C1_diagA": "C1 hex (diag A)",
    }

    all_eta = []
    for key in ["C3_fullA", "C3_diagA", "C1_fullA", "C1_diagA"]:
        data = sweeps.get(key)
        if data is None:
            continue
        eta = np.array([d["eta"] for d in data])
        bw = np.array([d["bandwidth_50"] for d in data])
        all_eta.extend(eta)

        # Power-law fit: log(BW) = α log(η) + log(C)
        mask = (eta > 0) & (bw > 0)
        if mask.sum() >= 2:
            log_eta, log_bw = np.log10(eta[mask]), np.log10(bw[mask])
            coeffs = np.polyfit(log_eta, log_bw, 1)
            alpha = coeffs[0]
            label = f"{labels[key]} (α={alpha:.2f})"
        else:
            label = labels[key]

        ax.scatter(eta, bw, marker=markers[key], color=colors[key],
                   s=60, zorder=5, label=label, edgecolors="white", linewidth=0.5)

    # Theoretical η² reference line
    if all_eta:
        eta_ref = np.logspace(np.log10(min(all_eta) * 0.7),
                              np.log10(max(all_eta) * 1.3), 50)
        # Normalize to match data at the geometric mean
        geo_mean = np.exp(np.mean(np.log(all_eta)))
        # Just plot η² shape (unknown prefactor) — normalize to pass through middle
        ref_bw = eta_ref**2
        # Scale to roughly match C3_diagA if available
        ref_data = sweeps.get("C3_diagA") or sweeps.get("C3_fullA")
        if ref_data:
            mid_idx = len(ref_data) // 2
            mid_eta = ref_data[mid_idx]["eta"]
            mid_bw = ref_data[mid_idx]["bandwidth_50"]
            scale = mid_bw / mid_eta**2
            ref_bw *= scale
        ax.plot(eta_ref, ref_bw, "--", color=LIGHT_GRAY, linewidth=2,
                label=r"$\propto \eta^2$ (theory)", zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Twist parameter $\eta = 2\sin(\theta/2)$")
    ax.set_ylabel(r"Miniband bandwidth $\Delta\varepsilon_{50}$")
    ax.set_title("V1: Miniband Bandwidth Scaling Law")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    # Add secondary x-axis for θ in degrees
    ax2 = ax.twiny()
    theta_ticks = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    eta_ticks = [2 * np.sin(np.radians(t / 2)) for t in theta_ticks]
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(eta_ticks)
    ax2.set_xticklabels([f"{t}°" for t in theta_ticks])
    ax2.set_xlabel(r"Twist angle $\theta$", labelpad=8)

    out = outdir / "V1_scaling_law.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# V2: IPR Localization Transition
# ══════════════════════════════════════════════════════════════════════
def plot_V2_localization(sweeps: dict, outdir: Path):
    """
    IPR of lowest modes vs twist angle.
    Compare qualitatively with Wang 2020 localization-delocalization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, cand_key in enumerate(["C3_fullA", "C3_diagA"]):
        ax = axes[ax_idx]
        data = sweeps.get(cand_key)
        if data is None:
            # Fallback to diagA
            cand_key_alt = cand_key.replace("fullA", "diagA")
            data = sweeps.get(cand_key_alt)
            cand_key = cand_key_alt
        if data is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        thetas = []
        ipr_mode0 = []
        ipr_mode1 = []
        ipr_median = []
        spread_mode0 = []

        for entry in data:
            theta = entry["theta_deg"]
            stats = load_mode_stats(cand_key, theta)
            if stats is None:
                continue
            thetas.append(theta)
            ipr_mode0.append(stats[0]["ipr"])
            ipr_mode1.append(stats[1]["ipr"] if len(stats) > 1 else np.nan)
            ipr_median.append(np.median([s["ipr"] for s in stats[:10]]))
            spread_mode0.append(stats[0]["spread"])

        if not thetas:
            ax.text(0.5, 0.5, "No mode stats", transform=ax.transAxes, ha="center")
            continue

        thetas = np.array(thetas)
        ax.semilogy(thetas, ipr_mode0, "o-", color=SKY_BLUE, label="Mode 0 (ground)")
        ax.semilogy(thetas, ipr_mode1, "s-", color=STARK_ORANGE, label="Mode 1")
        ax.semilogy(thetas, ipr_median, "D--", color=STEEL_BLUE, label="Median (modes 0-9)")

        ax.set_xlabel(r"Twist angle $\theta$ (°)")
        ax.set_ylabel("IPR (Inverse Participation Ratio)")
        ax.set_title(f"{'C3 Square' if 'C3' in cand_key else 'C1 Hex'} — {'Full A' if 'fullA' in cand_key else 'Diag A'}")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Annotate Wang 2020 connection
        ax.annotate("← localized (small θ)\n→ delocalized (large θ)\ncf. Wang et al. (2020)",
                    xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=8, color=LIGHT_GRAY,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=LIGHT_GRAY, alpha=0.8))

    fig.suptitle("V2: Localization Transition — IPR vs Twist Angle", fontsize=14, y=1.02)
    fig.tight_layout()
    out = outdir / "V2_localization_transition.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# V3: Per-Miniband Bandwidth → Magic Angle Search
# ══════════════════════════════════════════════════════════════════════
def plot_V3_magic_angle(sweeps: dict, outdir: Path):
    """
    Bandwidth of individual minibands vs θ.
    A bandwidth minimum would indicate a 'magic angle'.
    Compare with Dong et al. (2021) flat-band prediction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, (cand_key, cand_label) in enumerate([
        ("C3_fullA", "C3 Square (full A)"),
        ("C1_diagA", "C1 Hex (diag A)"),
    ]):
        ax = axes[ax_idx]
        data = sweeps.get(cand_key)
        if data is None:
            fallback = cand_key.replace("fullA", "diagA")
            data = sweeps.get(fallback)
            cand_label += " [fallback: diag A]"
        if data is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        thetas = np.array([d["theta_deg"] for d in data])
        n_modes = min(len(d["eigenvalues"]) for d in data)

        # Compute per-miniband-group bandwidths
        # Group modes into bands of ~5 (rough miniband grouping)
        n_per_band = 5
        n_bands_to_show = min(6, n_modes // n_per_band)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, n_bands_to_show))

        for b in range(n_bands_to_show):
            i_lo = b * n_per_band
            i_hi = min(i_lo + n_per_band, n_modes)
            bw_band = []
            for d in data:
                evals = np.array(d["eigenvalues"][i_lo:i_hi])
                bw_band.append(evals[-1] - evals[0])
            ax.semilogy(thetas, bw_band, "o-", color=cmap[b],
                        label=f"Modes {i_lo}-{i_hi-1}", markersize=5)

        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel("Miniband group bandwidth")
        ax.set_title(cand_label)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Mark if any band shows a minimum (potential magic angle)
        for b in range(n_bands_to_show):
            i_lo = b * n_per_band
            i_hi = min(i_lo + n_per_band, n_modes)
            bw_band = []
            for d in data:
                evals = np.array(d["eigenvalues"][i_lo:i_hi])
                bw_band.append(evals[-1] - evals[0])
            bw_arr = np.array(bw_band)
            # Check for interior minimum (not at endpoints)
            if len(bw_arr) > 2:
                min_idx = np.argmin(bw_arr)
                if 0 < min_idx < len(bw_arr) - 1:
                    ax.axvline(thetas[min_idx], color=cmap[b], alpha=0.3, linestyle=":")
                    ax.annotate(f"min @ {thetas[min_idx]:.1f}°",
                                xy=(thetas[min_idx], bw_arr[min_idx]),
                                xytext=(5, 10), textcoords="offset points",
                                fontsize=7, color=cmap[b])

    fig.suptitle("V3: Per-Miniband Bandwidth — Magic Angle Search\n"
                 "cf. Dong et al. (2021) PRL — flat bands at magic angle",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    out = outdir / "V3_magic_angle_search.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# V4: LDOS Enhancement Estimate
# ══════════════════════════════════════════════════════════════════════
def plot_V4_ldos_enhancement(sweeps: dict, outdir: Path):
    """
    Estimated LDOS enhancement ∝ 1/BW vs θ.
    Connects to Wang et al. (2025) Purcell factor measurement.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

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
        # LDOS enhancement ∝ 1/BW (relative to largest-θ value)
        ldos_enh = bw[-1] / bw  # normalized so enhancement at largest θ = 1
        ax.semilogy(thetas, ldos_enh, f"{marker}-", color=color, label=label,
                    markersize=7, linewidth=1.5)

    ax.set_xlabel(r"Twist angle $\theta$ (°)")
    ax.set_ylabel(r"LDOS enhancement $\propto 1/\mathrm{BW}$ (relative)")
    ax.set_title("V4: Estimated LDOS Enhancement from Flat Bands")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Annotate Wang 2025 connection
    ax.annotate("Wang et al. (2025) Sci. Adv.:\n"
                "Measured Purcell factor ×40\n"
                "from moiré flatband cavity",
                xy=(0.02, 0.98), xycoords="axes fraction",
                ha="left", va="top", fontsize=9, color=LIGHT_GRAY,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=LIGHT_GRAY, alpha=0.8))

    # Mark the EA validity boundary
    ax.axvline(3.0, color=LIGHT_GRAY, linestyle="--", alpha=0.5)
    ax.text(3.1, ax.get_ylim()[1] * 0.5, "EA validity\nboundary (C3)",
            fontsize=8, color=LIGHT_GRAY, va="center")

    out = outdir / "V4_ldos_enhancement.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# V5: Full-A vs Diag-A Comprehensive Comparison
# ══════════════════════════════════════════════════════════════════════
def plot_V5_fullA_vs_diagA(sweeps: dict, outdir: Path):
    """
    4-panel comparison: BW, mixing, gap_01, and BW ratio (full/diag) vs θ.
    Shows the effect of off-diagonal Berry connection across all angles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for cand, color_full, color_diag, label in [
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
            ax.semilogy(th_f, bw_f, "o-", color=color_full, label=f"{label} full-A")
        if diag:
            th_d = [d["theta_deg"] for d in diag]
            bw_d = [d["bandwidth_50"] for d in diag]
            ax.semilogy(th_d, bw_d, "s--", color=color_diag, label=f"{label} diag-A")
        ax.set_ylabel(r"$\Delta\varepsilon_{50}$ (bandwidth)")
        ax.set_title("Miniband Bandwidth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 1: Max mixing
        ax = axes[0, 1]
        if full:
            mix_f = [d["max_mixing"] for d in full]
            ax.plot(th_f, mix_f, "o-", color=color_full, label=f"{label} full-A")
        if diag:
            mix_d = [d["max_mixing"] for d in diag]
            ax.plot(th_d, mix_d, "s--", color=color_diag, label=f"{label} diag-A")
        ax.set_ylabel("Max interband mixing")
        ax.set_title("Interband Mixing Fraction")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Gap_01
        ax = axes[1, 0]
        if full:
            gap_f = [d["gap_01"] for d in full]
            ax.semilogy(th_f, np.abs(gap_f), "o-", color=color_full, label=f"{label} full-A")
        if diag:
            gap_d = [d["gap_01"] for d in diag]
            ax.semilogy(th_d, np.abs(gap_d), "s--", color=color_diag, label=f"{label} diag-A")
        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel(r"$|\Delta_{01}|$ (first gap)")
        ax.set_title("First Eigenvalue Gap")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 3: BW ratio full/diag
        ax = axes[1, 1]
        if full and diag:
            # Match angles
            th_common = sorted(set(th_f) & set(th_d))
            if th_common:
                bw_full_map = {d["theta_deg"]: d["bandwidth_50"] for d in full}
                bw_diag_map = {d["theta_deg"]: d["bandwidth_50"] for d in diag}
                ratio = [bw_full_map[t] / bw_diag_map[t] for t in th_common]
                ax.plot(th_common, ratio, "o-", color=color_full, label=label)
        ax.axhline(1.0, color=LIGHT_GRAY, linestyle="--", alpha=0.5)
        ax.set_xlabel(r"$\theta$ (°)")
        ax.set_ylabel(r"$\mathrm{BW}_\mathrm{full} / \mathrm{BW}_\mathrm{diag}$")
        ax.set_title("Bandwidth Ratio (Full-A / Diag-A)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("V5: Effect of Off-Diagonal Berry Connection Across All Angles",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = outdir / "V5_fullA_vs_diagA_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → Saved {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="External validation plots")
    parser.add_argument("--outdir", type=Path,
                        default=BASE / "thesis_results" / "T_external_validation",
                        help="Output directory for plots")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading sweep data...")
    sweeps = {}
    for key in SWEEP_PATHS:
        sweeps[key] = load_sweep(key)

    n_loaded = sum(1 for v in sweeps.values() if v is not None)
    if n_loaded == 0:
        print("\n✗ No sweep data found. Is the η-sweep still running?")
        print("  Check paths in SWEEP_PATHS dict at top of script.")
        sys.exit(1)

    print(f"\n{n_loaded}/{len(SWEEP_PATHS)} datasets loaded. Generating plots...\n")

    plot_V1_scaling_law(sweeps, args.outdir)
    plot_V2_localization(sweeps, args.outdir)
    plot_V3_magic_angle(sweeps, args.outdir)
    plot_V4_ldos_enhancement(sweeps, args.outdir)
    plot_V5_fullA_vs_diagA(sweeps, args.outdir)

    print(f"\n✓ All plots saved to {args.outdir}/")
    print("\nNext steps:")
    print("  1. Check V3 for bandwidth minima → potential magic angles")
    print("  2. Compare V1 slope with Dong et al. (2021) coupled-mode prediction")
    print("  3. Compare V2 localization transition with Wang et al. (2020)")
    print("  4. Connect V4 LDOS enhancement to Wang et al. (2025) Purcell factor")
    print("  5. Use V5 to quantify the full-A effect across the θ range")


if __name__ == "__main__":
    main()
