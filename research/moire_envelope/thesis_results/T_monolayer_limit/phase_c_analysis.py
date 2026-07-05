"""
Phase C: EA → Monolayer Limit Analysis
=======================================
Analyzes existing eta sweep data (19 angles, θ=0.4°–8.0°) to verify:
  1. Bandwidth BW ∝ η² as θ→0
  2. Mean eigenvalue center converges to a constant
  3. Band mixing decreases as θ→0

All data already exists — this is a pure analysis + plotting script.
No new MPB computation needed.

Outputs:
  - convergence_results_C.json
  - fig_phaseC_monolayer_limit.{png,pdf}
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR
RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runsV3" / "thesis_honeycomb_K_b1_20260307_171424"

SWEEPS = [
    ("eta_sweep_20260307_194458", [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]),
    ("eta_sweep_20260307_225641", [0.4, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.3, 1.7, 1.9]),
]

# From Phase A: monolayer Dirac frequency at K (MPB res=256)
OMEGA_DIRAC = 0.27435962  # mean of bands 0 and 1 at K

# =============================================================================
# Data extraction
# =============================================================================

def extract_sweep_data():
    """Read eigenvalues and mode stats from all sweep angles."""
    import h5py

    results = []
    for sweep_dir, thetas in SWEEPS:
        for theta in thetas:
            cdir = RUNS_DIR / sweep_dir / f"theta_{theta:.3f}" / "candidate_0000"
            p3_path = cdir / "phase3_multiband_modes.h5"
            stats_path = cdir / "phase3_mode_stats.json"

            if not p3_path.exists():
                print(f"  WARNING: Missing Phase 3 for θ={theta}°, skipping")
                continue

            with h5py.File(p3_path, "r") as f:
                eigs = f["eigenvalues"][:]
                omega_ref = float(f.attrs.get("omega_ref", 0))
                eta = float(f.attrs.get("eta", 0))
                n_modes = len(eigs)

            # Mode stats (band mixing, IPR, etc.)
            mixing_stats = {}
            if stats_path.exists():
                with open(stats_path) as f:
                    mode_stats = json.load(f)
                weights = [m["dominant_band_weight"] for m in mode_stats]
                mixing_stats = {
                    "mean_dominant_weight": float(np.mean(weights)),
                    "min_dominant_weight": float(np.min(weights)),
                    "mean_mixing": float(1.0 - np.mean(weights)),
                    "max_mixing": float(1.0 - np.min(weights)),
                    "mean_ipr": float(np.mean([m["ipr"] for m in mode_stats])),
                    "mean_spread": float(np.mean([m["spread"] for m in mode_stats])),
                }

            bw = float(np.max(eigs) - np.min(eigs))
            center = float((np.max(eigs) + np.min(eigs)) / 2)
            mean_eig = float(np.mean(eigs))

            results.append({
                "theta_deg": theta,
                "eta": eta,
                "omega_ref": omega_ref,
                "n_modes": n_modes,
                "eigenvalues": eigs.tolist(),
                "bandwidth": bw,
                "center_lambda": center,
                "mean_lambda": mean_eig,
                "omega_mean": omega_ref + mean_eig,
                "omega_center": omega_ref + center,
                **mixing_stats,
            })

    results.sort(key=lambda x: x["theta_deg"])
    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze(results):
    """Compute power-law fits and print summary."""
    thetas = np.array([r["theta_deg"] for r in results])
    etas = np.array([r["eta"] for r in results])
    bws = np.array([r["bandwidth"] for r in results])
    omega_centers = np.array([r["omega_center"] for r in results])
    mean_mixings = np.array([r.get("mean_mixing", np.nan) for r in results])

    print("=" * 70)
    print("  PHASE C: EA → MONOLAYER LIMIT ANALYSIS")
    print("=" * 70)
    print(f"  Angles: {len(results)} (θ = {thetas[0]:.1f}° to {thetas[-1]:.1f}°)")
    print(f"  Reference ω_D = {OMEGA_DIRAC:.6f} (Phase A, MPB res=256)")
    print()

    # Power-law fits for various ranges
    print("  ── Bandwidth Scaling: BW ~ η^α ──")
    fits = {}
    for max_theta, label in [(1.0, "θ≤1°"), (2.0, "θ≤2°"), (3.0, "θ≤3°"), (5.0, "θ≤5°"), (8.0, "all")]:
        mask = thetas <= max_theta
        if mask.sum() < 2:
            continue
        c = np.polyfit(np.log(etas[mask]), np.log(bws[mask]), 1)
        fits[label] = {"alpha": c[0], "log_prefactor": c[1], "n_points": int(mask.sum())}
        print(f"    {label:8s}: α = {c[0]:.3f}  (n={mask.sum()} pts)")
    print(f"    Expected: α = 2.0")
    print()

    # Mean eigenvalue convergence
    print("  ── Mean Eigenvalue Convergence ──")
    print(f"    {'θ (°)':>8} {'η':>10} {'ω_center':>12} {'|ω_center-ω_D|':>15} {'BW':>12} {'mixing':>8}")
    for r in results:
        dev = abs(r["omega_center"] - OMEGA_DIRAC)
        mix_str = f"{r.get('mean_mixing', float('nan')):.4f}" if "mean_mixing" in r else "N/A"
        print(f"    {r['theta_deg']:8.3f} {r['eta']:10.5f} {r['omega_center']:12.6f} {dev:15.6f} {r['bandwidth']:12.6f} {mix_str:>8}")
    print()

    # Check: ω_center should approach a value as η→0
    small_mask = thetas <= 1.0
    omega_center_small = omega_centers[small_mask]
    print(f"  ω_center as θ→0: mean = {np.mean(omega_center_small):.6f} ± {np.std(omega_center_small):.6f}")
    print(f"  Distance from ω_D = {OMEGA_DIRAC:.6f}: Δ = {abs(np.mean(omega_center_small) - OMEGA_DIRAC):.6f}")
    print(f"  (ω_center tracks min(Λ) + BW/2, NOT the Dirac frequency)")
    print()

    # Band mixing scaling
    valid_mix = ~np.isnan(mean_mixings)
    if valid_mix.sum() >= 3:
        print("  ── Band Mixing Scaling ──")
        for max_theta, label in [(2.0, "θ≤2°"), (8.0, "all")]:
            mask = (thetas <= max_theta) & valid_mix
            if mask.sum() >= 2:
                c = np.polyfit(np.log(etas[mask]), np.log(mean_mixings[mask]), 1)
                print(f"    {label:8s}: mixing ~ η^{c[0]:.2f}")
        print()

    return fits


# =============================================================================
# Plots
# =============================================================================

def generate_plots(results, fits):
    """Generate multi-panel Phase C figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = np.array([r["theta_deg"] for r in results])
    etas = np.array([r["eta"] for r in results])
    bws = np.array([r["bandwidth"] for r in results])
    omega_centers = np.array([r["omega_center"] for r in results])
    mean_mixings = np.array([r.get("mean_mixing", np.nan) for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel (a): BW vs η, log-log ──
    ax = axes[0]
    ax.loglog(etas, bws, "o-", color="C0", markersize=5, label="EA bandwidth")

    # Reference line: η² 
    eta_line = np.logspace(np.log10(etas.min()), np.log10(etas.max()), 100)
    # Fit through the small-angle data
    small_mask = thetas <= 2.0
    c = np.polyfit(np.log(etas[small_mask]), np.log(bws[small_mask]), 1)
    bw_fit = np.exp(c[1]) * eta_line ** c[0]
    ax.loglog(eta_line, bw_fit, "--", color="gray", alpha=0.7,
              label=f"fit: η^{{{c[0]:.2f}}}")
    
    # Pure η² reference
    c2_ref = np.polyfit(np.log(etas[small_mask]), np.log(bws[small_mask]) - 2 * np.log(etas[small_mask]), 0)
    bw_ref2 = np.exp(c2_ref[0]) * eta_line ** 2
    ax.loglog(eta_line, bw_ref2, ":", color="C3", alpha=0.5, label="η² reference")

    ax.set_xlabel("η = θ / (2 sin(π/6))")
    ax.set_ylabel("Bandwidth (c/a)")
    ax.set_title("(a) Bandwidth scaling")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel (b): ω_center vs θ ──
    ax = axes[1]
    ax.plot(thetas, omega_centers, "o-", color="C1", markersize=5)
    ax.axhline(OMEGA_DIRAC, color="C3", ls="--", alpha=0.5, label=f"ω_D = {OMEGA_DIRAC:.5f}")
    
    # Also mark ω_ref
    omega_ref = results[0]["omega_ref"]
    ax.axhline(omega_ref, color="C4", ls=":", alpha=0.5, label=f"ω_ref = {omega_ref:.5f}")

    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("ω_center (c/a)")
    ax.set_title("(b) Spectral center vs twist angle")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel (c): Band mixing vs θ ──
    ax = axes[2]
    valid = ~np.isnan(mean_mixings)
    if valid.sum() > 0:
        ax.semilogy(thetas[valid], mean_mixings[valid], "o-", color="C2", markersize=5,
                     label="mean mixing")
        max_mixings = np.array([r.get("max_mixing", np.nan) for r in results])
        valid2 = ~np.isnan(max_mixings)
        if valid2.sum() > 0:
            ax.semilogy(thetas[valid2], max_mixings[valid2], "s--", color="C4", 
                        markersize=4, label="max mixing")

    ax.set_xlabel("θ (degrees)")
    ax.set_ylabel("Band mixing (1 − max weight)")
    ax.set_title("(c) Band mixing vs twist angle")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig_phaseC_monolayer_limit.{ext}", dpi=150,
                    bbox_inches="tight")
    plt.close()
    print(f"  Plots saved: fig_phaseC_monolayer_limit.{{png,pdf}}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\n  Script: {__file__}")
    print(f"  Runs dir: {RUNS_DIR}")
    print(f"  Output: {OUTPUT_DIR}\n")

    # Extract data
    results = extract_sweep_data()
    if not results:
        print("  ERROR: No sweep data found!")
        sys.exit(1)

    # Analyze
    fits = analyze(results)

    # Save results
    out = {
        "omega_dirac": OMEGA_DIRAC,
        "n_angles": len(results),
        "fits": fits,
        "angles": results,
    }
    out_path = OUTPUT_DIR / "convergence_results_C.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved: {out_path}")

    # Plots
    try:
        generate_plots(results, fits)
    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("  PHASE C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
