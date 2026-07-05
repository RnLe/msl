"""
Tier 1 Candidate Scanner — Fast CSV-based ranking for moiré envelope candidates.

Reads Phase 0 CSV output (700k+ candidates) and ranks by estimated θ* — the
twist angle where V_depth / E_kin ≈ 1, i.e. where the moiré potential is
comparable to kinetic energy and bound states form.

Physics:
    M_eff_inv = curvature_trace / 2   (average ∂²ω/∂k² per dimension, isotropic)
    E_kin(θ)  = (curvature_trace / 4) · θ²   (θ in radians, |G_M| ≈ θ in 2π/a units)
    θ*(V)     = 2 · sqrt(V_depth / curvature_trace)

Since V_depth is unknown from Phase 0 (requires bilayer MPB), we:
    1. Compute a proxy  V_proxy = C · (ε_bg - 1)/ε_bg · ω₀
    2. Rank by θ* for several C values
    3. Also compute V/E_kin at fixed angles (1°, 3°, 5°)

The key insight: we want DISPERSIVE bands (large curvature → large E_kin) with
modest dielectric contrast (low V_depth). The existing S_total / valid_ea_flag
is too conservative — it penalizes curvature and rejects candidates we need.

Usage:
    python phasesV3/candidate_scanner_tier1.py <phase0_csv> [--config <yaml>] [--top N]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default filter / ranking parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    # Minimum band gap (in ω units) on EITHER side to ensure isolated subspace
    gap_min_threshold=0.0005,
    # Minimum curvature_trace to exclude truly flat bands (which give θ* → ∞)
    curvature_min=0.005,
    # Maximum band index (higher bands are less physical / harder to work with)
    max_band_index=20,
    # Minimum omega0 (exclude acoustic-like modes near ω=0)
    omega0_min=0.05,
    # Parabolic validity: k_parab must exceed this (= moiré BZ size at target θ)
    # For θ=5° = 0.087 rad, |G_M| ≈ 0.087.  We want k_parab > G_M comfortably.
    k_parab_min=0.03,
    # V_depth proxy calibration constants to sweep
    # V_proxy = C · (ε-1)/ε · ω₀
    C_values=[0.01, 0.05, 0.10, 0.20],
    # Reference C for primary ranking
    C_ref=0.05,
    # Target angles for V/E_kin computation (degrees)
    target_angles_deg=[1.0, 2.0, 3.0, 5.0],
    # How many candidates to output
    top_N=100,
    # Max candidates per band family in diversified view
    max_per_family=3,
)


def load_and_deduplicate(csv_path: str) -> pd.DataFrame:
    """Load Phase 0 CSV and remove duplicate rows."""
    df = pd.read_csv(csv_path)
    n_before = len(df)

    # Deduplicate on physical parameters (geometry + band + k-point)
    dedup_cols = [
        "lattice_type", "r_over_a", "eps_bg", "k_label",
        "band_index", "omega0", "curvature_trace",
    ]
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    n_after = len(df)
    logger.info(f"Loaded {n_before} rows, deduplicated to {n_after} ({n_before - n_after} removed)")
    return df


def apply_quality_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply basic quality filters to remove unsuitable candidates."""
    n0 = len(df)

    masks = {
        "omega0_min": df["omega0"] >= params["omega0_min"],
        "curvature_min": df["curvature_trace"] >= params["curvature_min"],
        "gap_min": df["gap_min"] >= params["gap_min_threshold"],
        "band_index": df["band_index"] <= params["max_band_index"],
    }

    # k_parab filter if column exists
    if "k_parab" in df.columns:
        masks["k_parab"] = df["k_parab"] >= params["k_parab_min"]

    combined = pd.Series(True, index=df.index)
    for name, mask in masks.items():
        n_fail = (~mask & combined).sum()
        combined &= mask
        logger.info(f"  Filter '{name}': removes {n_fail} candidates")

    df_filtered = df[combined].copy().reset_index(drop=True)
    logger.info(f"After quality filters: {len(df_filtered)} / {n0} candidates remain")
    return df_filtered


def compute_physics_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add physics-based columns for ranking."""
    # Effective mass inverse (per dimension, averaged)
    df["M_eff_inv"] = df["curvature_trace"] / 2.0

    # Dielectric contrast factor
    df["eps_contrast"] = (df["eps_bg"] - 1.0) / df["eps_bg"]

    # V_depth proxy for each calibration constant
    for C in params["C_values"]:
        col = f"V_proxy_C{C:.2f}"
        df[col] = C * df["eps_contrast"] * df["omega0"]

    # θ* for each C value (in degrees)
    for C in params["C_values"]:
        V_col = f"V_proxy_C{C:.2f}"
        theta_col = f"theta_star_C{C:.2f}"
        # θ* = 2 √(V/curv_trace), result in radians → convert to degrees
        df[theta_col] = np.degrees(
            2.0 * np.sqrt(df[V_col] / df["curvature_trace"])
        )

    # V/E_kin at specific angles for reference C
    C_ref = params["C_ref"]
    V_ref_col = f"V_proxy_C{C_ref:.2f}"
    for theta_deg in params["target_angles_deg"]:
        theta_rad = np.radians(theta_deg)
        E_kin = df["curvature_trace"] * theta_rad**2 / 4.0
        ratio_col = f"VEkin_{theta_deg:.0f}deg_C{C_ref:.2f}"
        df[ratio_col] = df[V_ref_col] / E_kin

    # Primary ranking column: θ* at reference C
    df["theta_star_ref"] = df[f"theta_star_C{C_ref:.2f}"]

    # --- Self-consistency: parabolic validity angle ---
    # k_parab is the k-range over which the band is parabolic.
    # The moiré samples |G_M| ≈ θ, so we need k_parab > θ.
    # theta_max_parab = the maximum twist angle where EA is trustworthy
    if "k_parab" in df.columns:
        df["theta_max_deg"] = np.degrees(df["k_parab"])
    else:
        df["theta_max_deg"] = 90.0  # no constraint

    # A candidate is "self-consistent" if θ* < θ_max (V/E_kin=1 within parabolic range)
    df["self_consistent"] = df["theta_star_ref"] < df["theta_max_deg"]

    # Usable angle window: [θ*, θ_max].  Width > 0 = good.
    df["angle_window_deg"] = (df["theta_max_deg"] - df["theta_star_ref"]).clip(lower=0)

    # --- KEY METRIC: V/E_kin at θ_max (the largest usable angle) ---
    # Even if θ* > θ_max, V/E_kin at θ_max may be in the target range [1, 10]
    C_ref = params["C_ref"]
    V_ref_col = f"V_proxy_C{C_ref:.2f}"
    theta_max_rad = np.radians(df["theta_max_deg"])
    E_kin_at_tmax = df["curvature_trace"] * theta_max_rad**2 / 4.0
    df["VEkin_at_thetamax"] = df[V_ref_col] / E_kin_at_tmax.clip(lower=1e-12)

    # Target quality: how close is V/E_kin(θ_max) to the sweet spot [1, 10]?
    # log-distance to center of target range (log(√10) ≈ 1.15)
    target_center = np.sqrt(10)  # geometric mean of [1, 10]
    df["target_quality"] = np.abs(np.log10(df["VEkin_at_thetamax"]) - np.log10(target_center))
    # In-range flag
    df["in_target_range"] = (df["VEkin_at_thetamax"] >= 0.5) & (df["VEkin_at_thetamax"] <= 30)

    # Also compute a "dispersiveness score" = curvature × gap quality
    # Higher is better: more kinetic energy + better isolated subspace
    df["dispersive_score"] = df["curvature_trace"] * np.sqrt(df["gap_min"])

    # Band family identifier for diversification
    # Group by lattice + k_label + band_index (varying ε_bg and r/a within family)
    df["family"] = df["lattice_type"] + "_" + df["k_label"] + "_b" + df["band_index"].astype(str)

    return df


def rank_candidates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Rank by θ* ascending (lowest magic angle = most practical).
    
    Note: V/E_kin at θ_max ≈ const for all candidates because k_parab = 0.2/√curv
    makes E_kin(k_parab) = 0.01 independent of curvature. So V/E_kin(θ_max) only
    depends on V_proxy, which is ~constant for similar (eps, omega0) values.
    The true discriminator is θ* — candidates with small θ* achieve V/E_kin~1 at
    practical twist angles.
    """
    df_sorted = df.sort_values(
        by=["theta_star_ref", "gap_min", "theta_max_deg"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    df_sorted["tier1_rank"] = range(1, len(df_sorted) + 1)

    # --- Diversified ranking: best N per family ---
    max_per_family = params.get("max_per_family", 3)
    df_sorted["_family_rank"] = df_sorted.groupby("family").cumcount() + 1
    df_diverse = df_sorted[df_sorted["_family_rank"] <= max_per_family].copy()

    df_sorted["_in_diverse"] = df_sorted.index.isin(df_diverse.index)
    return df_sorted


def print_summary(df: pd.DataFrame, params: dict, top_n: int = 30):
    """Print a formatted summary table of top candidates."""
    C_ref = params["C_ref"]

    # --- DIVERSIFIED TABLE (most useful) ---
    df_diverse = df[df["_in_diverse"]].copy().reset_index(drop=True)

    print("\n" + "=" * 130)
    print(f"TIER 1 DIVERSIFIED RANKING — Best {params.get('max_per_family', 3)} per band family, by θ* (C_ref={C_ref})")
    print("=" * 130)

    hdr = (
        f"{'#':>3s}  {'Family':>16s}  {'r/a':>5s}  {'ε_bg':>5s}  "
        f"{'Pol':>3s}  {'ω₀':>6s}  "
        f"{'curv':>7s}  {'gap_min':>7s}  {'θ*[°]':>6s}  {'θmax[°]':>7s}  "
        f"{'V/E@θmax':>8s}  {'InRng':>5s}  "
        f"{'V/E@3°':>7s}  {'k_parab':>7s}"
    )
    print(hdr)
    print("-" * 130)

    shown = 0
    for _, r in df_diverse.iterrows():
        if shown >= top_n:
            break
        in_rng = "Y" if r.in_target_range else "N"
        line = (
            f"{shown+1:3d}  {r.family:>16s}  {r.r_over_a:5.2f}  "
            f"{r.eps_bg:5.1f}  "
            f"{r.dominant_polarization:>3s}  {r.omega0:6.4f}  "
            f"{r.curvature_trace:7.4f}  {r.gap_min:7.4f}  "
            f"{r.theta_star_ref:6.1f}  {r.theta_max_deg:7.1f}  "
            f"{r.VEkin_at_thetamax:8.2f}  {in_rng:>5s}  "
        )
        col3 = f"VEkin_3deg_C{C_ref:.2f}"
        line += f"{r.get(col3, 0):7.1f}  "
        kp = r.get("k_parab", float("nan"))
        line += f"{kp:7.4f}"
        print(line)
        shown += 1

    # --- SUMMARY STATISTICS ---
    print(f"\n--- Distribution summary (all {len(df)} filtered candidates) ---")
    n_sc = df["self_consistent"].sum()
    n_in_range = df["in_target_range"].sum()
    print(f"  Self-consistent (θ* < θ_max): {n_sc} ({n_sc/len(df)*100:.1f}%)")
    print(f"  V/E_kin(θ_max) in [0.5, 30]: {n_in_range} ({n_in_range/len(df)*100:.1f}%)")
    print(f"  θ* range: {df.theta_star_ref.min():.1f}° – {df.theta_star_ref.max():.1f}°")
    print(f"  θ_max range: {df.theta_max_deg.min():.1f}° – {df.theta_max_deg.max():.1f}°")

    # Among in-range only
    ir = df[df["in_target_range"]]
    if len(ir) > 0:
        print(f"\n  Among in-target-range candidates ({len(ir)}):")
        print(f"    θ_max range: {ir.theta_max_deg.min():.1f}° – {ir.theta_max_deg.max():.1f}°")
        print(f"    θ* range: {ir.theta_star_ref.min():.1f}° – {ir.theta_star_ref.max():.1f}°")
        print(f"    V/E_kin(θ_max) range: {ir.VEkin_at_thetamax.min():.2f} – {ir.VEkin_at_thetamax.max():.2f}")
        print(f"    gap_min range: {ir.gap_min.min():.4f} – {ir.gap_min.max():.4f}")
        print(f"    With θ_max > 2°: {(ir.theta_max_deg > 2).sum()}")
        print(f"    With θ_max > 3°: {(ir.theta_max_deg > 3).sum()}")
        print(f"    With θ_max > 5°: {(ir.theta_max_deg > 5).sum()}")
        print(f"    Unique families: {ir.family.nunique()}")

    print(f"\n  Lattice distribution (all):")
    for lt, n in df.lattice_type.value_counts().items():
        print(f"    {lt}: {n}")

    print(f"\n  k-label distribution (all):")
    for kl, n in df.k_label.value_counts().items():
        print(f"    {kl}: {n}")

    print(f"\n  Unique band families: {df.family.nunique()}")
    print(f"  Top 10 families by count:")
    for fam, n in df.family.value_counts().head(10).items():
        print(f"    {fam}: {n}")


def print_sensitivity_table(df: pd.DataFrame, params: dict, top_n: int = 10):
    """Show how top candidates change across different C values."""
    C_values = params["C_values"]

    print("\n" + "=" * 100)
    print("SENSITIVITY ANALYSIS — θ* vs V_depth calibration constant C")
    print("=" * 100)

    top = df.head(top_n)

    hdr = f"{'Rank':>4s}  {'Lattice':>7s}  {'r/a':>5s}  {'ε_bg':>5s}  {'k':>2s}  {'Band':>4s}  "
    for C in C_values:
        hdr += f"{'θ*(C='+f'{C:.2f}'+')':>12s}  "
    print(hdr)
    print("-" * 100)

    for _, r in top.iterrows():
        line = (
            f"{r.tier1_rank:4d}  {r.lattice_type:>7s}  {r.r_over_a:5.2f}  "
            f"{r.eps_bg:5.1f}  {r.k_label:>2s}  {r.band_index:4d}  "
        )
        for C in C_values:
            col = f"theta_star_C{C:.2f}"
            line += f"{r[col]:12.1f}  "
        print(line)


def run_tier1(
    csv_path: str,
    params: Optional[dict] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full Tier 1 pipeline and return ranked DataFrame."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # 1. Load & deduplicate
    df = load_and_deduplicate(csv_path)

    # 2. Apply quality filters
    df = apply_quality_filters(df, params)

    if len(df) == 0:
        logger.warning("No candidates remain after filtering!")
        return df

    # 3. Compute physics columns
    df = compute_physics_columns(df, params)

    # 4. Rank
    df = rank_candidates(df, params)

    # 5. Print summary
    top_n = min(params.get("top_N", 100), len(df))
    print_summary(df, params, top_n=min(top_n, 50))
    print_sensitivity_table(df, params, top_n=min(top_n, 15))

    # 6. Save enriched CSV
    if output_csv is None:
        csv_p = Path(csv_path)
        output_csv = str(csv_p.parent / "tier1_ranked.csv")

    # Select key columns for output
    out_cols = [
        "tier1_rank", "candidate_id", "family", "lattice_type", "polarization",
        "r_over_a", "eps_bg", "k_label", "band_index", "omega0",
        "curvature_trace", "curvature_xx", "curvature_yy", "curvature_xy",
        "gap_above", "gap_below", "gap_min",
        "M_eff_inv", "eps_contrast", "theta_star_ref",
        "theta_max_deg", "angle_window_deg", "self_consistent",
        "VEkin_at_thetamax", "in_target_range", "target_quality",
        "dispersive_score",
        "valid_ea_flag", "k_parab", "k_parab_far",
        "n_subspace_bands", "subspace_bands",
        "dominant_polarization", "S_total",
    ]
    # Add V_proxy and theta_star columns
    for C in params["C_values"]:
        out_cols.append(f"V_proxy_C{C:.2f}")
        out_cols.append(f"theta_star_C{C:.2f}")
    # Add V/E_kin columns
    C_ref = params["C_ref"]
    for θ in params["target_angles_deg"]:
        out_cols.append(f"VEkin_{θ:.0f}deg_C{C_ref:.2f}")

    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in df.columns]

    # Apply family diversification to the output (not just display)
    top_N = params.get("top_N", 100)
    if params.get("diversify_output", True):
        df_diverse = df[df["_in_diverse"]].copy()
        df_out = df_diverse[out_cols].head(top_N)
        logger.info(f"Applied diversification (max {params.get('max_per_family', 3)} per family)")
    else:
        df_out = df[out_cols].head(top_N)

    df_out.to_csv(output_csv, index=False, float_format="%.6f")
    logger.info(f"Saved top {len(df_out)} candidates to {output_csv}")
    print(f"\nOutput saved to: {output_csv}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Tier 1 candidate scanner for moiré envelope approximation"
    )
    parser.add_argument("csv_path", help="Path to Phase 0 candidates CSV")
    parser.add_argument("--top", type=int, default=100, help="Number of top candidates")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--gap-min", type=float, default=None,
                        help="Minimum gap threshold (default: 0.0005)")
    parser.add_argument("--curv-min", type=float, default=None,
                        help="Minimum curvature_trace (default: 0.005)")
    parser.add_argument("--C-ref", type=float, default=None,
                        help="Reference V_depth calibration constant (default: 0.05)")

    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params["top_N"] = args.top
    if args.gap_min is not None:
        params["gap_min_threshold"] = args.gap_min
    if args.curv_min is not None:
        params["curvature_min"] = args.curv_min
    if args.C_ref is not None:
        params["C_ref"] = args.C_ref
        # Update C_ref in C_values if not present
        if args.C_ref not in params["C_values"]:
            params["C_values"].append(args.C_ref)
            params["C_values"].sort()

    run_tier1(args.csv_path, params=params, output_csv=args.output)


if __name__ == "__main__":
    main()
