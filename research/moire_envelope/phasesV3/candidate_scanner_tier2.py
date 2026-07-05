"""
Tier 2 Candidate Scanner — MPB bilayer validation for top Tier 1 candidates.

For each candidate from Tier 1, runs actual MPB eigensolves at a 3×3 registry
grid (9 frozen-registry bilayer configurations). Extracts:

    1. V_depth = max(ω_target) - min(ω_target)  across registry grid
    2. M_inv (2×2 Hessian) from 2D k-stencil at δ=0 (monolayer reference)
    3. ω₀(δ) map across registry grid
    4. vg_max at Γ (should be ≈ 0 at extremum)

Then computes actual V/E_kin and θ* with the true V_depth and M_inv.

Usage:
    python phasesV3/candidate_scanner_tier2.py <tier1_csv> [--n-registry 3] [--top N]
"""

import argparse
import ast
import json
import logging
import multiprocessing as mp_lib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# MPB threading: force single-threaded for multiprocessing compatibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    n_registry=3,          # 3×3 = 9 registry points
    dk=0.01,               # k-space step for finite-difference Hessian
    fd_order=2,            # FD order (2 = 3×3 stencil = 9 solves; 4 = 5×5 = 25)
    resolution=32,         # MPB grid resolution (32 = fast, 64 = production)
    n_workers=4,           # Parallel workers
    top_N=50,              # How many candidates to process
    target_angles_deg=[1.0, 2.0, 3.0, 5.0],
)


def parse_band_list(s):
    """Parse a string like '[0, 1, 2]' into a list of ints."""
    if isinstance(s, list):
        return s
    if isinstance(s, str):
        try:
            return list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return []
    return []


def k_label_to_k0(k_label, lattice_type):
    """Convert k-point label to MPB k-vector [kx, ky]."""
    if k_label == "Γ":
        return [0.0, 0.0]
    elif lattice_type == "square":
        if k_label == "X":
            return [0.5, 0.0]
        elif k_label == "M":
            return [0.5, 0.5]
    elif lattice_type == "hex":
        if k_label == "M":
            return [0.5, 0.0]
        elif k_label == "K":
            return [1.0 / 3.0, 1.0 / 3.0]
    raise ValueError(f"Unknown k_label={k_label} for lattice={lattice_type}")


def compute_registry_grid(n_reg):
    """Generate a uniform registry grid in [0, 1)²."""
    deltas = []
    for ix in range(n_reg):
        for iy in range(n_reg):
            delta = [ix / n_reg, iy / n_reg]
            deltas.append((ix, iy, delta))
    return deltas


def _mpb_worker(args):
    """Worker function for a single (candidate, registry_point) computation."""
    from phasesV3.phase1_mpb_v3 import (
        create_mpb_geometry,
        create_mpb_solver,
        compute_bands_at_k_stencil,
    )

    cand_idx, ix, iy, delta_frac, params = args

    geometry, lattice, bg_eps = create_mpb_geometry(
        params["lattice_type"],
        params["r_over_a"],
        params["eps_bg"],
        eps_hole=1.0,
        delta_frac=delta_frac,
    )

    ms = create_mpb_solver(
        geometry, lattice, bg_eps,
        params["max_band"],
        params["resolution"],
        params["polarization"],
    )

    result = compute_bands_at_k_stencil(
        ms,
        params["k0"],
        params["dk"],
        params["all_bands"],
        params["polarization"],
        params["fd_order"],
        extract_fields_at_center=False,
    )

    return (cand_idx, ix, iy, result)


def process_single_candidate(row, params, registry_grid):
    """Process one Tier 1 candidate: run MPB at all registry points.
    
    Returns a dict with:
        omega_map: (n_reg, n_reg, n_bands) — frequencies at each δ
        M_inv_map: (n_reg, n_reg, n_bands, 2, 2) — Hessians at each δ
        vg_map: (n_reg, n_reg, n_bands, 2) — group velocities
        V_depth: float — max-min frequency across registry (for target band)
        V_depths: (n_bands,) — per-band V_depth
        M_inv_ref: (n_bands, 2, 2) — Hessian at δ=0
    """
    lattice_type = row["lattice_type"]
    r_over_a = row["r_over_a"]
    eps_bg = row["eps_bg"]
    k_label = row["k_label"]
    band_index = int(row["band_index"])
    polarization = row.get("dominant_polarization", "TM")

    # Parse subspace bands
    all_bands = parse_band_list(row.get("subspace_bands", f"[{band_index}]"))
    if not all_bands:
        all_bands = [band_index]

    # Ensure target band is included
    if band_index not in all_bands:
        all_bands.append(band_index)
        all_bands.sort()

    # k-point
    k0 = k_label_to_k0(k_label, lattice_type)

    # Use k0 from CSV if available (more precise)
    if "k0_x" in row and "k0_y" in row:
        k0_csv = [row["k0_x"], row["k0_y"]]
        if not (np.isnan(k0_csv[0]) or np.isnan(k0_csv[1])):
            k0 = k0_csv

    max_band = max(all_bands) + 1

    mpb_params = {
        "lattice_type": lattice_type,
        "r_over_a": r_over_a,
        "eps_bg": eps_bg,
        "k0": k0,
        "dk": params["dk"],
        "all_bands": all_bands,
        "polarization": polarization,
        "fd_order": params["fd_order"],
        "resolution": params["resolution"],
        "max_band": max_band,
    }

    n_reg = params["n_registry"]
    n_bands = len(all_bands)

    omega_map = np.full((n_reg, n_reg, n_bands), np.nan)
    M_inv_map = np.full((n_reg, n_reg, n_bands, 2, 2), np.nan)
    vg_map = np.full((n_reg, n_reg, n_bands, 2), np.nan)

    # Build work items
    work = []
    for ix, iy, delta in registry_grid:
        work.append((0, ix, iy, delta, mpb_params))

    # Run sequentially (parallelism is across candidates, not within)
    for item in work:
        _, ix, iy, result = _mpb_worker(item)
        omega_map[ix, iy, :] = result["omega0"]
        M_inv_map[ix, iy, :, :, :] = result["M_inv"]
        vg_map[ix, iy, :, :] = result["vg"]

    # Find target band index in all_bands
    target_idx = all_bands.index(band_index)

    # V_depth: frequency range across registry for the target band
    omega_target = omega_map[:, :, target_idx]
    V_depth = float(np.nanmax(omega_target) - np.nanmin(omega_target))

    # Per-band V_depth
    V_depths = np.nanmax(omega_map, axis=(0, 1)) - np.nanmin(omega_map, axis=(0, 1))

    # Reference Hessian at δ=0 (monolayer)
    M_inv_ref = M_inv_map[0, 0, :, :, :]

    return {
        "omega_map": omega_map,
        "M_inv_map": M_inv_map,
        "vg_map": vg_map,
        "V_depth": V_depth,
        "V_depths": V_depths,
        "M_inv_ref": M_inv_ref,
        "target_idx": target_idx,
        "all_bands": all_bands,
    }


def compute_tier2_physics(row, mpb_result, params):
    """Compute physics metrics from MPB results."""
    V_depth = mpb_result["V_depth"]
    M_inv_ref = mpb_result["M_inv_ref"]
    target_idx = mpb_result["target_idx"]

    # 2D Hessian for target band at δ=0
    M_inv_2d = M_inv_ref[target_idx]  # shape: (2, 2)
    curv_trace_2d = M_inv_2d[0, 0] + M_inv_2d[1, 1]
    curv_det_2d = M_inv_2d[0, 0] * M_inv_2d[1, 1] - M_inv_2d[0, 1] * M_inv_2d[1, 0]

    # Condition number (ratio of eigenvalues — measures anisotropy)
    eigvals = np.linalg.eigvalsh(M_inv_2d)
    if min(abs(eigvals)) > 1e-12:
        cond_number = max(abs(eigvals)) / min(abs(eigvals))
    else:
        cond_number = np.inf

    # Average M_eff_inv (isotropic approximation)
    M_eff_inv_avg = abs(curv_trace_2d) / 2.0

    # θ* from actual V_depth and curvature
    if M_eff_inv_avg > 1e-12 and V_depth > 0:
        theta_star_rad = 2.0 * np.sqrt(V_depth / abs(curv_trace_2d))
        theta_star_deg = np.degrees(theta_star_rad)
    else:
        theta_star_deg = np.inf

    # V/E_kin at various angles
    V_Ekin = {}
    for theta_deg in params["target_angles_deg"]:
        theta_rad = np.radians(theta_deg)
        E_kin = abs(curv_trace_2d) * theta_rad**2 / 4.0
        if E_kin > 1e-15:
            V_Ekin[theta_deg] = V_depth / E_kin
        else:
            V_Ekin[theta_deg] = np.inf

    # Group velocity at δ=0 (should be ~0 at extremum)
    vg_ref = mpb_result["vg_map"][0, 0, target_idx]
    vg_norm = np.linalg.norm(vg_ref)

    # Registry frequency statistics
    omega_target = mpb_result["omega_map"][:, :, target_idx]
    omega_mean = float(np.nanmean(omega_target))
    omega_std = float(np.nanstd(omega_target))

    return {
        "V_depth": V_depth,
        "curv_trace_2d": curv_trace_2d,
        "curv_det_2d": curv_det_2d,
        "M_inv_xx": M_inv_2d[0, 0],
        "M_inv_xy": M_inv_2d[0, 1],
        "M_inv_yy": M_inv_2d[1, 1],
        "M_eff_inv_avg": M_eff_inv_avg,
        "cond_number": cond_number,
        "theta_star_deg": theta_star_deg,
        "V_Ekin": V_Ekin,
        "vg_norm_ref": vg_norm,
        "omega_mean": omega_mean,
        "omega_std": omega_std,
        "eigval_min": float(min(eigvals)),
        "eigval_max": float(max(eigvals)),
    }


def run_tier2(
    tier1_csv: str,
    params: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Run Tier 2 MPB validation on top candidates from Tier 1."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Load Tier 1 results
    df = pd.read_csv(tier1_csv)
    top_n = min(params.get("top_N", 50), len(df))
    df = df.head(top_n)
    logger.info(f"Processing top {len(df)} candidates from {tier1_csv}")

    # Registry grid
    registry_grid = compute_registry_grid(params["n_registry"])
    n_reg_pts = len(registry_grid)
    logger.info(f"Registry grid: {params['n_registry']}×{params['n_registry']} = {n_reg_pts} points")

    # Output directory
    if output_dir is None:
        output_dir = str(Path(tier1_csv).parent / "tier2_results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []
    t_total = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        cand_id = row.get("candidate_id", idx)
        family = row.get("family", "unknown")
        t_start = time.time()

        logger.info(
            f"[{idx+1}/{len(df)}] {family} r/a={row['r_over_a']:.2f} "
            f"eps={row['eps_bg']:.1f} k={row['k_label']} b={row['band_index']}"
        )

        try:
            mpb_result = process_single_candidate(row, params, registry_grid)
            physics = compute_tier2_physics(row, mpb_result, params)

            # Build result row
            res = {
                "tier2_rank": idx + 1,
                "candidate_id": cand_id,
                "family": family,
                "lattice_type": row["lattice_type"],
                "r_over_a": row["r_over_a"],
                "eps_bg": row["eps_bg"],
                "k_label": row["k_label"],
                "band_index": int(row["band_index"]),
                "polarization": row.get("dominant_polarization", "?"),
                "omega0_phase0": row.get("omega0", np.nan),
                "curv_trace_phase0": row.get("curvature_trace", np.nan),
            }
            res.update(physics)

            # Flatten V/E_kin dict
            for θ, val in physics["V_Ekin"].items():
                res[f"VEkin_{θ:.0f}deg"] = val

            # Store omega map as flat string for CSV
            omega_flat = mpb_result["omega_map"][:, :, mpb_result["target_idx"]]
            res["omega_map_flat"] = str(omega_flat.flatten().tolist())

            results.append(res)

            dt = time.time() - t_start
            logger.info(
                f"  → V_depth={physics['V_depth']:.6f}  curv_2d={physics['curv_trace_2d']:.4f}  "
                f"θ*={physics['theta_star_deg']:.1f}°  cond={physics['cond_number']:.1f}  "
                f"vg={physics['vg_norm_ref']:.4f}  ({dt:.1f}s)"
            )

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                "tier2_rank": idx + 1,
                "candidate_id": cand_id,
                "family": family,
                "lattice_type": row["lattice_type"],
                "r_over_a": row["r_over_a"],
                "eps_bg": row["eps_bg"],
                "k_label": row["k_label"],
                "band_index": int(row["band_index"]),
                "error": str(e),
            })

    total_time = time.time() - t_total
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Build results DataFrame
    df_results = pd.DataFrame(results)

    # Sort by θ* ascending
    if "theta_star_deg" in df_results.columns:
        df_results = df_results.sort_values("theta_star_deg").reset_index(drop=True)
        df_results["tier2_rank"] = range(1, len(df_results) + 1)

    # Save CSV
    csv_out = str(Path(output_dir) / "tier2_ranked.csv")
    drop_cols = [c for c in ["omega_map_flat", "V_Ekin"] if c in df_results.columns]
    df_results.drop(columns=drop_cols, errors="ignore").to_csv(csv_out, index=False, float_format="%.6f")
    logger.info(f"Saved results to {csv_out}")

    # Print summary table
    print_tier2_summary(df_results, params)

    return df_results


def print_tier2_summary(df, params):
    """Print formatted Tier 2 results."""
    print("\n" + "=" * 140)
    print("TIER 2 MPB VALIDATION RESULTS")
    print("=" * 140)

    hdr = (
        f"{'Rank':>4s}  {'Family':>16s}  {'r/a':>5s}  {'ε_bg':>5s}  "
        f"{'Pol':>3s}  {'V_depth':>8s}  {'curv_2D':>8s}  {'cond':>6s}  "
        f"{'θ*[°]':>6s}  {'vg_ref':>6s}  "
    )
    for θ in params["target_angles_deg"]:
        hdr += f"{'V/E@'+str(int(θ))+'°':>8s}  "
    hdr += f"{'curv_P0':>8s}"
    print(hdr)
    print("-" * 140)

    for _, r in df.iterrows():
        if "error" in r and pd.notna(r.get("error")):
            print(f"{r.get('tier2_rank', '?'):>4}  {r.get('family', '?'):>16s}  "
                  f"{r.get('r_over_a', 0):5.2f}  {r.get('eps_bg', 0):5.1f}  "
                  f"ERROR: {r.get('error', '')}")
            continue

        if "theta_star_deg" not in r or pd.isna(r.get("theta_star_deg")):
            continue

        line = (
            f"{r.tier2_rank:4d}  {r.get('family', ''):>16s}  "
            f"{r.r_over_a:5.2f}  {r.eps_bg:5.1f}  "
            f"{r.get('polarization', '?'):>3s}  "
            f"{r.V_depth:8.5f}  {r.curv_trace_2d:8.4f}  "
            f"{r.cond_number:6.1f}  "
            f"{r.theta_star_deg:6.1f}  {r.vg_norm_ref:6.4f}  "
        )
        for θ in params["target_angles_deg"]:
            col = f"VEkin_{θ:.0f}deg"
            val = r.get(col, np.nan)
            if np.isfinite(val):
                line += f"{val:8.1f}  "
            else:
                line += f"{'inf':>8s}  "
        line += f"{r.get('curv_trace_phase0', 0):8.4f}"
        print(line)

    # Summary
    valid = df.dropna(subset=["theta_star_deg"])
    valid = valid[valid["theta_star_deg"] < 1e6]
    if len(valid) > 0:
        print(f"\n--- Tier 2 Summary ({len(valid)} valid / {len(df)} total) ---")
        print(f"  θ* range: {valid.theta_star_deg.min():.1f}° – {valid.theta_star_deg.max():.1f}°")
        print(f"  V_depth range: {valid.V_depth.min():.6f} – {valid.V_depth.max():.6f}")
        print(f"  curv_trace_2d range: {valid.curv_trace_2d.min():.4f} – {valid.curv_trace_2d.max():.4f}")
        print(f"  Candidates with θ* < 5°: {(valid.theta_star_deg < 5).sum()}")
        print(f"  Candidates with θ* < 10°: {(valid.theta_star_deg < 10).sum()}")

        # Compare Phase 0 vs Phase 2 curvature
        if "curv_trace_phase0" in valid.columns:
            ratio = valid["curv_trace_2d"] / valid["curv_trace_phase0"].clip(lower=1e-12)
            print(f"\n  Phase 0 vs Phase 2 curvature ratio (2D/1D):")
            print(f"    Median: {ratio.median():.2f}")
            print(f"    Range: {ratio.min():.2f} – {ratio.max():.2f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tier 2 MPB validation scanner")
    parser.add_argument("tier1_csv", help="Path to Tier 1 ranked CSV")
    parser.add_argument("--top", type=int, default=50, help="Number of candidates to process")
    parser.add_argument("--n-registry", type=int, default=3, help="Registry grid size (NxN)")
    parser.add_argument("--resolution", type=int, default=32, help="MPB resolution")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--fd-order", type=int, default=2, choices=[2, 4],
                        help="Finite-difference order for Hessian (2=fast, 4=accurate)")

    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params["top_N"] = args.top
    params["n_registry"] = args.n_registry
    params["resolution"] = args.resolution
    params["n_workers"] = args.workers
    params["fd_order"] = args.fd_order

    run_tier2(args.tier1_csv, params=params, output_dir=args.output)


if __name__ == "__main__":
    main()
