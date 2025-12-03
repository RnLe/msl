"""Phase 2 (BLAZE): Prepare envelope-approximation data for BLAZE EA solver.

This is the BLAZE-pipeline version of Phase 2. It reads Phase 1 data produced by
phase1_blaze.py, applies supercell tiling if configured, and saves the prepared
data for BLAZE Phase 3's EA solver.

Unlike the legacy Phase 2 which assembles a sparse operator for scipy eigsh,
this version prepares the raw fields (V, M_inv, R_grid) that BLAZE's EA solver
will use to construct its own internal operator.

Note: The BLAZE EA solver uses the convention H = V + (η²/2) ∇·M⁻¹∇ (positive kinetic),
which differs from the legacy convention H = V - (η²/2) ∇·M⁻¹∇ (negative kinetic).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, cast

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_json, save_json
from common.plotting import plot_phase2_fields

# Import utility functions from the original phase2 (these are sign-agnostic)
from phases.phase2_ea_operator import (
    _regularize_mass_tensor,
    _grid_metrics,
    _tile_fields,
)


def log(msg: str):
    """Simple logging helper."""
    print(msg)


def resolve_blaze_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete BLAZE Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        base = Path(config.get("output_dir", "runs"))
        # Look for BLAZE-specific run directories first
        blaze_runs = sorted(base.glob("phase0_blaze_*"))
        if blaze_runs:
            run_dir = blaze_runs[-1]
            log(f"Auto-selected latest BLAZE run: {run_dir}")
        else:
            # Fall back to any phase0 runs
            phase0_runs = sorted(base.glob("phase0_*"))
            if not phase0_runs:
                raise FileNotFoundError(f"No phase0_* directories found in {base}")
            run_dir = phase0_runs[-1]
            log(f"Auto-selected latest Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runs")) / run_dir
        if candidate.exists():
            run_dir = candidate
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


def _discover_blaze_phase1_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that contain BLAZE Phase 1 data."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # Check for Phase 1 band data (same format as MPB pipeline)
        if (cdir / "phase1_band_data.h5").exists():
            discovered.append((cid, cdir))
    return discovered


def _load_blaze_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    """Load per-candidate metadata JSON, falling back to CSV if necessary."""
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception as exc:
            log(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return {"candidate_id": candidate_id}


def process_candidate(
    row: Dict,
    config: Dict,
    run_dir: Path,
    plot_fields: bool,
):
    """Process a single candidate through BLAZE Phase 2.
    
    Prepares V, M_inv, R_grid data (with optional tiling) for BLAZE EA solver.
    Does NOT build a sparse operator - that's done internally by BLAZE in Phase 3.
    """
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data for candidate {cid}: {h5_path}")

    log(f"  Candidate {cid}: loading BLAZE Phase 1 data from {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        vg_field = np.asarray(cast(h5py.Dataset, hf["vg"])[:]) if "vg" in hf else None
        eta_attr = hf.attrs.get("eta")
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))

    # Determine eta (small twist parameter)
    if eta_attr is not None:
        eta = float(eta_attr)
    else:
        eta_cfg = config.get("eta")
        if isinstance(eta_cfg, str) and eta_cfg.lower() == "auto":
            eta_cfg = None
        if eta_cfg is not None:
            eta = float(eta_cfg)
        else:
            a = float(row.get("a", np.nan))
            L_m = float(row.get("moire_length", np.nan))
            if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
                eta = a / L_m
            else:
                log("    WARNING: Missing eta in Phase 1 attrs and config; defaulting to 1.0")
                eta = 1.0

    log(f"    Using η = {eta:.6f}")

    # Regularize mass tensor
    min_mass_eig = config.get("phase2_min_mass_eig")
    mass_tensor = _regularize_mass_tensor(M_inv, min_mass_eig)

    # Handle supercell tiling
    tiles_cfg = config.get("phase2_supercell_tiles", [1, 1])
    if isinstance(tiles_cfg, int):
        tiles = (int(tiles_cfg), int(tiles_cfg))
    elif isinstance(tiles_cfg, (list, tuple)) and len(tiles_cfg) == 2:
        tiles = (int(tiles_cfg[0]), int(tiles_cfg[1]))
    else:
        raise ValueError("phase2_supercell_tiles must be an int or [Tx, Ty] list")
    if tiles[0] < 1 or tiles[1] < 1:
        raise ValueError("phase2_supercell_tiles entries must be ≥ 1")

    if tiles != (1, 1):
        log(f"    Tiling Phase 1 fields into a {tiles[0]}×{tiles[1]} supercell")
    R_grid, V, mass_tensor, vg_field = _tile_fields(R_grid, V, mass_tensor, vg_field, tiles)

    grid_info = _grid_metrics(R_grid)
    log(f"    Prepared data for BLAZE EA: grid {int(grid_info['Nx'])}×{int(grid_info['Ny'])}")

    # Save prepared data for BLAZE Phase 3
    h5_out = cdir / "phase2_blaze_data.h5"
    with h5py.File(h5_out, "w") as hf:
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("V", data=V, compression="gzip")
        hf.create_dataset("M_inv", data=mass_tensor, compression="gzip")
        if vg_field is not None:
            hf.create_dataset("vg", data=vg_field, compression="gzip")
        hf.attrs["eta"] = eta
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["tiles_x"] = tiles[0]
        hf.attrs["tiles_y"] = tiles[1]
    log(f"    Saved prepared data to {h5_out}")

    # Also save R_grid as numpy for compatibility
    np.save(cdir / "phase2_R_grid.npy", R_grid)

    # Compute field statistics
    eigvals = np.linalg.eigvalsh(mass_tensor.reshape(-1, 2, 2))
    stats = {
        "candidate_id": cid,
        "Nx": int(grid_info["Nx"]),
        "Ny": int(grid_info["Ny"]),
        "dx": grid_info["dx"],
        "dy": grid_info["dy"],
        "eta": eta,
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "V_mean": float(V.mean()),
        "M_inv_min_eig": float(eigvals.min()),
        "M_inv_max_eig": float(eigvals.max()),
        "M_inv_mean_eig": float(eigvals.mean()),
    }
    if vg_field is not None:
        vg_norm = np.linalg.norm(vg_field[..., :2], axis=-1)
        stats.update({
            "vg_norm_min": float(vg_norm.min()),
            "vg_norm_max": float(vg_norm.max()),
            "vg_norm_mean": float(vg_norm.mean()),
        })

    info_path = cdir / "phase2_operator_info.csv"
    pd.DataFrame([stats]).to_csv(info_path, index=False)
    log(f"    Wrote stats to {info_path}")

    # Save metadata
    meta = {
        "tiles": {"x": tiles[0], "y": tiles[1]},
        "pipeline": "blaze",
        "operator_convention": "positive_kinetic",  # H = V + T
        "note": "Data prepared for BLAZE EA solver (no prebuilt operator)",
    }
    save_json(meta, cdir / "phase2_operator_meta.json")

    # Write report
    _write_phase2_blaze_report(cdir, cid, grid_info, stats, eta, tiles, plot_fields)

    # Optional visualization
    if plot_fields:
        plot_phase2_fields(cdir, R_grid, V, mass_tensor, vg_field)
        log("    Rendered phase2_fields_visualization.png")
    else:
        log("    Skipped field visualization (set phase2_plot_fields: true to enable).")

    return {
        "success": True,
        "candidate_id": cid,
        "grid_size": (int(grid_info["Nx"]), int(grid_info["Ny"])),
    }


def _write_phase2_blaze_report(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    field_stats: Dict,
    eta: float,
    tiles: tuple,
    include_plot: bool,
):
    """Emit a human-readable summary of BLAZE Phase 2 results."""
    lines = [
        "# Phase 2 BLAZE Data Preparation Report",
        "",
        f"**Candidate**: {candidate_id:04d}",
        "",
        "## Discretization",
        f"- Grid: {int(grid_info['Nx'])} × {int(grid_info['Ny'])} points",
        f"- Δx = {grid_info['dx']:.5f}, Δy = {grid_info['dy']:.5f}",
        f"- Supercell tiling: {tiles[0]} × {tiles[1]}",
        f"- η (small twist parameter) = {eta:.6f}",
        "",
        "## Input Field Diagnostics",
        f"- Potential V(R): min {field_stats['V_min']:.6f}, max {field_stats['V_max']:.6f}, "
        f"mean {field_stats['V_mean']:.6f}",
        f"- M⁻¹ eigenvalues: min {field_stats['M_inv_min_eig']:.4f}, max {field_stats['M_inv_max_eig']:.4f}",
        "",
        "## BLAZE Pipeline Notes",
        "- This phase prepares data for BLAZE EA solver (Phase 3)",
        "- No sparse operator is prebuilt; BLAZE constructs it internally",
        "- Convention: H = V + (η²/2) ∇·M⁻¹∇ (positive kinetic term)",
        "",
    ]
    if "vg_norm_min" in field_stats:
        lines.insert(-3,
            f"- |v_g(R)|: min {field_stats['vg_norm_min']:.4e}, "
            f"max {field_stats['vg_norm_max']:.4e}, mean {field_stats['vg_norm_mean']:.4e}"
        )

    if include_plot:
        lines.append("- Field visualization saved alongside this report.")
    else:
        lines.append("- Field visualization skipped (set phase2_plot_fields: true to enable).")

    report_path = Path(cdir) / "phase2_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    log(f"    Wrote {report_path}")


def run_phase2_blaze(run_dir: str | Path, config: Dict):
    """Run BLAZE Phase 2 on all candidates in a run directory.
    
    Args:
        run_dir: Path to the BLAZE run directory (or 'auto'/'latest')
        config: Configuration dictionary
    """
    log("\n" + "=" * 70)
    log("PHASE 2 (BLAZE): Data Preparation for EA Solver")
    log("=" * 70)

    plot_fields = bool(config.get("phase2_plot_fields", False))

    run_dir = resolve_blaze_run_dir(run_dir, config)
    log(f"Using run directory: {run_dir}")

    # Load candidate list if available
    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    else:
        log(f"WARNING: {candidates_path} not found; relying solely on per-candidate metadata.")

    # Discover candidates with Phase 1 data
    discovered = _discover_blaze_phase1_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with BLAZE Phase 1 data found in {run_dir}. "
            "Run Phase 1 (BLAZE) before Phase 2."
        )

    # Optionally limit number of candidates
    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        discovered = discovered[:K]
        log(f"Processing {len(discovered)} candidate(s) (limited to K={K}).")
    else:
        log(f"Processing {len(discovered)} candidate(s).")

    results = []
    for cid, _ in discovered:
        row = _load_blaze_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            result = process_candidate(
                row,
                config,
                run_dir,
                plot_fields,
            )
            results.append(result)
        except FileNotFoundError as exc:
            log(f"  Skipping candidate {cid}: {exc}")
        except Exception as exc:
            log(f"  ERROR processing candidate {cid}: {exc}")
            import traceback
            traceback.print_exc()

    log(f"\nPhase 2 (BLAZE) completed: {len(results)}/{len(discovered)} candidates processed.\n")
    return results


def run_phase2(run_dir: str | Path, config_path: str | Path):
    """Entry point for command-line usage."""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return run_phase2_blaze(run_dir, config)


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 2 config."""
    return PROJECT_ROOT / "configs" / "phase2_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest BLAZE run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2("auto", default_config)
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2(sys.argv[1], default_config)
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase2(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python blaze_phases/phase2_blaze.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest BLAZE run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
