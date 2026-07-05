"""
Phase 3 (BLAZE): Envelope Approximation Eigensolver using BLAZE EA Mode

This is the BLAZE-pipeline version of Phase 3. It uses BLAZE's native EA
(Envelope Approximation) solver mode (available in blaze2d >= 0.2.0) to
compute eigenvalues of the envelope Hamiltonian:

    H ψ = E ψ

where:
    H = V(R) + (η²/2) ∇·M⁻¹(R)∇

Note: BLAZE uses positive kinetic convention (H = V + T), unlike the legacy
solver which uses negative kinetic (H = V - T).

BLAZE EA mode is significantly faster than scipy's sparse eigensolver,
using an optimized LOBPCG implementation in Rust.

LIMITATIONS:
- Group velocity (drift) term v_g·∇ is NOT currently supported by BLAZE EA.
  The drift term from Phase 1 data is ignored even if present.

Requirements:
- Phase 2 BLAZE data (phase2_blaze_data.h5) with V(R), M_inv(R), R_grid, eta
- blaze2d >= 0.3.0

Outputs:
- phase3_eigenstates.h5: Eigenvalues and eigenvectors
- phase3_eigenvalues.csv: Tabulated eigenvalue data
- phase3_report.md: Human-readable summary
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Tuple, cast

try:
    from blaze import BulkDriver
except ImportError:
    print("ERROR: blaze2d >= 0.3.0 required. Install with: pip install --upgrade blaze2d")
    sys.exit(1)

# Add parent directory to path for common imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json, save_json
from common.scoring import participation_ratio, field_entropy, localization_length
from common.plotting import plot_envelope_modes


def log(msg: str):
    """Simple logging helper."""
    print(msg, flush=True)


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


def _discover_phase1_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that contain Phase 1 data."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # Check for Phase 1 data (we need V, M_inv directly)
        if (cdir / "phase1_band_data.h5").exists():
            discovered.append((cid, cdir))
    return discovered


def _load_candidate_metadata(
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


def _grid_metrics(R_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics from R_grid."""
    Nx, Ny, _ = R_grid.shape
    x_vals = R_grid[..., 0]
    y_vals = R_grid[..., 1]
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    x_line = np.unique(R_grid[:, 0, 0])
    y_line = np.unique(R_grid[0, :, 1])
    dx = float(np.mean(np.diff(x_line))) if x_line.size > 1 else float("nan")
    dy = float(np.mean(np.diff(y_line))) if y_line.size > 1 else float("nan")
    return {
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "Lx": float(x_max - x_min + dx),
        "Ly": float(y_max - y_min + dy),
        "cell_area": float((x_max - x_min) * (y_max - y_min)),
    }


def _load_phase1_data(cdir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Load data needed for EA solver.
    
    Prefers phase2_blaze_data.h5 (tiled data from BLAZE Phase 2) if available,
    otherwise falls back to phase1_band_data.h5.
    
    Returns:
        R_grid: (Nx, Ny, 2) spatial coordinates
        V: (Nx, Ny) potential field
        M_inv: (Nx, Ny, 2, 2) inverse mass tensor
        omega_ref: reference frequency
        eta: small parameter
    """
    # Prefer Phase 2 prepared data (includes tiling)
    phase2_path = cdir / "phase2_blaze_data.h5"
    if phase2_path.exists():
        log(f"    Loading prepared data from {phase2_path.name}")
        with h5py.File(phase2_path, "r") as hf:
            R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
            V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
            M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
            omega_ref = float(hf.attrs.get("omega_ref", np.nan))
            eta = float(hf.attrs.get("eta", np.nan))
        return R_grid, V, M_inv, omega_ref, eta
    
    # Fall back to Phase 1 data
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    
    log(f"    Loading from {h5_path.name} (no Phase 2 data found)")
    with h5py.File(h5_path, "r") as hf:
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))
        eta = float(hf.attrs.get("eta", np.nan))
    
    return R_grid, V, M_inv, omega_ref, eta


def _export_binary_data(
    temp_dir: Path,
    V: np.ndarray,
    M_inv: np.ndarray,
) -> Tuple[Path, Path]:
    """
    Export V and M_inv to binary format expected by BLAZE EA solver.
    
    BLAZE expects:
    - potential.bin: Nx × Ny f64 values in row-major order
    - mass_inv.bin: Nx × Ny × 4 f64 values [m_xx, m_xy, m_yx, m_yy] row-major
    
    Returns:
        (potential_path, mass_inv_path)
    """
    Nx, Ny = V.shape
    
    # Potential: straightforward (Nx, Ny) -> row-major
    potential_path = temp_dir / "potential.bin"
    V.astype(np.float64).tofile(potential_path)
    
    # Mass inverse: (Nx, Ny, 2, 2) -> (Nx, Ny, 4) with [xx, xy, yx, yy]
    mass_inv_flat = np.zeros((Nx, Ny, 4), dtype=np.float64)
    mass_inv_flat[:, :, 0] = M_inv[:, :, 0, 0]  # m_xx
    mass_inv_flat[:, :, 1] = M_inv[:, :, 0, 1]  # m_xy
    mass_inv_flat[:, :, 2] = M_inv[:, :, 1, 0]  # m_yx
    mass_inv_flat[:, :, 3] = M_inv[:, :, 1, 1]  # m_yy
    
    mass_inv_path = temp_dir / "mass_inv.bin"
    mass_inv_flat.astype(np.float64).tofile(mass_inv_path)
    
    return potential_path, mass_inv_path


def _generate_ea_config(
    temp_dir: Path,
    potential_path: Path,
    mass_inv_path: Path,
    grid_info: Dict[str, float],
    eta: float,
    n_bands: int,
    config: Dict,
) -> Path:
    """
    Generate BLAZE TOML configuration for EA solver mode.
    """
    Nx = int(grid_info["Nx"])
    Ny = int(grid_info["Ny"])
    Lx = grid_info["Lx"]
    Ly = grid_info["Ly"]
    
    # Solver parameters from config
    max_iter = config.get("phase3_solver_maxiter", 500)
    tol = config.get("phase3_solver_tol", 1e-6)
    threads = config.get("blaze_threads", 8)
    
    config_content = f'''# BLAZE EA Config for Phase 3 - Auto-generated
# Envelope Approximation eigensolver

[bulk]
threads = {threads}
verbose = true

[solver]
type = "ea"

[ea]
potential = "{potential_path}"
mass_inv = "{mass_inv_path}"
eta = {eta}
domain_size = [{Lx}, {Ly}]
return_eigenvectors = true

[grid]
nx = {Nx}
ny = {Ny}
lx = {Lx}
ly = {Ly}

[eigensolver]
n_bands = {n_bands}
max_iter = {max_iter}
tol = {tol}

[output]
mode = "full"
directory = "{temp_dir}"
prefix = "ea_result"
'''
    
    config_path = temp_dir / "ea_config.toml"
    config_path.write_text(config_content)
    
    return config_path


def _run_blaze_ea(config_path: Path) -> Tuple[np.ndarray, np.ndarray | None, Tuple[int, int], Dict[str, Any]]:
    """
    Run BLAZE EA solver and extract results.
    
    Returns:
        eigenvalues: 1D array of eigenvalues
        eigenvectors: 3D array (n_modes, Nx, Ny) of complex eigenvectors, or None
        grid_dims: (Nx, Ny) tuple
        solver_info: Dict with solver metadata
    """
    driver = BulkDriver(str(config_path))
    
    log(f"    Running BLAZE EA solver ({driver.job_count} job)...")
    start_time = time.time()
    
    # Use streaming to get results (works with both old and new BLAZE versions)
    results = list(driver.run_streaming())
    
    elapsed = time.time() - start_time
    log(f"    BLAZE completed in {elapsed:.2f}s")
    
    if not results:
        raise RuntimeError("BLAZE EA solver returned no results")
    
    # EA mode returns a single result with eigenvalues
    result = results[0]
    
    if result.get("result_type") != "ea":
        raise RuntimeError(f"Expected EA result, got {result.get('result_type')}")
    
    eigenvalues = np.array(result["eigenvalues"])
    grid_dims = tuple(result.get("grid_dims", [0, 0]))
    
    # Extract eigenvectors if available (BLAZE >= 0.3.0)
    eigenvectors = None
    raw_eigenvectors = result.get("eigenvectors")
    if raw_eigenvectors is not None and len(raw_eigenvectors) > 0:
        try:
            # Eigenvectors are complex arrays of shape (nx*ny, 2) -> [re, im]
            n_modes = len(raw_eigenvectors)
            nx, ny = grid_dims
            eigenvectors = np.zeros((n_modes, nx, ny), dtype=np.complex128)
            
            for i, psi_raw in enumerate(raw_eigenvectors):
                psi_arr = np.array(psi_raw)
                # Convert [re, im] columns to complex
                psi_complex = psi_arr[:, 0] + 1j * psi_arr[:, 1]
                eigenvectors[i] = psi_complex.reshape((nx, ny))
            
            log(f"    Extracted {n_modes} eigenvectors from BLAZE")
        except Exception as exc:
            log(f"    WARNING: Failed to extract eigenvectors: {exc}")
            eigenvectors = None
    
    solver_info = {
        "n_iterations": result.get("n_iterations", -1),
        "converged": result.get("converged", True),
        "num_eigenvalues": result.get("num_eigenvalues", len(eigenvalues)),
        "total_time_secs": elapsed,
        "solver": "blaze_ea",
        "has_eigenvectors": eigenvectors is not None,
    }
    
    return eigenvalues, eigenvectors, grid_dims, solver_info


def _analyze_modes_with_eigenvectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    R_grid: np.ndarray,
    omega_ref: float,
) -> Tuple[np.ndarray, Iterable[Dict[str, float]]]:
    """
    Compute mode metrics from eigenvalues and eigenvectors.
    
    This is the full analysis with localization metrics.
    
    Args:
        eigenvalues: 1D array of eigenvalues
        eigenvectors: 3D array (n_modes, Nx, Ny) of complex eigenvectors
        R_grid: (Nx, Ny, 2) spatial coordinates
        omega_ref: reference frequency
    
    Returns:
        fields: 3D array (n_modes, Nx, Ny) - same as eigenvectors input
        rows: list of metric dicts per mode
    """
    Nx, Ny, _ = R_grid.shape
    n_modes = eigenvectors.shape[0]
    fields = eigenvectors  # Already in (n_modes, Nx, Ny) shape
    rows = []
    
    for idx in range(n_modes):
        field = fields[idx]
        abs_field = np.abs(field)
        abs2 = abs_field ** 2
        norm_l2 = float(np.linalg.norm(field.ravel()))
        delta_omega = eigenvalues[idx]
        delta_real = float(delta_omega.real if np.iscomplexobj(delta_omega) else delta_omega)
        omega_cav = float(omega_ref + delta_real)
        pr = float(participation_ratio(field))
        entropy = float(field_entropy(field))
        loc_len = float(localization_length(field, R_grid))
        weight = abs2 / max(abs2.sum(), 1e-20)
        center = np.tensordot(weight, R_grid, axes=([0, 1], [0, 1]))
        peak_idx = np.unravel_index(np.argmax(abs2), abs2.shape)
        peak_pos = R_grid[peak_idx]
        rows.append({
            "mode_index": idx,
            "delta_omega": delta_real,
            "omega_cavity": omega_cav,
            "norm_l2": norm_l2,
            "participation_ratio": pr,
            "entropy": entropy,
            "localization_length": loc_len,
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "peak_x": float(peak_pos[0]),
            "peak_y": float(peak_pos[1]),
            "max_amplitude": float(abs_field.max()),
        })
    
    return fields, rows


def _analyze_modes(
    eigenvalues: np.ndarray,
    R_grid: np.ndarray,
    omega_ref: float,
) -> Iterable[Dict[str, float]]:
    """
    Compute mode metrics from eigenvalues only.
    
    Note: BLAZE EA solver returns only eigenvalues, not eigenvectors,
    so localization metrics are not available.
    """
    rows = []
    for idx, delta_omega in enumerate(eigenvalues):
        delta_real = float(delta_omega.real if np.iscomplexobj(delta_omega) else delta_omega)
        omega_cav = float(omega_ref + delta_real)
        
        rows.append({
            "mode_index": idx,
            "delta_omega": delta_real,
            "omega_cavity": omega_cav,
            # Eigenvector-dependent metrics not available from BLAZE EA
            "participation_ratio": np.nan,
            "entropy": np.nan,
            "localization_length": np.nan,
            "center_x": np.nan,
            "center_y": np.nan,
            "peak_x": np.nan,
            "peak_y": np.nan,
            "max_amplitude": np.nan,
            "norm_l2": np.nan,
        })
    
    return rows


def _write_phase3_report(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    eigenvalues: np.ndarray,
    omega_ref: float,
    eta: float,
    solver_info: Dict[str, Any],
    mode_rows: Iterable[Dict[str, float]],
):
    """Write human-readable Phase 3 report."""
    report_path = cdir / "phase3_report.md"
    rows = list(mode_rows)
    has_eigenvectors = solver_info.get("has_eigenvectors", False)
    
    lines = [
        "# Phase 3 Envelope Solver Report (BLAZE EA)",
        "",
        f"**Candidate**: {candidate_id:04d}",
        "",
        "## Discretization",
        f"- Grid: {int(grid_info['Nx'])} × {int(grid_info['Ny'])} points",
        f"- Δx = {grid_info['dx']:.5f}, Δy = {grid_info['dy']:.5f}",
        f"- Domain: [{grid_info['Lx']:.4f} × {grid_info['Ly']:.4f}]",
        f"- η (small twist parameter) = {eta:.6f}",
        "",
        "## Solver",
        f"- Solver: BLAZE EA (LOBPCG)",
        f"- Iterations: {solver_info.get('n_iterations', 'N/A')}",
        f"- Converged: {solver_info.get('converged', 'N/A')}",
        f"- Time: {solver_info.get('total_time_secs', 0):.2f}s",
        f"- ω_ref = {omega_ref:.6f}",
        f"- Eigenvectors available: {'Yes' if has_eigenvectors else 'No'}",
        "",
        "## Lowest Eigenvalues (Δω from ω_ref)",
    ]
    
    # Include localization metrics if eigenvectors are available
    for row in rows[:min(12, len(rows))]:
        if has_eigenvectors and not np.isnan(row.get('participation_ratio', np.nan)):
            lines.append(
                f"- Mode {int(row['mode_index'])}: Δω = {row['delta_omega']:.6e}, "
                f"ω = {row['omega_cavity']:.6e}, PR = {row['participation_ratio']:.3f}, "
                f"ξ = {row['localization_length']:.3f}"
            )
        else:
            lines.append(
                f"- Mode {int(row['mode_index'])}: Δω = {row['delta_omega']:.6e}, "
                f"ω = {row['omega_cavity']:.6e}"
            )
    
    lines.extend([
        "",
        "## Notes",
    ])
    
    if has_eigenvectors:
        lines.extend([
            "- BLAZE EA solver (v0.3.0+) returned eigenvalues and eigenvectors",
            "- Full mode localization metrics computed",
            "- Cavity mode plots generated (phase3_cavity_modes.png, phase3_spectrum.png)",
        ])
    else:
        lines.extend([
            "- BLAZE EA solver returned eigenvalues only (no eigenvectors)",
            "- Mode localization metrics are not available",
            "- Upgrade to blaze2d >= 0.3.0 for eigenvector support",
        ])
    
    report_path.write_text("\n".join(lines) + "\n")
    log(f"    Wrote {report_path}")


def process_candidate(
    row: Dict,
    config: Dict,
    run_dir: Path,
):
    """
    Process a single candidate through BLAZE Phase 3.
    
    This:
    1. Loads Phase 1 data (V, M_inv)
    2. Exports binary files for BLAZE
    3. Generates EA config TOML
    4. Runs BLAZE EA solver
    5. Analyzes modes with full eigenvector metrics (if available)
    6. Saves results and generates cavity plots
    """
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    
    log(f"  Candidate {cid}: loading Phase 1 data...")
    
    # Load Phase 1 data
    R_grid, V, M_inv, omega_ref, eta = _load_phase1_data(cdir)
    
    # Handle missing eta
    if not np.isfinite(eta):
        a = float(row.get("a", np.nan))
        L_m = float(row.get("moire_length", np.nan))
        if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
            eta = a / L_m
        else:
            log("    WARNING: Missing eta; defaulting to 0.1")
            eta = 0.1
    
    if not np.isfinite(omega_ref):
        omega_ref = float(row.get("omega0", 0.0))
    
    grid_info = _grid_metrics(R_grid)
    n_modes = int(config.get("ea_n_modes", 12))
    
    log(f"    Grid: {int(grid_info['Nx'])}×{int(grid_info['Ny'])}, η={eta:.6f}")
    log(f"    Computing {n_modes} eigenvalues...")
    
    # Create temporary directory for BLAZE files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export binary data
        potential_path, mass_inv_path = _export_binary_data(temp_path, V, M_inv)
        
        # Generate EA config
        config_path = _generate_ea_config(
            temp_path,
            potential_path,
            mass_inv_path,
            grid_info,
            eta,
            n_modes,
            config,
        )
        
        # Run BLAZE EA solver
        eigenvalues, eigenvectors, grid_dims, solver_info = _run_blaze_ea(config_path)
    
    log(f"    Got {len(eigenvalues)} eigenvalues")
    
    # Analyze modes - use eigenvectors if available for full metrics
    if eigenvectors is not None:
        fields, mode_rows = _analyze_modes_with_eigenvectors(
            eigenvalues, eigenvectors, R_grid, omega_ref
        )
        log(f"    Computed full mode metrics with eigenvectors")
    else:
        fields = None
        mode_rows = list(_analyze_modes(eigenvalues, R_grid, omega_ref))
        log(f"    Eigenvectors not available; limited mode metrics")
    
    # Save eigenvalues and eigenvectors to HDF5
    eig_h5 = cdir / "phase3_eigenstates.h5"
    with h5py.File(eig_h5, "w") as hf:
        hf.create_dataset("eigenvalues", data=eigenvalues)
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        if fields is not None:
            hf.create_dataset("F", data=fields, compression="gzip")
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["solver"] = "blaze_ea"
        hf.attrs["n_iterations"] = solver_info.get("n_iterations", -1)
        hf.attrs["converged"] = solver_info.get("converged", True)
        hf.attrs["has_eigenvectors"] = fields is not None
    log(f"    Saved eigenstates to {eig_h5}")
    
    # Save CSV table
    df = pd.DataFrame(mode_rows)
    df.insert(0, "candidate_id", cid)
    df_path = cdir / "phase3_eigenvalues.csv"
    df.to_csv(df_path, index=False)
    log(f"    Wrote eigenvalue table to {df_path}")
    
    # Generate cavity mode plots if we have eigenvectors
    if fields is not None:
        eigenvalues_for_plots = eigenvalues.real if np.iscomplexobj(eigenvalues) else eigenvalues
        plot_envelope_modes(
            cdir,
            R_grid,
            fields,
            eigenvalues_for_plots,
            n_modes=n_modes,
            candidate_params=row,
        )
        log(f"    Generated cavity mode plots")
    else:
        log(f"    Skipped cavity plots (no eigenvectors)")
    
    # Write report
    _write_phase3_report(
        cdir, cid, grid_info, eigenvalues, omega_ref, eta, solver_info, mode_rows
    )
    
    # Save solver metadata
    save_json({
        **solver_info,
        "eta": eta,
        "omega_ref": omega_ref,
        "n_modes_requested": n_modes,
        "n_modes_returned": len(eigenvalues),
        "grid_nx": int(grid_info["Nx"]),
        "grid_ny": int(grid_info["Ny"]),
        "pipeline": "blaze",
    }, cdir / "phase3_solver_meta.json")
    
    return {
        "success": True,
        "candidate_id": cid,
        "n_eigenvalues": len(eigenvalues),
        "lowest_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else np.nan,
        "has_eigenvectors": fields is not None,
    }


def run_phase3_blaze(run_dir: str | Path, config: Dict):
    """
    Run BLAZE Phase 3 on all candidates in a run directory.
    
    Args:
        run_dir: Path to the BLAZE run directory (or 'auto'/'latest')
        config: Configuration dictionary
    """
    log("\n" + "=" * 70)
    log("PHASE 3 (BLAZE): Envelope Approximation Eigensolver")
    log("=" * 70)
    
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
    discovered = _discover_phase1_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 1 data found in {run_dir}. "
            "Run Phase 1 (BLAZE) before Phase 3."
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
        row = _load_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            result = process_candidate(row, config, run_dir)
            results.append(result)
            log(f"    ✓ Candidate {cid}: {result['n_eigenvalues']} eigenvalues, "
                f"E₀={result['lowest_eigenvalue']:.6e}")
        except FileNotFoundError as exc:
            log(f"  Skipping candidate {cid}: {exc}")
        except Exception as exc:
            log(f"  ERROR processing candidate {cid}: {exc}")
            import traceback
            traceback.print_exc()
    
    log(f"\nPhase 3 (BLAZE) completed: {len(results)}/{len(discovered)} candidates processed.\n")
    return results


def run_phase3(run_dir: str | Path, config_path: str | Path):
    """Entry point for command-line usage."""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return run_phase3_blaze(run_dir, config)


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 3 config."""
    return PROJECT_ROOT / "configs" / "phase3_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest BLAZE run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase3("auto", default_config)
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase3(sys.argv[1], default_config)
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase3(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python blaze_phases/phase3_blaze.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest BLAZE run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
