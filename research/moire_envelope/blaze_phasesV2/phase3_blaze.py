"""
Phase 3 (BLAZE): Envelope Approximation Eigensolver — V2 Pipeline

This is the BLAZE-pipeline version of Phase 3 for the V2 coordinate system.
It uses BLAZE's native EA (Envelope Approximation) solver mode to compute
eigenvalues of the envelope Hamiltonian in FRACTIONAL COORDINATES.

KEY V2 CHANGES from legacy BLAZE Phase 3:
1. Operates on unit square [0,1)² with domain_size = [1.0, 1.0]
2. Reads transformed mass tensor M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ from Phase 2
3. Stores eigenstates F(s1, s2) on fractional grid
4. Includes B_moire for Cartesian visualization transforms
5. pipeline_version="V2", coordinate_system="fractional"

The eigenvalue problem in fractional coordinates:
    H_EA ψ(s) = Δω ψ(s)

where:
    H_EA = V(s) + (η²/2) ∇_s · M̃⁻¹(s) ∇_s

Note: BLAZE uses positive kinetic convention (H = V + T), unlike the legacy
MPB solver which uses negative kinetic (H = V - T).

BLAZE EA solver configuration (from ea_moire_lattice.toml):
- [solver] type = "ea"
- [ea] potential, mass_inv, eta, domain_size, periodic
- [grid] nx, ny, lx, ly (for V2: lx=ly=1.0)
- [eigensolver] n_bands, max_iter, tol

Binary file format: f64 little-endian, row-major (C-order)
- Potential:   Ns1×Ns2 values
- Mass inverse: Ns1×Ns2×4 values [m_xx, m_xy, m_yx, m_yy] per grid point

Based on README_V2.md and examples/ea_moire_lattice.toml.
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
    from blaze2d import BulkDriver
except ImportError:
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


# =============================================================================
# Run Directory Resolution
# =============================================================================

def resolve_blaze_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to BLAZE Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        # V2: look in runsV2 directory for BLAZE runs only
        base = Path(config.get("output_dir", "runsV2"))
        blaze_runs = sorted(base.glob("phase0_blaze_*"))
        if not blaze_runs:
            raise FileNotFoundError(
                f"No BLAZE phase0 run directories found in {base}\n"
                f"  (Looking for phase0_blaze_*)\n"
                f"  Found MPB runs? Use explicit path instead."
            )
        run_dir = blaze_runs[-1]
        log(f"Auto-selected latest BLAZE Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runsV2")) / run_dir
        if candidate.exists():
            run_dir = candidate
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


def _discover_phase2_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that contain Phase 2 V2 data."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        # Check for Phase 2 BLAZE data
        h5_path = cdir / "phase2_blaze_data.h5"
        if h5_path.exists():
            # Verify it's V2 data
            try:
                with h5py.File(h5_path, 'r') as hf:
                    version = hf.attrs.get('pipeline_version', 'V1')
                    if version == 'V2':
                        discovered.append((cid, cdir))
                    else:
                        log(f"  Skipping {cdir.name}: not V2 pipeline data (version={version})")
            except Exception as e:
                log(f"  Warning: Could not read {h5_path}: {e}")
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


# =============================================================================
# Grid Metrics (V2: Fractional Coordinates)
# =============================================================================

def _grid_metrics_fractional(s_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics for fractional coordinate system (unit square)."""
    Ns1, Ns2, _ = s_grid.shape
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    return {
        "Ns1": Ns1,
        "Ns2": Ns2,
        "ds1": ds1,
        "ds2": ds2,
        "s1_min": 0.0,
        "s1_max": 1.0 - ds1,
        "s2_min": 0.0,
        "s2_max": 1.0 - ds2,
        "Ls1": 1.0,  # Unit square
        "Ls2": 1.0,
        "cell_area": 1.0,
    }


# =============================================================================
# Phase 2 Data Loading (V2)
# =============================================================================

def _load_phase2_data_v2(cdir: Path) -> Dict[str, Any]:
    """
    Load Phase 2 V2 data for EA solver.
    
    Returns dict with:
        s_grid: (Ns1, Ns2, 2) fractional coordinates
        R_grid: (Ns1, Ns2, 2) Cartesian coordinates (for visualization)
        V: (Ns1, Ns2) potential field
        M_inv_tilde: (Ns1, Ns2, 2, 2) transformed mass tensor
        omega_ref: reference frequency
        eta: small parameter
        B_moire: (2, 2) moiré basis matrix
    """
    h5_path = cdir / "phase2_blaze_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 V2 data: {h5_path}")
    
    log(f"    Loading V2 data from {h5_path.name}")
    with h5py.File(h5_path, "r") as hf:
        data = {
            "s_grid": np.asarray(cast(h5py.Dataset, hf["s_grid"])[:]),
            "R_grid": np.asarray(cast(h5py.Dataset, hf["R_grid"])[:]),
            "V": np.asarray(cast(h5py.Dataset, hf["V"])[:]),
            "M_inv_tilde": np.asarray(cast(h5py.Dataset, hf["M_inv_tilde"])[:]),
            "omega_ref": float(hf.attrs.get("omega_ref", np.nan)),
            "eta": float(hf.attrs.get("eta", np.nan)),
            "B_moire": np.asarray(hf.attrs.get("B_moire")) if "B_moire" in hf.attrs else np.eye(2),
            "B_mono": np.asarray(hf.attrs.get("B_mono")) if "B_mono" in hf.attrs else None,
            "theta_rad": float(hf.attrs.get("theta_rad", 0.0)),
            "pipeline_version": str(hf.attrs.get("pipeline_version", "V2")),
        }
        # Optional: original (untransformed) mass tensor
        if "M_inv" in hf:
            data["M_inv"] = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        # NOTE: vg is not passed to BLAZE EA solver - skipping
        # if "vg_tilde" in hf:
        #     data["vg_tilde"] = np.asarray(cast(h5py.Dataset, hf["vg_tilde"])[:])
    
    return data


# =============================================================================
# Binary Export for BLAZE EA
# =============================================================================

def _export_binary_data_v2(
    temp_dir: Path,
    V: np.ndarray,
    M_inv_tilde: np.ndarray,
) -> Tuple[Path, Path]:
    """
    Export V and M̃⁻¹ to binary format expected by BLAZE EA solver.
    
    BLAZE expects:
    - potential.bin: Ns1 × Ns2 f64 values in row-major order
    - mass_inv.bin: Ns1 × Ns2 × 4 f64 values [m_xx, m_xy, m_yx, m_yy] row-major
    
    For V2: Uses TRANSFORMED mass tensor M̃⁻¹ (not original M⁻¹).
    
    Returns:
        (potential_path, mass_inv_path)
    """
    Ns1, Ns2 = V.shape
    
    # Potential: straightforward (Ns1, Ns2) -> row-major
    potential_path = temp_dir / "potential.bin"
    V.astype(np.float64).tofile(potential_path)
    
    # Mass inverse (TRANSFORMED): (Ns1, Ns2, 2, 2) -> (Ns1, Ns2, 4) with [xx, xy, yx, yy]
    mass_inv_flat = np.zeros((Ns1, Ns2, 4), dtype=np.float64)
    mass_inv_flat[:, :, 0] = M_inv_tilde[:, :, 0, 0]  # m̃_xx
    mass_inv_flat[:, :, 1] = M_inv_tilde[:, :, 0, 1]  # m̃_xy
    mass_inv_flat[:, :, 2] = M_inv_tilde[:, :, 1, 0]  # m̃_yx
    mass_inv_flat[:, :, 3] = M_inv_tilde[:, :, 1, 1]  # m̃_yy
    
    mass_inv_path = temp_dir / "mass_inv.bin"
    mass_inv_flat.astype(np.float64).tofile(mass_inv_path)
    
    log(f"    Exported V ({V.shape}) and M̃⁻¹ ({M_inv_tilde.shape}) to binary")
    
    return potential_path, mass_inv_path


# =============================================================================
# BLAZE EA Configuration (V2)
# =============================================================================

def _generate_ea_config_v2(
    temp_dir: Path,
    potential_path: Path,
    mass_inv_path: Path,
    grid_info: Dict[str, float],
    eta: float,
    n_bands: int,
    config: Dict,
) -> Path:
    """
    Generate BLAZE TOML configuration for EA solver mode (V2).
    
    Key V2 difference: domain_size = [1.0, 1.0] (unit square in fractional coords)
    
    Based on examples/ea_moire_lattice.toml
    """
    Ns1 = int(grid_info["Ns1"])
    Ns2 = int(grid_info["Ns2"])
    
    # V2: Domain is unit square [0,1)²
    Ls1 = 1.0
    Ls2 = 1.0
    
    # Solver parameters from config
    max_iter = config.get("phase3_solver_maxiter", 500)
    tol = config.get("phase3_solver_tol", 1e-6)
    threads = config.get("blaze_threads", 0)  # 0 = auto (all cores)
    
    # Periodic boundary conditions (natural for unit square)
    periodic = config.get("phase3_periodic", True)
    
    config_content = f'''# BLAZE EA Config for Phase 3 V2 - Auto-generated
# Envelope Approximation eigensolver on unit square [0,1)²
#
# V2 formulation: H = V(s) + (η²/2) ∇_s · M̃⁻¹(s) ∇_s
# where M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ (transformed mass tensor)

# =============================================================================
# BULK SECTION
# =============================================================================

[bulk]
threads = {threads}                    # Number of parallel threads (0 = all cores)
verbose = true                         # Print progress messages

# =============================================================================
# SOLVER TYPE
# =============================================================================

[solver]
type = "ea"                            # Envelope Approximation solver

# =============================================================================
# EA-SPECIFIC CONFIGURATION
# =============================================================================

[ea]
# Binary input files (f64 little-endian, row-major)
potential = "{potential_path}"         # V(s): Ns1×Ns2 f64 values
mass_inv = "{mass_inv_path}"           # M̃⁻¹(s): Ns1×Ns2×4 f64 [m_xx, m_xy, m_yx, m_yy]

eta = {eta}                            # Small parameter η for kinetic term scale
domain_size = [{Ls1}, {Ls2}]           # V2: Unit square [0,1)²
periodic = {str(periodic).lower()}     # Periodic boundary conditions

# =============================================================================
# COMPUTATIONAL GRID
# =============================================================================

[grid]
nx = {Ns1}                             # Grid points in s1 direction
ny = {Ns2}                             # Grid points in s2 direction
lx = {Ls1}                             # Domain size s1 (unit square)
ly = {Ls2}                             # Domain size s2 (unit square)

# =============================================================================
# EIGENSOLVER CONFIGURATION
# =============================================================================

[eigensolver]
n_bands = {n_bands}                    # Number of eigenvalues to compute
max_iter = {max_iter}                  # Maximum LOBPCG iterations
tol = {tol}                            # Convergence tolerance

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

[output]
mode = "full"                          # "full" (one file per job) or "selective"
directory = "{temp_dir}"               # Output directory
prefix = "ea_result_v2"                # Filename prefix
'''
    
    config_path = temp_dir / "ea_config_v2.toml"
    config_path.write_text(config_content)
    
    log(f"    Generated BLAZE EA config: {config_path.name}")
    log(f"    Domain: [{Ls1}, {Ls2}] (unit square), Grid: {Ns1}×{Ns2}, η={eta:.6f}")
    
    return config_path


# =============================================================================
# BLAZE EA Execution
# =============================================================================

def _run_blaze_ea_v2(config_path: Path) -> Tuple[np.ndarray, np.ndarray | None, Tuple[int, int], Dict[str, Any]]:
    """
    Run BLAZE EA solver and extract results.
    
    Returns:
        eigenvalues: 1D array of eigenvalues (Δω from ω_ref)
        eigenvectors: 3D array (n_modes, Ns1, Ns2) of complex eigenvectors, or None
        grid_dims: (Ns1, Ns2) tuple
        solver_info: Dict with solver metadata
    """
    driver = BulkDriver(str(config_path))
    
    log(f"    Running BLAZE EA solver ({driver.job_count} job)...")
    start_time = time.time()
    
    # Use streaming to get results
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
            n_modes = len(raw_eigenvectors)
            ns1, ns2 = grid_dims
            eigenvectors = np.zeros((n_modes, ns1, ns2), dtype=np.complex128)
            
            for i, psi_raw in enumerate(raw_eigenvectors):
                psi_arr = np.array(psi_raw)
                # Convert [re, im] columns to complex
                psi_complex = psi_arr[:, 0] + 1j * psi_arr[:, 1]
                eigenvectors[i] = psi_complex.reshape((ns1, ns2))
            
            log(f"    Extracted {n_modes} eigenvectors from BLAZE")
        except Exception as exc:
            log(f"    WARNING: Failed to extract eigenvectors: {exc}")
            eigenvectors = None
    
    solver_info = {
        "n_iterations": result.get("n_iterations", -1),
        "converged": result.get("converged", True),
        "num_eigenvalues": result.get("num_eigenvalues", len(eigenvalues)),
        "total_time_secs": elapsed,
        "solver": "blaze_ea_v2",
        "has_eigenvectors": eigenvectors is not None,
        "coordinate_system": "fractional",
    }
    
    return eigenvalues, eigenvectors, grid_dims, solver_info


# =============================================================================
# Mode Analysis (V2: Fractional Coordinates)
# =============================================================================

def _analyze_modes_v2(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray | None,
    s_grid: np.ndarray,
    R_grid: np.ndarray,
    B_moire: np.ndarray,
    omega_ref: float,
) -> Tuple[np.ndarray | None, list[Dict[str, float]]]:
    """
    Analyze modes in V2 fractional coordinates.
    
    Localization metrics are computed in fractional space, then
    scaled to physical units using B_moire.
    
    Args:
        eigenvalues: 1D array of eigenvalues
        eigenvectors: 3D array (n_modes, Ns1, Ns2) or None
        s_grid: (Ns1, Ns2, 2) fractional coordinates
        R_grid: (Ns1, Ns2, 2) Cartesian coordinates
        B_moire: (2, 2) moiré basis matrix
        omega_ref: reference frequency
    
    Returns:
        fields: eigenvectors (or None)
        rows: list of metric dicts per mode
    """
    rows = []
    Ns1, Ns2, _ = s_grid.shape
    moire_area = abs(np.linalg.det(B_moire))  # Physical area of moiré cell
    
    for idx, delta_omega in enumerate(eigenvalues):
        delta_real = float(delta_omega.real if np.iscomplexobj(delta_omega) else delta_omega)
        omega_cav = float(omega_ref + delta_real)
        
        row = {
            "mode_index": idx,
            "delta_omega": delta_real,
            "omega_cavity": omega_cav,
        }
        
        if eigenvectors is not None and idx < eigenvectors.shape[0]:
            field = eigenvectors[idx]
            abs_field = np.abs(field)
            abs2 = abs_field ** 2
            norm_l2 = float(np.linalg.norm(field.ravel()))
            
            # Localization metrics (computed in fractional space)
            pr = float(participation_ratio(field))
            entropy = float(field_entropy(field))
            
            # Localization length in fractional coordinates
            loc_len_frac = float(localization_length(field, s_grid))
            # Scale to physical units: ξ_phys ≈ ξ_frac * sqrt(|det(B)|)
            loc_len_phys = loc_len_frac * np.sqrt(moire_area)
            
            # Center of mass in fractional and Cartesian
            weight = abs2 / max(abs2.sum(), 1e-20)
            center_frac = np.tensordot(weight, s_grid, axes=([0, 1], [0, 1]))
            center_cart = B_moire @ center_frac
            
            # Peak position
            peak_idx = np.unravel_index(np.argmax(abs2), abs2.shape)
            peak_frac = s_grid[peak_idx]
            peak_cart = R_grid[peak_idx]
            
            row.update({
                "norm_l2": norm_l2,
                "participation_ratio": pr,
                "entropy": entropy,
                "localization_length_frac": loc_len_frac,
                "localization_length": loc_len_phys,
                # Fractional coordinates
                "center_s1": float(center_frac[0]),
                "center_s2": float(center_frac[1]),
                "peak_s1": float(peak_frac[0]),
                "peak_s2": float(peak_frac[1]),
                # Cartesian coordinates (for visualization)
                "center_x": float(center_cart[0]),
                "center_y": float(center_cart[1]),
                "peak_x": float(peak_cart[0]),
                "peak_y": float(peak_cart[1]),
                "max_amplitude": float(abs_field.max()),
            })
        else:
            # No eigenvector data
            row.update({
                "norm_l2": np.nan,
                "participation_ratio": np.nan,
                "entropy": np.nan,
                "localization_length_frac": np.nan,
                "localization_length": np.nan,
                "center_s1": np.nan,
                "center_s2": np.nan,
                "peak_s1": np.nan,
                "peak_s2": np.nan,
                "center_x": np.nan,
                "center_y": np.nan,
                "peak_x": np.nan,
                "peak_y": np.nan,
                "max_amplitude": np.nan,
            })
        
        rows.append(row)
    
    return eigenvectors, rows


# =============================================================================
# Report Generation (V2)
# =============================================================================

def _write_phase3_report_v2(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    eigenvalues: np.ndarray,
    omega_ref: float,
    eta: float,
    B_moire: np.ndarray,
    solver_info: Dict[str, Any],
    mode_rows: list[Dict[str, float]],
):
    """Write human-readable Phase 3 V2 report."""
    report_path = cdir / "phase3_report.md"
    has_eigenvectors = solver_info.get("has_eigenvectors", False)
    
    lines = [
        "# Phase 3 Envelope Solver Report (BLAZE EA V2)",
        "",
        f"**Candidate**: {candidate_id:04d}",
        f"**Pipeline Version**: V2 (Fractional Coordinates)",
        "",
        "## Coordinate System",
        f"- **Type**: Fractional (unit square [0,1)²)",
        f"- Grid: {int(grid_info['Ns1'])} × {int(grid_info['Ns2'])} points",
        f"- Δs₁ = {grid_info['ds1']:.6f}, Δs₂ = {grid_info['ds2']:.6f}",
        f"- Domain: unit square [0,1) × [0,1)",
        "",
        "## Moiré Basis",
        f"- B_moire:",
        f"  - A₁ = [{B_moire[0, 0]:.4f}, {B_moire[1, 0]:.4f}]",
        f"  - A₂ = [{B_moire[0, 1]:.4f}, {B_moire[1, 1]:.4f}]",
        f"- |det(B_moire)| = {abs(np.linalg.det(B_moire)):.4f} (physical cell area)",
        "",
        "## Physics Parameters",
        f"- η (moiré scale ratio) = {eta:.6f}",
        f"- ω_ref = {omega_ref:.6f}",
        "",
        "## Solver",
        f"- Solver: BLAZE EA (LOBPCG) V2",
        f"- Iterations: {solver_info.get('n_iterations', 'N/A')}",
        f"- Converged: {solver_info.get('converged', 'N/A')}",
        f"- Time: {solver_info.get('total_time_secs', 0):.2f}s",
        f"- Eigenvectors available: {'Yes' if has_eigenvectors else 'No'}",
        "",
        "## Lowest Eigenvalues (Δω from ω_ref)",
    ]
    
    for row in mode_rows[:min(12, len(mode_rows))]:
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
        "## V2 Key Features",
        "- **Fractional coordinates**: Eigenstates F(s₁, s₂) on [0,1)²",
        "- **Transformed mass tensor**: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ",
        "- **Positive kinetic**: H = V + (η²/2) ∇_s · M̃⁻¹ ∇_s",
        "- **Localization metrics**: Computed in fractional space, scaled by √|det(B)|",
        "",
    ])
    
    if has_eigenvectors:
        lines.append("- Cavity mode plots generated (R_grid for visualization)")
    else:
        lines.append("- Upgrade to blaze2d >= 0.3.0 for eigenvector support")
    
    report_path.write_text("\n".join(lines) + "\n")
    log(f"    Wrote {report_path}")


# =============================================================================
# Candidate Processing (V2)
# =============================================================================

def process_candidate_v2(
    row: Dict,
    config: Dict,
    run_dir: Path,
):
    """
    Process a single candidate through BLAZE Phase 3 V2.
    
    Steps:
    1. Load Phase 2 V2 data (V, M̃⁻¹ on unit square)
    2. Export binary files for BLAZE
    3. Generate EA config TOML (domain_size=[1,1])
    4. Run BLAZE EA solver
    5. Analyze modes with V2 metrics
    6. Save results (s_grid as primary, R_grid for visualization)
    """
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    
    log(f"  Candidate {cid}: loading Phase 2 V2 data...")
    
    # Load Phase 2 V2 data
    data = _load_phase2_data_v2(cdir)
    s_grid = data["s_grid"]
    R_grid = data["R_grid"]
    V = data["V"]
    M_inv_tilde = data["M_inv_tilde"]
    omega_ref = data["omega_ref"]
    eta = data["eta"]
    B_moire = data["B_moire"]
    
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
    
    grid_info = _grid_metrics_fractional(s_grid)
    n_modes = int(config.get("ea_n_modes", 12))
    
    log(f"    Grid: {int(grid_info['Ns1'])}×{int(grid_info['Ns2'])}, η={eta:.6f}")
    log(f"    Computing {n_modes} eigenvalues on unit square...")
    
    # Create temporary directory for BLAZE files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export binary data (V2: uses transformed M̃⁻¹)
        potential_path, mass_inv_path = _export_binary_data_v2(temp_path, V, M_inv_tilde)
        
        # Generate EA config (V2: domain_size=[1,1])
        config_path = _generate_ea_config_v2(
            temp_path,
            potential_path,
            mass_inv_path,
            grid_info,
            eta,
            n_modes,
            config,
        )
        
        # Run BLAZE EA solver
        eigenvalues, eigenvectors, grid_dims, solver_info = _run_blaze_ea_v2(config_path)
    
    log(f"    Got {len(eigenvalues)} eigenvalues")
    
    # Analyze modes with V2 metrics
    fields, mode_rows = _analyze_modes_v2(
        eigenvalues, eigenvectors, s_grid, R_grid, B_moire, omega_ref
    )
    
    if eigenvectors is not None:
        log(f"    Computed full mode metrics with eigenvectors")
    else:
        log(f"    Eigenvectors not available; limited mode metrics")
    
    # Save eigenvalues and eigenvectors to HDF5 (V2 format)
    eig_h5 = cdir / "phase3_eigenstates.h5"
    with h5py.File(eig_h5, "w") as hf:
        hf.create_dataset("eigenvalues", data=eigenvalues)
        # V2: s_grid is primary
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")  # For visualization
        if fields is not None:
            hf.create_dataset("F", data=fields, compression="gzip")  # F(s1, s2)
        # Attributes
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["B_moire"] = B_moire
        hf.attrs["solver"] = "blaze_ea_v2"
        hf.attrs["n_iterations"] = solver_info.get("n_iterations", -1)
        hf.attrs["converged"] = solver_info.get("converged", True)
        hf.attrs["has_eigenvectors"] = fields is not None
        hf.attrs["pipeline_version"] = "V2"
        hf.attrs["coordinate_system"] = "fractional"
    log(f"    Saved eigenstates to {eig_h5}")
    
    # Save CSV table
    df = pd.DataFrame(mode_rows)
    df.insert(0, "candidate_id", cid)
    df_path = cdir / "phase3_eigenvalues.csv"
    df.to_csv(df_path, index=False)
    log(f"    Wrote eigenvalue table to {df_path}")
    
    # Generate cavity mode plots if we have eigenvectors (use R_grid for visualization)
    if fields is not None:
        eigenvalues_for_plots = eigenvalues.real if np.iscomplexobj(eigenvalues) else eigenvalues
        try:
            plot_envelope_modes(
                cdir,
                R_grid,  # Use Cartesian for visualization
                fields,
                eigenvalues_for_plots,
                n_modes=n_modes,
                candidate_params=row,
            )
            log(f"    Generated cavity mode plots")
        except Exception as e:
            log(f"    Warning: Could not generate plots: {e}")
    else:
        log(f"    Skipped cavity plots (no eigenvectors)")
    
    # Write report
    _write_phase3_report_v2(
        cdir, cid, grid_info, eigenvalues, omega_ref, eta, B_moire, solver_info, mode_rows
    )
    
    # Save solver metadata
    save_json({
        **solver_info,
        "eta": eta,
        "omega_ref": omega_ref,
        "n_modes_requested": n_modes,
        "n_modes_returned": len(eigenvalues),
        "grid_ns1": int(grid_info["Ns1"]),
        "grid_ns2": int(grid_info["Ns2"]),
        "pipeline": "blaze",
        "pipeline_version": "V2",
        "coordinate_system": "fractional",
        "B_moire_det": float(np.linalg.det(B_moire)),
    }, cdir / "phase3_solver_meta.json")
    
    return {
        "success": True,
        "candidate_id": cid,
        "n_eigenvalues": len(eigenvalues),
        "lowest_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else np.nan,
        "has_eigenvectors": fields is not None,
        "pipeline_version": "V2",
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_phase3_blaze_v2(run_dir: str | Path, config: Dict):
    """
    Run BLAZE Phase 3 V2 on all candidates in a run directory.
    
    Args:
        run_dir: Path to the BLAZE run directory (or 'auto'/'latest')
        config: Configuration dictionary
    """
    log("\n" + "=" * 70)
    log("PHASE 3 (BLAZE V2): Envelope Approximation Eigensolver")
    log("=" * 70)
    log("Coordinate system: Fractional (unit square [0,1)²)")
    log("Domain: [1.0, 1.0], Transformed mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ")
    log("")
    
    run_dir = resolve_blaze_run_dir(run_dir, config)
    log(f"Using run directory: {run_dir}")
    
    # Load candidate list if available
    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    else:
        log(f"WARNING: {candidates_path} not found; relying solely on per-candidate metadata.")
    
    # Discover candidates with Phase 2 V2 data
    discovered = _discover_phase2_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 2 V2 data found in {run_dir}. "
            "Run Phase 2 (BLAZE V2) before Phase 3."
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
            result = process_candidate_v2(row, config, run_dir)
            results.append(result)
            log(f"    ✓ Candidate {cid}: {result['n_eigenvalues']} eigenvalues, "
                f"E₀={result['lowest_eigenvalue']:.6e}")
        except FileNotFoundError as exc:
            log(f"  Skipping candidate {cid}: {exc}")
        except Exception as exc:
            log(f"  ERROR processing candidate {cid}: {exc}")
            import traceback
            traceback.print_exc()
    
    log(f"\nPhase 3 (BLAZE V2) completed: {len(results)}/{len(discovered)} candidates processed.\n")
    return results


def run_phase3(run_dir: str | Path, config_path: str | Path):
    """Entry point for command-line usage."""
    config = load_yaml(config_path)
    return run_phase3_blaze_v2(run_dir, config)


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 3 V2 config."""
    return PROJECT_ROOT / "configsV2" / "phase3_blaze.yaml"


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
            "Usage: python blaze_phasesV2/phase3_blaze.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest BLAZE run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
