"""
Phase 5 (BLAZE): Meep-based cavity validation with geometry previews — V2 Pipeline

This is the BLAZE-pipeline version of Phase 5 for the V2 coordinate system.
It uses Meep's FDTD solver with Harminv to compute Q-factors for EA-predicted
cavity modes, validating the envelope approximation results.

KEY V2 CHANGES from legacy Phase 5:
1. Reads from BLAZE V2 run directories (runsV2/phase0_blaze_*)
2. Uses phase1_band_data.h5 with V2 attributes (B_moire, fractional coords)
3. Uses phase3_eigenvalues.csv from BLAZE EA solver
4. Maintains same output format for comparison with legacy pipeline

This phase:
1. Loads Phase 1 geometry metadata (lattice, twist, r/a, eps)
2. Loads Phase 3 eigenvalues and mode localization data
3. Builds Meep geometry (twisted bilayer photonic crystal)
4. Runs Harminv to extract resonant Q-factors
5. Generates visualization (geometry preview, pulse spectrum, field animation)

Based on phases/phase5_meep_qfactor.py adapted for BLAZE V2 pipeline.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, cast

import h5py
import matplotlib.pyplot as plt

try:  # Matplotlib's mathtext parser may be unavailable on minimal builds
    from matplotlib.mathtext import MathTextParser
except Exception:  # pragma: no cover - optional dependency
    MathTextParser = None  # type: ignore

import meep as mp
import numpy as np
import pandas as pd

try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None

if MPI is not None:
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
else:
    MPI_COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1

IS_ROOT = MPI_RANK == 0
MPI_ENABLED = MPI_COMM is not None and MPI_SIZE > 1
MATH_TEXT_PARSER: MathTextParser | None = None  # type: ignore[valid-type]


def log(message: str):
    if IS_ROOT:
        print(message, flush=True)


def log_rank(message: str):
    prefix = f"[Rank {MPI_RANK}] " if MPI_ENABLED else ""
    print(f"{prefix}{message}", flush=True)


def mpi_barrier():
    if MPI_COMM is not None:
        MPI_COMM.Barrier()


def broadcast_value(value):
    if MPI_COMM is not None:
        return MPI_COMM.bcast(value if IS_ROOT else None, root=0)
    return value


def _get_cpu_count() -> int:
    """Get the number of physical CPU cores available."""
    try:
        # Try to get physical cores (not hyperthreads)
        import multiprocessing
        # os.cpu_count() returns logical cores; we try to get physical
        cpu_count = os.cpu_count() or 1
        # Check if we can get physical cores via /proc/cpuinfo on Linux
        if sys.platform == "linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                # Count physical cores (unique core ids per physical package)
                physical_ids = set()
                core_ids = set()
                current_physical = None
                for line in cpuinfo.split("\n"):
                    if line.startswith("physical id"):
                        current_physical = line.split(":")[1].strip()
                    elif line.startswith("core id") and current_physical is not None:
                        core_id = line.split(":")[1].strip()
                        physical_ids.add((current_physical, core_id))
                if physical_ids:
                    cpu_count = len(physical_ids)
            except Exception:
                pass
        return max(1, cpu_count)
    except Exception:
        return 1


def _setup_parallelization(config: Dict) -> Dict[str, int]:
    """
    Configure OpenMP threads for Meep parallelization.
    
    Meep uses OpenMP for shared-memory parallelism. This function sets
    OMP_NUM_THREADS based on config or auto-detects available cores.
    
    Returns:
        Dict with parallelization info (num_threads, mpi_size, etc.)
    """
    # Get configured number of threads (0 or null = auto)
    num_threads_cfg = config.get("phase5_num_threads")
    
    if num_threads_cfg is None or num_threads_cfg == 0:
        # Auto-detect: use all physical cores divided by MPI processes
        total_cores = _get_cpu_count()
        num_threads = max(1, total_cores // max(MPI_SIZE, 1))
    else:
        num_threads = int(num_threads_cfg)
    
    # Set OMP_NUM_THREADS environment variable
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    
    # Also set other OpenMP tuning variables for better performance
    if "OMP_SCHEDULE" not in os.environ:
        os.environ["OMP_SCHEDULE"] = "dynamic"
    if "OMP_PROC_BIND" not in os.environ:
        os.environ["OMP_PROC_BIND"] = "close"
    
    info = {
        "num_threads": num_threads,
        "mpi_size": MPI_SIZE,
        "mpi_rank": MPI_RANK,
        "total_cores": _get_cpu_count(),
        "is_parallel": MPI_ENABLED or num_threads > 1,
    }
    
    return info


def _log_parallelization_info(info: Dict[str, int]):
    """Log parallelization configuration."""
    log(f"Parallelization Configuration:")
    log(f"  Physical CPU cores: {info['total_cores']}")
    log(f"  OpenMP threads per process: {info['num_threads']}")
    log(f"  MPI processes: {info['mpi_size']}")
    if info['mpi_size'] > 1:
        log(f"  Total parallel workers: {info['num_threads'] * info['mpi_size']}")
    if info['is_parallel']:
        log(f"  Parallelization: ENABLED")
    else:
        log(f"  Parallelization: SERIAL (single thread)")


def _mathtext_can_render(text: str) -> bool:
    if not text or MathTextParser is None:
        return False
    global MATH_TEXT_PARSER
    expr = text.strip()
    if expr.startswith("$") and expr.endswith("$") and len(expr) >= 2:
        expr = expr[1:-1].strip()
    if not expr:
        return False
    if MATH_TEXT_PARSER is None:
        try:
            MATH_TEXT_PARSER = MathTextParser("path")
        except Exception:
            MATH_TEXT_PARSER = None
            return False
    try:
        MATH_TEXT_PARSER.parse(expr, dpi=72)
        return True
    except Exception:
        return False


# =============================================================================
# Project Setup
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json  # noqa: E402
from common.moire_utils import create_twisted_bilayer  # noqa: E402


# =============================================================================
# Run Directory Resolution (BLAZE V2 Specific)
# =============================================================================

def resolve_blaze_v2_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """
    Resolve 'auto'/'latest' shortcuts to BLAZE V2 Phase 0 run directories.
    
    CRITICAL: Only looks for phase0_blaze_* directories (NOT phase0_v2_library_*)
    to ensure we're using BLAZE pipeline data, not library-based phase0.
    """
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        base = Path(config.get("output_dir", "runsV2"))
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        # Only match BLAZE phase0 runs (not library-based)
        blaze_runs = sorted(base.glob("phase0_blaze_*"))
        if not blaze_runs:
            raise FileNotFoundError(
                f"No BLAZE V2 phase0 run directories found in {base}\n"
                f"  (Looking for phase0_blaze_*)\n"
                f"  Run Phase 0 (BLAZE) before Phase 5."
            )
        run_dir = blaze_runs[-1]
        log(f"Auto-selected latest BLAZE V2 Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runsV2")) / run_dir
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            run_dir = candidate
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


# =============================================================================
# Candidate Discovery (BLAZE V2)
# =============================================================================

def _discover_phase3_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that contain Phase 3 eigenvalues."""
    results: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase3_eigenvalues.csv").exists():
            results.append((cid, cdir))
    return results


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
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


def _load_phase1_metadata(cdir: Path) -> Dict[str, float]:
    """
    Load Phase 1 metadata from BLAZE V2 HDF5 file.
    
    BLAZE V2 stores all needed attributes in phase1_band_data.h5.
    """
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        attrs = hf.attrs
        
        def _attr(name: str, default):
            value = attrs.get(name, default)
            if isinstance(value, bytes):
                return value.decode()
            return value
        
        meta = {
            "omega_ref": float(attrs.get("omega_ref", math.nan)),
            "eta": float(attrs.get("eta", math.nan)),
            "moire_length": float(attrs.get("moire_length", math.nan)),
            "theta_deg": float(attrs.get("theta_deg", math.nan)),
            "a": float(attrs.get("a", 1.0)),
            "lattice_type": _attr("lattice_type", "hex"),
            "r_over_a": float(attrs.get("r_over_a", 0.2)),
            "eps_bg": float(attrs.get("eps_bg", 12.0)),
            # V2-specific attributes
            "pipeline_version": _attr("pipeline_version", "V2"),
            "coordinate_system": _attr("coordinate_system", "fractional"),
        }
        
        # Load B_moire if available (for transforms)
        if "B_moire" in attrs:
            meta["B_moire"] = np.array(attrs["B_moire"])
        
    return meta


def _select_mode_row(
    cdir: Path,
    config: Dict,
    return_table: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.DataFrame]:
    """Select the target mode from Phase 3 eigenvalues."""
    csv_path = cdir / "phase3_eigenvalues.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 3 eigenvalues missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Phase 3 eigenvalue table is empty for {cdir}")
    
    mode_setting = config.get("phase5_target_mode", 0)
    if isinstance(mode_setting, str) and mode_setting.lower() == "min_delta":
        selected = df.nsmallest(1, "delta_omega").iloc[0].copy()
        return (selected, df) if return_table else selected
    
    try:
        target_idx = int(mode_setting)
        row = df[df["mode_index"] == target_idx]
        if not row.empty:
            selected = row.iloc[0].copy()
            return (selected, df) if return_table else selected
    except Exception:
        pass
    
    df_sorted = df.sort_values("delta_omega")
    selected = df_sorted.iloc[0].copy()
    return (selected, df) if return_table else selected


# =============================================================================
# Utility Functions
# =============================================================================

def _format_significant(value: float, digits: int = 2) -> str:
    if not math.isfinite(value) or value == 0.0:
        return "0"
    formatted = f"{value:.{digits}g}"
    if "e" in formatted or "E" in formatted:
        numeric = float(formatted)
        formatted = f"{numeric:.{digits}g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _rotation_matrix(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def _coerce_pair(value: SupportsFloat | Sequence[SupportsFloat]) -> Tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = [float(item) for item in value]
        if len(seq) != 2:
            raise ValueError(
                "Expected a length-2 sequence for pair-valued configuration, "
                f"received length {len(seq)}"
            )
        return seq[0], seq[1]
    scalar = float(value)
    return scalar, scalar


def _compute_dynamic_fwidth(
    mode_row: pd.Series,
    modes: pd.DataFrame,
    config: Dict,
) -> float:
    """
    Compute dynamic FWHM based on distance to the next highest mode.
    
    For mode 0 (target), the FWHM is set to the distance to mode 1.
    This ensures the Gaussian pulse is well-separated from other modes.
    
    Args:
        mode_row: The selected target mode
        modes: DataFrame of all modes from Phase 3
        config: Configuration dict (fallback to phase5_source_fwidth)
    
    Returns:
        fwidth: The computed frequency width for the Gaussian source
    """
    fallback_fwidth = float(config.get("phase5_source_fwidth", 0.02))
    
    if modes is None or modes.empty:
        log(f"    Using fallback fwidth={fallback_fwidth:.4f} (no mode data)")
        return fallback_fwidth
    
    # Get target mode frequency
    target_freq = float(mode_row.get("omega_cavity", 0.0))
    if not math.isfinite(target_freq):
        # Try delta_omega + omega_ref reconstruction
        delta = float(mode_row.get("delta_omega", 0.0))
        omega_ref = float(mode_row.get("omega_ref", 0.0))
        target_freq = omega_ref + delta if math.isfinite(delta) else 0.0
    
    if not math.isfinite(target_freq) or target_freq == 0.0:
        log(f"    Using fallback fwidth={fallback_fwidth:.4f} (invalid target freq)")
        return fallback_fwidth
    
    # Get all mode frequencies
    mode_freqs = modes.get("omega_cavity")
    if mode_freqs is None:
        delta_series = modes.get("delta_omega")
        if delta_series is not None:
            omega_ref = float(mode_row.get("omega_cavity", target_freq) - mode_row.get("delta_omega", 0.0))
            mode_freqs = omega_ref + delta_series
    
    if mode_freqs is None:
        log(f"    Using fallback fwidth={fallback_fwidth:.4f} (no frequency data)")
        return fallback_fwidth
    
    # Sort frequencies and find the next mode
    sorted_freqs = sorted([float(f) for f in mode_freqs if math.isfinite(f)])
    
    if len(sorted_freqs) < 2:
        log(f"    Using fallback fwidth={fallback_fwidth:.4f} (only 1 mode)")
        return fallback_fwidth
    
    # Find target mode index in sorted list
    target_idx = -1
    min_diff = float('inf')
    for i, f in enumerate(sorted_freqs):
        diff = abs(f - target_freq)
        if diff < min_diff:
            min_diff = diff
            target_idx = i
    
    if target_idx < 0:
        log(f"    Using fallback fwidth={fallback_fwidth:.4f} (target not found)")
        return fallback_fwidth
    
    # Compute distance to the next highest mode
    if target_idx < len(sorted_freqs) - 1:
        next_freq = sorted_freqs[target_idx + 1]
        fwidth = abs(next_freq - target_freq) * 0.5  # Half the mode spacing
    else:
        # Target is the highest mode, use distance to previous mode
        prev_freq = sorted_freqs[target_idx - 1]
        fwidth = abs(target_freq - prev_freq) * 0.5  # Half the mode spacing
    
    # Apply minimum threshold
    min_fwidth = float(config.get("phase5_min_fwidth", 0.001))
    if fwidth < min_fwidth:
        log(f"    Computed fwidth={fwidth:.6f} below minimum, using {min_fwidth:.4f}")
        fwidth = min_fwidth
    
    log(f"    Dynamic FWHM: {fwidth:.6f} (half distance to next mode)")
    return fwidth


# =============================================================================
# Geometry Construction
# =============================================================================

def _lattice_points_in_bounds(
    a1: np.ndarray, 
    a2: np.ndarray, 
    bounds: Tuple[float, float, float, float], 
    pad: float
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad
    basis = np.column_stack([a1[:2], a2[:2]])
    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Lattice basis is singular") from exc
    corners = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymin],
        [xmax, ymax],
    ])
    frac = (basis_inv @ corners.T).T
    i_min = math.floor(frac[:, 0].min()) - 2
    i_max = math.ceil(frac[:, 0].max()) + 2
    j_min = math.floor(frac[:, 1].min()) - 2
    j_max = math.ceil(frac[:, 1].max()) + 2
    pts: List[np.ndarray] = []
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            vec = i * a1[:2] + j * a2[:2]
            if xmin <= vec[0] <= xmax and ymin <= vec[1] <= ymax:
                pts.append(vec)
    if not pts:
        return np.zeros((0, 2))
    return np.stack(pts)


def _point_inside_bounds(
    point: np.ndarray, 
    bounds: Tuple[float, float, float, float], 
    margin: float
) -> bool:
    xmin, xmax, ymin, ymax = bounds
    return (xmin + margin) <= point[0] <= (xmax - margin) and \
           (ymin + margin) <= point[1] <= (ymax - margin)


def _point_in_any_hole(
    point: np.ndarray, 
    holes: List[Tuple[np.ndarray, float]], 
    tol: float
) -> bool:
    for center, radius in holes:
        if np.linalg.norm(point - center) <= max(radius - tol, 0.0):
            return True
    return False


def _distance_to_nearest_hole_surface(
    point: np.ndarray,
    holes: List[Tuple[np.ndarray, float]],
) -> Tuple[float, np.ndarray | None, float]:
    """
    Compute distance from point to nearest hole surface.
    
    Returns:
        (distance, nearest_center, nearest_radius)
        distance > 0 means point is outside the hole
        distance < 0 means point is inside the hole
    """
    min_dist = float('inf')
    nearest_center = None
    nearest_radius = 0.0
    
    for center, radius in holes:
        dist_to_center = np.linalg.norm(point - center)
        dist_to_surface = dist_to_center - radius  # positive = outside, negative = inside
        
        if abs(dist_to_surface) < abs(min_dist):
            min_dist = dist_to_surface
            nearest_center = center
            nearest_radius = radius
    
    return min_dist, nearest_center, nearest_radius


def _candidate_directions(a1: np.ndarray, a2: np.ndarray) -> List[np.ndarray]:
    base_vectors = [
        0.5 * (a1 + a2), a1, a2, a1 - a2, 
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ]
    directions: List[np.ndarray] = []
    for vec in base_vectors:
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            continue
        unit = vec / norm
        directions.append(unit)
        directions.append(-unit)
    return directions


def _find_source_position(
    cavity_pos: np.ndarray,
    holes: List[Tuple[np.ndarray, float]],
    step: float,
    max_steps: int,
    bounds: Tuple[float, float, float, float],
    directions: List[np.ndarray],
    interface_margin_fraction: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """
    Find a suitable source position outside etched holes.
    
    The source is placed outside all holes, with an additional margin to avoid
    placing it directly at a hole interface. If a candidate position is too 
    close to any hole surface (within 10% of the radius), it's pushed further
    outward along the search direction.
    
    Args:
        cavity_pos: Center position of the cavity
        holes: List of (center, radius) tuples for all holes
        step: Step size for radial search
        max_steps: Maximum number of steps to try
        bounds: Simulation bounds (xmin, xmax, ymin, ymax)
        directions: Candidate search directions
        interface_margin_fraction: Fraction of radius to maintain as margin (default 0.10)
    
    Returns:
        (position, shift_distance)
    """
    # Check if cavity position is already outside all holes
    if not _point_in_any_hole(cavity_pos, holes, 1e-6):
        # Still check if we're too close to any surface
        dist_to_surface, nearest_center, nearest_radius = _distance_to_nearest_hole_surface(
            cavity_pos, holes
        )
        min_margin = interface_margin_fraction * nearest_radius if nearest_radius > 0 else 0
        
        if dist_to_surface >= min_margin:
            return cavity_pos, 0.0
        # Otherwise, we need to push outward
    
    if not directions:
        raise RuntimeError("No candidate directions to place the source")
    
    for step_idx in range(1, max_steps + 1):
        for direction in directions:
            candidate = cavity_pos + direction * (step * step_idx)
            
            if not _point_inside_bounds(candidate, bounds, margin=step):
                continue
            
            if _point_in_any_hole(candidate, holes, 1e-6):
                continue
            
            # Candidate is outside holes; check distance to nearest surface
            dist_to_surface, nearest_center, nearest_radius = _distance_to_nearest_hole_surface(
                candidate, holes
            )
            
            if nearest_center is None or nearest_radius <= 0:
                # No holes nearby, position is good
                shift = candidate - cavity_pos
                return candidate, float(np.linalg.norm(shift))
            
            # Compute minimum margin (10% of nearest hole radius)
            min_margin = interface_margin_fraction * nearest_radius
            
            if dist_to_surface >= min_margin:
                # Sufficient margin from all surfaces
                shift = candidate - cavity_pos
                return candidate, float(np.linalg.norm(shift))
            
            # Too close to interface - push outward along the direction away from hole center
            # Compute how much more we need to move
            additional_dist = min_margin - dist_to_surface + 1e-6  # small epsilon for safety
            
            # Push in the search direction (outward from cavity)
            pushed_candidate = candidate + direction * additional_dist
            
            # Verify the pushed position is valid
            if not _point_inside_bounds(pushed_candidate, bounds, margin=step):
                continue
            
            if _point_in_any_hole(pushed_candidate, holes, 1e-6):
                # Pushed into another hole - continue searching
                continue
            
            # Check distance again after pushing
            dist_after, _, _ = _distance_to_nearest_hole_surface(pushed_candidate, holes)
            if dist_after >= min_margin * 0.9:  # Allow slight tolerance
                shift = pushed_candidate - cavity_pos
                return pushed_candidate, float(np.linalg.norm(shift))
    
    raise RuntimeError("Could not place a source outside the etched region")


def _build_geometry(
    phase1_meta: Dict,
    config: Dict,
) -> Dict:
    """Build Meep geometry for twisted bilayer photonic crystal."""
    theta_deg = float(phase1_meta.get("theta_deg", 0.0))
    theta_rad = math.radians(theta_deg)
    lattice_type = str(phase1_meta.get("lattice_type", "hex"))
    a = float(phase1_meta.get("a", 1.0))
    r_over_a = float(phase1_meta.get("r_over_a", 0.2))
    eps_bg = float(phase1_meta.get("eps_bg", 12.0))
    moire_length = float(phase1_meta.get("moire_length", a))

    bilayer = create_twisted_bilayer(lattice_type, theta_deg, a)
    a1 = np.asarray(bilayer["a1"][:2], dtype=float)
    a2 = np.asarray(bilayer["a2"][:2], dtype=float)
    rot = _rotation_matrix(theta_rad)
    top_a1 = rot @ a1
    top_a2 = rot @ a2

    window_cells = cast(
        SupportsFloat | Sequence[SupportsFloat], 
        config.get("phase5_window_cells", 2.0)
    )
    win_x_cells, win_y_cells = _coerce_pair(window_cells)
    window_x = float(win_x_cells) * moire_length
    window_y = float(win_y_cells) * moire_length
    bounds = (-0.5 * window_x, 0.5 * window_x, -0.5 * window_y, 0.5 * window_y)
    
    # Handle null/None values in config
    pad_config = config.get("phase5_lattice_padding")
    pad = float(pad_config) if pad_config is not None else r_over_a * a

    bottom_points = _lattice_points_in_bounds(a1, a2, bounds, pad)
    top_points = _lattice_points_in_bounds(top_a1, top_a2, bounds, pad)

    bottom_shift = np.asarray(config.get("phase5_bottom_shift", [0.0, 0.0]), dtype=float)
    top_shift = np.asarray(config.get("phase5_top_shift", [0.0, 0.0]), dtype=float)
    if bottom_points.size:
        bottom_points = bottom_points + bottom_shift
    if top_points.size:
        top_points = top_points + top_shift

    radius = r_over_a * a
    hole_height_config = config.get("phase5_cylinder_height")
    hole_height = mp.inf if hole_height_config is None else hole_height_config
    air = mp.Medium(epsilon=1.0)
    z_sep = float(config.get("phase5_layer_separation", 0.0)) * 0.5
    geometry: List[mp.Cylinder] = []
    hole_centers: List[Tuple[np.ndarray, float]] = []
    
    for pt in bottom_points:
        geometry.append(
            mp.Cylinder(
                radius=radius,
                height=hole_height,
                center=mp.Vector3(float(pt[0]), float(pt[1]), -z_sep),
                material=air,
            )
        )
        hole_centers.append((np.asarray(pt, dtype=float), radius))
    
    for pt in top_points:
        geometry.append(
            mp.Cylinder(
                radius=radius,
                height=hole_height,
                center=mp.Vector3(float(pt[0]), float(pt[1]), z_sep),
                material=air,
            )
        )
        hole_centers.append((np.asarray(pt, dtype=float), radius))

    pml = float(config.get("phase5_pml_thickness", 2.0))
    cell_x = window_x + 2.0 * pml
    cell_y = window_y + 2.0 * pml
    cell_z = float(config.get("phase5_cell_height", 0.0))

    return {
        "geometry": geometry,
        "hole_centers": hole_centers,
        "bounds": bounds,
        "window_span": (window_x, window_y),
        "cell_span": (cell_x, cell_y, cell_z),
        "radius": radius,
        "a1": a1,
        "a2": a2,
        "top_a1": top_a1,
        "top_a2": top_a2,
        "eps_bg": eps_bg,
        "pml": pml,
        "counts": {"bottom": len(bottom_points), "top": len(top_points)},
        "bottom_points": bottom_points.copy() if bottom_points.size else bottom_points,
        "top_points": top_points.copy() if top_points.size else top_points,
    }


# =============================================================================
# Meep Simulation
# =============================================================================

def _build_simulation(ctx: Dict, resolution: float, sources=None) -> mp.Simulation:
    span_x, span_y, span_z = ctx["cell_span"]
    cell = mp.Vector3(span_x, span_y, span_z)
    boundary_layers = [mp.PML(thickness=ctx["pml"])] if ctx["pml"] > 0 else []
    sim = mp.Simulation(
        cell_size=cell,
        geometry=ctx["geometry"],
        resolution=resolution,
        boundary_layers=boundary_layers,
        default_material=mp.Medium(epsilon=ctx["eps_bg"]),
        sources=sources or [],
    )
    return sim


def _build_sources(
    component, 
    source_pos: np.ndarray, 
    freq: float, 
    fwidth: float,
    config: Dict
) -> List[mp.Source]:
    cutoff = float(config.get("phase5_source_cutoff", 3.0))
    amplitude = float(config.get("phase5_source_amplitude", 1.0))
    src = mp.GaussianSource(frequency=freq, fwidth=fwidth, cutoff=cutoff)
    return [
        mp.Source(
            src=src,
            component=component,
            center=mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0),
            amplitude=amplitude,
        )
    ]


def _run_meep(
    ctx: Dict,
    component,
    freq: float,
    fwidth: float,
    config: Dict,
    source_pos: np.ndarray,
) -> Tuple[List[Dict[str, float]], Dict[str, Any]]:
    resolution = float(config.get("phase5_resolution", 32))
    sources = _build_sources(component, source_pos, freq, fwidth, config)
    sim = _build_simulation(ctx, resolution, sources=sources)
    
    harminv_bw = float(config.get("phase5_harminv_bw", 0.05))
    harminv = mp.Harminv(
        component, 
        mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0), 
        freq, 
        harminv_bw
    )
    
    decay_dt = float(config.get("phase5_decay_dt", 50.0))
    decay_threshold = float(config.get("phase5_decay_threshold", 1e-9))
    run_time = float(config.get("phase5_run_time", 400.0))
    
    # Video capture settings
    # phase5_video_capture_dt: simulation time between frames (determines temporal resolution)
    # phase5_video_max_frames: maximum frames to capture (memory limit)
    video_capture_dt_cfg = config.get("phase5_video_capture_dt")
    if video_capture_dt_cfg is not None:
        video_capture_dt = float(video_capture_dt_cfg)
    else:
        # Default: capture every 3 simulation time units for good temporal resolution
        video_capture_dt = float(config.get("phase5_gif_capture_dt", 3.0))
    video_capture_dt = max(video_capture_dt, 0.5)  # Minimum 0.5 to avoid too many frames
    
    video_max_frames = int(config.get("phase5_video_max_frames", 
                                       config.get("phase5_gif_max_frames", 500)))
    
    # Spatial downsampling to reduce memory (critical for large simulations)
    # phase5_video_stride: pixel stride for spatial downsampling (1=full, 2=half, 4=quarter)
    # For 2L_m=110a @ res=32: original is 3520x3520, stride=4 gives 880x880
    # 
    # Auto-stride: if phase5_video_stride="auto", automatically compute based on grid size
    # to keep frame size reasonable (target ~1024x1024 max for memory safety)
    video_stride_cfg = config.get("phase5_video_stride", 1)
    window_x, window_y = ctx["window_span"]
    
    # Estimate grid dimensions
    grid_x = int(window_x * resolution)
    grid_y = int(window_y * resolution)
    max_grid_dim = max(grid_x, grid_y)
    
    if video_stride_cfg == "auto" or str(video_stride_cfg).lower() == "auto":
        # Auto-compute stride to keep max dimension <= 1024
        target_max_dim = int(config.get("phase5_video_auto_max_dim", 1024))
        if max_grid_dim > target_max_dim:
            video_stride = int(np.ceil(max_grid_dim / target_max_dim))
        else:
            video_stride = 1
        if IS_ROOT and video_stride > 1:
            log(f"    Auto-stride: {video_stride} (grid {grid_x}x{grid_y} -> ~{grid_x//video_stride}x{grid_y//video_stride})")
    else:
        video_stride = max(1, int(video_stride_cfg))
    
    # STREAMING VIDEO MODE: Write frames directly to disk during simulation
    # This is critical for very long simulations (1000s of timesteps) where
    # storing all frames in memory would cause OOM.
    # 
    # phase5_video_stream: true/false - Enable streaming mode (default: true for large sims)
    # When enabled, frames are written directly to MP4 during simulation.
    # When disabled, frames are stored in memory and written after simulation.
    #
    # Streaming mode uses ~0 MB for frames regardless of simulation length.
    # Non-streaming mode uses: n_frames * frame_size_mb
    
    expected_n_frames = min(video_max_frames, int(run_time / video_capture_dt) + 1)
    frame_size_mb = (grid_x // video_stride) * (grid_y // video_stride) * 4 / (1024 * 1024)
    total_frames_mb = expected_n_frames * frame_size_mb
    
    # Auto-enable streaming for large expected memory usage (> 2GB)
    stream_mode_cfg = config.get("phase5_video_stream")
    if stream_mode_cfg is None:
        # Auto-detect: stream if expected memory > 2GB
        video_stream_mode = total_frames_mb > 2000
    else:
        video_stream_mode = bool(stream_mode_cfg)
    
    if IS_ROOT:
        if video_stream_mode:
            log(f"    Streaming video mode: ENABLED (frames written directly to disk)")
        elif total_frames_mb > 2000:
            log(f"    WARNING: Estimated frame memory: {total_frames_mb/1024:.1f} GB")
            log(f"    Consider enabling phase5_video_stream: true")
    
    capture_frames = bool(config.get("phase5_capture_frames", True)) and video_capture_dt > 0
    store_frames = capture_frames and IS_ROOT
    
    # Video parameters for streaming
    sim_time_per_video_second = float(config.get("phase5_video_sim_time_per_second", 25.0))
    effective_fps = sim_time_per_video_second / max(video_capture_dt, 1e-6)
    effective_fps = max(10.0, min(60.0, effective_fps))
    effective_fps = int(round(effective_fps))
    
    # Prepare geometry overlay (computed once, reused for all frames)
    geometry_bg = None
    cmap = plt.colormaps.get_cmap("RdBu_r")
    
    # For streaming mode, we need to estimate vmax upfront or use adaptive normalization
    # We'll use a fixed normalization based on expected field amplitude
    # This is approximate but avoids needing to scan all frames
    streaming_vmax = float(config.get("phase5_video_vmax", 0.0))  # 0 = adaptive
    
    # Streaming video writer context
    video_writer = None
    video_path = None
    frame_count = [0]  # Use list for mutable closure
    vmax_running = [0.0]  # Track running max for adaptive normalization
    
    # Non-streaming storage
    frames: List[np.ndarray] = []
    frame_times: List[float] = []

    def _init_streaming_writer(output_path: Path) -> Any:
        """Initialize the streaming video writer."""
        nonlocal geometry_bg
        try:
            import imageio
            # Pre-compute geometry background
            sample_shape = (grid_y // video_stride, grid_x // video_stride)
            if ctx is not None:
                geometry_mask = _geometry_mask(sample_shape, ctx)
                geometry_bg = _geometry_background(geometry_mask)
            
            writer = imageio.get_writer(
                str(output_path),
                fps=effective_fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                pixelformat="yuv420p",
            )
            return writer
        except Exception as e:
            log(f"    WARNING: Could not initialize video writer: {e}")
            return None

    def _frame_to_rgb(frame: np.ndarray, vmax: float) -> np.ndarray:
        """Convert field array to RGB image."""
        if vmax <= 0:
            vmax = 1.0
        norm = (frame / vmax + 1.0) * 0.5
        norm = np.clip(norm, 0.0, 1.0)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        if geometry_bg is not None:
            rgba = _blend_with_background(rgba, geometry_bg)
        return rgba[..., :3]

    def _record_frame_streaming(sim_obj: mp.Simulation):
        """Record frame directly to video file (streaming mode)."""
        nonlocal video_writer
        if not capture_frames or not store_frames:
            return
        if frame_count[0] >= video_max_frames:
            return
        if video_writer is None:
            return
            
        arr = sim_obj.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(window_x, window_y, 0.0),
            component=component,
        )
        if arr is None:
            return
        
        # Transpose and downsample
        frame = np.array(arr, copy=True, dtype=np.float32).T
        if video_stride > 1:
            frame = frame[::video_stride, ::video_stride]
        
        # Update running vmax for adaptive normalization
        frame_max = float(np.max(np.abs(frame)))
        if frame_max > vmax_running[0]:
            vmax_running[0] = frame_max
        
        # Use fixed vmax if specified, otherwise use running max
        if streaming_vmax > 0:
            vmax = streaming_vmax
        else:
            # Use running max with some headroom
            vmax = max(vmax_running[0], 1e-10)
        
        # Convert to RGB and write
        rgb = _frame_to_rgb(frame, vmax)
        video_writer.append_data(rgb)
        frame_count[0] += 1
        
        # Progress logging
        if frame_count[0] % 100 == 0:
            log(f"    Written {frame_count[0]} frames (t={sim_obj.meep_time():.1f})...")

    def _record_frame_memory(sim_obj: mp.Simulation):
        """Record frame to memory (non-streaming mode)."""
        if not capture_frames:
            return
        if len(frames) >= video_max_frames:
            return
        arr = sim_obj.get_array(
            center=mp.Vector3(),
            size=mp.Vector3(window_x, window_y, 0.0),
            component=component,
        )
        if arr is None or not store_frames:
            return
        frame = np.array(arr, copy=True, dtype=np.float32).T
        if video_stride > 1:
            frame = frame[::video_stride, ::video_stride]
        frames.append(frame)
        frame_times.append(sim_obj.meep_time())

    decay_stop = mp.stop_when_fields_decayed(
        decay_dt,
        component,
        mp.Vector3(float(source_pos[0]), float(source_pos[1]), 0.0),
        decay_threshold,
    )
    
    # Choose recording function based on mode
    if video_stream_mode and store_frames:
        record_func = _record_frame_streaming
    else:
        record_func = _record_frame_memory
    
    # Initialize streaming writer if needed
    # Note: video_path will be set by caller, we use a temp path here
    temp_video_path = None
    if video_stream_mode and store_frames and capture_frames:
        import tempfile
        temp_video_path = Path(tempfile.mktemp(suffix=".mp4"))
        video_writer = _init_streaming_writer(temp_video_path)
        if video_writer is not None:
            log(f"    Streaming to temporary file during simulation...")
    
    # Progress logging callback (logs every ~50 simulation time units)
    progress_interval = float(config.get("phase5_progress_interval", 50.0))
    last_progress_time = [0.0]
    
    def _log_progress(sim_obj: mp.Simulation):
        t = sim_obj.meep_time()
        if t - last_progress_time[0] >= progress_interval:
            last_progress_time[0] = t
            if IS_ROOT:
                frames_info = f", {frame_count[0]} frames" if video_stream_mode else ""
                log(f"    Simulation progress: t={t:.1f}/{run_time:.0f}{frames_info}")
    
    log(f"    Starting Meep simulation (run_time={run_time:.0f})...")
    
    if video_capture_dt > 0:
        sim.run(
            mp.after_sources(harminv),
            mp.at_every(video_capture_dt, record_func),
            mp.at_every(progress_interval, _log_progress),
            decay_stop,
            until=run_time,
        )
    else:
        sim.run(
            mp.after_sources(harminv),
            mp.at_every(progress_interval, _log_progress),
            decay_stop,
            until=run_time,
        )
    
    log(f"    Simulation finished at t={sim.meep_time():.1f}")
    
    # Close streaming writer
    if video_writer is not None:
        try:
            video_writer.close()
            log(f"    Streaming complete: {frame_count[0]} frames written")
        except Exception as e:
            log(f"    WARNING: Error closing video writer: {e}")
    
    rows: List[Dict[str, float]] = []
    min_q = float(config.get("phase5_min_Q", 50.0))
    for mode in harminv.modes:
        if mode.Q < min_q:
            continue
        amp_val = getattr(mode, "amp", None)
        if amp_val is None:
            amp_val = getattr(mode, "alpha", None)
        amp_abs = float(abs(amp_val)) if amp_val is not None else float("nan")
        amp_real = float(np.real(amp_val)) if amp_val is not None else float("nan")
        amp_imag = float(np.imag(amp_val)) if amp_val is not None else float("nan")
        rows.append({
            "freq_meep": float(mode.freq),
            "Q": float(mode.Q),
            "decay": float(mode.decay),
            "amplitude_abs": amp_abs,
            "amplitude_real": amp_real,
            "amplitude_imag": amp_imag,
        })
    
    sim.reset_meep()
    if not store_frames:
        frames = []
    
    # Return frame data with streaming info
    frame_data = {
        "frames": frames,  # Empty if streaming mode
        "capture_dt": video_capture_dt,
        "frame_times": frame_times if (store_frames and not video_stream_mode) else [],
        "stream_mode": video_stream_mode,
        "stream_path": str(temp_video_path) if temp_video_path else None,
        "frame_count": frame_count[0] if video_stream_mode else len(frames),
        "effective_fps": effective_fps,
    }
    return rows, frame_data


# =============================================================================
# Results and Reporting
# =============================================================================

def _classify_quality(Q_val: float, config: Dict) -> Tuple[str, str]:
    minor = float(config.get("phase5_quality_minor", 250.0))
    good = float(config.get("phase5_quality_good", 1000.0))
    elite = float(config.get("phase5_quality_strong", 2500.0))
    if not math.isfinite(Q_val):
        return "unknown", "No Harminv data captured"
    if Q_val < minor:
        return "diffuse", f"Q < {minor:.0f}: leakage-dominated"
    if Q_val < good:
        return "incipient", f"{minor:.0f} ≤ Q < {good:.0f}: weak confinement"
    if Q_val < elite:
        return "cavity", f"{good:.0f} ≤ Q < {elite:.0f}: good cavity"
    return "elite", f"Q ≥ {elite:.0f}: strong cavity"


def _write_results(
    cdir: Path,
    rows: List[Dict[str, float]],
    cid: int,
    mode_index: int,
    omega_ea: float,
    source_shift: float,
    config: Dict,
) -> List[Dict[str, float]]:
    enriched = []
    for row in rows:
        rel_error = abs(row["freq_meep"] - omega_ea) / max(abs(omega_ea), 1e-12)
        quality_label, quality_note = _classify_quality(row["Q"], config)
        enriched.append({
            "candidate_id": cid,
            "mode_index": mode_index,
            "omega_ea": omega_ea,
            "omega_meep": row["freq_meep"],
            "Q": row["Q"],
            "decay": row["decay"],
            "amplitude_abs": row["amplitude_abs"],
            "amplitude_real": row["amplitude_real"],
            "amplitude_imag": row["amplitude_imag"],
            "relative_error": rel_error,
            "source_shift": source_shift,
            "quality_label": quality_label,
            "quality_note": quality_note,
        })
    
    results_path = cdir / "phase5_q_factor_results.csv"
    if enriched:
        pd.DataFrame(enriched).to_csv(results_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "candidate_id", "mode_index", "omega_ea", "omega_meep",
                "Q", "decay", "amplitude_abs", "amplitude_real", "amplitude_imag",
                "relative_error", "source_shift", "quality_label", "quality_note",
            ]
        ).to_csv(results_path, index=False)
    return enriched


def _quality_legend(config: Dict) -> List[str]:
    minor = float(config.get("phase5_quality_minor", 250.0))
    good = float(config.get("phase5_quality_good", 1000.0))
    elite = float(config.get("phase5_quality_strong", 2500.0))
    return [
        f"diffuse: Q < {minor:.0f}",
        f"incipient: {minor:.0f} ≤ Q < {good:.0f}",
        f"cavity: {good:.0f} ≤ Q < {elite:.0f}",
        f"elite: Q ≥ {elite:.0f}",
    ]


def _write_report(
    cdir: Path, 
    summary: Dict[str, float], 
    modes: List[Dict[str, float]], 
    config: Dict
):
    top_modes = sorted(modes, key=lambda row: row["Q"], reverse=True)
    lines = [
        "# Phase 5 Meep Validation Report (BLAZE V2)",
        "",
        f"**Candidate**: {summary['candidate_id']:04d}",
        f"**Pipeline Version**: V2 (BLAZE)",
        "",
        "## Geometry",
        f"- Window span: {summary['window_x']:.3f} × {summary['window_y']:.3f}",
        f"- Cylinders: {summary['n_bottom']} bottom + {summary['n_top']} top",
        f"- Source shift from EA peak: {summary['source_shift']:.4f}",
        "",
        "## Simulation",
        f"- Target mode index: {summary['mode_index']}",
        f"- ω_EA: {summary['omega_ea']:.6f}",
        f"- Ran Meep: {'yes' if summary['ran_meep'] else 'no'}",
    ]
    
    if summary["ran_meep"]:
        lines.extend([
            f"- Recorded Harminv modes: {summary['recorded_modes']}",
            f"- Best Q: {summary['best_q']:.1f}" if math.isfinite(summary['best_q']) else "- Best Q: N/A",
            "",
            "## Quality Classification",
        ])
        for label in _quality_legend(config):
            lines.append(f"- {label}")
        lines.append("")
        
        if top_modes:
            lines.append("## Top Modes by Q")
            lines.append("")
            lines.append("| ω_meep | Q | Relative Error | Label |")
            lines.append("|--------|---|----------------|-------|")
            for mode in top_modes[:5]:
                lines.append(
                    f"| {mode['omega_meep']:.6f} | {mode['Q']:.1f} | "
                    f"{mode['relative_error']:.2e} | {mode['quality_label']} |"
                )
    
    report_path = cdir / "phase5_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Visualization
# =============================================================================

def _render_geometry_preview(
    sim: mp.Simulation,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    out_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    sim.plot2D(ax=ax)
    ax.scatter(
        [source_pos[0]], [source_pos[1]],
        marker="o", s=3, c="C0", edgecolors="white", linewidths=0,
        label="Source", zorder=5,
    )
    ax.scatter(
        [cavity_pos[0]], [cavity_pos[1]],
        marker="o", s=3, c="C1", edgecolors="white", linewidths=0,
        label="EA peak", zorder=6,
    )
    ax.legend(loc="upper right")
    ax.set_title("Meep Geometry Preview (BLAZE V2)")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _render_geometry_preview_static(
    geometry_ctx: Dict,
    cavity_pos: np.ndarray,
    source_pos: np.ndarray,
    out_path: Path,
    dpi: int,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    bottom_points = geometry_ctx.get("bottom_points")
    top_points = geometry_ctx.get("top_points")
    if isinstance(bottom_points, np.ndarray) and bottom_points.size:
        ax.scatter(
            bottom_points[:, 0], bottom_points[:, 1],
            s=6, c="#0ea5e9", alpha=0.55, edgecolors="none",
            label="Bottom layer",
        )
    if isinstance(top_points, np.ndarray) and top_points.size:
        ax.scatter(
            top_points[:, 0], top_points[:, 1],
            s=6, c="#ec4899", alpha=0.55, edgecolors="none",
            label="Top layer",
        )
    ax.scatter(
        [source_pos[0]], [source_pos[1]],
        marker="o", s=8, c="C0", edgecolors="white", linewidths=0.3,
        label="Source", zorder=5,
    )
    ax.scatter(
        [cavity_pos[0]], [cavity_pos[1]],
        marker="o", s=7, c="C1", edgecolors="white", linewidths=0.3,
        label="EA peak", zorder=6,
    )
    ax.legend(loc="upper right")
    ax.set_title("Meep Geometry Preview (static)")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_gaussian_pulse(
    cdir: Path,
    mode_row: pd.Series,
    fwidth: float,
    config: Dict,
    modes: pd.DataFrame,
):
    cutoff = float(config.get("phase5_source_cutoff", 3.0))
    span_multiplier = max(cutoff, float(config.get("phase5_pulse_span_multiplier", 4.0)))
    freq = float(mode_row.get("omega_cavity", 0.0))
    if not math.isfinite(freq):
        freq = 0.0
    if fwidth <= 0:
        baseline = max(abs(freq) * 0.05, 0.01)
        fwidth = baseline
    sigma = fwidth / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    
    sorted_freqs: List[float] = []
    if modes is not None and not modes.empty:
        mode_freqs = modes.get("omega_cavity")
        if mode_freqs is None:
            delta_series = modes.get("delta_omega")
            if delta_series is not None:
                omega_ref = float(
                    mode_row.get("omega_cavity", freq) - mode_row.get("delta_omega", 0.0)
                )
                mode_freqs = omega_ref + delta_series
        if mode_freqs is not None:
            sorted_freqs = [float(value) for value in mode_freqs if math.isfinite(value)]
            sorted_freqs.sort()

    target_idx = -1
    if sorted_freqs:
        diffs = [abs(value - freq) for value in sorted_freqs]
        target_idx = int(np.argmin(diffs))

    axis_half_default = max(span_multiplier * fwidth, fwidth)
    min_display_half = max(0.5 * fwidth, 5e-3)
    axis_half = max(axis_half_default, min_display_half)
    if sorted_freqs:
        span_left = max(freq - sorted_freqs[0], 0.0)
        span_right = max(sorted_freqs[-1] - freq, 0.0)
        mode_half_span = max(span_left, span_right)
        buffer_frac = float(config.get("phase5_mode_axis_buffer", 0.05))
        dynamic_buffer = max(buffer_frac * max(mode_half_span, min_display_half), 1e-4)
        axis_half = max(mode_half_span + dynamic_buffer, min_display_half)

    f_min = freq - axis_half
    f_max = freq + axis_half
    if not math.isfinite(f_min) or not math.isfinite(f_max) or f_max <= f_min:
        f_min = freq - axis_half_default
        f_max = freq + axis_half_default
    
    x = np.linspace(f_min, f_max, 1024)
    denom = max(sigma, 1e-9)
    gaussian = np.exp(-0.5 * ((x - freq) / denom) ** 2)
    if gaussian.max() > 0:
        gaussian /= gaussian.max()
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=int(config.get("phase5_preview_dpi", 400)))
    ax.plot(x, gaussian, color="tab:blue", label="Gaussian pulse")
    ax.set_xlabel("Frequency (ω)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(f_min, f_max)
    ax.set_title("Phase 5 Source Spectrum (BLAZE V2)")

    half_amp = 0.5
    half_width = 0.5 * (2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma)
    left = max(freq - half_width, f_min)
    right = min(freq + half_width, f_max)
    fwhm_label = f"FWHM = {_format_significant(2.0 * half_width, digits=2)}"
    ax.hlines(half_amp, left, right, colors="tab:purple", linestyles="--", label=fwhm_label)

    ylim_top = 1.05
    if sorted_freqs:
        target_color = str(config.get("phase5_target_mode_color", "#d97706"))
        other_color = str(config.get("phase5_other_mode_color", "#1f2937"))
        other_labeled = False
        label_height = float(config.get("phase5_target_label_height", 1.08))
        label_text = str(config.get("phase5_target_label_text", r"$\omega_0$"))
        prefer_math = bool(config.get("phase5_target_label_math", True))
        use_math = prefer_math and _mathtext_can_render(label_text)
        fallback_text = (label_text.replace("$", "") or "omega0") if prefer_math else label_text
        if prefer_math and not use_math:
            log(
                f"    WARNING: MathText unavailable for phase5 target label"
                f" '{label_text}'. Rendering '{fallback_text}' instead."
            )
        for idx, value in enumerate(sorted_freqs):
            if idx == target_idx:
                ax.axvline(value, color=target_color, linewidth=1.3, label="Target mode")
                ax.text(
                    value, label_height,
                    label_text if use_math else fallback_text,
                    rotation=0, va="bottom", ha="center", fontsize=9, color=target_color,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4},
                )
                ylim_top = max(ylim_top, label_height + 0.05)
            else:
                ax.axvline(
                    value, color=other_color, alpha=0.85, linewidth=1.0,
                    label="Other modes" if not other_labeled else None,
                )
                other_labeled = True
    ax.set_ylim(-0.05, ylim_top)

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    out_path = cdir / "phase5_source_pulse.png"
    fig.savefig(out_path)
    plt.close(fig)


def _plot_q_factor_spectrum(
    cdir: Path,
    config: Dict,
    modes: Optional[pd.DataFrame] = None,
):
    """
    Plot Q-factor spectrum showing Harminv-detected modes as vertical bars.
    
    This plot uses the same x-axis range as the source pulse plot (or extends
    to cover all detected modes if they exceed that range). The y-axis shows
    Q-factor on a log scale.
    
    Args:
        cdir: Candidate directory containing phase5_q_factor_results.csv
        config: Configuration dictionary
        modes: Optional DataFrame of EA modes (for vertical mode lines)
    """
    results_path = cdir / "phase5_q_factor_results.csv"
    if not results_path.exists():
        log(f"    Q-factor results not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    if df.empty:
        log(f"    Q-factor results empty, skipping spectrum plot")
        return
    
    # Get frequencies and Q values
    freqs = df["omega_meep"].values
    Q_vals = df["Q"].values
    quality_labels = df["quality_label"].values if "quality_label" in df.columns else None
    
    # Get EA target frequency for reference
    omega_ea = df["omega_ea"].iloc[0] if "omega_ea" in df.columns else None
    
    # Determine x-axis range (same logic as source pulse plot)
    cutoff = float(config.get("phase5_source_cutoff", 3.0))
    span_multiplier = max(cutoff, float(config.get("phase5_pulse_span_multiplier", 4.0)))
    
    # Load phase3_eigenvalues.csv to get mode frequencies
    sorted_mode_freqs: List[float] = []
    if modes is not None and not modes.empty:
        mode_freqs_col = modes.get("omega_cavity")
        if mode_freqs_col is not None:
            sorted_mode_freqs = [float(v) for v in mode_freqs_col if math.isfinite(v)]
            sorted_mode_freqs.sort()
    
    # Calculate fwidth (same as in pulse plot)
    if omega_ea is not None and math.isfinite(omega_ea):
        freq_center = omega_ea
    else:
        freq_center = float(np.mean(freqs)) if len(freqs) > 0 else 0.5
    
    # Get fwidth from config or estimate
    fwidth_cfg = config.get("phase5_gaussian_fwidth")
    if fwidth_cfg is not None:
        fwidth = float(fwidth_cfg)
    else:
        # Estimate from mode spacing (simplified)
        if sorted_mode_freqs and len(sorted_mode_freqs) > 1:
            spacings = np.diff(sorted_mode_freqs)
            fwidth = float(np.min(spacings)) * 0.5 if len(spacings) > 0 else 0.01
        else:
            fwidth = max(abs(freq_center) * 0.05, 0.01)
    
    # Calculate axis range (matching source pulse logic)
    axis_half_default = max(span_multiplier * fwidth, fwidth)
    min_display_half = max(0.5 * fwidth, 5e-3)
    axis_half = max(axis_half_default, min_display_half)
    
    if sorted_mode_freqs:
        span_left = max(freq_center - sorted_mode_freqs[0], 0.0)
        span_right = max(sorted_mode_freqs[-1] - freq_center, 0.0)
        mode_half_span = max(span_left, span_right)
        buffer_frac = float(config.get("phase5_mode_axis_buffer", 0.05))
        dynamic_buffer = max(buffer_frac * max(mode_half_span, min_display_half), 1e-4)
        axis_half = max(mode_half_span + dynamic_buffer, min_display_half)
    
    f_min_pulse = freq_center - axis_half
    f_max_pulse = freq_center + axis_half
    
    # Extend range if detected modes exceed pulse range
    if len(freqs) > 0:
        f_min = min(f_min_pulse, float(np.min(freqs)) - 0.01)
        f_max = max(f_max_pulse, float(np.max(freqs)) + 0.01)
    else:
        f_min, f_max = f_min_pulse, f_max_pulse
    
    # Color mapping for quality labels
    quality_colors = {
        "elite": "#16a34a",      # Green
        "cavity": "#2563eb",     # Blue
        "incipient": "#d97706",  # Orange
        "diffuse": "#dc2626",    # Red
        "unknown": "#6b7280",    # Gray
    }
    
    # Create figure
    dpi = int(config.get("phase5_preview_dpi", 400))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    
    # Plot Q-factors as vertical bars
    bar_width = (f_max - f_min) / max(len(freqs) * 8, 50)  # Auto-scale bar width
    
    for i, (freq, Q, label) in enumerate(zip(freqs, Q_vals, 
            quality_labels if quality_labels is not None else ["unknown"] * len(freqs))):
        color = quality_colors.get(label, "#6b7280")
        ax.bar(freq, Q, width=bar_width, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    
    # Add EA mode lines (vertical dashed lines)
    if sorted_mode_freqs:
        target_color = str(config.get("phase5_target_mode_color", "#d97706"))
        other_color = str(config.get("phase5_other_mode_color", "#1f2937"))
        
        # Find target mode (closest to omega_ea)
        target_idx = -1
        if omega_ea is not None and math.isfinite(omega_ea):
            diffs = [abs(v - omega_ea) for v in sorted_mode_freqs]
            target_idx = int(np.argmin(diffs))
        
        other_labeled = False
        for idx, mode_freq in enumerate(sorted_mode_freqs):
            if f_min <= mode_freq <= f_max:
                if idx == target_idx:
                    ax.axvline(mode_freq, color=target_color, linewidth=1.2, 
                              linestyle="--", alpha=0.7, label="EA target", zorder=1)
                else:
                    ax.axvline(mode_freq, color=other_color, linewidth=0.8, 
                              linestyle=":", alpha=0.5, 
                              label="EA modes" if not other_labeled else None, zorder=1)
                    other_labeled = True
    
    # Add quality threshold lines
    minor = float(config.get("phase5_quality_minor", 250.0))
    good = float(config.get("phase5_quality_good", 1000.0))
    elite = float(config.get("phase5_quality_strong", 2500.0))
    
    ax.axhline(minor, color="#dc2626", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(good, color="#2563eb", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(elite, color="#16a34a", linestyle="--", alpha=0.4, linewidth=0.8)
    
    # Annotate thresholds on right edge
    ax.text(f_max, minor, f" Q={minor:.0f}", va="center", ha="left", fontsize=7, 
            color="#dc2626", alpha=0.7)
    ax.text(f_max, good, f" Q={good:.0f}", va="center", ha="left", fontsize=7, 
            color="#2563eb", alpha=0.7)
    ax.text(f_max, elite, f" Q={elite:.0f}", va="center", ha="left", fontsize=7, 
            color="#16a34a", alpha=0.7)
    
    # Configure axes
    ax.set_xlabel("Frequency (ω)")
    ax.set_ylabel("Q-factor")
    ax.set_yscale("log")
    ax.set_xlim(f_min, f_max)
    
    # Set y-axis limits based on data
    Q_min = max(float(np.min(Q_vals)) * 0.5, 1.0) if len(Q_vals) > 0 else 1.0
    Q_max = max(float(np.max(Q_vals)) * 2.0, elite * 2.0) if len(Q_vals) > 0 else elite * 2.0
    ax.set_ylim(Q_min, Q_max)
    
    ax.set_title("Phase 5 Q-Factor Spectrum (BLAZE V2)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle=":", which="both")
    
    # Add legend for quality colors
    from matplotlib.patches import Patch
    quality_patches = [
        Patch(facecolor=quality_colors["elite"], label=f"Elite (Q≥{elite:.0f})"),
        Patch(facecolor=quality_colors["cavity"], label=f"Cavity ({good:.0f}≤Q<{elite:.0f})"),
        Patch(facecolor=quality_colors["incipient"], label=f"Incipient ({minor:.0f}≤Q<{good:.0f})"),
        Patch(facecolor=quality_colors["diffuse"], label=f"Diffuse (Q<{minor:.0f})"),
    ]
    legend2 = ax.legend(handles=quality_patches, loc="upper left", fontsize=7, title="Quality")
    ax.add_artist(legend2)
    # Re-add the first legend
    if sorted_mode_freqs:
        ax.legend(loc="upper right", fontsize=8)
    
    fig.tight_layout()
    out_path = cdir / "phase5_q_spectrum.png"
    fig.savefig(out_path)
    plt.close(fig)
    log(f"    Saved Q-factor spectrum: {out_path.name}")


def _geometry_mask(shape: Tuple[int, int], ctx: Dict) -> np.ndarray:
    """Create a boolean mask for hole locations.
    
    After transposing Meep data, shape is (n_y, n_x) where:
    - First axis (rows) corresponds to y: from ymax (top) to ymin (bottom)
    - Second axis (cols) corresponds to x: from xmin (left) to xmax (right)
    """
    n_rows, n_cols = shape  # n_rows = y dimension, n_cols = x dimension
    xmin, xmax, ymin, ymax = ctx["bounds"]
    # x varies along columns (left to right)
    xs = np.linspace(xmin, xmax, n_cols)
    # y varies along rows (top to bottom in image = ymax to ymin)
    ys = np.linspace(ymax, ymin, n_rows)
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for center, radius in ctx["hole_centers"]:
        cx, cy = float(center[0]), float(center[1])
        rr = float(radius)
        mask |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr ** 2
    return mask


def _geometry_background(mask: np.ndarray) -> np.ndarray:
    bg = np.zeros(mask.shape + (4,), dtype=np.uint8)
    bg[..., :3] = 196
    bg[..., 3] = 210
    bg[mask, :3] = 255
    bg[mask, 3] = 255
    return bg


def _blend_with_background(
    field_rgba: np.ndarray, 
    background: np.ndarray, 
    alpha: float = 0.7
) -> np.ndarray:
    overlay = background.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay[..., :3] = (1.0 - alpha) * overlay[..., :3] + alpha * field_rgba[..., :3]
    overlay[..., 3] = 255
    return overlay


def _save_field_video(
    frames: List[np.ndarray],
    out_path: Path,
    sim_dt: float,
    target_fps: float,
    geometry_ctx: Optional[Dict] = None,
    config: Optional[Dict] = None,
):
    """
    Save field animation as MP4 video with consistent timing.
    
    Key design principles:
    1. STREAMING write to avoid OOM on large simulations
    2. Consistent simulation-time-to-video-time mapping across all videos
    3. Uses simulation dt to calculate proper frame timing
    
    Memory optimization: Uses streaming writer to process frames one at a time,
    avoiding the need to hold all RGB-converted frames in memory simultaneously.
    This is critical for large simulations (e.g., 3520x3520 @ 5000 timesteps).
    
    Args:
        frames: List of 2D field arrays (already transposed to image coords)
        out_path: Output path (will use .mp4 extension)
        sim_dt: Simulation time step between frames (Meep time units)
        target_fps: Target video framerate (frames per second)
        geometry_ctx: Optional geometry context for overlay
        config: Optional config for additional settings
    """
    if not frames:
        return
    
    # Ensure .mp4 extension
    out_path = out_path.with_suffix(".mp4")
    
    # Check for imageio availability
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        log("    WARNING: imageio not available, skipping video export")
        return
    
    # Normalize field values - compute vmax in a memory-efficient way
    # Process in chunks to avoid memory spike
    vmax = 0.0
    chunk_size = 100
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i+chunk_size]
        chunk_max = max(float(np.max(np.abs(f))) for f in chunk)
        vmax = max(vmax, chunk_max)
    if vmax <= 0:
        vmax = 1.0
    
    cmap = plt.colormaps.get_cmap("RdBu_r")
    
    # Pre-compute geometry background (same for all frames since geometry doesn't change)
    geometry_bg = None
    if geometry_ctx is not None and frames:
        geometry_mask = _geometry_mask(frames[0].shape, geometry_ctx)
        geometry_bg = _geometry_background(geometry_mask)
    
    # Calculate video parameters
    sim_time_per_video_second = 25.0
    if config is not None:
        sim_time_per_video_second = float(config.get("phase5_video_sim_time_per_second", 25.0))
    
    effective_fps = sim_time_per_video_second / max(sim_dt, 1e-6)
    effective_fps = max(10.0, min(60.0, effective_fps))
    effective_fps = round(effective_fps)
    
    n_frames = len(frames)
    log(f"    Video: {n_frames} frames, {effective_fps} fps, sim_dt={sim_dt:.3f}")
    log(f"    Video duration: {n_frames/effective_fps:.2f}s representing {n_frames*sim_dt:.1f} sim time units")
    log(f"    Frame shape: {frames[0].shape}, using streaming writer to save memory")
    
    def _convert_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert a single field array to RGB image."""
        norm = (frame / vmax + 1.0) * 0.5
        norm = np.clip(norm, 0.0, 1.0)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        if geometry_bg is not None:
            rgba = _blend_with_background(rgba, geometry_bg)
        # Convert RGBA to RGB for video (MP4 doesn't support alpha)
        return rgba[..., :3]
    
    try:
        # Use STREAMING writer to avoid OOM - process frames one at a time
        writer = imageio.get_writer(
            str(out_path),
            fps=effective_fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,  # Allow any resolution
            pixelformat="yuv420p",  # Ensure compatibility
        )
        
        # Process and write frames one at a time to minimize memory usage
        for i, frame in enumerate(frames):
            rgb = _convert_frame_to_rgb(frame)
            writer.append_data(rgb)
            # Free the RGB frame immediately after writing
            del rgb
            # Progress logging for long videos
            if (i + 1) % 200 == 0:
                log(f"    Written {i+1}/{n_frames} frames...")
        
        writer.close()
        log(f"    Saved video: {out_path.name}")
        
    except Exception as e:
        log(f"    WARNING: MP4 export failed ({e}), trying GIF fallback")
        # Fallback to GIF (also use streaming approach)
        gif_path = out_path.with_suffix(".gif")
        frame_duration = 1.0 / effective_fps
        try:
            # For GIF, we unfortunately need all frames at once for imageio
            # But we can at least convert them more efficiently
            rgba_images = []
            for frame in frames:
                norm = (frame / vmax + 1.0) * 0.5
                norm = np.clip(norm, 0.0, 1.0)
                rgba = (cmap(norm) * 255).astype(np.uint8)
                if geometry_bg is not None:
                    rgba = _blend_with_background(rgba, geometry_bg)
                rgba_images.append(rgba)
            try:
                import imageio.v3 as iio
                iio.imwrite(gif_path, rgba_images, duration=frame_duration, loop=0)
            except ImportError:
                imageio.mimwrite(gif_path, rgba_images, duration=frame_duration, loop=0)
            log(f"    Saved GIF fallback: {gif_path.name}")
        except Exception as e2:
            log(f"    WARNING: GIF fallback also failed: {e2}")


def _save_field_animation(
    frames: List[np.ndarray],
    out_path: Path,
    frame_duration: float,
    geometry_ctx: Optional[Dict] = None,
):
    if not frames:
        return
    try:
        import imageio.v3 as iio
    except ImportError:
        log("    WARNING: imageio not available, skipping GIF export")
        return

    vmax = max(float(np.max(np.abs(frame))) for frame in frames)
    if vmax <= 0:
        vmax = 1.0
    cmap = plt.colormaps.get_cmap("RdBu_r")
    
    geometry_images: List[np.ndarray] = []
    background = None
    if geometry_ctx is not None:
        for frame in frames:
            mask = _geometry_mask(frame.shape, geometry_ctx)
            geometry_images.append(_geometry_background(mask))
    
    images = []
    for idx, frame in enumerate(frames):
        norm = (frame / vmax + 1.0) * 0.5
        norm = np.clip(norm, 0.0, 1.0)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        if geometry_images:
            rgba = _blend_with_background(rgba, geometry_images[idx])
        images.append(rgba)
    
    iio.imwrite(out_path, images, duration=max(frame_duration, 0.01), loop=0)
    if geometry_images:
        overlay_path = out_path.with_name(out_path.stem + "_overlay.gif")
        iio.imwrite(overlay_path, images, duration=max(frame_duration, 0.01), loop=0)


def _apply_env_overrides(config: Dict):
    if _env_flag("PHASE5_PLOTS_ONLY"):
        config["phase5_run_meep"] = False


# =============================================================================
# Candidate Processing
# =============================================================================

def process_candidate(
    row: pd.Series | Dict[str, Any],
    config: Dict,
    run_dir: Path,
    shared_mode: bool = False,
) -> List[Dict[str, float]]:
    if row is None:
        return []
    if isinstance(row, dict):
        row = pd.Series(row)
    
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    
    if shared_mode and MPI_COMM is not None:
        phase1_meta = broadcast_value(_load_phase1_metadata(cdir) if IS_ROOT else None)
    else:
        phase1_meta = _load_phase1_metadata(cdir)
    
    if shared_mode and MPI_COMM is not None:
        mode_tuple = broadcast_value(
            _select_mode_row(cdir, config, return_table=True) if IS_ROOT else None
        )
        mode_row, modes = mode_tuple
    else:
        mode_row, modes = _select_mode_row(cdir, config, return_table=True)
    
    # Get cavity position from peak location
    # Phase 3 stores peak in fractional coords (s1, s2) ∈ [0, 1)² AND Cartesian (peak_x, peak_y)
    #
    # COORDINATE TRANSFORM (works for non-orthogonal bases like hexagonal):
    # 1. Fractional coords s ∈ [0, 1)² with origin at moiré cell corner
    # 2. The Meep geometry lattice also has origin at R = (0, 0)
    # 3. So fractional s = (0, 0) → Cartesian R = (0, 0) directly
    # 4. Transform: R = B_moire @ s (NO centering needed)
    #
    # The cavity minimum (AA stacking) occurs at s = (0, 0), which maps to R = (0, 0).
    # The Meep window is centered at origin, spanning [-L/2, L/2], so the cavity
    # is naturally at the center of the simulation domain.
    #
    peak_s1 = float(mode_row.get("peak_s1", math.nan))
    peak_s2 = float(mode_row.get("peak_s2", math.nan))
    
    if np.isfinite(peak_s1) and np.isfinite(peak_s2):
        peak_frac = np.array([peak_s1, peak_s2])
        
        # Transform to Cartesian using B_moire (works for non-orthogonal bases)
        # No centering needed: s = (0, 0) → R = (0, 0) directly
        B_moire = phase1_meta.get("B_moire")
        if B_moire is not None:
            B_moire = np.asarray(B_moire)
            cavity_pos = B_moire @ peak_frac
            if IS_ROOT:
                log(f"    Peak fractional: ({peak_s1:.4f}, {peak_s2:.4f})")
                log(f"    Cartesian: ({cavity_pos[0]:.4f}, {cavity_pos[1]:.4f})")
        else:
            # Fallback: use moire_length for approximate isotropic scaling
            moire_length = float(phase1_meta.get("moire_length", 1.0))
            # For isotropic case, need to map s ∈ [0,1) to R with origin at center
            # R = s * L_m directly (same logic as B_moire @ s)
            cavity_pos = peak_frac * moire_length
            if IS_ROOT:
                log(f"    WARNING: B_moire not available, using isotropic scaling")
    else:
        # Fallback to Cartesian peak_x/peak_y 
        # These should already be in the correct coordinate system
        peak_x = float(mode_row.get("peak_x", math.nan))
        peak_y = float(mode_row.get("peak_y", math.nan))
        if np.isfinite(peak_x) and np.isfinite(peak_y):
            # Direct use - peak_x, peak_y should be in Meep-compatible coords
            cavity_pos = np.array([peak_x, peak_y])
            if IS_ROOT:
                log(f"    Peak Cartesian: ({peak_x:.4f}, {peak_y:.4f})")
        else:
            cavity_pos = np.array([0.0, 0.0])
            if IS_ROOT:
                log(f"    WARNING: peak coordinates not found, using origin")
    
    geometry_ctx = _build_geometry(phase1_meta, config)
    directions = _candidate_directions(geometry_ctx["a1"], geometry_ctx["a2"])
    default_step = max(geometry_ctx["radius"] * 0.25, 0.05)
    step_cfg = config.get("phase5_source_step")
    step = float(step_cfg) if step_cfg is not None else default_step
    max_steps = int(config.get("phase5_source_max_steps", 400))
    source_pos, shift_distance = _find_source_position(
        cavity_pos,
        geometry_ctx["hole_centers"],
        step,
        max_steps,
        geometry_ctx["bounds"],
        directions,
    )
    
    # Log source position for coordinate verification
    if IS_ROOT:
        log(f"    Cavity position: ({cavity_pos[0]:.4f}, {cavity_pos[1]:.4f})")
        log(f"    Source position: ({source_pos[0]:.4f}, {source_pos[1]:.4f})")
        if shift_distance > 0:
            log(f"    Source shifted by: {shift_distance:.4f}")
    
    # Geometry preview
    preview_mode = str(config.get("phase5_preview_backend", "meep")).strip().lower()
    render_preview = bool(config.get("phase5_render_preview", True))
    if render_preview and preview_mode != "none":
        preview_path = cdir / "phase5_geometry_preview.png"
        dpi = int(config.get("phase5_preview_dpi", 400))
        if preview_mode == "meep" and IS_ROOT:
            resolution = float(config.get("phase5_resolution", 32))
            sim = _build_simulation(geometry_ctx, resolution)
            _render_geometry_preview(sim, cavity_pos, source_pos, preview_path, dpi)
            sim.reset_meep()
        elif preview_mode == "static" and IS_ROOT:
            _render_geometry_preview_static(
                geometry_ctx, cavity_pos, source_pos, preview_path, dpi
            )
    
    # Compute dynamic FWHM based on distance to next mode
    fwidth = _compute_dynamic_fwidth(mode_row, modes, config)
    
    # Plot pulse spectrum
    if IS_ROOT and bool(config.get("phase5_plot_pulse", True)):
        _plot_gaussian_pulse(cdir, mode_row, fwidth, config, modes)

    component_name = str(config.get("phase5_component", "Ez"))
    component = getattr(mp, component_name, None)
    if component is None:
        component = mp.Ez
    freq = float(mode_row.get("omega_cavity", mode_row.get("delta_omega", 0.0)))

    run_meep = bool(config.get("phase5_run_meep", True))
    rows: List[Dict[str, float]] = []
    frame_data: Dict = {"frames": [], "capture_dt": 0.0, "frame_times": []}
    if run_meep:
        rows, frame_data = _run_meep(geometry_ctx, component, freq, fwidth, config, source_pos)
    
    frames = frame_data.get("frames", [])
    capture_dt = frame_data.get("capture_dt", 3.0)

    summary = {
        "candidate_id": cid,
        "mode_index": int(mode_row.get("mode_index", 0)),
        "omega_ea": freq,
        "window_x": geometry_ctx["window_span"][0],
        "window_y": geometry_ctx["window_span"][1],
        "n_bottom": geometry_ctx["counts"]["bottom"],
        "n_top": geometry_ctx["counts"]["top"],
        "source_shift": shift_distance,
        "ran_meep": run_meep,
        "recorded_modes": len(rows),
        "best_q": max((r["Q"] for r in rows), default=float("nan")),
        "best_freq": max((r["freq_meep"] for r in rows), default=float("nan")),
    }
    
    enriched_rows: List[Dict[str, float]] = rows
    if run_meep and (not shared_mode or IS_ROOT):
        enriched_rows = _write_results(
            cdir, rows, cid, int(mode_row.get("mode_index", 0)),
            freq, shift_distance, config
        )
        _write_report(cdir, summary, enriched_rows, config)
    
    # Handle video output
    if IS_ROOT:
        video_path = cdir / "phase5_field_animation.mp4"
        stream_mode = frame_data.get("stream_mode", False)
        stream_path = frame_data.get("stream_path")
        
        if stream_mode and stream_path:
            # Streaming mode: move temp file to final location
            import shutil
            temp_path = Path(stream_path)
            if temp_path.exists():
                try:
                    shutil.move(str(temp_path), str(video_path))
                    log(f"    Video saved: {video_path.name} ({frame_data.get('frame_count', 0)} frames)")
                except Exception as e:
                    log(f"    WARNING: Could not move video file: {e}")
        elif frames:
            # Non-streaming mode: render frames to video
            target_fps = float(config.get("phase5_video_fps", 30.0))
            _save_field_video(
                frames, video_path, capture_dt, target_fps, geometry_ctx, config
            )
    
    # Plot Q-factor spectrum (after results are written)
    if IS_ROOT and run_meep and bool(config.get("phase5_plot_q_spectrum", True)):
        _plot_q_factor_spectrum(cdir, config, modes)
    
    rank_logging = bool(config.get("phase5_rank_logging", False))
    if IS_ROOT or rank_logging:
        log_rank(f"    Phase 5 artifacts written for candidate {cid}")
    mpi_barrier()
    return enriched_rows


# =============================================================================
# Main Runner
# =============================================================================

def run_phase5_blaze_v2(
    run_dir: str | Path, 
    config_path: str | Path, 
    plots_only: bool = False,
    candidate_id: int | None = None,
):
    """
    Run BLAZE Phase 5 V2 on all candidates in a run directory.
    
    Args:
        run_dir: Path to the BLAZE V2 run directory (or 'auto'/'latest')
        config_path: Path to Phase 5 configuration YAML
        plots_only: If True, skip Meep simulation and only generate plots
        candidate_id: If specified, only process this candidate
    """
    # Load config early so we can set up parallelization
    config = load_yaml(config_path)
    
    # Override config for plots-only mode
    if plots_only:
        config["phase5_run_meep"] = False
        config["phase5_render_preview"] = True
        config["phase5_plot_pulse"] = True
        config["K_candidates"] = None  # Process all candidates
        log("[--plots mode] Skipping Meep simulation, generating plots for all candidates.")
    if MPI_COMM is not None:
        config = broadcast_value(config)
    _apply_env_overrides(config)
    
    # Set up OpenMP threads for Meep parallelization
    parallel_info = _setup_parallelization(config)
    
    if MPI_ENABLED:
        version = MPI.Get_version()
        log(f"\nUsing MPI version {version[0]}.{version[1]}, {MPI_SIZE} processes")
    
    log("=" * 70)
    log("PHASE 5 (BLAZE V2): Meep Cavity Validation")
    log("=" * 70)
    
    # Log parallelization setup
    _log_parallelization_info(parallel_info)
    
    run_dir = resolve_blaze_v2_run_dir(run_dir, config)
    log(f"Using run directory: {run_dir}")
    
    parallel_mode = str(config.get("phase5_parallel_mode", "shared")).strip().lower()
    if parallel_mode not in {"shared", "scatter"}:
        log(f"Unknown phase5_parallel_mode='{parallel_mode}', defaulting to 'shared'.")
        parallel_mode = "shared"
    shared_mode = MPI_ENABLED and parallel_mode == "shared"
    if MPI_ENABLED:
        if shared_mode:
            log("MPI parallel mode: shared (all ranks cooperate on each candidate)")
        else:
            log("MPI parallel mode: scatter (candidates divided across ranks)")

    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if IS_ROOT and candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    elif IS_ROOT:
        log(f"WARNING: {candidates_path} not found; relying on per-candidate metadata only.")
    if MPI_COMM is not None:
        candidate_payload = (
            candidate_frame.to_dict(orient="list") if candidate_frame is not None else None
        )
        candidate_payload = broadcast_value(candidate_payload)
        candidate_frame = pd.DataFrame.from_dict(candidate_payload) if candidate_payload is not None else None

    discovered = _discover_phase3_candidates(run_dir) if IS_ROOT else None
    if MPI_COMM is not None:
        discovered = broadcast_value(discovered)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 3 eigenvalues found in {run_dir}. "
            "Run Phase 3 before Phase 5."
        )

    # Filter to specific candidate if requested
    if candidate_id is not None:
        discovered = [(cid, path) for cid, path in discovered if cid == candidate_id]
        if not discovered:
            raise FileNotFoundError(
                f"Candidate {candidate_id} not found in {run_dir}. "
                "Check that Phase 3 has been run for this candidate."
            )
        log(f"Processing candidate {candidate_id} only.")
    else:
        K = config.get("K_candidates")
        if isinstance(K, int) and K > 0:
            discovered = discovered[:K]
            log(f"Processing {len(discovered)} candidate directories (limited to K={K}).")
        else:
            log(f"Processing {len(discovered)} candidate directories (all Phase 3 outputs found).")

    aggregate_rows: List[Dict[str, float]] = []
    if shared_mode:
        for cid, _ in discovered:
            if IS_ROOT:
                row_dict = _load_candidate_metadata(run_dir, cid, candidate_frame)
            else:
                row_dict = None
            if MPI_COMM is not None:
                row_dict = broadcast_value(row_dict if IS_ROOT else None)
            if row_dict is None:
                continue
            try:
                if IS_ROOT:
                    log(f"  Candidate {cid}: running Phase 5 (shared mode)")
                process_rows = process_candidate(row_dict, config, Path(run_dir), shared_mode=True)
                if IS_ROOT and process_rows:
                    aggregate_rows.extend(process_rows)
            except FileNotFoundError as exc:
                log_rank(f"    Skipping candidate {cid}: {exc}")
            except Exception as exc:
                log_rank(f"    Phase 5 failed for candidate {cid}: {exc}")
                if MPI_COMM is not None:
                    MPI_COMM.Abort(1)
                else:
                    raise
    else:
        if MPI_ENABLED:
            assigned_candidates = [
                entry for idx, entry in enumerate(discovered) if idx % MPI_SIZE == MPI_RANK
            ]
            log_rank(f"Assigned {len(assigned_candidates)} of {len(discovered)} candidates to this rank")
        else:
            assigned_candidates = discovered

        aggregate_rows_local: List[Dict[str, float]] = []
        for cid, _ in assigned_candidates:
            row_dict = _load_candidate_metadata(run_dir, cid, candidate_frame)
            try:
                log_rank(f"  Candidate {cid}: running Phase 5")
                candidate_rows = process_candidate(row_dict, config, Path(run_dir))
                aggregate_rows_local.extend(candidate_rows)
            except FileNotFoundError as exc:
                log_rank(f"    Skipping candidate {cid}: {exc}")
            except Exception as exc:
                log_rank(f"    Phase 5 failed for candidate {cid}: {exc}")
                if MPI_COMM is not None:
                    MPI_COMM.Abort(1)
                else:
                    raise

        if MPI_COMM is not None:
            gathered_rows = MPI_COMM.gather(aggregate_rows_local, root=0)
            if IS_ROOT:
                for chunk in gathered_rows:
                    if chunk:
                        aggregate_rows.extend(chunk)
        else:
            aggregate_rows = aggregate_rows_local

    if IS_ROOT and aggregate_rows:
        out_path = Path(run_dir) / "phase5_q_factor_results.csv"
        pd.DataFrame(aggregate_rows).to_csv(out_path, index=False)
        log(f"Wrote aggregate results to {out_path}")

    log("Phase 5 (BLAZE V2) completed.\n")


def run_phase5(
    run_dir: str | Path, 
    config_path: str | Path, 
    plots_only: bool = False,
    candidate_id: int | None = None,
    qplot_only: bool = False,
):
    """Entry point for command-line usage.
    
    Args:
        run_dir: Path to run directory or 'auto'/'latest'
        config_path: Path to Phase 5 config YAML
        plots_only: If True, skip Meep simulation and only generate plots
        candidate_id: If specified, only process this candidate
        qplot_only: If True, only generate Q-factor spectrum plot from existing CSV
    """
    if qplot_only:
        return run_qplot_only(run_dir, config_path, candidate_id=candidate_id)
    return run_phase5_blaze_v2(run_dir, config_path, plots_only=plots_only, candidate_id=candidate_id)


def run_qplot_only(
    run_dir: str | Path,
    config_path: str | Path,
    candidate_id: int | None = None,
):
    """
    Generate only the Q-factor spectrum plot from existing phase5_q_factor_results.csv.
    
    Uses the same candidate selection logic as the rest of Phase 5.
    """
    config = load_yaml(config_path)
    run_dir = resolve_blaze_v2_run_dir(run_dir, config)
    
    log("=" * 70)
    log("PHASE 5 (BLAZE V2): Q-Factor Spectrum Plot Only")
    log("=" * 70)
    log(f"Using run directory: {run_dir}")
    
    # Discover candidates (same logic as main runner)
    discovered = _discover_phase3_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(f"No candidate directories found in {run_dir}")
    
    # Filter to specific candidate if requested
    if candidate_id is not None:
        discovered = [(cid, path) for cid, path in discovered if cid == candidate_id]
        if not discovered:
            raise FileNotFoundError(f"Candidate {candidate_id} not found in {run_dir}")
        log(f"Processing candidate {candidate_id} only.")
    else:
        # Use first/only candidate (matching default behavior)
        K = config.get("K_candidates")
        if isinstance(K, int) and K > 0:
            discovered = discovered[:K]
        log(f"Found {len(discovered)} candidate(s).")
    
    for cid, cdir in discovered:
        log(f"\n  Candidate {cid}:")
        
        # Check if results CSV exists
        results_path = cdir / "phase5_q_factor_results.csv"
        if not results_path.exists():
            log(f"    Skipping: {results_path.name} not found")
            continue
        
        # Load modes from phase3_eigenvalues.csv for vertical lines
        modes = None
        eigenvalues_path = cdir / "phase3_eigenvalues.csv"
        if eigenvalues_path.exists():
            try:
                modes = pd.read_csv(eigenvalues_path)
            except Exception as e:
                log(f"    WARNING: Could not load eigenvalues: {e}")
        
        # Generate the Q-spectrum plot
        _plot_q_factor_spectrum(cdir, config, modes)
    
    log("\nQ-factor spectrum plot generation completed.\n")


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 5 V2 config."""
    return PROJECT_ROOT / "configsV2" / "phase5_blaze.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5 (BLAZE V2): Meep-based cavity validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python blaze_phasesV2/phase5_blaze.py                    # Latest run, all candidates
  python blaze_phasesV2/phase5_blaze.py 2                  # Candidate 2 from latest run
  python blaze_phasesV2/phase5_blaze.py auto               # Same as no args
  python blaze_phasesV2/phase5_blaze.py runsV2/phase0_...  # Specific run, all candidates
  python blaze_phasesV2/phase5_blaze.py runsV2/... 2       # Specific run, candidate 2
  python blaze_phasesV2/phase5_blaze.py --plots            # Plots only, all candidates
  python blaze_phasesV2/phase5_blaze.py 2 --plots          # Plots only, candidate 2
  python blaze_phasesV2/phase5_blaze.py --qplot            # Q-spectrum plot only (from existing CSV)
  python blaze_phasesV2/phase5_blaze.py 2 --qplot          # Q-spectrum plot for candidate 2
"""
    )
    parser.add_argument(
        "run_dir", nargs="?", default="auto",
        help="Run directory path, 'auto'/'latest', or candidate number (default: auto)"
    )
    parser.add_argument(
        "candidate", nargs="?", default=None,
        help="Candidate number to process (when run_dir is a path)"
    )
    parser.add_argument(
        "--plots", action="store_true",
        help="Generate geometry preview and pulse plots only (skip Meep simulation)"
    )
    parser.add_argument(
        "--qplot", action="store_true",
        help="Generate only the Q-factor spectrum plot from existing phase5_q_factor_results.csv"
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to Phase 5 config YAML (default: configsV2/phase5_blaze.yaml)"
    )
    
    args = parser.parse_args()
    
    # Determine if run_dir is actually a candidate number
    run_dir = args.run_dir
    candidate_id = None
    
    # Check if run_dir is a pure integer (candidate number)
    if run_dir.isdigit():
        candidate_id = int(run_dir)
        run_dir = "auto"
    elif args.candidate is not None and args.candidate.isdigit():
        candidate_id = int(args.candidate)
    
    # Resolve config path
    if args.config is None:
        config_path = get_default_config_path()
        if not config_path.exists():
            raise SystemExit(f"Default config not found: {config_path}")
        print(f"Using default config: {config_path}")
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
    
    run_phase5(run_dir, config_path, plots_only=args.plots, candidate_id=candidate_id, qplot_only=args.qplot)
