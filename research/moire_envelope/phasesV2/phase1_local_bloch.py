"""
Phase 1: Local Bloch problems at frozen registry — V2 Pipeline

This implements the V2 formulation using FRACTIONAL COORDINATES for the moiré
unit cell sampling. See README_V2.md for the mathematical details.

KEY V2 CHANGES from legacy Phase 1:
1. Sample in fractional coordinates (s_1, s_2) ∈ [0,1)² instead of Cartesian
2. Corrected registry formula: δ(R) = (R(θ) - I) · R + τ  (NO division by η!)
3. Store s_grid as primary grid with B_moire attribute for Cartesian transforms

For each top-K candidate from Phase 0, this phase:
1. Builds a fractional coordinate grid (s_1, s_2) over the moiré unit cell
2. Computes the registry map δ(s) → monolayer stacking shift
3. For each grid point, runs a "frozen bilayer" MPB calculation:
   - Builds bilayer geometry with local stacking shift δ
   - Computes band structure at k₀ and nearby k-points
   - Extracts: ω₀(s), v_g(s), M⁻¹(s) (inverse mass tensor)
4. Assembles potential V(s) = ω₀(s) - ω_ref
5. Outputs HDF5 files with all EA inputs for Phase 2

Based on README_V2.md Section 4.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import time

# ============================================================================
# CRITICAL: Configure OpenMP threading BEFORE importing meep/mpb
# ============================================================================
# MPB uses OpenMP for internal parallelization. The thread count must be set
# before the OpenMP runtime is initialized (i.e., before importing meep).

def _get_num_threads_from_env_or_mpi():
    """Determine optimal thread count before MPI/meep imports."""
    # Check if user specified explicitly via environment
    if 'OMP_NUM_THREADS' in os.environ:
        return int(os.environ['OMP_NUM_THREADS'])
    
    # Check config-based setting via environment variable
    num_threads_env = os.environ.get('MSL_NUM_THREADS')
    if num_threads_env:
        return int(num_threads_env)
    
    # Auto-detect: check if we're running under MPI
    # MPI sets various environment variables we can check
    mpi_size = None
    for var in ['OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'MPI_LOCALNRANKS']:
        if var in os.environ:
            try:
                mpi_size = int(os.environ[var])
                break
            except ValueError:
                pass
    
    if mpi_size and mpi_size > 1:
        # Running with multiple MPI ranks: use 1 thread per rank
        return 1
    else:
        # Serial or single MPI rank: use all physical cores
        return os.cpu_count() or 16

# Set threading before any meep import
_num_threads = _get_num_threads_from_env_or_mpi()
os.environ['OMP_NUM_THREADS'] = str(_num_threads)
os.environ['OMP_PROC_BIND'] = 'false'  # Allow thread migration

# ============================================================================

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm is unavailable
    def tqdm(iterable, **_kwargs):
        return iterable

try:  # Allow Phase 1 to run even when mpi4py is not installed
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None

# Import common utilities (this imports meep/mpb)
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.mpb_utils import compute_local_band_at_registry
from common.geometry import build_lattice
from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json, load_json
from common.plotting import plot_phase1_fields, plot_phase1_lattice_panels


if MPI is not None:
    MPI_COMM = MPI.COMM_WORLD
    MPI_RANK = MPI_COMM.Get_rank()
    MPI_SIZE = MPI_COMM.Get_size()
else:  # Fallback for serial execution without MPI
    MPI_COMM = None
    MPI_RANK = 0
    MPI_SIZE = 1

IS_ROOT = MPI_RANK == 0
MPI_ENABLED = MPI_COMM is not None and MPI_SIZE > 1


def log(message):
    """Print from rank 0 only."""
    if IS_ROOT:
        print(message, flush=True)


def log_rank(message):
    """Print a message from any rank with a prefix (use sparingly)."""
    prefix = f"[Rank {MPI_RANK}] " if MPI_COMM is not None else ""
    print(f"{prefix}{message}", flush=True)


def mpi_barrier():
    """Synchronize all MPI ranks when MPI is available."""
    if MPI_COMM is not None:
        MPI_COMM.Barrier()


def format_eta(seconds: float) -> str:
    """Return a compact ETA string (e.g., 12m34s)."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def update_registry_progress(ticker, processed_total, total_points, start_time, increment):
    """Update tqdm ticker and return new processed_total."""
    if ticker is None or increment <= 0:
        return processed_total
    ticker.update(increment)
    processed_total += increment
    elapsed = time.time() - start_time
    if processed_total > 0 and elapsed > 1e-3:
        rate = processed_total / elapsed
        remaining = max((total_points - processed_total) / max(rate, 1e-6), 0.0)
        ticker.set_postfix_str(f"ETA {format_eta(remaining)}")
    return processed_total


def broadcast_from_root(value):
    """Broadcast a Python object from rank 0 to all ranks."""
    if MPI_COMM is not None:
        return MPI_COMM.bcast(value if IS_ROOT else None, root=0)
    return value


def run_root_only(action, *args, **kwargs):
    """Execute an action on rank 0 and propagate any exception to all ranks."""
    error = None
    if IS_ROOT:
        try:
            action(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive sync
            error = exc
    if MPI_COMM is not None:
        error = MPI_COMM.bcast(error, root=0)
    if error is not None:
        raise error


def run_root_with_result(action, *args, **kwargs):
    """Run an action on rank 0, broadcast result, and propagate errors."""
    result = None
    error = None
    if IS_ROOT:
        try:
            result = action(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive sync
            error = exc
    if MPI_COMM is not None:
        error = MPI_COMM.bcast(error, root=0)
    if error is not None:
        raise error
    if MPI_COMM is not None:
        result = MPI_COMM.bcast(result if IS_ROOT else None, root=0)
    return result


# ==============================================================================
# V2 Fractional Coordinate Functions
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """
    Build the monolayer lattice basis matrix B = (a1 | a2).
    
    Args:
        lattice_type: 'square', 'hex', or 'triangular'
        a: Lattice constant
        
    Returns:
        B: 2x2 matrix with columns as basis vectors
    """
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], 
                              [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        return a * np.array([[1.0, 0.5], 
                              [0.0, np.sqrt(3)/2]])
    elif lattice_type == 'rect':
        # Default rectangular with 1.5 aspect ratio
        return np.array([[a, 0.0], 
                         [0.0, 1.5*a]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """
    Compute the moiré lattice basis vectors.
    
    For small twist angle θ, the moiré period is L_m ≈ a/θ, so:
    B_moire ≈ B_mono / θ = η * B_mono where η = 1/θ
    
    Args:
        B_mono: 2x2 monolayer basis matrix
        theta_rad: Twist angle in radians
        
    Returns:
        B_moire: 2x2 moiré basis matrix
    """
    # η = L_m / a ≈ 1 / (2 sin(θ/2)) ≈ 1/θ for small θ
    eta = 1.0 / (2 * np.sin(theta_rad / 2))
    return eta * B_mono


def build_fractional_grid(Ns1: int, Ns2: int) -> np.ndarray:
    """
    Build uniform grid in fractional coordinates.
    
    This is the KEY V2 function: we sample the moiré unit cell
    on a regular grid in fractional coordinates (s1, s2) ∈ [0, 1)².
    
    Args:
        Ns1: Number of grid points in s1 direction
        Ns2: Number of grid points in s2 direction
        
    Returns:
        s_grid: array [Ns1, Ns2, 2] with s_grid[i,j] = (s1, s2)
                where s1 = i/Ns1, s2 = j/Ns2, both in [0, 1)
    """
    s1 = np.arange(Ns1) / Ns1  # [0, 1/N, 2/N, ..., (N-1)/N]
    s2 = np.arange(Ns2) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    return np.stack([S1, S2], axis=-1)


def fractional_to_cartesian(s_grid: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Transform fractional coordinates to Cartesian.
    
    Args:
        s_grid: [Ns1, Ns2, 2] fractional coordinates
        B: [2, 2] basis matrix with columns (A1, A2)
    
    Returns:
        R_grid: [Ns1, Ns2, 2] Cartesian positions
    """
    return np.einsum('ij,...j->...i', B, s_grid)


def compute_registry_fractional_v2(s_grid: np.ndarray, B_moire: np.ndarray, 
                                    B_mono: np.ndarray, theta_rad: float,
                                    tau_frac: np.ndarray) -> np.ndarray:
    """
    Compute registry shift in monolayer fractional coordinates.
    
    ⚠️ V2 CORRECTED FORMULA:
    δ(R) = (R(θ) - I) · R + τ
    
    NO division by η! The physical interpretation: when R spans one moiré 
    length L_m ≈ a/θ, the shift δ spans approximately one monolayer lattice 
    constant a. This creates a 1:1 mapping between moiré and monolayer cells.
    
    Args:
        s_grid: [Ns1, Ns2, 2] moiré fractional coordinates
        B_moire: [2, 2] moiré basis matrix (A1, A2)
        B_mono: [2, 2] monolayer basis matrix (a1, a2)
        theta_rad: twist angle in radians
        tau_frac: [2] stacking gauge in monolayer fractional coords
    
    Returns:
        delta_frac: [Ns1, Ns2, 2] registry in monolayer fractional coords, wrapped to [0,1)
    """
    # Convert moiré fractional → Cartesian
    R_grid = np.einsum('ij,...j->...i', B_moire, s_grid)
    
    # Rotation matrix
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_rot = np.array([[c, -s], [s, c]])
    
    # Registry shift in Cartesian: δ_cart = (R(θ) - I) · R
    # NO DIVISION BY η!
    delta_cart = np.einsum('ij,...j->...i', R_rot - np.eye(2), R_grid)
    
    # Convert to monolayer fractional coords
    B_mono_inv = np.linalg.inv(B_mono)
    delta_frac = np.einsum('ij,...j->...i', B_mono_inv, delta_cart)
    
    # Add stacking gauge and wrap to [0, 1)
    delta_frac = delta_frac + tau_frac
    delta_frac = delta_frac - np.floor(delta_frac)
    
    return delta_frac


def compute_eta_geometric(theta_rad: float) -> float:
    """
    Compute geometric moiré scale factor η_geom = L_m / a ≈ 1 / (2 sin(θ/2)).
    
    This is the ratio of moiré period to monolayer lattice constant.
    Used for: B_moire = η_geom * B_mono
    For small angles, η_geom ≈ 1/θ.
    """
    return 1.0 / (2 * np.sin(theta_rad / 2))


def compute_eta_physics(theta_rad: float) -> float:
    """
    Compute physics small parameter η = a / L_m ≈ 2 sin(θ/2) ≈ θ.
    
    This is the small parameter in the envelope Hamiltonian:
    H = V(s) + (η²/2) ∇_s · M̃⁻¹(s) ∇_s
    
    For θ = 1.1°: η ≈ 0.019
    """
    return 2 * np.sin(theta_rad / 2)


# ==============================================================================
# Candidate Parameter Extraction
# ==============================================================================

def extract_candidate_parameters(row):
    """
    Extract relevant parameters from candidate row
    
    Args:
        row: Pandas Series with candidate data
        
    Returns:
        dict: Parameter dictionary
    """
    params = {
        'candidate_id': int(row['candidate_id']),
        'lattice_type': row['lattice_type'],
        'a': float(row['a']),
        'r_over_a': float(row['r_over_a']),
        'eps_bg': float(row['eps_bg']),
        'band_index': int(row['band_index']),
        'k_label': row['k_label'],
        'k0_x': float(row['k0_x']),
        'k0_y': float(row['k0_y']),
        'omega0': float(row['omega0']),
    }
    
    # Handle polarization (from library-based Phase 0 with merged bands)
    if 'polarization' in row:
        params['polarization'] = row['polarization']
    if 'dominant_polarization' in row:
        params['dominant_polarization'] = row['dominant_polarization']
    if 'local_polarization' in row:
        params['local_polarization'] = row['local_polarization']
    
    # Handle optional moiré parameters (may not exist for monolayer runs)
    if 'theta_deg' in row:
        params['theta_deg'] = float(row['theta_deg'])
        params['theta_rad'] = math.radians(params['theta_deg'])
    
    if 'G_magnitude' in row:
        params['G_magnitude'] = float(row['G_magnitude'])
    
    if 'moire_length' in row:
        params['moire_length'] = float(row['moire_length'])
    
    return params


def ensure_moire_metadata(candidate_params: dict, config: dict) -> dict:
    """
    Ensure candidate dict contains twist + moiré info.
    
    Returns metadata dict with basis matrices and geometric parameters.
    """
    theta_deg = candidate_params.get('theta_deg')
    if theta_deg is None or (isinstance(theta_deg, float) and math.isnan(theta_deg)):
        theta_deg = config.get('default_theta_deg')
        if theta_deg is None:
            raise ValueError(
                "Candidate is missing theta_deg; specify 'default_theta_deg' in the Phase 1 "
                "config to define the moire twist applied to all Phase 0 monolayer candidates."
            )
    theta_deg = float(theta_deg)
    theta_rad = math.radians(theta_deg)
    
    lattice_type = candidate_params['lattice_type']
    a = candidate_params['a']
    
    # Build basis matrices (V2 approach)
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    
    # Compute moiré scale factors
    eta_geom = compute_eta_geometric(theta_rad)  # L_m / a (large, ~52)
    eta = compute_eta_physics(theta_rad)  # a / L_m (small, ~0.019) for Hamiltonian
    moire_length = eta_geom * a  # L_m ≈ a/θ
    
    # Store in candidate params
    candidate_params['theta_deg'] = theta_deg
    candidate_params['theta_rad'] = theta_rad
    candidate_params['eta'] = eta
    candidate_params['moire_length'] = moire_length
    
    # Compute reciprocal vector magnitude
    if lattice_type in ('hex', 'triangular'):
        G_magnitude = 4 * math.pi / (math.sqrt(3) * a) * math.sin(theta_rad / 2)
    else:
        G_magnitude = 2 * math.pi / a * theta_rad
    candidate_params['G_magnitude'] = G_magnitude
    
    return {
        'B_mono': B_mono,
        'B_moire': B_moire,
        'theta_rad': theta_rad,
        'eta': eta,
        'moire_length': moire_length,
        # Legacy compatibility: expose a1, a2 vectors
        'a1_vec': B_mono[:, 0],
        'a2_vec': B_mono[:, 1],
    }


# ==============================================================================
# Field Statistics
# ==============================================================================

def summarize_phase1_fields(omega0_grid, vg_grid, M_inv_grid, V_grid):
    """Compute quick diagnostic statistics for Phase 1 outputs."""
    vg_norm = np.linalg.norm(vg_grid, axis=-1)
    omega0_flat = omega0_grid.ravel()
    V_flat = V_grid.ravel()
    vg_flat = vg_norm.ravel()
    M_flat = M_inv_grid.reshape(-1, 2, 2)
    eigvals = np.linalg.eigvalsh(M_flat)

    stats = {
        'omega0_min': float(omega0_flat.min()),
        'omega0_max': float(omega0_flat.max()),
        'omega0_mean': float(omega0_flat.mean()),
        'omega0_std': float(omega0_flat.std()),
        'V_min': float(V_flat.min()),
        'V_max': float(V_flat.max()),
        'V_std': float(V_flat.std()),
        'vg_norm_min': float(vg_flat.min()),
        'vg_norm_max': float(vg_flat.max()),
        'vg_norm_mean': float(vg_flat.mean()),
        'M_inv_min_eig': float(eigvals.min()),
        'M_inv_max_eig': float(eigvals.max()),
        'M_inv_mean_eig': float(eigvals.mean()),
    }

    return stats


# ==============================================================================
# Bilayer Geometry Construction
# ==============================================================================

def build_bilayer_geometry_at_delta(base_params: dict, delta_frac: np.ndarray,
                                     layer_separation: float = 0.0) -> dict:
    """
    Build bilayer photonic crystal geometry with stacking shift δ
    
    The physical model: bottom layer is fixed at the origin, top layer is 
    displaced by δ. This matches the BLAZE convention and the envelope
    approximation derivation where the base monolayer doesn't rotate.
    
    Args:
        base_params: Base lattice parameters (lattice_type, r_over_a, eps_bg, a, B_mono)
        delta_frac: Fractional shift [2,] in monolayer lattice coordinates
        layer_separation: Vertical separation (for 3D, not used in 2D approximation)
        
    Returns:
        dict: Geometry parameters for MPB
    """
    # Build base lattice
    geom = build_lattice(
        base_params['lattice_type'],
        base_params['r_over_a'],
        base_params['eps_bg'],
        base_params['a']
    )
    
    # Get monolayer basis for coordinate conversion
    B_mono = base_params['B_mono']
    
    # Convert fractional delta to Cartesian for reference
    delta_cart = B_mono @ delta_frac
    
    # MPB's geometry 'center' argument uses LATTICE (fractional) coordinates
    # Physical model: bottom layer fixed at origin, top layer shifted by δ
    # This matches BLAZE convention and envelope approximation derivation
    hole_translations = [
        np.array([0.0, 0.0, 0.0]),                          # Bottom layer: fixed at origin
        np.array([delta_frac[0], delta_frac[1], 0.0]),      # Top layer: shifted by δ
    ]

    geom['delta_frac'] = delta_frac
    geom['delta_physical'] = delta_cart
    geom['layer_separation'] = layer_separation
    geom['hole_translations'] = hole_translations
    
    return geom


def _load_phase0p5_overrides(cdir: Path, config: dict) -> dict:
    """Load recommended MPB resolution/Δk from Phase 0.5 outputs if enabled."""
    if not bool(config.get('phase1_auto_from_phase0p5', True)):
        return {}
    rec_path = cdir / "phase0p5_recommended.json"
    if not rec_path.exists():
        return {}
    try:
        payload = load_json(rec_path)
    except Exception as exc:  # pragma: no cover - defensive path
        log(f"  WARNING: Failed to parse {rec_path}: {exc}")
        return {}
    selected = payload.get("selected") or {}
    resolution = selected.get("resolution")
    dk = selected.get("dk")
    if resolution is None or dk is None:
        log(f"  WARNING: {rec_path} missing 'resolution'/'dk'; ignoring recommendation.")
        return {}
    return {
        "resolution": float(resolution),
        "dk": float(dk),
        "score": float(selected.get("score", float("nan"))),
    }


# ==============================================================================
# Stencil Data Helpers
# ==============================================================================

def stencil_dict_to_array(omega_values: dict, fd_order: int) -> np.ndarray:
    """
    Convert stencil omega_values dict to 2D array.
    
    Args:
        omega_values: dict mapping (offset_x, offset_y) -> omega
        fd_order: 2 or 4 (determines stencil size)
        
    Returns:
        np.ndarray of shape (n_stencil, n_stencil) where n_stencil = 5 for order 4, 3 for order 2
    """
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
        n = 5
    else:
        offsets = [-1, 0, 1]
        n = 3
    
    arr = np.zeros((n, n))
    for ix, ox in enumerate(offsets):
        for iy, oy in enumerate(offsets):
            key = (ox, oy)
            if key in omega_values:
                arr[ix, iy] = omega_values[key]
            # For order 2, not all (ox, oy) pairs are present
    
    return arr


# ==============================================================================
# Point-wise Band Computation
# ==============================================================================

def compute_point_band_data(args):
    """
    Compute band data at a single grid point (for parallel execution)
    
    Args:
        args: Tuple of (i, j, s_vec, delta_frac, base_params, config)
        
    Returns:
        tuple: (i, j, omega0, vg, M_inv, stencil_data)
            stencil_data: dict with omega_values, dk, fd_order
    """
    i, j, s_vec, delta_frac, base_params, config = args
    
    # Build geometry for this registry point
    geom = build_bilayer_geometry_at_delta(base_params, delta_frac)
    
    # Compute local band data at k₀
    k0 = np.array([base_params['k0_x'], base_params['k0_y'], 0.0])
    band_idx = base_params['band_index']
    
    # Run MPB at this frozen geometry
    try:
        omega0, vg, M_inv, stencil_data = compute_local_band_at_registry(
            geom, k0, band_idx, config
        )
    except Exception as e:
        # If computation fails, use fallback values
        rank_txt = f"[Rank {MPI_RANK}] " if MPI_COMM is not None else ""
        print(f"{rank_txt}Warning: Failed at ({i},{j}) s={s_vec}: {e}", flush=True)
        omega0 = base_params['omega0']
        vg = np.zeros(2)
        M_inv = np.eye(2)
        stencil_data = None  # No stencil data on failure
    
    return (i, j, omega0, vg, M_inv, stencil_data)


# ==============================================================================
# Main Candidate Processing
# ==============================================================================

def process_candidate(candidate_params: dict, config: dict, run_dir: Path):
    """
    Process a single candidate through Phase 1 (V2)
    
    Args:
        candidate_params: Dictionary with candidate parameters
        config: Configuration dictionary
        run_dir: Run directory path
    """
    cid = candidate_params['candidate_id']
    log(f"\n=== Processing Candidate {cid} (V2 Pipeline) ===")
    log(f"  Lattice: {candidate_params['lattice_type']}")
    log(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    log(f"  Band {candidate_params['band_index']} at k={candidate_params['k_label']}")
    if candidate_params.get('polarization'):
        log(f"  Polarization: {candidate_params['polarization']}")
    if candidate_params.get('theta_deg') is not None:
        log(f"  Twist theta: {candidate_params['theta_deg']:.3f} deg")
    
    # Create candidate directory
    cdir = candidate_dir(run_dir, cid)
    run_root_only(lambda: cdir.mkdir(parents=True, exist_ok=True))
    
    # Check for Phase 0.5 convergence data
    phase0p5_exists = False
    if IS_ROOT:
        phase0p5_exists = (cdir / "phase0p5_report.md").exists()
    phase0p5_exists = broadcast_from_root(phase0p5_exists)
    if phase0p5_exists:
        log("  Found Phase 0.5 convergence report; using MPB settings validated there.")

    mpb_overrides = run_root_with_result(_load_phase0p5_overrides, cdir, config)
    mpb_config = dict(config)
    if mpb_overrides:
        mpb_config['phase1_resolution'] = mpb_overrides['resolution']
        mpb_config['phase1_dk'] = mpb_overrides['dk']
        score_txt = mpb_overrides.get('score')
        if score_txt is not None and math.isfinite(score_txt):
            log(f"  Phase 0.5 → resolution={mpb_overrides['resolution']:.0f}, "
                f"Δk={mpb_overrides['dk']:.4f} (score {score_txt:.2e})")
        else:
            log(f"  Phase 0.5 → resolution={mpb_overrides['resolution']:.0f}, "
                f"Δk={mpb_overrides['dk']:.4f}")
    else:
        log(f"  Using MPB settings from config: resolution={mpb_config.get('phase1_resolution', 'n/a')}, "
            f"Δk={mpb_config.get('phase1_dk', 'n/a')}")
    
    # === 1. Build moiré metadata and basis matrices ===
    moire_meta = ensure_moire_metadata(candidate_params, config)
    B_mono = moire_meta['B_mono']
    B_moire = moire_meta['B_moire']
    eta = moire_meta['eta']
    L_moire = moire_meta['moire_length']
    theta_rad = moire_meta['theta_rad']
    
    log(f"  Moiré scale factor η = {eta:.2f} (L_m = {L_moire:.3f})")
    
    # === 2. Build FRACTIONAL coordinate grid (V2 key change!) ===
    Ns1 = config.get('phase1_Ns1', config.get('phase1_Nx', 32))
    Ns2 = config.get('phase1_Ns2', config.get('phase1_Ny', 32))
    
    log(f"  Building fractional grid: {Ns1} x {Ns2} in (s₁, s₂) ∈ [0,1)²")
    s_grid = build_fractional_grid(Ns1, Ns2)
    
    # Also compute Cartesian grid for visualization (derived from s_grid)
    R_grid = fractional_to_cartesian(s_grid, B_moire)
    
    # Generate lattice visualization before heavy MPB computations
    log("  Rendering lattice visualization...")
    def _plot_lattice():
        try:
            plot_phase1_lattice_panels(cdir, candidate_params, moire_meta)
        except Exception as e:
            log(f"  WARNING: Lattice visualization failed: {e}")
    run_root_only(_plot_lattice)
    log("  Lattice visualization saved")
    
    # === 3. Compute registry map δ(s) using V2 formula ===
    tau_frac = np.array(config.get('tau', [0.0, 0.0]))
    
    log(f"  Computing registry map with V2 formula (NO η division)")
    log(f"    Stacking gauge τ = {tau_frac}")
    
    delta_frac = compute_registry_fractional_v2(
        s_grid, B_moire, B_mono, theta_rad, tau_frac
    )
    
    log(f"  Registry δ range: [{delta_frac.min():.4f}, {delta_frac.max():.4f}]")
    
    # Save candidate metadata
    run_root_only(lambda: save_json(candidate_params, cdir / "phase0_meta.json"))
    
    # === 4. Compute local band data at each grid point ===
    total_points = Ns1 * Ns2
    log(f"  Computing local band structure at {total_points} points...")
    
    # Prepare base parameters for MPB
    base_params = {
        'lattice_type': candidate_params['lattice_type'],
        'a': candidate_params['a'],
        'r_over_a': candidate_params['r_over_a'],
        'eps_bg': candidate_params['eps_bg'],
        'k0_x': candidate_params['k0_x'],
        'k0_y': candidate_params['k0_y'],
        'band_index': candidate_params['band_index'],
        'omega0': candidate_params['omega0'],
        'B_mono': B_mono,
        'a1_vec': moire_meta['a1_vec'],
        'a2_vec': moire_meta['a2_vec'],
    }
    
    # Resolve polarization for MPB (handle merged mode)
    polarization = candidate_params.get('polarization', 'te')
    if polarization == 'merged':
        # For merged candidates, use dominant_polarization or local_polarization
        polarization = candidate_params.get('dominant_polarization') or \
                       candidate_params.get('local_polarization') or 'te'
        log(f"  Resolved merged polarization to: {polarization}")
    mpb_config['polarization'] = polarization.lower()
    
    omega0_grid = np.zeros((Ns1, Ns2))
    vg_grid = np.zeros((Ns1, Ns2, 2))
    M_inv_grid = np.zeros((Ns1, Ns2, 2, 2))
    
    # Stencil data storage (V2 addition for re-processing capability)
    # For fd_order=4: 5x5 stencil; for fd_order=2: 3x3 stencil
    fd_order = int(mpb_config.get('phase1_fd_order', 4))
    n_stencil = 5 if fd_order == 4 else 3
    stencil_omega = np.zeros((Ns1, Ns2, n_stencil, n_stencil))
    stencil_offsets = np.array([-2, -1, 0, 1, 2] if fd_order == 4 else [-1, 0, 1])
    dk_actual = float(mpb_config.get('phase1_dk', 0.01))

    use_parallel = config.get('phase1_parallel', True)

    if MPI_ENABLED:
        if not use_parallel:
            log("  phase1_parallel=false but MPI ranks detected; continuing with MPI mode.")
        log(f"  Using MPI-distributed execution across {MPI_SIZE} ranks")
        omega0_local = np.zeros_like(omega0_grid)
        vg_local = np.zeros_like(vg_grid)
        M_inv_local = np.zeros_like(M_inv_grid)
        stencil_local = np.zeros_like(stencil_omega)

        progress_stride = max(1, int(config.get('phase1_registry_progress_stride', 32)))
        rank_verbose = bool(config.get('phase1_rank_logging', False))
        rank_verbose_stride = max(1, int(config.get('phase1_rank_logging_stride', 128)))

        registry_progress = None
        registry_processed_total = 0
        progress_start_time = time.time()
        if IS_ROOT:
            registry_progress = tqdm(
                total=total_points,
                desc="Registry Progress",
                unit="point",
                ncols=80
            )

        pending_since_update = 0
        rank_processed = 0
        rank_points_est = len(range(MPI_RANK, total_points, MPI_SIZE))

        for linear_idx in range(MPI_RANK, total_points, MPI_SIZE):
            i = linear_idx // Ns2
            j = linear_idx % Ns2
            args = (i, j, s_grid[i, j], delta_frac[i, j], base_params, mpb_config)
            _, _, omega0, vg, M_inv, stencil_data = compute_point_band_data(args)
            omega0_local[i, j] = omega0
            vg_local[i, j] = vg
            M_inv_local[i, j] = M_inv
            if stencil_data is not None:
                stencil_local[i, j] = stencil_dict_to_array(
                    stencil_data['omega_values'], fd_order
                )

            pending_since_update += 1
            rank_processed += 1

            if rank_verbose and (rank_processed % rank_verbose_stride == 0 or rank_processed == rank_points_est):
                log_rank(f"Processed {rank_processed}/{rank_points_est} points for candidate {cid}")

            if pending_since_update >= progress_stride:
                increment = pending_since_update
                pending_since_update = 0
                buffer = np.array(increment, dtype='i')
                MPI_COMM.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
                global_inc = int(buffer.item())
                if IS_ROOT:
                    registry_processed_total = update_registry_progress(
                        registry_progress, registry_processed_total, total_points,
                        progress_start_time, global_inc
                    )

        if pending_since_update > 0:
            buffer = np.array(pending_since_update, dtype='i')
            pending_since_update = 0
            MPI_COMM.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
            global_inc = int(buffer.item())
            if IS_ROOT:
                registry_processed_total = update_registry_progress(
                    registry_progress, registry_processed_total, total_points,
                    progress_start_time, global_inc
                )

        if registry_progress is not None:
            registry_progress.close()

        MPI_COMM.Allreduce(omega0_local, omega0_grid, op=MPI.SUM)
        MPI_COMM.Allreduce(vg_local, vg_grid, op=MPI.SUM)
        MPI_COMM.Allreduce(M_inv_local, M_inv_grid, op=MPI.SUM)
        MPI_COMM.Allreduce(stencil_local, stencil_omega, op=MPI.SUM)
    else:
        serial_desc = "  Computing bands"
        if use_parallel and total_points > 16:
            log("  phase1_parallel requested but MPI not available; running serial.")
        else:
            log("  Using serial execution")
        iterator = range(Ns1)
        if total_points > 16:
            iterator = tqdm(range(Ns1), desc=serial_desc, unit="row", ncols=80)
        for i in iterator:
            for j in range(Ns2):
                args = (i, j, s_grid[i, j], delta_frac[i, j], base_params, mpb_config)
                _, _, omega0, vg, M_inv, stencil_data = compute_point_band_data(args)
                omega0_grid[i, j] = omega0
                vg_grid[i, j] = vg
                M_inv_grid[i, j] = M_inv
                if stencil_data is not None:
                    stencil_omega[i, j] = stencil_dict_to_array(
                        stencil_data['omega_values'], fd_order
                    )

    log("  Completed local band calculations")
    
    # === 5. Compute potential V(s) ===
    omega_ref = choose_reference_frequency(omega0_grid, config)
    V_grid = omega0_grid - omega_ref
    
    log(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    log(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # === 6. Save to HDF5 (V2 format with s_grid as primary) ===
    h5_path = cdir / "phase1_band_data.h5"
    
    # Finite difference coefficients for stencil re-processing
    if fd_order == 4:
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        coeff_first = np.array([-0.5, 0, 0.5], dtype=float)
        coeff_second = np.array([1, -2, 1], dtype=float)
    
    def _write_hdf5():
        with h5py.File(h5_path, 'w') as hf:
            # V2: s_grid is the PRIMARY coordinate system
            hf.create_dataset("s_grid", data=s_grid, compression="gzip")
            hf.create_dataset("delta_frac", data=delta_frac, compression="gzip")
            
            # Field data
            hf.create_dataset("omega0", data=omega0_grid, compression="gzip")
            hf.create_dataset("vg", data=vg_grid, compression="gzip")
            hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
            hf.create_dataset("V", data=V_grid, compression="gzip")
            
            # Also store Cartesian grid for convenience (derived from s_grid)
            hf.create_dataset("R_grid", data=R_grid, compression="gzip")
            
            # === Stencil data for re-processing ===
            # This allows re-computing vg and M_inv with different FD settings
            # without re-running the expensive band calculations
            stencil_grp = hf.create_group("stencil")
            stencil_grp.create_dataset("omega", data=stencil_omega, compression="gzip")
            stencil_grp.create_dataset("offsets", data=stencil_offsets)
            stencil_grp.create_dataset("coeff_first", data=coeff_first)
            stencil_grp.create_dataset("coeff_second", data=coeff_second)
            stencil_grp.attrs["dk"] = dk_actual
            stencil_grp.attrs["fd_order"] = fd_order
            stencil_grp.attrs["n_stencil"] = n_stencil

            # Metadata
            hf.attrs["pipeline_version"] = "V2"
            hf.attrs["coordinate_system"] = "fractional"
            hf.attrs["omega_ref"] = omega_ref
            hf.attrs["eta"] = eta
            hf.attrs["theta_deg"] = candidate_params.get('theta_deg', 0.0)
            hf.attrs["theta_rad"] = candidate_params.get('theta_rad', 0.0)
            hf.attrs["band_index"] = candidate_params['band_index']
            hf.attrs["k0_x"] = candidate_params['k0_x']
            hf.attrs["k0_y"] = candidate_params['k0_y']
            hf.attrs["lattice_type"] = candidate_params['lattice_type']
            hf.attrs["r_over_a"] = candidate_params['r_over_a']
            hf.attrs["eps_bg"] = candidate_params['eps_bg']
            hf.attrs["a"] = candidate_params['a']
            hf.attrs["moire_length"] = L_moire
            hf.attrs["Ns1"] = Ns1
            hf.attrs["Ns2"] = Ns2
            
            # Store basis matrices for coordinate transforms
            hf.create_dataset("B_moire", data=B_moire)
            hf.create_dataset("B_mono", data=B_mono)
            
            if 'phase1_resolution' in mpb_config:
                hf.attrs['phase1_resolution'] = mpb_config['phase1_resolution']
            if 'phase1_dk' in mpb_config:
                hf.attrs['phase1_dk'] = mpb_config['phase1_dk']
            if candidate_params.get('polarization'):
                hf.attrs['polarization'] = candidate_params['polarization']

    run_root_only(_write_hdf5)
    log(f"  Saved band data to {h5_path} (V2 format with stencil data)")

    # === 7. Diagnostics summary ===
    stats_path = cdir / "phase1_field_stats.json"
    if IS_ROOT:
        stats = summarize_phase1_fields(omega0_grid, vg_grid, M_inv_grid, V_grid)
        stats['pipeline_version'] = 'V2'
        stats['coordinate_system'] = 'fractional'
        save_json(stats, stats_path)
        if stats['omega0_std'] < config.get('phase1_variation_tol', 1e-5):
            log("  WARNING: omega0_grid variation below tolerance; check MPB setup or registry mapping")
        log(f"  Saved field stats to {stats_path}")
    mpi_barrier()
    
    # === 8. Generate visualizations ===
    log(f"  Generating visualizations...")

    def _plot_fields():
        try:
            # Use R_grid for visualization (Cartesian view)
            plot_phase1_fields(cdir, R_grid, V_grid, vg_grid, M_inv_grid, candidate_params, moire_meta)
        except Exception as e:
            log(f"  WARNING: Field visualization failed: {e}")

    def _plot_stencil():
        try:
            from common.plotting import plot_stencil_comparison
            plot_stencil_comparison(
                save_path=cdir / "phase1_stencil_comparison.png",
                stencil_omega=stencil_omega,
                stencil_offsets=stencil_offsets,
                dk=dk_actual,
                omega_ref=omega_ref,
                candidate_params=candidate_params,
                solver_name="MPB"
            )
        except Exception as e:
            log(f"  WARNING: Stencil visualization failed: {e}")

    run_root_only(_plot_fields)
    run_root_only(_plot_stencil)
    log(f"=== Completed Candidate {cid} (V2) ===\n")
    mpi_barrier()


# ==============================================================================
# Main Driver
# ==============================================================================

def run_phase1(run_dir, config_path):
    """
    Main Phase 1 driver (V2 Pipeline)
    
    Args:
        run_dir: Path to run directory containing phase0_candidates.csv
                 Can be 'auto' or 'latest' to find most recent run
        config_path: Path to configuration YAML file
    """
    log("\n" + "="*70)
    log("PHASE 1: Local Bloch Problems — V2 Pipeline (Fractional Coordinates)")
    log("="*70)
    
    # Report threading configuration
    omp_threads = os.environ.get('OMP_NUM_THREADS', 'auto')
    if MPI_ENABLED:
        log(f"Parallelization: MPI with {MPI_SIZE} ranks, {omp_threads} OpenMP thread(s) per rank")
    else:
        log(f"Parallelization: Serial with {omp_threads} OpenMP thread(s)")
    
    # Load configuration
    config = load_yaml(config_path)
    log(f"Loaded config from: {config_path}")
    
    candidate_filter = os.getenv('MSL_PHASE1_CANDIDATE_ID')
    if candidate_filter is None:
        candidate_filter = config.get('phase1_candidate_id')
    if candidate_filter is not None:
        try:
            candidate_filter = int(candidate_filter)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid candidate ID '{candidate_filter}'. Must be an integer.")
    
    # Handle automatic run directory selection
    if run_dir in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runsV2'))
        # Look for MPB V2 phase0 runs (exclude BLAZE runs: phase0_blaze_*)
        all_runs = sorted(runs_base.glob('phase0_*'))
        phase0_runs = [r for r in all_runs if 'blaze' not in r.name.lower()]
        if not phase0_runs:
            raise FileNotFoundError(
                f"No MPB phase0 run directories found in {runs_base}\n"
                f"  (Looking for phase0_v2_* or phase0_*, excluding phase0_blaze_*)\n"
                f"  Found BLAZE runs? Use: python phasesV2/phase1_local_bloch.py <explicit_path>"
            )
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest MPB Phase 0 run: {run_dir}")
    
    # Load Phase 0 candidates
    run_dir = Path(run_dir)
    candidates_path = run_dir / "phase0_candidates.csv"
    
    if not candidates_path.exists():
        raise FileNotFoundError(f"Phase 0 candidates not found: {candidates_path}")
    
    candidates = pd.read_csv(candidates_path)
    log(f"Loaded {len(candidates)} candidates from Phase 0")
    
    # Check for V2 marker
    if 'pipeline_version' in candidates.columns:
        versions = candidates['pipeline_version'].unique()
        log(f"  Pipeline version(s) in candidates: {versions}")
    
    # Select top K candidates
    if candidate_filter is not None:
        top_candidates = candidates[candidates['candidate_id'] == candidate_filter]
        if top_candidates.empty:
            raise ValueError(f"Candidate ID {candidate_filter} not found in {candidates_path}")
        log(f"\nProcessing candidate {candidate_filter}:")
    else:
        K_candidates = config.get('K_candidates', 5)
        top_candidates = candidates.head(K_candidates)
        log(f"\nProcessing top {len(top_candidates)} candidates:")

    if IS_ROOT:
        for idx, row in top_candidates.iterrows():
            pol = row.get('polarization', 'n/a')
            print(f"  {row['candidate_id']}: {row['lattice_type']}, "
                  f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
                  f"band={row['band_index']}, k={row['k_label']}, pol={pol}, "
                  f"S_total={row['S_total']:.4f}")
    
    # Process each candidate
    log(f"\n{'='*70}")
    candidate_iter = list(top_candidates.iterrows())
    if IS_ROOT:
        iterator = tqdm(candidate_iter, desc="Phase 1 Progress", unit="candidate", ncols=80)
    else:
        iterator = candidate_iter
    for idx, row in iterator:
        candidate_params = extract_candidate_parameters(row)
        try:
            process_candidate(candidate_params, config, run_dir)
        except Exception as e:
            if IS_ROOT:
                print(f"ERROR processing candidate {candidate_params['candidate_id']}: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 1 (V2) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 to assemble EA operators (V2)")


def get_default_config_path() -> Path:
    """Return the path to the default MPB Phase 1 V2 config."""
    return Path(__file__).parent.parent / "configsV2" / "phase1.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase1("auto", str(default_config))
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase1(sys.argv[1], str(default_config))
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase1(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python phasesV2/phase1_local_bloch.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
