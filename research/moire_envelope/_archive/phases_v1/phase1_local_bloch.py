"""
Phase 1: Local Bloch problems at frozen registry

For each top-K candidate from Phase 0, this phase:
1. Builds a moiré spatial grid R(i,j) over one moiré unit cell
2. Computes the local registry map δ(R) based on twist geometry
3. For each R point, runs a "frozen bilayer" MPB calculation:
   - Builds bilayer geometry with local stacking shift δ(R)
   - Computes band structure at k₀ and nearby k-points
   - Extracts: ω₀(R), v_g(R), M⁻¹(R) (inverse mass tensor)
4. Assembles potential V(R) = ω₀(R) - ω_ref
5. Outputs HDF5 files with all EA inputs for Phase 2

Based on README.md Section 3.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import time

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm is unavailable
    def tqdm(iterable, **_kwargs):
        return iterable

try:  # Allow Phase 1 to run even when mpi4py is not installed
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.moire_utils import build_R_grid, compute_registry_map, create_twisted_bilayer
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
        print(message)


def log_rank(message):
    """Print a message from any rank with a prefix (use sparingly)."""
    prefix = f"[Rank {MPI_RANK}] " if MPI_COMM is not None else ""
    print(f"{prefix}{message}")


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
    
    # Handle optional moiré parameters (may not exist for monolayer runs)
    if 'theta_deg' in row:
        params['theta_deg'] = float(row['theta_deg'])
        params['theta_rad'] = math.radians(params['theta_deg'])
    
    if 'G_magnitude' in row:
        params['G_magnitude'] = float(row['G_magnitude'])
    
    if 'moire_length' in row:
        params['moire_length'] = float(row['moire_length'])
    
    return params


def ensure_moire_metadata(candidate_params, config):
    """Ensure candidate dict contains twist + moiré info using Phase 0 data or config."""
    theta_deg = candidate_params.get('theta_deg')
    if theta_deg is None or (isinstance(theta_deg, float) and math.isnan(theta_deg)):
        theta_deg = config.get('default_theta_deg')
        if theta_deg is None:
            raise ValueError(
                "Candidate is missing theta_deg; specify 'default_theta_deg' in the Phase 1 "
                "config to define the moire twist applied to all Phase 0 monolayer candidates."
            )
    theta_deg = float(theta_deg)
    lattice_type = candidate_params['lattice_type']
    a = candidate_params['a']
    bilayer = create_twisted_bilayer(lattice_type, theta_deg, a)

    # Trust explicit values when provided, otherwise use Rust helper outputs
    candidate_params['theta_deg'] = theta_deg
    candidate_params['theta_rad'] = bilayer['theta_rad']

    moire_length = candidate_params.get('moire_length')
    if moire_length is None or (isinstance(moire_length, float) and math.isnan(moire_length)):
        moire_length = bilayer['moire_length']
    candidate_params['moire_length'] = float(moire_length)

    G_mag = candidate_params.get('G_magnitude')
    if G_mag is None or (isinstance(G_mag, float) and math.isnan(G_mag)):
        G_mag = bilayer['G_magnitude']
    candidate_params['G_magnitude'] = float(G_mag)

    return {
        'a1_vec': bilayer['a1'][:2],
        'a2_vec': bilayer['a2'][:2],
        'moire_length': candidate_params['moire_length'],
        'theta_rad': candidate_params['theta_rad'],
    }


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


def build_bilayer_geometry_at_delta(base_params, delta_frac, layer_separation=0.0):
    """
    Build bilayer photonic crystal geometry with stacking shift δ
    
    Args:
        base_params: Base lattice parameters (lattice_type, r_over_a, eps_bg, a)
        delta_frac: Fractional shift [2,] in lattice coordinates
        layer_separation: Vertical separation (for 3D, not used in 2D approximation)
        
    Returns:
        dict: Geometry parameters for MPB
    """
    # For 2D effective medium approximation, treat bilayer as single layer
    # with modified dielectric based on stacking-dependent overlap
    
    # Build base lattice
    geom = build_lattice(
        base_params['lattice_type'],
        base_params['r_over_a'],
        base_params['eps_bg'],
        base_params['a']
    )

    # MPB's geometry 'center' argument uses LATTICE (fractional) coordinates,
    # where center=(cx, cy) places the object at position cx*a1 + cy*a2 in Cartesian.
    #
    # Physical model: bottom layer fixed at origin, top layer shifted by δ.
    # This matches the BLAZE convention and envelope approximation derivation
    # where the base monolayer doesn't rotate.
    a1_vec = base_params['a1_vec']
    a2_vec = base_params['a2_vec']
    
    # Compute Cartesian delta for reference/storage
    delta_vec = delta_frac[0] * a1_vec + delta_frac[1] * a2_vec

    # Represent bilayer as two hole arrays:
    # - Bottom layer: fixed at origin
    # - Top layer: shifted by δ in fractional coordinates
    hole_translations = [
        np.array([0.0, 0.0, 0.0]),                          # Bottom layer: fixed at origin
        np.array([delta_frac[0], delta_frac[1], 0.0]),      # Top layer: shifted by δ
    ]

    geom['delta_frac'] = delta_frac
    geom['delta_physical'] = delta_vec
    geom['layer_separation'] = layer_separation
    geom['hole_translations'] = hole_translations
    
    return geom


def _load_phase0p5_overrides(cdir: Path, config) -> dict:
    """Load recommended MPB resolution/Δk from Phase 0.5 outputs if enabled."""
    if not bool(config.get('phase1_auto_from_phase0p5', True)):
        return {}
    rec_path = cdir / "phase0p5_recommended.json"
    if not rec_path.exists():
        return {}
    try:
        payload = load_json(rec_path)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"  WARNING: Failed to parse {rec_path}: {exc}")
        return {}
    selected = payload.get("selected") or {}
    resolution = selected.get("resolution")
    dk = selected.get("dk")
    if resolution is None or dk is None:
        print(f"  WARNING: {rec_path} missing 'resolution'/'dk'; ignoring recommendation.")
        return {}
    return {
        "resolution": float(resolution),
        "dk": float(dk),
        "score": float(selected.get("score", float("nan"))),
    }


def compute_point_band_data(args):
    """
    Compute band data at a single R grid point (for parallel execution)
    
    Args:
        args: Tuple of (i, j, R_vec, delta_frac, base_params, config)
        
    Returns:
        tuple: (i, j, omega0, vg, M_inv)
    """
    i, j, R_vec, delta_frac, base_params, config = args
    
    # Build geometry for this registry point
    geom = build_bilayer_geometry_at_delta(base_params, delta_frac)
    
    # Compute local band data at k₀
    k0 = np.array([base_params['k0_x'], base_params['k0_y'], 0.0])
    band_idx = base_params['band_index']
    
    # Run MPB at this frozen geometry
    try:
        omega0, vg, M_inv, _stencil = compute_local_band_at_registry(
            geom, k0, band_idx, config
        )
    except Exception as e:
        # If computation fails, use fallback values
        rank_txt = f"[Rank {MPI_RANK}] " if MPI_COMM is not None else ""
        print(f"{rank_txt}Warning: Failed at ({i},{j}): {e}")
        omega0 = base_params['omega0']
        vg = np.zeros(2)
        M_inv = np.eye(2)
    
    return (i, j, omega0, vg, M_inv)


def process_candidate(candidate_params, config, run_dir):
    """
    Process a single candidate through Phase 1
    
    Args:
        candidate_params: Dictionary with candidate parameters
        config: Configuration dictionary
        run_dir: Run directory path
    """
    cid = candidate_params['candidate_id']
    log(f"\n=== Processing Candidate {cid} ===")
    log(f"  Lattice: {candidate_params['lattice_type']}")
    log(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    log(f"  Band {candidate_params['band_index']} at k={candidate_params['k_label']}")
    if candidate_params.get('theta_deg') is not None:
        log(f"  Twist theta: {candidate_params['theta_deg']:.3f} deg")
    
    # Create candidate directory
    cdir = candidate_dir(run_dir, cid)
    run_root_only(lambda: cdir.mkdir(parents=True, exist_ok=True))
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
            log(
                f"  Phase 0.5 recommendation → resolution={mpb_overrides['resolution']:.0f},"
                f" Δk={mpb_overrides['dk']:.4f} (relative score {score_txt:.2e})"
            )
        else:
            log(
                f"  Phase 0.5 recommendation → resolution={mpb_overrides['resolution']:.0f},"
                f" Δk={mpb_overrides['dk']:.4f}"
            )
    else:
        log(
            "  Using MPB settings from config: resolution=%s, Δk=%s"
            % (
                mpb_config.get('phase1_resolution', 'n/a'),
                mpb_config.get('phase1_dk', 'n/a'),
            )
        )
    
    # === 1. Build moiré spatial grid ===
    Nx = config.get('phase1_Nx', 32)
    Ny = config.get('phase1_Ny', 32)
    moire_meta = ensure_moire_metadata(candidate_params, config)
    L_moire = moire_meta['moire_length']

    # Generate lattice visualization before heavy MPB computations so we fail fast if Pillow is missing
    log("  Rendering lattice visualization...")
    def _plot_lattice():
        plot_phase1_lattice_panels(cdir, candidate_params, moire_meta)
    run_root_only(_plot_lattice)
    log("  Lattice visualization saved")

    R_grid = build_R_grid(Nx, Ny, L_moire, center=True)
    log(f"  Built R grid: {Nx} x {Ny}, L_moiré = {L_moire:.3f}")
    
    # === 2. Compute registry map δ(R) ===
    a1 = moire_meta['a1_vec']
    a2 = moire_meta['a2_vec']
    theta = moire_meta['theta_rad']
    
    # Get twist parameters
    tau = config.get('tau', np.zeros(2))
    eta_config = config.get('eta')
    eta_config_auto = isinstance(eta_config, str) and eta_config.lower() == 'auto'
    if eta_config is None or eta_config_auto:
        eta_physical = candidate_params['a'] / L_moire
        eta_source = 'geometry (a/L_moiré)'
    else:
        eta_physical = float(eta_config)
        eta_source = 'config override'
    candidate_params['eta'] = eta_physical
    eta = eta_physical  # Backwards compatibility for downstream references
    log(f"  Small parameter η_physical = {eta_physical:.6f} [{eta_source}]")

    registry_eta_cfg = config.get('phase1_registry_eta')
    if registry_eta_cfg is None:
        eta_for_registry = 1.0
        registry_source = 'legacy default (1.0)'
    elif isinstance(registry_eta_cfg, str):
        key = registry_eta_cfg.lower()
        if key in {'auto', 'physical'}:
            eta_for_registry = eta_physical
            registry_source = 'physical η'
        else:
            eta_for_registry = float(registry_eta_cfg)
            registry_source = 'config override'
    else:
        eta_for_registry = float(registry_eta_cfg)
        registry_source = 'config override'
    log(f"  Using η_registry = {eta_for_registry:.6f} for δ(R) [{registry_source}]")
    run_root_only(lambda: save_json(candidate_params, cdir / "phase0_meta.json"))
    
    delta_grid = compute_registry_map(R_grid, a1, a2, theta, tau, eta_for_registry)
    log(f"  Computed registry map δ(R)")
    
    # === 3. Compute local band data at each R ===
    total_points = Nx * Ny
    log(f"  Computing local band structure at {total_points} points...")
    
    # Prepare base parameters
    base_params = {
        'lattice_type': candidate_params['lattice_type'],
        'a': candidate_params['a'],
        'r_over_a': candidate_params['r_over_a'],
        'eps_bg': candidate_params['eps_bg'],
        'k0_x': candidate_params['k0_x'],
        'k0_y': candidate_params['k0_y'],
        'band_index': candidate_params['band_index'],
        'omega0': candidate_params['omega0'],
        'a1_vec': a1,
        'a2_vec': a2,
    }
    
    omega0_grid = np.zeros((Nx, Ny))
    vg_grid = np.zeros((Nx, Ny, 2))
    M_inv_grid = np.zeros((Nx, Ny, 2, 2))

    use_parallel = config.get('phase1_parallel', True)

    if MPI_ENABLED:
        if not use_parallel:
            log("  phase1_parallel=false but MPI ranks detected; continuing with MPI mode to keep ranks busy.")
        log(f"  Using MPI-distributed execution across {MPI_SIZE} ranks")
        omega0_local = np.zeros_like(omega0_grid)
        vg_local = np.zeros_like(vg_grid)
        M_inv_local = np.zeros_like(M_inv_grid)

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
        # Compute how many points this rank is responsible for (for optional verbose hooks)
        rank_points_est = len(range(MPI_RANK, total_points, MPI_SIZE))

        for linear_idx in range(MPI_RANK, total_points, MPI_SIZE):
            i = linear_idx // Ny
            j = linear_idx % Ny
            args = (i, j, R_grid[i, j], delta_grid[i, j], base_params, mpb_config)
            _, _, omega0, vg, M_inv = compute_point_band_data(args)
            omega0_local[i, j] = omega0
            vg_local[i, j] = vg
            M_inv_local[i, j] = M_inv

            pending_since_update += 1
            rank_processed += 1

            if rank_verbose and (rank_processed % rank_verbose_stride == 0 or rank_processed == rank_points_est):
                log_rank(
                    f"Processed {rank_processed}/{rank_points_est} registry points for candidate {cid}"
                )

            if pending_since_update >= progress_stride:
                increment = pending_since_update
                pending_since_update = 0
                buffer = np.array(increment, dtype='i')
                MPI_COMM.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
                global_inc = int(buffer.item())
                if IS_ROOT:
                    registry_processed_total = update_registry_progress(
                        registry_progress,
                        registry_processed_total,
                        total_points,
                        progress_start_time,
                        global_inc
                    )

        if pending_since_update > 0:
            buffer = np.array(pending_since_update, dtype='i')
            pending_since_update = 0
            MPI_COMM.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
            global_inc = int(buffer.item())
            if IS_ROOT:
                registry_processed_total = update_registry_progress(
                    registry_progress,
                    registry_processed_total,
                    total_points,
                    progress_start_time,
                    global_inc
                )

        if registry_progress is not None:
            registry_progress.close()

        MPI_COMM.Allreduce(omega0_local, omega0_grid, op=MPI.SUM)
        MPI_COMM.Allreduce(vg_local, vg_grid, op=MPI.SUM)
        MPI_COMM.Allreduce(M_inv_local, M_inv_grid, op=MPI.SUM)
    else:
        serial_desc = "  Computing bands"
        if use_parallel and total_points > 16:
            log("  phase1_parallel requested but MPI not available; running serial execution.")
        else:
            log("  Using serial execution")
        iterator = range(Nx)
        if total_points > 16:
            iterator = tqdm(range(Nx), desc=serial_desc, unit="row", ncols=80)
        for i in iterator:
            for j in range(Ny):
                args = (i, j, R_grid[i, j], delta_grid[i, j], base_params, mpb_config)
                _, _, omega0, vg, M_inv = compute_point_band_data(args)
                omega0_grid[i, j] = omega0
                vg_grid[i, j] = vg
                M_inv_grid[i, j] = M_inv

    log("  Completed local band calculations")
    
    # === 4. Compute potential V(R) ===
    omega_ref = choose_reference_frequency(omega0_grid, config)
    V_grid = omega0_grid - omega_ref
    
    log(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    log(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # === 5. Save to HDF5 ===
    h5_path = cdir / "phase1_band_data.h5"
    def _write_hdf5():
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("R_grid", data=R_grid, compression="gzip")
            hf.create_dataset("delta_grid", data=delta_grid, compression="gzip")
            hf.create_dataset("omega0", data=omega0_grid, compression="gzip")
            hf.create_dataset("vg", data=vg_grid, compression="gzip")
            hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
            hf.create_dataset("V", data=V_grid, compression="gzip")

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
            hf.attrs["Nx"] = Nx
            hf.attrs["Ny"] = Ny
            if 'phase1_resolution' in mpb_config:
                hf.attrs['phase1_resolution'] = mpb_config['phase1_resolution']
            if 'phase1_dk' in mpb_config:
                hf.attrs['phase1_dk'] = mpb_config['phase1_dk']

    run_root_only(_write_hdf5)
    log(f"  Saved band data to {h5_path}")

    # === 6. Diagnostics summary ===
    stats_path = cdir / "phase1_field_stats.json"
    if IS_ROOT:
        stats = summarize_phase1_fields(omega0_grid, vg_grid, M_inv_grid, V_grid)
        save_json(stats, stats_path)
        if stats['omega0_std'] < config.get('phase1_variation_tol', 1e-5):
            log("  WARNING: omega0_grid variation below tolerance; check MPB setup or registry mapping")
        log(f"  Saved field stats to {stats_path}")
    mpi_barrier()
    
    # === 7. Generate visualizations ===
    log(f"  Generating visualizations...")

    def _plot_fields():
        plot_phase1_fields(cdir, R_grid, V_grid, vg_grid, M_inv_grid, candidate_params, moire_meta)

    run_root_only(_plot_fields)
    log(f"=== Completed Candidate {cid} ===\n")
    mpi_barrier()


def run_phase1(run_dir, config_path):
    """
    Main Phase 1 driver
    
    Args:
        run_dir: Path to run directory containing phase0_candidates.csv
                 Can be 'auto' or 'latest' to find most recent phase0_real_run
        config_path: Path to configuration YAML file
    """
    log("\n" + "="*70)
    log("PHASE 1: Local Bloch Problems at Frozen Registry")
    log("="*70)
    
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
        runs_base = Path(config.get('output_dir', 'runs'))
        phase0_runs = sorted(runs_base.glob('phase0_real_run_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_real_run_* directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest Phase 0 run: {run_dir}")
    
    # Load Phase 0 candidates
    run_dir = Path(run_dir)
    candidates_path = run_dir / "phase0_candidates.csv"
    
    if not candidates_path.exists():
        raise FileNotFoundError(f"Phase 0 candidates not found: {candidates_path}")
    
    candidates = pd.read_csv(candidates_path)
    log(f"Loaded {len(candidates)} candidates from Phase 0")
    
    # Select top K candidates
    if candidate_filter is not None:
        top_candidates = candidates[candidates['candidate_id'] == candidate_filter]
        if top_candidates.empty:
            raise ValueError(
                f"Candidate ID {candidate_filter} not found in {candidates_path}"
            )
        log(f"\nProcessing candidate {candidate_filter}:")
    else:
        K_candidates = config.get('K_candidates', 5)
        top_candidates = candidates.head(K_candidates)
        log(f"\nProcessing top {len(top_candidates)} candidates:")

    if IS_ROOT:
        for idx, row in top_candidates.iterrows():
            print(f"  {row['candidate_id']}: {row['lattice_type']}, "
                  f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
                  f"band={row['band_index']}, k={row['k_label']}, "
                  f"S_total={row['S_total']:.4f}")
    
    # Process each candidate with progress bar
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
    log("PHASE 1 COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 to assemble EA operators")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mamba run -n msl python phase1_local_bloch.py <run_dir|auto|latest> [config_path]")
        print("\nArguments:")
        print("  run_dir: Path to Phase 0 run directory, or 'auto'/'latest' for most recent")
        print("  config_path: Optional config file (default: configs/phase1_real_run.yaml)")
        print("\nExamples:")
        print("  mamba run -n msl python phase1_local_bloch.py auto")
        print("  mamba run -n msl python phase1_local_bloch.py latest configs/phase1_real_run.yaml")
        print("  mamba run -n msl python phase1_local_bloch.py runs/phase0_real_run_20241113_120000")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) >= 3 else "configs/phase1_real_run.yaml"
    
    run_phase1(run_dir, config_path)
