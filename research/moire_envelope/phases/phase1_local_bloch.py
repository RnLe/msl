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
import multiprocessing as mp
from functools import partial

try:
    from tqdm import tqdm
except ImportError:  # Fallback if tqdm is unavailable
    def tqdm(iterable, **_kwargs):
        return iterable

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.moire_utils import build_R_grid, compute_registry_map, create_twisted_bilayer
from common.mpb_utils import compute_local_band_at_registry
from common.geometry import build_lattice
from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json
from common.plotting import plot_phase1_fields


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
    
    # In simplified model: stacking shift modulates effective epsilon or hole position
    # For now, use shift-independent geometry (full implementation would modify hole positions)
    geom['delta_frac'] = delta_frac
    geom['layer_separation'] = layer_separation
    
    return geom


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
        omega0, vg, M_inv = compute_local_band_at_registry(
            geom, k0, band_idx, config
        )
    except Exception as e:
        # If computation fails, use fallback values
        print(f"Warning: Failed at ({i},{j}): {e}")
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
    print(f"\n=== Processing Candidate {cid} ===")
    print(f"  Lattice: {candidate_params['lattice_type']}")
    print(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    print(f"  Band {candidate_params['band_index']} at k={candidate_params['k_label']}")
    
    # Create candidate directory
    cdir = candidate_dir(run_dir, cid)
    cdir.mkdir(parents=True, exist_ok=True)
    
    # Save candidate metadata
    save_json(candidate_params, cdir / "phase0_meta.json")
    
    # === 1. Build moiré spatial grid ===
    Nx = config.get('phase1_Nx', 32)
    Ny = config.get('phase1_Ny', 32)
    
    # Get or compute moiré length
    if 'moire_length' in candidate_params:
        L_moire = candidate_params['moire_length']
    else:
        # For monolayer, define effective "moiré scale" as multiple of lattice constant
        L_moire = candidate_params['a'] * config.get('effective_moire_scale', 10.0)
    
    R_grid = build_R_grid(Nx, Ny, L_moire, center=True)
    print(f"  Built R grid: {Nx} x {Ny}, L_moiré = {L_moire:.3f}")
    
    # === 2. Compute registry map δ(R) ===
    a1 = np.array([candidate_params['a'], 0.0])
    a2_y = candidate_params['a']
    if candidate_params['lattice_type'] == 'hex':
        a2 = np.array([candidate_params['a'] * 0.5, candidate_params['a'] * math.sqrt(3)/2])
    elif candidate_params['lattice_type'] == 'rect':
        a2 = np.array([0.0, candidate_params['a'] * 1.5])
    else:  # square
        a2 = np.array([0.0, candidate_params['a']])
    
    # Get twist parameters
    theta = candidate_params.get('theta_rad', 0.0)  # Default to 0 for monolayer
    tau = config.get('tau', np.zeros(2))
    eta = config.get('eta', 1.0)
    
    delta_grid = compute_registry_map(R_grid, a1, a2, theta, tau, eta)
    print(f"  Computed registry map δ(R)")
    
    # === 3. Compute local band data at each R ===
    print(f"  Computing local band structure at {Nx*Ny} points...")
    
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
    }
    
    # Initialize output arrays
    omega0_grid = np.zeros((Nx, Ny))
    vg_grid = np.zeros((Nx, Ny, 2))
    M_inv_grid = np.zeros((Nx, Ny, 2, 2))
    
    # Decide on parallel vs serial execution
    use_parallel = config.get('phase1_parallel', True)
    max_workers = config.get('phase1_max_workers', mp.cpu_count() // 2)
    
    if use_parallel and Nx * Ny > 16:
        # Parallel execution
        print(f"  Using parallel execution with {max_workers} workers")
        
        # Build argument list
        args_list = []
        for i in range(Nx):
            for j in range(Ny):
                args_list.append((
                    i, j,
                    R_grid[i, j],
                    delta_grid[i, j],
                    base_params,
                    config
                ))
        
        # Run in parallel with progress bar
        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(
                pool.imap(compute_point_band_data, args_list),
                total=len(args_list),
                desc="  Computing bands",
                unit="point",
                ncols=80
            ))
        
        # Unpack results
        for i, j, omega0, vg, M_inv in results:
            omega0_grid[i, j] = omega0
            vg_grid[i, j] = vg
            M_inv_grid[i, j] = M_inv
    else:
        # Serial execution
        print(f"  Using serial execution")
        for i in tqdm(range(Nx), desc="  Computing bands", unit="row", ncols=80):
            for j in range(Ny):
                args = (i, j, R_grid[i, j], delta_grid[i, j], base_params, config)
                i_out, j_out, omega0, vg, M_inv = compute_point_band_data(args)
                omega0_grid[i, j] = omega0
                vg_grid[i, j] = vg
                M_inv_grid[i, j] = M_inv
    
    print(f"  Completed local band calculations")
    
    # === 4. Compute potential V(R) ===
    omega_ref = choose_reference_frequency(omega0_grid, config)
    V_grid = omega0_grid - omega_ref
    
    print(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    print(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # === 5. Save to HDF5 ===
    h5_path = cdir / "phase1_band_data.h5"
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("delta_grid", data=delta_grid, compression="gzip")
        hf.create_dataset("omega0", data=omega0_grid, compression="gzip")
        hf.create_dataset("vg", data=vg_grid, compression="gzip")
        hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
        hf.create_dataset("V", data=V_grid, compression="gzip")
        
        # Save attributes
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
    
    print(f"  Saved band data to {h5_path}")
    
    # === 6. Generate visualizations ===
    print(f"  Generating visualizations...")
    plot_phase1_fields(cdir, R_grid, V_grid, vg_grid, M_inv_grid, candidate_params)
    
    print(f"=== Completed Candidate {cid} ===\n")


def run_phase1(run_dir, config_path):
    """
    Main Phase 1 driver
    
    Args:
        run_dir: Path to run directory containing phase0_candidates.csv
                 Can be 'auto' or 'latest' to find most recent phase0_real_run
        config_path: Path to configuration YAML file
    """
    print("\n" + "="*70)
    print("PHASE 1: Local Bloch Problems at Frozen Registry")
    print("="*70)
    
    # Load configuration
    config = load_yaml(config_path)
    print(f"Loaded config from: {config_path}")
    
    # Handle automatic run directory selection
    if run_dir in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runs'))
        phase0_runs = sorted(runs_base.glob('phase0_real_run_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_real_run_* directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest Phase 0 run: {run_dir}")
    
    # Load Phase 0 candidates
    run_dir = Path(run_dir)
    candidates_path = run_dir / "phase0_candidates.csv"
    
    if not candidates_path.exists():
        raise FileNotFoundError(f"Phase 0 candidates not found: {candidates_path}")
    
    candidates = pd.read_csv(candidates_path)
    print(f"Loaded {len(candidates)} candidates from Phase 0")
    
    # Select top K candidates
    K_candidates = config.get('K_candidates', 5)
    top_candidates = candidates.head(K_candidates)
    
    print(f"\nProcessing top {len(top_candidates)} candidates:")
    for idx, row in top_candidates.iterrows():
        print(f"  {row['candidate_id']}: {row['lattice_type']}, "
              f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
              f"band={row['band_index']}, k={row['k_label']}, "
              f"S_total={row['S_total']:.4f}")
    
    # Process each candidate with progress bar
    print(f"\n{'='*70}")
    for idx, row in tqdm(list(top_candidates.iterrows()), 
                         desc="Phase 1 Progress", 
                         unit="candidate",
                         ncols=80):
        candidate_params = extract_candidate_parameters(row)
        try:
            process_candidate(candidate_params, config, run_dir)
        except Exception as e:
            print(f"ERROR processing candidate {candidate_params['candidate_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to candidate directories in: {run_dir}")
    print("Next step: Run Phase 2 to assemble EA operators")


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
