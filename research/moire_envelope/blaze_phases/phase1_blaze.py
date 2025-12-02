"""
Phase 1 (BLAZE): Local Bloch problems at frozen registry using BLAZE solver

This is a high-performance rewrite of Phase 1 using the blaze2d package.
Instead of running MPB sequentially for each registry point, we leverage
BLAZE's bulk driver to sweep over atom positions in parallel.

Key differences from MPB-based Phase 1:
1. Uses 2-atom basis: Layer 1 (fixed) + Layer 2 (position swept)
2. Position sweep simulates the registry map δ(R) 
3. All registry points computed in a single parallel batch
4. ~100-250x faster than MPB

Outputs are compatible with Phase 2 (same HDF5 format).
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import time
import tempfile
import shutil

try:
    from blaze import BulkDriver
except ImportError:
    print("ERROR: blaze2d package not installed. Install with: pip install blaze2d")
    sys.exit(1)

# Import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.moire_utils import build_R_grid, compute_registry_map, create_twisted_bilayer
from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json, load_json
from common.plotting import plot_phase1_fields, plot_phase1_lattice_panels


def log(message):
    """Print message with flush."""
    print(message, flush=True)


def extract_candidate_parameters(row):
    """Extract relevant parameters from candidate row."""
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
    
    # Polarization - read from candidate if available, else default to TM
    if 'polarization' in row:
        params['polarization'] = row['polarization']
    
    if 'theta_deg' in row:
        params['theta_deg'] = float(row['theta_deg'])
        params['theta_rad'] = math.radians(params['theta_deg'])
    if 'G_magnitude' in row:
        params['G_magnitude'] = float(row['G_magnitude'])
    if 'moire_length' in row:
        params['moire_length'] = float(row['moire_length'])
    
    return params


def ensure_moire_metadata(candidate_params, config):
    """Ensure candidate dict contains twist + moiré info."""
    theta_deg = candidate_params.get('theta_deg')
    if theta_deg is None or (isinstance(theta_deg, float) and math.isnan(theta_deg)):
        theta_deg = config.get('default_theta_deg')
        if theta_deg is None:
            raise ValueError(
                "Candidate is missing theta_deg; specify 'default_theta_deg' in config"
            )
    theta_deg = float(theta_deg)
    lattice_type = candidate_params['lattice_type']
    a = candidate_params['a']
    bilayer = create_twisted_bilayer(lattice_type, theta_deg, a)

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
    """Compute diagnostic statistics for Phase 1 outputs."""
    vg_norm = np.linalg.norm(vg_grid, axis=-1)
    omega0_flat = omega0_grid.ravel()
    V_flat = V_grid.ravel()
    vg_flat = vg_norm.ravel()
    M_flat = M_inv_grid.reshape(-1, 2, 2)
    eigvals = np.linalg.eigvalsh(M_flat)

    return {
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


def lattice_type_to_blaze(lattice_type: str) -> str:
    """Convert our lattice type names to BLAZE format."""
    mapping = {
        'hex': 'triangular',
        'hexagonal': 'triangular',
        'triangular': 'triangular',
        'square': 'square',
        'sq': 'square',
    }
    return mapping.get(lattice_type.lower(), lattice_type.lower())


def generate_blaze_config(candidate_params, config, delta_grid, temp_dir) -> tuple:
    """
    Generate a BLAZE TOML config for sweeping over registry positions.
    
    The key insight: we model the bilayer as two atoms in the unit cell.
    - Atom 0 (Layer 1): Fixed at center (0.5, 0.5)
    - Atom 1 (Layer 2): Position swept according to delta_grid
    
    The registry shift δ is in fractional coordinates of the primitive cell.
    We reduce δ to [0, 1) and convert to atom position.
    
    For the k-path, we use a minimal stencil around k₀ for finite differences.
    
    IMPORTANT: k₀ from Phase 0 is in MPB's fractional reciprocal coords (60° convention).
    BLAZE uses 120° convention, so we must transform k₀ before use.
    
    Two modes:
    - Interpolated (default): Sample uniform grid in registry space, interpolate to actual points
    - Direct (blaze_direct_sampling=True): Sample every unique delta from delta_grid
    
    Returns:
        (config_path, k_point_labels, direct_mode_info)
        direct_mode_info is None for interpolated mode, or dict with mapping info for direct mode
    """
    lattice_type = lattice_type_to_blaze(candidate_params['lattice_type'])
    lattice_type_lower = candidate_params['lattice_type'].lower()
    r_over_a = candidate_params['r_over_a']
    eps_bg = candidate_params['eps_bg']
    resolution = config.get('blaze_resolution', config.get('phase1_resolution', 32))
    band_index = candidate_params['band_index']
    
    # We only need band_index + 1 bands (0-indexed, so band 1 needs 2 bands)
    num_bands = band_index + 1
    
    # k₀ from candidate (in MPB's fractional reciprocal coords for hex lattice)
    k0_x_mpb = candidate_params['k0_x']
    k0_y_mpb = candidate_params['k0_y']
    
    # Transform k₀ from MPB fractional reciprocal coords to BLAZE fractional reciprocal coords
    # 
    # The reciprocal lattice transformation is T_recip = (T_real^{-1})^T
    # where T_real = [[1, 1], [0, 1]] transforms MPB real-space fractional to BLAZE.
    # 
    # T_real^{-1} = [[1, -1], [0, 1]]
    # T_recip = (T_real^{-1})^T = [[1, 0], [-1, 1]]
    #
    # So: k_blaze = T_recip @ k_mpb
    #     k_blaze_x = k_mpb_x
    #     k_blaze_y = -k_mpb_x + k_mpb_y
    #
    if lattice_type_lower in ['hex', 'triangular']:
        k0_x = k0_x_mpb
        k0_y = -k0_x_mpb + k0_y_mpb
        log(f"  Transformed k₀: MPB ({k0_x_mpb:.6f}, {k0_y_mpb:.6f}) → BLAZE ({k0_x:.6f}, {k0_y:.6f})")
    else:
        k0_x = k0_x_mpb
        k0_y = k0_y_mpb
    
    # Finite difference step
    dk = config.get('blaze_dk', config.get('phase1_dk', 0.005))
    fd_order = config.get('blaze_fd_order', config.get('phase1_fd_order', 4))
    
    # Build k-point stencil around k₀ with warm-start-friendly ordering
    # For 4th order FD: need k0, k0±dk, k0±2dk in x and y
    # That's 5x5 = 25 points for the full grid (needed for d2_xy mixed derivative)
    #
    # Warm-start optimization: order k-points in a snake pattern so consecutive
    # solves are neighbors in k-space. Start from Γ and expand outward.
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
    else:
        offsets = [-1, 0, 1]
    
    # Build snake pattern: start at center, go right, step up, go left, etc.
    # This minimizes distance between consecutive k-points for warm start
    k_points = []
    k_point_labels = {}  # (ox, oy) -> index
    
    # Generate all offset pairs
    all_offsets = [(ox, oy) for oy in offsets for ox in offsets]
    
    # Sort by distance from origin, then snake within each "ring"
    # Group by max(|ox|, |oy|) to get concentric squares
    from itertools import groupby
    
    def ring_key(offset):
        return max(abs(offset[0]), abs(offset[1]))
    
    sorted_offsets = sorted(all_offsets, key=ring_key)
    
    # Within each ring, order for snake traversal
    ordered_offsets = []
    for ring, group in groupby(sorted_offsets, key=ring_key):
        ring_points = list(group)
        if ring == 0:
            # Center point
            ordered_offsets.extend(ring_points)
        else:
            # Sort ring points in clockwise order starting from right
            # For a square ring, go: right edge down, bottom edge left, left edge up, top edge right
            def angle_key(pt):
                import math
                return math.atan2(pt[1], pt[0])
            ring_points.sort(key=angle_key)
            ordered_offsets.extend(ring_points)
    
    # Build k-points in the optimized order
    for idx, (ox, oy) in enumerate(ordered_offsets):
        kx = k0_x + ox * dk
        ky = k0_y + oy * dk
        k_points.append(f"[{kx:.8f}, {ky:.8f}]")
        k_point_labels[(ox, oy)] = idx
    
    k_points_str = ", ".join(k_points)
    
    # Check if direct sampling mode is enabled
    direct_sampling = config.get('blaze_direct_sampling', False)
    
    # Lattice convention transformation for hex lattice
    lattice_type_lower = candidate_params['lattice_type'].lower()
    use_hex_transform = lattice_type_lower in ['hex', 'triangular']
    
    # Determine registry sampling resolution
    if direct_sampling:
        # DIRECT MODE: Use same resolution as spatial grid (Nx x Ny)
        # This means no interpolation - every grid point gets its own calculation
        Nx, Ny = delta_grid.shape[:2]
        n_registry_samples = max(Nx, Ny)  # Use the larger dimension for uniform sampling
        log(f"  Direct sampling mode: {n_registry_samples} x {n_registry_samples} = {n_registry_samples**2} registry samples")
        log(f"  (matches spatial grid resolution, no interpolation)")
    else:
        # INTERPOLATED MODE: Use coarse registry sampling (default 32)
        n_registry_samples = config.get('blaze_registry_samples', 32)
        log(f"  Interpolated mode: {n_registry_samples} x {n_registry_samples} = {n_registry_samples**2} registry samples")
    
    log(f"  K-stencil: {len(k_points)} points around k0=({k0_x:.4f}, {k0_y:.4f})")
    
    # Create uniform grid in [0, 1)
    registry_x = np.linspace(0, 1 - 1/n_registry_samples, n_registry_samples)
    registry_y = np.linspace(0, 1 - 1/n_registry_samples, n_registry_samples)
    
    # Sweep ranges
    dx_min, dx_max = float(registry_x.min()), float(registry_x.max())
    dy_min, dy_max = float(registry_y.min()), float(registry_y.max())
    dx_step = 1.0 / n_registry_samples
    dy_step = 1.0 / n_registry_samples
    
    # For atom position, we sweep [0, 1) in both directions
    pos_x_min = dx_min
    pos_x_max = dx_max
    pos_y_min = dy_min  
    pos_y_max = dy_max
    
    # Polarization - use candidate's polarization if available, else config, else default TM
    polarization = candidate_params.get('polarization', config.get('blaze_polarization', 'TM'))
    
    # Generate list of all k-indices (0 to n_k-1)
    k_indices = list(range(len(k_points)))
    k_indices_str = ", ".join(str(i) for i in k_indices)
    
    # We need the target band (1-indexed for output.selective)
    target_band = band_index + 1  # Convert 0-indexed to 1-indexed
    bands_str = str(target_band)
    
    # Default to 16 physical cores
    threads = config.get('blaze_threads', 16)
    
    # Single config template - sweep-based for both modes
    # (direct mode just uses finer step size)
    mode_label = "DIRECT" if direct_sampling else "INTERPOLATED"
    config_content = f'''# BLAZE config for Phase 1 - Candidate {candidate_params['candidate_id']}
# Auto-generated - do not edit ({mode_label} SAMPLING MODE)

# TOP LEVEL fields (must be before any [section] headers)
# FD stencil around k0 for Hessian computation
k_path = [{k_points_str}]
polarization = "{polarization}"

[bulk]
threads = {threads}
verbose = false

[geometry]
eps_bg = {eps_bg}

[geometry.lattice]
type = "{lattice_type}"
a = 1.0

# Layer 1 atom - fixed at center
[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {r_over_a}
eps_inside = 1.0

# Layer 2 atom - position swept (registry shift)
[[geometry.atoms]]
pos = [{pos_x_min:.8f}, {pos_y_min:.8f}]
radius = {r_over_a}
eps_inside = 1.0

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[eigensolver]
n_bands = {num_bands}
max_iter = 200
tol = 1e-6

[dielectric.smoothing]
mesh_size = 4

[ranges]
# First atom: fixed (no sweep)
[[ranges.atoms]]

# Second atom: sweep position for registry (fractional coords in [0,1))
[[ranges.atoms]]
pos_x = {{ min = {pos_x_min:.8f}, max = {pos_x_max:.8f}, step = {dx_step:.8f} }}
pos_y = {{ min = {pos_y_min:.8f}, max = {pos_y_max:.8f}, step = {dy_step:.8f} }}

[output]
mode = "selective"

[output.selective]
# All k-points in the FD stencil
k_indices = [{k_indices_str}]
# Only the target band (1-indexed)
bands = [{bands_str}]
'''
    
    config_path = Path(temp_dir) / "blaze_config.toml"
    config_path.write_text(config_content)
    
    # Save k_point_labels mapping for later use
    labels_path = Path(temp_dir) / "k_labels.json"
    save_json({str(k): v for k, v in k_point_labels.items()}, labels_path)
    
    # Return n_registry_samples so extraction knows what resolution was used
    return config_path, k_point_labels, n_registry_samples


def extract_band_data_from_blaze(results, candidate_params, config, delta_grid, k_point_labels, n_registry_samples=None):
    """
    Extract omega0, vg, M_inv from BLAZE results.
    
    BLAZE computes band data on a uniform registry grid [0,1) × [0,1).
    We interpolate to get values at the actual delta_grid points.
    
    When n_registry_samples matches the spatial grid size (direct mode),
    interpolation is minimal/exact. When it's smaller (interpolated mode),
    bilinear interpolation smooths the data.
    
    Args:
        results: List of BLAZE result dicts
        candidate_params: Candidate parameters
        config: Configuration
        delta_grid: (Nx, Ny, 2) array of registry shifts (fractional coords)
        k_point_labels: Dict mapping (ox, oy) offset tuples to k-point indices
        n_registry_samples: Number of registry samples used (from generate_blaze_config)
        
    Returns:
        omega0_grid, vg_grid, M_inv_grid
    """
    from scipy.interpolate import RegularGridInterpolator
    
    Nx, Ny = delta_grid.shape[:2]
    band_index = candidate_params['band_index']
    lattice_type = candidate_params['lattice_type']
    dk = config.get('blaze_dk', config.get('phase1_dk', 0.005))
    fd_order = config.get('blaze_fd_order', config.get('phase1_fd_order', 4))
    
    # Use provided n_registry_samples or fall back to config
    if n_registry_samples is None:
        n_registry_samples = config.get('blaze_registry_samples', 32)
    n_registry = n_registry_samples
    
    # BLAZE vs MPB lattice convention difference for hex/triangular:
    #
    # MPB uses 60° convention:  a1=[1,0], a2=[0.5, sqrt(3)/2]
    # BLAZE uses 120° convention: a1=[1,0], a2=[-0.5, sqrt(3)/2]
    #
    # For the SAME physical shift, the fractional coordinates differ!
    # Transformation: delta_blaze = T_real @ delta_mpb where T_real = [[1, 1], [0, 1]]
    #
    # Our delta_grid is in MPB convention (60°), so we need to convert
    # to BLAZE convention (120°) for the lookup.
    #
    # CRITICAL: The k-space coordinates and derivatives also need transformation!
    #
    # For reciprocal space (k-vectors):
    #   T_recip = (T_real^{-1})^T = [[1, 0], [-1, 1]]
    #   k_blaze_frac = T_recip @ k_mpb_frac
    #
    # For derivatives (transforming FROM BLAZE back TO MPB):
    #   T_recip^{-1} = T_real^T = [[1, 0], [1, 1]]
    #   vg_mpb = T_recip^{-1} @ vg_blaze
    #   H_mpb = T_recip^{-1} @ H_blaze @ (T_recip^{-1})^T
    #
    # Note: The k₀ is already transformed in generate_blaze_config(), so the
    # finite differences are computed in BLAZE k-space. We transform the 
    # resulting derivatives back to MPB k-space for output.
    
    if lattice_type in ['hex', 'triangular']:
        use_hex_transform = True
    else:
        use_hex_transform = False
    
    # Build offset list and FD coefficients
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        offsets = [-1, 0, 1]
        coeff_first = np.array([-0.5, 0, 0.5])
        coeff_second = np.array([1, -2, 1])
    
    # Build registry grid and interpolate (works for both modes, just different resolution)
    # Create arrays for the sampled registry grid
    registry_grid_omega0 = np.full((n_registry, n_registry), np.nan)
    registry_grid_vg_x = np.full((n_registry, n_registry), np.nan)
    registry_grid_vg_y = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_xx = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_yy = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_xy = np.full((n_registry, n_registry), np.nan)
    
    # Map BLAZE results to grid positions
    step = 1.0 / n_registry
    
    for r in results:
        atoms = r['params'].get('atoms', [])
        if len(atoms) < 2:
            continue
        
        pos = atoms[1]['pos']
        dx, dy = pos[0], pos[1]
        
        # Find grid indices (should be exact matches)
        ix = int(round(dx / step))
        iy = int(round(dy / step))
        
        if ix >= n_registry:
            ix = n_registry - 1
        if iy >= n_registry:
            iy = n_registry - 1
        
        bands = r['bands']  # bands[k_idx][band_idx]
        num_k = r['num_k_points']
        num_bands = r['num_bands']
        
        # With selective output, we only have 1 band (index 0)
        actual_band_idx = 0 if num_bands == 1 else min(band_index, num_bands - 1)
        
        # Extract omega values at each k-point
        omega_values = {}
        for (ox, oy), kidx in k_point_labels.items():
            if kidx < num_k and kidx < len(bands):
                omega_values[(ox, oy)] = bands[kidx][actual_band_idx]
        
        if (0, 0) not in omega_values:
            continue
        
        omega0 = omega_values[(0, 0)]
        registry_grid_omega0[ix, iy] = omega0
        
        # Compute derivatives using finite differences
        if fd_order == 4:
            vg_x = sum(coeff_first[idx] * omega_values.get((off, 0), omega0) 
                      for idx, off in enumerate(offsets)) / dk
            vg_y = sum(coeff_first[idx] * omega_values.get((0, off), omega0) 
                      for idx, off in enumerate(offsets)) / dk
            
            d2_xx = sum(coeff_second[idx] * omega_values.get((off, 0), omega0) 
                       for idx, off in enumerate(offsets)) / (dk ** 2)
            d2_yy = sum(coeff_second[idx] * omega_values.get((0, off), omega0) 
                       for idx, off in enumerate(offsets)) / (dk ** 2)
            
            d2_xy = 0.0
            for iox, ox in enumerate(offsets):
                for ioy, oy in enumerate(offsets):
                    d2_xy += coeff_first[iox] * coeff_first[ioy] * omega_values.get((ox, oy), omega0)
            d2_xy /= (dk ** 2)
        else:
            omega_xp = omega_values.get((1, 0), omega0)
            omega_xm = omega_values.get((-1, 0), omega0)
            omega_yp = omega_values.get((0, 1), omega0)
            omega_ym = omega_values.get((0, -1), omega0)
            omega_pp = omega_values.get((1, 1), omega0)
            omega_pm = omega_values.get((1, -1), omega0)
            omega_mp = omega_values.get((-1, 1), omega0)
            omega_mm = omega_values.get((-1, -1), omega0)
            
            vg_x = (omega_xp - omega_xm) / (2 * dk)
            vg_y = (omega_yp - omega_ym) / (2 * dk)
            
            d2_xx = (omega_xp - 2 * omega0 + omega_xm) / (dk ** 2)
            d2_yy = (omega_yp - 2 * omega0 + omega_ym) / (dk ** 2)
            d2_xy = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk ** 2)
        
        registry_grid_vg_x[ix, iy] = vg_x
        registry_grid_vg_y[ix, iy] = vg_y
        registry_grid_d2_xx[ix, iy] = d2_xx
        registry_grid_d2_yy[ix, iy] = d2_yy
        registry_grid_d2_xy[ix, iy] = d2_xy
    
    # Check coverage
    valid_count = np.sum(~np.isnan(registry_grid_omega0))
    log(f"  Filled {valid_count}/{n_registry*n_registry} registry grid points")

    # Fill any NaN values with nearest neighbor (shouldn't happen normally)
    if np.any(np.isnan(registry_grid_omega0)):
        from scipy.ndimage import distance_transform_edt
        
        for grid in [registry_grid_omega0, registry_grid_vg_x, registry_grid_vg_y,
                     registry_grid_d2_xx, registry_grid_d2_yy, registry_grid_d2_xy]:
            mask = np.isnan(grid)
            if np.any(mask) and not np.all(mask):
                # Find nearest valid value
                _, indices = distance_transform_edt(mask, return_indices=True)
                grid[mask] = grid[tuple(indices[:, mask])]
    
    # Create interpolators with periodic boundary conditions
    # Registry coordinates [0, 1)
    x_coords = np.linspace(0, 1 - step, n_registry)
    y_coords = np.linspace(0, 1 - step, n_registry)
    
    # For periodic interpolation, extend the grid
    def make_periodic_interp(grid):
        # Extend grid for periodicity: add one row/column at the end
        extended = np.zeros((n_registry + 1, n_registry + 1))
        extended[:n_registry, :n_registry] = grid
        extended[n_registry, :n_registry] = grid[0, :]  # Wrap x
        extended[:n_registry, n_registry] = grid[:, 0]  # Wrap y
        extended[n_registry, n_registry] = grid[0, 0]   # Corner
        
        x_ext = np.append(x_coords, 1.0)
        y_ext = np.append(y_coords, 1.0)
        
        return RegularGridInterpolator((x_ext, y_ext), extended, 
                                       method='linear', bounds_error=False, fill_value=None)
    
    interp_omega0 = make_periodic_interp(registry_grid_omega0)
    interp_vg_x = make_periodic_interp(registry_grid_vg_x)
    interp_vg_y = make_periodic_interp(registry_grid_vg_y)
    interp_d2_xx = make_periodic_interp(registry_grid_d2_xx)
    interp_d2_yy = make_periodic_interp(registry_grid_d2_yy)
    interp_d2_xy = make_periodic_interp(registry_grid_d2_xy)
    
    # Initialize output grids
    omega0_grid = np.zeros((Nx, Ny))
    vg_grid = np.zeros((Nx, Ny, 2))
    M_inv_grid = np.zeros((Nx, Ny, 2, 2))
    
    # Interpolate to actual delta_grid points
    log(f"  Interpolating to {Nx}x{Ny} = {Nx*Ny} grid points...")
    
    # The BLAZE geometry has:
    #   Atom 1 fixed at (0.5, 0.5) in BLAZE fractional coordinates (120° convention)
    #   Atom 2 swept from (0, 0) to (~1, ~1) in BLAZE fractional coordinates
    # So at grid position (gx, gy), the relative shift is (gx - 0.5, gy - 0.5)
    #
    # The delta_grid contains the registry shift in MPB convention (60°).
    # For hex lattice, we must transform to BLAZE convention (120°) before lookup.
    #
    # To find the corresponding grid position: 
    #   1. Transform delta_mpb to delta_blaze (for hex: delta_blaze = T @ delta_mpb)
    #   2. grid_pos = delta_blaze + 0.5
    
    if use_hex_transform:
        # Apply transformation: delta_blaze = T @ delta_mpb
        # T = [[1, 1], [0, 1]] means:
        #   delta_blaze_x = delta_mpb_x + delta_mpb_y
        #   delta_blaze_y = delta_mpb_y
        delta_blaze_x = delta_grid[:, :, 0] + delta_grid[:, :, 1]
        delta_blaze_y = delta_grid[:, :, 1]
    else:
        delta_blaze_x = delta_grid[:, :, 0]
        delta_blaze_y = delta_grid[:, :, 1]
    
    # Add 0.5 to convert from relative shift to grid position, then wrap to [0, 1)
    query_x = np.mod(delta_blaze_x + 0.5, 1.0)
    query_y = np.mod(delta_blaze_y + 0.5, 1.0)
    
    # Create query points
    query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
    
    # Interpolate all fields (these are in BLAZE k-space)
    omega0_flat = interp_omega0(query_points)
    vg_x_blaze_flat = interp_vg_x(query_points)
    vg_y_blaze_flat = interp_vg_y(query_points)
    d2_xx_blaze_flat = interp_d2_xx(query_points)
    d2_yy_blaze_flat = interp_d2_yy(query_points)
    d2_xy_blaze_flat = interp_d2_xy(query_points)
    
    omega0_grid = omega0_flat.reshape(Nx, Ny)
    
    # Transform derivatives from BLAZE k-space (fractional coords, 120° convention) 
    # to MPB k-space (fractional coords, 60° convention)
    #
    # The transformation matrix from MPB real-space fractional to BLAZE real-space fractional is:
    #   T_real = [[1, 1], [0, 1]]
    #
    # For reciprocal space, the transformation from MPB k-fractional to BLAZE k-fractional is:
    #   T_recip = (T_real^{-1})^T = [[1, 0], [-1, 1]]
    #
    # For derivatives (gradient and Hessian), we need the INVERSE transformation,
    # i.e., from BLAZE k-fractional to MPB k-fractional:
    #   T_recip^{-1} = T_real^T = [[1, 0], [1, 1]]
    #
    # Gradient transformation: vg_mpb = T_recip^{-1} @ vg_blaze
    #   vg_mpb_x = vg_blaze_x
    #   vg_mpb_y = vg_blaze_x + vg_blaze_y
    #
    # Hessian transformation: H_mpb = T_recip^{-1} @ H_blaze @ (T_recip^{-1})^T
    # With T_recip^{-1} = [[1, 0], [1, 1]] and (T_recip^{-1})^T = [[1, 1], [0, 1]]:
    #
    # Step 1: T_recip^{-1} @ H_blaze = [[1, 0], [1, 1]] @ [[a, b], [b, c]]
    #                                = [[a, b], [a+b, b+c]]
    #
    # Step 2: [[a, b], [a+b, b+c]] @ [[1, 1], [0, 1]]
    #       = [[a, a+b], [a+b, a+2b+c]]
    #
    # So: H_mpb[0,0] = d2_xx_blaze
    #     H_mpb[0,1] = H_mpb[1,0] = d2_xx_blaze + d2_xy_blaze
    #     H_mpb[1,1] = d2_xx_blaze + 2*d2_xy_blaze + d2_yy_blaze
    
    if use_hex_transform:
        log(f"  Applying k-space transformation (BLAZE 120° → MPB 60°)...")
        # Transform gradient: vg_mpb = T_recip^{-1} @ vg_blaze
        vg_x_flat = vg_x_blaze_flat
        vg_y_flat = vg_x_blaze_flat + vg_y_blaze_flat
        
        # Transform Hessian components
        d2_xx_flat = d2_xx_blaze_flat
        d2_xy_flat = d2_xx_blaze_flat + d2_xy_blaze_flat
        d2_yy_flat = d2_xx_blaze_flat + 2 * d2_xy_blaze_flat + d2_yy_blaze_flat
    else:
        # No transformation needed for square lattice
        vg_x_flat = vg_x_blaze_flat
        vg_y_flat = vg_y_blaze_flat
        d2_xx_flat = d2_xx_blaze_flat
        d2_xy_flat = d2_xy_blaze_flat
        d2_yy_flat = d2_yy_blaze_flat
    
    vg_grid[:, :, 0] = vg_x_flat.reshape(Nx, Ny)
    vg_grid[:, :, 1] = vg_y_flat.reshape(Nx, Ny)
    
    # Build M_inv tensor (Hessian of omega) - now in MPB k-space
    M_inv_grid[:, :, 0, 0] = d2_xx_flat.reshape(Nx, Ny)
    M_inv_grid[:, :, 0, 1] = d2_xy_flat.reshape(Nx, Ny)
    M_inv_grid[:, :, 1, 0] = d2_xy_flat.reshape(Nx, Ny)
    M_inv_grid[:, :, 1, 1] = d2_yy_flat.reshape(Nx, Ny)
    
    # Apply regularization to M_inv (same as original phase 1)
    # Preserve sign of eigenvalues (for band maxima), only regularize near-zero
    min_abs_eig = 1e-6
    for i in range(Nx):
        for j in range(Ny):
            M = M_inv_grid[i, j]
            eigvals, eigvecs = np.linalg.eigh(M)
            mask = np.abs(eigvals) < min_abs_eig
            eigvals = np.where(mask, np.sign(eigvals) * min_abs_eig, eigvals)
            eigvals = np.where(eigvals == 0, min_abs_eig, eigvals)
            M_inv_grid[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return omega0_grid, vg_grid, M_inv_grid


def process_candidate(candidate_params, config, run_dir):
    """Process a single candidate through Phase 1 using BLAZE."""
    cid = candidate_params['candidate_id']
    log(f"\n=== Processing Candidate {cid} (BLAZE) ===")
    log(f"  Lattice: {candidate_params['lattice_type']}")
    log(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    log(f"  Band {candidate_params['band_index']} at k={candidate_params['k_label']}")
    
    # Create candidate directory
    cdir = candidate_dir(run_dir, cid)
    cdir.mkdir(parents=True, exist_ok=True)
    
    # === 1. Build moiré spatial grid ===
    Nx = config.get('phase1_Nx', 32)
    Ny = config.get('phase1_Ny', 32)
    moire_meta = ensure_moire_metadata(candidate_params, config)
    L_moire = moire_meta['moire_length']
    
    # Generate lattice visualization
    log("  Rendering lattice visualization...")
    plot_phase1_lattice_panels(cdir, candidate_params, moire_meta)
    log("  Lattice visualization saved")
    
    R_grid = build_R_grid(Nx, Ny, L_moire, center=True)
    log(f"  Built R grid: {Nx} x {Ny}, L_moiré = {L_moire:.3f}")
    
    # === 2. Compute registry map δ(R) ===
    a1 = moire_meta['a1_vec']
    a2 = moire_meta['a2_vec']
    theta = moire_meta['theta_rad']
    
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
    eta = eta_physical
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
    
    save_json(candidate_params, cdir / "phase0_meta.json")
    
    delta_grid = compute_registry_map(R_grid, a1, a2, theta, tau, eta_for_registry)
    log(f"  Computed registry map δ(R)")
    
    # === 3. Run BLAZE for all registry points ===
    log(f"  Setting up BLAZE bulk driver...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path, k_point_labels, direct_mode_info = generate_blaze_config(candidate_params, config, delta_grid, temp_dir)
        
        driver = BulkDriver(str(config_path), threads=config.get('blaze_threads', 0))
        log(f"  BLAZE job count: {driver.job_count}")
        
        start_time = time.time()
        # Use run_collect which collects results in memory (returns (results, stats) tuple)
        results, stats = driver.run_collect()
        
        elapsed = time.time() - start_time
        log(f"  BLAZE completed {len(results)} jobs in {elapsed:.2f}s ({len(results)/elapsed:.1f} jobs/s)")
    
    # === 4. Extract band data from BLAZE results ===
    log(f"  Extracting band data from BLAZE results...")
    omega0_grid, vg_grid, M_inv_grid = extract_band_data_from_blaze(
        results, candidate_params, config, delta_grid, k_point_labels, direct_mode_info
    )
    
    log("  Completed local band calculations")
    
    # === 5. Compute potential V(R) ===
    omega_ref = choose_reference_frequency(omega0_grid, config)
    V_grid = omega0_grid - omega_ref
    
    log(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    log(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # === 6. Save to HDF5 ===
    h5_path = cdir / "phase1_band_data.h5"
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
        hf.attrs["solver"] = "blaze2d"
    
    log(f"  Saved band data to {h5_path}")
    
    # === 7. Diagnostics summary ===
    stats = summarize_phase1_fields(omega0_grid, vg_grid, M_inv_grid, V_grid)
    save_json(stats, cdir / "phase1_field_stats.json")
    variation_tol = float(config.get('phase1_variation_tol', 1e-5))
    if stats['omega0_std'] < variation_tol:
        log("  WARNING: omega0_grid variation below tolerance; check setup")
    log(f"  Saved field stats")
    
    # === 8. Generate visualizations ===
    log(f"  Generating visualizations...")
    plot_phase1_fields(cdir, R_grid, V_grid, vg_grid, M_inv_grid, candidate_params, moire_meta)
    
    log(f"=== Completed Candidate {cid} ===\n")


def run_phase1(run_dir, config_path):
    """Main Phase 1 driver using BLAZE."""
    log("\n" + "="*70)
    log("PHASE 1 (BLAZE): Local Bloch Problems at Frozen Registry")
    log("="*70)
    
    config = load_yaml(config_path)
    log(f"Loaded config from: {config_path}")
    
    candidate_filter = os.getenv('MSL_PHASE1_CANDIDATE_ID')
    if candidate_filter is None:
        candidate_filter = config.get('phase1_candidate_id')
    if candidate_filter is not None:
        try:
            candidate_filter = int(candidate_filter)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid candidate ID '{candidate_filter}'")
    
    # Handle automatic run directory selection
    if run_dir in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runs'))
        # Look for BLAZE phase0 runs first, then fall back to old phase0 runs
        phase0_runs = sorted(runs_base.glob('phase0_blaze_*'))
        if not phase0_runs:
            # Fall back to old naming convention for backwards compatibility
            phase0_runs = sorted(runs_base.glob('phase0_real_run_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_blaze_* or phase0_real_run_* directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest Phase 0 run: {run_dir}")
    
    run_dir = Path(run_dir)
    candidates_path = run_dir / "phase0_candidates.csv"
    
    if not candidates_path.exists():
        raise FileNotFoundError(f"Phase 0 candidates not found: {candidates_path}")
    
    candidates = pd.read_csv(candidates_path)
    log(f"Loaded {len(candidates)} candidates from Phase 0")
    
    if candidate_filter is not None:
        top_candidates = candidates[candidates['candidate_id'] == candidate_filter]
        if top_candidates.empty:
            raise ValueError(f"Candidate ID {candidate_filter} not found")
        log(f"\nProcessing candidate {candidate_filter}:")
    else:
        K_candidates = config.get('K_candidates', 5)
        top_candidates = candidates.head(K_candidates)
        log(f"\nProcessing top {len(top_candidates)} candidates:")
    
    for idx, row in top_candidates.iterrows():
        print(f"  {row['candidate_id']}: {row['lattice_type']}, "
              f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
              f"band={row['band_index']}, k={row['k_label']}, "
              f"S_total={row['S_total']:.4f}")
    
    log(f"\n{'='*70}")
    for idx, row in top_candidates.iterrows():
        candidate_params = extract_candidate_parameters(row)
        try:
            process_candidate(candidate_params, config, run_dir)
        except Exception as e:
            print(f"ERROR processing candidate {candidate_params['candidate_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 1 (BLAZE) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 to assemble EA operators")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase1_blaze.py <run_dir|auto|latest> [config_path]")
        print("\nArguments:")
        print("  run_dir: Path to Phase 0 run directory, or 'auto'/'latest' for most recent")
        print("  config_path: Optional config file (default: configs/phase1_blaze.yaml)")
        print("\nExamples:")
        print("  python phase1_blaze.py auto")
        print("  python phase1_blaze.py latest configs/phase1_blaze.yaml")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) >= 3 else "configs/phase1_blaze.yaml"
    
    run_phase1(run_dir, config_path)
