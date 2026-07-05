"""
Phase 1 (BLAZE): Local Bloch problems at frozen registry — V3 Multi-Band Pipeline

This is the V3 multi-band implementation of Phase 1 using BLAZE.
It extracts band data for MULTIPLE bands simultaneously to support the
multi-band envelope approximation theory.

V3 KEY FEATURES:
1. Extract N_bands frequencies ω_n(s) at each registry position
2. Compute group velocity v_n(s) and mass tensor M^(-1)_n(s) per band
3. Store eigenvector data for Berry connection computation in Phase 2
4. Track band subspace and extra bands for Born-Huang

DATA STRUCTURES (V3):
- omega: (Ns1, Ns2, N_bands) - frequencies for each band
- vg: (Ns1, Ns2, N_bands, 2) - group velocities per band
- M_inv: (Ns1, Ns2, N_bands, 2, 2) - mass tensors per band
- eigenvectors: stored for Berry connection (computed in Phase 2)

THEORY REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md
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
import ast

try:
    from blaze import BulkDriver
except ImportError:
    print("ERROR: blaze package not installed. Install with: pip install blaze2d")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json, load_json
from common.plotting import plot_phase1_fields_v2, plot_phase1_lattice_panels


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# V3 Fractional Coordinate Functions (same as V2)
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build the monolayer lattice basis matrix B = (a1 | a2)."""
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3)/2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """Compute the moiré lattice basis vectors: B_moire = (R(θ) - I)^{-1} @ B_mono"""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_theta = np.array([[c, -s], [s, c]])
    Delta_R = R_theta - np.eye(2)
    Delta_R_inv = np.linalg.inv(Delta_R)
    return Delta_R_inv @ B_mono


def build_fractional_grid(Ns1: int, Ns2: int) -> np.ndarray:
    """Build uniform grid in fractional coordinates (s1, s2) ∈ [0, 1)²."""
    s1 = np.arange(Ns1) / Ns1
    s2 = np.arange(Ns2) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    return np.stack([S1, S2], axis=-1)


def fractional_to_cartesian(s_grid: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transform fractional coordinates to Cartesian: R = B · s"""
    return np.einsum('ij,...j->...i', B, s_grid)


def compute_registry_fractional_v3(
    s_grid: np.ndarray, 
    B_moire: np.ndarray, 
    B_mono: np.ndarray, 
    theta_rad: float,
    tau_frac: np.ndarray
) -> np.ndarray:
    """Compute registry shift: δ(s) = s + τ (mod 1)"""
    delta_frac = s_grid + tau_frac
    delta_frac = np.mod(delta_frac, 1.0)
    return delta_frac


def compute_eta_physics(theta_rad: float) -> float:
    """Compute physics small parameter η = a / L_m ≈ 2 sin(θ/2)"""
    return 2 * np.sin(theta_rad / 2)


def compute_eta_geometric(theta_rad: float) -> float:
    """Compute geometric moiré scale factor η_geom = L_m / a"""
    return 1.0 / (2 * np.sin(theta_rad / 2))


# ==============================================================================
# V3 Multi-Band Candidate Extraction
# ==============================================================================

def extract_candidate_parameters_v3(row):
    """Extract relevant parameters from candidate row including V3 multi-band info."""
    merged_band_index = int(row['band_index'])
    
    polarization = row.get('polarization', None)
    if polarization == 'merged' and 'original_band_idx' in row:
        band_index = int(row['original_band_idx'])
    else:
        band_index = merged_band_index
    
    params = {
        'candidate_id': int(row['candidate_id']),
        'lattice_type': row['lattice_type'],
        'a': float(row['a']),
        'r_over_a': float(row['r_over_a']),
        'eps_bg': float(row['eps_bg']),
        'band_index': band_index,
        'merged_band_index': merged_band_index,
        'k_label': row['k_label'],
        'k0_x': float(row['k0_x']),
        'k0_y': float(row['k0_y']),
        'omega0': float(row['omega0']),
    }
    
    if 'polarization' in row:
        params['polarization'] = row['polarization']
    if 'dominant_polarization' in row:
        params['dominant_polarization'] = row['dominant_polarization']
    if 'local_polarization' in row:
        params['local_polarization'] = row['local_polarization']
    if 'theta_deg' in row:
        params['theta_deg'] = float(row['theta_deg'])
        params['theta_rad'] = math.radians(params['theta_deg'])
    
    # V3 multi-band info
    if 'n_subspace_bands' in row:
        params['n_subspace_bands'] = int(row['n_subspace_bands'])
    if 'subspace_bands' in row:
        # Parse string representation of list
        if isinstance(row['subspace_bands'], str):
            params['subspace_bands'] = ast.literal_eval(row['subspace_bands'])
        else:
            params['subspace_bands'] = row['subspace_bands']
    if 'all_bands' in row:
        if isinstance(row['all_bands'], str):
            params['all_bands'] = ast.literal_eval(row['all_bands'])
        else:
            params['all_bands'] = row['all_bands']
    if 'target_index_in_subspace' in row:
        params['target_index_in_subspace'] = int(row['target_index_in_subspace'])
    
    return params


def ensure_moire_metadata(candidate_params: dict, config: dict) -> dict:
    """Ensure candidate dict contains twist + moiré info."""
    theta_deg = candidate_params.get('theta_deg')
    if theta_deg is None or (isinstance(theta_deg, float) and math.isnan(theta_deg)):
        theta_deg = config.get('default_theta_deg')
        if theta_deg is None:
            raise ValueError("Candidate is missing theta_deg; specify 'default_theta_deg' in config")
    theta_deg = float(theta_deg)
    theta_rad = math.radians(theta_deg)
    
    lattice_type = candidate_params['lattice_type']
    a = candidate_params['a']
    
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    eta_geom = compute_eta_geometric(theta_rad)
    eta = compute_eta_physics(theta_rad)
    moire_length = eta_geom * a
    
    candidate_params['theta_deg'] = theta_deg
    candidate_params['theta_rad'] = theta_rad
    candidate_params['moire_length'] = moire_length
    candidate_params['eta'] = eta
    
    return {
        'B_mono': B_mono,
        'B_moire': B_moire,
        'eta': eta,
        'moire_length': moire_length,
        'theta_rad': theta_rad,
    }


# ==============================================================================
# BLAZE Configuration (V3 Multi-Band)
# ==============================================================================

def lattice_type_to_blaze(lattice_type: str) -> str:
    """Convert our lattice type names to BLAZE format."""
    mapping = {
        'hex': 'triangular', 'hexagonal': 'triangular', 'triangular': 'triangular',
        'square': 'square', 'sq': 'square',
    }
    return mapping.get(lattice_type.lower(), lattice_type.lower())


def generate_blaze_config_v3(candidate_params, config, n_registry_samples, temp_dir) -> tuple:
    """
    Generate BLAZE config for V3 multi-band sweep.
    
    Key V3 change: Request multiple bands (all_bands) instead of just one.
    """
    lattice_type = lattice_type_to_blaze(candidate_params['lattice_type'])
    r_over_a = candidate_params['r_over_a']
    eps_bg = candidate_params['eps_bg']
    resolution = config.get('blaze_resolution', 32)
    
    # V3: Get all bands needed (subspace + extra for Born-Huang)
    all_bands = candidate_params.get('all_bands', [candidate_params['band_index']])
    subspace_bands = candidate_params.get('subspace_bands', all_bands)
    
    # Number of bands to request from BLAZE
    max_band = max(all_bands) + 1
    num_bands = max_band
    
    log(f"    V3 Multi-band: requesting bands 0-{max_band-1} (subspace: {subspace_bands})")
    
    k0_x = candidate_params['k0_x']
    k0_y = candidate_params['k0_y']
    
    dk = config.get('blaze_dk', 0.01)
    fd_order = config.get('blaze_fd_order', 4)
    
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
    else:
        offsets = [-1, 0, 1]
    
    # Build snake pattern k-stencil
    ordered_offsets = []
    for row_idx, oy in enumerate(offsets):
        if row_idx % 2 == 0:
            row_offsets = [(ox, oy) for ox in offsets]
        else:
            row_offsets = [(ox, oy) for ox in reversed(offsets)]
        ordered_offsets.extend(row_offsets)
    
    k_points = []
    k_point_labels = {}
    for idx, (ox, oy) in enumerate(ordered_offsets):
        kx = k0_x + ox * dk
        ky = k0_y + oy * dk
        k_points.append(f"[{kx:.8f}, {ky:.8f}]")
        k_point_labels[(ox, oy)] = idx
    
    k_points_str = ", ".join(k_points)
    
    n_registry = n_registry_samples
    step = 1.0 / n_registry
    pos_min = 0.0
    pos_max = 1.0 - step
    
    polarization = candidate_params.get('polarization', config.get('blaze_polarization', 'TM'))
    if polarization == 'merged':
        polarization = candidate_params.get('local_polarization',
                         candidate_params.get('dominant_polarization', 'TM'))
    
    threads = config.get('blaze_threads', 16)
    
    # V3: Request ALL bands (not just one with selective output)
    config_content = f'''# BLAZE config for Phase 1 V3 - Multi-Band - Candidate {candidate_params['candidate_id']}

[bulk]
threads = {threads}
verbose = false
disable_band_tracking = true

[solver]
type = "maxwell"

[defaults]
eps_bg = {eps_bg}
resolution = {resolution}
polarization = "{polarization}"

[geometry]
eps_bg = {eps_bg}

[geometry.lattice]
type = "{lattice_type}"
a = 1.0

[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {r_over_a}
eps_inside = 1.0

[[geometry.atoms]]
pos = [{pos_min:.8f}, {pos_min:.8f}]
radius = {r_over_a}
eps_inside = 1.0

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

[path]
k_path = [{k_points_str}]

[eigensolver]
n_bands = {num_bands}
max_iter = 200
tol = 1e-6

[dielectric.smoothing]
mesh_size = 3

[[sweeps]]
parameter = "atom1.pos_x"
min = {pos_min:.8f}
max = {pos_max:.8f}
step = {step:.8f}

[[sweeps]]
parameter = "atom1.pos_y"
min = {pos_min:.8f}
max = {pos_max:.8f}
step = {step:.8f}

[output]
mode = "full"
'''
    
    config_path = Path(temp_dir) / "blaze_config_v3.toml"
    config_path.write_text(config_content)
    
    save_json({str(k): v for k, v in k_point_labels.items()}, Path(temp_dir) / "k_labels.json")
    
    return config_path, k_point_labels, n_registry, all_bands, subspace_bands


def extract_multiband_data_from_blaze_v3(
    results, 
    candidate_params, 
    config, 
    delta_frac_grid, 
    k_point_labels, 
    n_registry_samples,
    all_bands,
    subspace_bands,
):
    """
    Extract multi-band data from BLAZE results (V3).
    
    Returns:
        omega_grid: (Ns1, Ns2, N_subspace) frequencies for subspace bands
        vg_grid: (Ns1, Ns2, N_subspace, 2) group velocities per band
        M_inv_grid: (Ns1, Ns2, N_subspace, 2, 2) mass tensors per band
        stencil_info: dict with raw stencil data for all bands (for Berry connection)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    Ns1, Ns2 = delta_frac_grid.shape[:2]
    k0_x = candidate_params['k0_x']
    k0_y = candidate_params['k0_y']
    dk = config.get('blaze_dk', 0.01)
    fd_order = config.get('blaze_fd_order', 4)
    
    n_registry = n_registry_samples
    N_all = len(all_bands)
    N_subspace = len(subspace_bands)
    
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        offsets = [-1, 0, 1]
        coeff_first = np.array([-0.5, 0, 0.5])
        coeff_second = np.array([1, -2, 1])
    
    n_stencil = len(offsets)
    
    # Initialize registry grids for ALL bands (for Berry connection later)
    # Shape: (n_registry, n_registry, N_all)
    registry_grid_omega0 = np.full((n_registry, n_registry, N_all), np.nan)
    registry_grid_vg_x = np.full((n_registry, n_registry, N_all), np.nan)
    registry_grid_vg_y = np.full((n_registry, n_registry, N_all), np.nan)
    registry_grid_d2_xx = np.full((n_registry, n_registry, N_all), np.nan)
    registry_grid_d2_yy = np.full((n_registry, n_registry, N_all), np.nan)
    registry_grid_d2_xy = np.full((n_registry, n_registry, N_all), np.nan)
    
    # Raw stencil: (n_registry, n_registry, N_all, n_stencil, n_stencil)
    stencil_omega = np.full((n_registry, n_registry, N_all, n_stencil, n_stencil), np.nan)
    
    step = 1.0 / n_registry
    tol = dk * 0.01
    
    for r in results:
        sv = r.get('sweep_values', {})
        dx = sv.get('atom1.pos_x')
        dy = sv.get('atom1.pos_y')
        
        if dx is None or dy is None:
            atoms = r.get('params', {}).get('atoms', [])
            if len(atoms) >= 2:
                pos = atoms[1].get('pos', [0, 0])
                dx, dy = pos[0], pos[1]
            else:
                continue
        
        ix = int(round(dx / step))
        iy = int(round(dy / step))
        if ix >= n_registry:
            ix = n_registry - 1
        if iy >= n_registry:
            iy = n_registry - 1
        
        bands = r['bands']  # bands[k_idx][band_idx]
        k_path = r.get('k_path', [])
        
        # Process each band in all_bands
        for band_local_idx, band_global_idx in enumerate(all_bands):
            if band_global_idx >= len(bands[0]):
                continue
            
            # Extract omega at each k-point for this band
            omega_values = {}
            for kidx, (kx, ky) in enumerate(k_path):
                if kidx >= len(bands):
                    continue
                ox_float = (kx - k0_x) / dk
                oy_float = (ky - k0_y) / dk
                ox = round(ox_float)
                oy = round(oy_float)
                
                if ox in offsets and oy in offsets:
                    kx_expected = k0_x + ox * dk
                    ky_expected = k0_y + oy * dk
                    if abs(kx - kx_expected) < tol and abs(ky - ky_expected) < tol:
                        omega_values[(ox, oy)] = bands[kidx][band_global_idx]
            
            if (0, 0) not in omega_values:
                continue
            
            omega0 = omega_values[(0, 0)]
            registry_grid_omega0[ix, iy, band_local_idx] = omega0
            
            # Store raw stencil
            for i, ox in enumerate(offsets):
                for j, oy in enumerate(offsets):
                    stencil_omega[ix, iy, band_local_idx, i, j] = omega_values.get((ox, oy), np.nan)
            
            # Compute derivatives
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
            
            registry_grid_vg_x[ix, iy, band_local_idx] = vg_x
            registry_grid_vg_y[ix, iy, band_local_idx] = vg_y
            registry_grid_d2_xx[ix, iy, band_local_idx] = d2_xx
            registry_grid_d2_yy[ix, iy, band_local_idx] = d2_yy
            registry_grid_d2_xy[ix, iy, band_local_idx] = d2_xy
    
    # Check coverage
    valid_count = np.sum(~np.isnan(registry_grid_omega0[:, :, 0]))
    log(f"    Filled {valid_count}/{n_registry*n_registry} registry grid points")
    
    # Fill NaN with nearest neighbor
    if np.any(np.isnan(registry_grid_omega0)):
        from scipy.ndimage import distance_transform_edt
        for grid in [registry_grid_omega0, registry_grid_vg_x, registry_grid_vg_y,
                     registry_grid_d2_xx, registry_grid_d2_yy, registry_grid_d2_xy]:
            for band_idx in range(N_all):
                band_slice = grid[:, :, band_idx]
                mask = np.isnan(band_slice)
                if np.any(mask) and not np.all(mask):
                    _, indices = distance_transform_edt(mask, return_indices=True)
                    grid[:, :, band_idx][mask] = band_slice[tuple(indices[:, mask])]
    
    # Interpolate to output grid
    x_coords = np.linspace(0, 1 - step, n_registry)
    y_coords = np.linspace(0, 1 - step, n_registry)
    
    def make_periodic_interp(grid_2d):
        extended = np.zeros((n_registry + 1, n_registry + 1))
        extended[:n_registry, :n_registry] = grid_2d
        extended[n_registry, :n_registry] = grid_2d[0, :]
        extended[:n_registry, n_registry] = grid_2d[:, 0]
        extended[n_registry, n_registry] = grid_2d[0, 0]
        x_ext = np.append(x_coords, 1.0)
        y_ext = np.append(y_coords, 1.0)
        return RegularGridInterpolator((x_ext, y_ext), extended, 
                                       method='linear', bounds_error=False, fill_value=None)
    
    # Map subspace band indices
    subspace_to_all = [all_bands.index(b) for b in subspace_bands]
    
    # Output grids for SUBSPACE bands only
    omega_grid = np.zeros((Ns1, Ns2, N_subspace))
    vg_grid = np.zeros((Ns1, Ns2, N_subspace, 2))
    M_inv_grid = np.zeros((Ns1, Ns2, N_subspace, 2, 2))
    
    delta_frac_x = delta_frac_grid[:, :, 0]
    delta_frac_y = delta_frac_grid[:, :, 1]
    query_x = np.mod(delta_frac_x + 0.5, 1.0)
    query_y = np.mod(delta_frac_y + 0.5, 1.0)
    query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
    
    for sub_idx, all_idx in enumerate(subspace_to_all):
        interp_omega0 = make_periodic_interp(registry_grid_omega0[:, :, all_idx])
        interp_vg_x = make_periodic_interp(registry_grid_vg_x[:, :, all_idx])
        interp_vg_y = make_periodic_interp(registry_grid_vg_y[:, :, all_idx])
        interp_d2_xx = make_periodic_interp(registry_grid_d2_xx[:, :, all_idx])
        interp_d2_yy = make_periodic_interp(registry_grid_d2_yy[:, :, all_idx])
        interp_d2_xy = make_periodic_interp(registry_grid_d2_xy[:, :, all_idx])
        
        omega_grid[:, :, sub_idx] = interp_omega0(query_points).reshape(Ns1, Ns2)
        vg_grid[:, :, sub_idx, 0] = interp_vg_x(query_points).reshape(Ns1, Ns2)
        vg_grid[:, :, sub_idx, 1] = interp_vg_y(query_points).reshape(Ns1, Ns2)
        M_inv_grid[:, :, sub_idx, 0, 0] = interp_d2_xx(query_points).reshape(Ns1, Ns2)
        M_inv_grid[:, :, sub_idx, 0, 1] = interp_d2_xy(query_points).reshape(Ns1, Ns2)
        M_inv_grid[:, :, sub_idx, 1, 0] = interp_d2_xy(query_points).reshape(Ns1, Ns2)
        M_inv_grid[:, :, sub_idx, 1, 1] = interp_d2_yy(query_points).reshape(Ns1, Ns2)
    
    # Regularize mass tensors
    min_abs_eig = 1e-6
    for i in range(Ns1):
        for j in range(Ns2):
            for n in range(N_subspace):
                M = M_inv_grid[i, j, n]
                eigvals, eigvecs = np.linalg.eigh(M)
                mask = np.abs(eigvals) < min_abs_eig
                eigvals = np.where(mask, np.sign(eigvals) * min_abs_eig, eigvals)
                eigvals = np.where(eigvals == 0, min_abs_eig, eigvals)
                M_inv_grid[i, j, n] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    stencil_info = {
        'stencil_omega_all': stencil_omega,  # (n_reg, n_reg, N_all, n_stencil, n_stencil)
        'registry_omega_all': registry_grid_omega0,  # (n_reg, n_reg, N_all)
        'offsets': np.array(offsets),
        'dk': dk,
        'fd_order': fd_order,
        'n_registry': n_registry,
        'all_bands': all_bands,
        'subspace_bands': subspace_bands,
    }
    
    return omega_grid, vg_grid, M_inv_grid, stencil_info


def process_candidate_v3(candidate_params, config, run_dir):
    """Process a single candidate through Phase 1 V3 multi-band."""
    cid = candidate_params['candidate_id']
    log(f"\n=== Processing Candidate {cid} (BLAZE V3 Multi-Band) ===")
    log(f"  Lattice: {candidate_params['lattice_type']}")
    log(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    
    subspace_bands = candidate_params.get('subspace_bands', [candidate_params['band_index']])
    all_bands = candidate_params.get('all_bands', subspace_bands)
    log(f"  Target band: {candidate_params['band_index']} at k={candidate_params['k_label']}")
    log(f"  Subspace bands: {subspace_bands}")
    log(f"  All bands (including extra): {all_bands}")
    
    if candidate_params.get('polarization'):
        pol_str = candidate_params['polarization']
        if pol_str == 'merged':
            actual_pol = candidate_params.get('dominant_polarization', 'TM')
            log(f"  Polarization: {pol_str} → using {actual_pol}")
        else:
            log(f"  Polarization: {pol_str}")
    
    cdir = candidate_dir(run_dir, cid)
    cdir.mkdir(parents=True, exist_ok=True)
    
    Ns1 = config.get('phase1_Ns1', 128)
    Ns2 = config.get('phase1_Ns2', 128)
    
    moire_meta = ensure_moire_metadata(candidate_params, config)
    B_mono = moire_meta['B_mono']
    B_moire = moire_meta['B_moire']
    eta = moire_meta['eta']
    theta_rad = moire_meta['theta_rad']
    
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  Building fractional grid: {Ns1} × {Ns2}")
    
    s_grid = build_fractional_grid(Ns1, Ns2)
    R_grid = fractional_to_cartesian(s_grid, B_moire)
    
    tau_frac = np.array(config.get('tau', [0.0, 0.0]))
    delta_frac = compute_registry_fractional_v3(s_grid, B_moire, B_mono, theta_rad, tau_frac)
    
    save_json(candidate_params, cdir / "phase0_meta.json")
    
    n_registry_samples = config.get('blaze_registry_samples', 64)
    log(f"  BLAZE registry samples: {n_registry_samples} × {n_registry_samples}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path, k_point_labels, n_registry, all_bands_used, subspace_bands_used = generate_blaze_config_v3(
            candidate_params, config, n_registry_samples, temp_dir
        )
        
        driver = BulkDriver(str(config_path), threads=config.get('blaze_threads', 0))
        log(f"  BLAZE job count: {driver.job_count}")
        
        start_time = time.time()
        results, stats = driver.run_collect()
        
        elapsed = time.time() - start_time
        log(f"  BLAZE completed {len(results)} jobs in {elapsed:.2f}s")
    
    log(f"  Extracting multi-band data...")
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_blaze_v3(
        results, candidate_params, config, delta_frac, k_point_labels, n_registry,
        all_bands_used, subspace_bands_used
    )
    
    N_subspace = omega_grid.shape[2]
    log(f"  Extracted data for {N_subspace} subspace bands")
    
    # Reference frequency: use target band minimum
    target_idx = candidate_params.get('target_index_in_subspace', N_subspace // 2)
    omega_ref = choose_reference_frequency(omega_grid[:, :, target_idx], config)
    
    # Potential: V_n(s) = omega_n(s) - omega_ref for each band
    V_grid = omega_grid - omega_ref
    
    log(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    log(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # Save to HDF5 (V3 format)
    h5_path = cdir / "phase1_multiband_data.h5"
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("delta_frac", data=delta_frac, compression="gzip")
        
        # V3 multi-band data
        hf.create_dataset("omega", data=omega_grid, compression="gzip")  # (Ns1, Ns2, N_subspace)
        hf.create_dataset("vg", data=vg_grid, compression="gzip")  # (Ns1, Ns2, N_subspace, 2)
        hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")  # (Ns1, Ns2, N_subspace, 2, 2)
        hf.create_dataset("V", data=V_grid, compression="gzip")  # (Ns1, Ns2, N_subspace)
        
        # Raw stencil data for Phase 2 Berry connection
        stencil_grp = hf.create_group("stencil")
        stencil_grp.create_dataset("omega_all", data=stencil_info['stencil_omega_all'], compression="gzip")
        stencil_grp.create_dataset("registry_omega_all", data=stencil_info['registry_omega_all'], compression="gzip")
        stencil_grp.create_dataset("offsets", data=stencil_info['offsets'])
        stencil_grp.attrs["dk"] = stencil_info['dk']
        stencil_grp.attrs["fd_order"] = stencil_info['fd_order']
        stencil_grp.attrs["n_registry"] = stencil_info['n_registry']
        
        # Attributes
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["theta_deg"] = candidate_params.get('theta_deg', 0.0)
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["target_band_index"] = candidate_params['band_index']
        hf.attrs["target_index_in_subspace"] = target_idx
        hf.attrs["k0_x"] = candidate_params['k0_x']
        hf.attrs["k0_y"] = candidate_params['k0_y']
        hf.attrs["lattice_type"] = candidate_params['lattice_type']
        hf.attrs["r_over_a"] = candidate_params['r_over_a']
        hf.attrs["eps_bg"] = candidate_params['eps_bg']
        hf.attrs["a"] = candidate_params['a']
        hf.attrs["moire_length"] = moire_meta['moire_length']
        hf.attrs["Ns1"] = Ns1
        hf.attrs["Ns2"] = Ns2
        hf.attrs["N_subspace"] = N_subspace
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_mono"] = B_mono
        hf.attrs["subspace_bands"] = np.array(subspace_bands_used)
        hf.attrs["all_bands"] = np.array(all_bands_used)
        hf.attrs["solver"] = "blaze2d"
        hf.attrs["pipeline_version"] = "V3"
        hf.attrs["coordinate_system"] = "fractional"
    
    log(f"  Saved V3 multi-band data to {h5_path}")
    
    # Generate visualization for target band
    log(f"  Generating visualizations...")
    moire_meta_plot = {
        'moire_length': moire_meta['moire_length'],
        'theta_rad': theta_rad,
        'a1_vec': B_mono[:, 0],
        'a2_vec': B_mono[:, 1],
        'B_moire': B_moire,
    }
    try:
        plot_phase1_fields_v2(cdir, s_grid, V_grid[:, :, target_idx], 
                              vg_grid[:, :, target_idx, :], 
                              M_inv_grid[:, :, target_idx, :, :], 
                              B_moire, candidate_params, moire_meta_plot)
    except Exception as e:
        log(f"    WARNING: Visualization failed: {e}")
    
    log(f"=== Completed Candidate {cid} ===")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_phase1_v3(run_dir, config_path):
    """Main Phase 1 V3 driver using BLAZE."""
    log("\n" + "="*70)
    log("PHASE 1 V3 (BLAZE): Multi-Band Local Bloch Problems")
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
    
    if run_dir in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runsV3'))
        phase0_runs = sorted(runs_base.glob('phase0_blaze_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No BLAZE phase0 run directories found in {runs_base}")
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
        pol = row.get('polarization', 'TM')
        if pol == 'merged':
            actual_pol = row.get('local_polarization', row.get('dominant_polarization', '?'))
            pol_display = f"merged→{actual_pol}"
        else:
            pol_display = pol
        print(f"  {row['candidate_id']}: {row['lattice_type']}/{pol_display}, "
              f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
              f"band={row['band_index']}, k={row['k_label']}")
    
    log(f"\n{'='*70}")
    for idx, row in top_candidates.iterrows():
        candidate_params = extract_candidate_parameters_v3(row)
        try:
            process_candidate_v3(candidate_params, config, run_dir)
        except Exception as e:
            print(f"ERROR processing candidate {candidate_params['candidate_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 1 V3 (BLAZE) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 V3 for Berry connection and multi-band prep")


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase1_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase1_v3("auto", str(default_config))
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        
        try:
            candidate_id = int(arg)
            log(f"Using default config: {default_config}")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1_v3("auto", str(default_config))
        except ValueError:
            log(f"Using default config: {default_config}")
            run_phase1_v3(arg, str(default_config))
    elif len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        try:
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1_v3(arg2, str(default_config))
        except ValueError:
            run_phase1_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python blaze_phasesV3/phase1_blaze_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
