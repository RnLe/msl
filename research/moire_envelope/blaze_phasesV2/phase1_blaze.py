"""
Phase 1 (BLAZE): Local Bloch problems at frozen registry — V2 Pipeline

This is the high-performance BLAZE implementation of Phase 1 for the V2 pipeline
using FRACTIONAL COORDINATES. See README_V2.md for mathematical details.

KEY V2 CHANGES from legacy BLAZE Phase 1:
1. Sample in fractional coordinates (s_1, s_2) ∈ [0,1)² instead of Cartesian
2. Corrected registry formula: δ(R) = (R(θ) - I) · R + τ  (NO division by η!)
3. Store s_grid as primary grid with B_moire attribute for Cartesian transforms
4. Updated BLAZE 0.4.0 API: [[sweeps]] instead of [ranges]

The BLAZE approach models the bilayer as two atoms in the unit cell:
- Atom 0 (Layer 1): Fixed at center (0.5, 0.5)
- Atom 1 (Layer 2): Position swept according to registry δ

This is ~100-250x faster than MPB-based Phase 1.

Based on README_V2.md Section 4 and blaze2d 0.4.0 API.
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

try:
    from blaze import BulkDriver
except ImportError:
    print("ERROR: blaze package not installed. Install with: pip install blaze2d")
    sys.exit(1)

# Project root for config resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import common utilities
sys.path.insert(0, str(PROJECT_ROOT))
from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json, load_json
from common.plotting import plot_phase1_fields_v2, plot_phase1_lattice_panels


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# V2 Fractional Coordinate Functions
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """
    Build the monolayer lattice basis matrix B = (a1 | a2).
    
    Uses 60° convention for triangular/hex lattices (same as BLAZE).
    """
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], 
                              [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        # 60° convention: a1 = [1, 0], a2 = [0.5, √3/2]
        return a * np.array([[1.0, 0.5], 
                              [0.0, np.sqrt(3)/2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """
    Compute the moiré lattice basis vectors for a twisted bilayer.
    
    The correct formula is: B_moire = (R(θ) - I)^{-1} @ B_mono
    
    This ensures that traversing one moiré unit cell (s: 0→1) corresponds
    to the registry δ traversing one monolayer unit cell.
    
    For square lattices, this simplifies to η * B_mono (scaled + 90° rotated).
    For hexagonal lattices, the orientation differs from a simple scaling.
    
    The moiré length is still L_m = a / (2 sin(θ/2)) = η * a.
    """
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_theta = np.array([[c, -s], [s, c]])
    Delta_R = R_theta - np.eye(2)
    
    # (R(θ) - I) is invertible for θ ≠ 0 (det = 2 - 2cos(θ) = 4sin²(θ/2))
    Delta_R_inv = np.linalg.inv(Delta_R)
    return Delta_R_inv @ B_mono


def build_fractional_grid(Ns1: int, Ns2: int) -> np.ndarray:
    """
    Build uniform grid in fractional coordinates (s1, s2) ∈ [0, 1)².
    """
    s1 = np.arange(Ns1) / Ns1
    s2 = np.arange(Ns2) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    return np.stack([S1, S2], axis=-1)


def fractional_to_cartesian(s_grid: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Transform fractional coordinates to Cartesian: R = B · s"""
    return np.einsum('ij,...j->...i', B, s_grid)


def compute_registry_fractional_v2(
    s_grid: np.ndarray, 
    B_moire: np.ndarray, 
    B_mono: np.ndarray, 
    theta_rad: float,
    tau_frac: np.ndarray
) -> np.ndarray:
    """
    Compute registry shift in monolayer fractional coordinates.
    
    With the correct moiré basis B_moire = (R(θ) - I)^{-1} @ B_mono,
    the registry simplifies to: δ(s) = s + τ (mod 1)
    
    This is because:
      δ_frac = B_mono^{-1} @ (R(θ) - I) @ B_moire @ s
             = B_mono^{-1} @ (R(θ) - I) @ (R(θ) - I)^{-1} @ B_mono @ s
             = s
    
    The stacking gauge τ allows shifting the registry origin.
    """
    # With correct B_moire, delta = s + tau (simple!)
    delta_frac = s_grid + tau_frac
    
    # Wrap to [0, 1) with numerical tolerance
    delta_frac = np.mod(delta_frac, 1.0)
    
    return delta_frac


def compute_eta_geometric(theta_rad: float) -> float:
    """Compute geometric moiré scale factor η_geom = L_m / a ≈ 1 / (2 sin(θ/2)).
    
    This is the ratio of moiré period to monolayer lattice constant.
    Note: The moiré basis is computed via B_moire = (R(θ) - I)^{-1} @ B_mono,
    not as η_geom * B_mono, to ensure correct periodicity for all lattice types.
    However, η_geom still gives the correct moiré length: L_m = η_geom * a.
    """
    return 1.0 / (2 * np.sin(theta_rad / 2))


def compute_eta_physics(theta_rad: float) -> float:
    """Compute physics small parameter η = a / L_m ≈ 2 sin(θ/2) ≈ θ.
    
    This is the small parameter in the envelope Hamiltonian:
    H = V(s) + (η²/2) ∇_s · M̃⁻¹(s) ∇_s
    
    For θ = 1.1°: η ≈ 0.019
    """
    return 2 * np.sin(theta_rad / 2)


def compute_recommended_dk(
    L_m: float,
    a: float = 1.0,
    dk_fraction: float = 0.1,
    dk_min: float = 0.001,
    dk_max: float = 0.02,
) -> float:
    """
    Compute recommended k-step for finite difference derivatives.
    
    The parabolicity scale ℓ_k characterizes how far from k₀ the band
    remains approximately parabolic. For envelope theory to be valid,
    we need dk << ℓ_k.
    
    A reasonable heuristic: dk ~ 0.1 × (a / L_m) = 0.1 × η
    
    This ensures the k-step samples within the region where the effective
    mass is approximately constant, avoiding contamination from higher-order
    band structure features.
    
    For θ = 1.1°: η ≈ 0.019, so dk ~ 0.002 is appropriate.
    
    Args:
        L_m: Moiré length (in same units as a)
        a: Lattice constant (default 1.0 for normalized units)
        dk_fraction: Fraction of η to use as dk (default 0.1)
        dk_min: Minimum dk to prevent numerical noise issues
        dk_max: Maximum dk to stay in parabolic regime
    
    Returns:
        dk: Recommended finite difference step in fractional k-units
    """
    eta = a / L_m
    dk = dk_fraction * eta
    dk = max(dk_min, min(dk_max, dk))
    return dk


# ==============================================================================
# Candidate Parameter Extraction
# ==============================================================================

def extract_candidate_parameters(row):
    """Extract relevant parameters from candidate row."""
    # Get the merged band index (from combined TE+TM view)
    merged_band_index = int(row['band_index'])
    
    # For merged polarization mode, we need the original polarization-specific
    # band index. BLAZE runs in single-polarization mode (TE or TM), so it needs
    # the band index within that polarization, not the merged view.
    polarization = row.get('polarization', None)
    if polarization == 'merged' and 'original_band_idx' in row:
        # Use polarization-specific band index for BLAZE
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
        'merged_band_index': merged_band_index,  # Keep for reference
        'k_label': row['k_label'],
        'k0_x': float(row['k0_x']),
        'k0_y': float(row['k0_y']),
        'omega0': float(row['omega0']),
    }
    
    if 'polarization' in row:
        params['polarization'] = row['polarization']
    
    # For merged polarization mode, track the actual polarization to use
    if 'dominant_polarization' in row:
        params['dominant_polarization'] = row['dominant_polarization']
    if 'local_polarization' in row:
        params['local_polarization'] = row['local_polarization']
    
    if 'theta_deg' in row:
        params['theta_deg'] = float(row['theta_deg'])
        params['theta_rad'] = math.radians(params['theta_deg'])
    if 'G_magnitude' in row:
        params['G_magnitude'] = float(row['G_magnitude'])
    if 'moire_length' in row:
        params['moire_length'] = float(row['moire_length'])
    
    return params


def ensure_moire_metadata(candidate_params: dict, config: dict) -> dict:
    """Ensure candidate dict contains twist + moiré info."""
    theta_deg = candidate_params.get('theta_deg')
    if theta_deg is None or (isinstance(theta_deg, float) and math.isnan(theta_deg)):
        theta_deg = config.get('default_theta_deg')
        if theta_deg is None:
            raise ValueError(
                "Candidate is missing theta_deg; specify 'default_theta_deg' in config"
            )
    theta_deg = float(theta_deg)
    theta_rad = math.radians(theta_deg)
    
    lattice_type = candidate_params['lattice_type']
    a = candidate_params['a']
    
    # Build basis matrices
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    eta_geom = compute_eta_geometric(theta_rad)  # L_m / a (large, ~52)
    eta = compute_eta_physics(theta_rad)  # a / L_m (small, ~0.019) for Hamiltonian
    moire_length = eta_geom * a  # L_m = η_geom * a
    
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


# ==============================================================================
# BLAZE Configuration (v0.4.0 API)
# ==============================================================================

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


def generate_blaze_config_v2(candidate_params, config, n_registry_samples, temp_dir) -> tuple:
    """
    Generate a BLAZE TOML config for sweeping over registry positions.
    
    Uses BLAZE 0.4.0 API:
    - [[sweeps]] for parameter sweeps (not [ranges])
    - [path] preset for k-path specification
    - [output.selective] for filtered output
    
    Returns:
        (config_path, k_point_labels, n_registry_samples)
    """
    lattice_type = lattice_type_to_blaze(candidate_params['lattice_type'])
    r_over_a = candidate_params['r_over_a']
    eps_bg = candidate_params['eps_bg']
    resolution = config.get('blaze_resolution', config.get('phase1_resolution', 32))
    band_index = candidate_params['band_index']
    
    # We only need band_index + 1 bands
    num_bands = band_index + 1
    
    # k₀ from candidate (60° convention for both MPB and BLAZE)
    k0_x = candidate_params['k0_x']
    k0_y = candidate_params['k0_y']
    log(f"  k₀ = ({k0_x:.6f}, {k0_y:.6f}) [60° convention]")
    
    # Finite difference step and order
    dk_cfg = config.get('blaze_dk', config.get('phase1_dk', 'auto'))
    fd_order = config.get('blaze_fd_order', config.get('phase1_fd_order', 4))
    
    # Handle dynamic dk based on moiré length
    if dk_cfg == 'auto' or dk_cfg is None:
        L_m = candidate_params.get('moire_length', 52.0)  # Default for θ=1.1°
        a = candidate_params.get('a', 1.0)
        dk_fraction = config.get('blaze_dk_fraction', 0.1)
        dk = compute_recommended_dk(L_m, a, dk_fraction=dk_fraction)
        log(f"  Auto dk: {dk:.4f} (η = {a/L_m:.4f}, fraction = {dk_fraction})")
    else:
        dk = float(dk_cfg)
        L_m = candidate_params.get('moire_length', None)
        if L_m is not None:
            a = candidate_params.get('a', 1.0)
            eta = a / L_m
            log(f"  Fixed dk: {dk:.4f} (η = {eta:.4f}, dk/η = {dk/eta:.2f})")
        else:
            log(f"  Fixed dk: {dk:.4f}")
    
    # Build k-point stencil around k₀
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
    else:
        offsets = [-1, 0, 1]
    
    # Build k-points in CONNECTED SNAKE pattern for proper band tracking
    # BLAZE uses polar decomposition with subspace rotation tracking.
    # Jumping between distant k-points can confuse band tracking.
    # Use a meander/snake pattern: walk row-by-row, alternating direction.
    #
    # For 5x5 stencil (fd_order=4):
    #   Row oy=-2: ox = -2 → -1 → 0 → +1 → +2  (left to right)
    #   Row oy=-1: ox = +2 → +1 → 0 → -1 → -2  (right to left)
    #   Row oy= 0: ox = -2 → -1 → 0 → +1 → +2  (left to right)
    #   Row oy=+1: ox = +2 → +1 → 0 → -1 → -2  (right to left)
    #   Row oy=+2: ox = -2 → -1 → 0 → +1 → +2  (left to right)
    #
    # This ensures each k-point is adjacent to the previous one.
    
    ordered_offsets = []
    for row_idx, oy in enumerate(offsets):
        if row_idx % 2 == 0:
            # Even rows: left to right
            row_offsets = [(ox, oy) for ox in offsets]
        else:
            # Odd rows: right to left
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
    log(f"  K-stencil: {len(k_points)} points around k0=({k0_x:.4f}, {k0_y:.4f})")
    
    # Registry sampling grid in [0, 1)
    n_registry = n_registry_samples
    step = 1.0 / n_registry
    pos_min = 0.0
    pos_max = 1.0 - step
    
    # Polarization - handle merged mode
    polarization = candidate_params.get('polarization', config.get('blaze_polarization', 'TM'))
    if polarization == 'merged':
        # Use dominant_polarization (most common across k-path) or local_polarization (at k0)
        # Prefer local_polarization since we're computing at k0
        polarization = candidate_params.get('local_polarization',
                         candidate_params.get('dominant_polarization', 'TM'))
        log(f"  Using {polarization} polarization (from merged bands)")
    
    # All k-indices and target band
    k_indices_str = ", ".join(str(i) for i in range(len(k_points)))
    target_band = band_index + 1  # 1-indexed for output.selective
    
    threads = config.get('blaze_threads', 16)
    
    # BLAZE 0.4.0 TOML config using [[sweeps]] syntax
    config_content = f'''# BLAZE config for Phase 1 V2 - Candidate {candidate_params['candidate_id']}
# Auto-generated for blaze2d 0.4.0 API

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

# Layer 1 atom - fixed at center
[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {r_over_a}
eps_inside = 1.0

# Layer 2 atom - position swept (registry shift)
[[geometry.atoms]]
pos = [{pos_min:.8f}, {pos_min:.8f}]
radius = {r_over_a}
eps_inside = 1.0

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

# Explicit k-path: FD stencil around k0
[path]
k_path = [{k_points_str}]

[eigensolver]
n_bands = {num_bands}
max_iter = 200
tol = 1e-6

[dielectric.smoothing]
mesh_size = 3

# BLAZE 0.4.0: Use [[sweeps]] for parameter sweeps
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
mode = "selective"

[output.selective]
k_indices = [{k_indices_str}]
bands = [{target_band}]
'''
    
    config_path = Path(temp_dir) / "blaze_config.toml"
    config_path.write_text(config_content)
    
    # Save k_point_labels mapping
    labels_path = Path(temp_dir) / "k_labels.json"
    save_json({str(k): v for k, v in k_point_labels.items()}, labels_path)
    
    return config_path, k_point_labels, n_registry


def extract_band_data_from_blaze_v2(
    results, 
    candidate_params, 
    config, 
    delta_frac_grid, 
    k_point_labels, 
    n_registry_samples
):
    """
    Extract omega0, vg, M_inv from BLAZE results (V2 formulation).
    
    BLAZE computes band data on a uniform registry grid [0,1) × [0,1).
    We interpolate to get values at the actual delta_frac_grid points.
    
    Also stores raw stencil data for potential reprocessing.
    """
    from scipy.interpolate import RegularGridInterpolator
    
    Ns1, Ns2 = delta_frac_grid.shape[:2]
    band_index = candidate_params['band_index']
    k0_x = candidate_params['k0_x']
    k0_y = candidate_params['k0_y']
    dk = config.get('blaze_dk', config.get('phase1_dk', 0.005))
    fd_order = config.get('blaze_fd_order', config.get('phase1_fd_order', 4))
    
    n_registry = n_registry_samples
    
    # Build FD coefficients
    if fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        offsets = [-1, 0, 1]
        coeff_first = np.array([-0.5, 0, 0.5])
        coeff_second = np.array([1, -2, 1])
    
    n_stencil = len(offsets)  # 5 for 4th order, 3 for 2nd order
    
    # Create arrays for the sampled registry grid
    registry_grid_omega0 = np.full((n_registry, n_registry), np.nan)
    registry_grid_vg_x = np.full((n_registry, n_registry), np.nan)
    registry_grid_vg_y = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_xx = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_yy = np.full((n_registry, n_registry), np.nan)
    registry_grid_d2_xy = np.full((n_registry, n_registry), np.nan)
    
    # Raw stencil storage: shape (n_registry, n_registry, n_stencil, n_stencil)
    # stencil_omega[ix, iy, i, j] = omega at k = k0 + (offsets[i]*dk, offsets[j]*dk)
    stencil_omega = np.full((n_registry, n_registry, n_stencil, n_stencil), np.nan)
    
    step = 1.0 / n_registry
    
    for r in results:
        # Extract atom positions from sweep_values (BLAZE 0.4.0 API)
        sv = r.get('sweep_values', {})
        dx = sv.get('atom1.pos_x')
        dy = sv.get('atom1.pos_y')
        
        if dx is None or dy is None:
            # Fallback: try params dict
            atoms = r.get('params', {}).get('atoms', [])
            if len(atoms) >= 2:
                pos = atoms[1].get('pos', [0, 0])
                dx, dy = pos[0], pos[1]
            else:
                continue
        
        # Find grid indices
        ix = int(round(dx / step))
        iy = int(round(dy / step))
        
        if ix >= n_registry:
            ix = n_registry - 1
        if iy >= n_registry:
            iy = n_registry - 1
        
        bands = r['bands']  # bands[k_idx][band_idx]
        num_k = r['num_k_points']
        num_bands = r['num_bands']
        k_path = r.get('k_path', [])  # List of (kx, ky) tuples
        
        # With selective output, we only have 1 band (index 0)
        actual_band_idx = 0 if num_bands == 1 else min(band_index, num_bands - 1)
        
        # Extract omega values at each k-point using ACTUAL k-path coordinates
        # This is robust against any k-point reordering by BLAZE
        omega_values = {}
        tol = dk * 0.01  # Tolerance for matching k-points
        
        for kidx, (kx, ky) in enumerate(k_path):
            if kidx >= len(bands):
                continue
            # Determine which offset this k-point corresponds to
            ox_float = (kx - k0_x) / dk
            oy_float = (ky - k0_y) / dk
            ox = round(ox_float)
            oy = round(oy_float)
            
            # Verify it's a valid stencil offset
            if ox in offsets and oy in offsets:
                # Double-check the match is accurate
                kx_expected = k0_x + ox * dk
                ky_expected = k0_y + oy * dk
                if abs(kx - kx_expected) < tol and abs(ky - ky_expected) < tol:
                    omega_values[(ox, oy)] = bands[kidx][actual_band_idx]
        
        if (0, 0) not in omega_values:
            continue
        
        omega0 = omega_values[(0, 0)]
        registry_grid_omega0[ix, iy] = omega0
        
        # Store raw stencil values for reprocessing capability
        # stencil_omega[ix, iy, i, j] = omega at k = k0 + (offsets[i]*dk, offsets[j]*dk)
        for i, ox in enumerate(offsets):
            for j, oy in enumerate(offsets):
                stencil_omega[ix, iy, i, j] = omega_values.get((ox, oy), np.nan)
        
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

    # Fill any NaN values with nearest neighbor
    if np.any(np.isnan(registry_grid_omega0)):
        from scipy.ndimage import distance_transform_edt
        
        for grid in [registry_grid_omega0, registry_grid_vg_x, registry_grid_vg_y,
                     registry_grid_d2_xx, registry_grid_d2_yy, registry_grid_d2_xy]:
            mask = np.isnan(grid)
            if np.any(mask) and not np.all(mask):
                _, indices = distance_transform_edt(mask, return_indices=True)
                grid[mask] = grid[tuple(indices[:, mask])]
    
    # Create interpolators with periodic boundary conditions
    x_coords = np.linspace(0, 1 - step, n_registry)
    y_coords = np.linspace(0, 1 - step, n_registry)
    
    def make_periodic_interp(grid):
        extended = np.zeros((n_registry + 1, n_registry + 1))
        extended[:n_registry, :n_registry] = grid
        extended[n_registry, :n_registry] = grid[0, :]
        extended[:n_registry, n_registry] = grid[:, 0]
        extended[n_registry, n_registry] = grid[0, 0]
        
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
    omega0_grid = np.zeros((Ns1, Ns2))
    vg_grid = np.zeros((Ns1, Ns2, 2))
    M_inv_grid = np.zeros((Ns1, Ns2, 2, 2))
    
    log(f"  Interpolating to {Ns1}x{Ns2} = {Ns1*Ns2} grid points...")
    
    # The BLAZE geometry has Atom 0 at (0.5, 0.5), Atom 1 swept from (0, 0) to (~1, ~1)
    # The relative shift is (gx - 0.5, gy - 0.5), which we map to delta_frac
    delta_frac_x = delta_frac_grid[:, :, 0]
    delta_frac_y = delta_frac_grid[:, :, 1]
    
    # Convert delta_frac to BLAZE atom position (add 0.5 for center offset, wrap to [0,1))
    query_x = np.mod(delta_frac_x + 0.5, 1.0)
    query_y = np.mod(delta_frac_y + 0.5, 1.0)
    
    query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
    
    # Interpolate all fields
    omega0_flat = interp_omega0(query_points)
    vg_x_flat = interp_vg_x(query_points)
    vg_y_flat = interp_vg_y(query_points)
    d2_xx_flat = interp_d2_xx(query_points)
    d2_yy_flat = interp_d2_yy(query_points)
    d2_xy_flat = interp_d2_xy(query_points)
    
    omega0_grid = omega0_flat.reshape(Ns1, Ns2)
    vg_grid[:, :, 0] = vg_x_flat.reshape(Ns1, Ns2)
    vg_grid[:, :, 1] = vg_y_flat.reshape(Ns1, Ns2)
    
    # Build M_inv tensor (Hessian of omega)
    M_inv_grid[:, :, 0, 0] = d2_xx_flat.reshape(Ns1, Ns2)
    M_inv_grid[:, :, 0, 1] = d2_xy_flat.reshape(Ns1, Ns2)
    M_inv_grid[:, :, 1, 0] = d2_xy_flat.reshape(Ns1, Ns2)
    M_inv_grid[:, :, 1, 1] = d2_yy_flat.reshape(Ns1, Ns2)
    
    # Apply regularization to M_inv
    min_abs_eig = 1e-6
    for i in range(Ns1):
        for j in range(Ns2):
            M = M_inv_grid[i, j]
            eigvals, eigvecs = np.linalg.eigh(M)
            mask = np.abs(eigvals) < min_abs_eig
            eigvals = np.where(mask, np.sign(eigvals) * min_abs_eig, eigvals)
            eigvals = np.where(eigvals == 0, min_abs_eig, eigvals)
            M_inv_grid[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Package stencil metadata for saving
    stencil_info = {
        'stencil_omega': stencil_omega,  # Shape: (n_registry, n_registry, n_stencil, n_stencil)
        'offsets': np.array(offsets),    # e.g., [-2, -1, 0, 1, 2] for 4th order
        'dk': dk,
        'fd_order': fd_order,
        'n_registry': n_registry,
        'coeff_first': coeff_first,
        'coeff_second': coeff_second,
    }
    
    return omega0_grid, vg_grid, M_inv_grid, stencil_info


# ==============================================================================
# Candidate Processing
# ==============================================================================

def process_candidate_v2(candidate_params, config, run_dir):
    """Process a single candidate through Phase 1 V2 using BLAZE."""
    cid = candidate_params['candidate_id']
    log(f"\n=== Processing Candidate {cid} (BLAZE V2) ===")
    log(f"  Lattice: {candidate_params['lattice_type']}")
    log(f"  r/a: {candidate_params['r_over_a']:.3f}, eps_bg: {candidate_params['eps_bg']:.1f}")
    
    # Show band index with correction note if applicable
    band_idx = candidate_params['band_index']
    merged_idx = candidate_params.get('merged_band_index', band_idx)
    if merged_idx != band_idx:
        log(f"  Band {band_idx} at k={candidate_params['k_label']} (merged view: {merged_idx})")
    else:
        log(f"  Band {band_idx} at k={candidate_params['k_label']}")
    
    if candidate_params.get('polarization'):
        pol_str = candidate_params['polarization']
        if pol_str == 'merged':
            actual_pol = candidate_params.get('dominant_polarization') or \
                         candidate_params.get('local_polarization') or 'unknown'
            log(f"  Polarization: {pol_str} → using {actual_pol}")
        else:
            log(f"  Polarization: {pol_str}")
    
    # Create candidate directory
    cdir = candidate_dir(run_dir, cid)
    cdir.mkdir(parents=True, exist_ok=True)
    
    # === 1. Setup moiré geometry (V2 fractional coordinates) ===
    Ns1 = config.get('phase1_Ns1', config.get('phase1_Nx', 32))
    Ns2 = config.get('phase1_Ns2', config.get('phase1_Ny', 32))
    
    moire_meta = ensure_moire_metadata(candidate_params, config)
    B_mono = moire_meta['B_mono']
    B_moire = moire_meta['B_moire']
    eta = moire_meta['eta']
    theta_rad = moire_meta['theta_rad']
    
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  Building fractional grid: {Ns1} × {Ns2}")
    
    # === 2. Build fractional grid (V2 primary grid) ===
    s_grid = build_fractional_grid(Ns1, Ns2)
    R_grid = fractional_to_cartesian(s_grid, B_moire)
    
    # Generate lattice visualization (before heavy BLAZE computations)
    log(f"  Generating lattice visualization...")
    try:
        plot_phase1_lattice_panels(cdir, candidate_params, moire_meta)
        log(f"  Lattice visualization saved")
    except Exception as e:
        log(f"  WARNING: Lattice visualization failed: {e}")
    
    # === 3. Compute registry map (V2 corrected formula) ===
    tau_frac = np.array(config.get('tau', [0.0, 0.0]))
    delta_frac = compute_registry_fractional_v2(s_grid, B_moire, B_mono, theta_rad, tau_frac)
    log(f"  Computed V2 registry map (NO η division!)")
    
    # Save candidate metadata
    save_json(candidate_params, cdir / "phase0_meta.json")
    
    # === 4. Run BLAZE for all registry points ===
    n_registry_samples = config.get('blaze_registry_samples', 32)
    log(f"  BLAZE registry samples: {n_registry_samples} × {n_registry_samples}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path, k_point_labels, n_registry = generate_blaze_config_v2(
            candidate_params, config, n_registry_samples, temp_dir
        )
        
        driver = BulkDriver(str(config_path), threads=config.get('blaze_threads', 0))
        log(f"  BLAZE job count: {driver.job_count}")
        
        start_time = time.time()
        results, stats = driver.run_collect()
        
        elapsed = time.time() - start_time
        log(f"  BLAZE completed {len(results)} jobs in {elapsed:.2f}s ({len(results)/max(elapsed, 0.001):.1f} jobs/s)")
    
    # === 5. Extract band data from BLAZE results ===
    log(f"  Extracting band data from BLAZE results...")
    omega0_grid, vg_grid, M_inv_grid, stencil_info = extract_band_data_from_blaze_v2(
        results, candidate_params, config, delta_frac, k_point_labels, n_registry
    )
    
    log("  Completed local band calculations")
    
    # === 6. Compute potential V(s) ===
    omega_ref = choose_reference_frequency(omega0_grid, config)
    V_grid = omega0_grid - omega_ref
    
    log(f"  Reference frequency: ω_ref = {omega_ref:.6f}")
    log(f"  Potential range: V ∈ [{V_grid.min():.6f}, {V_grid.max():.6f}]")
    
    # === 7. Save to HDF5 (V2 format: s_grid as primary) ===
    h5_path = cdir / "phase1_band_data.h5"
    with h5py.File(h5_path, 'w') as hf:
        # V2: s_grid is primary
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")  # For visualization
        hf.create_dataset("delta_frac", data=delta_frac, compression="gzip")
        hf.create_dataset("omega0", data=omega0_grid, compression="gzip")
        hf.create_dataset("vg", data=vg_grid, compression="gzip")
        hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
        hf.create_dataset("V", data=V_grid, compression="gzip")
        
        # Raw stencil data for reprocessing
        # Shape: (n_registry, n_registry, n_stencil, n_stencil)
        # stencil_omega[ix, iy, i, j] = omega at registry (ix,iy), k = k0 + (offsets[i]*dk, offsets[j]*dk)
        stencil_grp = hf.create_group("stencil")
        stencil_grp.create_dataset("omega", data=stencil_info['stencil_omega'], compression="gzip")
        stencil_grp.create_dataset("offsets", data=stencil_info['offsets'])
        stencil_grp.create_dataset("coeff_first", data=stencil_info['coeff_first'])
        stencil_grp.create_dataset("coeff_second", data=stencil_info['coeff_second'])
        stencil_grp.attrs["dk"] = stencil_info['dk']
        stencil_grp.attrs["fd_order"] = stencil_info['fd_order']
        stencil_grp.attrs["n_registry"] = stencil_info['n_registry']
        
        # V2 attributes
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["theta_deg"] = candidate_params.get('theta_deg', 0.0)
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["band_index"] = candidate_params['band_index']
        hf.attrs["k0_x"] = candidate_params['k0_x']
        hf.attrs["k0_y"] = candidate_params['k0_y']
        hf.attrs["lattice_type"] = candidate_params['lattice_type']
        hf.attrs["r_over_a"] = candidate_params['r_over_a']
        hf.attrs["eps_bg"] = candidate_params['eps_bg']
        hf.attrs["a"] = candidate_params['a']
        hf.attrs["moire_length"] = moire_meta['moire_length']
        hf.attrs["Ns1"] = Ns1
        hf.attrs["Ns2"] = Ns2
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_mono"] = B_mono
        hf.attrs["solver"] = "blaze2d"
        hf.attrs["pipeline_version"] = "V2"
        hf.attrs["coordinate_system"] = "fractional"
    
    log(f"  Saved V2 band data to {h5_path}")
    
    # === 8. Diagnostics summary ===
    field_stats = summarize_phase1_fields(omega0_grid, vg_grid, M_inv_grid, V_grid)
    save_json(field_stats, cdir / "phase1_field_stats.json")
    
    variation_tol = float(config.get('phase1_variation_tol', 1e-5))
    if field_stats['omega0_std'] < variation_tol:
        log("  WARNING: omega0_grid variation below tolerance; check setup")
    
    # === 9. Generate visualizations ===
    log(f"  Generating visualizations...")
    # Plot with both fractional and real-space coordinates (2x4 layout)
    moire_meta_plot = {
        'moire_length': moire_meta['moire_length'],
        'theta_rad': theta_rad,
        'a1_vec': B_mono[:, 0],
        'a2_vec': B_mono[:, 1],
        'B_moire': B_moire,
    }
    plot_phase1_fields_v2(cdir, s_grid, V_grid, vg_grid, M_inv_grid, B_moire, candidate_params, moire_meta_plot)
    
    # Plot 3x3 stencil band diagrams
    try:
        from common.plotting import plot_stencil_comparison
        plot_stencil_comparison(
            save_path=cdir / "phase1_stencil_comparison.png",
            stencil_omega=stencil_info['stencil_omega'],
            stencil_offsets=stencil_info['offsets'],
            dk=stencil_info['dk'],
            omega_ref=omega0_grid.mean(),  # Use mean omega as reference
            candidate_params=candidate_params,
            solver_name="BLAZE"
        )
    except Exception as e:
        log(f"  WARNING: Stencil visualization failed: {e}")
    
    log(f"=== Completed Candidate {cid} ===")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_phase1(run_dir, config_path):
    """Main Phase 1 V2 driver using BLAZE."""
    log("\n" + "="*70)
    log("PHASE 1 V2 (BLAZE): Local Bloch Problems — Fractional Coordinates")
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
        runs_base = Path(config.get('output_dir', 'runsV2'))
        # Look for BLAZE V2 phase0 runs (phase0_blaze_*)
        phase0_runs = sorted(runs_base.glob('phase0_blaze_*'))
        if not phase0_runs:
            raise FileNotFoundError(
                f"No BLAZE phase0 run directories found in {runs_base}\n"
                f"  (Looking for phase0_blaze_*)\n"
                f"  Found MPB runs? Use: python blaze_phasesV2/phase1_blaze.py <explicit_path>"
            )
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest BLAZE Phase 0 run: {run_dir}")
    
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
        # Show actual polarization if merged
        if pol == 'merged':
            actual_pol = row.get('local_polarization', row.get('dominant_polarization', '?'))
            pol_display = f"merged→{actual_pol}"
        else:
            pol_display = pol
        print(f"  {row['candidate_id']}: {row['lattice_type']}/{pol_display}, "
              f"r/a={row['r_over_a']:.3f}, eps={row['eps_bg']:.1f}, "
              f"band={row['band_index']}, k={row['k_label']}, "
              f"S_total={row['S_total']:.4f}")
    
    log(f"\n{'='*70}")
    for idx, row in top_candidates.iterrows():
        candidate_params = extract_candidate_parameters(row)
        try:
            process_candidate_v2(candidate_params, config, run_dir)
        except Exception as e:
            print(f"ERROR processing candidate {candidate_params['candidate_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 1 V2 (BLAZE) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 V2 to assemble EA operators")


def get_default_config_path() -> Path:
    """Return the path to the default BLAZE Phase 1 V2 config."""
    return PROJECT_ROOT / "configsV2" / "phase1_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase1("auto", str(default_config))
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        
        # Check if the argument is a number (candidate ID)
        try:
            candidate_id = int(arg)
            # It's a number: use latest run with this specific candidate
            log(f"Using default config: {default_config}")
            log(f"Running Phase 1 for candidate {candidate_id} only")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1("auto", str(default_config))
        except ValueError:
            # Not a number: interpret as run_dir
            log(f"Using default config: {default_config}")
            run_phase1(arg, str(default_config))
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path (or candidate_id and run_dir)
        arg1, arg2 = sys.argv[1], sys.argv[2]
        
        # Check if first arg is a candidate ID (number)
        try:
            candidate_id = int(arg1)
            # First arg is candidate ID, second is run_dir
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            log(f"Using default config: {default_config}")
            log(f"Running Phase 1 for candidate {candidate_id} only")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1(arg2, str(default_config))
        except ValueError:
            # First arg is run_dir, second is config
            run_phase1(arg1, arg2)
    elif len(sys.argv) == 4:
        # Three arguments: candidate_id, run_dir, config_path
        try:
            candidate_id = int(sys.argv[1])
            log(f"Running Phase 1 for candidate {candidate_id} only")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1(sys.argv[2], sys.argv[3])
        except ValueError:
            raise SystemExit(
                "Usage: python blaze_phasesV2/phase1_blaze.py [candidate_id] [run_dir|auto] [config.yaml]\n"
                "       No arguments: uses latest run with default config (all candidates)\n"
                "       One number: uses latest run for that specific candidate\n"
                "       One path: uses specified run_dir with default config\n"
                "       Two arguments: run_dir and config, OR candidate_id and run_dir\n"
                "       Three arguments: candidate_id, run_dir, and config"
            )
    else:
        raise SystemExit(
            "Usage: python blaze_phasesV2/phase1_blaze.py [candidate_id] [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest run with default config (all candidates)\n"
            "       One number: uses latest run for that specific candidate\n"
            "       One path: uses specified run_dir with default config\n"
            "       Two arguments: run_dir and config, OR candidate_id and run_dir\n"
            "       Three arguments: candidate_id, run_dir, and config"
        )
