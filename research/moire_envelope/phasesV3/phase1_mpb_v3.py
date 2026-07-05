"""
Phase 1 (MPB): Local Bloch problems at frozen registry — V3 Multi-Band Pipeline

This is the V3 multi-band implementation of Phase 1 using MPB (MIT Photonic Bands).
It extracts band data for MULTIPLE bands simultaneously to support the
multi-band envelope approximation theory.

V3 KEY FEATURES:
1. Extract N_bands frequencies ω_n(s) at each registry position
2. Compute group velocity v_n(s) and mass tensor M^(-1)_n(s) per band
3. Store eigenvector data for Berry connection computation in Phase 2
4. Track band subspace and extra bands for Born-Huang

IMPORTANT: THE "UNIVERSAL MASTER MAP"
-------------------------------------
The output of this phase (phase1_multiband_data.h5) contains the "Universal Master Map"
of the local band structure shift ω_n(δ) over the configuration space δ.
- This map is INDEPENDENT of the moiré twist angle or lattice constant.
- It encodes the fundamental response of the monolayer bands to interlayer stacking.
- Once computed (expensive), it can be used to simulate ANY moiré geometry (cheap)
  by simply sampling the map along the path δ(r) ≈ θ z^ r.
- This decoupling allows for rapid optimization of cavity designs by scanning twist angles
  without re-running MPB.

DATA STRUCTURES (V3):
- omega: (Ns1, Ns2, N_bands) - frequencies for each band
- vg: (Ns1, Ns2, N_bands, 2) - group velocities per band
- M_inv: (Ns1, Ns2, N_bands, 2, 2) - mass tensors per band
- eigenvectors: stored for Berry connection (computed in Phase 2)

THEORY REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md
"""

# ==============================================================================
# CRITICAL: Set threading environment variables BEFORE importing numpy/meep/mpb
# This forces MPB/BLAS/OpenMP to run single-threaded so we can use Python 
# multiprocessing for parallelism instead.
# ==============================================================================
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'

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
    import meep as mp
    from meep import mpb
except ImportError:
    print("ERROR: meep package not installed. Install with: pip install meep")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, choose_reference_frequency, save_json, load_json
from common.plotting import plot_phase1_fields_v2, plot_phase1_lattice_panels


_log_fn = None


def log(message):
    """Print message with flush. Can be overridden by setting _log_fn."""
    if _log_fn is not None:
        _log_fn(message)
    else:
        print(message, flush=True)


# ==============================================================================
# V3 Fractional Coordinate Functions (same as V2)
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build the monolayer lattice basis matrix B = (a1 | a2)."""
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular', 'honeycomb'):
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
    if 'eps_hole' in row:
        params['eps_hole'] = float(row['eps_hole'])
    
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
# MPB Configuration (V3 Multi-Band)
# ==============================================================================

def lattice_type_to_mpb(lattice_type: str) -> str:
    """Convert our lattice type names to MPB-compatible format."""
    mapping = {
        'hex': 'triangular', 'hexagonal': 'triangular', 'triangular': 'triangular',
        'honeycomb': 'triangular',
        'square': 'square', 'sq': 'square',
    }
    return mapping.get(lattice_type.lower(), lattice_type.lower())


def create_mpb_geometry(lattice_type, r_over_a, eps_bg, eps_hole=1.0, delta_frac=None):
    """
    Create MPB geometry for a (frozen-registry) bilayer configuration.
    
    Args:
        lattice_type: 'hex', 'square', or 'honeycomb'
        r_over_a: cylinder radius / lattice constant
        eps_bg: background dielectric
        eps_hole: hole/rod dielectric (1.0 for air holes, or e.g. 11.56 for rods)
        delta_frac: fractional stacking shift [dx, dy] in [0,1)^2
        
    Returns:
        (geometry_list, lattice, eps_bg)
    """
    a = 1.0  # Normalized lattice constant
    
    # Build lattice
    if lattice_type in ('hex', 'triangular', 'honeycomb'):
        # Hexagonal (triangular) lattice
        basis1 = mp.Vector3(1, 0, 0)
        basis2 = mp.Vector3(0.5, math.sqrt(3)/2, 0)
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0), basis1=basis1, basis2=basis2)
    else:
        # Square lattice
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    # Create geometry: cylinders (holes or rods) in background
    r = r_over_a * a
    
    if lattice_type == 'honeycomb':
        # Honeycomb: two-atom basis at (0,0) and (1/3, 1/3) in lattice coords
        cyl_A = mp.Cylinder(
            radius=r,
            center=mp.Vector3(0, 0, 0),
            material=mp.Medium(epsilon=eps_hole)
        )
        cyl_B = mp.Cylinder(
            radius=r,
            center=mp.Vector3(1/3, 1/3, 0),
            material=mp.Medium(epsilon=eps_hole)
        )
        geometry = [cyl_A, cyl_B]
        
        # Second layer (bilayer): add shifted copies of both sublattices
        if delta_frac is not None:
            shift = delta_frac
            cyl_A2 = mp.Cylinder(
                radius=r,
                center=mp.Vector3(shift[0], shift[1], 0),
                material=mp.Medium(epsilon=eps_hole)
            )
            cyl_B2 = mp.Cylinder(
                radius=r,
                center=mp.Vector3(shift[0] + 1/3, shift[1] + 1/3, 0),
                material=mp.Medium(epsilon=eps_hole)
            )
            geometry.extend([cyl_A2, cyl_B2])
    else:
        # Standard: single atom at origin
        cyl1 = mp.Cylinder(
            radius=r,
            center=mp.Vector3(0, 0, 0),
            material=mp.Medium(epsilon=eps_hole)
        )
        geometry = [cyl1]
        
        # Second hole at shifted position (for bilayer)
        if delta_frac is not None:
            shift = delta_frac
            cyl2 = mp.Cylinder(
                radius=r,
                center=mp.Vector3(shift[0], shift[1], 0),
                material=mp.Medium(epsilon=eps_hole)
            )
            geometry.append(cyl2)
    
    return geometry, lattice, eps_bg


def create_mpb_solver(geometry, lattice, eps_bg, num_bands, resolution, polarization='TM'):
    """Create and configure MPB ModeSolver."""
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=eps_bg),
        num_bands=num_bands,
        resolution=resolution
    )
    return ms


def compute_bands_at_k_stencil(
    ms, k0, dk, all_bands, polarization='TM', fd_order=6,
    extract_fields_at_center=False
):
    """
    Compute frequencies at a k-point stencil for finite-difference derivatives.
    
    When extract_fields_at_center=True, also extracts Bloch fields and epsilon
    at the central k-point during the stencil sweep, avoiding a separate MPB run.
    
    Args:
        ms: MPB ModeSolver
        k0: central k-point [k0_x, k0_y]
        dk: step size in k-space
        all_bands: list of band indices to track
        polarization: 'TM' or 'TE'
        fd_order: 2, 4, or 6 for finite difference order (stencil sizes 3, 5, 7)
        extract_fields_at_center: if True, extract Bloch fields and epsilon at k0
        
    Returns:
        dict with stencil data:
        - omega_stencil: (n_bands, n_stencil, n_stencil) frequencies
        - omega0: (n_bands,) central frequencies
        - vg: (n_bands, 2) group velocities
        - M_inv: (n_bands, 2, 2) inverse mass tensors
        - bloch_fields: (N_bands, Nx, Ny, 3) complex64 [only if extract_fields_at_center]
        - epsilon: (Nx, Ny) float64 [only if extract_fields_at_center]
    """
    import os
    import meep as mp
    
    if fd_order == 6:
        offsets = [-3, -2, -1, 0, 1, 2, 3]
        coeff_first = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=float) / 60.0
        coeff_second = np.array([2, -27, 270, -490, 270, -27, 2], dtype=float) / 180.0
    elif fd_order == 4:
        offsets = [-2, -1, 0, 1, 2]
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        offsets = [-1, 0, 1]
        coeff_first = np.array([-0.5, 0, 0.5])
        coeff_second = np.array([1, -2, 1])
    
    n_stencil = len(offsets)
    n_bands = len(all_bands)
    max_band = max(all_bands) + 1
    center_idx = n_stencil // 2
    
    # Build k-point stencil
    omega_stencil = np.full((n_bands, n_stencil, n_stencil), np.nan)
    bloch_fields_center = None
    epsilon_center = None
    
    for ix, ox in enumerate(offsets):
        for iy, oy in enumerate(offsets):
            kx = k0[0] + ox * dk
            ky = k0[1] + oy * dk
            
            ms.k_points = [mp.Vector3(kx, ky, 0)]
            
            # Suppress C-level MPB output (contextlib can't catch C stdout)
            mp.verbosity(0)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                if polarization == 'TM':
                    ms.run_tm()
                else:
                    ms.run_te()
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(devnull_fd)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
            
            freqs = np.array(ms.all_freqs[0])
            
            for ib, band in enumerate(all_bands):
                if band < len(freqs):
                    omega_stencil[ib, ix, iy] = freqs[band]
            
            # At the central k-point, extract Bloch fields and epsilon
            if extract_fields_at_center and ix == center_idx and iy == center_idx:
                bloch_fields_center = _extract_bloch_fields(ms, all_bands, polarization)
                eps = ms.get_epsilon()
                epsilon_center = np.array(eps, dtype=np.float64)
                if epsilon_center.ndim == 3 and epsilon_center.shape[2] == 1:
                    epsilon_center = epsilon_center[:, :, 0]
    
    # Extract central values and compute derivatives
    omega0 = omega_stencil[:, center_idx, center_idx]
    
    # Group velocity
    vg = np.zeros((n_bands, 2))
    for ib in range(n_bands):
        vg_x = np.sum(coeff_first * omega_stencil[ib, :, center_idx]) / dk
        vg_y = np.sum(coeff_first * omega_stencil[ib, center_idx, :]) / dk
        vg[ib] = [vg_x, vg_y]
    
    # Mass tensor (inverse effective mass)
    M_inv = np.zeros((n_bands, 2, 2))
    for ib in range(n_bands):
        d2_xx = np.sum(coeff_second * omega_stencil[ib, :, center_idx]) / (dk ** 2)
        d2_yy = np.sum(coeff_second * omega_stencil[ib, center_idx, :]) / (dk ** 2)
        
        # Mixed derivative
        d2_xy = 0.0
        for ix, ox in enumerate(offsets):
            for iy, oy in enumerate(offsets):
                d2_xy += coeff_first[ix] * coeff_first[iy] * omega_stencil[ib, ix, iy]
        d2_xy /= (dk ** 2)
        
        M_inv[ib, 0, 0] = d2_xx
        M_inv[ib, 0, 1] = d2_xy
        M_inv[ib, 1, 0] = d2_xy
        M_inv[ib, 1, 1] = d2_yy
    
    result = {
        'omega_stencil': omega_stencil,
        'omega0': omega0,
        'vg': vg,
        'M_inv': M_inv,
    }
    
    if bloch_fields_center is not None:
        result['bloch_fields'] = bloch_fields_center
        result['epsilon'] = epsilon_center
    
    return result


def _compute_single_registry_point(args):
    """
    Worker function for multiprocessing: compute bands at a single registry point.
    
    Args:
        args: tuple of (ix, iy, delta_frac, params_dict)
              params_dict contains: lattice_type, r_over_a, eps_bg, k0, dk, 
                                    all_bands, polarization, fd_order, resolution, max_band
              Optional: export_bloch_fields (bool) to export Bloch functions for Born-Huang
    Returns:
        tuple of (ix, iy, result_dict)
        result_dict includes 'bloch_fields' and 'epsilon' if export_bloch_fields=True
    """
    ix, iy, delta_frac, params = args
    
    # Create geometry with this registry shift
    geometry, lattice, bg_eps = create_mpb_geometry(
        params['lattice_type'], params['r_over_a'], params['eps_bg'], 
        eps_hole=params.get('eps_hole', 1.0), delta_frac=delta_frac
    )
    
    # Create solver
    ms = create_mpb_solver(
        geometry, lattice, bg_eps, params['max_band'], 
        params['resolution'], params['polarization']
    )
    
    export_bloch_fields = params.get('export_bloch_fields', False)
    
    # Single stencil pass: computes bands AND extracts fields/epsilon at center
    # This eliminates the extra MPB run that was previously needed for field export
    result = compute_bands_at_k_stencil(
        ms, params['k0'], params['dk'], params['all_bands'], 
        params['polarization'], params['fd_order'],
        extract_fields_at_center=export_bloch_fields
    )
    
    return (ix, iy, result)


def _extract_bloch_fields(ms, band_indices, polarization):
    """
    Extract periodic Bloch functions u_n(r) for specified bands.
    
    Args:
        ms: MPB ModeSolver after running
        band_indices: list of 0-based band indices
        polarization: 'TM' or 'TE'
        
    Returns:
        bloch_fields: array of shape (N_bands, Nx, Ny, 3)
    """
    fields = []
    for band_idx in band_indices:
        # MPB uses 1-based indexing
        efield = ms.get_efield(band_idx + 1, bloch_phase=False)
        
        # efield is MPBArray with shape (Nx, Ny, Nz=1, 3) for 2D
        if efield.ndim == 4 and efield.shape[2] == 1:
            efield = efield[:, :, 0, :]  # Shape: (Nx, Ny, 3)
        
        fields.append(np.array(efield, dtype=np.complex64))
    
    return np.stack(fields, axis=0)


def run_mpb_registry_sweep(
    candidate_params, config, n_registry_samples, all_bands, subspace_bands
):
    """
    Run MPB at each registry point to build the local band data.
    
    Uses Python multiprocessing for parallelism (MPB runs single-threaded).
    
    Args:
        candidate_params: candidate parameters dict
        config: configuration dict
            - export_bloch_fields (bool): if True, extract Bloch functions for Born-Huang
        n_registry_samples: number of registry samples per direction
        all_bands: list of all band indices to compute
        subspace_bands: list of subspace band indices
        
    Returns:
        dict with registry grid data, including 'bloch_fields' if export_bloch_fields=True
    """
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    
    # Keep original lattice_type (e.g. 'honeycomb') for geometry construction
    lattice_type = candidate_params['lattice_type']
    r_over_a = candidate_params['r_over_a']
    eps_bg = candidate_params['eps_bg']
    
    k0_x = candidate_params['k0_x']
    k0_y = candidate_params['k0_y']
    k0 = [k0_x, k0_y]
    
    dk = config.get('mpb_dk', 0.06)
    fd_order = config.get('mpb_fd_order', 6)
    resolution = config.get('mpb_resolution', 32)
    
    # Number of parallel workers (default 16, use 1 for testing single-threaded behavior)
    n_workers = config.get('mpb_n_workers', 1)
    
    # Whether to export Bloch fields for Born-Huang computation
    export_bloch_fields = config.get('export_bloch_fields', False)
    
    polarization = candidate_params.get('polarization', config.get('mpb_polarization', 'TM'))
    if polarization == 'merged':
        polarization = candidate_params.get('local_polarization',
                         candidate_params.get('dominant_polarization', 'TM'))
    
    N_all = len(all_bands)
    N_subspace = len(subspace_bands)
    max_band = max(all_bands) + 1
    
    n_registry = n_registry_samples
    
    if fd_order == 6:
        n_stencil = 7
    elif fd_order == 4:
        n_stencil = 5
    else:
        n_stencil = 3
    
    log(f"    V3 Multi-band: computing bands {all_bands} (subspace: {subspace_bands})")
    log(f"    Polarization: {polarization}, resolution: {resolution}")
    log(f"    FD order: {fd_order} ({n_stencil}×{n_stencil} stencil), dk: {dk}")
    log(f"    Parallel workers: {n_workers} (CPU count: {cpu_count()})")
    
    if export_bloch_fields:
        # Estimate storage
        from phasesV3.bloch_fields import estimate_bloch_field_storage
        storage_est = estimate_bloch_field_storage(n_registry_samples, n_registry_samples, N_all, resolution)
        log(f"    Bloch field export ENABLED - Storage estimate: {storage_est}")
    
    # Registry grids for ALL bands
    registry_omega0 = np.full((n_registry, n_registry, N_all), np.nan)
    registry_vg = np.full((n_registry, n_registry, N_all, 2), np.nan)
    registry_M_inv = np.full((n_registry, n_registry, N_all, 2, 2), np.nan)
    stencil_omega = np.full((n_registry, n_registry, N_all, n_stencil, n_stencil), np.nan)
    
    # Pre-allocate Bloch fields and epsilon storage with known shapes
    # (avoids lazy allocation and ensures memory is reserved upfront)
    Nx = Ny = resolution  # MPB grid size matches resolution
    bloch_fields = None
    epsilon_grid = None
    if export_bloch_fields:
        bloch_fields = np.zeros(
            (n_registry, n_registry, N_all, Nx, Ny, 3),
            dtype=np.complex64
        )
        epsilon_grid = np.zeros(
            (n_registry, n_registry, Nx, Ny),
            dtype=np.float64
        )
    
    step = 1.0 / n_registry
    total_points = n_registry * n_registry
    
    # Prepare arguments for all registry points
    params_dict = {
        'lattice_type': lattice_type,
        'r_over_a': r_over_a,
        'eps_bg': eps_bg,
        'eps_hole': candidate_params.get('eps_hole', 1.0),
        'k0': k0,
        'dk': dk,
        'all_bands': all_bands,
        'polarization': polarization,
        'fd_order': fd_order,
        'resolution': resolution,
        'max_band': max_band,
        'export_bloch_fields': export_bloch_fields,
    }
    
    work_items = []
    for ix in range(n_registry):
        for iy in range(n_registry):
            delta_frac = np.array([ix * step, iy * step])
            work_items.append((ix, iy, delta_frac, params_dict))
    
    # Process results incrementally: copy to pre-allocated arrays and discard
    # each result immediately to avoid holding all results in memory at once.
    # Use imap_unordered for better load balancing across workers.
    def _process_result(ix, iy, result):
        """Copy result into pre-allocated arrays and discard."""
        registry_omega0[ix, iy] = result['omega0']
        registry_vg[ix, iy] = result['vg']
        registry_M_inv[ix, iy] = result['M_inv']
        stencil_omega[ix, iy] = result['omega_stencil']
        if 'bloch_fields' in result and bloch_fields is not None:
            bloch_fields[ix, iy] = result['bloch_fields']
        if 'epsilon' in result and epsilon_grid is not None:
            epsilon_grid[ix, iy] = result['epsilon']
    
    if n_workers == 1:
        # Single process mode (for testing single-threaded behavior)
        log(f"    Running in single-process mode for testing...")
        pbar = tqdm(total=total_points, desc="    MPB registry sweep", unit="pt")
        completed = 0
        t_sweep_start = time.time()
        last_log_time = t_sweep_start
        for args in work_items:
            ix, iy, result = _compute_single_registry_point(args)
            _process_result(ix, iy, result)
            del result  # free memory immediately
            completed += 1
            pbar.update(1)
            now = time.time()
            if now - last_log_time > 60:
                elapsed = now - t_sweep_start
                rate = completed / elapsed
                eta_s = (total_points - completed) / rate if rate > 0 else 0
                log(f"    Progress: {completed}/{total_points} ({100*completed/total_points:.0f}%) "
                    f"[{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, {rate:.1f} pt/s]")
                last_log_time = now
        pbar.close()
    else:
        # Multi-process mode with imap_unordered for better load balancing
        log(f"    Running with {n_workers} parallel workers...")
        with Pool(processes=n_workers) as pool:
            pbar = tqdm(total=total_points, desc="    MPB registry sweep", unit="pt")
            completed = 0
            t_sweep_start = time.time()
            last_log_time = t_sweep_start
            for ix, iy, result in pool.imap_unordered(
                _compute_single_registry_point, work_items, chunksize=4
            ):
                _process_result(ix, iy, result)
                del result  # free memory immediately
                completed += 1
                pbar.update(1)
                now = time.time()
                if now - last_log_time > 60:
                    elapsed = now - t_sweep_start
                    rate = completed / elapsed
                    eta_s = (total_points - completed) / rate if rate > 0 else 0
                    log(f"    Progress: {completed}/{total_points} ({100*completed/total_points:.0f}%) "
                        f"[{elapsed:.0f}s elapsed, ~{eta_s:.0f}s remaining, {rate:.1f} pt/s]")
                    last_log_time = now
            pbar.close()
    
    log(f"    Completed {total_points} registry points")
    
    output = {
        'registry_omega0': registry_omega0,
        'registry_vg': registry_vg,
        'registry_M_inv': registry_M_inv,
        'stencil_omega': stencil_omega,
        'n_registry': n_registry,
        'all_bands': all_bands,
        'subspace_bands': subspace_bands,
        'dk': dk,
        'fd_order': fd_order,
    }
    
    if bloch_fields is not None:
        output['bloch_fields'] = bloch_fields
        log(f"    Bloch fields shape: {bloch_fields.shape}")
    
    if epsilon_grid is not None:
        output['epsilon'] = epsilon_grid
        log(f"    Epsilon grid shape: {epsilon_grid.shape}")
    
    return output


def extract_multiband_data_from_mpb_v3(
    registry_data,
    delta_frac_grid,
    all_bands,
    subspace_bands,
):
    """
    Extract multi-band data from MPB registry sweep results (V3).
    
    Interpolates registry data to the full moiré grid.
    
    Returns:
        omega_grid: (Ns1, Ns2, N_subspace) frequencies for subspace bands
        vg_grid: (Ns1, Ns2, N_subspace, 2) group velocities per band
        M_inv_grid: (Ns1, Ns2, N_subspace, 2, 2) mass tensors per band
        stencil_info: dict with raw stencil data for all bands (for Berry connection)
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    
    Ns1, Ns2 = delta_frac_grid.shape[:2]
    
    registry_omega0 = registry_data['registry_omega0']
    registry_vg = registry_data['registry_vg']
    registry_M_inv = registry_data['registry_M_inv']
    stencil_omega = registry_data['stencil_omega']
    n_registry = registry_data['n_registry']
    dk = registry_data['dk']
    fd_order = registry_data['fd_order']
    
    N_all = len(all_bands)
    N_subspace = len(subspace_bands)
    
    # Fill NaN with nearest neighbor interpolation if needed
    for grid in [registry_omega0, registry_vg, registry_M_inv]:
        shape = grid.shape
        if len(shape) == 3:
            for band_idx in range(shape[2]):
                band_slice = grid[:, :, band_idx]
                mask = np.isnan(band_slice)
                if np.any(mask) and not np.all(mask):
                    _, indices = distance_transform_edt(mask, return_indices=True)
                    grid[:, :, band_idx][mask] = band_slice[tuple(indices[:, mask])]
        elif len(shape) == 4:
            for band_idx in range(shape[2]):
                for comp in range(shape[3]):
                    band_slice = grid[:, :, band_idx, comp]
                    mask = np.isnan(band_slice)
                    if np.any(mask) and not np.all(mask):
                        _, indices = distance_transform_edt(mask, return_indices=True)
                        grid[:, :, band_idx, comp][mask] = band_slice[tuple(indices[:, mask])]
        elif len(shape) == 5:
            for band_idx in range(shape[2]):
                for i in range(shape[3]):
                    for j in range(shape[4]):
                        band_slice = grid[:, :, band_idx, i, j]
                        mask = np.isnan(band_slice)
                        if np.any(mask) and not np.all(mask):
                            _, indices = distance_transform_edt(mask, return_indices=True)
                            grid[:, :, band_idx, i, j][mask] = band_slice[tuple(indices[:, mask])]
    
    # Interpolation setup
    step = 1.0 / n_registry
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
    
    # Map subspace band indices to all_bands indices
    subspace_to_all = [all_bands.index(b) for b in subspace_bands]
    
    # Output grids for SUBSPACE bands only
    omega_grid = np.zeros((Ns1, Ns2, N_subspace))
    vg_grid = np.zeros((Ns1, Ns2, N_subspace, 2))
    M_inv_grid = np.zeros((Ns1, Ns2, N_subspace, 2, 2))
    
    # Query points from delta_frac_grid
    delta_frac_x = delta_frac_grid[:, :, 0]
    delta_frac_y = delta_frac_grid[:, :, 1]
    query_x = np.mod(delta_frac_x + 0.5, 1.0)
    query_y = np.mod(delta_frac_y + 0.5, 1.0)
    query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
    
    for sub_idx, all_idx in enumerate(subspace_to_all):
        # Interpolate omega
        interp_omega = make_periodic_interp(registry_omega0[:, :, all_idx])
        omega_grid[:, :, sub_idx] = interp_omega(query_points).reshape(Ns1, Ns2)
        
        # Interpolate vg
        for comp in range(2):
            interp_vg = make_periodic_interp(registry_vg[:, :, all_idx, comp])
            vg_grid[:, :, sub_idx, comp] = interp_vg(query_points).reshape(Ns1, Ns2)
        
        # Interpolate M_inv
        for i in range(2):
            for j in range(2):
                interp_M = make_periodic_interp(registry_M_inv[:, :, all_idx, i, j])
                M_inv_grid[:, :, sub_idx, i, j] = interp_M(query_points).reshape(Ns1, Ns2)
    
    # Regularize mass tensors (avoid singular matrices)
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
        'stencil_omega_all': stencil_omega,
        'registry_omega_all': registry_omega0,
        'offsets': np.array(
            [-3, -2, -1, 0, 1, 2, 3] if fd_order == 6
            else [-2, -1, 0, 1, 2] if fd_order == 4
            else [-1, 0, 1]
        ),
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
    log(f"\n=== Processing Candidate {cid} (MPB V3 Multi-Band) ===")
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
    
    n_registry_samples = config.get('mpb_registry_samples', 32)
    log(f"  MPB registry samples: {n_registry_samples} × {n_registry_samples}")
    
    # Run MPB registry sweep
    start_time = time.time()
    registry_data = run_mpb_registry_sweep(
        candidate_params, config, n_registry_samples, all_bands, subspace_bands
    )
    elapsed = time.time() - start_time
    log(f"  MPB sweep completed in {elapsed:.2f}s")
    
    # Check if Bloch fields and epsilon were exported
    bloch_fields_data = registry_data.get('bloch_fields', None)
    epsilon_data = registry_data.get('epsilon', None)
    if bloch_fields_data is not None:
        log(f"  Bloch fields exported: shape {bloch_fields_data.shape}")
    if epsilon_data is not None:
        log(f"  Epsilon grid exported: shape {epsilon_data.shape}")
    
    log(f"  Extracting multi-band data...")
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_mpb_v3(
        registry_data, delta_frac, all_bands, subspace_bands
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
        hf.create_dataset("omega", data=omega_grid, compression="gzip")
        hf.create_dataset("vg", data=vg_grid, compression="gzip")
        hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
        hf.create_dataset("V", data=V_grid, compression="gzip")
        
        # Raw stencil data for Phase 2 Berry connection
        stencil_grp = hf.create_group("stencil")
        stencil_grp.create_dataset("omega_all", data=stencil_info['stencil_omega_all'], compression="gzip")
        stencil_grp.create_dataset("registry_omega_all", data=stencil_info['registry_omega_all'], compression="gzip")
        stencil_grp.create_dataset("offsets", data=stencil_info['offsets'])
        stencil_grp.attrs["dk"] = stencil_info['dk']
        stencil_grp.attrs["fd_order"] = stencil_info['fd_order']
        stencil_grp.attrs["n_registry"] = stencil_info['n_registry']
        
        # Save Bloch fields if exported (for Born-Huang computation)
        if bloch_fields_data is not None:
            from phasesV3.bloch_fields import save_bloch_fields
            save_bloch_fields(hf, bloch_fields_data, {
                'resolution': config.get('mpb_resolution', 32),
                'polarization': candidate_params.get('polarization', 
                               config.get('mpb_polarization', 'TM')),
            })
            log(f"  Saved Bloch fields for Born-Huang computation")
        
        # Save dielectric function ε(r; δ) for B-orthonormalization in Phase 2
        if epsilon_data is not None:
            hf.create_dataset("epsilon", data=epsilon_data, compression="lzf",
                              chunks=(1, 1, epsilon_data.shape[2], epsilon_data.shape[3]))
            log(f"  Saved ε(r; δ) grid: shape {epsilon_data.shape}, "
                f"range [{epsilon_data.min():.2f}, {epsilon_data.max():.2f}]")
        
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
        hf.attrs["subspace_bands"] = np.array(subspace_bands)
        hf.attrs["all_bands"] = np.array(all_bands)
        hf.attrs["solver"] = "mpb"
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
    """Main Phase 1 V3 driver using MPB."""
    log("\n" + "="*70)
    log("PHASE 1 V3 (MPB): Multi-Band Local Bloch Problems")
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
        phase0_runs = sorted(runs_base.glob('phase0_mpb_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No MPB phase0 run directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest Phase 0 run: {run_dir}")
    else:
        # Check if path exists as provided
        p_run = Path(run_dir)
        if not p_run.exists():
            # Check if it's a name inside output_dir
            runs_base = Path(config.get('output_dir', 'runsV3'))
            if (runs_base / run_dir).exists():
                run_dir = runs_base / run_dir
                log(f"Found run directory in output folder: {run_dir}")
    
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
    log("PHASE 1 V3 (MPB) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 2 V3 for Berry connection and multi-band prep")


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase1_mpb.yaml"


if __name__ == "__main__":
    # Guard against re-execution in multiprocessing worker processes.
    # Workers inherit __name__=="__main__" via fork, and MPB's MPI internals
    # can trigger module re-evaluation, causing cascading Pool creation.
    import multiprocessing as _mp
    if _mp.current_process().name != 'MainProcess':
        import sys as _sys
        _sys.exit(0)

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
            # Case 1: [candidate_id] [run_dir]
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
            run_phase1_v3(arg2, str(default_config))
        except ValueError:
            # Case 2: [run_dir] [candidate_id]
            try:
                candidate_id = int(arg2)
                default_config = get_default_config_path()
                if not default_config.exists():
                    raise SystemExit(f"Default config not found: {default_config}")
                os.environ['MSL_PHASE1_CANDIDATE_ID'] = str(candidate_id)
                run_phase1_v3(arg1, str(default_config))
            except ValueError:
                # Case 3: [run_dir] [config_path]
                run_phase1_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python phasesV3/phase1_mpb_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
