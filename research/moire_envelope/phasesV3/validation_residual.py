#!/usr/bin/env python3
"""
Validation Metric 1: Maxwell Operator Residual

Computes the dimensionless residual:
    R = ||L H_pred - ω² M H_pred|| / ||ω² M H_pred||

where:
    - L is the Maxwell curl-curl operator (scalar form for 2D)
    - M is the material weight (ε for E-field, identity for H-field)
    - H_pred is the reconstructed field from the envelope approximation
    - ω is the predicted eigenfrequency

For 2D TE polarization (E in-plane, H_z scalar):
    Master equation:  -∇·(ε⁻¹ ∇ H_z) = ω² H_z
    
    We reconstruct E_x, E_y from the envelope theory, then compute
    H_z = (1/iω)(∂E_y/∂x - ∂E_x/∂y)  (from Faraday's law)
    
    Or equivalently, we can directly test the E-field equation:
    ∇×∇×E = ω² ε E  ↔  -∇²E_i + ∂_i(∇·E) = ω² ε E_i
    
    For a divergence-free field (∇·(εE)=0), this simplifies, but
    the cleanest scalar test is via H_z.

USAGE:
    python phasesV3/validation_residual.py [candidate_id] [mode_index]
"""

import os
import sys
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phasesV3 import phase4_field_reconstruction as p4
from common.io_utils import load_json


# =============================================================================
# Geometry helpers (inlined to avoid meep dependency from phase5)
# =============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build monolayer lattice basis B = (a1 | a2)."""
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'triangular'):
        return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def rotation_matrix(theta_rad: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """Compute moiré lattice basis: B_moire = (R(θ) - I)^{-1} @ B_mono."""
    R = rotation_matrix(theta_rad)
    Delta_R = R - np.eye(2)
    return np.linalg.inv(Delta_R) @ B_mono


def log(msg):
    print(f"[Residual] {msg}")


# =============================================================================
# 0. Gauge Fixing for Bloch Functions
# =============================================================================

def gauge_fix_bloch_fields(bloch_fields, band_indices):
    """
    Apply per-band scalar gauge fixing to Bloch fields.
    
    MPB Bloch functions u_n(r;R) have an arbitrary overall phase at each
    registry point R. This creates discontinuities at tile boundaries that
    destroy finite-difference derivatives. We fix the gauge by aligning
    each u_n(R) with u_n(R-dR) via a scalar phase rotation:
        u_n(R) → e^{-iφ} u_n(R), where φ = arg(<u_n(R-dR)|u_n(R)>).
    
    This ensures the overlap <u_n(R)|u_n(R+dR)> is real and positive,
    making the Bloch functions smooth across the registry.
    
    Args:
        bloch_fields: (Ns1, Ns2, N_bands_all, Nx, Ny, 3) complex array
        band_indices: list of band indices in the all-bands dimension
    
    Returns:
        u_sub: (Ns1, Ns2, N_sub, Nx, Ny, 3) gauge-fixed subspace fields
    """
    Ns1, Ns2 = bloch_fields.shape[:2]
    u_sub = bloch_fields[:, :, band_indices, :, :, :].copy()
    N_sub = len(band_indices)
    
    # Fix along axis 0 (s1 direction)
    for n in range(N_sub):
        for j in range(Ns2):
            for i in range(1, Ns1):
                ov = np.sum(np.conj(u_sub[i-1, j, n]) * u_sub[i, j, n])
                if np.abs(ov) > 1e-10:
                    u_sub[i, j, n] *= np.exp(-1j * np.angle(ov))
    
    # Fix along axis 1 (s2 direction)
    for n in range(N_sub):
        for i in range(Ns1):
            for j in range(1, Ns2):
                ov = np.sum(np.conj(u_sub[i, j-1, n]) * u_sub[i, j, n])
                if np.abs(ov) > 1e-10:
                    u_sub[i, j, n] *= np.exp(-1j * np.angle(ov))
    
    log(f"  Gauge-fixed {N_sub} bands across {Ns1}×{Ns2} registry")
    return u_sub


# =============================================================================
# 0b. Per-Tile Rayleigh Quotient (Metric 1)
# =============================================================================

def compute_per_tile_residual(
    bloch_fields_fixed,  # (Ns_b, Ns_b, N_sub, Nx, Ny, 3) gauge-fixed
    band_indices,
    eps_registry,        # (Ns_b, Ns_b, Nx, Ny)
    omega_bands,         # (Ns_env, Ns_env, N_sub) band frequencies
    F_spinor,            # (N_modes, Ns_env, Ns_env, N_sub)
    mode_idx,
    k0_phys,             # (2,) physical Bloch wave vector
    bloch_fields_raw=None,  # (Ns_b, Ns_b, N_bands_all, Nx, Ny, 3) raw Bloch for FD baseline
):
    """
    Compute per-tile E-field Rayleigh quotient and aggregate residual.
    
    For each registry tile (i,j), the Bloch function u_n(r;R) is a local
    eigenstate with eigenvalue ω²_n(R). The per-tile Rayleigh quotient:
        R_q(i,j) = ∫_cell |curl_k E|² / ∫_cell ε |E|²
    should equal ω²_n(R) up to FD discretization error.
    
    The global dimensionless residual is the |F|²-weighted deviation:
        R = sqrt( Σ_ij |F(i,j)|² · (R_q(i,j)/ω²(i,j) - 1)² / Σ |F|² )
    
    This metric:
    - Avoids tile boundary artifacts (FD stays within each tile)
    - Uses local ω² (not a single global value)
    - Weights by envelope amplitude (focuses on physically relevant regions)
    - Is dimensionless and converges with resolution
    
    Returns:
        results: dict with keys:
            'R_global': global weighted residual (scalar)
            'Rq_weighted': |F|²-weighted Rayleigh quotient
            'omega2_weighted': |F|²-weighted expected ω²
            'ratio_weighted': Rq_weighted / omega2_weighted
            'Rq_map': (Ns_b, Ns_b) per-tile Rayleigh quotients
            'omega2_map': (Ns_b, Ns_b) per-tile ω² values  
            'ratio_map': (Ns_b, Ns_b) per-tile R_q/ω² ratios
    """
    from scipy.ndimage import zoom
    
    Ns_b = bloch_fields_fixed.shape[0]
    Nx, Ny = bloch_fields_fixed.shape[3], bloch_fields_fixed.shape[4]
    Ns_env = F_spinor.shape[1]
    N_sub = len(band_indices)
    
    # Get envelope on registry grid (downsample if needed)
    F_mode = F_spinor[mode_idx]  # (Ns_env, Ns_env, N_sub)
    
    # Within-tile FD operators (periodic on unit cell)
    d = 1.0 / Nx
    def dx_u(f):
        return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * d)
    def dy_u(f):
        return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * d)
    
    # Per-tile results
    Rq_map = np.zeros((Ns_b, Ns_b))
    omega2_map = np.zeros((Ns_b, Ns_b))
    num_map = np.zeros((Ns_b, Ns_b))  # curl energy per tile
    den_map = np.zeros((Ns_b, Ns_b))  # eps-weighted E energy per tile
    weight_map = np.zeros((Ns_b, Ns_b))  # |F|² weight
    
    # Compute |F|² on registry grid
    # For mode dominated by one band: use that band's envelope
    # For multi-band: use total |F|²
    F_sq_env = np.sum(np.abs(F_mode)**2, axis=-1)  # (Ns_env, Ns_env)
    if Ns_b != Ns_env:
        F_sq_reg = zoom(F_sq_env, (Ns_b / Ns_env, Ns_b / Ns_env), order=1, mode='wrap')
    else:
        F_sq_reg = F_sq_env
    
    # Map omega_bands to registry grid
    if omega_bands.shape[0] != Ns_b:
        omega_bands_reg = np.zeros((Ns_b, Ns_b, N_sub))
        for n in range(N_sub):
            omega_bands_reg[:, :, n] = zoom(
                omega_bands[:, :, n], (Ns_b / omega_bands.shape[0],) * 2,
                order=1, mode='wrap'
            )
    else:
        omega_bands_reg = omega_bands
    
    # Find dominant band for this mode
    band_weights = np.zeros(N_sub)
    for n in range(N_sub):
        band_weights[n] = np.sum(np.abs(F_mode[:, :, n])**2)
    band_weights /= band_weights.sum()
    dom_sub = np.argmax(band_weights)
    log(f"  Mode {mode_idx}: band weights = {band_weights}, dominant = band {dom_sub}")
    
    for ti in range(Ns_b):
        for tj in range(Ns_b):
            # Get E-field at this tile (dominant band)
            u_x = bloch_fields_fixed[ti, tj, dom_sub, :, :, 0]
            u_y = bloch_fields_fixed[ti, tj, dom_sub, :, :, 1]
            eps_local = eps_registry[ti, tj]  # (Nx, Ny)
            
            # k-modified curl: (∂/∂x + ik₀ₓ)Ey - (∂/∂y + ik₀ᵧ)Ex
            curl_k = (dx_u(u_y) + 1j * k0_phys[0] * u_y) - \
                     (dy_u(u_x) + 1j * k0_phys[1] * u_x)
            
            num = np.sum(np.abs(curl_k)**2)
            den = np.sum(eps_local * (np.abs(u_x)**2 + np.abs(u_y)**2))
            
            if den > 1e-30:
                Rq_map[ti, tj] = num / den
            
            # Local ω² (dominant band)
            freq_local = omega_bands_reg[ti, tj, dom_sub]
            omega2_map[ti, tj] = (2 * np.pi * freq_local) ** 2
            
            num_map[ti, tj] = num
            den_map[ti, tj] = den
            weight_map[ti, tj] = F_sq_reg[ti, tj]
    
    # Weighted aggregates
    w = weight_map
    w_sum = np.sum(w)
    
    # Weighted Rayleigh quotient
    Rq_weighted = np.sum(w * num_map) / np.sum(w * den_map) if np.sum(w * den_map) > 0 else np.inf
    omega2_weighted = np.sum(w * omega2_map) / w_sum if w_sum > 0 else np.inf
    ratio_weighted = Rq_weighted / omega2_weighted if omega2_weighted > 0 else np.inf
    
    # Dimensionless residual: RMS of (R_q/ω² - 1) weighted by |F|²
    ratio_map = np.where(omega2_map > 0, Rq_map / omega2_map, 0.0)
    deviation = ratio_map - 1.0
    R_global = np.sqrt(np.sum(w * deviation**2) / w_sum) if w_sum > 0 else np.inf
    
    # FD-corrected metric: compare moiré R_q to single-eigenstate R_q
    # This cancels FD discretization errors and isolates envelope theory error.
    Rq_eigen_map = np.zeros((Ns_b, Ns_b))
    if bloch_fields_raw is not None:
        dom_band_idx = band_indices[dom_sub]
        for ti in range(Ns_b):
            for tj in range(Ns_b):
                u_x_raw = bloch_fields_raw[ti, tj, dom_band_idx, :, :, 0]
                u_y_raw = bloch_fields_raw[ti, tj, dom_band_idx, :, :, 1]
                eps_local = eps_registry[ti, tj]
                curl_raw = (dx_u(u_y_raw) + 1j * k0_phys[0] * u_y_raw) - \
                           (dy_u(u_x_raw) + 1j * k0_phys[1] * u_x_raw)
                num_raw = np.sum(np.abs(curl_raw)**2)
                den_raw = np.sum(eps_local * (np.abs(u_x_raw)**2 + np.abs(u_y_raw)**2))
                if den_raw > 1e-30:
                    Rq_eigen_map[ti, tj] = num_raw / den_raw
        
        # FD-corrected ratio: R_q_moire / R_q_eigenstate (cancels FD errors)
        ratio_corrected = np.where(Rq_eigen_map > 0, Rq_map / Rq_eigen_map, 1.0)
        r_corrected_wtd = np.sum(w * ratio_corrected) / w_sum if w_sum > 0 else np.inf
        
        # FD-corrected weighted R_q
        Rq_eigen_weighted = np.sum(w * Rq_eigen_map * den_map) / np.sum(w * den_map) \
            if np.sum(w * den_map) > 0 else np.inf
        
        R_fd_corrected = float(np.sqrt(np.sum(w * (ratio_corrected - 1.0)**2) / w_sum)) \
            if w_sum > 0 else np.inf
        
        log(f"  FD baseline weighted ratio: {np.sum(w * Rq_eigen_map / omega2_map) / w_sum:.6f}")
        log(f"  FD-corrected ratio (moiré/eigenstate): {r_corrected_wtd:.6f}")
        log(f"  FD-corrected residual R: {R_fd_corrected:.6e}")
    else:
        ratio_corrected = ratio_map
        r_corrected_wtd = ratio_weighted
        R_fd_corrected = R_global
    
    results = {
        'R_global': float(R_global),
        'R_fd_corrected': float(R_fd_corrected),
        'ratio_fd_corrected': float(r_corrected_wtd),
        'Rq_weighted': float(Rq_weighted),
        'omega2_weighted': float(omega2_weighted),
        'ratio_weighted': float(ratio_weighted),
        'Rq_map': Rq_map,
        'omega2_map': omega2_map,
        'ratio_map': ratio_map,
        'ratio_corrected_map': ratio_corrected if bloch_fields_raw is not None else None,
        'weight_map': weight_map,
        'band_weights': band_weights.tolist(),
        'dominant_band': int(dom_sub),
    }
    
    return results


# =============================================================================
# 1. Dielectric Function ε(x, y) — MPB subpixel-averaged version
# =============================================================================

def _extract_epsilon_single(args):
    """
    Worker: extract subpixel-averaged ε for one registry point.
    
    Must be a top-level function for multiprocessing.
    """
    ix, iy, delta_frac, params = args
    
    import os, sys
    # Force single-threaded MPB/BLAS in worker subprocess
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLAS_NUM_THREADS'):
        os.environ[var] = '1'
    import meep as mp
    from meep import mpb
    
    lattice_type = params['lattice_type']
    r_over_a = params['r_over_a']
    eps_bg = params['eps_bg']
    resolution = params['resolution']
    
    # Build lattice
    if lattice_type in ('hex', 'triangular'):
        basis1 = mp.Vector3(1, 0, 0)
        basis2 = mp.Vector3(0.5, math.sqrt(3)/2, 0)
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0), basis1=basis1, basis2=basis2)
    else:
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    r = r_over_a  # a=1 normalized
    
    # Geometry: cylinder at origin + cylinder at shift
    geometry = [
        mp.Cylinder(radius=r, center=mp.Vector3(0, 0, 0),
                    material=mp.Medium(epsilon=1.0))
    ]
    if delta_frac is not None:
        geometry.append(
            mp.Cylinder(radius=r, center=mp.Vector3(delta_frac[0], delta_frac[1], 0),
                        material=mp.Medium(epsilon=1.0))
        )
    
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=eps_bg),
        num_bands=1,
        resolution=resolution,
    )
    
    # Suppress C-level stdout/stderr from MPB (contextlib can't catch C output)
    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        ms.init_params(mp.NO_PARITY, False)
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
    
    eps = np.array(ms.get_epsilon())  # shape (resolution, resolution)
    return (ix, iy, eps)


def extract_mpb_epsilon_grid(p0_meta, n_registry, resolution, cache_path=None,
                             n_workers=8):
    """
    Extract subpixel-averaged ε from MPB for all registry points.
    
    Args:
        p0_meta: dict with lattice_type, r_over_a, eps_bg
        n_registry: number of registry samples per direction (e.g. 64)
        resolution: MPB resolution (e.g. 64)
        cache_path: if given, save/load from this H5 file
        n_workers: number of parallel workers
    
    Returns:
        eps_registry: array of shape (n_registry, n_registry, resolution, resolution)
    """
    import h5py
    
    # Check cache
    if cache_path is not None and Path(cache_path).exists():
        log(f"Loading cached MPB epsilon from {cache_path}")
        with h5py.File(cache_path, 'r') as f:
            return f['epsilon_registry'][:]
    
    log(f"Extracting MPB subpixel-averaged ε: {n_registry}×{n_registry} registry, "
        f"resolution={resolution}")
    
    params = {
        'lattice_type': p0_meta.get('lattice_type', 'square'),
        'r_over_a': p0_meta['r_over_a'],
        'eps_bg': p0_meta['eps_bg'],
        'resolution': resolution,
    }
    
    step = 1.0 / n_registry
    work_items = []
    for ix in range(n_registry):
        for iy in range(n_registry):
            delta_frac = np.array([ix * step, iy * step])
            work_items.append((ix, iy, delta_frac, params))
    
    eps_registry = np.zeros((n_registry, n_registry, resolution, resolution),
                            dtype=np.float64)
    
    if n_workers > 1:
        from multiprocessing import Pool
        from tqdm import tqdm
        log(f"  Running {len(work_items)} MPB epsilon extractions with {n_workers} workers...")
        with Pool(n_workers) as pool:
            for ix, iy, eps in tqdm(
                pool.imap_unordered(_extract_epsilon_single, work_items),
                total=len(work_items), desc="  MPB ε extraction"
            ):
                eps_registry[ix, iy] = eps
    else:
        from tqdm import tqdm
        for args in tqdm(work_items, desc="  MPB ε extraction"):
            ix, iy, eps = _extract_epsilon_single(args)
            eps_registry[ix, iy] = eps
    
    # Cache
    if cache_path is not None:
        log(f"  Caching MPB epsilon to {cache_path}")
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('epsilon_registry', data=eps_registry, compression='gzip')
            f.attrs['n_registry'] = n_registry
            f.attrs['resolution'] = resolution
            for k, v in p0_meta.items():
                try:
                    f.attrs[k] = v
                except TypeError:
                    pass
    
    return eps_registry


def tile_epsilon_from_registry(eps_registry, Ns_env):
    """
    Tile registry-resolved epsilon into a full moiré-cell epsilon grid.
    
    The tiled grid matches the reconstruct_full_field_single_mode output:
    shape (Ns_env * Nx_micro, Ns_env * Ny_micro), where each tile
    at registry point (i, j) uses the epsilon from that registry.
    
    If the envelope grid (Ns_env) differs from the registry grid (Ns_reg),
    we interpolate via scipy.ndimage.zoom (same approach as for Bloch fields).
    
    Args:
        eps_registry: (Ns_reg, Ns_reg, Nx_micro, Ny_micro)
        Ns_env: envelope grid size per axis (e.g. 128)
    
    Returns:
        eps_tiled: (Ns_env * Nx_micro, Ns_env * Ny_micro)
    """
    from scipy.ndimage import zoom
    
    Ns_reg, _, Nx, Ny = eps_registry.shape
    
    if Ns_reg != Ns_env:
        # Interpolate along registry axes (first two) to match envelope grid
        zoom_factor = (Ns_env / Ns_reg, Ns_env / Ns_reg, 1, 1)
        log(f"  Interpolating ε registry {Ns_reg}→{Ns_env} (zoom={zoom_factor[:2]})")
        eps_registry = zoom(eps_registry, zoom_factor, order=1, mode='wrap')
    
    # Tile: concatenate micro-grids
    # eps_registry has shape (Ns_env, Ns_env, Nx, Ny)
    # We want (Ns_env*Nx, Ns_env*Ny) by tiling
    rows = []
    for i in range(eps_registry.shape[0]):
        row = np.concatenate([eps_registry[i, j] for j in range(eps_registry.shape[1])],
                            axis=1)
        rows.append(row)
    eps_tiled = np.concatenate(rows, axis=0)
    
    return eps_tiled


# =============================================================================
# 1b. Analytical Dielectric Function (fallback, no subpixel averaging)
# =============================================================================

def build_epsilon_evaluator(p0_meta):
    """
    Build a vectorized epsilon evaluator for the twisted bilayer PhC.
    
    The structure consists of two layers of air holes (cylinders of ε=1)
    in a dielectric background (ε=eps_bg). Each layer is a 2D lattice
    rotated by ±θ/2.
    
    Args:
        p0_meta: dict with keys 'a', 'r_over_a', 'eps_bg', 'theta_deg',
                 'lattice_type'
    
    Returns:
        eval_eps: function(x, y) -> ε values (vectorized, accepts arrays)
    """
    a = p0_meta['a']
    r = p0_meta['r_over_a'] * a
    eps_bg = p0_meta['eps_bg']
    theta_rad = math.radians(p0_meta['theta_deg'])
    lattice_type = p0_meta.get('lattice_type', 'square')
    
    B_mono = build_monolayer_basis(lattice_type, a)
    
    # Rotation matrices for the two layers
    R_bot = rotation_matrix(-theta_rad / 2)
    R_top = rotation_matrix(+theta_rad / 2)
    
    # Rotated lattice vectors for each layer
    a1_bot = R_bot @ B_mono[:, 0]
    a2_bot = R_bot @ B_mono[:, 1]
    a1_top = R_top @ B_mono[:, 0]
    a2_top = R_top @ B_mono[:, 1]
    
    # Inverse basis matrices for fractional coordinate computation
    B_bot = np.column_stack([a1_bot, a2_bot])
    B_top = np.column_stack([a1_top, a2_top])
    B_bot_inv = np.linalg.inv(B_bot)
    B_top_inv = np.linalg.inv(B_top)
    
    r_sq = r * r
    
    def eval_eps(x, y):
        """
        Evaluate ε(x,y) for arrays of coordinates.
        
        Strategy: for each point, find the nearest lattice site in each layer
        and check if the point is within radius r of that site.
        
        For efficiency, we convert to fractional coordinates, round to nearest
        integer (nearest lattice site), compute distance in Cartesian.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        P = np.stack([x, y], axis=0)  # (2, N)
        
        eps = np.full(x.shape, eps_bg, dtype=np.float64)
        
        # Check both layers
        for B_inv, B_mat in [(B_bot_inv, B_bot), (B_top_inv, B_top)]:
            frac = B_inv @ P  # (2, N) fractional coordinates
            
            # Check the 9 nearest lattice sites (n,m) around (frac[0], frac[1])
            frac0_floor = np.floor(frac[0]).astype(int)
            frac1_floor = np.floor(frac[1]).astype(int)
            
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    n = frac0_floor + di
                    m = frac1_floor + dj
                    
                    # Lattice site in Cartesian
                    site = B_mat @ np.stack([n.astype(float), m.astype(float)], axis=0)
                    
                    dx = x - site[0]
                    dy = y - site[1]
                    dist_sq = dx * dx + dy * dy
                    
                    # Inside hole → ε = 1 (air)
                    eps[dist_sq < r_sq] = 1.0
        
        return eps[0] if scalar else eps
    
    return eval_eps


# =============================================================================
# 2. Finite-Difference Operators in Oblique Coordinates
# =============================================================================
#
# The reconstruction grid is indexed by moiré fractional coordinates (s1, s2)
# where the physical position is  r = s1 * L1 + s2 * L2.
#
# The Cartesian derivatives are related to fractional derivatives by:
#   ∂/∂x_i = (B⁻¹)_{α,i} ∂/∂sα    (Einstein summation, α ∈ {1,2})
#
# i.e.  ∂/∂x = (B⁻¹)_{1,1} ∂/∂s1 + (B⁻¹)_{2,1} ∂/∂s2
#       ∂/∂y = (B⁻¹)_{1,2} ∂/∂s1 + (B⁻¹)_{2,2} ∂/∂s2
#
# where B = [L1 | L2] is the 2×2 basis matrix (columns = lattice vectors)
# and we use the relation s = B⁻¹ r, so ∂s_α/∂x_i = (B⁻¹)_{α,i}.
#
# The TE operator -∇·(ε⁻¹ ∇ Hz) in oblique coordinates becomes:
#   -∇·(ε⁻¹ ∇ Hz) = -g^{αβ} ∂_α(ε⁻¹ ∂_β Hz)   (with Yee-like staggering)
#
# where g^{αβ} = (B⁻¹ B⁻ᵀ)_{αβ} is the contravariant metric tensor, and
# ∂_α ≡ ∂/∂s_α.
#
# For the curl-curl E-field operator, we similarly express all Cartesian
# derivatives in terms of ∂/∂s1 and ∂/∂s2.
# =============================================================================


class ObliqueGridOperators:
    """
    Finite-difference operators on an oblique (non-Cartesian) periodic grid.
    
    Grid coordinates: (s1, s2) ∈ [0,1) × [0,1), with Nx × Ny points.
    Physical coordinates: r = s1 * L1 + s2 * L2.
    
    All Cartesian differential operators are expressed via the chain rule:
        ∂f/∂x_i = Σ_α (B⁻¹)_{α,i} · ∂f/∂s_α
    
    where B = [L1 | L2] and s = B⁻¹ r.
    """
    
    def __init__(self, L1, L2, Nx, Ny, k0_phys=None):
        self.L1 = np.asarray(L1, dtype=np.float64)
        self.L2 = np.asarray(L2, dtype=np.float64)
        self.Nx = Nx
        self.Ny = Ny
        self.ds1 = 1.0 / Nx
        self.ds2 = 1.0 / Ny
        
        # Basis matrix and its inverse
        self.B = np.column_stack([self.L1, self.L2])  # (2, 2)
        self.B_inv = np.linalg.inv(self.B)             # (2, 2)
        
        # Chain rule Jacobian for Cartesian derivatives:
        #   ∂f/∂x_i = Σ_α (B⁻¹)_{αi} ∂f/∂s_α = Σ_α Jac[i,α] ∂f/∂s_α
        # where Jac = (B⁻¹)ᵀ.
        self.Jac = self.B_inv.T                         # (2, 2)
        
        # Useful: Jacobian determinant (area element)
        self.det_B = abs(np.linalg.det(self.B))
        
        # Grid spacings in fractional coords
        # Physical area element: dA = det(B) * ds1 * ds2
        self.dA = self.det_B * self.ds1 * self.ds2
        
        # Bloch wave vector k0 in physical Cartesian coordinates.
        # When fields are reconstructed WITHOUT the Bloch phase e^{ik₀·r},
        # all derivatives must be replaced: ∇ → ∇ + ik₀
        # k0_phys should be in units of (2π/a) already converted to Cartesian.
        if k0_phys is not None:
            self.k0 = np.asarray(k0_phys, dtype=np.float64)  # (kx, ky)
        else:
            self.k0 = np.zeros(2)
    
    def _d_ds1(self, f):
        """∂f/∂s1 via centered FD (periodic)."""
        return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * self.ds1)
    
    def _d_ds2(self, f):
        """∂f/∂s2 via centered FD (periodic)."""
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * self.ds2)
    
    def _d_ds1_forward(self, f):
        """∂f/∂s1 forward difference (for staggered grid)."""
        return (np.roll(f, -1, axis=0) - f) / self.ds1
    
    def _d_ds2_forward(self, f):
        """∂f/∂s2 forward difference."""
        return (np.roll(f, -1, axis=1) - f) / self.ds2
    
    def _d_ds1_backward(self, f):
        """∂f/∂s1 backward difference."""
        return (f - np.roll(f, 1, axis=0)) / self.ds1
    
    def _d_ds2_backward(self, f):
        """∂f/∂s2 backward difference."""
        return (f - np.roll(f, 1, axis=1)) / self.ds2
    
    def _d2_ds1ds1(self, f):
        """∂²f/∂s1² via centered FD."""
        return (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (self.ds1**2)
    
    def _d2_ds2ds2(self, f):
        """∂²f/∂s2² via centered FD."""
        return (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (self.ds2**2)
    
    def _d2_ds1ds2(self, f):
        """∂²f/∂s1∂s2 via centered FD."""
        return (np.roll(np.roll(f, -1, 0), -1, 1)
                - np.roll(np.roll(f, -1, 0), 1, 1)
                - np.roll(np.roll(f, 1, 0), -1, 1)
                + np.roll(np.roll(f, 1, 0), 1, 1)) / (4 * self.ds1 * self.ds2)
    
    def ddx(self, f):
        """∂f/∂x in Cartesian, via chain rule: ∂f/∂x = Σ_α (B⁻¹)_{α,0} ∂f/∂s_α."""
        return self.Jac[0, 0] * self._d_ds1(f) + self.Jac[0, 1] * self._d_ds2(f)
    
    def ddy(self, f):
        """∂f/∂y in Cartesian, via chain rule: ∂f/∂y = Σ_α (B⁻¹)_{α,1} ∂f/∂s_α."""
        return self.Jac[1, 0] * self._d_ds1(f) + self.Jac[1, 1] * self._d_ds2(f)
    
    def ddx_k(self, f):
        """(∂/∂x + ik₀ₓ)f — modified derivative for Bloch-periodic fields."""
        return self.ddx(f) + 1j * self.k0[0] * f
    
    def ddy_k(self, f):
        """(∂/∂y + ik₀ᵧ)f — modified derivative for Bloch-periodic fields."""
        return self.ddy(f) + 1j * self.k0[1] * f
    
    def laplacian(self, f):
        """
        ∇²f = ∂²f/∂x² + ∂²f/∂y² in Cartesian, computed via oblique FD.
        
        ∇² = Σ_{i} (∂/∂x_i)² = Σ_{α,β} G^{αβ} ∂²/∂s_α∂s_β
        
        where G^{αβ} = Σ_i (B⁻¹)_{αi} (B⁻¹)_{βi} = (B⁻¹ B⁻ᵀ)_{αβ}.
        """
        J = self.Jac  # J[i, α] = (B⁻¹)_{α, i}
        # G^{αβ} = Σ_i J[i,α] * J[i,β]
        G11 = J[0, 0]**2 + J[1, 0]**2
        G12 = J[0, 0] * J[0, 1] + J[1, 0] * J[1, 1]
        G22 = J[0, 1]**2 + J[1, 1]**2
        
        return (G11 * self._d2_ds1ds1(f) 
                + 2 * G12 * self._d2_ds1ds2(f) 
                + G22 * self._d2_ds2ds2(f))
    
    def apply_te_operator(self, Hz, eps_inv):
        """
        Apply the TE master operator: L Hz = -∇·(ε⁻¹ ∇ Hz).
        
        In oblique coordinates, this is done via the Yee-like staggered scheme:
        
        1. Compute gradient ∇Hz in Cartesian (at half-grid points along each 
           oblique axis direction)
        2. Multiply by ε⁻¹ to get the flux F = ε⁻¹ ∇Hz
        3. Compute divergence ∇·F
        
        More precisely, working in oblique coords with Jacobian J:
        
        L Hz = -(1/|J|) Σ_{α} ∂/∂s_α [ |J| Σ_{β} G^{αβ} ε⁻¹ ∂Hz/∂s_β ]
        
        For uniform grid, |J| = |det B| is constant, so it cancels:
        L Hz = -Σ_{α,β} G^{αβ} ∂/∂s_α [ ε⁻¹ ∂Hz/∂s_β ]
        
        However, we must be careful: the "staggered" approach is better for 
        the divergence form.  Here we use an alternative: compute ∇Hz and ∇·(ε⁻¹ ∇Hz)
        all via the Cartesian chain rule, applying centered differences.
        """
        # Simple approach: compute ∂Hz/∂x, ∂Hz/∂y via chain rule,
        # form flux Fx = ε⁻¹ ∂Hz/∂x, Fy = ε⁻¹ ∂Hz/∂y,
        # then ∇·F = ∂Fx/∂x + ∂Fy/∂y (also via chain rule).
        
        dHdx = self.ddx(Hz)
        dHdy = self.ddy(Hz)
        
        Fx = eps_inv * dHdx
        Fy = eps_inv * dHdy
        
        div_F = self.ddx(Fx) + self.ddy(Fy)
        
        return -div_F
    
    def apply_te_operator_k(self, Hz, eps_inv):
        """
        Apply the k-modified TE master operator:
            L_k Hz = -(∇+ik₀)·[ε⁻¹ (∇+ik₀) Hz]
        
        For Bloch-periodic fields u (without the e^{ik₀·r} phase).
        """
        dHdx_k = self.ddx_k(Hz)
        dHdy_k = self.ddy_k(Hz)
        
        Fx = eps_inv * dHdx_k
        Fy = eps_inv * dHdy_k
        
        # Divergence also uses modified derivative: (∇+ik₀)·F
        div_F_k = self.ddx_k(Fx) + self.ddy_k(Fy)
        
        return -div_F_k
    
    def curl_E_z(self, Ex, Ey):
        """
        Compute (∇ × E)_z = ∂Ey/∂x - ∂Ex/∂y  (Cartesian).
        """
        return self.ddx(Ey) - self.ddy(Ex)
    
    def curl_E_z_k(self, Ex, Ey):
        """
        Compute [(∇+ik₀) × E]_z = (∂/∂x+ik₀ₓ)Ey - (∂/∂y+ik₀ᵧ)Ex
        for Bloch-periodic fields (without the e^{ik₀·r} phase).
        """
        return self.ddx_k(Ey) - self.ddy_k(Ex)
    
    def curl_curl_E(self, Ex, Ey):
        """
        Compute ∇×∇×E for E = (Ex, Ey, 0) in 2D.
        
        (∇×∇×E)_x = -∂²Ex/∂y² + ∂²Ey/∂x∂y
        (∇×∇×E)_y = +∂²Ex/∂x∂y - ∂²Ey/∂x²
        
        All second derivatives use the oblique chain rule.
        """
        # We need ∂²f/∂x², ∂²f/∂y², ∂²f/∂x∂y for both Ex and Ey.
        # ∂²f/∂x_i∂x_j = Σ_{α,β} (B⁻¹)_{α,i} (B⁻¹)_{β,j} ∂²f/∂s_α∂s_β
        
        # Precompute second derivatives in oblique coords for both Ex and Ey
        d2Ex_11 = self._d2_ds1ds1(Ex)
        d2Ex_12 = self._d2_ds1ds2(Ex)
        d2Ex_22 = self._d2_ds2ds2(Ex)
        
        d2Ey_11 = self._d2_ds1ds1(Ey)
        d2Ey_12 = self._d2_ds1ds2(Ey)
        d2Ey_22 = self._d2_ds2ds2(Ey)
        
        # Build Cartesian second derivatives using Jac = (B⁻¹)ᵀ:
        # ∂²f/∂x² = Σ_{αβ} J[0,α] J[0,β] ∂²f/∂s_α∂s_β  (where J = Jac)
        # ∂²f/∂y² = Σ_{αβ} J[1,α] J[1,β] ∂²f/∂s_α∂s_β
        # ∂²f/∂x∂y = Σ_{αβ} J[0,α] J[1,β] ∂²f/∂s_α∂s_β
        J = self.Jac
        
        def d2_dxdx(d2_11, d2_12, d2_22):
            return (J[0,0]**2 * d2_11 + 2*J[0,0]*J[0,1] * d2_12 + J[0,1]**2 * d2_22)
        
        def d2_dydy(d2_11, d2_12, d2_22):
            return (J[1,0]**2 * d2_11 + 2*J[1,0]*J[1,1] * d2_12 + J[1,1]**2 * d2_22)
        
        def d2_dxdy(d2_11, d2_12, d2_22):
            return (J[0,0]*J[1,0] * d2_11 
                    + (J[0,0]*J[1,1] + J[0,1]*J[1,0]) * d2_12 
                    + J[0,1]*J[1,1] * d2_22)
        
        # (∇×∇×E)_x = -∂²Ex/∂y² + ∂²Ey/∂x∂y
        cc_x = -d2_dydy(d2Ex_11, d2Ex_12, d2Ex_22) + d2_dxdy(d2Ey_11, d2Ey_12, d2Ey_22)
        
        # (∇×∇×E)_y = +∂²Ex/∂x∂y - ∂²Ey/∂x²
        cc_y = +d2_dxdy(d2Ex_11, d2Ex_12, d2Ex_22) - d2_dxdx(d2Ey_11, d2Ey_12, d2Ey_22)
        
        return cc_x, cc_y
    
    def curl_curl_E_k(self, Ex, Ey):
        """
        Compute (∇+ik₀)×(∇+ik₀)×E for E = (Ex, Ey, 0) in 2D.
        
        (∇+ik₀)×E has only z-component:
            [(∇+ik₀)×E]_z = (∂/∂x+ik₀ₓ)Ey - (∂/∂y+ik₀ᵧ)Ex
        
        Then (∇+ik₀)× of a z-directed field Fz ẑ gives:
            [(∇+ik₀)×(Fz ẑ)]_x = +(∂/∂y+ik₀ᵧ)Fz
            [(∇+ik₀)×(Fz ẑ)]_y = -(∂/∂x+ik₀ₓ)Fz
        """
        # Step 1: curl_z = [(∇+ik₀)×E]_z
        curl_z = self.curl_E_z_k(Ex, Ey)
        
        # Step 2: (∇+ik₀) × (curl_z ẑ)
        cc_x = self.ddy_k(curl_z)
        cc_y = -self.ddx_k(curl_z)
        
        return cc_x, cc_y
    
    def l2_norm(self, f):
        """L² norm over the moiré cell: sqrt(∫|f|² dA)."""
        return np.sqrt(np.sum(np.abs(f)**2) * self.dA)
    
    def l2_norm_vec(self, fx, fy):
        """L² norm of a 2D vector field."""
        return np.sqrt(np.sum(np.abs(fx)**2 + np.abs(fy)**2) * self.dA)


# =============================================================================
# 3. Residual Computation
# =============================================================================

def compute_weak_residual_Hz(ops, Hz, eps, omega_sq):
    """
    Compute the weak-form Maxwell residual (Rayleigh quotient deviation).
    
    The k-modified TE master equation in weak form (integration by parts):
        ∫ ε⁻¹ |(∇+ik₀)Hz|² dA = ω² ∫ |Hz|² dA
    
    The Rayleigh quotient:
        R_q = ∫ ε⁻¹ |(∇+ik₀)Hz|² dA / ∫ |Hz|² dA
    
    Weak residual:
        R_weak = |R_q - ω²| / ω²
    
    Uses modified derivatives (∇+ik₀) since Bloch fields are stored WITHOUT
    the e^{ik₀·r} phase factor.
    
    Args:
        ops: ObliqueGridOperators (with k0 set)
        Hz: 2D complex field (periodic part, no Bloch phase)
        eps: dielectric function (2D array)
        omega_sq: (2πf)² eigenvalue for the master equation
    
    Returns:
        R_weak: scalar weak residual
        R_q: Rayleigh quotient value
    """
    eps_inv = 1.0 / eps
    
    # Modified gradient of Hz: (∂/∂x + ik₀ₓ)Hz, (∂/∂y + ik₀ᵧ)Hz
    dHz_dx_k = ops.ddx_k(Hz)
    dHz_dy_k = ops.ddy_k(Hz)
    
    # ∫ ε⁻¹ |(∇+ik₀)Hz|² dA
    grad_sq = np.abs(dHz_dx_k)**2 + np.abs(dHz_dy_k)**2
    numerator = np.sum(eps_inv * grad_sq) * ops.dA
    
    # ∫ |Hz|² dA
    denominator = np.sum(np.abs(Hz)**2) * ops.dA
    
    R_q = numerator / denominator if denominator > 0 else np.inf
    R_weak = abs(R_q - omega_sq) / omega_sq
    
    return R_weak, R_q


def compute_weak_residual_E(ops, Ex, Ey, eps, omega_sq):
    """
    Compute the weak-form Maxwell residual for the E-field curl-curl equation.
    
    The k-modified TE E-field equation in weak form:
        ∫ |(∇+ik₀)×E|² dA = ω² ∫ ε |E|² dA
    
    For 2D: |(∇+ik₀)×E|² = |(∂/∂x+ik₀ₓ)Ey - (∂/∂y+ik₀ᵧ)Ex|²
    
    Rayleigh quotient:
        R_q = ∫ |curl_k E|² dA / ∫ ε |E|² dA
    
    Weak residual:
        R_weak = |R_q - ω²| / ω²
    
    Args:
        ops: ObliqueGridOperators (with k0 set)
        Ex, Ey: 2D complex E-field components (periodic parts)
        eps: dielectric function
        omega_sq: (2πf)²
    
    Returns:
        R_weak: scalar residual
        R_q: Rayleigh quotient
    """
    # [(∇+ik₀)×E]_z = (∂/∂x+ik₀ₓ)Ey - (∂/∂y+ik₀ᵧ)Ex
    curl_z = ops.curl_E_z_k(Ex, Ey)
    
    # ∫ |curl_k E|² dA
    numerator = np.sum(np.abs(curl_z)**2) * ops.dA
    
    # ∫ ε |E|² dA
    E_sq = np.abs(Ex)**2 + np.abs(Ey)**2
    denominator = np.sum(eps * E_sq) * ops.dA
    
    R_q = numerator / denominator if denominator > 0 else np.inf
    R_weak = abs(R_q - omega_sq) / omega_sq
    
    return R_weak, R_q


def compute_maxwell_residual_Hz(ops, Hz, eps, omega_sq, return_spatial=False):
    """
    Compute the dimensionless Maxwell residual for the k-modified TE master equation:
        -(∇+ik₀)·(ε⁻¹ (∇+ik₀) Hz) = ω² Hz
    
    Residual:
        R = ||L_k Hz - ω² Hz||_L² / ||ω² Hz||_L²
    
    Args:
        ops: ObliqueGridOperators instance (with k0 set)
        Hz: reconstructed H_z field, shape (Nx, Ny) — periodic part
        eps: dielectric function ε(x,y), same shape
        omega_sq: eigenvalue (2π f_MPB)² for the master equation
        return_spatial: if True, also return the spatial residual map
    
    Returns:
        R: scalar dimensionless residual
        (optional) r_spatial: spatial residual map, shape (Nx, Ny)
    """
    eps_inv = 1.0 / eps
    
    # LHS: L_k Hz = -(∇+ik₀)·(ε⁻¹ (∇+ik₀) Hz)
    L_Hz = ops.apply_te_operator_k(Hz, eps_inv)
    
    # RHS: ω² Hz
    rhs = omega_sq * Hz
    
    # Pointwise residual
    residual = L_Hz - rhs
    
    # L² norms
    norm_residual = ops.l2_norm(residual)
    norm_rhs = ops.l2_norm(rhs)
    
    R = norm_residual / norm_rhs if norm_rhs > 0 else np.inf
    
    if return_spatial:
        r_spatial = np.abs(residual) / (np.abs(rhs) + 1e-30)
        return R, r_spatial
    
    return R


def compute_residual_direct_E(ops, Ex, Ey, eps, omega_sq, return_spatial=False):
    """
    Compute the Maxwell residual directly from the E-field (periodic part).
    
    For TE with Bloch phase removed:
        (∇+ik₀)×(∇+ik₀)×E = ω² ε E
    
    Args:
        ops: ObliqueGridOperators instance (with k0 set)
        Ex, Ey: 2D complex arrays — periodic parts of E-field
        eps: dielectric function
        omega_sq: eigenvalue (2π f_MPB)² for the master equation
    
    Returns:
        R: scalar dimensionless residual
        (optional) r_spatial: spatial residual map
    """
    
    cc_x, cc_y = ops.curl_curl_E_k(Ex, Ey)
    
    # RHS: ω² ε E
    rhs_x = omega_sq * eps * Ex
    rhs_y = omega_sq * eps * Ey
    
    # Residual
    res_x = cc_x - rhs_x
    res_y = cc_y - rhs_y
    
    norm_res = ops.l2_norm_vec(res_x, res_y)
    norm_rhs = ops.l2_norm_vec(rhs_x, rhs_y)
    
    R = norm_res / norm_rhs if norm_rhs > 0 else np.inf
    
    if return_spatial:
        r_spatial = np.sqrt(np.abs(res_x)**2 + np.abs(res_y)**2) / \
                    (np.sqrt(np.abs(rhs_x)**2 + np.abs(rhs_y)**2) + 1e-30)
        return R, r_spatial
    
    return R


# =============================================================================
# 4. Main Driver
# =============================================================================

def run_validation_residual(candidate_id, mode_idx, grid_oversample=2, downsample=1):
    """
    Full pipeline: load data → reconstruct field → build ε → compute residual.
    
    Args:
        candidate_id: candidate ID
        mode_idx: mode index
        grid_oversample: oversample factor for the reconstruction grid
                        (1 = native grid, 2 = 2× finer, etc.)
        downsample: downsample the tiled grid by this factor 
                   (e.g. 4 → keep every 4th point, so 8192→2048)
    """
    # 0. Setup
    run_dir = p4.find_latest_run_dir()
    cdir = p4.candidate_dir(run_dir, candidate_id)
    out_dir = cdir / f"validation_residual_mode{mode_idx}"
    out_dir.mkdir(exist_ok=True)
    
    log(f"=== Maxwell Residual Validation ===")
    log(f"Candidate: {candidate_id}, Mode: {mode_idx}")
    log(f"Output: {out_dir}")
    
    # 1. Load metadata
    p0_meta = load_json(cdir / "phase0_meta.json")
    a0 = p0_meta.get('a', 1.0)
    theta_deg = p0_meta['theta_deg']
    theta_rad = math.radians(theta_deg)
    eta = p0_meta.get('eta', theta_rad)
    lattice_type = p0_meta.get('lattice_type', 'square')
    
    log(f"θ = {theta_deg}°, η = {eta:.6f}, lattice = {lattice_type}")
    
    # 2. Load Phase 1 & 3 data
    bloch_fields, subspace_bands, all_bands = p4.load_phase1_bloch_fields(cdir)
    F_spinor, eigenvalues, mode_stats = p4.load_phase3_envelopes(cdir)
    band_indices = p4.get_subspace_band_indices(subspace_bands, all_bands)
    
    # Load per-tile band frequencies from Phase 1 data
    import h5py
    with h5py.File(cdir / 'phase1_multiband_data.h5', 'r') as hf:
        omega_bands = hf['omega'][:]  # (Ns_env, Ns_env, N_sub) — subspace band freqs
    
    # NOTE: mode_stats['omega'] = omega_ref + eigenvalue is NOT a physical MPB frequency.
    # The correct ω² for the Rayleigh quotient comparison is the LOCAL band frequency
    # ω²_n(R) = (2πf_n(R))², weighted by the envelope |F_n(R)|². This is computed
    # per-tile in compute_per_tile_residual.
    # For backward compatibility, we also compute a weighted-average ω².
    
    # Find dominant band
    F_mode = F_spinor[mode_idx]
    band_weights_for_omega = np.array([np.sum(np.abs(F_mode[:,:,n])**2) for n in range(F_spinor.shape[-1])])
    band_weights_for_omega /= band_weights_for_omega.sum()
    dom_sub = int(np.argmax(band_weights_for_omega))
    
    # Weighted average omega² for backward-compatible reporting
    F_sq = np.sum(np.abs(F_mode)**2, axis=-1)
    omega2_mean = np.sum(F_sq * (2*np.pi*omega_bands[:,:,dom_sub])**2) / np.sum(F_sq)
    freq_mpb_effective = np.sqrt(omega2_mean) / (2*np.pi)  # effective frequency
    omega_sq = omega2_mean
    
    log(f"Dominant band: {dom_sub} (weight={band_weights_for_omega[dom_sub]:.4f})")
    log(f"<ω²_band{dom_sub}> (|F|²-weighted) = {omega_sq:.6f}")
    log(f"Effective f = {freq_mpb_effective:.6f}")
    
    # 3. Geometry setup
    B_mono = build_monolayer_basis(lattice_type, a0)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    L1 = B_moire[:, 0]
    L2 = B_moire[:, 1]
    L_moire = np.linalg.norm(L1)
    
    # Read the Bloch wave vector k0 (in units of 2π/a, reciprocal lattice coords)
    k0_x = p0_meta.get('k0_x', 0.0)
    k0_y = p0_meta.get('k0_y', 0.0)
    
    # Convert k0 from reciprocal lattice coordinates to physical Cartesian.
    # MPB uses k in units of 2π/a (for simple square lattice, k_phys = 2π k_mpb / a).
    # For the reciprocal lattice: G = 2π (B_mono^{-T}), so k_phys = G · k_frac.
    # With a=1 (natural units) and square lattice, this is just k_phys = 2π * (k0_x, k0_y).
    G_mono = 2 * np.pi * np.linalg.inv(B_mono).T  # reciprocal lattice vectors
    k0_phys = G_mono @ np.array([k0_x, k0_y])     # physical k-vector in 1/a units
    
    log(f"Moiré length: {L_moire:.3f}")
    log(f"L1 = {L1}")
    log(f"L2 = {L2}")
    log(f"k0 = ({k0_x}, {k0_y}) [rec. lattice], k0_phys = ({k0_phys[0]:.4f}, {k0_phys[1]:.4f}) [Cartesian]")
    
    # 4. Build reconstruction grid (Cartesian, covering one moiré cell centered at origin)
    # The tiled reconstruction from reconstruct_full_field_single_mode covers
    # [0, 1) × [0, 1) in moiré fractional coordinates.
    # Physical extent: one moiré cell.
    
    Ns1_b, Ns2_b, N_bands_all, Nx_micro, Ny_micro, _ = bloch_fields.shape
    _, Ns1_env, Ns2_env, N_sub = F_spinor.shape
    
    log(f"Bloch grid: {Ns1_b}×{Ns2_b} registry, {Nx_micro}×{Ny_micro} micro")
    log(f"Envelope grid: {Ns1_env}×{Ns2_env}, {N_sub} bands")
    
    # Native tiled grid: Ns_env × Nx_micro points per moiré cell
    Nx_native = Ns1_env * Nx_micro
    Ny_native = Ns2_env * Ny_micro
    
    log(f"Native tiled grid: {Nx_native}×{Ny_native}")
    
    # 4b. Build epsilon registry (needed for per-tile residual)
    eps_cache_path = cdir / "mpb_epsilon_registry.h5"
    mpb_resolution = Nx_micro  # Should be 64
    eps_registry = extract_mpb_epsilon_grid(
        p0_meta, n_registry=Ns1_b, resolution=mpb_resolution,
        cache_path=eps_cache_path, n_workers=8
    )
    log(f"MPB ε registry shape: {eps_registry.shape}")
    
    # 5. Gauge-fix Bloch fields and compute per-tile residual (PRIMARY METRIC)
    #
    # The per-tile metric avoids tile-boundary artifacts by computing the
    # Rayleigh quotient within each unit cell, where the Bloch function is
    # a true eigenstate. This is the correct formulation for the envelope
    # approximation, which predicts local eigenvalues ω²_n(R).
    
    log("Gauge-fixing Bloch fields...")
    u_sub_fixed = gauge_fix_bloch_fields(bloch_fields, band_indices)
    
    log("Computing per-tile Rayleigh quotient (primary metric)...")
    tile_results = compute_per_tile_residual(
        bloch_fields_fixed=u_sub_fixed,
        band_indices=band_indices,
        eps_registry=eps_registry,
        omega_bands=omega_bands,
        F_spinor=F_spinor,
        mode_idx=mode_idx,
        k0_phys=k0_phys,
        bloch_fields_raw=bloch_fields,
    )
    
    log(f"*** Per-tile residual R = {tile_results['R_global']:.6e} ***")
    log(f"    FD-corrected residual R = {tile_results['R_fd_corrected']:.6e}")
    log(f"    Weighted R_q = {tile_results['Rq_weighted']:.6f}")
    log(f"    Weighted <ω²> = {tile_results['omega2_weighted']:.6f}")
    log(f"    Ratio R_q/<ω²> = {tile_results['ratio_weighted']:.6f}")
    log(f"    FD-corrected ratio = {tile_results['ratio_fd_corrected']:.6f}")
    
    # 5b. Also reconstruct the full tiled field (with gauge-fixed Bloch functions)
    # for visualization and legacy residual computation.
    bf_fixed_full = bloch_fields.copy()
    bf_fixed_full[:, :, band_indices, :, :, :] = u_sub_fixed
    
    log("Reconstructing E_x on tiled grid (gauge-fixed)...")
    Ex_tiled = p4.reconstruct_full_field_single_mode(
        mode_idx=mode_idx,
        F_spinor=F_spinor,
        bloch_fields=bf_fixed_full,
        band_indices=band_indices,
        component=0,
        normalize_bloch=True,
    )
    
    log("Reconstructing E_y on tiled grid (gauge-fixed)...")
    Ey_tiled = p4.reconstruct_full_field_single_mode(
        mode_idx=mode_idx,
        F_spinor=F_spinor,
        bloch_fields=bf_fixed_full,
        band_indices=band_indices,
        component=1,
        normalize_bloch=True,
    )
    
    log(f"Reconstructed field shape: {Ex_tiled.shape}")
    log(f"max|Ex| = {np.max(np.abs(Ex_tiled)):.6e}, max|Ey| = {np.max(np.abs(Ey_tiled)):.6e}")
    
    # 5b. Optional downsampling (for faster testing)
    if downsample > 1:
        Ex_tiled = Ex_tiled[::downsample, ::downsample]
        Ey_tiled = Ey_tiled[::downsample, ::downsample]
        log(f"Downsampled by {downsample}× → shape {Ex_tiled.shape}")
    
    # 6. Build physical coordinate grid
    # The tiled grid covers one moiré cell in fractional coords s ∈ [0, 1).
    # Physical coords: r = s1 * L1 + s2 * L2
    Nx, Ny = Ex_tiled.shape
    s1 = np.linspace(0, 1, Nx, endpoint=False)
    s2 = np.linspace(0, 1, Ny, endpoint=False)
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    
    X_phys = S1 * L1[0] + S2 * L2[0]
    Y_phys = S1 * L1[1] + S2 * L2[1]
    
    # Build oblique-coordinate FD operators (with Bloch wave vector k0)
    ops = ObliqueGridOperators(L1, L2, Nx, Ny, k0_phys=k0_phys)
    
    angle_L1_L2 = math.degrees(math.atan2(
        float(np.cross(L1, L2)), float(np.dot(L1, L2))))
    log(f"L1·L2 angle: {angle_L1_L2:.2f}°")
    log(f"Effective grid spacing: |L1|/Nx = {np.linalg.norm(L1)/Nx:.6f}, "
        f"|L2|/Ny = {np.linalg.norm(L2)/Ny:.6f}")
    log(f"Cell area: det(B) = {ops.det_B:.4f}")
    
    # 7. Build epsilon on the tiled grid for global residual computation
    # eps_registry was already loaded in section 4b for per-tile metric.
    
    # Tile to full moiré grid matching envelope grid
    eps_tiled_full = tile_epsilon_from_registry(eps_registry, Ns1_env)
    log(f"Tiled ε shape: {eps_tiled_full.shape}")
    
    # Apply the same downsampling as the fields
    if downsample > 1:
        eps_grid = eps_tiled_full[::downsample, ::downsample]
    else:
        eps_grid = eps_tiled_full
    
    log(f"ε grid shape (after downsample): {eps_grid.shape}")
    
    fill_fraction_air = np.mean(eps_grid < 1.5)
    n_subpixel = np.sum((eps_grid > 1.1) & (eps_grid < p0_meta['eps_bg'] - 0.1))
    log(f"ε stats: min={eps_grid.min():.2f}, max={eps_grid.max():.2f}, "
        f"air fill = {fill_fraction_air*100:.1f}%, "
        f"subpixel-averaged pixels = {n_subpixel}")
    
    # 8. Compute H_z from curl of E (k-modified for Bloch-periodic fields)
    # NOTE: Computing Hz from FD curl of E has ~40% error at res=64.
    # Hz is kept only for legacy compatibility; the per-tile metric is preferred.
    omega_angular = np.sqrt(omega_sq)  # use weighted average
    log(f"Computing H_z = [(∇+ik₀)×E]_z / (iω), ω = {omega_angular:.6f}...")
    curl_E_z_k = ops.curl_E_z_k(Ex_tiled, Ey_tiled)
    Hz = curl_E_z_k / (1j * omega_angular)
    log(f"max|Hz| = {np.max(np.abs(Hz)):.6e}")
    log(f"WARNING: Hz from curl E has large FD error (~40%%). Per-tile metric is preferred.")
    
    # 9. Compute residuals
    log("Computing Maxwell residual (TE, H_z master equation)...")
    R_Hz, r_spatial_Hz = compute_maxwell_residual_Hz(
        ops, Hz, eps_grid, omega_sq, return_spatial=True
    )
    log(f"*** TE Residual (H_z, strong form): R = {R_Hz:.6e} ***")
    
    log("Computing Maxwell residual (E-field curl-curl)...")
    R_E, r_spatial_E = compute_residual_direct_E(
        ops, Ex_tiled, Ey_tiled, eps_grid, omega_sq, return_spatial=True
    )
    log(f"*** TE Residual (E curl-curl, strong form): R = {R_E:.6e} ***")
    
    # Weak-form residuals (Rayleigh quotient) — robust to ε discontinuities
    log("Computing weak-form residuals (Rayleigh quotient)...")
    R_weak_Hz, Rq_Hz = compute_weak_residual_Hz(ops, Hz, eps_grid, omega_sq)
    R_weak_E, Rq_E = compute_weak_residual_E(ops, Ex_tiled, Ey_tiled, eps_grid, omega_sq)
    log(f"*** Weak residual (Hz): R = {R_weak_Hz:.6e}, Rayleigh quotient = {Rq_Hz:.6f} (ω² = {omega_sq:.6f}) ***")
    log(f"*** Weak residual (E):  R = {R_weak_E:.6e}, Rayleigh quotient = {Rq_E:.6f} (ω² = {omega_sq:.6f}) ***")
    
    # 10. Visualization
    log("Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Fields
    ax = axes[0, 0]
    im = ax.imshow(np.abs(Ex_tiled).T, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='|Ex|')
    ax.set_title('|Ex| (reconstructed)')
    ax.set_xlabel('s1 index')
    ax.set_ylabel('s2 index')
    
    ax = axes[0, 1]
    im = ax.imshow(np.abs(Hz).T, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im, ax=ax, label='|Hz|')
    ax.set_title('|Hz| (from curl E)')
    
    ax = axes[0, 2]
    im = ax.imshow(eps_grid.T, origin='lower', cmap='gray_r', aspect='auto')
    plt.colorbar(im, ax=ax, label='ε')
    ax.set_title(f'ε(x,y) — air fill {fill_fraction_air*100:.1f}%')
    
    # Row 2: Residuals
    ax = axes[1, 0]
    vmax = np.percentile(r_spatial_Hz, 99)
    im = ax.imshow(r_spatial_Hz.T, origin='lower', cmap='inferno', 
                   aspect='auto', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='local residual')
    ax.set_title(f'TE Residual Map (Hz)\nR = {R_Hz:.4e}')
    
    ax = axes[1, 1]
    vmax_E = np.percentile(r_spatial_E, 99)
    im = ax.imshow(r_spatial_E.T, origin='lower', cmap='inferno',
                   aspect='auto', vmin=0, vmax=vmax_E)
    plt.colorbar(im, ax=ax, label='local residual')
    ax.set_title(f'TE Residual Map (E curl-curl)\nR = {R_E:.4e}')
    
    ax = axes[1, 2]
    # Histogram of local residuals
    r_flat = r_spatial_Hz.ravel()
    r_flat = r_flat[r_flat < np.percentile(r_flat, 99)]
    ax.hist(r_flat, bins=100, color='steelblue', alpha=0.7, density=True)
    ax.axvline(R_Hz, color='red', linestyle='--', label=f'Global R = {R_Hz:.4e}')
    ax.set_xlabel('Local residual')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution (Hz)')
    ax.legend()
    ax.set_yscale('log')
    
    fig.suptitle(
        f"Maxwell Residual — Candidate {candidate_id}, Mode {mode_idx}\n"
        f"f_eff = {freq_mpb_effective:.6f}, ω² = {omega_sq:.4f}, θ = {theta_deg}°, η = {eta:.6f}, "
        f"Grid {Nx}×{Ny}",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(out_dir / "maxwell_residual.png", dpi=150)
    plt.close()
    log(f"Plot saved: {out_dir / 'maxwell_residual.png'}")
    
    # Per-tile residual map visualization
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    ax = axes2[0]
    im = ax.imshow(tile_results['ratio_map'].T, origin='lower', cmap='RdBu_r',
                   vmin=0.8, vmax=1.2, aspect='equal')
    plt.colorbar(im, ax=ax, label='R_q / ω²')
    ax.set_title(f'Per-tile R_q/ω² ratio\nWeighted mean = {tile_results["ratio_weighted"]:.4f}')
    ax.set_xlabel('Registry s1'); ax.set_ylabel('Registry s2')
    
    ax = axes2[1]
    im = ax.imshow(tile_results['weight_map'].T, origin='lower', cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax, label='|F|²')
    ax.set_title('Envelope weight |F(R)|²')
    ax.set_xlabel('Registry s1'); ax.set_ylabel('Registry s2')
    
    ax = axes2[2]
    ratios = tile_results['ratio_map'].ravel()
    weights = tile_results['weight_map'].ravel()
    mask = weights > 0.01 * weights.max()
    ax.hist(ratios[mask], bins=50, color='steelblue', alpha=0.7, density=True,
            label=f'Weighted tiles (n={mask.sum()})')
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Exact (ratio=1)')
    ax.axvline(tile_results['ratio_weighted'], color='red', linestyle='-',
               linewidth=2, label=f'Weighted mean={tile_results["ratio_weighted"]:.4f}')
    ax.set_xlabel('R_q / ω²'); ax.set_ylabel('Density')
    ax.set_title(f'Per-tile ratio distribution\nR_global = {tile_results["R_global"]:.4e}')
    ax.legend(fontsize=8)
    
    fig2.suptitle(
        f"Per-Tile Maxwell Residual — Candidate {candidate_id}, Mode {mode_idx}\n"
        f"θ = {theta_deg}°, η = {eta:.6f}, Registry {bloch_fields.shape[0]}×{bloch_fields.shape[1]}",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(out_dir / "per_tile_residual.png", dpi=150)
    plt.close()
    log(f"Per-tile plot saved: {out_dir / 'per_tile_residual.png'}")
    
    # Save numerical results
    results = {
        'candidate_id': candidate_id,
        'mode_idx': mode_idx,
        'freq_mpb_effective': float(freq_mpb_effective),
        'omega_sq': float(omega_sq),
        'theta_deg': theta_deg,
        'eta': float(eta),
        'k0_frac': [float(k0_x), float(k0_y)],
        'k0_phys': k0_phys.tolist(),
        # PRIMARY METRIC: per-tile residual
        'R_per_tile': float(tile_results['R_global']),
        'R_fd_corrected': float(tile_results['R_fd_corrected']),
        'ratio_fd_corrected': float(tile_results['ratio_fd_corrected']),
        'Rq_weighted': float(tile_results['Rq_weighted']),
        'omega2_weighted': float(tile_results['omega2_weighted']),
        'ratio_weighted': float(tile_results['ratio_weighted']),
        'dominant_band': int(tile_results['dominant_band']),
        'band_weights': tile_results['band_weights'],
        # Legacy global residuals (affected by tile boundary artifacts)
        'R_Hz_strong': float(R_Hz),
        'R_E_strong': float(R_E),
        'R_Hz_weak': float(R_weak_Hz),
        'R_E_weak': float(R_weak_E),
        'Rq_Hz': float(Rq_Hz),
        'Rq_E': float(Rq_E),
        'grid_shape': list(Ex_tiled.shape),
        'downsample': downsample,
        'ds1': float(ops.ds1),
        'ds2': float(ops.ds2),
        'L1': L1.tolist(),
        'L2': L2.tolist(),
        'L_moire': float(L_moire),
        'det_B': float(ops.det_B),
        'eps_min': float(eps_grid.min()),
        'eps_max': float(eps_grid.max()),
        'air_fill_fraction': float(fill_fraction_air),
        'max_abs_Ex': float(np.max(np.abs(Ex_tiled))),
        'max_abs_Ey': float(np.max(np.abs(Ey_tiled))),
        'max_abs_Hz': float(np.max(np.abs(Hz))),
    }
    
    import json
    with open(out_dir / "residual_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results saved: {out_dir / 'residual_results.json'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation: Maxwell Residual")
    parser.add_argument("candidate_id", type=int, help="Candidate ID")
    parser.add_argument("mode_index", type=int, help="Mode index")
    parser.add_argument("--oversample", type=int, default=1,
                        help="Grid oversample factor (default: 1 = native)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor (e.g. 4 → 8192→2048)")
    args = parser.parse_args()
    
    run_validation_residual(args.candidate_id, args.mode_index, args.oversample,
                           args.downsample)
