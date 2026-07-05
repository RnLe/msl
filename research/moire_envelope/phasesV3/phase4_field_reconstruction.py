#!/usr/bin/env python3
"""
Phase 4: Full Field Reconstruction from Envelope Modes

Reconstructs the full electromagnetic field H(r) from:
  - Envelope functions F_n(R) from Phase 3
  - Bloch functions u_n(r; R) from Phase 1

The central two-scale ansatz:
    H(r) = e^{i k_0 · r} Σ_n F_n(R) · u_n(r; R)

where R = η·r is the slow (moiré) coordinate.

FEATURES:
1. Memory-efficient: Reconstructs one mode at a time, plots, discards.
2. Reusable reconstruction method for Phase 6 Meep initialization.
3. 10×10 visualization grid showing |H_z|² with microscopic Bloch structure.

USAGE:
    python phasesV3/phase4_field_reconstruction.py [candidate_id]
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
import h5py

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json


def log(msg):
    """Simple logging."""
    print(f"[Phase4] {msg}")


def find_latest_run_dir(base_name="phase0_mpb_v3"):
    """Find the latest run directory in runsV3 matching the base name."""
    runs_dir = PROJECT_ROOT / "runsV3"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        
    candidates = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(base_name)])
    if not candidates:
        raise FileNotFoundError(f"No runs found matching {base_name} in {runs_dir}")
        
    return candidates[-1]


def load_phase1_bloch_fields(cdir):
    """
    Load Bloch fields u_n(r; R) from Phase 1.
    
    Returns:
        bloch_fields: array of shape (Ns1, Ns2, N_bands, Nx, Ny, 3)
            - (Ns1, Ns2): moiré grid (registry sampling)
            - N_bands: number of bands in all_bands
            - (Nx, Ny): microscopic grid within monolayer cell
            - 3: field components (x, y, z)
        subspace_bands: indices of subspace bands within all_bands
        all_bands: all band indices computed
    """
    h5_path = cdir / "phase1_multiband_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 1 data not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        if 'bloch_fields' not in hf:
            raise KeyError("bloch_fields not found in Phase 1 data. "
                          "Re-run Phase 1 with export_bloch_fields=True")
        
        bloch_fields = hf['bloch_fields'][:]
        
        # Get band mapping
        if 'subspace_bands' in hf.attrs:
            subspace_bands = hf.attrs['subspace_bands']
        else:
            subspace_bands = None
            
        if 'all_bands' in hf.attrs:
            all_bands = hf.attrs['all_bands']
        else:
            all_bands = None
    
    log(f"  Loaded Bloch fields: shape {bloch_fields.shape}")
    
    # === MEMORY OPTIMIZATION: Reduce precision to float32 ===
    # This halves memory usage (8GB -> 4GB equivalent)
    if bloch_fields.dtype == np.float64:
        log("  Optimizing memory: Converting Bloch fields to float32")
        bloch_fields = bloch_fields.astype(np.float32)
        
    return bloch_fields, subspace_bands, all_bands


def load_phase3_envelopes(cdir):
    """
    Load envelope functions F_n(R) from Phase 3.
    
    Returns:
        F_spinor: array of shape (n_modes, Ns1, Ns2, N_subspace)
            - n_modes: number of computed envelope modes
            - (Ns1, Ns2): moiré grid
            - N_subspace: number of bands in subspace
        eigenvalues: array of shape (n_modes,)
        mode_stats: list of dicts with mode statistics
    """
    h5_path = cdir / "phase3_multiband_modes.h5"
    json_path = cdir / "phase3_mode_stats.json"
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 3 data not found: {h5_path}")
    
    with h5py.File(h5_path, 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        
        if 'F_spinor' in hf:
            F_spinor = hf['F_spinor'][:]
        elif 'F_envelope' in hf:
            F_spinor = hf['F_envelope'][:]
        elif 'eigenvectors' in hf:
            log("  Warning: Using raw eigenvectors, may need reshaping")
            F_spinor = hf['eigenvectors'][:]
        else:
            raise KeyError("Could not find F_spinor or eigenvectors in Phase 3 data")
    
    mode_stats = []
    if json_path.exists():
        mode_stats = load_json(json_path)
    
    log(f"  Loaded envelopes: {F_spinor.shape[0]} modes, grid {F_spinor.shape[1]}x{F_spinor.shape[2]}, {F_spinor.shape[3]} bands")
    return F_spinor, eigenvalues, mode_stats


def get_subspace_band_indices(subspace_bands, all_bands):
    """
    Map subspace band indices to indices within the bloch_fields array.
    
    Phase 1 stores bloch_fields for all_bands (e.g., bands [3,4,5,6]).
    Phase 3 uses subspace_bands (e.g., bands [4,5]).
    We need to find which indices in bloch_fields correspond to subspace_bands.
    
    Returns:
        band_indices: list of indices into bloch_fields dimension
    """
    if subspace_bands is None or all_bands is None:
        # Fallback: assume 1-to-1 mapping
        return list(range(len(subspace_bands) if subspace_bands is not None else 4))
    
    all_bands_list = list(all_bands)
    indices = []
    for sb in subspace_bands:
        if sb in all_bands_list:
            indices.append(all_bands_list.index(sb))
        else:
            raise ValueError(f"Subspace band {sb} not found in all_bands {all_bands}")
    
    return indices


def reconstruct_full_field_single_mode(
    mode_idx: int,
    F_spinor: np.ndarray,
    bloch_fields: np.ndarray,
    band_indices: list,
    component: int = 2,
    include_bloch_phase: bool = False,
    k0: np.ndarray = None,
    bloch_interp_cache: dict = None,
    normalize_bloch: bool = True,
) -> np.ndarray:
    """
    Reconstruct the full electromagnetic field H(r) for a single mode.
    
    This is the CORE REUSABLE METHOD for Phase 6 Meep initialization.
    
    The reconstruction formula:
        H(r, R) = e^{i k_0 · r} Σ_n F_n(R) · u_n(r; R)
    
    Args:
        mode_idx: Index of the mode in F_spinor
        F_spinor: Envelope functions, shape (n_modes, Ns1, Ns2, N_subspace)
        bloch_fields: Bloch functions, shape (Ns1_b, Ns2_b, N_bands, Nx, Ny, 3)
            Note: Bloch grid may differ from envelope grid; interpolation is done.
        band_indices: Mapping from subspace index to bloch_fields band index
        component: Field component to extract (0=x, 1=y, 2=z). Default z for TM.
        include_bloch_phase: If True, include e^{i k_0 · r} phase factor.
        k0: Crystal momentum (2D vector). Required if include_bloch_phase=True.
        bloch_interp_cache: Optional dict to cache interpolated Bloch fields.
            If provided, will store/retrieve interpolated u_n to avoid recomputation.
        normalize_bloch: If True (default), normalize each Bloch function so that
            <|u|²> = 1 over the unit cell. This ensures |E|² ≈ |F|² on average.
    
    Returns:
        H_full: Complex field array of shape (Ns1 * Nx, Ns2 * Ny)
            This is the full field on the tiled moiré cell.
            The field is complex-valued, suitable for Meep initialization.
    
    Notes:
        - For |H|² visualization, take np.abs(H_full)**2
        - For Meep initialization, use H_full.real and H_full.imag separately
        - The output grid has (Ns1 * Nx) x (Ns2 * Ny) points covering one moiré cell
    """
    from scipy.ndimage import zoom
    
    # Get dimensions
    Ns1_b, Ns2_b, N_bands_all, Nx, Ny, _ = bloch_fields.shape
    Ns1_env, Ns2_env, N_subspace = F_spinor[mode_idx].shape
    
    # Extract envelope for this mode: shape (Ns1_env, Ns2_env, N_subspace)
    F_mode = F_spinor[mode_idx]
    
    # Check if we need to interpolate Bloch fields to envelope grid
    need_interp = (Ns1_b != Ns1_env) or (Ns2_b != Ns2_env)
    
    # Use envelope grid size for output
    Ns1, Ns2 = Ns1_env, Ns2_env
    
    # Initialize output: full tiled field (complex64 for speed/memory)
    # Using complex64 (8 bytes/px) instead of complex128 (16 bytes/px)
    H_full = np.zeros((Ns1, Ns2, Nx, Ny), dtype=np.complex64)
    
    # Zoom factors for interpolation (if needed)
    if need_interp:
        zoom_factors = (Ns1_env / Ns1_b, Ns2_env / Ns2_b, 1, 1)  # Only zoom registry dims
    
    # Compute normalization factors for Bloch functions if requested
    # For TE modes (Ex, Ey), we compute <|u_x|² + |u_y|²> over the unit cell
    # and scale so that this average equals 1
    if normalize_bloch:
        # Check cache for normalization factors
        norm_cache_key = "bloch_norm_factors"
        if bloch_interp_cache is not None and norm_cache_key in bloch_interp_cache:
            norm_factors = bloch_interp_cache[norm_cache_key]
        else:
            # Compute <|u_x|² + |u_y|²> for each registry point and band
            # Shape: (Ns1_b, Ns2_b, N_bands)
            norm_factors_raw = np.zeros((Ns1_b, Ns2_b, N_bands_all))
            for band_idx in band_indices:
                u_x = bloch_fields[:, :, band_idx, :, :, 0]  # (Ns1_b, Ns2_b, Nx, Ny)
                u_y = bloch_fields[:, :, band_idx, :, :, 1]
                # Average over microscopic coords
                avg_intensity = np.mean(np.abs(u_x)**2 + np.abs(u_y)**2, axis=(2, 3))
                norm_factors_raw[:, :, band_idx] = avg_intensity
            
            # Interpolate normalization factors to envelope grid if needed
            if need_interp:
                from scipy.ndimage import zoom
                norm_factors = np.zeros((Ns1_env, Ns2_env, N_bands_all))
                for band_idx in band_indices:
                    norm_factors[:, :, band_idx] = zoom(
                        norm_factors_raw[:, :, band_idx], 
                        (Ns1_env / Ns1_b, Ns2_env / Ns2_b), 
                        order=1, mode='wrap'
                    )
            else:
                norm_factors = norm_factors_raw
            
            # Cache for reuse
            if bloch_interp_cache is not None:
                bloch_interp_cache[norm_cache_key] = norm_factors
    
    # Sum over subspace bands
    for sub_idx, band_idx in enumerate(band_indices):
        # Envelope amplitude for this band at each registry point
        F_n = F_mode[:, :, sub_idx]  # Shape: (Ns1, Ns2)
        
        # Check cache first
        cache_key = (band_idx, component)
        if bloch_interp_cache is not None and cache_key in bloch_interp_cache:
            u_n = bloch_interp_cache[cache_key]
        else:
            # Bloch function for this band
            u_n_raw = bloch_fields[:, :, band_idx, :, :, component]  # Shape: (Ns1_b, Ns2_b, Nx, Ny)
            
            # Interpolate Bloch field to envelope grid if needed
            # Use scipy.ndimage.zoom for FAST interpolation
            if need_interp:
                # Zoom real and imaginary parts separately
                u_n_real = zoom(u_n_raw.real, zoom_factors, order=1, mode='wrap')
                u_n_imag = zoom(u_n_raw.imag, zoom_factors, order=1, mode='wrap')
                u_n = u_n_real + 1j * u_n_imag
            else:
                u_n = u_n_raw.astype(np.complex64)
            
            # Store in cache if provided
            if bloch_interp_cache is not None:
                bloch_interp_cache[cache_key] = u_n
        
        # Multiply: F_n(R) * u_n(r; R)
        # Cast F_n to complex64 for consistency
        F_n_c64 = F_n.astype(np.complex64)
        
        # Apply Bloch normalization if requested
        # Divide by sqrt(<|u|²>) so that normalized <|u|²> = 1
        if normalize_bloch:
            norm_factor = norm_factors[:, :, band_idx]  # (Ns1, Ns2)
            # Avoid division by zero
            norm_factor = np.maximum(norm_factor, 1e-10)
            # Scale both F and u by 1/sqrt(norm) to get proper normalization
            # Since we want |E|² = |F|² · <|u|²> / <|u|²> = |F|²
            scale = (1.0 / np.sqrt(norm_factor)).astype(np.float32)
            H_full += F_n_c64[:, :, np.newaxis, np.newaxis] * scale[:, :, np.newaxis, np.newaxis] * u_n
        else:
            H_full += F_n_c64[:, :, np.newaxis, np.newaxis] * u_n
    
    # Optional: Add Bloch phase factor e^{i k_0 · r}
    if include_bloch_phase and k0 is not None:
        # Create microscopic coordinate grid
        # Assuming unit cell has size 1 in fractional coordinates
        rx = np.linspace(0, 1, Nx, endpoint=False)
        ry = np.linspace(0, 1, Ny, endpoint=False)
        RX, RY = np.meshgrid(rx, ry, indexing='ij')
        
        # Phase factor: exp(i * (k0_x * rx + k0_y * ry) * 2π)
        # Note: k0 is in units of 2π/a
        phase = np.exp(2j * np.pi * (k0[0] * RX + k0[1] * RY))
        
        # Apply to all registry points
        H_full *= phase[np.newaxis, np.newaxis, :, :]
    
    # Reshape to tiled view: (Ns1 * Nx, Ns2 * Ny)
    # The ordering: for each (i, j) moiré cell, we have (Nx, Ny) microscopic points
    # We want to tile them: row i contains all Nx points for moiré row i
    H_tiled = H_full.transpose(0, 2, 1, 3).reshape(Ns1 * Nx, Ns2 * Ny)
    
    return H_tiled


def reconstruct_full_field_for_meep(
    mode_idx: int,
    F_spinor: np.ndarray,
    bloch_fields: np.ndarray,
    band_indices: list,
    B_moire: np.ndarray,
    B_mono: np.ndarray,
    target_coords: tuple,
    component: int = 2,
    include_bloch_phase: bool = False,
    k0: np.ndarray = None,
    normalize_bloch: bool = True,
) -> np.ndarray:
    """
    Reconstruct the full field H(r) at arbitrary Cartesian coordinates.
    
    This version is designed for Meep initialization where we need to
    evaluate the field at specific (x, y) coordinates defined by the
    simulation grid.
    
    Args:
        mode_idx: Index of the mode in F_spinor
        F_spinor: Envelope functions, shape (n_modes, Ns1, Ns2, N_subspace)
        bloch_fields: Bloch functions, shape (Ns1, Ns2, N_bands, Nx, Ny, 3)
        band_indices: Mapping from subspace index to bloch_fields band index
        B_moire: Moiré basis vectors (2x2 matrix, columns are vectors)
        B_mono: Monolayer basis vectors (2x2 matrix)
        target_coords: Tuple (X_flat, Y_flat) of 1D arrays with Cartesian coordinates
        component: Field component (0=x, 1=y, 2=z)
        include_bloch_phase: Include e^{i k_0 · r} phase
        k0: Crystal momentum for phase factor
    
    Returns:
        H_values: Complex field values at target coordinates, shape (N_points,)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    X_flat, Y_flat = target_coords
    N_pts = len(X_flat)
    
    Ns1_b, Ns2_b, N_bands_all, Nx, Ny, _ = bloch_fields.shape
    # Get envelope dimensions
    _, Ns1_env, Ns2_env, _ = F_spinor.shape
    
    N_subspace = len(band_indices)
    
    # Extract envelope for this mode
    F_mode = F_spinor[mode_idx]  # Shape: (Ns1_env, Ns2_env, N_subspace)
    
    # Transform Cartesian coordinates to fractional coordinates
    # Moiré fractional: S_moire = B_moire^{-1} @ P
    # Monolayer fractional (for microscopic): S_mono = B_mono^{-1} @ P
    inv_B_moire = np.linalg.inv(B_moire)
    inv_B_mono = np.linalg.inv(B_mono)
    
    P_vec = np.vstack((X_flat, Y_flat))  # Shape: (2, N_pts)
    
    S_moire = inv_B_moire @ P_vec  # Shape: (2, N_pts)
    S_mono = inv_B_mono @ P_vec
    
    # Moiré fractional coords (for envelope lookup)
    # Center at 0: map to [-0.5, 0.5]
    s1_moire = S_moire[0, :]
    s2_moire = S_moire[1, :]
    
    # Microscopic coords (periodic in monolayer cell)
    ux = np.mod(S_mono[0, :], 1.0)
    uy = np.mod(S_mono[1, :], 1.0)
    
    # Build interpolators for envelope F (one per subspace band)
    # Envelope is defined on grid [0, 1] but we center at 0 -> [-0.5, 0.5]
    s_grid_env = (np.linspace(-0.5, 0.5, Ns1_env), np.linspace(-0.5, 0.5, Ns2_env))
    s_grid_bloch = (np.linspace(-0.5, 0.5, Ns1_b), np.linspace(-0.5, 0.5, Ns2_b))
    
    interp_F = []
    for sub_idx in range(N_subspace):
        F_n = F_mode[:, :, sub_idx]
        interp_r = RegularGridInterpolator(s_grid_env, F_n.real, bounds_error=False, fill_value=0)
        interp_i = RegularGridInterpolator(s_grid_env, F_n.imag, bounds_error=False, fill_value=0)
        interp_F.append((interp_r, interp_i))
    
    # Build interpolators for Bloch fields u_n(r; R)
    # u depends on both R (slow, moire) and r (fast, microscopic)
    # Stored as (Ns1, Ns2, band, Nx, Ny, component)
    # We need 4D interpolation: (s1_moire, s2_moire, ux, uy) -> u
    u_grid = (np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    
    interp_u = []
    for sub_idx, band_idx in enumerate(band_indices):
        u_n = bloch_fields[:, :, band_idx, :, :, component]  # Shape: (Ns1_b, Ns2_b, Nx, Ny)
        
        # 4D grid: using Bloch grid
        full_grid = (s_grid_bloch[0], s_grid_bloch[1], u_grid[0], u_grid[1])
        
        interp_r = RegularGridInterpolator(full_grid, u_n.real, bounds_error=False, fill_value=0)
        interp_i = RegularGridInterpolator(full_grid, u_n.imag, bounds_error=False, fill_value=0)
        interp_u.append((interp_r, interp_i))
    
    # Evaluate at target points
    pts_F = np.stack((s1_moire, s2_moire), axis=-1)  # Shape: (N_pts, 2)
    pts_u = np.stack((s1_moire, s2_moire, ux, uy), axis=-1)  # Shape: (N_pts, 4)
    
    H_values = np.zeros(N_pts, dtype=np.complex128)
    # Compute normalization factors for Bloch functions if requested
    if normalize_bloch:
        # Check cache for normalization factors (not shared with single mode function currently, 
        # so we compute per call if not passed explicitly - TODO: optimize if needed)
        
        # Compute <|u_x|² + |u_y|²> for each registry point and band
        # Shape: (Ns1_b, Ns2_b, N_bands)
        norm_factors_raw = np.zeros((Ns1_b, Ns2_b, N_bands_all))
        for band_idx in band_indices:
            u_x = bloch_fields[:, :, band_idx, :, :, 0]  # (Ns1, Ns2, Nx, Ny)
            u_y = bloch_fields[:, :, band_idx, :, :, 1]
            # Average over microscopic coords
            avg_intensity = np.mean(np.abs(u_x)**2 + np.abs(u_y)**2, axis=(2, 3))
            norm_factors_raw[:, :, band_idx] = avg_intensity
        
        # Interpolate normalization factors to envelope grid (moire coords)
        # We need to evaluate norm factors at S_moire
        
        # Build interpolator for norms using BLOCH grid because norm_factors_raw is on Bloch grid
        interp_norm = []
        for band_idx in band_indices:
             norm_n = norm_factors_raw[:, :, band_idx]
             r_int = RegularGridInterpolator(s_grid_bloch, norm_n, bounds_error=False, fill_value=1.0) # Default norm=1
             interp_norm.append(r_int)
    
    for sub_idx, band_idx in enumerate(band_indices):
        # Evaluate envelope
        F_r = interp_F[sub_idx][0](pts_F)
        F_i = interp_F[sub_idx][1](pts_F)
        F_val = F_r + 1j * F_i
        
        # Evaluate Bloch function
        u_r = interp_u[sub_idx][0](pts_u)
        u_i = interp_u[sub_idx][1](pts_u)
        u_val = u_r + 1j * u_i
        
        if normalize_bloch:
             # Evaluate norm factor at pts_F
             # norm_val = <|u|²>
             nf = interp_norm[sub_idx](pts_F)
             nf = np.maximum(nf, 1e-10)
             scale = 1.0 / np.sqrt(nf)
             H_values += F_val * u_val * scale
        else:
             H_values += F_val * u_val
    
    # Optional Bloch phase
    if include_bloch_phase and k0 is not None:
        phase = np.exp(2j * np.pi * (k0[0] * X_flat + k0[1] * Y_flat))
        H_values *= phase
    
    return H_values


def plot_reconstructed_modes(
    cdir,
    F_spinor,
    bloch_fields,
    band_indices,
    eigenvalues,
    mode_stats,
    n_rows=10,
    n_cols=10,
    single_mode_idx=None,
):
    """
    Create 10×10 grid plots of reconstructed |H_z|² for each mode.
    
    Memory-efficient: reconstructs one mode at a time.
    Uses caching to avoid recomputing Bloch field interpolation.
    
    Args:
        single_mode_idx: If provided, only plot this single mode (for debugging).
    """
    from tqdm import tqdm
    
    n_modes = F_spinor.shape[0]
    
    # Cache for interpolated Bloch fields (shared across modes)
    bloch_cache = {}
    
    # Single mode debugging mode
    if single_mode_idx is not None:
        log(f"=== DEBUG MODE: Single mode {single_mode_idx} ===")
        
        mode_idx = single_mode_idx
        
        # First check envelope
        F_mode = F_spinor[mode_idx]
        log(f"  F_mode shape: {F_mode.shape}")
        log(f"  F_mode range: [{np.min(np.abs(F_mode)):.2e}, {np.max(np.abs(F_mode)):.2e}]")
        log(f"  F_mode mean abs: {np.mean(np.abs(F_mode)):.2e}")
        
        # Check bloch fields - Note: stored fields are E-fields (Ex, Ey, Ez)
        # For 2D TE modes: Ex, Ey are non-zero; Ez = 0
        # For 2D TM modes: Ez is non-zero; Ex, Ey = 0
        for sub_idx, band_idx in enumerate(band_indices):
            for comp_idx, comp_name in enumerate(['Ex', 'Ey', 'Ez']):
                u_slice = bloch_fields[:, :, band_idx, :, :, comp_idx]
                maxval = np.max(np.abs(u_slice))
                if maxval > 1e-10:
                    log(f"  Bloch u[band={band_idx}] {comp_name}: max={maxval:.2e}")
        
        # Determine which components to use
        # Check Ex, Ey, Ez to find non-zero ones
        comp_x_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 0]))
        comp_y_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 1]))
        comp_z_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 2]))
        
        log(f"  Field component magnitudes: Ex={comp_x_max:.2e}, Ey={comp_y_max:.2e}, Ez={comp_z_max:.2e}")
        
        if comp_z_max > 1e-10:
            # TM mode: use Ez
            log("  Detected TM polarization (Ez dominant)")
            use_component = 2
            field_name = "E_z"
        else:
            # TE mode: use |E|² = |Ex|² + |Ey|²
            log("  Detected TE polarization (Ex, Ey in-plane)")
            use_component = -1  # Special flag for combined Ex+Ey
            field_name = "|E|"
        
        # Reconstruct full field - use detected component
        if use_component >= 0:
            # Single component (TM: Ez)
            E_field = reconstruct_full_field_single_mode(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                component=use_component,
                include_bloch_phase=False,
                bloch_interp_cache=bloch_cache,
            )
            intensity = np.abs(E_field)**2
        else:
            # TE mode: combine Ex and Ey
            E_x = reconstruct_full_field_single_mode(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                component=0,  # Ex
                include_bloch_phase=False,
                bloch_interp_cache=bloch_cache,
            )
            E_y = reconstruct_full_field_single_mode(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                component=1,  # Ey
                include_bloch_phase=False,
                bloch_interp_cache=bloch_cache,
            )
            E_field = E_x  # Use Ex for phase visualization
            intensity = np.abs(E_x)**2 + np.abs(E_y)**2
        
        # Log reconstruction results
        log(f"  {field_name} reconstructed shape: {E_field.shape}")
        log(f"  {field_name} range: [{np.min(np.abs(E_field)):.2e}, {np.max(np.abs(E_field)):.2e}]")
        log(f"  |{field_name}|² range: [{np.min(intensity):.2e}, {np.max(intensity):.2e}]")
        
        # Get mode stats
        if mode_stats and mode_idx < len(mode_stats):
            stats = mode_stats[mode_idx]
            omega = stats.get('omega', eigenvalues[mode_idx])
            spread = stats.get('spread', 0)
        else:
            omega = eigenvalues[mode_idx]
            spread = 0
        
        # === Compute coarse-grained intensity ===
        # Average |E|² over each 32x32 microscopic cell to get envelope-scale pattern
        Ns = F_spinor.shape[1]  # 128
        n_micro = bloch_fields.shape[3]  # 32
        
        # Reshape to (Ns1, n_micro, Ns2, n_micro) and average over micro dimensions
        intensity_coarse = intensity.reshape(Ns, n_micro, Ns, n_micro).mean(axis=(1, 3))
        log(f"  Coarse-grained |{field_name}|² shape: {intensity_coarse.shape}")
        log(f"  Coarse-grained range: [{np.min(intensity_coarse):.2e}, {np.max(intensity_coarse):.2e}]")
        
        # === Normalize for comparison ===
        F_prob = np.sum(np.abs(F_mode)**2, axis=2)  # |F|² summed over bands
        F_prob_norm = F_prob / np.max(F_prob) if np.max(F_prob) > 1e-15 else F_prob
        
        intensity_coarse_norm = intensity_coarse / np.max(intensity_coarse) if np.max(intensity_coarse) > 1e-15 else intensity_coarse
        intensity_full_norm = intensity / np.max(intensity) if np.max(intensity) > 1e-15 else intensity
        
        log(f"  |F|² max: {np.max(F_prob):.2e}")
        log(f"  |E|²_coarse max: {np.max(intensity_coarse):.2e}")
        log(f"  Ratio (coarse/F): {np.max(intensity_coarse)/np.max(F_prob):.2e}" if np.max(F_prob) > 1e-15 else "  Ratio: N/A")
        
        # Compute correlation between |F|² and coarse-grained |E|²
        F_flat = F_prob_norm.flatten()
        E_flat = intensity_coarse_norm.flatten()
        correlation = np.corrcoef(F_flat, E_flat)[0, 1]
        log(f"  Correlation(|F|², |E|²_coarse): {correlation:.4f}")
        
        # Compute the dominant band's Bloch support pattern for visualization
        dominant_band_idx = np.argmax([np.max(np.abs(F_mode[:,:,i])) for i in range(F_mode.shape[2])])
        log(f"  Dominant band in F_mode: {dominant_band_idx}")
        
        # === Create 1x3 comparison figure (User Requested) ===
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Envelope |F|² (normalized) - LOG SCALE
        F_log = np.log10(F_prob_norm + 1e-6)
        im1 = axes[0].imshow(F_log.T, origin='lower', cmap='hot', 
                                 aspect='equal', vmin=-4, vmax=0)
        axes[0].set_title(f'1. Envelope |F|² (log scale)\nM{mode_idx} ω={omega:.5f}')
        plt.colorbar(im1, ax=axes[0], label='log₁₀')
        
        # 2. Full resolution |E|² (Log scale per user preference for singular modes?)
        # Let's stick to Log Scale to match envelope, as "Field is sparse" was the complaint.
        E_full_log = np.log10(intensity_full_norm + 1e-6)
        im2 = axes[1].imshow(E_full_log.T, origin='lower', cmap='hot',
                                 aspect='equal', vmin=-4, vmax=0)
        axes[1].set_title(f'2. Reconstructed |{field_name}|² (log scale)')
        plt.colorbar(im2, ax=axes[1], label='log₁₀')
        
        # 3. Real(E) showing Bloch oscillations (normalized to p99.9)
        e_real = E_field.real
        p999 = np.percentile(np.abs(e_real), 99.9)
        vmax = p999 if p999 > 1e-15 else np.max(np.abs(e_real))
        
        e_real_norm = e_real / vmax if vmax > 1e-15 else e_real
        # Saturate at +/- 1 (which translates to +/- p99.9 of original)
        im3 = axes[2].imshow(e_real_norm.T, origin='lower', cmap='RdBu',
                                 aspect='equal', vmin=-1, vmax=1)
        axes[2].set_title(f'3. Real({field_name}) (sat. @ p99.9)\nShows Bloch oscillations')
        plt.colorbar(im3, ax=axes[2], label='Norm to p99.9')

        plt.suptitle(f'Phase 4 Reconstruction: Mode {mode_idx}', fontsize=16)
        plt.tight_layout()
        plot_path = cdir / f'phase4_singular_mode{mode_idx}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log(f"  Saved singular plot: {plot_path}")
        
        return
    
    # Normal mode: 10x10 grid
    n_plot = min(n_rows * n_cols, n_modes)
    
    log(f"Creating reconstructed field plots for {n_plot} modes...")
    
    # Sort by eigenvalue (frequency)
    sorted_indices = np.argsort(eigenvalues)
    
    # Check components just once on first mode to decide strategy
    m0_idx = sorted_indices[0]
    comp_z_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 2]))
    if comp_z_max > 1e-10:
        log("Batch plot: Using Hz (TM mode logic or consistent Hz plot)")
        # Actually for TM, Ez is dominant. For TE, Hz is dominant.
        # But previous code plotted Hz. Let's stick to Hz for consistency unless user wants E.
        # However, singular mode plots E. Let's adapt batch to plot E as well.
        # Wait, if I change to E, I might break consistency with file name "Hz".
        # Let's keep file name but plot intensity of E.
        pass
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
    axes = axes.flatten()
    
    for plot_idx in tqdm(range(n_plot), desc="Reconstructing modes (by freq)"):
        mode_idx = sorted_indices[plot_idx]
        ax = axes[plot_idx]
        
        # Use E-field logic to match singular plots
        if comp_z_max > 1e-10:
            # TM: Use Ez
            E_field = reconstruct_full_field_single_mode(mode_idx, F_spinor, bloch_fields, band_indices, 2, False, bloch_cache)
            intensity = np.abs(E_field)**2
        else:
            # TE: Use Ex+Ey (Intensity only)
             E_x = reconstruct_full_field_single_mode(mode_idx, F_spinor, bloch_fields, band_indices, 0, False, bloch_cache)
             E_y = reconstruct_full_field_single_mode(mode_idx, F_spinor, bloch_fields, band_indices, 1, False, bloch_cache)
             intensity = np.abs(E_x)**2 + np.abs(E_y)**2
        
        # Downsample strictly preserving peaks (Max pooling)
        # Target ~256 pixels max dimension for thumbnail
        nx, ny = intensity.shape
        block_size = max(1, nx // 256)
        
        if block_size > 1:
            # Reshape and max-pool
            # Handle cases where shape is not divisible by block_size
            sx = (nx // block_size) * block_size
            sy = (ny // block_size) * block_size
            
            intensity_view = intensity[:sx, :sy]
            intensity_down = intensity_view.reshape(nx // block_size, block_size, ny // block_size, block_size).max(axis=(1, 3))
        else:
            intensity_down = intensity
            
        # Plot with Log Scale
        intensity_log = np.log10(intensity_down + 1e-10)
        im = ax.imshow(intensity_log.T, origin='lower', cmap='hot', aspect='equal')
        
        if mode_stats and mode_idx < len(mode_stats):
            stats = mode_stats[mode_idx]
            omega = stats.get('omega', eigenvalues[mode_idx])
            spread = stats.get('spread', 0)
        else:
            omega = eigenvalues[mode_idx]
            spread = 0
        
        ax.set_title(f'M{mode_idx} ω={omega:.5f}\nσ={spread:.2f}', fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_plot, n_rows * n_cols):
        axes[idx].axis('off')
    
    plt.suptitle(f'Reconstructed |H_z|² (Full Bloch Structure) - Top {n_plot} Modes by Frequency', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plot_path = cdir / 'phase4_reconstructed_Hz_by_frequency.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"  Saved: {plot_path}")
    
    # Second plot: Real(E) showing Bloch oscillations
    log("Creating Real(E) plot...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
    axes = axes.flatten()
    
    for plot_idx in tqdm(range(n_plot), desc="Plotting Real(E)"):
        mode_idx = sorted_indices[plot_idx]
        ax = axes[plot_idx]
        
        # Consistent Field logic
        if comp_z_max > 1e-10:
            E_field = reconstruct_full_field_single_mode(mode_idx, F_spinor, bloch_fields, band_indices, 2, False, bloch_cache)
        else:
            # For TE, show Ex (phase relevant)
            E_field = reconstruct_full_field_single_mode(mode_idx, F_spinor, bloch_fields, band_indices, 0, False, bloch_cache)
        
        h_real = E_field.real
        vmax_abs = np.max(np.abs(h_real))
        p999 = np.percentile(np.abs(h_real), 99.9)
        vmax = p999 if p999 > 1e-10 else vmax_abs
        if vmax < 1e-10:
            vmax = 1.0
        
        # Downsample using striding (for signed real part, averaging wipes out oscillations)
        # But we must be careful not to stride perfectly out of phase
        # Given Bloch is 32x32 per unit cell, stride 16 essentially samples 2 per cell. 
        # Risky. Let's use stride 8. 8192/8 = 1024. A bit large but safe.
        nx, ny = h_real.shape
        stride = max(1, nx // 300) # Target 300x300
        
        im = ax.imshow(h_real[::stride, ::stride].T, origin='lower', cmap='RdBu', aspect='equal',
                       vmin=-vmax, vmax=vmax)
        
        if mode_stats and mode_idx < len(mode_stats):
            omega = mode_stats[mode_idx].get('omega', eigenvalues[mode_idx])
        else:
            omega = eigenvalues[mode_idx]
        
        ax.set_title(f'M{mode_idx} ω={omega:.5f}', fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    
    for idx in range(n_plot, n_rows * n_cols):
        axes[idx].axis('off')
    
    plt.suptitle(f'Real(H_z) with Bloch Oscillations - Top {n_plot} Modes by Frequency', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    plot_path = cdir / 'phase4_reconstructed_Hz_real.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log(f"  Saved: {plot_path}")


def run_phase4(candidate_id, single_mode=None):
    """Main Phase 4 execution.
    
    Args:
        candidate_id: ID of the candidate to process.
        single_mode: If provided, only process this single mode (for debugging).
    """
    run_dir = find_latest_run_dir()
    cdir = candidate_dir(run_dir, candidate_id)
    
    log(f"=== Phase 4: Full Field Reconstruction for Candidate {candidate_id} ===")
    log(f"Run Directory: {run_dir.name}")
    
    # Load Phase 1 data (Bloch fields)
    log("Loading Phase 1 Bloch fields...")
    bloch_fields, subspace_bands, all_bands = load_phase1_bloch_fields(cdir)
    
    # Load Phase 3 data (Envelopes)
    log("Loading Phase 3 envelope modes...")
    F_spinor, eigenvalues, mode_stats = load_phase3_envelopes(cdir)
    
    # Get band index mapping
    band_indices = get_subspace_band_indices(subspace_bands, all_bands)
    log(f"  Band mapping: subspace {subspace_bands} -> indices {band_indices}")
    
    # Verify dimensions match
    Ns1_bloch, Ns2_bloch = bloch_fields.shape[:2]
    Ns1_env, Ns2_env = F_spinor.shape[1:3]
    N_subspace = F_spinor.shape[3]
    
    if Ns1_bloch != Ns1_env or Ns2_bloch != Ns2_env:
        log(f"  Warning: Grid size mismatch! Bloch: {Ns1_bloch}x{Ns2_bloch}, Envelope: {Ns1_env}x{Ns2_env}")
        log(f"  Will interpolate Bloch fields to envelope grid.")
    
    if len(band_indices) != N_subspace:
        raise ValueError(f"Band count mismatch: {len(band_indices)} mapped bands vs {N_subspace} envelope bands")
    
    log(f"  Grid: {Ns1_env}x{Ns2_env} moiré points")
    log(f"  Microscopic: {bloch_fields.shape[3]}x{bloch_fields.shape[4]} points per cell")
    log(f"  Full resolution: {Ns1_env * bloch_fields.shape[3]} x {Ns2_env * bloch_fields.shape[4]}")
    
    # Generate plots
    plot_reconstructed_modes(
        cdir=cdir,
        F_spinor=F_spinor,
        bloch_fields=bloch_fields,
        band_indices=band_indices,
        eigenvalues=eigenvalues,
        mode_stats=mode_stats,
        single_mode_idx=single_mode,
    )
    
    log("Phase 4 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Full Field Reconstruction")
    parser.add_argument("candidate_id", type=int, help="Candidate ID to process")
    parser.add_argument("--mode", type=int, default=None, 
                        help="Single mode index to plot (for debugging). If not set, plots all modes.")
    
    args = parser.parse_args()
    run_phase4(args.candidate_id, single_mode=args.mode)
