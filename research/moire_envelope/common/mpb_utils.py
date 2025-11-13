"""
MPB (MIT Photonic Bands) utilities for band structure calculations
"""
import numpy as np
import meep as mp
from meep import mpb
import math
import sys
import os
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr


def create_mpb_geometry(geom_params):
    """
    Create MPB geometry from geometry parameters
    
    Args:
        geom_params: Dictionary with geometry parameters from build_lattice
        
    Returns:
        tuple: (geometry_list, lattice, eps_bg)
    """
    a = geom_params['lattice_constant']
    r = geom_params['radius']
    eps_bg = geom_params['eps_bg']
    eps_hole = geom_params['eps_hole']
    
    a1 = geom_params['a1']
    a2 = geom_params['a2']
    
    # Normalize lattice vectors
    a1_norm = np.linalg.norm(a1[:2])
    a2_norm = np.linalg.norm(a2[:2])
    
    # Create lattice for MPB
    lattice = mp.Lattice(
        size=mp.Vector3(a1_norm, a2_norm, 0),
        basis1=mp.Vector3(a1[0]/a1_norm, a1[1]/a1_norm, 0),
        basis2=mp.Vector3(a2[0]/a2_norm, a2[1]/a2_norm, 0)
    )
    
    # Create cylinder (hole) at the center
    geometry = [
        mp.Cylinder(r, material=mp.Medium(epsilon=eps_hole))
    ]
    
    return geometry, lattice, eps_bg


def compute_bandstructure(geom_params, config, k_points=None, num_bands=8):
    """
    Compute band structure using MPB
    
    Args:
        geom_params: Geometry parameters
        config: Configuration dictionary
        k_points: List of k-points to compute (optional)
        num_bands: Number of bands to compute
        
    Returns:
        dict: Band structure data with k_points, frequencies, and k_labels
    """
    geometry, lattice, eps_bg = create_mpb_geometry(geom_params)
    
    # Suppress MPB output for parallel execution
    quiet = config.get('quiet_mpb', True)
    
    devnull = None
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    if quiet:
        # Redirect both stdout and stderr to devnull
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
    
    try:
        # Create ModeSolver
        ms = mpb.ModeSolver(
            geometry=geometry,
            geometry_lattice=lattice,
            default_material=mp.Medium(epsilon=eps_bg),
            num_bands=num_bands,
            resolution=config.get('resolution', 32)
        )
        
        # Get high symmetry points if not provided
        if k_points is None:
            lattice_type = geom_params['lattice_type']
            from .geometry import high_symmetry_points
            hs_points = high_symmetry_points(lattice_type)
            k_labels = [kp[0] for kp in hs_points]
            k_vecs = [kp[1] for kp in hs_points]
            k_points = [mp.Vector3(*kv) for kv in k_vecs]
        else:
            k_labels = None
        
        # Run band calculation with interpolation
        k_interp_num = config.get('k_interp', 19)
        k_points_interp = mp.interpolate(k_interp_num, k_points)
        
        # Set k-points before running
        ms.k_points = k_points_interp
        
        # Run MPB for all k-points
        ms.run_te()
        
        # Extract frequencies from ms.all_freqs (shape: n_k x n_bands)
        # This is the correct MPB API - frequencies are stored in ms.all_freqs after run_te()
        freqs_array = np.array(ms.all_freqs)
        
        if len(freqs_array.shape) == 1:
            # If only one k-point, reshape
            freqs_array = freqs_array.reshape(1, -1)

        # Compute cumulative k-path distances for plotting
        def _vec_to_np(vec):
            return np.array([vec.x, vec.y, vec.z], dtype=float)

        k_path = np.zeros(len(k_points_interp))
        for idx in range(1, len(k_points_interp)):
            delta = _vec_to_np(k_points_interp[idx]) - _vec_to_np(k_points_interp[idx - 1])
            k_path[idx] = k_path[idx - 1] + np.linalg.norm(delta)

        k_label_positions = None
        k_break_indices = None
        if k_labels is not None and len(k_points) > 1:
            segment_lengths = []
            cumulative_targets = []
            total = 0.0
            for i in range(len(k_points) - 1):
                delta = _vec_to_np(k_points[i + 1]) - _vec_to_np(k_points[i])
                seg_len = np.linalg.norm(delta)
                segment_lengths.append(seg_len)
                total += seg_len
                cumulative_targets.append(total)

            k_break_indices = [0]
            for target in cumulative_targets:
                idx = int(np.argmin(np.abs(k_path - target)))
                k_break_indices.append(idx)

            k_label_positions = np.array([k_path[idx] for idx in k_break_indices], dtype=float)
        
        result = {
            'k_points': k_points_interp,
            'frequencies': freqs_array,
            'num_bands': num_bands,
            'lattice_type': geom_params['lattice_type'],
            'k_labels': k_labels,
            'k_interp': k_interp_num,
            'k_path': k_path,
            'k_label_positions': k_label_positions,
            'k_break_indices': k_break_indices,
        }
        
    finally:
        if quiet and devnull is not None:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
    
    return result


def compute_local_band_data(R_grid, delta_grid, candidate_params, config):
    """
    Compute local band structure data at each registry point
    
    This is a simplified placeholder - full implementation requires
    running MPB at each R point with local stacking shift
    
    Args:
        R_grid: Spatial grid [Nx, Ny, 2]
        delta_grid: Registry map [Nx, Ny, 2]
        candidate_params: Candidate parameters
        config: Configuration dict
        
    Returns:
        tuple: (omega0, vg, M_inv) arrays
    """
    Nx, Ny, _ = R_grid.shape
    
    # Placeholders - actual implementation would run MPB at each point
    omega0 = np.zeros((Nx, Ny))
    vg = np.zeros((Nx, Ny, 2))
    M_inv = np.zeros((Nx, Ny, 2, 2))
    
    # For now, use simplified analytical model
    # omega0(R) with some variation based on registry
    base_omega = candidate_params.get('omega0', 0.5)
    
    for i in range(Nx):
        for j in range(Ny):
            # Simple sinusoidal variation with registry
            delta = delta_grid[i, j]
            phase = 2 * np.pi * np.linalg.norm(delta)
            omega0[i, j] = base_omega * (1 + 0.1 * np.cos(phase))
            
            # Simplified effective mass tensor (isotropic)
            m_eff = 1.0 + 0.2 * np.sin(phase)
            M_inv[i, j] = np.eye(2) / m_eff
    
    return omega0, vg, M_inv


def fit_local_dispersion(bands, k_label, band_index, dk=0.01):
    """
    Fit local dispersion relation around a k-point
    
    Args:
        bands: Band structure data from compute_bandstructure
        k_label: Label of the k-point
        band_index: Band index to analyze (0-indexed)
        dk: k-space step for finite differences
        
    Returns:
        dict: Dispersion metrics (omega0, vg, curvature, etc.)
    """
    freqs = bands['frequencies']  # Shape: (n_k, n_bands)
    k_labels = bands.get('k_labels', [])
    
    # Find the index of the k-point with this label
    # The k-path is built as: k_points[0] --k_interp--> k_points[1] --k_interp--> k_points[2]
    # With mp.interpolate(k_interp, k_points), we get k_interp points BETWEEN each segment
    # For Γ → X → M with k_interp=19 and 3 high-sym points, we get 23 total k-points
    n_k = freqs.shape[0]
    
    if k_labels and k_label in k_labels:
        k_idx = None
        k_interp = bands.get('k_interp', 19)
        num_segments = len(k_labels) - 1
        
        for i, label in enumerate(k_labels):
            if label == k_label:
                # Position of high symmetry point i in the interpolated path
                # mp.interpolate distributes points evenly, so k_labels[i] is at:
                k_idx = i * (n_k - 1) // num_segments
                break
        
        if k_idx is None or k_idx >= n_k:
            k_idx = n_k // 2  # Default to middle
    else:
        k_idx = n_k // 2
    
    # Extract frequency at this k-point and band
    if band_index >= freqs.shape[1]:
        band_index = freqs.shape[1] - 1
    
    omega0 = freqs[k_idx, band_index]
    
    # Compute group velocity using finite differences
    if k_idx > 0 and k_idx < n_k - 1:
        domega_dk = (freqs[k_idx + 1, band_index] - freqs[k_idx - 1, band_index]) / 2
    else:
        domega_dk = 0.0
    
    vg_norm = abs(domega_dk)
    vg_x = domega_dk * 0.7071  # Assume diagonal direction
    vg_y = domega_dk * 0.7071
    
    # Compute curvature (second derivative)
    if k_idx > 0 and k_idx < n_k - 1:
        d2omega_dk2 = (freqs[k_idx + 1, band_index] - 2*freqs[k_idx, band_index] + 
                       freqs[k_idx - 1, band_index])
    else:
        d2omega_dk2 = 1.0
    
    # For 2D, assume isotropic curvature
    curvature_xx = abs(d2omega_dk2)
    curvature_yy = abs(d2omega_dk2)
    curvature_xy = 0.0
    curvature_trace = curvature_xx + curvature_yy
    curvature_det = curvature_xx * curvature_yy
    
    # Parabolic validity radius (rough estimate)
    if curvature_trace > 1e-6:
        k_parab = 0.2 / np.sqrt(curvature_trace)
    else:
        k_parab = 0.5
    
    # Spectral gaps
    gap_above = 0.1
    gap_below = 0.1
    if band_index < freqs.shape[1] - 1:
        gap_above = freqs[k_idx, band_index + 1] - omega0
    if band_index > 0:
        gap_below = omega0 - freqs[k_idx, band_index - 1]
    
    metrics = {
        'omega0': float(omega0),
        'vg_x': float(vg_x),
        'vg_y': float(vg_y),
        'vg_norm': float(vg_norm),
        'curvature_xx': float(curvature_xx),
        'curvature_xy': float(curvature_xy),
        'curvature_yy': float(curvature_yy),
        'curvature_trace': float(curvature_trace),
        'curvature_det': float(curvature_det),
        'k_parab': float(k_parab),
        'gap_above': float(gap_above),
        'gap_below': float(gap_below),
    }
    
    return metrics


def compute_local_band_at_registry(geom_params, k0, band_index, config):
    """
    Compute local band data at a specific k-point for frozen bilayer geometry
    
    This performs MPB calculation at k₀ and nearby k-points to extract:
    - ω₀(k₀): frequency at the reference k-point
    - v_g: group velocity ∇_k ω
    - M⁻¹: inverse effective mass tensor (curvature)
    
    Args:
        geom_params: Geometry parameters with 'delta_frac' for stacking shift
        k0: Reference k-vector [kx, ky, kz] in units of 2π/a
        band_index: Band index to analyze
        config: Configuration with MPB settings
        
    Returns:
        tuple: (omega0, vg[2], M_inv[2,2])
    """
    geometry, lattice, eps_bg = create_mpb_geometry(geom_params)
    
    # Finite difference step in k-space
    dk = config.get('phase1_dk', 0.01)  # In units of 2π/a
    
    # Suppress MPB output
    quiet = config.get('quiet_mpb', True)
    
    devnull = None
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    if quiet:
        devnull = open(os.devnull, 'w')
        sys.stdout = devnull
        sys.stderr = devnull
    
    try:
        # Create ModeSolver
        num_bands = config.get('num_bands', 8)
        resolution = config.get('phase1_resolution', 16)  # Lower res for speed
        
        ms = mpb.ModeSolver(
            geometry=geometry,
            geometry_lattice=lattice,
            default_material=mp.Medium(epsilon=eps_bg),
            num_bands=num_bands,
            resolution=resolution
        )
        
        # Ensure band_index is valid
        band_index = min(band_index, num_bands - 1)
        
        # Define k-points for finite difference stencil
        # Central point + 4 neighbors (±x, ±y)
        k0_vec = mp.Vector3(k0[0], k0[1], k0[2])
        kx_plus = mp.Vector3(k0[0] + dk, k0[1], k0[2])
        kx_minus = mp.Vector3(k0[0] - dk, k0[1], k0[2])
        ky_plus = mp.Vector3(k0[0], k0[1] + dk, k0[2])
        ky_minus = mp.Vector3(k0[0], k0[1] - dk, k0[2])
        
        # For second derivatives (curvature), need diagonal points
        kxy_pp = mp.Vector3(k0[0] + dk, k0[1] + dk, k0[2])
        kxy_pm = mp.Vector3(k0[0] + dk, k0[1] - dk, k0[2])
        kxy_mp = mp.Vector3(k0[0] - dk, k0[1] + dk, k0[2])
        kxy_mm = mp.Vector3(k0[0] - dk, k0[1] - dk, k0[2])
        
        k_points = [k0_vec, kx_plus, kx_minus, ky_plus, ky_minus,
                    kxy_pp, kxy_pm, kxy_mp, kxy_mm]
        
        ms.k_points = k_points
        
        # Run TE mode calculation
        ms.run_te()
        
        # Extract frequencies
        freqs_array = np.array(ms.all_freqs)
        if len(freqs_array.shape) == 1:
            freqs_array = freqs_array.reshape(-1, num_bands)
        
        # Extract values at each k-point
        omega0 = freqs_array[0, band_index]
        omega_xp = freqs_array[1, band_index]
        omega_xm = freqs_array[2, band_index]
        omega_yp = freqs_array[3, band_index]
        omega_ym = freqs_array[4, band_index]
        omega_pp = freqs_array[5, band_index]
        omega_pm = freqs_array[6, band_index]
        omega_mp = freqs_array[7, band_index]
        omega_mm = freqs_array[8, band_index]
        
        # Compute group velocity (first derivatives)
        vg_x = (omega_xp - omega_xm) / (2 * dk)
        vg_y = (omega_yp - omega_ym) / (2 * dk)
        vg = np.array([vg_x, vg_y])
        
        # Compute inverse mass tensor (second derivatives)
        # ∂²ω/∂kx² ≈ [ω(k+Δkx) - 2ω(k) + ω(k-Δkx)] / (Δk)²
        d2_xx = (omega_xp - 2*omega0 + omega_xm) / (dk**2)
        d2_yy = (omega_yp - 2*omega0 + omega_ym) / (dk**2)
        
        # Mixed derivative: ∂²ω/∂kx∂ky
        # Using centered finite difference formula
        d2_xy = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk**2)
        
        # Construct inverse mass tensor
        # M⁻¹ is the Hessian of ω(k)
        M_inv = np.array([
            [d2_xx, d2_xy],
            [d2_xy, d2_yy]
        ])
        
        # Ensure positive definiteness for stability (regularization)
        # If eigenvalues are negative, flip them to small positive values
        eigvals, eigvecs = np.linalg.eigh(M_inv)
        eigvals = np.where(eigvals < 1e-6, 1e-6, eigvals)
        M_inv = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
    finally:
        if quiet and devnull is not None:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            devnull.close()
    
    return float(omega0), vg.astype(float), M_inv.astype(float)


def compute_moire_bandstructure(candidate_params, config):
    """
    Compute full moiré supercell band structure
    
    This is used in Phase 4 for validation
    
    Args:
        candidate_params: Candidate parameters
        config: Configuration dict
        
    Returns:
        dict: Moiré band structure data
    """
    # Placeholder for Phase 4
    return {
        'frequencies': [],
        'k_points': [],
    }
