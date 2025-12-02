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
    
    translations = geom_params.get('hole_translations')
    if translations:
        geometry = [
            mp.Cylinder(
                r,
                center=mp.Vector3(float(shift[0]), float(shift[1]), float(shift[2]) if len(shift) == 3 else 0.0),
                material=mp.Medium(epsilon=eps_hole)
            )
            for shift in translations
        ]
    else:
        # Default: single hole at origin
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


def fit_local_dispersion(bands, k_label, band_index):
    """Extract local dispersion metrics for a monolayer band extremum."""
    freqs = bands['frequencies']  # Shape: (n_k, n_bands)
    k_labels = bands.get('k_labels', [])
    k_points = bands.get('k_points')
    k_path = bands.get('k_path')
    n_k = freqs.shape[0]
    if n_k < 3:
        raise ValueError("Need at least three k-points to fit dispersion")

    def _vec_to_np(vec):
        """Convert MPB Vector3 or numpy-like objects to numpy arrays."""
        try:
            return np.array([vec.x, vec.y, vec.z], dtype=float)
        except AttributeError:
            return np.array(vec, dtype=float)

    wrap_path = False
    if k_path is not None and len(k_path) == n_k and abs(k_path[-1] - k_path[0]) < 1e-9:
        wrap_path = True
    elif k_points is not None:
        try:
            first_vec = np.array([k_points[0].x, k_points[0].y, k_points[0].z])
            last_vec = np.array([k_points[-1].x, k_points[-1].y, k_points[-1].z])
            wrap_path = np.allclose(first_vec, last_vec)
        except AttributeError:
            pass
    
    # Find the index of the k-point with this label
    # The k-path is built as: k_points[0] --k_interp--> k_points[1] --k_interp--> k_points[2]
    # With mp.interpolate(k_interp, k_points), we get k_interp points BETWEEN each segment
    # For Γ → X → M with k_interp=19 and 3 high-sym points, we get 23 total k-points
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
    if wrap_path and k_idx == n_k - 1:
        # Use the first copy of the periodic point
        k_idx = 0
    
    # Extract frequency at this k-point and band
    if band_index >= freqs.shape[1]:
        band_index = freqs.shape[1] - 1
    
    omega0 = freqs[k_idx, band_index]
    
    # Compute group velocity using finite differences
    prev_idx = k_idx - 1
    next_idx = k_idx + 1
    if prev_idx < 0:
        prev_idx = n_k - 2 if wrap_path else 0
    if next_idx >= n_k:
        next_idx = 1 if wrap_path else n_k - 1
    if prev_idx == next_idx:
        prev_idx = max(prev_idx - 1, 0)
        next_idx = min(next_idx + 1, n_k - 1)

    omega_prev = freqs[prev_idx, band_index]
    omega_next = freqs[next_idx, band_index]

    # Determine actual k-space offsets (fallback to unit spacing)
    k_curr = None
    if k_points is not None:
        k_prev = _vec_to_np(k_points[prev_idx])
        k_curr = _vec_to_np(k_points[k_idx])
        k_next = _vec_to_np(k_points[next_idx])
        dk_prev = np.linalg.norm(k_curr - k_prev)
        dk_next = np.linalg.norm(k_next - k_curr)
        chord_vec = k_next - k_prev
    else:
        dk_prev = dk_next = 1.0
        chord_vec = np.array([1.0, 1.0, 0.0])
    chord_len = np.linalg.norm(chord_vec[:2])
    if chord_len < 1e-9:
        vg = np.zeros(2)
        domega_dk = 0.0
    else:
        domega_dk = (omega_next - omega_prev) / chord_len
        tangent = chord_vec[:2] / chord_len
        vg = domega_dk * tangent
    vg_norm = float(np.linalg.norm(vg))
    vg_x = float(vg[0])
    vg_y = float(vg[1])
    
    # Compute curvature (second derivative)
    if dk_prev > 1e-9 and dk_next > 1e-9:
        term_next = (omega_next - omega0) / dk_next
        term_prev = (omega0 - omega_prev) / dk_prev
        d2omega_dk2 = 2.0 * (term_next - term_prev) / (dk_prev + dk_next)
    else:
        d2omega_dk2 = 0.0
    
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

    # Additional symmetry sampling: compare near/far samples against quadratic fit
    parab_error_floor = 1e-4
    parab_error_threshold = 1.25

    def _neighbor_index(offset):
        idx = k_idx + offset
        if wrap_path:
            return idx % n_k
        if 0 <= idx < n_k:
            return idx
        return None

    def _neighbor_sample(offset):
        idx = _neighbor_index(offset)
        if idx is None or idx == k_idx:
            return None
        omega = freqs[idx, band_index]
        if k_points is not None and k_curr is not None:
            distance = float(np.linalg.norm(_vec_to_np(k_points[idx]) - k_curr))
        else:
            distance = float(abs(offset))
        if distance < 1e-12:
            return None
        return {
            'idx': idx,
            'omega': float(omega),
            'distance': distance,
        }

    near_samples = [s for s in (_neighbor_sample(-1), _neighbor_sample(1)) if s]
    far_samples = [s for s in (_neighbor_sample(-2), _neighbor_sample(2)) if s]

    def _parabola_error(samples):
        if not samples:
            return None
        errors = []
        for sample in samples:
            s = sample['distance']
            expected_delta = 0.5 * d2omega_dk2 * (s ** 2)
            actual_delta = sample['omega'] - omega0
            scale = max(abs(expected_delta), parab_error_floor)
            errors.append(abs(actual_delta - expected_delta) / scale)
        if not errors:
            return None
        return float(np.mean(errors))

    def _mean_distance(samples):
        if not samples:
            return None
        return float(np.mean([s['distance'] for s in samples]))

    parab_error_near = _parabola_error(near_samples)
    parab_error_far = _parabola_error(far_samples)
    near_span = _mean_distance(near_samples)
    far_span = _mean_distance(far_samples)

    k_parab_far = k_parab
    span_candidates = []
    if near_span is not None and (parab_error_near is None or parab_error_near <= parab_error_threshold):
        span_candidates.append(near_span)
    if far_span is not None and (parab_error_far is None or parab_error_far <= parab_error_threshold):
        span_candidates.append(far_span)
    if span_candidates:
        k_parab_far = max(k_parab_far, max(span_candidates))
    
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
        'k_parab_far': float(k_parab_far),
        'parab_error_near': float(parab_error_near) if parab_error_near is not None else None,
        'parab_error_far': float(parab_error_far) if parab_error_far is not None else None,
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
    
    # Finite difference step and stencil order in k-space
    dk = float(config.get('phase1_dk', 0.01))  # In units of 2π/a
    fd_order = int(config.get('phase1_fd_order', 4))
    if fd_order not in {2, 4}:
        raise ValueError(f"Unsupported phase1_fd_order={fd_order}; expected 2 or 4")
    
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
        labels: list[tuple[int, int]] = []
        k_points: list[mp.Vector3] = []
        if fd_order == 4:
            offsets = [-2, -1, 0, 1, 2]
            for ox in offsets:
                for oy in offsets:
                    labels.append((ox, oy))
                    k_points.append(mp.Vector3(k0[0] + ox * dk, k0[1] + oy * dk, k0[2]))
        else:
            labels = [
                (0, 0),
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            for ox, oy in labels:
                k_points.append(mp.Vector3(k0[0] + ox * dk, k0[1] + oy * dk, k0[2]))
        
        ms.k_points = k_points
        
        # Run TE mode calculation
        ms.run_te()
        
        # Extract frequencies
        freqs_array = np.array(ms.all_freqs)
        if len(freqs_array.shape) == 1:
            freqs_array = freqs_array.reshape(-1, num_bands)
        
        # Extract values at each k-point
        omega_values = {labels[idx]: freqs_array[idx, band_index] for idx in range(len(labels))}
        omega0 = float(omega_values[(0, 0)])

        if fd_order == 4:
            offsets = [-2, -1, 0, 1, 2]
            coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
            coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0

            vg_x = sum(
                coeff_first[idx] * omega_values[(offset, 0)] for idx, offset in enumerate(offsets)
            ) / dk
            vg_y = sum(
                coeff_first[idx] * omega_values[(0, offset)] for idx, offset in enumerate(offsets)
            ) / dk

            d2_xx = sum(
                coeff_second[idx] * omega_values[(offset, 0)] for idx, offset in enumerate(offsets)
            ) / (dk ** 2)
            d2_yy = sum(
                coeff_second[idx] * omega_values[(0, offset)] for idx, offset in enumerate(offsets)
            ) / (dk ** 2)

            d2_xy = 0.0
            for ix, ox in enumerate(offsets):
                for iy, oy in enumerate(offsets):
                    d2_xy += coeff_first[ix] * coeff_first[iy] * omega_values[(ox, oy)]
            d2_xy /= (dk ** 2)
        else:
            omega_xp = omega_values[(1, 0)]
            omega_xm = omega_values[(-1, 0)]
            omega_yp = omega_values[(0, 1)]
            omega_ym = omega_values[(0, -1)]
            omega_pp = omega_values[(1, 1)]
            omega_pm = omega_values[(1, -1)]
            omega_mp = omega_values[(-1, 1)]
            omega_mm = omega_values[(-1, -1)]

            vg_x = (omega_xp - omega_xm) / (2 * dk)
            vg_y = (omega_yp - omega_ym) / (2 * dk)

            d2_xx = (omega_xp - 2 * omega0 + omega_xm) / (dk ** 2)
            d2_yy = (omega_yp - 2 * omega0 + omega_ym) / (dk ** 2)
            d2_xy = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk ** 2)

        vg = np.array([vg_x, vg_y])
        
        # Construct inverse mass tensor
        # M⁻¹ is the Hessian of ω(k)
        M_inv = np.array([
            [d2_xx, d2_xy],
            [d2_xy, d2_yy]
        ])
        
        # Regularize mass tensor: avoid near-zero eigenvalues while preserving sign
        # Negative eigenvalues are physically meaningful (band maxima)
        eigvals, eigvecs = np.linalg.eigh(M_inv)
        min_abs_eig = 1e-6
        mask = np.abs(eigvals) < min_abs_eig
        eigvals = np.where(mask, np.sign(eigvals) * min_abs_eig, eigvals)
        # Handle exact zeros (sign returns 0) by defaulting to positive
        eigvals = np.where(eigvals == 0, min_abs_eig, eigvals)
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
