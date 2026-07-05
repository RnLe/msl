"""
Subpixel smoothing for FDFD epsilon grids.

Ported from analytic_geometry.rs — implements MPB-style subpixel smoothing:

    ε̃⁻¹ = P⟨ε⁻¹⟩ + (I − P)⟨ε⟩⁻¹

where P = n ⊗ n is the projection onto the interface normal, and
⟨ε⟩, ⟨ε⁻¹⟩ are area-weighted averages.

For TM polarization (Ez modes), the relevant scalar is ⟨ε⟩ (arithmetic mean),
since Ez is tangential to all in-plane interfaces.

For TE polarization (Hz modes), the full 2×2 tensor ε̃⁻¹ is needed.

Reference: A. Farjadpour et al., Optics Letters 31, 2972 (2006).
"""

import numpy as np
from typing import Tuple, Optional


def compute_filling_fraction_subgrid(
    pixel_center: np.ndarray,
    pixel_v1: np.ndarray,
    pixel_v2: np.ndarray,
    rod_center: np.ndarray,
    rod_radius: float,
    n_sub: int = 16,
) -> float:
    """
    Compute the filling fraction of a circle within a parallelogram pixel
    using sub-grid sampling.

    The pixel is defined by its center and two edge vectors:
        pixel spans from center - v1/2 - v2/2 to center + v1/2 + v2/2

    Args:
        pixel_center: (2,) center of the pixel in Cartesian coordinates
        pixel_v1: (2,) first edge vector of the pixel (e.g. L1/Nx)
        pixel_v2: (2,) second edge vector of the pixel (e.g. L2/Ny)
        rod_center: (2,) center of the circular rod
        rod_radius: radius of the rod
        n_sub: number of subdivisions per edge (total samples = n_sub²)

    Returns:
        Filling fraction in [0, 1].
    """
    r_sq = rod_radius * rod_radius

    # Sub-pixel centers in parametric coords t1, t2 ∈ [0, 1)
    # Offset from pixel corner: corner = center - v1/2 - v2/2
    corner = pixel_center - 0.5 * pixel_v1 - 0.5 * pixel_v2

    count = 0
    for i in range(n_sub):
        t1 = (i + 0.5) / n_sub
        for j in range(n_sub):
            t2 = (j + 0.5) / n_sub
            pt = corner + t1 * pixel_v1 + t2 * pixel_v2
            dx = pt[0] - rod_center[0]
            dy = pt[1] - rod_center[1]
            if dx * dx + dy * dy <= r_sq:
                count += 1

    return count / (n_sub * n_sub)


def compute_filling_fraction_subgrid_vectorized(
    pixel_centers: np.ndarray,
    pixel_v1: np.ndarray,
    pixel_v2: np.ndarray,
    rod_center: np.ndarray,
    rod_radius: float,
    n_sub: int = 16,
) -> np.ndarray:
    """
    Vectorized filling fraction computation for many pixels against one rod.

    Args:
        pixel_centers: (N, 2) array of pixel centers
        pixel_v1: (2,) first edge vector (same for all pixels)
        pixel_v2: (2,) second edge vector (same for all pixels)
        rod_center: (2,) center of the rod
        rod_radius: radius of the rod
        n_sub: subdivisions per edge

    Returns:
        (N,) array of filling fractions.
    """
    r_sq = rod_radius * rod_radius
    N = len(pixel_centers)

    # Corners: (N, 2)
    corners = pixel_centers - 0.5 * pixel_v1 - 0.5 * pixel_v2

    # Sub-pixel offsets: (n_sub*n_sub, 2)
    t1 = (np.arange(n_sub) + 0.5) / n_sub
    t2 = (np.arange(n_sub) + 0.5) / n_sub
    T1, T2 = np.meshgrid(t1, t2, indexing='ij')
    offsets = T1.ravel()[:, None] * pixel_v1 + T2.ravel()[:, None] * pixel_v2  # (n_sub², 2)

    # For each sub-pixel offset, check distance to rod center
    counts = np.zeros(N, dtype=np.int32)
    for offset in offsets:
        pts = corners + offset  # (N, 2)
        dx = pts[:, 0] - rod_center[0]
        dy = pts[:, 1] - rod_center[1]
        counts += (dx * dx + dy * dy <= r_sq).astype(np.int32)

    return counts / (n_sub * n_sub)


def compute_interface_normal(pixel_center: np.ndarray, rod_center: np.ndarray) -> np.ndarray:
    """
    Compute the interface normal for a circular rod at a pixel.

    For a circle, the outward normal is the radial direction from
    rod center toward pixel center.

    Args:
        pixel_center: (2,) pixel center
        rod_center: (2,) rod center

    Returns:
        (2,) unit normal vector, or [1, 0] if degenerate.
    """
    d = pixel_center - rod_center
    dist = np.linalg.norm(d)
    if dist < 1e-12:
        return np.array([1.0, 0.0])
    return d / dist


def compute_smoothed_dielectric(
    fill_frac: float,
    eps_inside: float,
    eps_outside: float,
    normal: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute smoothed dielectric using the MPB anisotropic smoothing formula.

    ε̃⁻¹ = P⟨ε⁻¹⟩ + (I − P)⟨ε⟩⁻¹

    where P = n ⊗ n.

    Args:
        fill_frac: fraction of pixel inside the rod [0, 1]
        eps_inside: permittivity inside the rod
        eps_outside: permittivity outside
        normal: (2,) unit interface normal

    Returns:
        avg_eps: area-weighted ⟨ε⟩
        avg_inv_eps: area-weighted ⟨ε⁻¹⟩
        inv_eps_tensor: (2,2) inverse permittivity tensor ε̃⁻¹
    """
    avg_eps = fill_frac * eps_inside + (1.0 - fill_frac) * eps_outside
    avg_inv_eps = fill_frac / eps_inside + (1.0 - fill_frac) / eps_outside

    # Uniform pixel — isotropic
    if fill_frac < 1e-10:
        inv = 1.0 / eps_outside
        return eps_outside, inv, np.array([[inv, 0.0], [0.0, inv]])
    if fill_frac > 1.0 - 1e-10:
        inv = 1.0 / eps_inside
        return eps_inside, inv, np.array([[inv, 0.0], [0.0, inv]])

    # MPB formula: ε̃⁻¹ = (1/⟨ε⟩) I + (⟨ε⁻¹⟩ − 1/⟨ε⟩) P
    inv_tangential = 1.0 / avg_eps
    inv_normal = avg_inv_eps

    nx, ny = normal[0], normal[1]
    P = np.array([[nx * nx, nx * ny],
                  [nx * ny, ny * ny]])

    delta = inv_normal - inv_tangential
    inv_eps_tensor = inv_tangential * np.eye(2) + delta * P

    return avg_eps, avg_inv_eps, inv_eps_tensor


def build_smoothed_eps_monolayer(
    resolution: int,
    a: float = 1.0,
    r_over_a: float = 0.2,
    eps_rod: float = 11.56,
    eps_bg: float = 1.0,
    n_sub: int = 16,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build subpixel-smoothed epsilon for a honeycomb monolayer unit cell.

    Returns both the scalar smoothed ε (for TM) and the binary ε (for comparison).
    The scalar smoothing uses ⟨ε⟩ = f·ε_rod + (1−f)·ε_bg (arithmetic mean),
    which is correct for TM since Ez is tangential to in-plane interfaces.

    Args:
        resolution: grid points per lattice constant
        a: lattice constant
        r_over_a: rod radius / a
        eps_rod: rod permittivity
        eps_bg: background permittivity
        n_sub: sub-grid resolution for filling fraction

    Returns:
        eps_binary: (Nx, Ny) binary (staircase) epsilon grid
        eps_smoothed: (Nx, Ny) subpixel-smoothed epsilon grid
        info: dict with geometry metadata and diagnostics
    """
    r = r_over_a * a

    # Lattice vectors (triangular)
    B = a * np.array([[1.0, 0.5],
                      [0.0, np.sqrt(3) / 2]])
    B_inv = np.linalg.inv(B)

    # Sublattice positions (honeycomb: 2 atoms per cell)
    sublattice_frac = np.array([[0.0, 0.0], [1.0 / 3, 1.0 / 3]])

    Nx = Ny = resolution
    # Pixel edge vectors in Cartesian
    v1 = B[:, 0] / Nx  # L1 / Nx
    v2 = B[:, 1] / Ny  # L2 / Ny

    # Grid of pixel centers in Cartesian
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    # Pixel centers shifted by half a pixel to center of pixel
    S1c = (np.arange(Nx) + 0.5) / Nx
    S2c = (np.arange(Ny) + 0.5) / Ny
    S1c, S2c = np.meshgrid(S1c, S2c, indexing='ij')

    X = S1 * B[0, 0] + S2 * B[0, 1]
    Y = S1 * B[1, 0] + S2 * B[1, 1]

    Xc = S1c * B[0, 0] + S2c * B[0, 1]
    Yc = S1c * B[1, 0] + S2c * B[1, 1]

    # Binary epsilon (same as existing code — uses grid-point-inside test)
    eps_binary = np.full((Nx, Ny), eps_bg, dtype=np.float64)
    XY = np.stack([X, Y], axis=0)

    for sub_frac in sublattice_frac:
        offset = B @ sub_frac
        shifted = XY - offset[:, None, None]
        frac = np.einsum('ij,jkl->ikl', B_inv, shifted)
        f_near = frac - np.round(frac)
        disp = np.einsum('ij,jkl->ikl', B, f_near)
        dist_sq = disp[0]**2 + disp[1]**2
        eps_binary[dist_sq < r**2] = eps_rod

    # Smoothed epsilon: start from binary, fix boundary pixels
    eps_smoothed = eps_binary.copy()

    pixel_diag = np.linalg.norm(v1 + v2)

    n_boundary = 0
    n_smoothed = 0

    for sub_frac in sublattice_frac:
        offset = B @ sub_frac

        shifted = XY - offset[:, None, None]
        frac_coords = np.einsum('ij,jkl->ikl', B_inv, shifted)
        f_near = frac_coords - np.round(frac_coords)
        disp = np.einsum('ij,jkl->ikl', B, f_near)
        dist = np.sqrt(disp[0]**2 + disp[1]**2)

        boundary_mask = np.abs(dist - r) < pixel_diag
        n_bdry = boundary_mask.sum()
        n_boundary += n_bdry

        if n_bdry == 0:
            continue

        # Vectorized: extract boundary pixel coordinates
        bi, bj = np.where(boundary_mask)
        pc_x = X[bi, bj]
        pc_y = Y[bi, bj]

        nlf_0 = frac_coords[0][bi, bj] - f_near[0][bi, bj]
        nlf_1 = frac_coords[1][bi, bj] - f_near[1][bi, bj]
        rod_x = B[0, 0] * nlf_0 + B[0, 1] * nlf_1 + offset[0]
        rod_y = B[1, 0] * nlf_0 + B[1, 1] * nlf_1 + offset[1]

        ff = _compute_filling_fractions_vectorized(
            pc_x, pc_y, rod_x, rod_y, v1, v2, r, n_sub=n_sub
        )

        valid = (ff > 1e-10) & (ff < 1.0 - 1e-10)
        valid_idx = np.where(valid)[0]
        n_smoothed += len(valid_idx)

        for k in valid_idx:
            eps_smoothed[bi[k], bj[k]] = ff[k] * eps_rod + (1.0 - ff[k]) * eps_bg

    info = {
        'B': B,
        'B_inv': B_inv,
        'L1': B[:, 0],
        'L2': B[:, 1],
        'B_super': B,  # for FDFD compatibility
        'Nx': Nx,
        'Ny': Ny,
        'resolution': resolution,
        'n_sub': n_sub,
        'n_boundary_candidates': n_boundary,
        'n_smoothed': n_smoothed,
        'pixel_v1': v1,
        'pixel_v2': v2,
        'pixel_diag': pixel_diag,
    }

    return eps_binary, eps_smoothed, info


def _compute_filling_fractions_vectorized(
    pixel_centers_x: np.ndarray,
    pixel_centers_y: np.ndarray,
    rod_centers_x: np.ndarray,
    rod_centers_y: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    rod_radius: float,
    n_sub: int = 16,
) -> np.ndarray:
    """
    Compute filling fractions for N boundary pixels in a fully vectorized way.

    All inputs are 1D arrays of length N (one per boundary pixel).
    Returns (N,) array of filling fractions.
    """
    N = len(pixel_centers_x)
    if N == 0:
        return np.empty(0)

    r_sq = rod_radius * rod_radius

    # Sub-pixel offsets: (n_sub², 2) relative to pixel corner
    t = (np.arange(n_sub) + 0.5) / n_sub
    T1, T2 = np.meshgrid(t, t, indexing='ij')
    # offsets[k] = T1[k]*v1 + T2[k]*v2, shape (n_sub², 2)
    t1_flat = T1.ravel()
    t2_flat = T2.ravel()
    n_samples = len(t1_flat)

    # Corner positions: pixel_center - v1/2 - v2/2
    corner_x = pixel_centers_x - 0.5 * v1[0] - 0.5 * v2[0]
    corner_y = pixel_centers_y - 0.5 * v1[1] - 0.5 * v2[1]

    # Process in chunks to control memory
    CHUNK = 8192
    counts = np.zeros(N, dtype=np.int32)

    for start in range(0, n_samples, CHUNK):
        end = min(start + CHUNK, n_samples)
        chunk_t1 = t1_flat[start:end]  # (C,)
        chunk_t2 = t2_flat[start:end]

        # Sub-pixel x,y for all N pixels × C samples
        # offset_x[c] = t1[c]*v1[0] + t2[c]*v2[0]
        off_x = chunk_t1 * v1[0] + chunk_t2 * v2[0]  # (C,)
        off_y = chunk_t1 * v1[1] + chunk_t2 * v2[1]

        # sample positions: (N,) + (C,) → broadcast to (N, C)
        sx = corner_x[:, None] + off_x[None, :]  # (N, C)
        sy = corner_y[:, None] + off_y[None, :]

        dx = sx - rod_centers_x[:, None]
        dy = sy - rod_centers_y[:, None]
        inside = (dx * dx + dy * dy) <= r_sq  # (N, C)
        counts += inside.sum(axis=1).astype(np.int32)

    return counts / (n_sub * n_sub)


def build_smoothed_eps_supercell(
    eps_binary: np.ndarray,
    supercell_info: dict,
    n_sub: int = 16,
    eps_rod: float = 11.56,
    eps_bg: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """
    Build subpixel-smoothed epsilon for a moiré supercell.

    Takes the existing binary epsilon grid and smooths boundary pixels
    using sub-grid sampling. Fully vectorized for performance.

    Args:
        eps_binary: (Nx, Ny) binary epsilon grid from build_supercell_eps
        supercell_info: geometry info dict from build_supercell_eps
        n_sub: sub-grid resolution
        eps_rod: rod permittivity
        eps_bg: background permittivity

    Returns:
        eps_smoothed: (Nx, Ny) smoothed epsilon grid
        smooth_info: dict with smoothing diagnostics
    """
    from .supercell_geometry import (
        build_monolayer_basis, get_sublattice_positions, rotation_matrix_2d
    )

    Nx = supercell_info['Nx']
    Ny = supercell_info['Ny']
    B_super = supercell_info['B_super']
    B_mono = supercell_info['B_mono']
    theta_rad = supercell_info['theta_rad']
    r = supercell_info['r_over_a'] * supercell_info['a']
    lattice_type = supercell_info['lattice_type']

    L1 = B_super[:, 0]
    L2 = B_super[:, 1]
    B_mono_inv = np.linalg.inv(B_mono)
    R = rotation_matrix_2d(theta_rad)
    B_rot = R @ B_mono
    B_rot_inv = np.linalg.inv(B_rot)

    sublattice_frac = get_sublattice_positions(lattice_type)

    # Pixel edge vectors
    v1 = L1 / Nx
    v2 = L2 / Ny
    pixel_diag = np.linalg.norm(v1 + v2)

    # Grid coordinates
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]
    XY = np.stack([X, Y], axis=0)

    eps_smoothed = eps_binary.copy()
    n_boundary = 0
    n_smoothed = 0

    # Process both layers
    layers = [
        (B_mono, B_mono_inv, "Layer 1 (unrotated)"),
        (B_rot, B_rot_inv, "Layer 2 (rotated)"),
    ]

    import time as _time
    for B_layer, B_layer_inv, label in layers:
        for sub_frac in sublattice_frac:
            t0 = _time.time()
            offset = B_layer @ sub_frac

            shifted = XY - offset[:, None, None]
            frac_coords = np.einsum('ij,jkl->ikl', B_layer_inv, shifted)
            f_near = frac_coords - np.round(frac_coords)
            disp = np.einsum('ij,jkl->ikl', B_layer, f_near)
            dist = np.sqrt(disp[0]**2 + disp[1]**2)

            boundary_mask = np.abs(dist - r) < pixel_diag
            n_bdry = boundary_mask.sum()
            n_boundary += n_bdry

            if n_bdry == 0:
                continue

            # Extract boundary pixel data (vectorized)
            bi, bj = np.where(boundary_mask)
            pc_x = X[bi, bj]
            pc_y = Y[bi, bj]

            # Nearest rod centers
            nlf_0 = frac_coords[0][bi, bj] - f_near[0][bi, bj]
            nlf_1 = frac_coords[1][bi, bj] - f_near[1][bi, bj]
            rod_x = B_layer[0, 0] * nlf_0 + B_layer[0, 1] * nlf_1 + offset[0]
            rod_y = B_layer[1, 0] * nlf_0 + B_layer[1, 1] * nlf_1 + offset[1]

            # Vectorized filling fraction
            ff = _compute_filling_fractions_vectorized(
                pc_x, pc_y, rod_x, rod_y, v1, v2, r, n_sub=n_sub
            )

            # Apply smoothing
            valid = (ff > 1e-10) & (ff < 1.0 - 1e-10)
            valid_idx = np.where(valid)[0]
            n_smoothed += len(valid_idx)

            for k in valid_idx:
                i, j = bi[k], bj[k]
                eps_eff = ff[k] * eps_rod + (1.0 - ff[k]) * eps_bg
                eps_smoothed[i, j] = max(eps_smoothed[i, j], eps_eff)

            dt = _time.time() - t0
            print(f"    {label}, sublattice {sub_frac}: {n_bdry} boundary, "
                  f"{len(valid_idx)} smoothed, {dt:.1f}s")

    smooth_info = {
        'n_boundary_candidates': n_boundary,
        'n_smoothed': n_smoothed,
        'n_sub': n_sub,
        'pixel_diag': pixel_diag,
        'pixel_v1_norm': np.linalg.norm(v1),
        'pixel_v2_norm': np.linalg.norm(v2),
    }

    return eps_smoothed, smooth_info
