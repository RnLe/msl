"""
Moiré supercell geometry builder for direct FDFD validation.

Constructs the dielectric function ε(x, y) for a twisted bilayer photonic
crystal on a commensurate moiré supercell. The geometry consists of two
superimposed layers of dielectric rods/holes, with the second layer twisted
by the commensurate twist angle θ.

The supercell is defined by lattice vectors L1, L2 (generally oblique).
We work in fractional supercell coordinates (s1, s2) ∈ [0, 1)² and use the
metric tensor g_ij = L_i · L_j for gradient operations.

Supports: square, hex (triangular), and honeycomb lattices.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .commensurate_utils import build_supercell_vectors, commensurate_twist_angle


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """2D rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """
    Build the monolayer Bravais lattice basis vectors.

    Returns:
        (2, 2) array: columns are lattice vectors a1, a2
    """
    if lattice_type == 'square':
        return a * np.array([[1.0, 0.0], [0.0, 1.0]])
    elif lattice_type in ('hex', 'hexagonal', 'triangular', 'honeycomb'):
        return a * np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def get_sublattice_positions(lattice_type: str) -> np.ndarray:
    """
    Get sublattice atom positions in fractional coordinates of the unit cell.

    Returns:
        (n_atoms, 2) array of fractional positions
    """
    if lattice_type == 'honeycomb':
        # Two atoms per unit cell at (0,0) and (1/3, 1/3) in fractional coords
        return np.array([[0.0, 0.0], [1.0 / 3, 1.0 / 3]])
    else:
        # Single atom at origin
        return np.array([[0.0, 0.0]])


def build_supercell_eps(
    lattice_type: str,
    m: int,
    n: int,
    a: float = 1.0,
    r_over_a: float = 0.2,
    eps_rod: float = 11.56,
    eps_bg: float = 1.0,
    Nx: int = 256,
    Ny: int = 256,
    subpixel_smoothing: bool = False,
    smoothing_Nsub: int = 8,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the dielectric function ε(x, y) on a moiré supercell grid.

    The geometry is two superimposed layers of a 2D photonic crystal:
    - Layer 1: unrotated, lattice vectors a1, a2
    - Layer 2: rotated by θ (commensurate twist angle for indices m, n)

    At each grid point, ε = eps_rod if the point is inside any rod (from
    either layer), otherwise ε = eps_bg. This models a "merged bilayer"
    where both layers are projected into the same 2D plane.

    The grid is defined in fractional supercell coordinates (s1, s2) ∈ [0,1)².
    Physical coordinates: x = B_super @ s, where B_super = [L1 | L2].

    Args:
        lattice_type: 'square', 'hex', 'triangular', or 'honeycomb'
        m, n: commensurate coincidence indices
        a: lattice constant
        r_over_a: rod radius / lattice constant
        eps_rod: rod dielectric constant
        eps_bg: background dielectric constant
        Nx, Ny: grid resolution (points per supercell side)
        subpixel_smoothing: if True, average eps over Nsub×Nsub sub-samples per pixel
        smoothing_Nsub: number of sub-pixel samples per dimension

    Returns:
        eps_grid: (Nx, Ny) array of dielectric values
        info: dict with geometry metadata
    """
    theta_rad = commensurate_twist_angle(lattice_type, m, n)
    theta_deg = np.degrees(theta_rad)

    # Monolayer basis vectors (columns of B_mono)
    B_mono = build_monolayer_basis(lattice_type, a)
    a1 = B_mono[:, 0]
    a2 = B_mono[:, 1]

    # Supercell lattice vectors
    B_super = build_supercell_vectors(lattice_type, m, n, a)
    L1 = B_super[:, 0]
    L2 = B_super[:, 1]

    # N_cells = supercell area / unit_cell_area
    area_unit = abs(np.cross(a1, a2))
    area_super = abs(np.cross(L1, L2))
    N_cells = int(round(area_super / area_unit))

    # Rotation matrix for layer 2
    R = rotation_matrix_2d(theta_rad)

    # Layer 2 basis vectors (rotated)
    a1_rot = R @ a1
    a2_rot = R @ a2

    # Rod radius
    r = r_over_a * a

    # Sublattice positions
    sublattice_frac = get_sublattice_positions(lattice_type)

    # Build grid in fractional supercell coordinates
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    # Convert to Cartesian: x = s1*L1 + s2*L2
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]

    # Layer 2 rotated basis
    B_rot = np.column_stack([a1_rot, a2_rot])
    B_rot_inv = np.linalg.inv(B_rot)
    B_mono_inv = np.linalg.inv(B_mono)

    def _fill_binary(Xp, Yp):
        """Binary rod assignment for given coordinates."""
        eps = np.full((Nx, Ny), eps_bg, dtype=np.float64)
        XY = np.stack([Xp, Yp], axis=0)
        for sub_pos in sublattice_frac:
            # Layer 1
            offset = B_mono @ sub_pos
            shifted = XY - offset[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_mono_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_mono, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps[dist_sq < r**2] = eps_rod
            # Layer 2
            offset_rot = B_rot @ sub_pos
            shifted = XY - offset_rot[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_rot_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_rot, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps[dist_sq < r**2] = eps_rod
        return eps

    if not subpixel_smoothing:
        eps_grid = _fill_binary(X, Y)
    else:
        # Sub-pixel smoothing: average ε over Nsub×Nsub sub-samples per pixel
        ds1 = 1.0 / Nx
        ds2 = 1.0 / Ny
        Nsub = smoothing_Nsub
        sub_offsets = (np.arange(Nsub) + 0.5) / Nsub - 0.5
        eps_grid = np.zeros((Nx, Ny), dtype=np.float64)
        for si in sub_offsets:
            for sj in sub_offsets:
                Xs = X + si * ds1 * L1[0] + sj * ds2 * L2[0]
                Ys = Y + si * ds1 * L1[1] + sj * ds2 * L2[1]
                eps_grid += _fill_binary(Xs, Ys)
        eps_grid /= (Nsub * Nsub)

    # Metric tensor for the supercell (needed by FDFD solver)
    g11 = np.dot(L1, L1)
    g12 = np.dot(L1, L2)
    g22 = np.dot(L2, L2)
    metric = np.array([[g11, g12], [g12, g22]])
    det_g = g11 * g22 - g12**2

    info = {
        'lattice_type': lattice_type,
        'm': m,
        'n': n,
        'a': a,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'r_over_a': r_over_a,
        'eps_rod': eps_rod,
        'eps_bg': eps_bg,
        'N_cells': N_cells,
        'Nx': Nx,
        'Ny': Ny,
        'B_super': B_super,
        'B_mono': B_mono,
        'metric': metric,
        'det_g': det_g,
        'L1': L1,
        'L2': L2,
        'area_super': area_super,
        'area_unit': area_unit,
        'subpixel_smoothing': subpixel_smoothing,
        'smoothing_Nsub': smoothing_Nsub if subpixel_smoothing else 0,
    }

    return eps_grid, info


def build_moire_supercell_eps(
    lattice_type: str,
    theta_rad: float,
    a: float = 1.0,
    r_over_a: float = 0.2,
    eps_rod: float = 11.56,
    eps_bg: float = 1.0,
    Nx: int = 256,
    Ny: int = 256,
    subpixel_smoothing: bool = True,
    smoothing_Nsub: int = 8,
) -> Tuple[np.ndarray, Dict]:
    """
    Build ε(x, y) on a SINGLE moiré supercell.

    Unlike build_supercell_eps which uses the commensurate supercell (containing
    multiple moiré cells), this uses the moiré lattice vectors directly.
    The resulting cell matches the EA's spatial domain exactly: one moiré period.

    For a square lattice, the moiré lattice vectors are:
        L_moire = a / (2 sin(θ/2))  ×  [ê₁, ê₂]   (square moiré cell)

    The ε map is generally NOT exactly periodic (the monolayer rods don't
    tile perfectly over one moiré period), but the *modulation* is periodic,
    which is exactly what the envelope approximation models.

    Subpixel smoothing (analogous to MPB's material averaging) can be
    enabled to reduce staircase artifacts at material boundaries.

    Args:
        lattice_type: 'square' (others not yet supported)
        theta_rad: twist angle in radians
        a: lattice constant
        r_over_a: rod radius / lattice constant
        eps_rod: rod dielectric constant
        eps_bg: background dielectric constant
        Nx, Ny: grid resolution (points per moiré cell side)
        subpixel_smoothing: if True, use sub-pixel material averaging
        smoothing_Nsub: number of sub-pixel samples per dimension (Nsub × Nsub)

    Returns:
        eps_grid: (Nx, Ny) array of dielectric values
        info: dict with geometry metadata
    """
    if lattice_type != 'square':
        raise NotImplementedError(f"Moiré supercell only implemented for 'square', got '{lattice_type}'")

    theta_deg = np.degrees(theta_rad)

    # Moiré lattice for square lattice: L_moire = a / (2 sin(θ/2))
    L_moire = a / (2.0 * np.sin(theta_rad / 2.0))

    # Moiré lattice vectors: rotated by +θ/2 relative to layer 1.
    # Verified numerically: +θ/2 gives cos(L1_comm, L1_nc) = 1.0,
    # matching the commensurate supercell vector direction exactly.
    c2, s2 = np.cos(theta_rad / 2.0), np.sin(theta_rad / 2.0)
    L1 = L_moire * np.array([c2, s2])
    L2 = L_moire * np.array([-s2, c2])
    B_super = np.column_stack([L1, L2])

    # Monolayer basis
    B_mono = build_monolayer_basis(lattice_type, a)
    a1 = B_mono[:, 0]
    a2 = B_mono[:, 1]

    # Rotation matrix for layer 2
    R = rotation_matrix_2d(theta_rad)
    a1_rot = R @ a1
    a2_rot = R @ a2
    B_rot = np.column_stack([a1_rot, a2_rot])

    # Rod radius
    r = r_over_a * a

    # Sublattice positions
    sublattice_frac = get_sublattice_positions(lattice_type)

    # Build grid in fractional moiré coordinates (s1, s2) ∈ [0, 1)²
    s1 = (np.arange(Nx) + 0.5) / Nx  # cell-centered
    s2 = (np.arange(Ny) + 0.5) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    # Cartesian coordinates
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]

    # Helper: compute rod fill fraction at a single point
    B_mono_inv = np.linalg.inv(B_mono)
    B_rot_inv = np.linalg.inv(B_rot)

    if not subpixel_smoothing:
        # Simple binary assignment (same as build_supercell_eps)
        eps_grid = np.full((Nx, Ny), eps_bg, dtype=np.float64)
        XY = np.stack([X, Y], axis=0)

        for sub_pos in sublattice_frac:
            # Layer 1
            offset = B_mono @ sub_pos
            shifted = XY - offset[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_mono_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_mono, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps_grid[dist_sq < r**2] = eps_rod

            # Layer 2
            offset_rot = B_rot @ sub_pos
            shifted = XY - offset_rot[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_rot_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_rot, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps_grid[dist_sq < r**2] = eps_rod
    else:
        # Sub-pixel smoothing: average ε over sub-pixel samples
        # This mimics MPB's analytic material averaging and dramatically
        # reduces staircase artifacts for high-contrast boundaries.
        eps_grid = np.zeros((Nx, Ny), dtype=np.float64)

        ds1 = 1.0 / Nx
        ds2 = 1.0 / Ny
        Nsub = smoothing_Nsub

        # Sub-pixel offsets (centered within pixel)
        sub_offsets = (np.arange(Nsub) + 0.5) / Nsub - 0.5  # [-0.5+0.5/N, ..., 0.5-0.5/N]

        for si in sub_offsets:
            for sj in sub_offsets:
                # Shifted coordinates
                Xs = X + si * ds1 * L1[0] + sj * ds2 * L2[0]
                Ys = Y + si * ds1 * L1[1] + sj * ds2 * L2[1]
                XYs = np.stack([Xs, Ys], axis=0)

                sub_eps = np.full((Nx, Ny), eps_bg, dtype=np.float64)

                for sub_pos in sublattice_frac:
                    # Layer 1
                    offset = B_mono @ sub_pos
                    shifted = XYs - offset[:, None, None]
                    frac = np.einsum('ij,jkl->ikl', B_mono_inv, shifted)
                    f_near = frac - np.round(frac)
                    disp = np.einsum('ij,jkl->ikl', B_mono, f_near)
                    dist_sq = disp[0]**2 + disp[1]**2
                    sub_eps[dist_sq < r**2] = eps_rod

                    # Layer 2
                    offset_rot = B_rot @ sub_pos
                    shifted = XYs - offset_rot[:, None, None]
                    frac = np.einsum('ij,jkl->ikl', B_rot_inv, shifted)
                    f_near = frac - np.round(frac)
                    disp = np.einsum('ij,jkl->ikl', B_rot, f_near)
                    dist_sq = disp[0]**2 + disp[1]**2
                    sub_eps[dist_sq < r**2] = eps_rod

                eps_grid += sub_eps

        eps_grid /= (Nsub * Nsub)

    # Metric tensor
    g11 = np.dot(L1, L1)
    g12 = np.dot(L1, L2)
    g22 = np.dot(L2, L2)
    metric = np.array([[g11, g12], [g12, g22]])
    det_g = g11 * g22 - g12**2

    area_unit = abs(np.cross(a1, a2))
    area_super = abs(np.cross(L1, L2))

    info = {
        'lattice_type': lattice_type,
        'a': a,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'r_over_a': r_over_a,
        'eps_rod': eps_rod,
        'eps_bg': eps_bg,
        'L_moire': L_moire,
        'Nx': Nx,
        'Ny': Ny,
        'B_super': B_super,
        'B_mono': B_mono,
        'metric': metric,
        'det_g': det_g,
        'L1': L1,
        'L2': L2,
        'area_super': area_super,
        'area_unit': area_unit,
        'subpixel_smoothing': subpixel_smoothing,
        'smoothing_Nsub': smoothing_Nsub if subpixel_smoothing else 0,
    }

    return eps_grid, info


def build_moire_bz_path(
    B_super: np.ndarray,
    lattice_type: str,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Build a k-path through the moiré Brillouin zone high-symmetry points.

    The moiré reciprocal lattice vectors are:
        G_super = 2π (B_super^{-T})

    High-symmetry points depend on lattice type.

    Args:
        B_super: (2, 2) supercell lattice vectors (columns)
        lattice_type: 'square', 'hex', or 'honeycomb'
        n_points: points per segment

    Returns:
        q_path: (n_total, 2) array of q-points in Cartesian k-space
        q_dist: (n_total,) array of cumulative distance along path
        labels: list of (index, label) for high-symmetry points
    """
    # Reciprocal lattice vectors of supercell
    G_super = 2 * np.pi * np.linalg.inv(B_super).T  # columns are G1, G2
    g1 = G_super[:, 0]
    g2 = G_super[:, 1]

    if lattice_type in ('hex', 'hexagonal', 'triangular', 'honeycomb'):
        # Γ → M → K → Γ
        Gamma = np.array([0.0, 0.0])
        M = 0.5 * g1
        K = (2 * g1 + g2) / 3.0
        points = [Gamma, M, K, Gamma]
        label_names = ['Γ_m', 'M_m', 'K_m', 'Γ_m']
    elif lattice_type == 'square':
        # Γ → X → M → Γ
        Gamma = np.array([0.0, 0.0])
        X = 0.5 * g1
        M = 0.5 * (g1 + g2)
        points = [Gamma, X, M, Gamma]
        label_names = ['Γ_m', 'X_m', 'M_m', 'Γ_m']
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    q_path = []
    q_dist = []
    labels = [(0, label_names[0])]
    cumulative = 0.0

    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        segment = np.linspace(0, 1, n_points, endpoint=(i == len(points) - 2))
        seg_points = p0[None, :] + segment[:, None] * (p1 - p0)[None, :]
        seg_dists = np.linalg.norm(np.diff(seg_points, axis=0), axis=1)

        for j, pt in enumerate(seg_points):
            if j == 0 and i > 0:
                continue  # avoid duplicating endpoints
            q_path.append(pt)
            if j > 0:
                cumulative += seg_dists[j - 1]
            q_dist.append(cumulative)

        labels.append((len(q_path) - 1, label_names[i + 1]))

    return np.array(q_path), np.array(q_dist), labels


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .commensurate_utils import enumerate_commensurate_angles

    # Quick test: build honeycomb supercell at ~3°
    angles = enumerate_commensurate_angles('honeycomb', 2.5, 3.5, max_N=500)
    if angles:
        best = angles[0]
        print(f"Building supercell: θ={best['theta_deg']:.2f}°, (m,n)=({best['m']},{best['n']}), N={best['N_cells']}")
        eps, info = build_supercell_eps(
            'honeycomb', best['m'], best['n'],
            a=1.0, r_over_a=0.2, eps_rod=11.56, eps_bg=1.0,
            Nx=512, Ny=512,
        )
        print(f"  ε grid: {eps.shape}, rod fraction: {(eps > 1.5).mean():.3f}")
        print(f"  Supercell: {info['N_cells']} cells, L1={info['L1']}, L2={info['L2']}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(eps.T, origin='lower', cmap='RdBu_r', aspect='equal')
        ax.set_title(f"ε(x,y) — honeycomb θ={best['theta_deg']:.2f}°, N={info['N_cells']}")
        fig.savefig('thesis_results/T_direct_validation/test_supercell.png', dpi=150)
        print("  Saved test_supercell.png")
