"""
Commensurate angle enumeration for moiré photonic crystals.

For a twisted bilayer with underlying Bravais lattice, commensurate twist angles
are those where the moiré superlattice is exactly periodic — i.e., a finite
supercell tiles the plane. This requires the twist angle to satisfy an algebraic
condition involving integer coincidence indices (m, n).

Triangular lattice (hex/honeycomb):
    Commensurate twist angles (Lopes dos Santos et al.):
        cos(θ) = (m² + 4mn + n²) / (2(m² + mn + n²))
    for coprime (m, n) with m >= n > 0 (or m > n >= 0 for small angles).
    The supercell has N = m² + mn + n² monolayer unit cells.
    Small angles require m ≈ n with both large (e.g. m=n+1).

Square lattice:
    Commensurate twist angles:
        cos(θ) = (m² - n²) / (m² + n²)    [i.e. θ = 2·arctan(n/m)]
    for coprime (m, n) with m > n > 0.
    The supercell has N = m² + n² monolayer unit cells.
"""

import numpy as np
from typing import List, Dict, Optional
from math import gcd


def _commensurate_angles_triangular(
    theta_min_deg: float,
    theta_max_deg: float,
    max_N: int,
) -> List[Dict]:
    """
    Enumerate commensurate twist angles for triangular lattice.

    Uses the Lopes dos Santos formula:
        cos(θ) = (m² + 4mn + n²) / (2(m² + mn + n²))

    The moiré supercell contains N = m² + m·n + n² monolayer unit cells.
    Small twist angles are obtained when m ≈ n (both large).
    """
    theta_min = np.radians(theta_min_deg)
    theta_max = np.radians(theta_max_deg)

    results = []
    # For small angles with m ≈ n, N ≈ 3m², so m_max ≈ sqrt(N_max/3)
    m_max = int(np.sqrt(max_N)) + 2

    for m in range(1, m_max + 1):
        for n in range(0, m + 1):
            if m == 0 and n == 0:
                continue
            if gcd(m, n) != 1:
                continue

            N = m * m + m * n + n * n
            if N > max_N:
                continue

            # Lopes dos Santos formula
            cos_theta = (m * m + 4 * m * n + n * n) / (2.0 * N)
            if abs(cos_theta) > 1.0:
                continue
            theta = np.arccos(cos_theta)

            if theta_min <= theta <= theta_max:
                results.append({
                    'theta_rad': float(theta),
                    'theta_deg': float(np.degrees(theta)),
                    'm': m,
                    'n': n,
                    'N_cells': N,
                    'lattice_family': 'triangular',
                })

    # Deduplicate by angle (some (m,n) pairs give the same angle)
    seen = set()
    unique = []
    for r in results:
        key = round(r['theta_deg'], 8)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Sort by N_cells (smallest supercell first), then by angle
    unique.sort(key=lambda r: (r['N_cells'], r['theta_deg']))
    return unique


def _commensurate_angles_square(
    theta_min_deg: float,
    theta_max_deg: float,
    max_N: int,
) -> List[Dict]:
    """
    Enumerate commensurate twist angles for square lattice.

    The twist angle between layers for indices (m, n) is:
        cos(θ) = (m² - n²) / (m² + n²)     [equivalently θ = 2·arctan(n/m)]

    The moiré supercell contains N = m² + n² monolayer unit cells.
    Small twist angles require m >> n.
    """
    theta_min = np.radians(theta_min_deg)
    theta_max = np.radians(theta_max_deg)

    results = []
    m_max = int(np.sqrt(max_N)) + 2

    for m in range(1, m_max + 1):
        for n in range(1, m):
            if gcd(m, n) != 1:
                continue

            N = m * m + n * n
            if N > max_N:
                continue

            # Twist angle: θ = 2·arctan(n/m)
            theta = 2.0 * np.arctan2(n, m)

            if theta_min <= theta <= theta_max:
                results.append({
                    'theta_rad': float(theta),
                    'theta_deg': float(np.degrees(theta)),
                    'm': m,
                    'n': n,
                    'N_cells': N,
                    'lattice_family': 'square',
                })

    seen = set()
    unique = []
    for r in results:
        key = round(r['theta_deg'], 8)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: (r['N_cells'], r['theta_deg']))
    return unique


def enumerate_commensurate_angles(
    lattice_type: str,
    theta_min_deg: float = 0.5,
    theta_max_deg: float = 10.0,
    max_N: int = 2000,
) -> List[Dict]:
    """
    Enumerate commensurate twist angles for a given lattice type.

    Args:
        lattice_type: 'square', 'hex', 'triangular', or 'honeycomb'
        theta_min_deg: minimum twist angle in degrees
        theta_max_deg: maximum twist angle in degrees
        max_N: maximum number of monolayer unit cells in the supercell

    Returns:
        List of dicts with keys: theta_deg, theta_rad, m, n, N_cells, lattice_family
        Sorted by N_cells (smallest first).
    """
    if lattice_type == 'square':
        return _commensurate_angles_square(theta_min_deg, theta_max_deg, max_N)
    elif lattice_type in ('hex', 'hexagonal', 'triangular', 'honeycomb'):
        return _commensurate_angles_triangular(theta_min_deg, theta_max_deg, max_N)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def estimate_fdfd_resources(N_cells: int, res_per_cell: int = 32, n_atoms: int = 1) -> Dict:
    """
    Estimate computational resources for an FDFD supercell solve.

    Args:
        N_cells: number of monolayer unit cells in the supercell
        res_per_cell: grid points per unit cell side (e.g. 32)
        n_atoms: atoms per unit cell (1 for hex/square, 2 for honeycomb)

    Returns:
        Dict with DOF count, estimated memory, feasibility flag
    """
    # The supercell is roughly sqrt(N) × sqrt(N) unit cells
    # Total grid: sqrt(N)*res × sqrt(N)*res = N * res²
    total_grid_points = N_cells * res_per_cell * res_per_cell
    dof = total_grid_points

    # Sparse matrix: ~5 nonzeros per row (2D 5-point stencil)
    nnz = 5 * dof
    # Memory: sparse CSR = 3 arrays: data(nnz*16B), indices(nnz*4B), indptr(dof*4B)
    sparse_mem_bytes = nnz * 16 + nnz * 4 + dof * 4

    # Shift-invert needs LU factorization. For banded sparse, fill-in is
    # roughly O(bandwidth * dof). Bandwidth ≈ sqrt(N)*res.
    bandwidth = int(np.sqrt(N_cells) * res_per_cell)
    # LU fill-in estimate: ~10-20× nnz for 2D problems
    lu_mem_bytes = 20 * nnz * 16

    # Eigenvector storage: n_modes * dof * 16B (complex128)
    n_modes = 50
    evec_mem_bytes = n_modes * dof * 16

    total_mem_bytes = sparse_mem_bytes + lu_mem_bytes + evec_mem_bytes
    total_mem_gb = total_mem_bytes / (1024**3)

    return {
        'N_cells': N_cells,
        'res_per_cell': res_per_cell,
        'total_grid_points': total_grid_points,
        'dof': dof,
        'nnz': nnz,
        'estimated_mem_gb': round(total_mem_gb, 2),
        'feasible_32gb': total_mem_gb < 24,  # leave headroom
        'feasible_64gb': total_mem_gb < 48,
    }


def select_best_angle(
    lattice_type: str,
    theta_min_deg: float = 2.0,
    theta_max_deg: float = 5.0,
    max_mem_gb: float = 24.0,
    res_per_cell: int = 32,
    prefer_largest_angle: bool = True,
) -> Optional[Dict]:
    """
    Select the best commensurate angle for validation: largest angle (smallest
    supercell) that fits in memory, within the validity window.

    Args:
        lattice_type: 'square', 'hex', 'triangular', or 'honeycomb'
        theta_min_deg: minimum angle (lower bound of validity window)
        theta_max_deg: maximum angle (upper bound for search)
        max_mem_gb: maximum memory budget in GB
        res_per_cell: FDFD grid points per unit cell side
        prefer_largest_angle: if True, pick largest feasible angle (smallest N)

    Returns:
        Best candidate dict or None if no feasible angle found
    """
    n_atoms = 2 if lattice_type == 'honeycomb' else 1

    # Search with generous max_N, then filter by memory
    candidates = enumerate_commensurate_angles(
        lattice_type, theta_min_deg, theta_max_deg, max_N=10000
    )

    feasible = []
    for c in candidates:
        resources = estimate_fdfd_resources(c['N_cells'], res_per_cell, n_atoms)
        c['resources'] = resources
        if resources['estimated_mem_gb'] < max_mem_gb:
            feasible.append(c)

    if not feasible:
        return None

    if prefer_largest_angle:
        # Largest angle = smallest supercell
        feasible.sort(key=lambda c: c['N_cells'])
        return feasible[0]
    else:
        # Smallest angle within range
        feasible.sort(key=lambda c: c['theta_deg'])
        return feasible[0]


def build_supercell_vectors(lattice_type: str, m: int, n: int, a: float = 1.0) -> np.ndarray:
    """
    Build the supercell lattice vectors for a commensurate moiré structure.

    For a triangular lattice with basis vectors a1, a2, the coincidence
    lattice vectors (common superlattice of both rotated layers) are:
        L1 = n·a1 + m·a2
        L2 = -m·a1 + (n+m)·a2
    These satisfy R(-θ)·L1 = m·a1 + n·a2 ∈ ℤ{a1,a2}, ensuring L1 is
    a lattice vector of both unrotated and rotated layers.

    For a square lattice:
        L1 = m·a1 + n·a2
        L2 = -n·a1 + m·a2  (90° rotation of L1)

    Args:
        lattice_type: 'square', 'hex', 'triangular', or 'honeycomb'
        m, n: coincidence indices
        a: lattice constant

    Returns:
        (2, 2) array: columns are supercell lattice vectors L1, L2
    """
    if lattice_type == 'square':
        a1 = a * np.array([1.0, 0.0])
        a2 = a * np.array([0.0, 1.0])
        L1 = m * a1 + n * a2
        L2 = -n * a1 + m * a2
    elif lattice_type in ('hex', 'hexagonal', 'triangular', 'honeycomb'):
        a1 = a * np.array([1.0, 0.0])
        a2 = a * np.array([0.5, np.sqrt(3) / 2])
        # Coincidence lattice vectors (valid superlattice of BOTH layers):
        # L1 = n·a1 + m·a2, L2 = -m·a1 + (n+m)·a2
        # This ensures R(-θ)·L1 = m·a1 + n·a2 ∈ ℤ{a1,a2}, so L1 is
        # also a lattice vector of the rotated layer.
        L1 = n * a1 + m * a2
        L2 = -m * a1 + (n + m) * a2
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    # Return as column matrix: B_super = [L1 | L2]
    return np.column_stack([L1, L2])


def commensurate_twist_angle(lattice_type: str, m: int, n: int) -> float:
    """
    Compute the exact commensurate twist angle for given indices.

    For triangular: cos(θ) = (m² + 4mn + n²) / (2(m² + mn + n²))
    For square: θ = 2·arctan(n/m)

    Returns:
        Twist angle in radians
    """
    if lattice_type == 'square':
        return 2.0 * np.arctan2(n, m)
    elif lattice_type in ('hex', 'hexagonal', 'triangular', 'honeycomb'):
        N = m * m + m * n + n * n
        cos_theta = (m * m + 4 * m * n + n * n) / (2.0 * N)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


if __name__ == '__main__':
    import json

    print("=" * 70)
    print("COMMENSURATE ANGLE ENUMERATION")
    print("=" * 70)

    for lt in ['honeycomb', 'hex', 'square']:
        print(f"\n--- {lt.upper()} lattice ---")
        angles = enumerate_commensurate_angles(lt, 1.0, 10.0, max_N=5000)
        for a in angles[:15]:
            res = estimate_fdfd_resources(a['N_cells'], 32, 2 if lt == 'honeycomb' else 1)
            feasible = "✓" if res['feasible_32gb'] else "✗"
            print(f"  θ={a['theta_deg']:6.2f}°  (m,n)=({a['m']},{a['n']})  "
                  f"N={a['N_cells']:5d}  DOF={res['dof']:>10,}  "
                  f"Mem≈{res['estimated_mem_gb']:6.1f} GB  {feasible}")

    print("\n--- Best angles for validation (within validity window) ---")
    for lt in ['honeycomb', 'hex', 'square']:
        best = select_best_angle(lt, theta_min_deg=2.0, theta_max_deg=5.0)
        if best:
            print(f"  {lt:12s}: θ={best['theta_deg']:.2f}° (m,n)=({best['m']},{best['n']}) "
                  f"N={best['N_cells']} Mem≈{best['resources']['estimated_mem_gb']:.1f} GB")
        else:
            print(f"  {lt:12s}: No feasible angle found in [2°, 5°] with 32 GB")
