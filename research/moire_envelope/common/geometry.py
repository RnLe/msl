"""
Geometry utilities for creating photonic crystal lattices
"""
import numpy as np
import math


def build_lattice(lattice_type: str, r_over_a: float, eps_bg: float, a: float = 1.0):
    """
    Build a 2D photonic crystal lattice geometry
    
    Args:
        lattice_type: One of 'square', 'hex', 'rect'
        r_over_a: Hole radius normalized by lattice constant
        eps_bg: Background dielectric constant
        a: Lattice constant (default 1.0)
        
    Returns:
        dict: Geometry parameters for MPB simulation
    """
    geometry = {
        'lattice_type': lattice_type,
        'lattice_constant': a,
        'r_over_a': r_over_a,
        'radius': r_over_a * a,
        'eps_bg': eps_bg,
        'eps_hole': 1.0,  # Air holes
    }
    
    # Define lattice vectors based on type
    if lattice_type == 'square':
        geometry['a1'] = np.array([a, 0.0, 0.0])
        geometry['a2'] = np.array([0.0, a, 0.0])
    elif lattice_type == 'hex':
        geometry['a1'] = np.array([a, 0.0, 0.0])
        geometry['a2'] = np.array([a * 0.5, a * math.sqrt(3) / 2, 0.0])
    elif lattice_type == 'rect':
        # Rectangular lattice with aspect ratio (can be parameterized later)
        geometry['a1'] = np.array([a, 0.0, 0.0])
        geometry['a2'] = np.array([0.0, a * 1.5, 0.0])  # Default aspect ratio
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    return geometry


def high_symmetry_points(lattice_type: str):
    """
    Get high symmetry points for a given lattice type
    
    Args:
        lattice_type: One of 'square', 'hex', 'rect'
        
    Returns:
        list: List of (label, k_vec) tuples
    """
    if lattice_type == 'square':
        # Square lattice: Γ, X, M
        points = [
            ('Γ', np.array([0.0, 0.0, 0.0])),
            ('X', np.array([0.5, 0.0, 0.0])),  # In units of 2π/a
            ('M', np.array([0.5, 0.5, 0.0])),
        ]
    elif lattice_type == 'hex':
        # Hexagonal lattice: Γ, M, K
        points = [
            ('Γ', np.array([0.0, 0.0, 0.0])),
            ('M', np.array([0.5, 0.0, 0.0])),
            ('K', np.array([1/3, 1/3, 0.0])),
        ]
    elif lattice_type == 'rect':
        # Rectangular lattice: Γ, X, Y, M
        points = [
            ('Γ', np.array([0.0, 0.0, 0.0])),
            ('X', np.array([0.5, 0.0, 0.0])),
            ('Y', np.array([0.0, 0.5, 0.0])),
            ('M', np.array([0.5, 0.5, 0.0])),
        ]
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    # Ensure the k-path closes back at Γ for plotting
    if points and points[-1][0] != points[0][0]:
        points.append((points[0][0], points[0][1].copy()))
    
    return points


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Create a 2D rotation matrix
    
    Args:
        theta: Rotation angle in radians
        
    Returns:
        2x2 rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])
