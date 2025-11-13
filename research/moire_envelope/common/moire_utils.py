"""
Moiré lattice utilities using Rust Python bindings
"""
import numpy as np
import math
import moire_lattice_py as ml


def create_twisted_bilayer(lattice_type: str, theta_deg: float, a: float = 1.0):
    """
    Create a twisted bilayer moiré lattice using Rust bindings
    
    Args:
        lattice_type: One of 'square', 'hex', 'rect'
        theta_deg: Twist angle in degrees
        a: Lattice constant (default 1.0)
        
    Returns:
        dict: Contains 'lattice' (PyLattice2D), 'a1', 'a2', 'theta_rad', 'moire_length'
    """
    theta_rad = math.radians(theta_deg)
    
    # Create base lattice using Rust bindings
    if lattice_type == 'square':
        base_lattice = ml.Lattice2D.from_basis_vectors(
            [a, 0.0, 0.0],
            [0.0, a, 0.0]
        )
    elif lattice_type == 'hex':
        base_lattice = ml.Lattice2D.from_basis_vectors(
            [a, 0.0, 0.0],
            [a * 0.5, a * math.sqrt(3) / 2, 0.0]
        )
    elif lattice_type == 'rect':
        base_lattice = ml.Lattice2D.from_basis_vectors(
            [a, 0.0, 0.0],
            [0.0, a * 1.5, 0.0]
        )
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    # Extract basis vectors from the direct basis matrix
    a1, a2, a3 = base_lattice.direct_basis().base_vectors()
    # We only need the first two for 2D
    a1 = a1
    a2 = a2
    
    # Compute moiré length scale
    # For small angles: L_m ≈ a / (2 * sin(θ/2)) ≈ a / θ
    if lattice_type == 'hex':
        # Hexagonal: L_m = a / (2 * sin(θ/2))
        moire_length = a / (2 * math.sin(theta_rad / 2))
    else:
        # Square/Rectangular: similar formula
        moire_length = a / (2 * math.sin(theta_rad / 2))
    
    # Compute moiré reciprocal vector magnitude
    # |G_m| ≈ 4π/(√3 a) * sin(θ/2) for hex
    # |G_m| ≈ 2π/a * θ for small angles (general)
    if lattice_type == 'hex':
        G_magnitude = 4 * math.pi / (math.sqrt(3) * a) * math.sin(theta_rad / 2)
    else:
        G_magnitude = 2 * math.pi / a * theta_rad
    
    return {
        'lattice': base_lattice,
        'a1': np.array(a1),
        'a2': np.array(a2),
        'theta_rad': theta_rad,
        'theta_deg': theta_deg,
        'a': a,
        'moire_length': moire_length,
        'G_magnitude': G_magnitude,
    }


def compute_moire_parameters(a: float, theta_deg: float, lattice_type: str = 'hex'):
    """
    Compute moiré lattice parameters
    
    Args:
        a: Monolayer lattice constant
        theta_deg: Twist angle in degrees
        lattice_type: Lattice type ('square', 'hex', 'rect')
        
    Returns:
        dict: Moiré parameters including length scale and reciprocal magnitude
    """
    theta_rad = math.radians(theta_deg)
    
    if lattice_type == 'hex':
        # Hexagonal moiré
        moire_length = a / (2 * math.sin(theta_rad / 2))
        G_magnitude = 4 * math.pi / (math.sqrt(3) * a) * math.sin(theta_rad / 2)
    else:
        # Square/Rectangular moiré (small angle approximation)
        moire_length = a / (2 * math.sin(theta_rad / 2))
        G_magnitude = 2 * math.pi / a * theta_rad
    
    return {
        'moire_length': moire_length,
        'G_magnitude': G_magnitude,
        'theta_rad': theta_rad,
        'a': a,
    }


def build_R_grid(Nx: int, Ny: int, moire_length: float, center: bool = True):
    """
    Build spatial grid for moiré unit cell
    
    Args:
        Nx: Number of points in x direction
        Ny: Number of points in y direction
        moire_length: Moiré length scale
        center: Whether to center the grid at origin (default True)
        
    Returns:
        np.ndarray: Shape (Nx, Ny, 2) array of R coordinates
    """
    if center:
        x = np.linspace(-moire_length/2, moire_length/2, Nx)
        y = np.linspace(-moire_length/2, moire_length/2, Ny)
    else:
        x = np.linspace(0, moire_length, Nx)
        y = np.linspace(0, moire_length, Ny)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    R_grid = np.stack([X, Y], axis=-1)
    
    return R_grid


def compute_registry_map(R_grid, a1, a2, theta, tau=None, eta=1.0):
    """
    Compute the local registry map δ(R) for a moiré lattice
    
    Args:
        R_grid: Array [Nx, Ny, 2] of spatial coordinates
        a1: First lattice vector [2,] or [3,]
        a2: Second lattice vector [2,] or [3,]
        theta: Twist angle in radians
        tau: Stacking gauge vector [2,] (default: [0, 0])
        eta: Twist center gauge (default: 1.0)
        
    Returns:
        np.ndarray: Shape (Nx, Ny, 2) of fractional shifts δ
    """
    if tau is None:
        tau = np.zeros(2)
    
    Nx, Ny, _ = R_grid.shape
    delta_grid = np.zeros((Nx, Ny, 2))
    
    # Build rotation matrix
    c = np.cos(theta)
    s = np.sin(theta)
    R_rot = np.array([[c, -s], [s, c]])
    I = np.eye(2)
    
    # Build lattice matrix from basis vectors
    lattice_mat = np.column_stack([a1[:2], a2[:2]])
    lattice_mat_inv = np.linalg.inv(lattice_mat)
    
    for i in range(Nx):
        for j in range(Ny):
            R_vec = R_grid[i, j, :]
            # Physical shift: (R(θ) - I) @ R / η + τ
            delta_physical = (R_rot - I) @ R_vec / eta + tau
            # Convert to fractional coordinates
            delta_frac = lattice_mat_inv @ delta_physical
            delta_grid[i, j, :] = delta_frac
    
    return delta_grid


def fractional_coordinates(r_physical, lattice_mat):
    """
    Convert physical coordinates to fractional lattice coordinates
    
    Args:
        r_physical: Physical position vector [2,]
        lattice_mat: Lattice matrix [2, 2] with columns as basis vectors
        
    Returns:
        np.ndarray: Fractional coordinates [2,]
    """
    lattice_mat_inv = np.linalg.inv(lattice_mat)
    return lattice_mat_inv @ r_physical
