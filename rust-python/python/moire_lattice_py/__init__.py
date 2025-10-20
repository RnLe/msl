"""
Python bindings for the moire-lattice library.

High-performance lattice calculations for photonic band structures,
moire patterns, and crystallographic operations.

This package provides 1:1 bindings to the Rust moire-lattice library,
including:
- 2D Bravais lattice generation and analysis (Lattice2D)
- Moire pattern calculations (Moire2D)
- Brillouin zone and Wigner-Seitz cell computations (Polyhedron)
- High symmetry points and k-paths (HighSymmetryData)
- Base matrix operations (BaseMatrixDirect, BaseMatrixReciprocal)

Example usage:
    >>> import moire_lattice_py as ml
    >>> # Create a square lattice
    >>> lattice = ml.Lattice2D.from_basis_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    >>> print(lattice.direct_bravais().name())
    'Square'
    
    >>> # Get Brillouin zone
    >>> bz = lattice.brillouin_zone()
    >>> print(f"BZ has {bz.num_vertices()} vertices")
    
    >>> # Generate k-path for band structure
    >>> k_path = lattice.generate_high_symmetry_k_path(50)
"""

from .moire_lattice_py import (
    # Base matrix types
    BaseMatrixDirect,
    BaseMatrixReciprocal,
    
    # Lattice types
    Bravais2D,
    Bravais3D,
    Centering,
    
    # Main lattice class
    Lattice2D,
    
    # Polyhedron for Brillouin zones and Wigner-Seitz cells
    Polyhedron,
    
    # High symmetry points and paths
    SymmetryPointLabel,
    HighSymmetryPoint,
    HighSymmetryPath,
    HighSymmetryData,
    
    # Moire lattice types
    Moire2D,
    MoireTransformation,
)

__all__ = [
    # Base matrix types
    "BaseMatrixDirect",
    "BaseMatrixReciprocal",
    
    # Lattice types
    "Bravais2D",
    "Bravais3D",
    "Centering",
    
    # Main lattice class
    "Lattice2D",
    
    # Polyhedron
    "Polyhedron",
    
    # High symmetry points
    "SymmetryPointLabel",
    "HighSymmetryPoint",
    "HighSymmetryPath",
    "HighSymmetryData",
    
    # Moire lattice
    "Moire2D",
    "MoireTransformation",
]

__version__ = "0.1.1"

