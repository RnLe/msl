"""
Moire Lattice Python Package

A high-performance Python wrapper for lattice and moire lattice calculations,
built on top of a Rust core library.

This package provides:
- 2D lattice generation and manipulation (PyLattice2D)
- Coordination analysis tools (PyCoordinationAnalysis)
- Polyhedron calculations (PyPolyhedron)  
- Various lattice construction functions
- Coordination number and packing analysis

Example usage:
    >>> import moire_lattice_py as ml
    >>> lattice = ml.create_square_lattice(1.0)
    >>> points = lattice.generate_points(5.0)
    >>> print(f"Generated {len(points)} lattice points")
    
    >>> # Analyze coordination
    >>> analysis = ml.PyCoordinationAnalysis()
    >>> coord_num = ml.py_coordination_number_2d(lattice, 1.5)
"""

from .moire_lattice_py import (
    # Classes
    PyLattice2D,
    PyCoordinationAnalysis,
    PyPolyhedron,
    
    # Lattice construction functions
    create_square_lattice,
    create_hexagonal_lattice, 
    create_rectangular_lattice,
    centered_rectangular_lattice_create,
    oblique_lattice_create,
    
    # Coordination analysis functions
    py_coordination_number_2d,
    py_nearest_neighbor_distance_2d,
    py_nearest_neighbors_2d,
    py_packing_fraction_2d,
    
    # Version info
    version,
)

__all__ = [
    # Classes
    "PyLattice2D",
    "PyCoordinationAnalysis", 
    "PyPolyhedron",
    
    # Lattice construction functions
    "create_square_lattice",
    "create_hexagonal_lattice",
    "create_rectangular_lattice",
    "centered_rectangular_lattice_create",
    "oblique_lattice_create",
    
    # Coordination analysis functions
    "py_coordination_number_2d",
    "py_nearest_neighbor_distance_2d", 
    "py_nearest_neighbors_2d",
    "py_packing_fraction_2d",
    
    # Version info
    "version",
]
