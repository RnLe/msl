"""
Moire Lattice Python Package

A high-performance Python wrapper for lattice and moire lattice calculations,
built on top of a Rust core library.

Example usage:
    >>> import moire_lattice_py as ml
    >>> lattice = ml.create_square_lattice(1.0)
    >>> points = lattice.generate_points(5.0)
    >>> print(f"Generated {len(points)} lattice points")
"""

from .moire_lattice_py import (
    PyLattice2D,
    create_square_lattice,
    create_hexagonal_lattice, 
    create_rectangular_lattice,
    version,
    __version__,
    __author__
)

__all__ = [
    "PyLattice2D",
    "create_square_lattice",
    "create_hexagonal_lattice",
    "create_rectangular_lattice", 
    "version",
    "__version__",
    "__author__"
]
