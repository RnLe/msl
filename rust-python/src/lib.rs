//! Python bindings for the moire-lattice library
//!
//! This module provides Python bindings for high-performance lattice calculations,
//! including 2D Bravais lattices, moire patterns, and crystallographic operations.

use pyo3::prelude::*;

mod base_matrix;
mod lattice_types;
mod lattice2d;
mod polyhedron;
mod high_symmetry_points;
mod moire2d;

use base_matrix::{PyBaseMatrixDirect, PyBaseMatrixReciprocal};
use lattice_types::{PyBravais2D, PyBravais3D, PyCentering};
use lattice2d::PyLattice2D;
use polyhedron::PyPolyhedron;
use high_symmetry_points::{PySymmetryPointLabel, PyHighSymmetryPoint, PyHighSymmetryPath, PyHighSymmetryData};
use moire2d::{PyMoire2D, PyMoireTransformation};

/// Python module for moire lattice calculations
#[pymodule]
fn moire_lattice_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Base matrix types
    m.add_class::<PyBaseMatrixDirect>()?;
    m.add_class::<PyBaseMatrixReciprocal>()?;
    
    // Lattice types
    m.add_class::<PyBravais2D>()?;
    m.add_class::<PyBravais3D>()?;
    m.add_class::<PyCentering>()?;
    
    // Main lattice class
    m.add_class::<PyLattice2D>()?;
    
    // Polyhedron for Brillouin zones and Wigner-Seitz cells
    m.add_class::<PyPolyhedron>()?;
    
    // High symmetry points and paths
    m.add_class::<PySymmetryPointLabel>()?;
    m.add_class::<PyHighSymmetryPoint>()?;
    m.add_class::<PyHighSymmetryPath>()?;
    m.add_class::<PyHighSymmetryData>()?;
    
    // Moire lattice types
    m.add_class::<PyMoire2D>()?;
    m.add_class::<PyMoireTransformation>()?;
    
    Ok(())
}
