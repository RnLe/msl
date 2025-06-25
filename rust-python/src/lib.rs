use pyo3::prelude::*;

// Module declarations
mod lattice;
mod symmetries;
mod moire_lattice;
mod utils;

// Import the existing types from our modules
use lattice::{PyLattice2D, PyPolyhedron, PyCoordinationAnalysis};
use lattice::{
    create_square_lattice, create_hexagonal_lattice, create_rectangular_lattice,
    oblique_lattice_create, centered_rectangular_lattice_create,
    py_coordination_number_2d, py_nearest_neighbors_2d, py_nearest_neighbor_distance_2d, py_packing_fraction_2d
};
use utils::version;

/// Python module definition
#[pymodule]
fn moire_lattice_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLattice2D>()?;
    m.add_class::<PyPolyhedron>()?;
    m.add_class::<PyCoordinationAnalysis>()?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(create_square_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_hexagonal_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_rectangular_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(oblique_lattice_create, m)?)?;
    m.add_function(wrap_pyfunction!(centered_rectangular_lattice_create, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    
    // Coordination analysis functions
    m.add_function(wrap_pyfunction!(py_coordination_number_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_nearest_neighbors_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_nearest_neighbor_distance_2d, m)?)?;
    m.add_function(wrap_pyfunction!(py_packing_fraction_2d, m)?)?;
    
    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Rene-Marcel Lehner")?;
    
    Ok(())
}
