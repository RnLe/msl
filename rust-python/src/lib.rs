use pyo3::prelude::*;

// Module declarations
mod lattice;
mod moire_lattice;
mod symmetries;
mod utils;

// Import the existing types from our modules
use lattice::{Polyhedron, PyCoordinationAnalysis, PyLattice2D};
use lattice::{
    centered_rectangular_lattice_create, create_hexagonal_lattice, create_rectangular_lattice,
    create_square_lattice, oblique_lattice_create, py_coordination_number_2d,
    py_nearest_neighbor_distance_2d, py_nearest_neighbors_2d, py_packing_fraction_2d,
};
use moire_lattice::{
    PyMoire2D, PyMoireBuilder, PyMoireTransformation, py_commensurate_moire,
    py_local_cell_at_point_preliminary, py_local_cells_preliminary, py_registry_centers,
    py_twisted_bilayer,
};
use utils::version;

/// Python module definition
#[pymodule]
fn moire_lattice_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLattice2D>()?;
    m.add_class::<Polyhedron>()?;
    m.add_class::<PyCoordinationAnalysis>()?;

    // Moiré lattice classes
    m.add_class::<PyMoire2D>()?;
    m.add_class::<PyMoireTransformation>()?;
    m.add_class::<PyMoireBuilder>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(create_square_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_hexagonal_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_rectangular_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(oblique_lattice_create, m)?)?;
    m.add_function(wrap_pyfunction!(centered_rectangular_lattice_create, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Moiré lattice functions
    m.add_function(wrap_pyfunction!(py_twisted_bilayer, m)?)?;
    m.add_function(wrap_pyfunction!(py_commensurate_moire, m)?)?;
    m.add_function(wrap_pyfunction!(py_registry_centers, m)?)?;
    m.add_function(wrap_pyfunction!(py_local_cells_preliminary, m)?)?;
    m.add_function(wrap_pyfunction!(py_local_cell_at_point_preliminary, m)?)?;

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
