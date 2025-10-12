use crate::lattice::lattice2d::PyLattice2D;
use moire_lattice::lattice::lattice_coordination_numbers::{
    coordination_number_2d as core_coordination_number_2d,
    nearest_neighbor_distance_2d as core_nearest_neighbor_distance_2d,
    nearest_neighbors_2d as core_nearest_neighbors_2d,
    packing_fraction_2d as core_packing_fraction_2d,
};
use pyo3::prelude::*;

/// Calculate coordination number for a 2D lattice
#[pyfunction]
pub fn py_coordination_number_2d(lattice: &PyLattice2D) -> usize {
    core_coordination_number_2d(&lattice.inner.bravais)
}

/// Find nearest neighbor positions for a 2D lattice
#[pyfunction]
pub fn py_nearest_neighbors_2d(lattice: &PyLattice2D) -> Vec<(f64, f64, f64)> {
    let neighbors = core_nearest_neighbors_2d(
        &lattice.inner.direct,
        &lattice.inner.bravais,
        lattice.inner.tol,
    );
    neighbors.into_iter().map(|v| (v.x, v.y, v.z)).collect()
}

/// Calculate nearest neighbor distance for a 2D lattice
#[pyfunction]
pub fn py_nearest_neighbor_distance_2d(lattice: &PyLattice2D) -> f64 {
    core_nearest_neighbor_distance_2d(&lattice.inner.direct, &lattice.inner.bravais)
}

/// Calculate 2D packing fraction
#[pyfunction]
pub fn py_packing_fraction_2d(lattice: &PyLattice2D) -> f64 {
    let (a, b) = lattice.inner.direct_lattice_parameters();
    core_packing_fraction_2d(&lattice.inner.bravais, (a, b))
}

/// Python wrapper for coordination analysis
#[pyclass]
pub struct PyCoordinationAnalysis;

#[pymethods]
impl PyCoordinationAnalysis {
    #[new]
    fn new() -> Self {
        PyCoordinationAnalysis
    }

    /// Get comprehensive coordination information for a lattice
    fn analyze_coordination(&self, lattice: &PyLattice2D) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            let coord_num = core_coordination_number_2d(&lattice.inner.bravais);
            let nn_distance =
                core_nearest_neighbor_distance_2d(&lattice.inner.direct, &lattice.inner.bravais);
            let neighbors = core_nearest_neighbors_2d(
                &lattice.inner.direct,
                &lattice.inner.bravais,
                lattice.inner.tol,
            );
            let neighbor_tuples: Vec<(f64, f64, f64)> =
                neighbors.into_iter().map(|v| (v.x, v.y, v.z)).collect();

            dict.set_item("coordination_number", coord_num)?;
            dict.set_item("nearest_neighbor_distance", nn_distance)?;
            dict.set_item("nearest_neighbors", neighbor_tuples)?;
            dict.set_item("lattice_type", format!("{:?}", lattice.inner.bravais))?;

            Ok(dict.into())
        })
    }

    /// Calculate packing fraction for given atomic radius
    fn calculate_packing_fraction(&self, lattice: &PyLattice2D, _atomic_radius: f64) -> f64 {
        let (a, b) = lattice.inner.direct_lattice_parameters();
        core_packing_fraction_2d(&lattice.inner.bravais, (a, b))
    }

    /// Find optimal atomic radius for maximum packing
    fn optimal_packing_radius(&self, lattice: &PyLattice2D) -> f64 {
        // For most lattices, optimal packing radius is half the nearest neighbor distance
        let nn_distance =
            core_nearest_neighbor_distance_2d(&lattice.inner.direct, &lattice.inner.bravais);
        nn_distance / 2.0
    }
}
