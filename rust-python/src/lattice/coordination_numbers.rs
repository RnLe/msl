use pyo3::prelude::*;
use moire_lattice::lattice::coordination_numbers;
use crate::lattice::lattice2d::PyLattice2D;

/// Calculate coordination number for a 2D lattice
#[pyfunction]
pub fn coordination_number_2d(lattice: &PyLattice2D) -> usize {
    coordination_numbers::coordination_number_2d(&lattice.inner.bravais_type())
}

/// Find nearest neighbor positions for a 2D lattice
#[pyfunction]
pub fn nearest_neighbors_2d(lattice: &PyLattice2D) -> Vec<(f64, f64, f64)> {
    let neighbors = coordination_numbers::nearest_neighbors_2d(
        lattice.inner.direct_basis(),
        &lattice.inner.bravais_type(),
        lattice.inner.tolerance()
    );
    neighbors.into_iter()
        .map(|v| (v.x, v.y, v.z))
        .collect()
}

/// Calculate nearest neighbor distance for a 2D lattice
#[pyfunction]
pub fn nearest_neighbor_distance_2d(lattice: &PyLattice2D) -> f64 {
    coordination_numbers::nearest_neighbor_distance_2d(
        lattice.inner.direct_basis(),
        &lattice.inner.bravais_type()
    )
}

/// Calculate 2D packing fraction
#[pyfunction]
pub fn packing_fraction_2d(lattice: &PyLattice2D, _radius: f64) -> f64 {
    let (a, b) = lattice.inner.lattice_parameters();
    coordination_numbers::packing_fraction_2d(&lattice.inner.bravais_type(), (a, b))
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
            
            let coord_num = coordination_numbers::coordination_number_2d(&lattice.inner.bravais_type());
            let nn_distance = coordination_numbers::nearest_neighbor_distance_2d(
                lattice.inner.direct_basis(),
                &lattice.inner.bravais_type()
            );
            let neighbors = coordination_numbers::nearest_neighbors_2d(
                lattice.inner.direct_basis(),
                &lattice.inner.bravais_type(),
                lattice.inner.tolerance()
            );
            let neighbor_tuples: Vec<(f64, f64, f64)> = neighbors
                .into_iter()
                .map(|v| (v.x, v.y, v.z))
                .collect();

            dict.set_item("coordination_number", coord_num)?;
            dict.set_item("nearest_neighbor_distance", nn_distance)?;
            dict.set_item("nearest_neighbors", neighbor_tuples)?;
            dict.set_item("lattice_type", format!("{:?}", lattice.inner.bravais_type()))?;
            
            Ok(dict.into())
        })
    }

    /// Calculate packing fraction for given atomic radius
    fn calculate_packing_fraction(&self, lattice: &PyLattice2D, _atomic_radius: f64) -> f64 {
        let (a, b) = lattice.inner.lattice_parameters();
        coordination_numbers::packing_fraction_2d(&lattice.inner.bravais_type(), (a, b))
    }

    /// Find optimal atomic radius for maximum packing
    fn optimal_packing_radius(&self, lattice: &PyLattice2D) -> f64 {
        // For most lattices, optimal packing radius is half the nearest neighbor distance
        let nn_distance = coordination_numbers::nearest_neighbor_distance_2d(
            lattice.inner.direct_basis(),
            &lattice.inner.bravais_type()
        );
        nn_distance / 2.0
    }
}
