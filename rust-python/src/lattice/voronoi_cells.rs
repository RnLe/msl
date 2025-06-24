use pyo3::prelude::*;
use moire_lattice::lattice::voronoi_cells;
use crate::lattice::{lattice2d::PyLattice2D, polyhedron::PyPolyhedron};
use nalgebra::Vector3;

/// Generate 2D lattice points by shell number
#[pyfunction]
pub fn generate_lattice_points_2d_by_shell(
    lattice: &PyLattice2D,
    max_shell: usize
) -> Vec<(f64, f64, f64)> {
    let points = voronoi_cells::generate_lattice_points_2d_by_shell(lattice.inner.direct_basis(), max_shell);
    points.into_iter()
        .map(|p| (p.x, p.y, p.z))
        .collect()
}

/// Generate 2D lattice points within a radius
#[pyfunction]
pub fn generate_lattice_points_2d_within_radius(
    lattice: &PyLattice2D,
    radius: f64
) -> Vec<(f64, f64, f64)> {
    let points = voronoi_cells::generate_lattice_points_2d_within_radius(lattice.inner.direct_basis(), radius);
    points.into_iter()
        .map(|p| (p.x, p.y, p.z))
        .collect()
}

/// Compute Wigner-Seitz cell for a 2D lattice
#[pyfunction]
pub fn compute_wigner_seitz_cell_2d(
    lattice: &PyLattice2D,
    tolerance: Option<f64>
) -> PyPolyhedron {
    let tol = tolerance.unwrap_or(lattice.inner.tolerance());
    let polyhedron = voronoi_cells::compute_wigner_seitz_cell_2d(lattice.inner.direct_basis(), tol);
    PyPolyhedron::new(polyhedron)
}

/// Compute first Brillouin zone for a 2D lattice
#[pyfunction]
pub fn compute_brillouin_zone_2d(
    lattice: &PyLattice2D,
    tolerance: Option<f64>
) -> PyPolyhedron {
    let tol = tolerance.unwrap_or(lattice.inner.tolerance());
    let polyhedron = voronoi_cells::compute_brillouin_zone_2d(lattice.inner.reciprocal_basis(), tol);
    PyPolyhedron::new(polyhedron)
}

/// Get the Wigner-Seitz cell from a lattice (cached)
#[pyfunction]
pub fn get_wigner_seitz_cell(lattice: &PyLattice2D) -> PyPolyhedron {
    PyPolyhedron::new(lattice.inner.wigner_seitz_cell().clone())
}

/// Get the first Brillouin zone from a lattice (cached)
#[pyfunction]
pub fn get_brillouin_zone(lattice: &PyLattice2D) -> PyPolyhedron {
    PyPolyhedron::new(lattice.inner.brillouin_zone().clone())
}

/// Python wrapper for Voronoi analysis
#[pyclass]
pub struct PyVoronoiAnalysis;

#[pymethods]
impl PyVoronoiAnalysis {
    #[new]
    fn new() -> Self {
        PyVoronoiAnalysis
    }

    /// Generate neighbor analysis for a lattice
    fn analyze_neighbors(&self, lattice: &PyLattice2D, max_shell: usize) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            // Generate points by shell
            let mut shell_data = Vec::new();
            for shell in 0..=max_shell {
                let points = voronoi_cells::generate_lattice_points_2d_by_shell(lattice.inner.direct_basis(), shell);
                let distances: Vec<f64> = points.iter().map(|p| p.norm()).collect();
                let avg_distance = if !distances.is_empty() {
                    distances.iter().sum::<f64>() / distances.len() as f64
                } else {
                    0.0
                };
                
                let shell_dict = PyDict::new(py);
                shell_dict.set_item("shell", shell)?;
                shell_dict.set_item("num_points", points.len())?;
                shell_dict.set_item("average_distance", avg_distance)?;
                shell_dict.set_item("points", points.into_iter().map(|p| (p.x, p.y, p.z)).collect::<Vec<_>>())?;
                
                shell_data.push(shell_dict);
            }
            
            dict.set_item("shells", shell_data)?;
            dict.set_item("lattice_type", format!("{:?}", lattice.inner.bravais_type()))?;
            dict.set_item("unit_cell_area", lattice.inner.cell_area())?;
            
            Ok(dict.into())
        })
    }

    /// Analyze the Wigner-Seitz cell properties
    fn analyze_wigner_seitz(&self, lattice: &PyLattice2D) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let ws_cell = lattice.inner.wigner_seitz_cell();
            
            dict.set_item("area", ws_cell.measure())?;
            dict.set_item("vertices", ws_cell.vertices().iter().map(|v| (v.x, v.y, v.z)).collect::<Vec<_>>())?;
            dict.set_item("num_vertices", ws_cell.vertices().len())?;
            dict.set_item("num_edges", ws_cell.edges().len())?;
            dict.set_item("unit_cell_area", lattice.inner.cell_area())?;
            dict.set_item("area_ratio", ws_cell.measure() / lattice.inner.cell_area())?;
            
            Ok(dict.into())
        })
    }

    /// Analyze the Brillouin zone properties
    fn analyze_brillouin_zone(&self, lattice: &PyLattice2D) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let bz = lattice.inner.brillouin_zone();
            
            dict.set_item("area", bz.measure())?;
            dict.set_item("vertices", bz.vertices().iter().map(|v| (v.x, v.y, v.z)).collect::<Vec<_>>())?;
            dict.set_item("num_vertices", bz.vertices().len())?;
            dict.set_item("num_edges", bz.edges().len())?;
            
            // Calculate reciprocal unit cell area
            let reciprocal_area = lattice.inner.reciprocal_basis().determinant().abs();
            dict.set_item("reciprocal_unit_cell_area", reciprocal_area)?;
            dict.set_item("area_ratio", bz.measure() / reciprocal_area)?;
            
            Ok(dict.into())
        })
    }

    /// Find lattice points within a circular region
    fn points_in_circle(&self, lattice: &PyLattice2D, radius: f64, center: Option<(f64, f64)>) -> Vec<(f64, f64, f64)> {
        let center = center.unwrap_or((0.0, 0.0));
        let points = voronoi_cells::generate_lattice_points_2d_within_radius(lattice.inner.direct_basis(), radius * 1.2);
        let center_vec = Vector3::new(center.0, center.1, 0.0);
        
        points.into_iter()
            .filter(|p| (p - center_vec).norm() <= radius)
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }

    /// Find lattice points within a rectangular region
    fn points_in_rectangle(&self, lattice: &PyLattice2D, width: f64, height: f64) -> Vec<(f64, f64, f64)> {
        lattice.inner.get_direct_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }
}
