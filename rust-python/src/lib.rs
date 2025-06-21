use pyo3::prelude::*;
use pyo3::types::PyDict;
use moire_lattice::lattice::{
    Lattice2D, 
    construction::{square_lattice, hexagonal_lattice, rectangular_lattice, oblique_lattice},
    voronoi_cells::generate_neighbor_points_2d
};
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Python wrapper for the 2D lattice structure
#[pyclass]
pub struct PyLattice2D {
    inner: Lattice2D,
}

#[pymethods]
impl PyLattice2D {
    #[new]
    #[pyo3(signature = (lattice_type, a, b=None, angle=None))]
    fn new(
        lattice_type: &str,
        a: f64,
        b: Option<f64>,
        angle: Option<f64>,
    ) -> PyResult<Self> {
        let lattice = match lattice_type.to_lowercase().as_str() {
            "square" => square_lattice(a),
            "rectangular" => {
                let b_val = b.unwrap_or(a);
                rectangular_lattice(a, b_val)
            },
            "hexagonal" | "triangular" => hexagonal_lattice(a),
            "oblique" => {
                let b_val = b.unwrap_or(a);
                let angle_val = angle.unwrap_or(90.0) * PI / 180.0; // Convert to radians
                oblique_lattice(a, b_val, angle_val)
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown lattice type: {}", lattice_type)
            )),
        };

        Ok(PyLattice2D { inner: lattice })
    }

    /// Get lattice parameters
    fn get_parameters(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let (a, b) = self.inner.lattice_parameters();
            let angle = self.inner.lattice_angle() * 180.0 / PI; // Convert to degrees
            dict.set_item("a", a)?;
            dict.set_item("b", b)?;
            dict.set_item("angle", angle)?;
            dict.set_item("lattice_type", format!("{:?}", self.inner.bravais))?;
            Ok(dict.into())
        })
    }

    /// Generate lattice points within a given radius
    #[pyo3(signature = (radius, center=(0.0, 0.0)))]
    fn generate_points(&self, radius: f64, center: (f64, f64)) -> PyResult<Vec<(f64, f64)>> {
        // Generate neighbor points and filter by radius
        let points = generate_neighbor_points_2d(self.inner.direct_basis(), radius * 1.5); // Add some margin
        
        let center_vec = Vector3::new(center.0, center.1, 0.0);
        let filtered_points: Vec<(f64, f64)> = points
            .into_iter()
            .filter(|p| {
                let dist = (p - center_vec).norm();
                dist <= radius
            })
            .map(|p| (p.x, p.y))
            .collect();
        
        Ok(filtered_points)
    }

    /// Get the area of the unit cell
    fn unit_cell_area(&self) -> f64 {
        self.inner.cell_area
    }

    /// Get lattice vectors as tuples
    fn lattice_vectors(&self) -> ((f64, f64), (f64, f64)) {
        let (a_vec, b_vec) = self.inner.primitive_vectors();
        ((a_vec.x, a_vec.y), (b_vec.x, b_vec.y))
    }

    /// Get reciprocal lattice vectors
    fn reciprocal_vectors(&self) -> ((f64, f64), (f64, f64)) {
        let g1 = self.inner.reciprocal_basis().column(0);
        let g2 = self.inner.reciprocal_basis().column(1);
        ((g1.x, g1.y), (g2.x, g2.y))
    }

    /// String representation
    fn __repr__(&self) -> String {
        let (a, b) = self.inner.lattice_parameters();
        let angle = self.inner.lattice_angle() * 180.0 / PI;
        format!("PyLattice2D({:?}, a={:.3}, b={:.3}, angle={:.1}°)",
            self.inner.bravais,
            a, b, angle
        )
    }
}

/// Utility functions for lattice operations
#[pyfunction]
fn create_square_lattice(a: f64) -> PyLattice2D {
    PyLattice2D {
        inner: square_lattice(a),
    }
}

#[pyfunction]
fn create_hexagonal_lattice(a: f64) -> PyLattice2D {
    PyLattice2D {
        inner: hexagonal_lattice(a),
    }
}

#[pyfunction]
fn create_rectangular_lattice(a: f64, b: f64) -> PyLattice2D {
    PyLattice2D {
        inner: rectangular_lattice(a, b),
    }
}

/// Get the version of the moire-lattice library
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Python module definition
#[pymodule]
fn moire_lattice_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLattice2D>()?;
    
    // Utility functions
    m.add_function(wrap_pyfunction!(create_square_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_hexagonal_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_rectangular_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    
    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Rene-Marcel Lehner")?;
    
    Ok(())
}
