use pyo3::prelude::*;
use moire_lattice::lattice::construction::*;
use std::f64::consts::PI;
use crate::lattice::lattice2d::PyLattice2D;

/// Create a square lattice with parameter a
#[pyfunction]
pub fn create_square_lattice(a: f64) -> PyLattice2D {
    PyLattice2D::new(square_lattice(a))
}

/// Create a rectangular lattice with parameters a, b
#[pyfunction]
pub fn create_rectangular_lattice(a: f64, b: f64) -> PyLattice2D {
    PyLattice2D::new(rectangular_lattice(a, b))
}

/// Create a hexagonal lattice with parameter a
#[pyfunction]
pub fn create_hexagonal_lattice(a: f64) -> PyLattice2D {
    PyLattice2D::new(hexagonal_lattice(a))
}

/// Create an oblique lattice with parameters a, b, and angle gamma (in degrees)
#[pyfunction]
pub fn create_oblique_lattice(a: f64, b: f64, gamma_degrees: f64) -> PyLattice2D {
    let gamma_radians = gamma_degrees * PI / 180.0;
    PyLattice2D::new(oblique_lattice(a, b, gamma_radians))
}

/// Create a centered rectangular lattice with parameters a, b
#[pyfunction]
pub fn create_centered_rectangular_lattice(a: f64, b: f64) -> PyLattice2D {
    PyLattice2D::new(centered_rectangular_lattice(a, b))
}

/// Scale a 2D lattice uniformly
#[pyfunction]
pub fn scale_lattice_2d(lattice: &PyLattice2D, scale: f64) -> PyLattice2D {
    PyLattice2D::new(scale_lattice_2d(&lattice.inner, scale))
}

/// Rotate a 2D lattice by angle (in degrees)
#[pyfunction]
pub fn rotate_lattice_2d(lattice: &PyLattice2D, angle_degrees: f64) -> PyLattice2D {
    let angle_radians = angle_degrees * PI / 180.0;
    PyLattice2D::new(rotate_lattice_2d(&lattice.inner, angle_radians))
}

/// Create a 2D supercell
#[pyfunction]
pub fn create_supercell_2d(lattice: &PyLattice2D, nx: i32, ny: i32) -> PyLattice2D {
    PyLattice2D::new(create_supercell_2d(&lattice.inner, nx, ny))
}

/// Python wrapper for creating lattices with a unified interface
#[pyclass]
pub struct PyLatticeConstructor;

#[pymethods]
impl PyLatticeConstructor {
    #[new]
    pub fn new() -> Self {
        PyLatticeConstructor
    }

    /// Create a lattice with a unified interface
    #[pyo3(signature = (lattice_type, a, b=None, angle=None))]
    pub fn create_lattice(
        &self,
        lattice_type: &str,
        a: f64,
        b: Option<f64>,
        angle: Option<f64>,
    ) -> PyResult<PyLattice2D> {
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
            "centered_rectangular" => {
                let b_val = b.unwrap_or(a);
                centered_rectangular_lattice(a, b_val)
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown lattice type: {}. Available types: square, rectangular, hexagonal, triangular, oblique, centered_rectangular", lattice_type)
            )),
        };

        Ok(PyLattice2D::new(lattice))
    }

    /// List available lattice types
    fn available_types(&self) -> Vec<String> {
        vec![
            "square".to_string(),
            "rectangular".to_string(),
            "hexagonal".to_string(),
            "triangular".to_string(),
            "oblique".to_string(),
            "centered_rectangular".to_string(),
        ]
    }
}
