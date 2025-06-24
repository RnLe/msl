use pyo3::prelude::*;
use moire_lattice::lattice::Lattice3D;

/// Python wrapper for the 3D lattice structure (stub for future implementation)
#[pyclass]
pub struct PyLattice3D {
    pub(crate) inner: Lattice3D,
}

#[pymethods]
impl PyLattice3D {
    /// Get lattice parameters (a, b, c)
    fn lattice_parameters(&self) -> (f64, f64, f64) {
        self.inner.lattice_parameters()
    }

    /// Get lattice angles (α, β, γ) in degrees
    fn lattice_angles(&self) -> (f64, f64, f64) {
        let (alpha, beta, gamma) = self.inner.lattice_angles();
        use std::f64::consts::PI;
        (alpha * 180.0 / PI, beta * 180.0 / PI, gamma * 180.0 / PI)
    }

    /// Get the unit cell volume
    fn unit_cell_volume(&self) -> f64 {
        self.inner.cell_volume()
    }

    /// Convert to 2D lattice by projecting to a-b plane
    fn to_2d(&self) -> crate::lattice::lattice2d::PyLattice2D {
        crate::lattice::lattice2d::PyLattice2D::new(self.inner.to_2d())
    }

    /// String representation
    fn __repr__(&self) -> String {
        let (a, b, c) = self.inner.lattice_parameters();
        let (alpha, beta, gamma) = self.lattice_angles();
        format!("PyLattice3D({:?}, a={:.3}, b={:.3}, c={:.3}, α={:.1}°, β={:.1}°, γ={:.1}°)",
            self.inner.bravais_type(),
            a, b, c, alpha, beta, gamma
        )
    }
}

impl PyLattice3D {
    pub fn new(inner: Lattice3D) -> Self {
        PyLattice3D { inner }
    }
}
