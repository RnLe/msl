use crate::lattice::PyLattice2D;
use crate::moire_lattice::{PyMoire2D, PyMoireTransformation};
use moire_lattice::moire_lattice::{MoireBuilder, commensurate_moire, twisted_bilayer};
use nalgebra::Matrix2;
use pyo3::prelude::*;

/// Python wrapper for the MoireBuilder
#[pyclass]
pub struct PyMoireBuilder {
    lattice: Option<moire_lattice::lattice::Lattice2D>,
    transformation: Option<moire_lattice::moire_lattice::MoireTransformation>,
    tolerance: f64,
}

#[pymethods]
impl PyMoireBuilder {
    /// Create a new MoireBuilder
    #[new]
    fn new() -> Self {
        PyMoireBuilder {
            lattice: None,
            transformation: None,
            tolerance: 1e-10,
        }
    }

    /// Set the base lattice
    fn with_base_lattice(&mut self, lattice: &PyLattice2D) -> PyResult<()> {
        self.lattice = Some(lattice.inner.clone());
        Ok(())
    }

    /// Set tolerance for calculations
    fn with_tolerance(&mut self, tol: f64) -> PyResult<()> {
        self.tolerance = tol;
        Ok(())
    }

    /// Set a rotation and uniform scaling transformation
    fn with_twist_and_scale(&mut self, angle: f64, scale: f64) -> PyResult<()> {
        self.transformation =
            Some(moire_lattice::moire_lattice::MoireTransformation::RotationScale { angle, scale });
        Ok(())
    }

    /// Set an anisotropic scaling transformation
    fn with_anisotropic_scale(&mut self, scale_x: f64, scale_y: f64) -> PyResult<()> {
        self.transformation = Some(
            moire_lattice::moire_lattice::MoireTransformation::AnisotropicScale {
                scale_x,
                scale_y,
            },
        );
        Ok(())
    }

    /// Set a shear transformation
    fn with_shear(&mut self, shear_x: f64, shear_y: f64) -> PyResult<()> {
        self.transformation =
            Some(moire_lattice::moire_lattice::MoireTransformation::Shear { shear_x, shear_y });
        Ok(())
    }

    /// Set a general 2x2 transformation matrix
    fn with_general_transformation(&mut self, matrix: Vec<Vec<f64>>) -> PyResult<()> {
        if matrix.len() != 2 || matrix[0].len() != 2 || matrix[1].len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix must be 2x2",
            ));
        }

        let mat = Matrix2::new(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]);

        self.transformation = Some(moire_lattice::moire_lattice::MoireTransformation::General(
            mat,
        ));
        Ok(())
    }

    /// Set transformation from PyMoireTransformation
    fn with_transformation(&mut self, transformation: &PyMoireTransformation) -> PyResult<()> {
        self.transformation = Some(transformation.inner.clone());
        Ok(())
    }

    /// Build the Moire2D lattice
    fn build(&self) -> PyResult<PyMoire2D> {
        let lattice = self.lattice.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Base lattice not set")
        })?;
        let transformation = self.transformation.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Transformation not set")
        })?;

        let mut builder = MoireBuilder::new()
            .with_base_lattice(lattice.clone())
            .with_tolerance(self.tolerance);

        builder = match transformation {
            moire_lattice::moire_lattice::MoireTransformation::RotationScale { angle, scale } => {
                builder.with_twist_and_scale(*angle, *scale)
            }
            moire_lattice::moire_lattice::MoireTransformation::AnisotropicScale {
                scale_x,
                scale_y,
            } => builder.with_anisotropic_scale(*scale_x, *scale_y),
            moire_lattice::moire_lattice::MoireTransformation::Shear { shear_x, shear_y } => {
                builder.with_shear(*shear_x, *shear_y)
            }
            moire_lattice::moire_lattice::MoireTransformation::General(mat) => {
                builder.with_general_transformation(*mat)
            }
        };

        let moire = builder
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        Ok(PyMoire2D { inner: moire })
    }
}

/// Create a simple twisted bilayer moiré pattern
#[pyfunction]
pub fn py_twisted_bilayer(lattice: &PyLattice2D, angle: f64) -> PyResult<PyMoire2D> {
    let moire = twisted_bilayer(lattice.inner.clone(), angle)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(PyMoire2D { inner: moire })
}

/// Create a moiré pattern with commensurate angle
#[pyfunction]
pub fn py_commensurate_moire(
    lattice: &PyLattice2D,
    m1: i32,
    m2: i32,
    n1: i32,
    n2: i32,
) -> PyResult<PyMoire2D> {
    let moire = commensurate_moire(lattice.inner.clone(), m1, m2, n1, n2)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(PyMoire2D { inner: moire })
}
