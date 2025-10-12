use crate::lattice::PyLattice2D;
use moire_lattice::moire_lattice::{Moire2D, MoireTransformation};
use nalgebra::{Matrix2, Vector3};
use pyo3::prelude::*;

/// Python wrapper for MoireTransformation enum
#[pyclass]
#[derive(Clone)]
pub struct PyMoireTransformation {
    pub(crate) inner: MoireTransformation,
}

#[pymethods]
impl PyMoireTransformation {
    /// Create a rotation and uniform scaling transformation
    #[staticmethod]
    fn rotation_scale(angle: f64, scale: f64) -> Self {
        PyMoireTransformation {
            inner: MoireTransformation::RotationScale { angle, scale },
        }
    }

    /// Create an anisotropic scaling transformation
    #[staticmethod]
    fn anisotropic_scale(scale_x: f64, scale_y: f64) -> Self {
        PyMoireTransformation {
            inner: MoireTransformation::AnisotropicScale { scale_x, scale_y },
        }
    }

    /// Create a shear transformation
    #[staticmethod]
    fn shear(shear_x: f64, shear_y: f64) -> Self {
        PyMoireTransformation {
            inner: MoireTransformation::Shear { shear_x, shear_y },
        }
    }

    /// Create a general 2x2 matrix transformation
    #[staticmethod]
    fn general(matrix: Vec<Vec<f64>>) -> PyResult<Self> {
        if matrix.len() != 2 || matrix[0].len() != 2 || matrix[1].len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix must be 2x2",
            ));
        }

        let mat = Matrix2::new(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]);

        Ok(PyMoireTransformation {
            inner: MoireTransformation::General(mat),
        })
    }

    /// Convert transformation to 2x2 matrix
    fn to_matrix(&self) -> Vec<Vec<f64>> {
        let mat = self.inner.to_matrix();
        vec![
            vec![mat[(0, 0)], mat[(0, 1)]],
            vec![mat[(1, 0)], mat[(1, 1)]],
        ]
    }

    /// Convert transformation to 3x3 matrix (2D embedded in 3D)
    fn to_matrix3(&self) -> Vec<Vec<f64>> {
        let mat = self.inner.to_matrix3();
        vec![
            vec![mat[(0, 0)], mat[(0, 1)], mat[(0, 2)]],
            vec![mat[(1, 0)], mat[(1, 1)], mat[(1, 2)]],
            vec![mat[(2, 0)], mat[(2, 1)], mat[(2, 2)]],
        ]
    }
}

/// Python wrapper for the 2D moiré lattice structure
#[pyclass]
pub struct PyMoire2D {
    pub(crate) inner: Moire2D,
}

#[pymethods]
impl PyMoire2D {
    /// Get the moiré lattice as a regular Lattice2D
    fn as_lattice2d(&self) -> PyLattice2D {
        PyLattice2D {
            inner: self.inner.as_lattice2d(),
        }
    }

    /// Get the primitive vectors of the moiré lattice
    fn primitive_vectors(&self) -> (Vec<f64>, Vec<f64>) {
        let (v1, v2) = self.inner.primitive_vectors();
        (vec![v1[0], v1[1], v1[2]], vec![v2[0], v2[1], v2[2]])
    }

    /// Get the moiré periodicity ratio (moiré to original lattice constant)
    fn moire_period_ratio(&self) -> f64 {
        self.inner.moire_period_ratio()
    }

    /// Check if a point belongs to lattice 1
    fn is_lattice1_point(&self, point: Vec<f64>) -> PyResult<bool> {
        if point.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Point must have 3 coordinates",
            ));
        }

        let vec = Vector3::new(point[0], point[1], point[2]);
        Ok(self.inner.is_lattice1_point(vec))
    }

    /// Check if a point belongs to lattice 2
    fn is_lattice2_point(&self, point: Vec<f64>) -> PyResult<bool> {
        if point.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Point must have 3 coordinates",
            ));
        }

        let vec = Vector3::new(point[0], point[1], point[2]);
        Ok(self.inner.is_lattice2_point(vec))
    }

    /// Get stacking type at a given position (AA, A, B, or None)
    fn get_stacking_at(&self, point: Vec<f64>) -> PyResult<Option<String>> {
        if point.len() != 3 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Point must have 3 coordinates",
            ));
        }

        let vec = Vector3::new(point[0], point[1], point[2]);
        Ok(self.inner.get_stacking_at(vec))
    }

    /// Get the first constituent lattice
    fn lattice_1(&self) -> PyLattice2D {
        PyLattice2D {
            inner: self.inner.lattice_1.clone(),
        }
    }

    /// Get the second constituent lattice
    fn lattice_2(&self) -> PyLattice2D {
        PyLattice2D {
            inner: self.inner.lattice_2.clone(),
        }
    }

    /// Get the transformation used to create the moiré pattern
    fn transformation(&self) -> PyMoireTransformation {
        PyMoireTransformation {
            inner: self.inner.transformation.clone(),
        }
    }

    /// Get the twist angle in radians
    fn twist_angle(&self) -> f64 {
        self.inner.twist_angle
    }

    /// Get the twist angle in degrees
    fn twist_angle_degrees(&self) -> f64 {
        self.inner.twist_angle.to_degrees()
    }

    /// Check if the moiré pattern is commensurate
    fn is_commensurate(&self) -> bool {
        self.inner.is_commensurate
    }

    /// Get coincidence indices if commensurate
    fn coincidence_indices(&self) -> Option<(i32, i32, i32, i32)> {
        self.inner.coincidence_indices
    }

    /// Get the unit cell area of the moiré lattice
    fn cell_area(&self) -> f64 {
        self.inner.cell_area
    }

    /// Get the Bravais lattice type
    fn bravais_type(&self) -> String {
        format!("{:?}", self.inner.bravais)
    }
}
