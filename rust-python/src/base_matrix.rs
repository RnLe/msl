//! Python bindings for BaseMatrix types

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{Matrix3, Vector3};
use moire_lattice::lattice::base_matrix::BaseMatrix;
use moire_lattice::interfaces::space::{Direct, Reciprocal};

/// Python wrapper for BaseMatrix<Direct>
#[pyclass(name = "BaseMatrixDirect")]
#[derive(Clone)]
pub struct PyBaseMatrixDirect {
    pub(crate) inner: BaseMatrix<Direct>,
}

#[pymethods]
impl PyBaseMatrixDirect {
    /// Create a 2D base matrix from two base vectors
    ///
    /// Args:
    ///     base_1: First base vector as [x, y, z] (z should be 0)
    ///     base_2: Second base vector as [x, y, z] (z should be 0)
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space base matrix
    #[staticmethod]
    fn from_base_vectors_2d(base_1: [f64; 3], base_2: [f64; 3]) -> PyResult<Self> {
        let v1 = Vector3::new(base_1[0], base_1[1], base_1[2]);
        let v2 = Vector3::new(base_2[0], base_2[1], base_2[2]);
        
        BaseMatrix::<Direct>::from_base_vectors_2d(v1, v2)
            .map(|inner| PyBaseMatrixDirect { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Create a 3D base matrix from three base vectors
    ///
    /// Args:
    ///     base_1: First base vector as [x, y, z]
    ///     base_2: Second base vector as [x, y, z]
    ///     base_3: Third base vector as [x, y, z]
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space base matrix
    #[staticmethod]
    fn from_base_vectors_3d(base_1: [f64; 3], base_2: [f64; 3], base_3: [f64; 3]) -> PyResult<Self> {
        let v1 = Vector3::new(base_1[0], base_1[1], base_1[2]);
        let v2 = Vector3::new(base_2[0], base_2[1], base_2[2]);
        let v3 = Vector3::new(base_3[0], base_3[1], base_3[2]);
        
        BaseMatrix::<Direct>::from_base_vectors_3d(v1, v2, v3)
            .map(|inner| PyBaseMatrixDirect { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Create a 2D base matrix from a 3x3 matrix (column-major)
    ///
    /// Args:
    ///     matrix: 3x3 matrix as a flat list of 9 elements (column-major order)
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space base matrix
    #[staticmethod]
    fn from_matrix_2d(matrix: [[f64; 3]; 3]) -> PyResult<Self> {
        let mat = Matrix3::from_column_slice(&[
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2],
        ]);
        
        BaseMatrix::<Direct>::from_matrix_2d(mat)
            .map(|inner| PyBaseMatrixDirect { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Create a 3D base matrix from a 3x3 matrix (column-major)
    ///
    /// Args:
    ///     matrix: 3x3 matrix as a flat list of 9 elements (column-major order)
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space base matrix
    #[staticmethod]
    fn from_matrix_3d(matrix: [[f64; 3]; 3]) -> PyResult<Self> {
        let mat = Matrix3::from_column_slice(&[
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2],
        ]);
        
        BaseMatrix::<Direct>::from_matrix_3d(mat)
            .map(|inner| PyBaseMatrixDirect { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Get the base matrix as a 3x3 array
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 matrix in column-major order
    fn base_matrix(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.base_matrix();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the three base vectors
    ///
    /// Returns:
    ///     Tuple[List[float], List[float], List[float]]: The three base vectors
    fn base_vectors(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let [v1, v2, v3] = self.inner.base_vectors();
        (
            [v1[0], v1[1], v1[2]],
            [v2[0], v2[1], v2[2]],
            [v3[0], v3[1], v3[2]],
        )
    }
    
    /// Get the determinant of the base matrix
    ///
    /// Returns:
    ///     float: The determinant
    fn determinant(&self) -> f64 {
        self.inner.determinant()
    }
    
    /// Get the inverse of the base matrix
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 inverse matrix
    fn inverse(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.inverse();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the transpose of the base matrix
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 transposed matrix
    fn transpose(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.transpose();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the metric tensor G = A^T * A
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 metric tensor
    fn metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Convert to reciprocal space basis
    ///
    /// Returns:
    ///     BaseMatrixReciprocal: The reciprocal space basis
    fn to_reciprocal(&self) -> PyResult<PyBaseMatrixReciprocal> {
        self.inner.to_reciprocal()
            .map(|inner| PyBaseMatrixReciprocal { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn __repr__(&self) -> String {
        format!("BaseMatrixDirect({:?})", self.inner.base_matrix())
    }
}

/// Python wrapper for BaseMatrix<Reciprocal>
#[pyclass(name = "BaseMatrixReciprocal")]
#[derive(Clone)]
pub struct PyBaseMatrixReciprocal {
    pub(crate) inner: BaseMatrix<Reciprocal>,
}

#[pymethods]
impl PyBaseMatrixReciprocal {
    /// Get the base matrix as a 3x3 array
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 matrix in column-major order
    fn base_matrix(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.base_matrix();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the three base vectors
    ///
    /// Returns:
    ///     Tuple[List[float], List[float], List[float]]: The three base vectors
    fn base_vectors(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let [v1, v2, v3] = self.inner.base_vectors();
        (
            [v1[0], v1[1], v1[2]],
            [v2[0], v2[1], v2[2]],
            [v3[0], v3[1], v3[2]],
        )
    }
    
    /// Get the determinant of the base matrix
    ///
    /// Returns:
    ///     float: The determinant
    fn determinant(&self) -> f64 {
        self.inner.determinant()
    }
    
    /// Get the inverse of the base matrix
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 inverse matrix
    fn inverse(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.inverse();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the transpose of the base matrix
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 transposed matrix
    fn transpose(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.transpose();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the metric tensor G = B^T * B
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 metric tensor
    fn metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Convert to direct space basis
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space basis
    fn to_direct(&self) -> PyResult<PyBaseMatrixDirect> {
        self.inner.to_direct()
            .map(|inner| PyBaseMatrixDirect { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    fn __repr__(&self) -> String {
        format!("BaseMatrixReciprocal({:?})", self.inner.base_matrix())
    }
}
