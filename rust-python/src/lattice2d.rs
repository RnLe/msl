//! Python bindings for Lattice2D

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{Matrix3, Vector3};
use moire_lattice::lattice::lattice2d::Lattice2D;
use moire_lattice::lattice::lattice_like_2d::LatticeLike2D;

use crate::base_matrix::{PyBaseMatrixDirect, PyBaseMatrixReciprocal};
use crate::lattice_types::PyBravais2D;
use crate::polyhedron::PyPolyhedron;
use crate::high_symmetry_points::PyHighSymmetryData;

/// Python wrapper for Lattice2D
#[pyclass(name = "Lattice2D")]
#[derive(Clone)]
pub struct PyLattice2D {
    pub(crate) inner: Lattice2D,
}

#[pymethods]
impl PyLattice2D {
    /// Create a new 2D lattice from a direct space basis matrix
    ///
    /// Args:
    ///     direct: 3x3 matrix as [[col0], [col1], [col2]] where each column is a basis vector
    ///
    /// Returns:
    ///     Lattice2D: The 2D lattice
    #[staticmethod]
    fn from_direct_matrix(direct: [[f64; 3]; 3]) -> PyResult<Self> {
        let mat = Matrix3::from_column_slice(&[
            direct[0][0], direct[0][1], direct[0][2],
            direct[1][0], direct[1][1], direct[1][2],
            direct[2][0], direct[2][1], direct[2][2],
        ]);
        
        Lattice2D::from_direct_matrix(mat)
            .map(|inner| PyLattice2D { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Create a 2D lattice from two basis vectors
    ///
    /// Args:
    ///     a1: First basis vector [x, y, z] (z should be 0)
    ///     a2: Second basis vector [x, y, z] (z should be 0)
    ///
    /// Returns:
    ///     Lattice2D: The 2D lattice
    #[staticmethod]
    fn from_basis_vectors(a1: [f64; 3], a2: [f64; 3]) -> PyResult<Self> {
        let mat = Matrix3::from_columns(&[
            Vector3::new(a1[0], a1[1], a1[2]),
            Vector3::new(a2[0], a2[1], a2[2]),
            Vector3::new(0.0, 0.0, 1.0),
        ]);
        
        Lattice2D::from_direct_matrix(mat)
            .map(|inner| PyLattice2D { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    /// Get the direct space basis matrix
    ///
    /// Returns:
    ///     BaseMatrixDirect: The direct space basis
    fn direct_basis(&self) -> PyBaseMatrixDirect {
        PyBaseMatrixDirect { inner: self.inner.direct_basis().clone() }
    }
    
    /// Get the reciprocal space basis matrix
    ///
    /// Returns:
    ///     BaseMatrixReciprocal: The reciprocal space basis
    fn reciprocal_basis(&self) -> PyBaseMatrixReciprocal {
        PyBaseMatrixReciprocal { inner: self.inner.reciprocal_basis().clone() }
    }
    
    /// Get the direct space Bravais lattice type
    ///
    /// Returns:
    ///     Bravais2D: The Bravais lattice type
    fn direct_bravais(&self) -> PyBravais2D {
        PyBravais2D { inner: self.inner.direct_bravais() }
    }
    
    /// Get the reciprocal space Bravais lattice type
    ///
    /// Returns:
    ///     Bravais2D: The reciprocal Bravais lattice type
    fn reciprocal_bravais(&self) -> PyBravais2D {
        PyBravais2D { inner: self.inner.reciprocal_bravais() }
    }
    
    /// Get the direct space metric tensor
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 metric tensor
    fn direct_metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.direct_metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the reciprocal space metric tensor
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 metric tensor
    fn reciprocal_metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.reciprocal_metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the Wigner-Seitz cell
    ///
    /// Returns:
    ///     Polyhedron: The Wigner-Seitz cell
    fn wigner_seitz(&self) -> PyPolyhedron {
        PyPolyhedron { inner: self.inner.wigner_seitz().clone() }
    }
    
    /// Get the Brillouin zone
    ///
    /// Returns:
    ///     Polyhedron: The first Brillouin zone
    fn brillouin_zone(&self) -> PyPolyhedron {
        PyPolyhedron { inner: self.inner.brillouin_zone().clone() }
    }
    
    /// Get direct space high symmetry points
    ///
    /// Returns:
    ///     HighSymmetryData: High symmetry points and paths
    fn direct_high_symmetry(&self) -> PyHighSymmetryData {
        PyHighSymmetryData { inner: self.inner.direct_high_symmetry().clone() }
    }
    
    /// Get reciprocal space high symmetry points
    ///
    /// Returns:
    ///     HighSymmetryData: High symmetry points and paths
    fn reciprocal_high_symmetry(&self) -> PyHighSymmetryData {
        PyHighSymmetryData { inner: self.inner.reciprocal_high_symmetry().clone() }
    }
    
    /// Get direct lattice parameters (a1, a2)
    ///
    /// Returns:
    ///     Tuple[float, float]: Lattice parameters
    fn direct_lattice_parameters(&self) -> (f64, f64) {
        self.inner.direct_lattice_parameters()
    }
    
    /// Get reciprocal lattice parameters (b1, b2)
    ///
    /// Returns:
    ///     Tuple[float, float]: Reciprocal lattice parameters
    fn reciprocal_lattice_parameters(&self) -> (f64, f64) {
        self.inner.reciprocal_lattice_parameters()
    }
    
    /// Get direct lattice angle (gamma)
    ///
    /// Returns:
    ///     float: Angle in radians
    fn direct_lattice_angle(&self) -> f64 {
        self.inner.direct_lattice_angle()
    }
    
    /// Get reciprocal lattice angle
    ///
    /// Returns:
    ///     float: Angle in radians
    fn reciprocal_lattice_angle(&self) -> f64 {
        self.inner.reciprocal_lattice_angle()
    }
    
    /// Compute direct lattice points in a rectangle
    ///
    /// Args:
    ///     width: Width of the rectangle
    ///     height: Height of the rectangle
    ///
    /// Returns:
    ///     List[List[float]]: List of lattice points as [x, y, z]
    fn compute_direct_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<[f64; 3]> {
        self.inner.compute_direct_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Compute reciprocal lattice points in a rectangle
    ///
    /// Args:
    ///     width: Width of the rectangle
    ///     height: Height of the rectangle
    ///
    /// Returns:
    ///     List[List[float]]: List of reciprocal lattice points as [x, y, z]
    fn compute_reciprocal_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<[f64; 3]> {
        self.inner.compute_reciprocal_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Generate a high symmetry k-point path for band structure calculations
    ///
    /// Args:
    ///     n_points_per_segment: Number of points to interpolate between each high symmetry point
    ///
    /// Returns:
    ///     List[List[float]]: List of k-points as [kx, ky, kz]
    fn generate_high_symmetry_k_path(&self, n_points_per_segment: u16) -> Vec<[f64; 3]> {
        self.inner.generate_high_symmetry_k_path(n_points_per_segment)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Check if a point is in the Brillouin zone
    ///
    /// Args:
    ///     k_point: Point to check as [kx, ky, kz]
    ///
    /// Returns:
    ///     bool: True if the point is in the Brillouin zone
    fn is_point_in_brillouin_zone(&self, k_point: [f64; 3]) -> bool {
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        self.inner.is_point_in_brillouin_zone(k)
    }
    
    /// Check if a point is in the Wigner-Seitz cell
    ///
    /// Args:
    ///     r_point: Point to check as [x, y, z]
    ///
    /// Returns:
    ///     bool: True if the point is in the Wigner-Seitz cell
    fn is_point_in_wigner_seitz_cell(&self, r_point: [f64; 3]) -> bool {
        let r = Vector3::new(r_point[0], r_point[1], r_point[2]);
        self.inner.is_point_in_wigner_seitz_cell(r)
    }
    
    /// Reduce a k-point to the first Brillouin zone
    ///
    /// Args:
    ///     k_point: Point to reduce as [kx, ky, kz]
    ///
    /// Returns:
    ///     List[float]: Reduced k-point as [kx, ky, kz]
    fn reduce_point_to_brillouin_zone(&self, k_point: [f64; 3]) -> [f64; 3] {
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        let reduced = self.inner.reduce_point_to_brillouin_zone(k);
        [reduced[0], reduced[1], reduced[2]]
    }
    
    fn __repr__(&self) -> String {
        format!("Lattice2D(direct_bravais={}, reciprocal_bravais={}, a={:.4}, b={:.4})",
                self.direct_bravais().name(),
                self.reciprocal_bravais().name(),
                self.direct_lattice_parameters().0,
                self.direct_lattice_parameters().1)
    }
}
