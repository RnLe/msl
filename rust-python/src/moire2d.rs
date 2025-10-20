//! Python bindings for Moire2D lattices

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::Matrix2;
use moire_lattice::moire_lattice::moire2d::{Moire2D, MoireTransformation};
use moire_lattice::lattice::lattice_like_2d::LatticeLike2D;

use crate::lattice2d::PyLattice2D;
use crate::base_matrix::{PyBaseMatrixDirect, PyBaseMatrixReciprocal};
use crate::lattice_types::PyBravais2D;
use crate::polyhedron::PyPolyhedron;
use crate::high_symmetry_points::PyHighSymmetryData;

/// Python wrapper for MoireTransformation
#[pyclass(name = "MoireTransformation")]
#[derive(Clone)]
pub struct PyMoireTransformation {
    pub(crate) inner: MoireTransformation,
}

#[pymethods]
impl PyMoireTransformation {
    /// Create a simple twist transformation: R(θ)
    ///
    /// Args:
    ///     angle: Rotation angle in radians
    ///
    /// Returns:
    ///     MoireTransformation: The twist transformation
    #[staticmethod]
    fn twist(angle: f64) -> Self {
        PyMoireTransformation { inner: MoireTransformation::Twist { angle } }
    }
    
    /// Create a rotation and uniform scaling transformation: s * R(θ)
    ///
    /// Args:
    ///     angle: Rotation angle in radians
    ///     scale: Uniform scale factor
    ///
    /// Returns:
    ///     MoireTransformation: The rotation-scale transformation
    #[staticmethod]
    fn rotation_scale(angle: f64, scale: f64) -> Self {
        PyMoireTransformation { 
            inner: MoireTransformation::RotationScale { angle, scale } 
        }
    }
    
    /// Create an anisotropic scaling transformation: diag(s_x, s_y)
    ///
    /// Args:
    ///     scale_x: Scale factor in x direction
    ///     scale_y: Scale factor in y direction
    ///
    /// Returns:
    ///     MoireTransformation: The anisotropic scale transformation
    #[staticmethod]
    fn anisotropic_scale(scale_x: f64, scale_y: f64) -> Self {
        PyMoireTransformation { 
            inner: MoireTransformation::AnisotropicScale { scale_x, scale_y } 
        }
    }
    
    /// Create a shear transformation
    ///
    /// Args:
    ///     shear_x: Shear in x direction
    ///     shear_y: Shear in y direction
    ///
    /// Returns:
    ///     MoireTransformation: The shear transformation
    #[staticmethod]
    fn shear(shear_x: f64, shear_y: f64) -> Self {
        PyMoireTransformation { 
            inner: MoireTransformation::Shear { shear_x, shear_y } 
        }
    }
    
    /// Create a general 2x2 matrix transformation
    ///
    /// Args:
    ///     matrix: 2x2 matrix as [[a, b], [c, d]]
    ///
    /// Returns:
    ///     MoireTransformation: The general transformation
    #[staticmethod]
    fn general(matrix: [[f64; 2]; 2]) -> Self {
        let mat = Matrix2::new(
            matrix[0][0], matrix[1][0],
            matrix[0][1], matrix[1][1],
        );
        PyMoireTransformation { inner: MoireTransformation::General(mat) }
    }
    
    /// Convert to 2x2 matrix form
    ///
    /// Returns:
    ///     List[List[float]]: 2x2 transformation matrix
    fn to_matrix(&self) -> [[f64; 2]; 2] {
        let mat = self.inner.to_matrix();
        [
            [mat[(0, 0)], mat[(1, 0)]],
            [mat[(0, 1)], mat[(1, 1)]],
        ]
    }
    
    /// Convert to 3x3 matrix form (embedding 2D transformation in 3D)
    ///
    /// Returns:
    ///     List[List[float]]: 3x3 transformation matrix
    fn to_matrix3(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.to_matrix3();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    fn __repr__(&self) -> String {
        match &self.inner {
            MoireTransformation::Twist { angle } => 
                format!("MoireTransformation.Twist(angle={:.4})", angle),
            MoireTransformation::RotationScale { angle, scale } => 
                format!("MoireTransformation.RotationScale(angle={:.4}, scale={:.4})", angle, scale),
            MoireTransformation::AnisotropicScale { scale_x, scale_y } => 
                format!("MoireTransformation.AnisotropicScale(scale_x={:.4}, scale_y={:.4})", scale_x, scale_y),
            MoireTransformation::Shear { shear_x, shear_y } => 
                format!("MoireTransformation.Shear(shear_x={:.4}, shear_y={:.4})", shear_x, shear_y),
            MoireTransformation::General(_) => 
                format!("MoireTransformation.General({:?})", self.to_matrix()),
        }
    }
}

/// Python wrapper for Moire2D
#[pyclass(name = "Moire2D")]
#[derive(Clone)]
pub struct PyMoire2D {
    pub(crate) inner: Moire2D,
}

#[pymethods]
impl PyMoire2D {
    /// Create a Moiré lattice from a base lattice and transformation
    ///
    /// Args:
    ///     base_lattice: The base Lattice2D
    ///     transformation: The MoireTransformation to apply
    ///
    /// Returns:
    ///     Moire2D: The Moiré lattice
    #[staticmethod]
    fn from_transformation(
        _base_lattice: &PyLattice2D,
        _transformation: &PyMoireTransformation,
    ) -> PyResult<Self> {
        // Note: The Rust implementation has a bug - it's calling from_transformation
        // on self, but it should be a static method. We'll work around this by
        // creating a temporary Moire2D first, then using the method.
        // For now, we'll need to expose this differently or fix the Rust code.
        // Let me check the implementation again...
        
        // Actually, looking at the Rust code, from_transformation is an instance method
        // but it should probably be a constructor. For Python bindings, we'll need to
        // call it differently. Since it's complex, let's create a wrapper that makes sense.
        
        Err(PyValueError::new_err(
            "from_transformation not yet fully implemented - Rust API needs adjustment"
        ))
    }
    
    /// Get the effective moiré lattice (returns a copy as Lattice2D interface)
    ///
    /// Note: This returns the effective lattice data through the LatticeLike2D interface
    /// The actual Moire2D struct stores lattice_1, lattice_2, and effective_lattice internally
    fn get_effective_lattice_data(&self) -> PyResult<PyLattice2D> {
        // Since effective_lattice is private, we reconstruct from the public interface
        let direct_mat = self.inner.direct_basis().base_matrix().clone();
        moire_lattice::lattice::lattice2d::Lattice2D::from_direct_matrix(direct_mat)
            .map(|inner| PyLattice2D { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    
    // Implement LatticeLike2D methods by forwarding to the inner Moire2D
    
    /// Get the direct space basis matrix
    fn direct_basis(&self) -> PyBaseMatrixDirect {
        PyBaseMatrixDirect { inner: self.inner.direct_basis().clone() }
    }
    
    /// Get the reciprocal space basis matrix
    fn reciprocal_basis(&self) -> PyBaseMatrixReciprocal {
        PyBaseMatrixReciprocal { inner: self.inner.reciprocal_basis().clone() }
    }
    
    /// Get the direct space Bravais lattice type
    fn direct_bravais(&self) -> PyBravais2D {
        PyBravais2D { inner: self.inner.direct_bravais() }
    }
    
    /// Get the reciprocal space Bravais lattice type
    fn reciprocal_bravais(&self) -> PyBravais2D {
        PyBravais2D { inner: self.inner.reciprocal_bravais() }
    }
    
    /// Get the direct space metric tensor
    fn direct_metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.direct_metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the reciprocal space metric tensor
    fn reciprocal_metric(&self) -> [[f64; 3]; 3] {
        let mat = self.inner.reciprocal_metric();
        [
            [mat[(0, 0)], mat[(1, 0)], mat[(2, 0)]],
            [mat[(0, 1)], mat[(1, 1)], mat[(2, 1)]],
            [mat[(0, 2)], mat[(1, 2)], mat[(2, 2)]],
        ]
    }
    
    /// Get the Wigner-Seitz cell
    fn wigner_seitz(&self) -> PyPolyhedron {
        PyPolyhedron { inner: self.inner.wigner_seitz().clone() }
    }
    
    /// Get the Brillouin zone
    fn brillouin_zone(&self) -> PyPolyhedron {
        PyPolyhedron { inner: self.inner.brillouin_zone().clone() }
    }
    
    /// Get direct space high symmetry points
    fn direct_high_symmetry(&self) -> PyHighSymmetryData {
        PyHighSymmetryData { inner: self.inner.direct_high_symmetry().clone() }
    }
    
    /// Get reciprocal space high symmetry points
    fn reciprocal_high_symmetry(&self) -> PyHighSymmetryData {
        PyHighSymmetryData { inner: self.inner.reciprocal_high_symmetry().clone() }
    }
    
    /// Get direct lattice parameters (a1, a2)
    fn direct_lattice_parameters(&self) -> (f64, f64) {
        self.inner.direct_lattice_parameters()
    }
    
    /// Get reciprocal lattice parameters (b1, b2)
    fn reciprocal_lattice_parameters(&self) -> (f64, f64) {
        self.inner.reciprocal_lattice_parameters()
    }
    
    /// Get direct lattice angle (gamma)
    fn direct_lattice_angle(&self) -> f64 {
        self.inner.direct_lattice_angle()
    }
    
    /// Get reciprocal lattice angle
    fn reciprocal_lattice_angle(&self) -> f64 {
        self.inner.reciprocal_lattice_angle()
    }
    
    /// Compute direct lattice points in a rectangle
    fn compute_direct_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<[f64; 3]> {
        self.inner.compute_direct_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Compute reciprocal lattice points in a rectangle
    fn compute_reciprocal_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<[f64; 3]> {
        self.inner.compute_reciprocal_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Generate a high symmetry k-point path
    fn generate_high_symmetry_k_path(&self, n_points_per_segment: u16) -> Vec<[f64; 3]> {
        self.inner.generate_high_symmetry_k_path(n_points_per_segment)
            .into_iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    fn __repr__(&self) -> String {
        format!("Moire2D(direct_bravais={}, a={:.4}, b={:.4})",
                self.direct_bravais().name(),
                self.direct_lattice_parameters().0,
                self.direct_lattice_parameters().1)
    }
}
