use std::f64::consts::PI;

use anyhow::{Error, Ok};
use nalgebra::{Matrix3, Vector3};

use crate::config::BASE_VECTOR_TOLERANCE;
use crate::interfaces::{Dimension, Space};

#[derive(Debug, Clone)]
pub struct BaseMatrix {
    base_matrix: Matrix3<f64>,
    space: Space,
    dimension: Dimension,
}

impl BaseMatrix {
    pub fn from_base_vectors_2d(
        base_1: Vector3<f64>,
        base_2: Vector3<f64>,
        space: Space,
    ) -> Result<Self, Error> {
        // Run tests on the base vectors to decide whether a base matrix can be constructed
        // Linearly non-dependent (also checks for zero vectors)
        if base_1.cross(&base_2).norm() < BASE_VECTOR_TOLERANCE {
            return Err(Error::msg(
                "Base vectors are either linearly dependent or too close to zero.",
            ));
        }

        // Coplanar with xy-plane (no z-component)
        if base_1[2].abs() > BASE_VECTOR_TOLERANCE || base_2[2].abs() > BASE_VECTOR_TOLERANCE {
            return Err(Error::msg(
                "Base vectors are not coplanar with the xy-plane (should have 0 for z-components).",
            ));
        }

        let base_matrix;

        // Ok to go. Construct both direct and reciprocal matrices (propagate error upwards)
        base_matrix = Matrix3::from_columns(&[base_1, base_2, Vector3::new(0.0, 0.0, 1.0)]);

        // Construct a matrix with the two provided base vectors and the cartesian base for z with length 1 (e_z = 0, 0, 1)
        Ok(BaseMatrix {
            base_matrix,
            space,
            dimension: Dimension::_2D,
        })
    }

    pub fn from_base_vectors_3d(
        base_1: Vector3<f64>,
        base_2: Vector3<f64>,
        base_3: Vector3<f64>,
        space: Space,
    ) -> Result<Self, Error> {
        // Pre-Construct the base matrix for operations
        let preliminary_base_matrix: Matrix3<f64> =
            Matrix3::from_columns(&[base_1, base_2, base_3]);

        // Run tests on the base vectors to decide whether a base matrix can be constructed
        // Linearly non-dependent
        if preliminary_base_matrix.determinant().abs() < BASE_VECTOR_TOLERANCE {
            return Err(Error::msg(
                "Determinant too small. Vectors are either too small or linearly dependent.",
            ));
        }

        let base_matrix;

        // Ok to go. Construct both direct and reciprocal matrices (propagate error upwards)
        base_matrix = preliminary_base_matrix;

        Ok(BaseMatrix {
            base_matrix,
            space,
            dimension: Dimension::_3D,
        })
    }

    pub fn from_matrix_2d(matrix: Matrix3<f64>, space: Space) -> Result<Self, Error> {
        Self::from_base_vectors_2d(matrix.column(0).into(), matrix.column(1).into(), space)
    }

    pub fn from_matrix_3d(matrix: Matrix3<f64>, space: Space) -> Result<Self, Error> {
        Self::from_base_vectors_3d(
            matrix.column(0).into(),
            matrix.column(1).into(),
            matrix.column(2).into(),
            space,
        )
    }

    /// This method applies the conversion to reciprocal space. It does NOT require the base matrix to be in direct space, as it converts whatever space it is in to the other space.
    ///
    /// This means: if the base matrix is already in reciprocal space, it will be converted to direct space.
    /// If it is in direct space, it will be converted to reciprocal space.
    pub fn apply_reciprocal_transformation(matrix: &BaseMatrix) -> Result<BaseMatrix, Error> {
        // A base matrix' inverse is guaranteed to exist (per constructor checks)
        let inverse = matrix.inverse();

        // Pass the inverted matrix to the constructor to ensure all checks are performed
        Self::from_matrix_3d(
            inverse,
            match matrix.space {
                Space::Real => Space::Reciprocal,
                Space::Reciprocal => Space::Real,
            },
        )
    }

    pub fn base_matrix(&self) -> &Matrix3<f64> {
        &self.base_matrix
    }

    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }

    pub fn determinant(&self) -> f64 {
        self.base_matrix.determinant()
    }

    pub fn inverse(&self) -> Matrix3<f64> {
        self.base_matrix
            .try_inverse()
            .expect("Matrix inversion failed. This should never happen as the constructor checks for invertibility.")
    }

    pub fn transpose(&self) -> Matrix3<f64> {
        self.base_matrix.transpose()
    }

    pub fn metric(&self) -> Matrix3<f64> {
        self.base_matrix.transpose() * self.base_matrix
    }

    pub fn base_vectors(&self) -> [Vector3<f64>; 3] {
        [
            self.base_matrix.column(0).into(),
            self.base_matrix.column(1).into(),
            self.base_matrix.column(2).into(),
        ]
    }
}
