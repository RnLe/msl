use std::f64::consts::PI;
use std::marker::PhantomData;

use anyhow::{Error, Ok};
use nalgebra::{Matrix3, Vector3};

use crate::config::BASE_VECTOR_TOLERANCE;
use crate::interfaces::Dimension;
use crate::interfaces::space::{Direct, Reciprocal};

#[derive(Debug, Clone)]
pub struct BaseMatrix<S> {
    base_matrix: Matrix3<f64>,
    dimension: Dimension,
    _s: std::marker::PhantomData<S>,
}

impl<S> BaseMatrix<S> {
    pub fn from_base_vectors_2d(base_1: Vector3<f64>, base_2: Vector3<f64>) -> Result<Self, Error> {
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

        // Ok to go
        let base_matrix = Matrix3::from_columns(&[base_1, base_2, Vector3::new(0.0, 0.0, 1.0)]);

        // Construct a matrix with the two provided base vectors and the cartesian base for z with length 1 (e_z = 0, 0, 1)
        Ok(Self {
            base_matrix,
            dimension: Dimension::_2D,
            _s: Default::default(),
        })
    }

    pub fn from_base_vectors_3d(
        base_1: Vector3<f64>,
        base_2: Vector3<f64>,
        base_3: Vector3<f64>,
    ) -> Result<Self, Error> {
        // Pre-Construct the base matrix for operations
        let base_matrix: Matrix3<f64> = Matrix3::from_columns(&[base_1, base_2, base_3]);

        // Run tests on the base vectors to decide whether a base matrix can be constructed
        // Linearly non-dependent
        if base_matrix.determinant().abs() < BASE_VECTOR_TOLERANCE {
            return Err(Error::msg(
                "Determinant too small. Vectors are either too small or linearly dependent.",
            ));
        }

        // Ok to go

        Ok(Self {
            base_matrix,
            dimension: Dimension::_3D,
            _s: Default::default(),
        })
    }

    pub fn from_matrix_2d(matrix: Matrix3<f64>) -> Result<Self, Error> {
        Self::from_base_vectors_2d(matrix.column(0).into(), matrix.column(1).into())
    }

    pub fn from_matrix_3d(matrix: Matrix3<f64>) -> Result<Self, Error> {
        Self::from_base_vectors_3d(
            matrix.column(0).into(),
            matrix.column(1).into(),
            matrix.column(2).into(),
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

fn a_star_from_a(matrix: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    // Physics convention: A^T B = 2pi I => B = 2pi A^{-T}
    let inverse = matrix.try_inverse()?.transpose();
    let tau = 2.0 * PI;
    Some(tau * inverse)
}

impl BaseMatrix<Direct> {
    /// Compute the reciprocal-space basis B = 2pi * A^{-T}
    pub fn to_reciprocal(&self) -> Result<BaseMatrix<Reciprocal>, Error> {
        let base_matrix = a_star_from_a(&self.base_matrix)
            .ok_or_else(|| Error::msg("Matrix inversion failed. This really shouldn't happen."))?;
        Ok(BaseMatrix::<Reciprocal> {
            base_matrix,
            dimension: self.dimension,
            _s: PhantomData,
        })
    }
}

impl BaseMatrix<Reciprocal> {
    /// Compute the real-space basis A = 2pi * B^{-T}
    pub fn to_direct(&self) -> Result<BaseMatrix<Direct>, Error> {
        let base_matrix = a_star_from_a(&self.base_matrix)
            .ok_or_else(|| Error::msg("Matrix inversion failed. This really shouldn't happen."))?;
        Ok(BaseMatrix::<Direct> {
            base_matrix,
            dimension: self.dimension,
            _s: PhantomData,
        })
    }
}
