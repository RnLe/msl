use anyhow::{Error, Ok};
use nalgebra::{Matrix3, Vector3};

use crate::interfaces::Dimension;

/// Tolerance for comparing base vectors
const BASE_VECTOR_TOLERANCE: f64 = 1e-10;

pub struct BaseMatrix {
    matrix: Matrix3<f64>,
    dimension: Dimension,
}

impl BaseMatrix {
    pub fn new_2_d(base_1: Vector3<f64>, base_2: Vector3<f64>) -> Result<BaseMatrix, Error> {
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

        // Construct a matrix with the two provided base vectors and the cartesian base for z with length 1 (e_z = 0, 0, 1)
        Ok(BaseMatrix {
            matrix: Matrix3::from_columns(&[base_1, base_2, Vector3::new(0.0, 0.0, 1.0)]),
            dimension: Dimension::_2D,
        })
    }

    pub fn new_3_d(
        base_1: Vector3<f64>,
        base_2: Vector3<f64>,
        base_3: Vector3<f64>,
    ) -> Result<BaseMatrix, Error> {
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

        Ok(BaseMatrix {
            matrix: preliminary_base_matrix,
            dimension: Dimension::_3D,
        })
    }

    pub fn matrix(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    pub fn dimension(&self) -> &Dimension {
        &self.dimension
    }

    pub fn determinant(&self) -> f64 {
        self.matrix.determinant()
    }

    pub fn inverse(&self) -> Matrix3<f64> {
        self.matrix.try_inverse().unwrap()
    }

    pub fn transpose(&self) -> Matrix3<f64> {
        self.matrix.transpose()
    }

    pub fn base_vectors(&self) -> [Vector3<f64>; 3] {
        [
            self.matrix.column(0).into(),
            self.matrix.column(1).into(),
            self.matrix.column(2).into(),
        ]
    }
}
