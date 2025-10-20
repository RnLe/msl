use std::f64::consts::PI;

use crate::interfaces::space::{Direct, Reciprocal};
use crate::lattice::base_matrix::BaseMatrix;
use crate::lattice::lattice2d::Lattice2D;
use crate::lattice::lattice_like_2d::LatticeLike2D;

use anyhow::{Error, Ok};
use nalgebra::{Matrix2, Matrix3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub enum Commensurability {
    Commensurate {
        coincidence_indices: (i32, i32, i32, i32),
        radius: f64,
    },
    NonCommensurate,
}

/// Transformation type for the second lattice relative to the first
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoireTransformation {
    /// Simple rotation: R(θ)
    Twist { angle: f64 },
    /// Rotation and uniform scaling: s * R(θ)
    RotationScale { angle: f64, scale: f64 },
    /// Anisotropic scaling: diag(s_x, s_y)
    AnisotropicScale { scale_x: f64, scale_y: f64 },
    /// Shear transformation
    Shear { shear_x: f64, shear_y: f64 },
    /// General 2x2 matrix transformation
    General(Matrix2<f64>),
}

impl MoireTransformation {
    /// Convert to 2x2 matrix form
    pub fn to_matrix(&self) -> Matrix2<f64> {
        match self {
            MoireTransformation::Twist { angle } => {
                let c = angle.cos();
                let s = angle.sin();
                Matrix2::new(c, -s, s, c)
            }
            MoireTransformation::RotationScale { angle, scale } => {
                let c = angle.cos();
                let s = angle.sin();
                *scale * Matrix2::new(c, -s, s, c)
            }
            MoireTransformation::AnisotropicScale { scale_x, scale_y } => {
                Matrix2::new(*scale_x, 0.0, 0.0, *scale_y)
            }
            MoireTransformation::Shear { shear_x, shear_y } => {
                Matrix2::new(1.0, *shear_x, *shear_y, 1.0)
            }
            MoireTransformation::General(mat) => *mat,
        }
    }

    /// Convert to 3x3 matrix form (embedding 2D transformation in 3D)
    pub fn to_matrix3(&self) -> Matrix3<f64> {
        let mat2 = self.to_matrix();
        let mut mat3 = Matrix3::identity();
        mat3[(0, 0)] = mat2[(0, 0)];
        mat3[(0, 1)] = mat2[(0, 1)];
        mat3[(1, 0)] = mat2[(1, 0)];
        mat3[(1, 1)] = mat2[(1, 1)];
        mat3
    }
}

/// A 2D Moiré lattice formed by two overlapping 2D lattices
#[derive(Debug, Clone)]
pub struct Moire2D {
    /// The large-scale Moiré lattice
    effective_lattice: Lattice2D,
    /// First constituent lattice
    lattice_1: Lattice2D,
    /// Second constituent lattice
    lattice_2: Lattice2D,
    /// Transformation applied to create lattice_2 from lattice_1
    transformation: MoireTransformation,
    /// Enumeration to track whether the two constituent lattices are commensurate or not
    commensurability: Commensurability,
}

impl Moire2D {
    pub fn from_transformation(
        &self,
        base_lattice: &impl LatticeLike2D,
        transformation: MoireTransformation,
    ) -> Result<Moire2D, Error> {
        // First, get the transformation matrix
        let transformation_matrix = transformation.to_matrix3();

        // Assume that the LatticeLike2D object is well formed; thus apply the transformation to the base vectors directly
        // This matrix will also be the new base matrix for lattice_2
        let direct_transformed_basis =
            &transformation_matrix * base_lattice.direct_basis().base_matrix();

        // Construct second lattice from the new basis
        let lattice_2 = Lattice2D::from_direct_matrix(direct_transformed_basis)?;

        // Construct the Moiré bases
        let (direct_moire_basis, _) = compute_moire_basis(base_lattice, &lattice_2)?;

        // Construct the effective lattice
        let effective_lattice = Lattice2D::from_direct_matrix(direct_moire_basis)?;

        // Clone the base_lattice into an owned Lattice2D
        let lattice_1 =
            Lattice2D::from_direct_matrix(base_lattice.direct_basis().base_matrix().clone())?;

        // TODO: Make commensurability checks

        Ok(Moire2D {
            effective_lattice,
            lattice_1,
            lattice_2,
            transformation,
            commensurability: Commensurability::NonCommensurate,
        })
    }
}

/// Compute the Moiré lattice basis given two constituent lattices
/// # Arguments
/// * `lattice_1` - First constituent lattice
/// * `lattice_2` - Second constituent lattice
/// # Returns
/// Tuple of (direct moiré basis, reciprocal moiré basis)
pub fn compute_moire_basis(
    lattice_1: &impl LatticeLike2D,
    lattice_2: &impl LatticeLike2D,
) -> Result<(Matrix3<f64>, Matrix3<f64>), Error> {
    let [g1, g2, _] = lattice_1.reciprocal_basis().base_vectors();
    let [g1_prime, g2_prime, _] = lattice_2.reciprocal_basis().base_vectors();

    // Moiré reciprocal vectors are differences
    let g_m1 = g1_prime - g1;
    let g_m2 = g2_prime - g2;

    // Build reciprocal basis matrix
    let mut reciprocal_moire = Matrix3::identity();
    reciprocal_moire.set_column(0, &g_m1);
    reciprocal_moire.set_column(1, &g_m2);

    // Convert to direct basis
    let direct_moire = reciprocal_moire
        .try_inverse()
        .ok_or(Error::msg("Moiré reciprocal basis is singular."))?
        .transpose()
        * (2.0 * PI);

    Ok((direct_moire, reciprocal_moire))
}

// Implement Lattice2D operations which essentially forward everything to and from the `effective_lattice`.
impl LatticeLike2D for Moire2D {
    fn direct_basis(&self) -> &BaseMatrix<Direct> {
        self.effective_lattice.direct_basis()
    }
    fn reciprocal_basis(&self) -> &BaseMatrix<Reciprocal> {
        self.effective_lattice.reciprocal_basis()
    }
    fn direct_bravais(&self) -> crate::prelude::Bravais2D {
        self.effective_lattice.direct_bravais()
    }
    fn reciprocal_bravais(&self) -> crate::prelude::Bravais2D {
        self.effective_lattice.reciprocal_bravais()
    }
    fn direct_metric(&self) -> &Matrix3<f64> {
        self.effective_lattice.direct_metric()
    }
    fn reciprocal_metric(&self) -> &Matrix3<f64> {
        self.effective_lattice.reciprocal_metric()
    }
    fn wigner_seitz(&self) -> &crate::lattice::Polyhedron {
        self.effective_lattice.wigner_seitz()
    }
    fn brillouin_zone(&self) -> &crate::lattice::Polyhedron {
        self.effective_lattice.brillouin_zone()
    }
    fn direct_high_symmetry(&self) -> &crate::symmetries::HighSymmetryData {
        self.effective_lattice.direct_high_symmetry()
    }
    fn reciprocal_high_symmetry(&self) -> &crate::symmetries::HighSymmetryData {
        self.effective_lattice.reciprocal_high_symmetry()
    }
    fn compute_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<nalgebra::Vector3<f64>> {
        self.effective_lattice
            .compute_direct_lattice_points_in_rectangle(width, height)
    }
    fn compute_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<nalgebra::Vector3<f64>> {
        self.effective_lattice
            .compute_reciprocal_lattice_points_in_rectangle(width, height)
    }
    fn generate_high_symmetry_k_path(
        &self,
        n_points_per_segment: u16,
    ) -> Vec<nalgebra::Vector3<f64>> {
        self.effective_lattice
            .generate_high_symmetry_k_path(n_points_per_segment)
    }
    fn is_point_in_brillouin_zone(&self, k_point: nalgebra::Vector3<f64>) -> bool {
        self.effective_lattice.is_point_in_brillouin_zone(k_point)
    }
    fn is_point_in_wigner_seitz_cell(&self, r_point: nalgebra::Vector3<f64>) -> bool {
        self.effective_lattice
            .is_point_in_wigner_seitz_cell(r_point)
    }
    fn reduce_point_to_brillouin_zone(
        &self,
        k_point: nalgebra::Vector3<f64>,
    ) -> nalgebra::Vector3<f64> {
        self.effective_lattice
            .reduce_point_to_brillouin_zone(k_point)
    }
}
