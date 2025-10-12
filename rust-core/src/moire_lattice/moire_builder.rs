use crate::lattice::lattice2d::Lattice2D;
use crate::moire_lattice::moire_validation_algorithms::{
    compute_moire_basis, find_commensurate_angles, validate_commensurability,
};
use crate::moire_lattice::moire2d::{Moire2D, MoireTransformation};
use nalgebra::Matrix2;

/// Builder for constructing Moire2D lattices
pub struct MoireBuilder {
    lattice_1: Option<Lattice2D>,
    transformation: Option<MoireTransformation>,
    tolerance: f64,
}

impl MoireBuilder {
    /// Create a new MoireBuilder
    pub fn new() -> Self {
        MoireBuilder {
            lattice_1: None,
            transformation: None,
            tolerance: 1e-10,
        }
    }

    /// Set the base lattice
    pub fn with_base_lattice(mut self, lattice: Lattice2D) -> Self {
        self.lattice_1 = Some(lattice);
        self
    }

    /// Set tolerance for calculations
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set a rotation and uniform scaling transformation
    pub fn with_twist_and_scale(mut self, angle: f64, scale: f64) -> Self {
        self.transformation = Some(MoireTransformation::RotationScale { angle, scale });
        self
    }

    /// Set an anisotropic scaling transformation
    pub fn with_anisotropic_scale(mut self, scale_x: f64, scale_y: f64) -> Self {
        self.transformation = Some(MoireTransformation::AnisotropicScale { scale_x, scale_y });
        self
    }

    /// Set a shear transformation
    pub fn with_shear(mut self, shear_x: f64, shear_y: f64) -> Self {
        self.transformation = Some(MoireTransformation::Shear { shear_x, shear_y });
        self
    }

    /// Set a general 2x2 transformation matrix
    pub fn with_general_transformation(mut self, matrix: Matrix2<f64>) -> Self {
        self.transformation = Some(MoireTransformation::General(matrix));
        self
    }

    /// Build the Moire2D lattice
    pub fn build(self) -> Result<Moire2D, String> {
        let lattice_1 = self.lattice_1.ok_or("Base lattice not set")?;
        let transformation = self.transformation.ok_or("Transformation not set")?;

        // Create the second lattice by transforming the first
        let lattice_2 = transform_lattice(&lattice_1, &transformation);

        // Check for commensurability
        let (is_commensurate, coincidence_indices) =
            validate_commensurability(&lattice_1, &lattice_2, self.tolerance);

        // Compute moiré basis vectors
        let moire_basis = compute_moire_basis(&lattice_1, &lattice_2, self.tolerance)?;

        // Create the moiré lattice structure
        let moire_lattice = Lattice2D::from_matrix(moire_basis, self.tolerance);

        // Extract twist angle
        let twist_angle = extract_twist_angle(&transformation);

        Ok(Moire2D {
            // Copy fields from moire_lattice
            direct: moire_lattice.direct,
            reciprocal: moire_lattice.reciprocal,
            bravais: moire_lattice.bravais,
            cell_area: moire_lattice.cell_area,
            metric: moire_lattice.metric,
            tol: moire_lattice.tol,
            sym_ops: moire_lattice.sym_ops,
            wigner_seitz_cell: moire_lattice.wigner_seitz_cell,
            brillouin_zone: moire_lattice.brillouin_zone,
            high_symmetry: moire_lattice.high_symmetry,
            // Moiré-specific fields
            lattice_1,
            lattice_2,
            transformation,
            twist_angle,
            is_commensurate,
            coincidence_indices,
        })
    }
}

/// Transform a lattice using the given transformation
fn transform_lattice(lattice: &Lattice2D, transformation: &MoireTransformation) -> Lattice2D {
    let transform_3d = transformation.to_matrix3();
    let new_direct = transform_3d * lattice.direct;
    Lattice2D::from_matrix(new_direct, lattice.tol)
}

/// Extract the rotation angle from a transformation
fn extract_twist_angle(transformation: &MoireTransformation) -> f64 {
    match transformation {
        MoireTransformation::RotationScale { angle, .. } => *angle,
        MoireTransformation::General(mat) => {
            // Extract angle from general matrix using SVD or polar decomposition
            // For now, use atan2 of the off-diagonal elements
            mat[(1, 0)].atan2(mat[(0, 0)])
        }
        _ => 0.0, // No rotation for pure scaling or shear
    }
}

/// Create a simple twisted bilayer moiré pattern
pub fn twisted_bilayer(lattice: Lattice2D, angle: f64) -> Result<Moire2D, String> {
    MoireBuilder::new()
        .with_base_lattice(lattice)
        .with_twist_and_scale(angle, 1.0)
        .build()
}

/// Create a moiré pattern with commensurate angle
pub fn commensurate_moire(
    lattice: Lattice2D,
    m1: i32,
    m2: i32,
    n1: i32,
    n2: i32,
) -> Result<Moire2D, String> {
    // Compute the exact commensurate angle
    let angles = find_commensurate_angles(&lattice, 100)?;

    // Find the angle corresponding to these indices
    let angle = angles
        .into_iter()
        .find(|(_, indices)| indices == &(m1, m2, n1, n2))
        .map(|(angle, _)| angle)
        .ok_or("No commensurate angle found for given indices")?;

    MoireBuilder::new()
        .with_base_lattice(lattice)
        .with_twist_and_scale(angle, 1.0)
        .build()
}
