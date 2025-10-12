use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// The five 2D Bravais lattices.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Bravais2D {
    Oblique,
    Rectangular,
    CenteredRectangular,
    Square,
    Hexagonal,
}

/// Centerings that make sense in 3D setting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Centering {
    Primitive,
    BodyCentered,
    FaceCentered,
    BaseCentered, // could be refined into A/B/C later
}

/// The seven 3D crystal systems, together with centering to cover all 14 Bravais types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Bravais3D {
    Triclinic(Centering),
    Monoclinic(Centering),
    Orthorhombic(Centering),
    Tetragonal(Centering),
    Trigonal(Centering),
    Hexagonal(Centering),
    Cubic(Centering),
}

/// Identify 2D Bravais lattice type from metric tensor.
pub fn identify_bravais_2d(metric: &Matrix3<f64>, tol: f64) -> Bravais2D {
    // Extract lattice parameters from metric tensor (2D only uses a, b)
    let a = metric[(0, 0)].sqrt();
    let b = metric[(1, 1)].sqrt();

    // Calculate γ angle (only relevant angle in 2D)
    let gamma = (metric[(0, 1)] / (a * b)).acos();

    // Check length relationships
    let a_eq_b = approx_equal(a, b, tol);

    // Check angle relationships
    let gamma_90 = is_right_angle(gamma, tol);
    let gamma_120 = is_120_degrees(gamma, tol);

    // For centered rectangular, check if we can find a centered cell
    // This happens when there's a lattice point at the center of the rectangle
    // A more accurate check would be to look at the actual lattice basis vectors
    // For now, assume simple rectangular construction creates primitive rectangular
    let is_centered_rect = false; // Disable heuristic that was causing misclassification

    // Identify 2D Bravais type
    match (a_eq_b, gamma_90, gamma_120, is_centered_rect) {
        // Square: a = b, γ = 90°
        (true, true, _, _) => Bravais2D::Square,

        // Hexagonal: a = b, γ = 120°
        (true, _, true, _) => Bravais2D::Hexagonal,

        // Centered Rectangular
        (_, _, _, true) => Bravais2D::CenteredRectangular,

        // Rectangular: a ≠ b, γ = 90°
        (false, true, _, false) => Bravais2D::Rectangular,

        // Oblique: general case
        _ => Bravais2D::Oblique,
    }
}

/// Identify 3D Bravais lattice type from metric tensor.
pub fn identify_bravais_3d(metric: &Matrix3<f64>, tol: f64) -> Bravais3D {
    // Extract lattice parameters from metric tensor
    let a = metric[(0, 0)].sqrt();
    let b = metric[(1, 1)].sqrt();
    let c = metric[(2, 2)].sqrt();

    // Calculate angles
    let alpha = (metric[(1, 2)] / (b * c)).acos();
    let beta = (metric[(0, 2)] / (a * c)).acos();
    let gamma = (metric[(0, 1)] / (a * b)).acos();

    // Check length relationships
    let a_eq_b = approx_equal(a, b, tol);
    let b_eq_c = approx_equal(b, c, tol);
    let a_eq_c = approx_equal(a, c, tol);
    let all_equal = a_eq_b && b_eq_c;

    // Check angle relationships
    let alpha_90 = is_right_angle(alpha, tol);
    let beta_90 = is_right_angle(beta, tol);
    let gamma_90 = is_right_angle(gamma, tol);
    let all_90 = alpha_90 && beta_90 && gamma_90;
    let gamma_120 = is_120_degrees(gamma, tol);

    // For now, assume primitive centering (P)
    // TODO: Implement centering detection based on additional constraints
    let centering = Centering::Primitive;

    // Identify crystal system based on constraints
    match (all_equal, a_eq_b, b_eq_c, a_eq_c, all_90, gamma_120) {
        // Cubic: a = b = c, α = β = γ = 90°
        (true, _, _, _, true, _) => Bravais3D::Cubic(centering),

        // Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
        (false, true, false, false, _, true) if alpha_90 && beta_90 => {
            Bravais3D::Hexagonal(centering)
        }

        // Trigonal: a = b = c, α = β = γ ≠ 90°
        (true, _, _, _, false, _) => Bravais3D::Trigonal(centering),

        // Tetragonal: a = b ≠ c, α = β = γ = 90°
        (false, true, false, false, true, _) => Bravais3D::Tetragonal(centering),

        // Orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
        (false, false, false, false, true, _) => Bravais3D::Orthorhombic(centering),

        // Monoclinic: a ≠ b ≠ c, α = γ = 90° ≠ β
        (false, false, false, false, false, _) if alpha_90 && gamma_90 && !beta_90 => {
            Bravais3D::Monoclinic(centering)
        }

        // Triclinic: a ≠ b ≠ c, α ≠ β ≠ γ ≠ 90°
        _ => Bravais3D::Triclinic(centering),
    }
}

/// Check if two values are approximately equal within tolerance
pub fn approx_equal(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Check if angle is approximately 90 degrees (π/2 radians)
pub fn is_right_angle(angle: f64, tol: f64) -> bool {
    approx_equal(angle, PI / 2.0, tol)
}

/// Check if angle is approximately 120 degrees (2π/3 radians)
pub fn is_120_degrees(angle: f64, tol: f64) -> bool {
    approx_equal(angle, 2.0 * PI / 3.0, tol)
}
