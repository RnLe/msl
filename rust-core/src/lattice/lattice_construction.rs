use crate::lattice::Lattice2D;
use nalgebra::Matrix3;

/// Standard lattice construction utilities for common 2D lattices

/// Create a square lattice with given lattice parameter
pub fn square_lattice(a: f64) -> Result<Lattice2D, anyhow::Error> {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::from_matrix(direct)
}

/// Create a rectangular lattice with given lattice parameters
pub fn rectangular_lattice(a: f64, b: f64) -> Result<Lattice2D, anyhow::Error> {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::from_matrix(direct)
}

/// Create a hexagonal lattice with given lattice parameter
pub fn hexagonal_lattice(a: f64) -> Result<Lattice2D, anyhow::Error> {
    let direct = Matrix3::new(
        a,
        -a / 2.0,
        0.0,
        0.0,
        a * 3.0_f64.sqrt() / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    Lattice2D::from_matrix(direct)
}

/// Create an oblique lattice with given parameters and angle
pub fn oblique_lattice(a: f64, b: f64, gamma: f64) -> Result<Lattice2D, anyhow::Error> {
    let direct = Matrix3::new(
        a,
        b * gamma.cos(),
        0.0,
        0.0,
        b * gamma.sin(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    Lattice2D::from_matrix(direct)
}

/// Create a centered rectangular lattice
pub fn centered_rectangular_lattice(a: f64, b: f64) -> Result<Lattice2D, anyhow::Error> {
    // This creates a primitive cell representation of a centered rectangular lattice
    let direct = Matrix3::new(a / 2.0, a / 2.0, 0.0, -b / 2.0, b / 2.0, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::from_matrix(direct)
}

/// Utility functions for lattice transformations

/// Rotate a 2D lattice by a given angle (in radians)
pub fn rotate_lattice_2d(lattice: &Lattice2D, angle: f64) -> Result<Lattice2D, anyhow::Error> {
    use nalgebra::Matrix3;

    let cos_theta = angle.cos();
    let sin_theta = angle.sin();

    // 2D rotation matrix embedded in 3D
    let rotation_matrix = Matrix3::new(
        cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0,
    );

    let rotated_direct = rotation_matrix * lattice.direct_matrix();
    Lattice2D::from_matrix(rotated_direct)
}

/// Scale a 2D lattice uniformly by a given factor
pub fn scale_lattice_2d(
    lattice: &Lattice2D,
    scale_factor: f64,
) -> Result<Lattice2D, anyhow::Error> {
    let scaled_direct = lattice.direct_matrix() * scale_factor;
    Lattice2D::from_matrix(scaled_direct)
}

/// Apply a transformation matrix to a lattice
pub fn transform_lattice_2d(
    lattice: &Lattice2D,
    transformation: &Matrix3<f64>,
) -> Result<Lattice2D, anyhow::Error> {
    let transformed_direct = transformation * lattice.direct_matrix();
    Lattice2D::from_matrix(transformed_direct)
}

/// Create a supercell from a lattice with given multiplicities
pub fn create_supercell_2d(
    lattice: &Lattice2D,
    n1: i32,
    n2: i32,
) -> Result<Lattice2D, anyhow::Error> {
    let supercell_matrix = Matrix3::new(n1 as f64, 0.0, 0.0, 0.0, n2 as f64, 0.0, 0.0, 0.0, 1.0);

    let supercell_direct = lattice.direct_matrix() * supercell_matrix;
    Lattice2D::from_matrix(supercell_direct)
}
