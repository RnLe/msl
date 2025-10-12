use crate::lattice::{Lattice2D, Lattice3D};
use nalgebra::Matrix3;

/// Standard lattice construction utilities for common 2D lattices

/// Create a square lattice with given lattice parameter
pub fn square_lattice(a: f64) -> Lattice2D {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::new(direct, 1e-10)
}

/// Create a rectangular lattice with given lattice parameters
pub fn rectangular_lattice(a: f64, b: f64) -> Lattice2D {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::new(direct, 1e-10)
}

/// Create a hexagonal lattice with given lattice parameter
pub fn hexagonal_lattice(a: f64) -> Lattice2D {
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
    Lattice2D::new(direct, 1e-10)
}

/// Create an oblique lattice with given parameters and angle
pub fn oblique_lattice(a: f64, b: f64, gamma: f64) -> Lattice2D {
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
    Lattice2D::new(direct, 1e-10)
}

/// Create a centered rectangular lattice
pub fn centered_rectangular_lattice(a: f64, b: f64) -> Lattice2D {
    // This creates a primitive cell representation of a centered rectangular lattice
    let direct = Matrix3::new(a / 2.0, a / 2.0, 0.0, -b / 2.0, b / 2.0, 0.0, 0.0, 0.0, 1.0);
    Lattice2D::new(direct, 1e-10)
}

/// Standard lattice construction utilities for common 3D lattices

/// Create a simple cubic lattice
pub fn simple_cubic_lattice(a: f64) -> Lattice3D {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a);
    Lattice3D::new(direct, 1e-10)
}

/// Create a body-centered cubic (BCC) lattice
pub fn body_centered_cubic_lattice(a: f64) -> Lattice3D {
    let direct = Matrix3::new(
        -a / 2.0,
        a / 2.0,
        a / 2.0,
        a / 2.0,
        -a / 2.0,
        a / 2.0,
        a / 2.0,
        a / 2.0,
        -a / 2.0,
    );
    Lattice3D::new(direct, 1e-10)
}

/// Create a face-centered cubic (FCC) lattice
pub fn face_centered_cubic_lattice(a: f64) -> Lattice3D {
    let direct = Matrix3::new(
        0.0,
        a / 2.0,
        a / 2.0,
        a / 2.0,
        0.0,
        a / 2.0,
        a / 2.0,
        a / 2.0,
        0.0,
    );
    Lattice3D::new(direct, 1e-10)
}

/// Create a hexagonal close-packed (HCP) lattice
pub fn hexagonal_close_packed_lattice(a: f64, c: f64) -> Lattice3D {
    let direct = Matrix3::new(
        a,
        -a / 2.0,
        0.0,
        0.0,
        a * 3.0_f64.sqrt() / 2.0,
        0.0,
        0.0,
        0.0,
        c,
    );
    Lattice3D::new(direct, 1e-10)
}

/// Create a tetragonal lattice
pub fn tetragonal_lattice(a: f64, c: f64) -> Lattice3D {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, c);
    Lattice3D::new(direct, 1e-10)
}

/// Create an orthorhombic lattice
pub fn orthorhombic_lattice(a: f64, b: f64, c: f64) -> Lattice3D {
    let direct = Matrix3::new(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, c);
    Lattice3D::new(direct, 1e-10)
}

/// Create a rhombohedral lattice
pub fn rhombohedral_lattice(a: f64, alpha: f64) -> Lattice3D {
    let cos_alpha = alpha.cos();
    let sin_alpha = alpha.sin();

    let direct = Matrix3::new(
        a,
        a * cos_alpha,
        a * cos_alpha,
        0.0,
        a * sin_alpha,
        a * (cos_alpha - cos_alpha.powi(2) / sin_alpha.powi(2)).sqrt(),
        0.0,
        0.0,
        a * (1.0 - 3.0 * cos_alpha.powi(2) + 2.0 * cos_alpha.powi(3)).sqrt() / sin_alpha,
    );
    Lattice3D::new(direct, 1e-10)
}

/// Utility functions for lattice transformations

/// Rotate a 2D lattice by a given angle (in radians)
pub fn rotate_lattice_2d(lattice: &Lattice2D, angle: f64) -> Lattice2D {
    use nalgebra::Matrix3;

    let cos_theta = angle.cos();
    let sin_theta = angle.sin();

    // 2D rotation matrix embedded in 3D
    let rotation_matrix = Matrix3::new(
        cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0,
    );

    let rotated_direct = rotation_matrix * lattice.direct_basis();
    Lattice2D::new(rotated_direct, lattice.tolerance())
}

/// Scale a 2D lattice uniformly by a given factor
pub fn scale_lattice_2d(lattice: &Lattice2D, scale_factor: f64) -> Lattice2D {
    let scaled_direct = lattice.direct_basis() * scale_factor;
    Lattice2D::new(scaled_direct, lattice.tolerance())
}

/// Scale a lattice uniformly
pub fn scale_lattice_3d(lattice: &Lattice3D, scale_factor: f64) -> Lattice3D {
    let scaled_direct = lattice.direct_basis() * scale_factor;
    Lattice3D::new(scaled_direct, lattice.tolerance())
}

/// Apply a transformation matrix to a lattice
pub fn transform_lattice_2d(lattice: &Lattice2D, transformation: &Matrix3<f64>) -> Lattice2D {
    let transformed_direct = transformation * lattice.direct_basis();
    Lattice2D::new(transformed_direct, lattice.tolerance())
}

/// Apply a transformation matrix to a lattice
pub fn transform_lattice_3d(lattice: &Lattice3D, transformation: &Matrix3<f64>) -> Lattice3D {
    let transformed_direct = transformation * lattice.direct_basis();
    Lattice3D::new(transformed_direct, lattice.tolerance())
}

/// Create a supercell from a lattice with given multiplicities
pub fn create_supercell_2d(lattice: &Lattice2D, n1: i32, n2: i32) -> Lattice2D {
    let supercell_matrix = Matrix3::new(n1 as f64, 0.0, 0.0, 0.0, n2 as f64, 0.0, 0.0, 0.0, 1.0);

    let supercell_direct = lattice.direct_basis() * supercell_matrix;
    Lattice2D::new(supercell_direct, lattice.tolerance())
}

/// Create a supercell from a 3D lattice with given multiplicities
pub fn create_supercell_3d(lattice: &Lattice3D, n1: i32, n2: i32, n3: i32) -> Lattice3D {
    let supercell_matrix = Matrix3::new(
        n1 as f64, 0.0, 0.0, 0.0, n2 as f64, 0.0, 0.0, 0.0, n3 as f64,
    );

    let supercell_direct = lattice.direct_basis() * supercell_matrix;
    Lattice3D::new(supercell_direct, lattice.tolerance())
}
