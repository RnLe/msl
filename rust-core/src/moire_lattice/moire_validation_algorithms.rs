use crate::lattice::lattice2d::Lattice2D;
use nalgebra::{Matrix3, Vector3};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Find commensurate angles for a given lattice
pub fn find_commensurate_angles(
    lattice: &Lattice2D,
    max_index: i32,
) -> Result<Vec<(f64, (i32, i32, i32, i32))>, String> {
    let mut angles = Vec::new();

    // Get lattice parameters for complex representation
    let (a1, a2) = lattice.direct_base_vectors();
    let tau = Complex64::new(a2[0] / a1.norm(), a2[1] / a1.norm());

    // Search over integer indices
    for m1 in -max_index..=max_index {
        for m2 in -max_index..=max_index {
            for n1 in -max_index..=max_index {
                for n2 in -max_index..=max_index {
                    // Skip if any index is zero or if gcd != 1
                    if m1 == 0 || m2 == 0 || n1 == 0 || n2 == 0 {
                        continue;
                    }
                    if gcd_four(m1.abs(), m2.abs(), n1.abs(), n2.abs()) != 1 {
                        continue;
                    }

                    // Compute angle using the formula from the theory
                    let angle = compute_twist_angle(m1, m2, n1, n2, tau);

                    // Only keep angles in [0, π]
                    if angle >= 0.0 && angle <= PI {
                        angles.push((angle, (m1, m2, n1, n2)));
                    }
                }
            }
        }
    }

    // Sort by angle
    angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    angles.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-10);

    Ok(angles)
}

/// Validate if two lattices form a commensurate moiré pattern
pub fn validate_commensurability(
    lattice_1: &Lattice2D,
    lattice_2: &Lattice2D,
    tolerance: f64,
) -> (bool, Option<(i32, i32, i32, i32)>) {
    // Try to find coincidence vectors
    let max_search = 50;

    for m1 in -max_search..=max_search {
        for m2 in -max_search..=max_search {
            if m1 == 0 && m2 == 0 {
                continue;
            }

            let v1 = lattice_1.fractional_to_cartesian(Vector3::new(m1 as f64, m2 as f64, 0.0));

            // Check if v1 is a lattice vector of lattice_2
            let v1_frac_in_l2 = lattice_2.cartesian_to_fractional(v1);

            let n1 = v1_frac_in_l2[0].round() as i32;
            let n2 = v1_frac_in_l2[1].round() as i32;

            let error = (v1_frac_in_l2[0] - n1 as f64).abs() + (v1_frac_in_l2[1] - n2 as f64).abs();

            if error < tolerance && (n1 != 0 || n2 != 0) {
                return (true, Some((m1, m2, n1, n2)));
            }
        }
    }

    (false, None)
}

/// Compute the moiré basis vectors from two lattices
pub fn compute_moire_basis(
    lattice_1: &Lattice2D,
    lattice_2: &Lattice2D,
    tolerance: f64,
) -> Result<Matrix3<f64>, String> {
    // Get reciprocal lattices
    let g1 = lattice_1.reciprocal.column(0).into_owned();
    let g2 = lattice_1.reciprocal.column(1).into_owned();
    let g1_prime = lattice_2.reciprocal.column(0).into_owned();
    let g2_prime = lattice_2.reciprocal.column(1).into_owned();

    // Moiré reciprocal vectors are differences
    let g_m1 = g1_prime - g1;
    let g_m2 = g2_prime - g2;

    // Check if vectors are non-zero
    if g_m1.norm() < tolerance || g_m2.norm() < tolerance {
        return Err("Lattices are too similar to form a moiré pattern".to_string());
    }

    // Build reciprocal basis matrix
    let mut reciprocal_moire = Matrix3::zeros();
    reciprocal_moire.set_column(0, &g_m1);
    reciprocal_moire.set_column(1, &g_m2);
    reciprocal_moire[(2, 2)] = 1.0; // z-component

    // Convert to direct basis
    let direct_moire = reciprocal_moire
        .try_inverse()
        .ok_or("Moiré reciprocal basis is singular")?
        .transpose()
        * (2.0 * PI);

    Ok(direct_moire)
}

/// Compute twist angle using the complex number formulation
fn compute_twist_angle(m1: i32, m2: i32, n1: i32, n2: i32, tau: Complex64) -> f64 {
    let u = m1 as f64 + m2 as f64 * tau;
    let z = n1 as f64 + n2 as f64 * tau;

    u.arg() - z.arg()
}

/// Greatest common divisor of four integers
fn gcd_four(a: i32, b: i32, c: i32, d: i32) -> i32 {
    gcd(gcd(a, b), gcd(c, d))
}

/// Greatest common divisor of two integers
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

/// Check if a moiré pattern has specific symmetries
pub fn analyze_moire_symmetry(moire: &Moire2D) -> Vec<String> {
    let mut symmetries = Vec::new();

    // Check if both lattices have the same Bravais type
    if moire.lattice_1.bravais == moire.lattice_2.bravais {
        symmetries.push("Same Bravais type".to_string());
    }

    // Check for specific angles that preserve symmetry
    let angle_deg = moire.twist_angle.to_degrees();

    // Hexagonal lattices
    if matches!(
        moire.lattice_1.bravais,
        crate::lattice::lattice_types::Bravais2D::Hexagonal
    ) {
        if (angle_deg % 60.0).abs() < moire.tol {
            symmetries.push("60° rotation symmetry preserved".to_string());
        }
        if (angle_deg - 30.0).abs() < moire.tol {
            symmetries.push("30° rotation (quasicrystalline)".to_string());
        }
    }

    // Square lattices
    if matches!(
        moire.lattice_1.bravais,
        crate::lattice::lattice_types::Bravais2D::Square
    ) {
        if (angle_deg % 90.0).abs() < moire.tol {
            symmetries.push("90° rotation symmetry preserved".to_string());
        }
        if (angle_deg - 45.0).abs() < moire.tol {
            symmetries.push("45° rotation".to_string());
        }
    }

    symmetries
}

/// Compute the effective moiré potential at a given point
pub fn moire_potential_at(
    moire: &Moire2D,
    point: Vector3<f64>,
    v_aa: f64, // Potential at AA stacking
    v_ab: f64, // Potential at AB stacking
) -> f64 {
    // Simple model: interpolate based on distance to nearest AA/AB regions
    // This is a placeholder - real implementation would use DFT or tight-binding

    match moire.get_stacking_at(point) {
        Some(stacking) => match stacking.as_str() {
            "AA" => v_aa,
            "A" | "B" => v_ab,
            _ => 0.0,
        },
        None => {
            // Interpolate based on nearby stacking
            // Simplified: just return average for now
            (v_aa + v_ab) / 2.0
        }
    }
}

// Re-export Moire2D for use in validation functions
use crate::moire_lattice::moire2d::Moire2D;
