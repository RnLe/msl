use crate::lattice::lattice2d::Lattice2D;
use crate::lattice::lattice_bravais_types::{Bravais2D, approx_equal};
use std::f64::consts::PI;

/// Check if angle is equivalent to 90° (considering crystallographic equivalences)
/// 
/// In crystallography, angles of 90°, 270° are equivalent for lattice classification.
/// We normalize to [0, 2π) and check against both possibilities.
fn is_equivalent_to_90_degrees(angle: f64, tol: f64) -> bool {
    // Normalize angle to [0, 2π) range
    let normalized = angle.rem_euclid(2.0 * PI);
    
    // Check against 90° (π/2) and 270° (3π/2)
    approx_equal(normalized, PI / 2.0, tol) || 
    approx_equal(normalized, 3.0 * PI / 2.0, tol)
}

/// Check if angle is equivalent to hexagonal angles (considering crystallographic equivalences)
/// 
/// In crystallography, angles of 60°, 120°, 240°, 300° are equivalent for hexagonal lattices.
/// We normalize to [0, 2π) and check against all possibilities.
fn is_equivalent_to_hexagonal_angle(angle: f64, tol: f64) -> bool {
    // Normalize angle to [0, 2π) range
    let normalized = angle.rem_euclid(2.0 * PI);
    
    // Check against 60° (π/3), 120° (2π/3), 240° (4π/3), 300° (5π/3)
    approx_equal(normalized, PI / 3.0, tol) ||           // 60°
    approx_equal(normalized, 2.0 * PI / 3.0, tol) ||     // 120°
    approx_equal(normalized, 4.0 * PI / 3.0, tol) ||     // 240°
    approx_equal(normalized, 5.0 * PI / 3.0, tol)        // 300°
}

/// Determine the Bravais lattice type from a Lattice2D instance.
/// 
/// This method analyzes the lattice structure to identify its Bravais type
/// based on lattice parameters and symmetry operations.
pub fn determine_bravais_type_2d(lattice: &Lattice2D) -> Bravais2D {
    let tol = lattice.tolerance();
    
    // Get lattice parameters
    let (a, b) = lattice.lattice_parameters();
    let gamma = lattice.lattice_angle();
    
    // Check length relationships
    let a_eq_b = approx_equal(a, b, tol);
    
    // Check angle relationships (with crystallographic equivalences)
    let gamma_90 = is_equivalent_to_90_degrees(gamma, tol);
    let gamma_hex = is_equivalent_to_hexagonal_angle(gamma, tol);
    
    // For centered rectangular, we need to check if the lattice has
    // a centering that distinguishes it from primitive rectangular
    let is_centered = check_for_centering_2d(lattice);
    
    // Determine Bravais type based on parameters and symmetry
    match (a_eq_b, gamma_90, gamma_hex, is_centered) {
        // Square: a = b, γ equivalent to 90°
        (true, true, false, _) => Bravais2D::Square,
        
        // Hexagonal: a = b, γ equivalent to hexagonal angles
        (true, false, true, _) => Bravais2D::Hexagonal,
        
        // Centered Rectangular: a ≠ b, γ equivalent to 90°, with centering
        (false, true, false, true) => Bravais2D::CenteredRectangular,
        
        // Rectangular: a ≠ b, γ equivalent to 90°, primitive
        (false, true, false, false) => Bravais2D::Rectangular,
        
        // Oblique: general case (including any impossible combinations)
        _ => Bravais2D::Oblique,
    }
}

/// Check if a 2D lattice has centering (for distinguishing centered rectangular from primitive).
/// 
/// A centered rectangular lattice has an additional lattice point at the center
/// of the primitive rectangular cell, effectively doubling the primitive cell.
/// 
/// For most standard constructions, we assume primitive unless explicitly centered.
fn check_for_centering_2d(_lattice: &Lattice2D) -> bool {
    // For now, we default to primitive (false) for all lattices
    // This can be refined later with more sophisticated centering detection
    // based on symmetry analysis or explicit construction parameters
    false
}

/// Validate that a Lattice2D's stored Bravais type matches its actual structure.
/// 
/// Returns true if the stored type matches the determined type, false otherwise.
pub fn validate_bravais_type_2d(lattice: &Lattice2D) -> bool {
    let stored_type = lattice.bravais_type();
    let determined_type = determine_bravais_type_2d(lattice);
    stored_type == determined_type
}

/// Get a detailed analysis of why a lattice has a particular Bravais type.
/// 
/// Returns a tuple of (determined_type, reason_string).
pub fn analyze_bravais_type_2d(lattice: &Lattice2D) -> (Bravais2D, String) {
    let tol = lattice.tolerance();
    let (a, b) = lattice.lattice_parameters();
    let gamma = lattice.lattice_angle();
    let gamma_deg = gamma.to_degrees();
    
    let a_eq_b = approx_equal(a, b, tol);
    let gamma_90 = is_equivalent_to_90_degrees(gamma, tol);
    let gamma_hex = is_equivalent_to_hexagonal_angle(gamma, tol);
    let is_centered = check_for_centering_2d(lattice);
    
    let bravais_type = determine_bravais_type_2d(lattice);
    
    let reason = match bravais_type {
        Bravais2D::Square => {
            format!("Square lattice: a = b = {:.6}, γ = {:.2}° ≈ 90°", a, gamma_deg)
        }
        Bravais2D::Hexagonal => {
            format!("Hexagonal lattice: a = b = {:.6}, γ = {:.2}° ≈ 60°/120°", a, gamma_deg)
        }
        Bravais2D::CenteredRectangular => {
            format!("Centered Rectangular lattice: a = {:.6}, b = {:.6} (a ≠ b), γ = {:.2}° ≈ 90°, with centering", 
                    a, b, gamma_deg)
        }
        Bravais2D::Rectangular => {
            format!("Rectangular lattice: a = {:.6}, b = {:.6} (a ≠ b), γ = {:.2}° ≈ 90°", 
                    a, b, gamma_deg)
        }
        Bravais2D::Oblique => {
            let mut details = Vec::new();
            if !a_eq_b {
                details.push(format!("a ≠ b ({:.6} vs {:.6})", a, b));
            }
            if !gamma_90 && !gamma_hex {
                details.push(format!("γ = {:.2}° (not 90° or hexagonal)", gamma_deg));
            }
            if is_centered {
                details.push("centered".to_string());
            }
            
            let detail_str = if details.is_empty() {
                "general case".to_string()
            } else {
                details.join(", ")
            };
            
            format!("Oblique lattice: a = {:.6}, b = {:.6}, γ = {:.2}° ({})", 
                    a, b, gamma_deg, detail_str)
        }
    };
    
    (bravais_type, reason)
}
