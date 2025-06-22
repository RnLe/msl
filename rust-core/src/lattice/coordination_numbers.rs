use nalgebra::{Matrix3, Vector3};
use crate::lattice::bravais_types::{Bravais2D, Bravais3D};

/// Calculate the coordination number (number of nearest neighbors) for a 2D lattice
pub fn coordination_number_2d(bravais: &Bravais2D) -> usize {
    match bravais {
        Bravais2D::Square => 4,
        Bravais2D::Hexagonal => 6,
        Bravais2D::Rectangular | Bravais2D::CenteredRectangular => 4,
        Bravais2D::Oblique => 4, // General case
    }
}

/// Calculate the coordination number for a 3D lattice
pub fn coordination_number_3d(bravais: &Bravais3D) -> usize {
    match bravais {
        Bravais3D::Cubic(_) => 6,  // Simple cubic has 6 nearest neighbors
        Bravais3D::Hexagonal(_) => 12,
        Bravais3D::Tetragonal(_) => 4,
        Bravais3D::Orthorhombic(_) => 6,
        Bravais3D::Trigonal(_) => 6,
        Bravais3D::Monoclinic(_) => 4,
        Bravais3D::Triclinic(_) => 4,
    }
}

/// Find nearest neighbor vectors for a 2D lattice
pub fn nearest_neighbors_2d(
    direct_basis: &Matrix3<f64>, 
    bravais: &Bravais2D, 
    _tol: f64
) -> Vec<Vector3<f64>> {
    let mut neighbors = Vec::new();
    let a1 = direct_basis.column(0);
    let a2 = direct_basis.column(1);
    
    match bravais {
        Bravais2D::Square => {
            // For square lattice: ±a1, ±a2
            neighbors.push(a1.into());
            neighbors.push((-a1).into());
            neighbors.push(a2.into());
            neighbors.push((-a2).into());
        },
        Bravais2D::Hexagonal => {
            // For hexagonal lattice: a1, a2, -(a1+a2), -a1, -a2, (a1+a2)
            neighbors.push(a1.into());
            neighbors.push(a2.into());
            neighbors.push((a1 + a2).into());
            neighbors.push((-a1).into());
            neighbors.push((-a2).into());
            neighbors.push((-a1 - a2).into());
        },
        Bravais2D::Rectangular | Bravais2D::CenteredRectangular => {
            // For rectangular lattice: ±a1, ±a2
            neighbors.push(a1.into());
            neighbors.push((-a1).into());
            neighbors.push(a2.into());
            neighbors.push((-a2).into());
        },
        Bravais2D::Oblique => {
            // For oblique lattice: ±a1, ±a2 (general case)
            neighbors.push(a1.into());
            neighbors.push((-a1).into());
            neighbors.push(a2.into());
            neighbors.push((-a2).into());
        },
    }
    
    neighbors
}

/// Find nearest neighbor vectors for a 3D lattice
pub fn nearest_neighbors_3d(
    direct_basis: &Matrix3<f64>, 
    bravais: &Bravais3D, 
    _tol: f64
) -> Vec<Vector3<f64>> {
    let mut neighbors = Vec::new();
    let a1 = direct_basis.column(0);
    let a2 = direct_basis.column(1);
    let a3 = direct_basis.column(2);
    
    match bravais {
        Bravais3D::Cubic(_) => {
            // For simple cubic: ±a1, ±a2, ±a3
            neighbors.push(a1.into());
            neighbors.push((-a1).into());
            neighbors.push(a2.into());
            neighbors.push((-a2).into());
            neighbors.push(a3.into());
            neighbors.push((-a3).into());
        },
        _ => {
            // TODO: Implement specific neighbor patterns for other 3D lattices
            // For now, use the basic cubic pattern
            neighbors.push(a1.into());
            neighbors.push((-a1).into());
            neighbors.push(a2.into());
            neighbors.push((-a2).into());
            neighbors.push(a3.into());
            neighbors.push((-a3).into());
        },
    }
    
    neighbors
}

/// Calculate the nearest neighbor distance for a 2D lattice
pub fn nearest_neighbor_distance_2d(direct_basis: &Matrix3<f64>, bravais: &Bravais2D) -> f64 {
    let neighbors = nearest_neighbors_2d(direct_basis, bravais, 1e-10);
    if neighbors.is_empty() {
        return 0.0;
    }
    
    neighbors.iter()
        .map(|neighbor| neighbor.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}

/// Calculate the nearest neighbor distance for a 3D lattice
pub fn nearest_neighbor_distance_3d(direct_basis: &Matrix3<f64>, bravais: &Bravais3D) -> f64 {
    let neighbors = nearest_neighbors_3d(direct_basis, bravais, 1e-10);
    if neighbors.is_empty() {
        return 0.0;
    }
    
    neighbors.iter()
        .map(|neighbor| neighbor.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}

/// Calculate packing fraction for 2D lattices (assuming circular atoms and touching spheres)
/// TODO: Implement proper packing fraction calculations
pub fn packing_fraction_2d(bravais: &Bravais2D, lattice_parameters: (f64, f64)) -> f64 {
    match bravais {
        Bravais2D::Square => {
            // For square lattice with touching spheres
            let _a = lattice_parameters.0;
            PI / 4.0 // π/4 ≈ 0.785
        },
        Bravais2D::Hexagonal => {
            // For hexagonal lattice (densest 2D packing)
            PI / (2.0 * 3.0_f64.sqrt()) // π/(2√3) ≈ 0.907
        },
        _ => {
            // Default case - needs proper implementation
            0.5
        }
    }
}

/// Calculate packing fraction for 3D lattices (assuming spherical atoms and touching spheres)
/// TODO: Implement proper packing fraction calculations for 3D
pub fn packing_fraction_3d(bravais: &Bravais3D, _lattice_parameters: (f64, f64, f64)) -> f64 {
    match bravais {
        Bravais3D::Cubic(centering) => {
            match centering {
                crate::lattice::bravais_types::Centering::Primitive => {
                    // Simple cubic
                    PI / 6.0 // π/6 ≈ 0.524
                },
                crate::lattice::bravais_types::Centering::BodyCentered => {
                    // Body-centered cubic
                    3.0_f64.sqrt() * PI / 8.0 // √3π/8 ≈ 0.680
                },
                crate::lattice::bravais_types::Centering::FaceCentered => {
                    // Face-centered cubic (densest 3D packing)
                    PI / (3.0 * 2.0_f64.sqrt()) // π/(3√2) ≈ 0.740
                },
                _ => 0.5, // Default
            }
        },
        _ => {
            // Default case - needs proper implementation
            0.5
        }
    }
}

use std::f64::consts::PI;
