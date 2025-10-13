//! Lattice module: Bravais lattices, Wigner–Seitz/Brillouin zones, coordination, and utilities.
//!
//! Quick reference
//! - Types: [`Lattice2D`], [`Lattice3D`], [`Bravais2D`], [`Bravais3D`], [`Centering`]
//! - Identification: [`identify_bravais_2d`], [`identify_bravais_3d`]
//! - Construction (2D): [`square_lattice`], [`rectangular_lattice`], [`hexagonal_lattice`], [`oblique_lattice`], [`centered_rectangular_lattice`]
//! - Construction (3D): [`simple_cubic_lattice`], [`body_centered_cubic_lattice`], [`face_centered_cubic_lattice`],
//!   [`tetragonal_lattice`], [`orthorhombic_lattice`], [`rhombohedral_lattice`], [`hexagonal_close_packed_lattice`]
//! - Transforms: [`scale_lattice_2d`], [`scale_lattice_3d`], [`rotate_lattice_2d`], [`transform_lattice_2d`], [`transform_lattice_3d`]
//! - Voronoi/WS/BZ: [`compute_wigner_seitz_cell_2d`], [`compute_wigner_seitz_cell_3d`], [`compute_brillouin_zone_2d`], [`compute_brillouin_zone_3d`]
//! - Coordination: [`coordination_number_2d`], [`coordination_number_3d`], [`nearest_neighbor_distance_2d`], [`nearest_neighbor_distance_3d`]
//!
//! See submodules for full details: [`lattice2d`], [`lattice3d`], [`lattice_construction`], [`voronoi_cells`], [`lattice_coordination_numbers`], [`lattice_types`].

// ======================== MODULE DECLARATIONS ========================
pub mod base_matrix;
pub mod lattice2d;
pub mod lattice_construction;
pub mod lattice_coordination_numbers;
pub mod lattice_like_2d;
pub mod lattice_types;
pub mod lattice_validations;
pub mod polyhedron;
pub mod voronoi_cells;

// ======================== RE-EXPORTED PUBLIC API (curated) ========================
#[doc(inline)]
pub use lattice2d::Lattice2D;

#[doc(inline)]
pub use polyhedron::Polyhedron;

pub use lattice_types::{
    Bravais2D, Bravais3D, Centering, identify_bravais_2d, identify_bravais_3d,
};

pub use voronoi_cells::{
    compute_brillouin_zone_2d, compute_brillouin_zone_3d, compute_wigner_seitz_cell_2d,
    compute_wigner_seitz_cell_3d, generate_lattice_points_2d_by_shell,
    generate_lattice_points_2d_within_radius, generate_lattice_points_3d_by_shell,
    generate_lattice_points_3d_within_radius,
};

pub use lattice_construction::{
    centered_rectangular_lattice, create_supercell_2d, hexagonal_lattice, oblique_lattice,
    rectangular_lattice, rotate_lattice_2d, scale_lattice_2d, square_lattice, transform_lattice_2d,
};

pub use lattice_validations::{
    analyze_bravais_type_2d, determine_bravais_type_2d, validate_bravais_type_2d,
};

// ======================== COORDINATION ANALYSIS ========================
pub use lattice_coordination_numbers::{
    coordination_number_2d, coordination_number_3d, nearest_neighbor_distance_2d,
    nearest_neighbor_distance_3d, nearest_neighbors_2d, nearest_neighbors_3d, packing_fraction_2d,
    packing_fraction_3d,
};

/// A convenience prelude for importing common lattice items.
pub mod prelude {
    #[doc(no_inline)]
    // Time will tell what belongs into the prelude; keep it lean
    pub use super::{Bravais2D, Lattice2D};
}
