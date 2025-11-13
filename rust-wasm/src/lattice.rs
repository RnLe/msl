//! Lattice module: Bravais lattices, Wigner–Seitz/Brillouin zones, coordination, and utilities.
//!
//! WASM bindings for 2D lattice structures and operations.

// ======================== MODULE DECLARATIONS ========================
pub mod base_matrix;
pub mod lattice2d;
pub mod lattice_algorithms;
pub mod lattice_construction;
pub mod lattice_like_2d;
pub mod lattice_types;
pub mod polyhedron;
pub mod voronoi_cells;

// ======================== RE-EXPORTED PUBLIC API ========================
#[doc(inline)]
pub use lattice2d::Lattice2D;

#[doc(inline)]
pub use polyhedron::Polyhedron;

pub use lattice_types::{Bravais2D, identify_bravais_2d};

pub use voronoi_cells::{
    compute_brillouin_zone_2d, compute_wigner_seitz_cell_2d, generate_lattice_points_2d_by_shell,
    generate_lattice_points_2d_within_radius,
};

pub use lattice_construction::{
    centered_rectangular_lattice, hexagonal_lattice, oblique_lattice, rectangular_lattice,
    rotate_lattice_2d, scale_lattice_2d, square_lattice, transform_lattice_2d,
};

pub use lattice_algorithms::lattice_points_in_rectangle;
