pub mod moire2d;
pub mod moire_builder;
pub mod moire_stacking_registries;
pub mod moire_validation; // new - monatomic registry utilities

// Re-export the main types for convenience
pub use moire_builder::{
    WasmMoireBuilder, create_commensurate_moire, create_magic_angle_graphene,
    create_twisted_bilayer,
};
pub use moire_validation::*;
pub use moire2d::WasmMoire2D;
// Re-export stacking registries helpers
pub use moire_stacking_registries::{
    compute_registry_centers_monatomic, compute_registry_centers_monatomic_from_layers,
    compute_registry_centers_monatomic_unwrapped, compute_registry_centers_monatomic_with_l,
    compute_registry_centers_monatomic_with_theta, get_moire_matrix_2x2, get_moire_primitives_2x2,
    get_monatomic_tau_set,
};
