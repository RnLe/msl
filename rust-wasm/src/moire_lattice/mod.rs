pub mod moire2d;
pub mod moire_builder;
pub mod moire_validation;
pub mod moire_stacking_registries; // new - monatomic registry utilities

// Re-export the main types for convenience
pub use moire2d::WasmMoire2D;
pub use moire_builder::{WasmMoireBuilder, create_twisted_bilayer, create_commensurate_moire, create_magic_angle_graphene};
pub use moire_validation::*;
// Re-export stacking registries helpers
pub use moire_stacking_registries::{
    get_monatomic_tau_set,
    get_moire_matrix_2x2,
    get_moire_primitives_2x2,
    compute_registry_centers_monatomic_from_layers,
    compute_registry_centers_monatomic_with_theta,
    compute_registry_centers_monatomic,
    compute_registry_centers_monatomic_unwrapped,
    compute_registry_centers_monatomic_with_l,
};
