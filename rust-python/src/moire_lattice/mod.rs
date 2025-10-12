// Moiré lattice module: Python wrappers for moiré lattice functionality

pub mod moire2d;
pub mod moire_builder;
pub mod moire_stacking_registries;
pub mod moire_validation_algorithms; // NEW: expose stacking registry helpers

// Re-export main types for convenience
pub use moire_builder::{PyMoireBuilder, py_commensurate_moire, py_twisted_bilayer};
pub use moire_stacking_registries::{
    py_local_cell_at_point_preliminary, py_local_cells_preliminary, py_registry_centers,
};
pub use moire2d::{PyMoire2D, PyMoireTransformation};
