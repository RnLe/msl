// Moiré lattice module: Python wrappers for moiré lattice functionality

pub mod moire2d;
pub mod moire_builder;
pub mod moire_validation_algorithms;
pub mod moire_stacking_registries; // NEW: expose stacking registry helpers

// Re-export main types for convenience
pub use moire2d::{PyMoire2D, PyMoireTransformation};
pub use moire_builder::{PyMoireBuilder, py_twisted_bilayer, py_commensurate_moire};
pub use moire_stacking_registries::{py_registry_centers, py_local_cells_preliminary, py_local_cell_at_point_preliminary};
