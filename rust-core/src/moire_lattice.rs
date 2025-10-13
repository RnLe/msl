// Moiré lattice module: Contains moiré pattern implementations and analysis tools
// This module provides structures and algorithms for 2D moiré lattices

// ======================== MODULE DECLARATIONS ========================
pub mod moire2d;

// ======================== MOIRÉ LATTICE STRUCTURES ========================
pub use moire2d::{
    Moire2D,             // struct - 2D moiré lattice formed by two overlapping lattices
    MoireTransformation, // enum - transformation types for creating moiré patterns
};
