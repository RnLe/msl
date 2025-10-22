//! Lattice types for WASM bindings
//!
//! Re-exports Bravais lattice types and identification functions

use wasm_bindgen::prelude::*;

// Re-export core types
pub use moire_lattice::lattice::lattice_types::{Bravais2D as CoreBravais2D, identify_bravais_2d};

/// 2D Bravais lattice classification (WASM-compatible)
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bravais2D {
    Square,
    Rectangular,
    CenteredRectangular,
    Hexagonal,
    Oblique,
}

impl From<CoreBravais2D> for Bravais2D {
    fn from(core: CoreBravais2D) -> Self {
        match core {
            CoreBravais2D::Square => Bravais2D::Square,
            CoreBravais2D::Rectangular => Bravais2D::Rectangular,
            CoreBravais2D::CenteredRectangular => Bravais2D::CenteredRectangular,
            CoreBravais2D::Hexagonal => Bravais2D::Hexagonal,
            CoreBravais2D::Oblique => Bravais2D::Oblique,
        }
    }
}

impl From<Bravais2D> for CoreBravais2D {
    fn from(wasm: Bravais2D) -> Self {
        match wasm {
            Bravais2D::Square => CoreBravais2D::Square,
            Bravais2D::Rectangular => CoreBravais2D::Rectangular,
            Bravais2D::CenteredRectangular => CoreBravais2D::CenteredRectangular,
            Bravais2D::Hexagonal => CoreBravais2D::Hexagonal,
            Bravais2D::Oblique => CoreBravais2D::Oblique,
        }
    }
}

// Note: wasm_bindgen doesn't support impl blocks on enums
// Methods must be provided as free functions or on wrapper structs
