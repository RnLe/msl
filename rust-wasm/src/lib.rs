//! # Moire Lattice WASM Bindings
//!
//! WebAssembly bindings for the Moire Lattice library, enabling high-performance
//! lattice calculations and moire physics simulations in web browsers.
//!
//! ## Overview
//!
//! This library provides WASM-compiled bindings for:
//! - 2D Bravais lattice generation and analysis
//! - Moire pattern calculations for twisted bilayer systems
//! - Crystallographic symmetry operations
//! - Voronoi cell and Brillouin zone analysis
//!
//! ## Usage
//!
//! The library is designed to be used from JavaScript/TypeScript via wasm-bindgen.
//! All public types are serializable to/from JavaScript objects.

use wasm_bindgen::prelude::*;

// Re-export console_error_panic_hook for better error messages
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

pub mod config;
pub mod interfaces;
pub mod lattice;
pub mod moire_lattice;
pub mod symmetries;
pub mod utils;

/// Version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Common result type used throughout the library
pub type Result<T> = std::result::Result<T, JsValue>;

/// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::lattice::{Bravais2D, Lattice2D};
    pub use crate::moire_lattice::Moire2D;
    pub use wasm_bindgen::prelude::*;
}
