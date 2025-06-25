use wasm_bindgen::prelude::*;

// Internal modules
mod common;
pub mod lattice;
pub mod moire_lattice;
pub mod symmetries;

// Re-export all public types and functions for backwards compatibility
pub use common::*;
pub use lattice::*;
pub use moire_lattice::*;

// Enable console logging and panic hooks for debugging
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Get the version of the library
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
