
//! # Moire Lattice
//!
//! A high-performance Rust library for lattice framework calculations, designed for photonic 
//! band structure computations and moire physics simulations.
//!
//! ## Overview
//!
//! This library provides efficient implementations for:
//! - 2D and 3D Bravais lattice generation
//! - Moire pattern calculations for twisted bilayer systems
//! - Crystallographic symmetry operations
//! - Voronoi cell analysis
//! - Foundation for photonic band structure solvers
//!
//! ## Quick Start
//!
//! ```rust
//! use moire_lattice::lattice::{Lattice2D, BravaisType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a hexagonal lattice
//! let lattice = Lattice2D::new(BravaisType::Hexagonal, 1.0, 1.0, 120.0)?;
//!
//! // Generate lattice points within radius
//! let points = lattice.generate_points(5.0)?;
//! println!("Generated {} lattice points", points.len());
//!
//! // Get lattice properties  
//! let area = lattice.unit_cell_area();
//! println!("Unit cell area: {:.4}", area);
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - **High Performance**: Optimized algorithms with optional parallelization
//! - **WebAssembly Ready**: Compile to WASM for browser applications
//! - **MPB Compatible**: Designed as foundation for photonic band structure calculations
//! - **Flexible**: Support for arbitrary lattice types and custom geometries
//!
//! ## Modules
//!
//! - [`lattice`]: Core lattice structures and algorithms
//! - [`moire_lattice`]: Moire pattern calculations and twisted bilayer physics
//! - [`symmetries`]: Crystallographic symmetry operations and point groups

pub mod lattice;
pub mod symmetries;
pub mod moire_lattice;

/// Common result type used throughout the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Re-export commonly used types for convenience
pub mod prelude {
    pub use crate::lattice::{Lattice2D, Lattice3D, Bravais2D, Bravais3D};
    pub use crate::Result;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
