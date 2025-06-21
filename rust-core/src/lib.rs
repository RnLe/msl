
//! Moire lattice simulation library
//! 
//! This library provides high-performance implementations for working with Bravais lattices,
//! moire patterns, and crystallographic calculations.

pub mod lattice;
pub mod symmetries;

/// Common result type used throughout the library
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
