/// Utility functions for the moire-lattice Python module
use pyo3::prelude::*;

/// Get the version of the moire-lattice library
#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
