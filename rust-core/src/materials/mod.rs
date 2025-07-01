// Materials module: Contains material property definitions for electromagnetic simulations
// This module provides comprehensive material modeling for heterogeneous electromagnetic structures

// ======================== MODULE DECLARATIONS ========================
pub mod material;

// ======================== MATERIAL TYPES & PROPERTIES ========================
pub use material::{
    Material,                      // struct - electromagnetic material with epsilon, mu, and refractive index
    CommonMaterials,               // struct - collection of predefined common materials (Silicon, Silica, etc.)
};

// ======================== RE-EXPORTS FOR CONVENIENCE ========================
// Re-export commonly used external types
pub use num_complex::Complex64;    // Complex number type for material properties
