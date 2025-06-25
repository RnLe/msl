// Symmetries module: Contains symmetry operations and high symmetry points
// This module provides crystallographic symmetry analysis and Brillouin zone navigation

// ======================== MODULE DECLARATIONS ========================
pub mod high_symmetry_points;
pub mod symmetry_operations;
pub mod symmetry_point_groups;

// ======================== HIGH SYMMETRY POINTS & PATHS ========================
pub use high_symmetry_points::{
    SymmetryPointLabel,             // enum - labels for high symmetry points (Gamma, M, K, X, Y, R, L, W, etc.)
    HighSymmetryPoint,              // struct - high symmetry point with label, position, and description
    HighSymmetryPath,               // struct - ordered path through high symmetry points for band structure
    HighSymmetryData,               // struct - collection of points and paths for a lattice type
    generate_2d_high_symmetry_points, // fn(bravais: &Bravais2D) -> HighSymmetryData - generates 2D high symmetry points
    generate_3d_high_symmetry_points, // fn(bravais: &Bravais3D) -> HighSymmetryData - generates 3D high symmetry points
    interpolate_path,               // fn(points: &[HighSymmetryPoint], n_points_per_segment: usize) -> Vec<Vector3<f64>> - interpolates k-path
};

// SymmetryPointLabel impl methods:
//   as_str(&self) -> &str                                          - returns string representation of label

// HighSymmetryPoint impl methods:
//   new(label: SymmetryPointLabel, position: Vector3<f64>, description: impl Into<String>) -> Self - creates new point

// HighSymmetryPath impl methods:
//   new(points: Vec<SymmetryPointLabel>) -> Self                   - creates path from point labels
//   with_n_points(self, n: usize) -> Self                         - sets number of interpolation points

// HighSymmetryData impl methods:
//   new() -> Self                                                  - creates empty high symmetry data
//   add_point(&mut self, point: HighSymmetryPoint)                - adds a high symmetry point
//   set_standard_path(&mut self, path: HighSymmetryPath)          - sets the standard band structure path
//   add_alternative_path(&mut self, name: impl Into<String>, path: HighSymmetryPath) - adds alternative path
//   get_point(&self, label: &SymmetryPointLabel) -> Option<&HighSymmetryPoint> - retrieves point by label
//   get_standard_path_points(&self) -> Vec<&HighSymmetryPoint>    - gets points along standard path

// ======================== SYMMETRY OPERATIONS ========================
pub use symmetry_operations::SymOp; // struct - crystallographic symmetry operation (rotation + translation)
// SymOp impl methods (stubbed for future implementation):
//   new() -> Self                                                  - creates identity symmetry operation
//   apply(&self, point: Vector3<f64>) -> Vector3<f64>             - applies symmetry operation to point
//   inverse(&self) -> Self                                         - returns inverse operation
//   compose(&self, other: &Self) -> Self                          - composes two operations

// ======================== POINT GROUP GENERATORS ========================
pub use symmetry_point_groups::{
    generate_symmetry_operations_2d, // fn(bravais: &Bravais2D) -> Vec<SymOp> - generates 2D symmetry operations for lattice type
    generate_symmetry_operations_3d, // fn(bravais: &Bravais3D) -> Vec<SymOp> - generates 3D symmetry operations for lattice type
    
    // === SPECIFIC CRYSTAL SYSTEM GENERATORS ===
    generate_cubic_operations,       // fn() -> Vec<SymOp> - generates cubic point group operations (Oh, Td, etc.)
    generate_hexagonal_operations,   // fn() -> Vec<SymOp> - generates hexagonal point group operations (D6h, C6v, etc.)
    generate_tetragonal_operations,  // fn() -> Vec<SymOp> - generates tetragonal point group operations (D4h, C4v, etc.)
    generate_orthorhombic_operations, // fn() -> Vec<SymOp> - generates orthorhombic point group operations (D2h, C2v, etc.)
    generate_trigonal_operations,    // fn() -> Vec<SymOp> - generates trigonal point group operations (D3d, C3v, etc.)
    generate_monoclinic_operations,  // fn() -> Vec<SymOp> - generates monoclinic point group operations (C2h, C2, etc.)
    generate_triclinic_operations,   // fn() -> Vec<SymOp> - generates triclinic point group operations (Ci, C1)
};
