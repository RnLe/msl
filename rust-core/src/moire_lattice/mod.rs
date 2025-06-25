// Moiré lattice module: Contains moiré pattern implementations and analysis tools
// This module provides structures and algorithms for 2D moiré lattices

// ======================== MODULE DECLARATIONS ========================
pub mod moire2d;
pub mod moire_builder;
pub mod moire_validation_algorithms;

// ======================== MOIRÉ LATTICE STRUCTURES ========================
pub use moire2d::{
    Moire2D,                        // struct - 2D moiré lattice formed by two overlapping lattices
    MoireTransformation,            // enum - transformation types for creating moiré patterns
};

// Moire2D impl methods:
//   as_lattice2d(&self) -> Lattice2D                              - get moiré lattice as regular Lattice2D
//   primitive_vectors(&self) -> (Vector3<f64>, Vector3<f64>)      - get moiré primitive vectors
//   moire_period_ratio(&self) -> f64                              - ratio of moiré to original lattice constant
//   is_lattice1_point(&self, point: Vector3<f64>) -> bool         - check if point belongs to lattice 1
//   is_lattice2_point(&self, point: Vector3<f64>) -> bool         - check if point belongs to lattice 2
//   get_stacking_at(&self, point: Vector3<f64>) -> Option<String> - get stacking type (AA, A, B) at point

// MoireTransformation variants:
//   RotationScale { angle: f64, scale: f64 }                      - uniform scaling and rotation
//   AnisotropicScale { scale_x: f64, scale_y: f64 }              - different scaling along axes
//   Shear { shear_x: f64, shear_y: f64 }                         - shear transformation
//   General(Matrix2<f64>)                                         - general 2x2 transformation matrix

// MoireTransformation impl methods:
//   to_matrix(&self) -> Matrix2<f64>                             - convert to 2x2 matrix form
//   to_matrix3(&self) -> Matrix3<f64>                            - convert to 3x3 matrix (2D embedded in 3D)

// ======================== MOIRÉ CONSTRUCTION ========================
pub use moire_builder::{
    MoireBuilder,                   // struct - builder pattern for constructing moiré lattices
    twisted_bilayer,                // fn(lattice: Lattice2D, angle: f64) -> Result<Moire2D, String>
    commensurate_moire,            // fn(lattice: Lattice2D, m1: i32, m2: i32, n1: i32, n2: i32) -> Result<Moire2D, String>
};

// MoireBuilder impl methods:
//   new() -> Self                                                 - create new builder
//   with_base_lattice(self, lattice: Lattice2D) -> Self         - set base lattice
//   with_tolerance(self, tol: f64) -> Self                       - set numerical tolerance
//   with_twist_and_scale(self, angle: f64, scale: f64) -> Self  - set rotation and scaling
//   with_anisotropic_scale(self, scale_x: f64, scale_y: f64) -> Self - set anisotropic scaling
//   with_shear(self, shear_x: f64, shear_y: f64) -> Self        - set shear transformation
//   with_general_transformation(self, matrix: Matrix2<f64>) -> Self - set general transformation
//   build(self) -> Result<Moire2D, String>                       - build the moiré lattice

// ======================== VALIDATION & ANALYSIS ========================
pub use moire_validation_algorithms::{
    find_commensurate_angles,       // fn(lattice: &Lattice2D, max_index: i32) -> Result<Vec<(f64, (i32, i32, i32, i32))>, String>
    validate_commensurability,      // fn(lattice_1: &Lattice2D, lattice_2: &Lattice2D, tolerance: f64) -> (bool, Option<(i32, i32, i32, i32)>)
    compute_moire_basis,           // fn(lattice_1: &Lattice2D, lattice_2: &Lattice2D, tolerance: f64) -> Result<Matrix3<f64>, String>
    analyze_moire_symmetry,        // fn(moire: &Moire2D) -> Vec<String> - analyze preserved symmetries
    moire_potential_at,            // fn(moire: &Moire2D, point: Vector3<f64>, v_aa: f64, v_ab: f64) -> f64 - compute effective potential
};
