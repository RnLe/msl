// Lattice module: Contains Bravais lattice implementations and related functionality
// This module provides comprehensive lattice structures and operations for crystallographic analysis

// ======================== MODULE DECLARATIONS ========================
pub mod bravais_types;
pub mod polyhedron;
pub mod voronoi_cells;
pub mod lattice2d;
pub mod lattice3d;
pub mod coordination_numbers;
pub mod construction;

// Test modules
mod _tests_lattice2d;
mod _tests_lattice3d;
mod _tests_polyhedron;

// ======================== BRAVAIS LATTICE TYPES & CLASSIFICATION ========================
pub use bravais_types::{
    Bravais2D,                      // enum - 2D Bravais lattice types (Square, Hexagonal, Rectangular, CenteredRectangular, Oblique)
    Bravais3D,                      // enum - 3D Bravais lattice types (Cubic, Tetragonal, Orthorhombic, Hexagonal, Trigonal, Monoclinic, Triclinic)
    Centering,                      // enum - lattice centering types (Primitive, Body, Face, Base)
    identify_bravais_2d,            // fn(metric: &Matrix3<f64>, tol: f64) -> Bravais2D - identifies 2D Bravais type from metric tensor
    identify_bravais_3d,            // fn(metric: &Matrix3<f64>, tol: f64) -> Bravais3D - identifies 3D Bravais type from metric tensor
};

// ======================== GEOMETRIC POLYHEDRONS ========================
pub use polyhedron::Polyhedron;    // struct - geometric polyhedron for Wigner-Seitz cells and Brillouin zones
// Polyhedron impl methods:
//   new() -> Self                               - creates empty polyhedron
//   contains_2d(&self, point: Vector3<f64>) -> bool - checks if point is inside 2D polyhedron
//   contains_3d(&self, point: Vector3<f64>) -> bool - checks if point is inside 3D polyhedron  
//   measure(&self) -> f64                       - returns area (2D) or volume (3D)
//   vertices(&self) -> &Vec<Vector3<f64>>       - returns polyhedron vertices
//   edges(&self) -> &Vec<(usize, usize)>        - returns edge connectivity
//   faces(&self) -> &Vec<Vec<usize>>            - returns face vertex indices

// ======================== VORONOI CELL CONSTRUCTION ========================
pub use voronoi_cells::{
    compute_wigner_seitz_cell_2d,        // fn(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron - computes 2D Wigner-Seitz cell
    compute_wigner_seitz_cell_3d,        // fn(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron - computes 3D Wigner-Seitz cell  
    compute_brillouin_zone_2d,           // fn(reciprocal_basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron - computes 2D first Brillouin zone
    compute_brillouin_zone_3d,           // fn(reciprocal_basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron - computes 3D first Brillouin zone
    generate_lattice_points_2d_by_shell, // fn(basis: &Matrix3<f64>, max_shell: usize) -> Vec<Vector3<f64>> - generates 2D neighbor lattice points by shell
    generate_lattice_points_3d_by_shell, // fn(basis: &Matrix3<f64>, max_shell: usize) -> Vec<Vector3<f64>> - generates 3D neighbor lattice points by shell
    generate_lattice_points_2d_within_radius, // fn(basis: &Matrix3<f64>, radius: f64) -> Vec<Vector3<f64>> - generates 2D neighbor points within radius
    generate_lattice_points_3d_within_radius, // fn(basis: &Matrix3<f64>, radius: f64) -> Vec<Vector3<f64>> - generates 3D neighbor points within radius
};

// ======================== 2D LATTICE STRUCTURE ========================
pub use lattice2d::Lattice2D;      // struct - 2D Bravais lattice embedded in 3D space
// Lattice2D impl methods:
//   new(direct: Matrix3<f64>, tol: f64) -> Self                    - constructs 2D lattice from basis vectors
//   frac_to_cart(&self, v_frac: Vector3<f64>) -> Vector3<f64>      - converts fractional to cartesian coordinates
//   cart_to_frac(&self, v_cart: Vector3<f64>) -> Vector3<f64>      - converts cartesian to fractional coordinates
//   lattice_parameters(&self) -> (f64, f64)                       - returns lattice constants a, b
//   lattice_angle(&self) -> f64                                    - returns lattice angle γ in radians
//   to_3d(&self, c_vector: Vector3<f64>) -> Lattice3D             - extends to 3D lattice with c-axis
//   primitive_vectors(&self) -> (Vector3<f64>, Vector3<f64>)       - returns primitive basis vectors a, b
//   in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool        - checks if k-point is in first BZ
//   reduce_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64> - reduces k-point to first BZ
//   get_high_symmetry_points_cartesian(&self) -> Vec<(String, Vector3<f64>)> - gets high symmetry points in cartesian coords
//   generate_k_path(&self, n_points_per_segment: usize) -> Vec<Vector3<f64>> - generates k-point path for band structure
//   bravais_type(&self) -> Bravais2D                               - returns Bravais lattice type
//   cell_area(&self) -> f64                                        - returns unit cell area
//   metric_tensor(&self) -> &Matrix3<f64>                          - returns metric tensor G = A^T * A
//   tolerance(&self) -> f64                                        - returns floating point tolerance
//   symmetry_operations(&self) -> &Vec<SymOp>                      - returns symmetry operations
//   wigner_seitz_cell(&self) -> &Polyhedron                       - returns Wigner-Seitz cell
//   brillouin_zone(&self) -> &Polyhedron                          - returns first Brillouin zone
//   high_symmetry_data(&self) -> &HighSymmetryData                - returns high symmetry points and paths
//   direct_basis(&self) -> &Matrix3<f64>                          - returns direct lattice basis matrix
//   reciprocal_basis(&self) -> &Matrix3<f64>                      - returns reciprocal lattice basis matrix

// ======================== 3D LATTICE STRUCTURE ========================
pub use lattice3d::Lattice3D;      // struct - 3D Bravais lattice
// Lattice3D impl methods:
//   new(direct: Matrix3<f64>, tol: f64) -> Self                    - constructs 3D lattice from basis vectors
//   frac_to_cart(&self, v_frac: Vector3<f64>) -> Vector3<f64>      - converts fractional to cartesian coordinates
//   cart_to_frac(&self, v_cart: Vector3<f64>) -> Vector3<f64>      - converts cartesian to fractional coordinates
//   lattice_parameters(&self) -> (f64, f64, f64)                  - returns lattice constants a, b, c
//   lattice_angles(&self) -> (f64, f64, f64)                      - returns lattice angles α, β, γ in radians
//   to_2d(&self) -> Lattice2D                                     - projects to 2D lattice (a, b vectors)
//   primitive_vectors(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) - returns primitive basis vectors a, b, c
//   in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool        - checks if k-point is in first BZ
//   reduce_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64> - reduces k-point to first BZ
//   get_high_symmetry_points_cartesian(&self) -> Vec<(String, Vector3<f64>)> - gets high symmetry points in cartesian coords
//   generate_k_path(&self, n_points_per_segment: usize) -> Vec<Vector3<f64>> - generates k-point path for band structure
//   bravais_type(&self) -> Bravais3D                               - returns Bravais lattice type
//   cell_volume(&self) -> f64                                      - returns unit cell volume
//   metric_tensor(&self) -> &Matrix3<f64>                          - returns metric tensor G = A^T * A
//   tolerance(&self) -> f64                                        - returns floating point tolerance
//   symmetry_operations(&self) -> &Vec<SymOp>                      - returns symmetry operations
//   wigner_seitz_cell(&self) -> &Polyhedron                       - returns Wigner-Seitz cell
//   brillouin_zone(&self) -> &Polyhedron                          - returns first Brillouin zone
//   high_symmetry_data(&self) -> &HighSymmetryData                - returns high symmetry points and paths
//   direct_basis(&self) -> &Matrix3<f64>                          - returns direct lattice basis matrix
//   reciprocal_basis(&self) -> &Matrix3<f64>                      - returns reciprocal lattice basis matrix

// ======================== COORDINATION ANALYSIS ========================
pub use coordination_numbers::{
    coordination_number_2d,         // fn(lattice: &Lattice2D) -> usize - calculates coordination number for 2D lattice
    coordination_number_3d,         // fn(lattice: &Lattice3D) -> usize - calculates coordination number for 3D lattice
    nearest_neighbors_2d,           // fn(lattice: &Lattice2D) -> Vec<Vector3<f64>> - finds nearest neighbor positions (2D)
    nearest_neighbors_3d,           // fn(lattice: &Lattice3D) -> Vec<Vector3<f64>> - finds nearest neighbor positions (3D)
    nearest_neighbor_distance_2d,   // fn(lattice: &Lattice2D) -> f64 - calculates nearest neighbor distance (2D)
    nearest_neighbor_distance_3d,   // fn(lattice: &Lattice3D) -> f64 - calculates nearest neighbor distance (3D)
    packing_fraction_2d,            // fn(lattice: &Lattice2D, radius: f64) -> f64 - calculates 2D packing fraction
    packing_fraction_3d,            // fn(lattice: &Lattice3D, radius: f64) -> f64 - calculates 3D packing fraction
};

// ======================== LATTICE CONSTRUCTION UTILITIES ========================
pub use construction::{
    // === 2D LATTICE CONSTRUCTORS ===
    square_lattice,                 // fn(a: f64) -> Lattice2D - creates square lattice with parameter a
    rectangular_lattice,            // fn(a: f64, b: f64) -> Lattice2D - creates rectangular lattice with parameters a, b
    hexagonal_lattice,              // fn(a: f64) -> Lattice2D - creates hexagonal lattice with parameter a
    oblique_lattice,                // fn(a: f64, b: f64, gamma: f64) -> Lattice2D - creates oblique lattice with angle γ
    centered_rectangular_lattice,   // fn(a: f64, b: f64) -> Lattice2D - creates centered rectangular lattice
    
    // === 3D LATTICE CONSTRUCTORS ===
    simple_cubic_lattice,           // fn(a: f64) -> Lattice3D - creates simple cubic lattice with parameter a
    body_centered_cubic_lattice,    // fn(a: f64) -> Lattice3D - creates body-centered cubic lattice
    face_centered_cubic_lattice,    // fn(a: f64) -> Lattice3D - creates face-centered cubic lattice
    hexagonal_close_packed_lattice, // fn(a: f64, c: f64) -> Lattice3D - creates hexagonal close-packed lattice
    tetragonal_lattice,             // fn(a: f64, c: f64) -> Lattice3D - creates tetragonal lattice
    orthorhombic_lattice,           // fn(a: f64, b: f64, c: f64) -> Lattice3D - creates orthorhombic lattice
    rhombohedral_lattice,           // fn(a: f64, alpha: f64) -> Lattice3D - creates rhombohedral lattice
    
    // === LATTICE TRANSFORMATIONS ===
    scale_lattice_2d,               // fn(lattice: &Lattice2D, scale: f64) -> Lattice2D - scales 2D lattice uniformly
    scale_lattice_3d,               // fn(lattice: &Lattice3D, scale: f64) -> Lattice3D - scales 3D lattice uniformly
    transform_lattice_2d,           // fn(lattice: &Lattice2D, matrix: &Matrix3<f64>) -> Lattice2D - applies linear transformation
    transform_lattice_3d,           // fn(lattice: &Lattice3D, matrix: &Matrix3<f64>) -> Lattice3D - applies linear transformation
    rotate_lattice_2d,              // fn(lattice: &Lattice2D, angle: f64) -> Lattice2D - rotates 2D lattice by angle
    create_supercell_2d,            // fn(lattice: &Lattice2D, nx: i32, ny: i32) -> Lattice2D - creates 2D supercell
    create_supercell_3d,            // fn(lattice: &Lattice3D, nx: i32, ny: i32, nz: i32) -> Lattice3D - creates 3D supercell
};