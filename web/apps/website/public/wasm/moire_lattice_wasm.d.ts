/* tslint:disable */
/* eslint-disable */
/**
 * Find commensurate angles for a given lattice
 */
export function find_commensurate_angles_wasm(lattice: WasmLattice2D, max_index: number): any;
/**
 * Validate commensurability between two lattices
 */
export function validate_commensurability_wasm(lattice_1: WasmLattice2D, lattice_2: WasmLattice2D, tolerance: number): any;
/**
 * Compute moiré basis vectors from two lattices
 */
export function compute_moire_basis_wasm(lattice_1: WasmLattice2D, lattice_2: WasmLattice2D, tolerance: number): Float64Array;
/**
 * Analyze symmetries preserved in the moiré pattern
 */
export function analyze_moire_symmetry_wasm(moire: WasmMoire2D): string[];
/**
 * Compute moiré potential at a given point
 */
export function moire_potential_at_wasm(moire: WasmMoire2D, x: number, y: number, v_aa: number, v_ab: number): number;
/**
 * Compute moiré potential over a grid for visualization
 */
export function compute_moire_potential_grid(moire: WasmMoire2D, x_min: number, x_max: number, y_min: number, y_max: number, nx: number, ny: number, v_aa: number, v_ab: number): any;
/**
 * Find magic angles (special commensurate angles with interesting properties)
 */
export function find_magic_angles(lattice: WasmLattice2D): any;
/**
 * Analyze the quality of a moiré pattern based on commensurability and period
 */
export function analyze_moire_quality(moire: WasmMoire2D): any;
/**
 * Get theoretical predictions for moiré properties
 */
export function get_moire_predictions(lattice_constant: number, twist_angle_degrees: number): any;
/**
 * Compute Wigner-Seitz cell for 2D lattice
 */
export function compute_wigner_seitz_2d(basis: Float64Array, tolerance: number): WasmPolyhedron;
/**
 * Compute Wigner-Seitz cell for 3D lattice
 */
export function compute_wigner_seitz_3d(basis: Float64Array, tolerance: number): WasmPolyhedron;
/**
 * Compute Brillouin zone for 2D lattice
 */
export function compute_brillouin_2d(reciprocal_basis: Float64Array, tolerance: number): WasmPolyhedron;
/**
 * Compute Brillouin zone for 3D lattice
 */
export function compute_brillouin_3d(reciprocal_basis: Float64Array, tolerance: number): WasmPolyhedron;
export function main(): void;
/**
 * Get the version of the library
 */
export function version(): string;
export function get_monatomic_tau_set(moire: WasmMoire2D): any;
export function get_moire_matrix_2x2(moire: WasmMoire2D): Float64Array;
export function get_moire_primitives_2x2(moire: WasmMoire2D): Float64Array;
export function compute_registry_centers_monatomic(moire: WasmMoire2D, d0x: number, d0y: number): any;
/**
 * Unwrapped registry centers: returns r_τ = M^{-1}(τ - d0) without wrapping into the moiré cell.
 * Use this to ensure continuity across twist-angle changes and perform wrapping/tiling in the UI.
 */
export function compute_registry_centers_monatomic_unwrapped(moire: WasmMoire2D, d0x: number, d0y: number): any;
export function compute_registry_centers_monatomic_from_layers(moire: WasmMoire2D, d0x: number, d0y: number): any;
export function compute_registry_centers_monatomic_with_theta(moire: WasmMoire2D, d0x: number, d0y: number): any;
export function compute_registry_centers_monatomic_with_l(moire: WasmMoire2D, d0x: number, d0y: number): any;
/**
 * Create a simple twisted bilayer moiré pattern
 */
export function create_twisted_bilayer(lattice: WasmLattice2D, angle_degrees: number): WasmMoire2D;
/**
 * Create a moiré pattern with commensurate angle
 */
export function create_commensurate_moire(lattice: WasmLattice2D, m1: number, m2: number, n1: number, n2: number): WasmMoire2D;
/**
 * Create twisted bilayer graphene moiré pattern with magic angle
 */
export function create_magic_angle_graphene(lattice: WasmLattice2D): WasmMoire2D;
/**
 * Create a series of moiré patterns with different twist angles
 */
export function create_twist_series(lattice: WasmLattice2D, start_angle: number, end_angle: number, num_steps: number): WasmMoire2D[];
/**
 * Get recommended twist angles for studying moiré patterns
 */
export function get_recommended_twist_angles(): Float64Array;
/**
 * Calculate the expected moiré period for a given twist angle and lattice constant
 */
export function calculate_moire_period(lattice_constant: number, twist_angle_degrees: number): number;
/**
 * Get transformation matrix for rotation and scaling
 */
export function get_rotation_scale_matrix(angle_degrees: number, scale: number): Float64Array;
/**
 * Get transformation matrix for anisotropic scaling
 */
export function get_anisotropic_scale_matrix(scale_x: number, scale_y: number): Float64Array;
/**
 * Get transformation matrix for shear
 */
export function get_shear_matrix(shear_x: number, shear_y: number): Float64Array;
/**
 * Identify Bravais lattice type for 2D from metric tensor
 */
export function identify_bravais_type_2d(metric: Float64Array, tolerance: number): WasmBravais2D;
/**
 * Identify Bravais lattice type for 3D from metric tensor
 */
export function identify_bravais_type_3d(metric: Float64Array, tolerance: number): WasmBravais3D;
/**
 * Create square lattice
 */
export function create_square_lattice(a: number): WasmLattice2D;
/**
 * Create hexagonal lattice
 */
export function create_hexagonal_lattice(a: number): WasmLattice2D;
/**
 * Create rectangular lattice
 */
export function create_rectangular_lattice(a: number, b: number): WasmLattice2D;
/**
 * Create centered rectangular lattice
 */
export function create_centered_rectangular_lattice(a: number, b: number): WasmLattice2D;
/**
 * Create oblique lattice
 */
export function create_oblique_lattice(a: number, b: number, gamma_degrees: number): WasmLattice2D;
/**
 * Create body-centered cubic lattice
 */
export function create_body_centered_cubic_lattice(a: number): WasmLattice3D;
/**
 * Create face-centered cubic lattice
 */
export function create_face_centered_cubic_lattice(a: number): WasmLattice3D;
/**
 * Create hexagonal close-packed lattice
 */
export function create_hexagonal_close_packed_lattice(a: number, c: number): WasmLattice3D;
/**
 * Create tetragonal lattice
 */
export function create_tetragonal_lattice(a: number, c: number): WasmLattice3D;
/**
 * Create orthorhombic lattice
 */
export function create_orthorhombic_lattice(a: number, b: number, c: number): WasmLattice3D;
/**
 * Create rhombohedral lattice
 */
export function create_rhombohedral_lattice(a: number, alpha_degrees: number): WasmLattice3D;
/**
 * Scale 2D lattice uniformly
 */
export function scale_2d_lattice(lattice: WasmLattice2D, scale: number): WasmLattice2D;
/**
 * Scale 3D lattice uniformly
 */
export function scale_3d_lattice(lattice: WasmLattice3D, scale: number): WasmLattice3D;
/**
 * Rotate 2D lattice by angle in degrees
 */
export function rotate_2d_lattice(lattice: WasmLattice2D, angle_degrees: number): WasmLattice2D;
/**
 * Create 2D supercell
 */
export function create_2d_supercell(lattice: WasmLattice2D, nx: number, ny: number): WasmLattice2D;
/**
 * Create 3D supercell
 */
export function create_3d_supercell(lattice: WasmLattice3D, nx: number, ny: number, nz: number): WasmLattice3D;
/**
 * Determine the Bravais lattice type from a 2D lattice
 */
export function determine_lattice_type_2d(lattice: WasmLattice2D): WasmBravais2D;
/**
 * Validate that a 2D lattice's stored Bravais type matches its actual structure
 */
export function validate_lattice_type_2d(lattice: WasmLattice2D): boolean;
/**
 * Get detailed analysis of why a lattice has a particular Bravais type
 */
export function analyze_lattice_type_2d(lattice: WasmLattice2D): any;
/**
 * Get the string representation of a Bravais type
 */
export function bravais_type_to_string(bravais_type: WasmBravais2D): string;
/**
 * Compare two Bravais types for equality
 */
export function bravais_types_equal(a: WasmBravais2D, b: WasmBravais2D): boolean;
/**
 * Check if two floating point values are approximately equal (exposed for testing)
 */
export function approx_equal_wasm(a: number, b: number, tolerance: number): boolean;
/**
 * Get lattice parameters from a 2D lattice (exposed for convenience)
 */
export function get_lattice_parameters_2d(lattice: WasmLattice2D): any;
/**
 * Validate multiple lattices at once (for efficiency)
 */
export function validate_multiple_lattices_2d(lattices_data: any): any;
/**
 * Analyze multiple lattices at once (for efficiency)
 */
export function analyze_multiple_lattices_2d(lattices_data: any): any;
/**
 * Determine Bravais types for multiple lattices at once (for efficiency)
 */
export function determine_multiple_lattice_types_2d(lattices_data: any): any;
/**
 * Check if an angle is equivalent to 90 degrees (considering crystallographic equivalences)
 * Useful for testing and validation in JavaScript
 */
export function is_angle_equivalent_to_90_degrees(angle_radians: number, tolerance: number): boolean;
/**
 * Check if an angle is equivalent to hexagonal angles (considering crystallographic equivalences)
 * Useful for testing and validation in JavaScript
 */
export function is_angle_equivalent_to_hexagonal(angle_radians: number, tolerance: number): boolean;
/**
 * Convert degrees to radians (convenience function for JavaScript)
 */
export function degrees_to_radians(degrees: number): number;
/**
 * Convert radians to degrees (convenience function for JavaScript)
 */
export function radians_to_degrees(radians: number): number;
/**
 * WASM wrapper for Bravais2D enum
 */
export enum WasmBravais2D {
  Square = 0,
  Hexagonal = 1,
  Rectangular = 2,
  CenteredRectangular = 3,
  Oblique = 4,
}
/**
 * WASM wrapper for Bravais3D enum
 */
export enum WasmBravais3D {
  Cubic = 0,
  Tetragonal = 1,
  Orthorhombic = 2,
  Hexagonal = 3,
  Trigonal = 4,
  Monoclinic = 5,
  Triclinic = 6,
}
/**
 * WASM wrapper for Centering enum
 */
export enum WasmCentering {
  Primitive = 0,
  BodyCentered = 1,
  FaceCentered = 2,
  BaseCentered = 3,
}
/**
 * WASM wrapper for MoireTransformation enum
 */
export enum WasmMoireTransformation {
  RotationScale = 0,
  AnisotropicScale = 1,
  Shear = 2,
  General = 3,
}
/**
 * WASM wrapper for 2D lattice
 */
export class WasmLattice2D {
  free(): void;
  /**
   * Create a new lattice from JavaScript parameters
   */
  constructor(params: any);
  /**
   * Generate lattice points within a radius
   */
  generate_points(radius: number, center_x: number, center_y: number): any;
  /**
   * Get lattice parameters as JavaScript object
   */
  get_parameters(): any;
  /**
   * Get unit cell area
   */
  unit_cell_area(): number;
  /**
   * Get lattice vectors as JavaScript object
   */
  lattice_vectors(): any;
  /**
   * Get reciprocal lattice vectors
   */
  reciprocal_vectors(): any;
  /**
   * Generate an SVG representation of the lattice
   */
  to_svg(width: number, height: number, radius: number): string;
  /**
   * Convert fractional to cartesian coordinates
   */
  frac_to_cart(fx: number, fy: number): any;
  /**
   * Convert cartesian to fractional coordinates
   */
  cart_to_frac(x: number, y: number): any;
  /**
   * Get Bravais lattice type
   */
  bravais_type(): WasmBravais2D;
  /**
   * Check if k-point is in Brillouin zone
   */
  in_brillouin_zone(kx: number, ky: number): boolean;
  /**
   * Reduce k-point to first Brillouin zone
   */
  reduce_to_brillouin_zone(kx: number, ky: number): any;
  /**
   * Get Wigner-Seitz cell
   */
  wigner_seitz_cell(): WasmPolyhedron;
  /**
   * Get Brillouin zone
   */
  brillouin_zone(): WasmPolyhedron;
  /**
   * Get coordination analysis
   */
  coordination_analysis(): any;
  /**
   * Get packing fraction for given atomic radius
   */
  packing_fraction(_radius: number): number;
  /**
   * Extend to 3D lattice with given c-vector
   */
  to_3d(cx: number, cy: number, cz: number): WasmLattice3D;
  /**
   * Generate lattice points by shell
   */
  generate_points_by_shell(max_shell: number): any;
  /**
   * Generate direct-space lattice points in a rectangle
   */
  get_direct_lattice_points_in_rectangle(width: number, height: number): any;
  /**
   * Generate reciprocal-space lattice points in a rectangle
   */
  get_reciprocal_lattice_points_in_rectangle(width: number, height: number): any;
  /**
   * Get high symmetry points in Cartesian coordinates
   */
  get_high_symmetry_points(): any;
  /**
   * Get high symmetry path data
   */
  get_high_symmetry_path(): any;
}
/**
 * WASM wrapper for 3D lattice
 */
export class WasmLattice3D {
  free(): void;
  /**
   * Create a new 3D lattice from JavaScript parameters
   */
  constructor(params: any);
  /**
   * Convert fractional to cartesian coordinates
   */
  frac_to_cart(fx: number, fy: number, fz: number): any;
  /**
   * Convert cartesian to fractional coordinates
   */
  cart_to_frac(x: number, y: number, z: number): any;
  /**
   * Get lattice parameters
   */
  lattice_parameters(): any;
  /**
   * Get lattice angles in degrees
   */
  lattice_angles(): any;
  /**
   * Get cell volume
   */
  cell_volume(): number;
  /**
   * Get Bravais lattice type
   */
  bravais_type(): WasmBravais3D;
  /**
   * Check if k-point is in Brillouin zone
   */
  in_brillouin_zone(kx: number, ky: number, kz: number): boolean;
  /**
   * Reduce k-point to first Brillouin zone
   */
  reduce_to_brillouin_zone(kx: number, ky: number, kz: number): any;
  /**
   * Generate 3D lattice points within radius
   */
  generate_points_3d(radius: number): any;
  /**
   * Generate 3D lattice points by shell
   */
  generate_points_3d_by_shell(max_shell: number): any;
  /**
   * Get Wigner-Seitz cell
   */
  wigner_seitz_cell(): WasmPolyhedron;
  /**
   * Get Brillouin zone
   */
  brillouin_zone(): WasmPolyhedron;
  /**
   * Get coordination analysis
   */
  coordination_analysis(): any;
  /**
   * Get packing fraction for given atomic radius
   */
  packing_fraction(_radius: number): number;
  /**
   * Convert to 2D lattice (projection onto a-b plane)
   */
  to_2d(): WasmLattice2D;
  /**
   * Get high symmetry points in Cartesian coordinates
   */
  get_high_symmetry_points(): any;
  /**
   * Get high symmetry path data
   */
  get_high_symmetry_path(): any;
}
/**
 * WASM wrapper for 2D moiré lattice
 */
export class WasmMoire2D {
  private constructor();
  free(): void;
  /**
   * Get the moiré lattice as a regular 2D lattice
   */
  as_lattice2d(): WasmLattice2D;
  /**
   * Get moiré primitive vectors as JavaScript object
   */
  primitive_vectors(): any;
  /**
   * Get the moiré periodicity ratio
   */
  moire_period_ratio(): number;
  /**
   * Check if a point belongs to lattice 1
   */
  is_lattice1_point(x: number, y: number): boolean;
  /**
   * Check if a point belongs to lattice 2
   */
  is_lattice2_point(x: number, y: number): boolean;
  /**
   * Get stacking type at a given position
   */
  get_stacking_at(x: number, y: number): string | undefined;
  /**
   * Get the twist angle in degrees
   */
  twist_angle_degrees(): number;
  /**
   * Get the twist angle in radians
   */
  twist_angle_radians(): number;
  /**
   * Check if the moiré lattice is commensurate
   */
  is_commensurate(): boolean;
  /**
   * Get coincidence indices if commensurate
   */
  coincidence_indices(): Int32Array | undefined;
  /**
   * Get the first constituent lattice
   */
  lattice_1(): WasmLattice2D;
  /**
   * Get the second constituent lattice
   */
  lattice_2(): WasmLattice2D;
  /**
   * Get unit cell area of the moiré lattice
   */
  cell_area(): number;
  /**
   * Get transformation matrix as JavaScript array (flattened 2x2 matrix)
   */
  transformation_matrix(): Float64Array;
  /**
   * Get lattice parameters as JavaScript object
   */
  get_parameters(): any;
  /**
   * Generate lattice points within a radius for visualization
   */
  generate_moire_points(radius: number): any;
  /**
   * Generate lattice 1 points within a radius
   */
  generate_lattice1_points(radius: number): any;
  /**
   * Generate lattice 2 points within a radius
   */
  generate_lattice2_points(radius: number): any;
  /**
   * Get stacking analysis for points within a radius
   */
  analyze_stacking_in_region(radius: number, grid_spacing: number): any;
  /**
   * Convert fractional to cartesian coordinates using moiré basis
   */
  frac_to_cart(fx: number, fy: number): any;
  /**
   * Convert cartesian to fractional coordinates using moiré basis
   */
  cart_to_frac(x: number, y: number): any;
}
/**
 * WASM wrapper for MoireBuilder
 */
export class WasmMoireBuilder {
  free(): void;
  /**
   * Create a new MoireBuilder
   */
  constructor();
  /**
   * Set the base lattice
   */
  with_base_lattice(lattice: WasmLattice2D): WasmMoireBuilder;
  /**
   * Set tolerance for calculations
   */
  with_tolerance(tolerance: number): WasmMoireBuilder;
  /**
   * Set a rotation and uniform scaling transformation
   */
  with_twist_and_scale(angle_degrees: number, scale: number): WasmMoireBuilder;
  /**
   * Set an anisotropic scaling transformation
   */
  with_anisotropic_scale(scale_x: number, scale_y: number): WasmMoireBuilder;
  /**
   * Set a shear transformation
   */
  with_shear(shear_x: number, shear_y: number): WasmMoireBuilder;
  /**
   * Set a general 2x2 transformation matrix (flattened array)
   */
  with_general_transformation(matrix: Float64Array): WasmMoireBuilder;
  /**
   * Build the Moire2D lattice
   */
  build(): WasmMoire2D;
  /**
   * Build with JavaScript parameters object
   */
  static build_with_params(lattice: WasmLattice2D, params: any): WasmMoire2D;
}
/**
 * WASM wrapper for Polyhedron
 */
export class WasmPolyhedron {
  private constructor();
  free(): void;
  /**
   * Check if a 2D point is inside the polyhedron
   */
  contains_2d(x: number, y: number): boolean;
  /**
   * Check if a 3D point is inside the polyhedron
   */
  contains_3d(x: number, y: number, z: number): boolean;
  /**
   * Get the measure (area for 2D, volume for 3D)
   */
  measure(): number;
  /**
   * Get polyhedron data as JavaScript object
   */
  get_data(): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmlattice2d_free: (a: number, b: number) => void;
  readonly wasmlattice2d_new: (a: any) => [number, number, number];
  readonly wasmlattice2d_generate_points: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice2d_get_parameters: (a: number) => [number, number, number];
  readonly wasmlattice2d_unit_cell_area: (a: number) => number;
  readonly wasmlattice2d_lattice_vectors: (a: number) => [number, number, number];
  readonly wasmlattice2d_reciprocal_vectors: (a: number) => [number, number, number];
  readonly wasmlattice2d_to_svg: (a: number, b: number, c: number, d: number) => [number, number];
  readonly wasmlattice2d_frac_to_cart: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmlattice2d_cart_to_frac: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmlattice2d_bravais_type: (a: number) => number;
  readonly wasmlattice2d_in_brillouin_zone: (a: number, b: number, c: number) => number;
  readonly wasmlattice2d_reduce_to_brillouin_zone: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmlattice2d_wigner_seitz_cell: (a: number) => number;
  readonly wasmlattice2d_brillouin_zone: (a: number) => number;
  readonly wasmlattice2d_coordination_analysis: (a: number) => [number, number, number];
  readonly wasmlattice2d_packing_fraction: (a: number, b: number) => number;
  readonly wasmlattice2d_to_3d: (a: number, b: number, c: number, d: number) => number;
  readonly wasmlattice2d_generate_points_by_shell: (a: number, b: number) => [number, number, number];
  readonly wasmlattice2d_get_direct_lattice_points_in_rectangle: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmlattice2d_get_reciprocal_lattice_points_in_rectangle: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmlattice2d_get_high_symmetry_points: (a: number) => [number, number, number];
  readonly wasmlattice2d_get_high_symmetry_path: (a: number) => [number, number, number];
  readonly __wbg_wasmlattice3d_free: (a: number, b: number) => void;
  readonly wasmlattice3d_new: (a: any) => [number, number, number];
  readonly wasmlattice3d_frac_to_cart: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice3d_cart_to_frac: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice3d_lattice_parameters: (a: number) => [number, number, number];
  readonly wasmlattice3d_lattice_angles: (a: number) => [number, number, number];
  readonly wasmlattice3d_cell_volume: (a: number) => number;
  readonly wasmlattice3d_bravais_type: (a: number) => number;
  readonly wasmlattice3d_in_brillouin_zone: (a: number, b: number, c: number, d: number) => number;
  readonly wasmlattice3d_reduce_to_brillouin_zone: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice3d_generate_points_3d: (a: number, b: number) => [number, number, number];
  readonly wasmlattice3d_generate_points_3d_by_shell: (a: number, b: number) => [number, number, number];
  readonly wasmlattice3d_wigner_seitz_cell: (a: number) => number;
  readonly wasmlattice3d_brillouin_zone: (a: number) => number;
  readonly wasmlattice3d_coordination_analysis: (a: number) => [number, number, number];
  readonly wasmlattice3d_packing_fraction: (a: number, b: number) => number;
  readonly wasmlattice3d_to_2d: (a: number) => number;
  readonly wasmlattice3d_get_high_symmetry_points: (a: number) => [number, number, number];
  readonly wasmlattice3d_get_high_symmetry_path: (a: number) => [number, number, number];
  readonly find_commensurate_angles_wasm: (a: number, b: number) => [number, number, number];
  readonly validate_commensurability_wasm: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_moire_basis_wasm: (a: number, b: number, c: number) => [number, number, number, number];
  readonly analyze_moire_symmetry_wasm: (a: number) => [number, number];
  readonly moire_potential_at_wasm: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly compute_moire_potential_grid: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
  readonly find_magic_angles: (a: number) => [number, number, number];
  readonly analyze_moire_quality: (a: number) => [number, number, number];
  readonly get_moire_predictions: (a: number, b: number) => [number, number, number];
  readonly __wbg_wasmpolyhedron_free: (a: number, b: number) => void;
  readonly wasmpolyhedron_contains_2d: (a: number, b: number, c: number) => number;
  readonly wasmpolyhedron_contains_3d: (a: number, b: number, c: number, d: number) => number;
  readonly wasmpolyhedron_measure: (a: number) => number;
  readonly wasmpolyhedron_get_data: (a: number) => [number, number, number];
  readonly compute_wigner_seitz_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_wigner_seitz_3d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_brillouin_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_brillouin_3d: (a: number, b: number, c: number) => [number, number, number];
  readonly main: () => void;
  readonly version: () => [number, number];
  readonly get_monatomic_tau_set: (a: number) => [number, number, number];
  readonly get_moire_matrix_2x2: (a: number) => [number, number, number, number];
  readonly get_moire_primitives_2x2: (a: number) => [number, number, number, number];
  readonly compute_registry_centers_monatomic: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_registry_centers_monatomic_unwrapped: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_registry_centers_monatomic_from_layers: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_registry_centers_monatomic_with_theta: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_registry_centers_monatomic_with_l: (a: number, b: number, c: number) => [number, number, number];
  readonly __wbg_wasmmoire2d_free: (a: number, b: number) => void;
  readonly wasmmoire2d_as_lattice2d: (a: number) => number;
  readonly wasmmoire2d_primitive_vectors: (a: number) => [number, number, number];
  readonly wasmmoire2d_moire_period_ratio: (a: number) => number;
  readonly wasmmoire2d_is_lattice1_point: (a: number, b: number, c: number) => number;
  readonly wasmmoire2d_is_lattice2_point: (a: number, b: number, c: number) => number;
  readonly wasmmoire2d_get_stacking_at: (a: number, b: number, c: number) => [number, number];
  readonly wasmmoire2d_twist_angle_degrees: (a: number) => number;
  readonly wasmmoire2d_twist_angle_radians: (a: number) => number;
  readonly wasmmoire2d_is_commensurate: (a: number) => number;
  readonly wasmmoire2d_coincidence_indices: (a: number) => [number, number];
  readonly wasmmoire2d_lattice_1: (a: number) => number;
  readonly wasmmoire2d_lattice_2: (a: number) => number;
  readonly wasmmoire2d_cell_area: (a: number) => number;
  readonly wasmmoire2d_transformation_matrix: (a: number) => [number, number];
  readonly wasmmoire2d_get_parameters: (a: number) => [number, number, number];
  readonly wasmmoire2d_generate_moire_points: (a: number, b: number) => [number, number, number];
  readonly wasmmoire2d_generate_lattice1_points: (a: number, b: number) => [number, number, number];
  readonly wasmmoire2d_generate_lattice2_points: (a: number, b: number) => [number, number, number];
  readonly wasmmoire2d_analyze_stacking_in_region: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmmoire2d_frac_to_cart: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmmoire2d_cart_to_frac: (a: number, b: number, c: number) => [number, number, number];
  readonly __wbg_wasmmoirebuilder_free: (a: number, b: number) => void;
  readonly wasmmoirebuilder_new: () => number;
  readonly wasmmoirebuilder_with_base_lattice: (a: number, b: number) => number;
  readonly wasmmoirebuilder_with_tolerance: (a: number, b: number) => number;
  readonly wasmmoirebuilder_with_twist_and_scale: (a: number, b: number, c: number) => number;
  readonly wasmmoirebuilder_with_anisotropic_scale: (a: number, b: number, c: number) => number;
  readonly wasmmoirebuilder_with_shear: (a: number, b: number, c: number) => number;
  readonly wasmmoirebuilder_with_general_transformation: (a: number, b: number, c: number) => [number, number, number];
  readonly wasmmoirebuilder_build: (a: number) => [number, number, number];
  readonly wasmmoirebuilder_build_with_params: (a: number, b: any) => [number, number, number];
  readonly create_twisted_bilayer: (a: number, b: number) => [number, number, number];
  readonly create_commensurate_moire: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
  readonly create_magic_angle_graphene: (a: number) => [number, number, number];
  readonly create_twist_series: (a: number, b: number, c: number, d: number) => [number, number, number, number];
  readonly get_recommended_twist_angles: () => [number, number];
  readonly calculate_moire_period: (a: number, b: number) => number;
  readonly get_rotation_scale_matrix: (a: number, b: number) => [number, number];
  readonly get_anisotropic_scale_matrix: (a: number, b: number) => [number, number];
  readonly get_shear_matrix: (a: number, b: number) => [number, number];
  readonly identify_bravais_type_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly identify_bravais_type_3d: (a: number, b: number, c: number) => [number, number, number];
  readonly create_square_lattice: (a: number) => [number, number, number];
  readonly create_hexagonal_lattice: (a: number) => [number, number, number];
  readonly create_rectangular_lattice: (a: number, b: number) => [number, number, number];
  readonly create_centered_rectangular_lattice: (a: number, b: number) => [number, number, number];
  readonly create_oblique_lattice: (a: number, b: number, c: number) => [number, number, number];
  readonly create_body_centered_cubic_lattice: (a: number) => [number, number, number];
  readonly create_face_centered_cubic_lattice: (a: number) => [number, number, number];
  readonly create_hexagonal_close_packed_lattice: (a: number, b: number) => [number, number, number];
  readonly create_tetragonal_lattice: (a: number, b: number) => [number, number, number];
  readonly create_orthorhombic_lattice: (a: number, b: number, c: number) => [number, number, number];
  readonly create_rhombohedral_lattice: (a: number, b: number) => [number, number, number];
  readonly scale_2d_lattice: (a: number, b: number) => number;
  readonly scale_3d_lattice: (a: number, b: number) => number;
  readonly rotate_2d_lattice: (a: number, b: number) => number;
  readonly create_2d_supercell: (a: number, b: number, c: number) => number;
  readonly create_3d_supercell: (a: number, b: number, c: number, d: number) => number;
  readonly determine_lattice_type_2d: (a: number) => number;
  readonly validate_lattice_type_2d: (a: number) => number;
  readonly analyze_lattice_type_2d: (a: number) => [number, number, number];
  readonly bravais_type_to_string: (a: number) => [number, number];
  readonly bravais_types_equal: (a: number, b: number) => number;
  readonly approx_equal_wasm: (a: number, b: number, c: number) => number;
  readonly get_lattice_parameters_2d: (a: number) => [number, number, number];
  readonly validate_multiple_lattices_2d: (a: any) => [number, number, number];
  readonly analyze_multiple_lattices_2d: (a: any) => [number, number, number];
  readonly determine_multiple_lattice_types_2d: (a: any) => [number, number, number];
  readonly is_angle_equivalent_to_90_degrees: (a: number, b: number) => number;
  readonly is_angle_equivalent_to_hexagonal: (a: number, b: number) => number;
  readonly degrees_to_radians: (a: number) => number;
  readonly radians_to_degrees: (a: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_4: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __externref_drop_slice: (a: number, b: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
