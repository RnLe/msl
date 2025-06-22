/* tslint:disable */
/* eslint-disable */
export function main(): void;
/**
 * Utility functions for creating common lattices
 */
export function create_square_lattice(a: number): WasmLattice2D;
export function create_hexagonal_lattice(a: number): WasmLattice2D;
export function create_rectangular_lattice(a: number, b: number): WasmLattice2D;
/**
 * Get the version of the library
 */
export function version(): string;
/**
 * Identify Bravais lattice type for 2D from metric tensor
 */
export function identify_bravais_type_2d(metric: Float64Array, tolerance: number): WasmBravais2D;
/**
 * Identify Bravais lattice type for 3D from metric tensor
 */
export function identify_bravais_type_3d(metric: Float64Array, tolerance: number): WasmBravais3D;
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
  readonly main: () => void;
  readonly __wbg_wasmlattice2d_free: (a: number, b: number) => void;
  readonly wasmlattice2d_new: (a: any) => [number, number, number];
  readonly wasmlattice2d_generate_points: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice2d_get_parameters: (a: number) => [number, number, number];
  readonly wasmlattice2d_unit_cell_area: (a: number) => number;
  readonly wasmlattice2d_lattice_vectors: (a: number) => [number, number, number];
  readonly wasmlattice2d_reciprocal_vectors: (a: number) => [number, number, number];
  readonly wasmlattice2d_to_svg: (a: number, b: number, c: number, d: number) => [number, number];
  readonly create_square_lattice: (a: number) => [number, number, number];
  readonly create_hexagonal_lattice: (a: number) => [number, number, number];
  readonly create_rectangular_lattice: (a: number, b: number) => [number, number, number];
  readonly version: () => [number, number];
  readonly __wbg_wasmpolyhedron_free: (a: number, b: number) => void;
  readonly wasmpolyhedron_contains_2d: (a: number, b: number, c: number) => number;
  readonly wasmpolyhedron_contains_3d: (a: number, b: number, c: number, d: number) => number;
  readonly wasmpolyhedron_measure: (a: number) => number;
  readonly wasmpolyhedron_get_data: (a: number) => [number, number, number];
  readonly __wbg_wasmlattice3d_free: (a: number, b: number) => void;
  readonly wasmlattice3d_new: (a: any) => [number, number, number];
  readonly wasmlattice3d_frac_to_cart: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice3d_cart_to_frac: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly wasmlattice3d_lattice_parameters: (a: number) => [number, number, number];
  readonly wasmlattice3d_lattice_angles: (a: number) => [number, number, number];
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
  readonly identify_bravais_type_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly identify_bravais_type_3d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_wigner_seitz_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_wigner_seitz_3d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_brillouin_2d: (a: number, b: number, c: number) => [number, number, number];
  readonly compute_brillouin_3d: (a: number, b: number, c: number) => [number, number, number];
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
  readonly wasmlattice3d_cell_volume: (a: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_export_3: WebAssembly.Table;
  readonly __externref_table_dealloc: (a: number) => void;
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
