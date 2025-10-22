/* tslint:disable */
/* eslint-disable */
/**
 * Helper function to create a moiré lattice from a base lattice and transformation
 * This is an alternative to using the Moire2D constructor directly
 */
export function createMoireLattice(base_lattice: Lattice2D, transformation: MoireTransformation): Moire2D;
/**
 * Helper functions for creating standard lattices
 */
export function create_square_lattice(a: number): Lattice2D;
export function create_rectangular_lattice(a: number, b: number): Lattice2D;
export function create_hexagonal_lattice(a: number): Lattice2D;
export function create_oblique_lattice(a: number, b: number, gamma: number): Lattice2D;
export function init_panic_hook(): void;
/**
 * Version information
 */
export function version(): string;
/**
 * 2D Bravais lattice classification (WASM-compatible)
 */
export enum Bravais2D {
  Square = 0,
  Rectangular = 1,
  CenteredRectangular = 2,
  Hexagonal = 3,
  Oblique = 4,
}
/**
 * WASM-compatible wrapper for 2D base matrix operations
 */
export class BaseMatrixDirect {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a 2D base matrix from two base vectors
   * Each vector should be provided as a 3-element array [x, y, z]
   */
  constructor(base_1: Float64Array, base_2: Float64Array);
  /**
   * Get the base matrix as a flat array (column-major order)
   */
  getMatrix(): Float64Array;
  /**
   * Get the determinant of the base matrix
   */
  determinant(): number;
}
/**
 * WASM-compatible wrapper for reciprocal space base matrix
 */
export class BaseMatrixReciprocal {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get the base matrix as a flat array (column-major order)
   */
  getMatrix(): Float64Array;
  /**
   * Get the determinant of the base matrix
   */
  determinant(): number;
}
/**
 * WASM-compatible wrapper for 2D lattice
 */
export class Lattice2D {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a 2D lattice from direct space basis vectors
   * The matrix should be provided as a flat array in column-major order (9 elements)
   */
  constructor(direct_matrix: Float64Array);
  /**
   * Get the direct space basis matrix as a flat array (column-major order)
   */
  getDirectBasis(): Float64Array;
  /**
   * Get the reciprocal space basis matrix as a flat array (column-major order)
   */
  getReciprocalBasis(): Float64Array;
  /**
   * Get the Bravais lattice type
   */
  getBravaisType(): Bravais2D;
  /**
   * Get the Wigner-Seitz cell vertices as a flat array
   * Each vertex is [x, y, z], returned as a flat array
   */
  getWignerSeitzVertices(): Float64Array;
  /**
   * Get the Brillouin zone vertices as a flat array
   * Each vertex is [x, y, z], returned as a flat array
   */
  getBrillouinZoneVertices(): Float64Array;
  /**
   * Generate direct lattice points in a rectangle
   */
  getDirectLatticePoints(width: number, height: number): Float64Array;
  /**
   * Generate reciprocal lattice points in a rectangle
   */
  getReciprocalLatticePoints(width: number, height: number): Float64Array;
  /**
   * Generate high symmetry k-path for band structure calculations
   */
  getHighSymmetryPath(n_points_per_segment: number): Float64Array;
  /**
   * Check if a point is in the Brillouin zone
   */
  isInBrillouinZone(k_point: Float64Array): boolean;
  /**
   * Reduce a k-point to the first Brillouin zone
   */
  reduceToBrillouinZone(k_point: Float64Array): Float64Array;
}
/**
 * WASM-compatible wrapper for Moire2D
 */
export class Moire2D {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a Moiré lattice from a base lattice and transformation
   */
  constructor(base_lattice: Lattice2D, transformation: MoireTransformation);
  /**
   * Get the effective moiré lattice direct space basis matrix
   */
  getEffectiveDirectBasis(): Float64Array;
  /**
   * Get the effective moiré lattice reciprocal space basis matrix
   */
  getEffectiveReciprocalBasis(): Float64Array;
  /**
   * Get the Bravais lattice type of the effective moiré lattice
   */
  getEffectiveBravaisType(): Bravais2D;
  /**
   * Get the Wigner-Seitz cell vertices of the effective lattice
   */
  getEffectiveWignerSeitzVertices(): Float64Array;
  /**
   * Get the Brillouin zone vertices of the effective lattice
   */
  getEffectiveBrillouinZoneVertices(): Float64Array;
  /**
   * Generate direct lattice points in a rectangle for the effective moiré lattice
   */
  getEffectiveDirectLatticePoints(width: number, height: number): Float64Array;
  /**
   * Generate reciprocal lattice points in a rectangle for the effective moiré lattice
   */
  getEffectiveReciprocalLatticePoints(width: number, height: number): Float64Array;
  /**
   * Generate high symmetry k-path for the effective moiré lattice
   */
  getEffectiveHighSymmetryPath(n_points_per_segment: number): Float64Array;
  /**
   * Check if a point is in the Brillouin zone of the effective lattice
   */
  isInEffectiveBrillouinZone(k_point: Float64Array): boolean;
  /**
   * Reduce a k-point to the first Brillouin zone of the effective lattice
   */
  reduceToEffectiveBrillouinZone(k_point: Float64Array): Float64Array;
  /**
   * Get the first constituent lattice
   */
  getLattice1(): Lattice2D;
  /**
   * Get the second constituent lattice
   */
  getLattice2(): Lattice2D;
  /**
   * Get the transformation that was applied
   */
  getTransformation(): MoireTransformation;
  /**
   * Check if the moiré lattice is commensurate
   */
  isCommensurate(): boolean;
}
/**
 * WASM-compatible wrapper for MoireTransformation
 */
export class MoireTransformation {
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Create a simple twist (rotation) transformation
   */
  constructor(angle: number);
  /**
   * Create a rotation and scaling transformation
   */
  static newRotationScale(angle: number, scale: number): MoireTransformation;
  /**
   * Create an anisotropic scaling transformation
   */
  static newAnisotropicScale(scale_x: number, scale_y: number): MoireTransformation;
  /**
   * Create a shear transformation
   */
  static newShear(shear_x: number, shear_y: number): MoireTransformation;
  /**
   * Create a general matrix transformation
   * Matrix should be provided as a flat array in column-major order (4 elements for 2x2)
   */
  static newGeneral(matrix: Float64Array): MoireTransformation;
  /**
   * Get the transformation matrix as a flat array (column-major order, 4 elements)
   */
  getMatrix2(): Float64Array;
  /**
   * Get the transformation matrix as a 3x3 matrix (embedding 2D in 3D)
   * Returns a flat array in column-major order (9 elements)
   */
  getMatrix3(): Float64Array;
}
/**
 * WASM-compatible wrapper for Polyhedron
 */
export class Polyhedron {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  /**
   * Get the vertices of the polyhedron as a flat array
   * Each vertex is represented as [x, y, z]
   */
  getVertices(): Float64Array;
  /**
   * Get the number of vertices
   */
  vertexCount(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_moiretransformation_free: (a: number, b: number) => void;
  readonly moiretransformation_new_twist: (a: number) => number;
  readonly moiretransformation_newRotationScale: (a: number, b: number) => number;
  readonly moiretransformation_newAnisotropicScale: (a: number, b: number) => number;
  readonly moiretransformation_newShear: (a: number, b: number) => number;
  readonly moiretransformation_newGeneral: (a: number, b: number) => [number, number, number];
  readonly moiretransformation_getMatrix2: (a: number) => [number, number];
  readonly moiretransformation_getMatrix3: (a: number) => [number, number];
  readonly __wbg_moire2d_free: (a: number, b: number) => void;
  readonly moire2d_new: (a: number, b: number) => [number, number, number];
  readonly moire2d_getEffectiveDirectBasis: (a: number) => [number, number];
  readonly moire2d_getEffectiveReciprocalBasis: (a: number) => [number, number];
  readonly moire2d_getEffectiveBravaisType: (a: number) => number;
  readonly moire2d_getEffectiveWignerSeitzVertices: (a: number) => [number, number];
  readonly moire2d_getEffectiveBrillouinZoneVertices: (a: number) => [number, number];
  readonly moire2d_getEffectiveDirectLatticePoints: (a: number, b: number, c: number) => [number, number];
  readonly moire2d_getEffectiveReciprocalLatticePoints: (a: number, b: number, c: number) => [number, number];
  readonly moire2d_getEffectiveHighSymmetryPath: (a: number, b: number) => [number, number];
  readonly moire2d_isInEffectiveBrillouinZone: (a: number, b: number, c: number) => [number, number, number];
  readonly moire2d_reduceToEffectiveBrillouinZone: (a: number, b: number, c: number) => [number, number, number, number];
  readonly moire2d_getLattice1: (a: number) => number;
  readonly moire2d_getLattice2: (a: number) => number;
  readonly moire2d_getTransformation: (a: number) => number;
  readonly moire2d_isCommensurate: (a: number) => number;
  readonly createMoireLattice: (a: number, b: number) => [number, number, number];
  readonly __wbg_lattice2d_free: (a: number, b: number) => void;
  readonly lattice2d_new: (a: number, b: number) => [number, number, number];
  readonly lattice2d_getDirectBasis: (a: number) => [number, number];
  readonly lattice2d_getReciprocalBasis: (a: number) => [number, number];
  readonly lattice2d_getBravaisType: (a: number) => number;
  readonly lattice2d_getWignerSeitzVertices: (a: number) => [number, number];
  readonly lattice2d_getBrillouinZoneVertices: (a: number) => [number, number];
  readonly lattice2d_getDirectLatticePoints: (a: number, b: number, c: number) => [number, number];
  readonly lattice2d_getReciprocalLatticePoints: (a: number, b: number, c: number) => [number, number];
  readonly lattice2d_getHighSymmetryPath: (a: number, b: number) => [number, number];
  readonly lattice2d_isInBrillouinZone: (a: number, b: number, c: number) => [number, number, number];
  readonly lattice2d_reduceToBrillouinZone: (a: number, b: number, c: number) => [number, number, number, number];
  readonly create_square_lattice: (a: number) => [number, number, number];
  readonly create_rectangular_lattice: (a: number, b: number) => [number, number, number];
  readonly create_hexagonal_lattice: (a: number) => [number, number, number];
  readonly create_oblique_lattice: (a: number, b: number, c: number) => [number, number, number];
  readonly init_panic_hook: () => void;
  readonly version: () => [number, number];
  readonly __wbg_basematrixdirect_free: (a: number, b: number) => void;
  readonly basematrixdirect_new: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly basematrixdirect_getMatrix: (a: number) => [number, number];
  readonly basematrixdirect_determinant: (a: number) => number;
  readonly __wbg_basematrixreciprocal_free: (a: number, b: number) => void;
  readonly basematrixreciprocal_getMatrix: (a: number) => [number, number];
  readonly basematrixreciprocal_determinant: (a: number) => number;
  readonly __wbg_polyhedron_free: (a: number, b: number) => void;
  readonly polyhedron_getVertices: (a: number) => [number, number];
  readonly polyhedron_vertexCount: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
