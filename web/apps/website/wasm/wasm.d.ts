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
