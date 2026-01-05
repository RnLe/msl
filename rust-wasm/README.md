# Moire Lattice WASM Bindings

WebAssembly bindings for the Moire Lattice library, enabling lattice calculations and moire physics simulations in web browsers.

## Overview

This package provides WASM-compiled bindings for the core `moire-lattice` library, focusing on:
- 2D Bravais lattice generation and analysis
- Moire pattern calculations for twisted bilayer systems
- Crystallographic symmetry operations
- Voronoi cell and Brillouin zone computations

## Project Structure

```
rust-wasm/
├── Cargo.toml                     # Project configuration
└── src/
    ├── lib.rs                     # Main library entry point
    ├── config.rs                  # Configuration constants
    ├── interfaces.rs              # Interface definitions
    ├── utils.rs                   # Utility functions for WASM
    ├── lattice/                   # Lattice module
    │   ├── base_matrix.rs         # Base matrix WASM wrappers
    │   ├── lattice2d.rs           # 2D lattice WASM wrappers
    │   ├── lattice_algorithms.rs  # Lattice algorithms re-exports
    │   ├── lattice_construction.rs # Lattice construction functions
    │   ├── lattice_like_2d.rs     # LatticeLike2D trait re-export
    │   ├── lattice_types.rs       # Bravais2D types
    │   ├── polyhedron.rs          # Polyhedron WASM wrappers
    │   └── voronoi_cells.rs       # Voronoi computation re-exports
    ├── symmetries/                # Symmetries module
    │   ├── high_symmetry_points.rs # High symmetry points re-exports
    │   └── symmetry_operations.rs  # Symmetry operations re-exports
    └── moire_lattice/             # Moiré lattice module
        └── moire2d.rs             # 2D moiré lattice WASM wrappers
```

## Module Organization

The project maintains the same folder structure as the core `rust-core` library for consistency:

- **lib.rs**: Main entry point with WASM initialization
- **lattice/**: All lattice-related structures and algorithms
  - Base matrices and lattice construction
  - 2D lattice operations
  - Voronoi cells and Brillouin zones
  - Lattice point generation algorithms
- **symmetries/**: Crystallographic symmetry operations
  - High symmetry points and paths
  - Symmetry operations
- **moire_lattice/**: Moiré pattern calculations
  - 2D moiré lattice structures
  - Transformation types for creating moiré patterns

## Key Features

### Lattice2D
- Create 2D lattices from basis vectors
- Access direct and reciprocal space bases
- Generate lattice points in rectangular regions
- Compute Wigner-Seitz cells and Brillouin zones
- Generate high-symmetry k-paths for band structures
- Check if points are in the Brillouin zone

### Moire2D
- Create moiré lattices from base lattice + transformation
- Support multiple transformation types:
  - Simple twist (rotation)
  - Rotation with scaling
  - Anisotropic scaling
  - Shear transformations
  - General matrix transformations
- Access effective moiré lattice properties

### Lattice Construction Helpers
- `create_square_lattice(a)`
- `create_rectangular_lattice(a, b)`
- `create_hexagonal_lattice(a)`
- `create_oblique_lattice(a, b, gamma)`

## Building

```bash
# Check for errors
cargo check

# Build the library
cargo build

# Build for WASM (requires wasm-pack)
wasm-pack build --target web --out-dir pkg
```

## Usage from JavaScript

```javascript
import init, { 
    Lattice2D, 
    create_hexagonal_lattice,
    MoireTransformation,
    Moire2D 
} from './pkg/moire_lattice_wasm.js';

await init();

// Create a hexagonal lattice
const lattice = create_hexagonal_lattice(1.0);

// Get basis vectors
const directBasis = lattice.getDirectBasis();
const reciprocalBasis = lattice.getReciprocalBasis();

// Get lattice points
const points = lattice.getDirectLatticePoints(10.0, 10.0);

// Get high symmetry k-path
const kPath = lattice.getHighSymmetryPath(50);

// Create a moiré transformation (twist angle in radians)
const twist = MoireTransformation.new_twist(Math.PI / 180 * 3.0);

// Create moiré lattice
const moireLattice = new Moire2D(lattice, twist);
// Or alternatively:
// const moireLattice = createMoireLattice(lattice, twist);

// Access constituent lattices
const lattice1 = moireLattice.getLattice1();
const lattice2 = moireLattice.getLattice2();

// Get the transformation
const transform = moireLattice.getTransformation();

// Check commensurability
const isCommensurate = moireLattice.isCommensurate();
```

## Design Decisions

1. **Re-exports over Wrappers**: Where possible, we re-export core library types directly to avoid duplication. Only types that need WASM-specific interfaces (like converting between Rust and JS data) are wrapped.

2. **Flat Arrays for Vectors/Matrices**: All vectors and matrices are exposed as flat `Vec<f64>` arrays in column-major order for easy JavaScript interop.

3. **Consistent Naming**: JavaScript-facing methods use camelCase (via `js_name` attribute) while maintaining Rust conventions internally.

4. **Error Handling**: All fallible operations return `Result<T, JsValue>` which automatically converts to JavaScript exceptions.

## Limitations

- Only 2D lattices are exposed (3D lattices from the core library are not included).
- Geometry and materials modules from the core library are not included as they are not required for moiré lattice calculations.

## Dependencies

- `moire-lattice`: Core lattice library (workspace dependency)
- `wasm-bindgen`: Rust-WASM bindings
- `nalgebra`: Linear algebra
- `serde` + `serde-wasm-bindgen`: Serialization for JS interop
- `console_error_panic_hook`: Better error messages in browser console

## Version

Current version: 0.1.1 (matches core library version)

## License

Licensed under MIT (same as core library)
