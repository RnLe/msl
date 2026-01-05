# Moire Lattice

[![Crates.io](https://img.shields.io/crates/v/moire-lattice.svg)](https://crates.io/crates/moire-lattice)
[![Documentation](https://docs.rs/moire-lattice/badge.svg)](https://docs.rs/moire-lattice)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rnle/msl)

A high-performance Rust library for lattice framework calculations, designed for photonic band structure computations and moire physics simulations.

## Features

- **High-Performance Lattice Generation**: Efficient 2D and 3D Bravais lattice implementations
- **Moire Pattern Calculations**: Advanced algorithms for twisted bilayer systems
- **Crystallographic Operations**: Symmetry operations, point groups, and space groups
- **Voronoi Cell Analysis**: Fast Voronoi tessellation for both 2D and 3D systems
- **Photonic Band Structure Ready**: Designed as foundation for MPB-like eigensolvers
- **WebAssembly Support**: Compile to WASM for browser-based applications
- **Python Bindings**: Optional Python interface via PyO3

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
moire-lattice = "0.1"
```

### Basic Usage

```rust
use moire_lattice::lattice::{Lattice2D, BravaisType};

// Create a hexagonal lattice
let hex_lattice = Lattice2D::new(
    BravaisType::Hexagonal,
    1.0,  // lattice constant a
    1.0,  // lattice constant b  
    120.0 // angle in degrees
)?;

// Generate lattice points within radius
let points = hex_lattice.generate_points(5.0)?;
println!("Generated {} lattice points", points.len());

// Get lattice properties
let area = hex_lattice.unit_cell_area();
let vectors = hex_lattice.lattice_vectors();
println!("Unit cell area: {:.4}", area);
```

### For Photonic Band Structure Calculations

```rust
use moire_lattice::lattice::Lattice2D;

// Enable photonic features
let lattice = Lattice2D::hexagonal(1.0)?;

// This lattice can now be used with your MPB-rust solver
// for photonic band structure calculations
```

## Feature Flags

- `default`: Includes standard library support
- `photonic`: Core features for photonic band structure calculations
- `mpb-compat`: Full compatibility layer for MPB replacement
- `parallel`: Parallel computing with Rayon
- `serde`: Serialization support
- `hdf5`: HDF5 file format support
- `plotting`: Visualization capabilities
- `voronoi2d`: 2D Voronoi cell calculations
- `ws3d_voro`: Fast 3D Wigner-Seitz cells

## Applications

This library is designed to be the foundation for:

- **Photonic Band Structure Solvers**: Replacement for MIT MPB
- **Moire Physics Simulations**: Twisted bilayer materials
- **Crystallographic Analysis**: Crystal structure calculations  
- **Computational Photonics**: Photonic crystal design
- **Materials Science**: Lattice-based material properties

## Performance

- Optimized for both CPU and WebAssembly targets
- Parallel algorithms using Rayon
- SIMD optimizations where applicable
- Memory-efficient data structures
- Benchmark suite included

## Integration

### With MPB-Rust

This library serves as the lattice framework for the upcoming `mpb-rust` photonic band structure solver:

```rust
// Future MPB-rust integration
use moire_lattice::lattice::Lattice2D;
use mpb_rust::solver::BandStructureSolver;

let lattice = Lattice2D::hexagonal(1.0)?;
let solver = BandStructureSolver::new(lattice);
let bands = solver.compute_bands(k_points)?;
```

### WebAssembly

Compile to WebAssembly for browser applications:

```bash
wasm-pack build --target web --features photonic
```

### Python Bindings

Install the Python package:

```bash
pip install moire-lattice-py
```

```python
import moire_lattice_py as ml

lattice = ml.create_hexagonal_lattice(1.0)
points = lattice.generate_points(5.0)
```

## Documentation

- [API Documentation](https://docs.rs/moire-lattice)
- [Examples](https://github.com/rnle/msl/tree/main/examples)
- [Benchmarks](https://github.com/rnle/msl/tree/main/benches)

## License

Licensed under the MIT License. See [LICENSE-MIT](LICENSE-MIT) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{moire_lattice,
  author = {Rene-Marcel Lehner},
  title = {Moire Lattice: High-Performance Lattice Framework for Photonic Calculations},
  url = {https://github.com/rnle/msl},
  year = {2025}
}
```
