# Moire Lattice Python Bindings

High-performance Python bindings for the `moire-lattice` Rust library, providing comprehensive tools for lattice and moire lattice calculations.

## Installation

### From PyPI (when published)
```bash
pip install moire-lattice-py
```

### Development Installation
```bash
# Install maturin for building Python extensions from Rust
pip install maturin

# Build and install in development mode
maturin develop

# Or build a wheel
maturin build
```

## Quick Start

```python
import moire_lattice_py as ml

# Create different lattice types
square = ml.create_square_lattice(1.0)
hex_lattice = ml.create_hexagonal_lattice(1.0) 
rect_lattice = ml.create_rectangular_lattice(1.0, 1.5)

# Generate lattice points within a radius
points = square.generate_points(radius=5.0)
print(f"Generated {len(points)} lattice points")

# Get lattice properties
print(f"Unit cell area: {square.unit_cell_area()}")
print(f"Lattice vectors: {square.lattice_vectors()}")
print(f"Reciprocal vectors: {square.reciprocal_vectors()}")
```

## Features

- **High Performance**: Built on Rust core for maximum computational efficiency
- **Multiple Lattice Types**: Support for square, rectangular, hexagonal, and oblique lattices
- **Lattice Generation**: Generate lattice points within specified regions
- **Reciprocal Space**: Calculate reciprocal lattice vectors and properties
- **Pythonic API**: Clean, intuitive interface following Python conventions

## API Reference

### Classes

#### `PyLattice2D`
Main lattice class supporting various 2D Bravais lattices.

**Constructor:**
- `PyLattice2D(lattice_type, a, b=None, angle=None)`
  - `lattice_type`: "square", "rectangular", "hexagonal", or "oblique"
  - `a`: First lattice parameter
  - `b`: Second lattice parameter (defaults to `a`)
  - `angle`: Lattice angle in degrees (defaults to 90°)

**Methods:**
- `generate_points(radius, center=(0, 0))`: Generate lattice points within radius
- `get_parameters()`: Get lattice parameters as dictionary
- `unit_cell_area()`: Calculate unit cell area
- `lattice_vectors()`: Get lattice vectors as tuples
- `reciprocal_vectors()`: Get reciprocal lattice vectors

### Utility Functions

- `create_square_lattice(a)`: Create square lattice with parameter `a`
- `create_hexagonal_lattice(a)`: Create hexagonal lattice with parameter `a`
- `create_rectangular_lattice(a, b)`: Create rectangular lattice with parameters `a`, `b`
- `version()`: Get library version

## Development

### Building from Source

1. Install Rust toolchain: https://rustup.rs/
2. Install maturin: `pip install maturin`
3. Build: `maturin develop`

### Testing

```bash
# Install test dependencies
pip install pytest numpy matplotlib

# Run tests
pytest tests/
```

## License

This project is licensed under: MIT License