# Moire Lattice Python Bindings

Python bindings for the `moire-lattice` Rust library. This package provides 1:1 mappings of the Rust API for lattice calculations, Brillouin zone analysis, and moiré pattern computations.

## Features

- **2D Bravais Lattices**: Create and analyze all 2D Bravais lattice types (Square, Hexagonal, Rectangular, etc.)
- **Brillouin Zone Calculations**: Automatic computation of first Brillouin zones and Wigner-Seitz cells
- **High Symmetry Points**: Built-in high symmetry points and k-paths for band structure calculations
- **Moiré Lattices**: Support for twisted bilayer and moiré pattern calculations
- **High Performance**: Rust-powered computations with Python convenience

## Installation

### Development Installation

```bash
# Activate your conda/mamba environment
mamba activate msl

# Install maturin if not already installed
pip install maturin

# Install in development mode
cd rust-python
maturin develop

# Or use the Makefile from the web directory
cd ../web
make build-python-dev
```

### Production Build

```bash
cd rust-python
maturin build --release
```

## Quick Start

```python
import moire_lattice_py as ml
import math

# Create a square lattice
lattice = ml.Lattice2D.from_basis_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
print(lattice.direct_bravais().name())  # "Square"

# Get Brillouin zone
bz = lattice.brillouin_zone()
print(f"BZ has {bz.num_vertices()} vertices")
print(f"BZ area: {bz.measure:.4f}")

# Get high symmetry points
hs = lattice.reciprocal_high_symmetry()
for label, point in hs.get_points():
    print(f"{label}: {point.position}")

# Generate k-path for band structure
k_path = lattice.generate_high_symmetry_k_path(50)  # 50 points per segment
```

## Testing

Run the test suite:

```bash
# Activate environment
mamba activate msl

# Run tests
python test_bindings.py
```

## License

MIT License - See LICENSE file for details
