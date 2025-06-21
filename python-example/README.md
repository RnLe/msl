# Python Example

This directory contains example scripts demonstrating the usage of the Python bindings for the moire-lattice library.

## Files

- `example.py`: Basic usage examples of the Python API

## Running the Examples

1. First, build and install the Python bindings:
   ```bash
   cd ../rust-python
   pip install maturin
   maturin develop
   ```

2. Run the example script:
   ```bash
   python example.py
   ```

## Requirements

- Python 3.8+
- maturin (for building the Rust extension)
- The moire-lattice Python package (built from `../rust-python`)

## Example Output

The script demonstrates:
- Creating different lattice types (square, hexagonal, rectangular, oblique)
- Generating lattice points within a specified radius
- Accessing lattice properties (unit cell area, lattice vectors, reciprocal vectors)
- Working with lattice parameters

For more advanced examples and API documentation, see the main project documentation.
