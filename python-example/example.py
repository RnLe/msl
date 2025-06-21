#!/usr/bin/env python3
"""
Example usage of the moire-lattice Python bindings.

This script demonstrates basic functionality of the Python wrapper
for lattice calculations.
"""

import sys
import os

# Add the local package path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rust-python', 'python'))

try:
    import moire_lattice_py as ml
    print(f"✓ Successfully imported moire_lattice_py version {ml.version()}")
except ImportError as e:
    print(f"✗ Failed to import moire_lattice_py: {e}")
    print("\nTo install the Python bindings:")
    print("1. Install maturin: pip install maturin")
    print("2. Build the package: cd rust-python && maturin develop")
    sys.exit(1)

def main():
    print("\n" + "="*50)
    print("Moire Lattice Python Example")
    print("="*50)
    
    # Create different lattice types
    print("\n1. Creating different lattice types:")
    
    square = ml.create_square_lattice(1.0)
    print(f"   Square lattice: {square}")
    
    hex_lattice = ml.create_hexagonal_lattice(1.0)
    print(f"   Hexagonal lattice: {hex_lattice}")
    
    rect_lattice = ml.create_rectangular_lattice(1.0, 1.5)
    print(f"   Rectangular lattice: {rect_lattice}")
    
    # Custom lattice
    custom = ml.PyLattice2D("oblique", 1.0, 1.2, 75.0)
    print(f"   Custom oblique lattice: {custom}")
    
    # Generate lattice points
    print("\n2. Generating lattice points:")
    
    radius = 3.0
    points = square.generate_points(radius)
    print(f"   Square lattice points within radius {radius}: {len(points)} points")
    
    points_hex = hex_lattice.generate_points(radius)
    print(f"   Hexagonal lattice points within radius {radius}: {len(points_hex)} points")
    
    # Display some points
    print(f"   First 10 square lattice points: {points[:10]}")
    
    # Lattice properties
    print("\n3. Lattice properties:")
    
    print(f"   Square lattice unit cell area: {square.unit_cell_area():.3f}")
    print(f"   Hexagonal lattice unit cell area: {hex_lattice.unit_cell_area():.3f}")
    
    vectors = square.lattice_vectors()
    print(f"   Square lattice vectors: {vectors}")
    
    reciprocal = square.reciprocal_vectors()
    print(f"   Square reciprocal vectors: {reciprocal}")
    
    # Parameters
    print("\n4. Lattice parameters:")
    params = custom.get_parameters()
    print(f"   Custom lattice parameters: {params}")
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
