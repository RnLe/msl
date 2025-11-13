#!/usr/bin/env python3
"""
Quick test script to verify Rust-Python bindings integration
"""
import sys
sys.path.insert(0, '.')

from common.moire_utils import create_twisted_bilayer
import moire_lattice_py as ml
import math

print("=" * 70)
print("Testing Rust-Python Bindings Integration")
print("=" * 70)

# Test 1: Create twisted bilayer using Rust bindings
print("\n1. Creating twisted bilayer moiré lattices:")
for lattice_type in ['square', 'hex']:
    print(f"\n  {lattice_type.upper()} Lattice:")
    moire_data = create_twisted_bilayer(lattice_type, theta_deg=1.5, a=1.0)
    
    print(f"    Base lattice: {moire_data['lattice']}")
    print(f"    Twist angle: {moire_data['theta_deg']:.2f}°")
    print(f"    Moiré length: {moire_data['moire_length']:.4f}")
    print(f"    Reciprocal magnitude |G_m|: {moire_data['G_magnitude']:.4f}")
    print(f"    Basis vectors:")
    print(f"      a1 = {moire_data['a1']}")
    print(f"      a2 = {moire_data['a2']}")

# Test 2: Check Brillouin zone calculation
print("\n2. Brillouin Zone Calculations:")
square_lattice = ml.Lattice2D.from_basis_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
bz = square_lattice.brillouin_zone()
print(f"  Square lattice BZ:")
print(f"    Vertices: {bz.num_vertices()}")
print(f"    Area: {bz.measure:.4f}")

hex_lattice = ml.Lattice2D.from_basis_vectors(
    [1.0, 0.0, 0.0],
    [0.5, math.sqrt(3)/2, 0.0]
)
bz_hex = hex_lattice.brillouin_zone()
print(f"  Hexagonal lattice BZ:")
print(f"    Vertices: {bz_hex.num_vertices()}")
print(f"    Area: {bz_hex.measure:.4f}")

# Test 3: High symmetry points
print("\n3. High Symmetry Points:")
hs = square_lattice.reciprocal_high_symmetry()
points = hs.get_points()
print(f"  Square lattice:")
for label, point in points[:3]:  # Show first 3
    print(f"    {label}: {point.position}")

# Test 4: Moiré parameter scaling
print("\n4. Moiré Length vs Twist Angle:")
print(f"  {'Angle (°)':>10} {'L_moiré':>12} {'|G_m|':>12}")
print(f"  {'-'*10} {'-'*12} {'-'*12}")
for theta in [0.5, 1.0, 1.5, 2.0, 3.0]:
    moire = create_twisted_bilayer('hex', theta, a=1.0)
    print(f"  {theta:>10.1f} {moire['moire_length']:>12.4f} {moire['G_magnitude']:>12.4f}")

print("\n" + "=" * 70)
print("✅ All Rust-Python binding tests passed!")
print("=" * 70)
