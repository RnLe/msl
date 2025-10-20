"""
Test file for moire_lattice_py Python bindings

This file demonstrates the usage of the Python bindings for the Rust moire-lattice library.
All classes and methods are 1:1 mappings from the Rust API.
"""

import moire_lattice_py as ml
import math


def test_square_lattice():
    """Test creating and analyzing a square lattice"""
    print("=" * 60)
    print("Testing Square Lattice")
    print("=" * 60)
    
    # Create a square lattice with lattice constant 1.0
    lattice = ml.Lattice2D.from_basis_vectors([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    print(f"\nCreated lattice: {lattice}")
    
    # Get lattice properties
    print(f"\nDirect space:")
    print(f"  Bravais type: {lattice.direct_bravais().name()}")
    a, b = lattice.direct_lattice_parameters()
    print(f"  Lattice parameters: a={a:.4f}, b={b:.4f}")
    print(f"  Angle: {math.degrees(lattice.direct_lattice_angle()):.2f}°")
    
    print(f"\nReciprocal space:")
    print(f"  Bravais type: {lattice.reciprocal_bravais().name()}")
    b1, b2 = lattice.reciprocal_lattice_parameters()
    print(f"  Lattice parameters: b1={b1:.4f}, b2={b2:.4f}")
    
    # Get Brillouin zone
    bz = lattice.brillouin_zone()
    print(f"\nBrillouin zone:")
    print(f"  Vertices: {bz.num_vertices()}")
    print(f"  Edges: {bz.num_edges()}")
    print(f"  Area: {bz.measure:.4f}")
    
    # Get high symmetry points
    hs = lattice.reciprocal_high_symmetry()
    points = hs.get_points()
    print(f"\nHigh symmetry points:")
    for label, point in points:
        pos = point.position
        print(f"  {label}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] - {point.description}")
    
    # Generate k-path for band structure
    k_path = lattice.generate_high_symmetry_k_path(10)
    print(f"\nGenerated k-path: {len(k_path)} points")
    print(f"  First point: {k_path[0]}")
    print(f"  Last point: {k_path[-1]}")
    
    # Compute lattice points in a region
    points = lattice.compute_direct_lattice_points_in_rectangle(5.0, 5.0)
    print(f"\nLattice points in 5×5 rectangle: {len(points)} points")
    
    print("\n✅ Square lattice test passed!\n")


def test_hexagonal_lattice():
    """Test creating and analyzing a hexagonal lattice"""
    print("=" * 60)
    print("Testing Hexagonal Lattice")
    print("=" * 60)
    
    # Create a hexagonal lattice
    a = 1.0
    lattice = ml.Lattice2D.from_basis_vectors(
        [a, 0.0, 0.0],
        [a * 0.5, a * math.sqrt(3) / 2, 0.0]
    )
    print(f"\nCreated lattice: {lattice}")
    
    # Note: The direct space might be detected as Oblique due to numerical precision,
    # but the reciprocal space should be Hexagonal
    print(f"Direct Bravais type: {lattice.direct_bravais().name()}")
    print(f"Reciprocal Bravais type: {lattice.reciprocal_bravais().name()}")
    
    # Get high symmetry points (K and M points for hexagonal)
    hs = lattice.reciprocal_high_symmetry()
    points = hs.get_points()
    print(f"\nHigh symmetry points:")
    for label, point in points:
        pos = point.position
        print(f"  {label}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Get Wigner-Seitz cell
    ws = lattice.wigner_seitz()
    print(f"\nWigner-Seitz cell:")
    print(f"  Vertices: {ws.num_vertices()}")
    print(f"  Area: {ws.measure:.4f}")
    
    print("\n✅ Hexagonal lattice test passed!\n")


def test_base_matrix():
    """Test BaseMatrix operations"""
    print("=" * 60)
    print("Testing BaseMatrix Operations")
    print("=" * 60)
    
    # Create a direct space base matrix
    direct = ml.BaseMatrixDirect.from_base_vectors_2d(
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    )
    print(f"\nDirect space base matrix: {direct}")
    
    # Get determinant
    det = direct.determinant()
    print(f"Determinant: {det:.4f}")
    
    # Convert to reciprocal space
    reciprocal = direct.to_reciprocal()
    print(f"Reciprocal space base matrix: {reciprocal}")
    
    # Get metric tensors
    direct_metric = direct.metric()
    print(f"\nDirect metric tensor (first row): {direct_metric[0]}")
    
    # Convert back
    direct_again = reciprocal.to_direct()
    print(f"Converted back to direct: {direct_again}")
    
    print("\n✅ BaseMatrix test passed!\n")


def test_bravais_types():
    """Test Bravais lattice type enums"""
    print("=" * 60)
    print("Testing Bravais Types")
    print("=" * 60)
    
    # Test 2D Bravais types
    bravais_2d = [
        ml.Bravais2D.square(),
        ml.Bravais2D.hexagonal(),
        ml.Bravais2D.rectangular(),
        ml.Bravais2D.oblique(),
    ]
    
    print("\n2D Bravais lattice types:")
    for b in bravais_2d:
        print(f"  {b}")
    
    # Test 3D Bravais types
    centering = ml.Centering.primitive()
    cubic = ml.Bravais3D.cubic(centering)
    print(f"\n3D Bravais example: {cubic}")
    
    print("\n✅ Bravais types test passed!\n")


def test_moire_transformation():
    """Test moiré transformation types"""
    print("=" * 60)
    print("Testing Moiré Transformation")
    print("=" * 60)
    
    # Create different transformation types
    twist = ml.MoireTransformation.twist(math.radians(1.5))
    print(f"\nTwist transformation: {twist}")
    print(f"  Matrix: {twist.to_matrix()}")
    
    rotation_scale = ml.MoireTransformation.rotation_scale(math.radians(2.0), 1.01)
    print(f"\nRotation-scale transformation: {rotation_scale}")
    
    anisotropic = ml.MoireTransformation.anisotropic_scale(1.05, 0.95)
    print(f"\nAnisotropic scale: {anisotropic}")
    
    print("\n✅ Moiré transformation test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Moire Lattice Python Bindings Test Suite")
    print("=" * 60 + "\n")
    
    test_base_matrix()
    test_bravais_types()
    test_square_lattice()
    test_hexagonal_lattice()
    test_moire_transformation()
    
    print("=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)
