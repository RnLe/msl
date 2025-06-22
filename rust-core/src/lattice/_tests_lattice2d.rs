#[cfg(test)]
mod _tests_lattice2d {
    use super::super::lattice2d::Lattice2D;
    use super::super::bravais_types::Bravais2D;
    use nalgebra::{Matrix3, Vector3};
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // Helper function to create a square lattice
    fn create_square_lattice(a: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a, 0.0, 0.0,
            0.0, a, 0.0,
            0.0, 0.0, 1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create a hexagonal lattice
    fn create_hexagonal_lattice(a: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a, -a * 0.5, 0.0,
            0.0, a * (3.0_f64).sqrt() / 2.0, 0.0,
            0.0, 0.0, 1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create a rectangular lattice
    fn create_rectangular_lattice(a: f64, b: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a, 0.0, 0.0,
            0.0, b, 0.0,
            0.0, 0.0, 1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create an oblique lattice
    fn create_oblique_lattice(a: f64, b: f64, gamma: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a, b * gamma.cos(), 0.0,
            0.0, b * gamma.sin(), 0.0,
            0.0, 0.0, 1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    #[test]
    fn test_new_square_lattice() {
        let lattice = create_square_lattice(1.0);
        
        assert_eq!(lattice.bravais_type(), Bravais2D::Square);
        assert!((lattice.cell_area() - 1.0).abs() < TOL);
        
        // Check reciprocal lattice
        let expected_reciprocal = 2.0 * PI;
        assert!((lattice.reciprocal[(0, 0)] - expected_reciprocal).abs() < TOL);
        assert!((lattice.reciprocal[(1, 1)] - expected_reciprocal).abs() < TOL);
        assert!(lattice.reciprocal[(0, 1)].abs() < TOL);
        assert!(lattice.reciprocal[(1, 0)].abs() < TOL);
    }

    #[test]
    fn test_new_hexagonal_lattice() {
        let lattice = create_hexagonal_lattice(1.0);
        
        assert_eq!(lattice.bravais_type(), Bravais2D::Hexagonal);
        let expected_area = (3.0_f64).sqrt() / 2.0;
        assert!((lattice.cell_area() - expected_area).abs() < TOL);
    }

    #[test]
    fn test_frac_to_cart_basic() {
        let lattice = create_square_lattice(2.0);
        
        // Test origin
        let origin = lattice.frac_to_cart(Vector3::new(0.0, 0.0, 0.0));
        assert!(origin.norm() < TOL);
        
        // Test unit vectors
        let v1 = lattice.frac_to_cart(Vector3::new(1.0, 0.0, 0.0));
        assert!((v1 - Vector3::new(2.0, 0.0, 0.0)).norm() < TOL);
        
        let v2 = lattice.frac_to_cart(Vector3::new(0.0, 1.0, 0.0));
        assert!((v2 - Vector3::new(0.0, 2.0, 0.0)).norm() < TOL);
        
        // Test diagonal
        let v_diag = lattice.frac_to_cart(Vector3::new(1.0, 1.0, 0.0));
        assert!((v_diag - Vector3::new(2.0, 2.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn test_cart_to_frac_basic() {
        let lattice = create_square_lattice(2.0);
        
        // Test origin
        let origin = lattice.cart_to_frac(Vector3::new(0.0, 0.0, 0.0));
        assert!(origin.norm() < TOL);
        
        // Test cartesian unit vectors
        let v1 = lattice.cart_to_frac(Vector3::new(2.0, 0.0, 0.0));
        assert!((v1 - Vector3::new(1.0, 0.0, 0.0)).norm() < TOL);
        
        let v2 = lattice.cart_to_frac(Vector3::new(0.0, 2.0, 0.0));
        assert!((v2 - Vector3::new(0.0, 1.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn test_frac_cart_roundtrip() {
        let lattice = create_hexagonal_lattice(1.0);
        
        let test_points = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.5, 0.5, 0.0),
            Vector3::new(-0.3, 0.7, 0.0),
            Vector3::new(2.5, -1.5, 0.0),
        ];
        
        for point in test_points {
            let cart = lattice.frac_to_cart(point);
            let frac_back = lattice.cart_to_frac(cart);
            assert!((frac_back - point).norm() < TOL);
        }
    }

    #[test]
    fn test_lattice_parameters() {
        // Square lattice
        let square = create_square_lattice(2.0);
        let (a, b) = square.lattice_parameters();
        assert!((a - 2.0).abs() < TOL);
        assert!((b - 2.0).abs() < TOL);
        
        // Rectangular lattice
        let rect = create_rectangular_lattice(2.0, 3.0);
        let (a, b) = rect.lattice_parameters();
        assert!((a - 2.0).abs() < TOL);
        assert!((b - 3.0).abs() < TOL);
        
        // Hexagonal lattice
        let hex = create_hexagonal_lattice(2.0);
        let (a, b) = hex.lattice_parameters();
        assert!((a - 2.0).abs() < TOL);
        assert!((b - 2.0).abs() < TOL);
    }

    #[test]
    fn test_lattice_angle() {
        // Square lattice - 90 degrees
        let square = create_square_lattice(1.0);
        let gamma = square.lattice_angle();
        assert!((gamma - PI / 2.0).abs() < TOL);
        
        // Hexagonal lattice - 120 degrees
        let hex = create_hexagonal_lattice(1.0);
        let gamma = hex.lattice_angle();
        assert!((gamma - 2.0 * PI / 3.0).abs() < TOL);
        
        // Oblique lattice - custom angle
        let angle = PI / 3.0; // 60 degrees
        let oblique = create_oblique_lattice(1.0, 1.0, angle);
        let gamma = oblique.lattice_angle();
        assert!((gamma - angle).abs() < TOL);
    }

    #[test]
    fn test_to_3d() {
        let lattice_2d = create_square_lattice(1.0);
        let c_vector = Vector3::new(0.0, 0.0, 2.0);
        let lattice_3d = lattice_2d.to_3d(c_vector);
        
        // Check that the first two basis vectors are preserved
        let (a_2d, b_2d) = lattice_2d.primitive_vectors();
        let direct_3d = lattice_3d.direct_basis();
        
        assert!((direct_3d.column(0) - a_2d).norm() < TOL);
        assert!((direct_3d.column(1) - b_2d).norm() < TOL);
        assert!((direct_3d.column(2) - c_vector).norm() < TOL);
    }

    #[test]
    fn test_primitive_vectors() {
        let lattice = create_rectangular_lattice(2.0, 3.0);
        let (a, b) = lattice.primitive_vectors();
        
        assert!((a - Vector3::new(2.0, 0.0, 0.0)).norm() < TOL);
        assert!((b - Vector3::new(0.0, 3.0, 0.0)).norm() < TOL);
    }

    #[test]
    fn test_reduce_to_brillouin_zone() {
        let lattice = create_square_lattice(1.0);
        
        // Point already in BZ
        let k1 = Vector3::new(0.0, 0.0, 0.0);
        let k1_reduced = lattice.reduce_to_brillouin_zone(k1);
        assert!((k1_reduced - k1).norm() < TOL);
        
        // Point outside BZ that should be folded back
        let k2 = Vector3::new(3.0 * PI, 0.0, 0.0);
        let k2_reduced = lattice.reduce_to_brillouin_zone(k2);
        assert!(k2_reduced[0].abs() < PI + TOL);
        
        // Point at BZ boundary
        let k3 = Vector3::new(PI, PI, 0.0);
        let k3_reduced = lattice.reduce_to_brillouin_zone(k3);
        assert!(lattice.in_brillouin_zone(k3_reduced));
    }

    #[test]
    fn test_in_brillouin_zone() {
        let lattice = create_square_lattice(1.0);
        
        // Center of BZ
        assert!(lattice.in_brillouin_zone(Vector3::new(0.0, 0.0, 0.0)));
        
        // Inside BZ
        assert!(lattice.in_brillouin_zone(Vector3::new(0.5 * PI, 0.5 * PI, 0.0)));
        
        // Outside BZ
        assert!(!lattice.in_brillouin_zone(Vector3::new(2.0 * PI, 0.0, 0.0)));
    }

    #[test]
    fn test_high_symmetry_points() {
        let lattice = create_square_lattice(1.0);
        let hs_points = lattice.get_high_symmetry_points_cartesian();
        
        // Check that we have some high symmetry points
        assert!(!hs_points.is_empty());
        
        // Check for Gamma point
        let gamma = hs_points.iter().find(|(label, _)| label == "Γ");
        assert!(gamma.is_some());
        let (_, gamma_pos) = gamma.unwrap();
        assert!(gamma_pos.norm() < TOL);
    }

    #[test]
    fn test_generate_k_path() {
        let lattice = create_square_lattice(1.0);
        let k_path = lattice.generate_k_path(10);
        
        // Should have multiple points
        assert!(k_path.len() > 20); // At least a few segments with 10 points each
        
        // First point should be at a high symmetry point
        let first = &k_path[0];
        let hs_points = lattice.get_high_symmetry_points_cartesian();
        let is_hs_point = hs_points.iter().any(|(_, pos)| (pos - first).norm() < TOL);
        assert!(is_hs_point);
    }

    #[test]
    fn test_edge_case_small_lattice() {
        let tiny = 1e-8;
        let lattice = create_square_lattice(tiny);
        
        assert_eq!(lattice.bravais_type(), Bravais2D::Square);
        assert!((lattice.cell_area() - tiny * tiny).abs() < TOL);
        
        // Reciprocal lattice should be very large
        let (a_recip, b_recip) = (
            lattice.reciprocal.column(0).norm(),
            lattice.reciprocal.column(1).norm()
        );
        assert!(a_recip > 1e6);
        assert!(b_recip > 1e6);
    }

    #[test]
    fn test_edge_case_large_lattice() {
        let huge = 1e8;
        let lattice = create_square_lattice(huge);
        
        assert_eq!(lattice.bravais_type(), Bravais2D::Square);
        assert!((lattice.cell_area() - huge * huge).abs() < huge * TOL);
        
        // Reciprocal lattice should be very small
        let (a_recip, b_recip) = (
            lattice.reciprocal.column(0).norm(),
            lattice.reciprocal.column(1).norm()
        );
        assert!(a_recip < 1e-6);
        assert!(b_recip < 1e-6);
    }

    #[test]
    fn test_edge_case_nearly_degenerate() {
        // Almost collinear vectors (very small angle)
        let a = 1.0;
        let b = 1.0;
        let gamma = 0.001; // Very small angle
        
        let lattice = create_oblique_lattice(a, b, gamma);
        
        // Should still identify as oblique
        assert_eq!(lattice.bravais_type(), Bravais2D::Oblique);
        
        // Area should be very small
        assert!(lattice.cell_area() < 0.01);
        
        // Angle should be preserved
        assert!((lattice.lattice_angle() - gamma).abs() < TOL);
    }

    #[test]
    #[should_panic(expected = "Direct basis must be invertible")]
    fn test_edge_case_singular_basis() {
        // Truly degenerate basis (collinear vectors)
        let direct = Matrix3::new(
            1.0, 2.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
        );
        let _ = Lattice2D::new(direct, TOL);
    }

    #[test]
    fn test_symmetry_operations() {
        // Square lattice symmetry operations
        let square = create_square_lattice(1.0);
        let num_square_ops = square.symmetry_operations().len();
        assert!(num_square_ops >= 4); // At least 4 operations
        
        // Hexagonal lattice symmetry operations
        let hex = create_hexagonal_lattice(1.0);
        let num_hex_ops = hex.symmetry_operations().len();
        assert!(num_hex_ops >= 3); // At least 3 operations
        
        // Rectangular lattice symmetry operations
        let rect = create_rectangular_lattice(1.0, 2.0);
        let num_rect_ops = rect.symmetry_operations().len();
        assert!(num_rect_ops >= 2); // At least 2 operations
    }

    #[test]
    fn test_wigner_seitz_cell() {
        let lattice = create_square_lattice(1.0);
        let ws_cell = lattice.wigner_seitz_cell();
        
        // Should have vertices
        assert!(!ws_cell.vertices().is_empty());
        
        // Center should be inside
        assert!(ws_cell.contains_2d(Vector3::new(0.0, 0.0, 0.0)));
        
        // Far away point should be outside
        assert!(!ws_cell.contains_2d(Vector3::new(10.0, 10.0, 0.0)));
        
        // Area should match unit cell area
        assert!((ws_cell.measure() - lattice.cell_area()).abs() < TOL);
    }

    #[test]
    fn test_brillouin_zone() {
        let lattice = create_square_lattice(1.0);
        let bz = lattice.brillouin_zone();
        
        // Should have vertices
        assert!(!bz.vertices().is_empty());
        
        // Gamma point should be inside
        assert!(bz.contains_2d(Vector3::new(0.0, 0.0, 0.0)));
        
        // Check that BZ has correct area
        let expected_area = (2.0 * PI).powi(2);
        assert!((bz.measure() - expected_area).abs() < 0.1);
    }
}
