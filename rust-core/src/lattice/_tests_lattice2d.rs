#[cfg(test)]
mod _tests_lattice2d {
    use super::super::lattice_types::Bravais2D;
    use super::super::lattice2d::Lattice2D;
    use nalgebra::{Matrix3, Vector3};
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // Helper function to create a square lattice
    fn create_square_lattice(a: f64) -> Lattice2D {
        let direct = Matrix3::new(a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, 1.0);
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create a hexagonal lattice
    fn create_hexagonal_lattice(a: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a,
            -a * 0.5,
            0.0,
            0.0,
            a * (3.0_f64).sqrt() / 2.0,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create a rectangular lattice
    fn create_rectangular_lattice(a: f64, b: f64) -> Lattice2D {
        let direct = Matrix3::new(a, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, 1.0);
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create an oblique lattice
    fn create_oblique_lattice(a: f64, b: f64, gamma: f64) -> Lattice2D {
        let direct = Matrix3::new(
            a,
            b * gamma.cos(),
            0.0,
            0.0,
            b * gamma.sin(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        Lattice2D::new(direct, TOL)
    }

    // Helper function to create a centered rectangular lattice
    fn create_centered_rectangular_lattice(a: f64, b: f64) -> Lattice2D {
        // For centered rectangular: a != b, and centering introduces additional lattice points
        // Use a lattice with basis vectors that create the centered rectangular symmetry
        let direct = Matrix3::new(a * 0.5, -a * 0.5, 0.0, b * 0.5, b * 0.5, 0.0, 0.0, 0.0, 1.0);
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
            lattice.reciprocal.column(1).norm(),
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
            lattice.reciprocal.column(1).norm(),
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
        let direct = Matrix3::new(1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
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

    #[test]
    fn test_direct_lattice_points_in_rectangle_square() {
        let lattice = create_square_lattice(1.0);

        // Small rectangle
        let points = lattice.get_direct_lattice_points_in_rectangle(2.0, 2.0);

        // Should contain origin
        assert!(points.iter().any(|p| p.norm() < TOL));

        // Should contain the 4 nearest neighbors
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(-1.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, 1.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, -1.0, 0.0)).norm() < TOL)
        );

        // Should contain diagonal corner points (on boundary of 2x2 rectangle)
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, 1.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(-1.0, -1.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, -1.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(-1.0, 1.0, 0.0)).norm() < TOL)
        );

        // Should NOT contain points clearly outside (like (2,0) or (0,2))
        assert!(
            !points
                .iter()
                .any(|p| (p - Vector3::new(2.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            !points
                .iter()
                .any(|p| (p - Vector3::new(0.0, 2.0, 0.0)).norm() < TOL)
        );

        // Larger rectangle (3x3)
        let points_large = lattice.get_direct_lattice_points_in_rectangle(3.0, 3.0);

        // For a 3x3 rectangle, the boundary is at ±1.5, so (2,0) is still outside
        // but we should have more points than the 2x2 case
        assert!(points_large.len() >= points.len());

        // Try an even larger rectangle (5x5) to include (2,0)
        let points_xl = lattice.get_direct_lattice_points_in_rectangle(5.0, 5.0);
        assert!(
            points_xl
                .iter()
                .any(|p| (p - Vector3::new(2.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points_large
                .iter()
                .any(|p| (p - Vector3::new(-1.0, -1.0, 0.0)).norm() < TOL)
        );

        // Count should be reasonable
        assert!(points_large.len() >= 9); // At least 3x3 grid
    }

    #[test]
    fn test_direct_lattice_points_in_rectangle_rectangular() {
        let lattice = create_rectangular_lattice(1.0, 2.0);

        // Rectangle 3x5
        let points = lattice.get_direct_lattice_points_in_rectangle(3.0, 5.0);

        // Check some expected points
        assert!(points.iter().any(|p| p.norm() < TOL)); // origin
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, 2.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, -2.0, 0.0)).norm() < TOL)
        );

        // Should contain (1, 2) but not (2, 2)
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, 2.0, 0.0)).norm() < TOL)
        );
        assert!(
            !points
                .iter()
                .any(|p| (p - Vector3::new(2.0, 2.0, 0.0)).norm() < TOL)
        );
    }

    #[test]
    fn test_direct_lattice_points_in_rectangle_hexagonal() {
        let lattice = create_hexagonal_lattice(1.0);

        // Rectangle that should capture hexagonal symmetry
        let points = lattice.get_direct_lattice_points_in_rectangle(3.0, 3.0);

        // Should contain origin
        assert!(points.iter().any(|p| p.norm() < TOL));

        // Check for 6-fold symmetry around origin
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(-0.5, (3.0_f64).sqrt() / 2.0, 0.0);

        // First shell of neighbors
        assert!(points.iter().any(|p| (p - a1).norm() < TOL));
        assert!(points.iter().any(|p| (p - a2).norm() < TOL));
        assert!(points.iter().any(|p| (p - (a2 - a1)).norm() < TOL));
        assert!(points.iter().any(|p| (p + a1).norm() < TOL));
        assert!(points.iter().any(|p| (p + a2).norm() < TOL));
        assert!(points.iter().any(|p| (p + (a2 - a1)).norm() < TOL));
    }

    #[test]
    fn test_reciprocal_lattice_points_in_rectangle() {
        let lattice = create_square_lattice(1.0);

        // Rectangle in reciprocal space
        let width = 4.0 * PI;
        let height = 4.0 * PI;
        let points = lattice.get_reciprocal_lattice_points_in_rectangle(width, height);

        // Should contain reciprocal lattice origin
        assert!(points.iter().any(|p| p.norm() < TOL));

        // Should contain first reciprocal lattice vectors
        let b1 = 2.0 * PI;
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(b1, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, b1, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(-b1, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, -b1, 0.0)).norm() < TOL)
        );
    }

    #[test]
    fn test_lattice_points_edge_case_empty_rectangle() {
        let lattice = create_square_lattice(1.0);

        // Zero width
        let points1 = lattice.get_direct_lattice_points_in_rectangle(0.0, 10.0);
        assert_eq!(points1.len(), 0);

        // Zero height
        let points2 = lattice.get_direct_lattice_points_in_rectangle(10.0, 0.0);
        assert_eq!(points2.len(), 0);

        // Both zero
        let points3 = lattice.get_direct_lattice_points_in_rectangle(0.0, 0.0);
        assert_eq!(points3.len(), 0);

        // Negative dimensions (treated as zero)
        let points4 = lattice.get_direct_lattice_points_in_rectangle(-5.0, 5.0);
        assert_eq!(points4.len(), 0);
    }

    #[test]
    fn test_lattice_points_edge_case_tiny_rectangle() {
        let lattice = create_square_lattice(1.0);

        // Rectangle smaller than unit cell
        let points = lattice.get_direct_lattice_points_in_rectangle(0.5, 0.5);

        // Should only contain origin
        assert_eq!(points.len(), 1);
        assert!(points[0].norm() < TOL);
    }

    #[test]
    fn test_lattice_points_edge_case_exact_boundaries() {
        let lattice = create_square_lattice(1.0);

        // Rectangle with width/height exactly at lattice points
        // Due to tolerance, points on boundary should be included
        let points = lattice.get_direct_lattice_points_in_rectangle(2.0, 2.0);

        // Points at ±1 should be included due to tolerance
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(1.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(-1.0, 0.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, 1.0, 0.0)).norm() < TOL)
        );
        assert!(
            points
                .iter()
                .any(|p| (p - Vector3::new(0.0, -1.0, 0.0)).norm() < TOL)
        );
    }

    #[test]
    fn test_lattice_points_large_rectangle() {
        let lattice = create_square_lattice(0.5);

        // Large rectangle
        let points = lattice.get_direct_lattice_points_in_rectangle(10.0, 10.0);

        // Should have many points (roughly (10/0.5)^2 = 400)
        assert!(points.len() > 300);
        assert!(points.len() < 500);

        // Check that all points are within rectangle
        for p in &points {
            assert!(p.x.abs() <= 5.0 + TOL);
            assert!(p.y.abs() <= 5.0 + TOL);
            assert!(p.z.abs() < TOL); // z should be zero for 2D lattice
        }
    }

    #[test]
    fn test_lattice_points_asymmetric_rectangle() {
        let lattice = create_square_lattice(1.0);

        // Very wide but short rectangle
        let points_wide = lattice.get_direct_lattice_points_in_rectangle(10.0, 1.0);

        // Should have points along x-axis
        for i in -5..=5 {
            assert!(
                points_wide
                    .iter()
                    .any(|p| (p - Vector3::new(i as f64, 0.0, 0.0)).norm() < TOL)
            );
        }

        // Should NOT have points far from x-axis
        assert!(!points_wide.iter().any(|p| p.y.abs() > 1.0));

        // Very tall but narrow rectangle
        let points_tall = lattice.get_direct_lattice_points_in_rectangle(1.0, 10.0);

        // Should have points along y-axis
        for i in -5..=5 {
            assert!(
                points_tall
                    .iter()
                    .any(|p| (p - Vector3::new(0.0, i as f64, 0.0)).norm() < TOL)
            );
        }

        // Should NOT have points far from y-axis
        assert!(!points_tall.iter().any(|p| p.x.abs() > 1.0));
    }

    #[test]
    fn test_lattice_points_oblique_lattice() {
        // 60 degree angle
        let lattice = create_oblique_lattice(1.0, 1.0, PI / 3.0);

        let points = lattice.get_direct_lattice_points_in_rectangle(3.0, 3.0);

        // Should contain origin
        assert!(points.iter().any(|p| p.norm() < TOL));

        // Should contain basis vectors
        let a1 = Vector3::new(1.0, 0.0, 0.0);
        let a2 = Vector3::new(0.5, (3.0_f64).sqrt() / 2.0, 0.0);

        assert!(points.iter().any(|p| (p - a1).norm() < TOL));
        assert!(points.iter().any(|p| (p - a2).norm() < TOL));

        // All points should be within rectangle
        for p in &points {
            assert!(p.x.abs() <= 1.5 + TOL);
            assert!(p.y.abs() <= 1.5 + TOL);
        }
    }

    #[test]
    fn test_reciprocal_lattice_points_hexagonal() {
        let lattice = create_hexagonal_lattice(1.0);

        // Get reciprocal lattice points
        let width = 8.0 * PI;
        let height = 8.0 * PI;
        let points = lattice.get_reciprocal_lattice_points_in_rectangle(width, height);

        // Should have reciprocal lattice vectors
        let b1: Vector3<f64> = lattice.reciprocal_basis().column(0).into();
        let b2: Vector3<f64> = lattice.reciprocal_basis().column(1).into();

        assert!(points.iter().any(|p| p.norm() < TOL)); // origin
        assert!(points.iter().any(|p| (p - b1).norm() < TOL));
        assert!(points.iter().any(|p| (p - b2).norm() < TOL));
        assert!(points.iter().any(|p| (p + b1).norm() < TOL));
        assert!(points.iter().any(|p| (p + b2).norm() < TOL));
    }

    #[test]
    fn test_lattice_points_consistency() {
        let lattice = create_rectangular_lattice(1.5, 2.5);

        // Same rectangle dimensions should give same number of points
        let points1 = lattice.get_direct_lattice_points_in_rectangle(5.0, 7.0);
        let points2 = lattice.get_direct_lattice_points_in_rectangle(5.0, 7.0);

        assert_eq!(points1.len(), points2.len());

        // Check that all points from first call are in second call
        for p1 in &points1 {
            assert!(points2.iter().any(|p2| (p1 - p2).norm() < TOL));
        }
    }

    #[test]
    fn test_wigner_seitz_vertex_counts() {
        // Square lattice should have 4 vertices
        let square = create_square_lattice(1.0);
        let ws_square = square.wigner_seitz_cell();
        println!("Square lattice vertices ({}): ", ws_square.vertices().len());
        for (i, v) in ws_square.vertices().iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }
        assert_eq!(
            ws_square.vertices().len(),
            4,
            "Square lattice WS cell should have 4 vertices, found {}",
            ws_square.vertices().len()
        );

        // Rectangular lattice should have 4 vertices
        let rect = create_rectangular_lattice(1.0, 2.0);
        let ws_rect = rect.wigner_seitz_cell();
        println!(
            "Rectangular lattice vertices ({}): ",
            ws_rect.vertices().len()
        );
        for (i, v) in ws_rect.vertices().iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }
        assert_eq!(
            ws_rect.vertices().len(),
            4,
            "Rectangular lattice WS cell should have 4 vertices, found {}",
            ws_rect.vertices().len()
        );

        // Hexagonal lattice should have 6 vertices
        let hex = create_hexagonal_lattice(1.0);
        let ws_hex = hex.wigner_seitz_cell();
        println!("Hexagonal lattice vertices ({}): ", ws_hex.vertices().len());
        for (i, v) in ws_hex.vertices().iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }
        assert_eq!(
            ws_hex.vertices().len(),
            6,
            "Hexagonal lattice WS cell should have 6 vertices, found {}",
            ws_hex.vertices().len()
        );

        // Centered rectangular lattice should have 6 vertices
        let cent_rect = create_centered_rectangular_lattice(1.0, 2.0);
        let ws_cent_rect = cent_rect.wigner_seitz_cell();
        println!(
            "Centered rectangular lattice vertices ({}): ",
            ws_cent_rect.vertices().len()
        );
        for (i, v) in ws_cent_rect.vertices().iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }
        assert_eq!(
            ws_cent_rect.vertices().len(),
            6,
            "Centered rectangular lattice WS cell should have 6 vertices, found {}",
            ws_cent_rect.vertices().len()
        );
    }

    #[test]
    fn test_hexagonal_wigner_seitz_vertices_exact() {
        let hex = create_hexagonal_lattice(1.0);
        let ws_cell = hex.wigner_seitz_cell();
        let vertices = ws_cell.vertices();

        // Expected vertices for hexagonal lattice with a = 1.0
        let sqrt3_inv = 1.0 / (3.0_f64).sqrt();
        let expected_vertices = vec![
            Vector3::new(0.0, -sqrt3_inv, 0.0),        // v0: (0.000, -1/√3)
            Vector3::new(0.5, -sqrt3_inv * 0.5, 0.0),  // v1: (+½, -1/(2√3))
            Vector3::new(0.5, sqrt3_inv * 0.5, 0.0),   // v2: (+½, +1/(2√3))
            Vector3::new(0.0, sqrt3_inv, 0.0),         // v3: (0.000, +1/√3)
            Vector3::new(-0.5, sqrt3_inv * 0.5, 0.0),  // v4: (-½, +1/(2√3))
            Vector3::new(-0.5, -sqrt3_inv * 0.5, 0.0), // v5: (-½, -1/(2√3))
        ];

        assert_eq!(
            vertices.len(),
            6,
            "Hexagonal WS cell should have exactly 6 vertices"
        );

        // Debug print current vertices to help identify the issue
        println!("Current vertices:");
        for (i, v) in vertices.iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }

        println!("Expected vertices:");
        for (i, v) in expected_vertices.iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }

        // Check that each expected vertex is present in the calculated vertices
        // Allow for some reordering and numerical tolerance
        for (i, expected) in expected_vertices.iter().enumerate() {
            let found = vertices.iter().any(|v| {
                (v[0] - expected[0]).abs() < 1e-6
                    && (v[1] - expected[1]).abs() < 1e-6
                    && v[2].abs() < 1e-10 // z should be essentially zero
            });
            assert!(
                found,
                "Expected vertex v{}: ({:.6}, {:.6}) not found in calculated vertices",
                i, expected[0], expected[1]
            );
        }

        // Also check that we don't have extra vertices
        for (i, vertex) in vertices.iter().enumerate() {
            let found = expected_vertices.iter().any(|expected| {
                (vertex[0] - expected[0]).abs() < 1e-6 && (vertex[1] - expected[1]).abs() < 1e-6
            });
            assert!(
                found,
                "Unexpected vertex found at index {}: ({:.6}, {:.6})",
                i, vertex[0], vertex[1]
            );
        }
    }

    #[test]
    fn test_square_wigner_seitz_vertices_exact() {
        let square = create_square_lattice(1.0);
        let ws_cell = square.wigner_seitz_cell();
        let vertices = ws_cell.vertices();

        // Expected vertices for unit square lattice WS cell
        let expected_vertices = vec![
            Vector3::new(-0.5, -0.5, 0.0), // bottom-left
            Vector3::new(0.5, -0.5, 0.0),  // bottom-right
            Vector3::new(0.5, 0.5, 0.0),   // top-right
            Vector3::new(-0.5, 0.5, 0.0),  // top-left
        ];

        assert_eq!(
            vertices.len(),
            4,
            "Square WS cell should have exactly 4 vertices"
        );

        // Debug print vertices
        println!("Square lattice vertices:");
        for (i, v) in vertices.iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }

        // Check vertices (allowing for reordering)
        for (i, expected) in expected_vertices.iter().enumerate() {
            let found = vertices.iter().any(|v| {
                (v[0] - expected[0]).abs() < 1e-10
                    && (v[1] - expected[1]).abs() < 1e-10
                    && v[2].abs() < 1e-10
            });
            assert!(
                found,
                "Expected vertex v{}: ({:.3}, {:.3}) not found in calculated vertices",
                i, expected[0], expected[1]
            );
        }
    }

    #[test]
    fn test_rectangular_wigner_seitz_vertices_exact() {
        let rect = create_rectangular_lattice(2.0, 1.0); // a=2, b=1
        let ws_cell = rect.wigner_seitz_cell();
        let vertices = ws_cell.vertices();

        // Expected vertices for 2x1 rectangular lattice WS cell
        let expected_vertices = vec![
            Vector3::new(-1.0, -0.5, 0.0), // bottom-left
            Vector3::new(1.0, -0.5, 0.0),  // bottom-right
            Vector3::new(1.0, 0.5, 0.0),   // top-right
            Vector3::new(-1.0, 0.5, 0.0),  // top-left
        ];

        assert_eq!(
            vertices.len(),
            4,
            "Rectangular WS cell should have exactly 4 vertices"
        );

        // Debug print vertices
        println!("Rectangular lattice vertices:");
        for (i, v) in vertices.iter().enumerate() {
            println!("  v{}: [{:8.3}, {:8.3}]", i, v[0], v[1]);
        }

        // Check vertices (allowing for reordering)
        for (i, expected) in expected_vertices.iter().enumerate() {
            let found = vertices.iter().any(|v| {
                (v[0] - expected[0]).abs() < 1e-10
                    && (v[1] - expected[1]).abs() < 1e-10
                    && v[2].abs() < 1e-10
            });
            assert!(
                found,
                "Expected vertex v{}: ({:.3}, {:.3}) not found in calculated vertices",
                i, expected[0], expected[1]
            );
        }
    }

    #[test]
    fn test_wigner_seitz_cell_properties() {
        // Test that WS cells have proper geometric properties

        // Square lattice
        let square = create_square_lattice(1.0);
        let ws_square = square.wigner_seitz_cell();
        assert!(
            (ws_square.measure() - 1.0).abs() < TOL,
            "Square WS cell area should be 1.0"
        );

        // Hexagonal lattice
        let hex = create_hexagonal_lattice(1.0);
        let ws_hex = hex.wigner_seitz_cell();
        let expected_hex_area = (3.0_f64).sqrt() / 2.0;
        assert!(
            (ws_hex.measure() - expected_hex_area).abs() < TOL,
            "Hexagonal WS cell area should be √3/2 ≈ {:.6}",
            expected_hex_area
        );

        // Rectangular lattice
        let rect = create_rectangular_lattice(2.0, 1.5);
        let ws_rect = rect.wigner_seitz_cell();
        assert!(
            (ws_rect.measure() - 3.0).abs() < TOL,
            "Rectangular WS cell area should be 3.0"
        );
    }

    #[test]
    fn test_wigner_seitz_contains_origin() {
        // All WS cells should contain the origin
        let lattices = vec![
            create_square_lattice(1.0),
            create_hexagonal_lattice(1.0),
            create_rectangular_lattice(1.0, 2.0),
            create_centered_rectangular_lattice(1.0, 2.0),
        ];

        for lattice in lattices {
            let ws_cell = lattice.wigner_seitz_cell();
            assert!(
                ws_cell.contains_2d(Vector3::new(0.0, 0.0, 0.0)),
                "WS cell should contain the origin for {:?} lattice",
                lattice.bravais_type()
            );
        }
    }

    #[test]
    fn test_wigner_seitz_symmetry() {
        // Test symmetry properties of WS cells

        // Square lattice: should have 4-fold rotational symmetry
        let square = create_square_lattice(1.0);
        let ws_square = square.wigner_seitz_cell();
        let test_point = Vector3::new(0.3, 0.2, 0.0);
        let rotated_90 = Vector3::new(-0.2, 0.3, 0.0); // 90° rotation
        let rotated_180 = Vector3::new(-0.3, -0.2, 0.0); // 180° rotation
        let rotated_270 = Vector3::new(0.2, -0.3, 0.0); // 270° rotation

        let in_original = ws_square.contains_2d(test_point);
        assert_eq!(
            ws_square.contains_2d(rotated_90),
            in_original,
            "Square WS cell should have 90° rotational symmetry"
        );
        assert_eq!(
            ws_square.contains_2d(rotated_180),
            in_original,
            "Square WS cell should have 180° rotational symmetry"
        );
        assert_eq!(
            ws_square.contains_2d(rotated_270),
            in_original,
            "Square WS cell should have 270° rotational symmetry"
        );

        // Hexagonal lattice: should have 6-fold rotational symmetry
        let hex = create_hexagonal_lattice(1.0);
        let ws_hex = hex.wigner_seitz_cell();
        let test_point_hex = Vector3::new(0.2, 0.1, 0.0);
        let in_hex_original = ws_hex.contains_2d(test_point_hex);

        // Test 60° rotation (rotation by π/3)
        let cos60 = 0.5;
        let sin60 = (3.0_f64).sqrt() / 2.0;
        let rotated_60 = Vector3::new(
            test_point_hex[0] * cos60 - test_point_hex[1] * sin60,
            test_point_hex[0] * sin60 + test_point_hex[1] * cos60,
            0.0,
        );
        assert_eq!(
            ws_hex.contains_2d(rotated_60),
            in_hex_original,
            "Hexagonal WS cell should have 60° rotational symmetry"
        );
    }

    #[test]
    fn test_lattice_points_no_duplicates() {
        let lattice = create_square_lattice(1.0);

        let points = lattice.get_direct_lattice_points_in_rectangle(5.0, 5.0);

        // Check for duplicates
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert!(
                    (points[i] - points[j]).norm() > TOL,
                    "Found duplicate points at indices {} and {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_debug_hexagonal_voronoi_computation() {
        use crate::lattice::voronoi_cells::generate_lattice_points_2d_by_shell;

        let hex = create_hexagonal_lattice(1.0);
        let basis = hex.direct_basis();

        println!("Hexagonal lattice basis vectors:");
        println!(
            "  a1: [{:.6}, {:.6}, {:.6}]",
            basis[(0, 0)],
            basis[(1, 0)],
            basis[(2, 0)]
        );
        println!(
            "  a2: [{:.6}, {:.6}, {:.6}]",
            basis[(0, 1)],
            basis[(1, 1)],
            basis[(2, 1)]
        );

        // Generate neighbors for shells 1, 2, 3
        for shell in 1..=3 {
            let neighbors = generate_lattice_points_2d_by_shell(basis, shell);
            println!("Shell {} neighbors ({} total):", shell, neighbors.len());
            for (i, neighbor) in neighbors.iter().enumerate() {
                let norm = neighbor.norm();
                println!(
                    "    neighbor[{}]: [{:8.6}, {:8.6}, {:8.6}] norm={:.6}",
                    i, neighbor[0], neighbor[1], neighbor[2], norm
                );
            }
        }

        // Test the Wigner-Seitz computation manually
        let ws_cell = hex.wigner_seitz_cell();
        println!("Final WS cell vertices ({}):", ws_cell.vertices().len());
        for (i, v) in ws_cell.vertices().iter().enumerate() {
            println!("  v{}: [{:8.6}, {:8.6}, {:8.6}]", i, v[0], v[1], v[2]);
        }
    }

    #[test]
    fn test_debug_brillouin_zone_vertices() {
        let hex = create_hexagonal_lattice(1.0);

        println!("=== DIRECT LATTICE ===");
        let ws_cell = hex.wigner_seitz_cell();
        println!(
            "Direct lattice WS cell vertices ({}):",
            ws_cell.vertices().len()
        );
        for (i, v) in ws_cell.vertices().iter().enumerate() {
            println!("  v{}: [{:8.6}, {:8.6}, {:8.6}]", i, v[0], v[1], v[2]);
        }
        println!("Direct lattice area: {:.6}", ws_cell.measure());

        println!("\n=== RECIPROCAL LATTICE ===");
        let bz = hex.brillouin_zone();
        println!("Brillouin zone vertices ({}):", bz.vertices().len());
        for (i, v) in bz.vertices().iter().enumerate() {
            println!("  v{}: [{:8.6}, {:8.6}, {:8.6}]", i, v[0], v[1], v[2]);
        }
        println!("Brillouin zone area: {:.6}", bz.measure());

        println!("\n=== BASIS VECTORS ===");
        let direct_basis = hex.direct_basis();
        let reciprocal_basis = hex.reciprocal_basis();

        println!("Direct basis vectors:");
        println!(
            "  a1: [{:.6}, {:.6}, {:.6}]",
            direct_basis[(0, 0)],
            direct_basis[(1, 0)],
            direct_basis[(2, 0)]
        );
        println!(
            "  a2: [{:.6}, {:.6}, {:.6}]",
            direct_basis[(0, 1)],
            direct_basis[(1, 1)],
            direct_basis[(2, 1)]
        );

        println!("Reciprocal basis vectors:");
        println!(
            "  b1: [{:.6}, {:.6}, {:.6}]",
            reciprocal_basis[(0, 0)],
            reciprocal_basis[(1, 0)],
            reciprocal_basis[(2, 0)]
        );
        println!(
            "  b2: [{:.6}, {:.6}, {:.6}]",
            reciprocal_basis[(0, 1)],
            reciprocal_basis[(1, 1)],
            reciprocal_basis[(2, 1)]
        );
    }
}
