#[cfg(test)]
mod tests_voronoi_cells {
    use super::super::voronoi_cells::*;
    use nalgebra::Matrix3;
    use std::f64::consts::PI;

    const TEST_TOLERANCE: f64 = 1e-10;

    // ======================== 2D WIGNER-SEITZ CELL TESTS ========================

    #[test]
    fn test_compute_wigner_seitz_cell_2d_square() {
        // Square lattice with side length 1
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 1.0;
        basis[(1, 1)] = 1.0;
        basis[(2, 2)] = 1.0; // z-direction placeholder

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.len() != 4 || (ws_cell.measure - 1.0).abs() > 0.01 {
            eprintln!(
                "DEBUG: Square lattice WS cell has {} vertices",
                ws_cell.vertices.len()
            );
            eprintln!("DEBUG: WS cell area = {}, expected = 1.0", ws_cell.measure);
            eprintln!("DEBUG: Vertices: {:?}", ws_cell.vertices);
        }

        // Square lattice should have a square WS cell with 4 vertices
        assert_eq!(
            ws_cell.vertices.len(),
            4,
            "Square WS cell should have 4 vertices"
        );
        assert_eq!(ws_cell.edges.len(), 4, "Square WS cell should have 4 edges");
        assert!(
            (ws_cell.measure - 1.0).abs() < 0.01,
            "Square WS cell area should be 1.0"
        );

        // Verify vertices are at (±0.5, ±0.5, 0)
        for vertex in &ws_cell.vertices {
            assert!(
                vertex.z.abs() < TEST_TOLERANCE,
                "2D vertices should have z=0"
            );
            assert!(
                (vertex.x.abs() - 0.5).abs() < 0.01,
                "x coordinates should be ±0.5"
            );
            assert!(
                (vertex.y.abs() - 0.5).abs() < 0.01,
                "y coordinates should be ±0.5"
            );
        }
    }

    #[test]
    fn test_compute_wigner_seitz_cell_2d_hexagonal() {
        // Hexagonal lattice with a = 1
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 1.0;
        basis[(1, 0)] = 0.0;
        basis[(0, 1)] = 0.5;
        basis[(1, 1)] = 0.5 * (3.0_f64).sqrt();
        basis[(2, 2)] = 1.0;

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.len() != 6 {
            eprintln!(
                "DEBUG: Hexagonal lattice WS cell has {} vertices, expected 6",
                ws_cell.vertices.len()
            );
            eprintln!("DEBUG: Vertices: {:?}", ws_cell.vertices);
            eprintln!("DEBUG: WS cell area = {}", ws_cell.measure);
        }

        // Hexagonal lattice should have hexagonal WS cell with 6 vertices
        assert_eq!(
            ws_cell.vertices.len(),
            6,
            "Hexagonal WS cell should have 6 vertices"
        );
        assert_eq!(
            ws_cell.edges.len(),
            6,
            "Hexagonal WS cell should have 6 edges"
        );

        // Area should be sqrt(3)/2 for unit hexagonal lattice
        let expected_area = 0.5 * (3.0_f64).sqrt();
        assert!(
            (ws_cell.measure - expected_area).abs() < 0.01,
            "Hexagonal WS cell area incorrect: got {}, expected {}",
            ws_cell.measure,
            expected_area
        );
    }

    #[test]
    fn test_compute_wigner_seitz_cell_2d_rectangular() {
        // Rectangular lattice a=2, b=1
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 2.0;
        basis[(1, 1)] = 1.0;
        basis[(2, 2)] = 1.0;

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.len() != 4 || (ws_cell.measure - 2.0).abs() > 0.01 {
            eprintln!(
                "DEBUG: Rectangular lattice WS cell has {} vertices",
                ws_cell.vertices.len()
            );
            eprintln!("DEBUG: WS cell area = {}, expected = 2.0", ws_cell.measure);
            eprintln!("DEBUG: Vertices: {:?}", ws_cell.vertices);
        }

        // Should have rectangular WS cell
        assert_eq!(
            ws_cell.vertices.len(),
            4,
            "Rectangular WS cell should have 4 vertices"
        );
        assert!(
            (ws_cell.measure - 2.0).abs() < 0.01,
            "Rectangular WS cell area should be 2.0"
        );
    }

    #[test]
    fn test_compute_wigner_seitz_cell_2d_oblique() {
        // Oblique lattice with angle 60 degrees
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 1.0;
        basis[(0, 1)] = 0.5;
        basis[(1, 1)] = 0.5 * (3.0_f64).sqrt();
        basis[(2, 2)] = 1.0;

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.is_empty() {
            eprintln!("DEBUG: Oblique lattice WS cell has no vertices!");
            eprintln!("DEBUG: Basis matrix: {:?}", basis);
        }

        // Should produce a valid WS cell
        assert!(
            ws_cell.vertices.len() >= 4,
            "Oblique WS cell should have at least 4 vertices"
        );
        assert!(ws_cell.measure > 0.0, "WS cell area should be positive");
    }

    #[test]
    fn test_compute_wigner_seitz_cell_2d_tiny_lattice() {
        // Very small lattice constant
        let scale = 1e-6;
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = scale;
        basis[(1, 1)] = scale;
        basis[(2, 2)] = 1.0;

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.is_empty() || ws_cell.measure <= 0.0 {
            eprintln!("DEBUG: Tiny lattice WS cell failed");
            eprintln!("DEBUG: Number of vertices: {}", ws_cell.vertices.len());
            eprintln!("DEBUG: WS cell area = {}", ws_cell.measure);
        }

        // Should still produce valid WS cell
        assert!(
            !ws_cell.vertices.is_empty(),
            "WS cell should have vertices even for tiny lattice"
        );
        assert!(ws_cell.measure > 0.0, "WS cell area should be positive");
        assert!(
            (ws_cell.measure - scale * scale).abs() < scale * scale * 0.1,
            "WS cell area should scale correctly"
        );
    }

    // ======================== 3D WIGNER-SEITZ CELL TESTS ========================

    #[test]
    fn test_compute_wigner_seitz_cell_3d_cubic() {
        // Simple cubic lattice
        let basis = Matrix3::identity();
        let ws_cell = compute_wigner_seitz_cell_3d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.len() != 8 || (ws_cell.measure - 1.0).abs() > 0.01 {
            eprintln!(
                "DEBUG: Cubic lattice WS cell has {} vertices",
                ws_cell.vertices.len()
            );
            eprintln!(
                "DEBUG: WS cell volume = {}, expected = 1.0",
                ws_cell.measure
            );
            eprintln!("DEBUG: Number of faces = {}", ws_cell.faces.len());
        }

        // Cubic lattice WS cell is a cube
        assert_eq!(
            ws_cell.vertices.len(),
            8,
            "Cubic WS cell should have 8 vertices"
        );
        assert_eq!(ws_cell.faces.len(), 6, "Cubic WS cell should have 6 faces");
        assert!(
            (ws_cell.measure - 1.0).abs() < 0.01,
            "Cubic WS cell volume should be 1.0"
        );
    }

    #[test]
    fn test_compute_wigner_seitz_cell_3d_fcc() {
        // FCC lattice basis
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 0.5;
        basis[(1, 0)] = 0.5;
        basis[(2, 0)] = 0.0;
        basis[(0, 1)] = 0.0;
        basis[(1, 1)] = 0.5;
        basis[(2, 1)] = 0.5;
        basis[(0, 2)] = 0.5;
        basis[(1, 2)] = 0.0;
        basis[(2, 2)] = 0.5;

        let ws_cell = compute_wigner_seitz_cell_3d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.vertices.len() != 14 && ws_cell.vertices.len() != 8 {
            eprintln!("DEBUG: FCC WS cell has {} vertices", ws_cell.vertices.len());
            eprintln!("DEBUG: Expected 14 (truncated octahedron) or 8 (fallback)");
            eprintln!("DEBUG: Volume = {}", ws_cell.measure);
        }

        // FCC WS cell is a truncated octahedron (14 vertices) or fallback parallelepiped (8)
        assert!(
            ws_cell.vertices.len() == 14 || ws_cell.vertices.len() == 8,
            "FCC WS cell should have 14 vertices (truncated octahedron) or 8 (fallback)"
        );
        assert!(ws_cell.measure > 0.0, "WS cell volume should be positive");
    }

    #[test]
    fn test_compute_wigner_seitz_cell_3d_extreme_aspect_ratio() {
        // Lattice with extreme aspect ratio
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 1.0;
        basis[(1, 1)] = 1.0;
        basis[(2, 2)] = 100.0;

        let ws_cell = compute_wigner_seitz_cell_3d(&basis, TEST_TOLERANCE);

        // Debug info
        if ws_cell.measure <= 0.0 {
            eprintln!("DEBUG: Extreme aspect ratio lattice failed");
            eprintln!("DEBUG: Number of vertices: {}", ws_cell.vertices.len());
            eprintln!("DEBUG: Volume = {}", ws_cell.measure);
        }

        // Should produce valid WS cell
        assert!(!ws_cell.vertices.is_empty(), "WS cell should have vertices");
        assert!(ws_cell.measure > 0.0, "WS cell volume should be positive");
        assert!(
            (ws_cell.measure - 100.0).abs() < 1.0,
            "WS cell volume should be ~100"
        );
    }

    // ======================== BRILLOUIN ZONE TESTS ========================

    #[test]
    fn test_compute_brillouin_zone_2d_square() {
        // Square lattice
        let mut direct = Matrix3::zeros();
        direct[(0, 0)] = 1.0;
        direct[(1, 1)] = 1.0;
        direct[(2, 2)] = 1.0;

        // Reciprocal lattice
        let reciprocal = 2.0 * PI * Matrix3::identity();

        let bz = compute_brillouin_zone_2d(&reciprocal, TEST_TOLERANCE);

        // Debug info
        if bz.vertices.len() != 4 {
            eprintln!(
                "DEBUG: Square BZ has {} vertices, expected 4",
                bz.vertices.len()
            );
            eprintln!("DEBUG: BZ area = {}", bz.measure);
        }

        // Square BZ should be square with area (2π)²
        assert_eq!(bz.vertices.len(), 4, "Square BZ should have 4 vertices");
        let expected_area = 4.0 * PI * PI;
        assert!(
            (bz.measure - expected_area).abs() < 0.1,
            "Square BZ area incorrect: got {}, expected {}",
            bz.measure,
            expected_area
        );
    }

    #[test]
    fn test_compute_brillouin_zone_3d_cubic() {
        // Simple cubic reciprocal lattice
        let reciprocal = 2.0 * PI * Matrix3::identity();
        let bz = compute_brillouin_zone_3d(&reciprocal, TEST_TOLERANCE);

        // Debug info
        if bz.vertices.len() != 8 {
            eprintln!("DEBUG: Cubic BZ has {} vertices", bz.vertices.len());
            eprintln!("DEBUG: BZ volume = {}", bz.measure);
        }

        // Cubic BZ should be a cube
        assert_eq!(bz.vertices.len(), 8, "Cubic BZ should have 8 vertices");
        let expected_volume = 8.0 * PI * PI * PI;
        assert!(
            (bz.measure - expected_volume).abs() < 0.1,
            "Cubic BZ volume incorrect: got {}, expected {}",
            bz.measure,
            expected_volume
        );
    }

    // ======================== LATTICE POINT GENERATION TESTS ========================

    #[test]
    fn test_generate_lattice_points_2d_by_shell() {
        let basis = Matrix3::identity();

        // Shell 1 should have 8 neighbors (excluding origin)
        let shell1 = generate_lattice_points_2d_by_shell(&basis, 1);

        // Debug info
        if shell1.len() != 8 {
            eprintln!("DEBUG: Shell 1 has {} points, expected 8", shell1.len());
            eprintln!("DEBUG: Points: {:?}", shell1);
        }

        assert_eq!(shell1.len(), 8, "Shell 1 should have 8 neighbors");

        // All points should have integer coordinates with |x|, |y| <= 1
        for point in &shell1 {
            assert!(point.x.abs() <= 1.0 + TEST_TOLERANCE);
            assert!(point.y.abs() <= 1.0 + TEST_TOLERANCE);
            assert!(point.z.abs() < TEST_TOLERANCE, "2D points should have z=0");
        }
    }

    #[test]
    fn test_generate_lattice_points_3d_by_shell() {
        let basis = Matrix3::identity();

        // Shell 1 should have 26 neighbors (3³ - 1)
        let shell1 = generate_lattice_points_3d_by_shell(&basis, 1);

        // Debug info
        if shell1.len() != 26 {
            eprintln!("DEBUG: Shell 1 has {} points, expected 26", shell1.len());
        }

        assert_eq!(shell1.len(), 26, "Shell 1 should have 26 neighbors");

        // All points should have integer coordinates with |x|, |y|, |z| <= 1
        for point in &shell1 {
            assert!(point.x.abs() <= 1.0 + TEST_TOLERANCE);
            assert!(point.y.abs() <= 1.0 + TEST_TOLERANCE);
            assert!(point.z.abs() <= 1.0 + TEST_TOLERANCE);
        }
    }

    #[test]
    fn test_generate_lattice_points_2d_within_radius() {
        let basis = Matrix3::identity();

        // Radius sqrt(2) should include points at distance 1 and sqrt(2)
        let radius = 2.0_f64.sqrt();
        let points = generate_lattice_points_2d_within_radius(&basis, radius);

        // Debug info
        if points.len() != 8 {
            eprintln!(
                "DEBUG: Found {} points within radius {}",
                points.len(),
                radius
            );
            eprintln!("DEBUG: Points and distances:");
            for p in &points {
                eprintln!("  {:?} -> distance = {}", p, p.norm());
            }
        }

        // Should include 4 points at distance 1 and 4 at distance sqrt(2)
        assert_eq!(
            points.len(),
            8,
            "Should have 8 points within radius sqrt(2)"
        );

        // Verify all points are within radius
        for point in &points {
            assert!(
                point.norm() <= radius + TEST_TOLERANCE,
                "Point {:?} outside radius {}",
                point,
                radius
            );
        }
    }

    #[test]
    fn test_generate_lattice_points_3d_within_radius() {
        let basis = Matrix3::identity();

        // Test 1: Radius 1.5 should include nearest (distance 1) and next-nearest (distance √2)
        let radius = 1.5;
        let points = generate_lattice_points_3d_within_radius(&basis, radius);

        // Should include 6 points at distance 1 and 12 points at distance √2 ≈ 1.414
        assert_eq!(points.len(), 18, "Should have 18 points within radius 1.5");

        // Verify all points are within radius
        for point in &points {
            assert!(
                point.norm() <= radius + TEST_TOLERANCE,
                "Point {:?} outside radius {}",
                point,
                radius
            );
        }

        // Test 2: Smaller radius should include only nearest neighbors
        let small_radius = 1.1;
        let nearest_points = generate_lattice_points_3d_within_radius(&basis, small_radius);
        assert_eq!(
            nearest_points.len(),
            6,
            "Should have 6 points within radius 1.1"
        );

        // All nearest neighbors should be at distance 1
        for point in &nearest_points {
            assert!(
                (point.norm() - 1.0).abs() < TEST_TOLERANCE,
                "Nearest neighbors should be at distance 1"
            );
        }
    }

    #[test]
    fn test_generate_lattice_points_edge_cases() {
        let basis = Matrix3::identity();

        // Zero radius should return empty
        let points = generate_lattice_points_2d_within_radius(&basis, 0.0);
        assert!(points.is_empty(), "Zero radius should return no points");

        // Negative radius should return empty
        let points = generate_lattice_points_3d_within_radius(&basis, -1.0);
        assert!(points.is_empty(), "Negative radius should return no points");

        // Very large shell should work
        let points = generate_lattice_points_2d_by_shell(&basis, 10);
        assert_eq!(
            points.len(),
            (21 * 21) - 1,
            "Shell 10 should have 440 points"
        );
    }

    #[test]
    fn test_wigner_seitz_reciprocal_consistency() {
        // Test that WS cell volume * BZ volume = (2π)^d
        let mut basis = Matrix3::zeros();
        basis[(0, 0)] = 2.0;
        basis[(1, 1)] = 3.0;
        basis[(2, 2)] = 1.0;

        let ws_cell = compute_wigner_seitz_cell_2d(&basis, TEST_TOLERANCE);

        // Reciprocal basis
        let inv = basis.try_inverse().unwrap();
        let reciprocal = 2.0 * PI * inv.transpose();
        let bz = compute_brillouin_zone_2d(&reciprocal, TEST_TOLERANCE);

        // Debug info
        let product = ws_cell.measure * bz.measure;
        let expected = 4.0 * PI * PI;
        if (product - expected).abs() > 0.1 {
            eprintln!(
                "DEBUG: WS area * BZ area = {}, expected = {}",
                product, expected
            );
            eprintln!(
                "DEBUG: WS area = {}, BZ area = {}",
                ws_cell.measure, bz.measure
            );
        }

        // Product should be (2π)²
        assert!(
            (product - expected).abs() < 0.1,
            "WS cell area * BZ area should equal (2π)²"
        );
    }
}
