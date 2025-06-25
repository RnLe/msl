#[cfg(test)]
mod tests_moire2d {
    use super::super::moire2d::{Moire2D, MoireTransformation};
    use crate::lattice::lattice_construction::{square_lattice, hexagonal_lattice};
    use nalgebra::{Matrix2, Vector3};
    use std::f64::consts::PI;

    // Helper function to create a simple test Moire2D
    fn create_test_moire() -> Moire2D {
        let lattice_1 = square_lattice(1.0);
        let lattice_2 = square_lattice(1.0);
        let transformation = MoireTransformation::RotationScale { 
            angle: PI / 6.0, 
            scale: 1.0 
        };
        
        // Create a basic Moire2D structure
        // Note: In real implementation, this would be done by the builder
        Moire2D {
            direct: lattice_1.direct,
            reciprocal: lattice_1.reciprocal,
            bravais: lattice_1.bravais,
            cell_area: lattice_1.cell_area * 4.0, // Approximate for test
            metric: lattice_1.metric,
            tol: 1e-10,
            sym_ops: lattice_1.sym_ops.clone(),
            wigner_seitz_cell: lattice_1.wigner_seitz_cell.clone(),
            brillouin_zone: lattice_1.brillouin_zone.clone(),
            high_symmetry: lattice_1.high_symmetry.clone(),
            lattice_1,
            lattice_2,
            transformation,
            twist_angle: PI / 6.0,
            is_commensurate: false,
            coincidence_indices: None,
        }
    }

    // ==================== MoireTransformation Tests ====================

    #[test]
    fn test_rotation_scale_to_matrix() {
        let transform = MoireTransformation::RotationScale { 
            angle: PI / 4.0, 
            scale: 2.0 
        };
        
        let mat = transform.to_matrix();
        let expected_cos = (PI / 4.0).cos();
        let expected_sin = (PI / 4.0).sin();
        
        assert!((mat[(0, 0)] - 2.0 * expected_cos).abs() < 1e-10,
            "Failed: mat[0,0] = {}, expected = {}", mat[(0, 0)], 2.0 * expected_cos);
        assert!((mat[(0, 1)] - (-2.0 * expected_sin)).abs() < 1e-10,
            "Failed: mat[0,1] = {}, expected = {}", mat[(0, 1)], -2.0 * expected_sin);
        assert!((mat[(1, 0)] - 2.0 * expected_sin).abs() < 1e-10,
            "Failed: mat[1,0] = {}, expected = {}", mat[(1, 0)], 2.0 * expected_sin);
        assert!((mat[(1, 1)] - 2.0 * expected_cos).abs() < 1e-10,
            "Failed: mat[1,1] = {}, expected = {}", mat[(1, 1)], 2.0 * expected_cos);
    }

    #[test]
    fn test_rotation_scale_edge_cases() {
        // Test zero angle
        let transform = MoireTransformation::RotationScale { angle: 0.0, scale: 1.5 };
        let mat = transform.to_matrix();
        assert!((mat[(0, 0)] - 1.5).abs() < 1e-10, "Zero angle rotation failed");
        assert!(mat[(0, 1)].abs() < 1e-10, "Zero angle should have no off-diagonal");
        
        // Test 180 degree rotation
        let transform = MoireTransformation::RotationScale { angle: PI, scale: 1.0 };
        let mat = transform.to_matrix();
        assert!((mat[(0, 0)] - (-1.0)).abs() < 1e-10, 
            "180 degree rotation failed: mat[0,0] = {}", mat[(0, 0)]);
        assert!(mat[(0, 1)].abs() < 1e-10, 
            "180 degree rotation has non-zero off-diagonal: {}", mat[(0, 1)]);
        
        // Test zero scale (degenerate case)
        let transform = MoireTransformation::RotationScale { angle: PI / 4.0, scale: 0.0 };
        let mat = transform.to_matrix();
        assert!(mat.norm() < 1e-10, "Zero scale should produce zero matrix");
    }

    #[test]
    fn test_anisotropic_scale_to_matrix() {
        let transform = MoireTransformation::AnisotropicScale { 
            scale_x: 2.0, 
            scale_y: 3.0 
        };
        
        let mat = transform.to_matrix();
        assert_eq!(mat[(0, 0)], 2.0, "scale_x not applied correctly");
        assert_eq!(mat[(1, 1)], 3.0, "scale_y not applied correctly");
        assert_eq!(mat[(0, 1)], 0.0, "Off-diagonal should be zero");
        assert_eq!(mat[(1, 0)], 0.0, "Off-diagonal should be zero");
    }

    #[test]
    fn test_shear_to_matrix() {
        let transform = MoireTransformation::Shear { 
            shear_x: 0.5, 
            shear_y: 0.3 
        };
        
        let mat = transform.to_matrix();
        assert_eq!(mat[(0, 0)], 1.0, "Diagonal should be 1 for shear");
        assert_eq!(mat[(1, 1)], 1.0, "Diagonal should be 1 for shear");
        assert_eq!(mat[(0, 1)], 0.5, "shear_x not applied correctly");
        assert_eq!(mat[(1, 0)], 0.3, "shear_y not applied correctly");
    }

    #[test]
    fn test_general_to_matrix() {
        let test_matrix = Matrix2::new(1.0, 2.0, 3.0, 4.0);
        let transform = MoireTransformation::General(test_matrix);
        
        let mat = transform.to_matrix();
        assert_eq!(mat, test_matrix, "General transformation not preserved");
    }

    #[test]
    fn test_to_matrix3() {
        let transform = MoireTransformation::RotationScale { 
            angle: PI / 3.0, 
            scale: 1.5 
        };
        
        let mat2 = transform.to_matrix();
        let mat3 = transform.to_matrix3();
        
        // Check 2x2 block is preserved
        assert_eq!(mat3[(0, 0)], mat2[(0, 0)], "2x2 block not preserved in 3x3");
        assert_eq!(mat3[(0, 1)], mat2[(0, 1)], "2x2 block not preserved in 3x3");
        assert_eq!(mat3[(1, 0)], mat2[(1, 0)], "2x2 block not preserved in 3x3");
        assert_eq!(mat3[(1, 1)], mat2[(1, 1)], "2x2 block not preserved in 3x3");
        
        // Check z-component is identity
        assert_eq!(mat3[(2, 2)], 1.0, "z-component should be 1");
        assert_eq!(mat3[(0, 2)], 0.0, "z-coupling should be 0");
        assert_eq!(mat3[(1, 2)], 0.0, "z-coupling should be 0");
        assert_eq!(mat3[(2, 0)], 0.0, "z-coupling should be 0");
        assert_eq!(mat3[(2, 1)], 0.0, "z-coupling should be 0");
    }

    // ==================== Moire2D Tests ====================

    #[test]
    fn test_as_lattice2d() {
        let moire = create_test_moire();
        let lattice = moire.as_lattice2d();
        
        assert_eq!(lattice.direct, moire.direct, "Direct basis not preserved");
        assert_eq!(lattice.reciprocal, moire.reciprocal, "Reciprocal basis not preserved");
        assert_eq!(lattice.bravais, moire.bravais, "Bravais type not preserved");
        assert_eq!(lattice.cell_area, moire.cell_area, "Cell area not preserved");
        assert_eq!(lattice.tol, moire.tol, "Tolerance not preserved");
    }

    #[test]
    fn test_primitive_vectors() {
        let moire = create_test_moire();
        let (a1, a2) = moire.primitive_vectors();
        
        let expected_a1: Vector3<f64> = moire.direct.column(0).into();
        let expected_a2: Vector3<f64> = moire.direct.column(1).into();
        
        assert_eq!(a1, expected_a1, "First primitive vector incorrect");
        assert_eq!(a2, expected_a2, "Second primitive vector incorrect");
        
        // Test orthogonality for square lattice
        if matches!(moire.bravais, crate::lattice::lattice_bravais_types::Bravais2D::Square) {
            let dot = a1.dot(&a2);
            assert!(dot.abs() < 1e-10, 
                "Square lattice vectors should be orthogonal, dot product = {}", dot);
        }
    }

    #[test]
    fn test_moire_period_ratio() {
        let moire = create_test_moire();
        let ratio = moire.moire_period_ratio();
        
        assert!(ratio > 0.0, "Period ratio must be positive");
        assert!(ratio.is_finite(), "Period ratio must be finite");
        
        // For our test case with 4x larger area
        let expected_ratio = 2.0;
        assert!((ratio - expected_ratio).abs() < 1e-10,
            "Period ratio = {}, expected = {}", ratio, expected_ratio);
    }

    #[test]
    fn test_is_lattice_point() {
        let moire = create_test_moire();
        
        // Test origin (should be on both lattices)
        let origin = Vector3::new(0.0, 0.0, 0.0);
        assert!(moire.is_lattice1_point(origin), 
            "Origin should be on lattice 1");
        assert!(moire.is_lattice2_point(origin), 
            "Origin should be on lattice 2");
        
        // Test lattice 1 point
        let a1 = moire.lattice_1.direct.column(0).into();
        assert!(moire.is_lattice1_point(a1), 
            "Lattice 1 basis vector should be on lattice 1");
        
        // Test non-lattice point
        let random_point = Vector3::new(0.123, 0.456, 0.0);
        assert!(!moire.is_lattice1_point(random_point), 
            "Random point should not be on lattice 1");
        assert!(!moire.is_lattice2_point(random_point), 
            "Random point should not be on lattice 2");
    }

    #[test]
    fn test_is_lattice_point_edge_cases() {
        let moire = create_test_moire();
        
        // Test point very close to lattice point (within tolerance)
        let a1: Vector3<f64> = moire.lattice_1.direct.column(0).into();
        let close_point = a1 + Vector3::new(moire.tol * 0.5, 0.0, 0.0);
        assert!(moire.is_lattice1_point(close_point), 
            "Point within tolerance should be considered on lattice");
        
        // Test point just outside tolerance
        let far_point = a1 + Vector3::new(moire.tol * 2.0, 0.0, 0.0);
        assert!(!moire.is_lattice1_point(far_point), 
            "Point outside tolerance should not be on lattice");
    }

    #[test]
    fn test_get_stacking_at() {
        let moire = create_test_moire();
        
        // Test origin (AA stacking)
        let origin = Vector3::new(0.0, 0.0, 0.0);
        let stacking = moire.get_stacking_at(origin);
        assert_eq!(stacking, Some("AA".to_string()), 
            "Origin should have AA stacking, got {:?}", stacking);
        
        // Test random point (no stacking)
        let random = Vector3::new(0.123, 0.456, 0.0);
        let stacking = moire.get_stacking_at(random);
        assert_eq!(stacking, None, 
            "Random point should have no stacking, got {:?}", stacking);
        
        // Test lattice 1 only point
        // This is tricky without proper moire construction, but we can test the logic
        let mut test_moire = moire.clone();
        // Artificially offset lattice 2
        test_moire.lattice_2.direct[(0, 0)] += 0.5;
        test_moire.lattice_2.direct[(1, 1)] += 0.5;
        
        let a1: Vector3<f64> = test_moire.lattice_1.direct.column(0).into();
        let stacking = test_moire.get_stacking_at(a1);
        if !test_moire.is_lattice2_point(a1) {
            assert_eq!(stacking, Some("A".to_string()), 
                "Lattice 1 only point should have A stacking");
        }
    }

    #[test]
    fn test_hexagonal_moire() {
        // Test with hexagonal lattices
        let lattice_1 = hexagonal_lattice(1.0);
        let lattice_2 = hexagonal_lattice(1.0);
        let transformation = MoireTransformation::RotationScale { 
            angle: PI / 30.0,  // Small angle for near-commensurate
            scale: 1.0 
        };
        
        let moire = Moire2D {
            direct: lattice_1.direct,
            reciprocal: lattice_1.reciprocal,
            bravais: lattice_1.bravais,
            cell_area: lattice_1.cell_area * 100.0, // Approximate
            metric: lattice_1.metric,
            tol: 1e-10,
            sym_ops: lattice_1.sym_ops.clone(),
            wigner_seitz_cell: lattice_1.wigner_seitz_cell.clone(),
            brillouin_zone: lattice_1.brillouin_zone.clone(),
            high_symmetry: lattice_1.high_symmetry.clone(),
            lattice_1,
            lattice_2,
            transformation,
            twist_angle: PI / 30.0,
            is_commensurate: false,
            coincidence_indices: None,
        };
        
        // Test that primitive vectors are extracted correctly
        let (a1, a2) = moire.primitive_vectors();
        assert!(a1.norm() > 0.0, "Primitive vector should have non-zero length");
        assert!(a2.norm() > 0.0, "Primitive vector should have non-zero length");
        
        // For hexagonal, check 120 degree angle
        let dot = a1.dot(&a2);
        let angle = (dot / (a1.norm() * a2.norm())).acos();
        let expected_angle = 2.0 * PI / 3.0; // 120 degrees
        assert!((angle - expected_angle).abs() < 0.1 || (angle - PI / 3.0).abs() < 0.1,
            "Hexagonal lattice should have 60 or 120 degree angles, got {} degrees", 
            angle * 180.0 / PI);
    }

    #[test]
    fn test_commensurate_moire() {
        let moire = create_test_moire();
        
        // Create a commensurate version
        let mut comm_moire = moire.clone();
        comm_moire.is_commensurate = true;
        comm_moire.coincidence_indices = Some((3, 1, -1, 3));
        
        assert!(comm_moire.is_commensurate, "Should be marked as commensurate");
        assert!(comm_moire.coincidence_indices.is_some(), 
            "Commensurate moire should have coincidence indices");
        
        let indices = comm_moire.coincidence_indices.unwrap();
        assert_eq!(indices, (3, 1, -1, 3), "Coincidence indices not preserved");
    }
}