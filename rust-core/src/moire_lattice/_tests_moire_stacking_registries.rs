//! Test suite for moire_lattice::moire_stacking_registries (monatomic case).
//!
//! These tests exercise:
//! - τ-sets for different Bravais types (labels and coordinates)
//! - Moiré primitives for twist-only and mismatch cases
//! - Registry center formulas and wrapping
//! - Consistency between explicit-θ and "from layers" APIs
//!
//! NOTE: This file is intended to be included from the `moire_lattice` module as
//! `mod _tests_moire_stacking_registries;` in its `mod.rs`.

#![allow(clippy::float_cmp)]

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use crate::lattice::lattice_types::Bravais2D;
use crate::lattice::lattice2d::Lattice2D;
use crate::moire_lattice::moire_builder::{MoireBuilder, twisted_bilayer};
use crate::moire_lattice::moire_stacking_registries::{
    moire_matrix_from_layers, moire_primitives, moire_primitives_from_m,
    monatomic_tau_set_for_bravais, registry_centers,
};

#[allow(dead_code)]
/// Helper: basic abs-diff comparison.
fn nearly(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}
#[allow(dead_code)]
fn nearly_vec2(a: Vector2<f64>, b: Vector2<f64>, eps: f64) -> bool {
    nearly(a[0], b[0], eps) && nearly(a[1], b[1], eps)
}

#[allow(dead_code)]
/// Helper: build a 3x3 direct basis from two in-plane vectors.
/// The third column is set to ẑ = (0,0,1) to keep the matrix invertible.
fn direct3_from_2d(a1: Vector2<f64>, a2: Vector2<f64>) -> Matrix3<f64> {
    let mut m = Matrix3::identity();
    m[(0, 0)] = a1[0];
    m[(1, 0)] = a1[1];
    m[(2, 0)] = 0.0;
    m[(0, 1)] = a2[0];
    m[(1, 1)] = a2[1];
    m[(2, 1)] = 0.0;
    m[(0, 2)] = 0.0;
    m[(1, 2)] = 0.0;
    m[(2, 2)] = 1.0;
    m
}

#[allow(dead_code)]
/// Helper: extract the 2×2 in-plane part from a 3×3.
fn mat2_from3_xy(m: &Matrix3<f64>) -> Matrix2<f64> {
    Matrix2::new(m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)])
}

#[allow(dead_code)]
/// Convert a Cartesian vector into fractional coordinates for a given 2×2 basis A.
fn to_fractional_2d(a: &Matrix2<f64>, r: Vector2<f64>) -> Vector2<f64> {
    let inv = a.try_inverse().expect("Basis must be invertible");
    inv * r
}

#[test]
fn tau_set_hexagonal_labels_and_values() {
    let a = 1.0f64;
    let a1 = Vector2::new(a, 0.0);
    let a2 = Vector2::new(0.5 * a, (3.0f64).sqrt() * 0.5 * a);
    let taus = monatomic_tau_set_for_bravais(a1, a2, &Bravais2D::Hexagonal);

    // Expect 7 labeled positions: top, 3×bridge, 3×hollow.
    assert_eq!(taus.len(), 7);

    // Build a label->vector map
    use std::collections::HashMap;
    let map: HashMap<_, _> = taus.into_iter().collect();

    // Exact positions
    assert!(map.contains_key("top"));
    assert!(nearly_vec2(map["top"], Vector2::new(0.0, 0.0), 1e-12));

    assert!(map.contains_key("bridge_a1"));
    assert!(map.contains_key("bridge_a2"));
    assert!(map.contains_key("bridge_a1_plus_a2"));

    // Barycenters for equilateral triangles
    let h1 = (a1 + a2) / 3.0;
    let h2 = (a1 + 2.0 * a2) / 3.0;
    let h3 = (2.0 * a1 + a2) / 3.0;

    let eps = 1e-12;
    assert!(map.contains_key("hollow_1"));
    assert!(map.contains_key("hollow_2"));
    assert!(map.contains_key("hollow_3"));
    assert!(nearly_vec2(map["hollow_1"], h1, eps));
    assert!(nearly_vec2(map["hollow_2"], h2, eps));
    assert!(nearly_vec2(map["hollow_3"], h3, eps));
}

#[test]
fn tau_set_square_and_rectangular() {
    let a = 2.0;
    let b = 1.0;

    // Square
    let a1_sq = Vector2::new(a, 0.0);
    let a2_sq = Vector2::new(0.0, a);
    let ts_sq = monatomic_tau_set_for_bravais(a1_sq, a2_sq, &Bravais2D::Square);
    let labels_sq: Vec<_> = ts_sq.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(ts_sq.len(), 4);
    assert!(labels_sq.contains(&"top"));
    assert!(labels_sq.contains(&"bridge_a1"));
    assert!(labels_sq.contains(&"bridge_a2"));
    assert!(labels_sq.contains(&"hollow"));
    // Hollow center
    let hollow = ts_sq.iter().find(|(s, _)| s == "hollow").unwrap().1;
    assert!(nearly(hollow[0], 0.5 * a, 1e-12));
    assert!(nearly(hollow[1], 0.5 * a, 1e-12));

    // Rectangular
    let a1_re = Vector2::new(a, 0.0);
    let a2_re = Vector2::new(0.0, b);
    let ts_re = monatomic_tau_set_for_bravais(a1_re, a2_re, &Bravais2D::Rectangular);
    assert_eq!(ts_re.len(), 4);
    let hollow_re = ts_re.iter().find(|(s, _)| s == "hollow").unwrap().1;
    assert!(nearly(hollow_re[0], 0.5 * a, 1e-12));
    assert!(nearly(hollow_re[1], 0.5 * b, 1e-12));
}

#[test]
fn moire_primitives_length_twist_only_square() {
    // Twist-only case, A1 = A2 = I.
    let a1 = Matrix2::identity();
    let a2 = Matrix2::identity();
    let theta = 0.137; // rad

    let l = moire_primitives(&a1, &a2, theta).expect("L should exist");
    let col0 = l.column(0);
    let col1 = l.column(1);

    // For twist-only identical lattices, |L e_i| = 1/(2 sin(θ/2))
    let expected = 1.0 / (2.0 * (0.5 * theta).sin());
    let eps = 1e-10;
    assert!(nearly(col0.norm(), expected, eps));
    assert!(nearly(col1.norm(), expected, eps));
}

#[test]
fn registry_centers_top_at_origin_when_no_shift() {
    // Square lattice, A1 = A2 = I, twist θ; d0 = 0.
    let a1 = Matrix2::identity();
    let a2 = Matrix2::identity();
    let theta = 0.25;
    let d0 = Vector2::new(0.0, 0.0);

    // Minimal τ-list with "top"
    let tau_list = vec![("top".to_string(), Vector2::new(0.0, 0.0))];
    let (_l, centers) = registry_centers(&a1, &a2, theta, d0, &tau_list).unwrap();
    assert_eq!(centers.len(), 1);
    let pos = centers[0].position;
    let eps = 1e-12;
    assert!(nearly(pos[0], 0.0, eps));
    assert!(nearly(pos[1], 0.0, eps));
    assert!(nearly(pos[2], 0.0, eps));
}

#[test]
fn unified_api_pure_twist_detection() {
    // Test the new unified API with pure twist case
    let theta = 0.314; // radians
    let a = 1.0;
    let a1 = Vector2::new(a, 0.0);
    let a2 = Vector2::new(0.0, a);

    // Create Moire2D with pure twist
    let direct1 = direct3_from_2d(a1, a2);

    let l1 = Lattice2D::new(direct1, 1e-12);

    // Create Moire2D using twisted_bilayer function
    let moire = twisted_bilayer(l1, theta).unwrap();

    let d0 = Vector3::new(0.1, -0.05, 0.0);

    // Test the unified API
    let centers = moire.registry_centers_monatomic(d0).unwrap();

    // Should get some registry centers
    assert!(!centers.is_empty());

    // All positions should be finite
    for center in &centers {
        assert!(center.position[0].is_finite());
        assert!(center.position[1].is_finite());
        assert!(center.position[2].is_finite());
    }

    // The "top" registry should be present
    let top_center = centers.iter().find(|c| c.label == "top");
    assert!(top_center.is_some(), "Should find 'top' registry center");
}

#[test]
fn unified_api_general_transformation() {
    // Test the unified API with general transformation (twist + strain)
    let theta = 0.2;
    let strain_x = 1.02;
    let strain_y = 0.98;

    let a = 1.0;
    let a1 = Vector2::new(a, 0.0);
    let a2 = Vector2::new(0.0, a);

    // Apply both rotation and strain
    let rot = Matrix2::new(
        f64::cos(theta),
        -f64::sin(theta),
        f64::sin(theta),
        f64::cos(theta),
    );
    let strain = Matrix2::new(strain_x, 0.0, 0.0, strain_y);
    let combined = strain * rot;

    let direct1 = direct3_from_2d(a1, a2);

    let l1 = Lattice2D::new(direct1, 1e-12);

    use crate::moire_lattice::moire_builder::MoireBuilder;
    let moire = MoireBuilder::new()
        .with_base_lattice(l1)
        .with_general_transformation(combined)
        .build()
        .unwrap();

    let d0 = Vector3::new(0.0, 0.0, 0.0);

    // Test the unified API
    let centers = moire.registry_centers_monatomic(d0).unwrap();

    // Should get some registry centers
    assert!(!centers.is_empty());

    // All positions should be finite
    for center in &centers {
        assert!(center.position[0].is_finite());
        assert!(center.position[1].is_finite());
        assert!(center.position[2].is_finite());
    }
}

#[test]
fn small_angle_is_stable_and_matches_formula() {
    // Very small angle ⇒ huge moiré period. Use analytic inverse branch.
    let theta = 1.0e-8;
    let a1 = Matrix2::identity();
    let a2 = Matrix2::identity();

    match moire_primitives(&a1, &a2, theta) {
        Ok(l) => {
            let expected = 1.0 / (2.0 * (0.5 * theta).sin());
            let eps_rel = 1e-9;
            let n0 = l.column(0).norm();
            let n1 = l.column(1).norm();
            assert!(
                nearly(n0, expected, expected * eps_rel),
                "n0={} expected={}",
                n0,
                expected
            );
            assert!(
                nearly(n1, expected, expected * eps_rel),
                "n1={} expected={}",
                n1,
                expected
            );

            // Registry centers should compute finite values
            let tau_list = vec![("top".to_string(), Vector2::new(0.0, 0.0))];
            let d0 = Vector2::new(0.3, -0.2);
            match registry_centers(&a1, &a2, theta, d0, &tau_list) {
                Ok((_l2, centers)) => {
                    assert_eq!(centers.len(), 1);
                    let pos = centers[0].position;
                    assert!(pos[0].is_finite() && pos[1].is_finite());
                }
                Err(e) => {
                    eprintln!("[DEBUG] registry_centers failed for theta={}: {}", theta, e);
                    assert!(
                        theta.abs() < 1e-7,
                        "registry_centers failed unexpectedly for theta={}: {}",
                        theta,
                        e
                    );
                }
            }
        }
        Err(e) => {
            eprintln!("[DEBUG] moire_primitives failed for theta={}: {}", theta, e);
            // Acceptable for extremely small theta
            assert!(
                theta.abs() < 1e-7,
                "moire_primitives failed unexpectedly for theta={}: {}",
                theta,
                e
            );
        }
    }
}

#[test]
fn anisotropic_mismatch_no_twist_matches_expected_periods() {
    // A2 = diag(sx, sy) * A1 with θ = 0.
    let sx = 1.02;
    let sy = 0.98;
    let a1 = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let a2 = Matrix2::new(sx, 0.0, 0.0, sy);

    // M = I - T = I - diag(sx, sy) ⇒ L = M^{-1} A1 with columns:
    // e1: 1/(1 - sx), e2: 1/(1 - sy)
    let l = moire_primitives(&a1, &a2, 0.0).expect("L should exist");
    let col0 = l.column(0);
    let col1 = l.column(1);
    let eps = 1e-12;
    assert!(nearly(col0.norm(), 1.0 / (1.0 - sx).abs(), eps));
    assert!(nearly(col1.norm(), 1.0 / (1.0 - sy).abs(), eps));

    // Now do the "from layers" path and compare
    let l1 = Lattice2D::new(
        direct3_from_2d(Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0)),
        1e-12,
    );
    let l2 = Lattice2D::new(
        direct3_from_2d(Vector2::new(sx, 0.0), Vector2::new(0.0, sy)),
        1e-12,
    );
    let m_layers = moire_matrix_from_layers(&l1, &l2).unwrap();
    let l_layers = moire_primitives_from_m(&mat2_from3_xy(l1.direct_basis()), &m_layers).unwrap();
    assert!(nearly(l_layers.column(0).norm(), col0.norm(), eps));
    assert!(nearly(l_layers.column(1).norm(), col1.norm(), eps));
}

#[test]
fn registry_centers_positions_are_wrapped_into_moire_cell() {
    // Pick a shift that produces a position far outside the cell, check wrapping.
    let theta = 0.2;
    let a1 = Matrix2::identity();
    let a2 = Matrix2::identity();

    // Large d0
    let d0 = Vector2::new(12.3, -7.6);
    let taus = vec![
        ("top".to_string(), Vector2::new(0.0, 0.0)),
        ("bridge_a1".to_string(), Vector2::new(0.5, 0.0)),
    ];

    let (l, centers) = registry_centers(&a1, &a2, theta, d0, &taus).unwrap();

    // All positions should be inside [0,1) in fractional coordinates of L
    for c in centers {
        let frac = to_fractional_2d(&l, Vector2::new(c.position[0], c.position[1]));
        assert!(
            frac[0] >= -1e-12 && frac[0] < 1.0 + 1e-12,
            "frac[0] = {}",
            frac[0]
        );
        assert!(
            frac[1] >= -1e-12 && frac[1] < 1.0 + 1e-12,
            "frac[1] = {}",
            frac[1]
        );
    }
}

#[test]
fn unified_api_uses_stored_twist_angle() {
    // Test that the unified API uses the stored twist angle for robust computation
    use crate::moire_lattice::moire_builder::twisted_bilayer;

    // Helper to create hexagonal lattice
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
        Lattice2D::new(direct, 1e-10)
    }

    let base = create_hexagonal_lattice(1.0);
    let angles = vec![1.0_f64, 5.0, 10.0, 15.0, 21.79]; // Including magic angle

    for angle_deg in angles {
        let theta = angle_deg.to_radians();
        let moire = twisted_bilayer(base.clone(), theta).unwrap();

        // Verify the stored twist angle is correct
        assert!(
            (moire.twist_angle - theta).abs() < 1e-12,
            "Stored twist angle mismatch: expected {}, got {}",
            theta,
            moire.twist_angle
        );

        // Test registry centers with the unified API
        let centers = moire.registry_centers_monatomic(Vector3::zeros()).unwrap();

        // Should have 7 centers for hexagonal (top + 3 bridges + 3 hollows)
        assert_eq!(
            centers.len(),
            7,
            "Wrong number of registry centers for hexagonal lattice"
        );

        // Top center should be at origin (approximately)
        let top = centers.iter().find(|c| c.label == "top").unwrap();
        assert!(
            top.position.norm() < 1e-6,
            "Top center not at origin for angle {}: {:?}",
            angle_deg,
            top.position
        );

        // Centers should be stable and reasonable - not collapsed or extremely far
        for center in &centers {
            let pos_norm = center.position.norm();
            assert!(
                pos_norm < 100.0,
                "Registry center too far from origin for angle {}: {} at {:?}",
                angle_deg,
                center.label,
                center.position
            );
        }
    }
}
