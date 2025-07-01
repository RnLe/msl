#[cfg(test)]
mod tests {
    use super::super::geometry2d_bounding_box::BoundingBox2D;
    use nalgebra::Vector2;

    // Helper function for approximate float comparison
    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    fn assert_vector_approx_eq(v1: Vector2<f64>, v2: Vector2<f64>, epsilon: f64) {
        if !approx_eq(v1.x, v2.x, epsilon) || !approx_eq(v1.y, v2.y, epsilon) {
            panic!("Vectors not approximately equal: {:?} != {:?}", v1, v2);
        }
    }

    #[test]
    fn test_new_with_valid_bounds() {
        let bbox = BoundingBox2D::new(
            Vector2::new(-5.0, -10.0),
            Vector2::new(5.0, 10.0)
        );
        
        assert_vector_approx_eq(bbox.min, Vector2::new(-5.0, -10.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(5.0, 10.0), 1e-10);
    }

    #[test]
    #[should_panic(expected = "Minimum coordinates must be less than or equal to maximum coordinates")]
    #[cfg(debug_assertions)]
    fn test_new_with_invalid_bounds() {
        // This should panic in debug mode
        let _bbox = BoundingBox2D::new(
            Vector2::new(5.0, 10.0),
            Vector2::new(-5.0, -10.0)
        );
    }

    #[test]
    fn test_from_center_size_edge_cases() {
        // Test with zero size
        let bbox = BoundingBox2D::from_center_size(
            Vector2::new(3.0, 4.0),
            Vector2::new(0.0, 0.0)
        );
        assert_vector_approx_eq(bbox.min, Vector2::new(3.0, 4.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(3.0, 4.0), 1e-10);
        
        // Test with negative center
        let bbox2 = BoundingBox2D::from_center_size(
            Vector2::new(-3.0, -4.0),
            Vector2::new(2.0, 2.0)
        );
        assert_vector_approx_eq(bbox2.min, Vector2::new(-4.0, -5.0), 1e-10);
        assert_vector_approx_eq(bbox2.max, Vector2::new(-2.0, -3.0), 1e-10);
    }

    #[test]
    fn test_from_points_edge_cases() {
        // Single point
        let single_point = vec![Vector2::new(5.0, 7.0)];
        let bbox = BoundingBox2D::from_points(single_point).unwrap();
        assert_vector_approx_eq(bbox.min, Vector2::new(5.0, 7.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(5.0, 7.0), 1e-10);
        
        // Collinear points
        let collinear = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 0.0),
        ];
        let bbox2 = BoundingBox2D::from_points(collinear).unwrap();
        assert_eq!(bbox2.height(), 0.0, "Height should be 0 for collinear horizontal points");
        assert_eq!(bbox2.width(), 2.0, "Width should be 2.0");
        
        // Points with same coordinates
        let same_points = vec![
            Vector2::new(3.0, 3.0),
            Vector2::new(3.0, 3.0),
            Vector2::new(3.0, 3.0),
        ];
        let bbox3 = BoundingBox2D::from_points(same_points).unwrap();
        assert!(bbox3.is_empty(), "Bounding box of identical points should be empty");
    }

    #[test]
    fn test_contains_edge_cases() {
        let bbox = BoundingBox2D::new(
            Vector2::new(-1.0, -1.0),
            Vector2::new(1.0, 1.0)
        );
        
        // Test exact boundaries
        assert!(bbox.contains(Vector2::new(-1.0, -1.0)), "Should contain min corner");
        assert!(bbox.contains(Vector2::new(1.0, 1.0)), "Should contain max corner");
        assert!(bbox.contains(Vector2::new(-1.0, 1.0)), "Should contain top-left corner");
        assert!(bbox.contains(Vector2::new(1.0, -1.0)), "Should contain bottom-right corner");
        
        // Test edges
        assert!(bbox.contains(Vector2::new(0.0, -1.0)), "Should contain bottom edge point");
        assert!(bbox.contains(Vector2::new(0.0, 1.0)), "Should contain top edge point");
        assert!(bbox.contains(Vector2::new(-1.0, 0.0)), "Should contain left edge point");
        assert!(bbox.contains(Vector2::new(1.0, 0.0)), "Should contain right edge point");
        
        // Test just outside
        let epsilon = 1e-10;
        assert!(!bbox.contains(Vector2::new(-1.0 - epsilon, 0.0)), "Should not contain point just outside left");
        assert!(!bbox.contains(Vector2::new(1.0 + epsilon, 0.0)), "Should not contain point just outside right");
        assert!(!bbox.contains(Vector2::new(0.0, -1.0 - epsilon)), "Should not contain point just outside bottom");
        assert!(!bbox.contains(Vector2::new(0.0, 1.0 + epsilon)), "Should not contain point just outside top");
    }

    #[test]
    fn test_contains_box_edge_cases() {
        let outer = BoundingBox2D::new(
            Vector2::new(-2.0, -2.0),
            Vector2::new(2.0, 2.0)
        );
        
        // Test identical boxes
        let identical = BoundingBox2D::new(
            Vector2::new(-2.0, -2.0),
            Vector2::new(2.0, 2.0)
        );
        assert!(outer.contains_box(&identical), "Box should contain itself");
        
        // Test touching edges
        let touching = BoundingBox2D::new(
            Vector2::new(-2.0, -2.0),
            Vector2::new(0.0, 0.0)
        );
        assert!(outer.contains_box(&touching), "Should contain box touching edges");
        
        // Test slightly outside
        let outside = BoundingBox2D::new(
            Vector2::new(-2.1, -2.0),
            Vector2::new(2.0, 2.0)
        );
        assert!(!outer.contains_box(&outside), "Should not contain box extending outside");
    }

    #[test]
    fn test_intersects_special_cases() {
        let bbox1 = BoundingBox2D::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 2.0)
        );
        
        // Test touching corners
        let corner_touch = BoundingBox2D::new(
            Vector2::new(2.0, 2.0),
            Vector2::new(3.0, 3.0)
        );
        assert!(bbox1.intersects(&corner_touch), "Boxes touching at corner should intersect");
        
        // Test touching edges
        let edge_touch = BoundingBox2D::new(
            Vector2::new(2.0, 0.0),
            Vector2::new(3.0, 2.0)
        );
        assert!(bbox1.intersects(&edge_touch), "Boxes touching at edge should intersect");
        
        // Test one box inside another
        let inside = BoundingBox2D::new(
            Vector2::new(0.5, 0.5),
            Vector2::new(1.5, 1.5)
        );
        assert!(bbox1.intersects(&inside), "Contained box should intersect");
        assert!(inside.intersects(&bbox1), "Intersection should be symmetric");
    }

    #[test]
    fn test_intersection_edge_cases() {
        let bbox1 = BoundingBox2D::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 2.0)
        );
        
        // Test edge intersection
        let edge_box = BoundingBox2D::new(
            Vector2::new(2.0, 0.0),
            Vector2::new(4.0, 2.0)
        );
        let intersection = bbox1.intersection(&edge_box);
        assert!(intersection.is_some(), "Edge touching boxes should have intersection");
        let int = intersection.unwrap();
        assert!(int.is_empty(), "Edge intersection should be empty (zero width)");
        
        // Test corner intersection
        let corner_box = BoundingBox2D::new(
            Vector2::new(2.0, 2.0),
            Vector2::new(4.0, 4.0)
        );
        let corner_int = bbox1.intersection(&corner_box);
        assert!(corner_int.is_some(), "Corner touching boxes should have intersection");
        let cint = corner_int.unwrap();
        assert!(cint.is_empty(), "Corner intersection should be empty");
    }

    #[test]
    fn test_union_special_cases() {
        // Test union with empty box
        let normal = BoundingBox2D::new(
            Vector2::new(1.0, 1.0),
            Vector2::new(3.0, 3.0)
        );
        let empty = BoundingBox2D::new(
            Vector2::new(5.0, 5.0),
            Vector2::new(5.0, 5.0)
        );
        
        let union = normal.union(&empty);
        assert_vector_approx_eq(union.min, Vector2::new(1.0, 1.0), 1e-10);
        assert_vector_approx_eq(union.max, Vector2::new(5.0, 5.0), 1e-10);
        
        // Test union of identical boxes
        let union2 = normal.union(&normal);
        assert_vector_approx_eq(union2.min, normal.min, 1e-10);
        assert_vector_approx_eq(union2.max, normal.max, 1e-10);
    }

    #[test]
    fn test_properties_edge_cases() {
        // Test empty box
        let empty = BoundingBox2D::new(
            Vector2::new(5.0, 5.0),
            Vector2::new(5.0, 5.0)
        );
        assert_eq!(empty.width(), 0.0, "Empty box should have zero width");
        assert_eq!(empty.height(), 0.0, "Empty box should have zero height");
        assert_eq!(empty.area(), 0.0, "Empty box should have zero area");
        assert!(empty.aspect_ratio().is_infinite(), "Empty box should have infinite aspect ratio");
        assert!(empty.is_empty(), "Should correctly identify as empty");
        
        // Test zero height box
        let flat = BoundingBox2D::new(
            Vector2::new(0.0, 5.0),
            Vector2::new(10.0, 5.0)
        );
        assert_eq!(flat.height(), 0.0, "Flat box should have zero height");
        assert_eq!(flat.width(), 10.0, "Flat box should have correct width");
        assert!(flat.aspect_ratio().is_infinite(), "Flat box should have infinite aspect ratio");
        
        // Test negative dimensions (in release mode)
        #[cfg(not(debug_assertions))]
        {
            let invalid = BoundingBox2D {
                min: Vector2::new(5.0, 5.0),
                max: Vector2::new(0.0, 0.0),
            };
            assert!(!invalid.is_valid(), "Should detect invalid box");
            assert!(invalid.is_empty(), "Invalid box should be considered empty");
        }
    }

    #[test]
    fn test_expand_edge_cases() {
        let bbox = BoundingBox2D::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 1.0)
        );
        
        // Test zero expansion
        let expanded0 = bbox.expand(0.0);
        assert_vector_approx_eq(expanded0.min, bbox.min, 1e-10);
        assert_vector_approx_eq(expanded0.max, bbox.max, 1e-10);
        
        // Test negative expansion (shrinking)
        let shrunk = bbox.expand(-0.25);
        assert_vector_approx_eq(shrunk.min, Vector2::new(0.25, 0.25), 1e-10);
        assert_vector_approx_eq(shrunk.max, Vector2::new(0.75, 0.75), 1e-10);
        
        // Test shrinking beyond center
        let over_shrunk = bbox.expand(-1.0);
        assert!(over_shrunk.width() <= 0.0, "Over-shrunk box should have non-positive width");
        assert!(over_shrunk.is_empty(), "Over-shrunk box should be empty");
        
        // Test asymmetric expansion
        let asym_expanded = bbox.expand_by(Vector2::new(1.0, 2.0));
        assert_vector_approx_eq(asym_expanded.min, Vector2::new(-1.0, -2.0), 1e-10);
        assert_vector_approx_eq(asym_expanded.max, Vector2::new(2.0, 3.0), 1e-10);
    }

    #[test]
    fn test_default() {
        let default_bbox = BoundingBox2D::default();
        assert_vector_approx_eq(default_bbox.min, Vector2::zeros(), 1e-10);
        assert_vector_approx_eq(default_bbox.max, Vector2::zeros(), 1e-10);
        assert!(default_bbox.is_empty(), "Default box should be empty");
        assert!(default_bbox.is_valid(), "Default box should be valid");
    }

    #[test]
    fn test_numerical_precision() {
        // Test with very small values
        let tiny = BoundingBox2D::new(
            Vector2::new(1e-15, 1e-15),
            Vector2::new(2e-15, 2e-15)
        );
        assert!(tiny.width() > 0.0, "Should handle tiny positive dimensions");
        
        // Test with very large values
        let huge = BoundingBox2D::new(
            Vector2::new(-1e15, -1e15),
            Vector2::new(1e15, 1e15)
        );
        assert_eq!(huge.width(), 2e15, "Should handle large dimensions");
        assert_eq!(huge.area(), 4e30, "Should handle large area calculations");
        
        // Test near-zero differences
        let near_zero = BoundingBox2D::new(
            Vector2::new(1.0, 1.0),
            Vector2::new(1.0 + 1e-10, 1.0 + 1e-10)
        );
        assert!(!near_zero.is_empty(), "Near-zero dimensions should not be considered empty");
    }
}
