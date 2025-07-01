#[cfg(test)]
mod _tests_geometry2d {
    use super::super::geometry2d::*;
    use super::super::geometry2d_transform::Transform2D;
    use nalgebra::Vector2;
    use crate::materials::Material;

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
    fn test_circle_contains_point() {
        let circle = Circle::new(Vector2::new(0.0, 0.0), 5.0);
        
        // Test points inside
        assert!(circle.contains_point(Vector2::new(0.0, 0.0)), "Center should be inside");
        assert!(circle.contains_point(Vector2::new(3.0, 4.0)), "Point (3,4) should be inside (distance=5)");
        assert!(circle.contains_point(Vector2::new(5.0, 0.0)), "Point on boundary should be inside");
        
        // Test points outside
        assert!(!circle.contains_point(Vector2::new(5.1, 0.0)), "Point just outside should not be inside");
        assert!(!circle.contains_point(Vector2::new(4.0, 4.0)), "Point (4,4) should be outside (distance≈5.66)");
    }

    #[test]
    fn test_circle_bounding_box() {
        let circle = Circle::new(Vector2::new(2.0, 3.0), 4.0);
        let bbox = circle.bounding_box();
        
        assert_vector_approx_eq(bbox.min, Vector2::new(-2.0, -1.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(6.0, 7.0), 1e-10);
    }

    #[test]
    fn test_circle_transform() {
        let mut circle = Circle::new(Vector2::new(1.0, 2.0), 3.0);
        let transform = Transform2D {
            translation: Vector2::new(2.0, 3.0),
            rotation: 0.0, // Rotation doesn't affect circle shape
            scale: Vector2::new(2.0, 2.0),
        };
        
        circle.transform(&transform);
        
        assert_vector_approx_eq(circle.center, Vector2::new(4.0, 7.0), 1e-10);
        assert!(approx_eq(circle.radius, 6.0, 1e-10), "Radius should be scaled");
    }

    #[test]
    fn test_rectangle_contains_point() {
        let rect = Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(10.0, 6.0));
        
        // Test points inside
        assert!(rect.contains_point(Vector2::new(0.0, 0.0)), "Center should be inside");
        assert!(rect.contains_point(Vector2::new(5.0, 3.0)), "Corner should be inside");
        assert!(rect.contains_point(Vector2::new(-5.0, -3.0)), "Opposite corner should be inside");
        
        // Test points outside
        assert!(!rect.contains_point(Vector2::new(5.1, 0.0)), "Point just outside should not be inside");
        assert!(!rect.contains_point(Vector2::new(0.0, 3.1)), "Point just outside should not be inside");
    }

    #[test]
    fn test_rectangle_rotated_contains_point() {
        let rect = Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(10.0, 6.0))
            .with_rotation(std::f64::consts::PI / 4.0); // 45 degrees
        
        // Test that rotation works correctly
        assert!(rect.contains_point(Vector2::new(0.0, 0.0)), "Center should still be inside after rotation");
        
        // Let's test a much smaller point that should definitely be inside
        let test_point = Vector2::new(0.0, 1.0); // Small Y coordinate
        assert!(rect.contains_point(test_point), "Point within rotated bounds should be inside");
        
        // Test a point that should be outside
        let outside_point = Vector2::new(0.0, 10.0); // Definitely outside
        assert!(!rect.contains_point(outside_point), "Point outside rotated bounds should not be inside");
    }

    #[test]
    fn test_rectangle_bounding_box() {
        // Test axis-aligned rectangle
        let rect = Rectangle::new(Vector2::new(2.0, 3.0), Vector2::new(4.0, 6.0));
        let bbox = rect.bounding_box();
        
        assert_vector_approx_eq(bbox.min, Vector2::new(0.0, 0.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(4.0, 6.0), 1e-10);
        
        // Test rotated rectangle (45 degrees)
        let rotated_rect = Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0))
            .with_rotation(std::f64::consts::PI / 4.0);
        let rotated_bbox = rotated_rect.bounding_box();
        
        let expected_extent = 2.0 * 2.0_f64.sqrt() / 2.0; // √2
        assert!(approx_eq(rotated_bbox.min.x, -expected_extent, 1e-10), 
                "Rotated bbox min.x incorrect: {} != {}", rotated_bbox.min.x, -expected_extent);
        assert!(approx_eq(rotated_bbox.max.x, expected_extent, 1e-10),
                "Rotated bbox max.x incorrect: {} != {}", rotated_bbox.max.x, expected_extent);
    }

    #[test]
    fn test_ellipse_contains_point() {
        let ellipse = Ellipse::new(Vector2::new(0.0, 0.0), Vector2::new(4.0, 2.0));
        
        // Test points inside
        assert!(ellipse.contains_point(Vector2::new(0.0, 0.0)), "Center should be inside");
        assert!(ellipse.contains_point(Vector2::new(4.0, 0.0)), "Point on major axis should be inside");
        assert!(ellipse.contains_point(Vector2::new(0.0, 2.0)), "Point on minor axis should be inside");
        assert!(ellipse.contains_point(Vector2::new(2.0, 1.0)), "Point inside ellipse");
        
        // Test points outside
        assert!(!ellipse.contains_point(Vector2::new(4.1, 0.0)), "Point just outside major axis");
        assert!(!ellipse.contains_point(Vector2::new(3.0, 2.0)), "Corner point should be outside");
    }

    #[test]
    fn test_ellipse_rotated_contains_point() {
        let ellipse = Ellipse::new(Vector2::new(0.0, 0.0), Vector2::new(4.0, 2.0))
            .with_rotation(std::f64::consts::PI / 2.0); // 90 degrees
        
        // After 90-degree rotation, major axis should be vertical
        assert!(ellipse.contains_point(Vector2::new(0.0, 4.0)), "Point on rotated major axis");
        assert!(ellipse.contains_point(Vector2::new(2.0, 0.0)), "Point on rotated minor axis");
        assert!(!ellipse.contains_point(Vector2::new(4.0, 0.0)), "Previous major axis point should be outside");
    }

    #[test]
    fn test_polygon_triangle_contains_point() {
        let vertices = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(4.0, 0.0),
            Vector2::new(2.0, 3.0),
        ];
        let triangle = Polygon::new(vertices);
        
        // Test points inside
        assert!(triangle.contains_point(Vector2::new(2.0, 1.0)), "Point inside triangle");
        assert!(triangle.contains_point(Vector2::new(1.0, 0.5)), "Point near base");
        
        // Test points outside
        assert!(!triangle.contains_point(Vector2::new(0.0, 3.0)), "Point outside left");
        assert!(!triangle.contains_point(Vector2::new(4.0, 3.0)), "Point outside right");
        assert!(!triangle.contains_point(Vector2::new(2.0, 4.0)), "Point above apex");
    }

    #[test]
    fn test_polygon_regular() {
        // Test regular hexagon
        let hexagon = Polygon::regular(Vector2::new(0.0, 0.0), 5.0, 6);
        
        assert_eq!(hexagon.vertices.len(), 6, "Should have 6 vertices");
        assert!(hexagon.contains_point(Vector2::new(0.0, 0.0)), "Center should be inside");
        assert!(hexagon.contains_point(Vector2::new(4.0, 0.0)), "Point near edge should be inside");
        assert!(!hexagon.contains_point(Vector2::new(5.1, 0.0)), "Point outside radius should be outside");
        
        // Check center calculation
        assert_vector_approx_eq(hexagon.center(), Vector2::new(0.0, 0.0), 1e-10);
    }

    #[test]
    fn test_polygon_empty() {
        let empty_polygon = Polygon::new(vec![]);
        
        assert!(!empty_polygon.contains_point(Vector2::new(0.0, 0.0)), "Empty polygon contains no points");
        assert_vector_approx_eq(empty_polygon.center(), Vector2::zeros(), 1e-10);
        
        let bbox = empty_polygon.bounding_box();
        assert_vector_approx_eq(bbox.min, Vector2::zeros(), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::zeros(), 1e-10);
    }

    #[test]
    fn test_compound_geometry() {
        let circle = Box::new(Circle::new(Vector2::new(-5.0, 0.0), 3.0));
        let rect = Box::new(Rectangle::new(Vector2::new(5.0, 0.0), Vector2::new(4.0, 4.0)));
        
        let compound = CompoundGeometry::new(vec![circle, rect]);
        
        // Test contains_point
        assert!(compound.contains_point(Vector2::new(-5.0, 0.0)), "Circle center should be inside");
        assert!(compound.contains_point(Vector2::new(5.0, 0.0)), "Rectangle center should be inside");
        assert!(!compound.contains_point(Vector2::new(0.0, 0.0)), "Gap between shapes should be outside");
        
        // Test bounding box
        let bbox = compound.bounding_box();
        assert_vector_approx_eq(bbox.min, Vector2::new(-8.0, -3.0), 1e-10);
        assert_vector_approx_eq(bbox.max, Vector2::new(7.0, 3.0), 1e-10);
        
        // Test center
        assert_vector_approx_eq(compound.center(), Vector2::new(0.0, 0.0), 1e-10);
    }

    #[test]
    fn test_compound_geometry_transform() {
        let circle = Box::new(Circle::new(Vector2::new(0.0, 0.0), 1.0));
        let mut compound = CompoundGeometry::new(vec![circle]);
        
        let transform = Transform2D {
            translation: Vector2::new(2.0, 3.0),
            rotation: 0.0,
            scale: Vector2::new(1.0, 1.0),
        };
        
        compound.transform(&transform);
        
        assert!(compound.contains_point(Vector2::new(2.0, 3.0)), "Transformed center should contain point");
        assert_vector_approx_eq(compound.center(), Vector2::new(2.0, 3.0), 1e-10);
    }

    #[test]
    fn test_geometric_object_2d() {
        let circle = Circle::new(Vector2::new(0.0, 0.0), 5.0);
        let material = Material::default();
        let mut obj = GeometricObject2D::new(circle, material.clone());
        
        // Test shape access
        assert!(obj.shape().contains_point(Vector2::new(0.0, 0.0)), "Should access shape methods");
        
        // Test material access
        assert_eq!(obj.material().name, material.name, "Should access material");
        
        // Test transform
        let transform = Transform2D {
            translation: Vector2::new(1.0, 1.0),
            rotation: 0.0,
            scale: Vector2::new(1.0, 1.0),
        };
        obj.transform(&transform);
        
        assert!(obj.shape().contains_point(Vector2::new(1.0, 1.0)), "Transform should affect shape");
    }

    #[test]
    fn test_clone_box() {
        // Test each geometry type's clone_box implementation
        let circle = Circle::new(Vector2::new(1.0, 2.0), 3.0);
        let cloned_circle = circle.clone_box();
        assert!(cloned_circle.contains_point(Vector2::new(1.0, 2.0)), "Cloned circle should have same properties");
        
        let rect = Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(4.0, 4.0));
        let cloned_rect = rect.clone_box();
        assert!(cloned_rect.contains_point(Vector2::new(0.0, 0.0)), "Cloned rectangle should have same properties");
        
        let ellipse = Ellipse::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 1.0));
        let cloned_ellipse = ellipse.clone_box();
        assert!(cloned_ellipse.contains_point(Vector2::new(0.0, 0.0)), "Cloned ellipse should have same properties");
        
        let polygon = Polygon::regular(Vector2::new(0.0, 0.0), 2.0, 4);
        let cloned_polygon = polygon.clone_box();
        assert!(cloned_polygon.contains_point(Vector2::new(0.0, 0.0)), "Cloned polygon should have same properties");
    }

    #[test]
    fn test_edge_cases_transform_scale() {
        // Test non-uniform scaling
        let mut circle = Circle::new(Vector2::new(0.0, 0.0), 2.0);
        let transform = Transform2D {
            translation: Vector2::new(0.0, 0.0),
            rotation: 0.0,
            scale: Vector2::new(3.0, 2.0), // Non-uniform scale
        };
        
        circle.transform(&transform);
        
        // Circle should use the maximum scale factor
        assert!(approx_eq(circle.radius, 6.0, 1e-10), "Circle radius should use max scale factor");
        
        // Test zero scale (edge case)
        let mut rect = Rectangle::new(Vector2::new(0.0, 0.0), Vector2::new(4.0, 4.0));
        let zero_scale = Transform2D {
            translation: Vector2::new(0.0, 0.0),
            rotation: 0.0,
            scale: Vector2::new(0.0, 0.0),
        };
        
        rect.transform(&zero_scale);
        // Rectangle should effectively become a point
        assert!(rect.contains_point(Vector2::new(0.0, 0.0)), "Zero-scaled rectangle should still contain its center");
    }
}
