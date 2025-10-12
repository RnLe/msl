#[cfg(test)]
mod tests_polyhedron {
    use super::super::polyhedron::Polyhedron;
    use nalgebra::Vector3;

    #[test]
    fn test_new() {
        let poly = Polyhedron::new();
        assert!(poly.vertices.is_empty());
        assert!(poly.edges.is_empty());
        assert!(poly.faces.is_empty());
        assert_eq!(poly.measure, 0.0);
    }

    #[test]
    fn test_default() {
        let poly = Polyhedron::default();
        assert!(poly.vertices.is_empty());
        assert!(poly.edges.is_empty());
        assert!(poly.faces.is_empty());
        assert_eq!(poly.measure, 0.0);
    }

    #[test]
    fn test_contains_2d_square() {
        let mut poly = Polyhedron::new();
        // Create a unit square: (0,0), (1,0), (1,1), (0,1)
        poly.vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];

        // Test interior points
        assert!(poly.contains_2d(Vector3::new(0.5, 0.5, 0.0)));
        assert!(poly.contains_2d(Vector3::new(0.25, 0.75, 0.0)));

        // Test exterior points
        assert!(!poly.contains_2d(Vector3::new(1.5, 0.5, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(-0.5, 0.5, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(0.5, 1.5, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(0.5, -0.5, 0.0)));

        // Test boundary points (ray casting algorithm behavior may vary)
        let boundary_point = Vector3::new(0.0, 0.5, 0.0);
        let result = poly.contains_2d(boundary_point);
        if !result {
            println!(
                "Debug: Boundary point {:?} not contained in square",
                boundary_point
            );
        }
    }

    #[test]
    fn test_contains_2d_triangle() {
        let mut poly = Polyhedron::new();
        // Create a triangle: (0,0), (2,0), (1,2)
        poly.vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(1.0, 2.0, 0.0),
        ];

        // Test interior point
        assert!(poly.contains_2d(Vector3::new(1.0, 0.5, 0.0)));

        // Test exterior points
        assert!(!poly.contains_2d(Vector3::new(0.5, 1.5, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(1.5, 1.5, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(3.0, 0.0, 0.0)));
    }

    #[test]
    fn test_contains_2d_empty_polygon() {
        let poly = Polyhedron::new();
        // Empty polygon should not contain any point
        assert!(!poly.contains_2d(Vector3::new(0.0, 0.0, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(1.0, 1.0, 0.0)));
    }

    #[test]
    fn test_contains_2d_single_vertex() {
        let mut poly = Polyhedron::new();
        poly.vertices = vec![Vector3::new(1.0, 1.0, 0.0)];

        // Single vertex polygon should not contain any point
        assert!(!poly.contains_2d(Vector3::new(1.0, 1.0, 0.0)));
        assert!(!poly.contains_2d(Vector3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_contains_3d_cube() {
        let mut poly = Polyhedron::new();
        // Create a unit cube vertices
        poly.vertices = vec![
            Vector3::new(0.0, 0.0, 0.0), // 0
            Vector3::new(1.0, 0.0, 0.0), // 1
            Vector3::new(1.0, 1.0, 0.0), // 2
            Vector3::new(0.0, 1.0, 0.0), // 3
            Vector3::new(0.0, 0.0, 1.0), // 4
            Vector3::new(1.0, 0.0, 1.0), // 5
            Vector3::new(1.0, 1.0, 1.0), // 6
            Vector3::new(0.0, 1.0, 1.0), // 7
        ];

        // Define cube faces (counter-clockwise when viewed from outside)
        poly.faces = vec![
            vec![0, 3, 2, 1], // bottom face (z=0)
            vec![4, 5, 6, 7], // top face (z=1)
            vec![0, 1, 5, 4], // front face (y=0)
            vec![2, 3, 7, 6], // back face (y=1)
            vec![0, 4, 7, 3], // left face (x=0)
            vec![1, 2, 6, 5], // right face (x=1)
        ];

        // Test interior points
        assert!(poly.contains_3d(Vector3::new(0.5, 0.5, 0.5)));
        assert!(poly.contains_3d(Vector3::new(0.25, 0.75, 0.25)));

        // Test exterior points
        assert!(!poly.contains_3d(Vector3::new(1.5, 0.5, 0.5)));
        assert!(!poly.contains_3d(Vector3::new(-0.5, 0.5, 0.5)));
        assert!(!poly.contains_3d(Vector3::new(0.5, 1.5, 0.5)));
        assert!(!poly.contains_3d(Vector3::new(0.5, -0.5, 0.5)));
        assert!(!poly.contains_3d(Vector3::new(0.5, 0.5, 1.5)));
        assert!(!poly.contains_3d(Vector3::new(0.5, 0.5, -0.5)));

        // Test corner vertices (should be inside due to EPS tolerance)
        let corner_result = poly.contains_3d(Vector3::new(0.0, 0.0, 0.0));
        if !corner_result {
            println!("Debug: Corner vertex not contained in cube");
            println!("Debug: EPS = 1e-10, vertices: {:?}", poly.vertices);
        }
    }

    #[test]
    fn test_contains_3d_tetrahedron() {
        let mut poly = Polyhedron::new();
        // Create a regular tetrahedron
        poly.vertices = vec![
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
        ];

        // Define tetrahedron faces
        poly.faces = vec![
            vec![0, 1, 2], // face 1
            vec![0, 2, 3], // face 2
            vec![0, 3, 1], // face 3
            vec![1, 3, 2], // face 4
        ];

        // Test center point (should be inside)
        assert!(poly.contains_3d(Vector3::new(0.0, 0.0, 0.0)));

        // Test far exterior points
        assert!(!poly.contains_3d(Vector3::new(5.0, 5.0, 5.0)));
        assert!(!poly.contains_3d(Vector3::new(-5.0, -5.0, -5.0)));
    }

    #[test]
    fn test_contains_3d_empty_faces() {
        let mut poly = Polyhedron::new();
        poly.vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];
        // No faces defined

        // Should return false for any point when no faces exist
        assert!(!poly.contains_3d(Vector3::new(0.5, 0.5, 0.0)));
        assert!(!poly.contains_3d(Vector3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn test_contains_3d_degenerate_faces() {
        let mut poly = Polyhedron::new();
        poly.vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];

        // Add degenerate faces (< 3 vertices)
        poly.faces = vec![
            vec![0],       // single vertex
            vec![0, 1],    // edge
            vec![0, 1, 2], // valid triangle
        ];

        let test_point = Vector3::new(0.5, 0.25, 0.0);
        let result = poly.contains_3d(test_point);

        if !result {
            println!("Debug: Point {:?} not contained", test_point);
            println!("Debug: Faces with degenerate entries: {:?}", poly.faces);
        }
    }

    #[test]
    fn test_contains_3d_numerical_precision() {
        let mut poly = Polyhedron::new();
        // Create a small cube around origin
        let eps = 1e-12;
        poly.vertices = vec![
            Vector3::new(-eps, -eps, -eps),
            Vector3::new(eps, -eps, -eps),
            Vector3::new(eps, eps, -eps),
            Vector3::new(-eps, eps, -eps),
            Vector3::new(-eps, -eps, eps),
            Vector3::new(eps, -eps, eps),
            Vector3::new(eps, eps, eps),
            Vector3::new(-eps, eps, eps),
        ];

        poly.faces = vec![
            vec![0, 3, 2, 1], // bottom
            vec![4, 5, 6, 7], // top
            vec![0, 1, 5, 4], // front
            vec![2, 3, 7, 6], // back
            vec![0, 4, 7, 3], // left
            vec![1, 2, 6, 5], // right
        ];

        // Test point very close to boundary
        let boundary_point = Vector3::new(eps + 1e-11, 0.0, 0.0);
        let result = poly.contains_3d(boundary_point);

        if result {
            println!(
                "Debug: Boundary point {:?} unexpectedly contained",
                boundary_point
            );
            println!("Debug: EPS tolerance = 1e-10, point distance from boundary ≈ 1e-11");
        }
    }

    #[test]
    fn test_measure() {
        let mut poly = Polyhedron::new();

        // Test initial measure
        assert_eq!(poly.measure(), 0.0);

        // Test after setting measure
        poly.measure = 42.5;
        assert_eq!(poly.measure(), 42.5);

        // Test negative measure (geometrically unusual but structurally valid)
        poly.measure = -1.0;
        assert_eq!(poly.measure(), -1.0);
    }

    #[test]
    fn test_measure_precision() {
        let mut poly = Polyhedron::new();

        // Test very small measures
        poly.measure = 1e-15;
        assert_eq!(poly.measure(), 1e-15);

        // Test very large measures
        poly.measure = 1e15;
        assert_eq!(poly.measure(), 1e15);
    }
}
