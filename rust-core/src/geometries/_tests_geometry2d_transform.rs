#[cfg(test)]
mod tests {
    use super::super::geometry2d_transform::Transform2D;
    use nalgebra::Vector2;
    use approx::{assert_abs_diff_eq};
    use std::f64::consts::PI;
    
    #[test]
    fn test_identity_transformation() {
        let transform = Transform2D::identity();
        let point = Vector2::new(3.0, 4.0);
        
        assert_eq!(transform.apply_to_point(point), point);
        assert!(transform.is_identity());
    }
    
    #[test]
    fn test_translation() {
        let transform = Transform2D::translation(Vector2::new(2.0, 3.0));
        let point = Vector2::new(1.0, 1.0);
        let transformed = transform.apply_to_point(point);
        
        assert_eq!(transformed, Vector2::new(3.0, 4.0));
        assert!(transform.is_translation_only());
        assert!(transform.is_rigid());
    }
    
    #[test]
    fn test_rotation() {
        let transform = Transform2D::rotation(PI / 2.0); // 90 degrees
        let point = Vector2::new(1.0, 0.0);
        let transformed = transform.apply_to_point(point);
        
        assert_abs_diff_eq!(transformed.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, 1.0, epsilon = 1e-10);
        assert!(transform.is_rigid());
    }
    
    #[test]
    fn test_scaling() {
        let transform = Transform2D::scaling(Vector2::new(2.0, 3.0));
        let point = Vector2::new(1.0, 1.0);
        let transformed = transform.apply_to_point(point);
        
        assert_eq!(transformed, Vector2::new(2.0, 3.0));
        assert!(!transform.is_rigid());
        assert!(!transform.is_similarity());
    }
    
    #[test]
    fn test_uniform_scaling() {
        let transform = Transform2D::uniform_scaling(2.0);
        let point = Vector2::new(1.0, 1.0);
        let transformed = transform.apply_to_point(point);
        
        assert_eq!(transformed, Vector2::new(2.0, 2.0));
        assert!(transform.is_similarity());
    }
    
    #[test]
    fn test_combined_transformation() {
        let transform = Transform2D::new()
            .scale(Vector2::new(2.0, 1.0))
            .rotate(PI / 4.0)
            .translate(Vector2::new(1.0, 1.0));
        
        let point = Vector2::new(1.0, 0.0);
        let transformed = transform.apply_to_point(point);
        
        // After scale: (2, 0)
        // After rotation: (√2, √2)
        // After translation: (1+√2, 1+√2)
        let sqrt2 = (2.0_f64).sqrt();
        assert_abs_diff_eq!(transformed.x, 1.0 + sqrt2, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, 1.0 + sqrt2, epsilon = 1e-10);
    }
    
    #[test]
    fn test_inverse_transformation() {
        let transform = Transform2D::new()
            .translate(Vector2::new(2.0, 3.0))
            .rotate(PI / 6.0)
            .scale(Vector2::new(2.0, 1.5));
        
        let point = Vector2::new(1.0, 1.0);
        let transformed = transform.apply_to_point(point);
        let inverse_transformed = transform.apply_inverse_to_point(transformed);
        
        assert_abs_diff_eq!(inverse_transformed.x, point.x, epsilon = 1e-10);
        assert_abs_diff_eq!(inverse_transformed.y, point.y, epsilon = 1e-10);
    }
    
    #[test]
    fn test_vector_transformation() {
        let transform = Transform2D::new()
            .scale(Vector2::new(2.0, 1.0))
            .rotate(PI / 2.0)
            .translate(Vector2::new(10.0, 20.0)); // Should be ignored for vectors
        
        let vector = Vector2::new(1.0, 0.0);
        let transformed = transform.apply_to_vector(vector);
        
        // After scale: (2, 0)
        // After rotation: (0, 2)
        // Translation is ignored
        assert_abs_diff_eq!(transformed.x, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(transformed.y, 2.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_composition() {
        let t1 = Transform2D::translation(Vector2::new(1.0, 2.0));
        let t2 = Transform2D::rotation(PI / 4.0);
        
        let composed = t1.then(&t2);
        let point = Vector2::new(0.0, 0.0);
        
        let result1 = composed.apply_to_point(point);
        let result2 = t2.apply_to_point(t1.apply_to_point(point));
        
        assert_abs_diff_eq!(result1.x, result2.x, epsilon = 1e-10);
        assert_abs_diff_eq!(result1.y, result2.y, epsilon = 1e-10);
    }
    
    #[test]
    fn test_matrix_conversion() {
        let transform = Transform2D::new()
            .translate(Vector2::new(2.0, 3.0))
            .rotate(PI / 6.0)
            .scale(Vector2::new(1.5, 2.0));
        
        let matrix = transform.to_matrix();
        let reconstructed = Transform2D::from_matrix(&matrix);
        
        // Test that they produce the same results
        let point = Vector2::new(1.0, 1.0);
        let result1 = transform.apply_to_point(point);
        let result2 = reconstructed.apply_to_point(point);
        
        assert_abs_diff_eq!(result1.x, result2.x, epsilon = 1e-10);
        assert_abs_diff_eq!(result1.y, result2.y, epsilon = 1e-10);
    }
}
