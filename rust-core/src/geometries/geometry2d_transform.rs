// 2D transformation module: Contains 2D geometric transformation functionality
// This module provides affine transformations for 2D geometric objects

use nalgebra::{Matrix3, Rotation2, Vector2};

/// 2D affine transformation
///
/// Represents a combination of translation, rotation, and scaling operations
/// that can be applied to 2D geometric objects. Transformations are applied
/// in the order: scale -> rotate -> translate.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform2D {
    /// Translation offset
    pub translation: Vector2<f64>,
    /// Rotation angle in radians (counterclockwise)
    pub rotation: f64,
    /// Scale factors for x and y axes
    pub scale: Vector2<f64>,
}

impl Transform2D {
    /// Create a new identity transformation (no change)
    pub fn new() -> Self {
        Self::identity()
    }

    /// Create an identity transformation
    pub fn identity() -> Self {
        Self {
            translation: Vector2::zeros(),
            rotation: 0.0,
            scale: Vector2::new(1.0, 1.0),
        }
    }

    /// Create a translation-only transformation
    ///
    /// # Arguments
    /// * `offset` - Translation offset
    pub fn translation(offset: Vector2<f64>) -> Self {
        Self {
            translation: offset,
            rotation: 0.0,
            scale: Vector2::new(1.0, 1.0),
        }
    }

    /// Create a rotation-only transformation
    ///
    /// # Arguments
    /// * `angle` - Rotation angle in radians (counterclockwise)
    pub fn rotation(angle: f64) -> Self {
        Self {
            translation: Vector2::zeros(),
            rotation: angle,
            scale: Vector2::new(1.0, 1.0),
        }
    }

    /// Create a scale-only transformation
    ///
    /// # Arguments
    /// * `scale` - Scale factors for x and y axes
    pub fn scaling(scale: Vector2<f64>) -> Self {
        Self {
            translation: Vector2::zeros(),
            rotation: 0.0,
            scale,
        }
    }

    /// Create a uniform scale transformation
    ///
    /// # Arguments
    /// * `factor` - Uniform scale factor
    pub fn uniform_scaling(factor: f64) -> Self {
        Self::scaling(Vector2::new(factor, factor))
    }

    /// Add a translation to this transformation
    ///
    /// # Arguments
    /// * `offset` - Translation offset to add
    pub fn translate(mut self, offset: Vector2<f64>) -> Self {
        self.translation += offset;
        self
    }

    /// Add a rotation to this transformation
    ///
    /// # Arguments
    /// * `angle` - Rotation angle in radians to add (counterclockwise)
    pub fn rotate(mut self, angle: f64) -> Self {
        self.rotation += angle;
        self
    }

    /// Add scaling to this transformation
    ///
    /// # Arguments
    /// * `scale` - Scale factors to multiply with current scaling
    pub fn scale(mut self, scale: Vector2<f64>) -> Self {
        self.scale.component_mul_assign(&scale);
        self
    }

    /// Add uniform scaling to this transformation
    ///
    /// # Arguments
    /// * `factor` - Uniform scale factor to multiply with current scaling
    pub fn scale_uniform(self, factor: f64) -> Self {
        self.scale(Vector2::new(factor, factor))
    }

    /// Apply the transformation to a point
    ///
    /// Transformations are applied in the order: scale -> rotate -> translate
    ///
    /// # Arguments
    /// * `point` - Point to transform
    ///
    /// # Returns
    /// The transformed point
    pub fn apply_to_point(&self, point: Vector2<f64>) -> Vector2<f64> {
        // Apply scale
        let scaled = Vector2::new(point.x * self.scale.x, point.y * self.scale.y);

        // Apply rotation
        let rotated = if self.rotation == 0.0 {
            scaled
        } else {
            Rotation2::new(self.rotation) * scaled
        };

        // Apply translation
        rotated + self.translation
    }

    /// Apply the inverse transformation to a point
    ///
    /// This is useful for transforming points from world space back to local space.
    /// Inverse operations are applied in reverse order: untranslate -> unrotate -> unscale
    ///
    /// # Arguments
    /// * `point` - Point to inverse transform
    ///
    /// # Returns
    /// The inverse transformed point
    ///
    /// # Panics
    /// Panics if any scale component is zero (transformation is not invertible)
    pub fn apply_inverse_to_point(&self, point: Vector2<f64>) -> Vector2<f64> {
        assert!(
            self.scale.x != 0.0 && self.scale.y != 0.0,
            "Cannot invert transformation with zero scale"
        );

        // Apply inverse translation
        let translated = point - self.translation;

        // Apply inverse rotation
        let rotated = if self.rotation == 0.0 {
            translated
        } else {
            Rotation2::new(-self.rotation) * translated
        };

        // Apply inverse scale
        Vector2::new(rotated.x / self.scale.x, rotated.y / self.scale.y)
    }

    /// Apply the transformation to a vector (ignoring translation)
    ///
    /// This is useful for transforming direction vectors or normals.
    /// Only scaling and rotation are applied.
    ///
    /// # Arguments
    /// * `vector` - Vector to transform
    ///
    /// # Returns
    /// The transformed vector
    pub fn apply_to_vector(&self, vector: Vector2<f64>) -> Vector2<f64> {
        // Apply scale
        let scaled = Vector2::new(vector.x * self.scale.x, vector.y * self.scale.y);

        // Apply rotation
        if self.rotation == 0.0 {
            scaled
        } else {
            Rotation2::new(self.rotation) * scaled
        }
    }

    /// Apply the inverse transformation to a vector (ignoring translation)
    ///
    /// # Arguments
    /// * `vector` - Vector to inverse transform
    ///
    /// # Returns
    /// The inverse transformed vector
    ///
    /// # Panics
    /// Panics if any scale component is zero (transformation is not invertible)
    pub fn apply_inverse_to_vector(&self, vector: Vector2<f64>) -> Vector2<f64> {
        assert!(
            self.scale.x != 0.0 && self.scale.y != 0.0,
            "Cannot invert transformation with zero scale"
        );

        // Apply inverse rotation
        let rotated = if self.rotation == 0.0 {
            vector
        } else {
            Rotation2::new(-self.rotation) * vector
        };

        // Apply inverse scale
        Vector2::new(rotated.x / self.scale.x, rotated.y / self.scale.y)
    }

    /// Compose this transformation with another transformation
    ///
    /// The resulting transformation applies this transformation first,
    /// then the other transformation.
    ///
    /// # Arguments
    /// * `other` - The transformation to compose with
    ///
    /// # Returns
    /// The composed transformation
    pub fn then(&self, other: &Transform2D) -> Transform2D {
        // This is a simplified composition that works well for most cases
        // For more complex cases, matrix multiplication would be more accurate
        Transform2D {
            translation: other.apply_to_point(self.translation),
            rotation: self.rotation + other.rotation,
            scale: Vector2::new(self.scale.x * other.scale.x, self.scale.y * other.scale.y),
        }
    }

    /// Convert the transformation to a 3x3 homogeneous matrix
    ///
    /// This is useful for more complex transformations or when interfacing
    /// with graphics libraries that expect matrix representations.
    ///
    /// # Returns
    /// A 3x3 homogeneous transformation matrix
    pub fn to_matrix(&self) -> Matrix3<f64> {
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();

        Matrix3::new(
            self.scale.x * cos_r,
            -self.scale.y * sin_r,
            self.translation.x,
            self.scale.x * sin_r,
            self.scale.y * cos_r,
            self.translation.y,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Create a transformation from a 3x3 homogeneous matrix
    ///
    /// This function extracts translation, rotation, and scale from a transformation matrix.
    /// Note that this may not work correctly for matrices with shear or non-uniform transformations.
    ///
    /// # Arguments
    /// * `matrix` - 3x3 homogeneous transformation matrix
    ///
    /// # Returns
    /// The extracted transformation
    pub fn from_matrix(matrix: &Matrix3<f64>) -> Self {
        // Extract translation
        let translation = Vector2::new(matrix[(0, 2)], matrix[(1, 2)]);

        // Extract scale and rotation from the 2x2 upper-left submatrix
        let a = matrix[(0, 0)]; // scale_x * cos(rotation)
        let b = matrix[(0, 1)]; // -scale_y * sin(rotation)
        let c = matrix[(1, 0)]; // scale_x * sin(rotation)
        let d = matrix[(1, 1)]; // scale_y * cos(rotation)

        // Extract rotation from the scale_x components
        let rotation = c.atan2(a);

        // Extract scale using the rotation
        let cos_r = rotation.cos();
        let sin_r = rotation.sin();

        // Use the most numerically stable computation
        let scale_x = if cos_r.abs() > sin_r.abs() {
            a / cos_r
        } else {
            c / sin_r
        };

        let scale_y = if cos_r.abs() > sin_r.abs() {
            d / cos_r
        } else {
            -b / sin_r
        };

        Transform2D {
            translation,
            rotation,
            scale: Vector2::new(scale_x, scale_y),
        }
    }

    /// Check if this is an identity transformation
    pub fn is_identity(&self) -> bool {
        const EPSILON: f64 = 1e-10;

        self.translation.norm() < EPSILON
            && self.rotation.abs() < EPSILON
            && (self.scale - Vector2::new(1.0, 1.0)).norm() < EPSILON
    }

    /// Check if this transformation includes only translation
    pub fn is_translation_only(&self) -> bool {
        const EPSILON: f64 = 1e-10;

        self.rotation.abs() < EPSILON && (self.scale - Vector2::new(1.0, 1.0)).norm() < EPSILON
    }

    /// Check if this transformation preserves distances (only translation and rotation)
    pub fn is_rigid(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.scale - Vector2::new(1.0, 1.0)).norm() < EPSILON
    }

    /// Check if this transformation preserves angles (uniform scaling, translation, and rotation)
    pub fn is_similarity(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.scale.x - self.scale.y).abs() < EPSILON
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}
