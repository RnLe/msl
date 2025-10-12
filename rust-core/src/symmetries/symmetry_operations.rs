use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// A single symmetry operation: rotation (integer‐matrix) + translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryOperation {
    /// Orthogonal rotation matrix with determinant ±1
    pub rotation: Matrix3<i8>,
    /// Fractional translation shift
    pub translation: Vector3<f64>,
}

impl SymmetryOperation {
    /// Create a new symmetry operation
    pub fn new(rotation: Matrix3<i8>, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Create identity operation
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Apply symmetry operation to a point
    pub fn apply(&self, point: Vector3<f64>) -> Vector3<f64> {
        let rotation_f64 = self.rotation.map(|x| x as f64);
        rotation_f64 * point + self.translation
    }

    /// Get the order of this symmetry operation (how many times to apply to get identity)
    /// TODO: Implement order calculation
    pub fn order(&self) -> usize {
        // Placeholder - proper implementation would calculate the actual order
        1
    }

    /// Check if this is the identity operation
    pub fn is_identity(&self) -> bool {
        self.rotation == Matrix3::identity() && self.translation.norm() < 1e-10
    }
}
