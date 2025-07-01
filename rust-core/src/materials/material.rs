// Material module: Contains material property definitions for electromagnetic simulations
// This module provides material properties including dielectric constants, refractive indices, and magnetic permeability

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Material type for electromagnetic simulations
/// 
/// This structure contains the fundamental electromagnetic properties needed for
/// solving Maxwell's equations in heterogeneous media. All materials are assumed
/// to be isotropic (no tensor properties).
/// 
/// # Fields
/// * `epsilon` - Complex relative permittivity (dielectric constant)
/// * `mu` - Complex relative permeability (magnetic permeability)  
/// * `refractive_index` - Complex refractive index (derived from epsilon and mu)
/// * `name` - Optional material identifier for debugging/visualization
/// 
/// # Physical Relations
/// The refractive index is related to the material parameters by:
/// n = sqrt(epsilon * mu)
/// 
/// For non-magnetic materials (mu = 1), this simplifies to:
/// n = sqrt(epsilon)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Material {
    /// Complex relative permittivity (dielectric constant)
    /// Real part: energy storage, Imaginary part: dielectric loss
    pub epsilon: Complex64,
    
    /// Complex relative permeability  
    /// Real part: magnetic response, Imaginary part: magnetic loss
    /// For most materials at optical frequencies, mu ≈ 1 + 0i
    pub mu: Complex64,
    
    /// Complex refractive index
    /// Real part: phase velocity, Imaginary part: absorption
    pub refractive_index: Complex64,
    
    /// Optional material name for identification
    pub name: Option<String>,
}

impl Material {
    /// Create a new material with specified electromagnetic properties
    /// 
    /// The refractive index is automatically calculated from epsilon and mu.
    /// 
    /// # Arguments
    /// * `epsilon` - Complex relative permittivity
    /// * `mu` - Complex relative permeability
    /// * `name` - Optional material name
    pub fn new(epsilon: Complex64, mu: Complex64, name: Option<String>) -> Self {
        let refractive_index = (epsilon * mu).sqrt();
        
        Self {
            epsilon,
            mu,
            refractive_index,
            name,
        }
    }
    
    /// Create a dielectric material (non-magnetic, mu = 1)
    /// 
    /// This is a convenience constructor for the most common case of
    /// non-magnetic dielectric materials.
    /// 
    /// # Arguments
    /// * `epsilon` - Complex relative permittivity
    /// * `name` - Optional material name
    pub fn dielectric(epsilon: Complex64, name: Option<String>) -> Self {
        Self::new(epsilon, Complex64::new(1.0, 0.0), name)
    }
    
    /// Create a material from refractive index (assumes non-magnetic)
    /// 
    /// This constructor calculates epsilon from the refractive index,
    /// assuming mu = 1 (non-magnetic material).
    /// 
    /// # Arguments
    /// * `refractive_index` - Complex refractive index
    /// * `name` - Optional material name
    pub fn from_refractive_index(refractive_index: Complex64, name: Option<String>) -> Self {
        let epsilon = refractive_index * refractive_index;
        let mu = Complex64::new(1.0, 0.0);
        
        Self {
            epsilon,
            mu,
            refractive_index,
            name,
        }
    }
    
    /// Update the refractive index based on current epsilon and mu values
    /// 
    /// This method should be called if epsilon or mu are modified directly
    /// to ensure consistency between the material parameters.
    pub fn update_refractive_index(&mut self) {
        self.refractive_index = (self.epsilon * self.mu).sqrt();
    }
    
    /// Check if the material is lossless (no imaginary parts)
    pub fn is_lossless(&self) -> bool {
        self.epsilon.im.abs() < f64::EPSILON && self.mu.im.abs() < f64::EPSILON
    }
    
    /// Check if the material is non-magnetic (mu ≈ 1)
    pub fn is_non_magnetic(&self) -> bool {
        (self.mu - Complex64::new(1.0, 0.0)).norm() < f64::EPSILON
    }
    
    /// Get the real part of the permittivity
    pub fn epsilon_real(&self) -> f64 {
        self.epsilon.re
    }
    
    /// Get the imaginary part of the permittivity (loss)
    pub fn epsilon_imag(&self) -> f64 {
        self.epsilon.im
    }
    
    /// Get the real part of the refractive index
    pub fn refractive_index_real(&self) -> f64 {
        self.refractive_index.re
    }
    
    /// Get the imaginary part of the refractive index (absorption)
    pub fn refractive_index_imag(&self) -> f64 {
        self.refractive_index.im
    }
}

impl Default for Material {
    /// Default material: vacuum/air (epsilon = 1, mu = 1, n = 1)
    fn default() -> Self {
        Self::new(
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Some("Vacuum".to_string())
        )
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        const TOLERANCE: f64 = 1e-12;
        
        (self.epsilon - other.epsilon).norm() < TOLERANCE &&
        (self.mu - other.mu).norm() < TOLERANCE
    }
}

/// Common material constants
pub struct CommonMaterials;

impl CommonMaterials {
    /// Vacuum/Air (epsilon = 1, mu = 1, n = 1)
    pub fn vacuum() -> Material {
        Material::default()
    }
    
    /// Silicon at 1550 nm (epsilon ≈ 12.25, n ≈ 3.5)
    pub fn silicon() -> Material {
        Material::dielectric(
            Complex64::new(12.25, 0.0),
            Some("Silicon".to_string())
        )
    }
    
    /// Silica glass (epsilon ≈ 2.13, n ≈ 1.46)
    pub fn silica() -> Material {
        Material::dielectric(
            Complex64::new(2.13, 0.0),
            Some("Silica".to_string())
        )
    }
    
    /// Gallium Arsenide at 1550 nm (epsilon ≈ 11.0, n ≈ 3.32)
    pub fn gaas() -> Material {
        Material::dielectric(
            Complex64::new(11.0, 0.0),
            Some("GaAs".to_string())
        )
    }
    
    /// Aluminum Oxide (Sapphire) (epsilon ≈ 9.8, n ≈ 1.76)
    pub fn al2o3() -> Material {
        Material::dielectric(
            Complex64::new(9.8, 0.0),
            Some("Al2O3".to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_material_creation() {
        let mat = Material::new(
            Complex64::new(4.0, 0.1),
            Complex64::new(1.0, 0.0),
            Some("Test".to_string())
        );
        
        assert_relative_eq!(mat.epsilon.re, 4.0);
        assert_relative_eq!(mat.epsilon.im, 0.1);
        assert_relative_eq!(mat.mu.re, 1.0);
        assert_relative_eq!(mat.mu.im, 0.0);
        
        // Check that refractive index is calculated correctly
        let expected_n = (Complex64::new(4.0, 0.1)).sqrt();
        assert_relative_eq!(mat.refractive_index.re, expected_n.re, epsilon = 1e-10);
        assert_relative_eq!(mat.refractive_index.im, expected_n.im, epsilon = 1e-10);
    }
    
    #[test]
    fn test_dielectric_material() {
        let mat = Material::dielectric(
            Complex64::new(2.25, 0.0),
            Some("Dielectric".to_string())
        );
        
        assert!(mat.is_non_magnetic());
        assert!(mat.is_lossless());
        assert_relative_eq!(mat.refractive_index_real(), 1.5); // sqrt(2.25) = 1.5
    }
    
    #[test]
    fn test_from_refractive_index() {
        let mat = Material::from_refractive_index(
            Complex64::new(1.5, 0.0),
            Some("Glass".to_string())
        );
        
        assert_relative_eq!(mat.epsilon_real(), 2.25); // 1.5^2 = 2.25
        assert_relative_eq!(mat.refractive_index_real(), 1.5);
    }
    
    #[test]
    fn test_common_materials() {
        let silicon = CommonMaterials::silicon();
        assert_relative_eq!(silicon.epsilon_real(), 12.25);
        
        let vacuum = CommonMaterials::vacuum();
        assert_relative_eq!(vacuum.epsilon_real(), 1.0);
        assert_relative_eq!(vacuum.refractive_index_real(), 1.0);
    }
    
    #[test]
    fn test_material_equality() {
        let mat1 = Material::dielectric(Complex64::new(2.0, 0.0), None);
        let mat2 = Material::dielectric(Complex64::new(2.0, 0.0), None);
        let mat3 = Material::dielectric(Complex64::new(2.1, 0.0), None);
        
        assert_eq!(mat1, mat2);
        assert_ne!(mat1, mat3);
    }
}
