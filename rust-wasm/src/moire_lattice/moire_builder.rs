use super::moire2d::WasmMoire2D;
use crate::lattice::WasmLattice2D;
use moire_lattice::moire_lattice::{
    MoireTransformation,
    moire_builder::{MoireBuilder, commensurate_moire, twisted_bilayer},
};
use nalgebra::Matrix2;
use serde::Deserialize;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// Parameters for creating moiré lattices
#[derive(Deserialize)]
pub struct MoireParams {
    pub transformation_type: String,
    pub angle_degrees: Option<f64>,
    pub scale: Option<f64>,
    pub scale_x: Option<f64>,
    pub scale_y: Option<f64>,
    pub shear_x: Option<f64>,
    pub shear_y: Option<f64>,
    pub matrix: Option<Vec<f64>>, // Flattened 2x2 matrix
    pub tolerance: Option<f64>,
}

/// WASM wrapper for MoireBuilder
#[wasm_bindgen]
pub struct WasmMoireBuilder {
    inner: MoireBuilder,
}

#[wasm_bindgen]
impl WasmMoireBuilder {
    /// Create a new MoireBuilder
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMoireBuilder {
        WasmMoireBuilder {
            inner: MoireBuilder::new(),
        }
    }

    /// Set the base lattice
    #[wasm_bindgen]
    pub fn with_base_lattice(mut self, lattice: &WasmLattice2D) -> WasmMoireBuilder {
        self.inner = self.inner.with_base_lattice(lattice.inner.clone());
        self
    }

    /// Set tolerance for calculations
    #[wasm_bindgen]
    pub fn with_tolerance(mut self, tolerance: f64) -> WasmMoireBuilder {
        self.inner = self.inner.with_tolerance(tolerance);
        self
    }

    /// Set a rotation and uniform scaling transformation
    #[wasm_bindgen]
    pub fn with_twist_and_scale(mut self, angle_degrees: f64, scale: f64) -> WasmMoireBuilder {
        let angle_radians = angle_degrees * PI / 180.0;
        self.inner = self.inner.with_twist_and_scale(angle_radians, scale);
        self
    }

    /// Set an anisotropic scaling transformation
    #[wasm_bindgen]
    pub fn with_anisotropic_scale(mut self, scale_x: f64, scale_y: f64) -> WasmMoireBuilder {
        self.inner = self.inner.with_anisotropic_scale(scale_x, scale_y);
        self
    }

    /// Set a shear transformation
    #[wasm_bindgen]
    pub fn with_shear(mut self, shear_x: f64, shear_y: f64) -> WasmMoireBuilder {
        self.inner = self.inner.with_shear(shear_x, shear_y);
        self
    }

    /// Set a general 2x2 transformation matrix (flattened array)
    #[wasm_bindgen]
    pub fn with_general_transformation(
        mut self,
        matrix: &[f64],
    ) -> Result<WasmMoireBuilder, JsValue> {
        if matrix.len() != 4 {
            return Err(JsValue::from_str(
                "Matrix must have exactly 4 elements [m00, m01, m10, m11]",
            ));
        }

        let mat = Matrix2::new(matrix[0], matrix[1], matrix[2], matrix[3]);
        self.inner = self.inner.with_general_transformation(mat);
        Ok(self)
    }

    /// Build the Moire2D lattice
    #[wasm_bindgen]
    pub fn build(self) -> Result<WasmMoire2D, JsValue> {
        match self.inner.build() {
            Ok(moire) => Ok(WasmMoire2D { inner: moire }),
            Err(e) => Err(JsValue::from_str(&e)),
        }
    }

    /// Build with JavaScript parameters object
    #[wasm_bindgen]
    pub fn build_with_params(
        lattice: &WasmLattice2D,
        params: &JsValue,
    ) -> Result<WasmMoire2D, JsValue> {
        let params: MoireParams = serde_wasm_bindgen::from_value(params.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse parameters: {}", e)))?;

        let mut builder = WasmMoireBuilder::new().with_base_lattice(lattice);

        if let Some(tol) = params.tolerance {
            builder = builder.with_tolerance(tol);
        }

        match params.transformation_type.to_lowercase().as_str() {
            "rotation" | "rotation_scale" | "twist" => {
                let angle = params.angle_degrees.unwrap_or(0.0);
                let scale = params.scale.unwrap_or(1.0);
                builder = builder.with_twist_and_scale(angle, scale);
            }
            "anisotropic" | "anisotropic_scale" => {
                let scale_x = params.scale_x.unwrap_or(1.0);
                let scale_y = params.scale_y.unwrap_or(1.0);
                builder = builder.with_anisotropic_scale(scale_x, scale_y);
            }
            "shear" => {
                let shear_x = params.shear_x.unwrap_or(0.0);
                let shear_y = params.shear_y.unwrap_or(0.0);
                builder = builder.with_shear(shear_x, shear_y);
            }
            "general" | "matrix" => {
                if let Some(matrix) = params.matrix {
                    builder = builder.with_general_transformation(&matrix)?;
                } else {
                    return Err(JsValue::from_str(
                        "Matrix required for general transformation",
                    ));
                }
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown transformation type: {}. Available: rotation, anisotropic, shear, general",
                    params.transformation_type
                )));
            }
        }

        builder.build()
    }
}

/// Create a simple twisted bilayer moiré pattern
#[wasm_bindgen]
pub fn create_twisted_bilayer(
    lattice: &WasmLattice2D,
    angle_degrees: f64,
) -> Result<WasmMoire2D, JsValue> {
    let angle_radians = angle_degrees * PI / 180.0;
    match twisted_bilayer(lattice.inner.clone(), angle_radians) {
        Ok(moire) => Ok(WasmMoire2D { inner: moire }),
        Err(e) => Err(JsValue::from_str(&e)),
    }
}

/// Create a moiré pattern with commensurate angle
#[wasm_bindgen]
pub fn create_commensurate_moire(
    lattice: &WasmLattice2D,
    m1: i32,
    m2: i32,
    n1: i32,
    n2: i32,
) -> Result<WasmMoire2D, JsValue> {
    match commensurate_moire(lattice.inner.clone(), m1, m2, n1, n2) {
        Ok(moire) => Ok(WasmMoire2D { inner: moire }),
        Err(e) => Err(JsValue::from_str(&e)),
    }
}

/// Create twisted bilayer graphene moiré pattern with magic angle
#[wasm_bindgen]
pub fn create_magic_angle_graphene(lattice: &WasmLattice2D) -> Result<WasmMoire2D, JsValue> {
    // Magic angle for twisted bilayer graphene is approximately 1.05 degrees
    create_twisted_bilayer(lattice, 1.05)
}

/// Create a series of moiré patterns with different twist angles
#[wasm_bindgen]
pub fn create_twist_series(
    lattice: &WasmLattice2D,
    start_angle: f64,
    end_angle: f64,
    num_steps: usize,
) -> Result<Vec<WasmMoire2D>, JsValue> {
    if num_steps == 0 {
        return Err(JsValue::from_str("Number of steps must be greater than 0"));
    }

    let mut moire_series = Vec::new();
    let step_size = if num_steps == 1 {
        0.0
    } else {
        (end_angle - start_angle) / (num_steps - 1) as f64
    };

    for i in 0..num_steps {
        let angle = start_angle + i as f64 * step_size;
        match create_twisted_bilayer(lattice, angle) {
            Ok(moire) => moire_series.push(moire),
            Err(e) => return Err(e),
        }
    }

    Ok(moire_series)
}

/// Get recommended twist angles for studying moiré patterns
#[wasm_bindgen]
pub fn get_recommended_twist_angles() -> Vec<f64> {
    vec![
        0.5,  // Very small angle
        1.05, // Magic angle for graphene
        2.0,  // Small angle
        5.0,  // Medium angle
        10.0, // Large angle
        21.8, // Special commensurate angle
        30.0, // Maximum distinct angle (due to 60° symmetry)
    ]
}

/// Calculate the expected moiré period for a given twist angle and lattice constant
#[wasm_bindgen]
pub fn calculate_moire_period(lattice_constant: f64, twist_angle_degrees: f64) -> f64 {
    let theta = twist_angle_degrees * PI / 180.0;
    // For small angles: L_M ≈ a / (2 * sin(θ/2)) ≈ a / θ (for θ in radians)
    if theta.abs() < 1e-6 {
        return f64::INFINITY; // Essentially no moiré pattern
    }
    lattice_constant / (2.0 * (theta / 2.0).sin())
}

/// Get transformation matrix for rotation and scaling
#[wasm_bindgen]
pub fn get_rotation_scale_matrix(angle_degrees: f64, scale: f64) -> Vec<f64> {
    let angle = angle_degrees * PI / 180.0;
    let c = angle.cos();
    let s = angle.sin();
    vec![scale * c, -scale * s, scale * s, scale * c]
}

/// Get transformation matrix for anisotropic scaling
#[wasm_bindgen]
pub fn get_anisotropic_scale_matrix(scale_x: f64, scale_y: f64) -> Vec<f64> {
    vec![scale_x, 0.0, 0.0, scale_y]
}

/// Get transformation matrix for shear
#[wasm_bindgen]
pub fn get_shear_matrix(shear_x: f64, shear_y: f64) -> Vec<f64> {
    vec![1.0, shear_x, shear_y, 1.0]
}
