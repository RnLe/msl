use crate::common::{Point, Point3D};
use crate::lattice::WasmLattice2D;
use moire_lattice::moire_lattice::{Moire2D, MoireTransformation};
use nalgebra::{Matrix2, Vector3};
use serde::Serialize;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// WASM wrapper for MoireTransformation enum
#[wasm_bindgen]
#[derive(Clone)]
pub enum WasmMoireTransformation {
    RotationScale,
    AnisotropicScale,
    Shear,
    General,
}

/// WASM wrapper for 2D moiré lattice
#[wasm_bindgen]
pub struct WasmMoire2D {
    pub(crate) inner: Moire2D,
}

#[wasm_bindgen]
impl WasmMoire2D {
    /// Get the moiré lattice as a regular 2D lattice
    #[wasm_bindgen]
    pub fn as_lattice2d(&self) -> WasmLattice2D {
        WasmLattice2D {
            inner: self.inner.as_lattice2d(),
        }
    }

    /// Get moiré primitive vectors as JavaScript object
    #[wasm_bindgen]
    pub fn primitive_vectors(&self) -> Result<JsValue, JsValue> {
        let (a_vec, b_vec) = self.inner.primitive_vectors();

        #[derive(Serialize)]
        struct Vectors {
            a: Point,
            b: Point,
        }

        let vectors = Vectors {
            a: Point {
                x: a_vec.x,
                y: a_vec.y,
            },
            b: Point {
                x: b_vec.x,
                y: b_vec.y,
            },
        };

        serde_wasm_bindgen::to_value(&vectors)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize vectors: {}", e)))
    }

    /// Get the moiré periodicity ratio
    #[wasm_bindgen]
    pub fn moire_period_ratio(&self) -> f64 {
        self.inner.moire_period_ratio()
    }

    /// Check if a point belongs to lattice 1
    #[wasm_bindgen]
    pub fn is_lattice1_point(&self, x: f64, y: f64) -> bool {
        let point = Vector3::new(x, y, 0.0);
        self.inner.is_lattice1_point(point)
    }

    /// Check if a point belongs to lattice 2
    #[wasm_bindgen]
    pub fn is_lattice2_point(&self, x: f64, y: f64) -> bool {
        let point = Vector3::new(x, y, 0.0);
        self.inner.is_lattice2_point(point)
    }

    /// Get stacking type at a given position
    #[wasm_bindgen]
    pub fn get_stacking_at(&self, x: f64, y: f64) -> Option<String> {
        let point = Vector3::new(x, y, 0.0);
        self.inner.get_stacking_at(point)
    }

    /// Get the twist angle in degrees
    #[wasm_bindgen]
    pub fn twist_angle_degrees(&self) -> f64 {
        self.inner.twist_angle * 180.0 / PI
    }

    /// Get the twist angle in radians
    #[wasm_bindgen]
    pub fn twist_angle_radians(&self) -> f64 {
        self.inner.twist_angle
    }

    /// Check if the moiré lattice is commensurate
    #[wasm_bindgen]
    pub fn is_commensurate(&self) -> bool {
        self.inner.is_commensurate
    }

    /// Get coincidence indices if commensurate
    #[wasm_bindgen]
    pub fn coincidence_indices(&self) -> Option<Vec<i32>> {
        self.inner
            .coincidence_indices
            .map(|(m1, m2, n1, n2)| vec![m1, m2, n1, n2])
    }

    /// Get the first constituent lattice
    #[wasm_bindgen]
    pub fn lattice_1(&self) -> WasmLattice2D {
        WasmLattice2D {
            inner: self.inner.lattice_1.clone(),
        }
    }

    /// Get the second constituent lattice
    #[wasm_bindgen]
    pub fn lattice_2(&self) -> WasmLattice2D {
        WasmLattice2D {
            inner: self.inner.lattice_2.clone(),
        }
    }

    /// Get unit cell area of the moiré lattice
    #[wasm_bindgen]
    pub fn cell_area(&self) -> f64 {
        self.inner.cell_area
    }

    /// Get transformation matrix as JavaScript array (flattened 2x2 matrix)
    #[wasm_bindgen]
    pub fn transformation_matrix(&self) -> Vec<f64> {
        let matrix = self.inner.transformation.to_matrix();
        vec![
            matrix[(0, 0)],
            matrix[(0, 1)],
            matrix[(1, 0)],
            matrix[(1, 1)],
        ]
    }

    /// Get lattice parameters as JavaScript object
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        struct MoireParams {
            twist_angle_degrees: f64,
            twist_angle_radians: f64,
            period_ratio: f64,
            is_commensurate: bool,
            coincidence_indices: Option<Vec<i32>>,
            cell_area: f64,
        }

        let params = MoireParams {
            twist_angle_degrees: self.twist_angle_degrees(),
            twist_angle_radians: self.twist_angle_radians(),
            period_ratio: self.moire_period_ratio(),
            is_commensurate: self.is_commensurate(),
            coincidence_indices: self.coincidence_indices(),
            cell_area: self.cell_area(),
        };

        serde_wasm_bindgen::to_value(&params)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize parameters: {}", e)))
    }

    /// Generate lattice points within a radius for visualization
    #[wasm_bindgen]
    pub fn generate_moire_points(&self, radius: f64) -> Result<JsValue, JsValue> {
        // Use the base lattice functionality
        let lattice = self.as_lattice2d();
        lattice.generate_points(radius, 0.0, 0.0)
    }

    /// Generate lattice 1 points within a radius
    #[wasm_bindgen]
    pub fn generate_lattice1_points(&self, radius: f64) -> Result<JsValue, JsValue> {
        let lattice1_wasm = WasmLattice2D {
            inner: self.inner.lattice_1.clone(),
        };
        lattice1_wasm.generate_points(radius, 0.0, 0.0)
    }

    /// Generate lattice 2 points within a radius
    #[wasm_bindgen]
    pub fn generate_lattice2_points(&self, radius: f64) -> Result<JsValue, JsValue> {
        let lattice2_wasm = WasmLattice2D {
            inner: self.inner.lattice_2.clone(),
        };
        lattice2_wasm.generate_points(radius, 0.0, 0.0)
    }

    /// Get stacking analysis for points within a radius
    #[wasm_bindgen]
    pub fn analyze_stacking_in_region(
        &self,
        radius: f64,
        grid_spacing: f64,
    ) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        struct StackingPoint {
            x: f64,
            y: f64,
            stacking: Option<String>,
        }

        let mut stacking_points = Vec::new();
        let steps = (2.0 * radius / grid_spacing) as i32;

        for i in -steps..=steps {
            for j in -steps..=steps {
                let x = i as f64 * grid_spacing;
                let y = j as f64 * grid_spacing;

                if x * x + y * y <= radius * radius {
                    let stacking = self.get_stacking_at(x, y);
                    stacking_points.push(StackingPoint { x, y, stacking });
                }
            }
        }

        serde_wasm_bindgen::to_value(&stacking_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize stacking data: {}", e)))
    }

    /// Convert fractional to cartesian coordinates using moiré basis
    #[wasm_bindgen]
    pub fn frac_to_cart(&self, fx: f64, fy: f64) -> Result<JsValue, JsValue> {
        let lattice = self.as_lattice2d();
        lattice.frac_to_cart(fx, fy)
    }

    /// Convert cartesian to fractional coordinates using moiré basis
    #[wasm_bindgen]
    pub fn cart_to_frac(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let lattice = self.as_lattice2d();
        lattice.cart_to_frac(x, y)
    }
}
