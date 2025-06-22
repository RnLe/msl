use wasm_bindgen::prelude::*;
use moire_lattice::lattice::{
    Lattice2D,
    construction::{square_lattice, hexagonal_lattice, rectangular_lattice, oblique_lattice},
    voronoi_cells::generate_lattice_points_2d_within_radius
};
use serde::{Deserialize, Serialize};
use nalgebra::Vector3;
use std::f64::consts::PI;

// Enable console logging and panic hooks for debugging
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Point structure for JavaScript interop
#[derive(Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Lattice parameters for JavaScript
#[derive(Serialize, Deserialize)]
pub struct LatticeParams {
    pub lattice_type: String,
    pub a: f64,
    pub b: Option<f64>,
    pub angle: Option<f64>,
}

/// WASM wrapper for 2D lattice
#[wasm_bindgen]
pub struct WasmLattice2D {
    inner: Lattice2D,
}

#[wasm_bindgen]
impl WasmLattice2D {
    /// Create a new lattice from JavaScript parameters
    #[wasm_bindgen(constructor)]
    pub fn new(params: &JsValue) -> Result<WasmLattice2D, JsValue> {
        let params: LatticeParams = serde_wasm_bindgen::from_value(params.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse parameters: {}", e)))?;

        let lattice = match params.lattice_type.to_lowercase().as_str() {
            "square" => square_lattice(params.a),
            "rectangular" => {
                let b_val = params.b.unwrap_or(params.a);
                rectangular_lattice(params.a, b_val)
            },
            "hexagonal" | "triangular" => hexagonal_lattice(params.a),
            "oblique" => {
                let b_val = params.b.unwrap_or(params.a);
                let angle_val = params.angle.unwrap_or(90.0) * PI / 180.0; // Convert to radians
                oblique_lattice(params.a, b_val, angle_val)
            },
            _ => return Err(JsValue::from_str(&format!("Unknown lattice type: {}", params.lattice_type))),
        };

        Ok(WasmLattice2D { inner: lattice })
    }

    /// Generate lattice points within a radius
    #[wasm_bindgen]
    pub fn generate_points(&self, radius: f64, center_x: f64, center_y: f64) -> Result<JsValue, JsValue> {
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius * 1.5); // Add some margin
        
        let center_vec = Vector3::new(center_x, center_y, 0.0);
        let filtered_points: Vec<Point> = points
            .into_iter()
            .filter(|p| {
                let dist = (p - center_vec).norm();
                dist <= radius
            })
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&filtered_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Get lattice parameters as JavaScript object
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> Result<JsValue, JsValue> {
        let (a, b) = self.inner.lattice_parameters();
        let angle = self.inner.lattice_angle() * 180.0 / PI; // Convert to degrees
        
        let params = LatticeParams {
            lattice_type: format!("{:?}", self.inner.bravais),
            a,
            b: Some(b),
            angle: Some(angle),
        };

        serde_wasm_bindgen::to_value(&params)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize parameters: {}", e)))
    }

    /// Get unit cell area
    #[wasm_bindgen]
    pub fn unit_cell_area(&self) -> f64 {
        self.inner.cell_area
    }

    /// Get lattice vectors as JavaScript object
    #[wasm_bindgen]
    pub fn lattice_vectors(&self) -> Result<JsValue, JsValue> {
        let (a_vec, b_vec) = self.inner.primitive_vectors();
        
        #[derive(Serialize)]
        struct Vectors {
            a: Point,
            b: Point,
        }
        
        let vectors = Vectors {
            a: Point { x: a_vec.x, y: a_vec.y },
            b: Point { x: b_vec.x, y: b_vec.y },
        };

        serde_wasm_bindgen::to_value(&vectors)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize vectors: {}", e)))
    }

    /// Get reciprocal lattice vectors
    #[wasm_bindgen]
    pub fn reciprocal_vectors(&self) -> Result<JsValue, JsValue> {
        let g1 = self.inner.reciprocal_basis().column(0);
        let g2 = self.inner.reciprocal_basis().column(1);
        
        #[derive(Serialize)]
        struct ReciprocalVectors {
            g1: Point,
            g2: Point,
        }
        
        let vectors = ReciprocalVectors {
            g1: Point { x: g1.x, y: g1.y },
            g2: Point { x: g2.x, y: g2.y },
        };

        serde_wasm_bindgen::to_value(&vectors)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize reciprocal vectors: {}", e)))
    }

    /// Generate an SVG representation of the lattice
    #[wasm_bindgen]
    pub fn to_svg(&self, width: f64, height: f64, radius: f64) -> String {
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius);
        
        let mut svg = format!(
            r#"<svg width="{}" height="{}" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            width, height, -radius, -radius, 2.0 * radius, 2.0 * radius
        );

        // Add lattice points
        for point in &points {
            if point.norm() <= radius {
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="0.05" fill="blue" />"#,
                    point.x, point.y
                ));
            }
        }

        // Add lattice vectors
        let (a_vec, b_vec) = self.inner.primitive_vectors();
        svg.push_str(&format!(
            r#"<line x1="0" y1="0" x2="{}" y2="{}" stroke="red" stroke-width="0.02" />"#,
            a_vec.x, a_vec.y
        ));
        svg.push_str(&format!(
            r#"<line x1="0" y1="0" x2="{}" y2="{}" stroke="green" stroke-width="0.02" />"#,
            b_vec.x, b_vec.y
        ));

        svg.push_str("</svg>");
        svg
    }
}

/// Utility functions for creating common lattices

#[wasm_bindgen]
pub fn create_square_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: square_lattice(a),
    })
}

#[wasm_bindgen]
pub fn create_hexagonal_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: hexagonal_lattice(a),
    })
}

#[wasm_bindgen]
pub fn create_rectangular_lattice(a: f64, b: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: rectangular_lattice(a, b),
    })
}

/// Get the version of the library
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
