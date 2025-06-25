use wasm_bindgen::prelude::*;
use moire_lattice::lattice::{
    Polyhedron,
    compute_wigner_seitz_cell_2d, compute_wigner_seitz_cell_3d,
    compute_brillouin_zone_2d, compute_brillouin_zone_3d,
};
use nalgebra::{Vector3, Matrix3};
use crate::common::{Point3D, PolyhedronData};

// ======================== POLYHEDRON WRAPPER ========================

/// WASM wrapper for Polyhedron
#[wasm_bindgen]
pub struct WasmPolyhedron {
    pub(crate) inner: Polyhedron,
}

#[wasm_bindgen]
impl WasmPolyhedron {
    /// Check if a 2D point is inside the polyhedron
    #[wasm_bindgen]
    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        let point = Vector3::new(x, y, 0.0);
        self.inner.contains_2d(point)
    }

    /// Check if a 3D point is inside the polyhedron
    #[wasm_bindgen]
    pub fn contains_3d(&self, x: f64, y: f64, z: f64) -> bool {
        let point = Vector3::new(x, y, z);
        self.inner.contains_3d(point)
    }

    /// Get the measure (area for 2D, volume for 3D)
    #[wasm_bindgen]
    pub fn measure(&self) -> f64 {
        self.inner.measure()
    }

    /// Get polyhedron data as JavaScript object
    #[wasm_bindgen]
    pub fn get_data(&self) -> Result<JsValue, JsValue> {
        let vertices: Vec<Point3D> = self.inner.vertices()
            .iter()
            .map(|v| Point3D { x: v.x, y: v.y, z: v.z })
            .collect();

        let data = PolyhedronData {
            vertices,
            edges: self.inner.edges().clone(),
            faces: self.inner.faces().clone(),
            measure: self.inner.measure(),
        };

        serde_wasm_bindgen::to_value(&data)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize polyhedron data: {}", e)))
    }
}

/// Compute Wigner-Seitz cell for 2D lattice
#[wasm_bindgen]
pub fn compute_wigner_seitz_2d(basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if basis.len() != 9 {
        return Err(JsValue::from_str("Basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(basis);
    let polyhedron = compute_wigner_seitz_cell_2d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Wigner-Seitz cell for 3D lattice
#[wasm_bindgen]
pub fn compute_wigner_seitz_3d(basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if basis.len() != 9 {
        return Err(JsValue::from_str("Basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(basis);
    let polyhedron = compute_wigner_seitz_cell_3d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Brillouin zone for 2D lattice
#[wasm_bindgen]
pub fn compute_brillouin_2d(reciprocal_basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if reciprocal_basis.len() != 9 {
        return Err(JsValue::from_str("Reciprocal basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(reciprocal_basis);
    let polyhedron = compute_brillouin_zone_2d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Brillouin zone for 3D lattice
#[wasm_bindgen]
pub fn compute_brillouin_3d(reciprocal_basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if reciprocal_basis.len() != 9 {
        return Err(JsValue::from_str("Reciprocal basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(reciprocal_basis);
    let polyhedron = compute_brillouin_zone_3d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}
