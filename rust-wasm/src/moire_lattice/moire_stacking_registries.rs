use wasm_bindgen::prelude::*;
use serde::Serialize;
use nalgebra::{Matrix2, Vector3};
use moire_lattice::moire_lattice::moire_stacking_registries::RegistryCenter as CoreRegistryCenter;
use crate::common::{Point, RegistryCenterData, RegistryCentersResult};
use super::moire2d::WasmMoire2D;

// WASM bindings for moiré stacking registry calculations.
// 
// API Evolution:
// - Recommended: Use `compute_registry_centers_monatomic()` for automatic numerical optimization (wrapped)
// - New: `compute_registry_centers_monatomic_unwrapped()` returns continuous, unwrapped positions
// - Deprecated: `compute_registry_centers_monatomic_from_layers()` and `compute_registry_centers_monatomic_with_theta()`
//   still work but are deprecated in favor of the unified API that auto-detects the best method

#[wasm_bindgen]
pub fn get_monatomic_tau_set(moire: &WasmMoire2D) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    struct TauItem { label: String, tau: Point }

    let items: Vec<TauItem> = moire
        .inner
        .monatomic_tau_set()
        .into_iter()
        .map(|(label, v): (String, Vector3<f64>)| TauItem {
            label,
            tau: Point { x: v.x, y: v.y },
        })
        .collect();

    serde_wasm_bindgen::to_value(&items)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize tau-set: {}", e)))
}

#[wasm_bindgen]
pub fn get_moire_matrix_2x2(moire: &WasmMoire2D) -> Result<Vec<f64>, JsValue> {
    let m: Matrix2<f64> = moire
        .inner
        .moire_matrix_from_layers_2d()
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]])
}

#[wasm_bindgen]
pub fn get_moire_primitives_2x2(moire: &WasmMoire2D) -> Result<Vec<f64>, JsValue> {
    let l: Matrix2<f64> = moire
        .inner
        .moire_primitives_from_layers_2d()
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(vec![l[(0, 0)], l[(0, 1)], l[(1, 0)], l[(1, 1)]])
}

fn to_registry_center_data(c: &CoreRegistryCenter) -> RegistryCenterData {
    RegistryCenterData {
        label: c.label.clone(),
        tau: Point { x: c.tau.x, y: c.tau.y },
        position: Point { x: c.position.x, y: c.position.y },
    }
}

#[wasm_bindgen]
pub fn compute_registry_centers_monatomic(
    moire: &WasmMoire2D,
    d0x: f64,
    d0y: f64,
) -> Result<JsValue, JsValue> {
    let centers = moire
        .inner
        .registry_centers_monatomic(Vector3::new(d0x, d0y, 0.0))
        .map_err(|e| JsValue::from_str(&e))?;

    let centers_js: Vec<RegistryCenterData> = centers.iter().map(to_registry_center_data).collect();
    serde_wasm_bindgen::to_value(&centers_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize registry centers: {}", e)))
}

/// Unwrapped registry centers: returns r_τ = M^{-1}(τ - d0) without wrapping into the moiré cell.
/// Use this to ensure continuity across twist-angle changes and perform wrapping/tiling in the UI.
#[wasm_bindgen]
pub fn compute_registry_centers_monatomic_unwrapped(
    moire: &WasmMoire2D,
    d0x: f64,
    d0y: f64,
) -> Result<JsValue, JsValue> {
    let centers = moire
        .inner
        .registry_centers_monatomic_unwrapped(Vector3::new(d0x, d0y, 0.0))
        .map_err(|e| JsValue::from_str(&e))?;

    let centers_js: Vec<RegistryCenterData> = centers.iter().map(to_registry_center_data).collect();
    serde_wasm_bindgen::to_value(&centers_js)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize unwrapped registry centers: {}", e)))
}

#[allow(deprecated)]
#[wasm_bindgen]
#[deprecated(since = "0.1.2", note = "Use compute_registry_centers_monatomic() for automatic numerical optimization")]
pub fn compute_registry_centers_monatomic_from_layers(
    moire: &WasmMoire2D,
    d0x: f64,
    d0y: f64,
) -> Result<JsValue, JsValue> {
    let (l, centers) = moire
        .inner
        .registry_centers_monatomic_from_layers(Vector3::new(d0x, d0y, 0.0))
        .map_err(|e| JsValue::from_str(&e))?;

    let centers_js: Vec<RegistryCenterData> = centers.iter().map(to_registry_center_data).collect();
    let result = RegistryCentersResult {
        l: vec![l[(0, 0)], l[(0, 1)], l[(1, 0)], l[(1, 1)]],
        centers: centers_js,
    };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize registry centers: {}", e)))
}

#[allow(deprecated)]
#[wasm_bindgen]
#[deprecated(since = "0.1.2", note = "Use compute_registry_centers_monatomic() for automatic numerical optimization")]
pub fn compute_registry_centers_monatomic_with_theta(
    moire: &WasmMoire2D,
    d0x: f64,
    d0y: f64,
) -> Result<JsValue, JsValue> {
    let (l, centers) = moire
        .inner
        .registry_centers_monatomic_with_theta(Vector3::new(d0x, d0y, 0.0))
        .map_err(|e| JsValue::from_str(&e))?;

    let centers_js: Vec<RegistryCenterData> = centers.iter().map(to_registry_center_data).collect();
    let result = RegistryCentersResult {
        l: vec![l[(0, 0)], l[(0, 1)], l[(1, 0)], l[(1, 1)]],
        centers: centers_js,
    };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize registry centers: {}", e)))
}

#[wasm_bindgen]
pub fn compute_registry_centers_monatomic_with_l(
    moire: &WasmMoire2D,
    d0x: f64,
    d0y: f64,
) -> Result<JsValue, JsValue> {
    let (l, centers) = moire
        .inner
        .registry_centers_monatomic_with_l(Vector3::new(d0x, d0y, 0.0))
        .map_err(|e| JsValue::from_str(&e))?;

    let centers_js: Vec<RegistryCenterData> = centers.iter().map(to_registry_center_data).collect();
    let result = RegistryCentersResult {
        l: vec![l[(0, 0)], l[(0, 1)], l[(1, 0)], l[(1, 1)]],
        centers: centers_js,
    };
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize registry centers: {}", e)))
}
