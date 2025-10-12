use crate::lattice::bravais_types::WasmBravais2D;
use crate::lattice::lattice2d::WasmLattice2D;
use moire_lattice::lattice::{
    analyze_bravais_type_2d, determine_bravais_type_2d, lattice_types::approx_equal,
    validate_bravais_type_2d,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ======================== LATTICE ANALYSIS RESULT ========================

/// Analysis result for JavaScript
#[derive(Serialize, Deserialize)]
pub struct LatticeAnalysisResult {
    pub bravais_type: String,
    pub reason: String,
    pub lattice_parameters: LatticeParameters,
}

#[derive(Serialize, Deserialize)]
pub struct LatticeParameters {
    pub a: f64,
    pub b: f64,
    pub gamma_degrees: f64,
    pub gamma_radians: f64,
}

// ======================== VALIDATION FUNCTIONS ========================

/// Determine the Bravais lattice type from a 2D lattice
#[wasm_bindgen]
pub fn determine_lattice_type_2d(lattice: &WasmLattice2D) -> WasmBravais2D {
    let bravais_type = determine_bravais_type_2d(&lattice.inner);
    bravais_type.into()
}

/// Validate that a 2D lattice's stored Bravais type matches its actual structure
#[wasm_bindgen]
pub fn validate_lattice_type_2d(lattice: &WasmLattice2D) -> bool {
    validate_bravais_type_2d(&lattice.inner)
}

/// Get detailed analysis of why a lattice has a particular Bravais type
#[wasm_bindgen]
pub fn analyze_lattice_type_2d(lattice: &WasmLattice2D) -> Result<JsValue, JsValue> {
    let (bravais_type, reason) = analyze_bravais_type_2d(&lattice.inner);
    let (a, b) = lattice.inner.lattice_parameters();
    let gamma = lattice.inner.lattice_angle();

    let result = LatticeAnalysisResult {
        bravais_type: format!("{:?}", bravais_type).to_lowercase(),
        reason,
        lattice_parameters: LatticeParameters {
            a,
            b,
            gamma_degrees: gamma.to_degrees(),
            gamma_radians: gamma,
        },
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize analysis result: {}", e)))
}

/// Get the string representation of a Bravais type
#[wasm_bindgen]
pub fn bravais_type_to_string(bravais_type: WasmBravais2D) -> String {
    match bravais_type {
        WasmBravais2D::Square => "square".to_string(),
        WasmBravais2D::Hexagonal => "hexagonal".to_string(),
        WasmBravais2D::Rectangular => "rectangular".to_string(),
        WasmBravais2D::CenteredRectangular => "centered_rectangular".to_string(),
        WasmBravais2D::Oblique => "oblique".to_string(),
    }
}

/// Compare two Bravais types for equality
#[wasm_bindgen]
pub fn bravais_types_equal(a: WasmBravais2D, b: WasmBravais2D) -> bool {
    // Convert to the core enum for proper comparison
    let a_core: moire_lattice::lattice::Bravais2D = match a {
        WasmBravais2D::Square => moire_lattice::lattice::Bravais2D::Square,
        WasmBravais2D::Hexagonal => moire_lattice::lattice::Bravais2D::Hexagonal,
        WasmBravais2D::Rectangular => moire_lattice::lattice::Bravais2D::Rectangular,
        WasmBravais2D::CenteredRectangular => {
            moire_lattice::lattice::Bravais2D::CenteredRectangular
        }
        WasmBravais2D::Oblique => moire_lattice::lattice::Bravais2D::Oblique,
    };

    let b_core: moire_lattice::lattice::Bravais2D = match b {
        WasmBravais2D::Square => moire_lattice::lattice::Bravais2D::Square,
        WasmBravais2D::Hexagonal => moire_lattice::lattice::Bravais2D::Hexagonal,
        WasmBravais2D::Rectangular => moire_lattice::lattice::Bravais2D::Rectangular,
        WasmBravais2D::CenteredRectangular => {
            moire_lattice::lattice::Bravais2D::CenteredRectangular
        }
        WasmBravais2D::Oblique => moire_lattice::lattice::Bravais2D::Oblique,
    };

    a_core == b_core
}

// ======================== UTILITY FUNCTIONS FOR TESTING ========================

/// Check if two floating point values are approximately equal (exposed for testing)
#[wasm_bindgen]
pub fn approx_equal_wasm(a: f64, b: f64, tolerance: f64) -> bool {
    approx_equal(a, b, tolerance)
}

/// Get lattice parameters from a 2D lattice (exposed for convenience)
#[wasm_bindgen]
pub fn get_lattice_parameters_2d(lattice: &WasmLattice2D) -> Result<JsValue, JsValue> {
    let (a, b) = lattice.inner.lattice_parameters();
    let gamma = lattice.inner.lattice_angle();

    let params = LatticeParameters {
        a,
        b,
        gamma_degrees: gamma.to_degrees(),
        gamma_radians: gamma,
    };

    serde_wasm_bindgen::to_value(&params)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize lattice parameters: {}", e)))
}

// ======================== BATCH VALIDATION FUNCTIONS ========================

/// Validate multiple lattices at once (for efficiency)
#[wasm_bindgen]
pub fn validate_multiple_lattices_2d(lattices_data: &JsValue) -> Result<JsValue, JsValue> {
    // Expect an array of lattice parameter objects
    let lattices_array: js_sys::Array = js_sys::Array::from(lattices_data);
    let results = js_sys::Array::new();

    for i in 0..lattices_array.length() {
        let lattice_data = lattices_array.get(i);

        // Try to deserialize the lattice data and create a WasmLattice2D
        let lattice = WasmLattice2D::new(&lattice_data).map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to create lattice at index {}: {}",
                i,
                e.as_string().unwrap_or_default()
            ))
        })?;

        let is_valid = validate_bravais_type_2d(&lattice.inner);
        results.push(&JsValue::from_bool(is_valid));
    }

    Ok(results.into())
}

/// Analyze multiple lattices at once (for efficiency)
#[wasm_bindgen]
pub fn analyze_multiple_lattices_2d(lattices_data: &JsValue) -> Result<JsValue, JsValue> {
    // Expect an array of lattice parameter objects
    let lattices_array: js_sys::Array = js_sys::Array::from(lattices_data);
    let results = js_sys::Array::new();

    for i in 0..lattices_array.length() {
        let lattice_data = lattices_array.get(i);

        // Try to deserialize the lattice data and create a WasmLattice2D
        let lattice = WasmLattice2D::new(&lattice_data).map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to create lattice at index {}: {}",
                i,
                e.as_string().unwrap_or_default()
            ))
        })?;

        let analysis_result = analyze_lattice_type_2d(&lattice)?;
        results.push(&analysis_result);
    }

    Ok(results.into())
}

/// Determine Bravais types for multiple lattices at once (for efficiency)
#[wasm_bindgen]
pub fn determine_multiple_lattice_types_2d(lattices_data: &JsValue) -> Result<JsValue, JsValue> {
    // Expect an array of lattice parameter objects
    let lattices_array: js_sys::Array = js_sys::Array::from(lattices_data);
    let results = js_sys::Array::new();

    for i in 0..lattices_array.length() {
        let lattice_data = lattices_array.get(i);

        // Try to deserialize the lattice data and create a WasmLattice2D
        let lattice = WasmLattice2D::new(&lattice_data).map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to create lattice at index {}: {}",
                i,
                e.as_string().unwrap_or_default()
            ))
        })?;

        let bravais_type = determine_bravais_type_2d(&lattice.inner);
        let wasm_type: WasmBravais2D = bravais_type.into();
        let type_string = bravais_type_to_string(wasm_type);
        results.push(&JsValue::from_str(&type_string));
    }

    Ok(results.into())
}

// ======================== ANGLE VALIDATION UTILITIES ========================

/// Check if an angle is equivalent to 90 degrees (considering crystallographic equivalences)
/// Useful for testing and validation in JavaScript
#[wasm_bindgen]
pub fn is_angle_equivalent_to_90_degrees(angle_radians: f64, tolerance: f64) -> bool {
    use std::f64::consts::PI;

    // Normalize angle to [0, 2π) range
    let normalized = angle_radians.rem_euclid(2.0 * PI);

    // Check against 90° (π/2) and 270° (3π/2)
    approx_equal(normalized, PI / 2.0, tolerance)
        || approx_equal(normalized, 3.0 * PI / 2.0, tolerance)
}

/// Check if an angle is equivalent to hexagonal angles (considering crystallographic equivalences)
/// Useful for testing and validation in JavaScript
#[wasm_bindgen]
pub fn is_angle_equivalent_to_hexagonal(angle_radians: f64, tolerance: f64) -> bool {
    use std::f64::consts::PI;

    // Normalize angle to [0, 2π) range
    let normalized = angle_radians.rem_euclid(2.0 * PI);

    // Check against 60° (π/3), 120° (2π/3), 240° (4π/3), 300° (5π/3)
    approx_equal(normalized, PI / 3.0, tolerance) ||           // 60°
    approx_equal(normalized, 2.0 * PI / 3.0, tolerance) ||     // 120°
    approx_equal(normalized, 4.0 * PI / 3.0, tolerance) ||     // 240°
    approx_equal(normalized, 5.0 * PI / 3.0, tolerance) // 300°
}

/// Convert degrees to radians (convenience function for JavaScript)
#[wasm_bindgen]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees (convenience function for JavaScript)
#[wasm_bindgen]
pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}
