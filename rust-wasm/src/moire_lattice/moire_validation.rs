use wasm_bindgen::prelude::*;
use moire_lattice::moire_lattice::moire_validation_algorithms::{
    find_commensurate_angles, validate_commensurability, compute_moire_basis,
    analyze_moire_symmetry, moire_potential_at,
};
use nalgebra::Vector3;
use serde::Serialize;
use crate::lattice::WasmLattice2D;
use super::moire2d::WasmMoire2D;

/// Find commensurate angles for a given lattice
#[wasm_bindgen]
pub fn find_commensurate_angles_wasm(
    lattice: &WasmLattice2D,
    max_index: i32,
) -> Result<JsValue, JsValue> {
    match find_commensurate_angles(&lattice.inner, max_index) {
        Ok(angles) => {
            #[derive(Serialize)]
            struct CommensurateAngle {
                angle_degrees: f64,
                angle_radians: f64,
                indices: Vec<i32>,
            }
            
            let result: Vec<CommensurateAngle> = angles
                .into_iter()
                .map(|(angle, (m1, m2, n1, n2))| CommensurateAngle {
                    angle_degrees: angle * 180.0 / std::f64::consts::PI,
                    angle_radians: angle,
                    indices: vec![m1, m2, n1, n2],
                })
                .collect();

            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize angles: {}", e)))
        }
        Err(e) => Err(JsValue::from_str(&e)),
    }
}

/// Validate commensurability between two lattices
#[wasm_bindgen]
pub fn validate_commensurability_wasm(
    lattice_1: &WasmLattice2D,
    lattice_2: &WasmLattice2D,
    tolerance: f64,
) -> Result<JsValue, JsValue> {
    let (is_commensurate, indices) = validate_commensurability(
        &lattice_1.inner,
        &lattice_2.inner,
        tolerance,
    );

    #[derive(Serialize)]
    struct CommensurabilityResult {
        is_commensurate: bool,
        coincidence_indices: Option<Vec<i32>>,
    }

    let result = CommensurabilityResult {
        is_commensurate,
        coincidence_indices: indices.map(|(m1, m2, n1, n2)| vec![m1, m2, n1, n2]),
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
}

/// Compute moiré basis vectors from two lattices
#[wasm_bindgen]
pub fn compute_moire_basis_wasm(
    lattice_1: &WasmLattice2D,
    lattice_2: &WasmLattice2D,
    tolerance: f64,
) -> Result<Vec<f64>, JsValue> {
    match compute_moire_basis(&lattice_1.inner, &lattice_2.inner, tolerance) {
        Ok(basis) => {
            // Convert 3x3 matrix to flattened array
            let mut result = Vec::with_capacity(9);
            for i in 0..3 {
                for j in 0..3 {
                    result.push(basis[(i, j)]);
                }
            }
            Ok(result)
        }
        Err(e) => Err(JsValue::from_str(&e)),
    }
}

/// Analyze symmetries preserved in the moiré pattern
#[wasm_bindgen]
pub fn analyze_moire_symmetry_wasm(moire: &WasmMoire2D) -> Vec<String> {
    analyze_moire_symmetry(&moire.inner)
}

/// Compute moiré potential at a given point
#[wasm_bindgen]
pub fn moire_potential_at_wasm(
    moire: &WasmMoire2D,
    x: f64,
    y: f64,
    v_aa: f64,
    v_ab: f64,
) -> f64 {
    let point = Vector3::new(x, y, 0.0);
    moire_potential_at(&moire.inner, point, v_aa, v_ab)
}

/// Compute moiré potential over a grid for visualization
#[wasm_bindgen]
pub fn compute_moire_potential_grid(
    moire: &WasmMoire2D,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    nx: usize,
    ny: usize,
    v_aa: f64,
    v_ab: f64,
) -> Result<JsValue, JsValue> {
    if nx == 0 || ny == 0 {
        return Err(JsValue::from_str("Grid dimensions must be greater than 0"));
    }

    #[derive(Serialize)]
    struct PotentialGrid {
        x_coords: Vec<f64>,
        y_coords: Vec<f64>,
        potential: Vec<Vec<f64>>,
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        nx: usize,
        ny: usize,
    }

    let dx = if nx == 1 { 0.0 } else { (x_max - x_min) / (nx - 1) as f64 };
    let dy = if ny == 1 { 0.0 } else { (y_max - y_min) / (ny - 1) as f64 };

    let mut x_coords = Vec::with_capacity(nx);
    let mut y_coords = Vec::with_capacity(ny);
    let mut potential = Vec::with_capacity(ny);

    for i in 0..nx {
        x_coords.push(x_min + i as f64 * dx);
    }

    for j in 0..ny {
        y_coords.push(y_min + j as f64 * dy);
        let mut row = Vec::with_capacity(nx);
        
        for i in 0..nx {
            let x = x_coords[i];
            let y = y_coords[j];
            let pot = moire_potential_at_wasm(moire, x, y, v_aa, v_ab);
            row.push(pot);
        }
        potential.push(row);
    }

    let grid = PotentialGrid {
        x_coords,
        y_coords,
        potential,
        x_min,
        x_max,
        y_min,
        y_max,
        nx,
        ny,
    };

    serde_wasm_bindgen::to_value(&grid)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize grid: {}", e)))
}

/// Find magic angles (special commensurate angles with interesting properties)
#[wasm_bindgen]
pub fn find_magic_angles(lattice: &WasmLattice2D) -> Result<JsValue, JsValue> {
    // Magic angles are typically small commensurate angles
    match find_commensurate_angles(&lattice.inner, 30) {
        Ok(angles) => {
            #[derive(Serialize)]
            struct MagicAngle {
                angle_degrees: f64,
                angle_radians: f64,
                indices: Vec<i32>,
                period_ratio: f64,
                is_magic: bool,
            }
            
            let result: Vec<MagicAngle> = angles
                .into_iter()
                .filter(|(angle, _)| {
                    let angle_deg = angle.abs() * 180.0 / std::f64::consts::PI;
                    angle_deg > 0.1 && angle_deg < 10.0 // Focus on small angles
                })
                .map(|(angle, (m1, m2, n1, n2))| {
                    let angle_deg = angle * 180.0 / std::f64::consts::PI;
                    let period_ratio = 1.0 / (2.0 * (angle / 2.0).sin());
                    let is_magic = (angle_deg - 1.05).abs() < 0.1; // Close to graphene magic angle
                    
                    MagicAngle {
                        angle_degrees: angle_deg,
                        angle_radians: angle,
                        indices: vec![m1, m2, n1, n2],
                        period_ratio,
                        is_magic,
                    }
                })
                .collect();

            serde_wasm_bindgen::to_value(&result)
                .map_err(|e| JsValue::from_str(&format!("Failed to serialize magic angles: {}", e)))
        }
        Err(e) => Err(JsValue::from_str(&e)),
    }
}

/// Analyze the quality of a moiré pattern based on commensurability and period
#[wasm_bindgen]
pub fn analyze_moire_quality(moire: &WasmMoire2D) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    struct MoireQuality {
        is_commensurate: bool,
        period_ratio: f64,
        twist_angle_degrees: f64,
        quality_score: f64,
        quality_description: String,
        symmetries: Vec<String>,
    }

    let period_ratio = moire.moire_period_ratio();
    let twist_angle = moire.twist_angle_degrees();
    let symmetries = analyze_moire_symmetry(&moire.inner);
    
    // Calculate quality score based on various factors
    let mut quality_score = 0.0;
    
    // Prefer commensurate patterns
    if moire.is_commensurate() {
        quality_score += 0.3;
    }
    
    // Prefer reasonable period ratios (not too large, not too small)
    if period_ratio > 2.0 && period_ratio < 100.0 {
        quality_score += 0.3;
    }
    
    // Prefer certain special angles
    if (twist_angle - 1.05).abs() < 0.1 {
        quality_score += 0.2; // Magic angle
    } else if twist_angle > 0.5 && twist_angle < 30.0 {
        quality_score += 0.1; // Reasonable angle range
    }
    
    // Bonus for preserved symmetries
    quality_score += symmetries.len() as f64 * 0.05;
    
    // Cap at 1.0
    quality_score = quality_score.min(1.0);
    
    let quality_description = if quality_score > 0.8 {
        "Excellent".to_string()
    } else if quality_score > 0.6 {
        "Good".to_string()
    } else if quality_score > 0.4 {
        "Fair".to_string()
    } else {
        "Poor".to_string()
    };

    let quality = MoireQuality {
        is_commensurate: moire.is_commensurate(),
        period_ratio,
        twist_angle_degrees: twist_angle,
        quality_score,
        quality_description,
        symmetries,
    };

    serde_wasm_bindgen::to_value(&quality)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize quality analysis: {}", e)))
}

/// Get theoretical predictions for moiré properties
#[wasm_bindgen]
pub fn get_moire_predictions(
    lattice_constant: f64,
    twist_angle_degrees: f64,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    struct MoirePredictions {
        twist_angle_degrees: f64,
        twist_angle_radians: f64,
        expected_period: f64,
        expected_period_ratio: f64,
        is_small_angle: bool,
        is_magic_angle: bool,
        theoretical_band_gap: Option<f64>,
    }

    let twist_angle_rad = twist_angle_degrees * std::f64::consts::PI / 180.0;
    let expected_period = if twist_angle_rad.abs() > 1e-6 {
        lattice_constant / (2.0 * (twist_angle_rad / 2.0).sin())
    } else {
        f64::INFINITY
    };
    
    let expected_period_ratio = expected_period / lattice_constant;
    let is_small_angle = twist_angle_degrees.abs() < 5.0;
    let is_magic_angle = (twist_angle_degrees.abs() - 1.05).abs() < 0.1;
    
    // Theoretical band gap for twisted bilayer graphene (simplified model)
    let theoretical_band_gap = if is_magic_angle {
        Some(2.0) // Approximate band gap in meV for magic angle graphene
    } else {
        None
    };

    let predictions = MoirePredictions {
        twist_angle_degrees,
        twist_angle_radians: twist_angle_rad,
        expected_period,
        expected_period_ratio,
        is_small_angle,
        is_magic_angle,
        theoretical_band_gap,
    };

    serde_wasm_bindgen::to_value(&predictions)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize predictions: {}", e)))
}
