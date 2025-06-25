use wasm_bindgen::prelude::*;
use moire_lattice::lattice::{Bravais2D, Bravais3D, Centering, identify_bravais_2d, identify_bravais_3d};
use nalgebra::Matrix3;

// ======================== BRAVAIS TYPE ENUMS ========================

/// WASM wrapper for Bravais2D enum
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmBravais2D {
    Square,
    Hexagonal,
    Rectangular,
    CenteredRectangular,
    Oblique,
}

impl From<Bravais2D> for WasmBravais2D {
    fn from(bravais: Bravais2D) -> Self {
        match bravais {
            Bravais2D::Square => WasmBravais2D::Square,
            Bravais2D::Hexagonal => WasmBravais2D::Hexagonal,
            Bravais2D::Rectangular => WasmBravais2D::Rectangular,
            Bravais2D::CenteredRectangular => WasmBravais2D::CenteredRectangular,
            Bravais2D::Oblique => WasmBravais2D::Oblique,
        }
    }
}

/// WASM wrapper for Bravais3D enum
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmBravais3D {
    Cubic,
    Tetragonal,
    Orthorhombic,
    Hexagonal,
    Trigonal,
    Monoclinic,
    Triclinic,
}

impl From<Bravais3D> for WasmBravais3D {
    fn from(bravais: Bravais3D) -> Self {
        match bravais {
            Bravais3D::Cubic(_) => WasmBravais3D::Cubic,
            Bravais3D::Tetragonal(_) => WasmBravais3D::Tetragonal,
            Bravais3D::Orthorhombic(_) => WasmBravais3D::Orthorhombic,
            Bravais3D::Hexagonal(_) => WasmBravais3D::Hexagonal,
            Bravais3D::Trigonal(_) => WasmBravais3D::Trigonal,
            Bravais3D::Monoclinic(_) => WasmBravais3D::Monoclinic,
            Bravais3D::Triclinic(_) => WasmBravais3D::Triclinic,
        }
    }
}

/// WASM wrapper for Centering enum
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum WasmCentering {
    Primitive,
    BodyCentered,
    FaceCentered,
    BaseCentered,
}

impl From<Centering> for WasmCentering {
    fn from(centering: Centering) -> Self {
        match centering {
            Centering::Primitive => WasmCentering::Primitive,
            Centering::BodyCentered => WasmCentering::BodyCentered,
            Centering::FaceCentered => WasmCentering::FaceCentered,
            Centering::BaseCentered => WasmCentering::BaseCentered,
        }
    }
}

/// Identify Bravais lattice type for 2D from metric tensor
#[wasm_bindgen]
pub fn identify_bravais_type_2d(metric: &[f64], tolerance: f64) -> Result<WasmBravais2D, JsValue> {
    if metric.len() != 9 {
        return Err(JsValue::from_str("Metric tensor must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(metric);
    let bravais = identify_bravais_2d(&matrix, tolerance);
    Ok(bravais.into())
}

/// Identify Bravais lattice type for 3D from metric tensor
#[wasm_bindgen]
pub fn identify_bravais_type_3d(metric: &[f64], tolerance: f64) -> Result<WasmBravais3D, JsValue> {
    if metric.len() != 9 {
        return Err(JsValue::from_str("Metric tensor must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(metric);
    let bravais = identify_bravais_3d(&matrix, tolerance);
    Ok(bravais.into())
}
