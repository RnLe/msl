use super::lattice2d::WasmLattice2D;
use super::lattice3d::WasmLattice3D;
use moire_lattice::lattice::lattice_construction::*;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// ======================== 2D LATTICE CONSTRUCTION FUNCTIONS ========================

/// Create square lattice
#[wasm_bindgen]
pub fn create_square_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: square_lattice(a),
    })
}

/// Create hexagonal lattice
#[wasm_bindgen]
pub fn create_hexagonal_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: hexagonal_lattice(a),
    })
}

/// Create rectangular lattice
#[wasm_bindgen]
pub fn create_rectangular_lattice(a: f64, b: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: rectangular_lattice(a, b),
    })
}

/// Create centered rectangular lattice
#[wasm_bindgen]
pub fn create_centered_rectangular_lattice(a: f64, b: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: centered_rectangular_lattice(a, b),
    })
}

/// Create oblique lattice
#[wasm_bindgen]
pub fn create_oblique_lattice(
    a: f64,
    b: f64,
    gamma_degrees: f64,
) -> Result<WasmLattice2D, JsValue> {
    let gamma = gamma_degrees * PI / 180.0;
    Ok(WasmLattice2D {
        inner: oblique_lattice(a, b, gamma),
    })
}

// ======================== 3D LATTICE CONSTRUCTION FUNCTIONS ========================

/// Create body-centered cubic lattice
#[wasm_bindgen]
pub fn create_body_centered_cubic_lattice(a: f64) -> Result<WasmLattice3D, JsValue> {
    Ok(WasmLattice3D {
        inner: body_centered_cubic_lattice(a),
    })
}

/// Create face-centered cubic lattice
#[wasm_bindgen]
pub fn create_face_centered_cubic_lattice(a: f64) -> Result<WasmLattice3D, JsValue> {
    Ok(WasmLattice3D {
        inner: face_centered_cubic_lattice(a),
    })
}

/// Create hexagonal close-packed lattice
#[wasm_bindgen]
pub fn create_hexagonal_close_packed_lattice(a: f64, c: f64) -> Result<WasmLattice3D, JsValue> {
    Ok(WasmLattice3D {
        inner: hexagonal_close_packed_lattice(a, c),
    })
}

/// Create tetragonal lattice
#[wasm_bindgen]
pub fn create_tetragonal_lattice(a: f64, c: f64) -> Result<WasmLattice3D, JsValue> {
    Ok(WasmLattice3D {
        inner: tetragonal_lattice(a, c),
    })
}

/// Create orthorhombic lattice
#[wasm_bindgen]
pub fn create_orthorhombic_lattice(a: f64, b: f64, c: f64) -> Result<WasmLattice3D, JsValue> {
    Ok(WasmLattice3D {
        inner: orthorhombic_lattice(a, b, c),
    })
}

/// Create rhombohedral lattice
#[wasm_bindgen]
pub fn create_rhombohedral_lattice(a: f64, alpha_degrees: f64) -> Result<WasmLattice3D, JsValue> {
    let alpha = alpha_degrees * PI / 180.0;
    Ok(WasmLattice3D {
        inner: rhombohedral_lattice(a, alpha),
    })
}

// ======================== LATTICE MANIPULATION FUNCTIONS ========================

/// Scale 2D lattice uniformly
#[wasm_bindgen]
pub fn scale_2d_lattice(lattice: &WasmLattice2D, scale: f64) -> WasmLattice2D {
    WasmLattice2D {
        inner: scale_lattice_2d(&lattice.inner, scale),
    }
}

/// Scale 3D lattice uniformly
#[wasm_bindgen]
pub fn scale_3d_lattice(lattice: &WasmLattice3D, scale: f64) -> WasmLattice3D {
    WasmLattice3D {
        inner: scale_lattice_3d(&lattice.inner, scale),
    }
}

/// Rotate 2D lattice by angle in degrees
#[wasm_bindgen]
pub fn rotate_2d_lattice(lattice: &WasmLattice2D, angle_degrees: f64) -> WasmLattice2D {
    let angle = angle_degrees * PI / 180.0;
    WasmLattice2D {
        inner: rotate_lattice_2d(&lattice.inner, angle),
    }
}

/// Create 2D supercell
#[wasm_bindgen]
pub fn create_2d_supercell(lattice: &WasmLattice2D, nx: i32, ny: i32) -> WasmLattice2D {
    WasmLattice2D {
        inner: create_supercell_2d(&lattice.inner, nx, ny),
    }
}

/// Create 3D supercell
#[wasm_bindgen]
pub fn create_3d_supercell(lattice: &WasmLattice3D, nx: i32, ny: i32, nz: i32) -> WasmLattice3D {
    WasmLattice3D {
        inner: create_supercell_3d(&lattice.inner, nx, ny, nz),
    }
}
