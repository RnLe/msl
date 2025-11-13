//! 2D Lattice structure for WASM bindings
//!
//! Wraps the core Lattice2D with WASM-compatible interfaces

use moire_lattice::lattice::lattice_like_2d::LatticeLike2D;
use nalgebra::{Matrix3, Vector3};
use wasm_bindgen::prelude::*;

use crate::lattice::lattice_types::Bravais2D;

// Re-export core type for internal use
pub use moire_lattice::lattice::Lattice2D as CoreLattice2D;

/// WASM-compatible wrapper for 2D lattice
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Lattice2D {
    inner: CoreLattice2D,
}

impl Lattice2D {
    pub fn from_core(inner: CoreLattice2D) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &CoreLattice2D {
        &self.inner
    }

    pub fn into_inner(self) -> CoreLattice2D {
        self.inner
    }
}

#[wasm_bindgen]
impl Lattice2D {
    /// Create a 2D lattice from direct space basis vectors
    /// The matrix should be provided as a flat array in column-major order (9 elements)
    #[wasm_bindgen(constructor)]
    pub fn new(direct_matrix: Vec<f64>) -> Result<Lattice2D, JsValue> {
        if direct_matrix.len() != 9 {
            return Err(JsValue::from_str("Matrix must have 9 elements"));
        }

        let mat = Matrix3::from_column_slice(&direct_matrix);

        CoreLattice2D::from_direct_matrix(mat)
            .map(|inner| Lattice2D { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the direct space basis matrix as a flat array (column-major order)
    #[wasm_bindgen(js_name = getDirectBasis)]
    pub fn get_direct_basis(&self) -> Vec<f64> {
        let mat = self.inner.direct_basis().base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the reciprocal space basis matrix as a flat array (column-major order)
    #[wasm_bindgen(js_name = getReciprocalBasis)]
    pub fn get_reciprocal_basis(&self) -> Vec<f64> {
        let mat = self.inner.reciprocal_basis().base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the Bravais lattice type
    #[wasm_bindgen(js_name = getBravaisType)]
    pub fn get_bravais_type(&self) -> Bravais2D {
        self.inner.direct_bravais().into()
    }

    /// Get the Wigner-Seitz cell vertices as a flat array
    /// Each vertex is [x, y, z], returned as a flat array
    #[wasm_bindgen(js_name = getWignerSeitzVertices)]
    pub fn get_wigner_seitz_vertices(&self) -> Vec<f64> {
        let ws = self.inner.wigner_seitz();
        let vertices = ws.vertices();
        let mut flat = Vec::with_capacity(vertices.len() * 3);
        for v in vertices {
            flat.push(v[0]);
            flat.push(v[1]);
            flat.push(v[2]);
        }
        flat
    }

    /// Get the Brillouin zone vertices as a flat array
    /// Each vertex is [x, y, z], returned as a flat array
    #[wasm_bindgen(js_name = getBrillouinZoneVertices)]
    pub fn get_brillouin_zone_vertices(&self) -> Vec<f64> {
        let bz = self.inner.brillouin_zone();
        let vertices = bz.vertices();
        let mut flat = Vec::with_capacity(vertices.len() * 3);
        for v in vertices {
            flat.push(v[0]);
            flat.push(v[1]);
            flat.push(v[2]);
        }
        flat
    }

    /// Generate direct lattice points in a rectangle
    #[wasm_bindgen(js_name = getDirectLatticePoints)]
    pub fn get_direct_lattice_points(&self, width: f64, height: f64) -> Vec<f64> {
        let points = self
            .inner
            .compute_direct_lattice_points_in_rectangle(width, height);
        let mut flat = Vec::with_capacity(points.len() * 3);
        for p in points {
            flat.push(p[0]);
            flat.push(p[1]);
            flat.push(p[2]);
        }
        flat
    }

    /// Generate reciprocal lattice points in a rectangle
    #[wasm_bindgen(js_name = getReciprocalLatticePoints)]
    pub fn get_reciprocal_lattice_points(&self, width: f64, height: f64) -> Vec<f64> {
        let points = self
            .inner
            .compute_reciprocal_lattice_points_in_rectangle(width, height);
        let mut flat = Vec::with_capacity(points.len() * 3);
        for p in points {
            flat.push(p[0]);
            flat.push(p[1]);
            flat.push(p[2]);
        }
        flat
    }

    /// Generate high symmetry k-path for band structure calculations
    #[wasm_bindgen(js_name = getHighSymmetryPath)]
    pub fn get_high_symmetry_path(&self, n_points_per_segment: u16) -> Vec<f64> {
        let points = self
            .inner
            .generate_high_symmetry_k_path(n_points_per_segment);
        let mut flat = Vec::with_capacity(points.len() * 3);
        for p in points {
            flat.push(p[0]);
            flat.push(p[1]);
            flat.push(p[2]);
        }
        flat
    }

    /// Check if a point is in the Brillouin zone
    #[wasm_bindgen(js_name = isInBrillouinZone)]
    pub fn is_in_brillouin_zone(&self, k_point: Vec<f64>) -> Result<bool, JsValue> {
        if k_point.len() != 3 {
            return Err(JsValue::from_str("Point must have 3 components"));
        }
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        Ok(self.inner.is_point_in_brillouin_zone(k))
    }

    /// Reduce a k-point to the first Brillouin zone
    #[wasm_bindgen(js_name = reduceToBrillouinZone)]
    pub fn reduce_to_brillouin_zone(&self, k_point: Vec<f64>) -> Result<Vec<f64>, JsValue> {
        if k_point.len() != 3 {
            return Err(JsValue::from_str("Point must have 3 components"));
        }
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        let reduced = self.inner.reduce_point_to_brillouin_zone(k);
        Ok(vec![reduced[0], reduced[1], reduced[2]])
    }
}

/// Helper functions for creating standard lattices
#[wasm_bindgen]
pub fn create_square_lattice(a: f64) -> Result<Lattice2D, JsValue> {
    moire_lattice::lattice::square_lattice(a)
        .map(Lattice2D::from_core)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn create_rectangular_lattice(a: f64, b: f64) -> Result<Lattice2D, JsValue> {
    moire_lattice::lattice::rectangular_lattice(a, b)
        .map(Lattice2D::from_core)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn create_hexagonal_lattice(a: f64) -> Result<Lattice2D, JsValue> {
    moire_lattice::lattice::hexagonal_lattice(a)
        .map(Lattice2D::from_core)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn create_oblique_lattice(a: f64, b: f64, gamma: f64) -> Result<Lattice2D, JsValue> {
    moire_lattice::lattice::oblique_lattice(a, b, gamma)
        .map(Lattice2D::from_core)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
