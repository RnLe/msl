//! Base matrix module for WASM bindings
//!
//! Wraps the core library's BaseMatrix with WASM-compatible interfaces

use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

// Re-export the core types
pub use moire_lattice::interfaces::space::{Direct, Reciprocal};
pub use moire_lattice::lattice::base_matrix::BaseMatrix as CoreBaseMatrix;

/// WASM-compatible wrapper for 2D base matrix operations
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct BaseMatrixDirect {
    inner: CoreBaseMatrix<Direct>,
}

#[wasm_bindgen]
impl BaseMatrixDirect {
    /// Create a 2D base matrix from two base vectors
    /// Each vector should be provided as a 3-element array [x, y, z]
    #[wasm_bindgen(constructor)]
    pub fn new(base_1: Vec<f64>, base_2: Vec<f64>) -> Result<BaseMatrixDirect, JsValue> {
        if base_1.len() != 3 || base_2.len() != 3 {
            return Err(JsValue::from_str("Base vectors must have 3 components"));
        }

        let v1 = Vector3::new(base_1[0], base_1[1], base_1[2]);
        let v2 = Vector3::new(base_2[0], base_2[1], base_2[2]);

        CoreBaseMatrix::<Direct>::from_base_vectors_2d(v1, v2)
            .map(|inner| BaseMatrixDirect { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the base matrix as a flat array (column-major order)
    #[wasm_bindgen(js_name = getMatrix)]
    pub fn get_matrix(&self) -> Vec<f64> {
        let mat = self.inner.base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the determinant of the base matrix
    #[wasm_bindgen(js_name = determinant)]
    pub fn determinant(&self) -> f64 {
        self.inner.determinant()
    }
}

/// WASM-compatible wrapper for reciprocal space base matrix
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct BaseMatrixReciprocal {
    inner: CoreBaseMatrix<Reciprocal>,
}

#[wasm_bindgen]
impl BaseMatrixReciprocal {
    /// Get the base matrix as a flat array (column-major order)
    #[wasm_bindgen(js_name = getMatrix)]
    pub fn get_matrix(&self) -> Vec<f64> {
        let mat = self.inner.base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the determinant of the base matrix
    #[wasm_bindgen(js_name = determinant)]
    pub fn determinant(&self) -> f64 {
        self.inner.determinant()
    }
}

// Internal types that don't need WASM bindings but are used internally
pub type BaseMatrix<S> = CoreBaseMatrix<S>;
