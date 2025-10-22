//! Polyhedron representation for WASM bindings
//!
//! Wraps polyhedron structures for Voronoi cells and Brillouin zones

use wasm_bindgen::prelude::*;

// Re-export core type
pub use moire_lattice::lattice::polyhedron::Polyhedron as CorePolyhedron;

/// WASM-compatible wrapper for Polyhedron
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Polyhedron {
    inner: CorePolyhedron,
}

impl Polyhedron {
    pub fn from_core(inner: CorePolyhedron) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &CorePolyhedron {
        &self.inner
    }
}

#[wasm_bindgen]
impl Polyhedron {
    /// Get the vertices of the polyhedron as a flat array
    /// Each vertex is represented as [x, y, z]
    #[wasm_bindgen(js_name = getVertices)]
    pub fn get_vertices(&self) -> Vec<f64> {
        let vertices = self.inner.vertices();
        let mut flat = Vec::with_capacity(vertices.len() * 3);
        for v in vertices {
            flat.push(v[0]);
            flat.push(v[1]);
            flat.push(v[2]);
        }
        flat
    }

    /// Get the number of vertices
    #[wasm_bindgen(js_name = vertexCount)]
    pub fn vertex_count(&self) -> usize {
        self.inner.vertices().len()
    }
}
