//! 2D Moiré lattice for WASM bindings
//!
//! Wraps the core Moire2D with WASM-compatible interfaces

use moire_lattice::lattice::lattice_like_2d::LatticeLike2D;
use nalgebra::{Matrix2, Vector3};
use wasm_bindgen::prelude::*;

use crate::lattice::Lattice2D;
use crate::lattice::lattice_types::Bravais2D;

// Re-export core types for internal use
pub use moire_lattice::moire_lattice::moire2d::{
    Moire2D as CoreMoire2D, MoireTransformation as CoreMoireTransformation,
};

/// WASM-compatible wrapper for MoireTransformation
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct MoireTransformation {
    inner: CoreMoireTransformation,
}

impl MoireTransformation {
    pub fn from_core(inner: CoreMoireTransformation) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &CoreMoireTransformation {
        &self.inner
    }

    pub fn into_inner(self) -> CoreMoireTransformation {
        self.inner
    }
}

#[wasm_bindgen]
impl MoireTransformation {
    /// Create a simple twist (rotation) transformation
    #[wasm_bindgen(constructor)]
    pub fn new_twist(angle: f64) -> MoireTransformation {
        MoireTransformation {
            inner: CoreMoireTransformation::Twist { angle },
        }
    }

    /// Create a rotation and scaling transformation
    #[wasm_bindgen(js_name = newRotationScale)]
    pub fn new_rotation_scale(angle: f64, scale: f64) -> MoireTransformation {
        MoireTransformation {
            inner: CoreMoireTransformation::RotationScale { angle, scale },
        }
    }

    /// Create an anisotropic scaling transformation
    #[wasm_bindgen(js_name = newAnisotropicScale)]
    pub fn new_anisotropic_scale(scale_x: f64, scale_y: f64) -> MoireTransformation {
        MoireTransformation {
            inner: CoreMoireTransformation::AnisotropicScale { scale_x, scale_y },
        }
    }

    /// Create a shear transformation
    #[wasm_bindgen(js_name = newShear)]
    pub fn new_shear(shear_x: f64, shear_y: f64) -> MoireTransformation {
        MoireTransformation {
            inner: CoreMoireTransformation::Shear { shear_x, shear_y },
        }
    }

    /// Create a general matrix transformation
    /// Matrix should be provided as a flat array in column-major order (4 elements for 2x2)
    #[wasm_bindgen(js_name = newGeneral)]
    pub fn new_general(matrix: Vec<f64>) -> Result<MoireTransformation, JsValue> {
        if matrix.len() != 4 {
            return Err(JsValue::from_str("Matrix must have 4 elements"));
        }
        let mat = Matrix2::from_column_slice(&matrix);
        Ok(MoireTransformation {
            inner: CoreMoireTransformation::General(mat),
        })
    }

    /// Get the transformation matrix as a flat array (column-major order, 4 elements)
    #[wasm_bindgen(js_name = getMatrix2)]
    pub fn get_matrix_2(&self) -> Vec<f64> {
        let mat = self.inner.to_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the transformation matrix as a 3x3 matrix (embedding 2D in 3D)
    /// Returns a flat array in column-major order (9 elements)
    #[wasm_bindgen(js_name = getMatrix3)]
    pub fn get_matrix_3(&self) -> Vec<f64> {
        let mat = self.inner.to_matrix3();
        mat.as_slice().to_vec()
    }
}

/// WASM-compatible wrapper for Moire2D
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Moire2D {
    inner: CoreMoire2D,
}

impl Moire2D {
    pub fn from_core(inner: CoreMoire2D) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &CoreMoire2D {
        &self.inner
    }

    pub fn into_inner(self) -> CoreMoire2D {
        self.inner
    }
}

#[wasm_bindgen]
impl Moire2D {
    /// Create a Moiré lattice from a base lattice and transformation
    #[wasm_bindgen(constructor)]
    pub fn new(
        base_lattice: &Lattice2D,
        transformation: &MoireTransformation,
    ) -> Result<Moire2D, JsValue> {
        use moire_lattice::moire_lattice::from_transformation;

        from_transformation(base_lattice.inner(), transformation.inner().clone())
            .map(Moire2D::from_core)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    /// Get the effective moiré lattice direct space basis matrix
    #[wasm_bindgen(js_name = getEffectiveDirectBasis)]
    pub fn get_effective_direct_basis(&self) -> Vec<f64> {
        let mat = self.inner.direct_basis().base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the effective moiré lattice reciprocal space basis matrix
    #[wasm_bindgen(js_name = getEffectiveReciprocalBasis)]
    pub fn get_effective_reciprocal_basis(&self) -> Vec<f64> {
        let mat = self.inner.reciprocal_basis().base_matrix();
        mat.as_slice().to_vec()
    }

    /// Get the Bravais lattice type of the effective moiré lattice
    #[wasm_bindgen(js_name = getEffectiveBravaisType)]
    pub fn get_effective_bravais_type(&self) -> Bravais2D {
        self.inner.direct_bravais().into()
    }

    /// Get the Wigner-Seitz cell vertices of the effective lattice
    #[wasm_bindgen(js_name = getEffectiveWignerSeitzVertices)]
    pub fn get_effective_wigner_seitz_vertices(&self) -> Vec<f64> {
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

    /// Get the Brillouin zone vertices of the effective lattice
    #[wasm_bindgen(js_name = getEffectiveBrillouinZoneVertices)]
    pub fn get_effective_brillouin_zone_vertices(&self) -> Vec<f64> {
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

    /// Generate direct lattice points in a rectangle for the effective moiré lattice
    #[wasm_bindgen(js_name = getEffectiveDirectLatticePoints)]
    pub fn get_effective_direct_lattice_points(&self, width: f64, height: f64) -> Vec<f64> {
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

    /// Generate reciprocal lattice points in a rectangle for the effective moiré lattice
    #[wasm_bindgen(js_name = getEffectiveReciprocalLatticePoints)]
    pub fn get_effective_reciprocal_lattice_points(&self, width: f64, height: f64) -> Vec<f64> {
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

    /// Generate high symmetry k-path for the effective moiré lattice
    #[wasm_bindgen(js_name = getEffectiveHighSymmetryPath)]
    pub fn get_effective_high_symmetry_path(&self, n_points_per_segment: u16) -> Vec<f64> {
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

    /// Check if a point is in the Brillouin zone of the effective lattice
    #[wasm_bindgen(js_name = isInEffectiveBrillouinZone)]
    pub fn is_in_effective_brillouin_zone(&self, k_point: Vec<f64>) -> Result<bool, JsValue> {
        if k_point.len() != 3 {
            return Err(JsValue::from_str("Point must have 3 components"));
        }
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        Ok(self.inner.is_point_in_brillouin_zone(k))
    }

    /// Reduce a k-point to the first Brillouin zone of the effective lattice
    #[wasm_bindgen(js_name = reduceToEffectiveBrillouinZone)]
    pub fn reduce_to_effective_brillouin_zone(
        &self,
        k_point: Vec<f64>,
    ) -> Result<Vec<f64>, JsValue> {
        if k_point.len() != 3 {
            return Err(JsValue::from_str("Point must have 3 components"));
        }
        let k = Vector3::new(k_point[0], k_point[1], k_point[2]);
        let reduced = self.inner.reduce_point_to_brillouin_zone(k);
        Ok(vec![reduced[0], reduced[1], reduced[2]])
    }

    /// Get the first constituent lattice
    #[wasm_bindgen(js_name = getLattice1)]
    pub fn get_lattice_1(&self) -> Lattice2D {
        Lattice2D::from_core(self.inner.lattice_1().clone())
    }

    /// Get the second constituent lattice
    #[wasm_bindgen(js_name = getLattice2)]
    pub fn get_lattice_2(&self) -> Lattice2D {
        Lattice2D::from_core(self.inner.lattice_2().clone())
    }

    /// Get the transformation that was applied
    #[wasm_bindgen(js_name = getTransformation)]
    pub fn get_transformation(&self) -> MoireTransformation {
        MoireTransformation::from_core(self.inner.transformation().clone())
    }

    /// Check if the moiré lattice is commensurate
    #[wasm_bindgen(js_name = isCommensurate)]
    pub fn is_commensurate(&self) -> bool {
        matches!(
            self.inner.commensurability(),
            moire_lattice::moire_lattice::moire2d::Commensurability::Commensurate { .. }
        )
    }
}

/// Helper function to create a moiré lattice from a base lattice and transformation
/// This is an alternative to using the Moire2D constructor directly
#[wasm_bindgen(js_name = createMoireLattice)]
pub fn create_moire_lattice(
    base_lattice: &Lattice2D,
    transformation: &MoireTransformation,
) -> Result<Moire2D, JsValue> {
    Moire2D::new(base_lattice, transformation)
}
