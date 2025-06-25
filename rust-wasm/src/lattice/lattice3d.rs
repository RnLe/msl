use wasm_bindgen::prelude::*;
use moire_lattice::lattice::{
    Lattice3D,
    lattice_construction::*,
    coordination_number_3d, nearest_neighbors_3d, nearest_neighbor_distance_3d, packing_fraction_3d,
    generate_lattice_points_3d_by_shell, generate_lattice_points_3d_within_radius,
};
use nalgebra::Vector3;
use std::f64::consts::PI;
use serde::Serialize;
use crate::common::{LatticeParams3D, CoordinationData, Point3D};
use super::bravais_types::WasmBravais3D;
use super::polyhedron::WasmPolyhedron;
use super::lattice2d::WasmLattice2D;

/// WASM wrapper for 3D lattice
#[wasm_bindgen]
pub struct WasmLattice3D {
    pub(crate) inner: Lattice3D,
}

#[wasm_bindgen]
impl WasmLattice3D {
    /// Create a new 3D lattice from JavaScript parameters
    #[wasm_bindgen(constructor)]
    pub fn new(params: &JsValue) -> Result<WasmLattice3D, JsValue> {
        let params: LatticeParams3D = serde_wasm_bindgen::from_value(params.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse parameters: {}", e)))?;

        let lattice = match params.lattice_type.to_lowercase().as_str() {
            "cubic" | "simple_cubic" => simple_cubic_lattice(params.a),
            "bcc" | "body_centered_cubic" => body_centered_cubic_lattice(params.a),
            "fcc" | "face_centered_cubic" => face_centered_cubic_lattice(params.a),
            "hcp" | "hexagonal_close_packed" => {
                let c_val = params.c.unwrap_or(params.a * 1.633); // ideal c/a ratio
                hexagonal_close_packed_lattice(params.a, c_val)
            },
            "tetragonal" => {
                let c_val = params.c.unwrap_or(params.a);
                tetragonal_lattice(params.a, c_val)
            },
            "orthorhombic" => {
                let b_val = params.b.unwrap_or(params.a);
                let c_val = params.c.unwrap_or(params.a);
                orthorhombic_lattice(params.a, b_val, c_val)
            },
            "rhombohedral" => {
                let alpha_val = params.alpha.unwrap_or(90.0) * PI / 180.0;
                rhombohedral_lattice(params.a, alpha_val)
            },
            _ => return Err(JsValue::from_str(&format!("Unknown 3D lattice type: {}", params.lattice_type))),
        };

        Ok(WasmLattice3D { inner: lattice })
    }

    /// Convert fractional to cartesian coordinates
    #[wasm_bindgen]
    pub fn frac_to_cart(&self, fx: f64, fy: f64, fz: f64) -> Result<JsValue, JsValue> {
        let frac = Vector3::new(fx, fy, fz);
        let cart = self.inner.frac_to_cart(frac);
        let point = Point3D { x: cart.x, y: cart.y, z: cart.z };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Convert cartesian to fractional coordinates
    #[wasm_bindgen]
    pub fn cart_to_frac(&self, x: f64, y: f64, z: f64) -> Result<JsValue, JsValue> {
        let cart = Vector3::new(x, y, z);
        let frac = self.inner.cart_to_frac(cart);
        let point = Point3D { x: frac.x, y: frac.y, z: frac.z };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Get lattice parameters
    #[wasm_bindgen]
    pub fn lattice_parameters(&self) -> Result<JsValue, JsValue> {
        let (a, b, c) = self.inner.lattice_parameters();
        let params = vec![a, b, c];
        serde_wasm_bindgen::to_value(&params)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize parameters: {}", e)))
    }

    /// Get lattice angles in degrees
    #[wasm_bindgen]
    pub fn lattice_angles(&self) -> Result<JsValue, JsValue> {
        let (alpha, beta, gamma) = self.inner.lattice_angles();
        let angles = vec![alpha * 180.0 / PI, beta * 180.0 / PI, gamma * 180.0 / PI];
        serde_wasm_bindgen::to_value(&angles)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize angles: {}", e)))
    }

    /// Get cell volume
    #[wasm_bindgen]
    pub fn cell_volume(&self) -> f64 {
        self.inner.cell_volume()
    }

    /// Get Bravais lattice type
    #[wasm_bindgen]
    pub fn bravais_type(&self) -> WasmBravais3D {
        self.inner.bravais_type().into()
    }

    /// Check if k-point is in Brillouin zone
    #[wasm_bindgen]
    pub fn in_brillouin_zone(&self, kx: f64, ky: f64, kz: f64) -> bool {
        let k_point = Vector3::new(kx, ky, kz);
        self.inner.in_brillouin_zone(k_point)
    }

    /// Reduce k-point to first Brillouin zone
    #[wasm_bindgen]
    pub fn reduce_to_brillouin_zone(&self, kx: f64, ky: f64, kz: f64) -> Result<JsValue, JsValue> {
        let k_point = Vector3::new(kx, ky, kz);
        let reduced = self.inner.reduce_to_brillouin_zone(k_point);
        let point = Point3D { x: reduced.x, y: reduced.y, z: reduced.z };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Generate 3D lattice points within radius
    #[wasm_bindgen]
    pub fn generate_points_3d(&self, radius: f64) -> Result<JsValue, JsValue> {
        let points = generate_lattice_points_3d_within_radius(self.inner.direct_basis(), radius);
        let js_points: Vec<Point3D> = points
            .into_iter()
            .map(|p| Point3D { x: p.x, y: p.y, z: p.z })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Generate 3D lattice points by shell
    #[wasm_bindgen]
    pub fn generate_points_3d_by_shell(&self, max_shell: usize) -> Result<JsValue, JsValue> {
        let points = generate_lattice_points_3d_by_shell(self.inner.direct_basis(), max_shell);
        let js_points: Vec<Point3D> = points
            .into_iter()
            .map(|p| Point3D { x: p.x, y: p.y, z: p.z })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Get Wigner-Seitz cell
    #[wasm_bindgen]
    pub fn wigner_seitz_cell(&self) -> WasmPolyhedron {
        WasmPolyhedron {
            inner: self.inner.wigner_seitz_cell().clone(),
        }
    }

    /// Get Brillouin zone
    #[wasm_bindgen]
    pub fn brillouin_zone(&self) -> WasmPolyhedron {
        WasmPolyhedron {
            inner: self.inner.brillouin_zone().clone(),
        }
    }

    /// Get coordination analysis
    #[wasm_bindgen]
    pub fn coordination_analysis(&self) -> Result<JsValue, JsValue> {
        let bravais_type = self.inner.bravais_type();
        let coord_num = coordination_number_3d(&bravais_type);
        let neighbors = nearest_neighbors_3d(self.inner.direct_basis(), &bravais_type, 1e-10);
        let distance = nearest_neighbor_distance_3d(self.inner.direct_basis(), &bravais_type);

        let neighbors_js: Vec<Point3D> = neighbors
            .into_iter()
            .map(|p| Point3D { x: p.x, y: p.y, z: p.z })
            .collect();

        let data = CoordinationData {
            coordination_number: coord_num,
            nearest_neighbors: neighbors_js,
            nearest_neighbor_distance: distance,
        };

        serde_wasm_bindgen::to_value(&data)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize coordination data: {}", e)))
    }

    /// Get packing fraction for given atomic radius
    #[wasm_bindgen]
    pub fn packing_fraction(&self, _radius: f64) -> f64 {
        let bravais_type = self.inner.bravais_type();
        let lattice_params = self.inner.lattice_parameters();
        packing_fraction_3d(&bravais_type, lattice_params)
    }

    /// Convert to 2D lattice (projection onto a-b plane)
    #[wasm_bindgen]
    pub fn to_2d(&self) -> WasmLattice2D {
        WasmLattice2D {
            inner: self.inner.to_2d(),
        }
    }

    /// Get high symmetry points in Cartesian coordinates
    #[wasm_bindgen]
    pub fn get_high_symmetry_points(&self) -> Result<JsValue, JsValue> {
        let points = self.inner.get_high_symmetry_points_cartesian();
        
        #[derive(Serialize)]
        struct HighSymmetryPoint {
            label: String,
            x: f64,
            y: f64,
        }
        
        let js_points: Vec<HighSymmetryPoint> = points
            .into_iter()
            .map(|(label, pos)| HighSymmetryPoint {
                label,
                x: pos.x,
                y: pos.y,
            })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize high symmetry points: {}", e)))
    }

    /// Get high symmetry path data
    #[wasm_bindgen]
    pub fn get_high_symmetry_path(&self) -> Result<JsValue, JsValue> {
        let data = self.inner.high_symmetry_data();
        
        #[derive(Serialize)]
        struct PathData {
            points: Vec<String>,
        }
        
        let path_data = PathData {
            points: data.standard_path.points.iter()
                .map(|label| label.as_str().to_string())
                .collect(),
        };

        serde_wasm_bindgen::to_value(&path_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize path data: {}", e)))
    }
}
