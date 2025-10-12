use super::bravais_types::WasmBravais2D;
use super::lattice3d::WasmLattice3D;
use super::polyhedron::WasmPolyhedron;
use crate::common::{CoordinationData, LatticeParams, Point, Point3D};
use moire_lattice::lattice::{
    Lattice2D, coordination_number_2d, generate_lattice_points_2d_by_shell,
    generate_lattice_points_2d_within_radius, lattice_construction::*,
    nearest_neighbor_distance_2d, nearest_neighbors_2d, packing_fraction_2d,
};
use nalgebra::Vector3;
use serde::Serialize;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

/// WASM wrapper for 2D lattice
#[wasm_bindgen]
pub struct WasmLattice2D {
    pub(crate) inner: Lattice2D,
}

#[wasm_bindgen]
impl WasmLattice2D {
    /// Create a new lattice from JavaScript parameters
    #[wasm_bindgen(constructor)]
    pub fn new(params: &JsValue) -> Result<WasmLattice2D, JsValue> {
        let params: LatticeParams = serde_wasm_bindgen::from_value(params.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to parse parameters: {}", e)))?;

        let lattice = match params.lattice_type.to_lowercase().as_str() {
            "square" => square_lattice(params.a),
            "rectangular" => {
                let b_val = params.b.unwrap_or(params.a);
                rectangular_lattice(params.a, b_val)
            }
            "hexagonal" | "triangular" => hexagonal_lattice(params.a),
            "oblique" => {
                let b_val = params.b.unwrap_or(params.a);
                let angle_val = params.angle.unwrap_or(90.0) * PI / 180.0; // Convert to radians
                oblique_lattice(params.a, b_val, angle_val)
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown lattice type: {}",
                    params.lattice_type
                )));
            }
        };

        Ok(WasmLattice2D { inner: lattice })
    }

    /// Generate lattice points within a radius
    #[wasm_bindgen]
    pub fn generate_points(
        &self,
        radius: f64,
        center_x: f64,
        center_y: f64,
    ) -> Result<JsValue, JsValue> {
        let points =
            generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius * 1.5); // Add some margin

        let center_vec = Vector3::new(center_x, center_y, 0.0);
        let filtered_points: Vec<Point> = points
            .into_iter()
            .filter(|p| {
                let dist = (p - center_vec).norm();
                dist <= radius
            })
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&filtered_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Get lattice parameters as JavaScript object
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> Result<JsValue, JsValue> {
        let (a, b) = self.inner.direct_lattice_parameters();
        let angle = self.inner.direct_lattice_angle() * 180.0 / PI; // Convert to degrees

        let params = LatticeParams {
            lattice_type: format!("{:?}", self.inner.bravais),
            a,
            b: Some(b),
            angle: Some(angle),
        };

        serde_wasm_bindgen::to_value(&params)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize parameters: {}", e)))
    }

    /// Get unit cell area
    #[wasm_bindgen]
    pub fn unit_cell_area(&self) -> f64 {
        self.inner.cell_area
    }

    /// Get lattice vectors as JavaScript object
    #[wasm_bindgen]
    pub fn lattice_vectors(&self) -> Result<JsValue, JsValue> {
        let (a_vec, b_vec) = self.inner.direct_base_vectors();

        #[derive(Serialize)]
        struct Vectors {
            a: Point,
            b: Point,
        }

        let vectors = Vectors {
            a: Point {
                x: a_vec.x,
                y: a_vec.y,
            },
            b: Point {
                x: b_vec.x,
                y: b_vec.y,
            },
        };

        serde_wasm_bindgen::to_value(&vectors)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize vectors: {}", e)))
    }

    /// Get reciprocal lattice vectors
    #[wasm_bindgen]
    pub fn reciprocal_vectors(&self) -> Result<JsValue, JsValue> {
        let b1 = self.inner.reciprocal_basis().column(0);
        let b2 = self.inner.reciprocal_basis().column(1);

        #[derive(Serialize)]
        struct Vectors {
            a: Point,
            b: Point,
        }

        let vectors = Vectors {
            a: Point { x: b1.x, y: b1.y },
            b: Point { x: b2.x, y: b2.y },
        };

        serde_wasm_bindgen::to_value(&vectors).map_err(|e| {
            JsValue::from_str(&format!("Failed to serialize reciprocal vectors: {}", e))
        })
    }

    /// Generate an SVG representation of the lattice
    #[wasm_bindgen]
    pub fn to_svg(&self, width: f64, height: f64, radius: f64) -> String {
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius);

        let mut svg = format!(
            r#"<svg width="{}" height="{}" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            width,
            height,
            -radius,
            -radius,
            2.0 * radius,
            2.0 * radius
        );

        // Add lattice points
        for point in &points {
            if point.norm() <= radius {
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="0.05" fill="blue" />"#,
                    point.x, point.y
                ));
            }
        }

        // Add lattice vectors
        let (a_vec, b_vec) = self.inner.direct_base_vectors();
        svg.push_str(&format!(
            r#"<line x1="0" y1="0" x2="{}" y2="{}" stroke="red" stroke-width="0.02" />"#,
            a_vec.x, a_vec.y
        ));
        svg.push_str(&format!(
            r#"<line x1="0" y1="0" x2="{}" y2="{}" stroke="green" stroke-width="0.02" />"#,
            b_vec.x, b_vec.y
        ));

        svg.push_str("</svg>");
        svg
    }

    /// Convert fractional to cartesian coordinates
    #[wasm_bindgen]
    pub fn frac_to_cart(&self, fx: f64, fy: f64) -> Result<JsValue, JsValue> {
        let frac = Vector3::new(fx, fy, 0.0);
        let cart = self.inner.fractional_to_cartesian(frac);
        let point = Point {
            x: cart.x,
            y: cart.y,
        };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Convert cartesian to fractional coordinates
    #[wasm_bindgen]
    pub fn cart_to_frac(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let cart = Vector3::new(x, y, 0.0);
        let frac = self.inner.cartesian_to_fractional(cart);
        let point = Point {
            x: frac.x,
            y: frac.y,
        };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Get Bravais lattice type
    #[wasm_bindgen]
    pub fn bravais_type(&self) -> WasmBravais2D {
        self.inner.bravais_type().into()
    }

    /// Check if k-point is in Brillouin zone
    #[wasm_bindgen]
    pub fn in_brillouin_zone(&self, kx: f64, ky: f64) -> bool {
        let k_point = Vector3::new(kx, ky, 0.0);
        self.inner.in_brillouin_zone(k_point)
    }

    /// Reduce k-point to first Brillouin zone
    #[wasm_bindgen]
    pub fn reduce_to_brillouin_zone(&self, kx: f64, ky: f64) -> Result<JsValue, JsValue> {
        let k_point = Vector3::new(kx, ky, 0.0);
        let reduced = self.inner.reduce_to_brillouin_zone(k_point);
        let point = Point {
            x: reduced.x,
            y: reduced.y,
        };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
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
        let coord_num = coordination_number_2d(&bravais_type);
        let neighbors = nearest_neighbors_2d(self.inner.direct_basis(), &bravais_type, 1e-10);
        let distance = nearest_neighbor_distance_2d(self.inner.direct_basis(), &bravais_type);

        let neighbors_js: Vec<Point3D> = neighbors
            .into_iter()
            .map(|p| Point3D {
                x: p.x,
                y: p.y,
                z: p.z,
            })
            .collect();

        let data = CoordinationData {
            coordination_number: coord_num,
            nearest_neighbors: neighbors_js,
            nearest_neighbor_distance: distance,
        };

        serde_wasm_bindgen::to_value(&data).map_err(|e| {
            JsValue::from_str(&format!("Failed to serialize coordination data: {}", e))
        })
    }

    /// Get packing fraction for given atomic radius
    #[wasm_bindgen]
    pub fn packing_fraction(&self, _radius: f64) -> f64 {
        let bravais_type = self.inner.bravais_type();
        let lattice_params = self.inner.direct_lattice_parameters();
        packing_fraction_2d(&bravais_type, lattice_params)
    }

    /// Extend to 3D lattice with given c-vector
    #[wasm_bindgen]
    pub fn to_3d(&self, cx: f64, cy: f64, cz: f64) -> WasmLattice3D {
        let c_vector = Vector3::new(cx, cy, cz);
        WasmLattice3D {
            inner: self.inner.to_3d(c_vector),
        }
    }

    /// Generate lattice points by shell
    #[wasm_bindgen]
    pub fn generate_points_by_shell(&self, max_shell: usize) -> Result<JsValue, JsValue> {
        let points = generate_lattice_points_2d_by_shell(self.inner.direct_basis(), max_shell);
        let js_points: Vec<Point> = points
            .into_iter()
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Generate direct-space lattice points in a rectangle
    #[wasm_bindgen]
    pub fn get_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Result<JsValue, JsValue> {
        let points = self
            .inner
            .get_direct_lattice_points_in_rectangle(width, height);
        let js_points: Vec<Point> = points
            .into_iter()
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Generate reciprocal-space lattice points in a rectangle
    #[wasm_bindgen]
    pub fn get_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Result<JsValue, JsValue> {
        let points = self
            .inner
            .get_reciprocal_lattice_points_in_rectangle(width, height);
        let js_points: Vec<Point> = points
            .into_iter()
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Get high symmetry points in Cartesian coordinates
    #[wasm_bindgen]
    pub fn get_high_symmetry_points(&self) -> Result<JsValue, JsValue> {
        let points = self.inner.reciprocal_high_symmetry_points_cartesian();

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

        serde_wasm_bindgen::to_value(&js_points).map_err(|e| {
            JsValue::from_str(&format!("Failed to serialize high symmetry points: {}", e))
        })
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
            points: data
                .standard_path
                .points
                .iter()
                .map(|label| label.as_str().to_string())
                .collect(),
        };

        serde_wasm_bindgen::to_value(&path_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize path data: {}", e)))
    }
}
