use wasm_bindgen::prelude::*;
use moire_lattice::lattice::{
    // Lattice structures
    Lattice2D, Lattice3D,
    
    // Bravais types and classification
    Bravais2D, Bravais3D, Centering,
    identify_bravais_2d, identify_bravais_3d,
    
    // Polyhedron
    Polyhedron,
    
    // Voronoi cells
    compute_wigner_seitz_cell_2d, compute_wigner_seitz_cell_3d,
    compute_brillouin_zone_2d, compute_brillouin_zone_3d,
    generate_lattice_points_2d_by_shell, generate_lattice_points_3d_by_shell,
    generate_lattice_points_2d_within_radius, generate_lattice_points_3d_within_radius,
    
    // Coordination analysis
    coordination_number_2d, coordination_number_3d,
    nearest_neighbors_2d, nearest_neighbors_3d,
    nearest_neighbor_distance_2d, nearest_neighbor_distance_3d,
    packing_fraction_2d, packing_fraction_3d,
    
    // Construction utilities
    construction::*,
};
use serde::{Deserialize, Serialize};
use nalgebra::{Vector3, Matrix3};
use std::f64::consts::PI;

// Enable console logging and panic hooks for debugging
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Point structure for JavaScript interop
#[derive(Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Lattice parameters for JavaScript
#[derive(Serialize, Deserialize)]
pub struct LatticeParams {
    pub lattice_type: String,
    pub a: f64,
    pub b: Option<f64>,
    pub angle: Option<f64>,
}

/// WASM wrapper for 2D lattice
#[wasm_bindgen]
pub struct WasmLattice2D {
    inner: Lattice2D,
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
            },
            "hexagonal" | "triangular" => hexagonal_lattice(params.a),
            "oblique" => {
                let b_val = params.b.unwrap_or(params.a);
                let angle_val = params.angle.unwrap_or(90.0) * PI / 180.0; // Convert to radians
                oblique_lattice(params.a, b_val, angle_val)
            },
            _ => return Err(JsValue::from_str(&format!("Unknown lattice type: {}", params.lattice_type))),
        };

        Ok(WasmLattice2D { inner: lattice })
    }

    /// Generate lattice points within a radius
    #[wasm_bindgen]
    pub fn generate_points(&self, radius: f64, center_x: f64, center_y: f64) -> Result<JsValue, JsValue> {
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius * 1.5); // Add some margin
        
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
        let (a, b) = self.inner.lattice_parameters();
        let angle = self.inner.lattice_angle() * 180.0 / PI; // Convert to degrees
        
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
        let (a_vec, b_vec) = self.inner.primitive_vectors();
        
        #[derive(Serialize)]
        struct Vectors {
            a: Point,
            b: Point,
        }
        
        let vectors = Vectors {
            a: Point { x: a_vec.x, y: a_vec.y },
            b: Point { x: b_vec.x, y: b_vec.y },
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

        serde_wasm_bindgen::to_value(&vectors)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize reciprocal vectors: {}", e)))
    }

    /// Generate an SVG representation of the lattice
    #[wasm_bindgen]
    pub fn to_svg(&self, width: f64, height: f64, radius: f64) -> String {
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius);
        
        let mut svg = format!(
            r#"<svg width="{}" height="{}" viewBox="{} {} {} {}" xmlns="http://www.w3.org/2000/svg">"#,
            width, height, -radius, -radius, 2.0 * radius, 2.0 * radius
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
        let (a_vec, b_vec) = self.inner.primitive_vectors();
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
}

/// Utility functions for creating common lattices

#[wasm_bindgen]
pub fn create_square_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: square_lattice(a),
    })
}

#[wasm_bindgen]
pub fn create_hexagonal_lattice(a: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: hexagonal_lattice(a),
    })
}

#[wasm_bindgen]
pub fn create_rectangular_lattice(a: f64, b: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: rectangular_lattice(a, b),
    })
}

/// Get the version of the library
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Additional data structures for WASM interop
#[derive(Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Serialize, Deserialize)]
pub struct LatticeParams3D {
    pub lattice_type: String,
    pub a: f64,
    pub b: Option<f64>,
    pub c: Option<f64>,
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub gamma: Option<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct PolyhedronData {
    pub vertices: Vec<Point3D>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub measure: f64,
}

#[derive(Serialize, Deserialize)]
pub struct CoordinationData {
    pub coordination_number: usize,
    pub nearest_neighbors: Vec<Point3D>,
    pub nearest_neighbor_distance: f64,
}

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

// ======================== POLYHEDRON WRAPPER ========================

/// WASM wrapper for Polyhedron
#[wasm_bindgen]
pub struct WasmPolyhedron {
    inner: Polyhedron,
}

#[wasm_bindgen]
impl WasmPolyhedron {
    /// Check if a 2D point is inside the polyhedron
    #[wasm_bindgen]
    pub fn contains_2d(&self, x: f64, y: f64) -> bool {
        let point = Vector3::new(x, y, 0.0);
        self.inner.contains_2d(point)
    }

    /// Check if a 3D point is inside the polyhedron
    #[wasm_bindgen]
    pub fn contains_3d(&self, x: f64, y: f64, z: f64) -> bool {
        let point = Vector3::new(x, y, z);
        self.inner.contains_3d(point)
    }

    /// Get the measure (area for 2D, volume for 3D)
    #[wasm_bindgen]
    pub fn measure(&self) -> f64 {
        self.inner.measure()
    }

    /// Get polyhedron data as JavaScript object
    #[wasm_bindgen]
    pub fn get_data(&self) -> Result<JsValue, JsValue> {
        let vertices: Vec<Point3D> = self.inner.vertices()
            .iter()
            .map(|v| Point3D { x: v.x, y: v.y, z: v.z })
            .collect();

        let data = PolyhedronData {
            vertices,
            edges: self.inner.edges().clone(),
            faces: self.inner.faces().clone(),
            measure: self.inner.measure(),
        };

        serde_wasm_bindgen::to_value(&data)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize polyhedron data: {}", e)))
    }
}

// ======================== 3D LATTICE WRAPPER ========================

/// WASM wrapper for 3D lattice
#[wasm_bindgen]
pub struct WasmLattice3D {
    inner: Lattice3D,
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

// ======================== ENHANCED 2D LATTICE WRAPPER ========================

#[wasm_bindgen]
impl WasmLattice2D {
    /// Convert fractional to cartesian coordinates
    #[wasm_bindgen]
    pub fn frac_to_cart(&self, fx: f64, fy: f64) -> Result<JsValue, JsValue> {
        let frac = Vector3::new(fx, fy, 0.0);
        let cart = self.inner.frac_to_cart(frac);
        let point = Point { x: cart.x, y: cart.y };
        serde_wasm_bindgen::to_value(&point)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize point: {}", e)))
    }

    /// Convert cartesian to fractional coordinates
    #[wasm_bindgen]
    pub fn cart_to_frac(&self, x: f64, y: f64) -> Result<JsValue, JsValue> {
        let cart = Vector3::new(x, y, 0.0);
        let frac = self.inner.cart_to_frac(cart);
        let point = Point { x: frac.x, y: frac.y };
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
        let point = Point { x: reduced.x, y: reduced.y };
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
    pub fn get_direct_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Result<JsValue, JsValue> {
        let points = self.inner.get_direct_lattice_points_in_rectangle(width, height);
        let js_points: Vec<Point> = points
            .into_iter()
            .map(|p| Point { x: p.x, y: p.y })
            .collect();

        serde_wasm_bindgen::to_value(&js_points)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize points: {}", e)))
    }

    /// Generate reciprocal-space lattice points in a rectangle
    #[wasm_bindgen]
    pub fn get_reciprocal_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Result<JsValue, JsValue> {
        let points = self.inner.get_reciprocal_lattice_points_in_rectangle(width, height);
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

// ======================== STANDALONE UTILITY FUNCTIONS ========================

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

/// Compute Wigner-Seitz cell for 2D lattice
#[wasm_bindgen]
pub fn compute_wigner_seitz_2d(basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if basis.len() != 9 {
        return Err(JsValue::from_str("Basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(basis);
    let polyhedron = compute_wigner_seitz_cell_2d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Wigner-Seitz cell for 3D lattice
#[wasm_bindgen]
pub fn compute_wigner_seitz_3d(basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if basis.len() != 9 {
        return Err(JsValue::from_str("Basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(basis);
    let polyhedron = compute_wigner_seitz_cell_3d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Brillouin zone for 2D lattice
#[wasm_bindgen]
pub fn compute_brillouin_2d(reciprocal_basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if reciprocal_basis.len() != 9 {
        return Err(JsValue::from_str("Reciprocal basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(reciprocal_basis);
    let polyhedron = compute_brillouin_zone_2d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

/// Compute Brillouin zone for 3D lattice
#[wasm_bindgen]
pub fn compute_brillouin_3d(reciprocal_basis: &[f64], tolerance: f64) -> Result<WasmPolyhedron, JsValue> {
    if reciprocal_basis.len() != 9 {
        return Err(JsValue::from_str("Reciprocal basis matrix must have 9 elements"));
    }
    
    let matrix = Matrix3::from_row_slice(reciprocal_basis);
    let polyhedron = compute_brillouin_zone_3d(&matrix, tolerance);
    Ok(WasmPolyhedron { inner: polyhedron })
}

// ======================== EXTENDED CONSTRUCTION FUNCTIONS ========================

/// Create centered rectangular lattice
#[wasm_bindgen]
pub fn create_centered_rectangular_lattice(a: f64, b: f64) -> Result<WasmLattice2D, JsValue> {
    Ok(WasmLattice2D {
        inner: centered_rectangular_lattice(a, b),
    })
}

/// Create oblique lattice
#[wasm_bindgen]
pub fn create_oblique_lattice(a: f64, b: f64, gamma_degrees: f64) -> Result<WasmLattice2D, JsValue> {
    let gamma = gamma_degrees * PI / 180.0;
    Ok(WasmLattice2D {
        inner: oblique_lattice(a, b, gamma),
    })
}

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
