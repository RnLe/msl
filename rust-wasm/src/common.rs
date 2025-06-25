use serde::{Deserialize, Serialize};

/// Point structure for JavaScript interop
#[derive(Serialize, Deserialize, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// 3D Point structure for JavaScript interop
#[derive(Serialize, Deserialize, Clone)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Lattice parameters for JavaScript
#[derive(Serialize, Deserialize)]
pub struct LatticeParams {
    pub lattice_type: String,
    pub a: f64,
    pub b: Option<f64>,
    pub angle: Option<f64>,
}

/// 3D Lattice parameters for JavaScript
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

/// Polyhedron data structure for JavaScript interop
#[derive(Serialize, Deserialize)]
pub struct PolyhedronData {
    pub vertices: Vec<Point3D>,
    pub edges: Vec<(usize, usize)>,
    pub faces: Vec<Vec<usize>>,
    pub measure: f64,
}

/// Coordination analysis data for JavaScript interop
#[derive(Serialize, Deserialize)]
pub struct CoordinationData {
    pub coordination_number: usize,
    pub nearest_neighbors: Vec<Point3D>,
    pub nearest_neighbor_distance: f64,
}
