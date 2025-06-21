use nalgebra::Vector3;
use serde::{Serialize, Deserialize};

/// Represents a polyhedron (2D polygon or 3D polyhedron) for Wigner-Seitz/Brillouin zones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polyhedron {
    /// Vertices of the polyhedron (in direct/reciprocal coordinates)
    pub vertices: Vec<Vector3<f64>>,
    /// Edges as pairs of vertex indices
    pub edges: Vec<(usize, usize)>,
    /// Faces as lists of vertex indices (empty for 2D)
    pub faces: Vec<Vec<usize>>,
    /// Volume (3D) or area (2D) of the polyhedron
    pub measure: f64,
}

impl Polyhedron {
    /// Create a new empty polyhedron
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            measure: 0.0,
        }
    }
    
    /// Check if a point is inside the polyhedron (2D version)
    pub fn contains_2d(&self, point: Vector3<f64>) -> bool {
        // Simple point-in-polygon test using ray casting
        let mut inside = false;
        let n = self.vertices.len();
        let (px, py) = (point[0], point[1]);
        
        for i in 0..n {
            let j = (i + 1) % n;
            let (xi, yi) = (self.vertices[i][0], self.vertices[i][1]);
            let (xj, yj) = (self.vertices[j][0], self.vertices[j][1]);
            
            if ((yi > py) != (yj > py)) && 
               (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        inside
    }
    
    /// Check if a point is inside the polyhedron (3D version)
    /// TODO: Implement proper 3D point-in-polyhedron test
    pub fn contains_3d(&self, _point: Vector3<f64>) -> bool {
        // Placeholder implementation
        // Full implementation would use proper 3D point-in-polyhedron algorithms
        false
    }
    
    /// Get the volume (3D) or area (2D) of the polyhedron
    pub fn measure(&self) -> f64 {
        self.measure
    }
    
    /// Get vertices as a reference
    pub fn vertices(&self) -> &Vec<Vector3<f64>> {
        &self.vertices
    }
    
    /// Get edges as a reference
    pub fn edges(&self) -> &Vec<(usize, usize)> {
        &self.edges
    }
    
    /// Get faces as a reference (empty for 2D)
    pub fn faces(&self) -> &Vec<Vec<usize>> {
        &self.faces
    }
}

impl Default for Polyhedron {
    fn default() -> Self {
        Self::new()
    }
}
