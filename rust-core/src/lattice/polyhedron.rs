use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// ε that controls the numerical tolerance (works for unit–cell sized data;
/// scale if the polyhedron spans many orders of magnitude).
const EPS: f64 = 1.0e-10;

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

            if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
        }
        inside
    }

    /// Return `true` if `point` lies inside – or on the boundary of – the
    /// convex polyhedron.  Works for any face ordering/orientation.
    pub fn contains_3d(&self, point: Vector3<f64>) -> bool {
        // Note: If at any point this becomes a bottleneck, consider using parry3d instead.
        // parry3d is a highly-tuned geometry kernel shipping highly vectorised collision queries (AVX2, SSE4.2).

        // A polyhedron must have at least one face with ≥3 vertices
        if self.faces.is_empty() {
            return false;
        }

        // Cheap approximate interior point: the arithmetic mean of all vertices.
        let centroid = self
            .vertices
            .iter()
            .fold(Vector3::zeros(), |acc, v| acc + v)
            / self.vertices.len() as f64;

        // Test the point against every face plane.
        for face in &self.faces {
            if face.len() < 3 {
                continue;
            } // degenerate face – skip

            // --- build the plane -------------------------------------------
            let v0 = self.vertices[face[0]];
            let v1 = self.vertices[face[1]];
            let v2 = self.vertices[face[2]];

            let mut normal = (v1 - v0).cross(&(v2 - v0)); // un-normalised

            // Ensure the normal points *outward*.
            if normal.dot(&(centroid - v0)) > 0.0 {
                normal = -normal;
            }

            // Signed distance of the query point to the plane.
            if normal.dot(&(point - v0)) > EPS {
                // One plane says “outside”  ⇒  entire test fails.
                return false;
            }
            // Otherwise continue with next face.
        }
        true
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
