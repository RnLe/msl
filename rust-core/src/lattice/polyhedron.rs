use crate::{config::LATTICE_TOLERANCE, interfaces::Dimension};
use nalgebra::Vector3;

/// Outward half-space of a Brillouin-zone edge in 2D (z=0 plane)
#[derive(Clone, Debug)]
pub(crate) struct HalfSpace2D {
    /// Outward unit normal of a BZ edge (z=0).
    pub normal: Vector3<f64>,
    /// Support value: distance from Γ along `normal` to the edge.
    pub h: f64,
    /// Associated shortest reciprocal vector for this edge: G = 2*h*normal.
    pub g: Vector3<f64>,
}

/// Signed polygon area; > 0 means CCW vertex ordering
pub(crate) fn polygon_signed_area_2d(verts: &[Vector3<f64>]) -> f64 {
    let n = verts.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = verts[i];
        let q = verts[(i + 1) % n];
        a += p.x * q.y - q.x * p.y;
    }
    0.5 * a
}

/// Build BZ half-spaces from the stored polygon (2D, z=0) in `Polyhedron`.
pub(crate) fn bz_halfspaces_from_poly(bz: &Polyhedron, tol: f64) -> Vec<HalfSpace2D> {
    let verts = &bz.vertices;
    assert!(verts.len() >= 3, "BZ requires at least a triangle");

    // Determine winding to choose outward normals correctly.
    let ccw = polygon_signed_area_2d(verts) > 0.0;

    let n = verts.len();
    let mut hs = Vec::with_capacity(n);

    for i in 0..n {
        let v0 = verts[i];
        let v1 = verts[(i + 1) % n];
        let e = v1 - v0; // edge direction in-plane

        // Right normal is outward if polygon is CCW; left if CW.
        let (nx, ny) = if ccw { (e.y, -e.x) } else { (-e.y, e.x) };
        let mut nvec = Vector3::new(nx, ny, 0.0);
        let len = (nvec.x * nvec.x + nvec.y * nvec.y).sqrt();
        assert!(len > tol, "Degenerate edge in BZ polygon");
        nvec /= len;

        // Support value: distance to the edge line from Γ along the outward normal.
        // Any point on the edge can be used; use midpoint to reduce numerical noise.
        let mid = 0.5 * (v0 + v1);
        let h = nvec.dot(&mid); // should be > 0 for a valid BZ edge

        // Associated shortest reciprocal vector for this face:
        let g = 2.0 * h * nvec;

        hs.push(HalfSpace2D { normal: nvec, h, g });
    }

    hs
}

/// Represents a polyhedron (2D polygon or 3D polyhedron) for Wigner-Seitz/Brillouin zones
#[derive(Debug, Clone)]
pub struct Polyhedron {
    /// Vertices of the polyhedron (in direct/reciprocal coordinates)
    pub vertices: Vec<Vector3<f64>>,
    /// Edges as pairs of vertex indices
    pub edges: Vec<(usize, usize)>,
    /// Faces as lists of vertex indices (empty for 2D)
    pub faces: Vec<Vec<usize>>,
    /// Volume (3D) or area (2D) of the polyhedron
    pub measure: f64,
    /// Dimension to classify what space this polyhedron lives in
    pub dimension: Dimension,
}

impl Polyhedron {
    /// Create a new empty polygon
    pub fn new_polygon() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            measure: 0.0,
            dimension: Dimension::_2D,
        }
    }

    /// Create a new empty polyhedron
    pub fn new_polyhedron() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            measure: 0.0,
            dimension: Dimension::_3D,
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
            if normal.dot(&(point - v0)) > LATTICE_TOLERANCE {
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
