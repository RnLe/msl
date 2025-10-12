use moire_lattice::lattice::polyhedron::Polyhedron;
use nalgebra::Vector3;
use pyo3::prelude::*;

/// Python wrapper for geometric polyhedron
#[pyclass]
pub struct PyPolyhedron {
    pub(crate) inner: Polyhedron,
}

#[pymethods]
impl PyPolyhedron {
    /// Check if a 2D point is inside the polyhedron
    fn contains_2d(&self, x: f64, y: f64, z: Option<f64>) -> bool {
        let z_val = z.unwrap_or(0.0);
        let point = Vector3::new(x, y, z_val);
        self.inner.contains_2d(point)
    }

    /// Check if a 3D point is inside the polyhedron
    fn contains_3d(&self, x: f64, y: f64, z: f64) -> bool {
        let point = Vector3::new(x, y, z);
        self.inner.contains_3d(point)
    }

    /// Get the measure (area for 2D, volume for 3D)
    fn measure(&self) -> f64 {
        self.inner.measure()
    }

    /// Get the vertices of the polyhedron
    fn vertices(&self) -> Vec<(f64, f64, f64)> {
        self.inner
            .vertices()
            .iter()
            .map(|v| (v.x, v.y, v.z))
            .collect()
    }

    /// Get the edges as pairs of vertex indices
    fn edges(&self) -> Vec<(usize, usize)> {
        self.inner.edges().clone()
    }

    /// Get the faces as lists of vertex indices
    fn faces(&self) -> Vec<Vec<usize>> {
        self.inner.faces().clone()
    }

    /// Get the number of vertices
    fn num_vertices(&self) -> usize {
        self.inner.vertices().len()
    }

    /// Get the number of edges
    fn num_edges(&self) -> usize {
        self.inner.edges().len()
    }

    /// Get the number of faces
    fn num_faces(&self) -> usize {
        self.inner.faces().len()
    }

    /// Get geometric properties as a dictionary
    fn geometric_properties(&self) -> PyResult<PyObject> {
        use pyo3::types::PyDict;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            dict.set_item("num_vertices", self.num_vertices())?;
            dict.set_item("num_edges", self.num_edges())?;
            dict.set_item("num_faces", self.num_faces())?;
            dict.set_item("measure", self.measure())?;

            // Calculate Euler characteristic (V - E + F should be 2 for convex polyhedra)
            let euler_char =
                self.num_vertices() as i32 - self.num_edges() as i32 + self.num_faces() as i32;
            dict.set_item("euler_characteristic", euler_char)?;

            Ok(dict.into())
        })
    }

    /// Check multiple points at once for efficiency
    fn contains_points_2d(&self, points: Vec<(f64, f64)>) -> Vec<bool> {
        points
            .into_iter()
            .map(|(x, y)| {
                let point = Vector3::new(x, y, 0.0);
                self.inner.contains_2d(point)
            })
            .collect()
    }

    /// Check multiple 3D points at once
    fn contains_points_3d(&self, points: Vec<(f64, f64, f64)>) -> Vec<bool> {
        points
            .into_iter()
            .map(|(x, y, z)| {
                let point = Vector3::new(x, y, z);
                self.inner.contains_3d(point)
            })
            .collect()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "PyPolyhedron(vertices={}, edges={}, faces={}, measure={:.6})",
            self.num_vertices(),
            self.num_edges(),
            self.num_faces(),
            self.measure()
        )
    }
}

impl PyPolyhedron {
    pub fn new(inner: Polyhedron) -> Self {
        PyPolyhedron { inner }
    }
}
