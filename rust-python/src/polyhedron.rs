//! Python bindings for Polyhedron type (Wigner-Seitz cells and Brillouin zones)

use pyo3::prelude::*;
use moire_lattice::lattice::polyhedron::Polyhedron;
use nalgebra::Vector3;

/// Python wrapper for Polyhedron
#[pyclass(name = "Polyhedron")]
#[derive(Clone)]
pub struct PyPolyhedron {
    pub(crate) inner: Polyhedron,
}

#[pymethods]
impl PyPolyhedron {
    /// Get the vertices of the polyhedron
    ///
    /// Returns:
    ///     List[List[float]]: List of vertices, each as [x, y, z]
    #[getter]
    fn vertices(&self) -> Vec<[f64; 3]> {
        self.inner.vertices.iter()
            .map(|v| [v[0], v[1], v[2]])
            .collect()
    }
    
    /// Get the edges of the polyhedron as pairs of vertex indices
    ///
    /// Returns:
    ///     List[Tuple[int, int]]: List of edges as (vertex_index_1, vertex_index_2)
    #[getter]
    fn edges(&self) -> Vec<(usize, usize)> {
        self.inner.edges.clone()
    }
    
    /// Get the faces of the polyhedron as lists of vertex indices
    ///
    /// Returns:
    ///     List[List[int]]: List of faces, each face is a list of vertex indices
    #[getter]
    fn faces(&self) -> Vec<Vec<usize>> {
        self.inner.faces.clone()
    }
    
    /// Get the measure (area for 2D, volume for 3D) of the polyhedron
    ///
    /// Returns:
    ///     float: The measure
    #[getter]
    fn measure(&self) -> f64 {
        self.inner.measure
    }
    
    /// Check if a point is inside the polyhedron (2D version)
    ///
    /// Args:
    ///     point: Point coordinates as [x, y, z]
    ///
    /// Returns:
    ///     bool: True if the point is inside
    fn contains_2d(&self, point: [f64; 3]) -> bool {
        let p = Vector3::new(point[0], point[1], point[2]);
        self.inner.contains_2d(p)
    }
    
    /// Get the number of vertices
    ///
    /// Returns:
    ///     int: Number of vertices
    fn num_vertices(&self) -> usize {
        self.inner.vertices.len()
    }
    
    /// Get the number of edges
    ///
    /// Returns:
    ///     int: Number of edges
    fn num_edges(&self) -> usize {
        self.inner.edges.len()
    }
    
    /// Get the number of faces
    ///
    /// Returns:
    ///     int: Number of faces
    fn num_faces(&self) -> usize {
        self.inner.faces.len()
    }
    
    fn __repr__(&self) -> String {
        format!("Polyhedron(vertices={}, edges={}, faces={}, measure={})", 
                self.num_vertices(), self.num_edges(), self.num_faces(), self.measure())
    }
}
