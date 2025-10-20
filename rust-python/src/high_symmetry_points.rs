//! Python bindings for high symmetry points and paths

use pyo3::prelude::*;
use moire_lattice::symmetries::high_symmetry_points::{
    SymmetryPointLabel, HighSymmetryPoint, HighSymmetryPath, HighSymmetryData
};
use nalgebra::Vector3;

/// Python wrapper for SymmetryPointLabel
#[pyclass(name = "SymmetryPointLabel")]
#[derive(Clone)]
pub struct PySymmetryPointLabel {
    pub(crate) inner: SymmetryPointLabel,
}

#[pymethods]
impl PySymmetryPointLabel {
    #[staticmethod]
    fn gamma() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::Gamma }
    }
    
    #[staticmethod]
    fn m() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::M }
    }
    
    #[staticmethod]
    fn k() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::K }
    }
    
    #[staticmethod]
    fn x() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::X }
    }
    
    #[staticmethod]
    fn y() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::Y }
    }
    
    #[staticmethod]
    fn r() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::R }
    }
    
    #[staticmethod]
    fn l() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::L }
    }
    
    #[staticmethod]
    fn w() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::W }
    }
    
    #[staticmethod]
    fn u() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::U }
    }
    
    #[staticmethod]
    fn s() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::S }
    }
    
    #[staticmethod]
    fn t() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::T }
    }
    
    #[staticmethod]
    fn z() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::Z }
    }
    
    #[staticmethod]
    fn a() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::A }
    }
    
    #[staticmethod]
    fn h() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::H }
    }
    
    #[staticmethod]
    fn p() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::P }
    }
    
    #[staticmethod]
    fn n() -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::N }
    }
    
    #[staticmethod]
    fn custom(name: String) -> Self {
        PySymmetryPointLabel { inner: SymmetryPointLabel::Custom(name) }
    }
    
    fn as_str(&self) -> &str {
        self.inner.as_str()
    }
    
    fn __repr__(&self) -> String {
        format!("SymmetryPointLabel('{}')", self.as_str())
    }
    
    fn __str__(&self) -> String {
        self.as_str().to_string()
    }
}

/// Python wrapper for HighSymmetryPoint
#[pyclass(name = "HighSymmetryPoint")]
#[derive(Clone)]
pub struct PyHighSymmetryPoint {
    pub(crate) inner: HighSymmetryPoint,
}

#[pymethods]
impl PyHighSymmetryPoint {
    /// Create a new high symmetry point
    ///
    /// Args:
    ///     label: The label for the point
    ///     position: Position as [x, y, z] in fractional coordinates
    ///     description: Description of the point
    #[new]
    fn new(label: PySymmetryPointLabel, position: [f64; 3], description: String) -> Self {
        let pos = Vector3::new(position[0], position[1], position[2]);
        let inner = HighSymmetryPoint::new(label.inner, pos, description);
        PyHighSymmetryPoint { inner }
    }
    
    /// Get the label
    #[getter]
    fn label(&self) -> PySymmetryPointLabel {
        PySymmetryPointLabel { inner: self.inner.label.clone() }
    }
    
    /// Get the position
    #[getter]
    fn position(&self) -> [f64; 3] {
        [self.inner.position[0], self.inner.position[1], self.inner.position[2]]
    }
    
    /// Get the description
    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }
    
    fn __repr__(&self) -> String {
        format!("HighSymmetryPoint({}, {:?}, '{}')", 
                self.inner.label.as_str(), self.position(), self.inner.description)
    }
}

/// Python wrapper for HighSymmetryPath
#[pyclass(name = "HighSymmetryPath")]
#[derive(Clone)]
pub struct PyHighSymmetryPath {
    pub(crate) inner: HighSymmetryPath,
}

#[pymethods]
impl PyHighSymmetryPath {
    /// Create a new path from a list of point labels
    ///
    /// Args:
    ///     points: List of SymmetryPointLabel objects
    #[new]
    fn new(points: Vec<PySymmetryPointLabel>) -> Self {
        let point_labels = points.into_iter().map(|p| p.inner).collect();
        PyHighSymmetryPath { inner: HighSymmetryPath::new(point_labels) }
    }
    
    /// Get the ordered list of point labels
    #[getter]
    fn points(&self) -> Vec<PySymmetryPointLabel> {
        self.inner.points.iter()
            .map(|label| PySymmetryPointLabel { inner: label.clone() })
            .collect()
    }
    
    /// Get the number of interpolation points
    #[getter]
    fn n_points(&self) -> Option<usize> {
        self.inner.n_points
    }
    
    /// Set the number of k-points for interpolation
    fn with_n_points(&self, n: usize) -> Self {
        PyHighSymmetryPath { inner: self.inner.clone().with_n_points(n) }
    }
    
    fn __repr__(&self) -> String {
        let labels: Vec<String> = self.inner.points.iter()
            .map(|l| l.as_str().to_string())
            .collect();
        format!("HighSymmetryPath([{}])", labels.join(" → "))
    }
}

/// Python wrapper for HighSymmetryData
#[pyclass(name = "HighSymmetryData")]
#[derive(Clone)]
pub struct PyHighSymmetryData {
    pub(crate) inner: HighSymmetryData,
}

#[pymethods]
impl PyHighSymmetryData {
    /// Create new high symmetry data
    #[new]
    fn new() -> Self {
        PyHighSymmetryData { inner: HighSymmetryData::new() }
    }
    
    /// Get all high symmetry points as a dictionary
    ///
    /// Returns:
    ///     Dict[str, HighSymmetryPoint]: Map of label strings to points
    fn get_points(&self) -> Vec<(String, PyHighSymmetryPoint)> {
        self.inner.points.iter()
            .map(|(label, point)| {
                (label.as_str().to_string(), PyHighSymmetryPoint { inner: point.clone() })
            })
            .collect()
    }
    
    /// Get a specific point by label
    ///
    /// Args:
    ///     label: The SymmetryPointLabel to look up
    ///
    /// Returns:
    ///     Optional[HighSymmetryPoint]: The point if found
    fn get_point(&self, label: &PySymmetryPointLabel) -> Option<PyHighSymmetryPoint> {
        self.inner.get_point(&label.inner)
            .map(|p| PyHighSymmetryPoint { inner: p.clone() })
    }
    
    /// Get the standard path
    #[getter]
    fn standard_path(&self) -> PyHighSymmetryPath {
        PyHighSymmetryPath { inner: self.inner.standard_path.clone() }
    }
    
    /// Get all points along the standard path
    fn get_standard_path_points(&self) -> Vec<PyHighSymmetryPoint> {
        self.inner.get_standard_path_points()
            .into_iter()
            .map(|p| PyHighSymmetryPoint { inner: p.clone() })
            .collect()
    }
    
    /// Get alternative paths
    fn get_alternative_paths(&self) -> Vec<(String, PyHighSymmetryPath)> {
        self.inner.alternative_paths.iter()
            .map(|(name, path)| (name.clone(), PyHighSymmetryPath { inner: path.clone() }))
            .collect()
    }
    
    fn __repr__(&self) -> String {
        format!("HighSymmetryData(points={}, standard_path={:?})", 
                self.inner.points.len(), self.inner.standard_path.points)
    }
}
