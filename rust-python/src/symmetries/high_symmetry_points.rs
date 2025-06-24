use pyo3::prelude::*;
use moire_lattice::symmetries::high_symmetry_points::*;
use crate::lattice::lattice2d::PyLattice2D;

/// Python wrapper for symmetry point labels
#[pyclass]
#[derive(Clone)]
pub struct PySymmetryPointLabel {
    pub(crate) inner: SymmetryPointLabel,
}

#[pymethods]
impl PySymmetryPointLabel {
    #[new]
    fn new(label: &str) -> PyResult<Self> {
        let point_label = match label.to_uppercase().as_str() {
            "GAMMA" | "Γ" => SymmetryPointLabel::Gamma,
            "M" => SymmetryPointLabel::M,
            "K" => SymmetryPointLabel::K,
            "X" => SymmetryPointLabel::X,
            "Y" => SymmetryPointLabel::Y,
            "R" => SymmetryPointLabel::R,
            "L" => SymmetryPointLabel::L,
            "W" => SymmetryPointLabel::W,
            "Z" => SymmetryPointLabel::Z,
            "A" => SymmetryPointLabel::A,
            "T" => SymmetryPointLabel::T,
            "S" => SymmetryPointLabel::S,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown symmetry point label: {}", label)
            )),
        };
        Ok(PySymmetryPointLabel { inner: point_label })
    }

    fn as_str(&self) -> String {
        self.inner.as_str().to_string()
    }

    fn __str__(&self) -> String {
        self.as_str()
    }

    fn __repr__(&self) -> String {
        format!("PySymmetryPointLabel('{}')", self.as_str())
    }

    /// Get all available symmetry point labels
    #[staticmethod]
    fn available_labels() -> Vec<String> {
        vec![
            "Gamma".to_string(), "M".to_string(), "K".to_string(), "X".to_string(),
            "Y".to_string(), "R".to_string(), "L".to_string(), "W".to_string(),
            "Z".to_string(), "A".to_string(), "T".to_string(), "S".to_string(),
        ]
    }
}

/// Python wrapper for high symmetry points
#[pyclass]
pub struct PyHighSymmetryPoint {
    pub(crate) inner: HighSymmetryPoint,
}

#[pymethods]
impl PyHighSymmetryPoint {
    /// Get the label
    fn label(&self) -> PySymmetryPointLabel {
        PySymmetryPointLabel { inner: self.inner.label }
    }

    /// Get the position in fractional coordinates
    fn position(&self) -> (f64, f64, f64) {
        let pos = self.inner.position;
        (pos.x, pos.y, pos.z)
    }

    /// Get the description
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    fn __repr__(&self) -> String {
        let pos = self.inner.position;
        format!("PyHighSymmetryPoint({}, ({:.3}, {:.3}, {:.3}), '{}')",
            self.inner.label.as_str(),
            pos.x, pos.y, pos.z,
            self.inner.description
        )
    }
}

/// Python wrapper for high symmetry paths
#[pyclass]
pub struct PyHighSymmetryPath {
    pub(crate) inner: HighSymmetryPath,
}

#[pymethods]
impl PyHighSymmetryPath {
    #[new]
    fn new(point_labels: Vec<String>) -> PyResult<Self> {
        let labels: Result<Vec<SymmetryPointLabel>, _> = point_labels
            .into_iter()
            .map(|s| match s.to_uppercase().as_str() {
                "GAMMA" | "Γ" => Ok(SymmetryPointLabel::Gamma),
                "M" => Ok(SymmetryPointLabel::M),
                "K" => Ok(SymmetryPointLabel::K),
                "X" => Ok(SymmetryPointLabel::X),
                "Y" => Ok(SymmetryPointLabel::Y),
                "R" => Ok(SymmetryPointLabel::R),
                "L" => Ok(SymmetryPointLabel::L),
                "W" => Ok(SymmetryPointLabel::W),
                "Z" => Ok(SymmetryPointLabel::Z),
                "A" => Ok(SymmetryPointLabel::A),
                "T" => Ok(SymmetryPointLabel::T),
                "S" => Ok(SymmetryPointLabel::S),
                _ => Err(format!("Unknown label: {}", s)),
            })
            .collect();
        
        match labels {
            Ok(labels) => Ok(PyHighSymmetryPath { 
                inner: HighSymmetryPath::new(labels) 
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e)),
        }
    }

    /// Set the number of interpolation points
    fn with_n_points(&mut self, n: usize) -> &mut Self {
        self.inner = self.inner.clone().with_n_points(n);
        self
    }

    /// Get the point labels as strings
    fn point_labels(&self) -> Vec<String> {
        self.inner.points.iter()
            .map(|label| label.as_str().to_string())
            .collect()
    }

    /// Get the number of interpolation points
    fn n_points(&self) -> usize {
        self.inner.n_points_per_segment
    }
}

/// Python wrapper for high symmetry data
#[pyclass]
pub struct PyHighSymmetryData {
    pub(crate) inner: HighSymmetryData,
}

#[pymethods]
impl PyHighSymmetryData {
    /// Get all high symmetry points
    fn get_points(&self) -> Vec<PyHighSymmetryPoint> {
        self.inner.points.values()
            .map(|point| PyHighSymmetryPoint { inner: point.clone() })
            .collect()
    }

    /// Get a specific point by label
    fn get_point(&self, label: &str) -> Option<PyHighSymmetryPoint> {
        let point_label = match label.to_uppercase().as_str() {
            "GAMMA" | "Γ" => SymmetryPointLabel::Gamma,
            "M" => SymmetryPointLabel::M,
            "K" => SymmetryPointLabel::K,
            "X" => SymmetryPointLabel::X,
            "Y" => SymmetryPointLabel::Y,
            _ => return None,
        };
        
        self.inner.get_point(&point_label)
            .map(|point| PyHighSymmetryPoint { inner: point.clone() })
    }

    /// Get the standard path
    fn get_standard_path(&self) -> Option<PyHighSymmetryPath> {
        self.inner.standard_path.as_ref()
            .map(|path| PyHighSymmetryPath { inner: path.clone() })
    }

    /// Get alternative paths
    fn get_alternative_paths(&self) -> Vec<(String, PyHighSymmetryPath)> {
        self.inner.alternative_paths.iter()
            .map(|(name, path)| (name.clone(), PyHighSymmetryPath { inner: path.clone() }))
            .collect()
    }

    /// Get the standard path points
    fn get_standard_path_points(&self) -> Vec<PyHighSymmetryPoint> {
        self.inner.get_standard_path_points()
            .into_iter()
            .map(|point| PyHighSymmetryPoint { inner: point.clone() })
            .collect()
    }
}

/// Generate high symmetry points for a 2D lattice
#[pyfunction]
pub fn generate_2d_high_symmetry_points(lattice: &PyLattice2D) -> PyHighSymmetryData {
    let data = generate_2d_high_symmetry_points(&lattice.inner.bravais_type());
    PyHighSymmetryData { inner: data }
}

/// Interpolate a path between high symmetry points
#[pyfunction]
pub fn interpolate_path(points: Vec<PyHighSymmetryPoint>, n_points_per_segment: usize) -> Vec<(f64, f64, f64)> {
    let rust_points: Vec<HighSymmetryPoint> = points.into_iter()
        .map(|p| p.inner)
        .collect();
    
    interpolate_path(&rust_points, n_points_per_segment)
        .into_iter()
        .map(|v| (v.x, v.y, v.z))
        .collect()
}

/// Get high symmetry information for a lattice
#[pyfunction]
pub fn get_high_symmetry_info(lattice: &PyLattice2D) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        
        let hs_data = lattice.inner.high_symmetry_data();
        
        // Get all points
        let points_dict = PyDict::new(py);
        for (label, point) in &hs_data.points {
            let point_dict = PyDict::new(py);
            point_dict.set_item("position", (point.position.x, point.position.y, point.position.z))?;
            point_dict.set_item("description", &point.description)?;
            points_dict.set_item(label.as_str(), point_dict)?;
        }
        dict.set_item("points", points_dict)?;
        
        // Get standard path
        if let Some(path) = &hs_data.standard_path {
            let path_labels: Vec<String> = path.points.iter()
                .map(|label| label.as_str().to_string())
                .collect();
            dict.set_item("standard_path", path_labels)?;
            dict.set_item("n_points_per_segment", path.n_points_per_segment)?;
        }
        
        // Get alternative paths
        if !hs_data.alternative_paths.is_empty() {
            let alt_paths_dict = PyDict::new(py);
            for (name, path) in &hs_data.alternative_paths {
                let path_labels: Vec<String> = path.points.iter()
                    .map(|label| label.as_str().to_string())
                    .collect();
                alt_paths_dict.set_item(name, path_labels)?;
            }
            dict.set_item("alternative_paths", alt_paths_dict)?;
        }
        
        dict.set_item("lattice_type", format!("{:?}", lattice.inner.bravais_type()))?;
        
        Ok(dict.into())
    })
}
