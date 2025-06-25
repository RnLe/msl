use pyo3::prelude::*;
use moire_lattice::lattice::lattice_bravais_types::*;
use nalgebra::Matrix3;

/// Python wrapper for 2D Bravais lattice types
#[pyclass]
#[derive(Clone)]
pub struct PyBravais2D {
    pub(crate) inner: Bravais2D,
}

#[pymethods]
impl PyBravais2D {
    #[new]
    fn new(bravais_type: &str) -> PyResult<Self> {
        let bravais = match bravais_type.to_lowercase().as_str() {
            "oblique" => Bravais2D::Oblique,
            "rectangular" => Bravais2D::Rectangular,
            "centered_rectangular" => Bravais2D::CenteredRectangular,
            "square" => Bravais2D::Square,
            "hexagonal" => Bravais2D::Hexagonal,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown Bravais type: {}. Available types: oblique, rectangular, centered_rectangular, square, hexagonal", bravais_type)
            )),
        };
        Ok(PyBravais2D { inner: bravais })
    }

    /// Get the string representation
    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Get the string representation
    fn __repr__(&self) -> String {
        format!("PyBravais2D('{:?}')", self.inner)
    }

    /// Check equality
    fn __eq__(&self, other: &PyBravais2D) -> bool {
        self.inner == other.inner
    }

    /// Get all available 2D Bravais types
    #[staticmethod]
    fn available_types() -> Vec<String> {
        vec![
            "Oblique".to_string(),
            "Rectangular".to_string(),
            "CenteredRectangular".to_string(),
            "Square".to_string(),
            "Hexagonal".to_string(),
        ]
    }

    /// Get the crystal system name
    fn crystal_system(&self) -> String {
        match self.inner {
            Bravais2D::Oblique => "Oblique".to_string(),
            Bravais2D::Rectangular => "Rectangular".to_string(),
            Bravais2D::CenteredRectangular => "Centered Rectangular".to_string(),
            Bravais2D::Square => "Square".to_string(),
            Bravais2D::Hexagonal => "Hexagonal".to_string(),
        }
    }

    /// Get the point group symbol
    fn point_group(&self) -> String {
        match self.inner {
            Bravais2D::Oblique => "p1".to_string(),
            Bravais2D::Rectangular => "p2mm".to_string(),
            Bravais2D::CenteredRectangular => "c2mm".to_string(),
            Bravais2D::Square => "p4mm".to_string(),
            Bravais2D::Hexagonal => "p6mm".to_string(),
        }
    }

    /// Get the lattice symmetry constraints
    fn symmetry_constraints(&self) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            match self.inner {
                Bravais2D::Oblique => {
                    dict.set_item("length_constraint", "a ≠ b")?;
                    dict.set_item("angle_constraint", "γ ≠ 90°")?;
                },
                Bravais2D::Rectangular => {
                    dict.set_item("length_constraint", "a ≠ b")?;
                    dict.set_item("angle_constraint", "γ = 90°")?;
                },
                Bravais2D::CenteredRectangular => {
                    dict.set_item("length_constraint", "a ≠ b")?;
                    dict.set_item("angle_constraint", "γ = 90°")?;
                    dict.set_item("centering", "centered")?;
                },
                Bravais2D::Square => {
                    dict.set_item("length_constraint", "a = b")?;
                    dict.set_item("angle_constraint", "γ = 90°")?;
                },
                Bravais2D::Hexagonal => {
                    dict.set_item("length_constraint", "a = b")?;
                    dict.set_item("angle_constraint", "γ = 120°")?;
                },
            }
            
            Ok(dict.into())
        })
    }
}

/// Identify 2D Bravais lattice type from lattice parameters
#[pyfunction]
pub fn identify_bravais_2d_from_parameters(a: f64, b: f64, gamma_degrees: f64, tol: Option<f64>) -> PyBravais2D {
    use std::f64::consts::PI;
    
    let tolerance = tol.unwrap_or(1e-6);
    let gamma = gamma_degrees * PI / 180.0;
    
    // Create a simple metric tensor from parameters
    let cos_gamma = gamma.cos();
    let metric = Matrix3::new(
        a * a, a * b * cos_gamma, 0.0,
        a * b * cos_gamma, b * b, 0.0,
        0.0, 0.0, 1.0
    );
    
    let bravais = identify_bravais_2d(&metric, tolerance);
    PyBravais2D { inner: bravais }
}

/// Get information about all 2D Bravais lattices
#[pyfunction]
pub fn get_all_bravais_2d_info() -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        
        let lattices = vec![
            ("Oblique", "a ≠ b, γ ≠ 90°", "p1"),
            ("Rectangular", "a ≠ b, γ = 90°", "p2mm"),
            ("CenteredRectangular", "a ≠ b, γ = 90°, centered", "c2mm"),
            ("Square", "a = b, γ = 90°", "p4mm"),
            ("Hexagonal", "a = b, γ = 120°", "p6mm"),
        ];
        
        for (name, constraints, point_group) in lattices {
            let lattice_dict = PyDict::new(py);
            lattice_dict.set_item("constraints", constraints)?;
            lattice_dict.set_item("point_group", point_group)?;
            dict.set_item(name, lattice_dict)?;
        }
        
        Ok(dict.into())
    })
}

/// Python wrapper for 3D centering types
#[pyclass]
#[derive(Clone)]
pub struct PyCentering {
    pub(crate) inner: Centering,
}

#[pymethods]
impl PyCentering {
    #[new]
    fn new(centering_type: &str) -> PyResult<Self> {
        let centering = match centering_type.to_lowercase().as_str() {
            "primitive" | "p" => Centering::Primitive,
            "body" | "body_centered" | "i" => Centering::BodyCentered,
            "face" | "face_centered" | "f" => Centering::FaceCentered,
            "base" | "base_centered" | "a" | "b" | "c" => Centering::BaseCentered,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown centering type: {}. Available types: primitive, body_centered, face_centered, base_centered", centering_type)
            )),
        };
        Ok(PyCentering { inner: centering })
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PyCentering('{:?}')", self.inner)
    }

    /// Get the Pearson symbol
    fn pearson_symbol(&self) -> String {
        match self.inner {
            Centering::Primitive => "P".to_string(),
            Centering::BodyCentered => "I".to_string(),
            Centering::FaceCentered => "F".to_string(),
            Centering::BaseCentered => "C".to_string(), // or A, B depending on which face
        }
    }

    /// Get multiplicity (number of lattice points per unit cell)
    fn multiplicity(&self) -> usize {
        match self.inner {
            Centering::Primitive => 1,
            Centering::BodyCentered => 2,
            Centering::FaceCentered => 4,
            Centering::BaseCentered => 2,
        }
    }
}
