//! Python bindings for Bravais lattice types

use pyo3::prelude::*;
use moire_lattice::lattice::lattice_types::{Bravais2D, Bravais3D, Centering};

/// Python wrapper for Bravais2D enum
#[pyclass(name = "Bravais2D")]
#[derive(Clone, Copy)]
pub struct PyBravais2D {
    pub(crate) inner: Bravais2D,
}

#[pymethods]
impl PyBravais2D {
    /// Create an Oblique lattice
    #[staticmethod]
    fn oblique() -> Self {
        PyBravais2D { inner: Bravais2D::Oblique }
    }
    
    /// Create a Rectangular lattice
    #[staticmethod]
    fn rectangular() -> Self {
        PyBravais2D { inner: Bravais2D::Rectangular }
    }
    
    /// Create a Centered Rectangular lattice
    #[staticmethod]
    fn centered_rectangular() -> Self {
        PyBravais2D { inner: Bravais2D::CenteredRectangular }
    }
    
    /// Create a Square lattice
    #[staticmethod]
    fn square() -> Self {
        PyBravais2D { inner: Bravais2D::Square }
    }
    
    /// Create a Hexagonal lattice
    #[staticmethod]
    fn hexagonal() -> Self {
        PyBravais2D { inner: Bravais2D::Hexagonal }
    }
    
    /// Get the name of the Bravais lattice type
    pub fn name(&self) -> &str {
        match self.inner {
            Bravais2D::Oblique => "Oblique",
            Bravais2D::Rectangular => "Rectangular",
            Bravais2D::CenteredRectangular => "CenteredRectangular",
            Bravais2D::Square => "Square",
            Bravais2D::Hexagonal => "Hexagonal",
        }
    }
    
    fn __repr__(&self) -> String {
        format!("Bravais2D.{}", self.name())
    }
    
    fn __str__(&self) -> String {
        self.name().to_string()
    }
    
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for Centering enum
#[pyclass(name = "Centering")]
#[derive(Clone, Copy)]
pub struct PyCentering {
    pub(crate) inner: Centering,
}

#[pymethods]
impl PyCentering {
    /// Primitive centering
    #[staticmethod]
    fn primitive() -> Self {
        PyCentering { inner: Centering::Primitive }
    }
    
    /// Body-centered
    #[staticmethod]
    fn body_centered() -> Self {
        PyCentering { inner: Centering::BodyCentered }
    }
    
    /// Face-centered
    #[staticmethod]
    fn face_centered() -> Self {
        PyCentering { inner: Centering::FaceCentered }
    }
    
    /// Base-centered
    #[staticmethod]
    fn base_centered() -> Self {
        PyCentering { inner: Centering::BaseCentered }
    }
    
    /// Get the name of the centering type
    pub fn name(&self) -> &str {
        match self.inner {
            Centering::Primitive => "Primitive",
            Centering::BodyCentered => "BodyCentered",
            Centering::FaceCentered => "FaceCentered",
            Centering::BaseCentered => "BaseCentered",
        }
    }
    
    fn __repr__(&self) -> String {
        format!("Centering.{}", self.name())
    }
    
    fn __str__(&self) -> String {
        self.name().to_string()
    }
    
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Python wrapper for Bravais3D enum
#[pyclass(name = "Bravais3D")]
#[derive(Clone, Copy)]
pub struct PyBravais3D {
    pub(crate) inner: Bravais3D,
}

#[pymethods]
impl PyBravais3D {
    /// Create a Triclinic lattice
    #[staticmethod]
    fn triclinic(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Triclinic(centering.inner) }
    }
    
    /// Create a Monoclinic lattice
    #[staticmethod]
    fn monoclinic(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Monoclinic(centering.inner) }
    }
    
    /// Create an Orthorhombic lattice
    #[staticmethod]
    fn orthorhombic(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Orthorhombic(centering.inner) }
    }
    
    /// Create a Tetragonal lattice
    #[staticmethod]
    fn tetragonal(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Tetragonal(centering.inner) }
    }
    
    /// Create a Trigonal lattice
    #[staticmethod]
    fn trigonal(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Trigonal(centering.inner) }
    }
    
    /// Create a Hexagonal lattice
    #[staticmethod]
    fn hexagonal(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Hexagonal(centering.inner) }
    }
    
    /// Create a Cubic lattice
    #[staticmethod]
    fn cubic(centering: &PyCentering) -> Self {
        PyBravais3D { inner: Bravais3D::Cubic(centering.inner) }
    }
    
    /// Get the name of the crystal system
    fn name(&self) -> String {
        match self.inner {
            Bravais3D::Triclinic(c) => format!("Triclinic({})", centering_name(c)),
            Bravais3D::Monoclinic(c) => format!("Monoclinic({})", centering_name(c)),
            Bravais3D::Orthorhombic(c) => format!("Orthorhombic({})", centering_name(c)),
            Bravais3D::Tetragonal(c) => format!("Tetragonal({})", centering_name(c)),
            Bravais3D::Trigonal(c) => format!("Trigonal({})", centering_name(c)),
            Bravais3D::Hexagonal(c) => format!("Hexagonal({})", centering_name(c)),
            Bravais3D::Cubic(c) => format!("Cubic({})", centering_name(c)),
        }
    }
    
    fn __repr__(&self) -> String {
        format!("Bravais3D.{}", self.name())
    }
    
    fn __str__(&self) -> String {
        self.name()
    }
    
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

fn centering_name(c: Centering) -> &'static str {
    match c {
        Centering::Primitive => "Primitive",
        Centering::BodyCentered => "BodyCentered",
        Centering::FaceCentered => "FaceCentered",
        Centering::BaseCentered => "BaseCentered",
    }
}
