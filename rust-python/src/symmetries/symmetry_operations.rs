use pyo3::prelude::*;
use moire_lattice::symmetries::symmetry_operations::SymOp;
use nalgebra::Vector3;

/// Python wrapper for symmetry operations
#[pyclass]
pub struct PySymOp {
    pub(crate) inner: SymOp,
}

#[pymethods]
impl PySymOp {
    #[new]
    fn new() -> Self {
        PySymOp { inner: SymOp::new() }
    }

    /// Apply symmetry operation to a point
    fn apply(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let point = Vector3::new(x, y, z);
        let transformed = self.inner.apply(point);
        (transformed.x, transformed.y, transformed.z)
    }

    /// Apply to multiple points at once
    fn apply_multiple(&self, points: Vec<(f64, f64, f64)>) -> Vec<(f64, f64, f64)> {
        points.into_iter()
            .map(|(x, y, z)| {
                let point = Vector3::new(x, y, z);
                let transformed = self.inner.apply(point);
                (transformed.x, transformed.y, transformed.z)
            })
            .collect()
    }

    /// Get the inverse operation
    fn inverse(&self) -> PySymOp {
        PySymOp { inner: self.inner.inverse() }
    }

    /// Compose with another symmetry operation
    fn compose(&self, other: &PySymOp) -> PySymOp {
        PySymOp { inner: self.inner.compose(&other.inner) }
    }

    /// Check if this is the identity operation
    fn is_identity(&self) -> bool {
        // This would need to be implemented in the Rust SymOp
        // For now, return a placeholder
        false
    }

    /// String representation
    fn __repr__(&self) -> String {
        "PySymOp(...)".to_string() // This would need more detailed info from Rust
    }
}

/// Generate symmetry operations for a 2D lattice
#[pyfunction]
pub fn generate_symmetry_operations_2d(lattice: &crate::lattice::lattice2d::PyLattice2D) -> Vec<PySymOp> {
    use moire_lattice::symmetries::point_groups::generate_symmetry_operations_2d;
    
    let ops = generate_symmetry_operations_2d(&lattice.inner.bravais_type());
    ops.into_iter()
        .map(|op| PySymOp { inner: op })
        .collect()
}

/// Python wrapper for symmetry analysis
#[pyclass]
pub struct PySymmetryAnalysis;

#[pymethods]
impl PySymmetryAnalysis {
    #[new]
    fn new() -> Self {
        PySymmetryAnalysis
    }

    /// Analyze all symmetry operations for a lattice
    fn analyze_symmetries(&self, lattice: &crate::lattice::lattice2d::PyLattice2D) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            
            let sym_ops = lattice.inner.symmetry_operations();
            dict.set_item("num_operations", sym_ops.len())?;
            dict.set_item("lattice_type", format!("{:?}", lattice.inner.bravais_type()))?;
            
            // Count operation types (this would need more detailed analysis in Rust)
            dict.set_item("point_group", self.get_point_group_symbol(&lattice.inner.bravais_type()))?;
            
            Ok(dict.into())
        })
    }

    /// Get point group symbol for a Bravais lattice
    fn get_point_group_symbol(&self, bravais: &moire_lattice::lattice::bravais_types::Bravais2D) -> String {
        match bravais {
            moire_lattice::lattice::bravais_types::Bravais2D::Oblique => "p1".to_string(),
            moire_lattice::lattice::bravais_types::Bravais2D::Rectangular => "p2mm".to_string(),
            moire_lattice::lattice::bravais_types::Bravais2D::CenteredRectangular => "c2mm".to_string(),
            moire_lattice::lattice::bravais_types::Bravais2D::Square => "p4mm".to_string(),
            moire_lattice::lattice::bravais_types::Bravais2D::Hexagonal => "p6mm".to_string(),
        }
    }

    /// Check if a point is equivalent under symmetry
    fn find_equivalent_points(&self, lattice: &crate::lattice::lattice2d::PyLattice2D, x: f64, y: f64, z: Option<f64>) -> Vec<(f64, f64, f64)> {
        let z_val = z.unwrap_or(0.0);
        let point = Vector3::new(x, y, z_val);
        let sym_ops = lattice.inner.symmetry_operations();
        
        let mut equivalent_points = Vec::new();
        for op in sym_ops {
            let transformed = op.apply(point);
            equivalent_points.push((transformed.x, transformed.y, transformed.z));
        }
        
        // Remove duplicates (would need a proper tolerance-based comparison in practice)
        equivalent_points.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
                .then(a.2.partial_cmp(&b.2).unwrap())
        });
        equivalent_points.dedup();
        
        equivalent_points
    }
}
