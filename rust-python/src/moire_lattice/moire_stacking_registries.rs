use pyo3::prelude::*;
use nalgebra::Vector3;
use moire_lattice::moire_lattice::moire_stacking_registries::RegistryCenter as CoreRegistryCenter;
use moire_lattice::moire_lattice::Moire2D;
use crate::moire_lattice::PyMoire2D;
use crate::lattice::PyLattice2D;

/// Convert core RegistryCenter to a simple Python-friendly dict (returned as PyObject)
fn registry_center_to_py(py: Python<'_>, c: &CoreRegistryCenter) -> PyObject {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("label", c.label.clone()).unwrap();
    dict.set_item("tau", vec![c.tau.x, c.tau.y, c.tau.z]).unwrap();
    dict.set_item("position", vec![c.position.x, c.position.y, c.position.z]).unwrap();
    dict.into()
}

/// Return wrapped registry centers (recommended API) for a given moiré lattice and global shift d0.
#[pyfunction]
pub fn py_registry_centers(moire: &PyMoire2D, d0x: f64, d0y: f64) -> PyResult<Vec<PyObject>> {
    let d0 = Vector3::new(d0x, d0y, 0.0);
    let centers = moire.inner.registry_centers_monatomic(d0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    Python::with_gil(|py| {
        Ok(centers.iter().map(|c| registry_center_to_py(py, c)).collect())
    })
}

/// Return a dict mapping each registry label -> { 'lattice': PyLattice2D, 'basis': [[x,y,z], ...] }
/// using the preliminary local cell construction (two-site basis: origin + tau).
#[pyfunction]
pub fn py_local_cells_preliminary(moire: &PyMoire2D, d0x: f64, d0y: f64) -> PyResult<PyObject> {
    let d0 = Vector3::new(d0x, d0y, 0.0);
    let map = moire.inner.local_bravais_and_basis_from_registries_preliminary(d0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    Python::with_gil(|py| {
        let out = pyo3::types::PyDict::new(py);
        for (label, (lat, basis)) in map.into_iter() {
            let sub = pyo3::types::PyDict::new(py);
            sub.set_item("lattice", PyLattice2D { inner: lat }).unwrap();
            let basis_list: Vec<Vec<f64>> = basis.into_iter().map(|v| vec![v.x, v.y, v.z]).collect();
            sub.set_item("basis", basis_list).unwrap();
            out.set_item(label, sub).unwrap();
        }
        Ok(out.into())
    })
}

/// Return the nearest preliminary local cell at a given Cartesian point (x,y) with global shift d0.
/// Returns (label, PyLattice2D, basis_vectors)
#[pyfunction]
pub fn py_local_cell_at_point_preliminary(
    moire: &PyMoire2D,
    x: f64,
    y: f64,
    d0x: f64,
    d0y: f64,
) -> PyResult<(String, PyLattice2D, Vec<Vec<f64>>)> {
    let d0 = Vector3::new(d0x, d0y, 0.0);
    let point = Vector3::new(x, y, 0.0);
    let (label, lat, basis) = moire.inner.local_bravais_and_basis_at_point_preliminary(point, d0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    let basis_vecs: Vec<Vec<f64>> = basis.into_iter().map(|v| vec![v.x, v.y, v.z]).collect();
    Ok((label, PyLattice2D { inner: lat }, basis_vecs))
}
