use pyo3::prelude::*;
use nalgebra::Vector3;
use moire_lattice::moire_lattice::{
    find_commensurate_angles, validate_commensurability, compute_moire_basis,
    analyze_moire_symmetry, moire_potential_at
};
use crate::lattice::PyLattice2D;
use crate::moire_lattice::PyMoire2D;

/// Find commensurate angles for a given lattice
#[pyfunction]
pub fn py_find_commensurate_angles(
    lattice: &PyLattice2D,
    max_index: i32,
) -> PyResult<Vec<(f64, (i32, i32, i32, i32))>> {
    let angles = find_commensurate_angles(&lattice.inner, max_index)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    Ok(angles)
}

/// Validate if two lattices form a commensurate moiré pattern
#[pyfunction]
pub fn py_validate_commensurability(
    lattice_1: &PyLattice2D,
    lattice_2: &PyLattice2D,
    tolerance: f64,
) -> (bool, Option<(i32, i32, i32, i32)>) {
    validate_commensurability(&lattice_1.inner, &lattice_2.inner, tolerance)
}

/// Compute the moiré basis vectors from two lattices
#[pyfunction]
pub fn py_compute_moire_basis(
    lattice_1: &PyLattice2D,
    lattice_2: &PyLattice2D,
    tolerance: f64,
) -> PyResult<Vec<Vec<f64>>> {
    let basis = compute_moire_basis(&lattice_1.inner, &lattice_2.inner, tolerance)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    // Convert Matrix3 to Vec<Vec<f64>>
    let mut result = Vec::new();
    for i in 0..3 {
        let mut row = Vec::new();
        for j in 0..3 {
            row.push(basis[(i, j)]);
        }
        result.push(row);
    }
    
    Ok(result)
}

/// Check if a moiré pattern has specific symmetries
#[pyfunction]
pub fn py_analyze_moire_symmetry(moire: &PyMoire2D) -> Vec<String> {
    analyze_moire_symmetry(&moire.inner)
}

/// Compute the effective moiré potential at a given point
#[pyfunction]
pub fn py_moire_potential_at(
    moire: &PyMoire2D,
    point: Vec<f64>,
    v_aa: f64,
    v_ab: f64,
) -> PyResult<f64> {
    if point.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Point must have 3 coordinates"
        ));
    }
    
    let vec = Vector3::new(point[0], point[1], point[2]);
    Ok(moire_potential_at(&moire.inner, vec, v_aa, v_ab))
}

/// Get twist angles in degrees instead of radians
#[pyfunction]
pub fn py_find_commensurate_angles_degrees(
    lattice: &PyLattice2D,
    max_index: i32,
) -> PyResult<Vec<(f64, (i32, i32, i32, i32))>> {
    let angles = find_commensurate_angles(&lattice.inner, max_index)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;
    
    // Convert angles from radians to degrees
    let angles_degrees = angles.into_iter()
        .map(|(angle, indices)| (angle.to_degrees(), indices))
        .collect();
    
    Ok(angles_degrees)
}

/// Helper function to get the magic angles for twisted bilayer graphene
#[pyfunction]
pub fn py_magic_angles_graphene(max_index: Option<i32>) -> PyResult<Vec<f64>> {
    let max_idx = max_index.unwrap_or(20);
    
    // Create a hexagonal lattice (graphene-like)
    let lattice = moire_lattice::lattice::lattice_construction::hexagonal_lattice(1.0);
    let py_lattice = PyLattice2D { inner: lattice };
    
    let angles = py_find_commensurate_angles_degrees(&py_lattice, max_idx)?;
    
    // Filter for "magic angles" (typically small angles with high commensurability)
    let magic_angles: Vec<f64> = angles.into_iter()
        .map(|(angle, _)| angle)
        .filter(|&angle| angle > 0.0 && angle < 10.0) // Focus on small angles
        .take(10) // Take first 10
        .collect();
    
    Ok(magic_angles)
}
