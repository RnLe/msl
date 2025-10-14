use super::lattice_polyhedron::Polyhedron;
use moire_lattice::lattice::{
    Lattice2D,
    lattice_construction::{
        centered_rectangular_lattice, hexagonal_lattice, oblique_lattice, rectangular_lattice,
        square_lattice,
    },
    lattice_coordination_numbers::{
        coordination_number_2d, nearest_neighbor_distance_2d, nearest_neighbors_2d,
        packing_fraction_2d,
    },
    voronoi_cells::generate_lattice_points_2d_by_shell,
};
use nalgebra::Vector3;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::f64::consts::PI;

/// Python wrapper for the 2D lattice structure
#[pyclass]
pub struct PyLattice2D {
    pub(crate) inner: Lattice2D,
}

#[pymethods]
impl PyLattice2D {
    #[new]
    #[pyo3(signature = (lattice_type, a, b=None, angle=None))]
    fn new(lattice_type: &str, a: f64, b: Option<f64>, angle: Option<f64>) -> PyResult<Self> {
        let lattice = match lattice_type.to_lowercase().as_str() {
            "square" => square_lattice(a),
            "rectangular" => {
                let b_val = b.unwrap_or(a);
                rectangular_lattice(a, b_val)
            }
            "hexagonal" | "triangular" => hexagonal_lattice(a),
            "oblique" => {
                let b_val = b.unwrap_or(a);
                let angle_val = angle.unwrap_or(90.0) * PI / 180.0; // Convert to radians
                oblique_lattice(a, b_val, angle_val)
            }
            "centered_rectangular" => {
                let b_val = b.unwrap_or(a);
                centered_rectangular_lattice(a, b_val)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown lattice type: {}. Available types: square, rectangular, hexagonal, triangular, oblique, centered_rectangular",
                    lattice_type
                )));
            }
        };

        Ok(PyLattice2D { inner: lattice })
    }

    /// Get lattice parameters
    fn get_parameters(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let (a, b) = self.inner.direct_lattice_parameters();
            let angle = self.inner.direct_lattice_angle() * 180.0 / PI; // Convert to degrees
            dict.set_item("a", a)?;
            dict.set_item("b", b)?;
            dict.set_item("angle", angle)?;
            dict.set_item("lattice_type", format!("{:?}", self.inner.bravais))?;
            Ok(dict.into())
        })
    }

    /// Get lattice parameters separately
    fn lattice_parameters(&self) -> (f64, f64) {
        self.inner.direct_lattice_parameters()
    }

    /// Get lattice angle in degrees
    fn lattice_angle(&self) -> f64 {
        self.inner.direct_lattice_angle() * 180.0 / PI
    }

    /// Convert fractional coordinates to cartesian
    fn frac_to_cart(&self, u: f64, v: f64, w: Option<f64>) -> (f64, f64, f64) {
        let w_val = w.unwrap_or(0.0);
        let frac = Vector3::new(u, v, w_val);
        let cart = self.inner.fractional_to_cartesian(frac);
        (cart.x, cart.y, cart.z)
    }

    /// Convert cartesian coordinates to fractional
    fn cart_to_frac(&self, x: f64, y: f64, z: Option<f64>) -> (f64, f64, f64) {
        let z_val = z.unwrap_or(0.0);
        let cart = Vector3::new(x, y, z_val);
        let frac = self.inner.cartesian_to_fractional(cart);
        (frac.x, frac.y, frac.z)
    }

    /// Get the primitive vectors as tuples
    fn primitive_vectors(&self) -> ((f64, f64, f64), (f64, f64, f64)) {
        let (a_vec, b_vec) = self.inner.direct_base_vectors();
        ((a_vec.x, a_vec.y, a_vec.z), (b_vec.x, b_vec.y, b_vec.z))
    }

    /// Get lattice vectors as tuples (for backward compatibility)
    fn lattice_vectors(&self) -> ((f64, f64), (f64, f64)) {
        let (a_vec, b_vec) = self.inner.direct_base_vectors();
        ((a_vec.x, a_vec.y), (b_vec.x, b_vec.y))
    }

    /// Get reciprocal lattice vectors
    fn reciprocal_vectors(&self) -> ((f64, f64), (f64, f64)) {
        let g1 = self.inner.reciprocal_basis().column(0);
        let g2 = self.inner.reciprocal_basis().column(1);
        ((g1.x, g1.y), (g2.x, g2.y))
    }

    /// Check if a k-point is in the first Brillouin zone
    fn in_brillouin_zone(&self, kx: f64, ky: f64, kz: Option<f64>) -> bool {
        let kz_val = kz.unwrap_or(0.0);
        let k_point = Vector3::new(kx, ky, kz_val);
        self.inner.in_brillouin_zone(k_point)
    }

    /// Reduce a k-point to the first Brillouin zone
    fn reduce_to_brillouin_zone(&self, kx: f64, ky: f64, kz: Option<f64>) -> (f64, f64, f64) {
        let kz_val = kz.unwrap_or(0.0);
        let k_point = Vector3::new(kx, ky, kz_val);
        let reduced = self.inner.reduce_to_brillouin_zone(k_point);
        (reduced.x, reduced.y, reduced.z)
    }

    /// Get high symmetry points in Cartesian coordinates
    fn get_high_symmetry_points(&self) -> Vec<(String, (f64, f64, f64))> {
        self.inner
            .reciprocal_high_symmetry_points_cartesian()
            .into_iter()
            .map(|(label, point)| (label, (point.x, point.y, point.z)))
            .collect()
    }

    /// Get high symmetry points in fractional coordinates (for MPB solver)
    fn get_high_symmetry_points_fractional(&self) -> Vec<(String, (f64, f64, f64))> {
        let hs_data = self.inner.high_symmetry_data();
        hs_data
            .points
            .iter()
            .map(|(label, point)| {
                let pos = point.position;
                (label.as_str().to_string(), (pos.x, pos.y, pos.z))
            })
            .collect()
    }

    /// Generate k-points along the standard high symmetry path
    fn generate_k_path(&self, n_points_per_segment: usize) -> Vec<(f64, f64, f64)> {
        self.inner
            .generate_k_path(n_points_per_segment)
            .into_iter()
            .map(|k| (k.x, k.y, k.z))
            .collect()
    }

    /// Get direct lattice points within a rectangular region
    fn get_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<(f64, f64, f64)> {
        self.inner
            .get_direct_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }

    /// Get reciprocal lattice points within a rectangular region
    fn get_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<(f64, f64, f64)> {
        self.inner
            .get_reciprocal_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }

    /// Get the Bravais lattice type as string
    fn bravais_type(&self) -> String {
        format!("{:?}", self.inner.bravais_type())
    }

    /// Get the area of the unit cell
    fn unit_cell_area(&self) -> f64 {
        self.inner.cell_area()
    }

    /// Get the tolerance used for floating point comparisons
    fn tolerance(&self) -> f64 {
        self.inner.tolerance()
    }

    /// Generate lattice points within a given radius (for backward compatibility)
    #[pyo3(signature = (radius, center=(0.0, 0.0)))]
    fn generate_points(&self, radius: f64, center: (f64, f64)) -> PyResult<Vec<(f64, f64)>> {
        use moire_lattice::lattice::voronoi_cells::generate_lattice_points_2d_within_radius;

        // Generate neighbor points and filter by radius
        let points = generate_lattice_points_2d_within_radius(&self.inner.direct, radius * 1.5); // Add some margin

        let center_vec = Vector3::new(center.0, center.1, 0.0);
        let filtered_points: Vec<(f64, f64)> = points
            .into_iter()
            .filter(|p| {
                let dist = (p - center_vec).norm();
                dist <= radius
            })
            .map(|p| (p.x, p.y))
            .collect();

        Ok(filtered_points)
    }

    /// Get coordination number for this lattice type
    fn coordination_number(&self) -> usize {
        coordination_number_2d(&self.inner.bravais)
    }

    /// Get nearest neighbors for this lattice type
    fn nearest_neighbors(&self) -> Vec<(f64, f64, f64)> {
        let neighbors =
            nearest_neighbors_2d(&self.inner.direct, &self.inner.bravais, self.inner.tol);
        neighbors.into_iter().map(|n| (n.x, n.y, n.z)).collect()
    }

    /// Get nearest neighbor distance for this lattice type
    fn nearest_neighbor_distance(&self) -> f64 {
        nearest_neighbor_distance_2d(&self.inner.direct, &self.inner.bravais)
    }

    /// Get packing fraction for this lattice type
    fn packing_fraction(&self) -> f64 {
        let (a, b) = self.inner.direct_lattice_parameters();
        packing_fraction_2d(&self.inner.bravais, (a, b))
    }

    /// String representation
    fn __repr__(&self) -> String {
        let (a, b) = self.inner.direct_lattice_parameters();
        let angle = self.inner.direct_lattice_angle() * 180.0 / PI;
        format!(
            "PyLattice2D({:?}, a={:.3}, b={:.3}, angle={:.1}°)",
            self.inner.bravais, a, b, angle
        )
    }

    /// Get the Wigner-Seitz cell (Voronoi cell in direct space)
    fn wigner_seitz_cell(&self) -> Polyhedron {
        Polyhedron::new(self.inner.wigner_seitz_cell.clone())
    }

    /// Get the Brillouin zone (Voronoi cell in reciprocal space)
    fn brillouin_zone(&self) -> Polyhedron {
        Polyhedron::new(self.inner.brillouin_zone.clone())
    }

    /// Get symmetry operations as (rotation matrix, translation vector)
    fn symmetry_operations(
        &self,
    ) -> Vec<(((i8, i8, i8), (i8, i8, i8), (i8, i8, i8)), (f64, f64, f64))> {
        self.inner
            .symmetry_operations()
            .iter()
            .map(|sym_op| {
                let rot = sym_op.rotation;
                let trans = sym_op.translation;
                (
                    (
                        (rot[(0, 0)], rot[(0, 1)], rot[(0, 2)]),
                        (rot[(1, 0)], rot[(1, 1)], rot[(1, 2)]),
                        (rot[(2, 0)], rot[(2, 1)], rot[(2, 2)]),
                    ),
                    (trans.x, trans.y, trans.z),
                )
            })
            .collect()
    }

    /// Generate lattice points within radius using shells
    fn generate_points_by_shell(&self, max_shell: usize) -> Vec<(f64, f64, f64)> {
        generate_lattice_points_2d_by_shell(&self.inner.direct, max_shell)
            .into_iter()
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }

    /// Access to direct basis matrix elements
    fn direct_basis(&self) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
        let basis = &self.inner.direct;
        (
            (basis[(0, 0)], basis[(1, 0)], basis[(2, 0)]),
            (basis[(0, 1)], basis[(1, 1)], basis[(2, 1)]),
            (basis[(0, 2)], basis[(1, 2)], basis[(2, 2)]),
        )
    }

    /// Access to reciprocal basis matrix elements
    fn reciprocal_basis(&self) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
        let basis = &self.inner.reciprocal;
        (
            (basis[(0, 0)], basis[(1, 0)], basis[(2, 0)]),
            (basis[(0, 1)], basis[(1, 1)], basis[(2, 1)]),
            (basis[(0, 2)], basis[(1, 2)], basis[(2, 2)]),
        )
    }

    /// Get metric tensor elements
    fn metric_tensor(&self) -> ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64)) {
        let metric = self.inner.metric_tensor();
        (
            (metric[(0, 0)], metric[(1, 0)], metric[(2, 0)]),
            (metric[(0, 1)], metric[(1, 1)], metric[(2, 1)]),
            (metric[(0, 2)], metric[(1, 2)], metric[(2, 2)]),
        )
    }
}

impl PyLattice2D {
    /// Create a PyLattice2D from an inner Lattice2D instance
    pub fn from_inner(inner: Lattice2D) -> Self {
        PyLattice2D { inner }
    }
}
