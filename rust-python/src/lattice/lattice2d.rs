use pyo3::prelude::*;
use pyo3::types::PyDict;
use moire_lattice::lattice::Lattice2D;
use nalgebra::Vector3;
use std::f64::consts::PI;

/// Python wrapper for the 2D lattice structure
#[pyclass]
pub struct PyLattice2D {
    pub(crate) inner: Lattice2D,
}

#[pymethods]
impl PyLattice2D {
    /// Get lattice parameters (a, b)
    fn get_parameters(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let (a, b) = self.inner.lattice_parameters();
            let angle = self.inner.lattice_angle() * 180.0 / PI; // Convert to degrees
            dict.set_item("a", a)?;
            dict.set_item("b", b)?;
            dict.set_item("angle", angle)?;
            dict.set_item("lattice_type", format!("{:?}", self.inner.bravais))?;
            Ok(dict.into())
        })
    }

    /// Get lattice parameters separately
    fn lattice_parameters(&self) -> (f64, f64) {
        self.inner.lattice_parameters()
    }

    /// Get lattice angle in degrees
    fn lattice_angle(&self) -> f64 {
        self.inner.lattice_angle() * 180.0 / PI
    }

    /// Convert fractional coordinates to cartesian
    fn frac_to_cart(&self, u: f64, v: f64, w: Option<f64>) -> (f64, f64, f64) {
        let w_val = w.unwrap_or(0.0);
        let frac = Vector3::new(u, v, w_val);
        let cart = self.inner.frac_to_cart(frac);
        (cart.x, cart.y, cart.z)
    }

    /// Convert cartesian coordinates to fractional
    fn cart_to_frac(&self, x: f64, y: f64, z: Option<f64>) -> (f64, f64, f64) {
        let z_val = z.unwrap_or(0.0);
        let cart = Vector3::new(x, y, z_val);
        let frac = self.inner.cart_to_frac(cart);
        (frac.x, frac.y, frac.z)
    }

    /// Get the primitive vectors as tuples
    fn primitive_vectors(&self) -> ((f64, f64, f64), (f64, f64, f64)) {
        let (a_vec, b_vec) = self.inner.primitive_vectors();
        ((a_vec.x, a_vec.y, a_vec.z), (b_vec.x, b_vec.y, b_vec.z))
    }

    /// Get lattice vectors as tuples (for backward compatibility)
    fn lattice_vectors(&self) -> ((f64, f64), (f64, f64)) {
        let (a_vec, b_vec) = self.inner.primitive_vectors();
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
        self.inner.get_high_symmetry_points_cartesian()
            .into_iter()
            .map(|(label, point)| (label, (point.x, point.y, point.z)))
            .collect()
    }

    /// Generate k-points along the standard high symmetry path
    fn generate_k_path(&self, n_points_per_segment: usize) -> Vec<(f64, f64, f64)> {
        self.inner.generate_k_path(n_points_per_segment)
            .into_iter()
            .map(|k| (k.x, k.y, k.z))
            .collect()
    }

    /// Get direct lattice points within a rectangular region
    fn get_direct_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<(f64, f64, f64)> {
        self.inner.get_direct_lattice_points_in_rectangle(width, height)
            .into_iter()
            .map(|p| (p.x, p.y, p.z))
            .collect()
    }

    /// Get reciprocal lattice points within a rectangular region
    fn get_reciprocal_lattice_points_in_rectangle(&self, width: f64, height: f64) -> Vec<(f64, f64, f64)> {
        self.inner.get_reciprocal_lattice_points_in_rectangle(width, height)
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

    /// Convert to 3D lattice by adding a c-axis
    fn to_3d(&self, cx: f64, cy: f64, cz: f64) -> PyResult<crate::lattice::lattice3d::PyLattice3D> {
        let c_vector = Vector3::new(cx, cy, cz);
        let lattice_3d = self.inner.to_3d(c_vector);
        Ok(crate::lattice::lattice3d::PyLattice3D { inner: lattice_3d })
    }

    /// Generate lattice points within a given radius (for backward compatibility)
    #[pyo3(signature = (radius, center=(0.0, 0.0)))]
    fn generate_points(&self, radius: f64, center: (f64, f64)) -> PyResult<Vec<(f64, f64)>> {
        use moire_lattice::lattice::voronoi_cells::generate_lattice_points_2d_within_radius;
        
        // Generate neighbor points and filter by radius
        let points = generate_lattice_points_2d_within_radius(self.inner.direct_basis(), radius * 1.5); // Add some margin
        
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

    /// String representation
    fn __repr__(&self) -> String {
        let (a, b) = self.inner.lattice_parameters();
        let angle = self.inner.lattice_angle() * 180.0 / PI;
        format!("PyLattice2D({:?}, a={:.3}, b={:.3}, angle={:.1}°)",
            self.inner.bravais,
            a, b, angle
        )
    }
}

impl PyLattice2D {
    pub fn new(inner: Lattice2D) -> Self {
        PyLattice2D { inner }
    }
}
