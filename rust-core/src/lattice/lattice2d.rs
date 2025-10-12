use anyhow::Error;
use nalgebra::{Matrix3, Vector3};

use crate::config::LATTICE_TOLERANCE;
use crate::interfaces::Space;
use crate::lattice::base_matrix::BaseMatrix;
use crate::lattice::lattice_types::{Bravais2D, identify_bravais_2d};
use crate::lattice::polyhedron::{Polyhedron, bz_halfspaces_from_poly};
use crate::lattice::voronoi_cells::{compute_brillouin_zone_2d, compute_wigner_seitz_cell_2d};
use crate::symmetries::high_symmetry_points::{HighSymmetryData, generate_2d_high_symmetry_points};
use crate::symmetries::symmetry_operations::SymmetryOperation;
use crate::symmetries::symmetry_point_groups::generate_symmetry_operations_2d;

/// A 2D Bravais lattice embedded in 3D space.
#[derive(Debug, Clone)]
pub struct Lattice2D {
    /// Real‐space basis vectors (columns), with third column typically for z-direction.
    pub direct_basis: BaseMatrix,
    /// Reciprocal-space basis vectors (columns), with third column typically for z-direction.
    pub reciprocal_basis: BaseMatrix,
    /// Real-space 2D Bravais lattice type.
    pub direct_bravais: Bravais2D,
    /// Reciprocal-space 2D Bravais lattice type.
    pub reciprocal_bravais: Bravais2D,
    /// Direct metric tensor G = A^T * A; where A is the direct basis matrix.
    pub direct_metric: Matrix3<f64>,
    /// Reciprocal metric tensor G* = B^T * B; where B is the reciprocal basis matrix.
    pub reciprocal_metric: Matrix3<f64>,
    /// Real-space symmetry operations based on bravais type.
    pub direct_symmetry_operations: Vec<SymmetryOperation>,
    /// Reciprocal-space symmetry operations based on bravais type.
    pub reciprocal_symmetry_operations: Vec<SymmetryOperation>,
    /// Wigner-Seitz cell in direct space
    pub wigner_seitz_cell: Polyhedron,
    /// First Brillouin zone in reciprocal space
    pub brillouin_zone: Polyhedron,
    /// Reciprocal high symmetry points and paths
    pub reciprocal_high_symmetry: HighSymmetryData,
    // Real-space high symmetry points and paths are omitted
}

impl Lattice2D {
    /// Construct a new 2D lattice from real‐space basis.
    pub fn from_matrix(direct: Matrix3<f64>) -> Result<Self, Error> {
        // Construct the base matrix (propagate errors)
        let direct_basis = BaseMatrix::from_matrix_2d(direct, Space::Real)?;
        let reciprocal_basis = BaseMatrix::apply_reciprocal_transformation(&direct_basis)?;
        // The BaseMatrix constructor ensures that these are proper base matrices. No checks necessary from here on out.

        // Compute metric tensors
        let direct_metric = direct_basis.metric();
        let reciprocal_metric = reciprocal_basis.metric();

        // Identify 2D Bravais type
        let direct_bravais = identify_bravais_2d(&direct_metric);
        let reciprocal_bravais = identify_bravais_2d(&reciprocal_metric);

        // Generate symmetry operations
        let direct_symmetry_operations = generate_symmetry_operations_2d(&direct_bravais);
        let reciprocal_symmetry_operations = generate_symmetry_operations_2d(&reciprocal_bravais);

        // Compute Wigner-Seitz cell and Brillouin zone
        let wigner_seitz_cell = compute_wigner_seitz_cell_2d(direct_basis.base_matrix());
        let brillouin_zone = compute_brillouin_zone_2d(reciprocal_basis.base_matrix());

        // Generate high symmetry points
        let reciprocal_high_symmetry = generate_2d_high_symmetry_points(&reciprocal_bravais);

        Ok(Lattice2D {
            direct_basis,
            reciprocal_basis,
            direct_bravais,
            reciprocal_bravais,
            direct_metric,
            reciprocal_metric,
            direct_symmetry_operations,
            reciprocal_symmetry_operations,
            wigner_seitz_cell,
            brillouin_zone,
            reciprocal_high_symmetry,
        })
    }

    /// Convert fractional (u,v,w) coords → cartesian.
    pub fn fractional_to_cartesian(
        &self,
        fractional_vector: Vector3<f64>,
        space: Space,
    ) -> Vector3<f64> {
        match space {
            Space::Real => self.direct_basis.base_matrix() * fractional_vector,
            Space::Reciprocal => self.reciprocal_basis.base_matrix() * fractional_vector,
        }
    }

    /// Convert cartesian coords → fractional (u,v,w).
    pub fn cartesian_to_fractional(
        &self,
        cartesian_vector: Vector3<f64>,
        space: Space,
    ) -> Vector3<f64> {
        match space {
            Space::Real => self.direct_basis.inverse() * cartesian_vector,
            Space::Reciprocal => self.reciprocal_basis.inverse() * cartesian_vector,
        }
    }

    /// Get 2D direct lattice parameters: a, b (lengths)
    pub fn direct_lattice_parameters(&self) -> (f64, f64) {
        let a1 = self.direct_metric[(0, 0)].sqrt();
        let a2 = self.direct_metric[(1, 1)].sqrt();
        (a1, a2)
    }

    /// Get 2D reciprocal lattice parameters: a, b (lengths)
    pub fn reciprocal_lattice_parameters(&self) -> (f64, f64) {
        let b1 = self.reciprocal_metric[(0, 0)].sqrt();
        let b2 = self.reciprocal_metric[(1, 1)].sqrt();
        (b1, b2)
    }

    /// Get 2D real space lattice angle: γ (in radians)
    pub fn direct_lattice_angle(&self) -> f64 {
        let (a1, a2) = self.direct_lattice_parameters();
        (self.direct_metric[(0, 1)] / (a1 * a2)).acos()
    }

    /// Get 2D reciprocal space lattice angle: γ* (in radians)
    pub fn reciprocal_lattice_angle(&self) -> f64 {
        let (b1, b2) = self.reciprocal_lattice_parameters();
        (self.reciprocal_metric[(0, 1)] / (b1 * b2)).acos()
    }

    /// Get the real-space base vectors as separate Vector3 objects
    pub fn direct_base_vectors(&self) -> [Vector3<f64>; 3] {
        self.direct_basis.base_vectors()
    }

    /// Get the reciprocal-space base vectors as separate Vector3 objects
    pub fn reciprocal_base_vectors(&self) -> [Vector3<f64>; 3] {
        self.reciprocal_basis.base_vectors()
    }

    /// Check if a point is in the first Brillouin zone
    pub fn in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool {
        self.brillouin_zone.contains_2d(k_point)
    }

    /// Initial centering into the primitive reciprocal cell (speeds up convergence for k-point-checks).
    fn center_to_primitive_cell(&self, k: Vector3<f64>) -> Vector3<f64> {
        // Convention: reciprocal basis includes 2π in columns; fractional coords are B^{-1} * k.
        let reciprocal_inverse = self.reciprocal_basis.inverse();
        let mut k_fractional = reciprocal_inverse * k;
        // Wrap to [-0.5, 0.5) for x,y; z left as is.
        for i in 0..2 {
            k_fractional[i] -= k_fractional[i].round();
        }
        self.reciprocal_basis.base_matrix() * k_fractional
    }

    /// Reduce a k-point to the first Brillouin zone
    pub fn reduce_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64> {
        // Build half-spaces directly from the precomputed BZ polygon (no BZ recomputation).
        let hs = bz_halfspaces_from_poly(&self.brillouin_zone, LATTICE_TOLERANCE);

        // First center to the primitive reciprocal cell to keep steps small.
        let mut k_curr = self.center_to_primitive_cell(k_point);

        // Iterate until all inequalities k·n <= h hold. Each correction reduces ||k||^2.
        let max_iter = 64; // very conservative for 2D
        for _ in 0..max_iter {
            let mut moved = false;
            for h in &hs {
                let s = k_curr.dot(&h.normal);
                if s > h.h + LATTICE_TOLERANCE {
                    // Outside across this face: fold back by G.
                    k_curr -= h.g;
                    moved = true;
                }
            }
            if !moved {
                // Snap tiny deviations to boundary for determinism and to avoid straddling.
                for h in &hs {
                    let s = k_curr.dot(&h.normal);
                    if (s - h.h).abs() <= 10.0 * LATTICE_TOLERANCE {
                        k_curr -= (s - h.h) * h.normal;
                    }
                }
                return k_curr;
            }
        }
        // Fallback (shouldn't happen): return current best.
        k_curr
    }

    /// Get reciprocal-space high symmetry points in Cartesian coordinates
    pub fn reciprocal_high_symmetry_points_cartesian(&self) -> Vec<(String, Vector3<f64>)> {
        self.reciprocal_high_symmetry
            .points
            .iter()
            .map(|(label, point)| {
                let k_cart = self.reciprocal_basis.base_matrix() * point.position;
                (label.as_str().to_string(), k_cart)
            })
            .collect()
    }

    /// Generate k-points along the standard reciprocal high symmetry path
    pub fn generate_k_path(&self, n_points_per_segment: usize) -> Vec<Vector3<f64>> {
        use crate::symmetries::high_symmetry_points::interpolate_path;

        let path_points = self.reciprocal_high_symmetry.get_standard_path_points();
        let path_points_owned: Vec<_> = path_points.into_iter().cloned().collect();
        let k_frac = interpolate_path(&path_points_owned, n_points_per_segment);

        // Convert to Cartesian coordinates
        k_frac
            .into_iter()
            .map(|k| self.reciprocal_basis.base_matrix() * k)
            .collect()
    }

    /// Core worker: enumerate all lattice sites n₁·a₁ + n₂·a₂ that lie
    /// inside an axis-aligned rectangle of size (width × height)
    /// centred at the origin.
    ///
    /// * `a1`, `a2` – primitive basis vectors (Cartesian, 3-component)
    /// * `width`, `height` – rectangle side lengths (≥ 0)
    /// * `tol` – numerical tolerance
    ///
    /// Returned `Vec` contains each point exactly once.
    fn lattice_points_in_rectangle(
        a1: Vector3<f64>,
        a2: Vector3<f64>,
        width: f64,
        height: f64,
    ) -> Vec<Vector3<f64>> {
        // Empty rectangle ⇒ empty result
        if width <= 0.0 || height <= 0.0 {
            return Vec::new();
        }

        // Half-sizes, slightly expanded by tolerance
        let half_w = 0.5 * width + LATTICE_TOLERANCE;
        let half_h = 0.5 * height + LATTICE_TOLERANCE;

        // Points inside the rectangle are certainly inside the circumscribed
        // circle of radius r_max.
        let r_max = (half_w * half_w + half_h * half_h).sqrt();

        // Integer bounds for enumeration.  +1 guarantees coverage even when
        // r_max is an exact multiple of |a_i|.
        let n1_max = (r_max / a1.norm()).ceil() as i32 + 1;
        let n2_max = (r_max / a2.norm()).ceil() as i32 + 1;

        let mut pts = Vec::new();
        for n1 in -n1_max..=n1_max {
            for n2 in -n2_max..=n2_max {
                let r = a1 * n1 as f64 + a2 * n2 as f64;

                // Fast axis-aligned check (ignore z-component)
                if r.x.abs() <= half_w && r.y.abs() <= half_h {
                    pts.push(r);
                }
            }
        }
        pts
    }

    /// Public helper: direct-space lattice points in (width × height)
    /// window centred at the origin.
    pub fn get_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<Vector3<f64>> {
        let [a1, a2, _] = self.direct_base_vectors();
        Self::lattice_points_in_rectangle(a1, a2, width, height)
    }

    /// Public helper: reciprocal-space lattice points in (width × height)
    /// window centred at the origin.
    pub fn get_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<Vector3<f64>> {
        let [b1, b2, _] = self.reciprocal_base_vectors();
        Self::lattice_points_in_rectangle(b1, b2, width, height)
    }

    /// Get the real-space bravais lattice type
    pub fn direct_bravais_type(&self) -> Bravais2D {
        self.direct_bravais
    }

    /// Get the reciprocal-space bravais lattice type
    pub fn reciprocal_bravais_type(&self) -> Bravais2D {
        self.reciprocal_bravais
    }

    /// Get the real-space metric tensor
    pub fn direct_metric_tensor(&self) -> &Matrix3<f64> {
        &self.direct_metric
    }

    /// Get the reciprocal-space metric tensor
    pub fn reciprocal_metric_tensor(&self) -> &Matrix3<f64> {
        &self.reciprocal_metric
    }

    /// Get all real-space symmetry operations
    pub fn direct_symmetry_operations(&self) -> &Vec<SymmetryOperation> {
        &self.direct_symmetry_operations
    }

    /// Get all reciprocal-space symmetry operations
    pub fn reciprocal_symmetry_operations(&self) -> &Vec<SymmetryOperation> {
        &self.reciprocal_symmetry_operations
    }

    /// Get the Wigner-Seitz cell
    pub fn wigner_seitz_cell(&self) -> &Polyhedron {
        &self.wigner_seitz_cell
    }

    /// Get the first Brillouin zone
    pub fn brillouin_zone(&self) -> &Polyhedron {
        &self.brillouin_zone
    }

    /// Get the reciprocal-space high symmetry data
    pub fn reciprocal_high_symmetry_data(&self) -> &HighSymmetryData {
        &self.reciprocal_high_symmetry
    }

    /// Get the real-space base matrix
    pub fn direct_base_matrix(&self) -> &BaseMatrix {
        &self.direct_basis
    }

    /// Get the reciprocal-space base matrix
    pub fn reciprocal_base_matrix(&self) -> &BaseMatrix {
        &self.reciprocal_basis
    }

    /// Get direct lattice basis vectors
    pub fn direct_matrix(&self) -> &Matrix3<f64> {
        &self.direct_basis.base_matrix()
    }

    /// Get reciprocal lattice basis vectors
    pub fn reciprocal_matrix(&self) -> &Matrix3<f64> {
        &self.reciprocal_basis.base_matrix()
    }
}
