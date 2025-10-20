use anyhow::Error;
use nalgebra::{Matrix3, Vector3};

use crate::config::LATTICE_TOLERANCE;
use crate::interfaces::space::{Direct, Reciprocal};
use crate::lattice::base_matrix::BaseMatrix;
use crate::lattice::lattice_like_2d::LatticeLike2D;
use crate::lattice::lattice_points_in_rectangle;
use crate::lattice::lattice_types::{Bravais2D, identify_bravais_2d};
use crate::lattice::polyhedron::{Polyhedron, bz_halfspaces_from_poly};
use crate::lattice::voronoi_cells::{compute_brillouin_zone_2d, compute_wigner_seitz_cell_2d};
use crate::symmetries::high_symmetry_points::{HighSymmetryData, generate_2d_high_symmetry_points};
use crate::symmetries::symmetry_operations::SymmetryOperation;
use crate::symmetries::symmetry_point_groups::generate_symmetry_operations_2d;

#[derive(Debug, Clone)]
struct LatticeData2D<S> {
    /// Real- or reciprocal-space basis vectors (columns)
    basis: BaseMatrix<S>,
    /// 2D bravais lattice type
    bravais: Bravais2D,
    /// Metric tensor G = A^T * A; where A is the base matrix.
    metric: Matrix3<f64>,
    /// Symmetry operations based on the bravais type.
    /// TODO: Move this to Bravais2D?
    symmetry_operations: Vec<SymmetryOperation>,
    /// Wigner-Seitz cell or Brillouin zone
    voronoi_cell: Polyhedron,
    /// High symmetry points and paths
    /// TODO: Also tied to Bravais2D, so move there?
    high_symmetry: HighSymmetryData,
}

impl<S> LatticeData2D<S> {
    pub fn basis(&self) -> &BaseMatrix<S> {
        &self.basis
    }
    pub fn bravais(&self) -> Bravais2D {
        self.bravais
    }
    pub fn metric(&self) -> &Matrix3<f64> {
        &self.metric
    }
    pub fn symmetry_operations(&self) -> &Vec<SymmetryOperation> {
        &self.symmetry_operations
    }
    pub fn voronoi_cell(&self) -> &Polyhedron {
        &self.voronoi_cell
    }
    pub fn high_symmetry(&self) -> &HighSymmetryData {
        &self.high_symmetry
    }
}

/// A 2D lattice embedded in 3D space.
///
/// All data lives in dual-space (real-space and reciprocal-space). Thus, all data is masked behind [`LatticeData2D`] structs.
#[derive(Debug, Clone)]
pub struct Lattice2D {
    direct_space: LatticeData2D<Direct>,
    reciprocal_space: LatticeData2D<Reciprocal>,
}

impl Lattice2D {
    /// Construct a new 2D lattice from a real‐space basis.
    pub fn from_direct_matrix(direct: Matrix3<f64>) -> Result<Self, Error> {
        // Construct the base matrix (propagate errors)
        let direct_basis = BaseMatrix::<Direct>::from_matrix_2d(direct)?;
        let reciprocal_basis = direct_basis.to_reciprocal()?;
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
        let direct_high_symmetry = generate_2d_high_symmetry_points(&direct_bravais);
        let reciprocal_high_symmetry = generate_2d_high_symmetry_points(&reciprocal_bravais);

        // Construct the two Lattice Data objects
        let direct_space = LatticeData2D {
            basis: direct_basis,
            bravais: direct_bravais,
            metric: direct_metric,
            symmetry_operations: direct_symmetry_operations,
            voronoi_cell: wigner_seitz_cell,
            high_symmetry: direct_high_symmetry,
        };

        let reciprocal_space = LatticeData2D {
            basis: reciprocal_basis,
            bravais: reciprocal_bravais,
            metric: reciprocal_metric,
            symmetry_operations: reciprocal_symmetry_operations,
            voronoi_cell: brillouin_zone,
            high_symmetry: reciprocal_high_symmetry,
        };

        Ok(Lattice2D {
            direct_space,
            reciprocal_space,
        })
    }

    /// Initial centering into the primitive reciprocal cell (speeds up convergence for k-point-checks).
    fn center_to_primitive_cell(&self, k: Vector3<f64>) -> Vector3<f64> {
        // Convention: reciprocal basis includes 2π in columns; fractional coords are B^{-1} * k.
        let reciprocal_inverse = self.reciprocal_basis().inverse();
        let mut k_fractional = reciprocal_inverse * k;
        // Wrap to [-0.5, 0.5) for x,y; z left as is.
        for i in 0..2 {
            k_fractional[i] -= k_fractional[i].round();
        }
        self.reciprocal_basis().base_matrix() * k_fractional
    }
}

impl LatticeLike2D for Lattice2D {
    fn direct_basis(&self) -> &BaseMatrix<Direct> {
        self.direct_space.basis()
    }
    fn reciprocal_basis(&self) -> &BaseMatrix<Reciprocal> {
        self.reciprocal_space.basis()
    }
    fn direct_bravais(&self) -> Bravais2D {
        self.direct_space.bravais()
    }
    fn reciprocal_bravais(&self) -> Bravais2D {
        self.reciprocal_space.bravais()
    }
    fn direct_metric(&self) -> &Matrix3<f64> {
        self.direct_space.metric()
    }
    fn reciprocal_metric(&self) -> &Matrix3<f64> {
        self.reciprocal_space.metric()
    }
    fn wigner_seitz(&self) -> &Polyhedron {
        self.direct_space.voronoi_cell()
    }
    fn brillouin_zone(&self) -> &Polyhedron {
        self.reciprocal_space.voronoi_cell()
    }
    fn direct_high_symmetry(&self) -> &HighSymmetryData {
        self.direct_space.high_symmetry()
    }
    fn reciprocal_high_symmetry(&self) -> &HighSymmetryData {
        self.reciprocal_space.high_symmetry()
    }
    fn compute_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<nalgebra::Vector3<f64>> {
        let [a1, a2, _] = self.direct_basis().base_vectors();
        lattice_points_in_rectangle(a1, a2, width, height)
    }
    fn compute_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<nalgebra::Vector3<f64>> {
        let [b1, b2, _] = self.reciprocal_basis().base_vectors();
        lattice_points_in_rectangle(b1, b2, width, height)
    }
    fn generate_high_symmetry_k_path(&self, n_points_per_segment: u16) -> Vec<Vector3<f64>> {
        use crate::symmetries::high_symmetry_points::interpolate_path;

        let path_points = self.reciprocal_high_symmetry().get_standard_path_points();
        let path_points_owned: Vec<_> = path_points.into_iter().cloned().collect();
        let k_frac = interpolate_path(&path_points_owned, n_points_per_segment);

        // Convert to Cartesian coordinates
        k_frac
            .into_iter()
            .map(|k| self.reciprocal_basis().base_matrix() * k)
            .collect()
    }
    fn is_point_in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool {
        self.brillouin_zone().contains_2d(k_point)
    }
    fn is_point_in_wigner_seitz_cell(&self, r_point: Vector3<f64>) -> bool {
        self.wigner_seitz().contains_2d(r_point)
    }
    fn reduce_point_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64> {
        // Build half-spaces directly from the precomputed BZ polygon (no BZ recomputation).
        let hs = bz_halfspaces_from_poly(&self.brillouin_zone(), LATTICE_TOLERANCE);

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
}
