use nalgebra::{Matrix3, Vector3};

use crate::{
    interfaces::space::{Direct, Reciprocal},
    lattice::{Bravais2D, Polyhedron, base_matrix::BaseMatrix},
    symmetries::HighSymmetryData,
};

pub trait LatticeLike2D {
    fn direct_basis(&self) -> &BaseMatrix<Direct>;
    fn reciprocal_basis(&self) -> &BaseMatrix<Reciprocal>;
    fn direct_bravais(&self) -> Bravais2D;
    fn reciprocal_bravais(&self) -> Bravais2D;
    fn direct_metric(&self) -> &Matrix3<f64>;
    fn reciprocal_metric(&self) -> &Matrix3<f64>;
    fn wigner_seitz(&self) -> &Polyhedron;
    fn brillouin_zone(&self) -> &Polyhedron;
    fn direct_high_symmetry(&self) -> &HighSymmetryData;
    fn reciprocal_high_symmetry(&self) -> &HighSymmetryData;
    fn direct_lattice_parameters(&self) -> (f64, f64) {
        let a1 = self.direct_metric()[(0, 0)].sqrt();
        let a2 = self.direct_metric()[(1, 1)].sqrt();
        (a1, a2)
    }
    fn reciprocal_lattice_parameters(&self) -> (f64, f64) {
        let b1 = self.reciprocal_metric()[(0, 0)].sqrt();
        let b2 = self.reciprocal_metric()[(1, 1)].sqrt();
        (b1, b2)
    }
    fn direct_lattice_angle(&self) -> f64 {
        let (a1, a2) = self.direct_lattice_parameters();
        (self.direct_metric()[(0, 1)] / (a1 * a2)).acos()
    }
    fn reciprocal_lattice_angle(&self) -> f64 {
        let (a1, a2) = self.reciprocal_lattice_parameters();
        (self.reciprocal_metric()[(0, 1)] / (a1 * a2)).acos()
    }
    fn compute_direct_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<Vector3<f64>>;
    fn compute_reciprocal_lattice_points_in_rectangle(
        &self,
        width: f64,
        height: f64,
    ) -> Vec<Vector3<f64>>;
    fn generate_high_symmetry_k_path(&self, n_points_per_segment: u16) -> Vec<Vector3<f64>>;
    fn is_point_in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool;
    fn is_point_in_wigner_seitz_cell(&self, r_point: Vector3<f64>) -> bool;
    fn reduce_point_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64>;
}
