use nalgebra::{Matrix3, Vector3};
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

use crate::lattice::lattice_bravais_types::{Bravais3D, identify_bravais_3d};
use crate::lattice::lattice_polyhedron::Polyhedron;
use crate::lattice::lattice_voronoi_cells::{compute_wigner_seitz_cell_3d, compute_brillouin_zone_3d};
use crate::symmetries::high_symmetry_points::{HighSymmetryData, generate_3d_high_symmetry_points};
use crate::symmetries::symmetry_point_groups::generate_symmetry_operations_3d;
use crate::symmetries::symmetry_operations::SymOp;

/// A 3D Bravais lattice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lattice3D {
    /// Real‐space basis vectors (columns).
    pub direct: Matrix3<f64>,
    /// Reciprocal‐space basis vectors (columns).
    pub reciprocal: Matrix3<f64>,
    /// 3D Bravais lattice type.
    pub bravais: Bravais3D,
    /// Unit cell volume = det(direct).
    pub cell_volume: f64,
    /// Metric tensor G = A^T * A.
    pub metric: Matrix3<f64>,
    /// Tolerance for float comparisons.
    pub tol: f64,
    /// Symmetry operations based on bravais type.
    pub sym_ops: Vec<SymOp>,
    /// Wigner-Seitz cell in direct space
    pub wigner_seitz_cell: Polyhedron,
    /// First Brillouin zone in reciprocal space
    pub brillouin_zone: Polyhedron,
    /// High symmetry points and paths
    pub high_symmetry: HighSymmetryData,
}

impl Lattice3D {
    /// Construct a new 3D lattice from real‐space basis.
    pub fn new(direct: Matrix3<f64>, tol: f64) -> Self {
        // 1) Compute metric tensor and volume
        let metric = direct.transpose() * direct;
        let cell_volume = direct.determinant();
        
        // 2) Compute reciprocal basis (2π‐convention)
        let reciprocal = {
            let inv = direct.try_inverse()
                .expect("Direct basis must be invertible for a true 3D lattice");
            (2.0 * PI) * inv.transpose()
        };
        
        // 3) Identify 3D Bravais type
        let bravais = identify_bravais_3d(&metric, tol);
        
        // 4) Generate symmetry operations
        let sym_ops = generate_symmetry_operations_3d(&bravais);
        
        // 5) Compute Wigner-Seitz cell and Brillouin zone
        let wigner_seitz_cell = compute_wigner_seitz_cell_3d(&direct, tol);
        let brillouin_zone = compute_brillouin_zone_3d(&reciprocal, tol);
        
        // 6) Generate high symmetry points
        let high_symmetry = generate_3d_high_symmetry_points(&bravais);
        
        Lattice3D {
            direct,
            reciprocal,
            bravais,
            cell_volume,
            metric,
            tol,
            sym_ops,
            wigner_seitz_cell,
            brillouin_zone,
            high_symmetry,
        }
    }

    /// Convert fractional (u,v,w) coords → cartesian.
    pub fn frac_to_cart(&self, v_frac: Vector3<f64>) -> Vector3<f64> {
        self.direct * v_frac
    }

    /// Convert cartesian coords → fractional (u,v,w).
    pub fn cart_to_frac(&self, v_cart: Vector3<f64>) -> Vector3<f64> {
        self.direct.try_inverse()
            .expect("Lattice basis is singular")
            * v_cart
    }

    /// Get lattice parameters: a, b, c (lengths)
    pub fn lattice_parameters(&self) -> (f64, f64, f64) {
        let a = self.metric[(0, 0)].sqrt();
        let b = self.metric[(1, 1)].sqrt();
        let c = self.metric[(2, 2)].sqrt();
        (a, b, c)
    }

    /// Get lattice angles: α, β, γ (in radians)
    pub fn lattice_angles(&self) -> (f64, f64, f64) {
        let (a, b, c) = self.lattice_parameters();
        
        // α = angle between b and c vectors
        let alpha = (self.metric[(1, 2)] / (b * c)).acos();
        // β = angle between a and c vectors  
        let beta = (self.metric[(0, 2)] / (a * c)).acos();
        // γ = angle between a and b vectors
        let gamma = (self.metric[(0, 1)] / (a * b)).acos();
        
        (alpha, beta, gamma)
    }

    /// Project to 2D lattice by removing third dimension
    pub fn to_2d(&self) -> crate::lattice::lattice2d::Lattice2D {
        // Extract the first two columns as the 2D basis
        let mut direct_2d = Matrix3::zeros();
        direct_2d.set_column(0, &self.direct.column(0));
        direct_2d.set_column(1, &self.direct.column(1));
        // Set a default z-direction
        direct_2d[(2, 2)] = 1.0;
        
        crate::lattice::lattice2d::Lattice2D::new(direct_2d, self.tol)
    }
    
    /// Get the primitive vectors as separate Vector3 objects
    pub fn primitive_vectors(&self) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        (
            self.direct.column(0).into(),
            self.direct.column(1).into(),
            self.direct.column(2).into(),
        )
    }
    
    /// Check if a point is in the first Brillouin zone
    pub fn in_brillouin_zone(&self, k_point: Vector3<f64>) -> bool {
        self.brillouin_zone.contains_3d(k_point)
    }
    
    /// Reduce a k-point to the first Brillouin zone
    pub fn reduce_to_brillouin_zone(&self, k_point: Vector3<f64>) -> Vector3<f64> {
        let mut k_frac = self.reciprocal.try_inverse()
            .expect("Reciprocal basis is singular") * k_point / (2.0 * PI);
        
        // Reduce to [-0.5, 0.5] in fractional coordinates
        for i in 0..3 {
            k_frac[i] = k_frac[i] - k_frac[i].round();
        }
        
        self.reciprocal * k_frac
    }
    
    /// Get high symmetry points in Cartesian coordinates
    pub fn get_high_symmetry_points_cartesian(&self) -> Vec<(String, Vector3<f64>)> {
        self.high_symmetry.points.iter()
            .map(|(label, point)| {
                let k_cart = self.reciprocal * point.position;
                (label.as_str().to_string(), k_cart)
            })
            .collect()
    }
    
    /// Generate k-points along the standard high symmetry path
    pub fn generate_k_path(&self, n_points_per_segment: usize) -> Vec<Vector3<f64>> {
        use crate::symmetries::high_symmetry_points::interpolate_path;
        
        let path_points = self.high_symmetry.get_standard_path_points();
        let path_points_owned: Vec<_> = path_points.into_iter().cloned().collect();
        let k_frac = interpolate_path(&path_points_owned, n_points_per_segment);
        
        // Convert to Cartesian coordinates
        k_frac.into_iter()
            .map(|k| self.reciprocal * k)
            .collect()
    }

    /// Get the Bravais lattice type
    pub fn bravais_type(&self) -> Bravais3D {
        self.bravais
    }

    /// Get the unit cell volume
    pub fn cell_volume(&self) -> f64 {
        self.cell_volume
    }

    /// Get the metric tensor
    pub fn metric_tensor(&self) -> &Matrix3<f64> {
        &self.metric
    }

    /// Get the tolerance used for floating point comparisons
    pub fn tolerance(&self) -> f64 {
        self.tol
    }

    /// Get all symmetry operations
    pub fn symmetry_operations(&self) -> &Vec<SymOp> {
        &self.sym_ops
    }

    /// Get the Wigner-Seitz cell
    pub fn wigner_seitz_cell(&self) -> &Polyhedron {
        &self.wigner_seitz_cell
    }

    /// Get the first Brillouin zone
    pub fn brillouin_zone(&self) -> &Polyhedron {
        &self.brillouin_zone
    }

    /// Get the high symmetry data
    pub fn high_symmetry_data(&self) -> &HighSymmetryData {
        &self.high_symmetry
    }

    /// Get direct lattice basis vectors
    pub fn direct_basis(&self) -> &Matrix3<f64> {
        &self.direct
    }

    /// Get reciprocal lattice basis vectors
    pub fn reciprocal_basis(&self) -> &Matrix3<f64> {
        &self.reciprocal
    }
}
