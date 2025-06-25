use nalgebra::{Matrix2, Matrix3, Vector3};
use serde::{Serialize, Deserialize};
use crate::lattice::lattice2d::Lattice2D;
use crate::lattice::lattice_bravais_types::Bravais2D;
use crate::lattice::lattice_polyhedron::Polyhedron;
use crate::symmetries::high_symmetry_points::HighSymmetryData;
use crate::symmetries::symmetry_operations::SymOp;

/// Transformation type for the second lattice relative to the first
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MoireTransformation {
    /// Simple rotation and uniform scaling: s * R(θ)
    RotationScale { angle: f64, scale: f64 },
    
    /// Anisotropic scaling: diag(s_x, s_y)
    AnisotropicScale { scale_x: f64, scale_y: f64 },
    
    /// Shear transformation
    Shear { shear_x: f64, shear_y: f64 },
    
    /// General 2x2 matrix transformation
    General(Matrix2<f64>),
}

impl MoireTransformation {
    /// Convert to 2x2 matrix form
    pub fn to_matrix(&self) -> Matrix2<f64> {
        match self {
            MoireTransformation::RotationScale { angle, scale } => {
                let c = angle.cos();
                let s = angle.sin();
                *scale * Matrix2::new(c, -s, s, c)
            }
            MoireTransformation::AnisotropicScale { scale_x, scale_y } => {
                Matrix2::new(*scale_x, 0.0, 0.0, *scale_y)
            }
            MoireTransformation::Shear { shear_x, shear_y } => {
                Matrix2::new(1.0, *shear_x, *shear_y, 1.0)
            }
            MoireTransformation::General(mat) => *mat,
        }
    }
    
    /// Convert to 3x3 matrix form (embedding 2D transformation in 3D)
    pub fn to_matrix3(&self) -> Matrix3<f64> {
        let mat2 = self.to_matrix();
        let mut mat3 = Matrix3::identity();
        mat3[(0, 0)] = mat2[(0, 0)];
        mat3[(0, 1)] = mat2[(0, 1)];
        mat3[(1, 0)] = mat2[(1, 0)];
        mat3[(1, 1)] = mat2[(1, 1)];
        mat3
    }
}

/// A 2D moiré lattice formed by two overlapping 2D lattices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moire2D {
    // === Inherited fields from Lattice2D ===
    /// Real-space basis vectors of the moiré lattice
    pub direct: Matrix3<f64>,
    /// Reciprocal-space basis vectors of the moiré lattice
    pub reciprocal: Matrix3<f64>,
    /// Bravais type of the moiré lattice
    pub bravais: Bravais2D,
    /// Unit cell area of the moiré lattice
    pub cell_area: f64,
    /// Metric tensor of the moiré lattice
    pub metric: Matrix3<f64>,
    /// Tolerance for float comparisons
    pub tol: f64,
    /// Symmetry operations of the moiré lattice
    pub sym_ops: Vec<SymOp>,
    /// Wigner-Seitz cell of the moiré lattice
    pub wigner_seitz_cell: Polyhedron,
    /// Brillouin zone of the moiré lattice
    pub brillouin_zone: Polyhedron,
    /// High symmetry points of the moiré lattice
    pub high_symmetry: HighSymmetryData,
    
    // === Moiré-specific fields ===
    /// First constituent lattice
    pub lattice_1: Lattice2D,
    /// Second constituent lattice
    pub lattice_2: Lattice2D,
    /// Transformation applied to create lattice_2 from lattice_1
    pub transformation: MoireTransformation,
    /// Twist angle between the two lattices (in radians)
    pub twist_angle: f64,
    /// Whether the moiré lattice is commensurate
    pub is_commensurate: bool,
    /// Coincidence indices (m1, m2, n1, n2) if commensurate
    pub coincidence_indices: Option<(i32, i32, i32, i32)>,
}

impl Moire2D {
    /// Get the moiré lattice as a regular Lattice2D
    pub fn as_lattice2d(&self) -> Lattice2D {
        Lattice2D {
            direct: self.direct,
            reciprocal: self.reciprocal,
            bravais: self.bravais,
            cell_area: self.cell_area,
            metric: self.metric,
            tol: self.tol,
            sym_ops: self.sym_ops.clone(),
            wigner_seitz_cell: self.wigner_seitz_cell.clone(),
            brillouin_zone: self.brillouin_zone.clone(),
            high_symmetry: self.high_symmetry.clone(),
        }
    }
    
    /// Get the primitive vectors of the moiré lattice
    pub fn primitive_vectors(&self) -> (Vector3<f64>, Vector3<f64>) {
        (self.direct.column(0).into(), self.direct.column(1).into())
    }
    
    /// Get the moiré periodicity (ratio of moiré to original lattice constant)
    pub fn moire_period_ratio(&self) -> f64 {
        (self.cell_area / self.lattice_1.cell_area).sqrt()
    }
    
    /// Check if a given point belongs to lattice 1
    pub fn is_lattice1_point(&self, point: Vector3<f64>) -> bool {
        let frac = self.lattice_1.cart_to_frac(point);
        (frac[0] - frac[0].round()).abs() < self.tol &&
        (frac[1] - frac[1].round()).abs() < self.tol
    }
    
    /// Check if a given point belongs to lattice 2
    pub fn is_lattice2_point(&self, point: Vector3<f64>) -> bool {
        let frac = self.lattice_2.cart_to_frac(point);
        (frac[0] - frac[0].round()).abs() < self.tol &&
        (frac[1] - frac[1].round()).abs() < self.tol
    }
    
    /// Get stacking type at a given position (AA, AB, BA, or neither)
    pub fn get_stacking_at(&self, point: Vector3<f64>) -> Option<String> {
        let on_l1 = self.is_lattice1_point(point);
        let on_l2 = self.is_lattice2_point(point);
        
        match (on_l1, on_l2) {
            (true, true) => Some("AA".to_string()),
            (true, false) => Some("A".to_string()),
            (false, true) => Some("B".to_string()),
            (false, false) => None,
        }
    }
}
