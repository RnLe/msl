use nalgebra::Vector3;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::lattice::bravais_types::{Bravais2D, Bravais3D};

/// Standard labels for high symmetry points in the Brillouin zone
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymmetryPointLabel {
    // General points
    Gamma,  // Γ: Center of BZ (0,0,0)
    
    // 2D specific points
    M,      // Middle of edge
    K,      // Corner of hexagonal BZ
    X,      // Face center in square/rectangular
    Y,      // Face center in rectangular
    
    // 3D specific points
    R,      // Corner of cubic BZ
    L,      // Center of hexagonal face
    W,      // Corner point
    U,      // Mid-edge point
    S,      // Face center
    T,      // Edge center
    Z,      // Face center
    A,      // Face center
    H,      // Corner of hexagonal BZ
    P,      // Corner point
    N,      // Face center
    
    // Custom point for special cases
    Custom(String),
}

impl SymmetryPointLabel {
    /// Get the conventional string representation
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gamma => "Γ",
            Self::M => "M",
            Self::K => "K",
            Self::X => "X",
            Self::Y => "Y",
            Self::R => "R",
            Self::L => "L",
            Self::W => "W",
            Self::U => "U",
            Self::S => "S",
            Self::T => "T",
            Self::Z => "Z",
            Self::A => "A",
            Self::H => "H",
            Self::P => "P",
            Self::N => "N",
            Self::Custom(s) => s,
        }
    }
}

/// A high symmetry point in the Brillouin zone
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighSymmetryPoint {
    /// Label for the point (Γ, K, M, etc.)
    pub label: SymmetryPointLabel,
    /// Position in reciprocal space (fractional coordinates of reciprocal lattice)
    pub position: Vector3<f64>,
    /// Description of the point's location
    pub description: String,
}

impl HighSymmetryPoint {
    /// Create a new high symmetry point
    pub fn new(label: SymmetryPointLabel, position: Vector3<f64>, description: impl Into<String>) -> Self {
        Self {
            label,
            position,
            description: description.into(),
        }
    }
}

/// A path through high symmetry points for band structure calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighSymmetryPath {
    /// Ordered list of point labels defining the path
    pub points: Vec<SymmetryPointLabel>,
    /// Optional number of k-points between each segment
    pub n_points: Option<usize>,
}

impl HighSymmetryPath {
    /// Create a new path from a list of points
    pub fn new(points: Vec<SymmetryPointLabel>) -> Self {
        Self {
            points,
            n_points: None,
        }
    }
    
    /// Set the number of k-points for interpolation
    pub fn with_n_points(mut self, n: usize) -> Self {
        self.n_points = Some(n);
        self
    }
}

/// Collection of high symmetry points and paths for a lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HighSymmetryData {
    /// Map of high symmetry points
    pub points: HashMap<SymmetryPointLabel, HighSymmetryPoint>,
    /// Standard path for band structure
    pub standard_path: HighSymmetryPath,
    /// Alternative paths that might be useful
    pub alternative_paths: Vec<(String, HighSymmetryPath)>,
}

impl HighSymmetryData {
    /// Create new high symmetry data
    pub fn new() -> Self {
        Self {
            points: HashMap::new(),
            standard_path: HighSymmetryPath::new(vec![]),
            alternative_paths: vec![],
        }
    }
    
    /// Add a high symmetry point
    pub fn add_point(&mut self, point: HighSymmetryPoint) {
        self.points.insert(point.label.clone(), point);
    }
    
    /// Set the standard path
    pub fn set_standard_path(&mut self, path: HighSymmetryPath) {
        self.standard_path = path;
    }
    
    /// Add an alternative path
    pub fn add_alternative_path(&mut self, name: impl Into<String>, path: HighSymmetryPath) {
        self.alternative_paths.push((name.into(), path));
    }
    
    /// Get a point by label
    pub fn get_point(&self, label: &SymmetryPointLabel) -> Option<&HighSymmetryPoint> {
        self.points.get(label)
    }
    
    /// Get all points along the standard path
    pub fn get_standard_path_points(&self) -> Vec<&HighSymmetryPoint> {
        self.standard_path.points.iter()
            .filter_map(|label| self.points.get(label))
            .collect()
    }
}

/// Generate high symmetry points for 2D Bravais lattices
pub fn generate_2d_high_symmetry_points(bravais: &Bravais2D) -> HighSymmetryData {
    let mut data = HighSymmetryData::new();
    
    // Gamma point is common to all lattices
    data.add_point(HighSymmetryPoint::new(
        SymmetryPointLabel::Gamma,
        Vector3::new(0.0, 0.0, 0.0),
        "Center of Brillouin zone",
    ));
    
    match bravais {
        Bravais2D::Square => {
            // Square lattice high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Center of square edge",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "Corner of square BZ",
            ));
            
            // Standard path: Γ → X → M → Γ
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Gamma,
            ]));
        },
        
        Bravais2D::Hexagonal => {
            // Hexagonal lattice high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.0, 0.0),
                "Middle of hexagon edge",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::K,
                Vector3::new(1.0/3.0, 1.0/3.0, 0.0),
                "Corner of hexagonal BZ",
            ));
            
            // Standard path: Γ → M → K → Γ
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::M,
                SymmetryPointLabel::K,
                SymmetryPointLabel::Gamma,
            ]));
        },
        
        Bravais2D::Rectangular => {
            // Rectangular lattice high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Center of edge along a",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::Y,
                Vector3::new(0.0, 0.5, 0.0),
                "Center of edge along b",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "Corner of rectangular BZ",
            ));
            
            // Standard path: Γ → X → M → Y → Γ
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Y,
                SymmetryPointLabel::Gamma,
            ]));
        },
        
        Bravais2D::CenteredRectangular => {
            // Centered rectangular has a more complex BZ
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Edge center",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::Y,
                Vector3::new(0.0, 0.5, 0.0),
                "Edge center",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "Corner of BZ",
            ));
            
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Y,
                SymmetryPointLabel::Gamma,
            ]));
        },
        
        Bravais2D::Oblique => {
            // Oblique lattice - most general case
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Edge midpoint",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::Y,
                Vector3::new(0.0, 0.5, 0.0),
                "Edge midpoint",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "BZ corner",
            ));
            
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Y,
                SymmetryPointLabel::Gamma,
            ]));
        },
    }
    
    data
}

/// Generate high symmetry points for 3D Bravais lattices
pub fn generate_3d_high_symmetry_points(bravais: &Bravais3D) -> HighSymmetryData {
    let mut data = HighSymmetryData::new();
    
    // Gamma point is common to all lattices
    data.add_point(HighSymmetryPoint::new(
        SymmetryPointLabel::Gamma,
        Vector3::new(0.0, 0.0, 0.0),
        "Center of Brillouin zone",
    ));
    
    use Bravais3D::*;
    match bravais {
        Cubic(_) => {
            // Simple cubic high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Face center",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "Edge center",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::R,
                Vector3::new(0.5, 0.5, 0.5),
                "Corner of cubic BZ",
            ));
            
            // Standard path for cubic: Γ → X → M → Γ → R → X | M → R
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::R,
                SymmetryPointLabel::X,
            ]));
            
            data.add_alternative_path(
                "Extended",
                HighSymmetryPath::new(vec![
                    SymmetryPointLabel::M,
                    SymmetryPointLabel::R,
                ]),
            );
        },
        
        Hexagonal(_) => {
            // Hexagonal 3D high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::A,
                Vector3::new(0.0, 0.0, 0.5),
                "Center of hexagonal face",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::H,
                Vector3::new(1.0/3.0, 1.0/3.0, 0.5),
                "Corner point",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::K,
                Vector3::new(1.0/3.0, 1.0/3.0, 0.0),
                "Corner of hexagonal face",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::L,
                Vector3::new(0.5, 0.0, 0.5),
                "Edge midpoint",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.0, 0.0),
                "Edge midpoint",
            ));
            
            // Standard hexagonal path: Γ → M → K → Γ → A → L → H → A | L → M | K → H
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::M,
                SymmetryPointLabel::K,
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::A,
                SymmetryPointLabel::L,
                SymmetryPointLabel::H,
                SymmetryPointLabel::A,
            ]));
        },
        
        Tetragonal(_) => {
            // Tetragonal high symmetry points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Face center",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::M,
                Vector3::new(0.5, 0.5, 0.0),
                "Edge center of square face",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::Z,
                Vector3::new(0.0, 0.0, 0.5),
                "Face center along c",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::R,
                Vector3::new(0.5, 0.5, 0.5),
                "Corner point",
            ));
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::A,
                Vector3::new(0.5, 0.0, 0.5),
                "Edge center",
            ));
            
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::M,
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::Z,
                SymmetryPointLabel::R,
                SymmetryPointLabel::A,
                SymmetryPointLabel::Z,
            ]));
        },
        
        // Add more crystal systems as needed
        _ => {
            // For other systems, provide basic points
            // This is a placeholder - each system needs its specific points
            data.add_point(HighSymmetryPoint::new(
                SymmetryPointLabel::X,
                Vector3::new(0.5, 0.0, 0.0),
                "Generic point",
            ));
            
            data.set_standard_path(HighSymmetryPath::new(vec![
                SymmetryPointLabel::Gamma,
                SymmetryPointLabel::X,
                SymmetryPointLabel::Gamma,
            ]));
        },
    }
    
    data
}

/// Interpolate k-points along a high symmetry path
pub fn interpolate_path(
    points: &[HighSymmetryPoint],
    n_points_per_segment: usize,
) -> Vec<Vector3<f64>> {
    let mut k_points = Vec::new();
    
    for i in 0..points.len() - 1 {
        let start = &points[i].position;
        let end = &points[i + 1].position;
        
        for j in 0..n_points_per_segment {
            let t = j as f64 / n_points_per_segment as f64;
            let k = start + t * (end - start);
            k_points.push(k);
        }
    }
    
    // Add the final point
    if let Some(last) = points.last() {
        k_points.push(last.position);
    }
    
    k_points
}
