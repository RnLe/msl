use nalgebra::{Vector2, Rotation2};
use crate::materials::Material;
use crate::geometries::geometry2d_bounding_box::BoundingBox2D;
use crate::geometries::geometry2d_transform::Transform2D;

/// Base trait for all 2D geometric shapes (without material)
pub trait Geometry2D: std::fmt::Debug {
    /// Check if a point is inside the geometry
    fn contains_point(&self, point: Vector2<f64>) -> bool;
    
    /// Get the bounding box of the geometry
    fn bounding_box(&self) -> BoundingBox2D;
    
    /// Get the center of the geometry
    fn center(&self) -> Vector2<f64>;
    
    /// Apply a transformation to the geometry
    fn transform(&mut self, transform: &Transform2D);
    
    /// Create a deep copy of the geometry
    fn clone_box(&self) -> Box<dyn Geometry2D>;
}

/// A geometric object that combines a shape with a material
#[derive(Debug)]
pub struct GeometricObject2D {
    pub shape: Box<dyn Geometry2D>,
    pub material: Material,
}

impl GeometricObject2D {
    /// Create a new geometric object from a shape and material
    pub fn new<G: Geometry2D + 'static>(shape: G, material: Material) -> Self {
        Self {
            shape: Box::new(shape),
            material,
        }
    }
    
    /// Get a reference to the shape
    pub fn shape(&self) -> &dyn Geometry2D {
        self.shape.as_ref()
    }
    
    /// Get a mutable reference to the shape
    pub fn shape_mut(&mut self) -> &mut dyn Geometry2D {
        self.shape.as_mut()
    }
    
    /// Get a reference to the material
    pub fn material(&self) -> &Material {
        &self.material
    }
    
    /// Set the material
    pub fn set_material(&mut self, material: Material) {
        self.material = material;
    }
    
    /// Apply a transformation to the underlying shape
    pub fn transform(&mut self, transform: &Transform2D) {
        self.shape.transform(transform);
    }
}

/// Circle geometry
#[derive(Debug, Clone)]
pub struct Circle {
    pub center: Vector2<f64>,
    pub radius: f64,
}

impl Circle {
    pub fn new(center: Vector2<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Geometry2D for Circle {
    fn contains_point(&self, point: Vector2<f64>) -> bool {
        (point - self.center).norm() <= self.radius
    }
    
    fn bounding_box(&self) -> BoundingBox2D {
        BoundingBox2D {
            min: self.center - Vector2::new(self.radius, self.radius),
            max: self.center + Vector2::new(self.radius, self.radius),
        }
    }
    
    fn center(&self) -> Vector2<f64> {
        self.center
    }
    
    fn transform(&mut self, transform: &Transform2D) {
        self.center = transform.apply_to_point(self.center);
        self.radius *= transform.scale.x.max(transform.scale.y);
    }
    
    fn clone_box(&self) -> Box<dyn Geometry2D> {
        Box::new(self.clone())
    }
}

/// Rectangle geometry (axis-aligned before transformation)
#[derive(Debug, Clone)]
pub struct Rectangle {
    pub center: Vector2<f64>,
    pub size: Vector2<f64>,
    pub transform: Transform2D,
}

impl Rectangle {
    pub fn new(center: Vector2<f64>, size: Vector2<f64>) -> Self {
        Self {
            center,
            size,
            transform: Transform2D::identity(),
        }
    }
    
    pub fn with_rotation(mut self, angle: f64) -> Self {
        self.transform.rotation = angle;
        self
    }
}

impl Geometry2D for Rectangle {
    fn contains_point(&self, point: Vector2<f64>) -> bool {
        // Handle degenerate case where scale is zero
        if self.transform.scale.x == 0.0 || self.transform.scale.y == 0.0 {
            // Rectangle becomes a line or point - only contains center
            return (point - self.center).norm() < 1e-10;
        }
        
        // Transform point to local coordinates
        let local_point = self.transform.apply_inverse_to_point(point - self.center);
        
        local_point.x.abs() <= self.size.x / 2.0 &&
        local_point.y.abs() <= self.size.y / 2.0
    }
    
    fn bounding_box(&self) -> BoundingBox2D {
        // Calculate corners in local space
        let half_size = self.size / 2.0;
        let corners = [
            Vector2::new(-half_size.x, -half_size.y),
            Vector2::new(half_size.x, -half_size.y),
            Vector2::new(half_size.x, half_size.y),
            Vector2::new(-half_size.x, half_size.y),
        ];
        
        // Transform corners to world space
        let transformed_corners: Vec<_> = corners.iter()
            .map(|&c| self.transform.apply_to_point(c) + self.center)
            .collect();
        
        // Find bounding box
        let min_x = transformed_corners.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
        let max_x = transformed_corners.iter().map(|c| c.x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = transformed_corners.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
        let max_y = transformed_corners.iter().map(|c| c.y).fold(f64::NEG_INFINITY, f64::max);
        
        BoundingBox2D {
            min: Vector2::new(min_x, min_y),
            max: Vector2::new(max_x, max_y),
        }
    }
    
    fn center(&self) -> Vector2<f64> {
        self.center
    }
    
    fn transform(&mut self, transform: &Transform2D) {
        self.center = transform.apply_to_point(self.center);
        self.transform.translation = transform.apply_to_point(self.transform.translation);
        self.transform.rotation += transform.rotation;
        self.transform.scale.component_mul_assign(&transform.scale);
    }
    
    fn clone_box(&self) -> Box<dyn Geometry2D> {
        Box::new(self.clone())
    }
}

/// Ellipse geometry
#[derive(Debug, Clone)]
pub struct Ellipse {
    pub center: Vector2<f64>,
    pub semi_axes: Vector2<f64>, // (a, b) where a is x-axis, b is y-axis
    pub rotation: f64, // rotation angle in radians
}

impl Ellipse {
    pub fn new(center: Vector2<f64>, semi_axes: Vector2<f64>) -> Self {
        Self {
            center,
            semi_axes,
            rotation: 0.0,
        }
    }
    
    pub fn with_rotation(mut self, angle: f64) -> Self {
        self.rotation = angle;
        self
    }
}

impl Geometry2D for Ellipse {
    fn contains_point(&self, point: Vector2<f64>) -> bool {
        // Transform point to local ellipse coordinates
        let translated = point - self.center;
        let rotated = Rotation2::new(-self.rotation) * translated;
        
        // Check ellipse equation: (x/a)² + (y/b)² <= 1
        let x_term = (rotated.x / self.semi_axes.x).powi(2);
        let y_term = (rotated.y / self.semi_axes.y).powi(2);
        
        x_term + y_term <= 1.0
    }
    
    fn bounding_box(&self) -> BoundingBox2D {
        // Calculate bounding box for rotated ellipse
        let cos_r = self.rotation.cos();
        let sin_r = self.rotation.sin();
        
        let x_extent = ((self.semi_axes.x * cos_r).powi(2) + 
                        (self.semi_axes.y * sin_r).powi(2)).sqrt();
        let y_extent = ((self.semi_axes.x * sin_r).powi(2) + 
                        (self.semi_axes.y * cos_r).powi(2)).sqrt();
        
        BoundingBox2D {
            min: self.center - Vector2::new(x_extent, y_extent),
            max: self.center + Vector2::new(x_extent, y_extent),
        }
    }
    
    fn center(&self) -> Vector2<f64> {
        self.center
    }
    
    fn transform(&mut self, transform: &Transform2D) {
        self.center = transform.apply_to_point(self.center);
        self.rotation += transform.rotation;
        self.semi_axes.component_mul_assign(&transform.scale);
    }
    
    fn clone_box(&self) -> Box<dyn Geometry2D> {
        Box::new(self.clone())
    }
}

/// Polygon geometry
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Vector2<f64>>,
    center: Vector2<f64>, // Cached centroid
}

impl Polygon {
    pub fn new(vertices: Vec<Vector2<f64>>) -> Self {
        let center = Self::compute_centroid(&vertices);
        Self { vertices, center }
    }
    
    fn compute_centroid(vertices: &[Vector2<f64>]) -> Vector2<f64> {
        if vertices.is_empty() {
            return Vector2::zeros();
        }
        
        let sum: Vector2<f64> = vertices.iter().sum();
        sum / vertices.len() as f64
    }
    
    /// Create a regular polygon
    pub fn regular(center: Vector2<f64>, radius: f64, sides: usize) -> Self {
        let mut vertices = Vec::with_capacity(sides);
        let angle_step = 2.0 * std::f64::consts::PI / sides as f64;
        
        for i in 0..sides {
            let angle = i as f64 * angle_step;
            vertices.push(center + Vector2::new(
                radius * angle.cos(),
                radius * angle.sin(),
            ));
        }
        
        Self { vertices, center }
    }
}

impl Geometry2D for Polygon {
    fn contains_point(&self, point: Vector2<f64>) -> bool {
        // Use ray casting algorithm
        let mut inside = false;
        let n = self.vertices.len();
        
        for i in 0..n {
            let v1 = self.vertices[i];
            let v2 = self.vertices[(i + 1) % n];
            
            if ((v1.y > point.y) != (v2.y > point.y)) &&
               (point.x < (v2.x - v1.x) * (point.y - v1.y) / (v2.y - v1.y) + v1.x) {
                inside = !inside;
            }
        }
        
        inside
    }
    
    fn bounding_box(&self) -> BoundingBox2D {
        if self.vertices.is_empty() {
            return BoundingBox2D::new(Vector2::zeros(), Vector2::zeros());
        }
        
        let min_x = self.vertices.iter().map(|v| v.x).fold(f64::INFINITY, f64::min);
        let max_x = self.vertices.iter().map(|v| v.x).fold(f64::NEG_INFINITY, f64::max);
        let min_y = self.vertices.iter().map(|v| v.y).fold(f64::INFINITY, f64::min);
        let max_y = self.vertices.iter().map(|v| v.y).fold(f64::NEG_INFINITY, f64::max);
        
        BoundingBox2D {
            min: Vector2::new(min_x, min_y),
            max: Vector2::new(max_x, max_y),
        }
    }
    
    fn center(&self) -> Vector2<f64> {
        self.center
    }
    
    fn transform(&mut self, transform: &Transform2D) {
        for vertex in &mut self.vertices {
            *vertex = transform.apply_to_point(*vertex);
        }
        self.center = Self::compute_centroid(&self.vertices);
    }
    
    fn clone_box(&self) -> Box<dyn Geometry2D> {
        Box::new(self.clone())
    }
}

/// Compound geometry containing multiple geometries
#[derive(Debug)]
pub struct CompoundGeometry {
    pub geometries: Vec<Box<dyn Geometry2D>>,
    center: Vector2<f64>,
}

impl CompoundGeometry {
    pub fn new(geometries: Vec<Box<dyn Geometry2D>>) -> Self {
        let center = if geometries.is_empty() {
            Vector2::zeros()
        } else {
            let sum: Vector2<f64> = geometries.iter()
                .map(|g| g.center())
                .sum();
            sum / geometries.len() as f64
        };
        
        Self { geometries, center }
    }
    
    pub fn add_geometry(&mut self, geometry: Box<dyn Geometry2D>) {
        self.geometries.push(geometry);
        self.update_center();
    }
    
    fn update_center(&mut self) {
        if !self.geometries.is_empty() {
            let sum: Vector2<f64> = self.geometries.iter()
                .map(|g| g.center())
                .sum();
            self.center = sum / self.geometries.len() as f64;
        }
    }
}

impl Geometry2D for CompoundGeometry {
    fn contains_point(&self, point: Vector2<f64>) -> bool {
        self.geometries.iter().any(|g| g.contains_point(point))
    }
    
    fn bounding_box(&self) -> BoundingBox2D {
        if self.geometries.is_empty() {
            return BoundingBox2D::new(Vector2::zeros(), Vector2::zeros());
        }
        
        let first_box = self.geometries[0].bounding_box();
        self.geometries[1..].iter()
            .fold(first_box, |acc, g| acc.union(&g.bounding_box()))
    }
    
    fn center(&self) -> Vector2<f64> {
        self.center
    }
    
    fn transform(&mut self, transform: &Transform2D) {
        for geometry in &mut self.geometries {
            geometry.transform(transform);
        }
        self.update_center();
    }
    
    fn clone_box(&self) -> Box<dyn Geometry2D> {
        let cloned_geometries = self.geometries.iter()
            .map(|g| g.clone_box())
            .collect();
        Box::new(CompoundGeometry::new(cloned_geometries))
    }
}

