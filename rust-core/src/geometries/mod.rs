// Geometries module: Contains geometric object definitions for electromagnetic simulations
// This module provides 2D geometric primitives for defining material distributions

// ======================== MODULE DECLARATIONS ========================
pub mod geometry2d;
pub mod geometry2d_bounding_box;
pub mod geometry2d_transform;

// Test modules
mod _tests_geometry2d;
mod _tests_geometry2d_bounding_box;
mod _tests_geometry2d_transform;

// ======================== CORE GEOMETRY TRAITS & OBJECTS ========================
pub use geometry2d::{
    GeometricObject2D, // struct - combines a shape with a material
    Geometry2D,        // trait - base trait for all 2D geometric shapes
};
// Geometry2D trait methods:
//   contains_point(&self, point: Vector2<f64>) -> bool       - checks if point is inside geometry
//   bounding_box(&self) -> BoundingBox2D                     - returns axis-aligned bounding box
//   center(&self) -> Vector2<f64>                            - returns geometric center
//   transform(&mut self, transform: &Transform2D)            - applies transformation to geometry
//   clone_box(&self) -> Box<dyn Geometry2D>                  - creates deep copy as boxed trait object

// GeometricObject2D impl methods:
//   new<G: Geometry2D + 'static>(shape: G, material: Material) -> Self - creates object from shape and material
//   shape(&self) -> &dyn Geometry2D                          - gets reference to underlying shape
//   shape_mut(&mut self) -> &mut dyn Geometry2D              - gets mutable reference to shape
//   material(&self) -> &Material                             - gets reference to material
//   set_material(&mut self, material: Material)              - sets new material
//   transform(&mut self, transform: &Transform2D)            - applies transformation to shape

// ======================== PRIMITIVE GEOMETRY TYPES ========================
pub use geometry2d::{
    Circle,    // struct - 2D circle geometry
    Ellipse,   // struct - 2D ellipse geometry (with rotation support)
    Polygon,   // struct - 2D polygon geometry (arbitrary vertices + regular polygon constructor)
    Rectangle, // struct - 2D rectangle geometry (with rotation support)
};

// Circle impl methods:
//   new(center: Vector2<f64>, radius: f64) -> Self           - creates circle with center and radius
//   + all Geometry2D trait methods

// Rectangle impl methods:
//   new(center: Vector2<f64>, size: Vector2<f64>) -> Self    - creates axis-aligned rectangle
//   with_rotation(self, angle: f64) -> Self                  - adds rotation to rectangle (builder pattern)
//   + all Geometry2D trait methods

// Ellipse impl methods:
//   new(center: Vector2<f64>, semi_axes: Vector2<f64>) -> Self - creates ellipse with semi-axes (a, b)
//   with_rotation(self, angle: f64) -> Self                  - adds rotation to ellipse (builder pattern)
//   + all Geometry2D trait methods

// Polygon impl methods:
//   new(vertices: Vec<Vector2<f64>>) -> Self                 - creates polygon from vertex list
//   regular(center: Vector2<f64>, radius: f64, sides: usize) -> Self - creates regular polygon
//   compute_centroid(vertices: &[Vector2<f64>]) -> Vector2<f64> - computes geometric centroid
//   + all Geometry2D trait methods

// ======================== COMPOSITE GEOMETRY TYPES ========================
pub use geometry2d::CompoundGeometry; // struct - collection of multiple 2D geometries
// CompoundGeometry impl methods:
//   new(geometries: Vec<Box<dyn Geometry2D>>) -> Self        - creates compound from geometry collection
//   add_geometry(&mut self, geometry: Box<dyn Geometry2D>)   - adds new geometry to compound
//   update_center(&mut self)                                 - recalculates center after modifications
//   + all Geometry2D trait methods

// ======================== GEOMETRIC UTILITIES ========================
pub use geometry2d_bounding_box::BoundingBox2D; // struct - 2D axis-aligned bounding box
// BoundingBox2D impl methods:
//   new(min: Vector2<f64>, max: Vector2<f64>) -> Self        - creates bounding box from corners
//   from_points(points: &[Vector2<f64>]) -> Self             - creates bounding box containing all points
//   contains_point(&self, point: Vector2<f64>) -> bool       - checks if point is inside box
//   intersects(&self, other: &BoundingBox2D) -> bool         - checks intersection with another box
//   union(&self, other: &BoundingBox2D) -> BoundingBox2D     - returns union of two bounding boxes
//   intersection(&self, other: &BoundingBox2D) -> Option<BoundingBox2D> - returns intersection if exists
//   width(&self) -> f64                                      - returns box width (x-dimension)
//   height(&self) -> f64                                     - returns box height (y-dimension)
//   area(&self) -> f64                                       - returns box area
//   center(&self) -> Vector2<f64>                            - returns box center point
//   expand(&self, margin: f64) -> BoundingBox2D              - expands box by margin in all directions
//   translate(&self, offset: Vector2<f64>) -> BoundingBox2D  - translates box by offset

pub use geometry2d_transform::Transform2D; // struct - 2D transformation (translation, rotation, scaling)
// Transform2D impl methods:
//   identity() -> Self                                       - creates identity transformation
//   translation(offset: Vector2<f64>) -> Self               - creates pure translation transform
//   rotation(angle: f64) -> Self                             - creates pure rotation transform (radians)
//   scaling(scale: Vector2<f64>) -> Self                     - creates pure scaling transform
//   uniform_scaling(scale: f64) -> Self                      - creates uniform scaling transform
//   new(translation: Vector2<f64>, rotation: f64, scale: Vector2<f64>) -> Self - creates general transform
//   apply_to_point(&self, point: Vector2<f64>) -> Vector2<f64> - applies transform to point
//   apply_inverse_to_point(&self, point: Vector2<f64>) -> Vector2<f64> - applies inverse transform
//   compose(&self, other: &Transform2D) -> Transform2D       - composes two transformations
//   inverse(&self) -> Transform2D                            - returns inverse transformation
//   to_matrix(&self) -> Matrix3<f64>                         - converts to 3x3 homogeneous matrix
//   from_matrix(matrix: &Matrix3<f64>) -> Self               - creates transform from matrix
