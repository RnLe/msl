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

// ======================== 2D GEOMETRY TYPES ========================
pub use geometry2d::{
    Geometry2D,                     // trait - base trait for all 2D geometric shapes
    GeometricObject2D,             // struct - combines a shape with a material
    Circle,                        // struct - 2D circle geometry
    Rectangle,                     // struct - 2D rectangle geometry (with rotation support)
    Ellipse,                       // struct - 2D ellipse geometry (with rotation support)
    Polygon,                       // struct - 2D polygon geometry (arbitrary vertices + regular polygon constructor)
    CompoundGeometry,              // struct - collection of multiple 2D geometries
};

pub use geometry2d_bounding_box::BoundingBox2D;   // struct - 2D axis-aligned bounding box
pub use geometry2d_transform::Transform2D;  // struct - 2D transformation (translation, rotation, scaling)