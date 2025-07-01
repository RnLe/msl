// 2D bounding box module: Contains 2D axis-aligned bounding box functionality
// This module provides efficient spatial bounds calculations for 2D geometries

use nalgebra::Vector2;

/// 2D axis-aligned bounding box
/// 
/// Represents a rectangular region in 2D space defined by minimum and maximum corners.
/// Used for efficient spatial queries, collision detection, and rendering optimizations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox2D {
    /// Minimum corner (bottom-left in standard coordinate system)
    pub min: Vector2<f64>,
    /// Maximum corner (top-right in standard coordinate system)
    pub max: Vector2<f64>,
}

impl BoundingBox2D {
    /// Create a new bounding box from minimum and maximum corners
    /// 
    /// # Arguments
    /// * `min` - Minimum corner coordinates
    /// * `max` - Maximum corner coordinates
    /// 
    /// # Panics
    /// This function will panic in debug mode if min coordinates are greater than max coordinates
    pub fn new(min: Vector2<f64>, max: Vector2<f64>) -> Self {
        debug_assert!(min.x <= max.x && min.y <= max.y, 
                     "Minimum coordinates must be less than or equal to maximum coordinates");
        Self { min, max }
    }
    
    /// Create a bounding box from a center point and size
    /// 
    /// # Arguments
    /// * `center` - Center point of the bounding box
    /// * `size` - Size (width, height) of the bounding box
    pub fn from_center_size(center: Vector2<f64>, size: Vector2<f64>) -> Self {
        let half_size = size / 2.0;
        Self {
            min: center - half_size,
            max: center + half_size,
        }
    }
    
    /// Create a bounding box from a collection of points
    /// 
    /// # Arguments
    /// * `points` - Iterator of points to bound
    /// 
    /// # Returns
    /// * `Some(BoundingBox2D)` if there are points to bound
    /// * `None` if the iterator is empty
    pub fn from_points<I>(points: I) -> Option<Self>
    where
        I: IntoIterator<Item = Vector2<f64>>,
    {
        let mut points_iter = points.into_iter();
        let first_point = points_iter.next()?;
        
        let mut min = first_point;
        let mut max = first_point;
        
        for point in points_iter {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
        }
        
        Some(Self { min, max })
    }
    
    /// Check if a point is inside the bounding box (inclusive of boundaries)
    /// 
    /// # Arguments
    /// * `point` - Point to test
    /// 
    /// # Returns
    /// `true` if the point is inside or on the boundary of the bounding box
    pub fn contains(&self, point: Vector2<f64>) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y
    }
    
    /// Check if this bounding box completely contains another bounding box
    /// 
    /// # Arguments
    /// * `other` - Other bounding box to test
    /// 
    /// # Returns
    /// `true` if this bounding box completely contains the other
    pub fn contains_box(&self, other: &BoundingBox2D) -> bool {
        self.min.x <= other.min.x && self.max.x >= other.max.x &&
        self.min.y <= other.min.y && self.max.y >= other.max.y
    }
    
    /// Check if this bounding box intersects with another bounding box
    /// 
    /// # Arguments
    /// * `other` - Other bounding box to test intersection with
    /// 
    /// # Returns
    /// `true` if the bounding boxes intersect or touch
    pub fn intersects(&self, other: &BoundingBox2D) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y
    }
    
    /// Compute the union of this bounding box with another
    /// 
    /// # Arguments
    /// * `other` - Other bounding box to union with
    /// 
    /// # Returns
    /// A new bounding box that contains both input bounding boxes
    pub fn union(&self, other: &BoundingBox2D) -> BoundingBox2D {
        BoundingBox2D {
            min: Vector2::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
            ),
            max: Vector2::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
            ),
        }
    }
    
    /// Compute the intersection of this bounding box with another
    /// 
    /// # Arguments
    /// * `other` - Other bounding box to intersect with
    /// 
    /// # Returns
    /// * `Some(BoundingBox2D)` if the boxes intersect
    /// * `None` if the boxes don't intersect
    pub fn intersection(&self, other: &BoundingBox2D) -> Option<BoundingBox2D> {
        if !self.intersects(other) {
            return None;
        }
        
        let min = Vector2::new(
            self.min.x.max(other.min.x),
            self.min.y.max(other.min.y),
        );
        let max = Vector2::new(
            self.max.x.min(other.max.x),
            self.max.y.min(other.max.y),
        );
        
        Some(BoundingBox2D { min, max })
    }
    
    /// Get the center point of the bounding box
    pub fn center(&self) -> Vector2<f64> {
        (self.min + self.max) / 2.0
    }
    
    /// Get the size (width, height) of the bounding box
    pub fn size(&self) -> Vector2<f64> {
        self.max - self.min
    }
    
    /// Get the width of the bounding box
    pub fn width(&self) -> f64 {
        self.max.x - self.min.x
    }
    
    /// Get the height of the bounding box
    pub fn height(&self) -> f64 {
        self.max.y - self.min.y
    }
    
    /// Get the area of the bounding box
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }
    
    /// Get the aspect ratio (width / height) of the bounding box
    /// 
    /// # Returns
    /// The aspect ratio, or `f64::INFINITY` if height is zero
    pub fn aspect_ratio(&self) -> f64 {
        let height = self.height();
        if height == 0.0 {
            f64::INFINITY
        } else {
            self.width() / height
        }
    }
    
    /// Expand the bounding box by a given margin in all directions
    /// 
    /// # Arguments
    /// * `margin` - Amount to expand in all directions
    pub fn expand(&self, margin: f64) -> BoundingBox2D {
        let margin_vec = Vector2::new(margin, margin);
        BoundingBox2D {
            min: self.min - margin_vec,
            max: self.max + margin_vec,
        }
    }
    
    /// Expand the bounding box by different margins in x and y directions
    /// 
    /// # Arguments
    /// * `margin` - Margins to expand (x_margin, y_margin)
    pub fn expand_by(&self, margin: Vector2<f64>) -> BoundingBox2D {
        BoundingBox2D {
            min: self.min - margin,
            max: self.max + margin,
        }
    }
    
    /// Check if the bounding box is empty (has zero area)
    pub fn is_empty(&self) -> bool {
        self.width() <= 0.0 || self.height() <= 0.0
    }
    
    /// Check if the bounding box is valid (min <= max for all coordinates)
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y
    }
}

impl Default for BoundingBox2D {
    /// Create a default empty bounding box at the origin
    fn default() -> Self {
        Self {
            min: Vector2::zeros(),
            max: Vector2::zeros(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_bounding_box_creation() {
        let bbox = BoundingBox2D::new(
            Vector2::new(-1.0, -2.0),
            Vector2::new(3.0, 4.0)
        );
        
        assert_eq!(bbox.min, Vector2::new(-1.0, -2.0));
        assert_eq!(bbox.max, Vector2::new(3.0, 4.0));
    }
    
    #[test]
    fn test_from_center_size() {
        let bbox = BoundingBox2D::from_center_size(
            Vector2::new(1.0, 1.0),
            Vector2::new(4.0, 2.0)
        );
        
        assert_eq!(bbox.min, Vector2::new(-1.0, 0.0));
        assert_eq!(bbox.max, Vector2::new(3.0, 2.0));
    }
    
    #[test]
    fn test_from_points() {
        let points = vec![
            Vector2::new(1.0, 2.0),
            Vector2::new(-1.0, 3.0),
            Vector2::new(2.0, -1.0),
        ];
        
        let bbox = BoundingBox2D::from_points(points).unwrap();
        assert_eq!(bbox.min, Vector2::new(-1.0, -1.0));
        assert_eq!(bbox.max, Vector2::new(2.0, 3.0));
        
        // Test empty iterator
        let empty_bbox = BoundingBox2D::from_points(std::iter::empty::<Vector2<f64>>());
        assert!(empty_bbox.is_none());
    }
    
    #[test]
    fn test_contains() {
        let bbox = BoundingBox2D::new(
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 2.0)
        );
        
        assert!(bbox.contains(Vector2::new(1.0, 1.0)));
        assert!(bbox.contains(Vector2::new(0.0, 0.0))); // boundary
        assert!(bbox.contains(Vector2::new(2.0, 2.0))); // boundary
        assert!(!bbox.contains(Vector2::new(-0.1, 1.0)));
        assert!(!bbox.contains(Vector2::new(1.0, 2.1)));
    }
    
    #[test]
    fn test_intersects() {
        let bbox1 = BoundingBox2D::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0));
        let bbox2 = BoundingBox2D::new(Vector2::new(1.0, 1.0), Vector2::new(3.0, 3.0));
        let bbox3 = BoundingBox2D::new(Vector2::new(3.0, 3.0), Vector2::new(4.0, 4.0));
        
        assert!(bbox1.intersects(&bbox2));
        assert!(bbox2.intersects(&bbox1));
        assert!(!bbox1.intersects(&bbox3));
    }
    
    #[test]
    fn test_union() {
        let bbox1 = BoundingBox2D::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0));
        let bbox2 = BoundingBox2D::new(Vector2::new(1.0, 1.0), Vector2::new(3.0, 3.0));
        
        let union = bbox1.union(&bbox2);
        assert_eq!(union.min, Vector2::new(0.0, 0.0));
        assert_eq!(union.max, Vector2::new(3.0, 3.0));
    }
    
    #[test]
    fn test_intersection() {
        let bbox1 = BoundingBox2D::new(Vector2::new(0.0, 0.0), Vector2::new(2.0, 2.0));
        let bbox2 = BoundingBox2D::new(Vector2::new(1.0, 1.0), Vector2::new(3.0, 3.0));
        let bbox3 = BoundingBox2D::new(Vector2::new(3.0, 3.0), Vector2::new(4.0, 4.0));
        
        let intersection = bbox1.intersection(&bbox2).unwrap();
        assert_eq!(intersection.min, Vector2::new(1.0, 1.0));
        assert_eq!(intersection.max, Vector2::new(2.0, 2.0));
        
        assert!(bbox1.intersection(&bbox3).is_none());
    }
    
    #[test]
    fn test_properties() {
        let bbox = BoundingBox2D::new(Vector2::new(1.0, 2.0), Vector2::new(5.0, 6.0));
        
        assert_eq!(bbox.center(), Vector2::new(3.0, 4.0));
        assert_eq!(bbox.size(), Vector2::new(4.0, 4.0));
        assert_relative_eq!(bbox.width(), 4.0);
        assert_relative_eq!(bbox.height(), 4.0);
        assert_relative_eq!(bbox.area(), 16.0);
        assert_relative_eq!(bbox.aspect_ratio(), 1.0);
    }
    
    #[test]
    fn test_expand() {
        let bbox = BoundingBox2D::new(Vector2::new(1.0, 1.0), Vector2::new(3.0, 3.0));
        
        let expanded = bbox.expand(1.0);
        assert_eq!(expanded.min, Vector2::new(0.0, 0.0));
        assert_eq!(expanded.max, Vector2::new(4.0, 4.0));
        
        let expanded_by = bbox.expand_by(Vector2::new(2.0, 1.0));
        assert_eq!(expanded_by.min, Vector2::new(-1.0, 0.0));
        assert_eq!(expanded_by.max, Vector2::new(5.0, 4.0));
    }
}
