use nalgebra::{Matrix2, Vector2, Vector3};
use std::f64::consts::PI;

use crate::lattice::Polyhedron;

const EPS: f64 = 1e-12;

/// Fill a rectangle [0, width] × [0, height] with lattice points.
///
/// This is a convenience wrapper around [`lattice_points_in_polygon_scanline`] that
/// creates a rectangular polygon and calls the general scanline algorithm.
///
/// # Arguments
/// * `a1` - First basis vector (2D, z=0)
/// * `a2` - Second basis vector (2D, z=0)
/// * `width` - Width of the rectangle
/// * `height` - Height of the rectangle
///
/// # Returns
/// Vector of lattice points inside the rectangle
pub fn lattice_points_in_rectangle(
    a1: Vector3<f64>,
    a2: Vector3<f64>,
    width: f64,
    height: f64,
) -> Vec<Vector3<f64>> {
    // Create rectangle polygon with 4 corners
    let vertices = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, height, 0.0),
        Vector3::new(width, height, 0.0),
        Vector3::new(width, 0.0, 0.0),
    ];

    let mut polygon = Polyhedron::new_polygon();
    polygon.vertices = vertices;

    lattice_points_in_polygon_scanline(a1, a2, polygon)
}

/// Fill a circle with lattice points by approximating it as a regular polygon.
///
/// # Arguments
/// * `a1` - First basis vector (2D, z=0)
/// * `a2` - Second basis vector (2D, z=0)
/// * `center` - Center of the circle (2D, z=0)
/// * `radius` - Radius of the circle
/// * `num_segments` - Number of segments to approximate the circle (default: 64)
///
/// # Returns
/// Vector of lattice points inside the circle
pub fn lattice_points_in_circle(
    a1: Vector3<f64>,
    a2: Vector3<f64>,
    center: Vector3<f64>,
    radius: f64,
    num_segments: Option<usize>,
) -> Vec<Vector3<f64>> {
    let num_segments = num_segments.unwrap_or(64);

    // Generate circle vertices (CCW order)
    let mut vertices = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let angle = 2.0 * PI * (i as f64) / (num_segments as f64);
        let x = center.x + radius * angle.cos();
        let y = center.y + radius * angle.sin();
        vertices.push(Vector3::new(x, y, 0.0));
    }

    let mut polygon = Polyhedron::new_polygon();
    polygon.vertices = vertices;

    lattice_points_in_polygon_scanline(a1, a2, polygon)
}

/// Fill a convex polygon with lattice points using the scanline algorithm.
///
/// This algorithm:
/// 1. Transforms the polygon vertices to lattice coordinates (k-space)
/// 2. Scans horizontal lines (u = integer) through the transformed polygon
/// 3. For each scanline, finds intersections with polygon edges
/// 4. Fills lattice points between intersection pairs (even-odd rule)
///
/// # Arguments
/// * `a1` - First basis vector (2D, z=0)
/// * `a2` - Second basis vector (2D, z=0)
/// * `polygon` - Convex polygon with vertices in CCW order
///
/// # Returns
/// Vector of lattice points inside the polygon
///
/// # Panics
/// Panics if the polygon has fewer than 3 vertices or if the basis vectors are degenerate
pub fn lattice_points_in_polygon_scanline(
    a1: Vector3<f64>,
    a2: Vector3<f64>,
    polygon: Polyhedron,
) -> Vec<Vector3<f64>> {
    assert!(
        polygon.vertices.len() >= 3,
        "Polygon must have at least 3 vertices"
    );

    // Build 2D basis matrix B = [a1 | a2] (columns are basis vectors)
    let basis = Matrix2::new(a1.x, a2.x, a1.y, a2.y);

    // Compute inverse to transform from real space to lattice coordinates
    let basis_inv = match basis.try_inverse() {
        Some(inv) => inv,
        None => return Vec::new(), // degenerate basis
    };

    // Transform polygon vertices to k-space (lattice coordinates)
    let mut kpoly: Vec<Vector2<f64>> = polygon
        .vertices
        .iter()
        .map(|v| basis_inv * Vector2::new(v.x, v.y))
        .collect();

    // Find integer u-range to scan
    let umin = kpoly.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let umax = kpoly.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let u_lo = umin.ceil() as i32;
    let u_hi = umax.floor() as i32;

    // Close the polygon for edge loop
    kpoly.push(kpoly[0]);

    let mut result = Vec::new();

    // Scan each integer u-line
    for m in u_lo..=u_hi {
        let u_line = m as f64;
        let mut v_hits = Vec::new();

        // Find intersections with polygon edges
        for e in 0..kpoly.len() - 1 {
            let p0 = kpoly[e];
            let p1 = kpoly[e + 1];

            // Skip horizontal edges (parallel to scanline)
            if (p1.x - p0.x).abs() < EPS {
                continue;
            }

            // Parametric intersection: u_line = p0.x + t * (p1.x - p0.x)
            let t = (u_line - p0.x) / (p1.x - p0.x);

            // Check if intersection is within edge segment [0, 1]
            if t >= -EPS && t <= 1.0 + EPS {
                let v_hit = p0.y + t * (p1.y - p0.y);
                v_hits.push(v_hit);
            }
        }

        if v_hits.is_empty() {
            continue;
        }

        // Sort intersections by v-coordinate
        v_hits.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Fill pairs of intersections (even-odd rule)
        let mut w = 0;
        while w + 1 < v_hits.len() {
            let v0 = v_hits[w];
            let v1 = v_hits[w + 1];

            // Fill integer lattice points between v0 and v1
            let n_lo = v0.ceil() as i32;
            let n_hi = v1.floor() as i32;

            for n in n_lo..=n_hi {
                // Transform back to real space: x = basis * [m, n]
                let lattice_coords = Vector2::new(m as f64, n as f64);
                let real_point = basis * lattice_coords;
                result.push(Vector3::new(real_point.x, real_point.y, 0.0));
            }

            w += 2;
        }
    }

    result
}
