// Voronoi cell and Wigner-Seitz cell computation for 2D and 3D lattices
//
// This module provides methods to compute Wigner-Seitz cells and Brillouin zones
// using either native half-space clipping or external libraries (voronoice, voro_rs)
// for more robust construction when needed.

// ======================== IMPORTS ========================
use nalgebra::{Matrix3, Vector2, Vector3};
use crate::lattice::polyhedron::Polyhedron;

#[cfg(feature = "voronoice")]
use voronoice::{VoronoiBuilder, Point as VPoint, BoundingBox};

#[cfg(feature = "ws3d_voro")]
use voro_rs::{
    pre_container::{PreContainerStd, PreContainer},
    container::{ContainerStd, Container1},
    cell::{VoroCell, VoroCellSgl},
};

use std::collections::HashSet;

// ======================== CONSTANTS ========================
const NUMERICAL_TOLERANCE: f64 = 1.0e-12;
#[cfg(feature = "ws3d_voro")]
const MIN_SAFE_SCALE: f64 = 1e-8;               // When a lattice vector is smaller than this, fallbacks are used
#[cfg(feature = "ws3d_voro")]
const MAX_SAFE_SCALE: f64 = 1e8;                // When a lattice vector is larger than this, fallbacks are used

// ======================== 2D WIGNER-SEITZ CELL ========================

/// Compute the 2D Wigner-Seitz cell centered at the origin
///
/// The Wigner-Seitz cell is the primitive cell of the direct lattice,
/// constructed as the region closer to the origin than to any other lattice point.
///
/// Parameters:
/// - `basis`: First two columns are the primitive vectors a₁, a₂
/// - `tolerance`: Numerical tolerance for computations
pub fn compute_wigner_seitz_cell_2d(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    #[cfg(feature = "voronoice")]
    {
        return compute_ws_cell_2d_voronoice(basis, tolerance);
    }
    
    #[allow(unreachable_code)]
    compute_ws_cell_2d_halfspace(basis, tolerance)
}

// ======================== 2D IMPLEMENTATIONS ========================

// Voronoice-based implementation for robust 2D Voronoi construction
#[cfg(feature = "voronoice")]
fn compute_ws_cell_2d_voronoice(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    // Extract the 2D lattice vectors
    let lattice_vector_a1 = basis.column(0);
    let lattice_vector_a2 = basis.column(1);
    
    // Calculate lattice vector norms for validation
    let a1_norm = lattice_vector_a1.norm();
    let a2_norm = lattice_vector_a2.norm();
    
    // Validate lattice vectors are not degenerate
    if a1_norm < tolerance || a2_norm < tolerance {
        panic!("Lattice vectors too small for meaningful Voronoi construction");
    }
    
    let max_lattice_norm = a1_norm.max(a2_norm);
    
    // Conservative radius to ensure we capture all nearest neighbors
    let search_radius = 3.0 * max_lattice_norm;
    let safe_search_radius = search_radius.max(10.0 * tolerance);

    // Generate lattice sites for Voronoi construction
    let mut voronoi_sites: Vec<VPoint> = Vec::new();
    voronoi_sites.push(VPoint { x: 0.0, y: 0.0 }); // Origin is the first site
    
    // Generate nearby lattice points
    let grid_extent_x = (safe_search_radius / a1_norm).ceil() as i32 + 2;
    let grid_extent_y = (safe_search_radius / a2_norm).ceil() as i32 + 2;
    
    // Clamp grid extents to reasonable limits
    let grid_extent_x = grid_extent_x.min(50).max(3);
    let grid_extent_y = grid_extent_y.min(50).max(3);

    // Add lattice points within search radius
    for index_x in -grid_extent_x..=grid_extent_x {
        for index_y in -grid_extent_y..=grid_extent_y {
            if index_x == 0 && index_y == 0 { 
                continue; // Skip origin as it's already added
            }
            
            let lattice_point = (index_x as f64) * lattice_vector_a1 + (index_y as f64) * lattice_vector_a2;
            if lattice_point.norm() <= safe_search_radius + tolerance {
                voronoi_sites.push(VPoint { x: lattice_point.x, y: lattice_point.y });
            }
        }
    }
    
    // Ensure we have enough sites for meaningful construction
    if voronoi_sites.len() < 4 {
        // Add artificial boundary sites as fallback
        voronoi_sites.push(VPoint { x: safe_search_radius, y: 0.0 });
        voronoi_sites.push(VPoint { x: 0.0, y: safe_search_radius });
        voronoi_sites.push(VPoint { x: -safe_search_radius, y: 0.0 });
        voronoi_sites.push(VPoint { x: 0.0, y: -safe_search_radius });
    }

    // Build Voronoi diagram
    let bounding_box_size = safe_search_radius * 1.5;
    let voronoi_diagram = VoronoiBuilder::default()
        .set_sites(voronoi_sites)
        .set_bounding_box(BoundingBox::new_centered_square(bounding_box_size))
        .build()
        .expect("Voronoi construction failed");

    // Extract the cell around the origin (site index 0)
    let origin_cell = voronoi_diagram.cell(0);
    let cell_vertices: Vec<Vector2<f64>> = origin_cell
        .iter_vertices()
        .map(|point| Vector2::new(point.x, point.y))
        .collect();

    if cell_vertices.is_empty() {
        panic!("No vertices found in Voronoi cell");
    }

    // Convert to Polyhedron structure
    let mut polyhedron = Polyhedron::new();
    
    // Remove duplicate vertices with tolerance-based comparison
    let mut unique_vertices: Vec<Vector2<f64>> = Vec::new();
    for vertex in &cell_vertices {
        let vertex_2d = Vector2::new(vertex.x, vertex.y);
        
        // Check if this vertex is already in the unique list
        let is_duplicate = unique_vertices.iter().any(|existing| {
            (existing - vertex_2d).norm() < tolerance * 1000.0
        });
        
        if !is_duplicate {
            unique_vertices.push(vertex_2d);
        }
    }
    
    // Add unique vertices as 3D points (z = 0 for 2D embedding)
    for vertex in &unique_vertices {
        polyhedron.vertices.push(Vector3::new(vertex.x, vertex.y, 0.0));
    }
    
    // Add edges connecting consecutive vertices
    let vertex_count = unique_vertices.len();
    for i in 0..vertex_count {
        polyhedron.edges.push((i, (i + 1) % vertex_count));
    }
    
    // Calculate cell area using unique vertices
    polyhedron.measure = calculate_polygon_area(&unique_vertices);
    
    polyhedron
}

// Native half-space clipping implementation (fallback)
fn compute_ws_cell_2d_halfspace(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {

    // Generate neighbors for Wigner-Seitz construction
    // Use multiple shells to ensure we get all relevant neighbors
    let mut all_neighbors = Vec::new();
    for shell in 1..=3 {
        all_neighbors.extend(generate_lattice_points_2d_by_shell(basis, shell));
    }
    
    // Find the nearest neighbor distance
    let nearest_distance = all_neighbors.iter()
        .map(|vector| vector.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);
    
    // Keep neighbors within reasonable distance for WS cell construction
    // Include up to second nearest neighbors to ensure proper cell construction
    let cutoff_distance = nearest_distance * 2.5;
    let relevant_neighbors: Vec<_> = all_neighbors.into_iter()
        .filter(|vector| vector.norm() <= cutoff_distance)
        .collect();

    // Start with a large bounding polygon
    let bounding_size = 2.0 * nearest_distance;
    let mut polygon_vertices: Vec<Vector2<f64>> = vec![
        Vector2::new(-bounding_size, -bounding_size),
        Vector2::new( bounding_size, -bounding_size),
        Vector2::new( bounding_size,  bounding_size),
        Vector2::new(-bounding_size,  bounding_size),
    ];

    // Clip against perpendicular bisectors of vectors to neighbors
    for neighbor in relevant_neighbors.iter() {
        let neighbor_2d = Vector2::new(neighbor.x, neighbor.y);
        let bisector_normal = neighbor_2d.normalize();
        let bisector_distance = 0.5 * neighbor_2d.norm();

        polygon_vertices = clip_polygon_by_halfspace(&polygon_vertices, &bisector_normal, bisector_distance);
        if polygon_vertices.is_empty() {
            break; // Degenerate case
        }
    }

    // Convert to Polyhedron structure
    let mut polyhedron = Polyhedron::new();
    
    // Remove duplicate vertices (within tolerance)
    let mut unique_vertices = Vec::new();
    for vertex in &polygon_vertices {
        let is_duplicate = unique_vertices.iter().any(|&existing: &Vector2<f64>| {
            (existing - vertex).norm() < tolerance * 100.0
        });
        if !is_duplicate {
            unique_vertices.push(*vertex);
        }
    }
    
    // Add vertices with coordinate cleanup for numerical stability
    let half_neighbor_distance = nearest_distance * 0.5;
    for vertex in &unique_vertices {
        let mut clean_x = vertex.x;
        let mut clean_y = vertex.y;
        
        // Snap to expected values if very close (reduces numerical noise)
        if (clean_x - half_neighbor_distance).abs() < tolerance * 1000.0 { 
            clean_x = half_neighbor_distance; 
        }
        if (clean_x + half_neighbor_distance).abs() < tolerance * 1000.0 { 
            clean_x = -half_neighbor_distance; 
        }
        if (clean_y - half_neighbor_distance).abs() < tolerance * 1000.0 { 
            clean_y = half_neighbor_distance; 
        }
        if (clean_y + half_neighbor_distance).abs() < tolerance * 1000.0 { 
            clean_y = -half_neighbor_distance; 
        }
        
        polyhedron.vertices.push(Vector3::new(clean_x, clean_y, 0.0));
    }
    
    // Add edges
    for i in 0..unique_vertices.len() {
        polyhedron.edges.push((i, (i + 1) % unique_vertices.len()));
    }
    
    polyhedron.measure = calculate_polygon_area(&unique_vertices);
    
    polyhedron
}

// ======================== 3D WIGNER-SEITZ CELL ========================

/// Compute the 3D Wigner-Seitz cell of a direct-space lattice
///
/// Parameters:
/// - `basis`: Columns are the primitive vectors a₁, a₂, a₃
/// - `tolerance`: Numerical tolerance for computations
pub fn compute_wigner_seitz_cell_3d(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    #[cfg(feature = "ws3d_voro")]
    {
        return compute_ws_cell_3d_voro(basis, tolerance);
    }
    
    #[cfg(not(feature = "ws3d_voro"))]
    {
        // Fallback: return the unit parallelepiped
        let _tolerance = tolerance; // Unused in fallback
        return create_parallelepiped_from_basis(basis);
    }
}

// ======================== 3D IMPLEMENTATIONS ========================

// Voro++ based implementation for robust 3D Voronoi construction
#[cfg(feature = "ws3d_voro")]
fn compute_ws_cell_3d_voro(basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    // Check for extreme lattice parameters
    let lattice_norms: Vec<f64> = (0..3).map(|i| basis.column(i).norm()).collect();
    let min_norm = lattice_norms.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_norm = lattice_norms.iter().fold(0.0_f64, |a, &b| a.max(b));
    
    // Fall back to parallelepiped for extreme cases
    if min_norm < MIN_SAFE_SCALE || max_norm > MAX_SAFE_SCALE {
        return create_parallelepiped_from_basis(basis);
    }

    // Generate neighbor lattice sites
    let neighbors = generate_neighbors_for_ws_3d(basis, tolerance);
    
    // Need at least 6 neighbors for meaningful 3D Voronoi cell
    if neighbors.len() < 6 {
        return create_parallelepiped_from_basis(basis);
    }

    // Set up Voro++ container
    let max_radius = neighbors.iter()
        .map(|v| v.norm())
        .fold(0.0, f64::max)
        .max(1e-10);
    let safe_radius = max_radius.max(1e-6).min(1e6);
    
    let box_min = [-safe_radius, -safe_radius, -safe_radius];
    let box_max = [ safe_radius,  safe_radius,  safe_radius];
    let periodic = [false, false, false];

    // Pre-fill particles (neighbors)
    let mut pre_container = PreContainerStd::new(box_min, box_max, periodic);
    for (index, neighbor) in neighbors.iter().enumerate() {
        pre_container.put(index as i32, [neighbor.x, neighbor.y, neighbor.z], 0.0);
    }
    let optimal_grids = pre_container.optimal_grids();

    let mut container: ContainerStd<'_> = ContainerStd::new(box_min, box_max, optimal_grids, periodic);
    pre_container.setup(&mut container);

    // Compute Voronoi cell at origin
    let cell_result = container.compute_ghost_cell([0.0, 0.0, 0.0], 0.0);
    
    let mut voronoi_cell: VoroCellSgl = match cell_result {
        Some(cell) => cell,
        None => return create_parallelepiped_from_basis(basis),
    };

    // Convert to Polyhedron structure
    let mut polyhedron = Polyhedron::new();

    // Extract vertices
    let vertices_raw = voronoi_cell.vertices_global([0.0, 0.0, 0.0]);
    for vertex_coords in vertices_raw.chunks(3) {
        polyhedron.vertices.push(Vector3::new(vertex_coords[0], vertex_coords[1], vertex_coords[2]));
    }

    // Extract faces with correct orientation
    let face_orders = voronoi_cell.face_orders();
    let face_indices = voronoi_cell.face_vertices();
    let mut index_offset = 0;
    
    for &vertex_count in &face_orders {
        let mut face = Vec::with_capacity(vertex_count as usize);
        for _ in 0..vertex_count {
            face.push(face_indices[index_offset] as usize);
            index_offset += 1;
        }
        
        // Ensure consistent CCW orientation
        if face.len() >= 3 {
            let vertex_a = polyhedron.vertices[face[0]];
            let vertex_b = polyhedron.vertices[face[1]];
            let vertex_c = polyhedron.vertices[face[2]];
            let face_normal = (vertex_b - vertex_a).cross(&(vertex_c - vertex_a));
            
            // If normal points inward, reverse the face
            if face_normal.dot(&vertex_a) < 0.0 {
                face.reverse();
            }
        }
        polyhedron.faces.push(face);
    }

    polyhedron.edges = extract_edges_from_faces(&polyhedron.faces);
    polyhedron.measure = voronoi_cell.volume();

    polyhedron
}

// ======================== BRILLOUIN ZONE COMPUTATION ========================

/// Compute the 2D first Brillouin zone (Wigner-Seitz cell of reciprocal lattice)
pub fn compute_brillouin_zone_2d(reciprocal_basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    compute_wigner_seitz_cell_2d(reciprocal_basis, tolerance)
}

/// Compute the 3D first Brillouin zone (Wigner-Seitz cell of reciprocal lattice)
pub fn compute_brillouin_zone_3d(reciprocal_basis: &Matrix3<f64>, tolerance: f64) -> Polyhedron {
    compute_wigner_seitz_cell_3d(reciprocal_basis, tolerance)
}

// ======================== LATTICE POINT GENERATION ========================

/// Generate 2D lattice points within specified shell index
///
/// Returns all integer linear combinations n·a₁ + m·a₂ where |n|, |m| ≤ max_shell
/// (excluding the origin)
pub fn generate_lattice_points_2d_by_shell(
    basis: &Matrix3<f64>,
    max_shell: usize,
) -> Vec<Vector3<f64>> {
    let vector_a1 = basis.column(0);
    let vector_a2 = basis.column(1);
    let shell_limit = max_shell as i32;

    let mut lattice_points = Vec::new();
    for n in -shell_limit..=shell_limit {
        for m in -shell_limit..=shell_limit {
            if n == 0 && m == 0 { 
                continue; // Skip origin
            }
            let point = (n as f64) * vector_a1 + (m as f64) * vector_a2;
            lattice_points.push(point.into());
        }
    }
    lattice_points
}

/// Generate 3D lattice points within specified shell index
///
/// Returns all integer linear combinations n·a₁ + m·a₂ + l·a₃ where |n|, |m|, |l| ≤ max_shell
/// (excluding the origin)
pub fn generate_lattice_points_3d_by_shell(
    basis: &Matrix3<f64>,
    max_shell: usize,
) -> Vec<Vector3<f64>> {
    let vector_a1 = basis.column(0);
    let vector_a2 = basis.column(1);
    let vector_a3 = basis.column(2);
    let shell_limit = max_shell as i32;

    let mut lattice_points = Vec::new();
    for n in -shell_limit..=shell_limit {
        for m in -shell_limit..=shell_limit {
            for l in -shell_limit..=shell_limit {
                if n == 0 && m == 0 && l == 0 { 
                    continue; // Skip origin
                }
                let point = (n as f64) * vector_a1 + (m as f64) * vector_a2 + (l as f64) * vector_a3;
                lattice_points.push(point.into());
            }
        }
    }
    lattice_points
}

/// Generate 2D lattice points within specified radius
///
/// Returns all lattice points n·a₁ + m·a₂ with distance ≤ radius from origin
/// (excluding the origin)
pub fn generate_lattice_points_2d_within_radius(
    basis: &Matrix3<f64>,
    radius: f64,
) -> Vec<Vector3<f64>> {
    if radius <= 0.0 { 
        return Vec::new(); 
    }

    // Estimate required shell based on shortest lattice vector
    let min_length = basis.column(0).norm().min(basis.column(1).norm());
    let estimated_shell = (radius / min_length).ceil() as usize;

    // Generate points and filter by radius
    generate_lattice_points_2d_by_shell(basis, estimated_shell)
        .into_iter()
        .filter(|point| point.norm() <= radius + NUMERICAL_TOLERANCE)
        .collect()
}

/// Generate 3D lattice points within specified radius
///
/// Returns all lattice points n·a₁ + m·a₂ + l·a₃ with distance ≤ radius from origin
/// (excluding the origin)
pub fn generate_lattice_points_3d_within_radius(
    basis: &Matrix3<f64>,
    radius: f64,
) -> Vec<Vector3<f64>> {
    if radius <= 0.0 { 
        return Vec::new(); 
    }

    // Estimate required shell based on shortest lattice vector
    let min_length = [0, 1, 2].iter()
        .map(|&i| basis.column(i).norm())
        .fold(f64::MAX, f64::min);
    let estimated_shell = (radius / min_length).ceil() as usize;

    // Generate points and filter by radius
    generate_lattice_points_3d_by_shell(basis, estimated_shell)
        .into_iter()
        .filter(|point| point.norm() <= radius + NUMERICAL_TOLERANCE)
        .collect()
}

// ======================== HELPER FUNCTIONS ========================

// Sutherland-Hodgman polygon clipping against a half-plane
fn clip_polygon_by_halfspace(
    polygon: &[Vector2<f64>],
    normal: &Vector2<f64>,
    distance: f64,
) -> Vec<Vector2<f64>> {
    if polygon.is_empty() { 
        return Vec::new(); 
    }
    
    let mut clipped_polygon = Vec::with_capacity(polygon.len());
    let mut previous_vertex = *polygon.last().unwrap();
    let mut previous_inside = normal.dot(&previous_vertex) - distance <= 0.0;
    
    for &current_vertex in polygon {
        let current_inside = normal.dot(&current_vertex) - distance <= 0.0;
        
        // Edge crosses the boundary
        if current_inside != previous_inside {
            let t = (distance - normal.dot(&previous_vertex)) / normal.dot(&(current_vertex - previous_vertex));
            let intersection = previous_vertex + (current_vertex - previous_vertex) * t;
            clipped_polygon.push(intersection);
        }
        
        if current_inside { 
            clipped_polygon.push(current_vertex); 
        }
        
        previous_vertex = current_vertex;
        previous_inside = current_inside;
    }
    
    clipped_polygon
}

// Calculate polygon area using the shoelace formula
fn calculate_polygon_area(vertices: &[Vector2<f64>]) -> f64 {
    let vertex_count = vertices.len();
    let mut area = 0.0;
    
    for i in 0..vertex_count {
        let (x_i, y_i) = (vertices[i].x, vertices[i].y);
        let (x_j, y_j) = (vertices[(i + 1) % vertex_count].x, vertices[(i + 1) % vertex_count].y);
        area += x_i * y_j - x_j * y_i;
    }
    
    0.5 * area.abs()
}

// Extract unique edges from face definitions
fn extract_edges_from_faces(faces: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut unique_edges: HashSet<(usize, usize)> = HashSet::new();
    
    for face in faces {
        // Add edges between consecutive vertices
        for window in face.windows(2) {
            add_normalized_edge(&mut unique_edges, window[0], window[1]);
        }
        // Close the face by connecting last to first
        if let (Some(&first), Some(&last)) = (face.first(), face.last()) {
            add_normalized_edge(&mut unique_edges, last, first);
        }
    }
    
    unique_edges.into_iter().collect()
}

// Add edge with normalized ordering (smaller index first)
fn add_normalized_edge(edges: &mut HashSet<(usize, usize)>, i: usize, j: usize) {
    if i < j { 
        edges.insert((i, j)); 
    } else { 
        edges.insert((j, i)); 
    }
}

// Generate neighbors for 3D Wigner-Seitz construction
#[cfg(feature = "ws3d_voro")]
fn generate_neighbors_for_ws_3d(basis: &Matrix3<f64>, tolerance: f64) -> Vec<Vector3<f64>> {
    let max_norm = (0..3)
        .map(|i| basis.column(i).norm())
        .fold(1e-10, f64::max);
    let search_radius = 3.0 * max_norm;
    let shell_count = (search_radius / max_norm).ceil() as usize + 1;
    let shell_count = shell_count.min(10).max(2); // Reasonable bounds
    
    generate_lattice_points_3d_by_shell(basis, shell_count)
        .into_iter()
        .filter(|v| v.norm() <= search_radius + tolerance)
        .collect()
}

// Create a parallelepiped from basis vectors (fallback for 3D)
fn create_parallelepiped_from_basis(basis: &Matrix3<f64>) -> Polyhedron {
    let mut polyhedron = Polyhedron::new();
    
    // Eight vertices of the parallelepiped
    let vertices = [
        Vector3::zeros(),
        basis.column(0).into(),
        basis.column(1).into(),
        basis.column(2).into(),
        (basis.column(0) + basis.column(1)).into(),
        (basis.column(0) + basis.column(2)).into(),
        (basis.column(1) + basis.column(2)).into(),
        (basis.column(0) + basis.column(1) + basis.column(2)).into(),
    ];
    polyhedron.vertices.extend_from_slice(&vertices);
    
    // Six faces of the parallelepiped
    polyhedron.faces = vec![
        vec![0, 2, 4, 1], // Bottom face
        vec![0, 1, 5, 3], // Front face
        vec![0, 3, 6, 2], // Left face
        vec![7, 5, 1, 4], // Top face
        vec![7, 6, 3, 5], // Back face
        vec![7, 4, 2, 6], // Right face
    ];
    
    polyhedron.edges = extract_edges_from_faces(&polyhedron.faces);
    polyhedron.measure = basis.determinant().abs();
    
    polyhedron
}
