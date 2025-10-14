use msl::lattice::{
    lattice_points_in_circle, lattice_points_in_polygon_scanline, lattice_points_in_rectangle,
    Polyhedron,
};
use nalgebra::Vector3;

fn main() {
    // Test 1: Rectangle
    println!("=== Test 1: Rectangle ===");
    let a1 = Vector3::new(0.7, 0.6, 0.0);
    let a2 = Vector3::new(0.2, -0.5, 0.0);
    let width = 30.0;
    let height = 10.0;

    let rect_points = lattice_points_in_rectangle(a1, a2, width, height);
    println!(
        "Rectangle [0, {}] x [0, {}] with basis vectors:",
        width, height
    );
    println!("  a1 = [{}, {}]", a1.x, a1.y);
    println!("  a2 = [{}, {}]", a2.x, a2.y);
    println!("  Found {} lattice points", rect_points.len());

    // Test 2: Circle
    println!("\n=== Test 2: Circle ===");
    let center = Vector3::new(250.0, 50.0, 0.0);
    let radius = 40.0;

    let circle_points = lattice_points_in_circle(a1, a2, center, radius, None);
    println!(
        "Circle at ({}, {}) with radius {}:",
        center.x, center.y, radius
    );
    println!("  Found {} lattice points", circle_points.len());

    // Test 3: Custom polygon (triangle)
    println!("\n=== Test 3: Triangle ===");
    let mut triangle = Polyhedron::new_polygon();
    triangle.vertices = vec![
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(20.0, 0.0, 0.0),
        Vector3::new(10.0, 15.0, 0.0),
    ];

    let triangle_points = lattice_points_in_polygon_scanline(a1, a2, triangle);
    println!("Triangle vertices:");
    println!("  (0, 0), (20, 0), (10, 15)");
    println!("  Found {} lattice points", triangle_points.len());

    // Verify some points are within bounds
    println!("\n=== Verification ===");
    let all_in_rect = rect_points
        .iter()
        .all(|p| p.x >= -1e-6 && p.x <= width + 1e-6 && p.y >= -1e-6 && p.y <= height + 1e-6);
    println!(
        "All rectangle points within bounds: {}",
        if all_in_rect { "✓" } else { "✗" }
    );

    let all_in_circle = circle_points.iter().all(|p| {
        let dx = p.x - center.x;
        let dy = p.y - center.y;
        dx * dx + dy * dy <= radius * radius + 1e-6
    });
    println!(
        "All circle points within radius: {}",
        if all_in_circle { "✓" } else { "✗" }
    );

    println!("\n=== Sample Points ===");
    println!("First 5 rectangle points:");
    for (i, p) in rect_points.iter().take(5).enumerate() {
        println!("  {}: ({:.3}, {:.3})", i, p.x, p.y);
    }

    println!("\nFirst 5 circle points:");
    for (i, p) in circle_points.iter().take(5).enumerate() {
        println!("  {}: ({:.3}, {:.3})", i, p.x, p.y);
    }
}
