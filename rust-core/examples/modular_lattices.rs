/// Example demonstrating the new modular lattice structure
///
/// This example shows how to create different types of lattices using the new
/// modular file structure. All functionality has been split into specialized
/// modules for better organization.
use moire_lattice::lattice::{
    // Lattice analysis functions
    coordination_number_2d,
    hexagonal_lattice,
    nearest_neighbor_distance_2d,
    simple_cubic_lattice,
    // Lattice constructors
    square_lattice,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Demonstrating the New Modular Lattice Structure ===\n");

    // Example 1: Create a square lattice using the constructor
    println!("1. Creating a square lattice:");
    let square = square_lattice(1.0);
    println!("   Bravais type: {:?}", square.bravais_type());
    println!("   Cell area: {:.3}", square.cell_area());
    println!(
        "   Coordination number: {}",
        coordination_number_2d(&square.bravais_type())
    );
    println!(
        "   Nearest neighbor distance: {:.3}\n",
        nearest_neighbor_distance_2d(square.direct_basis(), &square.bravais_type())
    );

    // Example 2: Create a hexagonal lattice
    println!("2. Creating a hexagonal lattice:");
    let hexagonal = hexagonal_lattice(1.0);
    println!("   Bravais type: {:?}", hexagonal.bravais_type());
    println!("   Cell area: {:.3}", hexagonal.cell_area());
    println!(
        "   Coordination number: {}",
        coordination_number_2d(&hexagonal.bravais_type())
    );
    println!(
        "   Lattice angle: {:.3} radians\n",
        hexagonal.lattice_angle()
    );

    // Example 3: Create a 3D cubic lattice
    println!("3. Creating a simple cubic lattice:");
    let cubic = simple_cubic_lattice(1.0);
    println!("   Bravais type: {:?}", cubic.bravais_type());
    println!("   Cell volume: {:.3}", cubic.cell_volume());
    let (a, b, c) = cubic.lattice_parameters();
    println!("   Lattice parameters: a={:.3}, b={:.3}, c={:.3}", a, b, c);
    let (alpha, beta, gamma) = cubic.lattice_angles();
    println!(
        "   Lattice angles: α={:.3}, β={:.3}, γ={:.3} radians\n",
        alpha, beta, gamma
    );

    // Example 4: Convert 2D to 3D lattice
    println!("4. Converting 2D square lattice to 3D:");
    let square_3d = square.to_3d(nalgebra::Vector3::new(0.0, 0.0, 2.0));
    println!("   Original 2D bravais type: {:?}", square.bravais_type());
    println!("   New 3D bravais type: {:?}", square_3d.bravais_type());
    println!("   3D cell volume: {:.3}\n", square_3d.cell_volume());

    // Example 5: Demonstrate coordinate conversion
    println!("5. Coordinate conversion:");
    let test_point = nalgebra::Vector3::new(0.5, 0.5, 0.0);
    let cartesian = square.frac_to_cart(test_point);
    let fractional = square.cart_to_frac(cartesian);
    println!(
        "   Fractional coordinates: [{:.3}, {:.3}, {:.3}]",
        test_point[0], test_point[1], test_point[2]
    );
    println!(
        "   Cartesian coordinates: [{:.3}, {:.3}, {:.3}]",
        cartesian[0], cartesian[1], cartesian[2]
    );
    println!(
        "   Back to fractional: [{:.3}, {:.3}, {:.3}]\n",
        fractional[0], fractional[1], fractional[2]
    );

    // Example 6: High symmetry points
    println!("6. High symmetry points:");
    let hs_points = square.get_high_symmetry_points_cartesian();
    for (label, point) in hs_points.iter().take(3) {
        println!(
            "   {}: [{:.3}, {:.3}, {:.3}]",
            label, point[0], point[1], point[2]
        );
    }

    println!("\n=== File Structure Information ===");
    println!("The lattice functionality has been organized into the following modules:");
    println!("- lattice2d.rs: 2D lattice implementation");
    println!("- lattice3d.rs: 3D lattice implementation");
    println!("- bravais_types.rs: Bravais lattice classification");
    println!("- polyhedron.rs: Wigner-Seitz cells and Brillouin zones");
    println!("- voronoi_cells.rs: Voronoi cell construction");
    println!("- coordination_numbers.rs: Coordination analysis");
    println!("- construction.rs: Lattice constructor utilities");
    println!("- symmetries/: Symmetry operations and high symmetry points");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_construction() {
        let square = square_lattice(1.0);
        assert_eq!(square.bravais_type(), Bravais2D::Square);
        assert!((square.cell_area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hexagonal_lattice() {
        let hex = hexagonal_lattice(1.0);
        assert_eq!(hex.bravais_type(), Bravais2D::Hexagonal);
        // Hexagonal lattice area = (√3/2) * a²
        let expected_area = (3.0_f64.sqrt() / 2.0);
        assert!((hex.cell_area() - expected_area).abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_conversion() {
        let square = square_lattice(2.0);
        let frac_point = nalgebra::Vector3::new(0.5, 0.25, 0.0);
        let cart_point = square.frac_to_cart(frac_point);
        let back_to_frac = square.cart_to_frac(cart_point);

        for i in 0..3 {
            assert!((frac_point[i] - back_to_frac[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_2d_to_3d_conversion() {
        let square_2d = square_lattice(1.0);
        let square_3d = square_2d.to_3d(nalgebra::Vector3::new(0.0, 0.0, 1.5));

        assert!((square_3d.cell_volume() - 1.5).abs() < 1e-10);
    }
}
