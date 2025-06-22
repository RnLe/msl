use nalgebra::{Matrix3, Vector3};

fn main() {
    // Rectangular lattice a=2, b=1
    let mut basis = Matrix3::zeros();
    basis[(0, 0)] = 2.0;
    basis[(1, 1)] = 1.0;
    basis[(2, 2)] = 1.0;
    
    println!("Basis matrix:");
    println!("{}", basis);
    
    // Generate neighbors by shell 1
    let mut neighbors = Vec::new();
    for n in -1..=1 {
        for m in -1..=1 {
            if n == 0 && m == 0 { continue; }
            let point = (n as f64) * basis.column(0) + (m as f64) * basis.column(1);
            neighbors.push(point);
        }
    }
    
    println!("\nNeighbors:");
    for (i, neighbor) in neighbors.iter().enumerate() {
        println!("{}: [{:.1}, {:.1}, {:.1}] distance: {:.3}", 
                 i, neighbor.x, neighbor.y, neighbor.z, neighbor.norm());
    }
    
    // Find nearest distance
    let nearest_distance = neighbors.iter()
        .map(|v| v.norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    println!("\nNearest distance: {:.3}", nearest_distance);
    
    // Filter nearest neighbors
    let nearest_neighbors: Vec<_> = neighbors.into_iter()
        .filter(|v| v.norm() <= nearest_distance * 1.01)
        .collect();
    
    println!("\nNearest neighbors:");
    for (i, neighbor) in nearest_neighbors.iter().enumerate() {
        println!("{}: [{:.1}, {:.1}, {:.1}] distance: {:.3}", 
                 i, neighbor.x, neighbor.y, neighbor.z, neighbor.norm());
    }
}
