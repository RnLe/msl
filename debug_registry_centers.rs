// Debug script to check registry center positions

fn main() {
    println!("Debug script to understand registry center positioning");
    
    // For a hexagonal lattice with a=1, the primitive vectors are:
    // a1 = (1, 0)
    // a2 = (-0.5, √3/2)
    
    let a = 1.0_f64;
    let a1 = (a, 0.0);
    let a2 = (-a * 0.5, a * (3.0_f64).sqrt() / 2.0);
    
    println!("Hexagonal lattice vectors:");
    println!("a1 = ({:.6}, {:.6})", a1.0, a1.1);
    println!("a2 = ({:.6}, {:.6})", a2.0, a2.1);
    
    // The expected registry centers for hexagonal should be:
    // - top: (0, 0)
    // - bridge_a1: 0.5 * a1 = (0.5, 0)
    // - bridge_a2: 0.5 * a2 = (-0.25, √3/4)
    // - bridge_a1_plus_a2: 0.5 * (a1 + a2) = (0.25, √3/4)
    // - hollow_1: (a1 + a2) / 3 = (1/6, √3/6)
    // - hollow_2: (a1 + 2*a2) / 3 = (-1/6, √3/3)
    // - hollow_3: (2*a1 + a2) / 3 = (1/2, √3/6)
    
    println!("\nExpected registry centers (before any moiré transformation):");
    println!("top: (0.000000, 0.000000)");
    println!("bridge_a1: ({:.6}, {:.6})", 0.5 * a1.0, 0.5 * a1.1);
    println!("bridge_a2: ({:.6}, {:.6})", 0.5 * a2.0, 0.5 * a2.1);
    println!("bridge_a1_plus_a2: ({:.6}, {:.6})", 0.5 * (a1.0 + a2.0), 0.5 * (a1.1 + a2.1));
    println!("hollow_1: ({:.6}, {:.6})", (a1.0 + a2.0) / 3.0, (a1.1 + a2.1) / 3.0);
    println!("hollow_2: ({:.6}, {:.6})", (a1.0 + 2.0 * a2.0) / 3.0, (a1.1 + 2.0 * a2.1) / 3.0);
    println!("hollow_3: ({:.6}, {:.6})", (2.0 * a1.0 + a2.0) / 3.0, (2.0 * a1.1 + a2.1) / 3.0);
    
    // For a 5° twist angle, the moiré period should be quite large
    // The moiré lattice vectors should scale these positions
    let theta = 5.0_f64.to_radians();
    println!("\nTwist angle: {:.6} radians ({:.1}°)", theta, theta.to_degrees());
    
    // For small twist angles, the moiré period length is approximately:
    // L_moiré ≈ a / θ for small θ
    let moire_period_approx = a / theta;
    println!("Approximate moiré period length: {:.3}", moire_period_approx);
    
    println!("\nScale factor from primitive cell to moiré supercell: ~{:.1}x", moire_period_approx);
}
