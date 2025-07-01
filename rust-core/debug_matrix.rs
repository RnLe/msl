use nalgebra::{Matrix3, Vector2, Vector3};

fn main() {
    let matrix = Matrix3::new(
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0
    );
    
    println!("Matrix created with Matrix3::new(1,2,3,4,5,6,7,8,9):");
    println!("{}", matrix);
    
    println!("matrix[(0,0)] = {}", matrix[(0,0)]);
    println!("matrix[(0,1)] = {}", matrix[(0,1)]);
    println!("matrix[(0,2)] = {}", matrix[(0,2)]);
    println!("matrix[(1,0)] = {}", matrix[(1,0)]);
    println!("matrix[(1,1)] = {}", matrix[(1,1)]);
    println!("matrix[(1,2)] = {}", matrix[(1,2)]);
    println!("matrix[(2,0)] = {}", matrix[(2,0)]);
    println!("matrix[(2,1)] = {}", matrix[(2,1)]);
    println!("matrix[(2,2)] = {}", matrix[(2,2)]);
    
    // Test matrix * vector
    let vec = Vector3::new(1.0, 1.0, 1.0);
    let result = matrix * vec;
    println!("Matrix * [1, 1, 1] = {:?}", result);
}
