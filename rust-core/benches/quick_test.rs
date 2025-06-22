use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use nalgebra::Matrix3;
use moire_lattice::lattice::{
    compute_wigner_seitz_2d, compute_wigner_seitz_3d,
    square_lattice, simple_cubic_lattice,
};

/// Simple benchmark to verify the basic setup works
fn bench_quick_voronoi_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_voronoi_test");
    
    // Simple square lattice
    let square_basis = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );
    
    // Simple cubic lattice
    let cubic_basis = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );
    
    let tol = 1e-10;
    
    group.bench_function("2d_square_wigner_seitz", |b| {
        b.iter(|| {
            compute_wigner_seitz_2d(black_box(&square_basis), black_box(tol))
        });
    });
    
    group.bench_function("3d_cubic_wigner_seitz", |b| {
        b.iter(|| {
            compute_wigner_seitz_3d(black_box(&cubic_basis), black_box(tol))
        });
    });
    
    group.bench_function("2d_lattice_creation", |b| {
        b.iter(|| {
            let _lattice = square_lattice(black_box(1.0));
        });
    });
    
    group.bench_function("3d_lattice_creation", |b| {
        b.iter(|| {
            let _lattice = simple_cubic_lattice(black_box(1.0));
        });
    });
    
    group.finish();
}

criterion_group!(quick_benches, bench_quick_voronoi_test);
criterion_main!(quick_benches);
