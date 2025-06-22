use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use nalgebra::Matrix3;
use moire_lattice::lattice::{
    compute_wigner_seitz_2d, compute_wigner_seitz_3d,
    compute_brillouin_zone_2d, compute_brillouin_zone_3d,
    generate_neighbor_points_2d, generate_neighbor_points_3d,
    generate_neighbor_points_2d_radius, generate_neighbor_points_3d_radius,
};

/// Single comprehensive benchmark that tests all Voronoi cell methods.
/// This can be run with different feature flags to compare backends:
/// - cargo bench (default features)
/// - cargo bench --no-default-features (fallback implementations)
/// - cargo bench --features voronoi_spade (with Spade for 2D)
/// - cargo bench --features ws3d_quickhull (with Qhull for 3D)
fn bench_all_voronoi_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi_methods");
    
    // Test matrices for 2D and 3D lattices
    let square_2d = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );
    
    let hex_2d = Matrix3::new(
        1.0, -0.5, 0.0,
        0.0, 3.0_f64.sqrt() / 2.0, 0.0,
        0.0, 0.0, 1.0,
    );
    
    let cubic_3d = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    );
    
    let fcc_3d = Matrix3::new(
        0.0, 0.5, 0.5,
        0.5, 0.0, 0.5,
        0.5, 0.5, 0.0,
    );
    
    let tolerance = 1e-10;
    
    // 2D Wigner-Seitz cells
    group.bench_function("wigner_seitz_2d_square", |b| {
        b.iter(|| {
            compute_wigner_seitz_2d(black_box(&square_2d), black_box(tolerance))
        })
    });
    
    group.bench_function("wigner_seitz_2d_hex", |b| {
        b.iter(|| {
            compute_wigner_seitz_2d(black_box(&hex_2d), black_box(tolerance))
        })
    });
    
    // 3D Wigner-Seitz cells
    group.bench_function("wigner_seitz_3d_cubic", |b| {
        b.iter(|| {
            compute_wigner_seitz_3d(black_box(&cubic_3d), black_box(tolerance))
        })
    });
    
    group.bench_function("wigner_seitz_3d_fcc", |b| {
        b.iter(|| {
            compute_wigner_seitz_3d(black_box(&fcc_3d), black_box(tolerance))
        })
    });
    
    // 2D Brillouin zones
    group.bench_function("brillouin_zone_2d_square", |b| {
        b.iter(|| {
            compute_brillouin_zone_2d(black_box(&square_2d), black_box(tolerance))
        })
    });
    
    group.bench_function("brillouin_zone_2d_hex", |b| {
        b.iter(|| {
            compute_brillouin_zone_2d(black_box(&hex_2d), black_box(tolerance))
        })
    });
    
    // 3D Brillouin zones
    group.bench_function("brillouin_zone_3d_cubic", |b| {
        b.iter(|| {
            compute_brillouin_zone_3d(black_box(&cubic_3d), black_box(tolerance))
        })
    });
    
    group.bench_function("brillouin_zone_3d_fcc", |b| {
        b.iter(|| {
            compute_brillouin_zone_3d(black_box(&fcc_3d), black_box(tolerance))
        })
    });
    
    // 2D neighbor generation by shell count
    group.bench_function("neighbors_2d_by_shell", |b| {
        b.iter(|| {
            generate_neighbor_points_2d(black_box(&hex_2d), black_box(5))
        })
    });
    
    // 3D neighbor generation by shell count
    group.bench_function("neighbors_3d_by_shell", |b| {
        b.iter(|| {
            generate_neighbor_points_3d(black_box(&cubic_3d), black_box(4))
        })
    });
    
    // 2D neighbor generation by radius
    group.bench_function("neighbors_2d_by_radius", |b| {
        b.iter(|| {
            generate_neighbor_points_2d_radius(black_box(&hex_2d), black_box(3.0))
        })
    });
    
    // 3D neighbor generation by radius
    group.bench_function("neighbors_3d_by_radius", |b| {
        b.iter(|| {
            generate_neighbor_points_3d_radius(black_box(&cubic_3d), black_box(2.5))
        })
    });
    
    group.finish();
}

criterion_group!(benches, bench_all_voronoi_methods);
criterion_main!(benches);
