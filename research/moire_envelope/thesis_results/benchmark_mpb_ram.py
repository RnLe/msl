"""Benchmark MPB RAM and time scaling across angles."""
import os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0

def measure_mpb(m_idx, n_idx, res, n_bands):
    gc.collect()
    L1 = np.array([m_idx, n_idx], dtype=float)
    L2 = np.array([-n_idx, m_idx], dtype=float)
    L_SUPER = np.sqrt(L1 @ L1)
    theta_rad = 2 * np.arctan2(n_idx, m_idx)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    B_super = np.column_stack([L1, L2])
    B_inv = np.linalg.inv(B_super)
    r_mpb = R_OVER_A / L_SUPER
    N_cells = m_idx**2 + n_idx**2
    grid = res * round(L_SUPER)

    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0), basis2=mp.Vector3(L2[0], L2[1], 0))
    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m_idx - 2, m_idx + n_idx + 2):
            for i2 in range(-n_idx - 2, m_idx + n_idx + 2):
                pos = i1 * a1 + i2 * a2
                frac = B_inv @ pos
                f1, f2 = frac[0] % 1.0, frac[1] % 1.0
                if f1 >= 0.5: f1 -= 1.0
                if f2 >= 0.5: f2 -= 1.0
                geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1, f2, 0),
                    material=mp.Medium(epsilon=EPS_ROD)))

    mp.verbosity(0)
    ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG), num_bands=n_bands,
        resolution=res, k_points=[mp.Vector3(0, 0, 0)])

    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    t0 = time.time()
    ms.run_tm()
    dt = time.time() - t0
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB on Linux

    return N_cells, grid, len(geometry), dt, rss

header = "%8s %6s %6s %4s %6s %6s %7s %8s" % ("(m,n)", "theta", "N", "res", "grid", "rods", "time", "RSS_MB")
print(header)
print("-" * 65)

# --- res=32, 10 bands ---
print("\n=== res=32, 10 bands ===")
for m in [8, 11, 14, 19, 25, 30, 35, 38]:
    n = 1
    theta = 2 * np.degrees(np.arctan2(n, m))
    N, grid, rods, dt, rss = measure_mpb(m, n, 32, 10)
    print("(%2d,%d) %5.2f° %6d %4d %6d %6d %6.2fs %7.0fMB" % (m, n, theta, N, 32, grid, rods, dt, rss))
    sys.stdout.flush()

# --- res=64, 10 bands ---
print("\n=== res=64, 10 bands ===")
for m in [8, 11, 14, 19, 25, 30, 35, 38]:
    n = 1
    theta = 2 * np.degrees(np.arctan2(n, m))
    N, grid, rods, dt, rss = measure_mpb(m, n, 64, 10)
    print("(%2d,%d) %5.2f° %6d %4d %6d %6d %6.2fs %7.0fMB" % (m, n, theta, N, 64, grid, rods, dt, rss))
    sys.stdout.flush()

# --- res=64, 100 bands ---
print("\n=== res=64, 100 bands ===")
for m in [8, 11, 14, 19, 25, 30, 35, 38]:
    n = 1
    theta = 2 * np.degrees(np.arctan2(n, m))
    N, grid, rods, dt, rss = measure_mpb(m, n, 64, 100)
    print("(%2d,%d) %5.2f° %6d %4d %6d %6d %6.2fs %7.0fMB" % (m, n, theta, N, 64, grid, rods, dt, rss))
    sys.stdout.flush()

# --- res=128, 10 bands (may OOM at large m) ---
print("\n=== res=128, 10 bands ===")
for m in [8, 11, 14, 19, 22, 25]:
    n = 1
    theta = 2 * np.degrees(np.arctan2(n, m))
    N, grid, rods, dt, rss = measure_mpb(m, n, 128, 10)
    print("(%2d,%d) %5.2f° %6d %4d %6d %6d %6.2fs %7.0fMB" % (m, n, theta, N, 128, grid, rods, dt, rss))
    sys.stdout.flush()
