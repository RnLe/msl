#!/usr/bin/env python3
"""
MPB supercell execution at progressively lower resolutions.
Designed to parallel the FDFD minimal-resolution experiment.
Supercell: (38,1) ~3.01°, 50 TM modes at the Gamma point.

Runs resolutions: [32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1]
Outputs: npz files with eigenvalues, grid configurations and computation times.
"""

import os, sys, time, json, resource
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import meep as mp
from meep import mpb

# ── Parameters ──────────────────────────────────────────────────
M_IDX, N_IDX = 38, 1
R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
N_MODES = 50

# Target resolutions to sweep from top to bottom
RESOLUTIONS = [32, 24, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1]

# Lattice math
L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)
N_cells = M_IDX**2 + N_IDX**2

print(f"Supercell: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°, N_cells={N_cells}")
print(f"Modes: {N_MODES} TM modes at Gamma")
print(f"Resolutions to test: {RESOLUTIONS}")
print("=" * 60)

# Setup coordinate transform for MPB geometry
c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)
r_mpb = R_OVER_A / L_SUPER

# Define overarching geometry once
lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
                     basis1=mp.Vector3(L1[0], L1[1], 0), 
                     basis2=mp.Vector3(L2[0], L2[1], 0))

print("Building supercell geometry rods...")
t0_geom = time.time()
geometry = []
for layer_rot in [np.eye(2), R_mat]:
    a1 = layer_rot @ np.array([1.0, 0.0])
    a2 = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-M_IDX - 2, M_IDX + N_IDX + 2):
        for i2 in range(-N_IDX - 2, M_IDX + N_IDX + 2):
            pos = i1 * a1 + i2 * a2
            frac = B_inv @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            # Centre coordinate in [-0.5, 0.5]
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1, f2, 0),
                                        material=mp.Medium(epsilon=EPS_ROD)))

print(f"Done. Placed {len(geometry)} rods in {time.time() - t0_geom:.2f}s.")
print("=" * 60)

CWD = os.path.dirname(os.path.abspath(__file__))
if not CWD or CWD == '.':
    CWD = os.getcwd()

# ── Sweep Loop ──────────────────────────────────────────────────
for res in RESOLUTIONS:
    N_grid = res * round(L_SUPER)
    dof = N_grid ** 2
    
    print(f"\n[{res} px/cell] MPB resolution sweep")
    print(f"Grid: {N_grid}x{N_grid} = {dof:,} cells")
    sys.stdout.flush()

    # Reset max RSS counter
    start_rusage = resource.getrusage(resource.RUSAGE_SELF)

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=N_MODES,
        resolution=res,
        k_points=[mp.Vector3(0, 0, 0)]
    )

    # Supress MPB C++ stdout spam but measure execution time
    mp.verbosity(0)
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    
    t0 = time.time()
    ms.run_tm()
    t_solve = time.time() - t0
    
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)

    # Note: ru_maxrss will capture the high-water mark up to this point
    end_rusage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = end_rusage.ru_maxrss / 1024

    freqs_mpb_raw = np.array(ms.all_freqs)[0]  # First k-point (Gamma)
    freqs_mpb = freqs_mpb_raw / L_SUPER

    print(f"  Complete in {t_solve:.2f}s | Max RSS: {rss_mb:.0f} MB")
    print(f"  ω range: [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}] a/2πc")

    npz_path = os.path.join(CWD, f"mpb_3deg_res{res}.npz")
    np.savez(npz_path, 
             freqs_mpb=freqs_mpb, 
             res=res, 
             t_mpb=t_solve,
             rss_mb=rss_mb,
             grid=N_grid,
             m=M_IDX, n=N_IDX, n_modes=N_MODES)
    
    print(f"  Saved -> {npz_path}")

print("\nAll MPB resolutions swept and saved successfully.")
