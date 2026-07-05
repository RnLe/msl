#!/usr/bin/env python3
"""
MPB TE-polarization eigensolve at Gamma for 8° and 3° Moiré supercells.
Resolution = 64 px/cell.
"""
import os, sys, time, resource
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import meep as mp
from meep import mpb

CWD = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()

CASES = [
    {'m': 14, 'n': 1, 'label': '8deg', 'n_modes': 30},
    {'m': 38, 'n': 1, 'label': '3deg', 'n_modes': 50},
]
RES_PX = 64
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0


def build_geometry(m, n, theta_rad, L_super, B_inv):
    r_mpb = R_OVER_A / L_super
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m - 2, m + n + 2):
            for i2 in range(-n - 2, m + n + 2):
                pos = i1 * a1 + i2 * a2
                frac = B_inv @ pos
                f1, f2 = frac[0] % 1.0, frac[1] % 1.0
                if f1 >= 0.5: f1 -= 1.0
                if f2 >= 0.5: f2 -= 1.0
                geometry.append(mp.Cylinder(
                    radius=r_mpb,
                    center=mp.Vector3(f1, f2, 0),
                    material=mp.Medium(epsilon=EPS_ROD)))
    return geometry


for case in CASES:
    m, n = case['m'], case['n']
    n_modes = case['n_modes']
    label = case['label']

    L1 = np.array([m, n], dtype=float)
    L2 = np.array([-n, m], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_rad = 2 * np.arctan2(n, m)
    theta_deg = np.degrees(theta_rad)
    B_super = np.column_stack([L1, L2])
    B_inv = np.linalg.inv(B_super)

    mpb_res = RES_PX * round(L_super)
    N_grid = mpb_res
    print(f"\n{'='*60}")
    print(f"MPB TE — {label} (m={m}, n={n}), θ={theta_deg:.2f}°")
    print(f"Resolution: {RES_PX} px/cell → grid {N_grid}×{N_grid}")
    print(f"Modes: {n_modes}")
    print(f"{'='*60}")

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0),
        basis2=mp.Vector3(L2[0], L2[1], 0))

    geometry = build_geometry(m, n, theta_rad, L_super, B_inv)
    print(f"Placed {len(geometry)} rods")

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=n_modes,
        resolution=mpb_res,
        k_points=[mp.Vector3(0, 0, 0)])

    mp.verbosity(0)
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)

    t0 = time.time()
    ms.run_te()          # <-- TE polarization
    t_solve = time.time() - t0

    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    freqs_raw = np.array(ms.all_freqs)[0]
    freqs = freqs_raw / L_super

    print(f"Solved in {t_solve:.1f}s | RSS: {rss_mb:.0f} MB")
    print(f"ω range: [{freqs[0]:.6f}, {freqs[-1]:.6f}] a/2πc")

    out = os.path.join(CWD, f"mpb_te_{label}_res{RES_PX}.npz")
    np.savez(out, freqs_mpb=freqs, res=RES_PX, n_modes=n_modes,
             m=m, n=n, theta_deg=theta_deg, grid=N_grid,
             t_mpb=t_solve, rss_mb=rss_mb)
    print(f"Saved → {out}")

print("\nAll MPB TE runs complete.")
