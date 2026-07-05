#!/usr/bin/env python3
"""One-shot: save corrected MPB 8deg 30-band solve."""
import os, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

m, n = 14, 1
L1 = np.array([m, n], dtype=float)
L2 = np.array([-n, m], dtype=float)
L_super = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(n, m)
c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)

R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
r_mpb = R_OVER_A / L_super

lattice = mp.Lattice(size=mp.Vector3(1,1,0),
    basis1=mp.Vector3(L1[0],L1[1],0), basis2=mp.Vector3(L2[0],L2[1],0))

geometry = []
for layer_rot in [np.eye(2), R_mat]:
    a1 = layer_rot @ np.array([1.0, 0.0])
    a2 = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-m-2, m+n+2):
        for i2 in range(-n-2, m+n+2):
            pos = i1*a1 + i2*a2
            frac = B_inv @ pos
            f1, f2 = frac[0]%1.0, frac[1]%1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1,f2,0),
                                        material=mp.Medium(epsilon=EPS_ROD)))

res_per_cell = 64
mpb_resolution = int(res_per_cell * round(L_super))
n_bands = 100
print(f'8deg: res={mpb_resolution}, {n_bands} bands, {len(geometry)} rods')

mp.verbosity(0)
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=n_bands, resolution=mpb_resolution,
    k_points=[mp.Vector3(0,0,0)])

import resource
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t0 = time.time()
ms.run_tm()
t_solve = time.time() - t0
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

freqs_raw = np.array(ms.all_freqs)[0]
freqs = freqs_raw / L_super
print(f'Done in {t_solve:.1f}s, freqs: [{freqs[0]:.6f}, {freqs[-1]:.6f}]')
print(f'Peak RSS: {rss_after:.0f} MB  (before: {rss_before:.0f} MB)')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_gamma_tm_mpb_corrected')
os.makedirs(DATA_DIR, exist_ok=True)
np.savez(os.path.join(DATA_DIR, 'mpb_tm_gamma_8deg_res64_100bands.npz'),
         freqs_all=freqs, freqs_raw=freqs_raw,
         n_bands=n_bands, m=m, n=n,
         resolution=mpb_resolution, res_per_cell=res_per_cell,
         t_solve=t_solve, L_super=L_super)
print(f'Saved to {DATA_DIR}/')
