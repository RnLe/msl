#!/usr/bin/env python3
"""
MPB TM solve at K point for honeycomb Dirac lattice, 8° angle, 64 px/cell.

Honeycomb: air holes (eps=1) in dielectric (eps=3.77658), r/a = 0.408425.
Two-atom basis: (0,0) and (1/3,1/3) in fractional coords.
Commensurate angle: (m,n) = (9,7), theta = 8.256°, N = 193.
"""
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

# --- Honeycomb lattice parameters ---
m, n = 9, 7
R_OVER_A = 0.408425
EPS_ROD = 1.0       # air holes
EPS_BG = 3.77658    # dielectric background

# Monolayer basis: triangular lattice
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3) / 2])

# Supercell vectors: L1 = m*a1 + n*a2, L2 = -n*a1 + (m+n)*a2
L1 = m * a1 + n * a2
L2 = -n * a1 + (m + n) * a2
N_cells = m * m + m * n + n * n
L_super = np.sqrt(N_cells)  # = |L1|

# Commensurate twist angle
cos_theta = (m * m + 4 * m * n + n * n) / (2.0 * N_cells)
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])

B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)

# Sublattice positions (fractional of monolayer unit cell)
sublattice_frac = np.array([[0.0, 0.0], [1.0 / 3, 1.0 / 3]])

# MPB radius: in units of supercell size
r_mpb = R_OVER_A / L_super

# MPB lattice definition
lattice = mp.Lattice(
    size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1[0], L1[1], 0),
    basis2=mp.Vector3(L2[0], L2[1], 0))

# Build geometry: cylinders for both layers, both sublattice positions
geometry = []
search_range = m + n + 3

for layer_rot in [np.eye(2), R_mat]:
    # Rotated monolayer basis
    a1_layer = layer_rot @ a1
    a2_layer = layer_rot @ a2
    B_layer = np.column_stack([a1_layer, a2_layer])

    for sub_pos in sublattice_frac:
        # Cartesian offset for this sublattice atom
        offset = B_layer @ sub_pos

        for i1 in range(-search_range, search_range + 1):
            for i2 in range(-search_range, search_range + 1):
                pos = i1 * a1_layer + i2 * a2_layer + offset
                frac = B_inv @ pos
                f1, f2 = frac[0] % 1.0, frac[1] % 1.0
                if f1 >= 0.5:
                    f1 -= 1.0
                if f2 >= 0.5:
                    f2 -= 1.0
                # Only include if inside the cell (with small margin)
                if abs(f1) <= 0.5 and abs(f2) <= 0.5:
                    geometry.append(mp.Cylinder(
                        radius=r_mpb,
                        center=mp.Vector3(f1, f2, 0),
                        material=mp.Medium(epsilon=EPS_ROD)))

# --- K-point in fractional superlattice reciprocal coords ---
# Monolayer K = (1/3)*b1 + (1/3)*b2
# In superlattice fractional: alpha = Q_K . L1 / (2pi), beta = Q_K . L2 / (2pi)
# Q_K . L1 = 2pi*(m+n)/3,  Q_K . L2 = 2pi*m/3
# alpha mod 1 = (m+n) % 3 / 3,  beta mod 1 = m % 3 / 3
alpha = ((m + n) % 3) / 3.0
beta = (m % 3) / 3.0
k_K = mp.Vector3(alpha, beta, 0)

res_per_cell = 64
mpb_resolution = int(res_per_cell * round(L_super))
n_bands = 50

print(f'Honeycomb 8deg K-point: theta={np.degrees(theta_rad):.3f}deg, N={N_cells}')
print(f'  res={mpb_resolution}, {n_bands} bands, {len(geometry)} cylinders')
print(f'  k-point (frac): ({alpha:.4f}, {beta:.4f})')
print(f'  L_super={L_super:.4f}')

mp.verbosity(0)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=n_bands,
    resolution=mpb_resolution,
    k_points=[k_K])

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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_k_tm_honeycomb_mpb')
os.makedirs(DATA_DIR, exist_ok=True)
np.savez(os.path.join(DATA_DIR, 'mpb_tm_k_8deg_res64_50bands.npz'),
         freqs_all=freqs, freqs_raw=freqs_raw,
         n_bands=n_bands, m=m, n=n,
         resolution=mpb_resolution, res_per_cell=res_per_cell,
         t_solve=t_solve, L_super=L_super,
         k_point=[alpha, beta, 0],
         theta_deg=np.degrees(theta_rad),
         N_cells=N_cells)
print(f'Saved to {DATA_DIR}/')
