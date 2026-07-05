#!/usr/bin/env python3
"""
MPB TM solve at Gamma point for hex rods lattice, 8° angle, 64 px/cell.

Hex rods: dielectric rods eps=12 in air, r/a = 0.220.
Single rod per cell at origin.
Commensurate angle: (m,n) = (9,7), theta = 8.256°, N = 193.

Extracts Bloch functions u_n(r) for all 50 bands.
"""
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

# --- Hex rods lattice parameters ---
m, n = 9, 7
R_OVER_A = 0.220
EPS_ROD = 12.0    # dielectric rods
EPS_BG = 1.0      # air background

# Monolayer basis: triangular lattice
a1 = np.array([1.0, 0.0])
a2 = np.array([0.5, np.sqrt(3) / 2])

# Coincidence lattice vectors (correct for bilayer commensurate):
# L1 = n*a1 + m*a2, L2 = -m*a1 + (n+m)*a2
L1 = n * a1 + m * a2
L2 = -m * a1 + (m + n) * a2
N_cells = m * m + m * n + n * n
L_super = np.sqrt(N_cells)

# Commensurate twist angle
cos_theta = (m * m + 4 * m * n + n * n) / (2.0 * N_cells)
theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])

B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)

# MPB radius: in fractional units (scaled by basis vector length)
r_mpb = R_OVER_A / L_super

# MPB lattice definition
lattice = mp.Lattice(
    size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1[0], L1[1], 0),
    basis2=mp.Vector3(L2[0], L2[1], 0))

# Build geometry: cylinders for both layers, single rod per unit cell
geometry = []
seen = set()
search_range = m + n + 3

for layer_rot in [np.eye(2), R_mat]:
    a1_layer = layer_rot @ a1
    a2_layer = layer_rot @ a2

    for i1 in range(-search_range, search_range + 1):
        for i2 in range(-search_range, search_range + 1):
            pos = i1 * a1_layer + i2 * a2_layer
            frac = B_inv @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            if f1 >= 0.5:
                f1 -= 1.0
            if f2 >= 0.5:
                f2 -= 1.0
            key = (round(f1, 6), round(f2, 6))
            if key not in seen and abs(f1) <= 0.5 and abs(f2) <= 0.5:
                seen.add(key)
                geometry.append(mp.Cylinder(
                    radius=r_mpb,
                    center=mp.Vector3(f1, f2, 0),
                    material=mp.Medium(epsilon=EPS_ROD)))

# --- Gamma point: k = (0, 0) ---
k_Gamma = mp.Vector3(0, 0, 0)

res_per_cell = 64
mpb_resolution = int(res_per_cell * round(L_super))
n_bands = 50

print(f'Hex rods 8deg Gamma-point: theta={np.degrees(theta_rad):.3f}deg, N={N_cells}')
print(f'  res={mpb_resolution}, {n_bands} bands, {len(geometry)} cylinders')
print(f'  k-point (frac): (0, 0)')
print(f'  L_super={L_super:.4f}')
import sys; sys.stdout.flush()

mp.verbosity(0)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=n_bands,
    resolution=mpb_resolution,
    k_points=[k_Gamma])

import resource
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
t0 = time.time()
ms.run_tm()
t_solve = time.time() - t0
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

freqs_raw = np.array(ms.all_freqs)[0]
freqs = freqs_raw / L_super
print(f'Solve done in {t_solve:.1f}s, freqs: [{freqs[0]:.6f}, {freqs[-1]:.6f}]')
print(f'Peak RSS: {rss_after:.0f} MB  (before: {rss_before:.0f} MB)')
sys.stdout.flush()

# --- Extract Bloch functions ---
print('Extracting Bloch functions (u_n) ...')
sys.stdout.flush()
bloch_fields = []
for band in range(1, n_bands + 1):
    ms.get_efield(band, bloch_phase=False)  # loads u_n(r) without e^{ikr}
    field = ms.get_curfield_as_array(bloch_phase=False)
    # TM scalar: field is (Nx, Ny) complex
    ez = field.copy()
    bloch_fields.append(ez)
    if band % 10 == 0:
        print(f'  extracted band {band}/{n_bands}')
        sys.stdout.flush()

bloch_fields = np.array(bloch_fields)  # (n_bands, Nx, Ny) complex
print(f'Bloch fields shape: {bloch_fields.shape}')

# --- Save ---
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_gamma_tm_hex')
os.makedirs(DATA_DIR, exist_ok=True)
outpath = os.path.join(DATA_DIR, 'mpb_tm_gamma_8deg_res64_50bands.npz')
np.savez(outpath,
         freqs_all=freqs, freqs_raw=freqs_raw,
         bloch_fields=bloch_fields,
         n_bands=n_bands, m=m, n=n,
         resolution=mpb_resolution, res_per_cell=res_per_cell,
         t_solve=t_solve, L_super=L_super,
         k_point=[0, 0, 0],
         theta_deg=np.degrees(theta_rad),
         N_cells=N_cells)
print(f'Saved to {outpath}')
