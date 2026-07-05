"""
MPB supercell eigenvalues: (29,1) ~ 3.95°, 64 px/cell, 50 bands.

Canonical square lattice: r/a=0.2, eps_rod=8.9, eps_bg=1.0, TM polarization.
Saves results to supercell_4deg_mpb_64.npz.
"""

import os, sys, time, resource
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np

# ── Parameters ──────────────────────────────────────────────────
M_IDX, N_IDX = 29, 1
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
PX_PER_CELL = 64
N_MODES = 50

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)  # ~29.017
N_CELLS = M_IDX**2 + N_IDX**2  # 842
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)
mpb_res = PX_PER_CELL * round(L_SUPER)  # 64 * 29 = 1856

print(f"Supercell: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°")
print(f"N_cells={N_CELLS}, L_super={L_SUPER:.3f}a")
print(f"Resolution: {PX_PER_CELL} px/cell → grid={mpb_res}x{mpb_res} = {mpb_res**2:,} pixels")
print(f"Bands: {N_MODES}")
print()

# ── MPB solve ───────────────────────────────────────────────────
import meep as mp
from meep import mpb

c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)
r_mpb = R_OVER_A / L_SUPER

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1[0], L1[1], 0),
    basis2=mp.Vector3(L2[0], L2[1], 0))

geometry = []
for layer_rot in [np.eye(2), R_mat]:
    a1 = layer_rot @ np.array([1.0, 0.0])
    a2 = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-M_IDX - 2, M_IDX + N_IDX + 2):
        for i2 in range(-N_IDX - 2, M_IDX + N_IDX + 2):
            pos = i1 * a1 + i2 * a2
            frac = B_inv @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(
                radius=r_mpb,
                center=mp.Vector3(f1, f2, 0),
                material=mp.Medium(epsilon=EPS_ROD)))

print(f"Rods placed: {len(geometry)}")
print(f"MPB resolution: {mpb_res} (= {PX_PER_CELL} px/cell × {round(L_SUPER)} cells)")

mp.verbosity(0)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=N_MODES,
    resolution=mpb_res,
    k_points=[mp.Vector3(0, 0, 0)])

# Suppress MPB stdout/stderr during solve
fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)

t0 = time.time()
ms.run_tm()
t_mpb = time.time() - t0

os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

freqs_mpb_raw = np.array(ms.all_freqs)[0]
freqs_mpb = freqs_mpb_raw / L_SUPER

print(f"\nMPB done: {N_MODES} bands in {t_mpb:.1f}s ({t_mpb/60:.1f} min)")
print(f"Peak RSS: {rss_mb:.0f} MB ({rss_mb/1024:.1f} GB)")
print(f"ω range: [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}] a/2πc")

# ── Save ────────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(outdir, 'supercell_4deg_mpb_64.npz')
np.savez(outfile,
         freqs_mpb=freqs_mpb,
         freqs_mpb_raw=freqs_mpb_raw,
         px_per_cell=PX_PER_CELL, res=mpb_res,
         n_modes=N_MODES,
         m=M_IDX, n=N_IDX, L_SUPER=L_SUPER,
         N_cells=N_CELLS, theta_deg=theta_deg,
         t_mpb=t_mpb, rss_mb=rss_mb)
print(f"Saved → {outfile}")

# ── Plot ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(1, N_MODES + 1), freqs_mpb, 'o-', ms=3, lw=0.8, color='#1f77b4')
ax.set_xlabel('Mode index')
ax.set_ylabel('Frequency  ω  [a / 2πc]')
ax.set_title(f'MPB TM eigenvalues — ({M_IDX},{N_IDX}) supercell, '
             f'θ={theta_deg:.2f}°, {PX_PER_CELL} px/cell')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, N_MODES + 1)

figfile = os.path.join(outdir, 'supercell_4deg_mpb_64.png')
fig.savefig(figfile, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot → {figfile}")
