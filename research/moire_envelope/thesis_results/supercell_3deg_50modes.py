"""
MPB vs FDFD supercell comparison: (38,1) ~3.01°, res=32, 50 modes.
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
M_IDX, N_IDX = 38, 1
R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
RES = 32
N_MODES = 50
# Use small sigma to get lowest eigenvalues (matching MPB's lowest-first approach)
SIGMA = 1e-4

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)
N_cells = M_IDX**2 + N_IDX**2
N_grid = RES * round(L_SUPER)

print(f"Supercell: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°, N_cells={N_cells}")
print(f"Grid: {N_grid}x{N_grid} = {N_grid**2:,} DOF, res={RES}/cell, {N_MODES} modes")
print()

# ── 1. MPB ──────────────────────────────────────────────────────
print("[1/4] MPB solve...", end=' ', flush=True)

import meep as mp
from meep import mpb

c, s = np.cos(theta_rad), np.sin(theta_rad)
R_mat = np.array([[c, -s], [s, c]])
B_super = np.column_stack([L1, L2])
B_inv = np.linalg.inv(B_super)
r_mpb = R_OVER_A / L_SUPER

lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1[0], L1[1], 0), basis2=mp.Vector3(L2[0], L2[1], 0))

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
            geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1, f2, 0),
                material=mp.Medium(epsilon=EPS_ROD)))

mp.verbosity(0)
ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG), num_bands=N_MODES,
    resolution=RES, k_points=[mp.Vector3(0, 0, 0)])

fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)
t0 = time.time()
ms.run_tm()
t_mpb = time.time() - t0
os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

freqs_mpb_raw = np.array(ms.all_freqs)[0]
freqs_mpb = freqs_mpb_raw / L_SUPER
print(f"done in {t_mpb:.2f}s, ω=[{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]")

# ── 2. FDFD build ──────────────────────────────────────────────
print("[2/4] FDFD ε + operator build...", end=' ', flush=True)

from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

t0 = time.time()
eps_grid, info = build_supercell_eps(
    lattice_type='square', m=M_IDX, n=N_IDX,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_grid, Ny=N_grid)
L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]), polarization='tm')
t_build = time.time() - t0
del eps_grid
print(f"done in {t_build:.2f}s, nnz={L_op.nnz:,}")

# ── 3. FDFD solve ──────────────────────────────────────────────
print(f"[3/4] FDFD eigensolver (k={N_MODES})...", flush=True)
t0 = time.time()
evals, evecs = eigsh(L_op, k=N_MODES, sigma=SIGMA, which='LM', maxiter=20000, tol=1e-10)
t_solve = time.time() - t0

idx = np.argsort(evals)
evals = evals[idx]
freqs_fdfd = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
print(f"      done in {t_solve:.2f}s, ω=[{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]")
del L_op, evecs

# ── 4. Save & Plot ──────────────────────────────────────────────
print("[4/4] Saving...", end=' ', flush=True)

outdir = os.path.dirname(os.path.abspath(__file__))
np.savez(os.path.join(outdir, 'supercell_3deg_50modes_comparison.npz'),
         freqs_mpb=freqs_mpb, freqs_fdfd=freqs_fdfd,
         evals_fdfd=evals, res=RES, n_modes=N_MODES,
         m=M_IDX, n=N_IDX, L_SUPER=L_SUPER,
         t_mpb=t_mpb, t_fdfd=t_solve)

fig, axes = plt.subplots(1, 3, figsize=(14, 7),
                         gridspec_kw={'width_ratios': [1, 1, 2]})

ax1 = axes[0]
for f in freqs_mpb:
    ax1.plot([0.2, 0.8], [f, f], 'b-', linewidth=0.8, alpha=0.7)
ax1.set_xlim(0, 1); ax1.set_ylabel('ω (a/2πc)')
ax1.set_title(f'MPB ({t_mpb:.1f}s)'); ax1.set_xticks([])
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
for f in freqs_fdfd:
    ax2.plot([0.2, 0.8], [f, f], 'r-', linewidth=0.8, alpha=0.7)
ax2.set_xlim(0, 1); ax2.set_title(f'FDFD ({t_solve:.1f}s)')
ax2.set_xticks([]); ax2.grid(True, alpha=0.3, axis='y')

ymin = min(freqs_mpb.min(), freqs_fdfd.min())
ymax = max(freqs_mpb.max(), freqs_fdfd.max())
margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.01
for ax in [ax1, ax2]:
    ax.set_ylim(ymin - margin, ymax + margin)
ax2.set_yticklabels([])

ax3 = axes[2]
n_c = min(len(freqs_mpb), len(freqs_fdfd))
ms_ = np.sort(freqs_mpb[:n_c])
fs_ = np.sort(freqs_fdfd[:n_c])
ax3.plot(range(n_c), ms_, 'b.-', markersize=4, label='MPB', alpha=0.8)
ax3.plot(range(n_c), fs_, 'r.-', markersize=4, label='FDFD', alpha=0.8)
ax3.set_xlabel('Mode index'); ax3.set_ylabel('ω (a/2πc)')
ax3.set_title('Sorted eigenvalue comparison')
ax3.legend(); ax3.grid(True, alpha=0.3)

fig.suptitle(f'MPB vs FDFD: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.1f}°, '
             f'res={RES}, {N_MODES} TM modes at Γ',
             fontsize=12, fontweight='bold')
fig.tight_layout()
figpath = os.path.join(outdir, 'fig_supercell_3deg_50modes_ladder.png')
fig.savefig(figpath, dpi=150)
print(f"saved {figpath}")

mask = (ms_ > 0.001) & (fs_ > 0.001)
if mask.any():
    rel_err = np.abs(ms_[mask] - fs_[mask]) / ms_[mask]
    print(f"\nAgreement: max={rel_err.max():.4%}, mean={rel_err.mean():.4%}, "
          f"<1%: {np.sum(rel_err<0.01)}/{len(rel_err)}")
