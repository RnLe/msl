"""
MPB vs FDFD supercell comparison: (38,1) at 3.01°, 100 modes.

IMPORTANT: MPB's Python `resolution` parameter sets grid_size = resolution * lattice.size.
With size=(1,1), resolution must be RES * round(L_SUPER) to get RES pixels per monolayer cell.
"""

import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────
M_IDX, N_IDX = 38, 1
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
RES = 32
N_MODES = 100

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)
N_cells = M_IDX**2 + N_IDX**2

print(f"Supercell: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°, "
      f"N_cells={N_cells}, L_super={L_SUPER:.2f}a")
print()

# ── 1. MPB ──────────────────────────────────────────────────────
print("=" * 60)
print("MPB solve")
print("=" * 60)

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

print("  Placing rods...", end='', flush=True)
t0 = time.time()
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
print(f" {len(geometry)} rods in {time.time()-t0:.1f}s")

mpb_res = RES * round(L_SUPER)  # MPB grid = resolution * size; size=(1,1) → need res*L
print(f"  Grid: ~{mpb_res}x{mpb_res}, res={RES}/cell")

mp.verbosity(0)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=N_MODES,
    resolution=mpb_res,
    k_points=[mp.Vector3(0, 0, 0)])

fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)

print(f"  Solving {N_MODES} TM bands...", flush=True)
t0 = time.time()
ms.run_tm()
t_mpb = time.time() - t0

os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

freqs_mpb_raw = np.array(ms.all_freqs)[0]
freqs_mpb = freqs_mpb_raw / L_SUPER

print(f"  [MPB DONE] {t_mpb:.1f}s, ω range: [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}] a/2πc")

# ── 2. FDFD ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("FDFD solve")
print("=" * 60)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

N_grid = RES * round(L_SUPER)

print(f"  Building ε grid ({N_grid}x{N_grid})...", end='', flush=True)
t0 = time.time()
eps_grid, info = build_supercell_eps(
    lattice_type='square', m=M_IDX, n=N_IDX,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_grid, Ny=N_grid)
print(f" {time.time()-t0:.1f}s")

print(f"  Building FDFD operator...", end='', flush=True)
t0 = time.time()
L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                           polarization='tm')
t_build = time.time() - t0
print(f" {t_build:.1f}s (DOF={N_grid**2:,}, nnz={L_op.nnz:,})")
del eps_grid

# Use sigma slightly negative to get the LOWEST eigenvalues (like MPB)
# The operator is PSD (eigenvalues >= 0), so sigma=-0.01 targets the bottom
sigma = -0.01
print(f"  Eigensolver: k={N_MODES}, σ={sigma} (shift-invert targeting lowest modes)")
print(f"  Solving...", flush=True)

t0 = time.time()
evals, evecs = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM',
                     maxiter=20000, tol=1e-10)
t_solve = time.time() - t0

idx = np.argsort(evals)
evals = evals[idx]
freqs_fdfd = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

print(f"  [FDFD DONE] {t_solve:.1f}s (total {t_build+t_solve:.1f}s)")
print(f"  ω range: [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}] a/2πc")

# ── 3. Save data ────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
npz_path = os.path.join(outdir, 'supercell_3deg_100modes_comparison.npz')
np.savez(npz_path,
         freqs_mpb=freqs_mpb, freqs_fdfd=freqs_fdfd,
         evals_fdfd=evals, res=RES, n_modes=N_MODES,
         m=M_IDX, n=N_IDX, L_SUPER=L_SUPER,
         theta_deg=theta_deg, N_cells=N_cells,
         t_mpb=t_mpb, t_fdfd=t_solve)
print(f"\n  Data saved: {npz_path}")

# ── 4. Plot eigenvalue ladders ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 1, 2]})

# Panel 1: MPB ladder
ax1 = axes[0]
for i, f in enumerate(freqs_mpb):
    ax1.plot([0.2, 0.8], [f, f], 'b-', linewidth=0.5, alpha=0.6)
ax1.set_xlim(0, 1)
ax1.set_ylabel('ω (a/2πc)')
ax1.set_title(f'MPB ({t_mpb:.1f}s)\nres={RES}')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3, axis='y')

# Panel 2: FDFD ladder
ax2 = axes[1]
for i, f in enumerate(freqs_fdfd):
    ax2.plot([0.2, 0.8], [f, f], 'r-', linewidth=0.5, alpha=0.6)
ax2.set_xlim(0, 1)
ax2.set_title(f'FDFD ({t_solve:.1f}s)\nres={RES}')
ax2.set_xticks([])
ax2.grid(True, alpha=0.3, axis='y')

# Same y-limits
ymin = min(freqs_mpb[0], freqs_fdfd[0])
ymax = max(freqs_mpb[-1], freqs_fdfd[-1])
margin = 0.02 * (ymax - ymin) if ymax > ymin else 0.01
for ax in [ax1, ax2]:
    ax.set_ylim(ymin - margin, ymax + margin)
ax2.set_yticklabels([])

# Panel 3: Overlay comparison
ax3 = axes[2]
n_compare = min(len(freqs_mpb), len(freqs_fdfd))
mpb_sorted = np.sort(freqs_mpb[:n_compare])
fdfd_sorted = np.sort(freqs_fdfd[:n_compare])

ax3.plot(range(n_compare), mpb_sorted, 'b.-', markersize=3, label='MPB', alpha=0.8)
ax3.plot(range(n_compare), fdfd_sorted, 'r.-', markersize=3, label='FDFD', alpha=0.8)
ax3.set_xlabel('Mode index')
ax3.set_ylabel('ω (a/2πc)')
ax3.set_title('Sorted eigenvalue comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

fig.suptitle(f'MPB vs FDFD: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°, '
             f'N_cells={N_cells}\n'
             f'res={RES}, {N_MODES} TM modes at Γ (lowest modes)',
             fontsize=12, fontweight='bold')
fig.tight_layout()

figpath = os.path.join(outdir, 'fig_supercell_3deg_100modes_ladder.png')
fig.savefig(figpath, dpi=150)
print(f"  Figure saved: {figpath}")

# ── 5. Summary statistics ──────────────────────────────────────
mask = (mpb_sorted > 0.001) & (fdfd_sorted > 0.001)
rel_err = np.abs(mpb_sorted[mask] - fdfd_sorted[mask]) / mpb_sorted[mask]
print()
print("Agreement (excluding zero mode):")
print(f"  Max  relative error: {rel_err.max():.4%}")
print(f"  Mean relative error: {rel_err.mean():.4%}")
print(f"  Median rel error:    {np.median(rel_err):.4%}")
n_good = np.sum(rel_err < 0.01)
print(f"  Modes with <1% error: {n_good}/{len(rel_err)}")
