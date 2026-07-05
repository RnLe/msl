"""
MPB vs FDFD supercell: (38,1) ~3.01°, res=64/cell, 100 TM modes at Γ.
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'  # no GUI backend
import numpy as np

M_IDX, N_IDX = 38, 1
R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
RES = 64
N_MODES = 100
TARGET_OMEGA = 0.053
SIGMA = (2 * np.pi * TARGET_OMEGA) ** 2

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)
N_cells = M_IDX**2 + N_IDX**2
N_grid = RES * round(L_SUPER)

print(f"(m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°, N_cells={N_cells}, "
      f"grid={N_grid}², DOF={N_grid**2:,}, res={RES}/cell, {N_MODES} modes")

# ── MPB ─────────────────────────────────────────────────────────
print("\n[1/3] MPB...", flush=True)
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

freqs_mpb = np.sort(np.array(ms.all_freqs)[0] / L_SUPER)
print(f"  MPB done: {t_mpb:.1f}s, ω=[{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]", flush=True)

# Free MPB objects
del ms, geometry, lattice

# ── FDFD ────────────────────────────────────────────────────────
print("[2/3] FDFD...", flush=True)
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
del eps_grid  # free grid RAM
print(f"  Build: {t_build:.1f}s, nnz={L_op.nnz:,}", flush=True)

t0 = time.time()
evals, _ = eigsh(L_op, k=N_MODES, sigma=SIGMA, which='LM', maxiter=20000, tol=1e-10)
t_solve = time.time() - t0
del L_op  # free operator RAM immediately
freqs_fdfd = np.sort(np.sqrt(np.maximum(evals, 0)) / (2 * np.pi))
print(f"  Solve: {t_solve:.1f}s, ω=[{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]", flush=True)

# ── Save ────────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
datapath = os.path.join(outdir, 'supercell_3deg_res64_100modes.npz')
np.savez(datapath,
    freqs_mpb=freqs_mpb, freqs_fdfd=freqs_fdfd,
    res=RES, n_modes=N_MODES, m=M_IDX, n=N_IDX,
    L_SUPER=L_SUPER, theta_deg=theta_deg,
    t_mpb=t_mpb, t_fdfd=t_solve)
print(f"  Data saved: {datapath}", flush=True)

# ── Plot ────────────────────────────────────────────────────────
print("[3/3] Plotting...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 8),
                         gridspec_kw={'width_ratios': [1, 1, 2]})

for i, f in enumerate(freqs_mpb):
    axes[0].plot([0.2, 0.8], [f, f], 'b-', lw=0.8, alpha=0.7)
for i, f in enumerate(freqs_fdfd):
    axes[1].plot([0.2, 0.8], [f, f], 'r-', lw=0.8, alpha=0.7)

ymin = min(freqs_mpb[0], freqs_fdfd[0])
ymax = max(freqs_mpb[-1], freqs_fdfd[-1])
margin = 0.02 * (ymax - ymin)
for i, (ax, title) in enumerate(zip(axes[:2], [f'MPB ({t_mpb:.1f}s)', f'FDFD ({t_solve:.1f}s)'])):
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')
axes[0].set_ylabel('ω (a/2πc)')
axes[1].set_yticklabels([])

n = min(len(freqs_mpb), len(freqs_fdfd))
axes[2].plot(range(n), freqs_mpb[:n], 'b.-', ms=3, label='MPB', alpha=0.8)
axes[2].plot(range(n), freqs_fdfd[:n], 'r.-', ms=3, label='FDFD', alpha=0.8)
axes[2].set_xlabel('Mode index'); axes[2].set_ylabel('ω (a/2πc)')
axes[2].set_title('Sorted eigenvalue comparison')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

fig.suptitle(f'MPB vs FDFD: ({M_IDX},{N_IDX}), θ={theta_deg:.1f}°, '
             f'res={RES}, {N_MODES} TM modes at Γ', fontsize=12, fontweight='bold')
fig.tight_layout()
figpath = os.path.join(outdir, 'fig_supercell_3deg_res64_100modes.png')
fig.savefig(figpath, dpi=150)
plt.close(fig)
print(f"  Plot saved: {figpath}")

# Stats
mask = (freqs_mpb[:n] > 1e-4) & (freqs_fdfd[:n] > 1e-4)
rel_err = np.abs(freqs_mpb[:n][mask] - freqs_fdfd[:n][mask]) / freqs_mpb[:n][mask]
print(f"\nAgreement: max={rel_err.max():.2%}, mean={rel_err.mean():.2%}, "
      f"<1%: {(rel_err<0.01).sum()}/{len(rel_err)}")
print("Done.")
