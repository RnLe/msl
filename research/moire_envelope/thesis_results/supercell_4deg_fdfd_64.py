"""
FDFD supercell eigenvalues: (29,1) ~ 3.95°, 64 px/cell, 50 bands.

Canonical square lattice: r/a=0.2, eps_rod=8.9, eps_bg=1.0, TM polarization.
Shift-invert with σ=0.01 (a/2πc).
Saves results to supercell_4deg_fdfd_64.npz.
"""

import os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

# ── Parameters ──────────────────────────────────────────────────
M_IDX, N_IDX = 29, 1
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
PX_PER_CELL = 64
N_MODES = 50
TARGET_OMEGA = 0.01  # shift-invert target in a/2πc

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)  # ~29.017
N_CELLS = M_IDX**2 + N_IDX**2  # 842
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
theta_deg = np.degrees(theta_rad)

N_grid = PX_PER_CELL * round(L_SUPER)  # 64 * 29 = 1856

print(f"Supercell: (m,n)=({M_IDX},{N_IDX}), θ={theta_deg:.2f}°")
print(f"N_cells={N_CELLS}, L_super={L_SUPER:.3f}a")
print(f"Resolution: {PX_PER_CELL} px/cell → grid={N_grid}x{N_grid} = {N_grid**2:,} DOF")
print(f"Bands: {N_MODES}, σ={TARGET_OMEGA} a/2πc")
print()

# ── Build ε grid ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

SMOOTHING_NSUB = 8
print(f"Building ε grid (subpixel smoothing: Nsub={SMOOTHING_NSUB})...")
t0 = time.time()
eps_grid, info = build_supercell_eps(
    lattice_type='square', m=M_IDX, n=N_IDX,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=SMOOTHING_NSUB)
t_eps = time.time() - t0
print(f"  ε grid: {N_grid}x{N_grid} = {N_grid**2:,} DOF, built in {t_eps:.1f}s")

# ── Build FDFD operator ────────────────────────────────────────
print("Building FDFD operator...")
t0 = time.time()
L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                           polarization='tm')
t_build = time.time() - t0
print(f"  Operator: nnz={L_op.nnz:,}, built in {t_build:.1f}s")

del eps_grid  # free ε grid memory
gc.collect()

# ── Shift-invert eigensolver (CHOLMOD preferred) ──────────────
sigma = (2 * np.pi * TARGET_OMEGA) ** 2
N_dof = L_op.shape[0]
print(f"  shift-invert σ = (2π·{TARGET_OMEGA})² = {sigma:.6f}")
print(f"  DOF = {N_dof:,}")

L_shifted = L_op - sigma * sp.eye(N_dof, format='csc')

try:
    from sksparse.cholmod import cholesky
    print(f"  CHOLMOD LDLᵀ factorization...", flush=True)
    t0 = time.time()
    factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
    t_factor = time.time() - t0
    print(f"  Factorization: {t_factor:.1f}s")

    OPinv = LinearOperator((N_dof, N_dof),
                           matvec=lambda b: factor(b),
                           dtype=L_op.dtype)

    print(f"  eigsh: {N_MODES} modes...", flush=True)
    t0 = time.time()
    evals, evecs = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM',
                         OPinv=OPinv, maxiter=20000, tol=1e-10)
    t_solve = time.time() - t0
    t_solve += t_factor  # include factorization in total solve time

    del factor, OPinv
except ImportError:
    print("  WARNING: sksparse not available, falling back to scipy SuperLU "
          "(may OOM for large grids).", flush=True)
    print(f"  Solving for {N_MODES} modes...", flush=True)
    t0 = time.time()
    evals, evecs = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM',
                         maxiter=20000, tol=1e-10)
    t_solve = time.time() - t0

del L_op, L_shifted
gc.collect()

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

idx = np.argsort(evals)
evals = evals[idx]
freqs_fdfd = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

t_total = t_eps + t_build + t_solve
print(f"\nFDFD done: {N_MODES} modes in {t_solve:.1f}s (total {t_total:.1f}s)")
print(f"Peak RSS: {rss_mb:.0f} MB ({rss_mb/1024:.1f} GB)")
print(f"ω range: [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}] a/2πc")

# ── Save ────────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
outfile = os.path.join(outdir, 'supercell_4deg_fdfd_64.npz')
np.savez(outfile,
         freqs_fdfd=freqs_fdfd,
         evals_fdfd=evals,
         px_per_cell=PX_PER_CELL, grid=N_grid,
         n_modes=N_MODES, sigma_omega=TARGET_OMEGA,
         m=M_IDX, n=N_IDX, L_SUPER=L_SUPER,
         N_cells=N_CELLS, theta_deg=theta_deg,
         t_eps=t_eps, t_build=t_build, t_solve=t_solve,
         t_total=t_total, rss_mb=rss_mb)
print(f"Saved → {outfile}")

# ── Plot: FDFD-only eigenvalue ladder ───────────────────────────
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6),
                                gridspec_kw={'width_ratios': [1, 2]})

# Left: horizontal-line ladder
for i, f in enumerate(freqs_fdfd):
    ax1.plot([0.2, 0.8], [f, f], '-', color='#d62728', linewidth=0.8, alpha=0.7)
ax1.set_xlim(0, 1)
ax1.set_xticks([])
ax1.set_ylabel('Frequency  ω  [a / 2πc]')
ax1.set_title('Eigenvalue ladder')
ax1.grid(True, alpha=0.3, axis='y')

# Right: ω vs mode index
ax2.plot(np.arange(1, N_MODES + 1), freqs_fdfd, 'o-', ms=3, lw=0.8,
         color='#d62728', label='FDFD')
ax2.set_xlabel('Mode index')
ax2.set_ylabel('Frequency  ω  [a / 2πc]')
ax2.set_title('Sorted eigenvalues')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, N_MODES + 1)
ax2.legend()

fig.suptitle(f'FDFD TM eigenvalues — ({M_IDX},{N_IDX}) supercell, '
             f'θ={theta_deg:.2f}°, {PX_PER_CELL} px/cell, σ={TARGET_OMEGA}',
             fontsize=11, fontweight='bold')
fig.tight_layout()

figfile = os.path.join(outdir, 'supercell_4deg_fdfd_64.png')
fig.savefig(figfile, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Plot → {figfile}")
