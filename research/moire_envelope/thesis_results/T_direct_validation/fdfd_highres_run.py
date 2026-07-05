"""
FDFD high-resolution run at res=60 for convergence study.
(30,29) supercell, ~9.4M DOF, CHOLMOD-accelerated.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from sksparse.cholmod import cholesky

out_dir = os.path.dirname(os.path.abspath(__file__))

M, N_mn = 30, 29
RES = 60
N_FDFD_MODES = 100
N_cells = M*M + M*N_mn + N_mn*N_mn
sqrt_N = np.sqrt(N_cells)
Nx = int(round(sqrt_N * RES))
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N_mn))

# Load envelope window center
with open('/home/renlephy/msl/research/moire_envelope/runsV3/'
          'thesis_honeycomb_K_b1_20260307_171424/'
          'eta_sweep_20260310_191610/sweep_results.json') as f:
    env_bh = json.load(f)[0]
env_freqs = np.sort(env_bh['omega_ref'] + np.array(env_bh['eigenvalues']))
env_center = 0.5 * (env_freqs.min() + env_freqs.max())
sigma = (2 * np.pi * env_center) ** 2

print(f"{'='*70}")
print(f"FDFD HIGH-RESOLUTION RUN")
print(f"(m,n)=({M},{N_mn}), θ={theta_deg:.4f}°, res={RES}")
print(f"Nx={Nx}, DOF={Nx*Nx:,}")
print(f"σ target: ω={env_center:.6f}")
print(f"{'='*70}\n")

# Build epsilon grid
t0 = time.time()
eps, info = build_supercell_eps(
    'honeycomb', m=M, n=N_mn, a=1.0,
    r_over_a=0.2, eps_rod=11.56, eps_bg=1.0,
    Nx=Nx, Ny=Nx,
)
print(f"Epsilon grid built: {time.time()-t0:.1f}s")

# Build operator
t0 = time.time()
L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
print(f"Operator built: {time.time()-t0:.1f}s, nnz={L.nnz:,}")
del eps  # free
gc.collect()

# Shift-invert
N_dof = Nx * Nx
L_shifted = (L - sigma * sp.eye(N_dof, format='csc')).tocsc()

print("CHOLMOD factorization...")
t0 = time.time()
factor = cholesky(L_shifted, beta=0, mode='simplicial')
t_factor = time.time() - t0
print(f"  Factorization: {t_factor:.1f}s")

del L_shifted  # free shifted matrix
gc.collect()

OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L.dtype)

print(f"eigsh ({N_FDFD_MODES} modes)...")
t0 = time.time()
evals, _ = eigsh(L, k=N_FDFD_MODES, sigma=sigma, which='LM',
                 OPinv=OPinv, maxiter=10000, tol=1e-8)
t_solve = time.time() - t0
print(f"  Eigensolver: {t_solve:.1f}s")
print(f"  Total: {t_factor + t_solve:.1f}s")

# Process
idx = np.argsort(evals)
evals = evals[idx]
fdfd_freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

# Save
fname = os.path.join(out_dir, f'fdfd_dirac_m{M}_n{N_mn}_res{RES}_v2.npz')
np.savez(fname, freqs=fdfd_freqs, evals=evals,
         m=M, n=N_mn, N_cells=N_cells, res=RES, Nx=Nx,
         n_modes=N_FDFD_MODES, omega_target=env_center,
         theta_deg=theta_deg, t_factor=t_factor, t_solve=t_solve)
print(f"\nSaved {os.path.basename(fname)}")
print(f"Freq range: [{fdfd_freqs.min():.6f}, {fdfd_freqs.max():.6f}]")

# Quick comparison
env_min, env_max = env_freqs.min(), env_freqs.max()
in_env = np.sum((fdfd_freqs >= env_min) & (fdfd_freqs <= env_max))
print(f"Modes in envelope window: {in_env}")
