"""Resolution convergence check for moiré miniband frequencies at Gamma."""
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from scipy.sparse.linalg import eigsh

EPS_BG, EPS_HOLE, R_OVER_A = 1.0, 11.56, 0.2
omega_ref = 0.182693
sigma = (2 * np.pi * omega_ref)**2

m, n = 8, 7
N_cells = m*m + m*n + n*n
theta = np.degrees(commensurate_twist_angle('honeycomb', m, n))

k_modes = 20  # focus on the 20 modes nearest omega_ref

print(f"Convergence check: (m,n)=({m},{n}), theta={theta:.3f} deg, N={N_cells}")
print(f"sigma = {sigma:.4f}, k_modes = {k_modes}")
print()

results = {}
for res in [12, 16, 20, 24]:
    Nx = int(round(np.sqrt(N_cells) * res))
    DOF = Nx * Nx
    mem_MB = DOF * 8 / 1e6  # rough sparse solve memory

    print(f"--- res = {res} pts/cell, Nx = {Nx}, DOF = {DOF:,} ---")
    eps, info = build_supercell_eps('honeycomb', m=m, n=n, a=1.0,
                                    r_over_a=R_OVER_A, eps_rod=EPS_HOLE,
                                    eps_bg=EPS_BG, Nx=Nx, Ny=Nx)

    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0

    t0 = time.time()
    evals, evecs = eigsh(L, k=k_modes, sigma=sigma, which='LM',
                         maxiter=5000, tol=1e-8)
    t_solve = time.time() - t0

    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

    results[res] = freqs
    print(f"  Build: {t_op:.1f}s, Solve: {t_solve:.1f}s")
    for i in range(min(20, len(freqs))):
        print(f"  mode {i:>2}: f = {freqs[i]:.6f}")
    print()

# Compare convergence
print("\n=== Convergence Table ===")
resolutions = sorted(results.keys())
header = "mode  " + "  ".join(f"res={r:>2}" for r in resolutions)
print(header)
for i in range(k_modes):
    row = f"  {i:>2}  "
    for r in resolutions:
        row += f"  {results[r][i]:.6f}"
    print(row)

# Relative changes between successive resolutions
print("\n=== Relative change (%) vs finest resolution ===")
finest = resolutions[-1]
header2 = "mode  " + "  ".join(f"res={r:>2}" for r in resolutions[:-1])
print(header2)
for i in range(k_modes):
    row = f"  {i:>2}  "
    for r in resolutions[:-1]:
        delta = abs(results[r][i] - results[finest][i]) / results[finest][i] * 100
        row += f"  {delta:.4f}%"
    print(row)
