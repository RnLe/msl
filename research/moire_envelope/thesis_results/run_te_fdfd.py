#!/usr/bin/env python3
"""
FDFD TE-polarization eigensolve at Gamma for 8°, 3°, and 1° Moiré supercells.
Resolution = 4 px/cell.
Uses build_fdfd_operator with polarization='te' (standard-form SPD).
"""
import os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

CWD = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
sys.path.insert(0, CWD)

from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator

CASES = [
    {'m': 14, 'n': 1, 'label': '8deg', 'n_modes': 30, 'sigma_omega': 0.02},
    {'m': 38, 'n': 1, 'label': '3deg', 'n_modes': 50, 'sigma_omega': 0.02},
    {'m': 114, 'n': 1, 'label': '1deg', 'n_modes': 50, 'sigma_omega': 0.02},
]
PX_PER_CELL = 4
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0

for case in CASES:
    m, n = case['m'], case['n']
    n_modes = case['n_modes']
    sigma_omega = case['sigma_omega']
    label = case['label']

    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_deg = np.degrees(2 * np.arctan2(n, m))
    N_grid = PX_PER_CELL * round(L_super)

    print(f"\n{'='*60}")
    print(f"FDFD TE — {label} (m={m}, n={n}), θ={theta_deg:.2f}°")
    print(f"Resolution: {PX_PER_CELL} px/cell → grid {N_grid}×{N_grid}")
    print(f"Modes: {n_modes}, σ_ω={sigma_omega}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Build dielectric grid
    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)

    # Build TE operator (standard form, SPD)
    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, info,
                               q_vec=np.array([0.0, 0.0]),
                               polarization='te')
    t_assembly = time.time() - t0
    print(f"Assembly: {t_assembly:.1f}s, nnz={L_op.nnz:,}, DOF={L_op.shape[0]:,}")

    # Shift-invert eigsh (standard form: L x = λ x)
    sigma = (2 * np.pi * sigma_omega) ** 2

    t0 = time.time()
    evals, evecs = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                         maxiter=20000, tol=1e-10)
    t_solve = time.time() - t0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    evals = np.sort(evals)
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

    print(f"Solved in {t_solve:.1f}s | RSS: {rss_mb:.0f} MB")
    print(f"ω range: [{freqs[0]:.6f}, {freqs[-1]:.6f}] a/2πc")

    out = os.path.join(CWD, f"fdfd_te_{label}_res{PX_PER_CELL}.npz")
    np.savez(out, freqs=freqs, evals=evals, grid=N_grid,
             px_per_cell=PX_PER_CELL, m=m, n=n, n_modes=n_modes,
             sigma_omega=sigma_omega, t_assembly=t_assembly,
             t_solve=t_solve, rss_mb=rss_mb)
    print(f"Saved → {out}")

    del L_op, eps_grid, evecs
    gc.collect()

print("\nAll FDFD TE runs complete.")
