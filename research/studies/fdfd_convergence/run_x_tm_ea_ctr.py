#!/usr/bin/env python3
"""
FDFD TM solve at X point at the EA bandwidth center frequencies.
Resolution 8 px/cell, 80 modes, all 4 angles.
Each angle uses the mean of its EA frequency window as the shift target.
"""
import gc
import os
import resource
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from scipy.sparse.linalg import eigsh

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.join(
    os.path.dirname(STUDY_DIR), '..',
    'moire_envelope', 'thesis_results')
sys.path.insert(0, os.path.abspath(THESIS_RESULTS))

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps

R_OVER_A    = 0.2
EPS_ROD     = 8.9
EPS_BG      = 1.0
Q_X         = np.array([np.pi, 0.0])
N_MODES     = 80
PX_PER_CELL = 8

# Per-angle EA bandwidth center frequencies (mean of EA eigenvalues)
ANGLES = [
    {'m': 14,  'n': 1, 'label': '8deg',  'sigma_omega': 0.240574},
    {'m': 29,  'n': 1, 'label': '4deg',  'sigma_omega': 0.241110},
    {'m': 57,  'n': 1, 'label': '2deg',  'sigma_omega': 0.240887},
    {'m': 114, 'n': 1, 'label': '1deg',  'sigma_omega': 0.240960},
]

DATA_DIR = os.path.join(STUDY_DIR, 'data_x_tm')
os.makedirs(DATA_DIR, exist_ok=True)


def run_angle(m, n, label, sigma_omega):
    fname = f'fdfd_tm_x_{label}_res{PX_PER_CELL}_fEActr.npz'
    dest = os.path.join(DATA_DIR, fname)
    if os.path.isfile(dest):
        print(f'  [skip] {fname}')
        return

    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    N_grid = PX_PER_CELL * round(L_super)

    print(f'  {label}  res={PX_PER_CELL}  σ_ω={sigma_omega:.6f}  '
          f'grid={N_grid}×{N_grid}  DOF={N_grid**2:,}  k={N_MODES} modes')
    sys.stdout.flush()

    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)

    t0 = time.time()
    L_op = build_fdfd_operator(
        eps_grid, info, q_vec=Q_X, polarization='tm')
    t_assembly = time.time() - t0

    sigma = (2 * np.pi * sigma_omega) ** 2
    t0 = time.time()
    evals, evecs = eigsh(
        L_op, k=N_MODES, sigma=sigma, which='LM',
        maxiter=20_000, tol=1e-10)
    t_solve = time.time() - t0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    evals = np.real_if_close(evals, tol=1000)
    idx = np.argsort(np.asarray(evals, dtype=float))
    evals = np.asarray(evals, dtype=float)[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

    np.savez(dest,
             freqs=freqs, evals=evals,
             grid=N_grid, px_per_cell=PX_PER_CELL,
             m=m, n=n, n_modes=N_MODES,
             sigma_omega=sigma_omega, q_vec=Q_X,
             t_assembly=t_assembly, t_solve=t_solve,
             rss_mb=rss_mb)

    print(f'  ✓ {t_solve:.1f}s  freq=[{freqs[0]:.6f}, {freqs[-1]:.6f}]')
    del L_op, eps_grid, evecs, evals, freqs
    gc.collect()


def main():
    print(f'FDFD TM X-point at EA center freqs, res={PX_PER_CELL}, {N_MODES} modes')
    for i, a in enumerate(ANGLES):
        print(f'\n[{i+1}/{len(ANGLES)}] {a["label"]}')
        run_angle(a['m'], a['n'], a['label'], a['sigma_omega'])
    print('\nDone.')


if __name__ == '__main__':
    main()
