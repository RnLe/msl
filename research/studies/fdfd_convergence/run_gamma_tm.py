#!/usr/bin/env python3
"""
FDFD TM convergence study at the Γ point (q = 0).

Same crystal & angles as the X-point study, but q_vec = (0, 0).
Resolutions: 1, 4, 8 px/cell.  Target freqs: 0.01, 0.05, 0.1, 0.2, 0.3, 0.4.
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

R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
Q_GAMMA = np.array([0.0, 0.0])

ANGLES = [
    {'m': 14,  'n': 1, 'label': '8deg'},
    {'m': 29,  'n': 1, 'label': '4deg'},
    {'m': 57,  'n': 1, 'label': '2deg'},
    {'m': 114, 'n': 1, 'label': '1deg'},
]

RESOLUTIONS = [1, 4, 8]
TARGET_FREQS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
N_MODES = 20

DATA_DIR = os.path.join(STUDY_DIR, 'data_gamma_tm')
os.makedirs(DATA_DIR, exist_ok=True)


def freq_tag(f: float) -> str:
    return f'f{f * 100:03.0f}'


def out_path(label, px, f):
    return os.path.join(DATA_DIR,
                        f'fdfd_tm_gamma_{label}_res{px}_{freq_tag(f)}.npz')


def run_single(m, n, label, px_per_cell, sigma_omega):
    dest = out_path(label, px_per_cell, sigma_omega)
    if os.path.isfile(dest):
        print(f'  [skip] {os.path.basename(dest)}')
        return True

    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    N_grid = px_per_cell * round(L_super)

    print(f'  {label}  res={px_per_cell}  σ_ω={sigma_omega}  '
          f'grid={N_grid}×{N_grid}  DOF={N_grid**2:,}')
    sys.stdout.flush()

    try:
        eps_grid, info = build_supercell_eps(
            lattice_type='square', m=m, n=n,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=N_grid, Ny=N_grid,
            subpixel_smoothing=True, smoothing_Nsub=8)

        t0 = time.time()
        L_op = build_fdfd_operator(
            eps_grid, info, q_vec=Q_GAMMA, polarization='tm')
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
                 grid=N_grid, px_per_cell=px_per_cell,
                 m=m, n=n, n_modes=N_MODES,
                 sigma_omega=sigma_omega, q_vec=Q_GAMMA,
                 t_assembly=t_assembly, t_solve=t_solve,
                 rss_mb=rss_mb)

        print(f'  ✓ {t_solve:.1f}s  freq=[{freqs[0]:.6f}, {freqs[-1]:.6f}]')
        del L_op, eps_grid, evecs, evals, freqs
        gc.collect()
        return True
    except Exception as exc:
        print(f'  ✗ FAILED: {exc}')
        gc.collect()
        return False


def main():
    cases = [(a, px, f)
             for a in ANGLES
             for px in RESOLUTIONS
             for f in TARGET_FREQS]
    total = len(cases)
    ok, fail = 0, 0
    t_start = time.time()
    print(f'Γ-point FDFD TM convergence — {total} runs')

    for i, (a, px, f) in enumerate(cases, 1):
        print(f'\n[{i}/{total}]  {a["label"]}  res={px}  f={f}')
        if run_single(a['m'], a['n'], a['label'], px, f):
            ok += 1
        else:
            fail += 1

    print(f'\nDone.  {ok} succeeded, {fail} failed, '
          f'{time.time() - t_start:.0f}s total')


if __name__ == '__main__':
    main()
