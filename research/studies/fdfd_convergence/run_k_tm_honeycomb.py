#!/usr/bin/env python3
"""
FDFD TM convergence study at the K point for honeycomb Dirac lattice.

Honeycomb lattice: a1=(1,0), a2=(0.5, sqrt(3)/2)
Two-atom basis: (0,0) and (1/3,1/3) fractional
Air holes (eps=1) in dielectric (eps=3.77658), r/a = 0.408425.

K point in Cartesian: Q_K = (2pi/3) * [1, 1/sqrt(3)]

Resolutions: 1, 4, 8 px/cell.
Target freqs: 0.05, 0.1, 0.2, 0.4, 0.749.
Modes per shift: 20.
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

R_OVER_A = 0.408425
EPS_ROD = 1.0       # air holes
EPS_BG = 3.77658    # dielectric background

# Monolayer K-point in Cartesian: Q_K = (1/3)*b1 + (1/3)*b2
# b1 = 2pi*(1, -1/sqrt(3)), b2 = 2pi*(0, 2/sqrt(3))
Q_K = (2.0 * np.pi / 3.0) * np.array([1.0, 1.0 / np.sqrt(3.0)])

# Honeycomb commensurate angles (triangular lattice formula)
# N_cells = m^2 + mn + n^2,  |L1| = sqrt(N_cells)
ANGLES = [
    {'m': 9,  'n': 7,  'label': '8deg'},   # theta = 8.256°, N = 193
    {'m': 9,  'n': 8,  'label': '4deg'},   # theta = 3.890°, N = 217
    {'m': 17, 'n': 16, 'label': '2deg'},   # theta = 2.005°, N = 817
    {'m': 67, 'n': 65, 'label': '1deg'},   # theta = 1.002°, N = 13069
]

RESOLUTIONS = [1, 4, 8]
TARGET_FREQS = [0.05, 0.1, 0.2, 0.4, 0.749]
N_MODES = 20

DATA_DIR = os.path.join(STUDY_DIR, 'data_k_tm_honeycomb')
os.makedirs(DATA_DIR, exist_ok=True)


def freq_tag(f: float) -> str:
    return f'f{f * 1000:04.0f}'


def out_path(label, px, f):
    return os.path.join(DATA_DIR,
                        f'fdfd_tm_k_{label}_res{px}_{freq_tag(f)}.npz')


def run_single(m, n, label, px_per_cell, sigma_omega):
    dest = out_path(label, px_per_cell, sigma_omega)
    if os.path.isfile(dest):
        print(f'  [skip] {os.path.basename(dest)}')
        return True

    # Honeycomb: |L1| = sqrt(m^2 + mn + n^2)
    N_cells = m * m + m * n + n * n
    L_super = np.sqrt(N_cells)
    N_grid = px_per_cell * round(L_super)

    print(f'  {label}  res={px_per_cell}  sigma_omega={sigma_omega}  '
          f'N_cells={N_cells}  grid={N_grid}x{N_grid}  DOF={N_grid**2:,}')
    sys.stdout.flush()

    try:
        eps_grid, info = build_supercell_eps(
            lattice_type='honeycomb', m=m, n=n,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=N_grid, Ny=N_grid,
            subpixel_smoothing=True, smoothing_Nsub=8)

        t0 = time.time()
        L_op = build_fdfd_operator(
            eps_grid, info, q_vec=Q_K, polarization='tm')
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
                 sigma_omega=sigma_omega, q_vec=Q_K,
                 t_assembly=t_assembly, t_solve=t_solve,
                 rss_mb=rss_mb)

        print(f'  -> {t_solve:.1f}s  freq=[{freqs[0]:.6f}, {freqs[-1]:.6f}]')
        del L_op, eps_grid, evecs, evals, freqs
        gc.collect()
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'  FAILED: {exc}')
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
    print(f'K-point honeycomb FDFD TM convergence — {total} runs')
    print(f'Q_K = [{Q_K[0]:.6f}, {Q_K[1]:.6f}]')

    for i, (a, px, f) in enumerate(cases, 1):
        print(f'\n[{i}/{total}]  {a["label"]}  res={px}  f={f}')
        if run_single(a['m'], a['n'], a['label'], px, f):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t_start
    print(f'\n{"="*50}')
    print(f'Done  {ok} ok / {fail} fail  in {elapsed:.0f}s')


if __name__ == '__main__':
    main()
