#!/usr/bin/env python3
"""Deep FDFD reference rungs for the strict campaign (square TM at X).

Extends the final-sprint ladder (run_x_tm_hires.py conventions, identical
solver/smoothing: build_supercell_eps Nsub=8 arithmetic, build_fdfd_operator,
shift-invert eigsh) with:
    2deg (57,1)  at 32 px/a   (3.33M DOF)
    1deg (114,1) at 16 px/a   (3.33M DOF)
Outputs land next to the originals in studies/fdfd_convergence/data_x_tm/.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
FDFD_STUDY = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'studies', 'fdfd_convergence'))
THESIS_RESULTS = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'moire_envelope', 'thesis_results'))
sys.path.insert(0, THESIS_RESULTS)

from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from T_direct_validation.supercell_geometry import build_supercell_eps  # noqa: E402

R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
Q_X = np.array([np.pi, 0.0])
N_MODES = 80

RUNS = [
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 32, 'sigma_omega': 0.240887},
    {'m': 114, 'n': 1, 'label': '1deg', 'px': 16, 'sigma_omega': 0.240956},
]

DATA_DIR = os.path.join(FDFD_STUDY, 'data_x_tm')


def run_one(m, n, label, px, sigma_omega):
    fname = f'fdfd_tm_x_{label}_res{px}_fEActr.npz'
    dest = os.path.join(DATA_DIR, fname)
    if os.path.isfile(dest):
        print(f'[skip] {fname}')
        return
    t0 = time.time()
    L1 = np.array([m, n], dtype=float)
    N_grid = px * round(float(np.sqrt(L1 @ L1)))
    eps, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)
    L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
    sigma = (2 * np.pi * sigma_omega) ** 2
    n_dof = L_op.shape[0]
    print(f'{label} res={px}: DOF={n_dof:,}  CHOLMOD factorizing...', flush=True)
    factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                      beta=0, mode='simplicial')
    op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
    print(f'{label}: factorized in {time.time()-t0:.0f}s, eigsh...', flush=True)
    vals = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM', OPinv=op_inv,
                 maxiter=20000, tol=1e-10, return_eigenvectors=False)
    freqs = np.sort(np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi))
    np.savez(dest, freqs=freqs, m=m, n=n, px=px,
             sigma_omega=sigma_omega, n_modes=N_MODES,
             elapsed_s=time.time() - t0)
    print(f'saved {fname}  f=[{freqs.min():.6f},{freqs.max():.6f}]  '
          f't={time.time()-t0:.0f}s')


if __name__ == '__main__':
    for r in RUNS:
        run_one(r['m'], r['n'], r['label'], r['px'], r['sigma_omega'])
