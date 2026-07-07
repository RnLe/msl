#!/usr/bin/env python3
"""FDFD reference ladders for the ASYM candidate (r1=0.20, r2=0.10, eps 8.9),
band-1 gap-edge manifold, centered cell at Q_X. px ladder for Richardson.
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
from sksparse.cholmod import cholesky

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'moire_envelope', 'thesis_results'))
sys.path.insert(0, THESIS_RESULTS)
sys.path.insert(0, HERE)

from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from supercell_asym import build_bilayer_eps_asym  # noqa: E402

Q_X = np.array([np.pi, 0.0])
RUNS = [
    # 2 deg (57,1): manifold bottom ~0.3700 at px16
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 16, 'sigma': 0.3670, 'k': 60},
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 32, 'sigma': 0.3670, 'k': 60},
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 48, 'sigma': 0.3670, 'k': 60},
    # 1 deg (113,1): zero-point halves -> bottom ~0.3681; denser ladder
    {'m': 113, 'n': 1, 'label': '1deg', 'px': 16, 'sigma': 0.3672, 'k': 80},
    {'m': 113, 'n': 1, 'label': '1deg', 'px': 24, 'sigma': 0.3672, 'k': 80},
]

for r in RUNS:
    m, n, label, px = r['m'], r['n'], r['label'], r['px']
    fname = f'fdfd_asym_x_{label}_res{px}.npz'
    dest = os.path.join(HERE, fname)
    if os.path.isfile(dest):
        print(f'[skip] {fname}')
        continue
    t0 = time.time()
    N_grid = px * round(float(np.sqrt(m * m + n * n)))
    eps, info = build_bilayer_eps_asym(m, n, r1=0.20, r2=0.10, eps1=8.9,
                                       eps2=8.9, eps_bg=1.0,
                                       Nx=N_grid, Ny=N_grid,
                                       smoothing_Nsub=8, cell='centered')
    L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
    sigma = (2 * np.pi * r['sigma']) ** 2
    n_dof = L_op.shape[0]
    print(f'{label} px{px}: DOF={n_dof:,} factorizing...', flush=True)
    factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                      beta=0, mode='simplicial')
    op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
    vals = eigsh(L_op, k=r['k'], sigma=sigma, which='LM', OPinv=op_inv,
                 maxiter=20000, tol=1e-10, return_eigenvectors=False)
    freqs = np.sort(np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi))
    np.savez(dest, freqs=freqs, m=m, n=n, px=px, sigma_omega=r['sigma'],
             n_modes=r['k'], r1=0.20, r2=0.10, eps=8.9)
    print(f'saved {fname}  f=[{freqs.min():.6f},{freqs.max():.6f}]  '
          f't={time.time()-t0:.0f}s', flush=True)
