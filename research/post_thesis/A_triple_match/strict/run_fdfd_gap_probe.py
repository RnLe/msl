#!/usr/bin/env python3
"""Decisive probe: is the 2 deg FDFD spectral gap [0.2265, 0.2267] real?

EA (Nb2+6rem AND Nb8) puts ~11 levels inside it; FDFD px16/32/48 all show it
empty. Rule out any shift-invert capture pathology by targeting sigma INSIDE
the gap: the 20 returned modes must bracket the gap (nearest levels at
~0.22649 and ~0.22672), with nothing in between.
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

from T_direct_validation.fdfd_solver import build_fdfd_operator  # noqa: E402
from T_direct_validation.supercell_geometry import build_supercell_eps  # noqa: E402

Q_X = np.array([np.pi, 0.0])
SIGMA_OMEGA = 0.22655        # inside the putative gap
N_MODES = 20
px, m, n = 32, 57, 1

t0 = time.time()
N_grid = px * round(float(np.sqrt(m * m + n * n)))
eps, info = build_supercell_eps(
    lattice_type='square', m=m, n=n, r_over_a=0.2,
    eps_rod=8.9, eps_bg=1.0, Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)
L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
n_dof = L_op.shape[0]
factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                  beta=0, mode='simplicial')
op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
vals = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM', OPinv=op_inv,
             maxiter=20000, tol=1e-10, return_eigenvectors=False)
freqs = np.sort(np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi))
print(f'sigma={SIGMA_OMEGA} px={px}  t={time.time()-t0:.0f}s')
for f in freqs:
    tag = '   <-- IN GAP' if 0.22650 < f < 0.22670 else ''
    print(f'  {f:.6f}{tag}')
