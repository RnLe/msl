#!/usr/bin/env python3
"""Band-edge (spectral bottom) FDFD reference at 2° — the EA-favorable window.

The March sprint targeted f≈0.241 (the MEAN of the Λ0 registry landscape,
semiclassically dense interior, level spacing ~1e-5). The EA's clean regime
is the spectrum bottom near min(Λ0)=0.2260, where levels are sparse and
envelopes smooth. This solves the lowest modes of the (57,1) supercell.
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

FDFD_STUDY = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'studies', 'fdfd_convergence'))
DATA_DIR = os.path.join(FDFD_STUDY, 'data_x_tm')

Q_X = np.array([np.pi, 0.0])
N_MODES = 40
SIGMA_OMEGA = 0.2270      # just above min(Lambda0) = 0.2260

RUNS = [
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 16},
    {'m': 57, 'n': 1, 'label': '2deg', 'px': 32},
]

for r in RUNS:
    m, n, label, px = r['m'], r['n'], r['label'], r['px']
    fname = f'fdfd_tm_x_{label}_res{px}_fBOTTOM.npz'
    dest = os.path.join(DATA_DIR, fname)
    if os.path.isfile(dest):
        print(f'[skip] {fname}')
        continue
    t0 = time.time()
    N_grid = px * round(float(np.sqrt(m * m + n * n)))
    eps, info = build_supercell_eps(
        lattice_type='square', m=m, n=n, r_over_a=0.2,
        eps_rod=8.9, eps_bg=1.0, Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)
    L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
    sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
    n_dof = L_op.shape[0]
    print(f'{label} res={px}: DOF={n_dof:,} factorizing...', flush=True)
    factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                      beta=0, mode='simplicial')
    op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
    vals = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM', OPinv=op_inv,
                 maxiter=20000, tol=1e-10, return_eigenvectors=False)
    freqs = np.sort(np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi))
    np.savez(dest, freqs=freqs, m=m, n=n, px=px,
             sigma_omega=SIGMA_OMEGA, n_modes=N_MODES)
    print(f'saved {fname}  f=[{freqs.min():.6f},{freqs.max():.6f}]  '
          f't={time.time()-t0:.0f}s', flush=True)
