#!/usr/bin/env python3
"""FDFD leg of the MPB<->FDFD cross-check at 4 deg (m,n)=(29,1).

The March MPB supercell lane (mpb_tm_x_4deg_res64_20bands.npz) ran at
supercell FRACTIONAL k = (0.5, 0) — NOT the fold of the monolayer X point
(which is (0.5, 0.5): e^{iQ_X·L1,2} = (e^{i pi m}, e^{-i pi n}) = (-1,-1)).
For an exact same-cell same-k cross-check we therefore solve FDFD at MPB's
k: q = 0.5 * b1_super (Cartesian), targeting the 20 lowest supercell modes
(MPB window f = 0.0102..0.0507 c/a). Index-aligned from mode 1 — the one
window where NO selection ambiguity exists at all.
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

N_MODES = 30
SIGMA_OMEGA = 0.008       # just below MPB's lowest mode 0.010185 c/a

RUNS = [
    {'m': 29, 'n': 1, 'label': '4deg', 'px': 16},
    {'m': 29, 'n': 1, 'label': '4deg', 'px': 32},
    {'m': 29, 'n': 1, 'label': '4deg', 'px': 64},
]

for r in RUNS:
    m, n, label, px = r['m'], r['n'], r['label'], r['px']
    fname = f'fdfd_tm_x_{label}_res{px}_fMPBK.npz'
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
    B_super = np.asarray(info['B_super'], dtype=float)   # columns L1, L2
    G = 2 * np.pi * np.linalg.inv(B_super).T             # columns b1, b2
    q_vec = 0.5 * G[:, 0]                                # MPB k = (0.5, 0) frac
    L_op = build_fdfd_operator(eps, info, q_vec, polarization='tm')
    sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
    n_dof = L_op.shape[0]
    print(f'{label} res={px}: DOF={n_dof:,} q={q_vec} factorizing...', flush=True)
    factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                      beta=0, mode='simplicial')
    op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
    vals = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM', OPinv=op_inv,
                 maxiter=20000, tol=1e-10, return_eigenvectors=False)
    freqs = np.sort(np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi))
    np.savez(dest, freqs=freqs, m=m, n=n, px=px, q_vec=q_vec,
             k_frac=np.array([0.5, 0.0]),
             sigma_omega=SIGMA_OMEGA, n_modes=N_MODES)
    print(f'saved {fname}  f=[{freqs.min():.6f},{freqs.max():.6f}]  '
          f't={time.time()-t0:.0f}s', flush=True)
