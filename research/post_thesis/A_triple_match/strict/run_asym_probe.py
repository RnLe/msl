#!/usr/bin/env python3
"""Go/no-go probe of the ASYMMETRIC candidate (r1=0.20, r2=0.10, eps=8.9):

  probe A: sigma inside the registry-common gap (0.3400) — the returned
           modes must bracket an EMPTY interval = the true global gap of
           the twisted structure (monolayer scan predicts ~[0.323, 0.366]).
  probe B: sigma at the predicted band-1 manifold bottom (0.3670) — the
           states here must show w_X + w_X' ~ 1 (X-star carriers): these
           are the isolated moire envelope states the EA must reproduce.

Usage: run_asym_probe.py [px] [m] [n] [sigmaA] [sigmaB]
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

px = int(sys.argv[1]) if len(sys.argv) > 1 else 16
m = int(sys.argv[2]) if len(sys.argv) > 2 else 57
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SIGMA_A = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3400
SIGMA_B = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3670
R_CUT = 0.45
Q_X = np.array([np.pi, 0.0])

N_grid = px * round(float(np.sqrt(m * m + n * n)))
eps, info = build_bilayer_eps_asym(m, n, r1=0.20, r2=0.10, eps1=8.9, eps2=8.9,
                                   eps_bg=1.0, Nx=N_grid, Ny=N_grid,
                                   smoothing_Nsub=8, cell='centered')
L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
n_dof = L_op.shape[0]
print(f'asym bilayer ({m},{n}) px={px} DOF={n_dof:,}', flush=True)

B_super = np.asarray(info['B_super'], dtype=float)
G = 2 * np.pi * np.linalg.inv(B_super).T
g1 = np.fft.fftfreq(N_grid) * N_grid
G1, G2 = np.meshgrid(g1, g1, indexing='ij')
qx = Q_X[0] + G1 * G[0, 0] + G2 * G[0, 1]
qy = Q_X[1] + G1 * G[1, 0] + G2 * G[1, 1]
qxr = (qx + np.pi) % (2 * np.pi) - np.pi
qyr = (qy + np.pi) % (2 * np.pi) - np.pi
dX = np.hypot(np.pi - np.abs(qxr), qyr)
dXp = np.hypot(qxr, np.pi - np.abs(qyr))

for tag, sig, k, vecs_wanted in [('A(gap)', SIGMA_A, 30, False),
                                 ('B(manifold)', SIGMA_B, 40, True)]:
    t0 = time.time()
    sigma = (2 * np.pi * sig) ** 2
    factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                      beta=0, mode='simplicial')
    op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
    if vecs_wanted:
        vals, vecs = eigsh(L_op, k=k, sigma=sigma, which='LM', OPinv=op_inv,
                           maxiter=20000, tol=1e-10)
    else:
        vals = eigsh(L_op, k=k, sigma=sigma, which='LM', OPinv=op_inv,
                     maxiter=20000, tol=1e-10, return_eigenvectors=False)
        vecs = None
    freqs = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)
    order = np.argsort(freqs)
    print(f'\n--- probe {tag}: sigma={sig}  t={time.time()-t0:.0f}s')
    for i in order:
        line = f'  {freqs[i]:.6f}'
        if vecs is not None:
            P = np.abs(np.fft.fft2(vecs[:, i].reshape(N_grid, N_grid))) ** 2
            P /= P.sum()
            wX = float(P[dX < R_CUT].sum())
            wXp = float(P[dXp < R_CUT].sum())
            lab = 'X ' if wX > 0.5 else ("X'" if wXp > 0.5 else
                                         ('bg' if wX + wXp < 0.3 else 'mix'))
            line += f'  wX={wX:5.3f} wX\'={wXp:5.3f}  {lab}'
        print(line)
