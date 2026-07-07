#!/usr/bin/env python3
"""Where do the asym-candidate gap-edge states carry their momentum?
Top Fourier peaks + BZ-region weight budget (Gamma/X/X'/M disks) per mode.
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

px, m, n = 16, 57, 1
SIG = 0.3670
K = 16
R_CUT = 0.45
Q_X = np.array([np.pi, 0.0])

N_grid = px * round(float(np.sqrt(m * m + n * n)))
eps, info = build_bilayer_eps_asym(m, n, r1=0.20, r2=0.10, eps1=8.9, eps2=8.9,
                                   eps_bg=1.0, Nx=N_grid, Ny=N_grid,
                                   smoothing_Nsub=8, cell='centered')
L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
n_dof = L_op.shape[0]
sigma = (2 * np.pi * SIG) ** 2
factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                  beta=0, mode='simplicial')
op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
t0 = time.time()
vals, vecs = eigsh(L_op, k=K, sigma=sigma, which='LM', OPinv=op_inv,
                   maxiter=20000, tol=1e-10)
freqs = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)
order = np.argsort(freqs)
print(f'solved {K} in {time.time()-t0:.0f}s')

B_super = np.asarray(info['B_super'], dtype=float)
G = 2 * np.pi * np.linalg.inv(B_super).T
g1 = np.fft.fftfreq(N_grid) * N_grid
G1, G2 = np.meshgrid(g1, g1, indexing='ij')
# CONVENTION (verified on the empty lattice): full Bloch field sampled;
# DFT bin (k1,k2) <-> momentum k1*b1 + k2*b2 exactly (offsets cancel).
qx = G1 * G[0, 0] + G2 * G[0, 1]
qy = G1 * G[1, 0] + G2 * G[1, 1]
qxr = (qx + np.pi) % (2 * np.pi) - np.pi
qyr = (qy + np.pi) % (2 * np.pi) - np.pi

STARS = {
    'Γ': lambda ax, ay: np.hypot(ax, ay),
    'X': lambda ax, ay: np.hypot(np.pi - np.abs(ax), ay),
    "X'": lambda ax, ay: np.hypot(ax, np.pi - np.abs(ay)),
    'M': lambda ax, ay: np.hypot(np.pi - np.abs(ax), np.pi - np.abs(ay)),
}
D = {k: f(qxr, qyr) for k, f in STARS.items()}

print("   f         wΓ     wX     wX'    wM     top-3 peaks (reduced BZ)")
for i in order:
    P = np.abs(np.fft.fft2(vecs[:, i].reshape(N_grid, N_grid))) ** 2
    P /= P.sum()
    ws = {k: float(P[d < R_CUT].sum()) for k, d in D.items()}
    flat = np.argsort(P.ravel())[::-1][:3]
    pk = " ".join(f"({qxr.ravel()[j]:+.2f},{qyr.ravel()[j]:+.2f}):{P.ravel()[j]:.2f}"
                  for j in flat)
    print(f'  {freqs[i]:.6f}  {ws["Γ"]:.3f}  {ws["X"]:.3f}  {ws["X\'"]:.3f}'
          f'  {ws["M"]:.3f}   {pk}')
