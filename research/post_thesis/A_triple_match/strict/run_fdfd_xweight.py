#!/usr/bin/env python3
"""FDFD eigenmode classification by X-star Fourier weight — the strict,
matching-free way to separate X-manifold envelope states from background
folded modes at the same supercell k.

For each eigenvector psi on the supercell grid: FFT -> weights |psi_hat|^2 on
q = Q + G_sc; reduce q to the monolayer BZ [-pi,pi)^2; X-star distance
d = min(|q - (pi,0)-star|, |q - (0,pi)-star|). Report, per mode:
  w_X(r_cut): total weight within r_cut of the X-star
  w_vX / w_vXp: split between the X=(pi,0) and X'=(0,pi) stars
  q_peak: monolayer-BZ position of the dominant Fourier component

Usage: run_fdfd_xweight.py [px] [sigma_omega] [n_modes] [m] [n]
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

px = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SIGMA_OMEGA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2266
N_MODES = int(sys.argv[3]) if len(sys.argv) > 3 else 80
m = int(sys.argv[4]) if len(sys.argv) > 4 else 57
n = int(sys.argv[5]) if len(sys.argv) > 5 else 1
R_CUT = 0.45

Q_X = np.array([np.pi, 0.0])

t0 = time.time()
N_grid = px * round(float(np.sqrt(m * m + n * n)))
eps, info = build_supercell_eps(
    lattice_type='square', m=m, n=n, r_over_a=0.2,
    eps_rod=8.9, eps_bg=1.0, Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)
L_op = build_fdfd_operator(eps, info, Q_X, polarization='tm')
sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
n_dof = L_op.shape[0]
print(f'DOF={n_dof:,} factorizing...', flush=True)
factor = cholesky((L_op - sigma * sp.eye(n_dof, format='csc')).tocsc(),
                  beta=0, mode='simplicial')
op_inv = LinearOperator((n_dof, n_dof), matvec=factor, dtype=L_op.dtype)
vals, vecs = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM', OPinv=op_inv,
                   maxiter=20000, tol=1e-10)
freqs = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)
order = np.argsort(freqs)
print(f'solved {N_MODES} modes in {time.time()-t0:.0f}s', flush=True)

# reciprocal setup
B_super = np.asarray(info['B_super'], dtype=float)          # columns L1, L2
G = 2 * np.pi * np.linalg.inv(B_super).T                    # columns b1, b2
Nx = Ny = N_grid
g1 = np.fft.fftfreq(Nx) * Nx                                 # integer indices
g2 = np.fft.fftfreq(Ny) * Ny
G1, G2 = np.meshgrid(g1, g2, indexing='ij')
# CONVENTION (verified on the empty lattice): the eigenvector is the FULL
# Bloch field sampled on the cell grid; DFT bin (k1,k2) corresponds to
# momentum k1*b1 + k2*b2 EXACTLY (the q-offset and fractional bin shift
# cancel identically: sum_i b_i (q.L_i)/2pi = q). Do NOT add Q.
qx = G1 * G[0, 0] + G2 * G[0, 1]
qy = G1 * G[1, 0] + G2 * G[1, 1]
# reduce to monolayer BZ [-pi, pi)^2
qxr = (qx + np.pi) % (2 * np.pi) - np.pi
qyr = (qy + np.pi) % (2 * np.pi) - np.pi
dX = np.hypot(np.pi - np.abs(qxr), qyr)        # distance to (pi,0) star
dXp = np.hypot(qxr, np.pi - np.abs(qyr))       # distance to (0,pi) star

rows = []
for i in order:
    psi = vecs[:, i].reshape(Nx, Ny)
    P = np.abs(np.fft.fft2(psi)) ** 2
    P /= P.sum()
    wX = float(P[dX < R_CUT].sum())
    wXp = float(P[dXp < R_CUT].sum())
    jpk = np.unravel_index(np.argmax(P), P.shape)
    rows.append((float(freqs[i]), wX, wXp,
                 float(qxr[jpk]), float(qyr[jpk])))

print(f'\n r_cut={R_CUT}   f        w_X     w_X\'    q_peak(monolayer BZ)')
for f, wX, wXp, qpx, qpy in rows:
    lab = 'X ' if wX > 0.5 else ("X'" if wXp > 0.5 else
                                 ('bg' if wX + wXp < 0.3 else '??'))
    print(f'  {f:.6f}  {wX:5.3f}  {wXp:5.3f}   ({qpx:+.3f},{qpy:+.3f})  {lab}')

out = os.path.join(HERE, f'fdfd_xweight_m{m}_px{px}_s{SIGMA_OMEGA}.npz')
np.savez(out, freqs=freqs[order],
         wX=np.array([r[1] for r in rows]),
         wXp=np.array([r[2] for r in rows]),
         qpeak=np.array([(r[3], r[4]) for r in rows]),
         px=px, m=m, n=n, sigma_omega=SIGMA_OMEGA, r_cut=R_CUT)
print('saved', out)
