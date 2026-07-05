#!/usr/bin/env python3
"""Run a commensurate square-lattice FDFD supercell solve near band 3 at M."""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from T_direct_validation.commensurate_utils import commensurate_twist_angle
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps


A = 1.0
R_OVER_A = 0.2
EPS_ROD = 11.56
EPS_BG = 1.0
OMEGA0 = 0.68457
SIGMA = (2.0 * np.pi * OMEGA0) ** 2

CASES = {
    '10deg': {'m': 11, 'n': 1},
    '7deg':  {'m': 17, 'n': 1},
    '4deg':  {'m': 29, 'n': 1},
    '2deg':  {'m': 57, 'n': 1},
}


def log(message):
    print(message, flush=True)


def default_output(case_name, resolution, n_modes):
    return SCRIPT_DIR / f'fdfd_square_{case_name}_res{resolution}_k{n_modes}.npz'


def solve_fdfd(case_name, resolution, n_modes, output_path, force):
    case = CASES[case_name]
    m_idx = case['m']
    n_idx = case['n']
    n_cells = m_idx ** 2 + n_idx ** 2
    l_super = math.sqrt(n_cells)
    nx = int(round(l_super * resolution))
    dof = nx * nx
    theta_deg = math.degrees(commensurate_twist_angle('square', m_idx, n_idx))

    if output_path.exists() and not force:
        log(f'Output already exists: {output_path}')
        return output_path

    log('=' * 72)
    log(f'Square commensurate FDFD: {case_name}  (m,n)=({m_idx},{n_idx})')
    log(f'  theta = {theta_deg:.4f} deg, N_cells = {n_cells}, L = {l_super:.4f}')
    log(f'  res = {resolution} px/cell, Nx = {nx}, DOF = {dof:,}')
    log(f'  sigma = {SIGMA:.6f}, n_modes = {n_modes}')
    log(f'  output = {output_path}')
    log('=' * 72)

    t0 = time.time()
    eps, info = build_supercell_eps(
        'square', m_idx, n_idx, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=nx, Ny=nx,
    )
    t_eps = time.time() - t0
    log(f'Built epsilon grid: {eps.shape}, range=[{eps.min():.2f}, {eps.max():.2f}], t={t_eps:.1f}s')

    t0 = time.time()
    operator = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    log(f'Built operator: nnz={operator.nnz:,}, dtype={operator.dtype}, t={t_op:.1f}s')

    del eps
    gc.collect()

    shifted = operator - SIGMA * sp.eye(dof, format='csc', dtype=operator.dtype)

    try:
        from sksparse.cholmod import cholesky

        t0 = time.time()
        factor = cholesky(shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        log(f'CHOLMOD factorization: t={t_factor:.1f}s')

        op_inv = LinearOperator(operator.shape, matvec=lambda vec: factor(vec), dtype=operator.dtype)
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=SIGMA, which='LM', OPinv=op_inv, maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0
    except ImportError:
        log('CHOLMOD unavailable, falling back to scipy shift-invert')
        t_factor = 0.0
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=SIGMA, which='LM', maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0

    del operator, shifted
    gc.collect()

    evals = np.sort(evals)
    freqs = np.sqrt(np.maximum(evals, 0.0)) / (2.0 * np.pi)
    log(f'Eigensolve: t={t_solve:.1f}s, freq range=[{freqs.min():.6f}, {freqs.max():.6f}]')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        freqs=freqs,
        evals=evals,
        case=case_name,
        m=m_idx,
        n=n_idx,
        theta_deg=theta_deg,
        omega0=OMEGA0,
        sigma=SIGMA,
        res_per_cell=resolution,
        Nx=nx,
        n_modes=n_modes,
        t_eps=t_eps,
        t_op=t_op,
        t_factor=t_factor,
        t_solve=t_solve,
    )
    log(f'Saved {output_path}')
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=sorted(CASES), required=True)
    parser.add_argument('--res', type=int, required=True, help='Pixels per monolayer unit cell.')
    parser.add_argument('--modes', type=int, default=50)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    output_path = args.out if args.out is not None else default_output(args.case, args.res, args.modes)
    solve_fdfd(args.case, args.res, args.modes, output_path, args.force)


if __name__ == '__main__':
    main()