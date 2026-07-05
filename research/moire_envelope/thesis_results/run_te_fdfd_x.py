#!/usr/bin/env python3
"""
FDFD TE eigensolve at the square-lattice X point for 8°, 3°, and 1° moire supercells.
Resolution, mode count, and sigma are configurable from the command line.
"""
import argparse
import gc
import os
import resource
import sys
import time

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
from scipy.sparse.linalg import eigsh

CWD = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
sys.path.insert(0, CWD)

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps

CASES = [
    {'m': 14, 'n': 1, 'label': '8deg', 'n_modes': 30},
    {'m': 38, 'n': 1, 'label': '3deg', 'n_modes': 50},
    {'m': 114, 'n': 1, 'label': '1deg', 'n_modes': 50},
]
PX_PER_CELL = 16
SIGMA_OMEGA = 0.33
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
Q_X = np.array([np.pi, 0.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--px-per-cell', type=int, default=PX_PER_CELL)
    parser.add_argument('--sigma-omega', type=float, default=SIGMA_OMEGA)
    parser.add_argument('--n-modes', type=int, default=0,
                        help='Override mode count for all angles; 0 keeps case defaults')
    parser.add_argument('--save-evecs', action='store_true',
                        help='Save eigenvectors in the output npz files')
    parser.add_argument('--angles', default='',
                        help='Comma-separated subset of labels to run, e.g. 1deg or 8deg,3deg')
    parser.add_argument('--out-tag', default='', help='Optional suffix inserted before .npz, e.g. sig002')
    args = parser.parse_args()

    px_per_cell = args.px_per_cell
    sigma_omega = args.sigma_omega
    out_tag = args.out_tag.strip('_')
    out_suffix = f'_{out_tag}' if out_tag else ''
    selected_angles = {item.strip() for item in args.angles.split(',') if item.strip()}

    for case in CASES:
        m, n = case['m'], case['n']
        n_modes = args.n_modes or case['n_modes']
        label = case['label']
        if selected_angles and label not in selected_angles:
            continue

        L1 = np.array([m, n], dtype=float)
        L_super = np.sqrt(L1 @ L1)
        theta_deg = np.degrees(2 * np.arctan2(n, m))
        N_grid = px_per_cell * round(L_super)

        print(f"\n{'=' * 60}")
        print(f"FDFD TE @ X — {label} (m={m}, n={n}), θ={theta_deg:.2f}°")
        print(f"Resolution: {px_per_cell} px/cell → grid {N_grid}×{N_grid}")
        print(f"Modes: {n_modes}, σ_ω={sigma_omega}, q=({Q_X[0]:.6f}, {Q_X[1]:.6f})")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        eps_grid, info = build_supercell_eps(
            lattice_type='square', m=m, n=n,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=N_grid, Ny=N_grid,
            subpixel_smoothing=True, smoothing_Nsub=8)

        t0 = time.time()
        L_op = build_fdfd_operator(
            eps_grid, info,
            q_vec=Q_X,
            polarization='te')
        t_assembly = time.time() - t0
        print(f"Assembly: {t_assembly:.1f}s, nnz={L_op.nnz:,}, DOF={L_op.shape[0]:,}")

        sigma = (2 * np.pi * sigma_omega) ** 2

        t0 = time.time()
        evals, evecs = eigsh(
            L_op, k=n_modes, sigma=sigma, which='LM',
            maxiter=20000, tol=1e-10)
        t_solve = time.time() - t0

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        evals = np.real_if_close(evals, tol=1000)
        idx = np.argsort(np.asarray(evals, dtype=float))
        evals = np.asarray(evals, dtype=float)[idx]
        evecs = evecs[:, idx]
        freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

        print(f"Solved in {t_solve:.1f}s | RSS: {rss_mb:.0f} MB")
        print(f"ω range: [{freqs[0]:.6f}, {freqs[-1]:.6f}] a/2πc")

        out = os.path.join(CWD, f"fdfd_te_x_{label}_res{px_per_cell}{out_suffix}.npz")
        save_kwargs = dict(
            freqs=freqs,
            evals=evals,
            grid=N_grid,
            px_per_cell=px_per_cell,
            m=m,
            n=n,
            n_modes=n_modes,
            sigma_omega=sigma_omega,
            q_vec=Q_X,
            t_assembly=t_assembly,
            t_solve=t_solve,
            rss_mb=rss_mb,
        )
        if args.save_evecs:
            save_kwargs['evecs'] = evecs
        np.savez(out, **save_kwargs)
        print(f"Saved → {out}")

        del L_op, eps_grid, evecs
        gc.collect()

    print("\nAll FDFD TE X-point runs complete.")


if __name__ == '__main__':
    main()