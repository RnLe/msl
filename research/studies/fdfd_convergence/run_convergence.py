#!/usr/bin/env python3
"""
FDFD TE convergence study at the square-lattice X point.

Sweeps over:
  - Angles:      1°, 2°, 4°, 8°  (commensurate approximations)
  - Resolutions:  1, 4, 8, 16 px per unit cell
  - Target freqs: 0.05, 0.1, 0.2, 0.3, 0.4  (c/a units)

Outputs one .npz per (angle, resolution, freq) triple into data/.
Skips runs whose output already exists (safe to re-run after failures).
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

# ── locate T_direct_validation on the import path ──────────────────
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.join(
    os.path.dirname(STUDY_DIR), '..',
    'moire_envelope', 'thesis_results')
sys.path.insert(0, os.path.abspath(THESIS_RESULTS))

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps

# ── physical constants ──────────────────────────────────────────────
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
Q_X = np.array([np.pi, 0.0])

# ── sweep axes ──────────────────────────────────────────────────────
ANGLES = [
    {'m': 14,  'n': 1, 'label': '8deg'},    # θ ≈ 8.13°  (fastest first)
    {'m': 29,  'n': 1, 'label': '4deg'},    # θ ≈ 3.95°
    {'m': 57,  'n': 1, 'label': '2deg'},    # θ ≈ 2.01°
    {'m': 114, 'n': 1, 'label': '1deg'},    # θ ≈ 1.005° (largest last)
]

RESOLUTIONS = [1, 4, 8, 16]                # px per unit cell
TARGET_FREQS = [0.05, 0.1, 0.2, 0.3, 0.4]  # σ_ω in c/a
N_MODES = 20

DATA_DIR = os.path.join(STUDY_DIR, 'data')


def freq_tag(f: float) -> str:
    """0.05 → 'f005', 0.4 → 'f040'."""
    return f'f{f * 100:03.0f}'


def out_path(label: str, px: int, f: float) -> str:
    return os.path.join(
        DATA_DIR,
        f'fdfd_te_x_{label}_res{px}_{freq_tag(f)}.npz')


def run_single(m: int, n: int, label: str,
               px_per_cell: int, sigma_omega: float) -> bool:
    """Run one FDFD eigensolve.  Returns True on success."""
    dest = out_path(label, px_per_cell, sigma_omega)
    if os.path.isfile(dest):
        print(f'  [skip] {os.path.basename(dest)} already exists')
        return True

    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_deg = np.degrees(2 * np.arctan2(n, m))
    N_grid = px_per_cell * round(L_super)

    print(f'\n  {label}  res={px_per_cell}  σ_ω={sigma_omega}  '
          f'grid={N_grid}×{N_grid}  DOF={N_grid**2:,}')
    sys.stdout.flush()

    try:
        eps_grid, info = build_supercell_eps(
            lattice_type='square', m=m, n=n,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=N_grid, Ny=N_grid,
            subpixel_smoothing=True, smoothing_Nsub=8)

        t0 = time.time()
        L_op = build_fdfd_operator(
            eps_grid, info, q_vec=Q_X, polarization='te')
        t_assembly = time.time() - t0

        sigma = (2 * np.pi * sigma_omega) ** 2
        t0 = time.time()
        evals, evecs = eigsh(
            L_op, k=N_MODES, sigma=sigma, which='LM',
            maxiter=20_000, tol=1e-10)
        t_solve = time.time() - t0

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        evals = np.real_if_close(evals, tol=1000)
        idx = np.argsort(np.asarray(evals, dtype=float))
        evals = np.asarray(evals, dtype=float)[idx]
        freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)

        np.savez(dest,
                 freqs=freqs, evals=evals,
                 grid=N_grid, px_per_cell=px_per_cell,
                 m=m, n=n, n_modes=N_MODES,
                 sigma_omega=sigma_omega, q_vec=Q_X,
                 t_assembly=t_assembly, t_solve=t_solve,
                 rss_mb=rss_mb)

        print(f'  ✓ solved {t_solve:.1f}s  RSS={rss_mb:.0f} MB  '
              f'freq=[{freqs[0]:.6f}, {freqs[-1]:.6f}]')
        print(f'    → {os.path.basename(dest)}')

        del L_op, eps_grid, evecs, evals, freqs
        gc.collect()
        return True

    except Exception as exc:
        print(f'  ✗ FAILED: {exc}')
        gc.collect()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='FDFD TE convergence study at the X point')
    parser.add_argument('--angles', default='',
                        help='Comma-separated angle labels to run (default: all)')
    parser.add_argument('--resolutions', default='',
                        help='Comma-separated px values (default: all)')
    parser.add_argument('--freqs', default='',
                        help='Comma-separated target freqs (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without running')
    args = parser.parse_args()

    sel_angles = ({s.strip() for s in args.angles.split(',') if s.strip()}
                  or {a['label'] for a in ANGLES})
    sel_px = ([int(x) for x in args.resolutions.split(',') if x.strip()]
              or RESOLUTIONS)
    sel_freq = ([float(x) for x in args.freqs.split(',') if x.strip()]
                or TARGET_FREQS)

    cases = [(a, px, f)
             for a in ANGLES if a['label'] in sel_angles
             for px in sorted(sel_px)
             for f in sorted(sel_freq)]

    total = len(cases)
    print(f'FDFD TE convergence study — {total} runs planned')
    print(f'Angles: {sorted(sel_angles)}')
    print(f'Resolutions: {sorted(sel_px)}')
    print(f'Target freqs: {sorted(sel_freq)}')
    print(f'Modes per run: {N_MODES}\n')

    if args.dry_run:
        for a, px, f in cases:
            tag = 'EXISTS' if os.path.isfile(out_path(a['label'], px, f)) else 'TODO'
            L = np.sqrt(a['m']**2 + a['n']**2)
            g = px * round(L)
            print(f'  [{tag}] {a["label"]}  res={px}  f={f}  grid={g}²  DOF={g**2:,}')
        return

    ok, fail = 0, 0
    t_total = time.time()
    for i, (a, px, f) in enumerate(cases, 1):
        print(f'\n{"=" * 60}')
        print(f'[{i}/{total}]  {a["label"]}  res={px}  target_freq={f}')
        if run_single(a['m'], a['n'], a['label'], px, f):
            ok += 1
        else:
            fail += 1

    elapsed = time.time() - t_total
    print(f'\n{"=" * 60}')
    print(f'Done.  {ok} succeeded, {fail} failed, {elapsed:.0f}s total')


if __name__ == '__main__':
    main()
