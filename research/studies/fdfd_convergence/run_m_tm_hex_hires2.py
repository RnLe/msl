#!/usr/bin/env python3
"""Additional FDFD runs: 1° at 8px, 2° at 16px."""
import gc, os, resource, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
from scipy.sparse.linalg import eigsh

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.join(
    os.path.dirname(STUDY_DIR), '..',
    'moire_envelope', 'thesis_results')
sys.path.insert(0, os.path.abspath(THESIS_RESULTS))

from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps

R_OVER_A, EPS_ROD, EPS_BG = 0.220, 12.0, 1.0
Q_M = np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
N_MODES = 50
DATA_DIR = os.path.join(STUDY_DIR, 'data_m_tm_hex')
os.makedirs(DATA_DIR, exist_ok=True)

CASES = [
    ({'m': 67, 'n': 65, 'label': '1deg'}, 8),   # 1° at 8px: grid=912
    ({'m': 17, 'n': 16, 'label': '2deg'}, 16),   # 2° at 16px: grid=464
]

def freq_tag(f): return f'f{f * 10000:05.0f}'
def out_path(label, px, f):
    return os.path.join(DATA_DIR, f'fdfd_tm_m_{label}_res{px}_{freq_tag(f)}.npz')

def run_single(m, n, label, px_per_cell, sigma_omega):
    dest = out_path(label, px_per_cell, sigma_omega)
    if os.path.isfile(dest):
        print(f'  [skip] {os.path.basename(dest)}')
        return True
    N_cells = m*m + m*n + n*n
    L_super = np.sqrt(N_cells)
    N_grid = px_per_cell * round(L_super)
    print(f'  {label}  res={px_per_cell}  grid={N_grid}x{N_grid}  DOF={N_grid**2:,}')
    sys.stdout.flush()
    try:
        eps_grid, info = build_supercell_eps(
            lattice_type='hex', m=m, n=n,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=N_grid, Ny=N_grid,
            subpixel_smoothing=True, smoothing_Nsub=8)
        t0 = time.time()
        L_op = build_fdfd_operator(eps_grid, info, q_vec=Q_M, polarization='tm')
        t_assembly = time.time() - t0
        sigma = (2 * np.pi * sigma_omega) ** 2
        t0 = time.time()
        evals, evecs = eigsh(L_op, k=N_MODES, sigma=sigma, which='LM',
                             maxiter=20_000, tol=1e-10)
        t_solve = time.time() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        evals = np.real_if_close(evals, tol=1000)
        idx = np.argsort(np.asarray(evals, dtype=float))
        evals = np.asarray(evals, dtype=float)[idx]
        freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        np.savez(dest, freqs=freqs, evals=evals, grid=N_grid,
                 px_per_cell=px_per_cell, m=m, n=n, n_modes=N_MODES,
                 sigma_omega=sigma_omega, q_vec=Q_M,
                 t_assembly=t_assembly, t_solve=t_solve, rss_mb=rss_mb)
        print(f'  -> {t_solve:.1f}s  freq=[{freqs[0]:.6f}, {freqs[-1]:.6f}]')
        del L_op, eps_grid, evecs, evals, freqs; gc.collect()
        return True
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'  FAILED: {exc}'); gc.collect()
        return False

def main():
    t_start = time.time()
    ok, fail = 0, 0
    for i, (a, px) in enumerate(CASES, 1):
        print(f'\n[{i}/{len(CASES)}]  {a["label"]}  res={px}')
        if run_single(a['m'], a['n'], a['label'], px, 0.2088):
            ok += 1
        else:
            fail += 1
    print(f'\nDone  {ok} ok / {fail} fail  in {time.time()-t_start:.0f}s')

if __name__ == '__main__':
    main()
