#!/usr/bin/env python3
"""
Validate LOBPCG+FFT vs CHOLMOD shift-invert — isolated subprocess per solver
for accurate peak-RSS measurement.

Test cases (all 64 px/cell, TM, Γ-point):
  1. (8,1)   θ=14.25°  grid=512    — small
  2. (14,1)  θ=8.17°   grid=896    — medium
  3. (29,1)  θ=3.95°   grid=1856   — large (CHOLMOD factor should dominate)
"""

import json, os, subprocess, sys, time
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'

PYTHON = sys.executable
CWD = os.path.dirname(os.path.abspath(__file__))
PX_PER_CELL = 64
N_MODES = 20

# ── Subprocess scripts ──────────────────────────────────────────
# Each script: build eps, build operator, solve, print JSON to stdout (last line).
# We suppress meep/mpb noise by importing them before the print.

CHOLMOD_SCRIPT = r'''
import os, sys, time, resource, gc, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

m, n, n_modes, px = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
wdir = sys.argv[5]

sys.path.insert(0, wdir)
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator

L1 = np.array([m, n], dtype=float)
L_super = np.sqrt(L1 @ L1)
N_grid = px * round(L_super)
SIGMA_OMEGA = 0.01

eps_grid, info = build_supercell_eps(
    lattice_type='square', m=m, n=n,
    r_over_a=0.2, eps_rod=8.9, eps_bg=1.0,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)

L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                           polarization='tm')
del eps_grid; gc.collect()

sigma = (2 * np.pi * SIGMA_OMEGA) ** 2
N_dof = L_op.shape[0]
L_shifted = L_op - sigma * sp.eye(N_dof, format='csc')

from sksparse.cholmod import cholesky
t0 = time.time()
factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L_op.dtype)
evals, _ = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                 OPinv=OPinv, maxiter=20000, tol=1e-10)
t_solve = time.time() - t0

del factor, OPinv, L_shifted, L_op; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
evals = np.sort(evals)
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print(json.dumps({"rss_mb": rss_kb / 1024, "t_solve": t_solve,
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

LOBPCG_SCRIPT = r'''
import os, sys, time, resource, gc, json
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

m, n, n_modes, px = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
wdir = sys.argv[5]

sys.path.insert(0, wdir)
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import solve_fdfd_lobpcg

L1 = np.array([m, n], dtype=float)
L_super = np.sqrt(L1 @ L1)
N_grid = px * round(L_super)

eps_grid, info = build_supercell_eps(
    lattice_type='square', m=m, n=n,
    r_over_a=0.2, eps_rod=8.9, eps_bg=1.0,
    Nx=N_grid, Ny=N_grid,
    subpixel_smoothing=True, smoothing_Nsub=8)

evals, _, timings = solve_fdfd_lobpcg(
    eps_grid, info,
    q_vec=np.array([0.0, 0.0]),
    n_modes=n_modes,
    tol=1e-8, maxiter=500, verbose=False)

del eps_grid; gc.collect()

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
freqs = (np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)).tolist()

print(json.dumps({"rss_mb": rss_kb / 1024, "t_solve": timings['total'],
                   "grid": N_grid, "freqs": freqs}), flush=True)
'''

# ── Runner ──────────────────────────────────────────────────────
def run_solver(script: str, m: int, n: int, n_modes: int, timeout: int = 900) -> dict:
    """Run solver in isolated subprocess, return parsed JSON."""
    proc = subprocess.run(
        [PYTHON, '-c', script, str(m), str(n), str(n_modes), str(PX_PER_CELL), CWD],
        capture_output=True, text=True, cwd=CWD, timeout=timeout)
    if proc.returncode != 0:
        print(f"    FAILED (rc={proc.returncode})")
        err = proc.stderr.strip().split('\n')
        for line in err[-5:]:
            print(f"    {line}")
        return None
    lines = proc.stdout.strip().split('\n')
    return json.loads(lines[-1])


# ── Main ────────────────────────────────────────────────────────
CASES = [
    (8,  1, "small"),
    (14, 1, "medium"),
    (29, 1, "large"),
]

results = []

for m, n, label in CASES:
    theta = np.degrees(2 * np.arctan2(n, m))
    L_super = np.sqrt(m**2 + n**2)
    grid = PX_PER_CELL * round(L_super)
    dof = grid**2

    print("=" * 70)
    print(f"({m},{n})  θ={theta:.2f}°  grid={grid}  DOF={dof:,}  [{label}]")
    print("=" * 70)

    # CHOLMOD
    print(f"  CHOLMOD ...", end=' ', flush=True)
    t0 = time.time()
    r_chol = run_solver(CHOLMOD_SCRIPT, m, n, N_MODES)
    wall_chol = time.time() - t0
    if r_chol:
        print(f"RSS={r_chol['rss_mb']:.0f} MB  t={r_chol['t_solve']:.1f}s  "
              f"(wall {wall_chol:.0f}s)")

    # LOBPCG
    print(f"  LOBPCG  ...", end=' ', flush=True)
    t0 = time.time()
    r_lob = run_solver(LOBPCG_SCRIPT, m, n, N_MODES)
    wall_lob = time.time() - t0
    if r_lob:
        print(f"RSS={r_lob['rss_mb']:.0f} MB  t={r_lob['t_solve']:.1f}s  "
              f"(wall {wall_lob:.0f}s)")

    if r_chol and r_lob:
        f_chol = np.array(r_chol['freqs'])
        f_lob = np.array(r_lob['freqs'])
        n_cmp = min(len(f_chol), len(f_lob))
        delta = np.abs(f_lob[:n_cmp] - f_chol[:n_cmp])
        rel = np.where(f_chol[:n_cmp] > 1e-12, delta / f_chol[:n_cmp], 0.0)
        print(f"  Δω: max|Δω|={delta.max():.2e}  max|Δω/ω|={rel.max():.2e}")

        results.append({
            'm': m, 'n': n, 'theta': theta, 'grid': grid, 'dof': dof,
            'rss_chol': r_chol['rss_mb'], 'rss_lob': r_lob['rss_mb'],
            't_chol': r_chol['t_solve'], 't_lob': r_lob['t_solve'],
            'max_rel': rel.max(), 'max_abs': delta.max(),
            'ram_ratio': r_lob['rss_mb'] / r_chol['rss_mb'],
        })

# ── Summary table ───────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'Case':>8s} {'grid':>6s} {'DOF':>10s} "
      f"{'CHOL MB':>8s} {'LOB MB':>8s} {'ratio':>6s} "
      f"{'CHOL s':>7s} {'LOB s':>7s} {'max|Δω/ω|':>11s}")
print("-" * 90)
for r in results:
    print(f"({r['m']:2d},{r['n']}) {r['grid']:6d} {r['dof']:10,d} "
          f"{r['rss_chol']:8.0f} {r['rss_lob']:8.0f} {r['ram_ratio']:6.2f} "
          f"{r['t_chol']:7.1f} {r['t_lob']:7.1f} {r['max_rel']:11.2e}")

# ── Save ────────────────────────────────────────────────────────
outfile = os.path.join(CWD, 'validate_lobpcg_isolated.npz')
if results:
    np.savez(outfile, **{k: np.array([r[k] for r in results]) for k in results[0]})
    print(f"\nSaved → {outfile}")

# ── Plot ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

if len(results) >= 2:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dofs = [r['dof'] for r in results]

    ax1.plot(dofs, [r['rss_chol'] for r in results], 'o-', color='#1f77b4',
             ms=7, lw=1.5, label='CHOLMOD')
    ax1.plot(dofs, [r['rss_lob'] for r in results], 's-', color='#d62728',
             ms=7, lw=1.5, label='LOBPCG')
    ax1.set_xlabel('DOF')
    ax1.set_ylabel('Peak RSS [MB]')
    ax1.set_title('Peak RAM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.plot(dofs, [r['t_chol'] for r in results], 'o-', color='#1f77b4',
             ms=7, lw=1.5, label='CHOLMOD')
    ax2.plot(dofs, [r['t_lob'] for r in results], 's-', color='#d62728',
             ms=7, lw=1.5, label='LOBPCG')
    ax2.set_xlabel('DOF')
    ax2.set_ylabel('Solve time [s]')
    ax2.set_title('Solve time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    fig.suptitle(f'CHOLMOD vs LOBPCG — isolated subprocess, {N_MODES} modes, '
                 f'{PX_PER_CELL} px/cell', fontweight='bold')
    fig.tight_layout()
    figfile = os.path.join(CWD, 'validate_lobpcg_isolated.png')
    fig.savefig(figfile, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot → {figfile}")
