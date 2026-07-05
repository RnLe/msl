#!/usr/bin/env python3
"""
Validate LOBPCG+FFT solver against CHOLMOD shift-invert baseline.

Runs both solvers on the same supercell and compares eigenvalues,
timing, and peak RAM.

Test cases:
  1. (8,1)  θ=14.25°  grid=512   small  — verify correctness
  2. (14,1) θ=8.17°   grid=896   medium — verify scaling
"""

import os, sys, time, resource, gc
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import (
    build_fdfd_operator,
    solve_fdfd_lobpcg,
)
from scipy.sparse.linalg import eigsh, LinearOperator

# ── Configuration ───────────────────────────────────────────────
PX_PER_CELL = 64
N_MODES = 20
CASES = [
    (8, 1, "small"),
    (14, 1, "medium"),
]

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def solve_cholmod(L_op, n_modes, sigma_omega=0.01):
    """Shift-invert CHOLMOD baseline."""
    sigma = (2 * np.pi * sigma_omega) ** 2
    N = L_op.shape[0]
    L_shifted = L_op - sigma * sp.eye(N, format='csc')

    from sksparse.cholmod import cholesky
    t0 = time.time()
    factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
    t_factor = time.time() - t0

    OPinv = LinearOperator((N, N), matvec=lambda b: factor(b), dtype=L_op.dtype)

    t1 = time.time()
    evals, evecs = eigsh(L_op, k=n_modes, sigma=sigma, which='LM',
                         OPinv=OPinv, maxiter=20000, tol=1e-10)
    t_eigsh = time.time() - t1

    del factor, OPinv, L_shifted
    gc.collect()

    idx = np.argsort(evals)
    return evals[idx], t_factor + t_eigsh

# ── Main ────────────────────────────────────────────────────────
results = []

for m, n, label in CASES:
    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_deg = np.degrees(2 * np.arctan2(n, m))
    N_grid = PX_PER_CELL * round(L_super)

    print("=" * 70)
    print(f"CASE: ({m},{n}) θ={theta_deg:.2f}° grid={N_grid} [{label}]")
    print("=" * 70)

    # Build epsilon grid
    t0 = time.time()
    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=0.2, eps_rod=8.9, eps_bg=1.0,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)
    t_eps = time.time() - t0
    print(f"  ε grid: {N_grid}×{N_grid} built in {t_eps:.1f}s")

    # ── CHOLMOD baseline ────────────────────────────────────────
    print(f"\n  --- CHOLMOD shift-invert ---")
    gc.collect()
    rss_before = peak_rss_mb()

    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                               polarization='tm')
    t_build = time.time() - t0

    evals_chol, t_chol = solve_cholmod(L_op, N_MODES)
    rss_chol = peak_rss_mb()
    freqs_chol = np.sqrt(np.maximum(evals_chol, 0)) / (2 * np.pi)

    del L_op
    gc.collect()

    print(f"  Build: {t_build:.1f}s, Solve: {t_chol:.1f}s, "
          f"RSS: {rss_chol:.0f} MB")
    print(f"  ω: [{freqs_chol[0]:.6f}, {freqs_chol[-1]:.6f}]")

    # ── LOBPCG ──────────────────────────────────────────────────
    print(f"\n  --- LOBPCG + FFT preconditioner ---")
    gc.collect()

    evals_lob, evecs_lob, timings = solve_fdfd_lobpcg(
        eps_grid, info,
        q_vec=np.array([0.0, 0.0]),
        n_modes=N_MODES,
        tol=1e-8,
        maxiter=500,
        verbose=True,
    )
    rss_lob = peak_rss_mb()
    freqs_lob = np.sqrt(np.maximum(evals_lob, 0)) / (2 * np.pi)

    del evecs_lob
    gc.collect()

    # ── Compare ─────────────────────────────────────────────────
    print(f"\n  --- Comparison ---")
    delta = np.abs(freqs_lob - freqs_chol)
    rel = np.where(freqs_chol > 1e-12,
                   delta / freqs_chol, 0.0)
    print(f"  max  |Δω|       = {delta.max():.3e}")
    print(f"  mean |Δω|       = {delta.mean():.3e}")
    print(f"  max  |Δω/ω|     = {rel.max():.3e}")
    print(f"  mean |Δω/ω|     = {rel.mean():.3e}")

    print(f"\n  Timing:  CHOLMOD {t_build + t_chol:.1f}s  vs  LOBPCG {timings['total']:.1f}s")
    print(f"  RAM:     CHOLMOD {rss_chol:.0f} MB  vs  LOBPCG {rss_lob:.0f} MB")
    print(f"  (Note: RSS is cumulative — CHOLMOD ran first)")

    results.append({
        'm': m, 'n': n, 'theta': theta_deg,
        'grid': N_grid, 'n_modes': N_MODES,
        't_chol': t_build + t_chol, 't_lob': timings['total'],
        'rss_chol': rss_chol, 'rss_lob': rss_lob,
        'max_delta_omega': delta.max(),
        'max_rel_error': rel.max(),
    })

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Case':>12s} {'grid':>6s} {'CHOLMOD s':>10s} {'LOBPCG s':>10s} "
      f"{'max |Δω/ω|':>12s}")
print("-" * 55)
for r in results:
    print(f"  ({r['m']},{r['n']}) {r['grid']:6d} "
          f"{r['t_chol']:10.1f} {r['t_lob']:10.1f} "
          f"{r['max_rel_error']:12.3e}")
