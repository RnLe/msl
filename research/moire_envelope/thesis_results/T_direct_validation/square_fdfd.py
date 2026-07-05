#!/usr/bin/env python3
"""
FDFD Eigensolve for Square Lattice (57,1) Twisted Bilayer at Γ (= folded M)
============================================================================

Solves the full supercell FDFD eigenproblem for a twisted square bilayer
at the commensurate angle θ = 2·arctan(1/57) ≈ 2.01°.

Because M = (0.5, 0.5) folds exactly to Γ for the (57,1) supercell
(both m and n are odd), we use q_vec = (0,0), making the operator
REAL symmetric → cheaper factorization and eigensolve.

Physical params:
    r/a = 0.2, ε_rod = 11.56, ε_bg = 1.0, TM polarization
    Band 3 at M: ω₀ = 0.68457 (c/a)
    σ = (2πω₀)² ≈ 18.50  (eigenvalue shift for band-3 neighbourhood)

Usage:
    python square_fdfd.py [--res 40] [--modes 50]
    nohup python square_fdfd.py --res 48 --modes 50 > square_fdfd.log 2>&1 &
"""

import sys, os, time, gc, argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

# Import as package from thesis_results/ (relative imports in supercell_geometry require this)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle

# ═══════════════════════════════════════════════════════════════
# Fixed physical parameters
# ═══════════════════════════════════════════════════════════════
M_MN, N_MN = 57, 1
LATTICE = 'square'
R_OVER_A = 0.2
EPS_ROD = 11.56
EPS_BG = 1.0
OMEGA0 = 0.68457          # c/a, TM band 3 at M from MPB res=128
SIGMA = (2 * np.pi * OMEGA0) ** 2   # ≈ 18.50

N_CELLS = M_MN**2 + N_MN**2          # 3250
L_SUPER = np.sqrt(N_CELLS)           # 57.01


def main():
    parser = argparse.ArgumentParser(description='FDFD solve for square (57,1)')
    parser.add_argument('--res', type=int, default=40,
                        help='Resolution (pixels per monolayer a). Default 40.')
    parser.add_argument('--modes', type=int, default=50,
                        help='Number of eigenvalues to find near sigma. Default 50.')
    args = parser.parse_args()

    RES = args.res
    N_MODES = args.modes
    Nx = int(round(L_SUPER * RES))
    N_dof = Nx * Nx

    theta_deg = np.degrees(commensurate_twist_angle(LATTICE, M_MN, N_MN))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir,
        f'fdfd_square_m{M_MN}_n{N_MN}_res{RES}_k{N_MODES}.npz')

    print(f"{'='*70}")
    print(f"  FDFD eigensolver — square lattice (m,n)=({M_MN},{N_MN})")
    print(f"  θ = {theta_deg:.4f}°, N_cells = {N_CELLS}")
    print(f"  res = {RES}, Nx = {Nx}, DOF = {N_dof:,}")
    print(f"  ω₀ = {OMEGA0:.5f} (c/a), σ = {SIGMA:.4f}")
    print(f"  q_vec = (0, 0)  [M folds to Γ exactly]")
    print(f"  n_modes = {N_MODES}")
    print(f"  Output: {os.path.basename(out_file)}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Check for existing output
    if os.path.exists(out_file):
        print(f"Output already exists: {out_file}")
        data = np.load(out_file)
        freqs = np.sort(data['freqs'])
        print(f"  {len(freqs)} modes, range [{freqs.min():.6f}, {freqs.max():.6f}]")
        return

    # ── 1. Build epsilon grid ──
    print("1. Building supercell epsilon grid...")
    t0 = time.time()
    eps, info = build_supercell_eps(
        LATTICE, M_MN, N_MN, a=1.0,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=Nx, Ny=Nx,
    )
    t_eps = time.time() - t0
    print(f"   eps shape: {eps.shape}, range [{eps.min():.2f}, {eps.max():.2f}]")
    print(f"   Build time: {t_eps:.1f}s")
    sys.stdout.flush()

    # ── 2. Build operator ──
    print(f"\n2. Building FDFD operator (TM, q=0)...")
    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    print(f"   L: {L.shape}, nnz={L.nnz:,}, dtype={L.dtype}")
    print(f"   Build time: {t_op:.1f}s")
    is_real = not np.issubdtype(L.dtype, np.complexfloating)
    print(f"   Real operator: {is_real}")
    sys.stdout.flush()

    del eps  # free ~200 MB
    gc.collect()

    # ── 3. Shift-invert factorization ──
    L_shifted = L - SIGMA * sp.eye(N_dof, format='csc')

    try:
        from sksparse.cholmod import cholesky
        print(f"\n3. CHOLMOD LDLᵀ factorization (DOF={N_dof:,})...")
        sys.stdout.flush()
        t0 = time.time()
        factor = cholesky(L_shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        print(f"   Factorization: {t_factor:.1f}s")
        sys.stdout.flush()

        OPinv = LinearOperator(
            (N_dof, N_dof),
            matvec=lambda b: factor(b),
            dtype=L.dtype,
        )

        # ── 4. Eigensolve ──
        print(f"\n4. eigsh ({N_MODES} modes near σ={SIGMA:.4f})...")
        sys.stdout.flush()
        t0 = time.time()
        evals, evecs = eigsh(
            L, k=N_MODES, sigma=SIGMA, which='LM',
            OPinv=OPinv, maxiter=5000, tol=1e-8,
        )
        t_solve = time.time() - t0
        print(f"   Eigensolver: {t_solve:.1f}s")

        del factor, OPinv  # free memory
    except ImportError:
        print("\n   WARNING: sksparse not available, falling back to scipy's "
              "built-in sparse LU (may be slow/memory-heavy).")
        print(f"\n3-4. eigsh ({N_MODES} modes near σ={SIGMA:.4f})...")
        sys.stdout.flush()
        t0 = time.time()
        t_factor = 0
        evals, evecs = eigsh(
            L, k=N_MODES, sigma=SIGMA, which='LM',
            maxiter=10000, tol=1e-10,
        )
        t_solve = time.time() - t0
        print(f"   Total solve: {t_solve:.1f}s")

    del L, L_shifted
    gc.collect()

    # ── 5. Post-process ──
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)  # c/a units

    print(f"\n5. Results:")
    print(f"   {len(freqs)} eigenfrequencies in c/a:")
    print(f"   min = {freqs.min():.6f}")
    print(f"   max = {freqs.max():.6f}")
    n_near = np.sum(np.abs(freqs - OMEGA0) < 0.01)
    print(f"   Within ±0.01 of ω₀ = {OMEGA0}: {n_near} modes")

    # Print the 20 closest to omega0
    d = np.abs(freqs - OMEGA0)
    close_idx = np.argsort(d)[:20]
    print(f"\n   20 closest to ω₀ = {OMEGA0:.5f}:")
    for i in close_idx:
        print(f"     [{i:3d}] ω = {freqs[i]:.6f}  Δ = {freqs[i]-OMEGA0:+.6f}")

    # ── 6. Save ──
    np.savez(out_file,
             freqs=freqs,
             evals=evals,
             # Don't save evecs (huge) — can recompute if needed
             m=M_MN, n=N_MN, N_cells=N_CELLS,
             res=RES, Nx=Nx, n_modes=N_MODES,
             omega0=OMEGA0, sigma=SIGMA,
             theta_deg=theta_deg,
             t_eps=t_eps, t_op=t_op, t_factor=t_factor, t_solve=t_solve)
    print(f"\n6. Saved → {os.path.basename(out_file)}")
    print(f"   Total wall time: {t_eps + t_op + t_factor + t_solve:.1f}s")


if __name__ == '__main__':
    main()
