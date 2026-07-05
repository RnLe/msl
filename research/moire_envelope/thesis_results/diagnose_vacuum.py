"""
Vacuum test: compare FDFD eigenvalues against analytic free-photon values.
No rods, eps=1 everywhere. This tests the FDFD operator in isolation.
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

N_MODES = 20

for m_idx, n_idx in [(11, 1), (38, 1)]:
    L1 = np.array([m_idx, n_idx], dtype=float)
    L2 = np.array([-n_idx, m_idx], dtype=float)
    L_SUPER = np.sqrt(L1 @ L1)
    N_cells = m_idx**2 + n_idx**2
    
    print(f"\n{'='*60}")
    print(f"({m_idx},{n_idx}): N_cells={N_cells}, L_super={L_SUPER:.4f}")
    print(f"{'='*60}")
    
    # Build vacuum eps (all 1.0)
    for RES in [32]:
        N_grid = RES * round(L_SUPER)
        eps_grid, info = build_supercell_eps(
            lattice_type='square', m=m_idx, n=n_idx,
            r_over_a=0.0, eps_rod=1.0, eps_bg=1.0,  # vacuum!
            Nx=N_grid, Ny=N_grid)
        
        # Verify all eps = 1
        assert np.allclose(eps_grid, 1.0), "eps should be all 1.0 for vacuum"
        
        # Check B_super in info
        B_super = info['B_super']
        print(f"  B_super = {B_super.tolist()}")
        B_inv = np.linalg.inv(B_super)
        g_contra = B_inv @ B_inv.T
        print(f"  g_contra = {g_contra.tolist()}")
        
        L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                                   polarization='tm')
        del eps_grid
        
        evals, _ = eigsh(L_op, k=N_MODES, sigma=-0.01, which='LM',
                         maxiter=20000, tol=1e-10)
        idx = np.argsort(evals)
        evals = evals[idx]
        freqs_fdfd = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        
        # Analytic: for vacuum, eigenvalues = |G|^2 where G = 2pi B^{-T} n
        # For square supercell with L1 perp L2, |L1|=|L2|=L:
        # lambda_n = (2pi)^2 (n1^2 + n2^2) / L^2
        # freq = sqrt(lambda) / (2pi) = sqrt(n1^2 + n2^2) / L
        
        # Generate analytic eigenvalues
        nmax = 5
        analytic_evals = []
        for n1 in range(-nmax, nmax+1):
            for n2 in range(-nmax, nmax+1):
                # G = 2pi B^{-T} @ [n1, n2]
                G = 2 * np.pi * B_inv.T @ np.array([n1, n2])
                lam = np.dot(G, G)
                analytic_evals.append(lam)
        analytic_evals = np.sort(analytic_evals)[:N_MODES]
        analytic_freqs = np.sqrt(np.maximum(analytic_evals, 0)) / (2 * np.pi)
        
        print(f"\n  res={RES}, N_grid={N_grid}")
        print(f"  Mode  FDFD_eval   Analytic    ratio    FDFD_freq   Analytic_freq")
        for i in range(min(10, N_MODES)):
            if analytic_evals[i] > 1e-10:
                ratio = evals[i] / analytic_evals[i]
            else:
                ratio = float('nan')
            print(f"  {i:4d}  {evals[i]:10.6f}  {analytic_evals[i]:10.6f}  {ratio:.6f}  {freqs_fdfd[i]:.6f}  {analytic_freqs[i]:.6f}")
