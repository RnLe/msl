"""Targeted solve at the Dirac manifold frequency window."""
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.commensurate_utils import commensurate_twist_angle
from scipy.sparse.linalg import eigsh

# Parameters
EPS_BG, EPS_HOLE, R_OVER_A = 1.0, 11.56, 0.2
omega_ref = 0.182693  # monolayer TM band 1 at K (from MPB)

# Start with the larger angle (8,7) for speed
for m, n in [(8, 7), (11, 10)]:
    N_cells = m*m + m*n + n*n
    theta = np.degrees(commensurate_twist_angle('honeycomb', m, n))
    sqrtN = np.sqrt(N_cells)
    res = 16
    Nx = int(round(sqrtN * res))

    print(f"\n{'='*70}")
    print(f"(m,n)=({m},{n}), theta={theta:.3f} deg, N={N_cells}, Nx={Nx}")
    print(f"{'='*70}")

    eps, info = build_supercell_eps('honeycomb', m=m, n=n, a=1.0,
                                    r_over_a=R_OVER_A, eps_rod=EPS_HOLE,
                                    eps_bg=EPS_BG, Nx=Nx, Ny=Nx)

    # Target sigma near omega_ref
    # eigenvalue lambda = (2*pi*f)^2, so sigma = (2*pi*omega_ref)^2
    sigma = (2 * np.pi * omega_ref)**2
    print(f"Targeting eigenvalues near lambda = {sigma:.4f} (omega_ref = {omega_ref})")

    # Solve at Gamma
    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    print(f"Operator: {time.time()-t0:.1f}s, DOF={L.shape[0]:,}")

    t0 = time.time()
    evals, evecs = eigsh(L, k=30, sigma=sigma, which='LM',
                         maxiter=5000, tol=1e-8)
    t_solve = time.time() - t0
    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    print(f"Eigensolver: {t_solve:.1f}s for 30 modes near omega_ref")

    print(f"\nFrequencies near omega_ref = {omega_ref}:")
    for i, f in enumerate(freqs):
        marker = " <-- omega_ref" if abs(f - omega_ref) < 0.005 else ""
        print(f"  mode {i:>3}: f = {f:.6f}  (delta = {f - omega_ref:+.6f}){marker}")

    # Also show bandwidth of modes within 10% of omega_ref
    mask = np.abs(freqs - omega_ref) < 0.1 * omega_ref
    if mask.any():
        f_near = freqs[mask]
        bw = f_near.max() - f_near.min()
        print(f"\nModes within 10% of omega_ref: {mask.sum()}")
        print(f"Bandwidth: {bw:.6f}")
        print(f"Bandwidth / omega_ref: {bw / omega_ref:.4f}")
