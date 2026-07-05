"""Quick feasibility test: can we solve the honeycomb moiré supercell in 32 GB?"""
import numpy as np
import sys, os, time, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from T_direct_validation.commensurate_utils import (
    enumerate_commensurate_angles, estimate_fdfd_resources,
    commensurate_twist_angle, build_supercell_vectors
)
from T_direct_validation.supercell_geometry import (
    build_supercell_eps, build_monolayer_basis, get_sublattice_positions, rotation_matrix_2d
)
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

# ── Step 1: Enumerate angles near 3° ──
print("=" * 70)
print("COMMENSURATE ANGLES FOR HONEYCOMB, 2°–6°")
print("=" * 70)
angles = enumerate_commensurate_angles('honeycomb', theta_min_deg=2.0, theta_max_deg=6.0, max_N=2000)
print(f"{'(m,n)':>8}  {'theta':>8}  {'N_cells':>8}  {'sqrt(N)':>8}")
for a in angles:
    sqrtN = np.sqrt(a['N_cells'])
    print(f"  ({a['m']},{a['n']}){'':<3}  {a['theta_deg']:>7.3f}°  {a['N_cells']:>8}  {sqrtN:>8.1f}")

# ── Step 2: Memory estimates ──
print(f"\n{'=' * 70}")
print("MEMORY ESTIMATES (for different resolutions per cell)")
print("=" * 70)
# For honeycomb TM with the corrected operator:
# DOF = Nx * Ny, where Nx = Ny = sqrt(N_cells) * res_per_cell
# The sparse matrix has ~5 nnz/row, but the LU decomposition dominates
target_angles = [(10, 11), (8, 7), (5, 4), (7, 6)]  # near 3° to 5°
for m, n in target_angles:
    N = m*m + m*n + n*n
    theta = np.degrees(commensurate_twist_angle('honeycomb', m, n))
    print(f"\n  (m,n)=({m},{n}), theta={theta:.3f}°, N_cells={N}")
    print(f"  {'res/cell':>10}  {'Nx=Ny':>8}  {'DOF':>10}  {'est_mem_GB':>12}  {'feasible':>10}")
    for res in [8, 12, 16, 24, 32]:
        sqrtN = np.sqrt(N)
        Nx = int(round(sqrtN * res))
        dof = Nx * Nx
        # Sparse matrix memory
        nnz = 5 * dof
        sparse_mem = nnz * 20  # data + indices
        # LU factorization (dominates): band ~ Nx, fill ~ 10-20x
        lu_mem = 15 * nnz * 16
        # Eigenvectors: 30 modes
        evec_mem = 30 * dof * 16
        total_gb = (sparse_mem + lu_mem + evec_mem) / 1e9
        ok = "YES" if total_gb < 24 else "NO"
        print(f"  {res:>10}  {Nx:>8}  {dof:>10,}  {total_gb:>12.2f}  {ok:>10}")

# ── Step 3: Quick coarse test ──
print(f"\n{'=' * 70}")
print("COARSE FEASIBILITY TEST")
print("=" * 70)

# Start with the largest angle (smallest supercell)
m, n = 8, 7
N_cells = m*m + m*n + n*n
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', m, n))
print(f"Testing (m,n)=({m},{n}), theta={theta_deg:.3f}°, N_cells={N_cells}")

# Very coarse first: 8 pts per cell
for res_per_cell in [8, 12, 16]:
    sqrtN = np.sqrt(N_cells)
    Nx = int(round(sqrtN * res_per_cell))
    print(f"\n  Resolution: {res_per_cell} pts/cell → Nx=Ny={Nx}, DOF={Nx*Nx:,}")

    tracemalloc.start()
    t0 = time.time()

    # Build epsilon on supercell
    eps, info = build_supercell_eps(
        'honeycomb', m=m, n=n, a=1.0, r_over_a=0.2,
        eps_rod=11.56, eps_bg=1.0, Nx=Nx, Ny=Nx,
    )
    t_eps = time.time() - t0
    print(f"  Epsilon built: {t_eps:.1f}s, shape={eps.shape}, "
          f"rod fraction={(eps > 5).mean():.4f}")

    # K-point at moire BZ center Gamma (simplest test)
    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    print(f"  Operator built: {t_op:.1f}s, shape={L.shape}, nnz={L.nnz:,}")

    # Solve for a few modes
    t0 = time.time()
    try:
        evals, evecs = eigsh(L, k=10, sigma=0.01, which='LM',
                             maxiter=5000, tol=1e-8)
        t_solve = time.time() - t0
        evals = np.sort(evals)
        freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        print(f"  Eigensolver: {t_solve:.1f}s")
        print(f"  First 5 frequencies: {freqs[:5]}")
    except Exception as e:
        t_solve = time.time() - t0
        print(f"  Eigensolver FAILED after {t_solve:.1f}s: {e}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Peak memory (Python): {peak / 1e9:.2f} GB")

print(f"\n{'=' * 70}")
print("Now testing (10,11) — the ~3.15° angle")
print("=" * 70)
m, n = 10, 11
N_cells = m*m + m*n + n*n
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', m, n))
print(f"(m,n)=({m},{n}), theta={theta_deg:.3f}°, N_cells={N_cells}")

for res_per_cell in [8, 12]:
    sqrtN = np.sqrt(N_cells)
    Nx = int(round(sqrtN * res_per_cell))
    print(f"\n  Resolution: {res_per_cell} pts/cell → Nx=Ny={Nx}, DOF={Nx*Nx:,}")

    tracemalloc.start()
    t0 = time.time()
    eps, info = build_supercell_eps(
        'honeycomb', m=m, n=n, a=1.0, r_over_a=0.2,
        eps_rod=11.56, eps_bg=1.0, Nx=Nx, Ny=Nx,
    )
    t_eps = time.time() - t0
    print(f"  Epsilon built: {t_eps:.1f}s, rod fraction={(eps > 5).mean():.4f}")

    t0 = time.time()
    L = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    print(f"  Operator built: {t_op:.1f}s, nnz={L.nnz:,}")

    t0 = time.time()
    try:
        evals, evecs = eigsh(L, k=10, sigma=0.01, which='LM',
                             maxiter=5000, tol=1e-8)
        t_solve = time.time() - t0
        evals = np.sort(evals)
        freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        print(f"  Eigensolver: {t_solve:.1f}s")
        print(f"  First 5 frequencies: {freqs[:5]}")
    except Exception as e:
        t_solve = time.time() - t0
        print(f"  Eigensolver FAILED after {t_solve:.1f}s: {e}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  Peak memory (Python): {peak / 1e9:.2f} GB")
