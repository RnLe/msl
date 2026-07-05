"""Benchmark MPB + FDFD at 3 angles for scaling extrapolation."""
import os, sys, time, tracemalloc
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(__file__))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator

R_OVER_A, EPS_ROD, EPS_BG = 0.2, 8.9, 1.0
RES = 32
N_MODES = 100
TARGET_OMEGA = 0.053
SIGMA = (2 * np.pi * TARGET_OMEGA) ** 2

CASES = [
    (8,  1),   # ~14.25° (closest to 15°)
    (11, 1),   # ~10.39°
    (14, 1),   # ~8.17°
]

for m_idx, n_idx in CASES:
    theta = 2 * np.degrees(np.arctan2(n_idx, m_idx))
    N_cells = m_idx**2 + n_idx**2
    L1 = np.array([m_idx, n_idx], dtype=float)
    L2 = np.array([-n_idx, m_idx], dtype=float)
    L_SUPER = np.sqrt(L1 @ L1)

    print(f"\n{'='*60}")
    print(f"(m,n)=({m_idx},{n_idx})  θ={theta:.2f}°  N_cells={N_cells}  L_super={L_SUPER:.2f}")
    print(f"{'='*60}")

    # --- MPB ---
    theta_rad = 2 * np.arctan2(n_idx, m_idx)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    B_super = np.column_stack([L1, L2])
    B_inv = np.linalg.inv(B_super)
    r_mpb = R_OVER_A / L_SUPER

    lattice = mp.Lattice(size=mp.Vector3(1,1,0),
        basis1=mp.Vector3(L1[0],L1[1],0), basis2=mp.Vector3(L2[0],L2[1],0))
    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m_idx-2, m_idx+n_idx+2):
            for i2 in range(-n_idx-2, m_idx+n_idx+2):
                pos = i1*a1 + i2*a2
                frac = B_inv @ pos
                f1, f2 = frac[0]%1.0, frac[1]%1.0
                if f1 >= 0.5: f1 -= 1.0
                if f2 >= 0.5: f2 -= 1.0
                geometry.append(mp.Cylinder(radius=r_mpb, center=mp.Vector3(f1,f2,0),
                    material=mp.Medium(epsilon=EPS_ROD)))

    mp.verbosity(0)
    ms = mpb.ModeSolver(geometry=geometry, geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG), num_bands=N_MODES,
        resolution=RES, k_points=[mp.Vector3(0,0,0)])
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    t0 = time.time()
    ms.run_tm()
    t_mpb = time.time() - t0
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)
    print(f"  MPB: {len(geometry)} rods, res={RES}, grid~{RES*round(L_SUPER)}², {N_MODES} bands in {t_mpb:.2f}s")

    # --- FDFD ---
    N_grid = RES * round(L_SUPER)
    tracemalloc.start()
    t0 = time.time()
    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m_idx, n=n_idx,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid)
    L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]), polarization='tm')
    t_build = time.time() - t0

    t0 = time.time()
    evals, _ = eigsh(L_op, k=N_MODES, sigma=SIGMA, which='LM', maxiter=20000, tol=1e-10)
    t_solve = time.time() - t0
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    print(f"  FDFD: grid={N_grid}², DOF={N_grid**2:,}, build={t_build:.2f}s, solve={t_solve:.2f}s, peak_RAM={peak_mb:.0f}MB")
    print(f"  FDFD total: {t_build+t_solve:.2f}s")
