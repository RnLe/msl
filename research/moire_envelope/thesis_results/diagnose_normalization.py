"""
Quick diagnostic: test MPB vs FDFD normalization at (11,1) res=32 and (38,1) res=32.
Also test vacuum (no rods) to isolate geometry vs operator issues.
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.fdfd_solver import build_fdfd_operator
from scipy.sparse.linalg import eigsh

import meep as mp
from meep import mpb

R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
RES = 32
N_MODES = 20

def run_comparison(m_idx, n_idx):
    L1 = np.array([m_idx, n_idx], dtype=float)
    L2 = np.array([-n_idx, m_idx], dtype=float)
    L_SUPER = np.sqrt(L1 @ L1)
    theta_rad = 2 * np.arctan2(n_idx, m_idx)
    N_cells = m_idx**2 + n_idx**2
    
    print(f"\n{'='*60}")
    print(f"({m_idx},{n_idx}): θ={np.degrees(theta_rad):.2f}°, N_cells={N_cells}, L_super={L_SUPER:.4f}")
    print(f"{'='*60}")
    
    # ── MPB ──
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    B_super = np.column_stack([L1, L2])
    B_inv = np.linalg.inv(B_super)
    r_mpb = R_OVER_A / L_SUPER
    
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0),
        basis2=mp.Vector3(L2[0], L2[1], 0))
    
    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m_idx - 2, m_idx + n_idx + 2):
            for i2 in range(-n_idx - 2, m_idx + n_idx + 2):
                pos = i1 * a1 + i2 * a2
                frac = B_inv @ pos
                f1, f2 = frac[0] % 1.0, frac[1] % 1.0
                if f1 >= 0.5: f1 -= 1.0
                if f2 >= 0.5: f2 -= 1.0
                geometry.append(mp.Cylinder(
                    radius=r_mpb,
                    center=mp.Vector3(f1, f2, 0),
                    material=mp.Medium(epsilon=EPS_ROD)))
    
    mp.verbosity(0)
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=N_MODES,
        resolution=RES,
        k_points=[mp.Vector3(0, 0, 0)])
    
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    ms.run_tm()
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)
    
    freqs_mpb_raw = np.array(ms.all_freqs)[0]
    freqs_mpb = freqs_mpb_raw / L_SUPER
    
    # ── FDFD ──
    N_grid = RES * round(L_SUPER)
    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m_idx, n=n_idx,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid)
    
    L_op = build_fdfd_operator(eps_grid, info, q_vec=np.array([0.0, 0.0]),
                               polarization='tm')
    del eps_grid
    
    evals, _ = eigsh(L_op, k=N_MODES, sigma=-0.01, which='LM',
                     maxiter=20000, tol=1e-10)
    idx = np.argsort(evals)
    evals = evals[idx]
    freqs_fdfd = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    
    # ── Compare ──
    print(f"  MPB raw range:  [{freqs_mpb_raw[0]:.6f}, {freqs_mpb_raw[-1]:.6f}]")
    print(f"  MPB /L_SUPER:   [{freqs_mpb[0]:.6f}, {freqs_mpb[-1]:.6f}]")
    print(f"  FDFD range:     [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}]")
    
    # Check both normalizations
    mask = (freqs_mpb > 0.001) & (freqs_fdfd > 0.001)
    if mask.sum() > 0:
        rel_err_divided = np.abs(freqs_mpb[mask] - freqs_fdfd[mask]) / freqs_mpb[mask]
        print(f"  With /L_SUPER:  max={rel_err_divided.max():.4%}, mean={rel_err_divided.mean():.4%}")
    
    mask2 = (freqs_mpb_raw > 0.001) & (freqs_fdfd > 0.001)
    if mask2.sum() > 0:
        rel_err_raw = np.abs(freqs_mpb_raw[mask2] - freqs_fdfd[mask2]) / freqs_mpb_raw[mask2]
        print(f"  Without /L_SUPER: max={rel_err_raw.max():.4%}, mean={rel_err_raw.mean():.4%}")
    
    # Print first few modes for comparison
    print(f"\n  Mode  MPB_raw     MPB/L     FDFD       ratio(MPB/L / FDFD)")
    for i in range(min(10, N_MODES)):
        if freqs_fdfd[i] > 1e-6:
            ratio = freqs_mpb[i] / freqs_fdfd[i]
        else:
            ratio = float('nan')
        print(f"  {i:4d}  {freqs_mpb_raw[i]:.6f}  {freqs_mpb[i]:.6f}  {freqs_fdfd[i]:.6f}  {ratio:.4f}")

run_comparison(11, 1)
run_comparison(38, 1)
