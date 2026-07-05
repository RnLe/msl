"""
Quick timing test for MPB at correct resolution for (38,1).
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import meep as mp
from meep import mpb

M_IDX, N_IDX = 38, 1
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
N_cells = M_IDX**2 + N_IDX**2

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
    a1_l = layer_rot @ np.array([1.0, 0.0])
    a2_l = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-M_IDX - 2, M_IDX + N_IDX + 2):
        for i2 in range(-N_IDX - 2, M_IDX + N_IDX + 2):
            pos = i1 * a1_l + i2 * a2_l
            frac = B_inv @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geometry.append(mp.Cylinder(
                radius=r_mpb,
                center=mp.Vector3(f1, f2, 0),
                material=mp.Medium(epsilon=EPS_ROD)))

print(f"({M_IDX},{N_IDX}): N_cells={N_cells}, L_super={L_SUPER:.2f}")
print(f"{len(geometry)} rods")

for RES in [32]:
    mpb_res = RES * round(L_SUPER)
    print(f"\n--- resolution={mpb_res} (= {RES}/cell * {round(L_SUPER)} cells) ---")
    print(f"    expected grid: {mpb_res}x{mpb_res}")
    
    for n_bands in [10, 100]:
        mp.verbosity(0)
        ms = mpb.ModeSolver(
            geometry=geometry,
            geometry_lattice=lattice,
            default_material=mp.Medium(epsilon=EPS_BG),
            num_bands=n_bands,
            resolution=mpb_res,
            k_points=[mp.Vector3(0, 0, 0)])
        
        fd = os.open(os.devnull, os.O_WRONLY)
        o1, o2 = os.dup(1), os.dup(2)
        os.dup2(fd, 1); os.dup2(fd, 2)
        
        t0 = time.time()
        ms.run_tm()
        t_mpb = time.time() - t0
        
        os.dup2(o1, 1); os.dup2(o2, 2)
        os.close(fd); os.close(o1); os.close(o2)
        
        freqs = np.array(ms.all_freqs)[0]
        eps = ms.get_epsilon()
        print(f"    n_bands={n_bands:3d}: {t_mpb:.1f}s, eps_shape={eps.shape}, "
              f"freq range=[{freqs[0]/L_SUPER:.6f}, {freqs[-1]/L_SUPER:.6f}]")
