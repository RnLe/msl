"""
Compare epsilon grids between MPB and FDFD for (38,1).
This will definitively show if the geometries match.
"""
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from T_direct_validation.supercell_geometry import build_supercell_eps

import meep as mp
from meep import mpb

M_IDX, N_IDX = 38, 1
R_OVER_A = 0.2
EPS_ROD = 8.9
EPS_BG = 1.0
RES = 32

L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
theta_rad = 2 * np.arctan2(N_IDX, M_IDX)
N_cells = M_IDX**2 + N_IDX**2
N_grid = RES * round(L_SUPER)

print(f"({M_IDX},{N_IDX}): θ={np.degrees(theta_rad):.2f}°, N_cells={N_cells}")
print(f"N_grid={N_grid}, L_SUPER={L_SUPER:.4f}")

# ── MPB epsilon ──
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

print(f"  MPB: {len(geometry)} rods placed")

mp.verbosity(0)
ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=lattice,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=1,
    resolution=RES,
    k_points=[mp.Vector3(0, 0, 0)])

# Need to init the grid to get epsilon
fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)
ms.init_params(mp.TM, False)
os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

eps_mpb = ms.get_epsilon()
print(f"  MPB epsilon grid shape: {eps_mpb.shape}")
print(f"  MPB grid dimensions: {ms.get_dims()}")

# ── FDFD epsilon ──
eps_fdfd, info = build_supercell_eps(
    lattice_type='square', m=M_IDX, n=N_IDX,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_grid, Ny=N_grid)
print(f"  FDFD epsilon grid shape: {eps_fdfd.shape}")

# ── Compare ──
# Fill fractions
fill_mpb = np.mean(eps_mpb > (EPS_BG + 0.1))
fill_fdfd = np.mean(eps_fdfd > (EPS_BG + 0.1))
print(f"\n  Fill fraction MPB:  {fill_mpb:.6f}")
print(f"  Fill fraction FDFD: {fill_fdfd:.6f}")

# Mean epsilon
print(f"  Mean eps MPB:  {eps_mpb.mean():.6f}")
print(f"  Mean eps FDFD: {eps_fdfd.mean():.6f}")

# Check if grids are same shape
if eps_mpb.shape == eps_fdfd.shape:
    diff = np.abs(eps_mpb - eps_fdfd)
    print(f"\n  Pixel-wise comparison:")
    print(f"    Max diff:  {diff.max():.6f}")
    print(f"    Mean diff: {diff.mean():.6f}")
    print(f"    Pixels where diff > 0.1: {(diff > 0.1).sum()} / {diff.size}")
elif eps_mpb.shape[0] != N_grid or eps_mpb.shape[1] != N_grid:
    print(f"\n  *** MPB grid size {eps_mpb.shape} DIFFERS from expected {N_grid}x{N_grid} ***")
    print(f"  This means MPB uses a DIFFERENT grid than FDFD!")
    print(f"  Expected grid: {N_grid} = RES * round(L_SUPER) = {RES} * {round(L_SUPER)}")
    
    # Check what MPB's actual resolution/grid is
    mpb_Nx, mpb_Ny = eps_mpb.shape[0], eps_mpb.shape[1]
    print(f"  MPB actual grid: {mpb_Nx} x {mpb_Ny}")
    print(f"  MPB grid / L_SUPER = {mpb_Nx/L_SUPER:.4f}  (expected: {RES})")
    print(f"  MPB grid / round(L_SUPER) = {mpb_Nx/round(L_SUPER)}")
    
    # Compare fill fractions (which are grid-size-independent)
    print(f"\n  Fill fraction MPB:  {fill_mpb:.6f}")
    print(f"  Fill fraction FDFD: {fill_fdfd:.6f}")
    
    # Interpolate and compare
    from scipy.interpolate import RegularGridInterpolator
    s1_mpb = np.linspace(0, 1, mpb_Nx, endpoint=False)
    s2_mpb = np.linspace(0, 1, mpb_Ny, endpoint=False)
    s1_fdfd = np.linspace(0, 1, N_grid, endpoint=False)
    s2_fdfd = np.linspace(0, 1, N_grid, endpoint=False)
    
    # Sample both grids at a few points and compare
    print(f"\n  Spot checks (sample eps at fractional coords):")
    test_points = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.25), (0.1, 0.1)]
    for f1, f2 in test_points:
        i_mpb = int(f1 * mpb_Nx) % mpb_Nx
        j_mpb = int(f2 * mpb_Ny) % mpb_Ny
        i_fdfd = int(f1 * N_grid) % N_grid
        j_fdfd = int(f2 * N_grid) % N_grid
        print(f"    ({f1:.2f}, {f2:.2f}): MPB={eps_mpb[i_mpb,j_mpb]:.4f}, FDFD={eps_fdfd[i_fdfd,j_fdfd]:.4f}")

# Also run for (11,1) as comparison
print("\n" + "="*60)
print("(11,1) for comparison:")
print("="*60)

M2, N2 = 11, 1
L1_11 = np.array([M2, N2], dtype=float)
L2_11 = np.array([-N2, M2], dtype=float)
L_S_11 = np.sqrt(L1_11 @ L1_11)
theta_11 = 2 * np.arctan2(N2, M2)
N_g_11 = RES * round(L_S_11)

c11, s11 = np.cos(theta_11), np.sin(theta_11)
R_11 = np.array([[c11, -s11], [s11, c11]])
B_s_11 = np.column_stack([L1_11, L2_11])
B_i_11 = np.linalg.inv(B_s_11)
r_m_11 = R_OVER_A / L_S_11

lat_11 = mp.Lattice(size=mp.Vector3(1, 1, 0),
    basis1=mp.Vector3(L1_11[0], L1_11[1], 0),
    basis2=mp.Vector3(L2_11[0], L2_11[1], 0))

geom_11 = []
for layer_rot in [np.eye(2), R_11]:
    a1_l = layer_rot @ np.array([1.0, 0.0])
    a2_l = layer_rot @ np.array([0.0, 1.0])
    for i1 in range(-M2 - 2, M2 + N2 + 2):
        for i2 in range(-N2 - 2, M2 + N2 + 2):
            pos = i1 * a1_l + i2 * a2_l
            frac = B_i_11 @ pos
            f1, f2 = frac[0] % 1.0, frac[1] % 1.0
            if f1 >= 0.5: f1 -= 1.0
            if f2 >= 0.5: f2 -= 1.0
            geom_11.append(mp.Cylinder(
                radius=r_m_11,
                center=mp.Vector3(f1, f2, 0),
                material=mp.Medium(epsilon=EPS_ROD)))

ms11 = mpb.ModeSolver(
    geometry=geom_11,
    geometry_lattice=lat_11,
    default_material=mp.Medium(epsilon=EPS_BG),
    num_bands=1,
    resolution=RES,
    k_points=[mp.Vector3(0, 0, 0)])

fd = os.open(os.devnull, os.O_WRONLY)
o1, o2 = os.dup(1), os.dup(2)
os.dup2(fd, 1); os.dup2(fd, 2)
ms11.init_params(mp.TM, False)
os.dup2(o1, 1); os.dup2(o2, 2)
os.close(fd); os.close(o1); os.close(o2)

eps_m11 = ms11.get_epsilon()
print(f"  MPB grid shape: {eps_m11.shape}")
print(f"  MPB grid dims: {ms11.get_dims()}")
print(f"  Expected N_grid: {N_g_11}")

eps_f11, _ = build_supercell_eps(
    lattice_type='square', m=M2, n=N2,
    r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
    Nx=N_g_11, Ny=N_g_11)

fill_m11 = np.mean(eps_m11 > (EPS_BG + 0.1))
fill_f11 = np.mean(eps_f11 > (EPS_BG + 0.1))
print(f"  Fill fraction MPB:  {fill_m11:.6f}")
print(f"  Fill fraction FDFD: {fill_f11:.6f}")
print(f"  Mean eps MPB:  {eps_m11.mean():.6f}")
print(f"  Mean eps FDFD: {eps_f11.mean():.6f}")

if eps_m11.shape == eps_f11.shape:
    diff = np.abs(eps_m11 - eps_f11)
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
elif eps_m11.shape[0] != N_g_11:
    print(f"  *** MPB grid {eps_m11.shape} DIFFERS from FDFD grid ({N_g_11},{N_g_11}) ***")
