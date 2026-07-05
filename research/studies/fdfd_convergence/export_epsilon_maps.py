#!/usr/bin/env python3
"""
Export the epsilon maps that FDFD and MPB "see" at resolution 32 px/cell.

Produces 8 PNGs (4 angles × 2 methods) in figures/epsilon_maps/.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(STUDY_DIR, 'figures', 'epsilon_maps')
os.makedirs(FIG_DIR, exist_ok=True)

THESIS_RESULTS = os.path.join(
    os.path.dirname(STUDY_DIR), '..',
    'moire_envelope', 'thesis_results')
sys.path.insert(0, os.path.abspath(THESIS_RESULTS))

# ── physics ─────────────────────────────────────────────────────────
R_OVER_A = 0.2
EPS_ROD  = 8.9
EPS_BG   = 1.0
RES      = 32        # px per unit cell

# ── palette (StyleGuide.md) ─────────────────────────────────────────
LIGHT_BROWN  = '#E3D5BF'

ANGLES = [
    {'m': 14,  'n': 1, 'label': '8deg', 'title': r'$\theta \approx 8.1°$'},
    {'m': 29,  'n': 1, 'label': '4deg', 'title': r'$\theta \approx 3.9°$'},
    {'m': 57,  'n': 1, 'label': '2deg', 'title': r'$\theta \approx 2.0°$'},
    {'m': 114, 'n': 1, 'label': '1deg', 'title': r'$\theta \approx 1.0°$'},
]


# ── FDFD epsilon (via build_supercell_eps) ──────────────────────────
def fdfd_epsilon(m, n):
    """Return (eps_grid, L_super) using the FDFD supercell builder at res=32."""
    from T_direct_validation.supercell_geometry import build_supercell_eps

    L_super = np.sqrt(m**2 + n**2)
    N_grid  = RES * round(L_super)

    eps, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)
    return eps, info


# ── MPB epsilon (via meep.mpb ModeSolver) ───────────────────────────
def mpb_epsilon(m, n):
    """Return 2-D epsilon array from MPB's internal dielectric grid at res=32."""
    import meep as mp
    from meep import mpb

    L1 = np.array([m, n], dtype=float)
    L2 = np.array([-n, m], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    theta_rad = 2 * np.arctan2(n, m)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R_mat = np.array([[c, -s], [s, c]])
    B_super = np.column_stack([L1, L2])
    B_inv   = np.linalg.inv(B_super)
    r_mpb   = R_OVER_A / L_super

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0),
        basis2=mp.Vector3(L2[0], L2[1], 0))

    geometry = []
    for layer_rot in [np.eye(2), R_mat]:
        a1 = layer_rot @ np.array([1.0, 0.0])
        a2 = layer_rot @ np.array([0.0, 1.0])
        for i1 in range(-m - 2, m + n + 2):
            for i2 in range(-n - 2, m + n + 2):
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
        num_bands=1,
        resolution=RES,
        k_points=[mp.Vector3(0, 0, 0)])

    ms.init_params(mp.NO_PARITY, False)
    eps = ms.get_epsilon()
    return np.array(eps), L_super


# ── plotting helper ─────────────────────────────────────────────────
def plot_eps(eps, title, outpath, vmin=EPS_BG, vmax=EPS_ROD):
    """Save a single epsilon map as PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(eps.T, origin='lower', cmap='inferno',
                   vmin=vmin, vmax=vmax, interpolation='none')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\varepsilon$', fontsize=12)
    cb.ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel('pixel $i_1$', fontsize=11)
    ax.set_ylabel('pixel $i_2$', fontsize=11)
    ax.tick_params(labelsize=9)
    fig.savefig(outpath, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved {outpath}')


# ── main ────────────────────────────────────────────────────────────
def main():
    # ── FDFD epsilon maps ───────────────────────────────────────────
    print('=== FDFD epsilon maps (subpixel smoothed, res=32) ===')
    for ang in ANGLES:
        m, n, label, title = ang['m'], ang['n'], ang['label'], ang['title']
        L_super = np.sqrt(m**2 + n**2)
        N_grid  = RES * round(L_super)
        print(f'  {label}  (m={m}, n={n})  grid={N_grid}×{N_grid}')
        eps, info = fdfd_epsilon(m, n)
        out = os.path.join(FIG_DIR, f'fdfd_eps_{label}_res{RES}.png')
        plot_eps(eps, f'FDFD  {title}  ({N_grid}×{N_grid}, smooth)', out)

    # ── MPB epsilon maps ────────────────────────────────────────────
    print('\n=== MPB epsilon maps (res=32) ===')
    for ang in ANGLES:
        m, n, label, title = ang['m'], ang['n'], ang['label'], ang['title']
        L_super = np.sqrt(m**2 + n**2)
        print(f'  {label}  (m={m}, n={n})')
        eps, _ = mpb_epsilon(m, n)
        out = os.path.join(FIG_DIR, f'mpb_eps_{label}_res{RES}.png')
        plot_eps(eps, f'MPB  {title}  ({eps.shape[0]}×{eps.shape[1]})', out)

    print('\nDone.')


if __name__ == '__main__':
    main()
