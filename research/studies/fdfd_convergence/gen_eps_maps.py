#!/usr/bin/env python3
"""
Generate and plot 64 px/cell epsilon maps for all 4 moiré angles.
Saves .npz files for EA pipeline and a 4-panel plot.
"""
import os, sys
import numpy as np

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_RESULTS = os.path.join(
    os.path.dirname(STUDY_DIR), '..',
    'moire_envelope', 'thesis_results')
sys.path.insert(0, os.path.abspath(THESIS_RESULTS))

from T_direct_validation.supercell_geometry import build_supercell_eps

R_OVER_A    = 0.2
EPS_ROD     = 8.9
EPS_BG      = 1.0
PX_PER_CELL = 64

ANGLES = [
    {'m': 14,  'n': 1, 'label': '8deg'},
    {'m': 29,  'n': 1, 'label': '4deg'},
    {'m': 57,  'n': 1, 'label': '2deg'},
    {'m': 114, 'n': 1, 'label': '1deg'},
]

DATA_DIR = os.path.join(STUDY_DIR, 'data_eps_maps')
FIG_DIR  = os.path.join(STUDY_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette ──
LIGHT_BROWN = '#E3D5BF'

results = []

for a in ANGLES:
    m, n, label = a['m'], a['n'], a['label']
    L1 = np.array([m, n], dtype=float)
    L_super = np.sqrt(L1 @ L1)
    N_grid = PX_PER_CELL * round(L_super)

    print(f'{label}: grid={N_grid}×{N_grid} ...', end=' ', flush=True)

    eps_grid, info = build_supercell_eps(
        lattice_type='square', m=m, n=n,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=N_grid, Ny=N_grid,
        subpixel_smoothing=True, smoothing_Nsub=8)

    # Reconstruct physical coordinates
    s1 = np.arange(N_grid) / N_grid
    s2 = np.arange(N_grid) / N_grid
    L1_vec = info['L1']
    L2_vec = info['L2']
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1_vec[0] + S2 * L2_vec[0]
    Y = S1 * L1_vec[1] + S2 * L2_vec[1]

    # Save for EA pipeline
    fname = f'eps_map_{label}_res{PX_PER_CELL}.npz'
    np.savez(os.path.join(DATA_DIR, fname),
             eps_grid=eps_grid,
             X=X, Y=Y,
             L1=L1_vec, L2=L2_vec,
             B_super=info['B_super'],
             m=m, n=n,
             theta_deg=info['theta_deg'],
             r_over_a=R_OVER_A,
             eps_rod=EPS_ROD, eps_bg=EPS_BG,
             px_per_cell=PX_PER_CELL,
             Nx=N_grid, Ny=N_grid)

    print(f'saved {fname}  eps range=[{eps_grid.min():.2f}, {eps_grid.max():.2f}]')
    results.append((label, info, eps_grid, X, Y))

# ── 4-panel plot ──
print('\nPlotting...')
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
axes = axes.ravel()

angle_titles = {
    '8deg': r'$\theta \approx 8.1°$',
    '4deg': r'$\theta \approx 3.9°$',
    '2deg': r'$\theta \approx 2.0°$',
    '1deg': r'$\theta \approx 1.0°$',
}

for idx, (label, info, eps_grid, X, Y) in enumerate(results):
    ax = axes[idx]
    # Use pcolormesh for non-rectangular (skewed) grids
    im = ax.pcolormesh(X, Y, eps_grid, shading='auto',
                       cmap='cividis', vmin=EPS_BG, vmax=EPS_ROD,
                       rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(angle_titles[label], fontsize=12)
    ax.set_xlabel('$x / a$', fontsize=10)
    ax.set_ylabel('$y / a$', fontsize=10)
    ax.tick_params(labelsize=9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.06)
    fig.colorbar(im, cax=cax, label=r'$\varepsilon$')

fig.suptitle(r'$\varepsilon(\mathbf{r})$ — moiré supercell (64 px/a, 8×8 subpixel)',
             fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

for ext in ('svg', 'png'):
    out = os.path.join(FIG_DIR, f'eps_maps_64px.{ext}')
    fig.savefig(out, bbox_inches='tight', dpi=150 if ext == 'png' else None)
    print(f'Saved {out}')
plt.close(fig)
