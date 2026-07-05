#!/usr/bin/env python3
"""
Generate epsilon material maps for non-commensurate moiré cells.
Emit WebP images at 8 px/cell for all four angles so the user can
visually confirm each is exactly one moiré cell.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY_DIR)

from common import (
    ANGLES, R_OVER_A, EPS_ROD, EPS_BG, A,
    build_moire_supercell_eps,
)

DATA_DIR = os.path.join(STUDY_DIR, 'data_eps')
FIG_DIR  = os.path.join(STUDY_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PX_PER_CELL = 8   # resolution for the material map preview


def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

    for ax, (label, theta_deg) in zip(axes, ANGLES.items()):
        theta_rad = np.radians(theta_deg)
        L_moire = A / (2.0 * np.sin(theta_rad / 2.0))
        N_grid = int(round(PX_PER_CELL * L_moire / A))

        print(f'{label}: θ = {theta_deg}°  L_m/a = {L_moire:.4f}  '
              f'grid = {N_grid}×{N_grid}')

        eps_grid, info = build_moire_supercell_eps(
            lattice_type='square',
            theta_rad=theta_rad,
            a=A,
            r_over_a=R_OVER_A,
            eps_rod=EPS_ROD,
            eps_bg=EPS_BG,
            Nx=N_grid,
            Ny=N_grid,
            subpixel_smoothing=True,
            smoothing_Nsub=8,
        )

        # Save data
        np.savez(
            os.path.join(DATA_DIR, f'eps_noncomm_{label}_res{PX_PER_CELL}.npz'),
            eps_grid=eps_grid,
            theta_deg=theta_deg,
            theta_rad=theta_rad,
            L_moire=L_moire,
            N_grid=N_grid,
            px_per_cell=PX_PER_CELL,
            L1=info['L1'],
            L2=info['L2'],
            B_super=info['B_super'],
        )

        # Plot
        extent = [0, L_moire, 0, L_moire]
        ax.imshow(
            eps_grid.T, origin='lower', extent=extent,
            cmap='gray_r', interpolation='nearest',
            vmin=EPS_BG, vmax=EPS_ROD,
        )
        ax.set_title(
            f'$\\theta = {theta_deg}°$\n'
            f'$L_m/a = {L_moire:.2f}$,  grid = {N_grid}²',
            fontsize=13,
        )
        ax.set_xlabel('$x / a$', fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel('$y / a$', fontsize=12)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        'Non-commensurate moiré cells  —  ε map at 8 px/cell  (TM, square)',
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    # Save as WebP (per style guide: complex images → WebP instead of SVG)
    out = os.path.join(FIG_DIR, 'eps_maps_noncomm_8px.webp')
    fig.savefig(out, bbox_inches='tight', dpi=200, format='webp')
    print(f'Saved {out}')
    plt.close(fig)

    # ── 3×3 tiling to check periodicity ─────────────────────────────
    fig3, axes3 = plt.subplots(1, 4, figsize=(24, 6.5))

    for ax, (label, theta_deg) in zip(axes3, ANGLES.items()):
        d = np.load(os.path.join(DATA_DIR,
                    f'eps_noncomm_{label}_res{PX_PER_CELL}.npz'))
        eps = d['eps_grid']
        L_moire = float(d['L_moire'])

        # Tile 3×3
        tiled = np.tile(eps, (3, 3))
        extent = [0, 3 * L_moire, 0, 3 * L_moire]
        ax.imshow(
            tiled.T, origin='lower', extent=extent,
            cmap='gray_r', interpolation='nearest',
            vmin=EPS_BG, vmax=EPS_ROD,
        )
        # Draw cell boundaries
        for k in range(1, 3):
            ax.axvline(k * L_moire, color='#EBA538', lw=0.8, ls='--', alpha=0.7)
            ax.axhline(k * L_moire, color='#EBA538', lw=0.8, ls='--', alpha=0.7)

        N_grid = int(d['N_grid'])
        ax.set_title(
            f'$\\theta = {theta_deg}°$\n'
            f'$L_m/a = {L_moire:.2f}$,  grid = {N_grid}²',
            fontsize=13,
        )
        ax.set_xlabel('$x / a$', fontsize=12)
        if ax is axes3[0]:
            ax.set_ylabel('$y / a$', fontsize=12)
        ax.tick_params(labelsize=10)

    fig3.suptitle(
        'Non-commensurate moiré cells  —  3×3 tiling at 8 px/cell',
        fontsize=14, y=1.02,
    )
    plt.tight_layout()

    out3 = os.path.join(FIG_DIR, 'eps_maps_noncomm_8px_3x3.webp')
    fig3.savefig(out3, bbox_inches='tight', dpi=200, format='webp')
    print(f'Saved {out3}')
    plt.close(fig3)

    # Also save individual per-angle maps
    for label, theta_deg in ANGLES.items():
        theta_rad = np.radians(theta_deg)
        L_moire = A / (2.0 * np.sin(theta_rad / 2.0))
        N_grid = int(round(PX_PER_CELL * L_moire / A))

        d = np.load(os.path.join(DATA_DIR,
                    f'eps_noncomm_{label}_res{PX_PER_CELL}.npz'))
        eps = d['eps_grid']

        fig_s, ax_s = plt.subplots(figsize=(7, 7))
        extent = [0, L_moire, 0, L_moire]
        ax_s.imshow(
            eps.T, origin='lower', extent=extent,
            cmap='gray_r', interpolation='nearest',
            vmin=EPS_BG, vmax=EPS_ROD,
        )
        ax_s.set_title(
            f'Non-commensurate  $\\theta = {theta_deg}°$\n'
            f'$L_m/a = {L_moire:.2f}$,  grid = {N_grid}²',
            fontsize=14,
        )
        ax_s.set_xlabel('$x / a$', fontsize=13)
        ax_s.set_ylabel('$y / a$', fontsize=13)
        ax_s.tick_params(labelsize=11)

        out_s = os.path.join(FIG_DIR, f'eps_noncomm_{label}_8px.webp')
        fig_s.savefig(out_s, bbox_inches='tight', dpi=200, format='webp')
        plt.close(fig_s)
        print(f'Saved {out_s}')

    plt.close(fig)
    print('Done.')


if __name__ == '__main__':
    main()
