#!/usr/bin/env python3
"""
5x5 subplot of lowest 25 MPB Bloch modes |u_n(r)|² at the Gamma point.
Hex rods lattice, 8° commensurate angle, 64 px/cell.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(STUDY_DIR, 'data_gamma_tm_hex')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

NPZ = os.path.join(DATA_DIR, 'mpb_tm_gamma_8deg_res64_50bands.npz')


def main():
    d = np.load(NPZ, allow_pickle=True)
    fields = d['bloch_fields']  # (50, Nx, Ny) complex
    freqs  = d['freqs_all']     # (50,) in units of c/a

    nrows, ncols = 5, 5
    n_modes = nrows * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.15})

    for idx in range(n_modes):
        ax = axes[idx // ncols, idx % ncols]
        u = fields[idx]
        intensity = np.abs(u) ** 2

        im = ax.imshow(intensity.T, origin='lower', cmap='inferno',
                       interpolation='bilinear',
                       norm=PowerNorm(gamma=0.25, vmin=0, vmax=intensity.max()))
        ax.set_title(f'$n={idx+1}$,  $f={freqs[idx]:.5f}$', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        r'$|u_n(\mathbf{r})|^2$ at $\Gamma$  —  hex rods $\varepsilon{=}12$, '
        r'$\theta \approx 8.3°$, MPB 64 px',
        fontsize=13, y=0.98)

    out_png = os.path.join(FIG_DIR, 'gamma_tm_hex_mpb_8deg_modes_5x5.png')
    out_svg = os.path.join(FIG_DIR, 'gamma_tm_hex_mpb_8deg_modes_5x5.svg')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_png}')
    print(f'Saved {out_svg}')


if __name__ == '__main__':
    main()
