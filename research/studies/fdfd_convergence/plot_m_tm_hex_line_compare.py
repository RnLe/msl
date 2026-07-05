#!/usr/bin/env python3
"""
Line plots comparing highest-res FDFD to EA for each angle.
1×4 subplots: mode index vs frequency.
"""
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(STUDY_DIR, 'data_m_tm_hex')
EA_DIR    = os.path.join(STUDY_DIR, 'data_ea_hex')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

STEEL_BLUE   = '#4D7B9E'
FDFD_PURPLE  = '#8E7CC3'
EA_GREEN     = '#6BAF6B'
DARK_STEEL   = '#4F5F6B'

ANGLES_ORDERED = ['1deg', '2deg', '4deg', '8deg']
RESOLUTIONS    = [1, 4, 8, 16]

ANGLE_LABELS = {
    '1deg': r'$\theta \approx 1.0°$',
    '2deg': r'$\theta \approx 2.0°$',
    '4deg': r'$\theta \approx 3.9°$',
    '8deg': r'$\theta \approx 8.3°$',
}

# Highest available resolution per angle
BEST_RES = {
    '1deg': 8,
    '2deg': 16,
    '4deg': 16,
    '8deg': 16,
}

BEST_RES_COLORS = {
    8:  STEEL_BLUE,
    16: FDFD_PURPLE,
}


def load_fdfd():
    """Return dict[angle][res] = sorted 1-D array of all eigenfrequencies."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'fdfd_tm_m_*.npz')))
    data = {}
    for fpath in files:
        m = re.match(r'fdfd_tm_m_(\w+deg)_res(\d+)_f\w+\.npz',
                     os.path.basename(fpath))
        if not m:
            continue
        angle, px = m.group(1), int(m.group(2))
        freqs = np.asarray(np.load(fpath)['freqs'], dtype=float)
        data.setdefault(angle, {}).setdefault(px, []).append(freqs)
    for angle in data:
        for px in data[angle]:
            data[angle][px] = np.unique(np.sort(np.concatenate(data[angle][px])))
    return data


def load_ea():
    """Return dict[angle] = sorted 1-D array of EA eigenfrequencies."""
    ea = {}
    for angle in ANGLES_ORDERED:
        fpath = os.path.join(EA_DIR, f'moire_{angle}.npz')
        if os.path.isfile(fpath):
            freqs = np.asarray(np.load(fpath)['frequencies'], dtype=float)
            ea[angle] = np.sort(freqs)
    return ea


def main():
    fdfd = load_fdfd()
    ea   = load_ea()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for col, angle in enumerate(ANGLES_ORDERED):
        ax = axes[col]
        best_px = BEST_RES[angle]
        fdfd_color = BEST_RES_COLORS[best_px]

        # FDFD: highest res
        fdfd_freqs = fdfd.get(angle, {}).get(best_px)
        if fdfd_freqs is not None:
            modes_fdfd = np.arange(1, len(fdfd_freqs) + 1)
            ax.plot(modes_fdfd, fdfd_freqs, 'o-',
                    color=fdfd_color, ms=3, lw=1.0,
                    label=f'FDFD {best_px} px', zorder=3)

        # EA
        ea_freqs = ea.get(angle)
        if ea_freqs is not None:
            modes_ea = np.arange(1, len(ea_freqs) + 1)
            ax.plot(modes_ea, ea_freqs, 's-',
                    color=EA_GREEN, ms=3, lw=1.0,
                    label='EA', zorder=2)

        ax.set_title(ANGLE_LABELS[angle], fontsize=11)
        ax.set_xlabel('Mode index', fontsize=10)
        if col == 0:
            ax.set_ylabel('Frequency  $f$  [$c/a$]', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc='best')

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        r'FDFD vs EA mode comparison  —  hex rods $\varepsilon{=}12$, $r/a{=}0.22$, M-point',
        fontsize=12, y=1.02)
    fig.tight_layout()

    out = os.path.join(FIG_DIR, 'm_tm_hex_line_compare.svg')
    fig.savefig(out, bbox_inches='tight')
    out_png = os.path.join(FIG_DIR, 'm_tm_hex_line_compare.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
