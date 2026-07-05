#!/usr/bin/env python3
"""
Spectrum ladder plot for the Γ-point FDFD TM convergence study.
Reads data_gamma_tm/*.npz, produces one SVG figure.
"""
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data_gamma_tm')
MPB_DIR  = os.path.join(STUDY_DIR, 'data_gamma_tm_mpb')
FIG_DIR  = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette (StyleGuide.md) ─────────────────────────────────────────
LIGHT_STEEL = '#A5C6DF'
SKY_BLUE    = '#4E9AE1'
STEEL_BLUE  = '#4D7B9E'
DARK_STEEL  = '#4F5F6B'
STARK_ORANGE = '#EBA538'

RES_COLORS = {1: LIGHT_STEEL, 4: SKY_BLUE, 8: STEEL_BLUE}
MPB_COLOR = STARK_ORANGE
MPB_X = 11  # x-position for MPB column on the plot

ANGLES_ORDERED = ['1deg', '2deg', '4deg', '8deg']
RESOLUTIONS = [1, 4, 8]
TARGET_FREQS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

ANGLE_LABELS = {
    '1deg': r'$\theta \approx 1.0°$',
    '2deg': r'$\theta \approx 2.0°$',
    '4deg': r'$\theta \approx 3.9°$',
    '8deg': r'$\theta \approx 8.1°$',
}


def load_all():
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'fdfd_tm_gamma_*.npz')))
    data = {}
    for fpath in files:
        m = re.match(r'fdfd_tm_gamma_(\w+deg)_res(\d+)_f(\d+)\.npz',
                     os.path.basename(fpath))
        if not m:
            continue
        angle, px, ftag = m.group(1), int(m.group(2)), int(m.group(3))
        freq_key = ftag / 100.0
        npz = np.load(fpath)
        entry = {k: npz[k] for k in npz.files}
        data.setdefault(angle, {}).setdefault(px, {})[freq_key] = entry
    return data


def load_mpb():
    """Load MPB Γ-point TM data: mpb[angle][freq] = freqs array."""
    files = sorted(glob.glob(os.path.join(MPB_DIR, 'mpb_tm_gamma_*.npz')))
    mpb = {}
    for fpath in files:
        m = re.match(r'mpb_tm_gamma_(\w+deg)_res\d+_f(\d+)\.npz',
                     os.path.basename(fpath))
        if not m:
            continue
        angle, ftag = m.group(1), int(m.group(2))
        freq_key = ftag / 100.0
        npz = np.load(fpath)
        freqs = np.asarray(npz['freqs'], dtype=float)
        # Only include if the selected freqs actually cover the target
        if np.max(freqs) >= freq_key * 0.9:
            mpb.setdefault(angle, {})[freq_key] = freqs
    return mpb


def main():
    data = load_all()
    mpb_data = load_mpb()
    n = sum(1 for a in data for p in data[a] for f in data[a][p])
    n_mpb = sum(1 for a in mpb_data for f in mpb_data[a])
    print(f'Loaded {n} FDFD + {n_mpb} MPB Γ-point result files')

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(
        len(ANGLES_ORDERED), len(TARGET_FREQS),
        figsize=(16, 10), sharex=True,
        gridspec_kw={'hspace': 0.35, 'wspace': 0.30})

    for row, angle in enumerate(ANGLES_ORDERED):
        for col, tf in enumerate(TARGET_FREQS):
            ax = axes[row, col]
            # FDFD data
            for res in RESOLUTIONS:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    continue
                freqs = np.asarray(entry['freqs'], dtype=float)
                x = np.full_like(freqs, res)
                ax.scatter(x, freqs, marker='_', s=80, linewidths=1.2,
                           color=RES_COLORS[res], zorder=3)

            # MPB overlay
            mpb_freqs = mpb_data.get(angle, {}).get(tf)
            if mpb_freqs is not None:
                x_mpb = np.full_like(mpb_freqs, MPB_X)
                ax.scatter(x_mpb, mpb_freqs, marker='_', s=80, linewidths=1.4,
                           color=MPB_COLOR, zorder=4)

            ax.set_xlim(-1, 14)
            xticks = list(RESOLUTIONS)
            xlabels = [str(r) for r in RESOLUTIONS]
            if mpb_data.get(angle):
                xticks.append(MPB_X)
                xlabels.append('MPB\n64')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels)
            if row == 0:
                ax.set_title(f'$f_{{\\mathrm{{target}}}} = {tf}$',
                             fontsize=9, pad=4)
            if col == 0:
                ax.set_ylabel(ANGLE_LABELS[angle], fontsize=8)
            if row == len(ANGLES_ORDERED) - 1:
                ax.set_xlabel('px / cell', fontsize=8)
            ax.tick_params(labelsize=7, direction='in',
                           which='both', top=True, right=True)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

    # Legend
    handles = [
        Line2D([], [], color=RES_COLORS[1], marker='_', linestyle='None',
               markersize=8, label='FDFD 1 px'),
        Line2D([], [], color=RES_COLORS[4], marker='_', linestyle='None',
               markersize=8, label='FDFD 4 px'),
        Line2D([], [], color=RES_COLORS[8], marker='_', linestyle='None',
               markersize=8, label='FDFD 8 px'),
        Line2D([], [], color=MPB_COLOR, marker='_', linestyle='None',
               markersize=8, markeredgewidth=1.4, label='MPB 64 px'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=8,
              framealpha=0.9, bbox_to_anchor=(0.98, 0.97))

    fig.suptitle(
        'TM eigenfrequencies at Γ — FDFD resolution sweep + MPB reference',
        fontsize=12, y=0.98)

    out = os.path.join(FIG_DIR, 'gamma_tm_spectrum_ladder.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
