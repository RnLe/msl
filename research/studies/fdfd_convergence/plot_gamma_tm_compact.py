#!/usr/bin/env python3
"""
Compact spectrum overview for the Γ-point TM convergence study.

Instead of a 4×6 subplot grid, this collapses all target-frequency windows
into a continuous frequency axis per angle.  Each angle gets one row; within
each row, resolution levels (1, 4, 8 px  + MPB 64 px) appear as horizontal
lanes, and eigenvalues are drawn as thin vertical lines inside their lane.
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
DATA_DIR  = os.path.join(STUDY_DIR, 'data_gamma_tm')
MPB_DIR   = os.path.join(STUDY_DIR, 'data_gamma_tm_mpb_corrected')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette (StyleGuide.md) ─────────────────────────────────────────
LIGHT_STEEL  = '#A5C6DF'
SKY_BLUE     = '#4E9AE1'
STEEL_BLUE   = '#4D7B9E'
DARK_STEEL   = '#4F5F6B'
STARK_ORANGE = '#EBA538'
LIGHT_BROWN  = '#E3D5BF'

ANGLES_ORDERED = ['1deg', '2deg', '4deg', '8deg']
RESOLUTIONS    = [1, 4, 8]
TARGET_FREQS   = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

ANGLE_LABELS = {
    '1deg': r'$\theta \approx 1.0°$',
    '2deg': r'$\theta \approx 2.0°$',
    '4deg': r'$\theta \approx 3.9°$',
    '8deg': r'$\theta \approx 8.1°$',
}

# Lane y-offsets and visual properties
LANE_LABELS = ['1 px', '4 px', '8 px', 'MPB\n64 px/a']
LANE_COLORS = [LIGHT_STEEL, SKY_BLUE, STEEL_BLUE, STARK_ORANGE]
N_LANES     = len(LANE_LABELS)
LANE_HEIGHT = 0.50          # vertical extent of each line
LANE_GAP    = 0.70          # spacing between lane centres


# ── data loaders ────────────────────────────────────────────────────
def load_fdfd():
    """Return dict[angle][res] = sorted 1-D array of all eigenfrequencies."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'fdfd_tm_gamma_*.npz')))
    data = {}
    for fpath in files:
        m = re.match(r'fdfd_tm_gamma_(\w+deg)_res(\d+)_f(\d+)\.npz',
                     os.path.basename(fpath))
        if not m:
            continue
        angle, px = m.group(1), int(m.group(2))
        freqs = np.asarray(np.load(fpath)['freqs'], dtype=float)
        data.setdefault(angle, {}).setdefault(px, []).append(freqs)
    # concatenate all target-frequency windows per (angle, res)
    for angle in data:
        for px in data[angle]:
            data[angle][px] = np.sort(np.concatenate(data[angle][px]))
    return data


def load_mpb():
    """Return dict[angle] = sorted 1-D array of all eigenfrequencies.
    Reads corrected MPB data (proper resolution per unit cell)."""
    files = sorted(glob.glob(os.path.join(MPB_DIR, 'mpb_tm_gamma_*.npz')))
    mpb = {}
    for fpath in files:
        m = re.match(r'mpb_tm_gamma_(\w+deg)_',
                     os.path.basename(fpath))
        if not m:
            continue
        angle = m.group(1)
        npz = np.load(fpath)
        freqs = np.asarray(npz['freqs_all'], dtype=float)
        mpb.setdefault(angle, []).append(freqs)
    for angle in mpb:
        mpb[angle] = np.sort(np.unique(np.concatenate(mpb[angle])))
    return mpb


# ── main plotting ───────────────────────────────────────────────────
def main():
    fdfd = load_fdfd()
    mpb  = load_mpb()
    n_fdfd = sum(len(fdfd[a][p]) for a in fdfd for p in fdfd[a])
    n_mpb  = sum(len(mpb[a]) for a in mpb)
    print(f'Loaded {n_fdfd} FDFD + {n_mpb} MPB eigenvalues')

    fig, axes = plt.subplots(
        len(ANGLES_ORDERED), 1,
        figsize=(12, 5.2), sharex=True,
        gridspec_kw={'hspace': 0.22})

    for row, angle in enumerate(ANGLES_ORDERED):
        ax = axes[row]

        # Determine which lanes this angle has
        has_mpb      = angle in mpb
        lanes_used   = N_LANES if has_mpb else N_LANES - 1
        lane_centres = np.arange(lanes_used) * LANE_GAP

        # Light horizontal bands behind each lane
        for i in range(lanes_used):
            ylo = lane_centres[i] - LANE_HEIGHT / 2
            yhi = lane_centres[i] + LANE_HEIGHT / 2
            ax.axhspan(ylo, yhi, color=LANE_COLORS[i], alpha=0.08, zorder=0)

        # FDFD eigenvalues as vertical lines
        for i, res in enumerate(RESOLUTIONS):
            freqs = fdfd.get(angle, {}).get(res)
            if freqs is None:
                continue
            ax.vlines(freqs, lane_centres[i] - LANE_HEIGHT / 2,
                      lane_centres[i] + LANE_HEIGHT / 2,
                      colors=LANE_COLORS[i], linewidths=0.45,
                      alpha=0.85, zorder=2)

        # MPB eigenvalues
        if has_mpb:
            i_mpb = 3
            ax.vlines(mpb[angle],
                      lane_centres[i_mpb] - LANE_HEIGHT / 2,
                      lane_centres[i_mpb] + LANE_HEIGHT / 2,
                      colors=LANE_COLORS[i_mpb], linewidths=0.55,
                      alpha=0.9, zorder=3)

        # Target-frequency markers (subtle dashed verticals)
        for tf in TARGET_FREQS:
            ax.axvline(tf, color=DARK_STEEL, ls=':', lw=0.4,
                       alpha=0.35, zorder=1)

        # Y-axis: lane labels
        ax.set_yticks(lane_centres)
        ax.set_yticklabels(LANE_LABELS[:lanes_used], fontsize=9)
        ax.set_ylim(lane_centres[0] - LANE_HEIGHT * 0.7,
                    lane_centres[-1] + LANE_HEIGHT * 0.7)
        ax.invert_yaxis()           # 1 px at top, MPB at bottom

        # Angle label on the right
        ax.text(1.01, 0.5, ANGLE_LABELS[angle], transform=ax.transAxes,
                fontsize=10, va='center', ha='left')

        # Cosmetics
        ax.tick_params(labelsize=9, direction='in',
                       which='both', top=(row == 0), right=False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.6)

    # Shared x-axis label
    axes[-1].set_xlabel('Frequency  $f$  [$c/a$]', fontsize=11)
    axes[-1].set_xlim(-0.01, 0.46)

    # Legend
    handles = [
        Line2D([], [], color=LANE_COLORS[0], lw=1.5, label='FDFD 1 px'),
        Line2D([], [], color=LANE_COLORS[1], lw=1.5, label='FDFD 4 px'),
        Line2D([], [], color=LANE_COLORS[2], lw=1.5, label='FDFD 8 px'),
        Line2D([], [], color=LANE_COLORS[3], lw=1.5, label='MPB 64 px'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               framealpha=0.9, edgecolor=LIGHT_BROWN,
               bbox_to_anchor=(0.98, 0.98), ncol=4)

    fig.suptitle(
        'TM eigenfrequencies at Γ  —  compact resolution comparison',
        fontsize=12, y=1.0)

    out = os.path.join(FIG_DIR, 'gamma_tm_spectrum_compact.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
