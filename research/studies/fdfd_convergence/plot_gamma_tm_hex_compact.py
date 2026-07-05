#!/usr/bin/env python3
"""
Compact spectrum overview for the Gamma-point TM convergence study (hex rods).

One horizontal panel per angle, resolution lanes with
vertical-line eigenvalues around the Gamma-point target frequency.
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
DATA_DIR  = os.path.join(STUDY_DIR, 'data_gamma_tm_hex')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────
LIGHT_STEEL  = '#A5C6DF'
SKY_BLUE     = '#4E9AE1'
STEEL_BLUE   = '#4D7B9E'
DARK_STEEL   = '#4F5F6B'
STARK_ORANGE = '#EBA538'
LIGHT_BROWN  = '#E3D5BF'
FDFD_PURPLE  = '#8E7CC3'
EA_GREEN     = '#6BAF6B'
MPB_GOLD     = '#D4920B'

ANGLES_ORDERED = ['1deg', '2deg', '4deg', '8deg']
RESOLUTIONS    = [1, 4, 8, 16]
TARGET_FREQS   = [0.05]

ANGLE_LABELS = {
    '1deg': r'$\theta \approx 1.0°$',
    '2deg': r'$\theta \approx 2.0°$',
    '4deg': r'$\theta \approx 3.9°$',
    '8deg': r'$\theta \approx 8.3°$',
}

LANE_LABELS = ['1 px', '4 px', '8 px', '16 px', 'EA', 'MPB 64 px']
LANE_COLORS = [LIGHT_STEEL, SKY_BLUE, STEEL_BLUE, FDFD_PURPLE, EA_GREEN, MPB_GOLD]
LANE_HEIGHT = 0.50
LANE_GAP    = 0.70


# ── data loader ─────────────────────────────────────────────────────
def load_fdfd():
    """Return dict[angle][res] = sorted 1-D array of all eigenfrequencies."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, 'fdfd_tm_gamma_*.npz')))
    data = {}
    for fpath in files:
        m = re.match(r'fdfd_tm_gamma_(\w+deg)_res(\d+)_f\w+\.npz',
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


# ── EA loader ───────────────────────────────────────────────────────
EA_FILES = {
    '8deg': 'gamma_commensurate_m14_n1_tiled.npz',
    '4deg': 'gamma_commensurate_m29_n1_tiled.npz',
    '2deg': 'gamma_commensurate_m57_n1_tiled.npz',
    '1deg': 'gamma_commensurate_m114_n1_tiled.npz',
}
EA_LANE_IDX = len(RESOLUTIONS)  # index 4 in LANE_LABELS / LANE_COLORS
MPB_LANE_IDX = EA_LANE_IDX + 1  # index 5

MPB_FILE = 'mpb_tm_gamma_8deg_res64_50bands.npz'


def load_mpb():
    """Return dict[angle] = sorted 1-D array of MPB frequencies."""
    data = {}
    fpath = os.path.join(DATA_DIR, MPB_FILE)
    if os.path.isfile(fpath):
        d = np.load(fpath, allow_pickle=True)
        freqs = np.sort(np.asarray(d['freqs_all'], dtype=float))
        data['8deg'] = freqs
    return data


def load_ea():
    """Return dict[angle] = sorted 1-D array of EA frequencies."""
    data = {}
    for angle, fname in EA_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        freqs = np.sort(np.asarray(d['frequencies'], dtype=float))
        data[angle] = freqs
    return data


# ── main plotting ───────────────────────────────────────────────────
def main():
    fdfd = load_fdfd()
    ea   = load_ea()
    mpb  = load_mpb()
    n_fdfd = sum(len(fdfd[a][p]) for a in fdfd for p in fdfd[a])
    n_ea   = sum(len(v) for v in ea.values())
    n_mpb  = sum(len(v) for v in mpb.values())
    print(f'Loaded {n_fdfd} FDFD + {n_ea} EA + {n_mpb} MPB eigenvalues')

    fig, axes = plt.subplots(
        len(ANGLES_ORDERED), 1,
        figsize=(12, 8.0), sharex=True,
        gridspec_kw={'hspace': 0.25})

    for row, angle in enumerate(ANGLES_ORDERED):
        ax = axes[row]

        # Figure out which lanes are active (FDFD + EA + MPB)
        lane_active = [res in fdfd.get(angle, {}) for res in RESOLUTIONS]
        if angle in ea:
            lane_active.append(True)
        else:
            lane_active.append(False)
        if angle in mpb:
            lane_active.append(True)
        else:
            lane_active.append(False)
        active_idx  = [i for i, a in enumerate(lane_active) if a]
        lanes_used  = len(active_idx)
        lane_centres = np.arange(lanes_used) * LANE_GAP
        lane_y = {orig: lane_centres[pos] for pos, orig in enumerate(active_idx)}

        for pos, orig in enumerate(active_idx):
            ylo = lane_centres[pos] - LANE_HEIGHT / 2
            yhi = lane_centres[pos] + LANE_HEIGHT / 2
            ax.axhspan(ylo, yhi, color=LANE_COLORS[orig], alpha=0.08, zorder=0)

        for i, res in enumerate(RESOLUTIONS):
            freqs = fdfd.get(angle, {}).get(res)
            if freqs is None or i not in lane_y:
                continue
            y = lane_y[i]
            ax.vlines(freqs, y - LANE_HEIGHT / 2, y + LANE_HEIGHT / 2,
                      colors=LANE_COLORS[i], linewidths=0.55,
                      alpha=0.85, zorder=2)

        # EA lane
        if angle in ea and EA_LANE_IDX in lane_y:
            y = lane_y[EA_LANE_IDX]
            ax.vlines(ea[angle], y - LANE_HEIGHT / 2, y + LANE_HEIGHT / 2,
                      colors=EA_GREEN, linewidths=0.55,
                      alpha=0.85, zorder=2)

        # MPB lane
        if angle in mpb and MPB_LANE_IDX in lane_y:
            y = lane_y[MPB_LANE_IDX]
            ax.vlines(mpb[angle], y - LANE_HEIGHT / 2, y + LANE_HEIGHT / 2,
                      colors=MPB_GOLD, linewidths=0.55,
                      alpha=0.85, zorder=2)

        for tf in TARGET_FREQS:
            ax.axvline(tf, color=STARK_ORANGE, ls='--', lw=0.8,
                       alpha=0.6, zorder=1)

        ax.set_yticks(lane_centres)
        ax.set_yticklabels([LANE_LABELS[i] for i in active_idx], fontsize=9)
        ax.set_ylim(lane_centres[0] - LANE_HEIGHT * 0.7,
                    lane_centres[-1] + LANE_HEIGHT * 0.7)
        ax.invert_yaxis()

        ax.text(1.01, 0.5, ANGLE_LABELS[angle], transform=ax.transAxes,
                fontsize=10, va='center', ha='left')

        ax.tick_params(labelsize=9, direction='in',
                       which='both', top=(row == 0), right=False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.6)

    axes[-1].set_xlabel('Frequency  $f$  [$c/a$]', fontsize=11)

    # Auto-range: pad around data
    all_freqs = []
    for a in fdfd:
        for p in fdfd[a]:
            all_freqs.append(fdfd[a][p])
    for a in ea:
        all_freqs.append(ea[a])
    for a in mpb:
        all_freqs.append(mpb[a])
    all_freqs = np.concatenate(all_freqs)
    fmin, fmax = all_freqs.min(), all_freqs.max()
    pad = 0.05 * (fmax - fmin) if fmax > fmin else 0.02
    axes[-1].set_xlim(fmin - pad, fmax + pad)

    # Legend — only lanes that appear in any panel
    all_active = set()
    for angle in ANGLES_ORDERED:
        for i, res in enumerate(RESOLUTIONS):
            if res in fdfd.get(angle, {}):
                all_active.add(i)
        if angle in ea:
            all_active.add(EA_LANE_IDX)
        if angle in mpb:
            all_active.add(MPB_LANE_IDX)
    legend_names = {
        0: 'FDFD 1 px', 1: 'FDFD 4 px', 2: 'FDFD 8 px', 3: 'FDFD 16 px',
        EA_LANE_IDX: 'EA',
        MPB_LANE_IDX: 'MPB 64 px',
    }
    handles = [
        Line2D([], [], color=LANE_COLORS[i], lw=1.5, label=legend_names[i])
        for i in sorted(all_active)
    ]
    handles.append(
        Line2D([], [], color=STARK_ORANGE, ls='--', lw=0.8,
               label=f'Target $f = {TARGET_FREQS[0]}$'))
    fig.legend(handles=handles, loc='upper right', fontsize=8,
               framealpha=0.9, edgecolor=LIGHT_BROWN,
               bbox_to_anchor=(0.98, 0.98), ncol=len(handles))

    fig.suptitle(
        r'TM eigenfrequencies at $\Gamma$  —  hex rods $\varepsilon{=}12$, $r/a{=}0.22$',
        fontsize=12, y=1.0)

    out = os.path.join(FIG_DIR, 'gamma_tm_hex_spectrum_compact.svg')
    fig.savefig(out, bbox_inches='tight')
    out_png = os.path.join(FIG_DIR, 'gamma_tm_hex_spectrum_compact.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'Saved {out}')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
