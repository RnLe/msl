#!/usr/bin/env python3
"""
Line plot comparing EA and highest-resolution FDFD eigenfrequencies
at the X point (TM).  1×4 subplots, one per twist angle.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(STUDY_DIR, 'data_x_tm')
EA_DIR    = os.path.join(STUDY_DIR, 'tm_commensurate_phase2')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────
EA_GREEN   = '#6BAF6B'
STEEL_BLUE = '#4D7B9E'
FDFD_PURPLE = '#8E7CC3'
FDFD_ROSE   = '#C27BA0'
FDFD_BRICK  = '#D4534E'

ANGLES = [
    {'label': '1deg', 'm': 114, 'n': 1, 'res': 8,
     'title': r'$\theta \approx 1.0°$',  'fdfd_color': STEEL_BLUE},
    {'label': '2deg', 'm': 57,  'n': 1, 'res': 16,
     'title': r'$\theta \approx 2.0°$',  'fdfd_color': FDFD_PURPLE},
    {'label': '4deg', 'm': 29,  'n': 1, 'res': 32,
     'title': r'$\theta \approx 3.9°$',  'fdfd_color': FDFD_ROSE},
    {'label': '8deg', 'm': 14,  'n': 1, 'res': 64,
     'title': r'$\theta \approx 8.1°$',  'fdfd_color': FDFD_BRICK},
]

MN_TO_LABEL = {(a['m'], a['n']): a['label'] for a in ANGLES}


def load_ea():
    ea = {}
    for a in ANGLES:
        fname = os.path.join(EA_DIR,
                             f'commensurate_m{a["m"]}_n{a["n"]}.npz')
        if os.path.isfile(fname):
            ea[a['label']] = np.sort(np.load(fname)['frequencies'])
    return ea


def load_fdfd():
    fdfd = {}
    for a in ANGLES:
        fname = os.path.join(DATA_DIR,
                             f'fdfd_tm_x_{a["label"]}_res{a["res"]}_fEActr.npz')
        if os.path.isfile(fname):
            fdfd[a['label']] = np.sort(np.load(fname)['freqs'])
    return fdfd


def main():
    ea   = load_ea()
    fdfd = load_fdfd()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)

    for ax, a in zip(axes, ANGLES):
        label = a['label']

        ea_f   = ea.get(label)
        fdfd_f = fdfd.get(label)

        if ea_f is not None:
            ax.plot(np.arange(1, len(ea_f) + 1), ea_f,
                    'o-', color=EA_GREEN, ms=3, lw=1.2, label='EA', zorder=3)
        if fdfd_f is not None:
            ax.plot(np.arange(1, len(fdfd_f) + 1), fdfd_f,
                    's-', color=a['fdfd_color'], ms=2.5, lw=1.0,
                    label=f'FDFD {a["res"]} px', zorder=2)

        ax.set_title(a['title'], fontsize=11)
        ax.set_xlabel('Mode index', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.tick_params(labelsize=9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel('Frequency  $f$  [$c/a$]', fontsize=10)

    fig.suptitle(
        'EA vs highest-resolution FDFD  —  X-point TM eigenfrequencies',
        fontsize=12, y=1.02)
    fig.tight_layout()

    for ext in ('svg', 'png'):
        out = os.path.join(FIG_DIR, f'x_tm_ea_vs_fdfd_line.{ext}')
        fig.savefig(out, bbox_inches='tight', dpi=200)
        print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
