#!/usr/bin/env python3
"""
Thesis-grade convergence plots for the FDFD TE X-point study.

Reads all .npz files from data/ and produces SVG figures in figures/.

Color palette from StyleGuide.md.
"""
import os
import glob
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── paths ───────────────────────────────────────────────────────────
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, 'data')
FIG_DIR = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── style-guide palette ────────────────────────────────────────────
SKY_BLUE       = '#4E9AE1'
STEEL_BLUE     = '#4D7B9E'
DARK_STEEL     = '#4F5F6B'
LIGHT_STEEL    = '#A5C6DF'
STARK_ORANGE   = '#EBA538'
DUSTY_ORANGE   = '#E3B064'
GENTLE_BROWN   = '#AB8954'
DUSTY_BROWN    = '#857255'
LIGHT_BROWN    = '#E3D5BF'

# Colors for the four resolutions (light → dark as px increases)
RES_COLORS = {
    1:  LIGHT_STEEL,
    4:  SKY_BLUE,
    8:  STEEL_BLUE,
    16: DARK_STEEL,
}

# Colors for the six target frequencies (browns → blues → oranges)
FREQ_COLORS = {
    0.01: LIGHT_BROWN,
    0.05: LIGHT_STEEL,
    0.1:  SKY_BLUE,
    0.2:  STEEL_BLUE,
    0.3:  DUSTY_ORANGE,
    0.4:  STARK_ORANGE,
}

# Markers for resolutions
RES_MARKERS = {1: 's', 4: 'D', 8: 'o', 16: '^'}

ANGLES_ORDERED = ['1deg', '2deg', '4deg', '8deg']
RESOLUTIONS = [1, 4, 8, 16]
TARGET_FREQS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

ANGLE_LABELS = {
    '1deg': r'$\theta \approx 1.0°$',
    '2deg': r'$\theta \approx 2.0°$',
    '4deg': r'$\theta \approx 3.9°$',
    '8deg': r'$\theta \approx 8.1°$',
}


def load_all() -> dict:
    """Load all .npz files into a nested dict: data[angle][px][freq] = {...}."""
    pattern = os.path.join(DATA_DIR, 'fdfd_te_x_*.npz')
    files = sorted(glob.glob(pattern))
    data = {}
    for fpath in files:
        base = os.path.basename(fpath)
        m = re.match(r'fdfd_te_x_(\w+deg)_res(\d+)_f(\d+)\.npz', base)
        if not m:
            continue
        angle, px, ftag = m.group(1), int(m.group(2)), int(m.group(3))
        freq_key = ftag / 100.0
        npz = np.load(fpath)
        entry = {k: npz[k] for k in npz.files}
        data.setdefault(angle, {}).setdefault(px, {})[freq_key] = entry
    return data


def _apply_thesis_style(fig, axes):
    """Shared formatting for thesis figures."""
    for ax in np.atleast_1d(axes).flat:
        ax.tick_params(direction='in', which='both', top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)


# ─────────────────────────────────────────────────────────────────────
#  Figure 1: Spectrum ladder — eigenfrequencies vs resolution
# ─────────────────────────────────────────────────────────────────────
def fig_spectrum_ladder(data: dict):
    """
    4×6 grid (rows=angles, cols=target_freq).
    Each panel: resolution on x-axis, eigenfrequencies as horizontal ticks.
    """
    fig, axes = plt.subplots(
        len(ANGLES_ORDERED), len(TARGET_FREQS),
        figsize=(16, 10), sharex=True,
        gridspec_kw={'hspace': 0.35, 'wspace': 0.30})

    for row, angle in enumerate(ANGLES_ORDERED):
        for col, tf in enumerate(TARGET_FREQS):
            ax = axes[row, col]
            for res in RESOLUTIONS:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    continue
                freqs = np.asarray(entry['freqs'], dtype=float)
                x = np.full_like(freqs, res)
                ax.scatter(x, freqs, marker='_', s=80, linewidths=1.2,
                           color=RES_COLORS[res], zorder=3)

            ax.set_xlim(-1, 20)
            ax.set_xticks(RESOLUTIONS)
            if row == 0:
                ax.set_title(f'$f_{{\\mathrm{{target}}}} = {tf}$',
                             fontsize=9, pad=4)
            if col == 0:
                ax.set_ylabel(ANGLE_LABELS[angle], fontsize=8)
            if row == len(ANGLES_ORDERED) - 1:
                ax.set_xlabel('px / cell', fontsize=8)

            ax.tick_params(labelsize=7)

    fig.suptitle(
        'FDFD TE eigenfrequencies at the X point — resolution sweep',
        fontsize=12, y=0.98)

    _apply_thesis_style(fig, axes)
    out = os.path.join(FIG_DIR, 'convergence_spectrum_ladder.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────
#  Figure 2: Convergence rate — relative error vs resolution
# ─────────────────────────────────────────────────────────────────────
def _match_eigenvalues(ref_freqs, test_freqs):
    """
    Match test eigenvalues to the nearest reference eigenvalue.
    Returns arrays of (ref, test) matched pairs.
    """
    ref = np.sort(ref_freqs)
    test = np.sort(test_freqs)
    matched_ref, matched_test = [], []
    used = set()
    for tv in test:
        diffs = np.abs(ref - tv)
        idx = np.argmin(diffs)
        if idx not in used:
            matched_ref.append(ref[idx])
            matched_test.append(tv)
            used.add(idx)
    return np.array(matched_ref), np.array(matched_test)


def fig_convergence_rate(data: dict):
    """
    One panel per angle.  Lines colored by target frequency.
    x-axis: px/cell (log).  y-axis: max relative error vs best reference (log).
    Uses highest available resolution as reference for each angle.
    """
    fig, axes = plt.subplots(1, len(ANGLES_ORDERED),
                             figsize=(14, 3.8), sharey=True,
                             gridspec_kw={'wspace': 0.12})

    for i, angle in enumerate(ANGLES_ORDERED):
        ax = axes[i]
        # Find highest available resolution for this angle
        avail_res = sorted(data.get(angle, {}).keys())
        ref_res = avail_res[-1] if avail_res else None
        if ref_res is None:
            continue

        for tf in TARGET_FREQS:
            ref_entry = data.get(angle, {}).get(ref_res, {}).get(tf)
            if ref_entry is None:
                continue
            ref_freqs = np.asarray(ref_entry['freqs'], dtype=float)

            px_vals, err_max = [], []
            for res in [r for r in RESOLUTIONS if r < ref_res]:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    continue
                test_freqs = np.asarray(entry['freqs'], dtype=float)
                rf, tf_matched = _match_eigenvalues(ref_freqs, test_freqs)
                if len(rf) == 0:
                    continue
                rel_err = np.abs(tf_matched - rf) / np.maximum(np.abs(rf), 1e-15)
                px_vals.append(res)
                err_max.append(np.max(rel_err))

            if px_vals:
                ax.plot(px_vals, err_max, 'o-', color=FREQ_COLORS[tf],
                        markersize=5, linewidth=1.2, label=f'$f={tf}$')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        lower_res = [r for r in RESOLUTIONS if r < ref_res]
        if lower_res:
            ax.set_xticks(lower_res)
            ax.set_xticklabels([str(r) for r in lower_res])
        ax.set_xlabel('px / cell', fontsize=9)
        ax.set_title(f'{ANGLE_LABELS[angle]}\n(ref: {ref_res} px)', fontsize=9)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.set_ylabel('Max relative error vs reference', fontsize=9)

    axes[-1].legend(fontsize=7, loc='upper right',
                    framealpha=0.9, edgecolor=LIGHT_BROWN)

    fig.suptitle(
        'Eigenfrequency convergence rate — FDFD TE at X',
        fontsize=11, y=1.02)

    _apply_thesis_style(fig, axes)
    out = os.path.join(FIG_DIR, 'convergence_rate.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────
#  Figure 3: Timing & DOF — computational cost
# ─────────────────────────────────────────────────────────────────────
def fig_cost(data: dict):
    """
    Two-row figure: top = solve time, bottom = DOF (grid²).
    One column per angle, lines colored by target frequency.
    Note: RSS was measured per-process (cumulative), so is unreliable.
    DOF = grid² is the deterministic problem size proxy.
    """
    fig, axes = plt.subplots(2, len(ANGLES_ORDERED),
                             figsize=(14, 6), sharex=True,
                             gridspec_kw={'hspace': 0.3, 'wspace': 0.2})

    for i, angle in enumerate(ANGLES_ORDERED):
        ax_time = axes[0, i]
        ax_dof = axes[1, i]
        for tf in TARGET_FREQS:
            px_vals, times, dofs = [], [], []
            for res in RESOLUTIONS:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    continue
                px_vals.append(res)
                times.append(float(entry['t_solve']))
                grid = int(entry['grid'])
                dofs.append(grid * grid)

            if px_vals:
                ax_time.plot(px_vals, times, 'o-', color=FREQ_COLORS[tf],
                             markersize=4, linewidth=1.1, label=f'$f={tf}$')
                ax_dof.plot(px_vals, dofs, 'o-', color=FREQ_COLORS[tf],
                            markersize=4, linewidth=1.1)

        for ax in (ax_time, ax_dof):
            ax.set_xscale('log', base=2)
            ax.set_xticks(RESOLUTIONS)
            ax.set_xticklabels(['1', '4', '8', '16'])
            ax.tick_params(labelsize=7)

        ax_time.set_yscale('log')
        ax_dof.set_yscale('log')
        ax_time.set_title(ANGLE_LABELS[angle], fontsize=9)
        ax_dof.set_xlabel('px / cell', fontsize=8)
        if i == 0:
            ax_time.set_ylabel('Solve time (s)', fontsize=8)
            ax_dof.set_ylabel('DOF (grid²)', fontsize=8)

    axes[0, -1].legend(fontsize=6, loc='upper left',
                       framealpha=0.9, edgecolor=LIGHT_BROWN)

    fig.suptitle(
        'Computational cost — FDFD TE at X',
        fontsize=11, y=0.98)

    _apply_thesis_style(fig, axes)
    out = os.path.join(FIG_DIR, 'convergence_cost.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────
#  Figure 5: Solve time vs DOF (collapse across angles)
# ─────────────────────────────────────────────────────────────────────
def fig_time_vs_dof(data: dict):
    """
    Single panel: DOF on x-axis, solve time on y.
    One marker per angle, all frequencies collapsed (mean ± band).
    Shows scaling law.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))

    angle_colors = {
        '1deg': DARK_STEEL, '2deg': STEEL_BLUE,
        '4deg': SKY_BLUE, '8deg': LIGHT_STEEL,
    }

    all_dofs, all_times = [], []
    for angle in ANGLES_ORDERED:
        dof_time = {}  # dof -> list of times
        for res in RESOLUTIONS:
            for tf in TARGET_FREQS:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    continue
                grid = int(entry['grid'])
                dof = grid * grid
                dof_time.setdefault(dof, []).append(float(entry['t_solve']))

        dofs = sorted(dof_time.keys())
        means = [np.mean(dof_time[d]) for d in dofs]
        mins_ = [np.min(dof_time[d]) for d in dofs]
        maxs_ = [np.max(dof_time[d]) for d in dofs]

        ax.plot(dofs, means, 'o-', color=angle_colors[angle],
                markersize=5, linewidth=1.2, label=ANGLE_LABELS[angle])
        ax.fill_between(dofs, mins_, maxs_, alpha=0.15,
                        color=angle_colors[angle])
        all_dofs.extend(dofs)
        all_times.extend(means)

    # Power-law reference line
    if len(all_dofs) > 1:
        d_arr = np.array(all_dofs, float)
        t_arr = np.array(all_times, float)
        mask = (d_arr > 0) & (t_arr > 0)
        if mask.sum() > 1:
            coeffs = np.polyfit(np.log10(d_arr[mask]), np.log10(t_arr[mask]), 1)
            d_ref = np.logspace(np.log10(d_arr[mask].min()),
                                np.log10(d_arr[mask].max()), 50)
            t_ref = 10**(coeffs[0] * np.log10(d_ref) + coeffs[1])
            ax.plot(d_ref, t_ref, '--', color=GENTLE_BROWN, linewidth=0.8,
                    label=f'$\\propto N^{{{coeffs[0]:.2f}}}$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('DOF ($N = \\mathrm{grid}^2$)', fontsize=9)
    ax.set_ylabel('Solve time (s)', fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9, edgecolor=LIGHT_BROWN)
    ax.set_title('Solve time scaling — FDFD TE at X', fontsize=10)

    _apply_thesis_style(fig, [ax])
    out = os.path.join(FIG_DIR, 'convergence_time_vs_dof.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────
#  Figure 4: Mode density — count of modes within ±10% of target
# ─────────────────────────────────────────────────────────────────────
def fig_mode_density(data: dict):
    """
    One panel per target frequency.
    Bar chart: x = angle, grouped bars colored by resolution.
    y = number of modes within ±10% of target_freq.
    """
    width = 0.18
    fig, axes = plt.subplots(1, len(TARGET_FREQS),
                             figsize=(14, 3.5), sharey=True,
                             gridspec_kw={'wspace': 0.12})

    for col, tf in enumerate(TARGET_FREQS):
        ax = axes[col]
        x_pos = np.arange(len(ANGLES_ORDERED))
        for j, res in enumerate(RESOLUTIONS):
            counts = []
            for angle in ANGLES_ORDERED:
                entry = data.get(angle, {}).get(res, {}).get(tf)
                if entry is None:
                    counts.append(0)
                    continue
                freqs = np.asarray(entry['freqs'], dtype=float)
                within = np.sum((freqs >= tf * 0.9) & (freqs <= tf * 1.1))
                counts.append(within)
            offset = (j - 1.5) * width
            ax.bar(x_pos + offset, counts, width * 0.9,
                   color=RES_COLORS[res], label=f'{res} px' if col == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([ANGLE_LABELS[a] for a in ANGLES_ORDERED],
                           fontsize=6, rotation=30, ha='right')
        ax.set_title(f'$f_{{\\mathrm{{target}}}} = {tf}$', fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel('Modes within ±10 %', fontsize=8)

    axes[0].legend(fontsize=7, loc='upper left',
                   framealpha=0.9, edgecolor=LIGHT_BROWN)

    fig.suptitle(
        'Mode density near target frequency — FDFD TE at X',
        fontsize=11, y=1.02)

    _apply_thesis_style(fig, axes)
    out = os.path.join(FIG_DIR, 'convergence_mode_density.svg')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main():
    data = load_all()
    n_files = sum(1 for a in data for p in data[a] for f in data[a][p])
    print(f'Loaded {n_files} result files')

    if n_files == 0:
        print('No data found — run run_convergence.py first.')
        return

    fig_spectrum_ladder(data)
    fig_convergence_rate(data)
    fig_cost(data)
    fig_time_vs_dof(data)
    fig_mode_density(data)
    print('\nAll figures saved to figures/')


if __name__ == '__main__':
    main()
