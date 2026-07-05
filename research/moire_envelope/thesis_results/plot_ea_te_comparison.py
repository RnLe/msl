#!/usr/bin/env python3
"""
EA TE self-comparison: 4 types × 3 angles = 12 runs.
Layout: 4 rows (types) × 3 columns (angles).
Each panel: ladder plot of eigenfrequencies.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

BASE = '/home/renlephy/msl/research/moire_envelope/thesis_results'

TYPES = [
    ('type1_1ret_0rem_1band', '1 ret, 0 rem → 1 band'),
    ('type2_1ret_5rem_1band', '1 ret, 5 rem → 1 band'),
    ('type3_4ret_0rem_1band', '4 ret, 0 rem → 1 band'),
    ('type4_4ret_0rem_4band', '4 ret, 0 rem → 4 bands'),
]
ANGLES = [
    ('8deg', '8.17°'),
    ('3deg', '3.01°'),
    ('1deg', '1.005°'),
]

# Colors per type
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def load_all():
    return load_all_for_tag('')


def dataset_stem(tkey: str, tag: str) -> str:
    if not tag:
        return tkey
    head, tail = tkey.split('_', 1)
    return f'{head}_{tag}_{tail}'


def load_all_for_tag(tag: str):
    data = {}
    for tkey, tlabel in TYPES:
        for akey, alabel in ANGLES:
            stem = dataset_stem(tkey, tag)
            f = np.load(f'{BASE}/ea_te_{stem}_{akey}.npz')
            data[(tkey, akey)] = {
                'freqs': f['frequencies'],
                'evals': f['eigenvalues'],
                'lambda_ref': float(f['lambda_ref']),
                'n_modes': int(f['n_modes']),
            }
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='', help='Optional dataset tag inserted after ea_te_, e.g. x')
    args = parser.parse_args()

    tag = args.tag.strip('_')
    data = load_all_for_tag(tag)
    point_label = 'Γ' if not tag else tag.upper()
    name_suffix = f'_{tag}' if tag else ''

    fig, axes = plt.subplots(4, 3, figsize=(22, 14), dpi=150)

    # Pre-compute shared x limits per column (widest range across all types, ignoring NaN)
    col_xlims = {}
    for col, (akey, alabel) in enumerate(ANGLES):
        all_f = np.concatenate([data[(t, akey)]['freqs'] for t, _ in TYPES])
        valid = all_f[~np.isnan(all_f)]
        if len(valid) == 0:
            col_xlims[col] = (-0.01, 0.01)
            continue
        span = valid.max() - valid.min()
        pad = max(span * 0.05, 0.001)
        col_xlims[col] = (valid.min() - pad, valid.max() + pad)

    for row, ((tkey, tlabel), color) in enumerate(zip(TYPES, COLORS)):
        for col, (akey, alabel) in enumerate(ANGLES):
            ax = axes[row, col]
            d = data[(tkey, akey)]
            freqs = d['freqs']
            n = len(freqs)

            # Ladder: vertical bars on a horizontal axis (skip NaN)
            valid = freqs[~np.isnan(freqs)]
            n_nan = n - len(valid)
            for f in valid:
                ax.vlines(f, -0.3, 0.3, color=color, linewidth=1.8, alpha=0.85)

            bw = valid[-1] - valid[0] if len(valid) > 1 else 0.0
            ax.set_xlim(*col_xlims[col])
            ax.set_ylim(-0.6, 0.6)
            ax.set_yticks([])

            # Info text inside the panel
            nan_str = f'\n({n_nan} NaN)' if n_nan > 0 else ''
            ax.text(0.02, 0.95,
                    f'{n} modes{nan_str}\nBW = {bw:.5f} c/a',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            if row == 0:
                ax.set_title(alabel, fontsize=14, fontweight='bold')
            if col == 0:
                ax.set_ylabel(tlabel, fontsize=10, fontweight='bold')
            if row == 3:
                ax.set_xlabel(r'Frequency $\nu$ [c/a]', fontsize=10)

    fig.suptitle('Envelope Approximation — TE Polarization\n'
                 f'Effect of retained / remote bands on Moiré spectrum at {point_label}',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = f'{BASE}/ea_te_self_comparison{name_suffix}.png'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved → {out}')

    # Also make an overlay version: all 4 types stacked per angle
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 5), dpi=150)

    for col, (akey, alabel) in enumerate(ANGLES):
        ax = axes2[col]
        for ypos, ((tkey, tlabel), color) in enumerate(zip(TYPES, COLORS)):
            d = data[(tkey, akey)]
            freqs = d['freqs']
            valid = freqs[~np.isnan(freqs)]
            for f in valid:
                ax.vlines(f, ypos - 0.3, ypos + 0.3, color=color, linewidth=1.8, alpha=0.85)

        # Global x range across all types for this angle (ignoring NaN)
        all_f = np.concatenate([data[(t, akey)]['freqs'] for t, _ in TYPES])
        all_valid = all_f[~np.isnan(all_f)]
        span = all_valid.max() - all_valid.min() if len(all_valid) > 1 else 0.01
        pad = max(span * 0.06, 0.001)
        ax.set_xlim(all_valid.min() - pad * 3.5, all_valid.max() + pad)

        ax.set_ylim(-0.6, 3.6)
        ax.set_yticks(range(4))
        ax.set_yticklabels([tl for _, tl in TYPES], fontsize=9)
        ax.set_title(alabel, fontsize=14, fontweight='bold')
        ax.set_xlabel(r'Frequency $\nu$ [c/a]', fontsize=10)
        ax.grid(axis='x', ls='--', alpha=0.3)

    fig2.suptitle(f'Envelope Approximation — TE at {point_label}\n'
                  'All 4 EA configurations compared per twist angle',
                  fontsize=15, fontweight='bold')
    fig2.tight_layout(rect=(0, 0, 1, 0.90))
    out2 = f'{BASE}/ea_te_self_comparison_overlay{name_suffix}.png'
    fig2.savefig(out2, bbox_inches='tight')
    print(f'Saved → {out2}')


if __name__ == '__main__':
    main()
