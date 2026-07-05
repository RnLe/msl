#!/usr/bin/env python3
"""
Plot frequency comparisons for the exact TE X-point audit datasets
against 8 px FDFD runs centered at f=0.05 using 30 modes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path('/home/renlephy/msl/research/moire_envelope/thesis_results')
ANGLES = [
    ('8deg', '8.17°', '8p17deg'),
    ('3deg', '3.01°', '3p01deg'),
]
FDFD_TAG = 'sig005_30'
OUT_NAME = 'exact_te_x_audit_f005_vs_fdfd_sig005_30_freq.png'


def load_angle(angle_key: str, angle_file: str) -> dict:
    audit = np.load(BASE / f'exact_te_x_audit_f005_{angle_file}.npz')
    fdfd = np.load(BASE / f'fdfd_te_x_{angle_key}_res8_{FDFD_TAG}.npz')

    lambda_ref = float(audit['lambda_ref'])
    sigma_shifted = float(audit['sigma'])
    target_frequency = 0.05

    compact_shifted = np.asarray(audit['compact_eigenvalues_shifted'], dtype=float)
    compact_freqs = np.asarray(audit['compact_frequencies'], dtype=float)
    exact_shifted = np.asarray(audit['exact_eigenvalues_shifted'], dtype=float)
    exact_freqs = np.asarray(audit['exact_frequencies'], dtype=float)

    fdfd_evals = np.asarray(fdfd['evals'], dtype=float)
    fdfd_shifted = fdfd_evals - lambda_ref
    fdfd_freqs = np.asarray(fdfd['freqs'], dtype=float)

    return {
        'lambda_ref': lambda_ref,
        'sigma_shifted': sigma_shifted,
        'target_frequency': target_frequency,
        'compact_shifted': compact_shifted,
        'compact_freqs': compact_freqs,
        'exact_shifted': exact_shifted,
        'exact_freqs': exact_freqs,
        'fdfd_shifted': fdfd_shifted,
        'fdfd_freqs': fdfd_freqs,
    }


def plot_frequency_series(ax, xvals, freqs, color, label, marker):
    real_mask = np.isfinite(freqs)
    ax.plot(xvals[real_mask], freqs[real_mask], color=color, lw=1.4, alpha=0.9)
    ax.scatter(xvals[real_mask], freqs[real_mask], s=30, marker=marker,
               color=color, linewidths=0.8, label=f'{label} (real freq)')


def ladder(ax, datasets, target_frequency):
    finite_sets = [vals[np.isfinite(vals)] for _, _, _, vals in datasets if np.any(np.isfinite(vals))]
    all_vals = np.concatenate(finite_sets)
    span = float(all_vals.max() - all_vals.min()) if len(all_vals) > 1 else 0.1
    pad = max(0.04 * span, 0.01)

    for ypos, (label, color, linestyle, vals) in enumerate(reversed(datasets)):
        valid = vals[np.isfinite(vals)]
        ax.hlines(ypos, all_vals.min() - pad, all_vals.max() + pad, color='k', lw=0.5, alpha=0.12)
        for value in valid:
            ax.vlines(value, ypos - 0.28, ypos + 0.28, color=color, linewidth=2.0, alpha=0.9, linestyles=linestyle)
        ax.text(all_vals.min() - 1.2 * pad, ypos, label, va='center', ha='right', color=color, fontsize=10)

    ax.axvline(target_frequency, color='#7f7f7f', lw=1.0, ls='--', alpha=0.8)
    ax.text(target_frequency, len(datasets) - 0.2, 'f = 0.05', color='#555555', fontsize=9, ha='center', va='bottom')
    ax.set_xlim(all_vals.min() - 4.2 * pad, all_vals.max() + pad)
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel('Frequency [c/a]')


def main():
    fig, axes = plt.subplots(len(ANGLES), 2, figsize=(18, 10), dpi=150,
                             gridspec_kw={'width_ratios': [1.3, 1.8]})
    if len(ANGLES) == 1:
        axes = np.asarray([axes])

    for row, (angle_key, angle_label, angle_file) in enumerate(ANGLES):
        data = load_angle(angle_key, angle_file)
        mode_idx = np.arange(1, len(data['fdfd_freqs']) + 1)

        ax1 = axes[row, 0]
        ax1.axhline(data['target_frequency'], color='#7f7f7f', lw=1.0, ls='--', alpha=0.8, label='target f = 0.05')
        plot_frequency_series(ax1, mode_idx, data['fdfd_freqs'], '#000000', 'FDFD 8 px', 's')
        plot_frequency_series(ax1, mode_idx, data['compact_freqs'], '#1f77b4', 'Compact', 'o')
        plot_frequency_series(ax1, mode_idx, data['exact_freqs'], '#d62728', 'Hermitianized exact', '^')
        ax1.set_title(f'{angle_label} — real frequencies')
        ax1.set_xlabel('Mode index n')
        ax1.set_ylabel('Frequency [c/a]')
        ax1.grid(True, ls='--', alpha=0.35)
        if row == 0:
            ax1.legend(fontsize=8, ncol=2)

        ax2 = axes[row, 1]
        ladder(
            ax2,
            [
                ('FDFD 8 px', '#000000', '-', data['fdfd_freqs']),
                ('Compact', '#1f77b4', '-', data['compact_freqs']),
                ('Hermitianized exact', '#d62728', '-', data['exact_freqs']),
            ],
            data['target_frequency'],
        )
        ax2.set_title(
            f"{angle_label} — frequency ladder; real-frequency counts: "
            f"FDFD={np.isfinite(data['fdfd_freqs']).sum()}/{len(data['fdfd_freqs'])}, "
            f"compact={np.isfinite(data['compact_freqs']).sum()}/{len(data['compact_freqs'])}, "
            f"exact={np.isfinite(data['exact_freqs']).sum()}/{len(data['exact_freqs'])}"
        )

    fig.suptitle(
        'TE at X — exact audit f=0.05 frequencies vs FDFD\n'
        'FDFD: 8 px/cell, 30 modes, sigma_omega = 0.05; audit branches: compact and hermitianized exact',
        fontsize=15,
        fontweight='bold',
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = BASE / OUT_NAME
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()