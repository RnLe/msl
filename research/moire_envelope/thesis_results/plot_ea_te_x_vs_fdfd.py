#!/usr/bin/env python3
"""
Compare X-point EA TE datasets against direct FDFD TE at X.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

BASE = '/home/renlephy/msl/research/moire_envelope/thesis_results'

TYPES = [
    ('type1_x_1ret_0rem_1band', '1 ret, 0 rem → 1 band', '#1f77b4'),
    ('type2_x_1ret_5rem_1band', '1 ret, 5 rem → 1 band', '#ff7f0e'),
    ('type3_x_4ret_0rem_1band', '4 ret, 0 rem → 1 band', '#2ca02c'),
    ('type4_x_4ret_0rem_4band', '4 ret, 0 rem → 4 bands', '#d62728'),
]
ANGLES = [
    ('8deg', '8.17°'),
    ('3deg', '3.01°'),
    ('1deg', '1.005°'),
]
ANGLE_FILE_TAG = {
    '8deg': '8p17deg',
    '3deg': '3p01deg',
    '1deg': '1deg',
}


def format_pattern(pattern: str, angle_key: str) -> str:
    return pattern.format(angle=angle_key, angle_file=ANGLE_FILE_TAG[angle_key])


def load_data(fdfd_res: int, fdfd_tag: str, ea_pattern: str, ea_label: str,
              ea_pattern_2: str, ea_label_2: str, selected_angles):
    data = {}
    fdfd_suffix = f'_{fdfd_tag.strip("_")}' if fdfd_tag else ''
    for akey, _ in selected_angles:
        fdfd = np.load(f'{BASE}/fdfd_te_x_{akey}_res{fdfd_res}{fdfd_suffix}.npz')['freqs']
        entry = {'fdfd': fdfd, 'ea': {}, 'ea_label': ea_label}
        if ea_pattern == 'legacy4':
            for tkey, _, _ in TYPES:
                freqs = np.load(f'{BASE}/ea_te_{tkey}_{akey}.npz')['frequencies']
                entry['ea'][tkey] = freqs
        else:
            freqs = np.load(f'{BASE}/{format_pattern(ea_pattern, akey)}')['frequencies']
            entry['ea'][ea_label] = freqs
        if ea_pattern_2:
            freqs2 = np.load(f'{BASE}/{format_pattern(ea_pattern_2, akey)}')['frequencies']
            entry['ea'][ea_label_2] = freqs2
        data[akey] = entry
    return data


def valid_bandwidth(freqs):
    valid = freqs[~np.isnan(freqs)]
    if len(valid) < 2:
        return 0.0
    return valid[-1] - valid[0]


def parse_extra_ea_specs(values):
    specs = []
    for raw in values:
        parts = raw.split('|')
        if len(parts) != 3:
            raise ValueError(
                f'Invalid --ea-spec value {raw!r}. Expected LABEL|PATTERN|COLOR.'
            )
        label, pattern, color = [part.strip() for part in parts]
        specs.append((label, pattern, color))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fdfd-res', type=int, default=16)
    parser.add_argument('--fdfd-tag', default='')
    parser.add_argument('--ea-pattern', default='legacy4', help='Filename pattern relative to BASE, use {angle}')
    parser.add_argument('--ea-label', default='EA')
    parser.add_argument('--ea-pattern-2', default='', help='Optional second filename pattern relative to BASE, use {angle}')
    parser.add_argument('--ea-label-2', default='EA 2')
    parser.add_argument('--ea-spec', action='append', default=[],
                        help='Additional EA overlay in the form LABEL|PATTERN|COLOR; PATTERN uses {angle}')
    parser.add_argument('--angles', default='',
                        help='Comma-separated subset such as 8deg,3deg')
    parser.add_argument('--out-name', default='ea_te_x_vs_fdfd.png')
    parser.add_argument('--title', default='TE X-point Comparison — FDFD vs Envelope Approximation')
    args = parser.parse_args()

    extra_specs = parse_extra_ea_specs(args.ea_spec)
    selected_angles = [
        pair for pair in ANGLES
        if not args.angles or pair[0] in {item.strip() for item in args.angles.split(',') if item.strip()}
    ]

    data = load_data(args.fdfd_res, args.fdfd_tag, args.ea_pattern, args.ea_label,
                     args.ea_pattern_2, args.ea_label_2, selected_angles)

    for akey, _ in selected_angles:
        for label, pattern, _ in extra_specs:
            data[akey]['ea'][label] = np.load(f'{BASE}/{format_pattern(pattern, akey)}')['frequencies']

    fig, axes = plt.subplots(len(selected_angles), 2, figsize=(20, 5.3 * len(selected_angles)), dpi=150,
                             gridspec_kw={'width_ratios': [1.1, 1.9]})
    if len(selected_angles) == 1:
        axes = np.asarray([axes])

    for row, (akey, alabel) in enumerate(selected_angles):
        d = data[akey]
        fdfd = d['fdfd']

        # Left: line plot using first len(fdfd) modes from each EA type.
        ax1 = axes[row, 0]
        mode_idx = np.arange(1, len(fdfd) + 1)
        ax1.plot(mode_idx, fdfd, color='black', lw=1.8, marker='s', ms=4,
                 label=f'FDFD X-point ({args.fdfd_res} px/cell)')
        if args.ea_pattern == 'legacy4':
            iterator = TYPES
        else:
            iterator = [(args.ea_label, args.ea_label, '#1f77b4')]
        if args.ea_pattern_2:
            iterator = iterator + [(args.ea_label_2, args.ea_label_2, '#d62728')]
        iterator = iterator + extra_specs
        for tkey, tlabel, color in iterator:
            freqs = d['ea'][tkey]
            n_cmp = min(len(fdfd), len(freqs))
            valid = freqs[:n_cmp]
            mask = ~np.isnan(valid)
            ax1.plot(mode_idx[:n_cmp][mask], valid[mask], color=color, lw=1.4,
                     marker='o', ms=3.5, alpha=0.85, label=tlabel)

        ax1.set_title(f'{alabel} — FDFD vs EA at X')
        ax1.set_xlabel('Mode index n')
        ax1.set_ylabel(r'Frequency $\nu$ [c/a]')
        ax1.grid(True, ls='--', alpha=0.35)
        if row == 0:
            ax1.legend(fontsize=8)

        # Right: stacked ladder plot of full spectra.
        ax2 = axes[row, 1]
        stacked = [('fdfd', f'FDFD {args.fdfd_res} px/cell', 'black', fdfd)]
        for tkey, tlabel, color in iterator:
            stacked.append((tkey, tlabel, color, d['ea'][tkey]))

        all_freqs = np.concatenate([arr[~np.isnan(arr)] for _, _, _, arr in stacked if np.any(~np.isnan(arr))])
        span = all_freqs.max() - all_freqs.min() if len(all_freqs) > 1 else 0.01
        pad = max(span * 0.06, 0.001)

        for ypos, (_, label, color, freqs) in enumerate(reversed(stacked)):
            valid = freqs[~np.isnan(freqs)]
            ax2.hlines(ypos, all_freqs.min() - pad, all_freqs.max() + pad,
                       color='k', lw=0.5, alpha=0.12)
            for f in valid:
                ax2.vlines(f, ypos - 0.28, ypos + 0.28, color=color, linewidth=2.0, alpha=0.9)
            ax2.text(all_freqs.min() - pad * 1.2, ypos, label,
                     va='center', ha='right', fontsize=10, color=color,
                     fontweight='bold' if color == 'black' else None)

        ax2.set_xlim(all_freqs.min() - pad * 3.8, all_freqs.max() + pad)
        ax2.set_ylim(-0.6, len(stacked) - 0.4)
        ax2.set_yticks([])
        ax2.set_title(f'{alabel} — Full ladder comparison')
        ax2.set_xlabel(r'Frequency $\nu$ [c/a]')

    fig.suptitle(args.title + '\n'
                 f'Square moire crystal, q = X, direct solver at {args.fdfd_res} px/cell',
                 fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f'{BASE}/{args.out_name}'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()