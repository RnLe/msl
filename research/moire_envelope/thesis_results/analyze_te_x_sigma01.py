#!/usr/bin/env python3
"""
Analyze X-point TE FDFD data at sigma_omega=0.1:
1. Compare px=4 and px=8 spectra, with an extra 1deg px=12 overlay if available.
2. Unfold px=8 eigenvectors to the monolayer BZ and identify dominant carriers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path('/home/renlephy/msl/research/moire_envelope/thesis_results')
ANGLES = [('8deg', '8.17°'), ('3deg', '3.01°'), ('1deg', '1.005°')]
Q_SUPER = np.array([np.pi, 0.0])
TOP_COMPONENTS = 64
AUDIT_FILES = {
    '8deg': 'square_x_te_term_audit_8p17deg.npz',
    '3deg': 'square_x_te_term_audit_3p01deg.npz',
}


def load_npz(angle: str, px: int) -> dict:
    path = BASE / f'fdfd_te_x_{angle}_res{px}_sig010.npz'
    return dict(np.load(path))


def load_npz_optional(angle: str, px: int):
    path = BASE / f'fdfd_te_x_{angle}_res{px}_sig010.npz'
    if not path.exists():
        return None
    return dict(np.load(path))


def load_audit_optional(angle: str):
    filename = AUDIT_FILES.get(angle)
    if filename is None:
        return None
    path = BASE / filename
    if not path.exists():
        return None
    return dict(np.load(path))


def reciprocal_from_direct_basis(basis: np.ndarray) -> np.ndarray:
    return 2 * np.pi * np.linalg.inv(basis).T


def fold_to_mono_frac(k_cart: np.ndarray, b_mono: np.ndarray) -> np.ndarray:
    frac = np.linalg.solve(b_mono, k_cart)
    return frac - np.round(frac)


def classify_frac(frac: np.ndarray) -> tuple[str, float]:
    candidates = []
    gamma = np.array([0.0, 0.0])
    x_points = [np.array([0.5, 0.0]), np.array([-0.5, 0.0]), np.array([0.0, 0.5]), np.array([0.0, -0.5])]
    m_points = [np.array([sx, sy]) for sx in (0.5, -0.5) for sy in (0.5, -0.5)]
    candidates.append(('Gamma', np.linalg.norm(frac - gamma)))
    candidates.extend(('X', np.linalg.norm(frac - pt)) for pt in x_points)
    candidates.extend(('M', np.linalg.norm(frac - pt)) for pt in m_points)
    label, dist = min(candidates, key=lambda item: item[1])
    if dist > 0.18:
        label = 'other'
    return label, dist


def dominant_carrier_stats(angle: str, d8: dict) -> list[dict]:
    m = int(d8['m'])
    n = int(d8['n'])
    nx = int(d8['grid'])
    ny = int(d8['grid'])
    evecs = d8['evecs']
    freqs = d8['freqs']

    l1 = np.array([m, n], dtype=float)
    l2 = np.array([-n, m], dtype=float)
    b_super = np.column_stack([l1, l2])
    b_mono = np.eye(2)
    g_super = reciprocal_from_direct_basis(b_super)
    g1 = g_super[:, 0]
    g2 = g_super[:, 1]
    b_mono_rec = reciprocal_from_direct_basis(b_mono)

    mode_rows = []
    aggregate = {'Gamma': 0.0, 'X': 0.0, 'M': 0.0, 'other': 0.0}

    for mode in range(evecs.shape[1]):
        field = evecs[:, mode].reshape(nx, ny)
        coeffs = np.fft.fft2(field, norm='ortho')
        power = np.abs(coeffs) ** 2
        flat = power.ravel()
        top_idx = np.argpartition(flat, -TOP_COMPONENTS)[-TOP_COMPONENTS:]

        carrier_weight = {'Gamma': 0.0, 'X': 0.0, 'M': 0.0, 'other': 0.0}
        best = {'weight': -1.0, 'label': 'other', 'frac': np.zeros(2), 'dist': 0.0}

        for idx in top_idx:
            i, j = np.unravel_index(idx, power.shape)
            ni = i if i <= nx // 2 else i - nx
            nj = j if j <= ny // 2 else j - ny
            k_cart = Q_SUPER + ni * g1 + nj * g2
            frac = fold_to_mono_frac(k_cart, b_mono_rec)
            label, dist = classify_frac(frac)
            weight = float(power[i, j])
            carrier_weight[label] += weight
            if weight > best['weight']:
                best = {'weight': weight, 'label': label, 'frac': frac, 'dist': dist}

        total_top = sum(carrier_weight.values())
        if total_top > 0:
            for key in carrier_weight:
                carrier_weight[key] /= total_top
                aggregate[key] += carrier_weight[key]

        mode_rows.append({
            'mode': mode + 1,
            'freq': float(freqs[mode]),
            'dominant_label': best['label'],
            'dominant_frac': best['frac'],
            'dominant_dist': best['dist'],
            'carrier_weight': carrier_weight,
        })

    for key in aggregate:
        aggregate[key] /= evecs.shape[1]

    return mode_rows, aggregate


def main():
    report_lines = [
        '# TE X-Point Sigma 0.1 Analysis',
        '',
        '## Setup',
        '',
        '- Polarization: TE',
        '- Direct solve: FDFD at X with total supercell Bloch vector `q = (π, 0)`',
        '- Sigma: `sigma_omega = 0.1`',
        '- Mode count: `50` for all three angles',
        '- Convergence comparison: `px = 4` versus `px = 8`, with an extra `px = 12` check at `1.005°`',
        '- Extra overlay: term-audit / EA spectra at `8.17°` and `3.01°`',
        '- Carrier analysis basis: unfold `px = 8` direct eigenvectors into the monolayer square-lattice Brillouin zone',
        '',
        '## px4 vs px8 spectral comparison',
        '',
        '| Angle | px4 range | px8 range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth px4 | Bandwidth px8 | |Δ bandwidth| |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), dpi=150, sharex=False)

    carrier_sections = []
    for row, (angle, angle_label) in enumerate(ANGLES):
        d4 = load_npz(angle, 4)
        d8 = load_npz(angle, 8)
        f4 = d4['freqs']
        f8 = d8['freqs']
        diff = f4 - f8
        rel = diff / np.maximum(np.abs(f8), 1e-12)
        bw4 = float(f4[-1] - f4[0])
        bw8 = float(f8[-1] - f8[0])

        report_lines.append(
            f'| {angle_label} | [{f4[0]:.6f}, {f4[-1]:.6f}] | [{f8[0]:.6f}, {f8[-1]:.6f}] | '
            f'{np.mean(np.abs(diff)):.6f} | {np.sqrt(np.mean(diff**2)):.6f} | '
            f'{np.mean(np.abs(rel))*100:.3f} | {np.max(np.abs(rel))*100:.3f} | '
            f'{bw4:.6f} | {bw8:.6f} | {abs(bw4-bw8):.6f} |'
        )

        ax = axes[row]
        mode_idx = np.arange(1, len(f4) + 1)
        ax.plot(mode_idx, f4, color='#ff7f0e', marker='o', ms=3, lw=1.3, label='px=4')
        ax.plot(mode_idx, f8, color='#1f77b4', marker='s', ms=3, lw=1.3, label='px=8')
        audit = load_audit_optional(angle)
        if audit is not None:
            fa = np.asarray(audit['frequencies'], dtype=float)
            ax.plot(
                np.arange(1, len(fa) + 1),
                fa,
                color='#9467bd',
                marker='x',
                ms=4,
                lw=1.1,
                ls='--',
                label='EA audit',
            )
        if angle == '1deg':
            d12 = load_npz_optional(angle, 12)
            d16 = load_npz_optional(angle, 16)
            if d12 is not None:
                f12 = d12['freqs']
                ax.plot(np.arange(1, len(f12) + 1), f12, color='#2ca02c', marker='^', ms=3, lw=1.2, label='px=12')
            if d16 is not None:
                f16 = d16['freqs']
                ax.plot(np.arange(1, len(f16) + 1), f16, color='#d62728', marker='D', ms=3, lw=1.2, label='px=16')
        ax.set_title(f'{angle_label} — sigma=0.1, TE at X')
        ax.set_xlabel('Mode index n')
        ax.set_ylabel('Frequency [c/a]')
        ax.grid(True, ls='--', alpha=0.35)
        if row == 0 or angle == '1deg' or audit is not None:
            ax.legend()

        if angle == '1deg':
            extra_rows = []
            for px in [12, 16]:
                dx = load_npz_optional(angle, px)
                if dx is None:
                    continue
                fx = dx['freqs']
                diff = fx - f8
                rel = diff / np.maximum(np.abs(f8), 1e-12)
                extra_rows.append(
                    f'| 1.005° | px={px} vs px=8 | [{fx[0]:.6f}, {fx[-1]:.6f}] | '
                    f'{np.mean(np.abs(diff)):.6f} | {np.sqrt(np.mean(diff**2)):.6f} | '
                    f'{np.mean(np.abs(rel))*100:.3f} | {np.max(np.abs(rel))*100:.3f} | '
                    f'{float(fx[-1]-fx[0]):.6f} | {abs(float(fx[-1]-fx[0]) - bw8):.6f} |'
                )
            if extra_rows:
                report_lines.extend([
                    '',
                    '## Additional 1.005° resolution checks',
                    '',
                    '| Angle | Comparison | Range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth | |Δ bandwidth vs px8| |',
                    '|---|---|---|---:|---:|---:|---:|---:|---:|',
                ])
                report_lines.extend(extra_rows)

        if audit is not None:
            fa = np.asarray(audit['frequencies'], dtype=float)
            diff_a = fa - f8
            rel_a = diff_a / np.maximum(np.abs(f8), 1e-12)
            bw_a = float(fa[-1] - fa[0])
            report_lines.extend([
                '',
                f'## Audit overlay comparison — {angle_label}',
                '',
                '| Comparison | Range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth | |Δ bandwidth vs px8| |',
                '|---|---|---:|---:|---:|---:|---:|---:|',
                f'| EA audit vs px=8 | [{fa[0]:.6f}, {fa[-1]:.6f}] | {np.mean(np.abs(diff_a)):.6f} | '
                f'{np.sqrt(np.mean(diff_a**2)):.6f} | {np.mean(np.abs(rel_a))*100:.3f} | {np.max(np.abs(rel_a))*100:.3f} | '
                f'{bw_a:.6f} | {abs(bw_a - bw8):.6f} |',
            ])

        mode_rows, aggregate = dominant_carrier_stats(angle, d8)
        counts = {'Gamma': 0, 'X': 0, 'M': 0, 'other': 0}
        for row_data in mode_rows:
            counts[row_data['dominant_label']] += 1

        carrier_sections.extend([
            '',
            f'## Carrier analysis — {angle_label}',
            '',
            f'- Dominant-label counts across 50 modes: Γ={counts["Gamma"]}, X={counts["X"]}, M={counts["M"]}, other={counts["other"]}',
            f'- Mean top-component carrier-family weight: Γ={aggregate["Gamma"]:.3f}, X={aggregate["X"]:.3f}, M={aggregate["M"]:.3f}, other={aggregate["other"]:.3f}',
            '',
            '| Representative modes | Frequency | Dominant carrier | Folded k in monolayer BZ (fractional) | Distance to labeled carrier |',
            '|---|---:|---|---|---:|',
        ])

        sample_indices = [0, 1, 2, 9, 19, 29, 39, 49]
        sample_indices = [idx for idx in sample_indices if idx < len(mode_rows)]
        for idx in sample_indices:
            row_data = mode_rows[idx]
            frac = row_data['dominant_frac']
            carrier_sections.append(
                f'| mode {row_data["mode"]} | {row_data["freq"]:.6f} | {row_data["dominant_label"]} | '
                f'({frac[0]:.3f}, {frac[1]:.3f}) | {row_data["dominant_dist"]:.3f} |'
            )

    report_lines.extend([
        '',
        '## Interpretation',
        '',
        '- `px = 8` is treated as the more resolved direct reference.',
        '- If the px4-vs-px8 differences remain small across the 50-mode window, then `4 px/cell` is still adequate at `sigma = 0.1` for this X-point target.',
        '- The carrier analysis is based on the unfolded Fourier content of the direct supercell eigenvectors, not on the EA carrier assumption.',
    ])
    report_lines.extend(carrier_sections)

    fig.suptitle('TE at X, sigma=0.1 — FDFD px4 vs px8 + EA audit overlays', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = BASE / 'fdfd_te_x_sigma01_px4_vs_px8.png'
    fig.savefig(fig_path, bbox_inches='tight')

    report_path = BASE / 'fdfd_te_x_sigma01_analysis.md'
    report_path.write_text('\n'.join(report_lines) + '\n')
    print(f'Saved → {fig_path}')
    print(f'Saved → {report_path}')


if __name__ == '__main__':
    main()