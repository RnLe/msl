#!/usr/bin/env python3
"""
Phase B: Band Tracking Field Study — Visualization & Analysis
=============================================================

Uses existing overnight Phase 1 data to:
1. Visualize band surfaces ω_n(δ) across the registry grid
2. Plot overlap score heatmap from tracking diagnostics
3. Identify crossing topology
4. Quantify tracking quality via smoothness metrics

Does NOT re-run any MPB computations — purely analytical.
"""

import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
OVERNIGHT_DIR = SCRIPT_DIR / 'overnight_validation' / 'run_20260313_004032'

OUTPUT_DIR = SCRIPT_DIR / 'phase_b_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)


def load_phase1_registry(angle_name):
    """Load raw Phase 1 HDF5 from overnight run."""
    h5_path = OVERNIGHT_DIR / angle_name / 'shared_phase1' / 'candidate_0000' / 'phase1_multiband_data.h5'
    if not h5_path.exists():
        print(f'  WARNING: {h5_path} not found')
        return None

    data = {}
    with h5py.File(h5_path, 'r') as hf:
        data['omega'] = hf['omega'][:]          # (Ns1, Ns2, N_subspace)
        data['vg'] = hf['vg'][:]                # (Ns1, Ns2, N_subspace, 2)
        data['M_inv'] = hf['M_inv'][:]          # (Ns1, Ns2, N_subspace, 2, 2)
        data['V'] = hf['V'][:]                  # (Ns1, Ns2, N_subspace)

        # Registry-level stencil data has ALL bands
        data['registry_omega_all'] = hf['stencil/registry_omega_all'][:]  # (n_reg, n_reg, N_all)
        n_reg = data['registry_omega_all'].shape[0]
        data['n_registry'] = n_reg

        # Full stencil data (for curvature analysis)
        if 'stencil/omega_all' in hf:
            data['stencil_omega_all'] = hf['stencil/omega_all'][:]  # (n_reg, n_reg, N_all, 7, 7)

        # Metadata
        data['omega_ref'] = float(hf.attrs['omega_ref'])
        data['eta'] = float(hf.attrs['eta'])
        data['theta_deg'] = float(hf.attrs['theta_deg'])
        data['moire_length'] = float(hf.attrs['moire_length'])
        data['Ns1'] = int(hf.attrs['Ns1'])
        data['Ns2'] = int(hf.attrs['Ns2'])
        data['N_subspace'] = int(hf.attrs['N_subspace'])
        data['subspace_bands'] = hf.attrs['subspace_bands'][:].tolist()
        data['all_bands'] = hf.attrs['all_bands'][:].tolist()

    # Load tracking diagnostics if available
    tracking_path = OVERNIGHT_DIR / angle_name / 'shared_phase1' / 'candidate_0000' / 'phase1_tracking_diagnostics.json'
    if tracking_path.exists():
        with open(tracking_path) as f:
            data['tracking_diag'] = json.load(f)

    return data


def plot_band_surfaces(data, angle_name, bands_to_plot=None):
    """
    Plot ω_n(δ) for specified bands across the registry grid.
    Shows the raw band landscape BEFORE interpolation to moiré grid.
    """
    registry_omega = data['registry_omega_all']  # (n_reg, n_reg, N_all)
    n_reg = data['n_registry']
    all_bands = data['all_bands']

    if bands_to_plot is None:
        bands_to_plot = data['subspace_bands']

    n_bands = len(bands_to_plot)
    fig, axes = plt.subplots(2, n_bands, figsize=(5 * n_bands, 9))
    if n_bands == 1:
        axes = axes.reshape(2, 1)

    s = np.linspace(0, 1, n_reg, endpoint=False)
    S1, S2 = np.meshgrid(s, s, indexing='ij')

    for i, band_idx in enumerate(bands_to_plot):
        all_idx = all_bands.index(band_idx)
        omega = registry_omega[:, :, all_idx]

        # Top row: 2D colormap
        ax = axes[0, i]
        im = ax.pcolormesh(S1, S2, omega, cmap='viridis', shading='auto')
        ax.set_title(f'Band {band_idx}: ω(δ)')
        ax.set_xlabel('δ₁')
        ax.set_ylabel('δ₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label='ω (c/a)')

        # Bottom row: curvature of ω surface (smoothness diagnostic)
        # Compute |∇ω| via finite differences (periodic)
        dw_d1 = np.roll(omega, -1, axis=0) - np.roll(omega, 1, axis=0)
        dw_d2 = np.roll(omega, -1, axis=1) - np.roll(omega, 1, axis=1)
        grad_mag = np.sqrt(dw_d1**2 + dw_d2**2)

        ax = axes[1, i]
        im = ax.pcolormesh(S1, S2, grad_mag, cmap='hot', shading='auto')
        ax.set_title(f'Band {band_idx}: |∇ω| (smoothness)')
        ax.set_xlabel('δ₁')
        ax.set_ylabel('δ₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label='|∇ω|')

    fig.suptitle(f'{angle_name}: Band ω surfaces on registry grid ({n_reg}×{n_reg})',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{angle_name}_band_surfaces.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {angle_name}_band_surfaces.png')


def plot_band_gaps(data, angle_name):
    """
    Plot energy gaps between adjacent bands across the registry grid.
    Shows where bands cross (gap → 0) and where they're well-separated.
    """
    registry_omega = data['registry_omega_all']  # (n_reg, n_reg, N_all)
    n_reg = data['n_registry']
    subspace = data['subspace_bands']
    all_bands = data['all_bands']

    # Show gaps between consecutive subspace bands
    n_gaps = len(subspace) - 1
    if n_gaps == 0:
        return

    fig, axes = plt.subplots(1, n_gaps + 1, figsize=(5 * (n_gaps + 1), 5))
    if n_gaps + 1 == 1:
        axes = [axes]

    s = np.linspace(0, 1, n_reg, endpoint=False)
    S1, S2 = np.meshgrid(s, s, indexing='ij')

    for i in range(n_gaps):
        b_lower = subspace[i]
        b_upper = subspace[i + 1]
        idx_lower = all_bands.index(b_lower)
        idx_upper = all_bands.index(b_upper)

        gap = registry_omega[:, :, idx_upper] - registry_omega[:, :, idx_lower]

        ax = axes[i]
        im = ax.pcolormesh(S1, S2, gap, cmap='RdYlGn', shading='auto',
                           norm=Normalize(vmin=min(0, gap.min()), vmax=max(0.01, gap.max())))
        ax.set_title(f'Gap: band {b_upper} − band {b_lower}')
        ax.set_xlabel('δ₁')
        ax.set_ylabel('δ₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Δω (c/a)')

        # Mark points where gap < threshold (crossings)
        crossing_mask = np.abs(gap) < 0.005  # threshold for "near-degenerate"
        if np.any(crossing_mask):
            ax.contour(S1, S2, gap, levels=[0.005], colors='red', linewidths=1)
            n_crossing = np.sum(crossing_mask)
            ax.text(0.02, 0.98, f'{n_crossing}/{n_reg**2} near-degen',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))

    # Last panel: gap between band below subspace and first subspace band
    b_below = subspace[0] - 1
    if b_below >= 0 and b_below in all_bands:
        idx_below = all_bands.index(b_below)
        idx_first = all_bands.index(subspace[0])
        gap_below = registry_omega[:, :, idx_first] - registry_omega[:, :, idx_below]

        ax = axes[-1]
        im = ax.pcolormesh(S1, S2, gap_below, cmap='RdYlGn', shading='auto')
        ax.set_title(f'Isolation gap: band {subspace[0]} − band {b_below}')
        ax.set_xlabel('δ₁')
        ax.set_ylabel('δ₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Δω (c/a)')

    fig.suptitle(f'{angle_name}: Band gaps on registry grid', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{angle_name}_band_gaps.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {angle_name}_band_gaps.png')


def plot_m_inv_diagnostics(data, angle_name):
    """
    Plot M_inv trace and determinant on the moiré grid.
    Sign flips indicate tracking errors (band identity changed).
    """
    M_inv = data['M_inv']  # (Ns1, Ns2, N_subspace, 2, 2)
    omega = data['omega']  # (Ns1, Ns2, N_subspace)
    Ns1, Ns2, N_sub = omega.shape

    fig, axes = plt.subplots(2, N_sub, figsize=(5 * N_sub, 9))
    if N_sub == 1:
        axes = axes.reshape(2, 1)

    s1 = np.linspace(0, 1, Ns1, endpoint=False)
    s2 = np.linspace(0, 1, Ns2, endpoint=False)
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    subspace = data['subspace_bands']

    for n in range(N_sub):
        tr = M_inv[:, :, n, 0, 0] + M_inv[:, :, n, 1, 1]
        det = M_inv[:, :, n, 0, 0] * M_inv[:, :, n, 1, 1] - M_inv[:, :, n, 0, 1]**2

        # Top: trace
        ax = axes[0, n]
        vmax = min(np.percentile(np.abs(tr), 95), 20)
        im = ax.pcolormesh(S1, S2, tr, cmap='coolwarm', shading='auto',
                           norm=Normalize(vmin=-vmax, vmax=vmax))
        ax.set_title(f'Band {subspace[n]}: Tr(M⁻¹)')
        ax.set_xlabel('s₁')
        ax.set_ylabel('s₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Count sign flips (tracking indicator)
        n_positive = np.sum(tr > 0)
        n_negative = np.sum(tr < 0)
        ax.text(0.02, 0.98, f'+:{n_positive} −:{n_negative}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

        # Bottom: determinant
        ax = axes[1, n]
        vmax_d = min(np.percentile(np.abs(det), 95), 100)
        im = ax.pcolormesh(S1, S2, det, cmap='coolwarm', shading='auto',
                           norm=Normalize(vmin=-vmax_d, vmax=vmax_d))
        ax.set_title(f'Band {subspace[n]}: det(M⁻¹)')
        ax.set_xlabel('s₁')
        ax.set_ylabel('s₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'{angle_name}: M⁻¹ diagnostics on moiré grid (post-tracking)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{angle_name}_M_inv_diagnostics.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {angle_name}_M_inv_diagnostics.png')


def plot_potential_landscape(data, angle_name):
    """Plot the effective potential V(R) = ω(R) - ω_ref on the moiré grid."""
    V = data['V']  # (Ns1, Ns2, N_subspace)
    omega = data['omega']
    Ns1, Ns2, N_sub = omega.shape
    subspace = data['subspace_bands']

    fig, axes = plt.subplots(1, N_sub, figsize=(5 * N_sub, 5))
    if N_sub == 1:
        axes = [axes]

    s1 = np.linspace(0, 1, Ns1, endpoint=False)
    s2 = np.linspace(0, 1, Ns2, endpoint=False)
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    for n in range(N_sub):
        ax = axes[n]
        im = ax.pcolormesh(S1, S2, V[:, :, n], cmap='coolwarm', shading='auto')
        ax.set_title(f'Band {subspace[n]}: V(R) = ω(R) − ω_ref')
        ax.set_xlabel('s₁')
        ax.set_ylabel('s₂')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label='V (c/a)')

        # Show range
        v_min, v_max = V[:, :, n].min(), V[:, :, n].max()
        ax.text(0.02, 0.02, f'range: [{v_min:.4f}, {v_max:.4f}]',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(f'{angle_name}: Effective potential on moiré grid',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{angle_name}_potential.png', dpi=150)
    plt.close(fig)
    print(f'  Saved {angle_name}_potential.png')


def compute_smoothness_metrics(data, angle_name):
    """
    Compute smoothness metrics for tracked band data.
    Large gradient discontinuities indicate tracking errors.
    """
    omega = data['omega']  # (Ns1, Ns2, N_subspace)
    M_inv = data['M_inv']  # (Ns1, Ns2, N_subspace, 2, 2)
    Ns1, Ns2, N_sub = omega.shape
    subspace = data['subspace_bands']

    metrics = {}
    for n in range(N_sub):
        w = omega[:, :, n]
        tr = M_inv[:, :, n, 0, 0] + M_inv[:, :, n, 1, 1]

        # Gradient of ω (periodic FD)
        dw_1 = np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)
        dw_2 = np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)
        grad_w = np.sqrt(dw_1**2 + dw_2**2)

        # Gradient of Tr(M_inv) (periodic FD)
        dtr_1 = np.roll(tr, -1, axis=0) - np.roll(tr, 1, axis=0)
        dtr_2 = np.roll(tr, -1, axis=1) - np.roll(tr, 1, axis=1)
        grad_tr = np.sqrt(dtr_1**2 + dtr_2**2)

        # Sign consistency of Tr(M_inv)
        n_positive = int(np.sum(tr > 0))
        n_negative = int(np.sum(tr < 0))

        metrics[f'band_{subspace[n]}'] = {
            'omega_range': float(w.max() - w.min()),
            'omega_mean': float(np.mean(w)),
            'omega_std': float(np.std(w)),
            'grad_omega_mean': float(np.mean(grad_w)),
            'grad_omega_max': float(np.max(grad_w)),
            'grad_omega_p95': float(np.percentile(grad_w, 95)),
            'tr_M_inv_mean': float(np.mean(tr)),
            'tr_M_inv_std': float(np.std(tr)),
            'tr_M_inv_positive': n_positive,
            'tr_M_inv_negative': n_negative,
            'tr_M_inv_sign_purity': float(max(n_positive, n_negative) / (Ns1 * Ns2)),
            'grad_tr_mean': float(np.mean(grad_tr)),
            'grad_tr_max': float(np.max(grad_tr)),
            'grad_tr_p95': float(np.percentile(grad_tr, 95)),
        }

    return metrics


def main():
    print('='*60)
    print('  Phase B: Band Tracking Field Study')
    print('='*60)
    print(f'  Overnight data: {OVERNIGHT_DIR}')
    print(f'  Output: {OUTPUT_DIR}')
    print()

    all_metrics = {}

    for angle_name in ['10deg', '7deg', '4deg', '2deg']:
        print(f'\n--- {angle_name} ---')
        data = load_phase1_registry(angle_name)
        if data is None:
            continue

        print(f'  θ={data["theta_deg"]:.2f}°, η={data["eta"]:.4f}')
        print(f'  Registry: {data["n_registry"]}×{data["n_registry"]}, '
              f'moiré grid: {data["Ns1"]}×{data["Ns2"]}')
        print(f'  Subspace bands: {data["subspace_bands"]}')
        print(f'  All bands: {data["all_bands"]}')

        if 'tracking_diag' in data:
            td = data['tracking_diag']
            print(f'  Tracking: {td.get("n_points_changed", "?")} points changed, '
                  f'min_score={td.get("match_score_min", "N/A")}')

        # 1. Band surfaces (registry-level, ALL bands)
        plot_band_surfaces(data, angle_name, bands_to_plot=data['subspace_bands'])

        # 2. Band gaps (registry-level)
        plot_band_gaps(data, angle_name)

        # 3. M_inv diagnostics (moiré-grid level, post-tracking)
        plot_m_inv_diagnostics(data, angle_name)

        # 4. Potential landscape
        plot_potential_landscape(data, angle_name)

        # 5. Smoothness metrics
        metrics = compute_smoothness_metrics(data, angle_name)
        all_metrics[angle_name] = metrics

        for band_key, m in metrics.items():
            print(f'  {band_key}:')
            print(f'    ω range: {m["omega_range"]:.4f}, grad_ω max: {m["grad_omega_max"]:.4f}')
            print(f'    Tr(M⁻¹): mean={m["tr_M_inv_mean"]:.2f}, sign_purity={m["tr_M_inv_sign_purity"]:.2%}')
            print(f'    grad_Tr max: {m["grad_tr_max"]:.2f}')

    # Save all metrics
    with open(OUTPUT_DIR / 'smoothness_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\n  Saved smoothness_metrics.json')

    # ── Summary report ──
    print('\n' + '='*60)
    print('  PHASE B SUMMARY: Band Crossing & Tracking Quality')
    print('='*60)
    for angle_name, metrics in all_metrics.items():
        print(f'\n  {angle_name}:')
        for band_key, m in metrics.items():
            sign_purity = m['tr_M_inv_sign_purity']
            status = '✓ CLEAN' if sign_purity > 0.95 else '⚠ MIXED' if sign_purity > 0.7 else '✗ BAD'
            print(f'    {band_key}: sign_purity={sign_purity:.2%} {status}, '
                  f'ω_range={m["omega_range"]:.4f}')


if __name__ == '__main__':
    main()
