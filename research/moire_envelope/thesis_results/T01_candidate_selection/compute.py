"""
T01: Candidate Selection Summary — Thesis Figure

Generates the candidate screening summary showing:
  - Tier 1 parameter space coverage
  - Tier 2 validated candidates with Hessian classification
  - Final 3 selected candidates highlighted
  - V/E_kin vs θ* landscape

Usage:
    python thesis_results/T01_candidate_selection/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
    PROJECT_ROOT,
)

TASK = "T01_candidate_selection"


def load_tier2_data():
    """Load all Tier 2 results (hex + square)."""
    base = PROJECT_ROOT / "runsV3" / "phase0_mpb_v3_allk_scan_20260209_152023"

    dfs = []
    for subdir in ["tier2_results", "tier2_square_results"]:
        csv_path = base / subdir / "tier2_ranked.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['source'] = subdir
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No Tier 2 results found")

    df = pd.concat(dfs, ignore_index=True)
    # Classify
    df['is_saddle'] = df['eigval_min'] * df['eigval_max'] < 0
    df['is_minimum'] = (df['eigval_min'] > 0)
    df['is_maximum'] = (df['eigval_max'] < 0)
    return df


def plot_screening_summary(df):
    """Two-panel figure: (a) V_depth vs cond_number, (b) V/E_kin vs θ*."""
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Selected candidates (our 3)
    selected = {
        'hex_M_b1': {'eps_bg': 9.9, 'r_over_a': 0.10, 'band_index': 1, 'lattice': 'hex'},
        'hex_M_b3': {'eps_bg': 9.8, 'r_over_a': 0.10, 'band_index': 3, 'lattice': 'hex'},
        'square_M_b3': {'eps_bg': 1.8, 'r_over_a': 0.15, 'band_index': 3, 'lattice': 'square'},
    }

    # --- Panel (a): θ* vs cond_number colored by type ---
    ax = axes[0]
    # Saddle points
    saddle = df[df.is_saddle]
    ax.scatter(saddle.theta_star_deg, saddle.cond_number,
               c='gray', alpha=0.4, s=20, marker='x', label='Saddle point', zorder=2)
    # Band maxima
    maxima = df[df.is_maximum & ~df.is_saddle]
    ax.scatter(maxima.theta_star_deg, maxima.cond_number,
               c='#CC79A7', alpha=0.5, s=25, marker='v', label='Band maximum', zorder=2)
    # Band minima (good candidates)
    minima = df[df.is_minimum & ~df.is_saddle]
    ax.scatter(minima.theta_star_deg, minima.cond_number,
               c='#56B4E9', alpha=0.5, s=25, marker='o', label='Band minimum', zorder=2)

    # Highlight selected
    for name, info in selected.items():
        mask = df.family == name
        if mask.sum() == 0:
            # Try first matching row
            mask = (
                (df.lattice_type == info['lattice']) &
                (abs(df.eps_bg - info['eps_bg']) < 0.2) &
                (df.band_index == info['band_index'])
            )
        if mask.sum() > 0:
            row = df[mask].iloc[0]
            ax.scatter(row.theta_star_deg, row.cond_number,
                       c=CANDIDATE_COLORS[name], s=150, marker=CANDIDATE_MARKERS[name],
                       edgecolors='black', linewidths=1.5, zorder=5,
                       label=CANDIDATE_LABELS[name])

    ax.set_xlabel(r'$\theta^*$ [deg]')
    ax.set_ylabel('Hessian condition number')
    ax.set_yscale('log')
    ax.set_xlim(0, 12)
    ax.axhline(y=1, color='green', ls='--', alpha=0.3, label='Perfect isotropy')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_title('(a) Tier 2 candidate landscape')

    # --- Panel (b): V/E_kin at 2° vs θ* ---
    ax = axes[1]
    practical = df[(df.theta_star_deg < 12) & (~df.is_saddle)]

    # Color by lattice type
    hex_mask = practical.lattice_type == 'hex'
    sq_mask = practical.lattice_type == 'square'

    ax.scatter(practical[hex_mask].theta_star_deg, practical[hex_mask].VEkin_2deg,
               c='#E69F00', alpha=0.4, s=20, label='Hex candidates')
    ax.scatter(practical[sq_mask].theta_star_deg, practical[sq_mask].VEkin_2deg,
               c='#0072B2', alpha=0.4, s=20, label='Square candidates')

    # Highlight selected
    for name, info in selected.items():
        mask = df.family == name
        if mask.sum() == 0:
            mask = (
                (df.lattice_type == info['lattice']) &
                (abs(df.eps_bg - info['eps_bg']) < 0.2) &
                (df.band_index == info['band_index'])
            )
        if mask.sum() > 0:
            row = df[mask].iloc[0]
            ax.scatter(row.theta_star_deg, row.VEkin_2deg,
                       c=CANDIDATE_COLORS[name], s=150, marker=CANDIDATE_MARKERS[name],
                       edgecolors='black', linewidths=1.5, zorder=5,
                       label=CANDIDATE_LABELS[name])

    # Target range
    ax.axhspan(1, 10, alpha=0.1, color='green', label=r'Target $V/E_{\rm kin} \in [1,10]$')
    ax.set_xlabel(r'$\theta^*$ [deg]')
    ax.set_ylabel(r'$V/E_{\rm kin}$ at $\theta=2°$')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 20)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title(r'(b) Potential depth at $\theta=2°$')

    fig.tight_layout()
    return fig


def plot_candidate_table(df):
    """Table figure with final 3 candidate parameters."""
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    columns = ['Label', 'Lattice', 'k', 'Band', 'Pol', 'r/a', 'ε', 'θ*', 'κ', 'V/E@2°']
    data = [
        ['C1', 'hex', 'M', '1', 'TE', '0.10', '9.9', '2.1°', '15.0', '1.1'],
        ['C2', 'hex', 'M', '3', 'TE', '0.10', '9.8', '4.2°', '35.5', '4.3'],
        ['C3', 'square', 'M', '3', 'TM', '0.15', '1.8', '2.5°', '1.0', '1.5'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#E8E8E8']*len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color candidate rows
    colors_list = [CANDIDATE_COLORS['hex_M_b1'],
                   CANDIDATE_COLORS['hex_M_b3'],
                   CANDIDATE_COLORS['square_M_b3']]
    for i, color in enumerate(colors_list):
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_facecolor(color + '30')  # 30 = alpha in hex

    ax.set_title('Selected Thesis Candidates', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T01: Candidate Selection Summary → {out_dir}")

    df = load_tier2_data()
    print(f"  Loaded {len(df)} Tier 2 candidates")

    fig1 = plot_screening_summary(df)
    save_figure(fig1, TASK, "T01_screening_landscape")

    fig2 = plot_candidate_table(df)
    save_figure(fig2, TASK, "T01_candidate_table")

    print("  T01 complete.")


if __name__ == "__main__":
    main()
