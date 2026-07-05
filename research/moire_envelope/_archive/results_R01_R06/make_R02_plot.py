#!/usr/bin/env python3
"""
R02 Plot: Multi-Band Miniband Dispersion
==========================================
Band structure ω(q) along moiré BZ path, colored by dominant band character.

Layout:
  - 2×2 panels for θ = {1.1, 3.0, 5.0, 8.0}°
  - Each shows ω(q) colored by dominant band, with DOS on sidepanel

Output: R02_miniband_dispersion.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent

# Band colors: 5-color qualitative palette
BAND_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
BAND_NAMES = ['Band 0 (hole)', 'Band 1 (elec)', 'Band 2 (hole)',
              'Band 3 (hole)', 'Band 4 (hole)']

def main():
    print("="*70)
    print("R02 Plot: Miniband Dispersion")
    print("="*70)

    # Load data
    data = np.load(OUTDIR / "R02_data.npz")
    with open(OUTDIR / "R02_data.json") as f:
        meta = json.load(f)

    thetas = meta['thetas']
    path_labels = meta['path_labels']
    n_modes = meta['n_modes']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ti, theta_str in enumerate(thetas):
        ax = axes[ti]
        prefix = f"t{theta_str}"

        evals = data[f"{prefix}_eigenvalues"]     # (n_q, n_modes)
        weights = data[f"{prefix}_band_weights"]   # (n_q, n_modes, N_bands)
        q_dist = data[f"{prefix}_q_distances"]     # (n_q,)

        info = meta['per_theta'][theta_str]
        ticks = info['tick_positions']
        theta_deg = info['theta_deg']
        eta = info['eta']
        omega_ref = info['omega_ref']

        n_q, nm = evals.shape
        N_bands = weights.shape[2]

        # Plot each miniband colored by dominant band character
        for mi in range(min(nm, n_modes)):
            # Dominant band at each q-point
            dom = np.argmax(weights[:, mi, :], axis=1)
            dom_weight = np.max(weights[:, mi, :], axis=1)

            # Use scatter with color = dominant band, alpha = weight
            for bi in range(N_bands):
                mask = dom == bi
                if np.any(mask):
                    ax.scatter(q_dist[mask], evals[mask, mi],
                              c=BAND_COLORS[bi], s=2, alpha=0.7,
                              rasterized=True, zorder=2)

        # Vertical lines at high-symmetry points
        for tp in ticks:
            ax.axvline(tp, color='gray', lw=0.5, ls='--', zorder=1)

        # Labels
        ax.set_xticks(ticks)
        ax.set_xticklabels(path_labels, fontsize=11)
        ax.set_ylabel(r'$\lambda = \omega - \omega_{\rm ref}$', fontsize=10)
        ax.set_title(f'$\\theta = {theta_deg}°$,  $\\eta = {eta:.4f}$', fontsize=11)

        # y-range: focus on the miniband region
        e_min, e_max = np.nanmin(evals), np.nanmax(evals)
        margin = (e_max - e_min) * 0.05
        ax.set_ylim(e_min - margin, e_max + margin)

        ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=BAND_COLORS[i],
                       markersize=6, label=BAND_NAMES[i]) for i in range(5)]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'Multi-Band Miniband Dispersion $\\omega(\\mathbf{q})$ along Moiré BZ',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R02_miniband_dispersion.{ext}"
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig)

    # ── Figure 2: Group velocity ───────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    axes2 = axes2.flatten()

    for ti, theta_str in enumerate(thetas):
        ax = axes2[ti]
        prefix = f"t{theta_str}"
        vg = data[f"{prefix}_group_velocity"]
        q_dist = data[f"{prefix}_q_distances"]
        weights = data[f"{prefix}_band_weights"]
        info = meta['per_theta'][theta_str]
        ticks = info['tick_positions']

        for mi in range(min(5, n_modes)):
            dom = np.argmax(weights[:, mi, :], axis=1)
            # Most common band for this miniband
            main_band = int(np.bincount(dom).argmax())
            ax.plot(q_dist, np.abs(vg[:, mi]),
                    color=BAND_COLORS[main_band], alpha=0.7, lw=1)

        for tp in ticks:
            ax.axvline(tp, color='gray', lw=0.5, ls='--')
        ax.set_xticks(ticks)
        ax.set_xticklabels(path_labels, fontsize=11)
        ax.set_ylabel(r'$|v_g| = |d\omega/dq|$', fontsize=10)
        ax.set_title(f'$\\theta = {info["theta_deg"]}°$', fontsize=11)
        ax.grid(True, alpha=0.2)

    fig2.suptitle('Group Velocity of Lowest Minibands', fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R02_group_velocity.{ext}"
        fig2.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig2)

    print("\nR02 plot complete.")


if __name__ == '__main__':
    main()
