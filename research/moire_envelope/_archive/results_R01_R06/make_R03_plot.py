#!/usr/bin/env python3
"""
R03 Plot: Envelope Mode Gallery
=================================
Gallery of envelope modes at multiple twist angles.

Layout:
  - Fig 1: 4×5 gallery of W(R) for lowest 20 modes at θ=1.1°
  - Fig 2: Comparison of lowest 5 modes at θ = {1.1, 3.0, 8.0}°
  - Fig 3: IPR and localization analysis

Output: R03_envelope_modes.png/.pdf, R03_mode_comparison.png/.pdf, R03_localization.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent
BAND_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

def main():
    print("="*70)
    print("R03 Plot: Envelope Mode Gallery")
    print("="*70)

    data = np.load(OUTDIR / "R03_data.npz")
    with open(OUTDIR / "R03_data.json") as f:
        meta = json.load(f)

    # ── Fig 1: Mode gallery at θ=1.1° ─────────────────────────────────────
    theta_ref = '1.100'
    W_ref = data[f"W_{theta_ref}"]  # (n_modes, Ns1, Ns2)
    s_grid = data[f"s_grid_{theta_ref}"]
    modes_ref = meta[theta_ref]['modes']
    Lm = meta[theta_ref]['L_moire']
    n_modes = min(20, W_ref.shape[0])

    nrows, ncols = 4, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))

    s1 = s_grid[:, 0, 0]
    s2 = s_grid[0, :, 1]

    for mi in range(n_modes):
        ax = axes[mi // ncols, mi % ncols]
        Wm = W_ref[mi]
        md = modes_ref[mi]
        dom = md['dominant_band']
        dw = md['dominant_weight']

        im = ax.pcolormesh(s1, s2, Wm.T, cmap='inferno', shading='auto')
        ax.set_aspect('equal')
        ax.set_title(f"M{mi}: B{dom}({dw:.0%})\n$\\lambda$={md['eigenvalue']:.5f}",
                     fontsize=7, color=BAND_COLORS[dom])
        ax.set_xticks([])
        ax.set_yticks([])

    theta_val = meta[theta_ref]['theta_deg']
    fig.suptitle(
        f"Envelope Modes $W(\\mathbf{{R}}) = \\sum_n |F_n|^2$ — "
        f"$\\theta={theta_val}°$, "
        f"$L_m={Lm:.0f}a$",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R03_envelope_modes.{ext}"
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig)

    # ── Fig 2: Comparison across angles ────────────────────────────────────
    thetas = list(meta.keys())
    n_compare = min(5, n_modes)
    fig2, axes2 = plt.subplots(n_compare, len(thetas), figsize=(4*len(thetas), 3.5*n_compare))

    for ti, theta_str in enumerate(thetas):
        W_t = data[f"W_{theta_str}"]
        sg = data[f"s_grid_{theta_str}"]
        s1_t = sg[:, 0, 0]
        s2_t = sg[0, :, 1]
        modes_t = meta[theta_str]['modes']

        for mi in range(n_compare):
            ax = axes2[mi, ti]
            Wm = W_t[mi]
            md = modes_t[mi]
            dom = md['dominant_band']

            im = ax.pcolormesh(s1_t, s2_t, Wm.T, cmap='inferno', shading='auto')
            ax.set_aspect('equal')

            if mi == 0:
                ax.set_title(f"$\\theta={meta[theta_str]['theta_deg']}°$\n"
                             f"$\\eta={meta[theta_str]['eta']:.4f}$",
                             fontsize=10, fontweight='bold')
            if ti == 0:
                ax.set_ylabel(f"Mode {mi}\nB{dom}({md['dominant_weight']:.0%})",
                              fontsize=8, color=BAND_COLORS[dom])

            # Inset text with PN
            ax.text(0.02, 0.98, f"PN={md['participation_number']:.0f}",
                    transform=ax.transAxes, fontsize=7, va='top',
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

            ax.set_xticks([])
            ax.set_yticks([])

    fig2.suptitle('Mode Comparison Across Twist Angles', fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R03_mode_comparison.{ext}"
        fig2.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig2)

    # ── Fig 3: Localization analysis ───────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: IPR vs mode index
    ax = axes3[0]
    for theta_str in thetas:
        modes_t = meta[theta_str]['modes']
        iprs = [m['ipr'] for m in modes_t]
        ax.semilogy(range(len(iprs)), iprs, 'o-', ms=3,
                    label=f"$\\theta={meta[theta_str]['theta_deg']}°$")
    ax.set_xlabel('Mode index')
    ax.set_ylabel('IPR')
    ax.set_title('Inverse Participation Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: Participation number vs mode
    ax = axes3[1]
    for theta_str in thetas:
        modes_t = meta[theta_str]['modes']
        pns = [m['participation_number'] for m in modes_t]
        ax.plot(range(len(pns)), pns, 'o-', ms=3,
                label=f"$\\theta={meta[theta_str]['theta_deg']}°$")
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Participation Number')
    ax.set_title('Mode Delocalization')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: Spread / L_moire vs mode
    ax = axes3[2]
    for theta_str in thetas:
        modes_t = meta[theta_str]['modes']
        spreads = [m['spread_over_Lm'] for m in modes_t]
        ax.plot(range(len(spreads)), spreads, 'o-', ms=3,
                label=f"$\\theta={meta[theta_str]['theta_deg']}°$")
    ax.axhline(0.5, color='gray', ls='--', lw=0.8, label=r'$\sigma/L_m = 0.5$ (delocalized)')
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\sigma / L_{\rm moiré}$')
    ax.set_title('Spatial Spread')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig3.suptitle('Localization Analysis of Envelope Modes', fontsize=13, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R03_localization.{ext}"
        fig3.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig3)

    print("\nR03 plot complete.")


if __name__ == '__main__':
    main()
