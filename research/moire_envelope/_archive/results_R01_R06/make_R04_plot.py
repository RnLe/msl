#!/usr/bin/env python3
"""
R04 Plot: Field Reconstruction & Mode Volume
==============================================
Visualize reconstructed fields, compare with envelope-only profiles.

Output: R04_field_reconstruction.png/.pdf, R04_mode_volume.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent


def main():
    print("="*70)
    print("R04 Plot: Field Reconstruction")
    print("="*70)

    with open(OUTDIR / "R04_data.json") as f:
        meta = json.load(f)

    fields = np.load(OUTDIR / "R04_fields.npz")
    modes = meta['modes']

    # ── Fig 1: Reconstructed fields vs envelope (3 modes) ─────────────────
    n_show = min(3, len([k for k in fields.keys() if k.startswith('field_')]))
    fig, axes = plt.subplots(n_show, 3, figsize=(14, 4.5*n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for mi in range(n_show):
        field = fields[f'field_{mi}']      # downsampled |E|²
        envelope = fields[f'envelope_{mi}']  # W(R) on envelope grid
        md = modes[mi]

        # Full reconstructed field
        ax = axes[mi, 0]
        im = ax.imshow(field.T, origin='lower', cmap='inferno', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Mode {mi}: $|H_z(\\mathbf{{r}})|^2$ (reconstructed)', fontsize=9)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        # Envelope-only
        ax = axes[mi, 1]
        im = ax.imshow(envelope.T, origin='lower', cmap='inferno', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Mode {mi}: $W(\\mathbf{{R}}) = \\sum_n |F_n|^2$ (envelope)', fontsize=9)
        ax.set_xlabel(r'$s_1$ index')
        ax.set_ylabel(r'$s_2$ index')

        # Zoom: center region of full field
        ax = axes[mi, 2]
        Nx, Ny = field.shape
        cx, cy = Nx // 2, Ny // 2
        w = min(Nx, Ny) // 4
        zoom = field[cx-w:cx+w, cy-w:cy+w]
        im = ax.imshow(zoom.T, origin='lower', cmap='inferno', aspect='equal')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Mode {mi}: Zoom (Bloch+envelope)', fontsize=9)

    fig.suptitle(
        f'Field Reconstruction — $\\theta={meta["theta_deg"]}°$, '
        f'$L_m={meta["L_moire"]:.0f}a$, grid={meta["full_grid"][0]}×{meta["full_grid"][1]}',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R04_field_reconstruction.{ext}"
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig)

    # ── Fig 2: Mode volume / area analysis ────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    mode_indices = [m['mode_index'] for m in modes]
    A_std = [m['A_eff_std'] for m in modes]
    A_ipr = [m['A_eff_ipr'] for m in modes]
    A_Lm2 = [m['A_eff_over_Lm2'] for m in modes]
    ldos = [m['ldos_enhancement'] for m in modes]
    corr = [m['envelope_correlation'] for m in modes]

    # Panel A: Mode area
    ax = axes2[0]
    ax.plot(mode_indices, A_Lm2, 'o-', ms=5, color='#e41a1c', label=r'$A_{\rm eff}/L_m^2$ (standard)')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='$L_m^2$')
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$A_{\rm eff} / L_m^2$')
    ax.set_title('Effective Mode Area')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: LDOS enhancement
    ax = axes2[1]
    ax.plot(mode_indices, ldos, 's-', ms=5, color='#377eb8')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('LDOS enhancement')
    ax.set_title(r'$\varepsilon|E|^2_{\max} / \langle\varepsilon|E|^2\rangle$')
    ax.grid(True, alpha=0.3)

    # Panel C: Envelope correlation
    ax = axes2[2]
    ax.plot(mode_indices, corr, 'd-', ms=5, color='#4daf4a')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Correlation')
    ax.set_title('Envelope vs Full-Field Correlation')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig2.suptitle('Mode Volume & Field Quality Analysis', fontsize=13, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R04_mode_volume.{ext}"
        fig2.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig2)

    print("\nR04 plot complete.")


if __name__ == '__main__':
    main()
