#!/usr/bin/env python3
"""
R01 Plot: Effective Hamiltonian Landscape
==========================================
Multi-panel figure showing the 5-band Hamiltonian parameter fields over
one moiré unit cell.

Layout: 
  - Top 5×4: V(R), |A(R)|, Tr[M⁻¹(R)], Φ_BH(R) for each band
  - Bottom row: off-diagonal coupling heatmap

Output: R01_hamiltonian_landscape.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent

def main():
    print("="*70)
    print("R01 Plot: Hamiltonian Landscape")
    print("="*70)

    # ── Load data ──────────────────────────────────────────────────────────
    data = np.load(OUTDIR / "R01_data.npz")
    V_diag      = data['V_diag']       # (Ns, Ns, Nb)
    A_diag_mag  = data['A_diag_mag']   # (Ns, Ns, Nb)
    M_trace     = data['M_trace']      # (Ns, Ns, Nb)
    M_aniso     = data['M_aniso']      # (Ns, Ns, Nb)
    Phi_diag    = data['Phi_diag']     # (Ns, Ns, Nb)
    s_grid      = data['s_grid']       # (Ns, Ns, 2)

    with open(OUTDIR / "R01_data.json") as f:
        meta = json.load(f)

    Nb = V_diag.shape[2]
    theta = meta['theta_deg']
    eta = meta['eta']
    sub_bands = meta['subspace_bands']
    target_idx = meta['target_index_in_subspace']
    band_stats = meta['band_stats']

    s1 = s_grid[:, 0, 0]  # fractional coords [0, 1)
    s2 = s_grid[0, :, 1]

    # ── Figure: 5 bands × 4 panels ────────────────────────────────────────
    fig, axes = plt.subplots(Nb, 4, figsize=(16, 4*Nb))

    col_labels = [
        r'$V_n(\mathbf{s})$ (potential)',
        r'$|\mathbf{A}_{nn}(\mathbf{s})|$ (Berry)',
        r'$\mathrm{Tr}[M^{-1}_{nn}(\mathbf{s})]$ (mass)',
        r'$\Phi^\mathrm{BH}_{nn}(\mathbf{s})$'
    ]

    for n in range(Nb):
        bs = band_stats[n]
        btype = bs['type']
        label = f"Band {n} ({btype})"
        if n == target_idx:
            label += " ★"

        # ── Column 0: Potential V_n(s) ─────────────────────────────────
        ax = axes[n, 0]
        Vn = V_diag[:, :, n]
        vmax = max(abs(Vn.min()), abs(Vn.max()))
        if vmax < 1e-10:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.pcolormesh(s1, s2, Vn.T, cmap='RdBu_r', norm=norm, shading='auto')
        plt.colorbar(im, ax=ax, format='%.3f')
        ax.set_ylabel(label, fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        if n == 0:
            ax.set_title(col_labels[0], fontsize=10)

        # ── Column 1: Berry connection |A_nn| ─────────────────────────
        ax = axes[n, 1]
        An = A_diag_mag[:, :, n]
        im = ax.pcolormesh(s1, s2, An.T, cmap='viridis', shading='auto')
        plt.colorbar(im, ax=ax, format='%.3f')
        ax.set_aspect('equal')
        if n == 0:
            ax.set_title(col_labels[1], fontsize=10)

        # ── Column 2: Mass trace ───────────────────────────────────────
        ax = axes[n, 2]
        Mt = M_trace[:, :, n]
        # Clip to avoid divergences dominating colorscale
        clip = float(np.percentile(np.abs(Mt), 98))
        if clip < 1e-10:
            clip = 1.0
        norm_m = TwoSlopeNorm(vmin=-clip, vcenter=0, vmax=clip)
        im = ax.pcolormesh(s1, s2, Mt.T, cmap='PuOr', norm=norm_m, shading='auto')
        plt.colorbar(im, ax=ax, format='%.1f')
        ax.set_aspect('equal')
        if n == 0:
            ax.set_title(col_labels[2], fontsize=10)

        # ── Column 3: Born-Huang Φ_BH ─────────────────────────────────
        ax = axes[n, 3]
        Ph = Phi_diag[:, :, n]
        pmax = max(abs(Ph.min()), abs(Ph.max()))
        if pmax < 1e-15:
            # BH is effectively zero — show that
            im = ax.pcolormesh(s1, s2, Ph.T, cmap='Greys', shading='auto')
            plt.colorbar(im, ax=ax, format='%.1e')
        else:
            norm_p = TwoSlopeNorm(vmin=-pmax, vcenter=0, vmax=pmax)
            im = ax.pcolormesh(s1, s2, Ph.T, cmap='PiYG', norm=norm_p, shading='auto')
            plt.colorbar(im, ax=ax, format='%.1e')
        ax.set_aspect('equal')
        if n == 0:
            ax.set_title(col_labels[3], fontsize=10)

    # Clean up tick labels (only bottom row gets x-labels)
    for n in range(Nb):
        for c in range(4):
            ax = axes[n, c]
            if n < Nb - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'$s_1$', fontsize=9)
            if c > 0:
                ax.set_yticklabels([])

    fig.suptitle(
        f"Effective Hamiltonian Landscape — "
        f"Square lattice, $\\varepsilon$=12, $r/a$=0.35, "
        f"$\\theta$={theta}°, $\\eta$={eta:.4f}\n"
        f"Bands {sub_bands}, target=Band {sub_bands[target_idx]} (★)",
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R01_hamiltonian_landscape.{ext}"
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")

    plt.close(fig)

    # ── Figure 2: Off-diagonal coupling matrix ─────────────────────────────
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4))

    coupling_names = ['Lambda', 'A_berry', 'v_drift', 'Phi_BH']
    coupling_labels = [r'$|\Lambda_{mn}|_{\max}$', r'$|A_{mn}|_{\max}$',
                       r'$|v_{mn}|_{\max}$', r'$|\Phi^{BH}_{mn}|_{\max}$']
    coupling_keys = ['Lambda_mn_max', 'A_mn_max', 'v_mn_max', 'Phi_mn_max']

    off_diag = meta['off_diag_coupling']

    for ci, (ckey, clabel) in enumerate(zip(coupling_keys, coupling_labels)):
        ax = axes2[ci]
        mat = np.zeros((Nb, Nb))
        for entry in off_diag:
            m, n = entry['pair']
            val = entry[ckey]
            mat[m, n] = val
            mat[n, m] = val  # symmetric
        # Add diagonal entries for reference
        for n in range(Nb):
            bs = band_stats[n]
            if 'Lambda' in ckey:
                mat[n, n] = bs['V_range']
            elif 'A' in ckey:
                mat[n, n] = bs['A_max']
            elif 'Phi' in ckey:
                mat[n, n] = bs['Phi_BH_max']

        im = ax.imshow(mat, cmap='YlOrRd', origin='upper')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(clabel, fontsize=10)
        ax.set_xticks(range(Nb))
        ax.set_yticks(range(Nb))
        ax.set_xticklabels([f"B{i}" for i in range(Nb)])
        ax.set_yticklabels([f"B{i}" for i in range(Nb)])
        # Annotate values
        for i in range(Nb):
            for j in range(Nb):
                val = mat[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                            fontsize=7, color='black' if val < mat.max()*0.6 else 'white')

    fig2.suptitle("Inter-Band Coupling Strengths", fontsize=12, fontweight='bold')
    fig2.tight_layout()

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R01_coupling_matrix.{ext}"
        fig2.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")

    plt.close(fig2)
    print("\nR01 plot complete.")


if __name__ == '__main__':
    main()
