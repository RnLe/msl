#!/usr/bin/env python3
"""
F06 — Plotting: Bloch Field Gauge & Normalization

Produces two figures:
  F06_before.png — Raw MPB Bloch fields (random gauge, inconsistent norms)
  F06_after.png  — After normalization + SVD parallel-transport gauge
                   With Row 5: ε-weighted orthogonality (correct inner product)

Layout per figure: 4-5 rows × N_bands columns
  Row 1: ||u_n||²  heatmap (normalization)
  Row 2: arg⟨u_n(R)|u_n(R+δR)⟩ heatmap (overlap phase along s1)
  Row 3: arg⟨u_n(R)|u_n(R+δR)⟩ heatmap (overlap phase along s2)
  Row 4: |⟨u_m|u_n⟩|_flat heatmap (flat L2 orthogonality)
  Row 5: |⟨u_m|ε|u_n⟩| heatmap (ε-weighted orthogonality — correct for E-fields)
  Row 6: |⟨u_m|ε|u_n⟩|_SVQB heatmap (after SVQB B-orthonormalization)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'


def make_figure(data_path, out_path, title_prefix, before=True, eps_data_path=None, svqb_data_path=None):
    """Create the 4/5/6-row × N_bands diagnostic figure.
    
    If eps_data_path is provided (and file exists), adds Row 5 showing
    ε-weighted orthogonality |⟨u_m|ε|u_n⟩| — the correct inner product.
    If svqb_data_path is provided, adds Row 6 showing SVQB B-orthonormality.
    """
    d = np.load(data_path)

    N_sub = int(d['N_sub'])
    Ns1 = int(d['Ns1'])
    Ns2 = int(d['Ns2'])
    
    # Check if epsilon data is available
    has_eps = (eps_data_path is not None and os.path.exists(eps_data_path))
    if has_eps:
        eps_d = np.load(eps_data_path)
        ortho_eps = eps_d['ortho_eps_raw']     # (Ns1, Ns2, N_sub, N_sub)
        offdiag_eps = eps_d['offdiag_eps_raw']  # (Ns1, Ns2)
    
    # Check if SVQB data is available
    has_svqb = (svqb_data_path is not None and os.path.exists(svqb_data_path))
    if has_svqb:
        svqb_d = np.load(svqb_data_path)
        ortho_svqb = svqb_d['ortho_svqb']       # (Ns1, Ns2, N_sub, N_sub)
        offdiag_svqb = svqb_d['offdiag_svqb']   # (Ns1, Ns2)
        gram_eigenvalues = svqb_d['gram_eigenvalues']  # (Ns1, Ns2, N_sub)

    # Cell-averaged norm for display
    norm_data = d['norm_sq_avg']   # (Ns1, Ns2, N_sub)
    norm_label = r'$\langle u_n | u_n \rangle_\Omega$'

    phase_s1 = d['phase_s1']       # (Ns1, Ns2, N_sub)
    phase_s2 = d['phase_s2']       # (Ns1, Ns2, N_sub)
    ortho_data = d['offdiag_max']  # (Ns1, Ns2)
    ortho_full = d['ortho']        # (Ns1, Ns2, N_sub, N_sub)
    mag_s1 = d['mag_s1']           # (Ns1, Ns2, N_sub)
    mag_s2 = d['mag_s2']           # (Ns1, Ns2, N_sub)

    band_labels = [f'Band {n}' for n in range(N_sub)]
    
    n_rows = 4 + (1 if has_eps else 0) + (1 if has_svqb else 0)

    fig, axes = plt.subplots(n_rows, N_sub + 1, figsize=(4.5 * (N_sub + 1), 4 * n_rows),
                             gridspec_kw={'width_ratios': [1]*N_sub + [1]})

    # ---- Row 1: Normalization heatmaps ----
    if before:
        vmin_n = norm_data.min()
        vmax_n = norm_data.max()
    else:
        dev = max(abs(norm_data.max() - 1), abs(1 - norm_data.min()), 0.01)
        vmin_n = 1 - dev
        vmax_n = 1 + dev

    for n in range(N_sub):
        ax = axes[0, n]
        im = ax.imshow(norm_data[:, :, n].T, origin='lower',
                       cmap='viridis', vmin=vmin_n, vmax=vmax_n,
                       aspect='equal', interpolation='nearest')
        ax.set_title(f'{band_labels[n]}\n{norm_label}', fontsize=10)
        ax.set_xlabel('$s_1$ index')
        ax.set_ylabel('$s_2$ index')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Row 1, last column: statistics text
    ax = axes[0, N_sub]
    ax.axis('off')
    stats_text = f"Normalization Statistics\n{'─'*30}\n"
    for n in range(N_sub):
        nd = norm_data[:, :, n]
        stats_text += (f"\nBand {n}:\n"
                       f"  mean = {nd.mean():.6f}\n"
                       f"  std  = {nd.std():.2e}\n"
                       f"  range = [{nd.min():.4f}, {nd.max():.4f}]\n")
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ---- Row 2: Phase along s1 ----
    if before:
        vmin_p, vmax_p = -np.pi, np.pi
    else:
        # After fixing s1, use tighter range to show structure
        max_abs = max(np.abs(phase_s1).max(), 0.5)
        vmin_p, vmax_p = -max_abs, max_abs

    for n in range(N_sub):
        ax = axes[1, n]
        im = ax.imshow(phase_s1[:, :, n].T, origin='lower',
                       cmap='twilight', vmin=-np.pi, vmax=np.pi,
                       aspect='equal', interpolation='nearest')
        ax.set_title(r'$\arg\langle u_n(s)|u_n(s+\delta s_1)\rangle$', fontsize=10)
        ax.set_xlabel('$s_1$ index')
        ax.set_ylabel('$s_2$ index')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    ax = axes[1, N_sub]
    ax.axis('off')
    stats_text = f"Phase (s₁) Statistics\n{'─'*30}\n"
    for n in range(N_sub):
        ps = phase_s1[:, :, n].ravel()
        ms = mag_s1[:, :, n].ravel()
        stats_text += (f"\nBand {n}:\n"
                       f"  phase σ  = {np.std(ps):.4f} rad\n"
                       f"  |ov| min = {ms.min():.4f}\n"
                       f"  |ov| mean= {ms.mean():.4f}\n")
    expected = np.pi / np.sqrt(3)
    stats_text += f"\nUniform random: σ = π/√3 = {expected:.3f}"
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # ---- Row 3: Phase along s2 ----
    for n in range(N_sub):
        ax = axes[2, n]
        im = ax.imshow(phase_s2[:, :, n].T, origin='lower',
                       cmap='twilight', vmin=-np.pi, vmax=np.pi,
                       aspect='equal', interpolation='nearest')
        ax.set_title(r'$\arg\langle u_n(s)|u_n(s+\delta s_2)\rangle$', fontsize=10)
        ax.set_xlabel('$s_1$ index')
        ax.set_ylabel('$s_2$ index')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])

    ax = axes[2, N_sub]
    ax.axis('off')
    stats_text = f"Phase (s₂) Statistics\n{'─'*30}\n"
    for n in range(N_sub):
        ps = phase_s2[:, :, n].ravel()
        ms = mag_s2[:, :, n].ravel()
        stats_text += (f"\nBand {n}:\n"
                       f"  phase σ  = {np.std(ps):.4f} rad\n"
                       f"  |ov| min = {ms.min():.4f}\n"
                       f"  |ov| mean= {ms.mean():.4f}\n")
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # ---- Row 4: Orthogonality ----
    pairs = [(0, 1), (1, 2), (0, 2)]
    pair_labels = [r'$|\langle u_0|u_1\rangle|$',
                   r'$|\langle u_1|u_2\rangle|$',
                   r'$|\langle u_0|u_2\rangle|$']

    for idx, (m, n) in enumerate(pairs):
        ax = axes[3, idx]
        ov_mn = ortho_full[:, :, m, n]
        vmax_o = max(ov_mn.max(), 0.05)
        im = ax.imshow(ov_mn.T, origin='lower',
                       cmap='hot_r', vmin=0, vmax=vmax_o,
                       aspect='equal', interpolation='nearest')
        ax.set_title(pair_labels[idx], fontsize=10)
        ax.set_xlabel('$s_1$ index')
        ax.set_ylabel('$s_2$ index')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    ax = axes[3, N_sub]
    ax.axis('off')
    stats_text = f"Orthogonality Statistics\n{'─'*30}\n"
    for m_i, n_i in pairs:
        vals = ortho_full[:, :, m_i, n_i].ravel()
        stats_text += (f"\n|⟨u_{m_i}|u_{n_i}⟩|:\n"
                       f"  mean = {vals.mean():.4f}\n"
                       f"  max  = {vals.max():.4f}\n"
                       f"  frac(<0.01) = {float(np.mean(vals < 0.01)):.3f}\n")
    stats_text += (f"\nmax offdiag (any pair):\n"
                   f"  mean = {ortho_data.mean():.4f}\n"
                   f"  max  = {ortho_data.max():.4f}\n")
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # ---- Row 5: ε-weighted orthogonality (only if epsilon data available) ----
    if has_eps:
        pairs = [(0, 1), (1, 2), (0, 2)]
        pair_labels_eps = [r'$|\langle u_0|\varepsilon|u_1\rangle|$',
                          r'$|\langle u_1|\varepsilon|u_2\rangle|$',
                          r'$|\langle u_0|\varepsilon|u_2\rangle|$']

        # Use same colorscale as Row 4's flat orthogonality for comparison
        # (but ε-weighted values should be ~100× smaller)
        for idx, (m, n) in enumerate(pairs):
            ax = axes[4, idx]
            ov_mn = ortho_eps[:, :, m, n]
            vmax_oe = max(ov_mn.max(), 0.005)
            im = ax.imshow(ov_mn.T, origin='lower',
                           cmap='hot_r', vmin=0, vmax=vmax_oe,
                           aspect='equal', interpolation='nearest')
            ax.set_title(pair_labels_eps[idx], fontsize=10)
            ax.set_xlabel('$s_1$ index')
            ax.set_ylabel('$s_2$ index')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        ax = axes[4, N_sub]
        ax.axis('off')
        stats_text = f"ε-weighted Orthogonality\n{'─'*30}\n"
        stats_text += "(correct inner product for\n E-field Bloch functions)\n"
        for m_i, n_i in pairs:
            vals = ortho_eps[:, :, m_i, n_i].ravel()
            flat_vals = ortho_full[:, :, m_i, n_i].ravel()
            stats_text += (f"\n|⟨u_{m_i}|ε|u_{n_i}⟩|:\n"
                           f"  mean = {vals.mean():.6f}\n"
                           f"  max  = {vals.max():.6f}\n"
                           f"  (flat: {flat_vals.mean():.4f} → {vals.mean():.6f})\n")
        stats_text += (f"\nmax offdiag (εw, any pair):\n"
                       f"  mean = {offdiag_eps.mean():.6f}\n"
                       f"  max  = {offdiag_eps.max():.6f}\n"
                       f"  (flat: {ortho_data.mean():.4f})\n")
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    # ---- Row 6: SVQB B-orthonormality (only if SVQB data available) ----
    if has_svqb:
        svqb_row = 4 + (1 if has_eps else 0)  # row index depends on whether eps row exists
        pairs = [(0, 1), (1, 2), (0, 2)]
        pair_labels_svqb = [r'$|\langle u_0|\varepsilon|u_1\rangle|_{\mathrm{SVQB}}$',
                            r'$|\langle u_1|\varepsilon|u_2\rangle|_{\mathrm{SVQB}}$',
                            r'$|\langle u_0|\varepsilon|u_2\rangle|_{\mathrm{SVQB}}$']

        for idx, (m, n) in enumerate(pairs):
            ax = axes[svqb_row, idx]
            ov_mn = ortho_svqb[:, :, m, n]
            vmax_os = max(ov_mn.max(), 1e-14)
            im = ax.imshow(ov_mn.T, origin='lower',
                           cmap='hot_r', vmin=0, vmax=vmax_os,
                           aspect='equal', interpolation='nearest')
            ax.set_title(pair_labels_svqb[idx], fontsize=10)
            ax.set_xlabel('$s_1$ index')
            ax.set_ylabel('$s_2$ index')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax, format='%.0e')

        ax = axes[svqb_row, N_sub]
        ax.axis('off')
        stats_text = f"SVQB B-Orthonormality\n{'─'*30}\n"
        stats_text += "(after eigendecomp-based\n B-orthonormalization)\n"
        for m_i, n_i in pairs:
            vals = ortho_svqb[:, :, m_i, n_i].ravel()
            stats_text += (f"\n|⟨u_{m_i}|ε|u_{n_i}⟩|:\n"
                           f"  mean = {vals.mean():.2e}\n"
                           f"  max  = {vals.max():.2e}\n")
        stats_text += (f"\nmax offdiag (any pair):\n"
                       f"  mean = {offdiag_svqb.mean():.2e}\n"
                       f"  max  = {offdiag_svqb.max():.2e}\n")
        # Add Gram condition number
        if gram_eigenvalues.shape[2] >= 2:
            cond = gram_eigenvalues[:,:,0] / np.maximum(gram_eigenvalues[:,:,-1], 1e-30)
            stats_text += (f"\nGram condition κ:\n"
                           f"  mean = {cond.mean():.2f}\n"
                           f"  max  = {cond.max():.2f}\n")
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("F06 — Generating plots...")

    make_figure(
        f'{FINDINGS}/F06_before_data.npz',
        f'{FINDINGS}/F06_before.png',
        'F06: Raw MPB Bloch Fields — Before Gauge Fixing',
        before=True,
    )

    make_figure(
        f'{FINDINGS}/F06_after_data.npz',
        f'{FINDINGS}/F06_after.png',
        'F06: Fixed Bloch Fields — After Normalization + Abelian Gauge + SVQB',
        before=False,
        eps_data_path=f'{FINDINGS}/F06_epsilon_data.npz',
        svqb_data_path=f'{FINDINGS}/F06_svqb_data.npz',
    )

    print("\nDone. Plots saved to findings/F06_before.png and F06_after.png")


if __name__ == '__main__':
    main()
