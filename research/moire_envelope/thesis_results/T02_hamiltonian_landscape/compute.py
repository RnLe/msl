"""
T02: Hamiltonian Landscape — Thesis Figure

Generates spatial maps of all Hamiltonian ingredients at default θ:
  - ω(s) / V(s) potential landscape (diagonal Λ)
  - Berry connection |A(s)| magnitude (diagonal + off-diagonal)  
  - Mass tensor M⁻¹(s) eigenvalues
  - Off-diagonal Λ coupling terms

3 columns (one per candidate) × 4 rows (Λ, |A|, M⁻¹, off-diag)

Usage:
    python thesis_results/T02_hamiltonian_landscape/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names, find_candidate_dir, load_phase2_data,
    load_phase0_meta,
    CANDIDATE_COLORS, CANDIDATE_LABELS,
)

TASK = "T02_hamiltonian_landscape"


def plot_hamiltonian_maps():
    """4-row × 3-col figure of Hamiltonian ingredients."""
    apply_thesis_style()
    names = get_candidate_names()
    n_cands = len(names)

    fig, axes = plt.subplots(4, n_cands, figsize=(4.5 * n_cands, 16))
    if n_cands == 1:
        axes = axes[:, np.newaxis]

    row_labels = [
        r'$V_{nn}(\mathbf{s}) = \omega_n(\mathbf{s}) - \omega_{\rm ref}$',
        r'$|\mathbf{A}_{nn}(\mathbf{s})|$ (Berry connection)',
        r'$\mathrm{tr}\,M^{-1}_{nn}(\mathbf{s})$',
        r'$|A_{01}(\mathbf{s})|$ (off-diagonal)',
    ]

    for col, name in enumerate(names):
        try:
            cand_dir = find_candidate_dir(name)
            data = load_phase2_data(cand_dir)
            meta = load_phase0_meta(cand_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            for row in range(4):
                axes[row, col].text(0.5, 0.5, f'No data\n{name}',
                                    transform=axes[row, col].transAxes,
                                    ha='center', va='center', fontsize=12)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        Lambda = data['Lambda']    # (Ns, Ns, Nb, Nb)
        A_berry = data['A_berry']  # (Ns, Ns, Nb, Nb, 2) complex
        M_inv = data['M_inv']      # (Ns, Ns, Nb, Nb, 2, 2)
        omega = data['omega']      # (Ns, Ns, Nb)
        target_idx = int(data.get('attr_target_index_in_subspace', 0))
        omega_ref = float(data.get('attr_omega_ref', 0))
        Ns = Lambda.shape[0]

        # Row 0: Potential V = Λ_{target,target}
        ax = axes[0, col]
        V = Lambda[:, :, target_idx, target_idx]
        im = ax.imshow(V.T, origin='lower', cmap='viridis',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$\omega \cdot a / (2\pi c)$')
        if col == 0:
            ax.set_ylabel(row_labels[0])

        # Row 1: |A_{nn}| diagonal Berry connection
        ax = axes[1, col]
        A_diag = A_berry[:, :, target_idx, target_idx, :]  # (Ns,Ns,2)
        A_mag = np.sqrt(np.abs(A_diag[:, :, 0])**2 + np.abs(A_diag[:, :, 1])**2)
        im = ax.imshow(A_mag.T, origin='lower', cmap='hot',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$|A|$')
        if col == 0:
            ax.set_ylabel(row_labels[1])

        # Row 2: tr(M⁻¹) for target band
        ax = axes[2, col]
        M_diag = M_inv[:, :, target_idx, target_idx, :, :]  # (Ns,Ns,2,2)
        M_trace = M_diag[:, :, 0, 0] + M_diag[:, :, 1, 1]
        vmax = np.percentile(np.abs(M_trace), 95)
        im = ax.imshow(M_trace.T, origin='lower', cmap='RdBu_r',
                       extent=[0, 1, 0, 1], vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$\mathrm{tr}\,M^{-1}$')
        if col == 0:
            ax.set_ylabel(row_labels[2])

        # Row 3: Off-diagonal |A_{0,1}|
        ax = axes[3, col]
        Nb = A_berry.shape[2]
        if Nb > 1:
            # Find the largest off-diagonal pair
            offdiag_idx = (target_idx, (target_idx + 1) % Nb)
            A_off = A_berry[:, :, offdiag_idx[0], offdiag_idx[1], :]
            A_off_mag = np.sqrt(np.abs(A_off[:, :, 0])**2 + np.abs(A_off[:, :, 1])**2)
            im = ax.imshow(A_off_mag.T, origin='lower', cmap='inferno',
                           extent=[0, 1, 0, 1])
            plt.colorbar(im, ax=ax, shrink=0.8, label=r'$|A_{mn}|$')
        if col == 0:
            ax.set_ylabel(row_labels[3])

        # Column title
        axes[0, col].set_title(CANDIDATE_LABELS[name], fontsize=12,
                                fontweight='bold',
                                color=CANDIDATE_COLORS[name])

        # Axis labels
        for row in range(4):
            axes[row, col].set_xlabel(r'$s_1$')
            if col == 0:
                pass  # ylabel already set
            axes[row, col].set_xticks([0, 0.5, 1])
            axes[row, col].set_yticks([0, 0.5, 1])

    fig.suptitle('Hamiltonian Landscape: Operator Fields over Configuration Space',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T02: Hamiltonian Landscape → {out_dir}")

    fig = plot_hamiltonian_maps()
    save_figure(fig, TASK, "T02_hamiltonian_landscape")

    print("  T02 complete.")


if __name__ == "__main__":
    main()
