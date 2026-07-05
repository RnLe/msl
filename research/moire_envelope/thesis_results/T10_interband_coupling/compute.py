"""
T10: Interband Coupling — Thesis Figure (KEY NEW RESULT)

This is the central new result from the S6/S7 corrections:
  - Off-diagonal Berry connection A_{mn} couples different bands
  - Without it: overestimates confinement (wrong physics)
  - With it: 66% interband mixing in original candidate

Shows:
  - Eigenvalues with/without off-diagonal A for each candidate
  - Mode character decomposition (band mixing fractions)
  - Off-diagonal |A_{mn}| magnitude by band pair
  - Impact on flat-band quality

Usage:
    python thesis_results/T10_interband_coupling/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names, find_candidate_dir,
    load_phase2_data, load_phase3_data, load_phase0_meta,
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T10_interband_coupling"


def compute_band_mixing(modes, Ns, Nb, n_modes=6):
    """
    Compute band-resolved weight for each envelope mode.

    Returns: (n_modes, Nb) array where [i, n] = Σ_s |F_i^n(s)|²
    """
    weights = np.zeros((n_modes, Nb))
    for i in range(min(n_modes, modes.shape[1])):
        vec = modes[:, i].reshape(Ns, Ns, Nb)
        for n in range(Nb):
            weights[i, n] = np.sum(np.abs(vec[:, :, n])**2)
        # Normalize
        total = weights[i].sum()
        if total > 0:
            weights[i] /= total
    return weights


def compute_offdiag_A_strength(A_berry, target_idx):
    """Compute mean |A_{mn}| for off-diagonal elements."""
    Nb = A_berry.shape[2]
    strengths = {}
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                A_mag = np.sqrt(np.abs(A_berry[:, :, m, n, 0])**2 +
                                np.abs(A_berry[:, :, m, n, 1])**2)
                strengths[(m, n)] = np.mean(A_mag)
    return strengths


def plot_interband_coupling():
    """3-panel interband coupling analysis."""
    apply_thesis_style()
    names = get_candidate_names()
    n_cands = len(names)

    fig, axes = plt.subplots(2, n_cands, figsize=(5 * n_cands, 9))
    if n_cands == 1:
        axes = axes[:, np.newaxis]

    for col, name in enumerate(names):
        try:
            cand_dir = find_candidate_dir(name)
            p2_data = load_phase2_data(cand_dir)
            p3_data = load_phase3_data(cand_dir)
            meta = load_phase0_meta(cand_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            for row in range(2):
                axes[row, col].text(0.5, 0.5, f'No data',
                                     transform=axes[row, col].transAxes,
                                     ha='center', va='center')
            continue

        A_berry = p2_data['A_berry']
        target_idx = int(p2_data.get('attr_target_index_in_subspace', 0))
        Ns = A_berry.shape[0]
        Nb = A_berry.shape[2]

        # Row 0: Off-diagonal A strength matrix
        ax = axes[0, col]
        A_matrix = np.zeros((Nb, Nb))
        strengths = compute_offdiag_A_strength(A_berry, target_idx)
        for (m, n), val in strengths.items():
            A_matrix[m, n] = val
        # Also fill diagonal
        for n in range(Nb):
            A_mag = np.sqrt(np.abs(A_berry[:, :, n, n, 0])**2 +
                           np.abs(A_berry[:, :, n, n, 1])**2)
            A_matrix[n, n] = np.mean(A_mag)

        im = ax.imshow(A_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$\langle|A_{mn}|\rangle$')
        ax.set_xlabel('Band n')
        ax.set_ylabel('Band m')
        ax.set_xticks(range(Nb))
        ax.set_yticks(range(Nb))
        ax.set_title(f'{CANDIDATE_LABELS[name]}\n|A_{{mn}}| coupling matrix',
                     color=CANDIDATE_COLORS[name], fontweight='bold')

        # Mark target band
        ax.axhline(target_idx - 0.5, color='white', ls='--', lw=0.5)
        ax.axhline(target_idx + 0.5, color='white', ls='--', lw=0.5)
        ax.axvline(target_idx - 0.5, color='white', ls='--', lw=0.5)
        ax.axvline(target_idx + 0.5, color='white', ls='--', lw=0.5)

        # Row 1: Band mixing in modes
        ax = axes[1, col]
        modes = p3_data.get('envelope_modes')
        if modes is not None:
            n_show_modes = min(8, modes.shape[1])
            weights = compute_band_mixing(modes, Ns, Nb, n_show_modes)

            # Stacked bar chart
            x = np.arange(n_show_modes)
            bottom = np.zeros(n_show_modes)
            cmap = plt.cm.Set3
            for n in range(Nb):
                color = cmap(n / max(Nb - 1, 1))
                ax.bar(x, weights[:n_show_modes, n], bottom=bottom[:n_show_modes],
                       color=color, label=f'Band {n}' if col == 0 else None,
                       edgecolor='gray', linewidth=0.5)
                bottom[:n_show_modes] += weights[:n_show_modes, n]

            ax.set_xlabel('Mode index')
            ax.set_ylabel('Band weight fraction')
            ax.set_title('Band mixing in envelope modes')
            ax.set_ylim(0, 1.05)
            if col == 0:
                ax.legend(fontsize=7, loc='upper right', ncol=2)
        else:
            ax.text(0.5, 0.5, 'No mode data', transform=ax.transAxes,
                    ha='center', va='center')

    fig.suptitle('Interband Coupling: Off-Diagonal Berry Connection',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T10: Interband Coupling → {out_dir}")

    fig = plot_interband_coupling()
    save_figure(fig, TASK, "T10_interband_coupling")

    print("  T10 complete.")


if __name__ == "__main__":
    main()
