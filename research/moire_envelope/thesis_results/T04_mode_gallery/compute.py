"""
T04: Mode Gallery — Thesis Figure

Generates envelope mode |F_n(s)|² plots for n=0..5 at selected twist angles.
Shows how the modes evolve from extended (large θ) to localized (small θ).

Layout: 3 columns (candidates) × 6 rows (modes n=0..5)
at a single representative θ near θ*.

Usage:
    python thesis_results/T04_mode_gallery/compute.py
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
    load_phase3_data, load_phase0_meta,
    CANDIDATE_COLORS, CANDIDATE_LABELS,
)

TASK = "T04_mode_gallery"
N_MODES_SHOW = 6


def compute_mode_density(modes, Ns, Nb, mode_idx):
    """
    Compute |F_n(s)|² from envelope mode vector.

    modes: shape (N_total,) or (N_total, n_modes)
    Returns: (Ns, Ns) density summed over bands
    """
    if modes.ndim == 1:
        vec = modes
    else:
        vec = modes[:, mode_idx]

    # Reshape: (Ns*Ns*Nb,) → (Ns, Ns, Nb)
    N_total = len(vec)
    if N_total == Ns * Ns * Nb:
        F = vec.reshape(Ns, Ns, Nb)
    else:
        # Try to infer
        Nb_infer = N_total // (Ns * Ns)
        F = vec.reshape(Ns, Ns, Nb_infer)

    # Sum |F_n|² over bands
    density = np.sum(np.abs(F)**2, axis=-1)  # (Ns, Ns)
    return density


def plot_mode_gallery():
    """Mode gallery: candidates × modes."""
    apply_thesis_style()
    names = get_candidate_names()
    n_cands = len(names)

    fig, axes = plt.subplots(N_MODES_SHOW, n_cands,
                              figsize=(4 * n_cands, 3 * N_MODES_SHOW))
    if n_cands == 1:
        axes = axes[:, np.newaxis]

    for col, name in enumerate(names):
        try:
            cand_dir = find_candidate_dir(name)
            data = load_phase3_data(cand_dir)
            meta = load_phase0_meta(cand_dir)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")
            for row in range(N_MODES_SHOW):
                axes[row, col].text(0.5, 0.5, f'No data',
                                     transform=axes[row, col].transAxes,
                                     ha='center', va='center')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        # Extract modes
        eigenvalues = data.get('eigenvalues', np.array([]))
        modes = data.get('envelope_modes', None)
        Ns = int(data.get('attr_Ns1', data.get('attr_Ns', 32)))
        Nb = int(data.get('attr_N_subspace', data.get('attr_Nb', 5)))

        if modes is None:
            print(f"  {name}: no envelope modes in h5")
            continue

        n_avail = min(N_MODES_SHOW, modes.shape[1] if modes.ndim > 1 else 1)

        for row in range(N_MODES_SHOW):
            ax = axes[row, col]
            if row >= n_avail:
                ax.set_visible(False)
                continue

            density = compute_mode_density(modes, Ns, Nb, row)
            im = ax.imshow(density.T, origin='lower', cmap='magma',
                           extent=[0, 1, 0, 1])
            plt.colorbar(im, ax=ax, shrink=0.7)

            E_val = eigenvalues[row] if row < len(eigenvalues) else 0
            ax.set_title(f'n={row}, E={E_val:.6f}', fontsize=9)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])

            if row == N_MODES_SHOW - 1:
                ax.set_xlabel(r'$s_1$')
            if col == 0:
                ax.set_ylabel(r'$s_2$')

        # Column header
        theta = meta.get('theta_deg', '?')
        axes[0, col].set_title(
            f'{CANDIDATE_LABELS[name]}\nθ={theta}°, n=0',
            fontsize=10, fontweight='bold',
            color=CANDIDATE_COLORS[name])

    fig.suptitle(r'Envelope Mode Gallery $|F_n(\mathbf{s})|^2$',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T04: Mode Gallery → {out_dir}")

    fig = plot_mode_gallery()
    save_figure(fig, TASK, "T04_mode_gallery")

    print("  T04 complete.")


if __name__ == "__main__":
    main()
