"""
T03: Miniband Dispersion — Thesis Figure

Generates the η-sweep results showing:
  - Envelope eigenvalues E_n(θ) for n=0..5 vs twist angle
  - Bandwidth (E_1 - E_0) vs θ
  - Flat-band ratio (BW / gap) vs θ
  - Comparison across 3 candidates

Requires: η-sweep data from the pipeline.

Usage:
    python thesis_results/T03_miniband_dispersion/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names, find_thesis_run_dir, load_eta_sweep_data,
    find_candidate_dir,
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T03_miniband_dispersion"


def load_all_sweeps():
    """Load η-sweep data for all candidates."""
    sweeps = {}
    for name in get_candidate_names():
        try:
            cand_dir = find_candidate_dir(name)
            data = load_eta_sweep_data(cand_dir)
            if data:
                sweeps[name] = data
                print(f"  {name}: {len(data)} angles")
            else:
                print(f"  {name}: no η-sweep data found")
        except FileNotFoundError as e:
            print(f"  {name}: {e}")
    return sweeps


def plot_miniband_dispersion(sweeps):
    """3-panel figure: (a) E_n vs θ, (b) bandwidth, (c) flat-band ratio."""
    apply_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel (a): E_n(θ) for each candidate ---
    ax = axes[0]
    for name, data in sweeps.items():
        thetas = [d['theta_deg'] for d in data]
        for i, d in enumerate(data):
            evals = d.get('eigenvalues')
            if evals is None:
                continue
            n_show = min(6, len(evals))
            for n in range(n_show):
                ax.plot(thetas[i], evals[n], 'o',
                        color=CANDIDATE_COLORS[name],
                        markersize=3, alpha=0.7)

        # Connect with lines for each mode
        if data:
            n_modes = min(6, len(data[0].get('eigenvalues', [])))
            for n in range(n_modes):
                vals = []
                ts = []
                for d in data:
                    evals = d.get('eigenvalues')
                    if evals is not None and n < len(evals):
                        vals.append(evals[n])
                        ts.append(d['theta_deg'])
                if vals:
                    label = CANDIDATE_LABELS[name] if n == 0 else None
                    ax.plot(ts, vals, '-', color=CANDIDATE_COLORS[name],
                            alpha=0.5, linewidth=1, label=label)

    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'$E_n$ [$\omega \cdot a / 2\pi c$]')
    ax.set_title('(a) Miniband dispersion')
    ax.legend(fontsize=8)

    # --- Panel (b): Bandwidth E_1 - E_0 vs θ ---
    ax = axes[1]
    for name, data in sweeps.items():
        bws = []
        thetas = []
        for d in data:
            evals = d.get('eigenvalues')
            if evals is not None and len(evals) >= 2:
                bws.append(evals[1] - evals[0])
                thetas.append(d['theta_deg'])
        if bws:
            ax.semilogy(thetas, bws, '-o', color=CANDIDATE_COLORS[name],
                        marker=CANDIDATE_MARKERS[name],
                        label=CANDIDATE_LABELS[name], markersize=6)

    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'Bandwidth $E_1 - E_0$')
    ax.set_title('(b) Ground-state bandwidth')
    ax.legend(fontsize=8)

    # --- Panel (c): Gap ratio ---
    ax = axes[2]
    for name, data in sweeps.items():
        ratios = []
        thetas = []
        for d in data:
            evals = d.get('eigenvalues')
            if evals is not None and len(evals) >= 3:
                bw = evals[1] - evals[0]
                gap = evals[2] - evals[1]
                if gap > 0:
                    ratios.append(bw / gap)
                    thetas.append(d['theta_deg'])
        if ratios:
            ax.semilogy(thetas, ratios, '-o', color=CANDIDATE_COLORS[name],
                        marker=CANDIDATE_MARKERS[name],
                        label=CANDIDATE_LABELS[name], markersize=6)

    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'Flat-band ratio $\Delta E / \mathrm{gap}$')
    ax.axhline(y=1, color='red', ls='--', alpha=0.3, label='BW = gap')
    ax.set_title('(c) Flat-band quality')
    ax.legend(fontsize=8)

    fig.suptitle('Miniband Dispersion vs Twist Angle',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T03: Miniband Dispersion → {out_dir}")

    sweeps = load_all_sweeps()
    if not sweeps:
        print("  No η-sweep data available. Run pipeline first.")
        return

    fig = plot_miniband_dispersion(sweeps)
    save_figure(fig, TASK, "T03_miniband_dispersion")

    print("  T03 complete.")


if __name__ == "__main__":
    main()
