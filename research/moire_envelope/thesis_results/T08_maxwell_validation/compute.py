"""
T08: Maxwell Validation — Thesis Figure (MOST IMPORTANT)

Compares envelope approximation eigenvalues with reference FDTD simulations.
This is the central validation figure for the thesis.

Shows:
  - EA eigenvalues vs FDTD eigenvalues (parity plot)
  - Relative error |ω_EA - ω_FDTD| / ω_FDTD vs mode index
  - Error dependence on θ (should improve for smaller θ)

Requires: FDTD data from Phase 6 (Meep validation) or Phase 5.

Usage:
    python thesis_results/T08_maxwell_validation/compute.py
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
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T08_maxwell_validation"


def find_fdtd_data(cand_dir: Path):
    """Look for Meep validation data (Phase 6/5)."""
    # Check for phase6 validation directories
    phase6_dirs = sorted(cand_dir.glob("phase6_val_*"))
    if phase6_dirs:
        return 'phase6', phase6_dirs

    # Check for phase5 data
    phase5_h5 = cand_dir / "phase5_meep_data.h5"
    if phase5_h5.exists():
        return 'phase5', [phase5_h5]

    return None, []


def plot_validation():
    """Validation parity plot and error analysis."""
    apply_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    has_data = False
    for name in get_candidate_names():
        try:
            cand_dir = find_candidate_dir(name)
            data_type, data_files = find_fdtd_data(cand_dir)

            if data_type is None:
                print(f"  {name}: no FDTD data found")
                continue

            has_data = True
            ea_data = load_phase3_data(cand_dir)
            ea_evals = ea_data.get('eigenvalues', np.array([]))

            print(f"  {name}: found {data_type} data ({len(data_files)} files)")

            # Load FDTD eigenvalues
            if data_type == 'phase6':
                fdtd_evals = []
                for dd in data_files:
                    json_path = dd / "validation_result.json"
                    if json_path.exists():
                        import json
                        with open(json_path) as f:
                            result = json.load(f)
                        fdtd_evals.append(result.get('fdtd_frequency', 0))

                if fdtd_evals and len(ea_evals) > 0:
                    n_compare = min(len(fdtd_evals), len(ea_evals))
                    ea_sub = ea_evals[:n_compare]
                    fdtd_sub = np.array(fdtd_evals[:n_compare])

                    # (a) Parity plot
                    ax = axes[0]
                    ax.scatter(fdtd_sub, ea_sub, c=CANDIDATE_COLORS[name],
                               s=50, marker=CANDIDATE_MARKERS[name],
                               label=CANDIDATE_LABELS[name])

                    # (b) Relative error
                    ax = axes[1]
                    rel_err = np.abs(ea_sub - fdtd_sub) / (fdtd_sub + 1e-30)
                    ax.plot(range(n_compare), rel_err, '-o',
                            color=CANDIDATE_COLORS[name],
                            marker=CANDIDATE_MARKERS[name],
                            label=CANDIDATE_LABELS[name])

        except FileNotFoundError as e:
            print(f"  {name}: {e}")

    if not has_data:
        print("  No FDTD validation data available.")
        print("  Run Meep validation (Phase 5/6) first.")
        plt.close(fig)
        return None

    # Finalize plots
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
    axes[0].set_xlabel(r'$\omega_{\rm FDTD}$')
    axes[0].set_ylabel(r'$\omega_{\rm EA}$')
    axes[0].set_title('(a) EA vs FDTD parity')
    axes[0].legend(fontsize=8)
    axes[0].set_aspect('equal')

    axes[1].set_xlabel('Mode index')
    axes[1].set_ylabel(r'$|\omega_{\rm EA} - \omega_{\rm FDTD}|/\omega_{\rm FDTD}$')
    axes[1].set_title('(b) Relative error by mode')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)

    axes[2].text(0.5, 0.5, 'Error vs θ\n(requires η-sweep + FDTD)',
                 transform=axes[2].transAxes, ha='center', va='center',
                 fontsize=12)
    axes[2].set_title('(c) Error convergence with θ')

    fig.suptitle('Maxwell Validation: Envelope Approximation vs FDTD',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T08: Maxwell Validation → {out_dir}")

    fig = plot_validation()
    if fig is not None:
        save_figure(fig, TASK, "T08_maxwell_validation")

    print("  T08 complete.")


if __name__ == "__main__":
    main()
