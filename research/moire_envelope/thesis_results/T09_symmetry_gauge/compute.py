"""
T09: Symmetry & Gauge Quality — Thesis Figure

Diagnostic figure showing:
  - [H, C4/C2] commutator norm before/after symmetrization
  - Berry connection gauge smoothness (|∇ × A| deviation)
  - Eigenvalue splitting under symmetry operations
  - Comparison of C4 (square) vs C2 (hex M-point) effectiveness

Usage:
    python thesis_results/T09_symmetry_gauge/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names, get_candidate, find_candidate_dir,
    load_phase2_data,
    CANDIDATE_COLORS, CANDIDATE_LABELS,
)
from symmetrize import (
    measure_error_scalar, measure_error_vector, measure_error_2tensor,
)

TASK = "T09_symmetry_gauge"


def plot_symmetry_diagnostics():
    """Bar chart of symmetry errors before/after symmetrization."""
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = get_candidate_names()
    errors_before = {}
    errors_after = {}

    for name in names:
        try:
            cand_dir = find_candidate_dir(name)
            cand = get_candidate(name)
            lattice = cand['lattice_type']
            sym_type = 'C4' if lattice == 'square' else 'C2'

            # Load unsymmetrized data if available
            unsym_h5 = cand_dir / "phase2_multiband_data_unsym.h5"
            sym_h5 = cand_dir / "phase2_multiband_data.h5"

            # Check for both files
            if unsym_h5.exists():
                import h5py
                with h5py.File(unsym_h5, 'r') as hf:
                    Lambda_orig = hf['Lambda'][:]
                    A_orig = hf['A_berry'][:]
                    M_orig = hf['M_inv'][:]
                    Ns = int(hf.attrs['Ns1'])

                err_L = measure_error_scalar(Lambda_orig, Ns, sym_type)
                err_A = measure_error_vector(A_orig, Ns, sym_type)
                err_M = measure_error_2tensor(M_orig, Ns, sym_type)
                errors_before[name] = {'Λ': err_L, 'A': err_A, 'M⁻¹': err_M}
                print(f"  {name} ({sym_type}) before: Λ={err_L:.2e}, A={err_A:.2e}, M={err_M:.2e}")

            if sym_h5.exists():
                import h5py
                with h5py.File(sym_h5, 'r') as hf:
                    Lambda_sym = hf['Lambda'][:]
                    A_sym = hf['A_berry'][:]
                    M_sym = hf['M_inv'][:]
                    Ns = int(hf.attrs['Ns1'])

                err_L = measure_error_scalar(Lambda_sym, Ns, sym_type)
                err_A = measure_error_vector(A_sym, Ns, sym_type)
                err_M = measure_error_2tensor(M_sym, Ns, sym_type)
                errors_after[name] = {'Λ': err_L, 'A': err_A, 'M⁻¹': err_M}
                print(f"  {name} ({sym_type}) after:  Λ={err_L:.2e}, A={err_A:.2e}, M={err_M:.2e}")

        except FileNotFoundError as e:
            print(f"  {name}: {e}")

    if not errors_before and not errors_after:
        print("  No data available for symmetry diagnostics.")
        plt.close(fig)
        return None

    # Plot grouped bar chart
    ax = axes[0]
    fields = ['Λ', 'A', 'M⁻¹']
    x = np.arange(len(fields))
    width = 0.25

    for i, name in enumerate(names):
        if name in errors_before:
            vals = [errors_before[name].get(f, 0) for f in fields]
            ax.bar(x + i * width - width, vals, width, alpha=0.5,
                   color=CANDIDATE_COLORS[name], label=f'{CANDIDATE_LABELS[name]} (before)')
        if name in errors_after:
            vals = [errors_after[name].get(f, 0) for f in fields]
            ax.bar(x + i * width - width, vals, width,
                   color=CANDIDATE_COLORS[name], edgecolor='black',
                   label=f'{CANDIDATE_LABELS[name]} (after)')

    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel('Relative symmetry error')
    ax.set_yscale('log')
    ax.set_title('(a) Symmetry errors by field')
    ax.legend(fontsize=7)

    # Panel (b): Berry curvature / gauge smoothness
    ax = axes[1]
    ax.text(0.5, 0.5, 'Berry curvature\n∇×A diagnostics\n(implemented after data available)',
            transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_title('(b) Gauge smoothness')

    fig.suptitle('Symmetry & Gauge Quality Diagnostics',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T09: Symmetry & Gauge → {out_dir}")

    fig = plot_symmetry_diagnostics()
    if fig is not None:
        save_figure(fig, TASK, "T09_symmetry_gauge")

    print("  T09 complete.")


if __name__ == "__main__":
    main()
