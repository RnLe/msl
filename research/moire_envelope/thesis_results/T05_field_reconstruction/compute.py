"""
T05: Field Reconstruction — Thesis Figure

Reconstructs full electromagnetic field E(r) from envelope × Bloch:
  E(r) ≈ Σ_n F_n(r_slow) u_n(r_fast; r_slow) exp(ik₀·r)

Shows:
  - Moiré-scale pattern (slow modulation)
  - Zoom into unit cells (fast oscillation)
  - Comparison with direct FDTD (if available from T08)

Requires: Phase 1 Bloch fields + Phase 3 envelope modes.

Usage:
    python thesis_results/T05_field_reconstruction/compute.py
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

TASK = "T05_field_reconstruction"


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T05: Field Reconstruction → {out_dir}")

    names = get_candidate_names()
    for name in names:
        try:
            cand_dir = find_candidate_dir(name)
            meta = load_phase0_meta(cand_dir)
            print(f"  {name}: candidate dir found at {cand_dir}")

            # Check for Bloch field data
            bloch_h5 = cand_dir / "phase1_multiband_data.h5"
            if bloch_h5.exists():
                import h5py
                with h5py.File(bloch_h5, 'r') as hf:
                    keys = list(hf.keys())
                    has_fields = 'bloch_fields' in keys or 'eigvecs_flat' in keys
                    print(f"    Phase 1 keys: {keys[:10]}...")
                    print(f"    Has Bloch fields: {has_fields}")
            else:
                print(f"    Phase 1 data not found")

            # Phase 4 reconstruction uses phasesV3/phase4_field_reconstruction.py
            # We'll call it when data is available
        except FileNotFoundError as e:
            print(f"  {name}: {e}")

    print("\n  T05: Field reconstruction requires Phase 1 Bloch fields.")
    print("  Will generate plots after pipeline completes.")
    print("  T05 placeholder complete.")


if __name__ == "__main__":
    main()
