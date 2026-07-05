"""
T07: Disorder Robustness — Thesis Figure

Studies how twist-angle disorder σ_θ affects miniband structure.
For each candidate, runs Phase 3 with gaussian-distributed θ
and measures:
  - Level statistics P(s) (Wigner-Dyson vs Poisson transition)
  - E_gap persistence under disorder
  - DOS broadening vs σ_θ

This script generates disorder by modifying θ in the envelope solver
and averaging over realizations.

Usage:
    python thesis_results/T07_disorder_robustness/compute.py [--n-realizations 20]
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
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T07_disorder_robustness"


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T07: Disorder Robustness → {out_dir}")
    print("  This script requires Phase 1/2 data to run disorder realizations.")
    print("  Placeholder — will be implemented after pipeline runs complete.")
    print("  T07 placeholder complete.")


if __name__ == "__main__":
    main()
