#!/usr/bin/env python3
"""
Example: Running Phase 1 Pipeline

This script demonstrates how to run Phase 1 on the output of Phase 0.

Usage:
    mamba run -n msl python run_phase1_example.py <run_dir>

Example:
    mamba run -n msl python run_phase1_example.py runs/phase0_real_run_20241113_120000
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phases.phase1_local_bloch import run_phase1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable run directories:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "phase0_candidates.csv").exists():
                    print(f"  {run_dir}")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    # Check if run directory exists
    if not Path(run_dir).exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Check if phase0_candidates.csv exists
    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    if not candidates_path.exists():
        print(f"Error: Phase 0 candidates not found: {candidates_path}")
        print("Please run Phase 0 first to generate candidates.")
        sys.exit(1)
    
    # Use quick test config by default
    config_path = "configs/phase1_quick_test.yaml"
    
    if len(sys.argv) >= 3:
        config_path = sys.argv[2]
    
    print("="*70)
    print("Phase 1 Example Run")
    print("="*70)
    print(f"Run directory: {run_dir}")
    print(f"Configuration: {config_path}")
    print()
    
    # Run Phase 1
    run_phase1(run_dir, config_path)
    
    print("\n" + "="*70)
    print("Phase 1 Complete!")
    print("="*70)
    print(f"\nResults saved in candidate directories under: {run_dir}")
    print("\nFor each candidate, you should find:")
    print("  - phase1_band_data.h5: HDF5 file with all field data")
    print("  - phase1_fields_visualization.png: Visualization of V(R), v_g(R), M⁻¹(R)")
    print("  - phase0_meta.json: Candidate metadata")
    print("\nNext step: Run Phase 2 to assemble EA operators")


if __name__ == "__main__":
    main()
