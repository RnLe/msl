#!/usr/bin/env python
"""
Watch script: monitors honeycomb η-sweep and runs analysis upon completion.

Usage:
    python thesis_results/T_honeycomb_analysis/watch_and_analyze.py
"""
import sys, time, json, subprocess
from pathlib import Path

SWEEP_DIR = Path("/home/renlephy/msl/research/moire_envelope/runsV3/thesis_honeycomb_K_b1_20260307_171424/eta_sweep_20260307_192037")
PYTHON = "/home/renlephy/.local/share/mamba/envs/msl/bin/python"
PROJECT_ROOT = Path("/home/renlephy/msl/research/moire_envelope")
EXPECTED_ANGLES = 8


def count_completed():
    """Count how many angles have completed Phase 3."""
    count = 0
    for theta_dir in sorted(SWEEP_DIR.glob("theta_*")):
        h5 = theta_dir / "candidate_0000" / "phase3_multiband_modes.h5"
        if h5.exists():
            count += 1
    return count


def check_sweep_json():
    """Check if sweep_results.json exists with all results."""
    results_path = SWEEP_DIR / "sweep_results.json"
    if not results_path.exists():
        return False, 0
    with open(results_path) as f:
        data = json.load(f)
    success = [d for d in data if 'error' not in d]
    return len(success) == EXPECTED_ANGLES, len(success)


def main():
    print(f"Watching: {SWEEP_DIR}")
    print(f"Expected: {EXPECTED_ANGLES} angles")
    print()
    
    while True:
        n_done = count_completed()
        sweep_done, n_json = check_sweep_json()
        print(f"  [{time.strftime('%H:%M:%S')}] Angles done: {n_done}/{EXPECTED_ANGLES}, "
              f"JSON: {n_json}/{EXPECTED_ANGLES}, Complete: {sweep_done}")
        
        if sweep_done:
            print(f"\n{'='*60}")
            print("  η-SWEEP COMPLETE! Running full analysis...")
            print(f"{'='*60}")
            
            # Run T03
            print("\n[1] Running T03 miniband dispersion...")
            subprocess.run([PYTHON, "thesis_results/T03_miniband_dispersion/compute.py"],
                           cwd=str(PROJECT_ROOT))
            
            # Run T11
            print("\n[2] Running T11 validation suite...")
            subprocess.run([PYTHON, "thesis_results/T11_miniband_validation/compute.py", "--all"],
                           cwd=str(PROJECT_ROOT))
            
            # Run honeycomb comparison
            print("\n[3] Running cross-candidate comparison...")
            subprocess.run([PYTHON, "thesis_results/T_honeycomb_analysis/comparison.py"],
                           cwd=str(PROJECT_ROOT))
            
            # Run full honeycomb analysis
            print("\n[4] Running full honeycomb analysis...")
            subprocess.run([PYTHON, "thesis_results/T_honeycomb_analysis/compute.py"],
                           cwd=str(PROJECT_ROOT))
            
            print(f"\n{'='*60}")
            print("  ALL ANALYSIS COMPLETE")
            print(f"{'='*60}")
            break
        
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
