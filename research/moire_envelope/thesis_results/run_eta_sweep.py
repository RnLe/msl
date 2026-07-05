"""
Thesis η-sweep wrapper.

Wraps the existing eta_sweep.py to accept explicit run directories
(thesis runs are named thesis_* which won't be found by the standard
find_latest_run_dir(base_name="phase0_mpb_v3") auto-detection).

Usage:
    python thesis_results/run_eta_sweep.py hex_M_b1
    python thesis_results/run_eta_sweep.py hex_M_b3 --n_modes 50
    python thesis_results/run_eta_sweep.py square_M_b3 --angles 1.0 2.0 3.0
    python thesis_results/run_eta_sweep.py --all
"""

import sys, argparse, time
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

from thesis_utils import (
    load_candidates_yaml, get_candidate_names, find_thesis_run_dir,
)

# Import the real eta_sweep machinery
import eta_sweep


DEFAULT_ANGLES = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
DEFAULT_N_MODES = 50   # matches n_envelope_eigenstates in configs


def run_thesis_eta_sweep(
    candidate_name: str,
    theta_list: list = None,
    n_modes: int = DEFAULT_N_MODES,
):
    """
    Run η-sweep for a single thesis candidate.

    1. Find the thesis run directory for this candidate
    2. Monkey-patch the p4.find_latest_run_dir to return it
    3. Call eta_sweep.run_eta_sweep with candidate_id=0
       (thesis runs have exactly one candidate per run directory)
    """
    if theta_list is None:
        theta_list = DEFAULT_ANGLES

    run_dir = find_thesis_run_dir(candidate_name)
    print(f"\n{'='*70}")
    print(f"  Thesis η-sweep: {candidate_name}")
    print(f"  Run dir: {run_dir}")
    print(f"  Angles: {theta_list}")
    print(f"  Modes: {n_modes}")
    print(f"{'='*70}")

    # Verify Phase 1/2 data exists
    cand_dir = run_dir / "candidate_0000"
    phase1_h5 = cand_dir / "phase1_multiband_data.h5"
    phase2_h5 = cand_dir / "phase2_multiband_data.h5"
    if not phase1_h5.exists():
        raise FileNotFoundError(f"Phase 1 data missing: {phase1_h5}")
    if not phase2_h5.exists():
        print(f"  WARNING: Phase 2 data missing — η-sweep will run Phase 2 per angle")

    # Monkey-patch find_latest_run_dir to return our thesis run_dir
    # eta_sweep.py imports phase4_field_reconstruction directly (not via phasesV3.),
    # so we must patch the same module object it references.
    import phase4_field_reconstruction as p4_direct
    original_find = p4_direct.find_latest_run_dir
    p4_direct.find_latest_run_dir = lambda base_name=None: run_dir

    config_overrides = {
        'include_born_huang': False,      # validated negligible
        'include_drift_term': True,
        'include_offdiag_A': True,        # CRITICAL: enable full Berry coupling
        'use_parallel_transport_gauge': True,
        'n_extra_bands': 4,
        'mpb_fd_order': 4,
    }

    try:
        results, sweep_dir = eta_sweep.run_eta_sweep(
            candidate_id=0,     # thesis runs have one candidate each
            theta_list=theta_list,
            n_modes=n_modes,
            config_overrides=config_overrides,
        )
    finally:
        # Restore original function
        p4_direct.find_latest_run_dir = original_find

    return results, sweep_dir


def main():
    parser = argparse.ArgumentParser(
        description="Thesis η-sweep wrapper for named candidates"
    )
    parser.add_argument(
        "candidate", nargs='?', default=None,
        help="Candidate name (e.g. hex_M_b1) or --all"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run η-sweep for all thesis candidates"
    )
    parser.add_argument(
        "--angles", type=float, nargs='+', default=None,
        help=f"Twist angles (deg). Default: {DEFAULT_ANGLES}"
    )
    parser.add_argument(
        "--n_modes", type=int, default=DEFAULT_N_MODES,
        help=f"Envelope modes per angle. Default: {DEFAULT_N_MODES}"
    )
    args = parser.parse_args()

    if args.all:
        names = get_candidate_names()
    elif args.candidate:
        names = [args.candidate]
    else:
        parser.error("Provide a candidate name or --all")

    t0 = time.time()
    for name in names:
        print(f"\n{'#'*70}")
        print(f"  CANDIDATE: {name}")
        print(f"{'#'*70}")
        try:
            run_thesis_eta_sweep(name, args.angles, args.n_modes)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
