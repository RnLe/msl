#!/usr/bin/env python3
"""
Phase C: High-Resolution Phase 1 (Universal Bloch-field extraction)
===================================================================

Phase 1 only: computes the θ-independent "Universal Master Map" of Bloch
fields ω_n(δ), u_n(r;δ) over the monolayer stacking space at high resolution.

Parameters:
  - mpb_resolution = 128        (per-cell MPB grid, up from 64)
  - mpb_registry_samples = 128  (stacking-space sampling)
  - phase1_Ns = 128             (envelope grid)

This data is reusable for:
  - Phase C monolayer-limit sweep (Phase 2+3 at 32 angles, run separately)
  - Any other downstream analysis needing high-resolution Bloch fields

Architecture:
  Phase 1 is θ-independent — run once with 16 multiprocessing workers.
  16384 MPB eigensolves at res=128 (~2.9h estimated).

Memory: ~3.5 GB per worker + final HDF5 ~16 GB compressed on disk.

Usage:
    nohup python phase_c_redo.py > phase_c_redo.log 2>&1 &
"""

import sys, os

# CRITICAL: Set threading env vars BEFORE importing numpy/scipy/mpb.
# MPB internal OMP/BLAS threading thrashes with Python multiprocessing.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import math, json, time, gc, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

# Flush stdout for nohup real-time logging
sys.stdout.reconfigure(line_buffering=True)

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase1_mpb_v3 as p1
import phase2_mpb_v3 as p2
import phase3_mpb_v3 as p3
import phase4_field_reconstruction as p4

sys.path.insert(0, str(THESIS_DIR))
try:
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False
    print("WARNING: symmetrize module not found, will skip C6 symmetrization")

# Import eta_sweep helpers
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))
from eta_sweep import (
    compute_moire_params, patch_h5_theta, patch_meta_theta,
    run_single_theta, collect_observables, plot_sweep_results,
)
from common.io_utils import candidate_dir, load_json

# =============================================================================
# Parameters
# =============================================================================

MPB_RESOLUTION = 128        # per-cell MPB grid resolution
REGISTRY_SAMPLES = 128      # stacking-space sampling
NS = 128                    # envelope grid (moiré scale)
N_MODES = 50                # envelope modes per angle
FD_ORDER = 4
N_WORKERS = 16              # Phase 1 multiprocessing workers

# 32 angles: 0.1° to 2.0° in 0.1° steps + 2.5° to 8.0° in 0.5° steps
THETA_LIST = [round(0.1 * i, 1) for i in range(1, 21)] + \
             [round(2.0 + 0.5 * i, 1) for i in range(1, 13)]
# = [0.1, 0.2, ..., 2.0, 2.5, 3.0, ..., 8.0]

# Source candidate (honeycomb K-point, TM, θ=1.1°)
SOURCE_CDIR = PROJECT_ROOT / "runsV3" / "thesis_honeycomb_K_b1_20260307_171424" / "candidate_0000"

# Output
OUTPUT_DIR = SCRIPT_DIR / "phase_c_redo_run"

# =============================================================================
# Phase 1: Universal stacking-space map (θ-independent)
# =============================================================================

def run_phase1(work_dir):
    """Run Phase 1 at reg=128, res=128 with 16 workers."""
    cdir = work_dir / "candidate_0000"
    p1_h5 = cdir / "phase1_multiband_data.h5"

    if p1_h5.exists():
        print(f"Phase 1 already exists at {p1_h5}, skipping.")
        return 0.0

    # Load candidate parameters (meta already in output dir from prior setup)
    meta_path = cdir / "phase0_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"phase0_meta.json not found at {meta_path}. "
            f"Copy it from the source candidate first."
        )
    with open(meta_path) as f:
        candidate_params = json.load(f)

    config_p1 = {
        'phase1_Ns1': NS,
        'phase1_Ns2': NS,
        'mpb_resolution': MPB_RESOLUTION,
        'mpb_registry_samples': REGISTRY_SAMPLES,
        'mpb_dk': 0.01,
        'mpb_fd_order': FD_ORDER,
        'mpb_polarization': candidate_params.get('dominant_polarization', 'TM'),
        'export_bloch_fields': True,
        'mpb_n_workers': N_WORKERS,
        'tau': [0.0, 0.0],
        'default_theta_deg': candidate_params.get('theta_deg', 1.1),
    }

    print(f"\n{'='*70}")
    print(f"  PHASE 1: MPB Bloch-field extraction")
    print(f"  Resolution: {MPB_RESOLUTION}, Registry: {REGISTRY_SAMPLES}, Ns: {NS}")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Output: {work_dir}")
    print(f"{'='*70}")

    t0 = time.time()
    p1.process_candidate_v3(candidate_params, config_p1, work_dir)
    wall = time.time() - t0
    print(f"Phase 1 done in {wall:.1f}s ({wall/3600:.2f}h)")
    gc.collect()

    # Verify output
    with h5py.File(p1_h5, 'r') as hf:
        bf = hf['bloch_fields']
        print(f"  bloch_fields: shape={bf.shape}, dtype={bf.dtype}, "
              f"nbytes={bf.nbytes/1e9:.1f} GB")

    return wall


# =============================================================================
# Sweep: Phase 2+3 per angle
# =============================================================================

def run_sweep(work_dir, sweep_dir):
    """Run Phase 2+3+C6 for each angle, with continuous saving."""
    source_cdir = work_dir / "candidate_0000"

    # Sweep config (all flags ON)
    config = {
        'include_born_huang': True,
        'include_drift_term': True,
        'use_parallel_transport_gauge': True,
        'n_extra_bands': 4,
        'mpb_fd_order': FD_ORDER,
        'include_offdiag_A': True,  # off-diagonal Berry connection
        'symmetry_type': 'C6',     # honeycomb K-point symmetry
    }

    # Save sweep config
    sweep_config = {
        'mpb_resolution': MPB_RESOLUTION,
        'registry_samples': REGISTRY_SAMPLES,
        'Ns': NS,
        'n_modes': N_MODES,
        'theta_list': THETA_LIST,
        'config': config,
        'source_dir': str(SOURCE_CDIR),
        'timestamp': datetime.now().isoformat(),
    }
    with open(sweep_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    # Check for existing partial results (resumability)
    results_path = sweep_dir / 'sweep_results_partial.json'
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        completed_thetas = {r['theta_deg'] for r in all_results if 'error' not in r}
        print(f"Resuming: {len(completed_thetas)} angles already done")
    else:
        all_results = []
        completed_thetas = set()

    total_angles = len(THETA_LIST)
    t_sweep_start = time.time()

    for i, theta_deg in enumerate(THETA_LIST):
        if theta_deg in completed_thetas:
            print(f"\n[{i+1}/{total_angles}] θ = {theta_deg}° already done, skipping")
            continue

        print(f"\n{'#'*70}")
        print(f"  ANGLE {i+1}/{total_angles}: θ = {theta_deg}°")
        elapsed = time.time() - t_sweep_start
        if i > len(completed_thetas):
            done_count = i - len(completed_thetas)
            if done_count > 0:
                rate = elapsed / done_count
                remaining = (total_angles - i) * rate
                print(f"  Elapsed: {elapsed/3600:.2f}h, "
                      f"ETA: {remaining/3600:.2f}h remaining")
        print(f"{'#'*70}")

        try:
            result = run_single_theta(
                theta_deg, source_cdir, sweep_dir, config, N_MODES
            )
            all_results.append(result)

            # Continuous saving
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

            completed_thetas.add(theta_deg)
            gc.collect()

        except Exception as e:
            print(f"\n  ERROR at θ = {theta_deg}°: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'theta_deg': theta_deg,
                'eta': 2 * math.sin(math.radians(theta_deg) / 2),
                'error': str(e),
            })
            # Save even errors
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

    return all_results


# =============================================================================
# Analysis + plotting
# =============================================================================

def run_analysis(all_results, sweep_dir):
    """Generate analysis plots and summary."""
    valid = [r for r in all_results if 'error' not in r]
    if len(valid) < 2:
        print("Not enough valid results for analysis")
        return

    # Use eta_sweep's built-in plotting
    plot_sweep_results(valid, sweep_dir)

    # Additional analysis: power-law fits
    etas = np.array([r['eta'] for r in valid])
    bws = np.array([r['bandwidth_50'] for r in valid])
    mixings = np.array([r['max_mixing'] for r in valid])
    omega_refs = np.array([r['omega_ref'] for r in valid])
    centers = np.array([r['omega_ref'] + np.mean(r['eigenvalues']) for r in valid])

    # BW ~ η^α fit (log-log linear regression)
    mask = etas > 0.001  # exclude any zero
    if mask.sum() >= 3 and np.all(bws[mask] > 0):
        log_eta = np.log(etas[mask])
        log_bw = np.log(bws[mask])
        coeffs = np.polyfit(log_eta, log_bw, 1)
        alpha_bw = coeffs[0]
        print(f"\nBandwidth scaling: BW ~ η^{alpha_bw:.3f} (theory: η²)")

    # Mixing ~ η^β fit
    mask_mix = (etas > 0.001) & (mixings > 1e-15)
    if mask_mix.sum() >= 3:
        log_mixing = np.log(mixings[mask_mix])
        coeffs_mix = np.polyfit(np.log(etas[mask_mix]), log_mixing, 1)
        beta_mix = coeffs_mix[0]
        print(f"Mixing scaling: mixing ~ η^{beta_mix:.3f}")

    # Spectral center convergence
    print(f"\nSpectral center (smallest θ): {centers[np.argmin(etas)]:.6f}")
    print(f"Spectral center (largest θ):  {centers[np.argmax(etas)]:.6f}")
    print(f"ω_ref (Dirac freq at K):      {omega_refs[0]:.6f}")

    # Save final summary
    summary = {
        'n_angles': len(valid),
        'n_errors': len(all_results) - len(valid),
        'theta_range': [min(r['theta_deg'] for r in valid),
                        max(r['theta_deg'] for r in valid)],
        'mpb_resolution': MPB_RESOLUTION,
        'registry_samples': REGISTRY_SAMPLES,
        'Ns': NS,
        'results': valid,
    }
    with open(sweep_dir / 'phase_c_redo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {sweep_dir / 'phase_c_redo_summary.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    t_total_start = time.time()
    print(f"\n{'='*70}")
    print(f"  PHASE 1: High-Resolution Bloch-field Extraction")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Resolution: MPB={MPB_RESOLUTION}, Registry={REGISTRY_SAMPLES}, Ns={NS}")
    print(f"  Mode: Phase 1 only (sweep deferred)")
    print(f"{'='*70}")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_dir = OUTPUT_DIR / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ──
    print("\n" + "="*70)
    print("  STEP 1: Phase 1 — Universal Bloch-field extraction")
    print("="*70)
    wall_p1 = run_phase1(OUTPUT_DIR)

    # ── Phase 1 only mode ──
    # The sweep (Phase 2+3) can be run later by a separate invocation.
    # The high-res Phase 1 data is useful for all downstream analyses.
    wall_total = time.time() - t_total_start

    print(f"\n{'='*70}")
    print(f"  PHASE 1 COMPLETE (Phase 1 only mode)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Phase 1: {wall_p1:.0f}s ({wall_p1/3600:.2f}h)")
    print(f"  Total:   {wall_total:.0f}s ({wall_total/3600:.2f}h)")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Phase 2+3 sweep can be run later.")
    print(f"{'='*70}")

    # Save wall times
    with open(OUTPUT_DIR / 'wall_times.json', 'w') as f:
        json.dump({
            'phase1_s': wall_p1,
            'total_s': wall_total,
            'mode': 'phase1_only',
        }, f, indent=2)


if __name__ == '__main__':
    main()
