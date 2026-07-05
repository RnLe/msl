#!/usr/bin/env python3
"""
Phase 1: High-Resolution Bloch-Field Extraction for Square Lattice M-point
==========================================================================

Generates the θ-independent "Universal Master Map" of Bloch fields
ω_n(δ), u_n(r;δ) for a square lattice at the M point, TM polarization.

Candidate: square lattice, r/a=0.2, ε_rod=11.56, ε_bg=1.0
   Band 3 at M = (0.5, 0.5) — isolated with gap ~0.18 on both sides
   ω₀(M, band 3) ≈ 0.68457 (c/a)

Parameters:
  - mpb_resolution = 128
  - mpb_registry_samples = 128
  - phase1_Ns = 128
  - n_workers = 16

Output: phase1_multiband_data.h5 (~12 GB with Bloch field export)

Usage:
    nohup python square_phase1.py > square_phase1.log 2>&1 &
"""

import sys, os

# CRITICAL: Set threading env vars BEFORE importing numpy/scipy/mpb.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import json, time, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent  # moire_envelope
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase1_mpb_v3 as p1

# =============================================================================
# Parameters
# =============================================================================

MPB_RESOLUTION = 128
REGISTRY_SAMPLES = 128
NS = 128
FD_ORDER = 4
N_WORKERS = 16

OUTPUT_DIR = SCRIPT_DIR / "square_M_b3_phase1_run"

# Phase 0 meta for this candidate (manually constructed)
CANDIDATE_PARAMS = {
    "candidate_id": 0,
    "lattice_type": "square",
    "a": 1.0,
    "r_over_a": 0.2,
    "eps_bg": 1.0,
    "eps_hole": 11.56,       # dielectric material of the rods
    "band_index": 3,         # 0-indexed: 4th TM band at M, isolated
    "k_label": "M",
    "k0_x": 0.5,
    "k0_y": 0.5,
    "omega0": 0.68457,       # from MPB at res=128
    "polarization": "TM",
    "dominant_polarization": "TM",
    "local_polarization": "TM",
    "n_subspace_bands": 1,
    "subspace_bands": [3],           # single isolated band
    "all_bands": [0, 1, 2, 3, 4, 5, 6, 7],  # extra bands for Born-Huang
    "target_index_in_subspace": 0,
    "theta_deg": 2.01,       # default (doesn't matter for Phase 1 registry)
    "theta_rad": 0.035089,
    "moire_length": 28.51,   # a / (2*sin(theta/2))
    "eta": 0.035087,         # 2*sin(theta/2)
}

# Phase 1 config
CONFIG_P1 = {
    'phase1_Ns1': NS,
    'phase1_Ns2': NS,
    'mpb_resolution': MPB_RESOLUTION,
    'mpb_registry_samples': REGISTRY_SAMPLES,
    'mpb_dk': 0.01,
    'mpb_fd_order': FD_ORDER,
    'mpb_polarization': 'TM',
    'export_bloch_fields': True,
    'mpb_n_workers': N_WORKERS,
    'tau': [0.0, 0.0],
    'default_theta_deg': CANDIDATE_PARAMS['theta_deg'],
}

# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  Square Lattice Phase 1: High-Resolution Bloch-Field Extraction")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Resolution: MPB={MPB_RESOLUTION}, Registry={REGISTRY_SAMPLES}, Ns={NS}")
    print(f"  Band: {CANDIDATE_PARAMS['band_index']} at {CANDIDATE_PARAMS['k_label']}")
    print(f"  ω₀ ≈ {CANDIDATE_PARAMS['omega0']:.5f} (c/a)")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing Phase 1 data
    cdir = OUTPUT_DIR / "candidate_0000"
    p1_h5 = cdir / "phase1_multiband_data.h5"
    if p1_h5.exists():
        print(f"\nPhase 1 already exists at {p1_h5}, skipping.")
        with h5py.File(p1_h5, 'r') as hf:
            if 'bloch_fields' in hf:
                bf = hf['bloch_fields']
                print(f"  bloch_fields: shape={bf.shape}, dtype={bf.dtype}, "
                      f"nbytes={bf.nbytes/1e9:.1f} GB")
        return

    # Run Phase 1
    print(f"\nRunning Phase 1...")
    p1.process_candidate_v3(CANDIDATE_PARAMS, CONFIG_P1, OUTPUT_DIR)

    wall = time.time() - t0
    print(f"\nPhase 1 complete in {wall:.1f}s ({wall/3600:.2f}h)")
    gc.collect()

    # Verify output
    if p1_h5.exists():
        with h5py.File(p1_h5, 'r') as hf:
            print(f"\n=== Output verification ===")
            for key in hf.keys():
                ds = hf[key]
                if hasattr(ds, 'shape'):
                    print(f"  {key}: shape={ds.shape}, dtype={ds.dtype}")
            if 'bloch_fields' in hf:
                bf = hf['bloch_fields']
                print(f"\n  bloch_fields: {bf.nbytes/1e9:.1f} GB")
            print(f"  omega_ref = {hf.attrs.get('omega_ref', 'N/A')}")
            print(f"  theta_deg = {hf.attrs.get('theta_deg', 'N/A')}")
    else:
        print(f"\nERROR: Phase 1 output not found at {p1_h5}!")

    # Save wall time
    with open(OUTPUT_DIR / 'wall_times.json', 'w') as f:
        json.dump({'phase1_s': wall, 'phase1_h': wall/3600}, f, indent=2)


if __name__ == '__main__':
    main()
