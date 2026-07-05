#!/usr/bin/env python
"""
Re-extract observables from an already-completed η-sweep.
All Phase 2+3 data is already computed — this script just reads it,
runs the N=1 convergence test, and produces plots.

Usage:
  python eta_sweep_reextract.py <sweep_dir>
"""

import sys, json, math, time
from pathlib import Path
import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "phasesV3"))

from phasesV3 import phase3_mpb_v3 as p3
from phasesV3 import phase4_field_reconstruction as p4
from common.io_utils import load_json

from eta_sweep import (
    compute_moire_params, run_nband_convergence, plot_sweep_results,
)


def reextract_single(work_dir, source_cdir):
    """Extract observables from existing Phase 2+3 data in work_dir."""
    work_dir = Path(work_dir)

    # Read theta from the Phase 1 HDF5 attrs
    with h5py.File(work_dir / 'phase1_multiband_data.h5', 'r') as hf:
        theta_deg = float(hf.attrs['theta_deg'])
        eta = float(hf.attrs['eta'])
        moire_length = float(hf.attrs['moire_length'])
        omega_ref = float(hf.attrs['omega_ref'])
        B_moire = hf.attrs['B_moire']

    moire_params = {
        'theta_deg': theta_deg,
        'theta_rad': math.radians(theta_deg),
        'eta': eta,
        'B_moire': B_moire,
        'moire_length': moire_length,
    }

    print(f"\n  θ = {theta_deg}°, η = {eta:.6f}")

    # Load Phase 3 results
    F_spinor, eigenvalues, mode_stats = p4.load_phase3_envelopes(work_dir)
    n_modes = len(eigenvalues)
    N_sub = F_spinor.shape[-1]

    # Eigenvalue spectrum (Observable B)
    evals = eigenvalues[:min(n_modes, 50)]
    bandwidth = float(evals[-1] - evals[0]) if len(evals) > 1 else 0.0
    gap_01 = float(evals[1] - evals[0]) if len(evals) > 1 else 0.0

    # Band composition for ALL modes
    band_compositions = []
    for m in range(n_modes):
        F_mode = F_spinor[m]
        bw = np.array([np.sum(np.abs(F_mode[:, :, n])**2) for n in range(N_sub)])
        bw /= bw.sum()
        band_compositions.append({
            'mode': m,
            'weights': bw.tolist(),
            'dominant': int(np.argmax(bw)),
            'max_weight': float(np.max(bw)),
            'mixing': float(1.0 - np.max(bw)),
        })

    max_mixing = max(bc['mixing'] for bc in band_compositions)

    # N-band convergence (Observable A)
    print(f"  Running N=1 band convergence test...")
    nband_results = run_nband_convergence(work_dir, moire_params, n_modes=20)
    evals_N1 = np.array(nband_results['eigenvalues_N1'][:20])

    # Find target band modes in N=3
    meta = load_json(work_dir / 'phase0_meta.json')
    target_sub = meta.get('target_index_in_subspace', 0)

    target_modes_N3 = [m for m in range(len(band_compositions))
                       if band_compositions[m]['dominant'] == target_sub]

    if target_modes_N3:
        lambda_0_N3 = float(eigenvalues[target_modes_N3[0]])
    else:
        lambda_0_N3 = float(eigenvalues[0])
    lambda_0_N1 = float(evals_N1[0])
    delta_lambda_N = abs(lambda_0_N3 - lambda_0_N1)

    return {
        'theta_deg': theta_deg,
        'eta': eta,
        'moire_length': moire_length,
        'eigenvalues': evals.tolist(),
        'bandwidth_50': bandwidth,
        'gap_01': gap_01,
        'omega_ref': omega_ref,
        'max_mixing': max_mixing,
        'band_compositions': band_compositions[:20],  # save first 20 for JSON
        'lambda_0_N3': lambda_0_N3,
        'lambda_0_N1': lambda_0_N1,
        'delta_lambda_N': delta_lambda_N,
        'R_fd_corrected': None,
        'ratio_fd_corrected': None,
        'wall_time_s': 0.0,
        'n_target_modes': len(target_modes_N3),
    }


def main():
    if len(sys.argv) < 2:
        # Auto-find latest sweep
        run_dir = p4.find_latest_run_dir()
        sweep_dirs = sorted([d for d in run_dir.iterdir()
                            if d.is_dir() and d.name.startswith('eta_sweep')])
        if not sweep_dirs:
            print("No sweep directories found")
            sys.exit(1)
        sweep_dir = sweep_dirs[-1]
    else:
        sweep_dir = Path(sys.argv[1])

    print(f"Sweep dir: {sweep_dir}")

    # Find source candidate dir
    config_path = sweep_dir / 'sweep_config.json'
    if config_path.exists():
        cfg = load_json(config_path)
        source_cdir = Path(cfg['source_dir'])
    else:
        run_dir = p4.find_latest_run_dir()
        source_cdir = Path(run_dir) / "candidate_0000"

    # Find all theta subdirectories
    theta_dirs = sorted([d for d in sweep_dir.iterdir()
                        if d.is_dir() and d.name.startswith('theta_')])

    print(f"Found {len(theta_dirs)} angle directories")

    all_results = []
    for td in theta_dirs:
        cdir = td / "candidate_0000"
        if not (cdir / 'phase3_multiband_modes.h5').exists():
            print(f"  Skipping {td.name}: no Phase 3 data")
            continue
        try:
            r = reextract_single(cdir, source_cdir)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR in {td.name}: {e}")
            import traceback; traceback.print_exc()

    # Save results
    out_path = sweep_dir / 'sweep_results_reextracted.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    # Plot
    if len(all_results) >= 2:
        plot_sweep_results(all_results, sweep_dir)

    # Print summary table
    etas = np.array([r['eta'] for r in all_results])
    sort_idx = np.argsort(etas)
    results_sorted = [all_results[i] for i in sort_idx]

    print(f"\n{'='*100}")
    print(f"  {'θ (°)':>8} {'η':>10} {'L_m':>8} {'λ₀(N=3)':>12} {'λ₀(N=1)':>12} "
          f"{'|Δλ|':>12} {'BW':>10} {'gap₀₁':>10} {'mixing':>10}")
    print(f"{'='*100}")
    for r in results_sorted:
        print(f"  {r['theta_deg']:8.3f} {r['eta']:10.6f} {r['moire_length']:8.2f} "
              f"{r['lambda_0_N3']:12.6f} {r['lambda_0_N1']:12.6f} "
              f"{r['delta_lambda_N']:12.4e} {r['bandwidth_50']:10.6f} "
              f"{r['gap_01']:10.6f} {r['max_mixing']:10.2e}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
