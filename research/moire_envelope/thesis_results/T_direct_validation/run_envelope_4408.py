"""Run envelope approximation at θ=4.408° for honecomb K b1 (TM) candidate.

Directly calls the eta_sweep machinery with the correct TM run directory.
"""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phasesV3'))
sys.path.insert(0, str(PROJECT_ROOT / 'thesis_results'))

import eta_sweep

# Force the correct TM run directory (not TE)
RUN_DIR = PROJECT_ROOT / 'runsV3' / 'thesis_honeycomb_K_b1_20260307_171424'
CAND_DIR = RUN_DIR / 'candidate_0000'

print(f"Run directory: {RUN_DIR}")
print(f"Candidate dir: {CAND_DIR}")

# Monkey-patch find_latest_run_dir
import phase4_field_reconstruction as p4_direct
p4_direct.find_latest_run_dir = lambda base_name=None: RUN_DIR

config_overrides = {
    'include_born_huang': False,
    'include_drift_term': True,
    'include_offdiag_A': True,
    'use_parallel_transport_gauge': True,
    'n_extra_bands': 4,
    'mpb_fd_order': 4,
}

results, sweep_dir = eta_sweep.run_eta_sweep(
    candidate_id=0,
    theta_list=[4.408],
    n_modes=50,
    config_overrides=config_overrides,
)

print(f"\nSweep completed. Results in: {sweep_dir}")
print(f"Number of results: {len(results)}")
for r in results:
    import numpy as np
    evals = np.array(r['eigenvalues'])
    print(f"  θ={r['theta_deg']}°: {len(evals)} eigenvalues")
    print(f"    omega_ref = {r['omega_ref']}")
    print(f"    lambda range: [{evals[0]:.8f}, {evals[-1]:.8f}]")
    print(f"    physical freq range: [{r['omega_ref']+evals[0]:.6f}, {r['omega_ref']+evals[-1]:.6f}]")
