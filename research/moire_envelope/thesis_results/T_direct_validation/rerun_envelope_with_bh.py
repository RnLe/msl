"""
Re-run envelope at θ=1.1213° with Born-Huang ENABLED in Phase 2.
Then produce definitive comparison plots using Hungarian matching.
"""
import numpy as np
import json
import sys, os, time
from pathlib import Path

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '..', 'phasesV3'))

from T_direct_validation.commensurate_utils import commensurate_twist_angle
import eta_sweep
import phase4_field_reconstruction as p4_direct

M, N = 30, 29
theta_deg = np.degrees(commensurate_twist_angle('honeycomb', M, N))

run_dir = Path('/home/renlephy/msl/research/moire_envelope/runsV3/thesis_honeycomb_K_b1_20260307_171424')
original_find = p4_direct.find_latest_run_dir
p4_direct.find_latest_run_dir = lambda base_name=None: run_dir

print(f"Re-running envelope at θ = {theta_deg:.4f}° WITH Born-Huang")
print(f"Run dir: {run_dir}")

config_overrides = {
    'include_born_huang': True,    # ← NOW ENABLED
    'include_drift_term': True,
    'include_offdiag_A': True,
    'use_parallel_transport_gauge': True,
    'n_extra_bands': 4,
    'mpb_fd_order': 4,
}

t0 = time.time()
try:
    results, sweep_dir = eta_sweep.run_eta_sweep(
        candidate_id=0,
        theta_list=[theta_deg],
        n_modes=50,
        config_overrides=config_overrides,
    )
    env_data = results[0]
    print(f"\nEnvelope done in {time.time()-t0:.1f}s")
    print(f"Sweep dir: {sweep_dir}")
finally:
    p4_direct.find_latest_run_dir = original_find

# Extract and compare
env_evals = np.array(env_data['eigenvalues'])
env_freqs = np.sort(env_data['omega_ref'] + env_evals)
print(f"\nWith Born-Huang:")
print(f"  {len(env_freqs)} modes, BW = {env_freqs.max()-env_freqs.min():.6f}")
print(f"  Range: [{env_freqs.min():.6f}, {env_freqs.max():.6f}]")

# Compare with no-BH run
with open('/home/renlephy/msl/research/moire_envelope/runsV3/thesis_honeycomb_K_b1_20260307_171424/eta_sweep_20260310_181650/sweep_results.json') as f:
    old_data = json.load(f)[0]
old_evals = np.array(old_data['eigenvalues'])
old_freqs = np.sort(old_data['omega_ref'] + old_evals)

print(f"\nWithout Born-Huang (previous run):")
print(f"  Range: [{old_freqs.min():.6f}, {old_freqs.max():.6f}]")

diff = np.sort(env_evals) - np.sort(old_evals)
print(f"\nEigenvalue shift from Born-Huang:")
print(f"  mean|Δ| = {np.mean(np.abs(diff)):.6e}")
print(f"  max|Δ|  = {np.max(np.abs(diff)):.6e}")
print(f"  BH shifts eigenvalues by {np.mean(np.abs(diff))/(env_freqs.max()-env_freqs.min())*100:.2f}% of BW")

# Save the new sweep directory path for the plotting script
out_dir = os.path.dirname(os.path.abspath(__file__))
info = {
    'sweep_dir': str(sweep_dir),
    'theta_deg': theta_deg,
    'born_huang': True,
    'n_modes': 50,
}
with open(os.path.join(out_dir, 'envelope_bh_run_info.json'), 'w') as f:
    json.dump(info, f, indent=2)
print(f"\nSaved run info to envelope_bh_run_info.json")
