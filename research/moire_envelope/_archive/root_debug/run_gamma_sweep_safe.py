#!/usr/bin/env python
"""Run η-sweep for Gamma-point crystal, saving output to log file."""
import sys, os, signal

# Ignore SIGINT in this process
signal.signal(signal.SIGINT, signal.SIG_IGN)

# Change to working directory
os.chdir('/home/renlephy/msl/research/moire_envelope')

# Redirect stdout/stderr to log file
log_path = 'runsV3/eta_sweep_gamma_log.txt'
log_file = open(log_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'phasesV3')

# Import and run
from eta_sweep import run_eta_sweep

results, sweep_dir = run_eta_sweep(
    candidate_id=0,
    theta_list=[0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0, 8.0],
    n_modes=50,
)

print(f"\nSweep complete! Results in: {sweep_dir}", flush=True)
log_file.close()
