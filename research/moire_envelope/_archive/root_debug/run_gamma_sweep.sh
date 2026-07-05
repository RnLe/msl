#!/bin/bash
# Run η-sweep for Gamma-point crystal
# Usage: bash run_gamma_sweep.sh
cd /home/renlephy/msl/research/moire_envelope
/home/renlephy/.local/share/mamba/envs/msl/bin/python -u eta_sweep.py \
    --candidate_id 0 \
    --n_modes 50 \
    | tee runsV3/eta_sweep_gamma_log.txt
echo "Sweep complete at $(date)"
