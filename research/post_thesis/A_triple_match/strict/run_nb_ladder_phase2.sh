#!/usr/bin/env bash
# Nb-ladder phase-2: exact-TM at 4/2/1 deg for Nb in {4,6,8} (n_remote=0)
# plus the X' valley at matching cases. n_modes=25/k for window coverage.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export MSL_LATTICE_TYPE=square MSL_POLARIZATION=TM
for NB in 4 6 8; do
  echo "=== Nb=$NB $(date +%H:%M) ==="
  mamba run -n msl python A_triple_match/strict/strict_commensurate.py \
    --phase1 A_triple_match/strict/phase1_x_${NB}r0/square_x_tm_phase1.npz \
    --out A_triple_match/strict/phase2_x_${NB}r0 \
    --cases 29,1 57,1 114,1 --n-modes 25 --k-fold
done
echo "=== NB LADDER PHASE2 DONE $(date +%H:%M) ==="
