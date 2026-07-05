#!/usr/bin/env bash
# A2 remote-band ladder: 4 sequential Phase-1 runs on the golden crystal.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for NR in 0 4 8 16; do
  echo "=== phase1 n_remote=$NR $(date +%H:%M:%S) ==="
  mamba run -n msl python lib/phase1_blaze_v4.py \
    --config A_triple_match/a2_config_base.yaml \
    --n-remote $NR \
    --output-dir A_triple_match/phase1_nrem$NR
done
echo "=== LADDER COMPLETE $(date +%H:%M:%S) ==="
