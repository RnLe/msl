#!/usr/bin/env bash
# Nb-ladder phase-2 at PRODUCTION settings (reg128 phase-1), 4-k folding,
# exact-TM. Focus: m29 (4°, where truncation error lives) and m57 (2°).
# Includes the Nb=2(+6rem) baseline from the slimmed March archive.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export MSL_LATTICE_TYPE=square MSL_POLARIZATION=TM

echo "=== Nb=2 baseline (March reg128 archive) m29 $(date +%H:%M) ==="
mamba run -n msl python A_triple_match/strict/strict_commensurate.py \
  --phase1 A_triple_match/strict/phase1_x_reg128_slim/square_x_tm_phase1.npz \
  --out A_triple_match/strict/phase2_x_2r6_reg128_m29 \
  --cases 29,1 --n-modes 10 --k-fold

for NB in 4 6 8; do
  echo "=== Nb=$NB reg128 $(date +%H:%M) ==="
  mamba run -n msl python A_triple_match/strict/strict_commensurate.py \
    --phase1 A_triple_match/strict/phase1_x_${NB}r0_reg128/square_x_tm_phase1.npz \
    --out A_triple_match/strict/phase2_x_${NB}r0_reg128 \
    --cases 29,1 57,1 --n-modes 15 --k-fold
done
echo "=== NB LADDER PHASE2 DONE $(date +%H:%M) ==="
