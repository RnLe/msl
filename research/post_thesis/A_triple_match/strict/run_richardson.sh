#!/usr/bin/env bash
# Richardson refinement: does the FDFD X-manifold converge TO the momentum model
# as px grows? At weak coupling the layer-2 rods are sub-pixel at px16 (r₂×px
# px radius), so the px16 residual is FDFD discretization error, not model error.
# The model (MPB res-64) resolves the rods fully = the px→∞ limit. Show FDFD
# ground state → model as px16→px32→px48. Each output npz is its own checkpoint.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4

# tag  r2     px  sigma  nmodes
# NOTE: px48 (7.5M DOF) needs ~65 GB — OOMs at 40 GB; px16→px32 gives a valid
# 1/px² 2-point extrapolation, so we stop at px32.
RUNS=(
  "040 0.040 32 0.428 80"
  "054 0.054 32 0.417 80"
  "070 0.070 32 0.401 80"
)
for row in "${RUNS[@]}"; do
  read tag r2 px sig nm <<< "$row"
  OUT="fdfd_xman_r2_${tag}_px${px}.npz"
  if [[ -f "$OUT" ]]; then echo "=== $OUT cached, skip ==="; continue; fi
  echo "=== Richardson r2=$r2 px=$px sigma=$sig  $(date +%H:%M:%S) ==="
  python fdfd_xmanifold.py 57 "$px" "$sig" "$nm" "$OUT" "$r2" 2>&1 \
    | grep -vE "^Using MPI" | tail -3
done
echo "=== Richardson done $(date +%H:%M:%S) ==="
