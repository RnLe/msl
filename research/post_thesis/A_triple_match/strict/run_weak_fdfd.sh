#!/usr/bin/env bash
# Weak-coupling crossover — HEAVY stage: FDFD X-manifold ground truth (px16) for
# each new candidate r2. Sequential (one CHOLMOD factorization at a time is the
# RAM bottleneck ~832k DOF at m=57). Each solve's output npz is its own
# checkpoint: a re-run skips finished candidates, so a WSL crash resumes.
# sigma = scan well-bottom (min L1(X;s)); the gap below the manifold is empty so
# shift-invert captures the manifold from the bottom up. m=57 (2°), n=1.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4

# tag  r2     sigma  nmodes   (sigma = refine manifold-bottom, just inside)
CANDS=(
  "070 0.070 0.402 120"
  "054 0.054 0.418 120"
  "040 0.040 0.429 120"
  "031 0.031 0.434 120"
)

for row in "${CANDS[@]}"; do
  read tag r2 sig nm <<< "$row"
  OUT="fdfd_xman_r2_${tag}.npz"
  if [[ -f "$OUT" ]]; then echo "=== r2=$r2: $OUT cached, skip ==="; continue; fi
  echo "=== FDFD r2=$r2 sigma=$sig nmodes=$nm  $(date +%H:%M:%S) ==="
  python fdfd_xmanifold.py 57 16 "$sig" "$nm" "$OUT" "$r2" 2>&1 \
    | grep -vE "^Using MPI" | tail -3
done
echo "=== FDFD stage done $(date +%H:%M:%S) ==="
