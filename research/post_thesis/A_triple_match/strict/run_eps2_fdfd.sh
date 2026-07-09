#!/usr/bin/env bash
# eps2 crossover — FDFD X-manifold ground truth (px16). At r2=0.10 the rods are
# 1.6px (well-resolved), so px16 FDFD is accurate and the model↔FDFD residual is
# pure MODEL error. eps2=8.9 reuses the existing anchor (fdfd_xman_2deg.npz).
# sigma ≈ supercell manifold bottom (model bottom − small). Sequential, guarded.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4

# eps2=8.9 == r2=0.10 anchor: reuse the already-computed X-manifold
[[ -f fdfd_e89.npz ]] || cp fdfd_xman_2deg.npz fdfd_e89.npz

# tag  eps2  sigma  nmodes
CANDS=(
  "e50 5.0 0.407 120"
  "e35 3.5 0.421 120"
  "e25 2.5 0.431 120"
  "e20 2.0 0.435 120"
)
for row in "${CANDS[@]}"; do
  read tag e2 sig nm <<< "$row"
  OUT="fdfd_${tag}.npz"
  if [[ -f "$OUT" ]]; then echo "=== $OUT cached, skip ==="; continue; fi
  echo "=== FDFD eps2=$e2 sigma=$sig  $(date +%H:%M:%S) ==="
  # fdfd_xmanifold.py <m> <px> <sigma> <nmodes> <out> [r2] [r1] [eps2]
  python fdfd_xmanifold.py 57 16 "$sig" "$nm" "$OUT" 0.10 0.20 "$e2" 2>&1 \
    | grep -vE "^Using MPI" | tail -3
done
echo "=== eps2 FDFD done $(date +%H:%M:%S) ==="
