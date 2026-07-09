#!/usr/bin/env bash
# Weak-coupling crossover — LIGHT stage (MPB E_ref + Λ landscape + momentum
# model) for each candidate r2. All fast/low-RAM; each step skips if its output
# already exists, so a crash resumes cleanly. m=57 (2°) throughout.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1

# tag  r2     jmax  win_lo  win_hi
CANDS=(
  "100 0.100 14 0.370 0.383"
  "070 0.070 14 0.405 0.420"
  "054 0.054 14 0.420 0.432"
  "040 0.040 14 0.430 0.440"
  "031 0.031 14 0.434 0.442"
)

for row in "${CANDS[@]}"; do
  read tag r2 jmax wlo whi <<< "$row"
  echo "=== r2=$r2 (tag $tag) $(date +%H:%M:%S) ==="
  EREF="eref_r2_${tag}.npz"; LAM="lambda_r2_${tag}.npz"; MOM="momentum_kp_r2_${tag}.npz"

  if [[ -f "$EREF" ]]; then echo "  eref: cached"; else
    python momentum_kp_ref.py --r2 "$r2" --jmax "$jmax" --out "$EREF" \
      2>&1 | grep -v "^Using MPI" | sed 's/^/  eref: /'
  fi
  if [[ -f "$LAM" ]]; then echo "  lambda: cached"; else
    python lambda_landscape.py --r2 "$r2" --nreg 128 --res 64 --out "$LAM" \
      2>&1 | grep -v "^Using MPI" | sed 's/^/  lambda: /'
  fi
  # momentum model always (re)runs: cheap, and pulls the latest eref/lambda
  python momentum_kp_moire.py --eref "$EREF" --phase1 "$LAM" --ncut 6 \
    --window "$wlo" "$whi" --out "$MOM" \
    2>&1 | grep -E "edge d|in-window|index-aligned" | sed 's/^/  mom: /'
done
echo "=== LIGHT stage done $(date +%H:%M:%S) ==="
