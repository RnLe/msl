#!/usr/bin/env bash
# CLEAN weak-coupling crossover via the DIELECTRIC-CONTRAST knob eps2, at fixed
# r2=0.10 (rods stay well-resolved: 1.6px at FDFD px16, 6.4px at MPB res-64), so
# the ONLY variable is coupling and the residual is pure MODEL error — no
# sub-pixel confound (unlike the r2 knob, where weak⟺tiny rods). Layer 1 fixed
# (r1=0.20, eps=8.9); lowering eps2 weakens the moiré potential while keeping
# the isolating layer-1 gap open. m=57 (2°). Skip-guarded / resumable.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1

# tag  eps2   jmax
CANDS=(
  "e89 8.9 14"
  "e50 5.0 14"
  "e35 3.5 14"
  "e25 2.5 14"
  "e20 2.0 14"
)
for row in "${CANDS[@]}"; do
  read tag e2 jmax <<< "$row"
  echo "=== eps2=$e2 (tag $tag) $(date +%H:%M:%S) ==="
  EREF="eref_${tag}.npz"; LAM="lambda_${tag}.npz"; MOM="momentum_${tag}.npz"
  if [[ -f "$EREF" ]]; then echo "  eref: cached"; else
    python momentum_kp_ref.py --r2 0.10 --eps1 8.9 --eps2 "$e2" --jmax "$jmax" \
      --out "$EREF" 2>&1 | grep -vE "^Using MPI|^$" | sed 's/^/  eref: /'
  fi
  if [[ -f "$LAM" ]]; then echo "  lambda: cached"; else
    python lambda_landscape.py --r2 0.10 --eps 8.9 --eps2 "$e2" --nreg 128 \
      --res 64 --out "$LAM" 2>&1 | grep -vE "^Using MPI|^$" | sed 's/^/  lambda: /'
  fi
  python momentum_kp_moire.py --eref "$EREF" --phase1 "$LAM" --ncut 6 \
    --out "$MOM" 2>&1 | grep -E "edge d" | sed 's/^/  mom: /'
done
echo "=== eps2 LIGHT done $(date +%H:%M:%S) ==="
