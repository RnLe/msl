#!/usr/bin/env bash
# Weak-coupling crossover — GALERKIN certificate. Variational proof that at weak
# coupling a SINGLE reference + few bands converges to the FDFD floor (contrast
# with the strong-coupling +6-7e-3 plateau, §9). Single-ref (nref=1) at the
# well-bottom registry sbar (from refine_candidate), Nb=2/4/6, gcut=4, px16.
# Per-reference + _HS checkpoints (galerkin_recip) make it crash-resumable.
#
# Usage: run_weak_galerkin.sh <r2> <tag> <sbar_x> <sbar_y>
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=4
r2="${1:?r2}"; tag="${2:?tag}"; sx="${3:?sbar_x}"; sy="${4:?sbar_y}"
FDFD="fdfd_xman_r2_${tag}.npz"
if [[ ! -f "$FDFD" ]]; then echo "MISSING $FDFD (run FDFD stage first)"; exit 1; fi

for nb in 2 4 6; do
  OUT="grecip_r2_${tag}_nb${nb}.npz"
  echo "=== Galerkin r2=$r2 Nb=$nb sbar=($sx,$sy)  $(date +%H:%M:%S) ==="
  python galerkin_recip.py --m 57 --px 16 --r2 "$r2" --nref 1 --nbands "$nb" \
    --gcut 4 --sbar "$sx" "$sy" --fdfd "$FDFD" --out "$OUT" \
    2>&1 | grep -vE "^Using MPI" | grep -E "S-rank|window|Galerkin window|FDFD" | tail -4
done
echo "=== Galerkin r2=$r2 done $(date +%H:%M:%S) ==="
