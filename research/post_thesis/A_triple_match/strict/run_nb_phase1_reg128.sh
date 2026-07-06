#!/usr/bin/env bash
# Nb-ladder phase-1 at PRODUCTION registry 128, with crash-resume:
# the blaze sweep occasionally aborts with heap corruption at reg128
# ("corrupted double-linked list"); the sweep is checkpointed, so we simply
# retry — each attempt resumes from the surviving checkpoints.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for NB in 4 6 8; do
  echo "=== reg128 Nb=$NB $(date +%H:%M) ==="
  OUT="A_triple_match/strict/phase1_x_${NB}r0_reg128"
  for ATTEMPT in 1 2 3 4 5; do
    if mamba run -n msl python lib/phase1_blaze_v4.py \
        --config A_triple_match/strict/phase1_square_x.yaml \
        --cases square_x --n-retained $NB --n-remote 0 \
        --registry-grid-size 128 --threads 8 \
        --output-dir "$OUT"; then
      echo "--- Nb=$NB completed on attempt $ATTEMPT"
      break
    else
      echo "--- Nb=$NB attempt $ATTEMPT crashed; resuming from checkpoints"
    fi
  done
  NB=$NB mamba run -n msl python - <<'PY'
import zipfile, os
nb = os.environ["NB"]
src = f"A_triple_match/strict/phase1_x_{nb}r0_reg128/square_x_tm_phase1.npz"
if not os.path.isfile(src):
    raise SystemExit(f"MISSING after retries: {src}")
DROP_SUB = ("hermitized_eigenvectors", "_r_derivatives_", "_r_second_derivatives_")
zin = zipfile.ZipFile(src)
tmp = src + ".slim"
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED)
for it in zin.infolist():
    if any(s in it.filename for s in DROP_SUB):
        continue
    zout.writestr(it, zin.read(it.filename))
zout.close(); zin.close()
os.replace(tmp, src)
print("slimmed:", src, round(os.path.getsize(src)/1e6, 1), "MB")
PY
done
echo "=== REG128 NB PHASE1 DONE $(date +%H:%M) ==="
