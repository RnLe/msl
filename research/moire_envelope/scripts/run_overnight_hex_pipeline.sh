#!/usr/bin/env bash
# =============================================================================
# Overnight Pipeline: hex_M_b1 (C1) — Full Phase 1→2→Sym→3→η-sweep
# =============================================================================
# Expected runtime: 4-6 hours
#   Phase 1 (128×128 MPB registry, 8 bands, res=64): ~2h
#   Phase 2 (Berry connection + gauge): ~15 min
#   Symmetrize (C2 for hex M-point): ~5 min
#   Phase 3 (50 envelope modes): ~10 min
#   η-sweep (8 angles × ~15 min): ~2h
#   Miniband dispersion: ~5 min
#   T11 validation: ~2 min
#
# Usage: nohup bash run_overnight_hex_pipeline.sh > runsV3/overnight_hex_pipeline.log 2>&1 &
# =============================================================================

set -euo pipefail

# --- Environment setup ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # scripts/ lives one level below the pipeline root
PYTHON="/home/renlephy/.local/share/mamba/envs/msl/bin/python"

RUN_DIR="runsV3/thesis_hex_M_b1_20260209_173724"
CONFIG="configsV3/thesis_hex_M_b1.yaml"
CAND_DIR="$RUN_DIR/candidate_0000"
LOG_PREFIX="runsV3/thesis_hex_M_b1"

echo "============================================================"
echo "  OVERNIGHT PIPELINE: hex_M_b1 (C1)"
echo "  Started: $(date)"
echo "  Python: $PYTHON"
echo "  Run dir: $RUN_DIR"
echo "  Config: $CONFIG"
echo "============================================================"

# --- Step 1: Phase 1 (MPB local Bloch fields) ---
echo ""
echo "============================================================"
echo "  STEP 1: Phase 1 — MPB Local Bloch Fields"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u phasesV3/phase1_mpb_v3.py "$RUN_DIR" "$CONFIG" 2>&1 | tee "${LOG_PREFIX}_phase1.log"
echo "  Phase 1 completed: $(date)"

# Verify Phase 1 output
if [ ! -f "$CAND_DIR/phase1_multiband_data.h5" ]; then
    echo "ERROR: Phase 1 output not found! Aborting pipeline."
    exit 1
fi
echo "  Phase 1 output verified: $(ls -lh "$CAND_DIR/phase1_multiband_data.h5")"

# --- Step 2: Phase 2 (Berry connection + gauge fixing) ---
echo ""
echo "============================================================"
echo "  STEP 2: Phase 2 — Berry Connection & Gauge Fixing"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u phasesV3/phase2_mpb_v3.py "$RUN_DIR" "$CONFIG" 2>&1 | tee "${LOG_PREFIX}_phase2.log"
echo "  Phase 2 completed: $(date)"

if [ ! -f "$CAND_DIR/phase2_multiband_data.h5" ]; then
    echo "ERROR: Phase 2 output not found! Aborting pipeline."
    exit 1
fi
echo "  Phase 2 output verified: $(ls -lh "$CAND_DIR/phase2_multiband_data.h5")"

# --- Step 3: C2 Symmetrization (hex M-point) ---
echo ""
echo "============================================================"
echo "  STEP 3: C2 Symmetrization"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u thesis_results/symmetrize.py "$CAND_DIR" --sym C2 2>&1 | tee "${LOG_PREFIX}_symmetrize.log"
echo "  Symmetrization completed: $(date)"

# Determine symmetrized filename — symmetrize.py also replaces phase2_multiband_data.h5
# with the symmetrized version and makes a _unsym.h5 backup automatically.
SYM_H5="$CAND_DIR/phase2_multiband_data_c2sym.h5"
if [ ! -f "$SYM_H5" ]; then
    echo "WARNING: C2 sym output not found at expected path, checking alternatives..."
    SYM_H5=$(ls "$CAND_DIR"/phase2_multiband_data_*sym*.h5 2>/dev/null | head -1 || true)
    if [ -z "$SYM_H5" ]; then
        echo "WARNING: No symmetrized file found. Proceeding with unsymmetrized data."
        SYM_H5="$CAND_DIR/phase2_multiband_data.h5"
    fi
fi
echo "  Using symmetrized data: $(ls -lh "$SYM_H5")"

# --- Step 4: Phase 3 (Envelope Eigensolver, 50 modes) ---
echo ""
echo "============================================================"
echo "  STEP 4: Phase 3 — Envelope Eigensolver (50 modes)"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u phasesV3/phase3_mpb_v3.py "$RUN_DIR" "$CONFIG" 2>&1 | tee "${LOG_PREFIX}_phase3.log"
echo "  Phase 3 completed: $(date)"

if [ ! -f "$CAND_DIR/phase3_mode_stats.json" ]; then
    echo "ERROR: Phase 3 output not found! Aborting pipeline."
    exit 1
fi
echo "  Phase 3 output verified: $(ls -lh "$CAND_DIR/phase3_multiband_modes.h5")"

# --- Step 5: η-sweep (8 angles × 50 modes) ---
echo ""
echo "============================================================"
echo "  STEP 5: η-sweep (8 angles, 50 modes)"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u thesis_results/run_eta_sweep.py hex_M_b1 --n_modes 50 2>&1 | tee "${LOG_PREFIX}_eta_sweep.log"
echo "  η-sweep completed: $(date)"

# --- Step 6: Miniband dispersion E_n(q) ---
echo ""
echo "============================================================"
echo "  STEP 6: Miniband Dispersion E_n(q)"
echo "  Started: $(date)"
echo "============================================================"
# Find the symmetrized Phase 2 HDF5 to use
DISP_H5="$SYM_H5"
DISP_OUT="thesis_results/T03_miniband_dispersion/hex_M_b1_bz_dispersion/"
mkdir -p "$DISP_OUT"
$PYTHON -u results_bands/compute_miniband_structure.py \
    --h5 "$DISP_H5" \
    --outdir "$DISP_OUT" \
    --nmodes 20 \
    --nq 10 2>&1 | tee "${LOG_PREFIX}_miniband.log"
echo "  Miniband dispersion completed: $(date)"

# --- Step 7: T11 Validation Suite ---
echo ""
echo "============================================================"
echo "  STEP 7: T11 Dense Miniband Validation"
echo "  Started: $(date)"
echo "============================================================"
$PYTHON -u thesis_results/T11_miniband_validation/compute.py hex_M_b1 2>&1 | tee "${LOG_PREFIX}_T11.log"
echo "  T11 validation completed: $(date)"

# --- Summary ---
echo ""
echo "============================================================"
echo "  OVERNIGHT PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "============================================================"
echo ""
echo "  Outputs:"
echo "    Phase 1: $(ls -lh "$CAND_DIR/phase1_multiband_data.h5" 2>/dev/null || echo 'MISSING')"
echo "    Phase 2: $(ls -lh "$CAND_DIR/phase2_multiband_data.h5" 2>/dev/null || echo 'MISSING')"
echo "    Symm:    $(ls -lh "$SYM_H5" 2>/dev/null || echo 'MISSING')"
echo "    Phase 3: $(ls -lh "$CAND_DIR/phase3_multiband_modes.h5" 2>/dev/null || echo 'MISSING')"
echo "    Sweep:   $(find "$RUN_DIR" -name 'sweep_results.json' -print 2>/dev/null || echo 'NOT FOUND')"
echo "    T03:     $(ls "$DISP_OUT"/*.png 2>/dev/null | wc -l) figures"
echo "    T11:     $(find thesis_results/T11_miniband_validation/ -name '*hex*' -print 2>/dev/null | wc -l) files"
echo ""
echo "  Logs saved to: ${LOG_PREFIX}_*.log"
