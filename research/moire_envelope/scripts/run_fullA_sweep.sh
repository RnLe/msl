#!/bin/bash
# =============================================================================
# Re-run η-sweep + T03 + T11 with include_offdiag_A=True
# Phase 3 already completed for both candidates — only downstream steps needed.
# =============================================================================

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="/home/renlephy/.local/share/mamba/envs/msl/bin/python"

# Force single-threaded MPB/BLAS — MPB's internal threading is harmful.
# Parallelism must come from Python multiprocessing only.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLAS_NUM_THREADS=1

LOGFILE="runsV3/fullA_sweep_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=============================================="
echo "FULL-A SWEEP PIPELINE (Phase 3 already done)"
echo "Started: $(date)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "=============================================="

# ============================================================================
# STEP 1: η-sweep for C3 (square_M_b3) with full off-diagonal A
# ============================================================================
echo ""
echo ">>> STEP 1: η-sweep for C3 (square_M_b3)"
echo "    Started: $(date)"
$PYTHON thesis_results/run_eta_sweep.py square_M_b3 --n_modes 50
echo "    Completed: $(date)"

# ============================================================================
# STEP 2: η-sweep for C1 (hex_M_b1) with full off-diagonal A
# ============================================================================
echo ""
echo ">>> STEP 2: η-sweep for C1 (hex_M_b1)"
echo "    Started: $(date)"
$PYTHON thesis_results/run_eta_sweep.py hex_M_b1 --n_modes 50
echo "    Completed: $(date)"

# ============================================================================
# STEP 3: T03 miniband dispersion for C3
# ============================================================================
echo ""
echo ">>> STEP 3: T03 miniband dispersion for C3 (square_M_b3)"
echo "    Started: $(date)"
$PYTHON results_bands/compute_miniband_structure.py \
    --h5 runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/phase2_multiband_data_c4sym.h5 \
    --outdir thesis_results/T03_miniband_dispersion/bz_dispersion \
    --nmodes 20 --nq 10
echo "    Completed: $(date)"

# ============================================================================
# STEP 4: T03 miniband dispersion for C1
# ============================================================================
echo ""
echo ">>> STEP 4: T03 miniband dispersion for C1 (hex_M_b1)"
echo "    Started: $(date)"
$PYTHON results_bands/compute_miniband_structure.py \
    --h5 runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/phase2_multiband_data_c2sym.h5 \
    --outdir thesis_results/T03_miniband_dispersion/hex_M_b1_bz_dispersion \
    --nmodes 20 --nq 10
echo "    Completed: $(date)"

# ============================================================================
# STEP 5: T11 validation for C3
# ============================================================================
echo ""
echo ">>> STEP 5: T11 validation for C3 (square_M_b3)"
echo "    Started: $(date)"
$PYTHON thesis_results/T11_miniband_validation/compute.py square_M_b3
echo "    Completed: $(date)"

# ============================================================================
# STEP 6: T11 validation for C1
# ============================================================================
echo ""
echo ">>> STEP 6: T11 validation for C1 (hex_M_b1)"
echo "    Started: $(date)"
$PYTHON thesis_results/T11_miniband_validation/compute.py hex_M_b1
echo "    Completed: $(date)"

echo ""
echo "=============================================="
echo "FULL-A SWEEP PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
