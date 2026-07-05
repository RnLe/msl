#!/bin/zsh
# =============================================================================
# Re-run Phase 3 + η-sweep + T03 + T11 with include_offdiag_A=True
# =============================================================================
# This script re-runs ALL pipeline stages that depend on the Hamiltonian
# assembly with full off-diagonal Berry connection enabled.
#
# The config YAML files already have include_offdiag_A: true.
# The Phase 3 code now reads this flag and passes it to the Hamiltonian.
#
# Created: 2026-03-07
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

LOGFILE="runsV3/fullA_rerun_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=============================================="
echo "FULL-A RE-RUN PIPELINE"
echo "Started: $(date)"
echo "Log: $LOGFILE"
echo "=============================================="

# ============================================================================
# STEP 1: Re-run Phase 3 for C3 (square_M_b3) with include_offdiag_A=True
# ============================================================================
echo ""
echo ">>> STEP 1: Phase 3 for C3 (square_M_b3) with full off-diagonal A"
echo "    Started: $(date)"
$PYTHON phasesV3/phase3_mpb_v3.py \
    runsV3/thesis_square_M_b3_20260209_173724 \
    configsV3/thesis_square_M_b3.yaml
echo "    Completed: $(date)"

# ============================================================================
# STEP 2: Re-run Phase 3 for C1 (hex_M_b1) with include_offdiag_A=True
# ============================================================================
echo ""
echo ">>> STEP 2: Phase 3 for C1 (hex_M_b1) with full off-diagonal A"
echo "    Started: $(date)"
$PYTHON phasesV3/phase3_mpb_v3.py \
    runsV3/thesis_hex_M_b1_20260209_173724 \
    configsV3/thesis_hex_M_b1.yaml
echo "    Completed: $(date)"

# ============================================================================
# STEP 3: η-sweep for C3 (square_M_b3) with full off-diagonal A
# ============================================================================
echo ""
echo ">>> STEP 3: η-sweep for C3 (square_M_b3) with full off-diagonal A"
echo "    Started: $(date)"
$PYTHON thesis_results/run_eta_sweep.py square_M_b3 --n_modes 50
echo "    Completed: $(date)"

# ============================================================================
# STEP 4: η-sweep for C1 (hex_M_b1) with full off-diagonal A
# ============================================================================
echo ""
echo ">>> STEP 4: η-sweep for C1 (hex_M_b1) with full off-diagonal A"
echo "    Started: $(date)"
$PYTHON thesis_results/run_eta_sweep.py hex_M_b1 --n_modes 50
echo "    Completed: $(date)"

# ============================================================================
# STEP 5: T03 miniband dispersion for C3
# ============================================================================
echo ""
echo ">>> STEP 5: T03 miniband dispersion for C3 (square_M_b3)"
echo "    Started: $(date)"
$PYTHON results_bands/compute_miniband_structure.py \
    --h5 runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/phase2_multiband_data_c4sym.h5 \
    --outdir thesis_results/T03_miniband_dispersion/bz_dispersion \
    --nmodes 20 --nq 10
echo "    Completed: $(date)"

# ============================================================================
# STEP 6: T03 miniband dispersion for C1
# ============================================================================
echo ""
echo ">>> STEP 6: T03 miniband dispersion for C1 (hex_M_b1)"
echo "    Started: $(date)"
$PYTHON results_bands/compute_miniband_structure.py \
    --h5 runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/phase2_multiband_data_c2sym.h5 \
    --outdir thesis_results/T03_miniband_dispersion/hex_M_b1_bz_dispersion \
    --nmodes 20 --nq 10
echo "    Completed: $(date)"

# ============================================================================
# STEP 7: T11 validation for C3
# ============================================================================
echo ""
echo ">>> STEP 7: T11 validation for C3 (square_M_b3)"
echo "    Started: $(date)"
$PYTHON thesis_results/T11_miniband_validation/compute.py square_M_b3
echo "    Completed: $(date)"

# ============================================================================
# STEP 8: T11 validation for C1
# ============================================================================
echo ""
echo ">>> STEP 8: T11 validation for C1 (hex_M_b1)"
echo "    Started: $(date)"
$PYTHON thesis_results/T11_miniband_validation/compute.py hex_M_b1
echo "    Completed: $(date)"

echo ""
echo "=============================================="
echo "FULL-A RE-RUN PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "KEY OUTPUTS:"
echo "  C3 Phase 3: runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/phase3_mode_stats.json"
echo "  C1 Phase 3: runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/phase3_mode_stats.json"
echo "  C3 η-sweep: runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/eta_sweep_*/sweep_results.json"
echo "  C1 η-sweep: runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/eta_sweep_*/sweep_results.json"
echo "  T03 C3:     thesis_results/T03_miniband_dispersion/bz_dispersion/miniband_data.json"
echo "  T03 C1:     thesis_results/T03_miniband_dispersion/hex_M_b1_bz_dispersion/miniband_data.json"
echo ""
echo "Compare with backed-up diagA results:"
echo "  phase3_mode_stats_diagA.json"
echo "  phase3_multiband_modes_diagA.h5"
