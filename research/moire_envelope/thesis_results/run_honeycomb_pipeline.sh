#!/bin/bash
# =============================================================================
# Honeycomb Candidate Pipeline Runner
# =============================================================================
# Runs the full Phase 1 → 2 → symmetrize(C6) → 3 → η-sweep pipeline
# for the honeycomb K-point Dirac cone candidate (photonic twisted bilayer graphene).
#
# Usage:
#   bash thesis_results/run_honeycomb_pipeline.sh
#   bash thesis_results/run_honeycomb_pipeline.sh --phase1    # only Phase 1
#   bash thesis_results/run_honeycomb_pipeline.sh --eta       # only η-sweep
#
# Must be run from: research/moire_envelope/

set -euo pipefail

PYTHON="/home/renlephy/.local/share/mamba/envs/msl/bin/python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

LABEL="honeycomb_K_b1"
CONFIG="configsV3/thesis_honeycomb_K_b1.yaml"

# Parse flags
DO_PHASE1=false
DO_PHASE2=false
DO_SYM=false
DO_PHASE3=false
DO_ETA=false

if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
    DO_PHASE1=true; DO_PHASE2=true; DO_SYM=true; DO_PHASE3=true; DO_ETA=true
else
    for arg in "$@"; do
        case "$arg" in
            --phase1) DO_PHASE1=true ;;
            --phase2) DO_PHASE2=true ;;
            --sym)    DO_SYM=true ;;
            --phase3) DO_PHASE3=true ;;
            --eta)    DO_ETA=true ;;
            *) echo "Unknown flag: $arg"; exit 1 ;;
        esac
    done
fi

# Find latest thesis run directory
RD=$(ls -1d runsV3/thesis_${LABEL}_* 2>/dev/null | sort | tail -1)
if [[ -z "$RD" ]]; then
    echo "ERROR: No run directory found for $LABEL."
    echo "Run: python thesis_results/setup_thesis_candidates.py"
    exit 1
fi

echo "============================================================"
echo "  HONEYCOMB PIPELINE: $LABEL"
echo "  Run dir: $RD"
echo "  Config:  $CONFIG"
echo "============================================================"
echo

# Phase 1: Local Bloch problems (~30 min at 128×128, res=64)
if $DO_PHASE1; then
    echo "--- Phase 1: Local Bloch Problems (MPB) ---"
    $PYTHON phasesV3/phase1_mpb_v3.py "$RD" "$CONFIG"
    echo
fi

# Phase 2: Berry connection + gauge fixing
if $DO_PHASE2; then
    echo "--- Phase 2: Berry Connection & Gauge Fixing ---"
    $PYTHON phasesV3/phase2_mpb_v3.py "$RD" "$CONFIG"
    echo
fi

# Symmetrization: C6 for honeycomb K-point
if $DO_SYM; then
    CAND_DIR="$RD/candidate_0000"
    echo "--- Symmetrize: C6 ---"
    $PYTHON thesis_results/symmetrize.py "$CAND_DIR" --sym C6
    echo
fi

# Phase 3: Envelope eigensolver
if $DO_PHASE3; then
    echo "--- Phase 3: Envelope Eigensolver ---"
    $PYTHON phasesV3/phase3_mpb_v3.py "$RD" "$CONFIG"
    echo
fi

# η-sweep: multiple twist angles
if $DO_ETA; then
    echo "--- η-sweep: Multiple Twist Angles ---"
    $PYTHON thesis_results/run_eta_sweep.py "$LABEL" \
        --angles 0.5 0.8 1.0 1.5 2.0 3.0 5.0 8.0 \
        --n_modes 50
    echo
fi

echo "============================================================"
echo "  HONEYCOMB PIPELINE COMPLETE"
echo "============================================================"
