#!/bin/bash
# =============================================================================
# Thesis Pipeline Runner — Full Phase 1→2→sym→3→η-sweep for all candidates
# =============================================================================
# Usage:
#   ./thesis_results/run_pipeline.sh [--setup] [--phase1] [--phase2] [--sym] [--phase3] [--eta]
#   ./thesis_results/run_pipeline.sh --all
#
# If no flags given, runs the full pipeline.
# Must be run from: research/moire_envelope/

set -euo pipefail

PYTHON="/home/renlephy/.local/share/mamba/envs/msl/bin/python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

# Candidate labels
CANDIDATES=("hex_M_b1" "hex_M_b3" "square_M_b3")
CONFIGS=("configsV3/thesis_hex_M_b1.yaml" "configsV3/thesis_hex_M_b3.yaml" "configsV3/thesis_square_M_b3.yaml")

# Parse flags
DO_SETUP=false
DO_PHASE1=false
DO_PHASE2=false
DO_SYM=false
DO_PHASE3=false
DO_ETA=false

if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
    DO_SETUP=true; DO_PHASE1=true; DO_PHASE2=true; DO_SYM=true; DO_PHASE3=true; DO_ETA=true
else
    for arg in "$@"; do
        case "$arg" in
            --setup)  DO_SETUP=true ;;
            --phase1) DO_PHASE1=true ;;
            --phase2) DO_PHASE2=true ;;
            --sym)    DO_SYM=true ;;
            --phase3) DO_PHASE3=true ;;
            --eta)    DO_ETA=true ;;
            *) echo "Unknown flag: $arg"; exit 1 ;;
        esac
    done
fi

# ===========================================================================
# Find or create thesis run directories
# ===========================================================================
find_thesis_run() {
    local label="$1"
    # Find latest thesis run directory for this candidate
    local match
    match=$(ls -1d runsV3/thesis_${label}_* 2>/dev/null | sort | tail -1)
    if [[ -z "$match" ]]; then
        echo ""
    else
        echo "$match"
    fi
}

echo "============================================================"
echo "  THESIS PIPELINE RUNNER"
echo "============================================================"
echo "Working directory: $BASE_DIR"
echo

# Step 0: Setup candidate run directories
if $DO_SETUP; then
    echo "[SETUP] Creating thesis candidate run directories..."
    $PYTHON thesis_results/setup_thesis_candidates.py
    echo
fi

# Discover run directories
declare -A RUN_DIRS
for label in "${CANDIDATES[@]}"; do
    rd=$(find_thesis_run "$label")
    if [[ -z "$rd" ]]; then
        echo "ERROR: No run directory found for $label. Run with --setup first."
        exit 1
    fi
    RUN_DIRS[$label]="$rd"
    echo "  $label → $rd"
done
echo

# Step 1: Phase 1 — Local Bloch problems (expensive: ~30 min each at 128×128, res=64)
if $DO_PHASE1; then
    echo "============================================================"
    echo "  PHASE 1: Local Bloch Problems (MPB)"
    echo "============================================================"
    for i in "${!CANDIDATES[@]}"; do
        label="${CANDIDATES[$i]}"
        config="${CONFIGS[$i]}"
        rd="${RUN_DIRS[$label]}"
        echo
        echo "--- Phase 1: $label ---"
        echo "  Run dir: $rd"
        echo "  Config:  $config"
        $PYTHON phasesV3/phase1_mpb_v3.py "$rd" "$config"
    done
    echo
fi

# Step 2: Phase 2 — Berry connection + gauge fixing
if $DO_PHASE2; then
    echo "============================================================"
    echo "  PHASE 2: Berry Connection & Gauge Fixing"
    echo "============================================================"
    for i in "${!CANDIDATES[@]}"; do
        label="${CANDIDATES[$i]}"
        config="${CONFIGS[$i]}"
        rd="${RUN_DIRS[$label]}"
        echo
        echo "--- Phase 2: $label ---"
        $PYTHON phasesV3/phase2_mpb_v3.py "$rd" "$config"
    done
    echo
fi

# Step 3: Symmetrization (C4 for square, C2 for hex M-point)
if $DO_SYM; then
    echo "============================================================"
    echo "  SYMMETRIZATION"
    echo "============================================================"
    for label in "${CANDIDATES[@]}"; do
        rd="${RUN_DIRS[$label]}"
        cand_dir="$rd/candidate_0000"
        echo
        echo "--- Symmetrize: $label ---"
        if [[ "$label" == square_* ]]; then
            echo "  Applying C4 symmetrization..."
            $PYTHON thesis_results/symmetrize.py "$cand_dir" --sym C4
        else
            echo "  Applying C2 symmetrization (hex M-point)..."
            $PYTHON thesis_results/symmetrize.py "$cand_dir" --sym C2
        fi
    done
    echo
fi

# Step 4: Phase 3 — Envelope eigensolver
if $DO_PHASE3; then
    echo "============================================================"
    echo "  PHASE 3: Envelope Eigensolver"
    echo "============================================================"
    for i in "${!CANDIDATES[@]}"; do
        label="${CANDIDATES[$i]}"
        config="${CONFIGS[$i]}"
        rd="${RUN_DIRS[$label]}"
        echo
        echo "--- Phase 3: $label ---"
        $PYTHON phasesV3/phase3_mpb_v3.py "$rd" "$config"
    done
    echo
fi

# Step 5: η-sweep (multiple twist angles)
if $DO_ETA; then
    echo "============================================================"
    echo "  η-SWEEP: Multiple Twist Angles"
    echo "============================================================"
    for label in "${CANDIDATES[@]}"; do
        rd="${RUN_DIRS[$label]}"
        echo
        echo "--- η-sweep: $label ---"
        # The eta_sweep needs to find the Phase 1 data in the run directory.
        # We need to set it as the "latest" or pass explicitly.
        $PYTHON thesis_results/run_eta_sweep.py "$rd" \
            --angles 0.5 0.8 1.0 1.5 2.0 3.0 5.0 8.0 \
            --n_modes 100
    done
    echo
fi

echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo "Run T01-T10 analysis scripts next:"
echo "  python thesis_results/T01_candidate_selection/compute.py"
echo "  python thesis_results/T02_hamiltonian_landscape/compute.py"
echo "  ..."
