#!/bin/bash
# Launch miniband convergence for honeycomb at all 3 angles
# Phase 2 is recomputed at each angle (theta-dependent)
# Background run with nohup

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/renlephy/.local/share/mamba/envs/msl/bin/python"
LOG="$SCRIPT_DIR/run_miniband_convergence.log"

echo "========================================" | tee "$LOG"
echo "  Miniband convergence run" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

cd "$SCRIPT_DIR"

$PYTHON convergence_miniband.py \
    --only honeycomb \
    --angles 0.5,1.1,3.0 \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "  Finished: $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
