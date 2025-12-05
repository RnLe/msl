# blaze_phasesV2 — BLAZE Pipeline V2
# ===================================
# This package contains the V2 pipeline phases for the BLAZE solver
# using fractional coordinates.
# See README_V2.md for the mathematical formulation and migration details.
#
# Pipeline Phases:
# - Phase 0: Candidate search (library-based or BLAZE MPB)
# - Phase 1: Local Bloch problems at frozen registry
# - Phase 2: Prepare envelope-approximation data
# - Phase 3: Envelope Approximation eigensolver
# - Phase 5: Meep-based cavity validation (Q-factor)
