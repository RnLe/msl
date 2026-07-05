# ─────────────────────────────────────────────────────────────────────────────
# Moiré Envelope Approximation — V3 pipeline runner
#
# Current generation: phasesV3/ (MPB) and blaze_phasesV3/ (Blaze2D), driven by
# configsV3/*.yaml. Older generations live in _archive/ (see _archive/README.md).
#
# Typical thesis-candidate workflow (see README.md for details):
#   make thesis_candidates                 # create runsV3/thesis_* run dirs
#   make phase1 CONFIG=configsV3/thesis_hex_M_b1.yaml RUN_DIR=<run_dir>
#   make phase2 CONFIG=configsV3/thesis_hex_M_b1.yaml
#   make symmetrize RUN_DIR=<run_dir>
#   make phase3 CONFIG=configsV3/thesis_hex_M_b1.yaml
#   make eta_sweep RUN_DIR=<run_dir>
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash

ENV_NAME ?= msl
MAMBA    ?= mamba
PYTHON   ?= python
RUN      := $(MAMBA) run -n $(ENV_NAME) $(PYTHON)

# "auto" = pick the latest matching run directory
RUN_DIR  ?= auto
CONFIG   ?=

.PHONY: help
help:
	@grep -E '^[a-z_0-9]+:' Makefile | grep -v '^help' | cut -d: -f1 | sort

# ── Phase 0: candidate search ────────────────────────────────────────────────
.PHONY: phase0 phase0_blaze
phase0:
	$(RUN) phasesV3/phase0_library_v3.py $(or $(CONFIG),configsV3/phase0_mpb.yaml)

phase0_blaze:
	$(RUN) blaze_phasesV3/phase0_library_v3.py $(or $(CONFIG),configsV3/phase0_blaze.yaml)

# ── Phase 1: local Bloch problems on the registry grid ───────────────────────
.PHONY: phase1 phase1_blaze
phase1:
	$(RUN) phasesV3/phase1_mpb_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase1_mpb.yaml)

phase1_blaze:
	$(RUN) blaze_phasesV3/phase1_blaze_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase1_blaze.yaml)

# ── Phase 2: EA operator assembly (Λ, v, M⁻¹, Berry A, Born–Huang Φ) ─────────
.PHONY: phase2 phase2_blaze
phase2:
	$(RUN) phasesV3/phase2_mpb_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase2_mpb.yaml)

phase2_blaze:
	$(RUN) blaze_phasesV3/phase2_blaze_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase2_blaze.yaml)

# ── Point-group symmetrization (C4/C2/C6, between Phase 2 and Phase 3) ───────
.PHONY: symmetrize
symmetrize:
	$(RUN) corrections_findings/S4b_c4_symmetrize.py $(RUN_DIR)

# ── Phase 3: envelope eigensolve ─────────────────────────────────────────────
.PHONY: phase3 phase3_blaze
phase3:
	$(RUN) phasesV3/phase3_mpb_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase3_mpb.yaml)

phase3_blaze:
	$(RUN) blaze_phasesV3/phase3_blaze_v3.py $(RUN_DIR) $(or $(CONFIG),configsV3/phase3_blaze.yaml)

# ── η-sweep: twist-angle sweep of the assembled Hamiltonian ──────────────────
.PHONY: eta_sweep
eta_sweep:
	$(RUN) phasesV3/eta_sweep.py --run-dir $(RUN_DIR)

# ── Thesis candidates: create runsV3/thesis_* run dirs (C1/C2/C3) ────────────
.PHONY: thesis_candidates
thesis_candidates:
	$(RUN) thesis_results/setup_thesis_candidates.py

# ── Tests ────────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(MAMBA) run -n $(ENV_NAME) pytest tests/ -v
