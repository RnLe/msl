SHELL := /bin/bash

ENV_NAME ?= msl
MPIRUN ?= mpirun
NPROC ?= 4
RUN_DIR ?= auto
PHASE0_CONFIG ?= configs/phase0_real_run.yaml
PHASE1_CONFIG ?= configs/phase1_real_run.yaml
PHASE2_CONFIG ?= configs/phase2_real_run.yaml
PHASE3_CONFIG ?= configs/phase3_config.yaml
PHASE4_CONFIG ?= configs/phase4_config.yaml
PHASE5_CONFIG ?= configs/phase5_config.yaml
MAMBA ?= mamba
PYTHON ?= python

.PHONY: phase0
phase0:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase0_candidate_search.py $(PHASE0_CONFIG)

.PHONY: phase1_serial

.PHONY: phase0_grid

.PHONY: phase1_%
phase1_%:
	MSL_PHASE1_CANDIDATE_ID=$* $(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase1_local_bloch.py latest $(PHASE1_CONFIG)
phase0_grid:
	MSL_PHASE0_DISABLE_OPT=1 $(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase0_candidate_search.py $(PHASE0_CONFIG)

.PHONY: phase0_optimize
phase0_optimize:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase0_candidate_search.py optimize $(PHASE0_CONFIG)

.PHONY: phase1
phase1:
	$(MAMBA) run -n $(ENV_NAME) $(MPIRUN) -n $(NPROC) $(PYTHON) phases/phase1_local_bloch.py $(RUN_DIR) $(PHASE1_CONFIG)

.PHONY: phase1_serial
phase1_serial:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase1_local_bloch.py $(RUN_DIR) $(PHASE1_CONFIG)

.PHONY: phase2
phase2:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase2_ea_operator.py $(RUN_DIR) $(PHASE2_CONFIG)

.PHONY: phase3
phase3:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase3_ea_solver.py $(RUN_DIR) $(PHASE3_CONFIG)

.PHONY: phase4
phase4:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase4_validation.py $(RUN_DIR) $(PHASE4_CONFIG)

.PHONY: phase5
phase5:
	$(MAMBA) run -n $(ENV_NAME) $(MPIRUN) -n $(NPROC) $(PYTHON) phases/phase5_meep_qfactor.py $(RUN_DIR) $(PHASE5_CONFIG)

.PHONY: phase5_serial
phase5_serial:
	$(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase5_meep_qfactor.py $(RUN_DIR) $(PHASE5_CONFIG)

.PHONY: phase5_plots
phase5_plots:
	PHASE5_PLOTS_ONLY=1 $(MAMBA) run -n $(ENV_NAME) $(PYTHON) phases/phase5_meep_qfactor.py $(RUN_DIR) $(PHASE5_CONFIG)
