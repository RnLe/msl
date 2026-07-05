# Research Data Manifest

This repository publishes the **methods** (code, configs, docs) of the thesis
research in full. The **raw data** produced by the pipeline totals ~185 GB and
is not tracked by git. This file documents what exists, where it lives on the
research machine, how it was generated, and what small curated subset *is*
committed.

**Availability:** the full raw datasets are available from the author on
request (rene.marcel.lehner@gmail.com). All of them are regenerable from the
committed code and configs; approximate regeneration commands are listed below.

---

## 1. What is committed (curated subset)

Tracked binary/data files are added explicitly with `git add -f` against the
global `*.npz` / `*.h5` / `*.pkl` ignore rules. Selection rule: **< 10 MB and
either feeds a thesis figure or is a summary table/fit**.

| Committed set | Contents |
| --- | --- |
| `moire_envelope/thesis_results/*.npz` (< 10 MB each) | Eigenvalue-ladder comparisons (EA vs FDFD vs MPB vs hybrid), solver cross-checks, term audits |
| `moire_envelope/thesis_results/figures/` | Final composite figures (PNG) |
| `moire_envelope/thesis_results/T*/` small files | Per-study summaries: JSON/CSV fits, scaling-law tables, magic-angle scan tables, small NPZ |
| `moire_envelope/runsV3/*/phase0_config.json`, `phase0_candidates.csv` | Provenance of every Phase-0 run (candidate definitions) |
| `moire_envelope/runsV3/thesis_*/eta_sweep_*/sweep_config.json`, `sweep_results.json` | η-sweep (twist-angle sweep) results behind the bandwidth-scaling and miniband analyses |
| `moire_envelope/phasesV4/mpb_blaze_validation/output/*.png` | MPB ↔ Blaze2D validation report figures |
| `studies/fdfd_convergence/figures/*.svg` | FDFD convergence study figures (thesis-grade) |

## 2. Local-only data (not in git)

### 2.1 Pipeline runs — `moire_envelope/runsV3/` (~88 GB)

Each run directory contains Phase 0–3 outputs: candidate CSVs, per-candidate
`phase1_multiband_data.h5` (Bloch eigenvectors on the registry grid — the
dominant cost), Phase-2 operator fields, Phase-3 envelope modes, η-sweeps.

| Run directory | Size | Role |
| --- | --- | --- |
| `thesis_square_M_b3_20260209_173724` | 13 GB | **C3 candidate** (square, M, 5 bands) — complete Phase 0–3 + C4 sym + η-sweep |
| `thesis_hex_M_b1_20260209_173724` | 15 GB | **C1 candidate** (hex, M, 4 bands) — complete Phase 0–3 + C2 sym + η-sweep |
| `thesis_honeycomb_K_b1_20260307_171424` | 9.3 GB | **C_hc candidate** (honeycomb, K, Dirac pair) — incl. magic-angle fine sweep |
| `thesis_honeycomb_K_b1_TE_20260308_001513` | 7.3 GB | Honeycomb TE variant (tangential-field run) |
| `thesis_hex_M_b3_20260209_173724` | 20 KB | C2 candidate — Phase 0 only (deferred) |
| `phase0_mpb_v3_allk_scan_20260209_152023` | 349 MB | All-k screening scan (feeds T01 candidate selection) |
| `phase0_mpb_v3_candidate_search_b20_20260217_132202` | 1.3 GB | 20-band candidate search (feeds T01) |
| `phase0_mpb_v3_2026020*` (5 dirs) | ~44 GB | Earlier Phase-0 scans (incl. outdated Γ-point candidate) |
| `phase0_blaze_v3_20260203_001132` | 246 MB | Blaze2D-based Phase-0 comparison run |

Regenerate: `thesis_results/setup_thesis_candidates.py`, then per candidate
`phasesV3/phase1_mpb_v3.py <run_dir> configsV3/thesis_<label>.yaml` → phase2 →
`corrections_findings/S4b_c4_symmetrize.py` → phase3 → `phasesV3/eta_sweep.py`.
See `moire_envelope/README.md`.

### 2.2 Validation studies — `moire_envelope/thesis_results/` (~81 GB untracked part)

| Directory | Size | Contents |
| --- | --- | --- |
| `T_direct_validation/` (subdirs) | 52 GB | FDFD supercell reference solves (H5/NPZ per angle/resolution), overnight validation campaigns. Top-level FDFD solver code (`fdfd_solver.py`, `supercell_geometry.py`) **is** tracked |
| `T_monolayer_limit/` (subdirs) | 17 GB | Monolayer-limit consistency check (single 17 GB Phase-1 H5) |
| `T_convergence/` (subdirs) | 12 GB | Resolution-convergence H5 stacks |
| root `*.npz` ≥ 10 MB | ~1.2 GB | Full-eigenvector audit files (`2ret_*_reg64_*`, `exact_te_x_audit_*`, `eps_maps_*`, larger `fdfd_te_x_*`) |

### 2.3 FDFD convergence study — `studies/fdfd_convergence/` (~2.2 GB)

`data_eps_maps/` 1.6 GB, `data_gamma_tm_hex/` 330 MB, `tm_commensurate_phase2/`
234 MB, plus ~10 smaller `data_*` dirs (NPZ per angle × resolution × target
frequency). Regenerate: `run_convergence.py` (batch runner, skip-on-exist).

### 2.4 Legacy data — `moire_envelope/_archive/` (~1.5 GB)

`findings_F01_F06/` 765 MB, `runsV2/` 554 MB, `runs_v1/` 173 MB,
`results_R01_R06/` 49 MB. Outputs of the archived V1/V2 pipelines; kept for
provenance only. See `_archive/README.md`.

### 2.5 Out-of-repo working copies — `_local/` (never committed)

| Path | Contents |
| --- | --- |
| `_local/thesis/` | Pre-submission in-repo thesis working copy; superseded by the separate thesis repository |
| `_local/root_v4_sprint/` | Final-sprint working copies (`common/`, `magic_angle_output/`, `runsV3/`) of the V4 Blaze-based studies whose canonical home is the **blaze2d** repository (`thesis/research/phasesV4/studies/`) |
| `_local/build_src/` | Meep/MPB source trees used to build the simulation environment (build scripts: `research/scripts/`) |

---

*Sizes measured 2026-07-05. Directory layout after the July 2026 cleanup; see
`research/README.md` for the research map.*
