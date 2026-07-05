# Benchmark Progress: EA Validation for Thesis

> Last updated: 2026-03-11 (Phase C redo Phase 1 running)

## Overview

Three benchmark workstreams to complete the thesis validation of the multi-band two-scale envelope approximation. Builds on the existing FDFD direct validation (θ≈1.1°, res=40) that established:
- Mean |Δω| = 23.2×10⁻⁶ (0.80% of EA bandwidth)
- BW ratio = 0.9845, KS p = 1.000

---

## Phase A: MPB Monolayer Resolution Convergence

**Goal:** Determine the minimum MPB resolution needed for converged monolayer band frequencies. This informs the `mpb_resolution` parameter for phases B and C.

- [x] **A1.** Create script `T_mpb_resolution/mpb_resolution_convergence.py`
- [x] **A2.** Run MPB on monolayer honeycomb unit cell (2-atom basis, TM) at k=K
  - Swept `mpb_resolution` ∈ {16, 32, 48, 64, 96, 128, 192, 256}
  - Recorded band frequencies for bands 1–8
- [x] **A3.** Generate convergence plot: frequency vs resolution + relative error from res=256
- [x] **A4.** Determine convergence characteristics
- [ ] **A5.** Document result — set `mpb_resolution` for remaining phases

### Phase A Results (2026-03-11)

**Convergence rate:** ~O(h²) (algebraic, not exponential). Richardson extrapolation confirms order p ≈ 2.0.

**Absolute frequency convergence** (Dirac bands, vs res=256 reference):

| res | max |Δω|/ω | Status |
|-----|-------------|--------|
| 16 | 3.70×10⁻³ | not converged |
| 32 | 8.33×10⁻⁴ | not converged |
| 48 | 4.24×10⁻⁴ | not converged |
| 64 | 2.02×10⁻⁴ | not converged |
| 96 | 9.31×10⁻⁵ | marginal |
| 128 | 4.13×10⁻⁵ | marginal |
| 192 | 1.12×10⁻⁵ | marginal |

**Bilayer difference convergence** (Λ = ω_bilayer − ω_monolayer, what EA uses):

| res | max |δ(Δω)| | Comparable to |
|-----|-------------|--------|
| 32 | 1.32×10⁻⁴ | 5× larger than EA-FDFD residual |
| 64 | 2.37×10⁻⁵ | ~equal to EA-FDFD residual (23×10⁻⁶) |
| 96 | 4.77×10⁻⁶ | ~5× below EA-FDFD residual |
| 128 | 6.18×10⁻⁶ | ~4× below EA-FDFD residual |

**Key insight:** Systematic MPB error partially cancels in bilayer−monolayer differences. At res=64, the difference error (~2.4×10⁻⁵) is comparable to the EA-FDFD residual (~2.3×10⁻⁵). The thesis production config (res=64) is therefore well-matched to the current accuracy level.

**Note:** This test probes frequency convergence only. Berry connection and Born-Huang terms depend on Bloch function spatial resolution (field derivatives), which has its own convergence behavior. A full EA pipeline convergence test (Phase B) is needed to assess the combined effect.

**Decision gate:** Phase A result determines `mpb_resolution` for B and C. Recommendation: res=32 for cheap scaling tests (Phase C), res=64 for accuracy-critical runs (Phase B).

---

## Phase B: EA Multi-Axis Resolution Convergence ✅

**Goal:** Systematically disentangle three resolution axes and show eigenvalue convergence.
**Completed:** 2026-03-11, runtime 1.40 hours (5027s), mpb_resolution=64 (from Phase A)

### B1: Registry Sampling Convergence ✅

Fix Ns=128, mpb_resolution=64.

- [x] **B1.1** Run full pipeline at `registry_samples=32` (1,024 MPB runs) — 156s
- [x] **B1.2** Run full pipeline at `registry_samples=64` (4,096 MPB runs) — 522s
- [x] **B1.3** Use existing thesis run at `registry_samples=128` (Phase 3 only) — 14s
- [x] **B1.4** Compare 50 EA eigenvalues at each registry value
- [x] **B1.5** Plot: eigenvalue convergence vs registry_samples
- [x] **B1.6** Determine if registry=128 is converged → **YES**, self-error ~3×10⁻⁴

**Result:** Non-monotonic convergence — reg=64 worse than reg=32 (aliasing/symmetry effect). reg=128 well-converged. EA-FDFD residual unchanged across all registries (~1000–1300×10⁻⁶), confirming FDFD-limited.

### B2: Hamiltonian Grid Ns Convergence ✅

Fix registry_samples=128, mpb_resolution=64. Phase 3 reruns only.

- [x] **B2.1** Run at Ns ∈ {32, 48, 64, 96, 128} — done
- [x] **B2.2** Extend to Ns=192 — done
- [x] **B2.3** Extend to Ns=256 — done
- [x] **B2.4** Plot: eigenvalue convergence vs Ns
- [x] **B2.5** Plateau confirmed by Ns=128 (error < 3×10⁻⁴)

**Result:** Power-law error ~Ns⁻⁰·⁵⁵, slower than O(h²). Non-monotonic pairwise rates. EA-FDFD residual ~1000–1450×10⁻⁶ independent of Ns.

### B3: "Honest" Combined Convergence (registry = Ns) ✅

- [x] **B3.1** registry=Ns=64 — resampled Phase 2 from B1 + Phase 3 — 3s
- [x] **B3.2** registry=Ns=128 — reused existing data — 15s
- [x] **B3.3** registry=Ns=192 — full pipeline (36,864 MPB runs) — 4144s (1.15h)
- [x] **B3.4** registry=Ns=256 — **skipped** (B3.3 confirmed convergent, not needed)
- [x] **B3.5** Compare each against FDFD(res=40) → mean|Δ| ≈ 1075–1347×10⁻⁶
- [x] **B3.6** EA–FDFD floor does NOT drop below 10⁻³ → confirms FDFD is the bottleneck

**Result:** Pairwise rate ~0.92 (first-order). BW narrows 4.9→2.9→2.6 mλ. Clean convergence.

### B4: Summary ✅

- [x] **B4.1** Multi-panel figure generated: `fig_phaseB_convergence.{png,pdf}`
- [x] **B4.2** All points compared against FDFD(res=40) — residual floor at ~10⁻³ independent of EA resolution
- [x] **B4.3** Results documented in `validation_summary.md` §5

### Phase B Key Conclusions

1. **EA internally convergent** — BW narrows from ~5.1 to ~2.6 mλ
2. **FDFD-limited** — EA-FDFD residual ~10⁻³ unchanged across all EA resolutions
3. **Production config (reg=128, Ns=128) is well-converged** — self-error ~3×10⁻⁴ ≪ FDFD residual ~10⁻³

---

## Phase C: EA → Monolayer Limit (IN PROGRESS — Phase 1 running)

**Goal:** Verify that EA eigenvalues recover monolayer band values as η→0 (zero twist).

### Phase C Redo: High-Resolution Phase 1 (running as of 2026-03-11)

**Status:** Phase 1 running (PID 4814), ETA ~2.9h. Phase 2+3 sweep deferred.

The original Phase C analysis (below) used existing sweep data at res=64, reg=128. The Phase C redo produces a high-resolution Phase 1 dataset at **res=128, reg=128** that serves two purposes:
1. **Thesis validation:** Higher-res Phase C sweep (32 angles, 0.1°–8.0°) — run separately later
2. **Research:** Universal high-res Bloch-field data for all downstream moiré analyses

**Parameters:**
- `mpb_resolution = 128` (up from 64)
- `mpb_registry_samples = 128`
- `phase1_Ns = 128`
- 16 multiprocessing workers, single-threaded MPB (OMP/BLAS flags set before all imports)
- 16,384 MPB eigensolves total

**Threading verification:** Each worker runs with nlwp=1 (single thread). OMP/BLAS/MKL flags are set at module level in both `phase_c_redo.py` and `phase1_mpb_v3.py` before any numpy import.

**Output:** `T_monolayer_limit/phase_c_redo_run/candidate_0000/phase1_multiband_data.h5`
- Expected bloch_fields shape: (128, 128, 6, 128, 128, 3) complex64 = 38.7 GB uncompressed
- Compressed (LZF) estimate: ~13 GB on disk

**Script:** `T_monolayer_limit/phase_c_redo.py`

### Original Phase C Analysis (completed 2026-03-11)

Pure analysis of 19 existing angles (res=64, reg=128) — no new computation.

### Existing Data Inventory

**19 angles computed** from previous eta sweeps (all with registry=128, mpb_res=64, Ns=128, 50 modes, C6-symmetrized Phase 2):
- `eta_sweep_20260307_194458`: θ ∈ {0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}
- `eta_sweep_20260307_225641`: θ ∈ {0.4, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95, 1.3, 1.7, 1.9}

### C1: Analytical Argument

- [ ] **C1.1** Write derivation (thesis writing task, not computational)
- [ ] **C1.2** Include in thesis validation section

### C2: Numerical Verification ✅

- [x] **C2.1** Reference $\omega_0 = 0.27436$ from Phase A (MPB res=256 at K)
- [x] **C2.2** Consolidated sweep results from 19 existing HDF5 files
- [x] **C2.3** Additional angles not needed — 19 angles sufficient
- [x] **C2.4** ~~C6 re-run~~ — data already C6-symmetrized
- [x] **C2.5** Plot: BW vs η (log-log) → α = 1.81, close to expected 2.0
- [x] **C2.6** Plot: ω_center vs θ → converges to 0.24277 ± 0.00001
- [x] **C2.7** Plot: band mixing vs θ → decreases as mixing ~ η^1.71
- [ ] **C2.8** Write monolayer limit analysis for thesis

### Phase C Key Results

1. **BW ~ η^1.81** — close to theoretical η², deviation likely from linear T_drift term
2. **ω_center converges** to 0.2428 (std = 1.3×10⁻⁵ across 11 angles ≤1°)
3. **Band mixing vanishes** — 0.83% at θ=0.4° → 19.6% at θ=8.0°
4. All data from existing sweeps; no new MPB computation needed

### Output Files

| File | Description |
|---|---|
| `T_monolayer_limit/phase_c_analysis.py` | Analysis script |
| `T_monolayer_limit/convergence_results_C.json` | All results (19 angles) |
| `T_monolayer_limit/fig_phaseC_monolayer_limit.{png,pdf}` | 3-panel figure |

---

## Execution Order

```
Phase A ✅ → mpb_resolution=64
    ↓
Phase B ✅ → EA internally convergent, FDFD-limited
    ↓
Phase C (initial) ✅ → BW ~ η^1.81 (19 angles, res=64)
    ↓
Phase C redo Phase 1 🔄 → high-res Bloch fields (res=128, reg=128) — running
    ↓
Phase C redo sweep (deferred) → Phase 2+3 at 32 angles with high-res data
    ↓
Remaining: C1 (analytical derivation) + thesis writing
```

---

## Key Files

| File | Purpose |
|---|---|
| `phasesV3/phase1_mpb_v3.py` | MPB band extraction, registry sweep |
| `phasesV3/phase2_mpb_v3.py` | Berry connection, BH potential, interpolation |
| `phasesV3/phase3_mpb_v3.py` | Hamiltonian assembly + eigsolve |
| `phasesV3/eta_sweep.py` | Full pipeline orchestrator |
| `configsV3/thesis_honeycomb_K_b1.yaml` | Thesis production config |
| `thesis_results/T_convergence/convergence_test.py` | Existing Ns/n_modes convergence |
| `thesis_results/T_convergence/phase_b_convergence.py` | Phase B runner (completed) |
| `thesis_results/T_convergence/convergence_results_B.json` | Phase B results |
| `thesis_results/T_convergence/fig_phaseB_convergence.{png,pdf}` | Phase B convergence plots |
| `thesis_results/T_monolayer_limit/` | Phase C analysis + results |
| `thesis_results/T_monolayer_limit/phase_c_redo.py` | Phase C redo: high-res Phase 1 (Phase 2+3 sweep deferred) |
| `thesis_results/T_direct_validation/` | FDFD solver + comparison scripts |
| `thesis_results/T_mpb_resolution/` | Phase A scripts + results |

---

## Resolution Parameter Reference

| Context | Ns | mpb_resolution | registry_samples | n_modes |
|---|---|---|---|---|
| Thesis production | 128 | 64 | 128 | 50 |
| Generic defaults | 128 | 32 | 64 | 6 |
| Phase A sweep | N/A | {16,32,48,64,96,128} | N/A | N/A |
| Phase B1 sweep | 128 | 64 | {32,64,128} | 50 |
| Phase B2 sweep | {32,48,64,96,128,192,256} | 64 | 128 | 50 |
| Phase B3 sweep | {64,128,192} | 64 | =Ns | 50 |
| Phase C sweep (existing) | 128 | 64 | 128 | 50 |
| Phase C redo (Phase 1) | 128 | **128** | **128** | N/A |
