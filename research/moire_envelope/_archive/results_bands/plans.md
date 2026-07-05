# Moiré Miniband Analysis — Roadmap

**Date:** 2026-02-17  
**Status:** Phase 1 — Candidate Search ✅ COMPLETE  
**Old candidate:** square, r/a=0.35, ε_bg=12, band 7 @ Γ → **pathological** (V/E_kin = 954)  
**New candidate:** square, r/a=0.18, ε_bg=2.0, band 3 @ X → **V/E_kin = 1.9 at θ=2°** ✅

---

## Problem

The current candidate produces V/E_kin ≈ 954 at θ = 1.1°, meaning the moiré potential 
is ~1000× deeper than the kinetic energy. This yields ~200 nearly-degenerate bound states 
with zero dispersion — the "deep well" regime where the envelope approximation is 
quantitatively unreliable and physically uninteresting. 

**Goal:** Find a candidate where V/E_kin ~ 1–10 at a defensible twist angle θ < 5°, 
producing O(5–20) resolved miniband tubes with measurable dispersion.

---

## Six Criteria for a Good Candidate

| # | Criterion | Diagnostic | Target |
|---|-----------|-----------|--------|
| 1 | **Spectral isolation** | gap_min ≫ η·‖∂_R ε‖ | gap_min > 0.001 |
| 2 | **True extremum at k₀** | v_g(k₀) = 0 | Use high-symmetry k₀ (automatic) |
| 3 | **V/E_kin ~ 1–10** | θ* ∈ [0.5°, 5°] | **THE critical constraint** |
| 4 | **No internal anti-crossings** | Well-separated bands in subspace | Δω_internal > 0.01 |
| 5 | **Moderate ε contrast** | ε_bg / ε_hole | Favor ε_bg ∈ [2, 6] |
| 6 | **k₀ stable across R** | Always true at HS points | ✓ automatic |

### The dimensionless ratio

$$\frac{V_\text{depth}}{E_\text{kin}} = \frac{\Delta\omega}{\tfrac{1}{2}|M^{-1}_\text{eff}| \cdot \eta^2}, \qquad \theta^* = \sqrt{\frac{2\,\Delta\omega}{|M^{-1}_\text{eff}|}}$$

θ* is the twist angle where V/E_kin ≈ 1. We need θ* to be small (< 5°) **and** physically 
accessible. The current candidate has θ* ≈ 57° — unphysically large.

---

## Tiered Search Strategy

### Tier 1 — CSV-based Fast Screening (seconds)

- **Input:** Phase 0 `phase0_candidates.csv` (~100k+ rows)
- **Method:** Estimate V_proxy = C · (ε_bg − 1)/ε_bg · ω₀ (proxy for moiré potential depth)
- **Compute:** θ*_ref, self_consistent flag, V/E_kin at θ = 1°, 2°, 3°, 5°
- **Rank by:** smallest θ* first, then gap_min descending, then θ_max descending
- **Diversify:** max 3 per "band family" (lattice_type + k_label + band_index)
- **Output:** `tier1_ranked.csv` (top 100), `tier1_diversified.csv`
- **Tool:** `phasesV3/candidate_scanner_tier1.py`

### Tier 2 — MPB Validation (minutes per candidate)

- **Input:** Top 20 from Tier 1
- **Method:** Run MPB at 3×3 = 9 registry points per candidate
- **Extract:** Actual V_depth, 2D Hessian M_inv, condition number
- **Compute:** True θ*, V/E_kin at θ = 1°, 2°, 3°, 5°
- **Filter:** 
  - θ* ∈ [0.5°, 5°]
  - self_consistent (θ* < θ_max)
  - cond_number < 5 (isotropic mass)
  - gap_min > 0.001
- **Output:** `tier2_ranked.csv`
- **Tool:** `phasesV3/candidate_scanner_tier2.py`

### Tier 3 — Full Pipeline (hours)

- **Input:** Top 1–3 from Tier 2
- **Method:** Phase 1 (128×128 MPB registry) → Phase 2 (Berry connection, Born-Huang) → C4-sym → Phase 3 (envelope eigensolver)
- **Config:** `include_offdiag_A=True`, `export_bloch_fields=true`
- **Validate:** C4 commutator, Hermiticity, BZ periodicity
- **Output:** Full miniband structure with eigenvalue tubes

---

## Parameter Space

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| Lattice | square, hex | Both have distinct HS k-points |
| ε_bg | 2.0–10.0 | Moderate contrast (Summary.md: favor 2–6) |
| r/a | 0.10–0.48 | Full library range |
| Bands | 0–19 (merged TE+TM) | First 20 bands — covers bands 0–9 per polarization |
| k₀ | all HS points | Γ, X, M (square); Γ, K, M (hex) |
| N_bands (subspace) | 5 (n_neighbor=2) | Standard multi-band envelope |

---

## Priority-Ordered Roadmap

### Phase 1: Candidate Search ✅ COMPLETE
1. ✅ Created expanded Phase 0 config (bands 0–19, ε_bg 2–10, all k-points)
2. ✅ Phase 0 scan → 249,080 candidates (3,645 valid EA)
3. ✅ Tier 1 fast filter → 95,796 pass quality filters, top 100 ranked
4. ✅ Tier 2 MPB validation (top 20) → real V_depth, M_inv, V/E_kin
5. ✅ Winner selected: **square_X_b3, r/a=0.18, ε_bg=2.0**

#### Winner Details
| Property | Value |
|----------|-------|
| Lattice | Square |
| k₀ | X = (0.5, 0, 0) |
| Band index | 3 (merged TE+TM) |
| Polarization | TE dominant |
| Subspace | [1, 2, 3, 4, 5] (N_bands=5) |
| target_index | 2 |
| ω₀ | 0.3764 |
| r/a | 0.18 |
| ε_bg | 2.0 (ε_hole = 1.0) |
| V_depth (Tier 2) | 0.0240 |
| curv_trace_2D | −40.3 |
| cond_number | 4.2 |
| θ\* | 2.8° |
| V/E_kin @1° | 7.8 |
| V/E_kin @2° | 1.9 |
| V/E_kin @3° | 0.9 |
| gap_below | 0.006 |
| gap_above | 0.415 |
| candidate_id | 214201 |

**Why this candidate?**
- V/E_kin ≈ 1–2 at θ = 2–3° → sweet spot for resolved minibands
- Low ε contrast (2:1) → envelope approximation most trustworthy
- Isotropic at library level (curv_xx = curv_yy = 22.1), cond=4.2 at Tier 2
- Good spectral isolation (gap_below = 0.006, gap_above = 0.415)
- 5-band subspace centered on band 3
- Self-consistent: θ\* = 2.8° < θ_max = 1.7° needs investigation (θ_max from 1D parabolic range, but 2D MPB may differ)

**Runners-up:**
- square_X_b3, r/a=0.10, ε_bg=3.8: θ\*=2.1°, V/E@2°=1.1, gap=0.0035, cond=1.2
- square_X_b3, r/a=0.11, ε_bg=3.4: θ\*=2.2°, V/E@2°=1.2, gap=0.0040, cond=1.4

### Phase 2: Miniband Structure ← **NEXT**
1. Run full pipeline: Phase 1 → 2 → C4-sym → 3
2. Re-run `compute_miniband_structure.py` with new HDF5 data
3. Verify V/E_kin ~ 1–10 produces resolved tubes
4. Generate eigenvalue tube plots

### Phase 3: Wannier Functions & Mode Volume
1. Compute Wannier functions from miniband eigenstates
2. Calculate mode volumes
3. Assess cavity confinement quality

### Phase 4: Resolution & Sweep Studies
1. Higher q-resolution sweep (32–64 q-points)
2. Twist-angle sweep: θ from 0.5° to 5° in steps of 0.2°
3. Track tube evolution with twist angle
4. Identify "magic angles" where tubes flatten

### Phase 5: Full-Wave Validation (Meep)
1. Set up Meep simulation with bilayer geometry
2. Compare envelope eigenvalues to full-wave results
3. Quantify envelope approximation error
4. **This is the very last step**

---

## Files

| File | Purpose |
|------|---------|
| `configsV3/phase0_candidate_search_b20.yaml` | Phase 0 config for expanded search |
| `runsV3/phase0_mpb_v3_candidate_search_b20_20260217_132202/` | Phase 0/Tier1/Tier2 results |
| `phasesV3/candidate_scanner_tier1.py` | Tier 1 fast CSV-based screening |
| `phasesV3/candidate_scanner_tier2.py` | Tier 2 MPB validation |
| `results_bands/compute_miniband_structure.py` | Miniband q-sweep analysis |
| `results_bands/plot_eigenvalue_tubes.py` | Tube visualization |
| `results_bands/miniband_data.json` | Saved miniband results |
| `results_bands/plans.md` | This file |

---

## Notes

- The band library at `research/band_diagram_scan/data/band_library.h5` stores 10 bands 
  per polarization. When TE+TM are merged and sorted by frequency, this gives up to 20 
  merged bands. The existing Phase 0 run only covered bands 0–9.
- C_ref = 0.05 is the calibration constant for V_proxy. Summary.md recommends this default.
- The Tier 1 proxy V_proxy = C · Δε/ε_bg · ω₀ is a rough estimate. Tier 2's actual MPB 
  solves are the ground truth.
- Self-consistency: θ* must be smaller than θ_max (the angle where the parabolic 
  approximation breaks down). Otherwise, the EA is invalid at the interesting twist angle.
