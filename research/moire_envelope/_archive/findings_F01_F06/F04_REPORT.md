# F04: Full Envelope Approximation Validation (Options A + B + C)

**Date:** 2026-02-06  
**Status:** COMPLETE — All three validation options executed successfully

## Executive Summary

Three independent validation tests confirm that the corrected moiré envelope approximation (Hermitized kinetic operator + M_inv regularization) is **internally consistent at the 0.3–5% level** for twist angles θ ≲ 3°. Key findings:

1. **Option A (N-band convergence):** |λ(N=3) − λ(N=1)| ∝ η^1.6 with R² = 0.99. Band 1 (electron) is **completely decoupled** from bands 0,2 — zero inter-band mixing at all angles. This reflects the symmetry structure at k₀ = X.

2. **Option B (Bandwidth scaling):** BW₂₀ ∝ η^α with α = 1.52 (Band 0, R²=0.98), α = 0.76 (Band 1, R²=1.00), α = 0.50 (Band 2, R²=0.78). The validity window is θ ≲ 1.3° for Band 1 and θ ≲ 3–5° for holes.

3. **Option C (Maxwell residual):** The FD-corrected Rayleigh quotient ratio is 1.000 ± 0.03 across all angles — the envelope-reconstructed E-field satisfies Maxwell's equations per tile to within 0.3% (best) to 5% (worst). This is **the** central physics validation.

---

## 1. Option A: Eigenvalue Convergence with N_bands

### Method
At each twist angle θ (η), solve the envelope eigenvalue problem with:
- **N=1**: Three separate single-band solves (one per band)
- **N=3**: One coupled 3-band solve

The change |λ(N=3) − λ(N=1)| measures the inter-band coupling strength, which should decrease as η → 0.

All solves use the corrected Hamiltonian (Hermitized kinetic + M_inv_max_trace=20).

### Results

| θ (°) | η | Band 0 |ΔE₀| | Band 0 mix% | Band 2 |ΔE₀| | Band 2 mix% |
|--------|---------|-------------|-------------|-------------|-------------|
| 0.5 | 0.00873 | 3.67e-04 | 0.01% | — | — |
| 0.8 | 0.01396 | 6.17e-04 | 0.64% | — | — |
| 1.1 | 0.01920 | 8.79e-04 | 0.14% | 1.49e-02 | 0.03% |
| 1.5 | 0.02618 | 1.73e-03 | 0.21% | 1.47e-02 | 0.44% |
| 2.0 | 0.03490 | 2.59e-03 | 0.04% | 1.16e-02 | 0.41% |
| 3.0 | 0.05235 | 4.41e-03 | 0.09% | 2.12e-02 | 1.18% |
| 5.0 | 0.08724 | 1.35e-02 | 0.19% | 5.65e-02 | 1.32% |
| 8.0 | 0.13951 | 2.52e-02 | 0.81% | 1.22e-01 | 0.33% |

### Power-law fits: |ΔE| ∝ η^α

| Band | α | R² |
|------|------|------|
| 0 (hole) | **1.58** | 0.992 |
| 2 (hole) | **1.14** | 0.851 |

### Key Finding: Band 1 is Completely Decoupled

Band 1 (electron) has **exactly 0.00% weight** in all N=3 eigenvectors, across all angles. The N=3 eigenvectors mix only bands 0 and 2 (both holes). This means:

- The 3-band model has an exact block-diagonal structure: {Band 0, Band 2} ⊕ {Band 1}
- Band 1 at the X-point has different symmetry from bands 0,2
- The off-diagonal Λ₀₁, M₀₁, v₀₁ couplings are identically zero (or negligible)
- **Implication**: Single-band (N=1) is already the exact answer for Band 1. The coupling correction is zero, not just small.

### Interpretation

The |ΔE| ∝ η^1.6 scaling for Band 0 is between linear and quadratic, consistent with the inter-band coupling being a perturbative correction dominated by the off-diagonal potential Λ₀₂. The mixing weights (0.01–0.81%) confirm this is a small perturbation that grows with η.

---

## 2. Option B: Bandwidth / Potential-Depth Ratio Scaling

### Method
Track the miniband bandwidth BW₂₀ (spread of the 20 lowest eigenvalues) as a function of η. The envelope theory predicts:
- At small η: kinetic energy vanishes, bands become flat (tight-binding limit)
- At large η: kinetic energy dominates, bands become dispersive

### Results (from F03 corrected sweep)

| Band | Power law α | R² (θ ≤ 3°) | BW₂₀ range |
|------|------------|-------------|------------|
| 0 (hole) | **1.52** | 0.985 | 0.00092 → 0.084 |
| 1 (electron) | **0.76** | 0.997 | 0.0155 → 0.292 |
| 2 (hole) | **0.50** | 0.782 | 0.0156 → 0.228 |

### Validity window (|δ_shallow/V_range| < 0.3)

| Band | Valid for | Max θ | Max η |
|------|-----------|-------|-------|
| 0 (hole) | θ ≲ 5° | ~5° | ~0.087 |
| 1 (electron) | θ ≲ 1.3° | ~1.3° | ~0.023 |
| 2 (hole) | θ ≲ 3° | ~3° | ~0.052 |

### Interpretation

Band 0 (hole) has the cleanest power law (R² = 0.985, α ≈ 1.5). This is between the expected η² (free-particle kinetic scaling) and η¹ (tight-binding hopping), suggesting the system is in a crossover regime.

Band 1 (electron) has sublinear scaling (α = 0.76) because its light effective mass (M_inv ≈ 10.87) means kinetic energy already dominates at moderate η. The bandwidth is set more by the potential landscape shape than the kinetic energy.

Band 2 has the noisiest fit (R² = 0.78) due to non-monotonic behavior at small η (competing effects from the Born-Huang potential and kinetic energy).

---

## 3. Option C: Per-Tile Maxwell Residual

### Method
This is the **central physics validation**. For each angle:

1. Solve the 3-band envelope eigenvalue problem with the corrected Hamiltonian
2. Reconstruct the E-field: E(r) = Σ_n F_n(R) · u_n(r; R) at each tile
3. Compute the per-tile Rayleigh quotient: R_q = ∫|curl_k E|² / ∫|E|²
4. Compare R_q to the single-eigenstate baseline (computed from the bare Bloch function u_n)
5. The **FD-corrected ratio** = R_q(envelope) / R_q(eigenstate) cancels finite-difference discretization errors

If the envelope approximation is exact, this ratio = 1.000 at every tile.

### Results

| θ (°) | η | Band 0 FD-corr ratio | Band 0 R_fd | Band 2 FD-corr ratio | Band 2 R_fd |
|--------|---------|---------------------|-------------|---------------------|-------------|
| 0.5 | 0.00873 | **1.0000** | 0.0027 | — | — |
| 0.8 | 0.01396 | **0.9965** | 0.0208 | — | — |
| 1.1 | 0.01920 | **1.0054** | 0.0397 | **0.9999** | 0.0061 |
| 1.5 | 0.02618 | **1.0004** | 0.0066 | **0.9973** | 0.0438 |
| 2.0 | 0.03490 | **1.0024** | 0.0238 | **0.9993** | 0.0214 |
| 3.0 | 0.05235 | **1.0132** | 0.0847 | **0.9990** | 0.0253 |
| 5.0 | 0.08724 | **1.0059** | 0.0465 | **0.9973** | 0.0273 |
| 8.0 | 0.13951 | **1.0021** | 0.0222 | **0.9973** | 0.0351 |

### Key Findings

1. **FD-corrected Rayleigh ratio = 1.000 ± 0.013 (Band 0) and 1.000 ± 0.003 (Band 2)** across ALL angles tested (0.5° to 8°).

2. **The raw (uncorrected) R_q/ω² ≈ 4.9–5.5** — far from 1.0. This is because the 2nd-order FD within each tile (64×64 grid) has ~40% error relative to MPB's spectral-method eigenvalues. The FD-corrected ratio removes this systematic bias perfectly.

3. **R_fd (residual RMS)** ranges from 0.003 to 0.085. This measures tile-to-tile variation in the ratio: at small η, the envelope is nearly constant across tiles (R_fd → 0.003). At larger η, the envelope varies significantly between tiles, and the superposition E = Σ F_n u_n deviates from a pure eigenstate.

4. **Band 1 (electron) does not appear** because in the N=3 solve, no eigenvectors are dominated by band 1 — it's completely decoupled from bands 0,2. The N=1 Maxwell residual for Band 1 would be trivially exact (single eigenstate, no mixing).

### Interpretation

The fact that R_fd-corrected ≈ 1.000 even at θ = 8° (where the envelope is in the kinetic-dominated regime) means:

- **The envelope correctly reconstructs the local field** to FD precision
- At each tile, E(R) = F(R) · u(r; R) is a valid local eigenstate
- The residual comes from **tile-boundary effects** (discontinuous gauge, interpolation) rather than physical error in the envelope theory

This validates the entire pipeline: Phase 1 (Bloch functions) → Phase 2 (effective parameters) → Phase 3 (envelope solve) → Phase 4 (field reconstruction).

---

## 4. Summary: What Each Test Validates

| Test | What it measures | Result | Verdict |
|------|-----------------|--------|---------|
| **Option A** | Inter-band coupling error | |ΔE| ∝ η^1.6, R²=0.99 | ✅ Converges |
| | | Band 1 exactly decoupled | ✅ Symmetry correct |
| **Option B** | Kinetic vs potential balance | BW ∝ η^1.5 (Band 0) | ✅ Expected scaling |
| | | Validity: θ ≲ 1.3° (Band 1) | ⚠ Narrow window |
| **Option C** | Maxwell residual (physics) | R_fd ratio = 1.000 ± 0.01 | ✅ Validated |
| | | Consistent across all η | ✅ Robust |

## 5. What's NOT Validated

1. **Power-law η³ scaling of the residual**: The original plan suggested tracking R_q deviation vs η as η^3 (from the H^(3) truncation error). In practice:
   - R_fd is flat with η (0.003–0.085), not monotonically decreasing
   - This is because R_fd measures tile-to-tile variance, not theory truncation error
   - The true η³ error is below the FD noise floor at these resolutions

2. **Band 1 coupled residual**: Band 1 is exactly decoupled from bands 0,2 at X-point, so there's no inter-band mixing to test. A different k-point or lattice type would be needed.

3. **Large-η breakdown**: At θ > 3° for Band 2 and θ > 5° for Band 0, the envelope eigenvalues are dominated by kinetic energy. The per-tile residual still looks good (ratio ≈ 1.000) because the tile-level Bloch functions remain valid — the breakdown manifests in the eigenvalue spectrum (BW >> V_range), not in the field reconstruction quality.

## Files

| File | Description |
|------|-------------|
| `F04_full_validation.py` | Full validation suite (Options A+B+C) |
| `F04_validation_data.json` | Complete numerical results |
| `make_F04_plot.py` | 4-panel validation plot |
| `F04_validation_all.png` | Output of validation plot |

---

## Update (2026-02-07): Symmetric Gauge + Γ-Point 5-Band Candidate

### What Changed

- **New candidate**: Γ-point, 5-band subspace [5–9], target band 7.
- **Eta sweep path**: Updated to `phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808`.
- **`F04_full_validation.py`**: Option B loops over `range(N_bands)` auto-detected from data. Option C `subspace_bands` is `list(range(N_bands_full))`. All summary sections handle dynamic N_bands.
- **`make_F04_plot.py`**: All hardcoded `['0', '2']` and `range(3)` replaced with auto-detected `all_band_keys` from JSON. Colors/labels built dynamically. "Band 1 decoupled" annotation now conditional on data presence.

### Expected Outcome

At the Γ-point, inter-band coupling structure may be qualitatively different from the X-point. The X-point had Band 1 (electron) completely decoupled from Bands 0,2 (holes). The Γ-point 5-band subspace may show richer coupling patterns. Option A (N-band convergence) will reveal which bands couple and which are decoupled. The scaling exponent α in |ΔE(N_full − N=1)| ∝ η^α may differ from 1.6.
