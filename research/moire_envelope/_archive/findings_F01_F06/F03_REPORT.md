# F03: Kinetic Operator Prefactor & Hermiticity Investigation

**Date:** 2025-02-07  
**Status:** RESOLVED — Three issues found, two bugs, one physical limitation

## Executive Summary

Investigation of the η² prefactor in the kinetic operator revealed:

1. **η² cancellation is CORRECT** — the code's prefactor `0.5/(2π)² ≈ 0.01267` is dimensionally correct (verified to machine precision by comparing dimensionless-theory vs physical-code coordinates)
2. **The kinetic operator is NON-HERMITIAN** — `M(x)·L` ≠ `L·M(x)` when M varies in space. The non-Hermiticity is 32% at θ=0.5° and 57% at θ=5°. This is a code bug.
3. **M_inv has 1166% spatial variation** — driven by near-degeneracy of bands 0 and 1 at 4.2% of grid points (gap₀₁ → 0). This is a physical limitation of the single-band model.

## 1. The η² Question (RESOLVED)

### Background
The theory (doc §8-9) gives the kinetic term as:
$$\hat{H}^{(2)} \ni \frac{1}{2} M^{-1}_{ij}(\mathbf{R}) (-i\mathcal{D}_i)(-i\mathcal{D}_j)$$

with an explicit η² prefactor in the full equation `Λ + η H^(1) + η² H^(2)`.

The code (phase3_mpb_v3.py line 497) uses `prefactor = 0.5/(2π)²` with NO explicit η².

### Resolution

The η² from the theory cancels against `L² = (a/η)²` when converting the dimensionless derivative ∂/∂R (theory's coordinate) to the physical derivative ∂/∂x (code's coordinate):

```
Theory: η² · ½ M^{-1}_theory · (-i∂_R)²
      = η² · ½ M^{-1}_theory · L² · (-i∂_x)²     [since ∂_R = L·∂_x]
      = η² · ½ · 2f₀·M^{-1}_MPB · (1/η²)·(-i∂_x)²   [L=1/η, M_theory=2f₀M_MPB]
      = f₀ · M^{-1}_MPB · (-i∂_x)²

Dividing by 8π²f₀ (to convert λ̃-eigenvalue to f-eigenvalue):
      = M^{-1}_MPB/(8π²) · (-i∂_x)²
      = [0.5/(2π)²] · M^{-1}_MPB · (-i∂_x)²   ✓
```

**Verified numerically**: Building H in dimensionless s-coordinates (theory's formulation with η²) vs physical x-coordinates (code's formulation without η²) gives identical eigenvalues to machine precision (max diff = 7.99e-15).

### Why Earlier Tests Seemed to Need η²

When we manually multiplied M_inv by η² in earlier tests, the kinetic energy was artificially suppressed by ~η²≈0.008, making eigenvalues stay near the potential. This looked "physical" but was actually underpredicting the kinetic energy by 100×. The correct physics (without extra η²) gives large kinetic energy because M_inv is huge at hot spots.

## 2. Non-Hermiticity Bug (FOUND)

### The Problem

The code builds the kinetic operator as:
```python
K = -prefactor * (M11_diag @ L1_full + M22_diag @ L2_full + 2*M12_diag @ D1@D2)
```

This computes `M(x) · ∇²ψ`, which is NOT self-adjoint when M varies in space:
```
(M·L)† = L†·M† = L·M ≠ M·L
```

Measured non-Hermiticity: `|H - H†|/|H|` = 32% at θ=0.5° and 57% at θ=5°.

### Physical Origin

The parent operator L^(2) = -∇_R · (ε^{-1} ∇_R) IS manifestly self-adjoint (divergence form). After projection onto the band subspace, the kinetic part should remain Hermitian. The notation `M^{-1}(-iD)(-iD)` in the theory is implicitly **Weyl-ordered** (symmetrized):

$$K = \frac{1}{4}[M^{-1}_{ij}(-iD_i)(-iD_j) + (-iD_j)(-iD_i)M^{-1}_{ij}]$$

### Fix

Symmetrize the kinetic operator: `K_sym = (K + K†)/2`. However, at large M variation this produces negative eigenvalues in the kinetic-only operator, indicating the approximation breaks down.

## 3. M_inv Divergence at Near-Degeneracy Points (PHYSICAL LIMITATION)

### The Problem

Band 1's effective mass varies by 30× (trace: 4.24 to 131.05, mean 10.87) across the moiré cell. This variation is **independent of θ** — it's an intrinsic property of the band structure at k₀=(0.5,0).

### Root Cause

The 4.2% of grid points where M_trace > 50 correspond to positions where **bands 0 and 1 are nearly degenerate**: gap₀₁ = 0.004 at hot spots vs 0.088 elsewhere. This is the standard k·p mass divergence: M^{-1} ∝ v²/Δ where Δ→0.

### Consequence

The single-band envelope approximation REQUIRES that the chosen band be well-separated from all other bands across the entire moiré cell. When the gap closes, the single-band mass diverges and:
- Kinetic energy at hot spots overwhelms the potential
- The Hamiltonian becomes effectively non-Hermitian (due to M·L ordering)
- Eigenvalues become unphysical

### Solution Path

1. **Regularize M_inv**: Clamp |M^{-1}| to a maximum value (loses accuracy at hot spots)
2. **Use multi-band model**: Include bands 0+1 together (correct physics but more complex)
3. **Exclude hot-spot regions**: Mask out near-degeneracy points (ad hoc)

## 4. Summary of Energy Scales

| θ (deg) | η | L (a) | T₁/V_range | Regime |
|---------|------|-------|------------|--------|
| 0.5 | 0.0087 | 114.6 | 0.004 | Deep tight-binding ✓ |
| 0.8 | 0.0140 | 71.6 | 0.011 | Tight-binding ✓ |
| 1.5 | 0.0262 | 38.2 | 0.039 | Intermediate ✓ |
| 3.0 | 0.0524 | 19.1 | 0.155 | Intermediate ⚠ |
| 5.0 | 0.0872 | 11.5 | 0.429 | Kinetic comparable ⚠ |
| 8.0 | 0.1395 | 7.2 | 1.097 | Kinetic dominates ❌ |

Where T₁ = (mean M_inv) / (2L²) is the kinetic energy of the first moiré Bloch state, and V_range is the potential modulation depth.

## 5. Numerical Verification Data

### Full diag eigenvalues at Ns=16 (single-band, band 1):

| θ | E_min (code) | E_min (theory dim'less) | |E_code - E_theory|_max |
|------|-------------|----------------------|----------------------|
| 0.5° | 0.041841 | — | — |
| 1.5° | 0.069145 | — | — |
| 5.0° | 0.233317 | 0.078750 (matches code ✓) | 7.99e-15 |
| 8.0° | 0.445653 | — | — |

### M_inv statistics per band:

| Band | Type | Mean | Min | Max | Var% | Neg eig% |
|------|------|------|-----|-----|------|----------|
| 0 | hole | -9.32 | -130.1 | -0.31 | 1392% | 100% |
| 1 | electron | 10.87 | 4.24 | 131.05 | 1166% | 0% |
| 2 | hole | -10.30 | -111.4 | -2.19 | 1060% | 100% |

---

## 6. Corrected Sweep Results (Post-Fix)

After applying both fixes (kinetic Hermitization + M_inv regularization with max_trace=20), a full 8-angle × 3-band sweep was run at Ns=128.

**All 24 solves converged.** No eigenvalues below V_min.

### Per-band results table

| θ (°) | η | Band | E₀ (c/a) | BW₂₀ (c/a) | δ/V_range | Status |
|--------|---------|------|-----------|-------------|-----------|--------|
| 0.5 | 0.00873 | 0 (h) | 0.09093 | 0.00092 | −0.004 | ✓ |
| 0.5 | 0.00873 | 1 (e) | 0.03697 | 0.01546 | +0.122 | ✓ |
| 0.5 | 0.00873 | 2 (h) | 0.33000 | 0.01563 | +0.045 | ✓ |
| 0.8 | 0.01396 | 0 (h) | 0.09061 | 0.00160 | −0.006 | ✓ |
| 0.8 | 0.01396 | 1 (e) | 0.04427 | 0.02174 | +0.198 | ✓ |
| 0.8 | 0.01396 | 2 (h) | 0.32948 | 0.02426 | −0.019 | ✓ |
| 1.5 | 0.02618 | 0 (h) | 0.08948 | 0.00391 | −0.015 | ✓ |
| 1.5 | 0.02618 | 1 (e) | 0.06217 | 0.03533 | +0.383 | ⚠ |
| 1.5 | 0.02618 | 2 (h) | 0.33675 | 0.02833 | −0.107 | ✓ |
| 3.0 | 0.05235 | 0 (h) | 0.08465 | 0.01399 | −0.056 | ✓ |
| 3.0 | 0.05235 | 1 (e) | 0.11222 | 0.06135 | +0.902 | ⚠ |
| 3.0 | 0.05235 | 2 (h) | 0.32729 | 0.04473 | −0.161 | ✓ |
| 5.0 | 0.08724 | 0 (h) | 0.07192 | 0.03878 | −0.148 | ✓ |
| 5.0 | 0.08724 | 1 (e) | 0.21506 | 0.11103 | +1.968 | ❌ kinetic |
| 5.0 | 0.08724 | 2 (h) | 0.29309 | 0.10507 | −0.364 | ⚠ |
| 8.0 | 0.13951 | 0 (h) | 0.05042 | 0.08407 | −0.330 | ⚠ |
| 8.0 | 0.13951 | 1 (e) | 0.41881 | 0.29213 | +4.080 | ❌ kinetic |
| 8.0 | 0.13951 | 2 (h) | 0.23610 | 0.22766 | −0.874 | ❌ kinetic |

Legend: ✓ = valid envelope regime, ⚠ = marginal (0.3 < |δ/V| < 1), ❌ = kinetic-dominated (|δ/V| > 1)

### Power-law fits: BW₂₀ ∝ η^α

| Band | α (θ ≤ 3°) | α (all angles) | R² (θ ≤ 3°) |
|------|------------|----------------|-------------|
| 0 (hole) | **1.52** | 1.69 | 0.985 |
| 1 (electron) | **0.76** | 1.00 | 0.997 |
| 2 (hole) | **0.50** | 0.93 | 0.782 |

**Observations:**
- **Band 0 (hole):** BW ∝ η^1.5 — between linear and quadratic. This band has the most stable power law (R² = 0.985). The hole stays well-bound in the potential at all angles tested (|δ/V| < 0.33 up to θ = 8°).
- **Band 1 (electron):** BW ∝ η^0.76 — sublinear scaling. This band has extremely light effective mass (M_inv = 10.87), causing the kinetic energy to dominate at moderate angles. The envelope breaks down at **θ ≈ 1.1–1.5°** (δ/V crosses 0.3).
- **Band 2 (hole):** BW ∝ η^0.5 — square-root scaling. Non-monotonic local exponents (−0.80 to 1.67) suggest convergence issues or competing effects. Breaks down at **θ ≈ 3–5°**.

### Validity regime (|δ/V_range| < 0.3 criterion)

| Band | Valid for | Max θ | Max η |
|------|-----------|-------|-------|
| 0 (hole) | θ ≲ 5° | ~5° | ~0.087 |
| 1 (electron) | θ ≲ 1.1° | ~1.3° | ~0.023 |
| 2 (hole) | θ ≲ 3° | ~3° | ~0.052 |

**Key finding:** Band 1 (electron) has the narrowest validity window because its positive effective mass (~10.87) creates kinetic energy that quickly overwhelms the potential modulation (~0.097). By θ = 3°, the ground state is 90% kinetic energy above V_min.

## 7. Conclusions

1. **The η² cancellation is rigorously correct.** The code's prefactor 0.5/(2π)² is the dimensionally consistent choice for eigenvalues in f-units (c/a) with derivatives in physical x-coordinates (units of a). No η² factor is missing.

2. **Two bugs were found and fixed:**
   - Non-Hermitian kinetic operator (fixed by symmetrization)
   - Missing M_inv regularization option (added with configurable max_trace)

3. **The envelope approximation has a limited validity window.** For Candidate 0 (square lattice, r/a=0.29, ε=7.9, k₀=X), the single-band electron envelope is only valid for θ ≲ 1.3° (η ≲ 0.023). The hole bands are valid to θ ≈ 3–5°. Beyond these limits, kinetic energy dominates and the envelope predictions are unphysical.

4. **The narrow validity for Band 1 is driven by M_inv hot spots** where bands 0 and 1 nearly touch (gap₀₁ → 0.004). A multi-band (bands 0+1) model would handle these points correctly, potentially extending the validity to larger angles.

## Files

| File | Description |
|------|-------------|
| `F03_eta_prefactor_analysis.py` | Definitive η² test (full diag + dimensionless comparison) |
| `F03_REPORT.md` | This report |
| `F03_data.json` | Raw numerical data from η² tests |
| `make_F03_plot.py` | 4-panel diagnostic plot (M_inv, gap, T/V, non-Hermiticity) |
| `F03_kinetic_analysis.png` | Output of diagnostic plot |
| `F03_resweep.py` | Full 8-angle × 3-band corrected sweep script |
| `sweep_results_F03_corrected.json` | Complete sweep results (20 eigenvalues per solve) |
| `make_F03_sweep_plot.py` | 4-panel corrected sweep analysis plot |
| `F03_sweep_corrected.png` | Output of sweep analysis plot |

---

## Update (2026-02-07): Symmetric Gauge + Γ-Point 5-Band Candidate

### What Changed

- **New candidate**: Γ-point, r/a=0.35, ε=12.0, 5-band subspace [5–9].
- **Eta sweep path**: Updated from `phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258` to `phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808` (8 angles: 0.5°–8.0°).
- **`make_F03_plot.py` updated**: Panel (a) uses `target_index_in_subspace` from HDF5. Panel (b) computes gap to nearest neighbor dynamically. Panel (c) auto-detects `N_bands` and extends colors/labels. Panel (d) handles arbitrary band count.
- **`F03_resweep.py`**: Path updated.

### Expected Outcome

The η² prefactor and Hermiticity findings are independent of k₀ and N_bands — these are structural properties of the envelope formulation. The M_inv hot-spot analysis may differ: the Γ-point 5-band subspace has different inter-band gaps and degeneracy structure, likely affecting where non-Hermiticity becomes significant.
