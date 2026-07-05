# S1–S2 Diagnostic Findings Report
## Systematic Audit of the Moiré Envelope Pipeline

**Date**: Stage 1–2 audit  
**System**: Square lattice, a=1.0, r/a=0.35, ε_bg=12.0, k₀=Γ, TE, band 7, subspace [5-9]  
**Data**: `runsV3/phase0_mpb_v3_20260206_152443/candidate_0000/`

---

> ## ⚠️ PARTIAL RETRACTION (2026-02-08)
>
> **Finding 1 ("Subspace NOT closed under C4") is RETRACTED.** The C4 test used throughout this report was **C4 invariance** (is the subspace self-symmetric?), which is only valid at C4 fixed points. The correct test is **C4 equivariance** (does C4-rotating the subspace at R give the subspace at C4·R?). With the correct test, the original [5-9] subspace is **99.4% equivariant**.
>
> The "98.3% failure" was a testing artifact. See [`FINDINGS_S3.md`](FINDINGS_S3.md) for details.
>
> **Findings 2–6 remain valid.** M_inv divergence, Berry connection noise, and the (32,32) fixed-point defect are real issues, now understood as consequences of anti-crossing smoothness defects at ~10% of points.

---

## Executive Summary

~~The 5-band subspace [5-9] is **fundamentally broken** for the envelope approximation.~~ **RETRACTED**: The C4 test was wrong (invariance instead of equivariance). The subspace is 99.4% C4-equivariant.

The **real** issues are: (1) subspace smoothness failures at 9.4% of points due to anti-crossings, causing (2) M_inv divergence at >50% of points and (3) Berry connection noise |Im(A)| up to 1.8. These are localized defects, not global subspace failure.

---

## Critical Findings

### ~~Finding 1: Subspace NOT closed under C4 (FATAL)~~ → RETRACTED

> **This finding used the wrong C4 test (invariance instead of equivariance).** See [FINDINGS_S3.md](FINDINGS_S3.md).
>
> With the correct equivariance test: 99.4% of R-points pass (min σ > 0.9).
> The invariance numbers below are included for reference only.

| Metric (INVARIANCE — wrong test) | Value |
|--------|-------|
| % of R-points with \|det(M₅)\| < 0.5 | 98.3% ← artifact |
| % of R-points with min σ(M₅) < 0.5 | 96.7% ← artifact |

The 5×5 overlap matrix M_mn = ⟨u_m|C₄u_n⟩/(||u_m||||u_n||) tests invariance (same-point). At generic R, C4 maps R → C4·R ≠ R, so the subspace at R has no reason to be self-symmetric. The correct test compares C4-rotated states at R with states at C4·R (equivariance).

**Why invariance fails at generic points:** A subspace carrying a non-trivial C4 representation (like the E-rep at bands 7,8) transforms non-trivially under C4. Only at fixed points (δ=0, δ=0.5) does invariance = equivariance.

**Subspace size scan (also RETRACTED):**

| Subspace | Invariance min_sv > 0.9 (wrong) | Equivariance > 0.9 (correct) |
|----------|:-------------------------------:|:----------------------------:|
| Band 7 alone | 0.0% | **100.0%** |
| Bands 7,8 | 1.3% | **99.9%** |
| Bands 6,7,8 | 0.8% | **99.9%** |
| Bands 5–9 (5) | 0.6% | **99.4%** |
| Bands 4–9 (6) | 0.0% | **99.4%** |
| Bands 4–10 (7) | 0.0% | **99.9%** |

ALL subspaces pass equivariance. The invariance "failures" were the wrong test.

**What IS real at δ=(0.5,0.5):** At this C4 fixed point, invariance = equivariance, and it genuinely fails (equivariance = 0.0001). Bands 7,8 are separated by only 3×10⁻⁶ here, causing degenerate-subspace issues.

### Finding 2: δ=(0,0) IS perfectly C4 — validating the test

At δ=(0,0), where both cylinders coincide:
- Bands 5, 6, 9: individual C4 fidelity > 0.999
- Bands 7, 8: form a perfect 2D E-representation (|det|=0.9995, eigenvalues ±i)
- Full 5-band eigenvalue phases: {-π, -π/2, 0, +π/2, +π} — exactly A⊕B⊕E⊕B

This confirms the C4 test algorithm is correct and the issue is genuinely about subspace validity at other R-points.

### Finding 3: M⁻¹ divergent at majority of points (FATAL)

| Threshold |Tr(M⁻¹)| | Band 5 | Band 6 | Band 7 | Band 8 | Band 9 |
|----------|--------|--------|--------|--------|--------|
| > 5 | 51.1% | 51.8% | 32.3% | 36.7% | **62.8%** |
| > 10 | 28.0% | 31.7% | 17.7% | 21.6% | **44.4%** |
| > 20 | 13.3% | 15.0% | 7.7% | 10.5% | **28.3%** |

This is not "isolated hot spots" — the effective mass is wildly fluctuating across more than half the moiré cell. The k-stencil FD derivative follows the wrong branch at anti-crossings, producing spurious curvatures.

### Finding 4: Berry connection has massive imaginary parts

For a properly gauged diagonal Berry connection A_nn(R), the imaginary part should vanish (since A_nn = -i⟨u_n|ε|∂u_n/∂R⟩ is real when u is real-gauged). Instead:

- |Im(A)|_max ≈ **1.8** for all bands
- Real part std ≈ 0.3–0.4

The gauge fix is not achieving a real gauge, or the fields are too noisy for FD derivatives, or both.

### Finding 5: Off-diagonal v_drift is identically zero

The inter-band drift velocity v_drift_mn for m≠n is exactly 0.0. This means the pipeline never computes inter-band coupling for this term, removing a potentially important piece of physics.

### Finding 6: Potential Λ(R) IS approximately C4

Despite all other failures, the diagonal potential Λ_nn(R) = ω_n(R) - ω_ref satisfies C4 to within 0.2%. This is because ω_n(R) is a scalar eigenvalue that doesn't depend on gauge or field phases.

---

## OK Findings

| Check | Status | Notes |
|-------|--------|-------|
| Band ordering | ✓ | No crossings within [5-9] (by frequency) |
| Λ_nn = ω_n - ω_ref | ✓ | Exact match (machine precision) |
| ε-Gram condition | ✓ | cond(G) < 1.09 everywhere (SVQB should handle) |
| ε symmetry at δ=(0,0) | ≈✓ | max|ε - C₄ε|/max(ε) = 1.8e-4 (discretization) |
| vg at Γ (bands 5–7) | ✓ | max|vg| < 0.01 (consistent with Γ) |

---

## Diagnosis: Why the Physics is Wrong — UPDATED

~~The envelope approximation requires a well-defined, isolated subspace that maintains its identity as R varies. The pipeline selects bands by index [5-9] at every R-point. The fatal assumption was that band indices track physical character through anti-crossings.~~ 

**CORRECTED (2026-02-08):** The index-based subspace IS valid — it is 99.4% C4-equivariant (guaranteed by crystal symmetry). The real problems are:

1. **Anti-crossing smoothness defects (9.4% of points):** Where bands from outside [5-9] nearly touch bands inside, the Bloch functions change rapidly. FD derivatives at these points produce noisy M_inv and A_berry.

2. **The (32,32) fixed-point defect:** At δ=(0.5,0.5), bands 7 and 8 are separated by only 3×10⁻⁶, creating a near-degenerate pair that doesn't resolve into proper C4 irreps.

3. **Gauge disorder:** MPB assigns random phases to eigenstates. The BFS gauge fix only partially addresses this, leaving residual noise in the Berry connection.

The M_inv divergence (50%+ of points) and Berry connection noise (|Im(A)| up to 1.8) are **consequences** of these smoothness defects, not independent problems.

---

## What Needs to Change — UPDATED

### ~~Option A: Physical band tracking~~ → NOT NEEDED
The subspace is already valid. Band tracking (S3, S3b) was attempted and found to be unnecessary (and harmful — it broke equivariance).

### ~~Option B: Wannier function approach~~ → NOT NEEDED
Same — the subspace doesn't need reconstruction, just smoothing at defect points.

### Option A (new): Regularize anti-crossing defects
Flag the ~386 points where subspace smoothness < 0.5. At these points, median-filter the derived quantities (M_inv, A_berry) from neighbors. This removes the FD noise without changing the subspace.

### Option B (new): Term-by-term Hamiltonian diagnosis
Build H incrementally (Λ only → add M_inv → add A) to identify which term breaks C4 in the eigenmodes. This tells us exactly where regularization is most needed.

### Option C (new): Handle (32,32) fixed point
At the single genuine C4 defect, either interpolate from neighbors or resolve the degenerate E-representation explicitly.

---

## Next Steps — UPDATED

1. **Stage 4: Term-by-term Hamiltonian build** — H = Λ only → add M_inv → add A → identify which breaks C4
2. **Stage 5: Grid convergence** — test at Ns = 16, 32, 64, 128
3. **Stage 6: Regularize defect points** — median-filter M_inv and A at the 386 anti-crossing points
4. **Stage 7: Analytical cross-checks** — free-particle limit, Mathieu comparison

---

## Files

| Script | Purpose | Key Output |
|--------|---------|------------|
| `S1_single_R_audit.py` | Point-wise checks at 4 R-points | Orthonormality, gaps, vg, M_inv |
| `S1b_C4_fixed.py` | Corrected C4 test + degenerate-pair analysis | C4 fidelity, E-rep verification |
| `S2_field_symmetry.py` | Full 64×64/128×128 field maps | Closure, gaps, Gram, V, M⁻¹, A, Φ |
| `S2b_subspace_sizes.py` | Tests 1–8 band subspaces | Closure vs subspace size |
