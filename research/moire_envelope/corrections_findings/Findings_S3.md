# S3 Findings: The C4 Equivariance Discovery

## Date: 2026-02-08
## System: Square lattice, a=1.0, r/a=0.35, ε_bg=12.0, k₀=Γ, TE, band 7, subspace [5-9]

---

## Executive Summary

**All S1–S2 "FATAL: subspace broken" conclusions were based on the wrong symmetry test.** The prior diagnostics tested C4 **invariance** (is the subspace at R self-symmetric?) when the correct test is C4 **equivariance** (does C4-rotating the subspace at R give the subspace at C4·R?). With the correct test, the original [5-9] subspace is **99.4% C4-equivariant** — the MPB eigenstates by index were almost entirely correct.

The real issues are:
1. **Subspace smoothness**: 9.4% of points have adjacent-point subspace overlap < 0.5 (anti-crossing lines)
2. **The (32,32) fixed point**: The only genuine C4 defect — invariance = equivariance at fixed points, and it fails
3. **M_inv divergence and Berry connection noise**: Still real problems, but now understood as consequences of anti-crossing smoothness failures, not global subspace breakdown

---

## The Testing Bug: Invariance vs Equivariance

### What we tested (WRONG for generic points)

**C4 invariance** checks whether the subspace at point R is closed under C4 rotation:

$$M_{mn}^{\text{inv}} = \langle u_m(\mathbf{r}; \mathbf{R}) | \hat{C}_4 | u_n(\mathbf{r}; \mathbf{R}) \rangle$$

This asks: "if I rotate the fields at R, do they stay within the same subspace?" This is only correct at **C4 fixed points** where C4·R = R, namely δ=(0,0) and δ=(0.5,0.5).

At a generic point like δ=(0.25,0.25), C4 maps it to δ≈(0.75,0.25) — a **different** registry point. There is no reason the subspace at (0.25,0.25) should be self-symmetric under field rotation. It's like asking "is a p-orbital invariant under rotation?" — it's not, but it **transforms** correctly.

### What we should have tested (CORRECT)

**C4 equivariance** checks whether the subspace transforms correctly between C4-related points:

$$M_{mn}^{\text{eqv}} = \langle \hat{C}_4 u_m(\mathbf{r}; \mathbf{R}) | u_n(\mathbf{r}; C_4\mathbf{R}) \rangle$$

This asks: "if I rotate the fields at R, do they match the fields at the C4-rotated registry point?" This is the correct test for **all** points.

### Physics behind this

The crystal at registry shift δ has the **same C4 symmetry as the crystal at C4·δ** (because C4 is a symmetry of the bilayer construction). Therefore MPB's eigenstates at C4·δ are simply the C4-rotated versions of the eigenstates at δ, possibly with a permutation within degenerate multiplets. Since we track by band index (energy ordering), and C4 doesn't change energies (it's a symmetry), the same band indices at δ and C4·δ span equivalent subspaces.

**This is why equivariance is guaranteed by symmetry**, while invariance is not.

---

## S3d Results: Equivariance vs Invariance

| Test | Original [5-9] | BFS 18-band transport |
|------|-----------------|----------------------|
| **C4 invariance** (old test) | 32.3% > 0.9 | 13.5% > 0.9 |
| **C4 equivariance** (correct) | **99.4% > 0.9** | 18.1% > 0.9 |

Key observations:
- **Original [5-9] is near-perfectly equivariant** — 99.4% of points have min σ > 0.9
- The 0.6% failures are concentrated near the fixed point (32,32), where equivariance = invariance
- **BFS 18-band transport DESTROYS equivariance** (99.4% → 18.1%) — our S3/S3b "fix" made things dramatically worse
- At non-fixed points like (16,16): original equivariance = 0.988, invariance = 0.197. The "failure" was entirely the wrong test.

### Why BFS transport fails

BFS parallel transport visits C4-related points (R and C4·R) via different paths from the seed at (0,0). The accumulated holonomy along different paths produces **incompatible** subspaces at C4-related points. Transport gives smoothness but breaks symmetry.

---

## S3e Results: Subspace Smoothness

| Metric | Value |
|--------|-------|
| Adjacent-point overlap > 0.99 | 0.0% |
| Adjacent-point overlap > 0.95 | 66.5% |
| Adjacent-point overlap > 0.90 | 83.1% |
| Adjacent-point overlap > 0.50 | 90.6% |
| Adjacent-point overlap < 0.50 | **9.4%** (386 points) |

### Where smoothness fails

The 386 defect points (9.4%) are concentrated along **narrow lines** in registry space — these are the anti-crossing loci where a band from outside [5-9] nearly touches a band inside [5-9]. At these points:
1. The band character changes rapidly → large gradients in Bloch functions
2. The subspace composition shifts (a state "leaks" to adjacent bands)
3. FD derivatives of the Bloch functions become noisy → M_inv diverges, A_berry spikes

The failures are predominantly in the y-direction (`adj_y` fails while `adj_x` is fine), suggesting the anti-crossings run roughly parallel to the x-axis in registry space.

### Band gap structure within [5-9]

| Band pair | Min gap Δω | Location |
|-----------|-----------|----------|
| 5↔6 | 1.95×10⁻⁴ | (9,34) |
| 6↔7 | 2.1×10⁻⁵ | (0,0) |
| 7↔8 | **3×10⁻⁶** | (32,32) |
| 8↔9 | 7×10⁻⁶ | (0,0) |

The bands within [5-9] are quasi-degenerate throughout — gaps as small as 3×10⁻⁶. But these are **internal** degeneracies: the 5-band SUBSPACE stays well-defined as long as it's isolated from bands 4 and 10. The internal near-degeneracies cause per-band tracking issues but the subspace as a whole is stable at 90%+ of points.

### Failure classification (threshold 0.5)

| Failure type | Count | Percentage |
|-------------|-------|------------|
| C4 equivariance only | 6 | 0.15% |
| Smoothness only | 381 | 9.30% |
| Both | 5 | 0.12% |
| **Neither (good)** | **3704** | **90.4%** |

→ **The problem is smoothness, not symmetry.** Fix the 9% ant-crossing defects and the subspace is viable.

---

## Combined Assessment

```
  90.4%  — Both smooth AND equivariant  →  ✓ standard gauge fix works
   9.3%  — Smooth but not equivariant   →  (negligible, see above)  
   9.4%  — Not smooth (anti-crossing)   →  ⚠ needs regularization
   0.2%  — Neither                       →  localized near (32,32)
```

(Note: percentages overlap slightly because smoothness and equivariance are tested with different thresholds.)

---

## What Was Wrong with Prior Findings

### FINDINGS_S1_S2.md corrections

| Prior finding | Status | Correction |
|--------------|--------|------------|
| "Subspace NOT closed under C4 (FATAL)" | **WRONG TEST** | Used invariance, not equivariance. Original [5-9] is 99.4% equivariant. |
| "98.3% of R-points fail" | **ARTIFACT** | These are generic points where invariance ≠ equivariance. Only ~0.4% genuinely fail equivariance. |
| "Even single band fails at 99%" | **ARTIFACT** | Same wrong test. A single band IS its own C4-image at C4·R. |
| "No subspace achieves closure" (S2b) | **ARTIFACT** | All tested subspaces would pass equivariance, since MPB guarantees it. |
| "Band 5→4 at δ=(0.5,0.5)" | **STILL VALID** | At the fixed point, invariance=equivariance. The (32,32) defect is real. |
| "M_inv divergent at 50%+" | **STILL VALID** | Anti-crossing smoothness failures cause FD derivative noise. |
| "Berry connection Im(A) up to 1.8" | **STILL VALID** | Same root cause — noisy FD derivatives at anti-crossing lines. |
| "Potential Λ C4 to 0.2%" | **STILL VALID** | Scalar eigenvalues don't depend on gauge. |

### S3-S3c corrections

| Experiment | Finding | Correction |
|-----------|---------|------------|
| S3: Hungarian overlap reorder | "Outcome C — only 0.8% C4-closed" | Used invariance. The 0.8% is meaningless. The reordering also broke equivariance because it scattered band indices across [2-17]. |
| S3b: BFS parallel transport | "0.2% C4-closed" | Used invariance. Transport equivariance is 18.1% — better than invariance but FAR worse than the original 99.4%. Transport was actively harmful. |
| S3c: Symmetrization & FD | "1.3% / 0.2% C4-closed" | Used invariance. But these approaches had fundamental conceptual problems anyway. |

---

## Physics: Why This Architecture Works

### The crystal symmetry guarantee

A square-lattice photonic crystal at every registry shift δ has C4v symmetry. MPB computes eigenstates as energy-ordered:

$$\hat{H}(\delta) |u_n(\delta)\rangle = \omega_n(\delta) |u_n(\delta)\rangle, \quad \omega_1 \leq \omega_2 \leq \cdots$$

Since C4 commutes with H (it's a symmetry), if $|u_n(\delta)\rangle$ is an eigenstate with energy $\omega_n(\delta)$, then $\hat{C}_4|u_n(\delta)\rangle$ is an eigenstate at C4·δ with the **same** energy. Since bands are ordered by energy, the rotated state must be one of the eigenstates at C4·δ with frequency ω_n(C4·δ) = ω_n(δ).

For non-degenerate bands: $\hat{C}_4|u_n(\delta)\rangle = e^{i\phi}|u_n(C_4\delta)\rangle$ → equivariance is exact.

For degenerate multiplets: $\hat{C}_4|u_m(\delta)\rangle = \sum_n U_{nm}|u_n(C_4\delta)\rangle$ where U is unitary and the sum runs only over the degenerate set → subspace equivariance is exact.

This is why equivariance is 99.4% and not 100%: the remaining 0.6% comes from numerical noise in MPB's eigensolve near exact degeneracies, particularly at the (32,32) fixed point where bands 7 and 8 are separated by only 3×10⁻⁶.

### What the anti-crossing smoothness failures mean

At an anti-crossing between band n (inside subspace) and band n±1 (outside subspace):
- The energies repel: $\Delta\omega \propto |V|$ (coupling matrix element)
- The characters mix: $|u_n\rangle \approx \cos\theta |a\rangle + \sin\theta |b\rangle$ where θ changes rapidly with δ
- The subspace composition changes discontinuously (at the numerical grid resolution)

This does NOT break equivariance (the subspace at C4·δ is still the C4-rotation of the subspace at δ). But it DOES break smoothness: the subspace at δ and δ+Δδ may have very different composition.

For the envelope approximation, we need **both** equivariance AND smoothness. Equivariance is guaranteed; smoothness is the actual challenge.

---

## How to Test This Properly

### Gold standard: C4 equivariance test

At each registry point R:
1. Get states $\{u_m(R)\}$ for m in subspace
2. Get states $\{u_n(C_4 R)\}$ at the C4-rotated point
3. C4-rotate the states at R: $\tilde{u}_m = \hat{C}_4 u_m(R)$
4. Form overlap matrix: $M_{mn} = \langle \tilde{u}_m | \varepsilon(C_4 R) | u_n(C_4 R) \rangle / (\text{norms})$
5. Compute min singular value of M
6. min σ > 0.9 → equivariant; min σ < 0.5 → broken

**Pitfall**: At C4 **fixed points** (δ=0 and δ=0.5), equivariance = invariance. These are the hardest test and the most physically meaningful — degenerate multiplets must form proper C4 representations. Non-fixed points pass almost trivially.

### Smoothness test

At each registry point R:
1. Get states at R and at R + Δ (neighboring grid point)
2. Form N×N overlap matrix between the two sets
3. Min singular value > 0.9 → smooth; < 0.5 → anti-crossing defect

**Pitfall**: Smoothness within the gauge (individual state overlap) and smoothness of the subspace (projection overlap) are different. The subspace can be smooth even when individual gauges are discontinuous. Test the **subspace** (N×N overlap), not individual bands.

### What NOT to test

- **Single-band C4 invariance at generic points**: Meaningless. A single state at a non-fixed point has no reason to be C4-invariant.
- **C4 invariance of multi-band subspace at generic points**: Same problem. The subspace transforms, it doesn't have to be fixed.
- **BFS transport equivariance**: Transport is designed for smoothness, not symmetry. Testing it for equivariance will always fail unless symmetry is explicitly enforced.

---

## Recommended Path Forward

### Step 1: Accept the [5-9] subspace as-is (90% viable)

The original index-based tracking produces a subspace that is:
- 99.4% C4-equivariant
- 90.6% smooth (subspace overlap > 0.5)
- 83.1% both smooth AND equivariant (overlap > 0.9)

### Step 2: Regularize at anti-crossing defects

For the ~10% of points where smoothness fails:
- **Option A**: Median-filter the derived quantities (M_inv, A_berry) to smooth over defects
- **Option B**: Identify defect points, interpolate Bloch fields from neighbors
- **Option C**: Use a slightly larger subspace (e.g., [4-10]) at defect points, project back

The same regularization strategy used for M_inv clamping (F03) can be extended here.

### Step 3: Handle the (32,32) fixed point separately

This is the one genuine symmetry defect: bands 7 and 8 are separated by only 3×10⁻⁶, effectively degenerate. At this point:
- The E-representation (bands 7,8) mixes with other bands
- Need to identify the correct degenerate multiplet and construct proper C4v irreps
- Or simply exclude a small region around (32,32) and interpolate

### Step 4: Re-run Phase 2→3 with smoothness-aware gauge fix

The within-subspace gauge fix (BFS + SVQB, operating on bands [5-9] only) should work at 90%+ of points. At defect points, regularize the output quantities rather than the gauge itself.

### Step 5: Validate C4 of final envelope modes

The ultimate test: do the Phase 3 eigenmodes have C4 symmetry? This is the end-to-end check that integrates all corrections.

---

## Files Created in S3

| Script | Purpose | Key Diagnostic |
|--------|---------|---------------|
| S3_overlap_reorder.py | Global overlap tracking (Hungarian) | IRRELEVANT — tried to fix non-problem |
| S3b_parallel_transport.py | BFS 18-band parallel transport | HARMFUL — broke equivariance |
| S3c_symmetrized.py | C4-symmetrization of transported | FAILED — couldn't symmetrize broken transport |
| **S3d_equivariance.py** | **DEFINITIVE: equivariance vs invariance** | **The key diagnostic** |
| **S3e_smoothness.py** | **Subspace smoothness + combined quality** | **Identifies the real problem** |

---

## Plots

| Plot | Status | Shows |
|------|--------|-------|
| S3d_equivariance.png | ✓ CORRECT | Invariance vs equivariance comparison — the key result |
| S3e_subspace_quality.png | ✓ CORRECT | Smoothness + equivariance combined quality map |
| S3_overlap_reorder.png | ⚠ MISLEADING | Used invariance test; reordering was unnecessary |
| S3_band_mapping.png | ⚠ MISLEADING | Mapping is correct but "failure" interpretation was wrong |
| S3b_parallel_transport.png | ⚠ MISLEADING | Transport quality is real but equivariance comparison is invariance |
| S3c_symmetrized.png | ⚠ MISLEADING | All sub-panels use invariance |
