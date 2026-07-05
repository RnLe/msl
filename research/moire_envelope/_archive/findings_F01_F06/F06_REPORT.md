# F06 — Bloch Field Gauge & Normalization Diagnostic

## Executive Summary

Raw MPB Bloch fields have **two critical problems** that corrupt all downstream
results (Berry connection, Born-Huang potential, envelope Hamiltonian):

1. **Normalization**: Raw MPB fields satisfy ∫ε|E|²dV = 1 (ε-weighted), but the
   pipeline expects ⟨u|u⟩_Ω = Σ|u|²/(NxNy) = 1 (cell-averaged flat norm).
   Raw norms vary by 4× across the registry grid and differ 2× between bands.

2. **Gauge**: MPB assigns an arbitrary complex phase to each eigenfield at each
   registry point. The phase jumps are uniformly random (σ = π/√3 ≈ 1.81 rad),
   making finite-difference Berry connection computations meaningless.

**Fix 1 (normalization)** is trivial: divide by √(⟨u|u⟩_Ω) → perfectly uniform norms.

**Fix 2 (gauge)** is partially successful: Abelian (per-band scalar) parallel
transport reduces the s₁ phase standard deviation from 1.8 → 0.18 rad. The s₂
direction improves from 1.8 → 1.17 rad but remains significantly rough due to
non-trivial Berry curvature in the registry Brillouin zone.

**Critical discovery**: The existing SVD non-Abelian gauge fix
(`apply_parallel_transport_gauge` in `phase2_mpb_v3.py`) **completely fails**
because it uses the flat inner product ⟨u_m|u_n⟩ for cross-band overlaps, but
MPB eigenstates are orthogonal under the ε-weighted inner product ∫ε u_m*·u_n dA.
This causes the gauge rotation to mix non-orthogonal components, destroying both
normalization and gauge coherence.

---

## Data & Configuration

| Parameter       | Value |
|-----------------|-------|
| Crystal         | Square lattice, a=1.0, r/a=0.29, ε_bg=7.9 |
| k₀              | X-point (0.5, 0) |
| Subspace        | Bands 0–2 (N_sub = 3) |
| Registry grid   | 64 × 64 |
| Unit cell grid  | 64 × 64 × 3 components |
| Sweep angle     | θ = 2.0° (Bloch fields are θ-independent) |
| Sweep dir       | `runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258` |

---

## Problem 1: Normalization Inconsistency

### What MPB gives us

MPB normalizes eigenmodes under the **ε-weighted norm**:
$$\int_\Omega \varepsilon(\mathbf{r})\, |\mathbf{E}_{n\mathbf{k}}(\mathbf{r})|^2 \, d^2r = 1$$

This is physically correct but means the flat (unweighted) norm varies depending
on where the field concentrates relative to the dielectric structure.

### Raw norm measurements

| Band | ⟨u|u⟩_Ω mean | ⟨u|u⟩_Ω std | ⟨u|u⟩_Ω range |
|------|-------------|------------|---------------|
| 0    | 0.2515      | 0.1158     | [0.147, 0.612] |
| 1    | 0.4495      | 0.0874     | [0.281, 0.622] |
| 2    | 0.2216      | 0.0474     | [0.172, 0.466] |

Key observations:
- **Band-dependent norms**: Band 1 has 2× the flat norm of Band 2, reflecting
  different spatial concentration relative to ε(r).
- **Registry-dependent norms**: Within each band, the norm varies by a factor
  of 2–4 across the shift grid. This is because the local band structure
  (and hence ε-weighted vs flat mismatch) changes with the registry shift.
- **Convention mismatch**: Phase 2 (`phase2_mpb_v3.py`) renormalized to
  Σ|u|² = 1 (flat L2), while `bloch_fields.py` and Born-Huang use
  Σ|u|²/(NxNy) = 1 (cell-averaged). These differ by a factor of NxNy = 4096.

### After Fix 1

Cell-averaged normalization (⟨u|u⟩_Ω = 1):

| Band | ⟨u|u⟩_Ω mean | ⟨u|u⟩_Ω std |
|------|-------------|------------|
| 0    | 1.000000    | 9.3e-8     |
| 1    | 1.000000    | 9.3e-8     |
| 2    | 1.000000    | 9.3e-8     |

Trivially perfect. The Abelian gauge fix does not mix bands, so norms remain
exactly 1.0 after gauge fixing (confirmed in the post-fix diagnostic).

---

## Problem 2: Random Gauge (Phase)

### The Problem

At each registry point (s₁, s₂), MPB independently solves the eigenvalue
problem. The resulting eigenvector u_n(s) has an arbitrary overall phase
e^(iφ_n(s)). This phase is uncorrelated between neighboring registry points.

**Diagnostic metric**: The overlap phase between neighbors,
$$\phi_{s_1}(s) = \arg \langle \tilde{u}_n(s) | \tilde{u}_n(s + \delta s_1) \rangle$$
where $\tilde{u} = u/\|u\|$.

For a smooth gauge: $\sigma(\phi) \ll 1$.
For random gauge:   $\sigma(\phi) = \pi/\sqrt{3} \approx 1.814$ rad.

### Raw gauge measurements

| Band | σ(φ_s₁) | σ(φ_s₂) | Expected random |
|------|---------|---------|-----------------|
| 0    | 1.816   | 1.815   | 1.814           |
| 1    | 1.830   | 1.784   | 1.814           |
| 2    | 1.801   | 1.751   | 1.814           |

All perfectly match the uniform-random prediction. **The phases ARE random.**

Despite this, the overlap **magnitudes** are mostly large:

| Band | frac(|ov| > 0.99) s₁ | min|ov| s₁ | frac(|ov| > 0.99) s₂ | min|ov| s₂ |
|------|-----------------------|------------|---------------------|----|
| 0    | 0.924                 | 0.213      | 0.924               | 0.072 |
| 1    | 0.929                 | 0.213      | 0.929               | 0.073 |
| 2    | 0.827                 | 0.000      | 0.827               | 0.019 |

This means the band **character** (spatial pattern) is smooth — only the **phase**
is random. Band 2 has a topological singularity where |ov| = 0.000 (the band
character rotates by 180°, likely a π-Berry-phase vortex).

### After Abelian Gauge Fix

The Abelian fix aligns each band's phase independently (no cross-band mixing):

| Band | σ(φ_s₁) before → after | σ(φ_s₂) before → after |
|------|----------------------|----------------------|
| 0    | 1.816 → **0.182**    | 1.815 → **1.166**    |
| 1    | 1.830 → **0.182**    | 1.784 → **1.166**    |
| 2    | 1.801 → **0.224**    | 1.751 → **1.576**    |

**s₁ direction**: 10× improvement. Residual σ ≈ 0.18 is dominated by the
open-chain boundary (the last grid point wraps to the first, and the accumulated
transport phase creates a single large phase jump — the Zak phase / Berry holonomy).

**s₂ direction**: Only ~35% improvement for bands 0–1, essentially unchanged for
band 2. This is a **fundamental limitation**: the 2D gauge cannot be simultaneously
smooth in both directions when the Berry curvature is non-zero. The residual
s₂ phase reflects the accumulated Berry phase around plaquettes.

---

## Problem 3: Non-Orthogonality Under Flat Inner Product

### Discovery

MPB eigenstates are orthogonal under the **ε-weighted** inner product:
$$\int \varepsilon(\mathbf{r})\, \mathbf{u}_m^*(\mathbf{r}) \cdot \mathbf{u}_n(\mathbf{r})\, d^2r = \delta_{mn}$$

But the pipeline uses the **flat** inner product everywhere:
$$\langle u_m | u_n \rangle_\mathrm{flat} = \sum_{\mathbf{r}} u_m^*(\mathbf{r}) \cdot u_n(\mathbf{r})$$

| Band pair | mean |⟨u_m|u_n⟩_flat| | max  | frac(< 0.01) |
|-----------|------|------|--------------|
| (0, 1)    | 0.000 | 0.000 | 1.000        |
| (0, 2)    | 0.142 | 0.644 | 0.160        |
| (1, 2)    | 0.028 | 0.652 | 0.860        |

Bands 0 and 1 happen to be perfectly orthogonal under the flat inner product
(likely due to symmetry — different irreps). But bands 0–2 and 1–2 have
**significant non-orthogonality** (up to 0.65!) under the flat product.

### Consequences

1. **SVD non-Abelian gauge fix fails**: The `apply_parallel_transport_gauge()`
   function computes overlaps using the flat inner product and applies SVD
   rotations. These rotations mix non-orthogonal bands, **destroying** both
   normalization and phase coherence. Our test showed the non-Abelian fix
   actually WORSENED all metrics (phase σ unchanged, norms destabilized,
   cross-band leakage increased).

2. **Berry connection**: The non-Abelian Berry connection A_{j,mn} = i⟨u_m|∂_j u_n⟩
   should use the ε-weighted inner product for consistency with the Hamiltonian.
   Using the flat inner product introduces spurious off-diagonal terms.

3. **Born-Huang potential**: Similarly uses flat inner product, leading to
   artifacts from the ε-weighting mismatch.

### The Abelian gauge fix avoids this

By fixing each band's phase independently (scalar rotation, no cross-band mixing),
the Abelian gauge preserves whatever orthogonality structure exists. Norms stay
exactly 1.0 after gauge fixing (confirmed numerically).

---

## What the Plots Show

### F06_before.png — Raw MPB State

- **Row 1 (Normalization)**: Wildly varying ⟨u|u⟩_Ω across the grid and between
  bands. Band 1 is brightest (~0.45), Band 2 dimmest (~0.22).
- **Row 2 (s₁ Phase)**: Uniform random noise — the "snow" pattern confirms
  completely random gauge.
- **Row 3 (s₂ Phase)**: Same random noise pattern.
- **Row 4 (Orthogonality)**: Bands 0–1 perfectly orthogonal (dark). Bands 0–2
  show localized non-orthogonality hotspots up to 0.64.

### F06_after.png — After Normalization + Abelian Gauge Fix

- **Row 1 (Normalization)**: Perfectly uniform = 1.0 (all green, no variation).
- **Row 2 (s₁ Phase)**: Nearly zero everywhere — gauge successfully smoothed
  along s₁. One thin stripe at the periodic boundary may be visible (Zak phase).
- **Row 3 (s₂ Phase)**: Still has significant structure — this is **real Berry
  curvature**, not a fixable artifact. Band 2 is worst (topological singularity).
- **Row 4 (Flat Orthogonality)**: Exactly the same as before (Abelian fix preserves it).
  |⟨u_0|u_2⟩|_flat up to 0.64 — expected, NOT a bug (wrong inner product).
- **Row 5 (ε-weighted Orthogonality)**: Near-perfect zeros across the entire grid.
  |⟨u_m|ε|u_n⟩| ≤ 0.021 everywhere (vs flat max of 0.644). This confirms that
  the E-field Bloch functions from MPB ARE orthogonal under the correct,
  ε-weighted inner product, as theory requires.
- **Row 6 (SVQB B-Orthonormality)**: Uniformly zero at machine precision.
  |⟨u_m|ε|u_n⟩|_SVQB ≤ 10⁻¹⁵ everywhere. SVQB removes the ~0.02 residual
  that MPB's eigensolver leaves behind, achieving exact B-orthonormality.

---

## ε-Weighted Orthogonality: The Correct Inner Product

### Theory

Phase 1 extracts E-fields from MPB via `ms.get_efield(band, bloch_phase=False)`.
These satisfy the **generalized eigenvalue problem**:
$$\nabla \times \nabla \times \mathbf{E}_n = \varepsilon(\mathbf{r})\, \left(\frac{\omega_n}{c}\right)^2 \mathbf{E}_n$$

The correct orthogonality relation for this formulation is the **ε-weighted inner product**:
$$\langle u_m | \varepsilon | u_n \rangle = \int_\Omega \varepsilon(\mathbf{r})\, \mathbf{u}_m^*(\mathbf{r}) \cdot \mathbf{u}_n(\mathbf{r})\, d^2r = \delta_{mn}$$

This is NOT the flat L2 inner product ⟨u_m|u_n⟩ that the pipeline currently uses.

### Extracting ε from MPB

MPB performs **sophisticated subpixel averaging** at dielectric boundaries.
The exact ε(r) on the grid cannot be reconstructed analytically — it must be
extracted directly from MPB via:
```python
ms.init_params(mp.NO_PARITY, False)
eps = ms.get_epsilon()  # returns (Nx, Ny) array with subpixel averaging
```
Importantly, ε(r; δ) changes at each registry point because the bilayer shift δ
moves the second cylinder.

### Results: ε-weighted vs flat orthogonality

Extracted ε(r) from MPB at all 64×64 registry points (4096 MPB calls, ~10s):

| Band pair | Flat max |⟨u_m\|u_n⟩| | ε-weighted max |⟨u_m\|ε\|u_n⟩| | Reduction |
|-----------|------------------------|-------------------------------|-----------|
| (0, 1)    | 0.0004                 | 0.0004                        | ~1×       |
| (0, 2)    | **0.644**              | **0.021**                     | **31×**   |
| (1, 2)    | **0.652**              | **0.015**                     | **43×**   |

| Metric | Flat | ε-weighted | Improvement |
|--------|------|------------|-------------|
| Max off-diagonal (any pair) | 0.652 | 0.021 | **31×** |
| Mean off-diagonal           | 0.169 | 0.0025 | **68×** |

The ε-weighted off-diagonal overlaps are consistent with MPB's eigensolver
convergence tolerance (~10⁻³ to 10⁻² for eigenvectors). **The orthogonality
is essentially perfect** under the correct inner product.

### Why the residual ~0.02 is not alarming

MPB's default eigenvalue tolerance is 10⁻⁷, which translates to eigenvector
accuracy of approximately √(tol) ≈ 10⁻³ to 10⁻². The max |⟨u_m|ε|u_n⟩| = 0.021
occurs at isolated registry points (likely near band crossings or high-curvature
regions) and is fully consistent with numerical precision. The mean of 0.0025
confirms overall excellent orthogonality.

However, if exact B-orthonormality is required (e.g., for Berry connection
computation), the residual can be eliminated entirely using SVQB (see below).

---

## SVQB B-Orthonormalization

### Motivation

While MPB's eigensolver produces Bloch functions that are approximately
ε-orthonormal (residual ~0.02), exact B-orthonormality is important for:
- Computing the non-Abelian Berry connection accurately
- Ensuring the Born-Huang potential matrix is Hermitian
- Preventing numerical artifacts from accumulating in the envelope Hamiltonian

### The SVQB Algorithm

SVQB (from the Blaze photonic eigensolver) is more stable than Gram-Schmidt
near band degeneracies. At each registry point:

1. **Pre-normalize** each band's Bloch vector to unit B-norm (B = diag(ε))
2. **Form Gram matrix** G = X^H · (BX) — a 3×3 Hermitian matrix
3. **Eigendecompose** G = Q Λ Q^H
4. **Rank-reveal**: drop eigenvalues with λᵢ/λ_max < 10⁻¹² (none dropped here)
5. **Transform** X_new = X_old · Q · Λ^{-1/2}

After SVQB: X_new^H · B · X_new = I exactly (to machine precision).

### Results

| Metric | Raw MPB | Simple ε-norm | SVQB |
|--------|---------|---------------|------|
| max |⟨u_0\|ε\|u_1⟩| | 0.000397 | 0.000397 | **6.3×10⁻¹⁶** |
| max |⟨u_0\|ε\|u_2⟩| | 0.021078 | 0.021078 | **1.0×10⁻¹⁵** |
| max |⟨u_1\|ε\|u_2⟩| | 0.015161 | 0.015161 | **7.1×10⁻¹⁶** |
| max |1 - diag| | ~0 | ~0 | **1.7×10⁻¹⁵** |
| Rank deficiency | — | — | **0/4096** points |
| Gram κ (condition) | — | — | mean=1.01, max=1.04 |

Key observations:
- **SVQB achieves machine-epsilon B-orthonormality** — 10¹³× better than simple
  ε-normalization.
- **No rank deficiency** at any registry point. The Gram condition number is
  excellent (max 1.04), indicating no near-degeneracies in this 3-band subspace.
- **Simple ε-normalization is inadequate**: dividing by √⟨u|ε|u⟩ fixes the
  diagonal but cannot correct cross-band contamination. SVQB's eigendecomposition
  rotates within the subspace to achieve exact orthogonality.

### Why not Gram-Schmidt?

At high-symmetry k-points (Γ, X, M), bands become degenerate. Two Bloch functions
spanning the same 2D eigenspace will have nearly parallel components. Gram-Schmidt
subtracts one from the other, leaving a vector with norm ~ ε_machine — catastrophic
cancellation. SVQB handles this via the eigendecomposition of the Gram matrix,
rotating within the degenerate subspace instead of subtracting.

### Implementation

See `findings/F06_svqb_orthonormalize.py` and `findings/svqb_guide.md` for the
complete algorithm. The Python implementation follows the Rust implementation in
Blaze, with all accumulations in float64 for numerical stability.

---

## Implications for the Pipeline

### What must change

1. **Phase 1** (`phase1_mpb_v3.py`):
   - Currently saves raw MPB E-fields with NO normalization. Should normalize to
     ⟨u|u⟩_Ω = Σ|u|²/NxNy = 1 immediately after extraction.
   - **Must also extract and store ε(r; δ)** at each registry point, using
     `ms.get_epsilon()` after `ms.init_params()`. This is fast (~10s for 4096
     points, no eigensolve needed) and essential for correct inner products.

2. **Phase 2** (`phase2_mpb_v3.py`):
   - Replace SVD non-Abelian gauge with Abelian per-band gauge
   - Use proper 2D gauge: seed row + columns (not sequential axis-0 then axis-1)
   - Fix the flat/cell-averaged normalization convention inconsistency
   - **Berry connection** A_{j,mn} = i⟨u_m|ε|∂_j u_n⟩ must use ε-weighted inner product
   - **Born-Huang potential** V_{mn} must also use ε-weighted inner product

3. **Alternative**: Switch to **H-fields** (`ms.get_hfield()`), which satisfy
   the standard eigenvalue problem Θ H = (ω/c)² H and are orthogonal under
   the flat L2 inner product. This avoids the ε-weighting complexity entirely
   but requires re-deriving the envelope Hamiltonian in the H-field formulation.

### Recommended approach

The **E-field + ε-weighted inner product + SVQB** approach is theoretically rigorous
and closest to our envelope approximation derivation:
- Store ε(r; δ) alongside Bloch fields in Phase 1 (trivial cost: ~10s)
- Apply SVQB B-orthonormalization with B = diag(ε) at each registry point (~7s)
- Use ε-weighted overlaps for gauge fixing, Berry connection, Born-Huang
- The Abelian gauge fix already works correctly (doesn't need ε)
- For the non-Abelian Berry connection, ε-weighting + SVQB is essential

### What is fundamentally limited

- The s₂ gauge roughness is real Berry curvature, not an artifact. Derivatives
  in the s₂ direction will always have a "gauge-artifact" component that must be
  absorbed into the Berry connection.
- Band 2 has a topological singularity (|ov| = 0 at some point in the registry
  BZ). This means Band 2 cannot have a globally smooth gauge — it requires a
  topological treatment (e.g., patch-based gauge or Wilson-loop approach).
- Non-orthogonality under the flat inner product means we cannot trust off-diagonal
  Berry connection elements without ε correction. Diagonal elements (Abelian Berry)
  should be reliable.

---

## Files Produced

| File | Description |
|------|-------------|
| `findings/F06_gauge_diagnostic.py` | Main diagnostic script (compute before/after) |
| `findings/F06_epsilon_orthogonality.py` | ε extraction from MPB + ε-weighted orthogonality |
| `findings/F06_svqb_orthonormalize.py` | SVQB B-orthonormalization + comparison |
| `findings/svqb_guide.md` | SVQB algorithm guide (from Blaze) |
| `findings/make_F06_plot.py` | Plotting script (6-row × 4-col heatmaps) |
| `findings/F06_before_data.npz` | Raw MPB state: norms, phases, orthogonality |
| `findings/F06_after_data.npz` | After normalization + Abelian gauge fix |
| `findings/F06_epsilon_data.npz` | ε(r; δ) grid + ε-weighted orthogonality matrices |
| `findings/F06_svqb_data.npz` | SVQB results: B-ortho fields, Gram eigenvalues |
| `findings/F06_before.png` | Before heatmap (4 rows) |
| `findings/F06_after.png` | After heatmap (6 rows: norm, gauge, flat ortho, ε-ortho, SVQB) |
| `findings/F06_REPORT.md` | This report |

---

## Update (2026-02-07): BFS Gauge Fix Resolves s₁/s₂ Asymmetry

### Root Cause Identified

The seed-row+columns Abelian gauge algorithm treated s₁ and s₂ asymmetrically:
- **s₂ direction** was fixed first (seed row along s₂ at row 0)
- **s₁ direction** was fixed second (columns propagating from the seed row)

This produced 6.4× worse phase variance in s₂ (σ = 1.17 rad) vs s₁ (σ = 0.18 rad), directly breaking C4 symmetry in the Berry connection and Born-Huang potential.

### Fix Applied

**BFS from center + Zak phase linear ramp** (implemented in `phase2_mpb_v3.py`):

1. **BFS (breadth-first search)**: Start at center point `(Ns1//2, Ns2//2)` and expand isotropically to 4-connected neighbors using `collections.deque`. Each visited point's phase is aligned with its parent via overlap maximization. This treats s₁ and s₂ identically.

2. **Zak phase linear ramp**: After BFS, the field at `(Ns1-1, j)` may differ from `(0, j)` by a residual phase (the Zak phase). Rather than forcing a discontinuous jump, the residual is distributed as a linear ramp across all rows (and similarly for columns). This preserves periodic boundary smoothness.

### Diagnostics

New per-band metrics logged: `min_ov`, `n_singular`, `n_aligned`, `zak_phi1`, `zak_phi2`, `post_boundary_std_s1`, `post_boundary_std_s2`. Post-fix, s₁ and s₂ boundary std should be comparable (symmetric).

### Impact

This fix is upstream of all other findings. Re-running Phase 2+3 with the BFS gauge should restore C4 symmetry in mode plots and may quantitatively change Berry connection magnitudes, Born-Huang values, and some scaling exponents reported in F02–F05.
