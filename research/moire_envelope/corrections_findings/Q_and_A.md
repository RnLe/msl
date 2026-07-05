# Questions & Answers — Envelope Approximation Physics

**Date:** 2025-02-09  
**Context:** Post-S4 diagnostic — root causes identified (BFS gauge → A_berry ~200% C4 error, anti-crossing noise → M_inv 31–39% C4 error for clustered bands).

---

## Q1: Should we use a different test candidate?

**Short answer:** Not yet — but plan a validation run on a simpler candidate after C4-symmetrization succeeds.

### Current candidate

| Property     | Value                                       |
|-------------|---------------------------------------------|
| Lattice      | Square, a = 1.0                              |
| r/a          | 0.35                                         |
| ε_bg         | 12.0 (holes: ε = 1.0)                        |
| k₀           | Γ                                            |
| Polarization  | TE                                           |
| Target band   | 7 (merged index), subspace [5, 6, 7, 8, 9]   |
| θ             | 1.1°, η = 0.0192, L_moire = 52.09a           |

**Why keep it for now:**
1. It's the most-studied candidate — all S1–S4 diagnostics are done here.
2. The five quasi-degenerate bands represent the *hardest* regime for the multi-band envelope approximation. If C4-symmetrization works here, it proves the method in the most challenging case.
3. The C4 symmetry breaking is now fully characterized (S4), so we know exactly what the symmetrization must fix.

**Why consider switching later:**
1. The five subspace bands are quasi-degenerate (internal gaps as small as 3×10⁻⁶), causing anti-crossing noise in FD derivatives → 31–39% C4 error in M_inv for bands 3,4 and 14.3% of M_inv grid points hitting the regularization clamp.
2. The kinetic/potential ratio is ~1/2830 — far too small for well-separated bound states. All 20 eigenmodes cluster within ~10⁻⁴ of V_max, making them nearly indistinguishable and hypersensitive to small perturbations.
3. A simpler candidate (fewer bands, better isolation, moderate kinetic/potential ratio) would provide a much cleaner validation target.

**Available alternatives:**

| Candidate | Lattice | r/a  | ε    | k₀      | Bands  | Notes                    |
|-----------|---------|------|------|---------|--------|--------------------------|
| Current   | Square  | 0.35 | 12.0 | Γ       | [5–9]  | 5-band, quasi-degenerate |
| B         | Square  | 0.29 | 7.9  | X-point | [0–2]  | 3-band, better isolated  |
| C         | Hex     | 0.29 | 3.7  | Γ       | [11–15]| 5-band, hex symmetry     |
| Custom    | —       | —    | —    | —       | —      | ~19k geometries in band library |

**Recommendation:** Complete Phase A (C4-symmetrization) and Phase B (A=0 test) on the current candidate. Then run the same pipeline on candidate B (X-point, 3 bands) as a validation cross-check.

---

## Q2: What parameters unleash the full power of the envelope approximation?

The envelope approximation works best when **six conditions** are simultaneously satisfied. The current candidate satisfies some but fails others critically.

### The six criteria (ranked by importance)

#### 1. Spectral isolation (large gap between subspace and exterior bands)

The interband coupling from slow modulation scales as:

$$\langle u_m | \partial_{R_j} u_n \rangle \sim \frac{\langle u_m | (\partial_{R_j} \mathcal{L}) | u_n \rangle}{\lambda_n - \lambda_m}$$

When the gap Δλ between the subspace and exterior bands is small, this coupling diverges → the adiabatic approximation breaks down → the effective mass tensor M_inv becomes unreliable.

**Current candidate:** Gap to exterior bands is moderate (~0.03), but *internal* gaps within the subspace [5–9] are tiny (3×10⁻⁶ to 2×10⁻⁴). This is the root cause of the M_inv noise.

**What to look for:** `S_gap` score in Phase 0 scoring. Want Δλ ≫ η · max|∂_R ε|.

#### 2. True extremum at k₀ (so drift vanishes)

At a band extremum, ∇_k ω(k₀) = 0 and the drift term vanishes identically. The mass tensor M_inv = ∂²ω/∂k² is then the leading kinetic contribution at O(η²), giving a clean Schrödinger-like equation.

- **True minimum:** M_inv > 0 (electron-like). Potential *well* traps modes.
- **True maximum:** M_inv < 0 (hole-like). Potential *hill* traps modes (inverted problem, still tractable).
- **Saddle point:** M_inv has mixed signs. Kinetic operator is *hyperbolic* rather than elliptic. Bound states are much harder to form.

**Current candidate:** Target band appears to be at a maximum (hole-like), which is fine. However, the drift is not exactly zero (v_drift max ≈ 3×10⁻² at some R-points), suggesting the extremum may shift with registry.

**What to look for:** `S_linear` penalty (non-zero group velocity at k₀). Verify ∇_k ω = 0 at k₀ for *all* registry values, not just the reference.

#### 3. Moderate kinetic/potential ratio (O(1)–O(10))

This is the **key dimensionless ratio** governing bound state physics:

$$\frac{V_{\text{depth}}}{E_{\text{kinetic}}} = \frac{\max(\Lambda) - \min(\Lambda)}{0.5 \cdot M^{-1}_{\text{typical}} / L_{\text{moire}}^2}$$

| Regime | Ratio | Physics |
|--------|-------|---------|
| Too small (< 1) | V_depth ≪ E_kin | No bound states, free-particle-like |
| **Optimal (1–10)** | V_depth ~ E_kin | **Few well-separated bound states, cleanly resolvable** |
| Too large (≫ 100) | V_depth ≫ E_kin | Many nearly-degenerate states clustered at band edge |

**Current candidate: ratio ≈ 2830** → deep into the "too large" regime. All 20 eigenmodes cluster within 10⁻⁴ of V_max. This makes:
- Individual mode classification unreliable (tiny C4 breaking can reorder modes)
- Size convergence extremely slow (modes barely differ from the flat-band limit)
- Numerical noise dominant over physical kinetic corrections

**How to improve:**
- **Increase twist angle θ**: larger η → larger kinetic scale E_kin ~ η². Going from θ=1.1° to θ=5° increases η² by ~20×.
- **Choose a more dispersive band**: larger |M_inv| → larger kinetic scale.
- **Choose a shallower potential**: smaller ε contrast → smaller V_depth.

**What to look for:** `S_flat` score (curvature magnitude). Ideally want `S_flat` not too high (flat → large mass → tiny kinetic scale).

#### 4. Stable k₀ across all R (no k-drift)

The two-scale ansatz uses a *global* carrier momentum: ψ(r) = e^{ik₀·r} Σ F_n(R) u_n(r;R). Making k₀ depend on R would put slow-scale dependence into the fast carrier phase, invalidating the derivative separation.

If the band extremum drifts with R — i.e., k_ext(R) ≠ k₀ — then the fixed-k₀ potential λ(R, k₀) can create a "fake well" that doesn't correspond to any real local spectral feature (see Q3 for full explanation).

**What to look for:** At a grid of R-values, compute ω(k, R) on a small k-grid around k₀. Verify the extremum stays at k₀ within tolerance ~ η.

#### 5. Modest geometry modulation

Even with η ≪ 1, if the *amplitude* of the slow modulation is strong (large index contrast, sharp geometric features), then ∂_R ε is large and the non-adiabatic couplings become non-perturbative.

**Current candidate:** ε contrast = 12.0/1.0 = 12:1, which is quite large. This contributes to strong off-diagonal Λ and large non-adiabatic coupling.

**What to look for:** `S_sym` score and the magnitude of off-diagonal Λ_{mn} relative to the gaps.

#### 6. Internal subspace isolation (no anti-crossings within subspace)

Within the N-band subspace, bands should not have near-degeneracies as a function of k near k₀. Anti-crossings cause:
- FD derivatives of ω(k) to be noisy (curvature changes sign abruptly)
- Berry connection A to spike (phase winding at avoided crossings)  
- Effective mass M_inv to diverge locally (1/Δ blowup)

**Current candidate:** Bands 5–9 have internal gaps as small as 3×10⁻⁶ → severe anti-crossing effects → M_inv errors concentrated in bands 3,4 (the ones nearest the quasi-degeneracy).

### Summary scorecard for current candidate

| Criterion | Status | Severity |
|-----------|--------|----------|
| 1. Spectral isolation (exterior) | ✓ OK | — |
| 2. True extremum | ~ Marginal | Moderate (drift not zero everywhere) |
| 3. Kinetic/potential ratio | ✗ FAIL | **Critical (ratio ≈ 2830)** |
| 4. k₀ stability | ? Unknown | Needs diagnostic |
| 5. Geometry modulation | ~ Large | Moderate (ε contrast 12:1) |
| 6. Internal isolation | ✗ FAIL | **Critical (gaps ~ 10⁻⁶)** |

---

## Q3: Why would k-point tracking / drift correction help?

### The physics of k-drift in moiré systems

The envelope approximation expands the Bloch dispersion around a fixed momentum k₀:

$$\omega_n(\mathbf{k}_0 + \Delta\mathbf{k}; \mathbf{R}) \approx \omega_n(\mathbf{k}_0; \mathbf{R}) + \mathbf{v}_n(\mathbf{R}) \cdot \Delta\mathbf{k} + \frac{1}{2} M^{-1}_{ij,n}(\mathbf{R}) \Delta k_i \Delta k_j$$

At the expansion point k₀, we *want* the group velocity (drift) to vanish:

$$\mathbf{v}_n(\mathbf{R}) = \nabla_{\mathbf{k}} \omega_n\big|_{\mathbf{k}_0} = 0$$

because then the leading envelope-scale physics is dominated by the O(η²) kinetic term rather than the O(η) drift term.

### Why does k₀ drift?

In a moiré structure, the local crystal environment at position R differs from the reference crystal. The dielectric function varies:

$$\varepsilon(\mathbf{r}; \mathbf{R}) \neq \varepsilon(\mathbf{r}; \mathbf{R}')$$

This means the **band structure itself changes with R**. A band that has its extremum at k₀ = Γ in the reference crystal may have its extremum shifted to k₀ + δk(R) at a different registry point. The drift velocity is:

$$v_n(\mathbf{R}) = \nabla_{\mathbf{k}}\omega_n(\mathbf{k}_0; \mathbf{R}) \neq 0 \quad\text{when } k_0 \neq k_{\text{ext}}(\mathbf{R})$$

### Three consequences of uncompensated k-drift

**1. Fake potential wells.** The "potential" in the envelope equation is λ(R, k₀) — the band energy evaluated at the fixed expansion point. If the true band edge is at k_ext(R) ≠ k₀, then:

- λ(R, k₀) ≠ min_k ω(k, R) (or max, for a hole band)
- The potential landscape can show a "beautiful well" that doesn't correspond to any real spectral feature
- Meep sees no bound mode, or a mode at a completely different location

**2. The drift term dominates.** When v(R) ≠ 0, the drift term is O(η) while the kinetic term is O(η²). For small twist angles this ratio η/η² = 1/η can be enormous (~50 for θ=1.1°). The drift term becomes the dominant dynamics, swamping the kinetic corrections that create bound states.

**3. The envelope equation becomes first-order.** A large drift term converts the Schrödinger-like equation (2nd order, supports localized states) into a transport equation (1st order, propagating waves). Bound states require the 2nd-order kinetic term to compete with the potential.

### Why not make k₀ depend on R?

Making k₀(R) position-dependent would break the two-scale ansatz. The wavefunction ansatz is:

$$\psi(\mathbf{r}) = e^{i\mathbf{k}_0 \cdot \mathbf{r}} \sum_n F_n(\mathbf{R})\, u_n(\mathbf{r}; \mathbf{R})$$

The carrier phase e^{ik₀·r} is a **fast-scale** object. If k₀ depends on R (the slow scale), this puts slow-scale modulation into the fast carrier phase, invalidating the clean derivative separation:

$$\nabla_{\mathbf{x}} = \frac{1}{a}\left(\nabla_{\mathbf{r}} + \eta\,\nabla_{\mathbf{R}}\right)$$

### The multi-band alternative

Instead of tracking k₀(R), the multi-band approach captures the same physics through **off-diagonal interband coupling**. When k₀ shifts, the envelope "leaks" amplitude into adjacent bands via the off-diagonal drift and kinetic terms. Including enough bands in the subspace lets the envelope equation automatically account for the shifting extremum — the carrier momentum adjustment is encoded in the band-mixing coefficients F_n(R).

At a true band extremum, the diagonal drift vanishes. But **off-diagonal drift** (coupling between bands m ≠ n in the subspace) can still be large near degeneracies, capturing the interband physics that a single-band model with drifting k₀ would need.

### Practical diagnostic: is k-drift a problem for our candidate?

**Test protocol:**
1. At a grid of R-values (e.g., 16×16 across the moiré cell), compute ω(k, R) on a small k-grid (e.g., 5×5 around Γ).
2. For each R, locate the actual extremum k_ext(R) by interpolation.
3. Map |k_ext(R) − k₀| across the moiré cell.
4. If max|k_ext − k₀| ≫ η ≈ 0.019, the fixed-k₀ assumption is problematic.

**Current evidence:** v_drift max ≈ 3×10⁻² is nonzero but modest. Whether this indicates a real k-shift or just numerical noise needs investigation.

### Summary

| Aspect | Fixed k₀ | Tracking k₀(R) | Multi-band |
|--------|----------|----------------|------------|
| Two-scale separation | ✓ Valid | ✗ Breaks ansatz | ✓ Valid |
| Drift term | May be O(η), too large | Zero by construction | Captured by off-diagonal coupling |
| Potential | May be "fake well" | True band edge | True subspace potential |
| Implementation | Simple | Hard (position-dependent phase) | Current approach |
| When it works | Extremum stable at k₀ | Always | Always (if subspace large enough) |

**Bottom line:** k-point tracking is the *wrong* solution (breaks the formalism). The right approach is multi-band with enough bands to capture the drift physics through interband coupling. But it's critical to *verify* that the extremum is actually at k₀ — if it's not, the potential landscape is misleading.

---

## Implications for Next Steps

These three answers converge on the same message:

1. **C4-symmetrize first** (Phase A) — this is the immediate fix for the gauge-induced symmetry breaking, independent of candidate choice or k-drift concerns.

2. **Diagnose k-drift** (new Phase D task) — verify the band extremum stays at Γ across all R. If it drifts, the potential well may be a numerical artifact.

3. **Consider the kinetic/potential ratio** — the current ratio of ~2830 is pathologically large, causing eigenvalue clustering. Either increase θ or choose a more dispersive band.

4. **Validate on a simpler candidate** — after proving the method on the current (hard) case, cross-check on the X-point candidate (3 bands, likely better isolation and ratio).
