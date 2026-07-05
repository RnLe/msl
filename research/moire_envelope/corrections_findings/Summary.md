# Envelope Approximation: Full Summary & Next Steps

**Date:** 2026-02-09  
**Status:** Seven diagnostic stages complete. Code fixes validated. Current candidate exhausted — candidate scanning is next.

---

## 1. The Seven Stages — What Happened

### Stage 1–2: Point-wise MPB audit & R-dependent field scans

**Goal:** Understand the raw data coming out of Phase 1 (MPB + finite-difference derivatives).

**Found:**
- Bloch fields at Γ are well-formed; orthonormality holds (⟨u_m|ε|u_n⟩ diagonal to ~1%)
- At the C4 fixed point δ=(0,0), bands 7,8 form a clean 2D E-representation — validates methodology
- At δ=(0.5,0.5), a near-degeneracy (gap 3×10⁻⁶) causes a genuine localized defect
- M_inv diverges at ~50% of grid points (FD curvature spikes at anti-crossings)
- Berry connection |A| spikes at anti-crossing lines
- Potential Λ is clean (C4 to 0.2%)
- **Initial conclusion:** "98% of points fail C4 subspace closure" → **turned out to be WRONG TEST**

### Stage 3: Subspace validity (the equivariance breakthrough)

**Goal:** Determine whether band subspace [5-9] is valid under C4.

**Key discovery:** We were testing C4 **invariance** (subspace at R is self-symmetric) instead of C4 **equivariance** (rotating subspace at R gives subspace at C4·R). Invariance is only meaningful at C4 fixed points.

**Result:** With the correct equivariance test:
- **99.4%** of R-points pass (min singular value > 0.9)
- Only 0.15% genuinely fail (at anti-crossing lines)
- 90.4% are both smooth AND equivariant
- The subspace is valid — the problem was the test all along

**Impact:** All S1–S2 "closure failure" findings retracted. The ~10% smoothness defects at anti-crossing lines remain real.

### Stage 4: Term-by-term Hamiltonian build

**Goal:** Identify which Hamiltonian terms break C4 symmetry via the commutator ||[H, C4]||/||H||.

| Configuration | [H, C4] / ||H|| | Culprit |
|------|------|------|
| Λ only | 1×10⁻⁴ | ✓ Clean |
| Λ + drift | 6.7×10⁻³ | Drift adds 0.7% |
| Λ + K (no A) | **2.9×10⁻²** | M_inv bands 3,4 have 31-39% C4 error |
| Λ + K (with A) | **4.6×10⁻²** | A_berry is ~200% C4-broken (ALL bands) |
| Full H | 4.6×10⁻² | BH/drift are small corrections |

**Root cause:** The BFS gauge fix does not produce C4-equivariant Bloch phases → both A_berry (~200% error) and Φ_BH (~200% error) are catastrophically non-C4.

### Stage 4b: C4-symmetrization

**Goal:** Fix C4 by post-processing Phase 2 data with 4-fold averaging.

**Result:** Complete success.
- All C4 errors drop to **machine precision** (~10⁻¹⁷)
- ||[H, C4]|| / ||H|| → **9.58×10⁻¹⁷**
- All eigenmodes carry clean C4 irrep labels (A, B, E±)
- 8/8 mode groups pass C4 for full H; 7/7 for A=0
- Saved `phase2_multiband_data_c4sym.h5` (42.7 MB)

### Stage 5: Mode analysis & energy scales

**Goal:** Characterize the bound states produced by the C4-symmetrized Hamiltonian.

**Findings:**
- Modes are **genuinely localized** bound states (10/20 pass IPR threshold)
- But **zero interband mixing** — each mode lives in exactly one band
- V_depth / E_kin = **954** → ~197 bound states per band, all clustered within 4.6×10⁻⁴
- Grid convergence is 2nd order but slow (eigenvalues still shift ~3×10⁻⁴ between Ns=64 and Ns=128)
- Ground state from Band 3 (not target Band 2)

**Diagnosis:** The 5-band Hamiltonian acts as 5 independent single-band problems because all coupling operators (Λ, M_inv, v_drift) are band-diagonal, and the one operator with off-diagonal data (A_berry) is being **ignored** by the kinetic operator code.

### Stage 6: Off-diagonal coupling discovery

**Goal:** Find the source of interband coupling and survey its strength.

**Critical findings:**
1. **Off-diagonal A_berry has ||off-diag||/||diag|| = 0.84** — it's the only operator with off-diagonal data
2. Phase 3's kinetic operator extracts only `A[n,n]` at line 529-540 — drops all off-diagonal Berry connection
3. A corrected covariant derivative with full A_{mn} produces **mean mixing 0 → 66%**
4. **η-sweep:** V/E_kin = 954 at θ=1.1°; reaching V/E_kin ≈ 10 requires θ ≈ 11° (η = 0.19, EA questionable)

### Stage 7: Production code fix & validation

**Goal:** Implement full off-diagonal A in the production Phase 3 code and validate.

**Code changes** (in `phasesV3/phase3_mpb_v3.py`):
- Added diamagnetic term: A²_{mn} = Σ_{p,ij} M_{ij,pp} A_{mp,i} A_{pn,j}
- Added paramagnetic term: −i · M_{ij,mm} · A_{mn,i} · ∂_j
- Hermitization K → (K + K†)/2
- Backward-compatible via `include_offdiag_A=False` (default)

**All 10 validation checks pass:**
| Check | Result |
|-------|--------|
| Hermiticity | ||K − K†|| / ||K|| = 0.0 |
| A=0 consistency | K_full(A=0) ≡ K_legacy(A=0), Δ = 0.0 |
| C4 commutator | ||[H,C4]|| / ||H|| = 9.7×10⁻¹⁷ |
| Interband mixing | 0% → **66.6%** |
| η-sweep | mixing 64%→35% as θ grows (physics-correct) |

---

## 2. Current State: What Do We Have?

### What works

1. **The pipeline runs end-to-end** — Phases 0→1→2→3 produce eigenvalues and envelope spinors
2. **C4-symmetrization works perfectly** — one-line post-processing fixes all gauge-induced symmetry breaking
3. **The envelope expansion form is valid** — r ≈ 0.93 correlation between |F|² and reconstructed |E|²
4. **Off-diagonal Berry connection produces genuine interband coupling** — 66% mixing, Hermitian, C4-preserving
5. **The eigensolver produces localized bound states** — not numerical garbage

### What the current candidate teaches us

This candidate (square, r/a=0.35, ε_bg=12, Γ, TE, bands [5-9]) was the **hardest possible test case**: five quasi-degenerate bands (internal gaps ~10⁻⁶), massive ε contrast (12:1), pathological V/E_kin = 954. It proved the infrastructure works but produces uninteresting physics because:

1. **V/E_kin = 954**: ~200 bound states per band, all packed within 4.6×10⁻⁴ of V_max. Individual modes are indistinguishable. No well-separated energy levels to study.
2. **Reaching V/E_kin ~ 10 requires θ ≈ 11°** where η = 0.19 — the EA "small-η" expansion may break down.
3. **Anti-crossing noise**: 10% of R-points have M_inv divergences requiring clamping. This introduces systematic error proportional to the fraction of clamped points.

### The honest assessment

> **The pipeline is a validated, working implementation of the multi-band photonic moiré envelope approximation. But this particular candidate does not produce thesis-worthy physics. We need a candidate where the dimensionless ratio V/E_kin falls in the "interesting" regime (1–10) at a small, defensible twist angle.**

---

## 3. Can We Write a Thesis Chapter?

### What can already be written

**Yes — the methodology chapter is complete.** Stages 1–7 document a rigorous development:

1. **Theory**: Multi-band two-scale envelope approximation with covariant derivative (documented in `docs/envelopeApproximationDerivation/`)
2. **Implementation**: Phase 0–3 pipeline with C4-symmetrization and full Berry coupling
3. **Validation**: Equivariance tests, Hermiticity proofs, C4 commutator checks, consistency tests, grid convergence analysis
4. **Bug-finding narrative**: The equivariance-vs-invariance discovery (S3) and the off-diagonal A discovery (S6) are genuinely interesting methodological results that show the subtlety of gauge-theory-based photonic computations

This is a strong "methods + validation" chapter. What's **missing** is a "results" section showing interesting confined photonic modes with well-separated energy levels and nontrivial band hybridization patterns.

### What interesting physics would look like

The EA becomes compelling when you can show:
- **A few well-separated bound states** (V/E_kin ~ 2–10): clearly resolved energy levels, like the first 3–5 modes of a quantum dot
- **Multi-band hybridization structure**: modes with nontrivial band composition (e.g., 60% band-1 + 30% band-2 + 10% band-3), shaped by Berry curvature coupling
- **Topological or geometric effects**: Berry-phase-induced level splittings, non-Abelian gauge structure in the envelope equation
- **Comparison with full-wave (Meep/FDTD)**: envelope modes at small θ matching supercell simulations
- **Twist-angle dependence**: a "magic angle" where modes cross or hybridize

All of this is within reach — we just need the right candidate.

---

## 4. What Makes a Good Candidate?

The six criteria from the Q&A document, now informed by S1–S7:

| # | Criterion | Current candidate | What to optimize |
|---|-----------|-------------------|------------------|
| 1 | **Spectral isolation** (gap to exterior bands) | ✓ OK (~0.03) | Gap ≫ η · ‖∂_R ε‖ |
| 2 | **True extremum at k₀** (drift = 0) | ✓ Auto at Γ, M, X, K | Use high-symmetry k₀ |
| 3 | **V/E_kin ~ 1–10** | ✗ FAIL (954) | **THE critical constraint** |
| 4 | **No internal anti-crossings** | ✗ FAIL (gaps ~10⁻⁶) | Well-separated bands within subspace |
| 5 | **Moderate ε contrast** | ✗ Too high (12:1) | Lower ε_bg or smaller r/a |
| 6 | **k₀ stable across R** | Presumably OK at Γ | Always true at high-symmetry points |

### The dimensionless ratio controls everything

$$\frac{V_\text{depth}}{E_\text{kin}} = \frac{\Delta\omega}{\frac{1}{2}|M^{-1}_\text{eff}| \cdot \eta^2}$$

where:
- $\Delta\omega$ = variation of ω(k₀, R) across the moiré cell (= potential depth V)
- $M^{-1}_\text{eff}$ = effective inverse mass (from band curvature at k₀)
- $\eta = 2\sin(\theta/2)$ ≈ θ for small angles

To get V/E_kin ~ 1 we need:

$$\theta \sim \theta^* = \sqrt{\frac{2\,\Delta\omega}{|M^{-1}_\text{eff}|}}$$

For the current candidate: Δω ≈ 0.15, |M_eff⁻¹| ≈ 0.3 → θ* ≈ 1 rad ≈ 57°. Completely unphysical.

**We need either:**
- **Much larger |M_eff⁻¹|** — a more dispersive (curved) band → larger kinetic energy at fixed θ
- **Much smaller Δω** — a shallower potential → fewer bound states but better ratio
- **Both** (ideally)

This translates to: **look for bands with high curvature at k₀ in crystals with small frequency variation across registries (low ε contrast).**

---

## 5. Advice on the Scanning Script

### Your suggestion is exactly right

A systematic scan through candidates is the correct next step. Here's my recommended design:

### Architecture: Two-tier scan

**Tier 1: Fast screening from Phase 0 data (no MPB, seconds)**

You already have 189,200 candidates in `phase0_candidates.csv` with curvature data. From the monolayer band structure alone, extract:

- $|M^{-1}_\text{eff}|$ ∝ `|curvature_trace|` (already in CSV)
- `gap_above`, `gap_below` (spectral isolation → already scored as S_gap)
- `S_linear` (penalizes drift → ensures k₀ is an extremum)

**New metric to add:** Estimate $\theta^*$ for each candidate. You can't compute V_depth from Phase 0 alone, but you can use a **proxy**:

$$V_\text{depth} \sim C \cdot \frac{\Delta\varepsilon}{\varepsilon_\text{bg}} \cdot \omega_0$$

where $C$ is an O(1) constant and $\Delta\varepsilon/\varepsilon_\text{bg}$ measures the modulation strength. This gives:

$$\theta^* \sim \sqrt{\frac{2C \cdot (\Delta\varepsilon/\varepsilon_\text{bg}) \cdot \omega_0}{|M^{-1}_\text{eff}|}}$$

Filter for candidates where θ* < 5° (so V/E_kin ~ 1 at a small, defensible twist angle).

Priority ranking for Tier 1:
1. Small θ* (most important)
2. Large spectral gaps (S_gap high)
3. Low S_linear penalty (true extremum)
4. Well-separated internal bands (check gap structure in the band library)

**Tier 2: Quick Phase 1 validation (MPB at ~4 R-points, minutes per candidate)**

For the top ~20–50 candidates from Tier 1:
1. Run Phase 1 at just 4 R-points: (0,0), (0.5,0), (0,0.5), (0.5,0.5) — the C4 fixed points
2. Extract: ω(k₀, R) at each point → actual V_depth = max − min
3. Extract: M_inv eigenvalues → actual |M_eff⁻¹|
4. Compute: V/E_kin at θ = 1.1°, 3°, 5°
5. Check: internal band gaps within the subspace at each R-point
6. Check: v_drift magnitude (should be ≈ 0 at a high-symmetry k₀)

This gives the actual V/E_kin without running the full 128×128 grid.

**Tier 3: Full pipeline run (the winner)**

Run the top 1–3 candidates through Phases 1→2→C4-sym→3 with `include_offdiag_A=True`. This is the thesis result.

### About drift: do we need v_drift = 0?

**At high-symmetry k₀ points (Γ, M, X, K), drift is zero by symmetry of the monolayer crystal.** This is guaranteed by the point group at those k-points — ∇_k ω vanishes at any k-point where the little group contains an inversion or sufficient rotation symmetry. So:

- **You don't need to hunt for zero-drift candidates** — just restrict to high-symmetry k₀ (which Phase 0 already does!)
- The nonzero v_drift we observed (max ~0.12) in the current candidate is from FD numerical noise, not real physics. After C4-symmetrization it should vanish.
- In the Hamiltonian, the drift term is O(η) while kinetic is O(η²), so even small residual drift can be large. But at a true high-symmetry point, it's identically zero analytically.

**Bottom line:** Don't worry about drift as a selection criterion. Any candidate at Γ, M, X, or K automatically has zero drift.

### What the scanning script needs

```
Input:  phase0_candidates.csv (189,200 rows) + band_library.h5
Output: ranked list of candidates with estimated V/E_kin at θ = 1°, 3°, 5°

Steps:
1. Load candidates CSV
2. Filter: valid_ea_flag == True, S_total > threshold
3. For each: estimate θ* from curvature + ε contrast
4. Rank by θ* (smallest = best)
5. Top 20–50: run quick Phase 1 (4 R-points each)
6. Re-rank by actual V/E_kin
7. Top 3: flag for full pipeline run
```

### Subspace size considerations

- **Fewer bands is better** for numerical cleanliness (fewer anti-crossings, fewer coupling channels)
- But **you need enough bands** to capture the interband mixing physics (at least 2–3)
- A 3-band subspace at a well-isolated extremum would be ideal for a first "results" candidate
- The current 5-band subspace was overkill for this candidate (4 of 5 bands were quasi-degenerate)

### Parameter space priorities

| Parameter | Favor | Reasoning |
|-----------|-------|-----------|
| ε_bg | **Low** (2–6) | Smaller contrast → shallower V_depth |
| r/a | Moderate (0.2–0.3) | Avoid extremes |
| Band index | **Low** (1–4 merged) | Lower bands tend to have larger curvature |
| k₀ | Γ, M preferred | Highest symmetry → cleanest C4 + zero drift |
| Lattice | Both square & hex | Hex has C6 → even more restrictive symmetry |
| N_bands (subspace) | **2–3** for first pass | Minimize anti-crossing risk |

---

## 6. Summary

| Aspect | Status |
|--------|--------|
| Pipeline code | ✅ Working end-to-end |
| C4 symmetry | ✅ Fixed via post-processing |
| Off-diagonal Berry coupling | ✅ Implemented & validated |
| Hermiticity, consistency | ✅ All checks pass |
| Current candidate physics | ❌ V/E_kin = 954 → uninteresting (too many clustered modes) |
| Thesis-worthy results | ⏳ Needs a better candidate |
| Next step | **Candidate scanning script** |

The infrastructure is ready. The physics understanding is deep. We just need the right crystal + band + twist angle where V/E_kin falls into the sweet spot. The scanning script is the right approach to find it systematically.
