# Corrections & Findings — Systematic Rebuild

> **Status (post-thesis):** this S1–S7 diagnostic arc validated the V3
> methodology (gauge fixing, symmetrization, term-by-term Hamiltonian audit).
> All fixes documented here are folded into the production pipeline
> (`../phasesV3/`); the folder is kept as the record of *why* the pipeline is
> correct. `S4b_c4_symmetrize.py` remains an active pipeline step.

## 0. Key Findings Summary

> **Full reports: [`FINDINGS_S1_S2.md`](FINDINGS_S1_S2.md) (stages 1–2), [`FINDINGS_S3.md`](FINDINGS_S3.md) (stage 3 + equivariance discovery), and [`FINDINGS_S4.md`](FINDINGS_S4.md) (stage 4 — term-by-term Hamiltonian)**

### ~~FATAL: Subspace [5-9] is not closed under C4~~ → **WRONG TEST**

**RETRACTED (2026-02-08).** This finding was based on C4 **invariance** testing, which checks if the subspace at R is self-symmetric under C4. The correct test is C4 **equivariance** — whether rotating the subspace at R gives the subspace at C4·R. Invariance is only meaningful at C4 fixed points (δ=0 and δ=0.5).

With the **correct** equivariance test:
- **99.4%** of R-points have equivariance min σ > 0.9
- **0.15%** of R-points genuinely fail equivariance
- The "98.3% failure" was entirely a testing artifact

See [FINDINGS_S3.md](FINDINGS_S3.md) for the full explanation of why this happens (crystal C4 symmetry guarantees equivariance of energy-ordered eigenstates).

### CONFIRMED: Subspace smoothness failures at ~10% of points

The **real** issue: 9.4% of registry points have adjacent-point subspace overlap < 0.5, caused by anti-crossings where bands from outside [5-9] exchange character with bands inside. These occur along narrow lines in registry space.

### CONFIRMED: M⁻¹ introduces 3% C4 breaking (bands 3,4 worst)

S4 measured `||[H,C4]||/||H||` for M_inv alone (no A): **2.93%**. Bands 0–2 have acceptable C4 tensor error (~0.6%), but bands 3,4 have **31–39% C4 error** in `M(C4·R) vs C4·M(R)·C4ᵀ`. The divergences (14.3% of points clamped at |eig|=20) plus poor C4 at anti-crossing bands make M_inv the second-largest C4 breaker.

### ~~STILL VALID~~ → **CRITICAL: Berry connection is 200% C4-broken**

S4 revealed `A_berry` does not transform as a vector under C4 at ALL — C4 error is ~200% for **every** band. This is not just anti-crossing noise; the BFS gauge fix does not enforce C4-equivariant phases. A_berry is the single largest source of C4 breaking in the Hamiltonian (increases `||[H,C4]||/||H||` from 3% → 4.6%).

### CRITICAL: Φ_BH is 200% C4-broken

Born-Huang potential has the same ~200% C4 error as A_berry. Both are computed from `∂u/∂R` which inherits the gauge disorder.

### CONFIRMED OK: δ=(0,0) is perfectly C4

Validates the test methodology. Bands 7,8 correctly form a 2D E-representation. (At fixed points, invariance = equivariance, so this was always a valid test.)

### CONFIRMED OK: Potential Λ(R) is C4 to within 0.2%

Scalar eigenvalues don't depend on gauge or field phases.

### CONFIRMED: δ=(0.5,0.5) fixed point is genuinely problematic

At this C4 fixed point, equivariance = invariance, and it fails (near-degeneracy of bands 7,8 with gap 3×10⁻⁶). This is a localized point defect, not a global issue.

### ✅ C4-symmetrization FIXES the Hamiltonian (S4b)

Post-processing all Phase 2 data with 4-fold averaging:
$$Q^{\text{sym}}(R) = \frac{1}{4}\sum_{n=0}^{3} \mathcal{T}_{C_4^n}[Q(C_4^{-n} R)]$$
reduces all C4 errors from the 2–200% level to **machine precision** (10⁻¹⁷). Consequently:
- `||[H, C4]||/||H||` drops from 4.57×10⁻² to **9.58×10⁻¹⁷**
- All eigenmodes carry clean C4 irrep labels (A, B, E±)
- 8/8 mode groups pass C4 for full H; 7/7 for A=0 variant
- Ground state: E doublet (full H) or B singlet (A=0)

### OFF-DIAGONAL BERRY CONNECTION enables interband mixing (S6)

Phase 3's kinetic operator ignores off-diagonal A_berry (only uses A_{nn}). The data has ||A_offdiag||/||A_diag|| = **0.84** — comparable magnitude! A corrected covariant derivative using all A_{mn} terms produces:
- Mean interband mixing: **0 → 0.66** (standard → corrected)
- Each mode hybridizes across 2-4 bands (50-72% mixing) instead of living in a single band
- **This is the missing interband coupling channel** — Λ, v_drift, M_inv, Φ_BH are all band-diagonal

### This candidate is PATHOLOGICAL for the envelope approximation (S6)

V_depth/E_kin = 954 at θ=1.1° means ~197 bound states per band, all clustered within 4.6×10⁻⁴. Reaching V/E_kin ≈ 10 requires θ ≈ 11° where η=0.19 and the EA (which assumes η ≪ 1) is questionable.

---

## 1. Current State Assessment

### 1.1 What the pipeline does

The moiré envelope approximation pipeline computes confined photonic modes in twisted photonic crystals via a multi-band effective Hamiltonian, in four phases:

| Phase | Input | Output | Grid |
|-------|-------|--------|------|
| **Phase 0** | Band library (MPB precompute) | Crystal candidate (lattice, k₀, bands) | — |
| **Phase 1** | Candidate + θ | ω(s), vg(s), M_inv(s), u_n(r;s) at k₀ | 128×128 (scalars) / 64×64 (fields) |
| **Phase 2** | Phase 1 data | Λ, A_berry, M_inv, v_drift, Φ_BH (gauge-fixed) | 128×128 |
| **Phase 3** | Phase 2 data | Eigenvalues + envelope spinors F_n(R) | 128×128×5 → eigsh |

**Crystal**: square lattice, a=1.0, r/a=0.35, ε_bg=12.0, ε_hole=1.0.
**Target band**: band 7 (merged TE+TM index), TE polarization, at Γ-point (k₀=0).
**Subspace**: bands [5,6,7,8,9] → N_bands=5, target_index=2.
**Reference frequency**: ω_ref = 0.7913 c/a.

### 1.2 What works (infrastructure)

- **End-to-end code flow**: Phases 0→1→2→3 run without errors and produce files.
- **Eigensolve converges**: eigsh finds 100+ modes in <5s on 5120-dim matrices.
- **Field reconstruction** (Phase 4): Can multiply F_n(R)·u_n(r;R) and recover full-wave fields.
- **Envelope correlation**: r ≈ 0.93 between envelope |F|² and reconstructed |E|² — meaning the *expansion form* is valid, even if the Hamiltonian terms are wrong.
- **Scaling machinery**: eta sweep over 8 angles works.
- **C4 equivariance of subspace**: 99.4% of points pass, guaranteed by crystal symmetry.

### 1.3 What is broken / suspect

#### ❌ C4 symmetry violation in envelope modes (Phase 3 output)

The ground-state envelope |F(R)|² should exhibit C4 symmetry. **It does not.** Now that we understand the subspace is correct, the remaining causes are:

1. **Anti-crossing smoothness defects** (~10% of points): FD derivatives at these points produce noisy M_inv and A_berry, which feed into the kinetic operator
2. **Gauge disorder** in u_n(r;R): random phases from MPB + incomplete BFS gauge fix → noisy Berry connection
3. **M_inv divergence**: consequence of (1), currently handled by clamping, but clamping may break C4
4. **The (32,32) fixed-point defect**: bands nearly degenerate → degenerate-subspace issues

#### ❌ No convergence with system size

Eigenvalues and mode shapes change qualitatively between Ns=32 and Ns=128. Likely caused by the Berry connection and M_inv noise contaminating the kinetic operator at all scales.

#### ⚠️ Born-Huang Φ_BH ≈ 1.5×10⁻³ (seems real but tiny)

200× smaller than potential well depth (~0.3). Physically plausible for a mostly-smooth subspace (90%+ smooth).

#### ⚠️ Off-diagonal drift v_drift = 0 (diagonal only)

Phase 2 sets v_drift_mn = vg_n · δ_mn. Off-diagonal elements never computed. Likely OK for Γ-point (vg=0 by symmetry).

#### ⚠️ M_inv is diagonal in bands

Only diagonal blocks M_inv[:,:,n,n,:,:] populated. Off-diagonal mass (Löwdin correction) not computed.

---

## 2. Theory Reference

See `docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md` for the complete derivation. The envelope Hamiltonian is:

$$\hat{H} = \underbrace{\Lambda(\mathbf{R})}_{\text{potential}} + \underbrace{\eta \sum_i v^{(i)} (-i\mathcal{D}_i)}_{\text{drift}} + \underbrace{\frac{\eta^2}{2} \sum_{ij} M^{-1}_{ij} (-i\mathcal{D}_i)(-i\mathcal{D}_j)}_{\text{kinetic}} + \underbrace{\eta^2 \Phi_{BH}(\mathbf{R})}_{\text{Born-Huang}}$$

where $\mathcal{D}_j = \partial_{R_j} - i A_j(\mathbf{R})$ is the gauge-covariant derivative.

### C4 symmetry requirements

The envelope Hamiltonian must commute with C4 to produce symmetric eigenmodes. This requires:
- **Λ(C4·R) = Λ(R)**: ✓ guaranteed (scalar eigenvalues)
- **A(C4·R) = C4·A(R)**: requires proper gauge (covariant transformation)
- **M_inv(C4·R) = C4·M_inv(R)·C4ᵀ**: requires smooth derivatives
- **Subspace at C4·R = C4·(subspace at R)**: ✓ 99.4% equivariant (guaranteed by crystal symmetry)

The key insight from S3d: the **subspace** transforms correctly. The problem is that the **derived quantities** (A, M_inv) are computed from noisy FD derivatives at anti-crossing defects and don't transform correctly.

---

## 3. Diagnostic Stages — Checklist

### Stage 1: Single-R-point MPB audit ────────────────────────────

- [x] **S1.1** Pick representative R-points: δ=(0,0), δ=(0.25,0), δ=(0.5,0), δ=(0.5,0.5)
- [x] **S1.3** Check Bloch function orthonormality: ⟨u_m|ε|u_n⟩ → diag 1.00–1.10, off-diag up to 0.01
- [x] **S1.4** C4 symmetry at **fixed points**: **PERFECT at δ=(0,0)** (E-rep validated). **BROKEN at δ=(0.5,0.5)** (near-degenerate bands, genuine defect)
- [x] **S1.5** Band ordering: no frequency crossings within [5-9], but internal gaps as small as 3×10⁻⁶
- [x] **S1.6** k-stencil: vg≈0 for bands 5–7 (OK). M⁻¹ divergent at 50%+ (consequence of anti-crossing defects)

### Stage 2: R-dependent field symmetry scan ─────────────────────

- [x] **S2.1** ~~Subspace C4 closure~~ → **TEST WAS WRONG** (used invariance, not equivariance). Retested in S3d: 99.4% equivariant.
- [x] **S2.2** Λ_nn(R): C4 to within 0.2% ✓. Depth ~0.15–0.23
- [x] **S2.3** v_drift: diagonal small (OK), off-diagonal exactly ZERO ⚠
- [x] **S2.4** M_inv: divergent at 50%+ points ❌ — caused by anti-crossing smoothness defects
- [x] **S2.5** A_berry: |Im| up to 1.8 ❌ — caused by FD noise at anti-crossing defects
- [x] **S2.6** Φ_BH: ~1.5e-3, uniform, real. OK.
- [x] **S2.7** ~~No subspace achieves C4 closure~~ → **WRONG TEST** (invariance; all would pass equivariance)
- [x] **S2.extra** Band mapping at (32,32): band 5→4 leakage confirmed at this fixed point only

### Stage 3: Subspace validity & band tracking ───────────────────

> **See [`FINDINGS_S3.md`](FINDINGS_S3.md) for the complete analysis.**

- [x] **S3.1** S3: Global overlap band reordering (Hungarian) → *IRRELEVANT: tried to fix non-problem using wrong test*
- [x] **S3.2** S3b: BFS 18-band parallel transport → *HARMFUL: broke equivariance from 99.4% → 18.1%*
- [x] **S3.3** S3c: C4 symmetrization / fundamental domain → *FAILED: wrong conceptual basis*
- [x] **S3.4** S3d: **DEFINITIVE** equivariance vs invariance test → **Original [5-9] is 99.4% C4-equivariant**
- [x] **S3.5** S3e: Subspace smoothness + combined quality → **90.4% both smooth AND equivariant**. 9.4% have smoothness defects at anti-crossing lines.

**Stage 3 verdict:** The subspace is valid. The problem was the test, not the physics. The remaining issues are localized anti-crossing smoothness defects (10%) and the (32,32) fixed-point defect.

### Stage 4: Term-by-term Hamiltonian build ──────────────────────

> **See [`FINDINGS_S4.md`](FINDINGS_S4.md) for the complete analysis.**

- [x] **S4.1** H = Λ only → `||[H,C4]||/||H||` = **1.0e-4** ✓ (eigsh singular — Λ is spatially diagonal, no spatial coupling)
- [x] **S4.2** H = Λ + drift → `||[H,C4]||/||H||` = **6.7e-3** ⚠ (drift breaks C4 0.7%; v_drift ≠ 0 at generic R expected)
- [x] **S4.3** H = Λ + K (no A) → `||[H,C4]||/||H||` = **2.9e-2** ❌ (M_inv bands 3,4 have 31–39% C4 error)
- [x] **S4.4** H = Λ + K (with A) → `||[H,C4]||/||H||` = **4.6e-2** ❌ (A adds 56% more C4 breaking; A_berry is 200% C4-broken)
- [x] **S4.5** H = full → `||[H,C4]||/||H||` = **4.6e-2** ❌ (same as S4.4; BH+drift are small corrections)

**Stage 4 verdict:** C4 symmetry breaking is **dominated by two terms**: (1) M_inv contributes 3% via noisy band-3,4 mass tensors; (2) A_berry contributes an additional 1.6% via 200% gauge-broken Berry connection. The potential Λ is essentially perfect (0.01%). The BFS gauge fix does NOT produce C4-equivariant phases → both A_berry (all bands ~200% error) and Φ_BH (~200% error) are catastrophically non-C4.

### Stage 4b: C4-symmetrization post-processing ──────────────────

> **Script: `S4b_c4_symmetrize.py`**

- [x] **S4b.1** C4-symmetrize all Phase 2 quantities → all errors drop to machine precision (~10⁻¹⁷)
- [x] **S4b.2** Rebuild H from symmetrized data → `||[H,C4]||/||H||` = **9.58×10⁻¹⁷** ✓
- [x] **S4b.3** Original H eigenmodes: 0/12 pass C4 (all closure < 0.37)
- [x] **S4b.4** C4-sym full H: **8/8 mode groups pass C4** (closure = 1.000000 for all)
- [x] **S4b.5** C4-sym A=0: **7/7 mode groups pass C4** — cleanest case, B singlet ground state
- [x] **S4b.6** C4-sym Λ+K only (no drift): 3/7 pass — degenerate E subspaces need drift to resolve
- [x] **S4b.7** Saved symmetrized data to `phase2_multiband_data_c4sym.h5` (42.7 MB)

**Key findings:**
- Ground state: E doublet (full) or B singlet (A=0) — Berry connection affects spectrum structure
- Clean degeneracy patterns: E pairs exactly degenerate, singlets well-separated
- Eigenvalue spread: [0.13126, 0.13149] (full), [0.13125, 0.13154] (A=0) — still clustered near V_max
- Λ+K-only partial C4 failure is NOT a physics issue — ARPACK returns arbitrary combinations within exact degenerate subspaces; drift lifts degeneracy and reveals C4 eigenstates

**Stage 4b verdict:** C4-symmetrization completely fixes the symmetry problem. Phase A ✅ and Phase B ✅ both succeed. The remaining question is whether the modes are physically meaningful (eigenvalue clustering, mode localization, convergence).

### Stage 5: Mode analysis, grid convergence & energy scales ────────────────

> **Script: `S5_mode_analysis.py`**

- [x] **S5.1** Energy scale analysis: V_depth/E_kin = **954** → deep trapping, ~197 bound states/band
- [x] **S5.2** Mode localization: **10/20 localized** (IPR > 10×extended), 10/20 intermediate, **0 extended**
- [x] **S5.3** Band composition: **ZERO interband mixing** — each mode weight=1.000 in exactly one band
- [x] **S5.4** Grid convergence: Ns=32→64→128, ratio=3.5 → **2nd-order convergent** but not fully converged
- [x] **S5.5** Free-particle cross-check: q=0 matches exactly, higher q differ due to FD truncation (expected)
- [x] **S5.6** Potential landscape: V_max=0.131381 for target band, modes within ±2.3×10⁻⁴ of V_max

**Key findings:**
- The EA produces **genuinely localized bound states**, not numerical garbage
- Zero interband mixing means the 5-band Hamiltonian acts as **5 independent single-band problems**
  - Root cause: Λ is perfectly diagonal (off-diag = 0), M_inv and v_drift are diagonal-only (Phase 2 limitation), and A=0
  - Off-diagonal mass tensor M_inv_{mn} (Löwdin correction) is never computed
  - With internal gaps ~10⁻⁶, these off-diagonal couplings should be enormous (∝ 1/gap)
- Ground state is from **Band 3** (not the target Band 2) — bands compete for lowest energy
- Grid convergence is 2nd-order (consistent with 4th-order FD + block-averaging), but eigenvalues shift ~3×10⁻⁴ between Ns=64 and Ns=128

**Stage 5 verdict:** The C4-symmetrized single-band EA works correctly for each band individually. The multi-band structure is currently inert (no interband coupling). The deep trapping (V/E_kin ≈ 954) causes eigenvalue clustering — ~20 modes packed within 4.6×10⁻⁴.

### Stage 6: Off-diagonal coupling + twist angle sweep ──────────

Script: `S6_eta_sweep_coupling.py`

**Part A — Off-diagonal coupling survey:**
- [x] **S6.1** Off-diagonal magnitudes: **Only A_berry has off-diagonal data** (||off-diag||/||diag|| = 0.84)
  - Λ off-diagonal: exactly 0 (diagonal by definition)
  - M_inv off-diagonal: exactly 0 (Phase 2 only computes diagonal)
  - v_drift off-diagonal: exactly 0 (Phase 2 only computes diagonal)
  - Φ_BH off-diagonal: ratio 0.093 (C4-sym introduced tiny values, but Born-Huang is a placeholder)
- [x] **S6.2** Phase 3 kinetic operator analyzed: **IGNORES off-diagonal A_berry** (extracts only n,n)
  - Code at `phase3_mpb_v3.py` L529-540: `M_inv_flat[indices] = M_inv_reshaped[:, n, n, :, :]`
  - Off-diagonal Berry connection EXISTS in data but is never used in H assembly
- [x] **S6.3** Built corrected kinetic op with FULL covariant derivative (diamagnetic + paramagnetic terms)
  - H_corr nnz = 4,996,978 (vs 2,047,954 standard) — 2.4× denser from off-diagonal coupling
  - Mean interband mixing: **0.656** (vs 0.000 standard) → massive interband mixing when A off-diag is used!
  - Eigenvalue shifts: Δε(corr−std) = −2.8×10⁻⁵ to −9.4×10⁻⁵ (systematic downshift)
  - Standard + A=0 have ZERO mixing; corrected has 50-72% mixing per mode
  - **⇒ Off-diagonal A_berry is the DOMINANT interband coupling mechanism**

**Part B — Twist angle (η) sweep (Ns=64, A=0):**
- [x] **S6.4** Energy scale predictions confirmed:
  - θ=1.1°: V/E_kin=946, ~196 bound states → deeply trapped, clustered
  - θ=5°: V/E_kin=46, ~9 bound states → moderate trapping
  - θ=10°: V/E_kin=12, ~2 bound states → near optimal
  - θ=15°: V/E_kin=5, ~1 bound state → kinetic-dominated
- [x] **S6.5** Eigensolve across 7 θ-values:
  - θ=1.1°: 12/12 localized, spread=2.87×10⁻⁴ (deeply trapped)
  - θ=5°: 9/12 localized, spread=3.64×10⁻³ (well-resolved levels)
  - θ=10°: 6/12 localized, spread=1.03×10⁻² (approaching continuum)
  - θ=15°: 1/12 localized, spread=2.40×10⁻² (too few bound states)
  - **Zero interband mixing at ALL θ** when using standard kinetic op (since A off-diag is ignored)
- [x] **S6.6** Optimal θ for V/E_kin ≈ 10 requires θ ≈ **10.7°** (η ≈ 0.187) — EA validity questionable

**Stage 6 verdict:** Two critical findings:
1. **Off-diagonal A_berry enables massive interband coupling** (mean mixing 0 → 66%) but Phase 3 kinetic operator drops it. The corrected covariant derivative with all A_{mn} terms produces physically meaningful interband hybridization.
2. **This candidate is pathological**: V/E_kin ≈ 954 at θ=1.1°, and reaching V/E_kin ~ 10 requires θ ≈ 11° where the EA (which assumes η ≪ 1) may break down. A candidate with larger |M_eff| or smaller V_depth is needed for physically meaningful small-θ results.

### Stage 7: Production off-diagonal A validation ✅ ──────────

- [x] **S7.1** Legacy path unchanged: 20 modes, 0% mixing, band-diagonal ✓
- [x] **S7.2** Full Berry coupling: 66.6% mean mixing (matches S6), K nnz 2M → 5M
- [x] **S7.3** Hermiticity: ||K - K†||/||K|| = 0.0 (perfectly Hermitian) ✓
- [x] **S7.4** A=0 consistency: K_full(A=0) ≡ K_legacy(A=0), Δ = 0.0 ✓
- [x] **S7.5** Diagonal-A cross-check: eigenvalues Δε = 7×10⁻⁵ (paramagnetic-only shift)
- [x] **S7.6** C4 commutator: ||[H,C4]||/||H|| = 9.7×10⁻¹⁷ ✓
- [x] **S7.7** Mode gallery: genuine multi-band profiles, weight spread across B0–B4
- [x] **S7.8** η-sweep: mixing decreases 64%→35% as θ increases 1.1°→10° (physics-correct)

**Stage 7 verdict:** The production code fix is correct, self-consistent, and preserves all symmetries. Off-diagonal Berry connection produces the dominant interband coupling mechanism. The implementation is backward-compatible (default `include_offdiag_A=False`).

### Stage 8: Analytical cross-checks (FUTURE) ─────────────────────

- [ ] **S8.1** Single-band Mathieu: set N=1, A=0, M_inv=constant → compare with known solutions
- [ ] **S8.2** Free-particle limit: Λ=constant, A=0 → eigenvalues = ω₀ + η²/(2m)·|q|². Verify.
- [ ] **S8.3** Sum rule: ∑ₙ ωₙ = Tr(Λ)

---

## 4. Key Hypotheses — Updated

| # | Hypothesis | Test | Status |
|---|-----------|------|--------|
| ~~H1~~ | ~~Band crossings invalidate subspace~~ | ~~S2.7~~ | **RETRACTED: wrong C4 test. Subspace is 99.4% equivariant.** |
| H2 | Gauge noise in A causes C4 breaking | S4.3 vs S4.4 | **CONFIRMED: A adds 56% more C4 breaking (2.9%→4.6%).** |
| H3 | M_inv divergences cause C4 breaking | S4.3: [H,C4] = 2.9% | **CONFIRMED: Bands 3,4 have 31-39% C4 tensor error.** |
| H4 | Anti-crossing smoothness defects corrupt FD derivatives | S3e + S4 | **CONFIRMED: M_inv clamped at 14.3% of points; bands 3,4 worst.** |
| H5 | FD kinetic operator has wrong prefactor | S4 prefactor check | **RESOLVED: η² = 1/L² verified to machine precision.** |
| H6 | Off-diagonal M_inv needed for accuracy | S6 Part A | **PARTIALLY: A_berry off-diag (not M_inv) drives 66% interband mixing; M_inv off-diag still zero** |
| H7 | (32,32) fixed-point defect propagates to eigenmodes | S4 | **SUBSUMED: the global C4 problem is much larger than any point defect.** |
| **H8** | **BFS gauge produces non-C4 phases → all ∂u/∂R quantities broken** | **S4 input C4 test** | **CONFIRMED: A_berry ALL bands ~200% C4 error, Φ_BH ~200%.** |
| **H9** | **v_drift ≠ 0 at Γ for generic R** | **S4.2** | **CONFIRMED: |v|_max = 0.118. Expected physics (not at extremum at generic R).** |
| **H10** | **Phase 3 kinetic op ignores off-diagonal A → no interband coupling** | **S6.2-S6.3** | **CONFIRMED: standard H has 0.000 mixing; corrected covariant derivative has 0.656 mixing** |
| **H11** | **V/E_kin pathological for this candidate at small θ** | **S6.4-S6.6** | **CONFIRMED: V/E_kin=954 at θ=1.1°; need θ≈11° for V/E_kin≈10 (EA breaks down)** |

---

## 5. Recommended Path Forward

Based on S3 (subspace OK) and S4 (input C4 broken), the pipeline's physics is sound but the **computed operator data** breaks C4 symmetry. The three culprits and their severity:

| Source | C4 error | Mechanism |
|--------|----------|----------|
| **A_berry** | ~200% all bands | BFS gauge is non-C4; `∂u/∂R` inherits random phases |
| **M_inv** bands 3,4 | 31–39% | Anti-crossing defects + FD noise at degeneracies |
| **Φ_BH** | ~200% | Same gauge issue as A_berry (both from `∂u/∂R`) |
| **Λ** | 0.2% | ✓ OK |
| **M_inv** bands 0–2 | 0.6% | ✓ OK |

### Phase A: C4-symmetrize all operator data ✅ DONE

Completed in S4b. All C4 errors → machine precision, `[H,C4]` → 9.58×10⁻¹⁷, 8/8 mode groups pass C4.

### Phase B: Test with A=0 ✅ DONE

Completed in S4b. A=0 gives 7/7 mode groups passing C4. B singlet ground state. Practically identical eigenvalues to Λ+K only (drift/BH have negligible effect on eigenvalues, but drift is needed for clean degenerate-subspace resolution).

### Phase C: Mode analysis & eigenvalue clustering ✅ DONE

Completed in S5. Modes ARE localized (10/20 pass IPR threshold). Zero interband mixing → each band operates independently. Grid convergence is 2nd order. V_depth/E_kin = 954 → too deep, but bound states are physical.

### Phase D: Validate analytical limits (partially done in S5)

Free-particle limit tested in S5: q=0 state matches exactly, higher-q modes differ due to FD truncation (expected, not a bug). Remaining:
1. **Single-band Mathieu**: N=1, A=0, constant M → compare with Mathieu equation solutions
2. ~~**Off-diagonal coupling investigation**: compute M_inv_{mn} (m≠n) and Λ_{mn} → measure interband mixing strength~~ → **DONE in S6** (A_berry off-diag provides 66% mixing)

### Phase E: Address pathological V/E_kin ratio ✅ CHARACTERIZED in S6

Confirmed in S6: V_depth/E_kin = 954 at θ=1.1°. Sweep shows:
- θ=5° gives V/E_kin=46 (9 bound states, well-resolved)
- θ=10° gives V/E_kin=12 (optimal but EA questionable)
- θ=15° gives V/E_kin=5 (too few bound states, η=0.26 not small)
- **Conclusion**: this candidate is pathological — need larger |M_eff| or smaller V_depth

### Phase F: Fix Phase 3 kinetic operator ✅ DONE

S6 discovered that Phase 3's kinetic operator ignores off-diagonal A_berry. Implemented and validated in S7.

**Code changes** (`phasesV3/phase3_mpb_v3.py`):
- Added `_build_band_block_diagonal(vals, N_s, N_bands, N_total)` helper for vectorized sparse block-diagonal construction
- Modified `build_multiband_kinetic_operator(... include_offdiag_A=False)`:
  - When `True`: adds diamagnetic A² term (`Σ_{p,ij} M_{ij,pp} A_{mp,i} A_{pn,j}`) via two vectorized `einsum` calls, plus paramagnetic term (`−i · M_{ij,mm} · A_{mn,i} · ∂_j`) via block-diagonal sparse matrices. Hermitization captures adjoint terms.
  - When `False`: legacy behavior (diagonal A² only, no paramagnetic)
- Modified `assemble_multiband_hamiltonian(... include_offdiag_A=False)` to propagate flag

**S7 validation results**:
| Check | Result | Status |
|-------|--------|--------|
| Hermiticity | `\|\|K - K†\|\|/\|\|K\|\|` = 0.0 | ✓ Machine precision |
| A=0 consistency | `K_full(A=0) ≡ K_legacy(A=0)`, Δ = 0.0 | ✓ Exact match |
| Diagonal-A eigenvalues | Δε = 7.1×10⁻⁵ (6 modes) | ✓ Close (para-shift only) |
| C4 commutator | `\|\|[H,C4]\|\|/\|\|H\|\|` = 9.7×10⁻¹⁷ | ✓ Machine precision |
| Interband mixing | 0% (legacy) → **66.6%** (full A) | ✓ Matches S6 |
| Eigenvalue spectrum | 20 modes, spread reduced vs legacy | ✓ Physical |

**η-sweep with full coupling** (Ns=64):
| θ (°) | V/E_kin | mix (full) | loc modes |
|--------|---------|------------|-----------|
| 1.1 | 946 | 63.7% | 3/12 |
| 3.0 | 127 | 66.1% | 4/12 |
| 5.0 | 46 | 51.3% | 0/12 |
| 7.0 | 23 | 44.3% | 3/12 |
| 10.0 | 12 | 35.0% | 5/12 |

Mixing DECREASES with θ: at large angles kinetic energy dominates, making A_berry relatively less important. All modes show genuine multi-band character (weight spread across B0–B4, no single band >45%).

**Remaining** (lower priority): Compute off-diagonal M_inv in Phase 2 (currently M_{m≠n}=0 → Laplacian still band-diagonal). Less critical since V≫K at small θ, and A_berry dominates coupling.

### Phase G: New candidate selection (FUTURE)

Select a candidate with better V/E_kin ratio at small θ:
- Need: larger |M_eff| (more dispersive band) or smaller V_depth (shallower potential)
- Options: different band, different lattice, different ε contrast

---

## 6. File Structure

```
corrections_findings/
├── GUIDE.md                              ← this file
├── FINDINGS_S1_S2.md                     ← Stage 1–2 findings (⚠ C4 closure test was wrong)
├── FINDINGS_S3.md                        ← Stage 3 findings (equivariance discovery + smoothness)
├── S1_single_R_audit.py                  ← S1: point-wise checks (OK — non-C4 findings still valid)
├── S1b_C4_fixed.py                       ← S1: C4 at fixed points (OK — invariance correct at fixed pts)
├── S2_field_symmetry.py                  ← S2: field scans (⚠ closure scan used invariance)
├── S2b_subspace_sizes.py                 ← S2: subspace sizes (⚠ used invariance — all would pass eqv)
├── S3_overlap_reorder.py                 ← S3: Hungarian reordering (irrelevant — wrong test)
├── S3b_parallel_transport.py             ← S3: BFS transport (harmful — broke equivariance)
├── S3c_symmetrized.py                    ← S3: symmetrization (failed — wrong basis)
├── S3d_equivariance.py                   ← S3: ✓ DEFINITIVE equivariance vs invariance test
├── S3e_smoothness.py                     ← S3: ✓ DEFINITIVE smoothness + combined quality
├── S3_definitive_diagnostic.py           ← S3: ✓ Clean summary with correct tests only
├── S4_hamiltonian_termbyterm.py          ← S4: ✓ Term-by-term H build, C4 commutator, eigenmode C4
├── S4b_c4_symmetrize.py                 ← S4b: ✓ C4-symmetrization of all Phase 2 data
├── S5_mode_analysis.py                  ← S5: ✓ Mode localization, band composition, grid convergence
├── S6_eta_sweep_coupling.py             ← S6: ✓ Off-diagonal A coupling + twist angle η-sweep
├── S7_offdiag_A_validation.py           ← S7: ✓ Production off-diagonal A validation (10 tests)
├── QUESTIONS_AND_ANSWERS.md              ← Physics Q&A: candidate choice, EA criteria, k-drift
└── plots/                                ← output figures (see plot inventory below)
```

### Plot Inventory

| Plot | Status | Description |
|------|--------|-------------|
| S1_band_gaps.png | ✓ Valid | Band gap structure (doesn't use C4 test) |
| S1b_C4_fields_delta00.png | ✓ Valid | C4 at fixed point δ=(0,0) — invariance = equivariance here |
| S1b_C4_fields_delta05.png | ✓ Valid | C4 at fixed point δ=(0.5,0.5) — shows genuine defect |
| S1b_epsilon.png | ✓ Valid | Epsilon field comparison |
| S2_Minv_singleband.png | ✓ Valid | M_inv divergence (doesn't use C4 test) |
| S2_band_gaps_map.png | ✓ Valid | Band gap map (scalar data) |
| S2_berry_diagonal.png | ✓ Valid | Berry connection |A| (doesn't use C4 test) |
| S2_born_huang.png | ✓ Valid | Born-Huang potential |
| S2_gram_quality.png | ✓ Valid | Gram matrix condition number |
| S2_potential_diagonal.png | ✓ Valid | Λ_nn potential landscapes |
| S2_potential_offdiag.png | ✓ Valid | Off-diagonal Λ_mn |
| S2_subspace_closure.png | ⚠ **MISLEADING** | Used invariance test — 98% "failure" is artifact |
| S2b_subspace_sizes.png | ⚠ **MISLEADING** | Used invariance — all subspaces would pass equivariance |
| S3_overlap_reorder.png | ⚠ Obsolete | Tried to fix non-problem |
| S3_band_mapping.png | ⚠ Obsolete | Band mapping at (32,32) is valid, but "failure" framing wrong |
| S3b_parallel_transport.png | ⚠ Obsolete | Transport quality real, but equivariance test wrong |
| S3c_symmetrized.png | ⚠ Obsolete | All used invariance |
| **S3d_equivariance.png** | **✓ KEY RESULT** | Equivariance vs invariance — proves subspace correct |
| **S3e_subspace_quality.png** | **✓ KEY RESULT** | Smoothness + equivariance combined |
| S3_definitive.png | ✓ Summary | Combined overview of correct diagnostics |
| **S4_hamiltonian_termbyterm.png** | **✓ KEY RESULT** | [H,C4] commutator + eigenspectra per config |
| **S4_mode_gallery.png** | **✓ KEY RESULT** | Mode profiles (4 lowest) for all 5 configurations |
| **S4b_c4_symmetrization.png** | **✓ KEY RESULT** | Before/after C4 error + eigenvalue comparison |
| **S4b_mode_gallery.png** | **✓ KEY RESULT** | Mode profiles for original vs C4-sym configs |
| **S5_mode_analysis.png** | **✓ KEY RESULT** | Energy scales, IPR, localization, grid convergence, band weights |
| **S5_mode_gallery.png** | **✓ KEY RESULT** | Real-space |F|² for 12 lowest modes (C4-sym, A=0) |
| **S6_eta_sweep_coupling.png** | **✓ KEY RESULT** | V/E_kin vs θ, eigenvalue spectra, interband mixing comparison, off-diag coupling magnitudes |
| **S7_offdiag_validation.png** | **✓ KEY RESULT** | Mode gallery: 6 lowest modes × 3 configs (A=0, legacy, full A) — shows genuine multi-band profiles |
| **S7_offdiag_summary.png** | **✓ KEY RESULT** | 6-panel summary: eigenvalue spectra, band weights, mixing vs θ, localization, V/E_kin scaling |

---

## 7. Notation Reference

| Symbol | Code variable | Phase | Shape | Description |
|--------|--------------|-------|-------|-------------|
| Λ_mn(R) | `Lambda` | 2 | (128,128,5,5) | Diagonal potential matrix |
| v^(i)_mn | `v_drift` | 2 | (128,128,5,5,2) | Drift velocity (diagonal only) |
| M⁻¹_ij,nn | `M_inv` | 2 | (128,128,5,5,2,2) | Inverse mass tensor (diagonal only) |
| A_j,mn | `A_berry` | 2 | (128,128,5,5,2) | Berry connection (complex) |
| Φ_BH,mn | `Phi_BH` | 2 | (128,128,5,5) | Born-Huang potential |
| u_n(r;R) | `bloch_fields` | 1 | (64,64,18,32,32,3) | Bloch E-fields (idx 0-17 = all_bands) |
| ε(r;R) | `epsilon` | 1 | (64,64,32,32) | Local permittivity |
| ω_n(R) | `omega` | 1 | (128,128,5) | Band frequencies |
| F_n(R) | `eigenvectors` | 3 | (81920,100) | Envelope spinors |

**Index convention for bloch_fields**: axis 2 indexes into `all_bands = [0..17]`. The subspace bands [5,6,7,8,9] are at indices 5-9.

**Coordinate systems**:
- s ∈ [0,1)²: fractional registry coordinate
- R = s · L_moire: physical position (units of a)
- r ∈ Ω: fast coordinate within unit cell (units of a)
- dR = L_moire / Ns: physical grid spacing

**C4 transformation**:
- Registry: (ix, iy) → ((Nr-iy) % Nr, ix)
- Unit cell: (jx, jy) → ((Nx-jy) % Nx, jx)
- Vector field: (Ex, Ey, Ez) → (-Ey, Ex, Ez)
- **Equivariance**: C4·subspace(R) = subspace(C4·R) — correct test for generic R
- **Invariance**: C4·subspace(R) = subspace(R) — only valid at C4 fixed points (R = C4·R)
