# Stage 4 Findings: Term-by-term Hamiltonian Diagnostic

**Date:** 2026-02-08
**Script:** `S4_hamiltonian_termbyterm.py`

---

## 1. Purpose

Identify which term in the envelope Hamiltonian breaks C4 symmetry. This is done by:
1. Checking C4 symmetry of each **input operator** (Λ, M_inv, A_berry, v_drift, Φ_BH)
2. Computing `||[H, C4]||_F / ||H||_F` — the **commutator test** — for each partial Hamiltonian
3. Solving for eigenmodes and testing their C4 symmetry

---

## 2. Key Results

### 2.1 Prefactor Verification ✓

The kinetic prefactor is correct:
- η² = 3.6858 × 10⁻⁴
- 1/L² = 3.6858 × 10⁻⁴
- Match: **1.47 × 10⁻¹⁶ relative error** (machine precision)

This definitively rules out hypothesis H5 (wrong kinetic prefactor). The V3 code's `0.5/(2π)²` with physical grid spacing `dR` correctly reproduces the theoretical `0.5·η²·M_inv`.

### 2.2 C4 Symmetry of Input Operators — THE SMOKING GUN

| Operator | C4 transformation law | C4 error | Assessment |
|----------|----------------------|----------|------------|
| **Λ(R)** | Scalar: Λ(C4·R) = Λ(R) | 0.19% | ✓ Excellent |
| **M_inv bands 0–2** | Tensor: M(C4·R) = C4·M(R)·C4ᵀ | 0.6% | ✓ Acceptable |
| **M_inv bands 3,4** | Tensor: M(C4·R) = C4·M(R)·C4ᵀ | **31–39%** | ✗ BAD |
| **A_berry ALL bands** | Vector: A(C4·R) = C4·A(R) | **191–198%** | ✗✗ CATASTROPHIC |
| **Φ_BH** | Scalar: Φ(C4·R) = Φ(R) | **200%** | ✗✗ CATASTROPHIC |
| **v_drift** | Vector: v(C4·R) = C4·v(R) | — | max |v|=0.118 (not zero) |

**Interpretation:**

- **A_berry ~200% C4 error for ALL bands** means the Berry connection is completely uncorrelated under C4. The error of ~2 (vs signal of ~1) means `A(C4·R)` and `C4·A(R)` are essentially unrelated. This is NOT an anti-crossing effect — it affects ALL five bands equally.

- **Root cause:** The BFS Abelian gauge fix aligns phases via a flood-fill from the grid center. This makes the gauge smooth (adjacent points have consistent phases), but it does NOT enforce C4 equivariance. The gauge choice at point R has no relationship to the gauge choice at C4·R. Since `A = i⟨u|∂u⟩` depends directly on the gauge, the Berry connection inherits the gauge's lack of C4 symmetry.

- **Φ_BH suffers identically** because it's also computed from `∂u/∂R`: Φ_mn = Σ_j ⟨∂_j u_m | (1-P) | ∂_j u_n⟩. Same gauge-dependent derivatives, same catastrophic C4 error.

- **M_inv is frequency-based** (`∂²ω/∂k²`), so it doesn't depend on the gauge of Bloch functions. Its C4 error comes from FD noise at anti-crossing degeneracies (bands 3,4 are closest to band crossings).

- **v_drift ≠ 0** is expected physics: at generic registry points R, the local crystal ε(r; R) does NOT have C4 symmetry (only the moiré pattern overall has C4). So group velocity ∂ω/∂k at Γ can be nonzero for individual R.

### 2.3 [H, C4] Commutator — Progressive Degradation

| Configuration | ||[H,C4]||/||H|| | Eigenmode C4 | Interpretation |
|:--|:--:|:--:|:--|
| **S4.1: Λ only** | **1.0 × 10⁻⁴** | (singular) | Λ is essentially C4-perfect |
| **S4.2: Λ + drift** | **6.7 × 10⁻³** | 0/7 pass | Drift adds 0.7% C4 breaking |
| **S4.3: Λ + K (no A)** | **2.9 × 10⁻²** | 0/20 pass | M_inv adds **2.3%** additional breaking |
| **S4.4: Λ + K (with A)** | **4.6 × 10⁻²** | 0/20 pass | A (|A|² term) adds **1.6%** more |
| **S4.5: Full H** | **4.6 × 10⁻²** | 0/20 pass | BH + drift correction negligible |

**Key observations:**
- Each term **adds** C4 breaking progressively
- The dominant contributions: M_inv (2.3%) + A_berry's |A|² (1.6%) + drift (0.7%)
- The |A|² diamagnetic term alone adds significant C4 breaking — even without the paramagnetic cross-terms that are missing from the code

### 2.4 Eigenvalue Spectrum

All configurations produce eigenvalues clustered near V_max = 0.1314 for band 2 (target):

| Config | ε₀ | ε₉ | Spread (ε₉-ε₀) |
|--------|-----|-----|-----------------|
| Λ+drift | 0.131314 | 0.131359 | 4.4 × 10⁻⁵ |
| Λ+K(0) | 0.131171 | 0.131414 | 2.4 × 10⁻⁴ |
| Λ+K(A) | 0.131253 | 0.131421 | 1.7 × 10⁻⁴ |
| Full | 0.131258 | 0.131419 | 1.6 × 10⁻⁴ |

- All eigenvalues are positive and close to V_max → modes sit at the **top** of the potential well → **hole-like (inverted) band**
- The kinetic term K provides spatial coupling and spreads eigenvalues (without K, they're nearly degenerate)
- Total eigenvalue bandwidth ~10⁻⁴ is comparable to the kinetic scale `0.5·M_inv·η²`

### 2.5 Eigenmode C4 Quality

**S4.1 (Λ only):** Eigsh failed (singular). This is expected: H=Λ is purely diagonal (no spatial coupling), so the shift-invert matrix `(H-σI)` has exact zeros wherever Λ_nn(R)=σ. The "eigenvalue spectrum" is simply the set of all Λ_nn(R) values at all grid points. There are no spatially extended modes.

**S4.2 (Λ+drift):** All modes come in pairs with C4 overlap ≈ 0. Unitarity error ≈ √2 means C4 maps each mode completely outside its degenerate subspace. This tells us the drift operator mixes different spatial regions in a non-C4 way.

**S4.3 (K, no A):** Some modes (0, 2, 3, 18) have decent C4 overlap (~0.98), suggesting they avoid the defective regions of M_inv. But most modes (1, 4–17, 19) have terrible C4. The pattern: modes near V_max (the continuum edge) are poorly defined and have random C4, while modes pulled further below V_max by kinetic energy have better C4.

**S4.4 (K, with A):** Adding A makes everything worse. No mode above 0.55 C4 overlap. The |A|² term acts as a spatially-varying potential that is 200% C4-broken, contaminating ALL modes.

**S4.5 (Full):** Essentially same as S4.4 with Born-Huang correction on top.

---

## 3. What's Good

1. **Prefactors are correct** — η² = 1/L² verified to 10⁻¹⁶. No unit conversion bugs.
2. **Λ is C4-symmetric** to 0.02% — the potential V(R) = ω(R) - ω_ref is fine.
3. **M_inv is C4-acceptable for bands 0–2** (~0.6% error).
4. **The [H,C4] commutator test is definitive** — it directly measures Hamiltonian symmetry without ambiguity from eigenmode degeneracy.
5. **The problem hierarchy is clear**: A_berry > M_inv(bands 3,4) > v_drift > Λ.
6. **The code structure supports term-by-term testing** via `include_drift`, `include_kinetic`, `include_born_huang` flags.

## 4. What's Bad

1. **A_berry is 200% C4-broken for ALL bands** — the BFS gauge does not produce C4-equivariant phases. This is the single worst problem.
2. **Φ_BH is equally broken** — same root cause (gauge-dependent ∂u/∂R).
3. **M_inv bands 3,4 are 31–39% C4-broken** — anti-crossing defects at these bands produce noisy FD curvatures.
4. **14.3% of M_inv points are clamped** at |eig|=20 — the regularization prevents divergence but doesn't fix C4.
5. **No eigenmode passes C4 at the 0.99 level** in ANY configuration.
6. **Paramagnetic cross-terms missing from code** — the kinetic operator uses `-∂² + |A|²` but omits `-i(A·∂ + ∂·A)`. Even if A were C4-correct, the covariant derivative is incomplete.

## 5. What's Confusing

1. **v_drift is not zero at Γ** — expected for generic R (local crystal has no C4), but the original derivation assumed k₀ is at a band extremum where v_g=0. If the band is NOT at an extremum at Γ, the drift term is O(η) and could be larger than the kinetic term O(η²). This needs investigation: is the target band actually at a minimum/maximum/saddle at Γ?

2. **S4.1 singular** — H=Λ is trivially C4 but we couldn't verify eigenmodes. The "modes" would just be delta functions at each grid point. This is correct behavior but uninformative for mode-level C4 testing.

3. **Eigenvalue clustering** — all modes within ~10⁻⁴ of V_max suggests we're probing the **continuum edge**, not well-separated bound states. For a hole band, the ground state should be at V_max minus kinetic corrections. But the near-degeneracy of all 20 modes makes C4 classification unreliable (small C4 breaking can reorder modes).

4. **S4.2 shows C4 overlap ≈0 with unitarity error ≈√2** — the drift operator is spatially coupling different points via `-i v·∂`, but v(R) is not C4-equivariant (0.7% [H,C4]). This small C4 breaking produces modes that are completely non-C4 because the modes are nearly degenerate → any small perturbation mixes them.

---

## 6. Root Cause Chain

```
Crystal C4 symmetry ─┬─→ ω(R, k₀) C4-equivariant   ─→ Λ C4 ✓  (directly from frequencies)
                     ├─→ M_inv C4-equivariant        ─→ M_inv C4 ~ ✓ for bands 0-2
                     │                                    (from ∂²ω/∂k², frequency-based)
                     │                                    M_inv C4 ✗ for bands 3,4
                     │                                    (anti-crossing noise in FD curvature)
                     │
                     └─→ u(r; R) C4-equivariant UP TO GAUGE
                              │
                              ├─→ BFS gauge fix: smooth but NOT C4
                              │
                              ├─→ A = i⟨u|∂u⟩ inherits gauge randomness → 200% C4 error
                              │
                              └─→ Φ_BH = ⟨∂u|(1-P)|∂u⟩ inherits gauge randomness → 200% C4 error
```

**The fundamental issue:** The BFS gauge produces spatially smooth phases (good for avoiding FD derivative disasters) but does NOT produce C4-equivariant phases. A true C4-equivariant gauge would require: if u(r; R) is the gauge choice at R, then u(C4·r; C4·R) must be the gauge choice at C4·R (up to a global phase). The BFS flood-fill from center has no mechanism to enforce this.

---

## 7. Recommended Next Steps

### Immediate: C4-symmetrize all operator data

For any quantity Q(R) with known C4 transformation law T:

$$Q^{\text{sym}}(R) = \frac{1}{4}\sum_{n=0}^{3} T^n\left[Q(C_4^{-n} R)\right]$$

- **Scalars** (Λ, Φ_BH): average values at R, C4·R, C4²·R, C4³·R
- **Vectors** (A, v_drift): average with C4 rotation of vector components
- **2-tensors** (M_inv): average with C4 rotation of tensor components

This is guaranteed to give [H,C4] = 0 and requires NO re-running of Phase 1/2.

### Then: Investigate whether C4-symmetric H produces physical modes

1. Do C4-symmetric eigenmodes appear?
2. Are they localized (bound states) or extended (scattering)?
3. Do eigenvalues converge with system size?

### If modes are still unphysical after C4 fix:

1. Investigate sigma selection — are we targeting the right energy?
2. Test single-band (N=1) with constant mass → Mathieu equation
3. Check if the potential well is deep enough for bound states

---

## 8. Data Reference

| Quantity | Value |
|----------|-------|
| Grid | 128×128×5 (N_total = 81,920) |
| η | 0.019198 |
| L_moire | 52.088 a |
| dR | 0.407 a |
| ω_ref | 0.7913 |
| V_max (band 2) | +0.1314 |
| V_min (band 2) | -0.1519 |
| Potential depth | 0.283 |
| Kinetic scale (typical) | ~10⁻⁴ (for lowest Fourier mode) |
| M_inv clamped points | 11,755/81,920 (14.3%) |
| Kinetic prefactor | 0.5/(2π)² = 1.267 × 10⁻² |
