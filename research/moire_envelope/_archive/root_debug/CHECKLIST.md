# Moiré Envelope Method Inspection Checklist

This document tracks the verification of the "Moiré Envelope" pipeline phases.
Objective: Ensure physical correctness, numerical stability, and symmetry preservation.

## Phase 1: Local Band Abstraction (MPB)
**Goal:** Extract smooth local band parameters ($\omega$, $v_g$, $M^{-1}$) and Bloch fields on the moiré grid.

- [x] **Data Ranges & Units**
    - [x] `omega` ($\omega$): Should be dimensionless ($a/\lambda$). Typical range [0.1, 1.0].
        - *Observation:* Range [0.60, 0.88]. OK.
    - [x] `vg` (Group Velocity): Should be small (drift). Zero at extrema.
        - *Observation:* Max ~0.02. OK.
    - [x] `M_inv` (Inverse Mass): Curvature of bands. Can be large near crossings.
        - *Observation:* Range [-117, +55]. **Mixed signs**. Indicates saddle points or band mixing. User expects minimum (positive), but data shows strong negative components.
    - [x] `V` (Potential): $\omega(R) - \omega_{ref}$. Should be smooth.
        - *Observation:* Smoothness metric ~4e-3. Looks continuous.
- [x] **Symmetry (C4 / C6)**
    - [x] Potential $V(R)$ should reflect lattice symmetry.
        - *Observation:* Error ~7e-4. Good.
- [x] **Bloch Fields (The Noise Source)**
    - [x] Check phase coherence between $R$ and $R+\delta$.
        - *Observation:* **FAIL**. Max phase jump ~3 rad. StdDev ~1.7 rad. This confirms Phase 1 raw output has random gauge. **Must be fixed in Phase 2.**

## Phase 2: Gauge Fixing & Operator Construction
**Goal:** Fix the random gauge and compute smooth geometric potentials ($\mathbf{A}$, $\Phi_{BH}$).

- [x] **Gauge Fixing Performance**
    - [x] Apply `parallel_transport_gauge`.
        - *Observation:* Implemented in Phase 2. $A \approx 0$ confirms gauge condition.
    - [x] Re-check phase smoothness after fixing.
        - *Observation:* Code assumes smoothness. Inspection shows "BH Max Adjacent Diff" ~2e-3, which is reasonably smooth.
    - [ ] **Symmetry Check**: Does the Axis 0 -> Axis 1 transport break $C_4$ symmetry significantly?
        - *Observation:* BH Symmetry Error ~2e-3. Acceptable.
- [x] **Born-Huang Potential ($\Phi_{BH}$)**
    - [x] Magnitude: Should be small perturbation (order $\eta^2$).
        - *Observation:* **Fixed.** Was large due to unnormalized fields. Now range is $10^{-4}$, consistent with kinetic term magnitude.
    - [x] Smoothness: Should not look like white noise.
        - *Observation:* Validated.
    - [x] Symmetry: Should reflect lattice symmetry + gauge choice artifacts.
        - *Observation:* Final modes show degenerate pairs (e.g., Mode 7/8), confirming symmetry.
- [x] **Berry Connection ($\mathbf{A}$)**
    - [x] Real/Imaginary parts: Diagonal $\mathbf{A}_{nn}$ should be real (if Hermitian).
    - [x] Magnitude.
        - *Observation:* **Fixed.** Was ~565, now ~0.8 (physical units).
        - *Root Cause:* Bloch fields were not normalized ($\langle u|u \rangle \approx 207$). Added normalization step in Phase 2.

## Phase 3: Envelope Solver
**Goal:** Solve the effective Hamiltonian correctly.

- [x] **Scaling Factors**
    - [x] **Kinetic Term**: $\frac{1}{2} D_{phys} M^{-1} D_{phys}$.
        - *Correction:* Removed extra $\eta^2$ factor.
    - [x] **Drift Term**: $v_g \cdot D_{phys}$.
        - *Correction:* Replaced $\eta$ with $1/2\pi$ for unit conversion.
    - [x] **Born-Huang**: $\Phi_{BH}$.
        - *Correction:* Removed extra $\eta^2$ factor.
        - *Correction 2:* Fixed normalization of source fields.
- [x] **Negative Mass / Hole Band Handling**
    - [x] **Problem**: Solver finding "sprinkled dots" ($E \ll V_{min}$) due to negative mass continuum.
    - [x] **Fix**: Analyze $M^{-1}$ trace. If negative, target $V_{max}$ (hole states). If positive, target $V_{min}$ (electron states).
    - [x] **Verification**: Ran Phase 3. Auto-detected hole band ($M^{-1} \approx -9.3$). Set $\sigma \approx 0.147$. Found modes at $\omega \approx 0.88$ ($E \approx 0.147$). No more -35 eigenvalues.
- [x] **Eigenvalues**
    - [x] Should be bound states within the potential well (between $\min(V)$ and $\max(V)$).
        - *Observation:* Finding modes near $E \approx 0.147$, which is effectively the "top" of the hole well.
    - [x] Should not be localized to single pixels (IPR analysis).
        - *Observation:* Spread ~0.4 (unit cell fraction). Seems localized but physical.
    - [x] **Mode Purity**: Modes should be dominated by single bands if coupling is weak.
        - *Observation:* **Confirmed.** Modes are now >99% single-band character. (Previously high mixing was artifact of huge BH term).
    - [x] **Degeneracy**: C4 symmetry should produce doublets.
        - *Observation:* **Confirmed.** Found degenerate pairs (e.g. Mode 7/8).

## Action Items
1. [x] Inspect Phase 1 (Completed).
2. [x] Re-run Phase 2 with gauge fixing (Verification Complete).
3. [x] Inspect Phase 2 outputs (Symmetry, Smoothness) (Completed).
4. [x] Run Phase 3 with new dynamic mass detection (Completed).
5. [x] Verify Mode 0 is a proper cavity mode (not noise) (Completed).

