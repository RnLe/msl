# Finding F01: No Discrete Bound States in Single-Band Moiré Envelope

**Date:** 2026-02-06  
**System:** Square lattice, a=1.0, r/a=0.29, ε=7.9, k₀=(0.5,0) [X-point]  
**Parameters:** θ = 1.5°, η = 0.026, Ns = 128×128

## Summary

The single-band envelope Hamiltonian $H_0 = \frac{1}{2} M^{-1}_{ij}(\mathbf{R})\,\partial_i\partial_j + \Lambda_0(\mathbf{R})$ on a **periodic** moiré grid does **not** have discrete bound states. The spectrum is purely continuous (Bloch-band-like).

## Evidence

We solved the eigenvalue problem using `scipy.sparse.linalg.eigsh` with shift-invert targeting σ = V_max = 0.0914, and swept the number of requested modes k:

| k | n_bound | λ₀ (ground) | E_bind = V_max − λ₀ | E_bind/η² |
|---|---------|-------------|---------------------|-----------|
| 10 | 5 | 0.0907 | 7.1×10⁻⁴ | 1.0 |
| 20 | 12 | 0.0890 | 2.4×10⁻³ | 3.5 |
| 50 | 25 | 0.0860 | 5.4×10⁻³ | 7.9 |
| 100 | 50 | 0.0803 | 1.1×10⁻² | 16.2 |
| 200 | 102 | 0.0686 | 2.3×10⁻² | 33.3 |
| 500 | 264 | 0.0346 | 5.7×10⁻² | 82.8 |
| 1000 | 560 | −0.0183 | 1.1×10⁻¹ | 160.0 |

**λ₀ drifts without bound as k increases.** The "binding energy" grows as k^1.1 — it is an artifact of how many eigenvalues eigsh returns, not a physical quantity.

Furthermore, examining the eigenvalue spacings near V_max shows **no gap**: the spacing across V_max (3.9×10⁻⁴) is comparable to the mean spacing on either side. V_max sits inside a continuous band, not at a band edge.

## Physical Interpretation

This is actually the **expected** behavior for a periodic Hamiltonian:

1. **Periodicity**: The moiré potential Λ₀(R) is periodic with the moiré lattice. The Hamiltonian therefore has Bloch-band structure — its spectrum consists of continuous bands, not discrete levels.

2. **No confining boundary**: Unlike a quantum dot or defect cavity, the moiré unit cell repeats infinitely. There is no mechanism to produce truly discrete bound states.

3. **Hole-band physics**: Band 0 has negative effective mass (M⁻¹ < 0). This means kinetic energy is negative, and "bound states" would form near V_max (the potential maximum). But V_max is just an interior point of a continuous moiré miniband.

## Consequence for Validation

**What is NOT a valid observable:**
- E_bind = V_max − λ₀ (depends on k, not physical)
- Any scaling law derived from E_bind vs η

**What ARE valid observables:**
- **Moiré miniband bandwidth** — width of the cluster of eigenvalues near V_max
- **Band gap structure** — gaps between distinct moiré minibands
- **Flatness ratio** — gap/bandwidth (measures how "flat" the moiré bands are)
- **N-band coupling strength** — how much inter-band coupling shifts eigenvalues
- **Eigenvalue convergence with Ns** — grid resolution convergence (separate from k-convergence)

## Files

- **Plot:** `F01_no_discrete_bound_states.png`
- **Data:** `F01_data.json`
- **Script:** `make_F01_plot.py`

---

## Update (2026-02-07): Symmetric Gauge + Γ-Point 5-Band Candidate

### What Changed

- **New candidate**: Γ-point, square lattice a=1.0, r/a=0.35, ε=12.0, TE, band 7 (target), 5-band subspace [5–9].
- **Gauge fix**: BFS from center + Zak phase linear ramp replaced the old seed-row+columns Abelian gauge, which broke C4 symmetry (s₂ had 6.4× worse phase variance than s₁).
- **`make_F01_plot.py` fully rewritten**: Now loads data directly from `phase2_multiband_data.h5` instead of hardcoded arrays. Auto-detects `N_subspace`, `target_idx`, `band_type`, `V_max`, `V_min`, `eta`. Builds single-band Hamiltonian dynamically and runs `eigsh` with k = [10, 20, 50, 100, 200, 500].

### Expected Outcome

The core finding — no discrete bound states in single-band envelope — should hold identically for the Γ-point candidate. The spectrum remains a continuous moiré miniband. The updated plot will show the new potential parameters and band type.
