# EA vs FDFD Validation — Square (11,1), θ = 10.39°

## Setup

| Parameter | Value |
|-----------|-------|
| Lattice | Square, a = 1.0, r/a = 0.2, ε_rod = 11.56, TM |
| Supercell | (m,n) = (11,1), N_cells = 122, L = √122 ≈ 11.045 |
| Target | Band 3 at M-point, ω₀ = 0.68457 |
| FDFD | 128 px/cell → 1414², DOF = 2.0M, CHOLMOD shift-invert |
| EA registry | 32×32, MPB res = 32, 7×7 stencil (dk = 0.06) |
| EA grid | 128×128 moiré grid, single-band (band 3) |

## Results

**Best configuration:** trace-clamped M⁻¹ with max |Tr| = 2.0

| Metric | Value |
|--------|-------|
| RMS error | 4.85 × 10⁻³ (11.5% of FDFD bw) |
| Max error | 10.3 × 10⁻³ |
| EA bandwidth | 57.1 × 10⁻³ |
| FDFD bandwidth | 42.3 × 10⁻³ |

Error profile: EA spans slightly wider than FDFD. Central modes (indices 12–20) match within ±3 × 10⁻³. Tails diverge up to 10 × 10⁻³.

## Regularization sweep

The raw inverse effective mass Tr(M⁻¹) ranges from 2.4 (monolayer, δ = 0) to 14.5 (bilayer band crossings). This 6× inflation makes the unregularized kinetic term dominate, broadening the EA spectrum to 3× the FDFD bandwidth.

| Trace clamp | Bandwidth × 10³ | RMS × 10³ | Notes |
|-------------|-----------------|-----------|-------|
| V-only | 9.7 | 10.72 | Too narrow — no kinetic |
| raw | 123.6 | 24.32 | 3× too wide — M⁻¹ diverges |
| 5.0 | 106.3 | 20.06 | |
| 3.0 | 79.9 | 11.83 | |
| **2.0** | **57.1** | **4.85** | **optimal** |
| 1.0 | 27.5 | 4.89 | |
| 0.5 | 13.0 | 8.92 | |

Optimal clamp ≈ 2.0, slightly below monolayer Tr(M⁻¹) = 2.43. The bilayer band-crossing inflation is entirely spurious at this twist angle.

## Clamping method matters

Two implementations were tested:

1. **Trace clamping** (manual): scale entire 2×2 tensor so |Tr(M⁻¹)| ≤ mt.
   Preserves anisotropy ratio. At mt = 2.0 → RMS = 4.85.

2. **Eigenvalue clamping** (built-in `_regularize_M_inv`): decompose into eigenvectors,
   clip each eigenvalue to [−mt, mt]. Allows Tr up to 2 × mt.
   At mt = 2.0 → RMS = 17.45 (trace reaches 4.0).

The trace clamp is strictly tighter and physically better motivated: the relevant scale is the scalar effective mass at the band extremum, not individual directional curvatures.

## Multi-band vs single-band

A 4-band (bands 3–6) Hamiltonian was tested. Without inter-band Berry connection and Born-Huang terms, the 4 bands decouple into independent single-band problems. The 50 modes nearest σ = 0 then include spurious contributions from bands 4, 5, 6 (which have V crossing zero at various δ), worsening RMS to 10.975.

**Conclusion:** At θ ≈ 10° with no Bloch-field data, single-band with trace-regularized M⁻¹ is the right approach. Multi-band requires exported Bloch fields and Berry connection computation to couple the bands properly.

## Files

- `ea_final_comparison.py` — comparison script (sweep + plots)
- `square_3way/fig_ea_vs_fdfd_final.png` — 4-panel figure
- `square_3way/ea_singleband_mt2_results.npz` — saved eigenvalues
- `square_3way/ea_multiband_registry.npz` — 32×32 × 10-band registry
