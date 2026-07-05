# Cross-Candidate Comparison Report

Generated: 2026-03-07 20:54:46

## Overview

This report compares three photonic moiré crystal candidates:
- **C3** (`square_M_b3`): Square lattice, M-point, band 3 (5-band subspace)
- **C1** (`hex_M_b1`): Hexagonal lattice, M-point, band 1 (4-band subspace)
- **C_hc** (`honeycomb_K_b1`): Honeycomb (triangular + 2-atom basis), K-point Dirac cone (2-band subspace)

## Phase 2 Parameters

| Parameter | C3 (square_M_b3) | C1 (hex_M_b1) | C_hc (honeycomb_K_b1) |
|-----------|-------------------|----------------|-----------------------|
| N_subspace | 5 | 4 | 2 |
| ω range | [0.5287, 1.1841] | [0.1838, 0.3534] | [0.2269, 0.2570] |
| Λ₀₀ range | [-0.0415, -0.0398] | [-0.0177, -0.0070] | [-0.0070, 0.0088] |
| |Λ₀₁| max | 0.0000 | 0.0000 | 0.0000 |
| |A₀₀| max | 1.1494 | 5.0800 | 1.2265 |
| |A₀₁| max | 1.2809 | 2.9874 | 1.2389 |
| Tr(M⁻¹₀₀) | 65.9683 | -34.6662 | -10.3110 |
| Tr(M⁻¹₁₁) | 24.4056 | 35.7697 | 5.3633 |

## Phase 3 Base Run (θ ≈ 1.1°)

| Metric | C3 | C1 | C_hc |
|--------|----|----|------|
| BW₅₀ | 0.010201 | 0.004755 | 0.003261 |
| Gap E₁-E₀ | 0.000512 | 0.000018 | 0.000033 |

## η-Sweep Results (Full Berry Connection)

### C3: sq M b3

| θ (deg) | η | BW₅₀ | Max mixing | Gap E₁-E₀ |
|---------|---|-------|------------|-----------|
| 0.5 | 0.008727 | 0.002499 | 0.660 | 0.000047 |
| 0.8 | 0.013963 | 0.005326 | 0.718 | 0.000079 |
| 1.0 | 0.017453 | 0.007998 | 0.722 | 0.000115 |
| 1.5 | 0.026179 | 0.019023 | 0.707 | 0.000915 |
| 2.0 | 0.034905 | 0.031779 | 0.702 | 0.000213 |
| 3.0 | 0.052354 | 0.067616 | 0.704 | 0.000899 |
| 5.0 | 0.087239 | 0.193710 | 0.704 | 0.006462 |
| 8.0 | 0.139513 | 0.506598 | 0.694 | 0.006700 |

Power-law fit: BW ~ η^2.037 (a = 27.9833)

### C1: hex M b1

| θ (deg) | η | BW₅₀ | Max mixing | Gap E₁-E₀ |
|---------|---|-------|------------|-----------|
| 0.5 | 0.008727 | 0.001313 | 0.707 | 0.000018 |
| 0.8 | 0.013963 | 0.002826 | 0.636 | 0.000102 |
| 1.0 | 0.017453 | 0.003712 | 0.652 | 0.000059 |
| 1.5 | 0.026179 | 0.006893 | 0.655 | 0.000050 |
| 2.0 | 0.034905 | 0.011958 | 0.654 | 0.000287 |
| 3.0 | 0.052354 | 0.026664 | 0.628 | 0.000414 |
| 5.0 | 0.087239 | 0.071579 | 0.674 | 0.002484 |
| 8.0 | 0.139513 | 0.197180 | 0.646 | 0.001183 |

Power-law fit: BW ~ η^2.092 (a = 12.1151)

### C_hc: hc K Dirac

| θ (deg) | η | BW₅₀ | Max mixing | Gap E₁-E₀ |
|---------|---|-------|------------|-----------|
| 0.5 | 0.008727 | 0.000656 | 0.016 | 0.000027 |
| 0.8 | 0.013963 | 0.001475 | 0.103 | 0.000007 |
| 1.0 | 0.017453 | 0.002290 | 0.074 | 0.000081 |
| 1.5 | 0.026179 | 0.004586 | 0.175 | 0.000155 |
| 2.0 | 0.034905 | 0.007996 | 0.197 | 0.000085 |
| 3.0 | 0.052354 | 0.018447 | 0.367 | 0.000703 |
| 5.0 | 0.087239 | 0.046918 | 0.457 | 0.000210 |
| 8.0 | 0.139513 | 0.116953 | 0.469 | 0.003009 |

Power-law fit: BW ~ η^1.922 (a = 5.1514)

## Key Findings

### 1. Honeycomb Dirac Cone Candidate
- The honeycomb candidate has **zero inter-band moiré potential coupling** (|Λ₀₁| = 0)
- Inter-band coupling comes **entirely through the off-diagonal Berry connection** (|A₀₁| ≈ 1.24)
- This is the photonic analogue of **twisted bilayer graphene**: Dirac cone + Berry phase
- 2-band subspace (vs 4-5 for other candidates) = maximally clean Dirac physics

### 2. Effective Mass Asymmetry
- C_hc: Band 0 = HOLE (Tr(M⁻¹) = -10.31), Band 1 = ELECTRON (Tr(M⁻¹) = +5.36)
- This electron-hole asymmetry will produce asymmetric miniband spectra
- The large |M⁻¹| magnitudes indicate strong dispersion near K-point

### 3. Berry Connection Dominance
- For C_hc, the Berry connection provides the **only** inter-band coupling mechanism
- This makes it a pure gauge-field-mediated phenomenon
- Validates the importance of the full non-Abelian Berry connection treatment
