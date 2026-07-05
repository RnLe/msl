# TE X-Point Sigma 0.1 Analysis

## Setup

- Polarization: TE
- Direct solve: FDFD at X with total supercell Bloch vector `q = (π, 0)`
- Sigma: `sigma_omega = 0.1`
- Mode count: `50` for all three angles
- Convergence comparison: `px = 4` versus `px = 8`, with an extra `px = 12` check at `1.005°`
- Extra overlay: term-audit / EA spectra at `8.17°` and `3.01°`
- Carrier analysis basis: unfold `px = 8` direct eigenvectors into the monolayer square-lattice Brillouin zone

## px4 vs px8 spectral comparison

| Angle | px4 range | px8 range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth px4 | Bandwidth px8 | |Δ bandwidth| |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8.17° | [0.025392, 0.198626] | [0.026955, 0.211920] | 0.008990 | 0.009655 | 6.207 | 7.166 | 0.173233 | 0.184965 | 0.011732 |

## Audit overlay comparison — 8.17°

| Comparison | Range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth | |Δ bandwidth vs px8| |
|---|---|---:|---:|---:|---:|---:|---:|
| EA audit vs px=8 | [0.355160, 0.393801] | 0.231222 | 0.234545 | 227.459 | 1220.923 | 0.038641 | 0.146324 |
| 3.01° | [0.085537, 0.112562] | [0.084487, 0.115702] | 0.001415 | 0.001748 | 1.395 | 3.333 | 0.027025 | 0.031215 | 0.004190 |

## Audit overlay comparison — 3.01°

| Comparison | Range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth | |Δ bandwidth vs px8| |
|---|---|---:|---:|---:|---:|---:|---:|
| EA audit vs px=8 | [0.369723, 0.380180] | 0.274419 | 0.274481 | 275.267 | 337.910 | 0.010457 | 0.020758 |
| 1.005° | [0.098345, 0.101574] | [0.098117, 0.101692] | 0.000164 | 0.000196 | 0.165 | 0.351 | 0.003229 | 0.003574 | 0.000345 |

## Additional 1.005° resolution checks

| Angle | Comparison | Range | Mean abs diff | RMSE | Mean abs rel % | Max abs rel % | Bandwidth | |Δ bandwidth vs px8| |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1.005° | px=12 vs px=8 | [0.098222, 0.101640] | 0.000149 | 0.000191 | 0.149 | 0.440 | 0.003418 | 0.000156 |

## Interpretation

- `px = 8` is treated as the more resolved direct reference.
- If the px4-vs-px8 differences remain small across the 50-mode window, then `4 px/cell` is still adequate at `sigma = 0.1` for this X-point target.
- The carrier analysis is based on the unfolded Fourier content of the direct supercell eigenvectors, not on the EA carrier assumption.

## Carrier analysis — 8.17°

- Dominant-label counts across 50 modes: Γ=0, X=21, M=0, other=29
- Mean top-component carrier-family weight: Γ=0.001, X=0.396, M=0.026, other=0.577

| Representative modes | Frequency | Dominant carrier | Folded k in monolayer BZ (fractional) | Distance to labeled carrier |
|---|---:|---|---|---:|
| mode 1 | 0.026955 | X | (-0.495, -0.071) | 0.071 |
| mode 2 | 0.026960 | X | (0.500, 0.000) | 0.000 |
| mode 3 | 0.058193 | X | (0.434, -0.076) | 0.101 |
| mode 10 | 0.095701 | X | (-0.439, 0.147) | 0.159 |
| mode 20 | 0.134299 | X | (-0.490, -0.142) | 0.142 |
| mode 30 | 0.164510 | other | (-0.287, 0.015) | 0.214 |
| mode 40 | 0.186363 | other | (0.485, 0.213) | 0.214 |
| mode 50 | 0.211920 | other | (0.221, -0.091) | 0.239 |

## Carrier analysis — 3.01°

- Dominant-label counts across 50 modes: Γ=0, X=50, M=0, other=0
- Mean top-component carrier-family weight: Γ=0.000, X=0.955, M=0.003, other=0.042

| Representative modes | Frequency | Dominant carrier | Folded k in monolayer BZ (fractional) | Distance to labeled carrier |
|---|---:|---|---|---:|
| mode 1 | 0.084487 | X | (0.396, -0.029) | 0.108 |
| mode 2 | 0.084488 | X | (-0.396, 0.029) | 0.108 |
| mode 3 | 0.086020 | X | (-0.394, -0.024) | 0.108 |
| mode 10 | 0.090950 | X | (-0.424, 0.107) | 0.132 |
| mode 20 | 0.097406 | X | (0.451, -0.133) | 0.142 |
| mode 30 | 0.104671 | X | (0.368, 0.023) | 0.134 |
| mode 40 | 0.109121 | X | (-0.496, -0.158) | 0.158 |
| mode 50 | 0.115702 | X | (0.452, -0.159) | 0.166 |

## Carrier analysis — 1.005°

- Dominant-label counts across 50 modes: Γ=0, X=50, M=0, other=0
- Mean top-component carrier-family weight: Γ=0.000, X=0.998, M=0.000, other=0.002

| Representative modes | Frequency | Dominant carrier | Folded k in monolayer BZ (fractional) | Distance to labeled carrier |
|---|---:|---|---|---:|
| mode 1 | 0.098117 | X | (-0.420, -0.105) | 0.132 |
| mode 2 | 0.098118 | X | (-0.420, -0.105) | 0.132 |
| mode 3 | 0.098283 | X | (0.377, 0.034) | 0.128 |
| mode 10 | 0.098797 | X | (0.387, -0.071) | 0.134 |
| mode 20 | 0.099397 | X | (0.466, -0.132) | 0.136 |
| mode 30 | 0.100151 | X | (0.440, -0.123) | 0.137 |
| mode 40 | 0.100715 | X | (0.396, -0.089) | 0.137 |
| mode 50 | 0.101692 | X | (0.378, -0.062) | 0.137 |
