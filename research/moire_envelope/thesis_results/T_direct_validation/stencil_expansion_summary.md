# Stencil Expansion and K-Dependent Miniband Pipeline

## Overview

The envelope approximation (EA) pipeline uses a three-phase architecture to compute moiré minibands from monolayer photonic crystal data. Phase 1 samples a k-stencil around a chosen carrier momentum $k_0$ at each registry point $\delta$, extracting scalar band parameters $\omega_n$, $\mathbf{v}_g$, $M^{-1}$, and Bloch functions $u_n(\mathbf{r}; k_0, \delta)$. Phase 2 processes the Bloch fields into Berry connection $\mathbf{A}(\delta)$ and Born-Huang potential $\Phi_\mathrm{BH}(\delta)$. Phase 3 assembles the envelope Hamiltonian $H(\mathbf{K})$ at moiré wavevectors $\mathbf{K}$ and diagonalises it to obtain miniband eigenvalues $\omega(\mathbf{K})$.

The stencil expansion upgrades Phase 1 to support large-angle moiré supercells (up to $\sim 10°$) and enables Phase 3 to compute K-dependent miniband dispersion rather than only the $\Gamma$-point eigenvalues.

## What the pipeline does

### Phase 1: Registry sweep and k-stencil

For each registry shift $\delta$ on a coarse grid (typically $32 \times 32$), MPB solves the monolayer eigenproblem at a grid of k-points centred on $k_0$:

$$\omega_n(k_0 + \delta k_i, k_0 + \delta k_j; \delta) \quad \text{on a } 7 \times 7 \text{ stencil}$$

with spacing $dk = 0.06$ (in $2\pi/a$ units). This yields 49 eigenvalues per registry point per band. Bloch functions $u_n(\mathbf{r}; k_0, \delta)$ are extracted only at the centre k-point.

From the stencil, 6th-order finite-difference coefficients give:

- **First derivatives** (group velocity): $\mathbf{v}_g = \nabla_k \omega$ using $[-1, 9, -45, 0, 45, -9, 1]/60$
- **Second derivatives** (inverse mass): $M^{-1}_{ij} = \partial^2\omega / \partial k_i \partial k_j$ using $[2, -27, 270, -490, 270, -27, 2]/180$

These are stored in Phase 1 output alongside the raw stencil eigenvalues for later polynomial fitting.

### Phase 2: Gauge fixing, Berry connection, Born-Huang

Operates on the Bloch fields from Phase 1. Run once at $k_0$:

1. **Abelian gauge fix** — BFS-based phase alignment ensuring smooth $u_n(\mathbf{r}; \delta)$ across the registry grid, with Zak phase distribution across periodic boundaries.
2. **SVQB B-orthonormalisation** — $\varepsilon$-weighted Gram–Schmidt ensuring $\langle u_m | \varepsilon | u_n \rangle = \delta_{mn}$ to machine precision.
3. **Berry connection** — $A_{mn,j}(\delta) = i \langle u_m | \varepsilon | \partial_{\delta_j} u_n \rangle$, computed via 4th-order FD on the registry grid.
4. **Born-Huang potential** — $\Phi_{mn}(\delta) = \sum_j \langle \partial_{\delta_j} u_m | (1 - P) | \partial_{\delta_j} u_n \rangle_\varepsilon$, projecting out the subspace.

Output: $\Lambda(\delta)$, $\mathbf{v}_\mathrm{drift}(\delta)$, $M^{-1}(\delta)$, $\mathbf{A}(\delta)$, $\Phi_\mathrm{BH}(\delta)$.

### Phase 3: Hamiltonian assembly and K-sweep

The envelope equation

$$\left[\Lambda(\mathbf{R}) + \eta\, \hat{H}^{(1)}(\mathbf{R}, \mathcal{D}) + \eta^2\, \hat{H}^{(2)}(\mathbf{R}, \mathcal{D}) \right] F(\mathbf{R}) = \Delta\lambda\, F(\mathbf{R})$$

is discretised on the moiré grid ($128 \times 128$). For miniband dispersion, the envelope carries a Bloch factor $F_n(\mathbf{R}) = e^{i\mathbf{K}\cdot\mathbf{R}} \tilde{F}_n(\mathbf{R})$, which shifts the covariant derivative $\mathcal{D}_j \to iK_j + \mathcal{D}_j$. The scalar band parameters are re-evaluated at $k_0 + \mathbf{K}$ from the stencil polynomial fit.

## Why we expanded the stencil

The moiré BZ excursion scale is $|\mathbf{K}_\mathrm{max}| \approx 2\sin(\theta/2) \cdot |k_0 - k_\mathrm{sym}|$. For a square lattice at the M-point:

| Twist angle | $\|\mathbf{K}_\mathrm{max}\|$ ($2\pi/a$) | Old stencil radius ($dk=0.01$, $5\times5$) | New stencil radius ($dk=0.06$, $7\times7$) |
|:-----------:|:-------:|:-----:|:------:|
| 1°          | 0.009   | 0.02 ✓ | 0.18 ✓ |
| 5°          | 0.044   | 0.02 ✗ | 0.18 ✓ |
| 10°         | 0.087   | 0.02 ✗ | 0.18 ✓ |

The old 5×5/$dk=0.01$ stencil had a support radius of only $0.02$ — too small for any angle beyond $\sim 2°$. Beyond the stencil, polynomial extrapolation fails catastrophically (67% error at 10°). The new 7×7/$dk=0.06$ stencil covers the full excursion with a safety margin of $2\times$.

### What we gained

1. **Accuracy at large angles.** Validation shows the new stencil stays below 2% interpolation error out to 10°, versus 67% for the old one.
2. **K-dependent miniband dispersion.** With reliable $\omega(\mathbf{K})$, $\mathbf{v}_g(\mathbf{K})$, $M^{-1}(\mathbf{K})$, Phase 3 can sweep the moiré BZ rather than only solving at $\Gamma$.
3. **6th-order FD coefficients.** The wider stencil permits 6th-order accurate derivatives, giving $\sim 1700\times$ improvement in FD truncation error over 4th-order at the same $dk$.

## Theoretical justification: why K enters only through scalar parameters

The two-scale ansatz is

$$\Psi(\mathbf{r}) = e^{i\mathbf{k}_0 \cdot \mathbf{r}} \sum_n F_n(\mathbf{R})\, u_n(\mathbf{r};\, \mathbf{k}_0,\, \delta(\mathbf{R}))$$

The Bloch functions $u_n$ are always evaluated at the fixed carrier momentum $k_0$, **never** at $k_0 + \mathbf{K}$. The moiré wavevector $\mathbf{K}$ is a slow-scale quantity that enters only through the envelope $F_n(\mathbf{R})$. Consequently:

| Quantity | Depends on $\delta$? | Depends on moiré $\mathbf{K}$? |
|----------|:---:|:---:|
| $u_n(\mathbf{r}; k_0, \delta)$ | Yes | **No** |
| $\Lambda_n(\delta)$, $\mathbf{v}_g(\delta)$, $M^{-1}(\delta)$ | Yes | **No** (but can be improved by stencil polynomial) |
| $\mathbf{A}_\mathrm{berry}(\delta)$, $\Phi_\mathrm{BH}(\delta)$ | Yes | **No** |
| $F_n(\mathbf{R})$ | n/a | **Yes** |
| $H(\mathbf{K})$ | n/a | **Yes** (via $\nabla \to i\mathbf{K} + \nabla$) |

This means Phase 2 (Berry/Born-Huang) runs **once** at $k_0$. It does not need to be re-run for each $\mathbf{K}$. The K-sweep in Phase 3 is pure linear algebra.

### Connection between the two approaches

At quadratic order, using K-interpolated stencil data (Approach B) and the standard $\nabla \to \mathbf{K} + \nabla$ substitution (Approach A) are algebraically identical:

$$\underbrace{\Lambda_0 + \mathbf{v}_{g,0}\cdot\mathbf{K} + \tfrac{1}{2}\mathbf{K} \cdot M_0^{-1} \cdot \mathbf{K}}_{\Lambda_\mathbf{K}} + \underbrace{(\mathbf{v}_{g,0} + M_0^{-1}\cdot\mathbf{K})}_{\mathbf{v}_{g,\mathbf{K}}} \cdot (-i\nabla) + (-i\nabla)\cdot M_0^{-1} \cdot (-i\nabla)$$

At higher polynomial order (quartic), the stencil approach captures beyond-quadratic curvature of the band surface — a systematic improvement over the standard EA.

## How to use the pipeline

### Running Phase 1 with the expanded stencil

Phase 1 accepts `dk` and `fd_order` parameters. The production defaults are:

```
dk = 0.06        # stencil spacing in 2π/a
fd_order = 6     # 6th-order FD → 7×7 stencil grid
```

The stencil eigenvalues are saved in the Phase 1 HDF5 output as `stencil_omega` with shape `(n_registry, n_registry, N_bands, 7, 7)`.

### Computing miniband dispersion (Phase 3)

`solve_moire_band_structure()` takes a path of K-points, pre-fits 2D polynomials on the stencil once, then for each K:

1. Evaluates $\omega_n(\mathbf{K})$, $\mathbf{v}_{g,n}(\mathbf{K})$, $M^{-1}_n(\mathbf{K})$ from the polynomial fit
2. Uses the frozen $\mathbf{A}_\mathrm{berry}$ and $\Phi_\mathrm{BH}$ from Phase 2
3. Assembles $H(\mathbf{K})$ and diagonalises

No MPB re-solves. No re-running Phase 2. Each K-point costs only a matrix assembly + sparse eigsolve.

### When to apply

- **Small angles ($\theta \lesssim 2°$)**: The old 5×5/$dk=0.01$ stencil suffices at $\Gamma$ only. Use the new stencil if miniband dispersion is needed.
- **Moderate angles ($2°$–$10°$)**: The expanded stencil is required. Without it, polynomial extrapolation beyond the old stencil produces meaningless results.
- **K=0 only (any angle)**: Use the raw FD centre values from the stencil, not the polynomial fit, to avoid the polynomial's fitting bias ($\sim 0.8\%$ at $\Gamma$ for quadratic).

## Key files

| File | Role |
|------|------|
| `phasesV3/phase1_mpb_v3.py` | Phase 1: registry sweep, k-stencil extraction, Bloch field extraction |
| `phasesV3/phase2_mpb_v3.py` | Phase 2: gauge fix, Berry connection, Born-Huang potential |
| `phasesV3/phase3_mpb_v3.py` | Phase 3: Hamiltonian assembly, eigsolve, `solve_moire_band_structure()` |
| `phasesV3/stencil_interpolation.py` | Polynomial fitting and K-evaluation of stencil data |
| `phasesV3/bloch_fields.py` | Born-Huang computation from stored Bloch fields |
| `T_direct_validation/validate_stencil.py` | 3-panel validation: coverage, accuracy, angle comparison |
| `T_direct_validation/square_supercell_3way.py` | Three-way MPB vs FDFD vs EA comparison driver |
| `T_direct_validation/square_monolayer_comparison.py` | Monolayer MPB vs FDFD agreement (0.025%) |

## Validation results

- **Monolayer**: MPB and FDFD agree to 0.025% on band 3 at M ($\omega = 0.6846$).
- **Polynomial fit**: Quadratic RMS residual $3.4 \times 10^{-3}$, quartic $2.9 \times 10^{-4}$ on real band surface. Machine precision on synthetic test surfaces.
- **Old vs new stencil at 10°**: Old stencil error 67%, new stencil error <2%.
- **FD coefficient improvement**: 6th-order coefficients give $\sim 1700\times$ lower truncation error than 4th-order at the same grid spacing.
