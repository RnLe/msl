# F02: Kinetic Operator Dominance & Anomalous Miniband Scaling

**Finding ID**: F02  
**Date**: 2025-02-07  
**Severity**: Critical — changes interpretation of all Phase 3 eigenvalues  
**Status**: Confirmed  

## Summary

The finite-difference kinetic operator $K = \frac{1}{2(2\pi)^2} M^{-1}_{ij} \partial_{R_i} \partial_{R_j}$ has a **diagonal norm 56× larger** than the moiré potential modulation $\Delta\Lambda$ at $\theta = 0.5°$ (and still 5× at $\theta = 8°$). This is caused by:

1. **Grid Nyquist effect**: With $N_s = 128$ grid points per direction and physical grid spacing $dR = L_{\mathrm{moire}}/N_s$, the kinetic energy at the Nyquist frequency scales as $\sim N_s^2$, far exceeding the potential.

2. **Effective mass hot spots**: $M^{-1}_{\mathrm{trace}}$ varies from 4.2 to **131** across the moiré cell (vs mean 10.9). At near-degeneracy points $s \approx (0, 0.2)$, the Bloch bands approach each other and $M_{\mathrm{eff}} \to 0$, creating extreme kinetic energy concentrations.

## Quantitative Evidence

### Kinetic vs Potential Scales (Band 1, electron, $\theta = 0.5°$)

| Quantity | Value |
|---|---|
| $\Delta\Lambda = V_{\max} - V_{\min}$ | 0.097 |
| Kinetic diagonal max | **5.44** |
| Kinetic diagonal mean | 0.44 |
| $T/V$ ratio (max) | 56 |
| $T_{q_1}$ (first moiré K-point) | $3.5 \times 10^{-6}$ |
| $T_{\mathrm{Nyquist}} / T_{q_1}$ | $\sim N_s^2/4 \approx 4096$ |

### Scaling Test: Artificial Kinetic Reduction

Scaling the kinetic operator by a factor $\alpha$ in $H = \Lambda + \alpha K$:

| Scale $\alpha$ | $\lambda_0 - V_{\min}$ | BW$_{20}$ |
|---|---|---|
| 1.0 | $1.17 \times 10^{-2}$ | $1.55 \times 10^{-2}$ |
| 0.1 | $4.00 \times 10^{-3}$ | $6.47 \times 10^{-3}$ |
| 0.01 | $1.34 \times 10^{-3}$ | $4.10 \times 10^{-3}$ |
| $10^{-3}$ | $2.25 \times 10^{-4}$ | $3.52 \times 10^{-3}$ |
| $10^{-6}$ | $2.41 \times 10^{-7}$ | $3.52 \times 10^{-3}$ |

The eigenvalue gap $\delta = \lambda_0 - V_{\min}$ scales linearly with kinetic strength.  
BW$_{20}$ saturates at 0.0035 (= intrinsic potential spread of 20 nearest grid values).

### Effective Mass Distribution ($M^{-1}_{\mathrm{trace}}$)

| Band | Type | Mean | Median | 95th %ile | Max | Max/Mean |
|---|---|---|---|---|---|---|
| 0 | hole | −9.3 | −3.9 | 44.8 | −130.1 | 14× |
| 1 | electron | 10.9 | 5.2 | 45.2 | 131.0 | 12× |
| 2 | hole | −10.3 | −5.3 | 40.3 | −111.4 | 11× |

## Power-Law Fits: BW$_{20} \sim \eta^\alpha$

| Band | Type | $\alpha$ (BW$_{20}$) | $\alpha$ ($\delta_{\mathrm{shallow}}$) |
|---|---|---|---|
| 0 | hole | 1.67 | 1.69 |
| 1 | electron | 1.01 | 1.27 |
| 2 | hole | 1.29 | 0.82 |

None of these are $\eta^2$ as naively expected. The anomalous exponents arise from:
- **Grid artifacts**: BW$_{20}$ is k-dependent (F01), contaminating the scaling
- **M_inv inhomogeneity**: Extreme values create non-perturbative corrections
- **Drift term**: $O(\eta)$ contribution competes with $O(\eta^2)$ kinetic at small $\eta$

## Physical Interpretation

The moiré Hamiltonian $H = \Lambda(R) + \eta \, v_{\mathrm{drift}} \cdot (-i\nabla_R) + \frac{1}{2(2\pi)^2} M^{-1}(-i\nabla_R)^2$ operates on a periodic grid with physical spacing $dR = L_{\mathrm{moire}}/N_s$. The key finding is that:

1. **The grid resolution $N_s$ introduces spurious high-frequency kinetic modes** that are not present in the physical moiré system. The physical moiré BZ contains only modes with $|K| \le |b_{\mathrm{moire}}|/2 = \pi/L_{\mathrm{moire}}$, but the grid supports modes up to $\pi/dR = \pi N_s / L_{\mathrm{moire}}$, i.e., $N_s$ times more.

2. **The eigsh eigenvalues near the band edge are NOT pure miniband states** — they are contaminated by kinetic coupling to high-K modes. The eigenvalue gap $\lambda_0 - V_{\mathrm{ext}}$ is dominated by this coupling, not by miniband physics.

3. **This is NOT a bug** — it is the correct physics of the discretized envelope equation. The resolution $N_s = 128$ is needed to resolve the smooth potential, but it introduces kinetic energy states far beyond the first moiré BZ. A spectral (Fourier) solver would naturally truncate at a chosen K-cutoff.

## Grid Resolution Convergence Test

Subsampling the $N_s = 128$ grid to lower resolutions (Band 1, $\theta = 0.5°$):

| $N_s$ | $\lambda_0$ | $\delta = \lambda_0 - V_{\min}$ | $T_{\mathrm{Nyquist}}$ |
|---|---|---|---|
| 16 | 0.04189 | 0.01668 | 0.034 |
| 32 | 0.03966 | 0.01445 | 0.114 |
| 64 | 0.03827 | 0.01306 | 0.424 |
| 128 | 0.03697 | 0.01177 | 1.696 |

Fit: $\delta(N_s) = 0.0086 + 0.028 / N_s^{0.44}$

**The eigenvalues converge slowly ($N_s^{-0.44}$) toward a finite $\delta_\infty \approx 0.009$.** This is the physical zero-point energy of the moiré potential well — the ground state wavefunction has finite kinetic energy even in the $N_s \to \infty$ limit. At $N_s = 128$, $\delta$ is within ~30% of the extrapolated value.

The slow convergence exponent (0.44 vs the expected 2 for smooth functions) is caused by the $M^{-1}$ hot spots creating near-singular kinetic energy profiles.

## Implications

- **$\delta_{\mathrm{shallow}}$ IS a physical observable** (zero-point energy of the moiré potential well), but has ~30% grid error at $N_s = 128$
- **BW$_{20}$ is NOT physical** — still dominated by k-dependent sampling artifact (F01)
- **Scaling with $\eta$**: $\delta_\infty(\eta) \approx 0.009$ at $\eta = 0.0087$, need multi-$N_s$ extrapolation at each $\eta$ for clean scaling law
- **For thesis**: The position-dependent effective mass creates slow convergence; a regularized or Fourier-space solver would converge faster

## Reproducibility

- Data: `F02_data.json` (power-law fits, all observables)
- Plot: `F02_miniband_scaling.png` (6-panel figure)
- Analysis: `analyze_scaling.py`, `diagnose_hamiltonian.py`
- Input: `sweep_results_corrected.json` in eta_sweep directory

---

## Update (2026-02-07): Symmetric Gauge + Γ-Point 5-Band Candidate

### What Changed

- **New candidate**: Γ-point, r/a=0.35, ε=12.0, 5-band subspace [5–9], target band 7.
- **`make_F02_plot.py` updated**: Data source changed to `sweep_results_F03_corrected.json`. Auto-detects `N_bands` from data (handles 3 or 5 bands). Extended color/marker/label arrays. All hardcoded `range(3)` replaced with `range(N_bands)`.
- **Gauge fix**: BFS + Zak ramp ensures isotropic gauge smoothness across the registry grid.

### Expected Outcome

The kinetic operator dominance finding will persist: ||K||/||V|| is dictated by M⁻¹ eigenvalues (effective mass tensor) which don't depend on the gauge algorithm. The 5-band subspace may show different coupling structure — in particular, whether the Γ-point bands are all coupled or if some are symmetry-decoupled (unlike the X-point where Band 1 was completely decoupled).
