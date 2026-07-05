# Finding F01: No Discrete Bound States in the Single-Band Moiré Envelope

**Date:** 2026-02-06  
**System:** Square lattice, a=1.0, r/a=0.29, ε=7.9, k₀=(0.5,0) [X-point]  
**θ = 1.5°, η = 0.026, band 0 (hole-like, M⁻¹ < 0)**

---

## Summary

The single-band envelope Hamiltonian $\hat{H} = \Lambda(\mathbf{R}) + \frac{1}{2} M_{ij}^{-1}(\mathbf{R}) \, D_i D_j + \ldots$ on the periodic moiré grid has a **continuous spectrum**. There are **no discrete bound states** below $V_{\max}$.

This means that "binding energy" $E_{\text{bind}} = V_{\max} - \lambda_0$ is **not a physical observable** — it depends on the number of modes $k$ requested from `eigsh`, and grows indefinitely as $k$ increases.

## Evidence

| k (modes) | n_bound | $\lambda_0$ | $E_{\text{bind}}$ | $E_{\text{bind}} / \eta^2$ |
|-----------|---------|-------------|-------------------|---------------------------|
| 10        | 5       | 0.09045     | 0.00093           | 1.4                       |
| 20        | 9       | 0.08945     | 0.00193           | 2.8                       |
| 50        | 24      | 0.08591     | 0.00547           | 8.0                       |
| 100       | 49      | 0.07988     | 0.01151           | 16.8                      |
| 200       | 100     | 0.06868     | 0.02270           | 33.1                      |
| 500*      | 264     | 0.03464     | 0.05674           | 82.8                      |
| 1000*     | 560     | −0.01831    | 0.10969           | 160.0                     |

\* from earlier session logs

### Key observations

1. **$\lambda_0$ drifts without bound** as $k$ increases — it drops from 0.090 to −0.018 with no sign of convergence.
2. **$E_{\text{bind}} / \eta^2$ scales as $k^{1.1}$** — it is a monotone function of the spectral window, not a material property.
3. **The top eigenvalue $\lambda_{\text{top}} = 0.0912$ is k-independent** — this IS physical: it's the highest moiré miniband state for this band.
4. **No spectral gap at $V_{\max}$** — eigenvalues pass smoothly through $V_{\max} = 0.0914$ with no visible gap or edge, confirming a continuous band.

## Physical Interpretation

For a **periodic** potential $\Lambda(\mathbf{R})$ on a moiré superlattice, the spectrum is band-like (Bloch's theorem at the moiré scale), not atomic-like. The Hamiltonian $H = K + V$ with periodic $V$ on a torus yields moiré **minibands**, not discrete levels.

- **$V_{\max}$ is not a band edge.** The kinetic operator $K$ (with $M^{-1} < 0$ for hole bands) creates moiré Bloch waves that have eigenvalues both above and below $V_{\max}$.
- **The eigenvalue density near $V_{\max}$** reflects the moiré density of states, not bound-state counting.
- **"Binding energy"** as commonly defined ($E_{\text{bind}} = V_{\max} - \lambda_0$) measures the total bandwidth of the miniband below $V_{\max}$, which grows with the spectral window sampled.

## Correct Observables

Instead of "binding energy", the physically meaningful quantities for the moiré envelope theory are:

1. **Moiré miniband bandwidth** $W = \lambda_{\max} - \lambda_{\min}$ (for a fixed, well-resolved set of miniband states)
2. **Flatness ratio** $\Delta_{\text{gap}} / W$ (gap to bandwidth ratio — indicates flat-band physics)  
3. **Band mixing weight** $1 - \max_n |c_n|^2$ (measures inter-band hybridization from multi-band coupling)
4. **Moiré DOS structure** (Van Hove singularities, gap structure)

## Implications for the η-Sweep

- Any scaling law derived from $E_{\text{bind}}$ is meaningless.
- The η-sweep should instead track **miniband structure**: how the bandwidth, gaps, and flatness ratio evolve with twist angle.
- The N-band convergence test ($N_{\text{sub}} = 1$ vs 3) should compare **miniband bandwidths**, not individual eigenvalues.

## Plot

![F01_no_discrete_bound_states](F01_no_discrete_bound_states.png)

- **Panel A:** Ground state eigenvalue drifts without convergence as k increases.
- **Panel B:** $E_{\text{bind}}/\eta^2$ grows as $k^{1.1}$ — purely an artifact.
- **Panel C:** No spectral gap at $V_{\max}$ — eigenvalues are dense on both sides.
- **Panel D:** Eigenvalue staircase grows with k; the top (shallowest) state is stable.
