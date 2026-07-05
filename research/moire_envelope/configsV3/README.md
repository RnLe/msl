# V3 Multi-Band Pipeline Configuration

This directory contains configuration files for the **V3 multi-band envelope approximation pipeline**.

## Theory Reference

See `/docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md` for the complete derivation.

## Key V3 Features

### Multi-Band Subspace
- Tracks N bands simultaneously (N = 2×n_neighbor_bands + 1)
- Default: 5 bands (n_neighbor_bands=2)
- Eigenfunction is a spinor: F(s) ∈ ℂ^N at each grid point

### Berry Connection (Non-Abelian Gauge Field)
- A_j,mn(s) = i⟨u_m|∂_j u_n⟩
- Computed via finite-difference overlaps between registry samples
- Parallel transport gauge applied in Phase 2 for gauge fixing

### Born-Huang Potential
- Φ_mn(s) = Σ_j ⟨∂_j u_m|(1-P)|∂_j u_n⟩
- Captures out-of-subspace mixing effects
- Requires extra bands beyond tracked subspace (n_extra_bands=4)

### Drift Term (Group Velocity)
- v^(i)_mn(s) = ⟨u_m|V_i|u_n⟩ where V_i = ∂L_0/∂k_i
- Retained even at band extrema for off-diagonal coupling

## Configuration Files

| File | Description |
|------|-------------|
| `phase0_blaze.yaml` | Candidate search with multi-band parameters |
| `phase1_blaze.yaml` | Local Bloch problems for N bands |
| `phase2_blaze.yaml` | Berry connection, Born-Huang, mass tensors |
| `phase3_blaze.yaml` | Multi-band envelope eigensolver |

## Key Parameters

### `n_neighbor_bands` (all phases)
Number of bands above/below the target to include. With n=2:
- Total bands in subspace: 5
- Band indices: [target-2, target-1, target, target+1, target+2]

### `n_extra_bands` (phases 0-2)
Additional bands for Born-Huang calculation. With n=4:
- Total bands computed: N_bands + 2×n_extra_bands = 13
- Extra bands provide out-of-subspace projector (1-P)

### `include_born_huang` (phases 2-3)
Whether to compute and use Born-Huang potential.

### `include_drift_term` (phases 2-3)
Whether to include group velocity coupling term.

### `include_berry_connection` (phase 3)
Whether to use gauge-covariant derivatives.

## Running the Pipeline

```bash
# Phase 0: Candidate search
python blaze_phasesV3/phase0_library_v3.py configsV3/phase0_blaze.yaml

# Phase 1: Local Bloch (use 'auto' for latest Phase 0 run)
python blaze_phasesV3/phase1_blaze_v3.py auto configsV3/phase1_blaze.yaml

# Phase 2: Data preparation with Berry connection
python blaze_phasesV3/phase2_blaze_v3.py auto configsV3/phase2_blaze.yaml

# Phase 3: Multi-band eigensolver
python blaze_phasesV3/phase3_blaze_v3.py auto configsV3/phase3_blaze.yaml
```

## Envelope Hamiltonian

The full multi-band envelope equation:

$$
\Big[\Lambda(\mathbf{s}) - \lambda_{\mathrm{ref}}I + \eta \hat{H}^{(1)}(\mathbf{s}, \mathcal{D}) + \eta^2 \hat{H}^{(2)}(\mathbf{s}, \mathcal{D}) + U_{\mathrm{BH}}(\mathbf{s})\Big] F(\mathbf{s}) = \Delta\lambda F(\mathbf{s})
$$

where:
- $\mathcal{D}_j = \partial_{s_j} - i A_j(\mathbf{s})$ is the gauge-covariant derivative
- $\hat{H}^{(1)}F = \sum_{i,n} v^{(i)}_{mn} (-i \mathcal{D}_i F)_n$ is the drift term
- $\hat{H}^{(2)}F = \frac{1}{2}\sum_{i,j,n} M^{-1}_{ij,mn} (-i\mathcal{D}_i)(-i\mathcal{D}_j) F_n$ is the kinetic term
