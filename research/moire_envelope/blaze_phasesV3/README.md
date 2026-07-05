# blaze_phasesV3 - Multi-Band Envelope Approximation Pipeline

This package implements the **V3 multi-band envelope approximation** for moiré photonic crystals, based on the theory in `docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md`.

## Key Features (V3 vs V2)

| Feature | V2 (Single-Band) | V3 (Multi-Band) |
|---------|------------------|-----------------|
| Band tracking | Single target band | N_subspace bands simultaneously |
| Berry connection | Not included | A_mn(s) = i⟨u_m\|∂u_n⟩ |
| Born-Huang potential | Not included | Φ_mn from remote bands |
| Drift term | Dropped | Retained: η × v_mn · ∇F |
| Gauge fixing | Natural gauge | Parallel transport (SVD) |
| Envelope | Scalar F(s) | Spinor F_n(s) ∈ ℂ^N |

## Theory Background

The multi-band envelope equation reads:

$$
\sum_n \left[ \Lambda_{mn}(s) + \eta\, v_{mn}(s) \cdot (-i\nabla) + \eta^2 \left( \frac{1}{2} D_i M^{-1}_{mn,ij} D_j + \Phi^{BH}_{mn}(s) \right) \right] F_n(s) = E\, F_m(s)
$$

where:
- **Λ_mn(s)** = diagonal potentials (band energies relative to reference)
- **v_mn(s)** = drift velocity matrix (group velocity for diagonal)
- **D_i = -i∂_i + A_i** = gauge-covariant derivative with Berry connection
- **M^{-1}_mn** = inverse effective mass tensor
- **Φ^{BH}_mn** = Born-Huang potential from remote bands
- **η = 2sin(θ/2)** = small parameter for twist angle θ

## Pipeline Phases

### Phase 0: Library Search (`phase0_library_v3.py`)
- Searches band library for extrema candidates
- **V3 addition**: Computes `n_subspace_bands`, `subspace_bands`, `all_bands` (including extra for Born-Huang)
- Output: `phase0_candidates.csv` with multi-band metadata

### Phase 1: Local Bloch Problems (`phase1_blaze_v3.py`)
- Runs BLAZE at registry points across moiré unit cell
- **V3 addition**: Extracts N_bands simultaneously, stores raw stencil data
- Output: `phase1_multiband_data.h5`
  - `omega`: (Ns1, Ns2, N_subspace) frequencies
  - `vg`: (Ns1, Ns2, N_subspace, 2) group velocities
  - `M_inv`: (Ns1, Ns2, N_subspace, 2, 2) mass tensors
  - `stencil/`: raw data for Berry connection calculation

### Phase 2: Berry Connection & Born-Huang (`phase2_blaze_v3.py`)
- **NEW in V3**: Computes non-Abelian Berry connection
- **NEW in V3**: Computes Born-Huang potential from extra bands
- Applies parallel transport gauge via SVD
- Output: `phase2_multiband_data.h5`
  - `Lambda`: (Ns1, Ns2, N, N) potential matrix
  - `A_berry`: (Ns1, Ns2, N, N, 2) Berry connection
  - `Phi_BH`: (Ns1, Ns2, N, N) Born-Huang potential
  - `v_drift`: (Ns1, Ns2, N, N, 2) drift velocity matrix
  - `M_inv`: (Ns1, Ns2, N, N, 2, 2) mass tensor matrix

### Phase 3: Envelope Solver (`phase3_blaze_v3.py`)
- Assembles full multi-band Hamiltonian as sparse block matrix
- Solves eigenvalue problem for spinor envelopes
- Output: `phase3_multiband_modes.h5`
  - `eigenvalues`: (n_modes,) cavity mode frequencies
  - `F_spinor`: (n_modes, Ns1, Ns2, N_subspace) envelope functions
  - Mode statistics in `phase3_mode_stats.json`

## Configuration (configsV3/)

### `phase0_blaze.yaml`
```yaml
n_neighbor_bands: 2      # Bands above/below target (N_subspace = 2*2+1 = 5)
n_extra_bands: 4         # Additional bands for Born-Huang
```

### `phase1_blaze.yaml`
```yaml
phase1_Ns1: 128          # Grid resolution
phase1_Ns2: 128
blaze_registry_samples: 64
blaze_fd_order: 4        # Finite difference order
```

### `phase2_blaze.yaml`
```yaml
include_born_huang: true
include_drift_term: true
use_parallel_transport_gauge: true
born_huang_coupling: 1.0
```

### `phase3_blaze.yaml`
```yaml
n_modes: 20              # Number of cavity modes to compute
include_drift_term: true
include_kinetic_term: true
include_born_huang: true
fd_order: 4
```

## Usage

```bash
# Run full pipeline
python blaze_phasesV3/phase0_library_v3.py
python blaze_phasesV3/phase1_blaze_v3.py
python blaze_phasesV3/phase2_blaze_v3.py
python blaze_phasesV3/phase3_blaze_v3.py

# Run for specific candidate
python blaze_phasesV3/phase1_blaze_v3.py 42
python blaze_phasesV3/phase2_blaze_v3.py 42
python blaze_phasesV3/phase3_blaze_v3.py 42

# Run with specific run directory
python blaze_phasesV3/phase1_blaze_v3.py runsV3/phase0_blaze_20240101_120000
```

## Data Flow

```
Library HDF5 → Phase 0 → candidates.csv
                           ↓
                       Phase 1 → phase1_multiband_data.h5
                           ↓
                       Phase 2 → phase2_multiband_data.h5 (+ Berry, Born-Huang)
                           ↓
                       Phase 3 → phase3_multiband_modes.h5 (spinor envelopes)
```

## Important Notes

### Berry Connection Approximation
Since BLAZE doesn't export eigenvector fields, the Berry connection is currently approximated. For accurate non-Abelian Berry connection, true Bloch function overlaps are needed. The parallel transport gauge ensures smooth gauge fixing.

### Born-Huang Potential
The Born-Huang potential Φ_mn captures adiabatic corrections from remote bands not in the subspace. It's computed using:
$$
\Phi_{mn} \approx \sum_\alpha \frac{1}{(\omega_\alpha - \omega_m)(\omega_\alpha - \omega_n)} \times \text{(curvature coupling)}
$$

### Coordinate System
All computations use **fractional coordinates** (s1, s2) ∈ [0,1)² on the moiré unit cell, with `B_moire` transforming to Cartesian.

## Dependencies

- `numpy`, `scipy`, `h5py`, `pandas`, `matplotlib`
- `blaze2d` (BLAZE photonic crystal solver)
- `pyyaml` for configuration

## References

- Theory: `docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md`
- Validation: `docs/envelopeApproximationDerivation/6_ValidationStrategiesAndPitfalls.md`
