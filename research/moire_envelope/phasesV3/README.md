# phasesV3 - Multi-Band Envelope Approximation Pipeline (MPB-based)

This package implements the **V3 multi-band envelope approximation** for moiré photonic crystals using **MPB (MIT Photonic Bands)**, based on the theory in `docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md`.

## Key Features (V3 vs V2)

| Feature | V2 (Single-Band) | V3 (Multi-Band) |
|---------|------------------|-----------------|
| Band tracking | Single target band | N_subspace bands simultaneously |
| Berry connection | Not included | A_mn(s) = i⟨u_m\|∂u_n⟩ |
| Born-Huang potential | Not included | Φ_mn from remote bands |
| Drift term | Dropped | Retained: η × v_mn · ∇F |
| Gauge fixing | Natural gauge | Parallel transport (SVD) |
| Envelope | Scalar F(s) | Spinor F_n(s) ∈ ℂ^N |

## Solver: MPB

This pipeline uses **MPB (MIT Photonic Bands)** instead of Blaze2D for band structure calculations. MPB is a well-established tool for photonic crystal simulations with:

- Direct eigenmode computation
- Support for TM and TE polarizations
- Accurate dispersion and field calculations

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

### Phase 1: Local Bloch Problems (`phase1_mpb_v3.py`)
- Runs **MPB** at registry points across moiré unit cell
- **V3 addition**: Extracts N_bands simultaneously, stores raw stencil data
- Output: `phase1_multiband_data.h5`
  - `omega`: (Ns1, Ns2, N_subspace) frequencies
  - `vg`: (Ns1, Ns2, N_subspace, 2) group velocities
  - `M_inv`: (Ns1, Ns2, N_subspace, 2, 2) mass tensors
  - `stencil/`: raw data for Berry connection calculation

### Phase 2: Berry Connection & Born-Huang (`phase2_mpb_v3.py`)
- **NEW in V3**: Computes non-Abelian Berry connection
- **NEW in V3**: Computes Born-Huang potential from extra bands
- Applies parallel transport gauge via SVD
- Output: `phase2_multiband_data.h5`
  - `Lambda`: (Ns1, Ns2, N, N) potential matrix
  - `A_berry`: (Ns1, Ns2, N, N, 2) Berry connection
  - `Phi_BH`: (Ns1, Ns2, N, N) Born-Huang potential
  - `v_drift`: (Ns1, Ns2, N, N, 2) drift velocity matrix
  - `M_inv`: (Ns1, Ns2, N, N, 2, 2) mass tensor matrix

### Phase 3: Envelope Solver (`phase3_mpb_v3.py`)
- Assembles full multi-band Hamiltonian as sparse block matrix
- Solves eigenvalue problem for spinor envelopes
- Output: `phase3_multiband_modes.h5`
  - `eigenvalues`: (n_modes,) cavity mode frequencies
  - `F_spinor`: (n_modes, Ns1, Ns2, N_subspace) envelope functions
  - Mode statistics in `phase3_mode_stats.json`

## Configuration (configsV3/)

### `phase0_mpb.yaml`
```yaml
run_name: mpb_v3
n_neighbor_bands: 2      # Bands above/below target (N_subspace = 2*2+1 = 5)
n_extra_bands: 4         # Additional bands for Born-Huang
```

### `phase1_mpb.yaml`
```yaml
phase1_Ns1: 128          # Grid resolution
phase1_Ns2: 128
mpb_registry_samples: 32 # Registry grid for MPB
mpb_fd_order: 4          # Finite difference order
mpb_resolution: 32       # MPB resolution
```

### `phase2_mpb.yaml`
```yaml
include_born_huang: true
include_drift_term: true
use_parallel_transport_gauge: true
born_huang_coupling: 1.0
```

### `phase3_mpb.yaml`
```yaml
n_modes: 20              # Number of cavity modes to compute
include_drift_term: true
include_kinetic_term: true
include_born_huang: true
fd_order: 4
```

### `phase5_mpb.yaml`
```yaml
phase5_mode_selection: min_spread   # Select most localized mode
phase5_resolution_per_a: 64         # 64 pixels per lattice constant
phase5_supercell_tiles: [2, 2]      # 2×2 moiré supercell
phase5_pml_thickness: 2.0           # PML in lattice constants
phase5_ringdown_time: 200.0         # Ringdown measurement time
```

## Usage

```bash
# Run the full pipeline
python phasesV3/phase0_library_v3.py configsV3/phase0_mpb.yaml
python phasesV3/phase1_mpb_v3.py auto
python phasesV3/phase2_mpb_v3.py auto
python phasesV3/phase3_mpb_v3.py auto
python phasesV3/phase5_meep_v3.py auto         # Full FDTD validation
python phasesV3/phase5_meep_v3.py auto --test  # Test mode (plots + estimates)

# Process specific candidate
python phasesV3/phase1_mpb_v3.py 5 auto  # Process candidate ID 5
```

## Output Directory Structure

```
runsV3/
└── phase0_mpb_v3_YYYYMMDD_HHMMSS/
    ├── phase0_candidates.csv
    ├── phase0_config.json
    ├── phase0_top_candidates_bands.png
    └── candidate_0/
        ├── phase0_meta.json
        ├── phase1_multiband_data.h5
        ├── phase1_fields_fractional.png
        ├── phase2_multiband_data.h5
        ├── phase2_multiband_fields.png
        ├── phase3_multiband_modes.h5
        ├── phase3_mode_stats.json
        ├── phase3_envelope_modes_by_spread.png
        ├── phase5_geometry_meep.png      # Meep plot2D output
        ├── phase5_simulation_setup.png   # Custom bilayer plot
        ├── phase5_simulation.mp4         # Full simulation video
        ├── phase5_results.json           # Q-factors and metrics
        └── phase5_harminv_modes.csv      # Detected resonances
```

## Phase 5: Meep FDTD Validation

Phase 5 performs full electromagnetic FDTD simulation using Meep to validate the envelope approximation predictions.

### Features
- **Continuous-wave source** at the predicted cavity frequency
- **Multiple Q-factor methods**:
  - Harminv ringdown analysis
  - Energy decay fitting: U(t) ∝ exp(-2γt) → Q = ω/(2γ)
  - Power loss method: Q = ωU/P
- **Streaming video** (MP4) of field evolution
- **High resolution**: 64 pixels per lattice constant for single-atom resolution
- **Large simulations**: 2×2 moiré supercell (~48M pixels typical)

### Test Mode
The `--test` flag runs without simulation to:
1. Generate geometry plots (MPB-style + custom bilayer plot)
2. Estimate computational resources (RAM, time, grid size)
3. Verify video generation (ffmpeg)

```bash
python phasesV3/phase5_meep_v3.py auto --test
```

### Resource Estimates (typical)
For a 2×2 moiré supercell at 64 px/a:
- Grid: ~7000 × 7000 pixels
- Memory: ~6 GB
- Time: ~2 minutes per candidate

## Comparison with Blaze Pipeline

This MPB-based pipeline (`phasesV3/`) is functionally equivalent to the Blaze-based pipeline (`blaze_phasesV3/`). The key difference is the band structure solver:

| Aspect | MPB (this pipeline) | Blaze |
|--------|---------------------|-------|
| Solver | MIT Photonic Bands | Blaze2D |
| Speed | Moderate | Fast |
| Accuracy | High (established) | To be verified |
| Field export | Supported | Limited |

Both pipelines produce identical output formats, so downstream analysis tools work with either.

## Requirements

- Python 3.8+
- meep (with mpb)
- numpy, scipy, h5py, pandas
- matplotlib, tqdm
- ffmpeg (for video generation in Phase 5)

Install meep with:
```bash
conda install -c conda-forge pymeep
```

## See Also

- `blaze_phasesV3/` - Blaze2D-based pipeline (faster)
- `docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md` - Theory derivation
- `docs/envelopeApproximationDerivation/6_ValidationStrategiesAndPitfalls.md` - Validation methods
