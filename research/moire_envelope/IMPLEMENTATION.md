# Envelope Approximation Implementation - Phase 0

## Overview

This is the implementation of Phase 0 of the Photonic Moiré Envelope Approximation pipeline. Phase 0 performs candidate search and scoring to identify promising photonic crystal moiré lattice designs for further analysis.

## What was implemented

### Directory Structure
```
moire_envelope/
├── common/              # Shared utilities
│   ├── __init__.py
│   ├── geometry.py      # Lattice geometry utilities
│   ├── moire_utils.py   # Moiré lattice construction using Rust bindings
│   ├── mpb_utils.py     # MPB band structure utilities
│   ├── io_utils.py      # I/O and configuration management
│   ├── scoring.py       # Candidate scoring functions
│   └── plotting.py      # Visualization utilities
├── phases/              # Phase implementations
│   ├── __init__.py
│   └── phase0_candidate_search.py  # Phase 0 implementation
├── configs/             # Configuration files
│   ├── base_config.yaml
│   ├── search_config_square.yaml
│   └── search_config_hex.yaml
└── runs/                # Output directory for runs
```

### Key Features

#### 1. Moiré Lattice Construction
- Uses **Rust-Python bindings** from `moire_lattice_py` for high-performance lattice calculations
- Supports square, hexagonal, and rectangular lattice types
- Computes moiré parameters (length scale, reciprocal vector magnitude)

#### 2. Candidate Search
- Explores design space over:
  - Lattice types (square, hex, rect)
  - Twist angles (0.5° - 3.0°)
  - Hole radii (r/a)
  - Background dielectric constants
  - High symmetry points and band indices
  
#### 3. Scoring System
- **Band flatness score**: Favors flat bands (small curvature)
- **Spectral isolation score**: Favors large bandgaps
- **Parabolic validity score**: Ensures EA approximation is valid
- **Group velocity score**: Favors band extrema (v_g ≈ 0)
- **Dielectric contrast score**: Based on material properties

#### 4. Simplified Model
For rapid design space exploration, Phase 0 uses a simplified analytical model instead of full MPB calculations. This allows:
- Fast candidate generation (810 candidates in ~0.17s)
- Exploration of broad parameter space
- Identification of promising regions for detailed study

### Output Format

Phase 0 generates:
- `phase0_candidates.csv`: Complete candidate table with all metrics and scores
- `config.yaml`: Copy of configuration for reproducibility

Each candidate row contains:
- Geometry parameters (lattice_type, theta_deg, r_over_a, eps_bg)
- k-point and band information
- Dispersion metrics (omega0, curvature, group velocity)
- Moiré parameters (G_magnitude, moire_length)
- Individual scores and total score
- EA validity flag

## Usage

### Running Phase 0

```bash
# Activate environment
mamba activate msl

# Run with base configuration
cd /home/renlephy/msl/research/moire_envelope
python phases/phase0_candidate_search.py configs/base_config.yaml

# Run with lattice-specific configuration
python phases/phase0_candidate_search.py configs/search_config_square.yaml
python phases/phase0_candidate_search.py configs/search_config_hex.yaml
```

### Example Output

```
======================================================================
Phase 0: Candidate Search & Scoring
======================================================================

Search space:
  Lattice types: ['square', 'hex']
  Twist angles: 5 points in [0.5°, 3.0°]
  Hole radii (r/a): [0.2, 0.3, 0.4]
  Background ε: [4.0, 6.0, 9.0]
  Target bands: [3, 4, 5]

Generated 810 candidates

Top 10 candidates:
 candidate_id lattice_type  theta_deg  r_over_a  eps_bg k_label  band_index  S_total  valid_ea_flag
          156       square      1.125       0.4     9.0       X           3 0.698726          False
           72       square      0.500       0.4     9.0       Γ           3 0.697951           True
          153       square      1.125       0.4     9.0       Γ           3 0.697552          False
          ...
```

## Configuration

Configuration files use YAML format with parameters for:

### Search Parameters
- `lattice_types`: List of lattice types to explore
- `theta_range`, `n_theta`: Twist angle range and number of points
- `r_over_a_list`: List of hole radii (normalized)
- `eps_bg_list`: List of background dielectric constants
- `target_bands`: Band indices to analyze

### Scoring Parameters
- `w_flat`, `w_gap`, `w_parab`, `w_vg`, `w_contrast`: Scoring weights
- `alpha_parab`, `beta_parab`: EA validity criteria
- Reference values: `kappa_0`, `Delta_0`, `v_0`, `eps_bg_max`

### Phase Control
- `K_candidates`: Number of top candidates for Phase 1
- `use_simplified_model`: Use fast analytical model vs full MPB

## Next Steps

After Phase 0, the top candidates can be fed into:

- **Phase 1**: Local Bloch problems at frozen registry points
- **Phase 2**: Envelope operator assembly
- **Phase 3**: Envelope eigenvalue solver
- **Phase 4**: Validation against full moiré simulation
- **Phase 5**: Meep Q-factor analysis

The framework is set up to support the full pipeline as described in the comprehensive guide.

## Dependencies

- Python ≥ 3.10
- `moire_lattice_py` (Rust Python bindings) - built via `make build-python-dev`
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `pyyaml` for configuration
- `meep`, `mpb` (for future full simulations)

## Notes

1. **Rust Integration**: Successfully uses the Rust Python bindings for moiré lattice construction, demonstrating the framework's hybrid Rust/Python architecture.

2. **Simplified Model**: Phase 0 uses analytical approximations for rapid exploration. For production use with specific materials, enable full MPB calculations by setting `use_simplified_model: false` in the config.

3. **Scalability**: The implementation generates hundreds of candidates in milliseconds, making it suitable for optimization loops and large parameter sweeps.

4. **Extensibility**: The modular design with `common/` utilities makes it easy to add new lattice types, scoring functions, or analysis methods.
