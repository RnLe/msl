# V2 Configurations

This directory contains configuration files for the V2 pipeline.

The V2 pipeline uses **fractional coordinates** for sampling the moiré unit cell.
See `README_V2.md` in the parent directory for full documentation.

## Configuration Files

### Phase 0 (Candidate Search)

- `phase0_library.yaml` — For the MPB/eigsh pipeline using the HDF5 band library
- `phase0_blaze.yaml` — For the BLAZE custom solver pipeline

### Phase 1 (Local Bloch Problems)

- `phase1.yaml` — Local Bloch problems with V2 fractional coordinates

All configurations output to `runsV2/` and mark output with `pipeline_version: V2`.

### Key V2 Parameters

Phase 1 and later phases use fractional coordinate parameters:

```yaml
# Grid is defined in fractional coordinates (s1, s2) ∈ [0,1)²
phase1_Ns1: 64  # Grid points in s1 direction  
phase1_Ns2: 64  # Grid points in s2 direction

# Stacking gauge in fractional monolayer coordinates
tau: [0.0, 0.0]
```

**V2 Key Differences:**
1. Fractional grid `s_grid` is primary (not Cartesian `R_grid`)
2. Registry formula: `δ(R) = (R(θ) - I) · R + τ` — NO division by η!
3. Output HDF5 includes `B_moire` and `B_mono` matrices for coordinate transforms

## Usage

```bash
# Phase 0: MPB/eigsh pipeline
python phasesV2/phase0_candidate_search_library.py configsV2/phase0_library.yaml

# Phase 0: BLAZE pipeline  
python blaze_phasesV2/phase0_library.py configsV2/phase0_blaze.yaml

# Phase 1: Local Bloch problems (V2)
python phasesV2/phase1_local_bloch.py auto  # Uses latest phase0 run
python phasesV2/phase1_local_bloch.py latest configsV2/phase1.yaml
```
