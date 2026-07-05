# V2 Run Outputs

This directory contains output from the V2 pipeline runs.

The V2 pipeline uses **fractional coordinates** for sampling the moiré unit cell,
which provides true periodic boundary conditions on any Bravais lattice.

See `README_V2.md` in the parent directory for full documentation.

## Structure

Each run creates a timestamped subdirectory:

```
runsV2/
  phase0_v2_library_YYYYMMDD_HHMMSS/
    config.yaml                    # Copy of configuration used
    phase0_candidates.csv          # All scored candidates
    phase0_top_candidates_bands.png  # Band diagram visualization
    
  phase1_v2_YYYYMMDD_HHMMSS/
    candidate_XXXX/
      phase1_band_data.h5          # s_grid in fractional coords
    ...
```

## Key Differences from V1

1. **Fractional grid**: `s_grid` is in $(s_1, s_2) \in [0,1)^2$ coordinates
2. **Corrected stacking shift**: No division by η in the registry formula
3. **Pipeline version marker**: Output CSVs include `pipeline_version: V2` column
