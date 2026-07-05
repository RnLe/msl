# Thesis Results — Moiré Envelope Approximation

## Selected Candidates

| ID | Label | Lattice | k-point | Band | Pol | r/a  | ε_bg | θ\*  | cond | V/E@2° | Type |
|----|-------|---------|---------|------|-----|------|------|------|------|--------|------|
| C1 | `hex_M_b1`     | hex    | M | 1 | TE | 0.10 |  9.9 | 2.1° | 15.0 |  1.1 | min+ |
| C2 | `hex_M_b3`     | hex    | M | 3 | TE | 0.10 |  9.8 | 4.2° | 35.5 |  4.3 | min+ |
| C3 | `square_M_b3`  | square | M | 3 | TM | 0.15 |  1.8 | 2.5° |  1.0 |  1.5 | min+ |

### Selection Rationale

- **C1 (hex_M_b1):** Best hex candidate. Band minimum at M, moderate anisotropy
  (cond=15), low contrast ε=9.9, practical θ\*=2.1°, V/E@2°=1.1 (weakly bound regime).
  Gap isolation: 10.5 mΩ.

- **C2 (hex_M_b3):** Second hex candidate. Deeper potential (V_depth=0.019),
  higher band with θ\*=4.2°. V/E@2°=4.3 gives access to strongly bound regime.
  Provides comparison for interband coupling effects (higher Berry connection).

- **C3 (square_M_b3):** Best square candidate. **Perfectly isotropic** (cond=1.0,
  exact M_xx = M_yy by C₄ symmetry). TM polarization adds diversity.
  Low contrast ε=1.8 (ideal EA validity window). θ\*=2.5°.
  Largest gap isolation of all candidates: 28.9 mΩ.

## Directory Structure

```
thesis_results/
├── README.md                   ← this file
├── candidates.yaml             ← candidate definitions (machine-readable)
├── setup_thesis_candidates.py  ← creates Phase 0 run dirs + phase0_candidates.csv
├── run_pipeline.sh             ← master runner: Phase 1→2→C4sym→3→η-sweep
│
├── T01_candidate_selection/    ← Tier 1/2 screening summary figures
├── T02_hamiltonian_landscape/  ← ω(s), V(s), A(s), M⁻¹(s) maps
├── T03_miniband_dispersion/    ← η-sweep: E_n(η) and bandwidth vs θ
├── T04_mode_gallery/           ← envelope |F_n(s)|² for n=0..5 at select θ
├── T05_field_reconstruction/   ← full E(r) from envelope × Bloch
├── T06_scaling_laws/           ← E_gap, BW, IPR vs η; power-law fits
├── T07_disorder_robustness/    ← twist-angle disorder σ_θ study
├── T08_maxwell_validation/     ← FDTD vs envelope eigenvalues
├── T09_symmetry_gauge/         ← C4/C2 commutator, gauge smoothness
├── T10_interband_coupling/     ← A_off-diag effect: with/without comparison
└── figures/                    ← final publication-ready composites
```

## Pipeline Execution Order

1. `python setup_thesis_candidates.py` — creates `runsV3/thesis_*` run dirs
2. Phase 1: `python phasesV3/phase1_mpb_v3.py <run_dir> configsV3/thesis_<label>.yaml`
3. Phase 2: `python phasesV3/phase2_mpb_v3.py auto configsV3/thesis_<label>.yaml`
4. C4/C2 symmetrization: `python corrections_findings/S4b_c4_symmetrize.py <run_dir>`
5. Phase 3: `python phasesV3/phase3_mpb_v3.py auto configsV3/thesis_<label>.yaml`
6. η-sweep: `python eta_sweep.py --run-dir <run_dir>`
7. T01–T10 scripts: `python thesis_results/T0X_<name>/compute.py`

## Key Physics Parameters

- **Off-diagonal Berry connection:** Always ON (`include_offdiag_A=True`)
- **Born-Huang potential:** OFF (validated negligible in S5)
- **C4-symmetrization:** Applied to square candidates post-Phase 2
- **C2-symmetrization:** Applied to hex M-point candidates post-Phase 2
- **Production settings:** resolution=64, registry=128×128, fd_order=4
- **η-sweep angles:** θ = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]°
