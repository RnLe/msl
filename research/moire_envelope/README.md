# Moiré Envelope Approximation — Pipeline

Production pipeline of the thesis *"Photonic Band Theory of Moiré Crystals: A
Two-Scale Approach"*. It computes the multi-band envelope-approximation (EA)
Hamiltonian for twisted bilayer photonic crystals and solves for moiré
envelope modes — without ever building a moiré supercell.

**Theory:** `../../docs/envelopeApproximationDerivation/` (docs 1–6, culminating
in `5_FinalMultiBandTwoScaleEA.md`). The envelope equation solved here:

```
Σₙ [ Λ_mn(s) + η v_mn(s)·(−i∇) + η² ( ½ Dᵢ M⁻¹_mn,ij Dⱼ + Φ^BH_mn(s) ) ] Fₙ(s) = E F_m(s)
```

with registry coordinate `s`, twist parameter `η = a/L ∝ θ`, band-energy
landscape Λ, group-velocity drift v, inverse-mass tensor M⁻¹ (Löwdin-corrected),
Berry connection (inside the covariant derivative D), and Born–Huang potential Φ.

## Pipeline phases (V3 — current)

| Phase | Script (MPB) | Script (Blaze2D) | Output |
| --- | --- | --- | --- |
| 0 — candidate search | `phasesV3/phase0_library_v3.py` | `blaze_phasesV3/phase0_library_v3.py` | `runsV3/<run>/phase0_candidates.csv` |
| 1 — local Bloch problems | `phasesV3/phase1_mpb_v3.py` | `blaze_phasesV3/phase1_blaze_v3.py` | `phase1_multiband_data.h5` (eigenvalues + eigenvectors on the registry grid) |
| 2 — EA operator assembly | `phasesV3/phase2_mpb_v3.py` | `blaze_phasesV3/phase2_blaze_v3.py` | Λ, v, M⁻¹, Berry A, Born–Huang Φ fields |
| sym — point-group symmetrization | `corrections_findings/S4b_c4_symmetrize.py` | — | C4/C2/C6-symmetrized Phase-2 fields |
| 3 — envelope eigensolve | `phasesV3/phase3_mpb_v3.py` | `blaze_phasesV3/phase3_blaze_v3.py` | Envelope modes F_n(s), eigenvalues |
| η-sweep — twist-angle sweep | `phasesV3/eta_sweep.py` | — | `eta_sweep_*/sweep_results.json` (bandwidths, IPR, gaps vs θ) |

Convenience targets for all of the above: `Makefile` (run `make help`).

### Thesis-candidate workflow

```bash
make thesis_candidates                                        # runsV3/thesis_* dirs
python phasesV3/phase1_mpb_v3.py <run_dir> configsV3/thesis_hex_M_b1.yaml
python phasesV3/phase2_mpb_v3.py auto configsV3/thesis_hex_M_b1.yaml
python corrections_findings/S4b_c4_symmetrize.py <run_dir>
python phasesV3/phase3_mpb_v3.py auto configsV3/thesis_hex_M_b1.yaml
python phasesV3/eta_sweep.py --run-dir <run_dir>
```

End-to-end runners used for the thesis runs: `scripts/run_overnight_hex_pipeline.sh`,
`scripts/run_fullA_sweep.sh` (invoke from anywhere; they cd to this directory).

## Directory map

| Directory | Status | Contents |
| --- | --- | --- |
| `phasesV3/` | **current** | MPB-based multi-band pipeline + η-sweep, field reconstruction, Meep validation phases; `MPB_units.md` documents unit conventions |
| `blaze_phasesV3/` | **current** | Blaze2D-based variant of phases 0–3 |
| `phasesV4/` | **current** | MPB ↔ Blaze2D cross-validation suite + `ea_pipeline_guide.md` (Blaze2D `EAExtractor` API guide). The full V4 studies live in the [blaze2d repo](https://github.com/RnLe/blaze2d) (`thesis/research/phasesV4/`) |
| `configsV3/` | **current** | YAML configs incl. the thesis candidates (`thesis_*.yaml`) |
| `common/` | **current** | Shared geometry / IO / MPB / plotting / scoring utilities |
| `corrections_findings/` | **reference** | S1–S7 diagnostic arc that validated the methodology (gauge fixing, symmetrization, Hamiltonian term-by-term audit). All fixes are folded into V3; kept as the record of *why* the pipeline is correct |
| `thesis_results/` | **results** | T01–T11 + T_* studies behind the thesis figures; `master_plan.md` and `final_thesis_direction.md` document the final state of the research. Also home of the FDFD reference solver (`T_direct_validation/fdfd_solver.py`) |
| `scripts/` | helpers | Inspection helpers, normalization checks, overnight runners |
| `tests/` | tests | Phase 1/2 unit tests (`make test`) |
| `runsV3/` | **data (local-only)** | Raw pipeline runs (~88 GB). Only `phase0_config.json`, `phase0_candidates.csv` and η-sweep summaries are committed — see `../DATA.md` |
| `_archive/` | archive | Superseded V1/V2 generations — see `_archive/README.md` |

## Solvers

- **MPB** (via the `msl` conda env, `../environment.yml`) — reference solver.
- **[Blaze2D](https://github.com/RnLe/blaze2d)** (`pip install blaze2d`) — our
  Rust PWE eigensolver with full eigenvector extraction, built for the dense
  registry sweeps this pipeline needs. Cross-validated against MPB in
  `phasesV4/mpb_blaze_validation/` (report: `phasesV4/blaze_mpb_validation_report.md`).
- **FDFD** (`thesis_results/T_direct_validation/fdfd_solver.py`) — independent
  finite-difference frequency-domain reference for direct supercell validation;
  convergence study in `../studies/fdfd_convergence/`.

## Key results (thesis)

- Universal miniband bandwidth scaling **BW ∝ η^1.92** across square, hex and
  honeycomb lattices (η-sweep data committed under `runsV3/thesis_*/eta_sweep_*/`).
- Multiband Berry coupling is non-perturbative (dominant-band fraction drops to
  0.31; IPR shifts up to 5×).
- Honeycomb K-point: interband coupling is **purely geometric** (Λ₀₁ ≡ 0 by
  symmetry, Berry-only), with a photonic magic angle near θ ≈ 0.7°.
- Full narrative and honest status assessment: `thesis_results/master_plan.md`,
  `thesis_results/final_thesis_direction.md`.
