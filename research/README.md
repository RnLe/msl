# Research

Research code and data behind the Master's thesis
**"Photonic Band Theory of Moiré Crystals: A Two-Scale Approach"**
(Rene-Marcel Lehner, TU Dortmund University, March 2026).

The research concluded with the thesis submission. This tree preserves the full
method history: the production pipeline, the validation studies, and the
archived earlier generations. Raw data (~185 GB) is local-only and documented
in [`DATA.md`](DATA.md).

## Map

| Folder | Status | Contents |
| --- | --- | --- |
| [`moire_envelope/`](moire_envelope/) | **core** | The multi-band envelope-approximation pipeline (Phase 0–3, symmetrization, η-sweeps), the S1–S7 methodology diagnostics, the FDFD reference solver, and all thesis result studies (T01–T11, T_*) |
| [`studies/fdfd_convergence/`](studies/fdfd_convergence/) | **validation** | Systematic FDFD convergence study (angle × resolution × target frequency) that established the coarse-grid FDFD reference used in the thesis validations chapter |
| [`scripts/`](scripts/) | tooling | Build scripts for the Meep/MPB simulation stack |
| [`_archive/`](_archive/) | archive | The V0 envelope-approximation prototype and eight early exploration studies — see [`_archive/README.md`](_archive/README.md) |
| `environment.yml` | env | Conda/mamba environment (`msl`) with MPB, Meep, NumPy/SciPy stack |

## How the thesis maps onto this tree

| Thesis chapter | Where the work lives |
| --- | --- |
| Ch. 2 — Photonic moiré crystals (two-atomic crystal approximation, registry maps) | Theory: [`../docs/envelopeApproximationDerivation/`](../docs/envelopeApproximationDerivation/); early demos in `_archive/explorations/` |
| Ch. 3 — Envelope approximation | Derivation docs 1–6 + `moire_envelope/phasesV3/` implementation |
| Ch. 4 — Blaze2D | The [blaze2d repository](https://github.com/RnLe/blaze2d) (solver, benchmarks, and the V4 Blaze-native EA studies under `thesis/research/phasesV4/`) |
| Ch. 5 — Validations | `moire_envelope/thesis_results/` (EA vs FDFD vs MPB comparisons, internal checks) + `studies/fdfd_convergence/` |

## Pipeline generations

The envelope approximation went through five generations; keeping this history
public is deliberate:

1. **V0** (`_archive/envelope_approximation_v0/`) — first prototype, flat scripts.
2. **V1** (`moire_envelope/_archive/phases_v1/`) — run-directory structure, Γ-point candidate search, single-band.
3. **V2** (`moire_envelope/_archive/phasesV2/`) — band-library candidate search, still single-band.
4. **V3** (`moire_envelope/phasesV3/`, `blaze_phasesV3/`) — **the thesis pipeline**: multi-band spinor envelope with Berry connection, Born–Huang potential, drift term, gauge fixing, point-group symmetrization.
5. **V4** (blaze2d repo, `thesis/research/phasesV4/`) — V3 theory extracted directly through Blaze2D's Rust `EAExtractor` API; magic-angle and band-0 sweep studies.

A word of honesty: this was active research up to the submission deadline. Not
every study in `thesis_results/` made it into the thesis, and some result sets
explore directions that turned out inconclusive. The status notes in
`moire_envelope/thesis_results/master_plan.md` and
`final_thesis_direction.md` state what is solid and what remains open.
