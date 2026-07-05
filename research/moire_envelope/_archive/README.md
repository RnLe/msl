# Archive — superseded pipeline generations

Everything in this folder is **kept for provenance, not for use**. The current
pipeline is `../phasesV3/` (MPB) and `../blaze_phasesV3/` (Blaze2D) with
`../configsV3/`; the final Blaze-based V4 studies live in the
[blaze2d repository](https://github.com/RnLe/blaze2d) under
`thesis/research/phasesV4/`.

## Generation history

| Generation | Folders here | Era | Solver | Physics | Why superseded |
| --- | --- | --- | --- | --- | --- |
| **V1** | `phases_v1/`, `configs_v1/`, `runs_v1/` | Oct–Nov 2025 | MPB | Single-band envelope, Γ-point candidate search, Meep Q-factor phases | Single-band model proved quantitatively wrong (interband coupling is non-perturbative); superseded by multi-band V3 |
| **V1 (Blaze)** | `blaze_phases_v1/` | Nov 2025 | Blaze2D (early) | Blaze port of the V1 phases | Same physics limitations as V1 |
| **V2** | `phasesV2/`, `configsV2/`, `blaze_phasesV2/`, `runsV2/` | Dec 2025 | MPB + Blaze2D | Library-based candidate search, still single-band | No Berry connection, no Born–Huang potential, drift term dropped |
| **V3** | *(current — lives in `../phasesV3/`, `../blaze_phasesV3/`)* | Jan–Mar 2026 | MPB + Blaze2D | Multi-band spinor envelope: Berry connection A_mn, Born–Huang Φ_mn, drift term, parallel-transport gauge fixing, C4/C2/C6 symmetrization | — |
| **V4** | *(lives in the blaze2d repo)* | Mar 2026 | Blaze2D `EAExtractor` | Same theory as V3, extracted directly through Blaze2D's Rust API; magic-angle and band-0 sweep studies | — |

## Other folders

| Folder | Contents |
| --- | --- |
| `examples/` | V1-era usage examples (`run_phase1_example.py` imports the V1 phases) mixed with early Blaze2D TOML examples |
| `results_R01_R06/` | Feb 2026 result sets from the pre-symmetrization pipeline (old Γ-point candidate, no off-diagonal Berry) — numbers outdated |
| `findings_F01_F06/` | Feb 2026 findings arc; F01 (no discrete states), F03 (Hermiticity fix) and F06 (gauge/normalization) remain valid as methodology history, the specific numbers do not |
| `results_bands/`, `results_bands_old/` | Miniband-structure scripts + data from the old candidate; script logic was folded into the V3 η-sweep/T03 tooling |
| `root_debug/` | One-off debugging and comparison scripts from the pipeline-root era (solver output comparisons, y-flip investigations, Bloch-phase checks) |
| `bulk_output/` | Small leftover outputs from bulk two-atom-basis sweeps |

The definitive S1–S7 diagnostic arc that *validated* the V3 methodology is
**not** archived — it lives in `../corrections_findings/` because its findings
document why the production pipeline is correct.

Raw data sizes and availability: see `../../DATA.md`.
