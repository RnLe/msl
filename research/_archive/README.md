# Archive — early research phases

Work in this folder predates the production envelope-approximation pipeline in
`../moire_envelope/`. It is kept for provenance and to document the research
arc; none of it feeds the final thesis results.

## `envelope_approximation_v0/`

The very first envelope-approximation prototype (mid 2025): Phases 0–5 in
single flat scripts (lattice setup → local Bloch via MPB → EA operator →
envelope solve → validation → Meep FDTD). Superseded by the versioned
`moire_envelope` pipeline (V1 → V3), which reworked every one of these steps.

## `explorations/`

Exploratory studies from the first half of the thesis, roughly in order:

| Folder | What it explored | Status |
| --- | --- | --- |
| `lattice_algorithms/` | Filling arbitrary areas with lattice points (Rust-core algorithm demo) | Folded into `rust-core` |
| `python-example/` | First notebooks using the Python bindings: lattice visualization, moiré basics, photonic band diagrams | Didactic only; the website (`web/`) supersedes these as demos |
| `animations/` | Twisted-bilayer animation of the moiré effect | One-off visualization |
| `stacking_shift_map/` | Registry (stacking-shift) map construction demo | Concept became `moire_envelope` Phase 0 registry maps |
| `moire_cavity_exploration/` | Early hunt for localized cavity modes in moiré cells; monolayer prefilter + local-cell demos | Superseded by the Phase 0 candidate search; the "isolated cavity mode" picture was later refuted (dense overlapping minibands) |
| `band_diagram_scan/` | `bandlib`: a packaged MPB scan tool sweeping lattice type, ε and r/a | Superseded by the Phase 0 band library |
| `photonic_bm_model/` | Photonic Bistritzer–MacDonald model; Dirac-point search and confirmation | Background study for the honeycomb K-point (Dirac) candidate; the thesis' multiband EA generalizes beyond this 2-band model |
| `multi_moire_construction/` | ML side-experiment: inferring lattice parameters from moiré patterns (TensorFlow) | Not used in the thesis. Model weights/tensorboard logs are untracked (local only) |
