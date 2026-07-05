# Moiré Superlattice (MSL)

Companion repository to the Master's thesis
**"Photonic Band Theory of Moiré Crystals: A Two-Scale Approach"**
(Rene-Marcel Lehner, TU Dortmund University, Condensed Matter Theory, March 2026).

It combines a Rust framework for moiré lattice physics, an interactive
documentation website, and the complete research pipeline — code, configs,
docs, and curated results — of the thesis.

---

## The project ecosystem

| Repository | Role |
| --- | --- |
| **msl** (this repo) | Rust lattice framework, interactive website, and the envelope-approximation research pipeline with validation studies |
| [**blaze2d**](https://github.com/RnLe/blaze2d) | Blaze2D — plane-wave eigensolver for 2D photonic crystals written in Rust (`pip install blaze2d`), purpose-built for this research; also hosts the final Blaze-native EA studies |
| Thesis repository | Typst source of the thesis manuscript and its figures |

---

## The research in one paragraph

Twisting two photonic crystals against each other produces a moiré lattice
whose period far exceeds that of the crystals it is built from — and which
standard solvers cannot treat: at generic angles no periodic unit cell exists
at all. This thesis develops a **multiband envelope approximation** that starts
from the local Bloch modes of the twisted bilayer and yields an effective
Hamiltonian whose ingredients — band energies, group velocities, effective
masses, Berry connections, and Born–Huang potentials — live on the registry
domain and are independent of the twist angle. The pipeline is validated
against an independent FDFD solver (itself checked against MPB), confirms the
predicted quadratic bandwidth scaling BW ∝ θ², and resolves geometric moiré
physics (e.g. purely Berry-mediated interband coupling at Dirac points) that
no supercell calculation can access.

---

## Repository structure

```text
msl/
├── rust-core/         Rust library: lattice algorithms, moiré physics
├── rust-python/       Python bindings via PyO3/Maturin
├── rust-wasm/         WebAssembly bindings for the website
├── web/               Next.js + Nextra documentation site with live components
├── docs/              Envelope-approximation theory derivations (docs 1–6)
├── research/          The thesis research (see research/README.md)
│   ├── moire_envelope/    Production EA pipeline + thesis results
│   ├── studies/           FDFD convergence study
│   └── _archive/          Early prototypes and explorations (kept public)
└── Cargo.toml         Workspace configuration
```

**Where to start reading:**

- Research overview and thesis-chapter map: [`research/README.md`](research/README.md)
- Pipeline how-to: [`research/moire_envelope/README.md`](research/moire_envelope/README.md)
- Theory derivations: [`docs/envelopeApproximationDerivation/`](docs/envelopeApproximationDerivation/)
- Data policy and raw-data manifest: [`research/DATA.md`](research/DATA.md)

---

## Rust framework

The core library (`rust-core/`) provides efficient algorithms for
crystallographic and moiré physics:

**Lattice operations** — all five 2D Bravais lattice types, Wigner–Seitz cell
and Brillouin zone computation, reciprocal transformations, coordination
analysis, lattice point generation within arbitrary polygons and radii.

**Moiré lattice support** — twisted bilayer construction from arbitrary
transformations, effective moiré lattice vectors, commensurability detection,
rotation/scaling/shear transformations.

**Bindings** — `rust-python/` (PyO3 + Maturin) and `rust-wasm/`
(browser-ready WebAssembly).

```bash
cargo build --workspace
cargo test --workspace
```

---

## Interactive website

`web/` hosts a Nextra-based documentation site with interactive moiré
visualizations (React + Konva + WASM): <https://rnle.github.io/msl/>

```bash
cd web && pnpm install && pnpm dev
```

---

## Research pipeline

The envelope-approximation pipeline (MPB- or Blaze2D-backed) lives in
`research/moire_envelope/`:

```bash
cd research/moire_envelope
make phase0        # candidate search
make phase1        # local Bloch problems on the registry grid (MPB)
make phase2        # EA operator assembly (Λ, v, M⁻¹, Berry A, Born–Huang Φ)
make symmetrize    # point-group symmetrization
make phase3        # envelope eigensolve
make eta_sweep     # twist-angle sweep
```

Dependencies are managed via the conda/mamba environment
`research/environment.yml` (MPB, Meep, NumPy/SciPy) plus `pip install blaze2d`
for the Blaze2D-backed variant.

### Data availability

The pipeline's raw output (~185 GB of HDF5/NPZ run data) is not tracked in
git. A curated set of small key-result files **is** committed, and
[`research/DATA.md`](research/DATA.md) documents every local dataset — what it
contains, how it was generated, and how to regenerate it. Full datasets are
available from the author on request.

### Research status

The research concluded with the thesis submission (March 2026). Superseded
pipeline generations are archived — not deleted — under
`research/_archive/` and `research/moire_envelope/_archive/`, with READMEs
explaining each generation. Honest status notes on what is solid and what
remains open: `research/moire_envelope/thesis_results/master_plan.md`.

---

## Author

Rene-Marcel Lehner \
TU Dortmund University \
<rene.lehner@tu-dortmund.de>
