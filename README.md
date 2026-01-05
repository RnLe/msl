# Moire Superlattice (MSL)

A Master's thesis project at TU Dortmund University combining a Rust framework for Moire lattice physics, interactive research visualizations, and original theoretical work on photonic cavity modes.

---

## Overview

This repository contains three interconnected modules:

| Module | Description |
|--------|-------------|
| **rust-core, rust-python, rust-wasm** | Rust library for 2D/3D lattice computations with Python and WebAssembly bindings |
| **web** | Interactive documentation website built with Next.js and Nextra, featuring live React components |
| **research** | Hands-on Master's thesis work: envelope approximation theory for Moire photonic crystals |

---

## Rust Framework

The core library (`rust-core/`) provides efficient algorithms for crystallographic and Moire physics:

**Lattice Operations**
- All five 2D Bravais lattice types with construction helpers
- Wigner-Seitz cell and Brillouin zone computation
- Reciprocal lattice transformations
- Coordination number and nearest-neighbor analysis
- Lattice point generation within arbitrary polygons and radii

**Moire Lattice Support**
- Twisted bilayer construction from arbitrary transformations
- Effective Moire lattice vector computation
- Commensurability detection
- Support for rotation, scaling, shear, and general matrix transformations

**Bindings**
- `rust-python/`: PyO3-based Python package via Maturin
- `rust-wasm/`: WebAssembly bindings for browser-based applications

---

## Interactive Website

The `web/` directory hosts a Nextra-based documentation site with interactive visualizations.

---

## Research: Envelope Approximation Theory

The central theoretical contribution of this thesis is an **envelope approximation method** for predicting localized photonic cavity modes in twisted bilayer photonic crystals.

**Physical Motivation**

When two periodic photonic crystals are stacked with a small twist angle, they form a Moire superlattice with a much larger period. The local stacking configuration varies smoothly across the Moire cell, creating an effective potential landscape that can trap light.

**The Envelope Approximation**

Instead of solving Maxwell's equations on the full Moire supercell (computationally expensive; effectively impossible right now), the envelope approximation treats the problem as:

1. **Phase 0**: Construct the Moire lattice and compute the registry (stacking shift) map
2. **Phase 1**: Solve local Bloch problems at each stacking configuration using MPB
3. **Phase 2**: Assemble an effective Hamiltonian: kinetic, drift, and potential terms
4. **Phase 3**: Solve for envelope eigenstates to predict cavity mode frequencies
5. **Phase 4-5**: Validation against perturbation theory and FDTD simulations (Meep)

The resulting eigenvalue problem captures the essential physics of Moire-induced photonic cavities while reducing computational cost by orders of magnitude.

---

## Project Structure

```
msl/
├── rust-core/         Rust library: lattice algorithms, Moire physics
├── rust-python/       Python bindings via PyO3
├── rust-wasm/         WebAssembly bindings
├── web/               Next.js documentation with interactive components
├── research/          Master's thesis research code
└── Cargo.toml         Workspace configuration
```

Each module contains a `Makefile` for common operations.

---

## Getting Started

**Rust Framework**
```bash
cargo build --workspace
cargo test --workspace
```

**Python Bindings**
```bash
cd rust-python
pip install maturin
maturin develop
```

**WebAssembly**
```bash
cd rust-wasm
wasm-pack build --target web --out-dir pkg
```

**Website**
Visit https://rnle.github.io/msl/
Or run the website locally:
```bash
cd web
pnpm install
pnpm dev
```

**Research Pipeline**
```bash
cd research/moire_envelope
make phase0   # Candidate search
make phase1   # Local Bloch problems (requires MPB)
make phase2   # Operator assembly
make phase3   # Eigenvalue solve
# etc.
```

---

## Technical Details

**Rust Core Features**
- Voronoi tessellation via Voronoice (2D) and Voro-RS (3D)
- Parallel computation with Rayon
- Designed for photonic band structure calculations (MPB-compatible foundations)

**Website Stack**
- Next.js 15 with Nextra documentation theme
- React components with Konva for canvas rendering
- WASM integration for real-time lattice computation
- MDX for combining prose with interactive elements

**Research Dependencies**
- MIT Photonic Bands (MPB) for band structure calculations
- Meep for FDTD validation
- NumPy, SciPy for numerical computation
- Managed via Conda environment (`research/environment.yml`)

---

## License

MIT

---

## Author

Rene-Marcel Lehner  
TU Dortmund University  
rene.lehner@tu-dortmund.de
