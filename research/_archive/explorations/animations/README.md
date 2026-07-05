# Moiré Animation Assets

This directory contains utilities for producing visual animations based on the `moire_lattice_py` bindings.

## Contents

- `twisted_triangular_animation.py` &mdash; generates a GIF showing the evolution of the moiré superlattice produced by twisting a triangular (hexagonal) monolayer. The animation displays the base lattice on the right, the moiré lattice on the left, and overlays the live twist angle together with the ratio of the moiré lattice constant relative to the monolayer lattice constant.

## Usage

```bash
cd research/animations
python twisted_triangular_animation.py
```

The script saves `twisted_triangular_moire.gif` in the same directory. Ensure the `moire_lattice_py` package is available (e.g., by running `maturin develop` in `rust-python/` beforehand) and that Pillow support for GIF writing is installed (Matplotlib pulls it in automatically when Pillow is present).
