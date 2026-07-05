# FDFD TE Convergence Study — Square Lattice, X Point

Systematic convergence study of the FDFD TE eigensolver at the X point
of a square-lattice photonic crystal moiré supercell.

## Crystal parameters

| Parameter | Value |
|-----------|-------|
| Lattice   | Square |
| r/a       | 0.2 |
| ε_rod     | 8.9 |
| ε_bg      | 1.0 |
| q-vector  | X = (π, 0) |
| Polarization | TE |

## Sweep axes

| Axis | Values |
|------|--------|
| Twist angles | 1° (m=114), 2° (m=57), 4° (m=29), 8° (m=14) |
| Resolutions  | 1, 4, 8, 16 px per unit cell |
| Target freqs | 0.05, 0.1, 0.2, 0.3, 0.4 (c/a) |
| Modes/run    | 20 |

Total: 4 × 4 × 5 = 80 runs.

## Files

- `run_convergence.py` — Batch runner (skip-on-exist, safe to re-run)
- `plot_convergence.py` — Thesis-grade SVG plotter
- `data/` — Raw `.npz` results (one per run)
- `figures/` — Output SVG plots
