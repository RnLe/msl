# Phase 0 Scoring Visualizer Plan

## Goals

- Provide an intuitive, web-based way to tune the five Phase 0 score weights (`w_flat`, `w_gap`, `w_parab`, `w_vg`, `w_contrast`).
- Demonstrate how curvature, derivative (group velocity) and spectral gap translate into extrema quality before the user edits weights.
- Offer copy-ready weight presets that stay numerically consistent with the Python scoring utility.

## UI Sketch

1. **Extremum Playground**
   - Plot a quadratic band approximation `f(k) = f0 + v k - 0.5 * K * k^2`.
   - Sliders: `K` (curvature) and `|v|` (first-derivative magnitude).
   - Show the computed maximum, tangent slope, and overlay two horizontal lines to visualize the "spectral gap" above and below the extremum.
   - Re-render in real time so users immediately see how flatter bands or larger slopes change the extremum quality metrics (`S_flat`, `S_gap`, `S_vg`).

2. **Scoring Weight Explorer**
   - Five sliders (one per weight) constrained to sum to 1.0.
   - Synthetic candidate gallery (3×3 grid) where each tile is a stacked bar encoding the weighted contribution of all five metrics, colored by total score.
   - Hover tooltip / legend listing raw metric values to connect the visual cues back to the scoring terms (`curvature`, `gap`, `parabolic width`, `group velocity`, `contrast`).
   - Button to copy the exact YAML snippet for the current weights.

## Data & Logic

- Use the same formulas as `research/moire_envelope/common/scoring.py` for normalization constants and default weights.
- Sample candidates are generated on the fly (static arrays) to avoid backend calls but still cover "good", "mediocre", and "bad" regimes for all five metrics.

## Deliverables

- New MDX research page that imports the React component and documents how to interpret the visuals.
- Component lives in `web/apps/website/src/components/Phase0ScoringExplorer.tsx` (tree-shake friendly, client-side only).
- README entry in this folder to remind future contributors of the relationship between the Next.js demo and Phase 0 configs.
