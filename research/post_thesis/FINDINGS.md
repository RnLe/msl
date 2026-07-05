# Post-Thesis Findings Log

## Session 2026-07-05 (evening) — strict campaign, V4-exact path
### ⚠ Supersedes the afternoon session's A1/A2 sections below

**Corrections after deep audit (user was right on all counts):**
- The **V4 exact-TM path** (`tm_operator_model="exact"`, blaze2d
  `run_commensurate_phase2.py` + `assemble_exact_tm_hamiltonian`) is the
  authoritative final pipeline — Berry-free by construction (A=None; all
  registry dependence via direct matrix-element fields), so the afternoon's
  "gauge noise dooms V4 → use V3" conclusion is **wrong** for the exact path
  (it applies only to the compact path with raw off-diagonal Berry, which the
  final runs did not use).
- The **Hungarian-matched "golden benchmark" (Mar 11) is rejected as a
  protocol**: it silently dropped 16 of 64 FDFD window modes. The final
  sprint (Mar 21–27, `studies/fdfd_convergence`) already used the strict
  protocol: sorted lanes + index-aligned compares, no matching.
- Timeline: `fdfd_solver.py` last touched Mar 27 03:43 (submission day);
  authoritative EA data = `tm_commensurate_phase2/` (exact-TM, Nb=2(+6 rem),
  Ns=64, 2×2 tiling → 4 k-points × 10 modes = 40 pooled).

### 🔑 V0 discovery — the missing X′ valley (mode-count mystery solved)

The thesis square-X comparison pooled ONE valley. But the square lattice has
two inequivalent X points (X=(½,0)·2π, X′=(0,½)·2π) folding to the same
supercell momentum:

- FDFD spectra at 2° and 4° consist of **exactly degenerate pairs**
  (median pair splitting ~1e-17 — symmetry-protected valley doubling).
- EA ladder **duplicated ×2** vs FDFD, strict common window, index-aligned:

| θ | β=θ/γ | EA(×2) vs FDFD mean rel | FDFD self-drift (same window) |
| --- | --- | --- | --- |
| 1.005° | 0.037 | **0.012 %** (vs res8) | res16 rung queued |
| 2.01° | 0.075 | **0.067 %** (vs res16) | 1.6e-4 res8→16 → **FDFD-limited** |
| 3.94° | 0.149 (marginal zone) | 0.55 % (vs res32) | 3.6e-4 → genuine EA error |

So with correct valley bookkeeping the Nb=2 exact-TM EA already matches FDFD
at the reference solver's own floor for β ≤ 0.075 — and the θ⁴-like error
growth into the marginal zone is exactly where band truncation should bite.
Remaining protocol gap: EA solved only 10 modes/k, so it under-fills the
window (68/48/58 EA modes vs 80 FDFD); strict runs will use ≥25 modes/k.

Baselines saved: `A_triple_match/strict/v0_strict_{1,2,4}deg.json`;
thesis figures regenerate bit-for-bit (`plot_x_tm_compact.py`,
`plot_x_tm_line_compare.py`).

### In flight (background)

1. Phase-1 exact extraction (reg 64, res 64): `square_x_prime` 2ret+6rem
   (legitimize the valley doubling by an actual X′ run — spectra must match
   X to solver precision), then X with **Nb ∈ {4,6,8}, n_remote=0** (the
   many-bands ladder). → `A_triple_match/strict/phase1_*/`
2. Deep FDFD rungs: 2° at 32 px/a, 1° at 16 px/a (3.3M DOF each)
   → `studies/fdfd_convergence/data_x_tm/*_fEActr.npz`

### Next session recipe

1. `strict_commensurate.py --phase1 <phase1_xp_2r6/...npz> --cases 57,1 114,1`
   → X′ vs X spectrum identity check (V1-sym PASS criterion: ≤ solver tol).
2. Same driver over the Nb ladder (n_modes 25/k) → `strict_eval.py` with both
   valleys pooled vs the deep FDFD rungs → does the 4° (and residual 2°)
   error collapse with Nb? (the central many-bands claim)
3. V2 null tests (frozen registry, monolayer limit) + chase
   `first_order_remainder` scale (34.7 at reg8 pilot vs 2796 in the final
   archive — likely spikes at isolated registry points; map it).
4. MPB lane at 4° for the triple figure.

Pilot facts: exact-TM phase-1 at reg8 converges (`all_converged: true`);
X′ pilot k0 corner verified; runners: `strict_commensurate.py`,
`strict_eval.py`, `run_fdfd_deep.py` (all in `A_triple_match/strict/`).

---

## Session 2026-07-05 (afternoon) — SUPERSEDED where marked

### A1 — Hungarian benchmark reproduction **[SUPERSEDED — protocol rejected]**

Reproduced the Mar-11 result (50/50 Hungarian, mean |Δω| 23e-6 = 0.8% BW) —
retained only as historical context; the matching protocol hides unmatched
FDFD modes and is not used going forward.

### Blaze ingredient verification ✅ (still valid)

Golden honeycomb crystal: Dirac pair at bands 0–1, ω_D = 0.2744 at
K = (1/3,2/3) frac — matches MPB-V3 exactly.

### A2 V4-compact ladder **[REINTERPRETED]**

The remote-band shifts (n_remote 0→4 moves eigenvalues by ~91e-6) remain a
valid quantification of Löwdin dressing. The "gauge noise" finding
(|A₀₁| spikes to 56 at gap closings) is real but applies to the **compact**
path only; the exact-TM path never uses the extracted Berry connection.
Datasets kept: `A_triple_match/phase1_nrem{0,4,8,16}` (honeycomb, reg 64).

### Infrastructure ✅

blaze dev rebuilt (`maturin develop --release`); `EAExtractor` renamed →
`OperatorDataExtractor` (blaze2d 99a4442, May 2026) — compat shim in
`lib/phase1_blaze_v4.py`.
