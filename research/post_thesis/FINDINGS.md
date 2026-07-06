# Post-Thesis Findings Log

## Session 2026-07-06 — identity discrimination & the EA grid floor

### Valley identity: three-way discrimination (m57/m114, Γ-lane, Nb=2+6rem)

| Comparison | isolates | mean \|Δf\| @2° | @1° |
| --- | --- | --- | --- |
| X_today vs X′_today (same code+build) | pure valley physics + grid | 3.0e-4 | 8.2e-5 |
| X_today vs X_archive (same valley) | Mar-22→Mar-29 code + blaze build drift | 3.4e-4 | 1.6e-4 |

FDFD says the valley degeneracy is EXACT (pair splitting ~1e-17), so the
X↔X′ difference is **not physics** — together with the same-scale code-drift
term it pins the **EA pipeline discretization floor at reg64/Ns64 ≈ 1–3×10⁻⁴**
— the same magnitude as the EA↔FDFD residual at 2°. The current accuracy
bottleneck is the EA grid, not the operator physics.

Consequences for the strict campaign:
1. Valley pooling by duplication is justified physically (FDFD-exact
   degeneracy) even though the discretized EA splits it at the grid floor.
2. Production settings move to reg128/Ns128 (resolution rung in progress
   using the March 128-registry exact archive).
3. The Mar-29 phase2 refactor replaced the Mar-22 4-k folding with
   `supercell_tiling`; `strict_commensurate.py --k-fold` restores the
   archive protocol (Γ_m/X1_m/X2_m/M_m explicit solves).

### V2 frozen-registry null test — v1 construction invalid (my design flaw)

Naively freezing all Phase-1 fields at one registry keeps the ∂_R-derived
couplings (slow-gradient ε terms / direct_b / γ pieces) at nonzero constant
values — these implicitly carry the moiré-map Jacobian, so the "frozen"
operator does not correspond to a uniform crystal. Result: modes at
f≈0.1131 vs the local band-0 value 0.2722 (spurious branch from the
inconsistent constant drift couplings), BW 5.7e-4. **Not** a pipeline
failure. A correct null test must zero every ∂_R-derived field and keep
only Λ/metric/velocity constant — deferred; the empirical EA↔FDFD ladder
is the stronger assembly check regardless.
Artifacts: `strict/v2_frozen/` (kept as the cautionary example).

### OOM forensics

The reg-128 exact archive npz is 13 GB, of which 8.6 GB hermitized
eigenvectors + 8×0.54 GB real-space ε/ρ derivative grids are **loaded and
interpolated by phase-2 but never consumed by the assembly** (audit-tool
payload). This is what OOM-killed the deep-FDFD run last night (30 GB RSS
kill in dmesg) and two resolution-rung attempts (20 GB). Fix: zip-level slim
copy without the 9 dead members (13 GB → 149 MB),
`strict/phase1_x_reg128_slim/`. Future phase-1 runs keep
`archive_exact_tm_hermitized_eigenvectors: false`.

### 🔑 Resolution-rung verdict — reg128/Ns128 is the production setting

Solving the March reg-128 exact archive (slimmed) at Ns=128, m57 Γ-lane:

| EA setting | vs FDFD res16, in-window, valley-doubled, Γ-lane | 
| --- | --- |
| reg64 / Ns64 | mean 1.19e-3, max 2.8e-3 (lane-resolved; the pooled-4k number looked better by averaging) |
| **reg128 / Ns128** | **mean 7.1e-5, max 2.3e-4** — 17× better, at the FDFD-res16 reference's own error scale |

Grid shift reg64→reg128 is ~1e-3 — larger than every effect studied at
reg64. All further ladder work runs at reg128/Ns128; the Nb∈{4,6,8}
phase-1 sweeps are being repeated at registry 128 (cheap: blaze sweeps
~80 s at reg64, ~6 min at reg128). The remaining EA↔FDFD gap is now
FDFD-reference-limited → deep rungs (2°@32px, 1°@16px) in flight.

*(pending: deep-FDFD rungs, reg128 Nb ladder, final strict table)*

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
