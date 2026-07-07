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

### Deep FDFD rungs + the 2° headline

CHOLMOD shift-invert (the March pattern; scipy's default SuperLU OOMs at
31 GB on the same 3.3M-DOF problem) delivered both rungs in ~8 min each:

| rung | FDFD self-drift |
| --- | --- |
| 2° res16→res32 | mean 7.0e-5, max 9.9e-5 |
| 1° res8→res16 | mean 6.8e-5, max 1.2e-4 |

**Headline: at 2° (β=0.075), reg128-EA (Γ-lane, valley-doubled) vs
FDFD-res32 = mean 5.2e-5 (0.022%), max 1.5e-4 — the EA↔FDFD residual is
now BELOW the FDFD reference's own resolution drift.** At production
settings the exact-TM EA is indistinguishable from full Maxwell within the
reference's own convergence error, inside the accuracy zone.

Protocol lesson: per-lane `n_modes` must over-cover the FDFD window —
10 modes/lane at reg128 under-fills it (each lane returns the 10 nearest σ),
which masquerades as count mismatch. Production: ≥25 modes/lane
(re-run in flight).

### Upstream bug: blaze registry sweep heap corruption at reg128

`extract_registry_sweep` (via phase1_blaze_v4, threads 12, Nb=4/rem0,
exact fields on) aborts with glibc `corrupted double-linked list` at ~22%
of a 128×128 sweep; reg64 ran clean five times. Mitigation: checkpointed
crash-resume retries + threads 8 (`run_nb_phase1_reg128.sh`). To report
upstream in blaze2d with this reproduction.

### 🔑🔑 The window-choice discovery — why "the eigenvalues looked nothing alike"

Deepening the strict protocol exposed a **level-density mismatch**: in the
March window (f≈0.241) the 4-lane EA carries ~2.5× more levels than FDFD
(whose 80-nearest-σ span pins the true density unambiguously). Envelope
eigenvectors there have a flat/white Fourier signature (≈75% weight outside
the inner quarter-zone — the random-vector value). Resolution:

- The Λ₀(registry) landscape at X spans ω ∈ [0.2260, 0.2734], mean 0.2440.
- **The March comparison window [0.2398, 0.2422] sits deep in the landscape
  INTERIOR** — envelope excitation numbers ~10³, legitimately oscillatory
  envelopes, level spacing ~1e-5, semiclassically dense.
- In such a window: (i) index-aligned residuals are meaningless (any two
  dense ladders agree to ~spacing/2 — this **retroactively downgrades the
  earlier "5.2e-5 at 2°" headline** to density-limited pseudo-agreement,
  and equally the Hungarian-era numbers); (ii) the only meaningful strict
  metric is integrated level density — and there EA and FDFD disagree
  (~2.5×, cause TBD: mass/η-scaling of the envelope DOS or folding
  bookkeeping).
- The EA is a band-edge (k·p at X) expansion: its clean, testable regime is
  the **spectral bottom** near min Λ₀ = 0.2260 — sparse levels, smooth
  envelopes, count-exact mode identity possible. The March sprint pinned
  its σ to the landscape MEAN (0.241 ≈ ⟨Λ₀⟩) — the semiclassical worst
  case. This is the structural reason the thesis-era ladders "looked
  nothing alike."

**Campaign redirected to the band-edge window** (σ_ω = 0.2270): FDFD
res16/32 references + EA 4-lane bottom solves in flight. The dense-window
Nb ladder was stopped (question malformed); the Nb ladder will rerun at
the bottom window where truncation physics is actually testable.

### Band-edge (σ_ω=0.2270) results — the countable regime

FDFD bottom refs (2°, res16/32): lowest mode 0.226459, self-drift mean
5.0e-5. All FDFD levels are exact pairs (1e-14). EA 4-lane rungs:

| rung | EA spectral edge | edge error vs FDFD | density (same span, no doubling) |
| --- | --- | --- | --- |
| Nb2+6rem | 0.226416 | −4.3e-5 (**within FDFD drift**) | ~1.25× (40 vs 32) |
| Nb4 rem0 | 0.225744 | −7.2e-4 | ~1.4× |
| Nb6 rem0 | 0.225161 | −1.3e-3 | (span shifted) |

Two structural findings:

1. **Valley doubling retracted.** Cluster table at the edge: FDFD first
   cluster ×4 ↔ EA pooled-4-lane first cluster ×4 (offset −4.9e-5), with
   NO doubling. The 4-lane pool already represents the full supercell
   content; the earlier "×2 valley" match in the dense window was a
   density coincidence (2.5×/2 ≈ 1.25× residual — see below). The X↔X′
   lane-level identity (grid-floor equal) is the mechanism: the lanes of
   one carrier already span both valleys' folded content.
2. **Consistent ~1.25× EA density excess** in BOTH windows (edge: 40 vs
   32; interior: 100 vs 80). A clean rational factor — the remaining
   folding/multiplicity bookkeeping question. Root-causing needs the
   explicit (57,1) supercell-momentum ↔ moiré-k mapping (pencil-and-paper
   + small script; next session's first task).
3. **Naive many-bands fails in the exact-TM operator**: the spectral edge
   softens BELOW truth as Nb grows (−4e-5 → −7e-4 → −1.3e-3 for
   Nb 2→4→6). The truncated-η² exact operator is not variational; adding
   explicit bands over-binds. Nb=2(+6rem Löwdin) gives the most accurate
   edge. The professor's rule needs the compensating higher-order terms
   before large retained windows pay off — a concrete theory task.

**What stands as genuinely validated:** the EA's spectral EDGE at
production settings is exact within the FDFD reference's own convergence
error (−4.3e-5 vs drift 5.0e-5) at 2°, β=0.075 — a strict, matching-free,
count-anchored statement.

*(open: 1.25× multiplicity bookkeeping; Nb8 edge rung (rerunning);
1° edge window; density root cause)*

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

---

## Session 2026-07-07 — Triangle protocol: the MPB leg, and the deliverable figure

**Goal (user directive):** one clean figure + data that compares MPB/FDFD/EA
faithfully (eigenvalue stripes), unambiguously recreating the same
eigenvalues, with every run parameter stated.

### Discovery: the March MPB lane was at the wrong supercell momentum

`data_x_tm_mpb_corrected/mpb_tm_x_4deg_res64_20bands.npz` (Mar 22) ran at
supercell fractional k = (1/2, 0). But the fold of the monolayer X point onto
the (m,1) supercell is k = (1/2, 1/2): the Bloch phases are
e^{iQ_X·L1} = e^{iπm} = −1 and e^{iQ_X·L2} = e^{−iπn} = −1. Every March FDFD
lane ran at Q_X = (π,0) ≡ (1/2,1/2). **The March MPB↔FDFD comparison compared
different supercell momenta** — one more structural reason the thesis-era
lanes never lined up, independent of the window-choice problem.

Additionally, MPB's window is structurally disjoint from the EA's: MPB has no
shift-invert and computes the lowest N supercell bands (4°: 20 bands span
f = 0.0102–0.0507 c/a). Reaching the band-edge window f ≈ 0.2265 at ≤4° would
need ~600+ bands — infeasible. **No triple overlap window exists at in-zone
angles.** The rigorous comparison is therefore a triangle:

1. **MPB ↔ FDFD** (both exact): same (29,1) cell, same k=(1/2,0) — FDFD rerun
   at MPB's k (`run_fdfd_mpbk.py`), lowest 20 modes, index-aligned from
   absolute mode 1 (zero selection ambiguity).
2. **FDFD ↔ EA**: band-edge window at 2° (in-zone β=0.075), strict
   edge-anchored ladders (session-4 protocol).

### Triangle leg 1 result: MPB ≡ FDFD to sub-ppm-of-window

FDFD at MPB's k, (29,1), lowest 20 modes, vs MPB res64/cell:

| FDFD px | mean abs Δf (c/a) | max abs Δf |
|---|---|---|
| 16 | (lowest mode already 0.010185 = MPB to 6 digits) | |
| 32 | 3.6e-7 | 9.1e-7 |

Degenerate pairs reproduced exactly. The FDFD reference is hereby certified
against an independent exact method at the 1e-6 c/a level (px64 leg pending).

### Nb ladder at the 2° edge re-verified from disk — NON-monotonic

Per-lane modes.json re-read (all four rungs, lanes Γ_m/X1_m/X2_m/M_m):
Nb2+6rem pooled edge 0.226416 (Δ = −4.3e-5), Nb4 0.225744 (−7.1e-4),
Nb6 0.225161 (−1.3e-3), **Nb8 partial lanes (Γ 0.226427, X1 0.226406) come
BACK to ≈ −4e-5**. Session-4's "monotonic Nb-softening" is amended:
the η²-truncated exact operator is non-variational and the edge error is
non-monotonic in the retained-band count; Nb2+6rem (Löwdin) and Nb8 are the
accurate rungs, Nb4/Nb6 truncate mid-multiplet. Theory note still owed.

### New lanes this session

- `run_fdfd_mpbk.py`: FDFD at MPB's k=(1/2,0), (29,1), σ_ω=0.008, 30 modes,
  px 16/32/64 (px64 = 3.44M DOF, enabled by the RAM raise to 47 GB).
- `run_fdfd_bottom.py` extended: 4° (29,1) bottom refs px16/32 at Q_X, and a
  2° px48 rung (7.5M DOF) to sharpen the FDFD drift bar.
- EA 4° band-edge lanes: `phase2_x_2r6_BOTTOM_m29` (Nb2+6rem, reg128/Ns128,
  4-lane fold, 15 modes/lane, target 0.2270) — landed: 60 modes,
  f = [0.224675, 0.229535], t = 432 s. dom_band(0) ≈ 62–64% (strong band-1
  mixing at 4°, consistent with marginal zone β = 0.147).
- Nb8 2° band-edge lanes resumed via new `--resume` flag in
  `strict_commensurate.py` (skips k-lanes with existing modes.json).

### Deliverable

`strict_triple_figure.py` → `fig_strict_triple.{pdf,png}` +
`strict_triple_data.json`: panels A (MPB↔FDFD stripes + per-mode residual),
B (2° edge stripes FDFD/EA), C (4° edge stripes), D (Nb ladder vs FDFD drift
band), E (edge cluster table), parameter footer with polarization, contrast,
r/a, k-points, grids, solver settings. Regenerates from disk; missing lanes
reported, never faked.

### DISCOVERY — the (m,1) supercell is a 2× CENTERED cell; exact valley bookkeeping

Hunting the universal exact ×2 degeneracy of the FDFD/MPB supercell spectra
(present at BOTH supercell k=(1/2,1/2) and k=(1/2,0), so not explainable by a
little-group 2D irrep alone) led to an exact structural fact:

**τ = (L1+L2)/2 is a lattice vector of BOTH layers.** For (m,n)=(29,1):
τ=(14,15) = 15·Re1 + 14·Re2 exactly (integer arithmetic via the
commensuration identities cosθ=(m²−n²)/(m²+n²), sinθ=2mn/(m²+n²)); for
(57,1): τ=(28,29). Verified numerically: ε-map autocorrelation at (1/2,1/2)
equals the autocorrelation peak to machine precision at both angles. This
holds whenever m,n are both odd.

Consequences:
1. Every FDFD/MPB supercell run of the campaign (and the thesis) used a
   **2× non-primitive centered cell**. The primitive commensurate cell has
   area (m²+n²)/2. All exact ×2 pairs = two primitive momenta folding onto
   one supercell momentum (FUTURE OPTIMIZATION: primitive-cell FDFD at half
   DOF, valley-resolved references).
2. At Q_X: monolayer X folds to primitive k=(0,1/2), X′ to (1/2,0) — the
   two folded primitive momenta ARE the two valleys, exactly degenerate by
   C4 (point group of the ε map about the twist center = full C4v,
   symmorphic; FFT symmetry search found no glide).
3. EA lane bookkeeping, exact form: envelope periodicity over the primitive
   cell (P1 ≈ a_m1−a_m2, P2 ≈ a_m1+a_m2) selects moiré lanes
   {Γ_m, M_m} ↔ valley X and {X1_m, X2_m} ↔ valley X′. **Verified**: the
   two lane-pair multisets agree to mean 2.2e-5 / max 9.1e-5 (= EA grid
   floor) at 2°, Nb2+6rem. The 4-lane pool = full two-valley content —
   the session-4 doubling retraction now has its exact mechanism.
4. Caveat kept honest: the CML is congruent to 2×(moiré lattice) but rotated
   by ~θ/2 — NOT a sublattice (L1 ∉ Λ_m exactly). The lane fold is exact
   only at EA order; the O(η) frame skew is part of the EA's error budget.

### The density excess is LOCALIZED, not a multiplicity factor

Level-counting functions N(f) at the 2° edge (FDFD px48 vs EA Nb2+6rem
pool): FDFD is cluster-stepped with a clean plateau (N=4) across
0.2265–0.2267; the EA staircase is nearly linear and climbs through that
FDFD spectral gap (~11 EA levels inside it). So the former "~1.25×" excess
(window-dependent; ×1.47 in [0.2262,0.2275]) is **envelope-ladder
compression** — EA minibands squeezed and partially filling true gaps — the
same η²-truncation physics as the edge softening, NOT a folding/multiplicity
error (that bookkeeping is now exact, see above). Next lever: the
compression should shrink with better η² closure (Nb8 pool / higher-order
terms), measurable directly as N(f) convergence.

### 4° deep-edge FDFD: no spectral edge exists at 4°

DEEP rungs (σ_ω=0.2225, 60 modes, px16/32/48): folded background modes
continue down to at least 0.2163 with cluster gaps ~1–2e-3 — the clean
"edge with a gap below" is a 2° feature (at Q_X). Panel C therefore
compares density/cluster positions in a ball-complete window; per-level
association at β=0.147 is impossible (cluster shifts ≳ cluster gaps),
which IS the accuracy-zone prediction made visible.

### 2° edge claim sharpened by the px48 rung (7.5M DOF, post-RAM-raise)

FDFD edge ladder: px16 0.226438 → px32 0.226459 → px48 0.226472;
Richardson f∞ = 0.226482, remaining drift 1.3e-5. EA(Nb2+6rem) edge
0.226416 → **Δ = −5.6e-5 (vs px48) / −6.6e-5 (vs f∞), now RESOLVED as a
real EA deficit** (≈5× the combined numerical floors; 0.029% relative) —
supersedes session-4's "within FDFD drift" (which was true vs the res16/32
pair). Magnitude consistent with the expected η³/higher-order residual at
β=0.075. The MPB↔FDFD leg closed at px64: mean |Δf| = 2.9e-7, max 7.4e-7,
FDFD 32→64 self-drift 1.7e-7 (i.e. MPB and FDFD agree at FDFD's own
convergence level).

### Final Nb ladder at the 2° edge (all rungs pooled, vs FDFD px48 = 0.226472)

| rung | pooled edge | Δ edge |
|---|---|---|
| Nb2+6rem | 0.226416 | −5.6e-5 |
| Nb4 | 0.225744 | −7.3e-4 |
| Nb6 | 0.225161 | −1.3e-3 |
| Nb8 | 0.226381 | −9.1e-5 |

V-shaped: the truncation is worst when the retained set cuts mid-multiplet
(Nb4/6) and recovers at Nb8; Löwdin dressing (Nb2+6rem) is the best rung.
Nb8 M_m-lane mode characters show dom_band down to ~23–25% (bands 1–2
dominated) in the window — the many-band mixing is physical, but the η²
closure isn't yet good enough to beat the dressed 2-band model.

**Deliverable frozen this session**: `fig_strict_triple.{pdf,png}` +
`strict_triple_data.json` (all lanes, all parameters, regenerable via
`strict_triple_figure.py`).
