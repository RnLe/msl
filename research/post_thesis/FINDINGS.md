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

---

## Session 2026-07-07 (continued) — Faithfulness audit: the X-manifold dissolves

**User directive:** the EA↔FDFD comparison must be EXACT (eigenvalue-faithful)
or it cannot be trusted. Audit result: it was not — and the reason is deeper
than truncation error. Chain of evidence:

1. **Ladder mismatch at 2°**: FDFD shows a real spectral gap
   [0.22649, 0.22672] (verified by an in-gap shift-invert probe); EA
   (Nb2+6rem AND Nb8) puts ~11 levels inside it. EA spacing above the edge is
   ~3× too small. Not a resolvent artifact (rem=0 rung identical).
2. **Session-4 "edge" was a capture artifact**: the same probe found FDFD
   levels at 0.226232 — BELOW the supposed edge 0.226459. The "clean gap
   below the edge" was the boundary of a 40-mode shift-invert ball.
   **RETRACTED: the 2° edge-agreement claims (−4.3e-5 / −5.6e-5) compared
   EA against levels that are not X-manifold states at all (see 3).**
3. **X-weight classification** (`run_fdfd_xweight.py`: per-eigenvector
   Fourier weight near the X-star, reduced to the monolayer BZ — strict,
   matching-free labeling): EVERY FDFD mode in [0.2255, 0.2280] at Q_X has
   w_X ≈ 0 (max 0.013; total X weight across 80 modes ≈ 0.05 states).
   Dominant carriers sit at generic k on the band-0 iso-frequency contour.
   Same at 0.269 (band-1-at-X window): all background.
4. **Mechanism**: band 0 of the local two-rod crystal is a connected band
   (ω→0 at Γ). At any f in the "X window", other registry regions propagate
   (local AA gap ≈ [0.24,0.29] vs AB gap shifted up by the a/√2 morph → no
   registry-common gap at ε=8.9, r=0.2). The EA's discrete X-patch envelope
   states are resonances embedded in that continuum and dissolve — no
   individual FDFD eigenvalue corresponds to them. **Eigenvalue-exact
   EA↔FDFD agreement is impossible IN PRINCIPLE for the thesis's central
   case (square TM, band 0/1 at X, ε=8.9, r/a=0.2).** This — not matching
   protocol, not window choice — is the root cause of the thesis-era
   "eigenvalues look nothing alike".

**Requirement for a faithful candidate** (new design criterion): the target
manifold must sit inside a REGISTRY-COMMON gap:
max_s max_k ω_n(k;s) < f_window < min_s min_k ω_{n+1}(k;s). Then every
supercell state in the window is a moiré envelope state (nothing to
hybridize with), counts are exact, and EA↔FDFD can match level-by-level.
Search running: `scan_common_gap.py` (MPB, two-rod crystal, full BZ × full
registry torus, (ε, r) scan).

Side finding kept: the (113,1) 1.014° EA lanes (reg128, 40/lane, 380 s)
are on disk (`phase2_x_2r6_BOTTOM_m113`) — edge 0.226206, per-lane spacing
~2e-5 — for later use once reinterpreted against classified FDFD.

### THE PROPER CANDIDATE (found by design): asymmetric bilayer, band-1 gap edge

`scan_common_gap.py`: NO (ε, r) equal-rod square-TM bilayer has a
registry-common gap (the AA→AB √2-morph is too violent; thesis case
overlap 0.055). `scan_common_gap2.py` + `refine_candidate.py`: making
layer 2 a weak perturber fixes it by continuity:

**Candidate: square TM, layer-1 rods ε=8.9 r=0.20, layer-2 rods ε=8.9
r=0.10.** Registry-common gap [0.32251, 0.36608] (width 0.0436, 16×16 k,
9×9 s, res 48). Band-1 landscape: global minimum 0.36608 EXACTLY at the
X′ point at s=(0,½) (X valley: s=(½,0) by symmetry) — the manifold bottom
is X-star-carried, k·p at X valid. Λ₁(X-star; s) modulation depth ≈ 0.072
(strong moiré potential). Band-2 headroom 0.121 → remote-truncation zone
β_rem: 2°→0.124, 1°→0.062. Retained {0,1} + 6 Löwdin remotes unchanged;
band-0 fully retained so the 0↔1 coupling is exact in the model.

Infrastructure: `base_atoms` already supports per-atom radii (no blaze
changes!); `supercell_asym.py` (per-layer radii + optional primitive cell);
`strict_commensurate.py --target-band`; X-weight classifier unchanged.

### PRE-REGISTERED exactness criteria (fixed BEFORE any comparison is run)

Window: [manifold bottom − margin, bottom + W], W sized for ≥ 20 FDFD
levels, at Q_X on the (57,1) centered cell (then (113,1) if truncation
visible). The comparison PASSES as **exact** iff ALL of:
1. Enumeration-complete on both sides (σ-bracketing verified below and
   above the window on both FDFD and EA).
2. Every FDFD state in-window has X-star weight w_X + w_X′ > 0.5
   (isolation confirmed; else FAIL with diagnosis).
3. Mode counts EXACTLY equal in-window (EA = pooled 4 lanes, no doubling).
4. Index-aligned residuals: mean |Δf| ≤ max(3×combined numerical floor,
   3e-5 c/a) AND max |Δf| ≤ 1e-4 c/a, floors measured independently
   (FDFD: Richardson residual of the px ladder; EA: reg64↔reg128 shift +
   C4 lane-pairing spread).
5. N(f) staircases: no insertion/deletion anywhere in-window (counts match
   level-by-level, not just in total).
Anything less is reported as its measured value, not "exact".

### THREE OPERATOR-LEVEL ROOT CAUSES FOUND AND FIXED (frozen-symbol audit)

The asym candidate's isolated manifold made the EA operator's diseases
individually visible for the first time. Diagnostic tool: freeze all
phase-1 fields at one registry (s_min) — the assembled operator must then
reproduce the local band dispersion λ₁(X+q; s_min) exactly (plane-wave
symbol). Audit chain:

**(1) The ∂_Rε-derived exact-TM fields are grid-divergent artifacts.**
γ₁ ("first_order_remainder"), γ₂ ("direct_gamma2") and direct_b are
finite-difference derivatives of the DISCRETIZED rod boundary: |γ₁| ~
Δε·res ≈ 450, |γ₂| ~ Δε·res² ≈ 2.5e4–3.8e4 (both archives! the March
red-flag "remainder_abs_max ≈ 2796" was this). η²γ₂ injects a spurious
pseudo-potential of ±19–31 λ-units — larger than the entire physical
landscape (±0.5) — into EVERY exact-TM spectrum ever assembled. FIX:
`core_only` assembly mode (keep Λ + v·Π + direct_metric kinetic + v-only
Löwdin; all boundary-safe matrix elements). Frozen symbol with the core:
reproduces MPB dispersion at ±g₁ to 2.7e-4 λ. The true (properly
regularized) γ-terms are physically O(η)-small; computing them for hard
rods needs surface-integral formulas, not grid FD — future blaze work.

**(2) band_lo>0 archives: matrices are retained-first ordered.** The Rust
extractor emits (n_total×n_total) exact-TM matrices as [retained bands,
remotes]; the assembly slices them with absolute band_lo-based indices —
correct for band_lo=0 (all past runs), silently wrong for band-1 targeting.
Discriminated empirically (retained-first symbol fits MPB 2× better:
rms 0.014 vs 0.029 λ). FIX: permutation to absolute order in the npz
loader (+ MSL_BAND_LO env override; band_lo-aware λ_ref in the runner).

**(3) FERMION DOUBLING in the envelope kinetic term.** The exact-TM
kinetic was assembled as Π@Π — the square of the first-derivative FD
stencil, whose discrete symbol vanishes at the Nyquist momentum as well as
at 0. The operator therefore supports FOUR interleaved copies of the whole
envelope spectrum (verified: frozen-coefficient operator gave every level
exactly ×4). With s-dependent coefficients the copies mix — THE mechanism
behind the ~4–6× overdense, structure-washed, softened EA ladders seen at
both candidates (the "envelope-ladder compression" of session 5 and, in
retrospect, a large part of why thesis-era EA spectra never looked like
FDFD). FIX: true second-derivative stencils for the diagonal kinetic
(Π_a² = −L_a + 2(2πk_a)(−iD_a) + (2πk_a)²); doublers pushed +13 λ out of
the window. Frozen ladder now matches the plane-wave symbol with correct
degeneracies (×1 ground, ×2 pairs).

**Post-fix pilot (2°, reg64, Nb=1 band-1, clean core):** in-window density
now ≈ correct (32 EA vs 28 FDFD states in [0.370, 0.3763]); first clusters
align to +2.5e-4 / +4.7e-4. Remaining defects — a sub-gap spurious branch
(0.3658–0.3685) and one interloper quadruplet — are quantitatively
consistent with the measured quartic k·p truncation along the soft g₂
direction (frozen symbol: −0.028 λ at |q|=g, growing to −0.07 at 2g):
genuine asymptotic-order error, predicted to shrink ~×16 (branch depth) /
×4 (per-level) at 1°. Production run at (113,1) will measure the η-scaling
directly.

---

## Session 2026-07-07 (cont.) — Operator audit on the isolated manifold: 4 bugs, and a strong-coupling wall

The asym candidate's ISOLATED manifold (verified: all in-window FDFD states
X/X'-carried, w=0.60-0.69, band-2 headroom 0.12) finally let us audit the EA
operator against a clean target. Diagnostic = the frozen-registry symbol
(constant fields → operator must reproduce the local band dispersion
λ₁(X+q;s_min)) plus the "ground-above-floor" invariant (an envelope ground
state cannot lie below min Λ).

**BUG #4 (found+fixed): the Löwdin remote term was non-Hermitian in a way
global-hermitization can't repair.** Assembly built
`H -= left @ res @ right` with `right = v_qp·Π_p` an INDEPENDENT operator,
not `left†`. The correct 2nd-order downfolding is
`H_PQ (E−H_QQ)⁻¹ H_QP` with `H_QP=H_PQ†`, i.e. `left @ res @ left†`
(manifestly Hermitian, res real-diagonal). The old `right` differs from
`left†` by `[Π,v]` (a velocity-gradient commutator) — invisible at a frozen
registry, but once fields vary in space it injects a spurious attractive
potential. Symptom: the exact-TM envelope ground sank ~0.23λ BELOW the
potential floor (term-toggle: kinetic-only ground −0.976 above floor
−1.100 ✓; +old-Löwdin −1.208 ✗). Fix `left@res@left†` restores
ground-above-floor. Frozen dispersion: kinetic-only rms 0.113 (isotropic,
WRONG — the mass anisotropy lives entirely in the Löwdin), kin+Löwdin
0.030 — so the Hermitian Löwdin is both correct AND necessary.

**The strong-coupling wall (unresolved; the current frontier).** With all
four fixes and Nb=2 (bands {0,1}, exact 0↔1 coupling), the band EDGE matches
beautifully — EA bottom 0.369989 vs FDFD 0.370047 = **6×10⁻⁵** — but the
full ladder is ~5-9× OVER-DENSE (EA 137 vs FDFD 16 states/cell in
[0.370,0.383]; FDFD verified complete via a 300-mode re-solve). Two
distinct over-density channels, both diagnosed:
  1. **Band-0 k·p doublers** (Nb=2): the over-dense states are 69% Nyquist-
     weight and 75% band-0 — band-0's parabola, expanded around X, is
     extrapolated on the Ns=64 moiré grid to momenta up to ~32·g_moiré where
     k·p is meaningless, folding spurious band-0 states into the band-1
     window. A sharp momentum cutoff removes them but distorts the physical
     high-q content (bottom shifts to 0.351) — too crude.
  2. **Intrinsic over-counting** (band_lo=1, band 0 only via smooth Löwdin —
     states are now Nyquist-clean, wt 0.03): STILL ~5× over-dense. Root
     cause: the moiré potential depth Λ₁(X;s) spans 2.1λ while the η²-small
     kinetic quantum is ~0.024λ (ratio ~90). The operator is potential-
     dominated (nearly diagonal), so its eigenvalues pile at grid-sampled
     Λ(s) values → far more low minibands than FDFD's true count. FDFD, with
     the SAME potential, shows few — so the EA k·p envelope, truncated at
     2nd order in q with a real-space grid, does not reproduce the true
     miniband count in this strong-moiré regime.

Interpretation: eigenvalue-EXACT ladder matching needs either (a) a
momentum-space k·p with an explicit validity cutoff |q|<q_c (band-limited
envelope), sized to the actual moiré folding rather than the registry grid;
or (b) a shallower-moiré candidate (weaker layer-2 perturber, r₂≈0.05→depth
0.5λ) to enter the weak-coupling regime where 2nd-order k·p is quantitative;
or (c) higher-order-in-q terms. The band EDGE (the physically robust,
matching-free quantity) IS reproduced to 6×10⁻⁵ — that stands. The full
faithful ladder does not yet; the obstruction is now precisely characterized
rather than mysterious.

**Fixes committed this session** (all real correctness improvements, verified
by the frozen symbol + ground-above-floor invariant): (i) diagonal kinetic
via true 2nd-derivative stencils (fermion-doubling of Π@Π removed);
(ii) retained-first→absolute matrix permutation for band_lo>0; (iii) γ/direct_b
`core_only` toggle (dR-ε grid divergence); (iv) Hermitian Löwdin
`left@res@left†`. Tooling: `assemble_checkpoints.py` (subprocess workaround
for the blaze bulk-load stack-smash), `supercell_asym.py`, `--target-band`,
`MSL_BAND_LO`, per-run σ in the bottom FDFD runner, `run_fdfd_xweight.py`
(X-star classifier, empty-lattice-calibrated momentum map).

### The wall localized: kinetic is EXACT, the strong potential is the problem

Free-particle limit (Λ≡const, drift & Löwdin off, uniform direct_metric):
the assembled EA spectrum matches the analytic folded square-lattice
free-particle ladder E=(dm·metric)(2πn)² to **1.9×10⁻⁶** with exact
multiplicities (1,4,4,4,...). So the kinetic stencil, the metric/η² scaling,
and the plane-wave basis are all CORRECT — the operator core is sound.

The over-density is therefore purely a potential-coupling effect, and it has
the WRONG SIGN relative to truth. Free-particle count in the window ≈44
states; FDFD (with the moiré potential) shows 16 — the true potential pushes
states OUT (up) of the window; EA shows 137 — the modelled potential pulls
states IN. A single-/few-band envelope with Λ=λ₁(X;s) as a static potential
of depth 2.4λ (=0.076 in f, a real single band, no crossing) OVER-BINDS
because that "depth" is ~90× the η²-kinetic quantum (0.028λ): the two-scale
EA's assumption that the O(1) registry modulation is a gentle slow potential
fails when the modulation rivals the inter-band gaps. This is the intrinsic
reason the thesis EA↔FDFD spectra "looked nothing alike" — not a matching
protocol or a code bug, but small-angle moiré being a STRONG-coupling problem
that 2nd-order-in-q, few-band k·p cannot render as an eigenvalue-exact
static-potential envelope equation.

CONSEQUENCE for the hero plot: the band EDGE (matching-free, physically
robust) is reproduced to 6×10⁻⁵ and is a legitimate, honest result. The
full eigenvalue-exact LADDER is not achievable in this formulation for this
regime. Genuine routes (each a real research step, none a quick tweak):
(a) momentum-space k·p with a validity cutoff |q|<q_c and many more retained
bands so the "potential" is diagonalised, not perturbed; (b) a much weaker
moiré modulation (thin layer-2 perturber and/or a near-degenerate band pair
where the registry modulation is genuinely small); (c) a genuinely flat
target band so depth≪gap. The campaign's rigorous negative result — WHY the
naive comparison fails, quantified — is itself the main scientific output.

### RESOLVED: momentum-space k·p reproduces the FDFD moiré manifold

Full derivation + results in `A_triple_match/strict/STRONG_COUPLING_ANALYSIS.md`.
Replacing the real-space parabolic k·p (which over-binds ×5–9 in the strong-
modulation regime) with a Bistritzer–MacDonald-style momentum-space model —
diagonal = EXACT local band-1 dispersion `E_ref(k₀+k+G)` (MPB), off-diagonal =
`Ṽ(ΔG)=FFT_s[Λ₁(X;s)]`, plane-wave cutoff, 4 folded moiré lanes — **eliminates
the over-binding**:

| θ | FDFD X-manifold (w_X>0.5) | EA count | edge offset | de-trended shape resid | span ratio |
|---|---|---|---|---|---|
| 2.01° | 24 (×4 quads) | 24 | +2.7e-3 | mean 8.2e-4, max 2.1e-3 | 1.11 |
| 1.01° | 88 (×4 quads) | 88 | +1.8e-3 | mean 1.4e-3, max 2.6e-3 | 1.25 |

Structure (fourfold valley×C4 clusters, N(f) ordering) reproduced. Residual =
a rigid edge offset scaling `∝ η^{0.6–0.75}` (→0 as θ→0; separable-approx
under-binding) + an ~8e-4 miniband-shape residual at 2° (≈ FDFD's own px16
drift 6e-4). NOT yet eigenvalue-exact to the FDFD floor (η-order offset +
10–25% bandwidth overestimate), but the qualitative failure is now a
quantitative agreement with a provably converging error. Also key: only 24 of
64 (2°) / 88 of 120 (1°) FDFD in-window states are X-band-manifold; the rest
are band-0/background leakage the single-band EA correctly excludes — the
"under-density" vanished once FDFD was filtered by X-star weight.
Deliverable: `fig_momentum_hero.{pdf,png}` + `momentum_hero_data.json`.
Next (incremental on this scaffold): k-resolved coupling Ṽ(ΔG;k+G,k+G') to
remove the offset; 2-band momentum model to tighten the bandwidth.

### Exact Galerkin continuum model + regime map (STRONG_COUPLING_ANALYSIS §7–8)

Pushed toward eigenvalue-exactness. New method: **Galerkin (Rayleigh–Ritz)
projection of the true supercell TM operator onto reference-Bloch states** at
the folded moiré momenta — `H c = λ S c`, `H=⟨E|−∇²|E⟩` (Q_X-shifted kinetic
on the periodic parts), `S=⟨E|ε_bl|E⟩`, exact Fourier field construction,
canonical orthogonalization. Variational ⇒ converges to FDFD; contains the
ε⁻¹-weighted Bloch form factors + inter-band coupling that the §5 heuristic
drops. Code: `galerkin_moire.py`.

**Validated (7,1) θ=16.3° vs FDFD** (bottom-12, index-aligned): mean|Δf|
3.7e-3→1.8e-3→1.1e-3→8.2e-4 over N_b=2→8, monotone — the exact method is
correct. Sanity gates pass (single-momentum+ref-ε → local bands to 1e-4;
free-particle exact).

**Small-angle regime finding:** at (57,1) θ=2° the single-reference basis
under-populates the 24-state manifold (mean-registry ref: 4/24; well-bottom
ref: 2/24). The manifold modes have **registry-varying Bloch character** that
a fixed frame cannot span with few bands — eigenvalue-exactness there needs a
**registry-adapted** (multi-frame) basis, the momentum-space analogue of the
thesis EA's `u_n(r;R)`+Berry construction. Not curable by reference choice.

**Regime map (§8):** two HARD walls — (1) no registry-common gap ⇒ dissolution
(thesis case, impossible at any N_b); (2) β=θ/γ≳1 ⇒ two-scale breakdown
(Galerkin converges but needs ~full basis). Inside the valid region, the SOFT
cost is V/E_kin: weak modulation ⇒ few-band single-ref exact (honeycomb-K
Λ₀₁≡0, shallow candidates); strong modulation (asym 2°, V/E_kin~86) ⇒
registry-adapted frames required. **Verdict:** eigenvalue-exactness is
achievable (variational convergence proven); the cost is angle- and
strength-dependent per the map. The §5 momentum model (24/24, 8e-4 shape,
+2.7e-3→0 offset) sits one controlled approximation from the exact ladder;
the two closing routes are multi-reference Galerkin (strong) or k-resolved
form-factor coupling (weak). Deliverable: `fig_exact_model.{pdf,png}`,
`exact_model_figure.py`.

### Registry-adapted Galerkin: convergence + the position-locking plateau (§9)

Built the memory-robust reciprocal-space Galerkin engine (`galerkin_recip.py`:
sparse plane-wave coeffs + per-basis FFT-convolution ε-coupling; <1 GB at px16,
aliasing-free, checkpointed; validated vs real-space on (7,1) — 2.6× closer to
FDFD, no undersampling) and the registry-adapted multi-reference basis
(`galerkin_multiref.py`). (57,1) 2° vs FDFD X-manifold (24 states, bottom
0.37005), in-window [0.365,0.385]:

| basis | count | bottom | Δ |
|---|---|---|---|
| single ref | 1 | 0.36824 | (non-manifold) |
| 9 frames | 10 | 0.37680 | +6.8e-3 |
| 16 frames | 10 | 0.37654 | +6.5e-3 (plateau) |
| 4 frames, G_c=6 | 10 | 0.37749 | +7.4e-3 |

**Registry-adaptation works (1→10 states with more frames) but PLATEAUS** at
+6–7e-3 above the FDFD ground state: 9→16 frames barely moves (S-rank saturates
→ redundant frames), and G_c 4→6 gains only ~2e-3. **Fundamental reason:** the
true state's local Bloch character is *locked to the moiré position* via s(R);
the momentum-space basis e^{ip·r}u_n(s_k) carries a *fixed* registry and cannot
synthesize that position-registry correlation from a coarse frame grid — which
is exactly why the thesis EA uses the real-space *continuously* registry-adapted
frame u_n(r;R)+Berry. **Verdict:** the strong-coupling small-angle exact ladder
is reachable in principle (Galerkin is convergent) but not efficiently from
fixed momentum-space frames; the efficient exact vehicle is the real-space
continuously-adapted envelope carrying the EXACT local dispersion (unifying the
registry-adaptation the momentum model lacks with the exact dispersion the
thesis operator lacked). Full derivation + verdict in STRONG_COUPLING_ANALYSIS
§9. The practical §5 momentum model (24/24, 8e-4 shape, offset→0) stands as the
delivered result; §9 maps exactly what closing the last 10⁻³ requires and why.

---

## Weak-coupling test + a coupling-bug correction + the structural verdict (§10)

Tested §8.3's prediction (weak coupling → cheap few-parameter exactness) with two
controlled coupling sweeps at m=57/2°: r₂ (rod size, ΔΛ 2.44→0.21 λ,
V/E_kin 86→8) and ε₂ (contrast at fixed r₂=0.10, ΔΛ 2.44→1.13 λ). Key results:

- **The r₂ knob is unsafe (resolution trap).** Weak coupling ⟺ tiny rods: layer-2
  rod radius = r₂·px px → 1.6px at r₂=0.10 but **0.64px (sub-pixel) at r₂=0.040**.
  The FDFD ground truth is then discretization-limited (Richardson px16→px32 at
  r₂=0.040: 0.42995→0.43129, straddling the model). A naïve r₂ sweep shows a
  **spurious** ground-residual dip (~2e-5 at r₂=0.070) — a coincidental crossing,
  not exactness. The ε₂ knob (rods fixed 1.6px) is the clean probe.

- **Adversarial audit found a real transpose bug** in `momentum_kp_moire.py` (the
  Ṽ coupling paired the strong s_x-registry harmonic with the wrong moiré-recip
  axis; geometry-validated via FFT of the real-space V(r)=Λ(s(r))). Fix:
  `Vhat[(n2-m2),(n1-m1)]`. **Present in the §5/§6 delivered result too.**

- **The §6 headline residual was ~96 % artifact.** Reported 2° ground residual
  +2.74e-3 = transpose bug (+1.86e-3) + FDFD sub-pixel (−0.86e-3) + **TRUE
  +1.8e-5**. Corrected, the model ground **energy is exact to ~2e-5**. The
  "offset ∝ η^0.7 → 0" claim of §6 is **RETRACTED** — corrected offset is
  θ-independent (η^−0.01, +8.8e-4 at both 2°/1°), = the FDFD sub-pixel floor.

- **Structural verdict (correction-invariant): NOT eigenvalue-exact.** The
  single-band X-only model breaks the symmetry-protected 4-fold ground degeneracy
  (FDFD 1.7e-10 → model **1.17e-4 at 2°**, → 8.9e-7 at 1°, i.e. →0 as θ→0) and
  over-splits the miniband fine-structure (~13×). Weakening the coupling does NOT
  restore exactness (mixed: over-split shrinks, degeneracy-break grows); only θ→0
  suppresses the valley error. §8.3's "weak → cheap exact" is **not borne out**
  for either few-parameter vehicle (scalar momentum model here; fixed-frame
  Galerkin §9, same registry/valley reason). Exactness needs the X⊕X′
  valley-coupled (form-factor) operator or the full solve.

Lessons: the ground-residual-average is a poor exactness proxy (conflated bug +
sub-pixel + reference-registry DC); use the degeneracy-break/over-split; FDFD
must be Richardson-extrapolated even at 1.6px; adversarial cross-checking was
load-bearing (caught the bug the validated-anchor reproduction did not). Full
write-up: STRONG_COUPLING_ANALYSIS §10; figure `fig_weak_verdict.{png,pdf}`.

---

## The two-valley (X⊕X′) completion — corrects §9, a real path to exactness (§11)

Question: can the EA reproduce FDFD exactly? Traced the §10 non-exactness to the
model being **single-valley** (X=(π,0) only), and tested the fix (second carrier
X′=(0,π)). Adversarially verified (verdict: holds).

- **Premise (measured):** the 2° FDFD ground quadruplet (4-fold to 1.7e-10) is
  **2-at-X + 2-at-X′** (w_X≈w_X′≈0.61, w_M=0.000; `valley_composition_2deg.npz`).
  A single-X model must miss half. X′ folds to Q_X (X′−X=(−28,29)·b_sup, integer)
  but is a half-integer in moiré units → ~40 half-g steps from X, unreachable by
  cutoff at 2° (only ~2 cells at (7,1)/16°).
- **Fix = drop-in for the Galerkin** (`galerkin_recip.py --two-valley`): add an
  X′-centered momenta patch; kinetic |X+G|², basis_coeffs, ε-coupling all
  valley-agnostic (X′−X is an integer supercell-G shift).
- **RESULT (nref=9, gcut=4, m=57):** two-valley takes the captured manifold
  **10→24 (=FDFD count)** and lifts the ground **+6.8e-3 → +1.5e-3 (4.5×, b0) /
  +2.4e-3 (2.8×, b1 clean)**; S-rank 2684→4921. Mechanism at (7,1): X′-shell entry
  (gcut 3→4) converges the ground-split 2.09e-4→1.48e-4 toward FDFD 1.25e-4.
- **It's the VALLEY not basis size:** single-valley SATURATES (9→16 fr: +5054 fns,
  +191 rank, −2.6e-4), X′-block adds +2237 rank, −5.3e-3 (~9–16× more efficient);
  and w_M=0 ⇒ X′ is the only subspace with weight.
- **CORRECTS §9:** the +6-7e-3 plateau was substantially a **missing-X′-carrier
  truncation in the same momentum-space vehicle**, NOT the "fundamental
  position-registry locking needing real-space" §9 claimed ("stalls regardless of
  frame count/cutoff" was true only vs more X-frames).
- **Honest verdict:** DEMONSTRATED = valley-attributed 2.8–4.5× plateau-lift +
  full count + §9 correction + a variational method that converges in the
  complete-basis limit. NOT DEMONSTRATED = eigenvalue-exactness: bottom still
  +1.5e-3, ground a SINGLET (FDFD is a 1.7e-10 4-fold). Uncoupled X′ recovers the
  COUNT not the symmetry-protected DEGENERACY. **Remaining step:** X↔X′
  valley-COUPLED (form-factor) operator (§8.4b + §10.3), optionally on the
  real-space registry-adapted envelope (§9). No fundamental obstruction (2° inside
  §8.1-8.2 walls). Write-up: STRONG_COUPLING_ANALYSIS §11; `fig_two_valley.{png,pdf}`.

---

## Convergence of the two-valley Galerkin — exact at (7,1), conditioning-limited at 2° (§12)

Does the two-valley Galerkin CONVERGE to the FDFD floor? (variational theorem says
it must in the complete-basis limit). Built the machinery + tested:

- **Memory-lean eigensolve** (`galerkin_recip.py`, audit-designed): `eigh(S, evr,
  subset_by_value)` (O(Nb) workspace) + matrix-free `Hp` from sparse C — breaks the
  OOM wall (was 15GB→OOM at gcut/nref pushes). Gate: reproduces committed gcut=4 to
  1.4e-11. Added band-1-weight classifier (Galerkin analogue of FDFD w_X) to filter
  band-0 pollution + flag spurious states.
- **(7,1) CONVERGES to the floor:** valley-complete band ladder edge |Δf| vs FDFD
  = 3.4e-4(nb2)→8.5e-5(nb8)→**3.4e-5(nb16)**, monotone. The method IS
  eigenvalue-exact once the valley is present + basis complete (tractable cell).
- **2° gcut converges per rank BUT the fixed-frame basis is conditioning-limited:**
  at matched rank, gcut=5 beats gcut=4 (rank 2675→+2.0e-3 vs gcut=4 ~+5.8e-3) → gcut
  helps (convergence). But the well-conditioned rank caps ~4900 (redundant plane
  waves); beyond it near-singular S emits SPURIOUS sub-floor states (Δ<0, violates
  variational bound). Best CLEAN 2° bottom = +1.5e-3 (gcut=4/s_tol=1e-6) — a
  conditioning floor of THIS formulation, not a completeness wall.
- **Verdict:** EA CAN reproduce FDFD eigenvalue-for-eigenvalue (shown +3.4e-5 at
  (7,1)); at 2° the valley lifts the plateau 4.5×, residual is convergence throttled
  by fixed-frame ill-conditioning. Efficient 2° exactness → the real-space
  continuously-registry-adapted envelope (§9) carrying BOTH valleys (§11) — unifies
  exact dispersion (§4/5) + registry adaptation (§9) + valley (§11); the defined
  next program. Write-up: STRONG_COUPLING_ANALYSIS §12; `fig_exactness_ladder.{png,pdf}`.

## §13 — Foundations for the exactness program (floor + the symmetry of the 4-fold)

An adversarial design review surfaced two foundational
issues; both resolved by direct computation, and they reshape the picture.

- **Floor reconciliation.** The Galerkin is SPECTRAL (exact |X+G|² kinetic) → its variational
  floor is the CONTINUUM ground (from above); prior residuals were quoted vs res16 FINITE-DIFFERENCE
  FDFD (0.370047, from below), differing by the size of the gap. Frozen-candidate res16/32/48 ladder
  is O(1/px²) → **continuum floor = 0.370907 ± 5.7e-6** (`floor_reconciliation.py`). Re-baselined,
  the best clean 2° bottom is **+6.3e-4** (not +1.5e-3); the gcut5 state stays sub-floor (spurious).
  Action: repoint the Galerkin `--fdfd`/sub-floor threshold to 0.370907. (Galerkin's own px16/res64
  discretization ~1e-4 → falsifiable Stage-2 target ~3e-4.)
- **Space group = p4 (chiral), and the true even-grid C4 is a ROLL, not np.rot90.** `stage0c`:
  C4 about origin is exact via `A[:,(-arange N)%N].T`; np.rot90 is a half-pixel off (NOT a symmetry).
  No mirrors (twist is chiral, r1≠r2) → no glides, no 2D screws → symmorphic p4.
- **The ground 4-fold = REGULAR REP of C4 = TWO exact 2-folds + an emergent merge.** `stage0b`:
  χ(E)=4, χ(C4)=0 (eigenphases {−i,1,i,−1}), χ(C2)=0 → A⊕B⊕¹E⊕²E, one each; C4 maps X↔X′. Fine
  structure (`stage1_finestructure.py`) resolves it by C2=C4² into two pairs, each RIGOROUSLY
  degenerate (≤1e-15) at all angles: **{¹E,²E}** (C2=−1) T-protected (T-rep test: T:¹E↔²E, T:A→A,
  T:B→B); **{A,B}** (C2=+1) exact but T-INDEPENDENT → a HIDDEN symmetry (open question; A,B=(X1±X′1)/√2
  degenerate iff Re⟨X1|H|X′1⟩=0). The EMERGENT part is the inter-C2-sector split (→ the 4-fold):
  1.25e-4 (16°) → 1.7e-10 (2°) → 2.1e-11 (1°), θ→0 (BM-type valley physics). Answers "missing physics
  vs gauge?": neither — two rigorous 2-folds + an emergent small-angle symmetry.
- **Consequences.** Stage 1 fix splits THREE ways: C4+T-closure restores ¹E≡²E exactly; A≡B needs the
  basis to respect the hidden symmetry (measure — if A,B split, that identifies it); the inter-sector
  merge is a convergence/emergent target. Stage 4: plain single-X EA is C2-invariant (C2 fixes X,
  rigorous p4) → recovers 2 of 4 = **1/2** per quadruplet (not 1/4). NB: the (7,1)/16° manifold §12
  validated (f≈0.067) has w_X=0 — the band-1-at-X manifold has dissolved at β≫1 (§8.2); same C4
  fine-structure but a different band (a symmetry testbed, not an X-proxy). Write-up: §13.

## §14 — Fixing the 4-fold: the C4-irrep-projected Galerkin

The fix for the 4-fold, tested at the 16° testbed and the 2° target:

- **Naive C4-closure fails** (`stage1_c4basis.py`): generating the X′ block as the exact C4-image of
  X (index permutation n→nG₀+C4·n on the sparse plane-wave coeffs; self-check populates 12544/12544
  of the independent X′ indices) does NOT restore the 2-fold (min gap 2.3e-5) — the canonical
  orthogonalisation + eigensolve don't enforce the symmetry.
- **Explicit C4-irrep projection fixes it** (`stage1_c4proj.py`): project each X-seed onto the four
  C4 irreps v_χ=¼Σ_k χ̄ᵏ Pᵏ C[:,b], assemble H,S per irrep block, solve. ¹E,²E are conjugate blocks
  (T-images) → identical spectra by construction. Results: **max|f(¹E)−f(²E)| = 9.2e-12 (m=7 gcut3),
  7.0e-12 (m=57 2° over 634 levels)** — the rigorous T-protected 2-fold restored to ~1e-11 (from
  ~2e-5 un-projected). **A≡B split CONVERGES** (2.7e-5 gcut3,nb2 → 5.9e-6 gcut4,nb3): the hidden
  symmetry is respected in the complete-basis limit (convergence target, not obstruction). The
  **emergent inter-sector split** comes out 1.0–1.3e-4, matching FDFD's 1.25e-4 at 16°.
- **Verdict:** completely-fix-the-4-fold = C4-irrep projection (¹E≡²E EXACT, A≡B convergent, merge =
  emergent θ→0). Bonus: block-diagonalises H,S into 4 smaller better-conditioned blocks — a free win
  for Stage-2 conditioning. Open: the exact hidden A≡B symmetry (not p4, T, or {g|τ}·T). Write-up: §14.
  *(Superseded in mechanism by §15: A≡B is T_{P1}-protected; the "convergent" split was an engine artifact.)*

## §15 — The hidden symmetry identified (T_{P1}) + the engine's momentum-grid defect

Re-audit closed §14's open question AND found a real engine defect:

- **Hidden translation PROVEN**: τ=(L1+L2)/2 = P1 = ((m−1)/2,(m+1)/2) is an integer lattice vector
  of BOTH layers (exact rational algebra, m=7/57/113; ε_bl invariant to 0.0). The centered (m,1)
  cell is a 2× SUPERCELL of the true primitive crystal (`supercell_asym` cell='primitive').
- **T_{P1}·C4 = −C4·T_{P1} at Q_X** ({C4|L1}, e^{−iπm}=−1 for odd m); T²=+1. Only 2D irreps ⇒
  EVERY Q_X level is an exact {λ,−λ} doublet: {A,B} and {¹E,²E} — ONE rigorous symmetry explains
  both §13 2-folds (T redundant). Verified on the FDFD 4-fold: anticommutator 3.7e-15, T-maps
  A↔B, ¹E↔²E with |amp|=1.000000. The 4-fold = two T_{P1}-doublets + the EMERGENT valley merge.
- **Falsifier fired → engine defect**: iso-spectrality test (stage_a2_sector) FAILED → traced to
  the momentum grid: admissible Q_X momenta are p = X + INTEGER(j₁b₁+j₂b₂); the engine's
  half-integer steps make every odd-j coefficient index exactly .5 and `basis_coeffs`' np.rint
  ties-to-even SNAPS them — ~¾ of the historical basis was silently aliased (wrong momenta, mixed
  T-parity, near-duplicates). §7–§12 energies STAND (variational upper bounds); the §12
  conditioning wall is now prime-suspected to be aliasing-induced (re-test at 2°, Stage B).
- **Corrected engine (integer grid) — machine-exact structure**: seeds have uniform T-parity;
  sectors decouple exactly (ε_bl Fourier support = even-sum sublattice = primitive reciprocals);
  **A–B split = 1.7e-16, ¹E–²E = 2.2e-15** (old: 2.7e-5). Model lowest-4 = two exact doublets with
  emergent split 1.0e-4 vs FDFD 1.25e-4. The complete rigorous symmetry structure of the spectrum
  is now exact BY CONSTRUCTION; only the emergent merge remains as physics/convergence.
  Scripts: stage_a_tp1.py, stage_a2_sector.py, stage_a2_integer.py. Write-up: §15.
- **T_{P1} IS the valley** (§15.6, stage_a4_primitive.py): primitive-cell FDFD (DOF halved) at the
  two folded momenta: q₋ ≡ X′ mod primitive reciprocals — the hidden-translation quantum number is
  the VALLEY INDEX; valleys = C4-related Bloch momenta in the primitive frame (the moiré K/K′).
  Sector ladders identical to 3e-16; emergent split cleanly resolved within a sector at 16°
  (1.16e-4). Primitive Richardson family cross-validates the floor (0.370879 vs 0.370907, 2.8e-5).
- **§15.8 — floor pinned EXACTLY + the aliased-content lesson**: the engine's complete-basis
  operator (spectral kinetic + SAMPLED ε) densely diagonalized at m=7 (stage_a3_dense.py):
  ε-sampling offset vs continuum = **+2.4e-5 only** (retires the ~1e-4 floor worry; §12's (7,1)
  +3.4e-5 → +2.8e-5 vs continuum, survives). **2° target floor = 0.37093(3)**. Cautionary records:
  trig-upsample FD Richardson = wrong object (Gibbs, 0.3651 artifact — stage_a3_floor.py kept with
  warning); MINRES-ARPACK interior solve correct but too slow (stage_a3_spectral.py). ALSO: the
  corrected clean fixed-frame engine (galerkin_sector.py) with the old engine's IDENTICAL
  admissible content puts ZERO states in the 2° manifold window (≥+7e-3) — worse than the aliased
  engine (+2.4e-3): the old ~4500 snapped odd-j vectors were accidentally-useful admissible
  content. The fixed-frame reference-Bloch basis is not the efficient vehicle at strong
  modulation, cleanly or aliased. Measured envelope support of the FDFD ground (what any basis
  must cover): 50/90/99/99.9% of Fourier weight within 2.5/5.6/15.4/18.8 |b_prim| of the stars.

## §16-§17 — Stage B delivered: the valley-windowed PWE and the eigenvalue-exact ladder

(§16, Jul 10: `pwe_valley.py` — the exact solver on k = X + g_mono + G_env; m=7 to +5e-6 with
259 PWs; 2° budget-controlled to +8.7e-4 at the dense-RAM limit {18,18,18,12,6}; the §16.3
excluded-weight budget tracks every rung; §16.5 sector bookkeeping. Write-up: SCA §16.)

**§17 (Aug 26) — the ladder matched.** `pwe_valley_iter.py`: matrix-free iterative windows
(S·c = 2 FFTs on the fixed N² grid — window-size-independent; ARPACK shift-invert σ_f=0.3722 +
Jacobi-MINRES; pencil real-symmetric). Gates: FFT-vs-dense matvec 8e-16; eigsh vs dense eigh
≤2e-10; the {18,18,18,12,6} dense anchor reproduced to ≤4e-11 (bottom +8.74e-4 exactly).

- px16 window ladder: +8.74e-4 → +7.40e-4 ({24,24,24,16,8}) → +6.21e-4 ({40,40,40,32,16,8},
  g≤5, Nb=120k). Fits Δ = δ + r·B with **δ=4.8e-4, r=0.36**: a window-independent wall.
- **The wall = the px16 ε-sampling offset at 2°** (px32 rerun: all states drop ~¾δ; O(1/px²)).
  §15.8's m=7 calibration (+2.4e-5) does NOT transfer — the offset is state-dependent.
  This also explains the budget drift (res16-state tails are FD-damped).
- **The matched ladder**: {40,40,40,32,16,8} at px16+px32, per-state Richardson, vs the
  res16/32/48-extrapolated FDFD quadruplets (all 9 built; q0 = 0.370907±5.7e-6), sorted +
  index-aligned (strict, matching-free). All 14 even-sector residuals positive; per-quadruplet
  best **+1.6–1.9e-4 for q0–q5** (q6 +3.9e-4) = exactly the remaining r·B window term
  (0.36·3.95e-4 = 1.4e-4). Raw px32 bottom +2.77e-4 — under the ≤3e-4 Stage-B target unassisted.
  Odd sector = exact C4 image (§15.6) → 28 states covered. Path below 1e-4: one more window
  rung and/or px48 — mechanical, budget-predicted, no walls.
- **EA dossier at 2°** (`pwe_ea_fidelity.py`, odd sector, renv12/g3): M0 plain EA spans 95.4%
  of the true state, M1 96.3%, M2 (3×3 registries) 99.5% — but ALL rungs assign the energy
  ≈+1.5e-2: coverage is cheap, energy accuracy needs the full window. The one-valley "plain EA
  recovers 1/2 per quadruplet" statement stands structurally (§16).

Scripts: pwe_valley_iter.py, budget_window.py, fdfd_ladder_richardson.py, fig_exact_ladder.py.
Deliverables: fig_exact_ladder.{png,pdf}, exact_ladder_data.npz, fdfd_ladder_2deg.npz.
Write-up: STRONG_COUPLING_ANALYSIS §17.

## Section 18 (Aug 27) — Ground-up re-audit: two retractions, theory errata, the v5 rebuild

A full adversarial re-audit of the derivation, both implementations, and the historical
benchmarks. Every claim below was decided by a direct test (dense same-pencil references,
exact integer algebra, controlled reruns); scripts and the full verdict dossier are local
(_local/audit_*). Two standing results are retracted, several are newly confirmed robust.

**Retraction 1 — the band-order fix of section 15-era work was itself the bug.** A dense
Hellmann-Feynman reference on the identical discrete pencil shows the Rust extractor
emits the exact-TM matrices in ABSOLUTE eigenvalue order (deviation 1.6e-4 = solver
floor) — not retained-first (deviation O(1)). The MSL_BAND_LO permutation in the vendored
phase2 therefore corrupted the velocity/gamma1 blocks of every band_lo=1 assembly. The
permutation is removed. Tainted and needing re-assembly: the band-1 exact-TM ladders,
including the "band_lo=1 ladder still 5x over-dense" strong-coupling evidence (channel b
of section 4; channel a and the Nb=2 band-edge 6e-5 match used band_lo=0 and stand).
The valley-PWE/FDFD work of sections 15-17 never touches the extractor and stands.

**Retraction 2 — the golden 1.12-degree benchmark is not state-identity evidence.**
(i) Sector algebra: the EA ran at carrier K=(2/3,1/3) with periodic envelope, which
folds to supercell sector (1/3,2/3); all archived FDFD references are at Gamma. A
controlled two-sector rerun at res16 shows the two sectors differ by 6.7 mean level
spacings in the EA window — the historical comparison was made against the wrong
spectrum. (ii) Solving in the correct sector does NOT produce a match (Hungarian mean
5.6e-5 vs 5.1e-5 at Gamma, both consistent with a spacing-preserving null; the archived
2.3e-5 sits at the 42nd percentile of a cyclic-shift null — a rigid random shift of the
same pool matches equally well). (iii) The number is not code-stable: the geometry/solver
path drifted ~1.3e-5 per mode since the March runs with no surviving March copy.
Consequence: the eigenvalue-level validation of the EA remains open in BOTH campaigns;
an honest golden-system rerun needs the corrected pipeline, sector, and protocol.

**Theory errata (recorded for the write-up; derivation notes local).** Confirmed by
exact algebra/numerics: the registry map as written is missing 1/eta under the stated
dimensionless convention (delta(R) = -J R exactly); the mean-frame lift omits a
cos(theta/2) dilation at the retained order; the first-order symmetrized remainder needs
the commutator (1/2)[A_i, v] (the appendix's anticommutator belongs to the other
convention); the appendix Loewdin split is sign-flipped and non-Hermitian as displayed
(the safe form is the factorized -S^dag R_Q S, which the mass-tensor equation already
matches); unrestricted frame gauge requires the full matrix Lambda to rotate; the
expanded second-order figure double-counts the direct remainder. The exact first-order
operator statements and the flux-form direct term verified correct — the raw projected
operator is a sound implementation target.

**Blaze kernel findings (fixes queued in blaze2d):** the exact-TM coefficient derivative
is built on the Bloch k0+G table, injecting exactly -i k0 eps^-1 into gamma1 (16% of its
norm at X-like carriers, machine-exact match to the defect model); the k+G near-zero
clamp leaks an artificial x-direction into k-derivative exports at Gamma (1.6e-5); solver
convergence flags do not certify eigenpairs (true errors 70-4000x the requested tolerance
on a 16x16 dense cross-check; no residuals exported). Blaze maps the FFT Nyquist row to
+N/2 (numpy uses -N/2) — required knowledge for any dense cross-check.

**Confirmed robust:** the FDFD field back-transform had the inverse power of sqrt(eps)
(fixed), but X/X' valley classifications shift only a few percent under the correction —
no label flips; the dissolution verdict and the 24-state manifold count stand. The
centered-cell/valley/T_P1 structure of sections 15-17 is untouched by any finding.

**The v5 rebuild (research/post_thesis/lib_v5/, all tested):** exact integer
lattice/sector/coset algebra (Smith normal form; the hex K corner fixed — V4 configs had
(1/3,1/3), an interior point at 58% of the corner distance); tamper-detecting run
manifests that refuse ambiguous band order; oracle layer (lifted-basis Ritz, exact
Feshbach downfold, principal angles, inertia interval counts, lifted-residual
certificates — injected-defect detection verified); doubler-free envelope kinetic
(flux-form diagonal + centered cross terms: single symbol zero over the full discrete BZ,
second-order, Hermitian by construction; the V4 centered-derivative symbol has four);
smooth finite-Fourier material family with exact registry derivatives; exact lifted
moire reference via integer harmonic maps (layer-2 matrix A2 = B0^-1 R^-1 B0 A, integer
for commensurate cells) passing the zero-modulation fold-union test at 1e-8. The
valley-PWE window builder gained orbit-closed union windows (the previous intersection
silently shrank the advertised cutoff; intersect mode kept for section 16/17
reproducibility).

Next: raw projected operator + the tiny synthetic end-to-end case, blaze kernel repairs,
then the smooth weak-bilayer validation ladder toward the eta-scaling result.

## Section 19 (Aug 27) — Raw projection validated, kernel repairs landed, smooth candidate frozen

**Blaze kernel repairs (blaze2d 2df27d5, pushed):** G-only coefficient-derivative tables
(the -i k0 eps^-1 defect gone to 7e-16), raw/preconditioner k+G table split (the Gamma
clamp leak in exported velocities now exactly 0), post-solve Rayleigh-Ritz certification
with fresh per-band residuals and B-orthogonality exported on every path plus an optional
fail_on_residual gate, and blaze.build_info() provenance. Full workspace suite green.

**Raw projected operator (lib_v5/raw_projection.py) — the corrected theory works.**
Built on the product space (slow torus grid x monolayer plane waves) by direct operator
composition: the hermitized TM lifted operator is exactly quadratic in (D_r + eta D_R),
so the three orders are assembled without any hand-expanded coefficient algebra, and all
orders are Hermitian by construction (odd slow grids). Two exact identities make the
tests sharp: for finite-Fourier bilayers the phase registry map delta(s) = (A2 - A) s
reproduces the twisted material identically, and one hermitized-collocation
discretization family must be used on both sides of any comparison (the truncated
generalized pencil differs at O(truncation): (P eps P)^{-1/2} != P rho P).

- Frozen-registry complete-frame gate: product-space spectrum = monolayer family at the
  shifted momenta to 1e-8, square and hex (oblique), generic off-symmetry carriers.
- **Headline: raw projection vs direct supercell projection (same trial space, same
  family) deviates by 3.2e-7 / 1.3e-7 / 8.4e-8 in lambda for square (m,1), m = 5/7/9 —
  about 1e-8 in frequency — with fitted order eta^2.3.** The corrected raw envelope
  operator tracks the exact lift at the target accuracy scale of the whole program.
- Generic-carrier warning reproduced in miniature: with v != 0 the frozen single-band
  drift term makes the projected spectrum dive unphysically as the envelope grid grows
  (bottom 2.9 -> 0.98 for Ns 7 -> 15 vs lambda_1 = 7.5) — validation carriers must be
  dispersion extrema, and window comparisons need certified counts (recorded as a test).
- Gauge: random U(2) frame rotations are isospectral to 1e-9.

**Smooth validation candidate frozen (A_triple_match/smooth/candidate_hexM.py,
assert-verified):** hex lattice, eps0 = 9, detuned three-star host (2.95/2.25/2.60) +
weak layer-2 star 0.12 — a SINGLE-valley manifold: band-1 floor at M2 = (0, 1/2)
(time-reversal invariant, so no TR partner; the detuning splits the three M points,
next-M separation +0.011), window V = 0.023, registry-common full gap below +0.148,
band-2 headroom +0.611, min-eps 0.84. V/E_kin spans 0.1 -> 0.8 over commensurate
(4,3) -> (9,8) with a strongly anisotropic mass (0.25/2.9); every cell in the family is
37-217 primitive cells, so the full reference is dense-solvable across the ladder.
Known refinement for the production runs: widen the host gap (beta = theta/gamma ~ 1 at
the small-angle end as parametrized).

Next: the L7 reference stack on the frozen candidate (dense PWE + certified matrix-free
+ FDFD on the identical analytic map), then the tracked-cluster eta-scaling family.

## Section 20 (Aug 28) — THE HERO MEASUREMENT: triple match at 8e-7 in frequency, scaling law eta^3.8

The goal of the whole program — an eigenvalue-exact envelope-vs-brute-force comparison —
is delivered on the frozen smooth single-valley candidate (candidate_hexM: detuned hex
host, band-1 floor at the time-reversal-invariant M2, no valley partner of any kind).
All numbers below are ground-state (the manifold's V-window holds exactly one state per
angle at these kinetic-dominated parameters), sector-exact, count-certified.

**Reference certification (two independent solvers).** The hermitized-collocation PWE
reference (dense below 26k plane waves, else bottom-block Lanczos with certified
residuals <= 8e-12) and the FDFD leg (exact analytic dielectric sampling, res 16/24/32,
fitted order p = 2.000 at every angle, extrapolation uncertainty ~4e-7 in lambda) agree
to |PWE - FDFD| = 2.5-2.9e-7 in lambda — 1.7e-8 in frequency — uniformly across the
family. Every solver consumes the identical 7-coefficient analytic dielectric.

**Three model/scaling families (all knobs certified: Ns 17 vs 21 drift 6e-15, gmax_mono
4 vs 5 and fine 192 vs 256 at the 1e-6 level):**

1. Frozen-frame single-band raw projection, fixed a2 = 0.12: deviation 2.0-2.4e-3 in
   lambda, exponent eta^-0.18 — an eta-INDEPENDENT floor. The frozen frame cannot follow
   the registry rotation of the Bloch state; the floor is a material property (~ second
   order in the interlayer coupling), now measured cleanly.
2. Registry-adapted single-band frame (the frame u1(delta(R)) enters the product-space
   trial tensor per slow point; the slow spectral derivative differentiates THROUGH the
   frame, so every frame-derivative/Berry/leakage contribution enters mechanically, and
   the trial space stays exactly orthonormal): 15x better, 1.35-1.94e-4 in lambda
   (8-12e-6 in f), exponent eta^-0.41. Still not asymptotic — as it must be: at fixed
   material the family is nonuniform (V/E_kin ~ eta^-2), the localization-sharpened
   envelope eats the naive eta-gains.
3. **The uniform asymptotic family (a2 proportional to eta^2, V/E_kin fixed): deviation
   1.42e-4 -> 6.6e-5 -> 3.5e-5 -> 1.28e-5 in lambda over (5,4) -> (9,8), fitted
   exponent eta^3.79.** At the (9,8) landing point:

       |EA - PWE reference| = 1.28e-5 lambda = 7.9e-7 in f
       |EA - FDFD|          = 1.31e-5 lambda = 8.1e-7 in f
       |PWE - FDFD|         = 2.7e-7 lambda  = 1.7e-8 in f

   Envelope approximation, plane-wave reference, and finite-difference solver agree on
   the moire manifold state below 1e-6 in frequency, with a measured convergence law.

**Negative results worth as much as the positive ones:** multi-band FROZEN frames are
poisoned at this carrier (the remote bands have nonzero slope at M2; their envelope
parabolas over-extend and dump spurious states into the window — the [0,1,2] frame put
7 spurious states there). The frozen floor and the nonuniform-family exponents are the
quantitative demonstration of WHY registry adaptation and scaled-asymptotics matter.

Open threads recorded: the (9,8) upper window (1.676-1.679) mixes M3-valley reference
states with EA envelope excitations and is not yet state-resolved (ground-state claim
only; needs the principal-angle machinery + deeper Lanczos coverage); extending the
ladder to multi-state windows means smaller angles or shallower kinetics; the analogous
eta^2-scaled run on the OTHER rungs (frozen, multi-band with a momentum patch) would
complete the model-hierarchy phase diagram.

Machinery: smooth/hero_engine.py (certified PWE reference + lazily projected raw
operator, frozen or adapted frames), hero_family.py / hero_adapted.py / hero_scaled.py
(the three families), fdfd_leg.py (independent leg, exact sampling, fitted-order
extrapolation), fig_hero_scaling.py (the scaling figure). Data local:
hero_{family,adapted,scaled}.npz, fdfd_leg_ladders.npz, fdfd_scaled.npz.

## Section 21 (Aug 28) — Sector ladders resolved: the upper window is the folded band-1 tower

Follow-up to the section-20 open thread ("the (9,8) upper window mixes M3-valley
reference states with EA envelope excitations"): both halves of that sentence were
wrong, in a good way. The upper window states are neither M3 nor envelope excitations —
they are the FOLDED BAND-1 LADDER of the same M2 sector, and the whole window is now
state-resolved in all three solvers.

**The three M valleys fold to distinct supercell sectors.** Exact integer computation
of kappa_s = A^T kappa0 + kappa_env mod 1 for all three M points, across every (m,n) in
the family: the three folded sectors are pairwise distinct at every angle. A
sector-resolved spectrum therefore holds ONLY the M2 tower — there is no valley
admixture to disentangle, by arithmetic rather than by projection. (This is the same
sector algebra that retired the golden benchmark in section 18, now working for us.)

**The folded ladder identified.** The states above the manifold rung are band-1 Bloch
states at the supercell-commensurate monolayer momenta nearest M2 in-sector, folded
down by the moire cell. Their spacing follows 0.5 * c_min * |b|^2 / N_cells (leading
in-sector folded k): +0.032 predicted at (9,8) vs +0.0320 measured; the same
commensuration arithmetic says (6,5) and (7,6) have a single in-range rung and (9,8)
has four in [1.60, 1.695] — exactly what all solvers show.

**Wide-window cross-checks (fixed material, wide PWE Lanczos + wide FDFD re-extraction
from stored raw levels):**

- (9,8): FDFD [1.6464554, 1.6784612, 1.6804573, 1.6851343] vs PWE [1.6464551,
  1.6784609, 1.6804571, 1.6851340] — all four rungs agree to 3e-7 in lambda (2e-8 in
  f). The two independent references certify the ENTIRE sector tower, not just the
  ground state. The adapted EA at fixed material tracks the tower at its known
  nonuniform-family floor (1.9e-4 lambda on the ground rung, up to ~1e-3 on the top
  rung, i.e. 1e-5 to 7e-5 in f) plus one spurious near-degenerate pair below the first
  folded rung — single-band envelope, next order in the tower.
- (9,8) scaled family (a2 ~ eta^2): three rungs in window, references again 3e-7 on
  every rung; EA ground at 8e-7 f (the hero number), EA top tower rung at 7e-6 f,
  spurious pair dashed. The tower error being ~10x the ground error is consistent with
  the folded rungs living farther from the M2 carrier where the single-band quadratic
  model degrades.
- (6,5): single rung, FDFD 1.6477087 vs PWE 1.6477084 vs EA 1.6475564. Consistent.

**One open discrepancy, recorded honestly:** at (7,6) fixed material the wide PWE run
finds a second rung at 1.6909375 (Lanczos residual certificate 3.3e-11 — it IS an
eigenvalue of the PWE operator) but the FDFD raw levels, which span past 2.1, have no
state between 1.648 and 1.76. The adapted EA puts a pair at 1.693, weakly supporting
the PWE side. This is a live instance of the audit report's warning that shift-invert
eigsh does not certify intervals — one of the two legs mis-enumerates here, and only
LDL inertia counts on both operators can adjudicate. Excluded from the figures;
flagged for the inertia-certification pass.

Figures: fig_hero_ladders.py renders fig_ladder_family (the manifold rung at all four
angles, three solvers overlaid + the eta^3.8 residual panel), fig_ladder_landing (the
(9,8) landing point on a 1e-6-f axis with the FDFD extrapolation band), and
fig_ladder_tower (side-by-side rung ladders FDFD | PWE | EA, scaled and
fixed-material, matched rungs connected and annotated, unmatched EA states dashed).
Data local: ladder_fdfd_wide.npz, ladder65/76/98_unscaled.npz.

## Section 22 (Aug 28) — Wide ladders: a sparse certified reference, and the M3 ceiling on single-valley matching

Follow-up to section 21, driven by the question "how far up does the ladder match?".
Answering it needed a reference that can reach 40-100 states without an hours-long
Lanczos climb, and that can prove it has not skipped one.

**The sparse Galerkin pencil (ladder_wide.py).** The commensurate moire dielectric of
finite-Fourier layers has an EXACT 13-term supercell Fourier series, so the plane-wave
mass matrix S = eps_hat is sparse (13 bands). The TM pencil K u = lambda S u with
K = diag|k_s + G|^2 is then a sparse symmetric pencil, which buys two things:

1. Shift-invert reaches interior window states directly, instead of climbing through
   the N_cells band-0 states underneath them. At (9,8): **21 s versus 5551 s** for the
   hermitized-collocation Lanczos of section 21 — a 250x speedup — returning the same
   four rungs to all seven digits (1.6464551, 1.6784609, 1.6804571, 1.6851340), and
   converged from cutoff 3.5 through 6 (1.64645513 at every one).
2. The LDL^T factorization of (K - lambda S) gives the EXACT number of eigenvalues below
   lambda by Sylvester inertia. Every window census below is certified, not assumed.

**The (7,6) discrepancy of section 21 is resolved, against the FDFD leg.** Certified
census on [1.60, 1.76] is 5 at (7,6), and 1.6909375 is one of them. The state is real:
a third, independent discretization plus an exact inertia count confirm the plane-wave
side. The FDFD leg under-enumerated (its extraction was capped at SIGMA + R_COVER).
Section 21's "one of the two legs mis-enumerates" is settled — it was FDFD.

**Correction to section 21 — the M3 basin, not just the M3 point.** The three M *points*
do fold to distinct supercell sectors (that stands). But the sector's momenta
{M2 + G_moire} sample the whole monolayer Brillouin zone, so points NEAR M3 are in this
sector even though M3 itself is not. M3 sits +0.0355 above the M2 floor and M1 +0.0754,
so above +0.0355 the sector carries M3-basin states and above +0.0754 M1-basin states
as well. "This sector holds the M2 tower only" is true only below +0.0355. That is the
ceiling on what any single-valley envelope theory can match here.

**Measured wide ladders** (all counts inertia-certified; registry map det(A2 - A) = 1
verified, so the folding is a clean bijection and the extra envelope levels below are a
model artifact, not a folding bug):

- (18,17), 919 cells, fixed material, +0.148 window: 47 states. **|PWE - FDFD| =
  3-8e-8 in frequency on every one of the 47** — two independent full-Maxwell solvers
  agreeing state by state across the whole gap. EA matches 13.
- (18,17), scaled family: 49 states, EA matches 11, reaching reference index 36,
  deviations 6.2e-8 to 1.4e-5 in f.
- (32,31), 2977 cells, scaled family, +0.10 window: **85 states**, EA matches 17,
  reaching reference index 76, deviations **9.5e-9 to 9.0e-6 in f** (median 3.0e-6).

**What the envelope model does to a tower.** Its spectrum CONTAINS the true M2-basin
levels to ~1e-6 in f, and adds extra levels around them. At (18,17)-scaled the true
first shell splits into three doublets (+0.00711/+0.00715, +0.02833/+0.02833,
+0.05081/+0.05084) — the six nearest folded momenta separated by a strongly anisotropic
M2 mass (roughly 15:1). The envelope operator reproduces the +0.0071 and +0.0283 pairs
exactly but puts six states near +0.007, i.e. it under-resolves the mass anisotropy in
the upper shells while getting the individual matched levels right.

**Why 40+ MATCHED rungs is out of reach at these angles.** The anisotropy makes the
M2-basin tower sparse: only 3 states within +0.024 at 919 cells. The matchable count
scales as roughly 0.135 * N_cells * E with E <= 0.0355, so 40 matched rungs needs
N_cells ~ 8300, about 0.6 degrees — a supercell several times larger than anything run
here. The honest statement is that the comparison spans 85 states, the two references
agree on all of them, and the envelope theory tracks a certified subset at 1e-8 to 1e-5.

Machinery: ladder_wide.py (sparse pencil, inertia census, three legs, --scaled for the
uniform asymptotic family), fig_ladder_wide.py (the wide ladder figure). Data local:
ladder_wide_{1817,1817s,3231s}.npz.

## Section 23 (Aug 29) — The anatomy closed: a-priori validity domains, the frame hierarchy, and every state accounted for

Section 22 left a fair objection standing: some rungs matched exquisitely, others
were missing, and the envelope model produced extra levels — "correct as far as it
goes" is not a validation. This section closes that gap. Every unmatched and every
extra level now has a verified mechanical explanation, the model's error is
PREDICTED before any reference is run, and the upgraded single-band model matches
every state of a pre-declared domain.

**The per-state diagnosis (valley_diagnosis.py).** Sparse-pencil eigenvectors give
each reference state's valley weights (they are essentially binary — every window
state is a pure valley state) and dominant folded momentum; the EA eigenvectors give
each envelope state's dominant harmonic. At (18,17)-scaled: 22/22 unmatched
reference states are other-valley or above-ceiling states (100%), 44/46 extra EA
levels carry out-of-domain harmonics (96%, the remaining 2 are degeneracy-inflated
partners of matched shells), and all 5 in-domain states matched at 6.2e-8 to 2.5e-7
in f. The "same momenta, mis-energized" picture is confirmed with receipts: each
extra EA level's dominant harmonic points at a momentum whose true band-1 energy is
far above the window (+0.19 to +0.53) — the model's continued surface assigns it a
wrong low energy, while the reference holds the same momentum at its true energy
(unmatched, other basin).

**The sharper finding — direction, not distance.** At (32,31)-scaled the domain
grows to 21 states and the picture refines: the fixed-frame surface h11(kappa) =
u1(M2)^H C(M2+kappa) u1(M2) is nearly ISOTROPIC, while the true band-1 surface is
strongly anisotropic (the first heavy-direction shell sits 9x above the first light
one). The missing heavy mass lives in the remote-band k.p coupling that a frame
frozen at one momentum cannot carry. Consequence: at the same |kappa| = 0.133|b|,
light-direction rungs match at 3.7e-7 while heavy-direction rungs err at 1.5e-5+.
The dispersion gap |h11 - E_true|(kappa) is computable from the MONOLAYER ALONE and
predicts the fixed-frame model's per-rung error before any supercell is solved
(fig_ladder_domain, right panel): predicted 3.8e-7/1.6e-6/3.7e-6/7.2e-6 on the
light shells vs measured 3.4e-7...7.2e-6; predicted ~1e-3 on the heavy shells vs
measured 4e-4...1.1e-3 (the restricted model; level repulsion makes the prediction
an upper-bound-flavored estimate).

**The two stacked approximations, measured separately (hierarchy_ladder.py).** The
user-facing question was: how much error is the two-crystal/envelope step, and how
much is the expansion around the carrier? Four models against the same reference:

  model                          (32,31), 21 in-domain rungs, dev in f
  fixed-frame single-band EA     3.4e-7 (light shells) ... 1.1e-3 (heavy shells)
  fixed-frame three-band EA      no cure (2.3e-7 ... 6.2e-5; at (18,17) it was
                                 strictly worse than single-band — the frozen
                                 remote frames add over-extension, not mass)
  exact-frame Ritz, band 1       5.4e-12 ... 5.9e-8   (median 4.4e-10)
  exact-frame Ritz, bands 0-2    same to within nanounits — band mixing by the
                                 moire potential is negligible here

The exact-frame model (lifted_ritz: one exact Bloch function of the
registry-averaged monolayer per in-domain folded momentum — "many real bands"
resummed into the k-dependent frame) is the envelope idea with the expansion error
removed. Its 4e-10 median residual IS the two-scale/one-band/one-valley
approximation error at these couplings: essentially zero. Everything else ever seen
in the tower comparisons was dispersion error of the fixed frame, and it is
predictable a priori.

**The reviewer-proof protocol.** Declared from monolayer dispersion alone, before
any comparison: (i) the domain = harmonics in the M2 basin at or below the M3 floor
(the ceiling above which other-valley states legitimately enter the sector and no
single-valley claim is made); (ii) the fixed-frame model additionally claims only
rungs with a-priori dispersion gap <= 1e-5 in f; (iii) models are built ON the
domain (momentum-restricted trial spaces — lazy_project grew a slow_modes argument,
verified against the position-basis spectrum at 3e-15), so counts are fixed by
construction; (iv) reference censuses are certified by Sylvester inertia; (v)
matching is sorted 1:1, no tolerance, no Hungarian.

**The big run: (55,54), N_cells = 8911, theta = 0.607 deg (ladder_wide.py bigrun).**
The a-priori census says 51 in-domain states. Measured, FDFD-only reference (res
16/20 on 2.28M and 3.56M unknowns, census certified by LDL inertia on the FDFD
matrix at both resolutions, extraction equal to the census at both):

- FDFD claims below the ceiling: 51 — equal to the a-priori census; eigenvector
  classification confirms 51/51 claimed states are M2-dominant.
- Exact-frame envelope model, sorted 1:1 over ALL 51 rungs: |dev in f| = 4.7e-8
  min, 5.8e-8 median, 8.1e-8 max. The measured reference error itself (FDFD
  extrapolation vs the certified plane-wave pencil, calibrated on the 21 in-domain
  states at (32,31)) is 3.2-5.2e-8 in f — the 51-rung comparison is
  REFERENCE-LIMITED: the model matches the full-Maxwell solver to within the
  solver's own discretization error, on every state it claims, with counts fixed
  before the comparison.
- Fixed-frame model: claims its 13 a-priori rungs, all 13 within 5e-8 to 6.8e-6 —
  inside the declared 1e-5 limit. Its trial count below the ceiling (59) does not
  close against the census (51): the count failure of the fixed frame is itself
  part of the record, and only the exact-frame model achieves count closure.
- Ops note: the first attempt at res 16/24 (5.1M unknowns) was OOM-killed during
  the CHOLMOD factorization; res 16/20 fits and its extrapolation error is
  calibrated above.

Figures: fig_valley_geometry.py (the momentum-space anatomy: basins, folded
lattice, domain patch, and the two-direction dispersion cut), fig_ladder_domain.py
(the (32,31) three-column ladder + the a-priori error closure), fig_ladder_big.py
(the (55,54) headline ladder). Corrections to earlier sections: section 21's
"sector holds only the M2 tower" holds only below the M3 floor (already noted in
section 22); section 22's "40+ matched rungs needs ~8300 cells" was about the
fixed-frame model — with exact frames the whole domain matches and the count is
set by the domain census (51 at 8911 cells). Data local: diag_*.npz, hier_*.npz,
ea_dom_*.npz, ladder_big_5554.npz.

## Section 24 (Aug 30) — The envelope approximation completed: the thesis's own eta^2 term, and the resummed model that closes everything

The question after section 23 was whether the dynamic frame is an addition to the
theory or already inside it. It is inside it: the eta^2 Lowdin remote-band dressing
is part of the V4 operator family (historically its best rung, "Nb2+6rem"),
manifest_lowdin_v1 is a named model in lib_v5/manifest.py that the v5 rebuild never
implemented, and oracles.feshbach was built as its adjudicator. The v5 campaign
validated the raw rung only — the EA was truncated one rung early, not wrong. This
section implements the missing rungs, measures the order-by-order convergence, and
lands the resummed model as a standalone monolayer-only computation.

**The mass sum rule, closed at the symbol level (order_ladder.py).** The raw
fixed-frame surface is an EXACTLY isotropic parabola — h11(kappa) = E1 +
(u1^H R^2 u1)|kappa|^2, coefficient 0.247, equal to the true light-direction
curvature 0.2464. The true heavy curvature 2.3503 (9.5:1) is carried entirely by
the second-order k.p sum: ONE remote band (band 0) contributes +2.159, two bands
close the sum to 0.08%, and the full sum closes to the finite-difference exact
value at 1e-5 relative (velocity elements cross-checked by finite differences).
Also diagnosed: the section-23 ea3 failure was the algebra, not the content —
DIAGONALIZING P+Q lets remote parabolas dump spurious states into the window,
FOLDING Q into P is the same trial content with none of that.

**The order ladder, measured (fold_model in hierarchy_ladder.py; fig_order_ladder).**
At (32,31)-scaled, per-rung deviation over all 21 in-domain states, median (max):

  raw fixed frame                    8.3e-04  (1.1e-03)
  + Lowdin fold, 1 remote band       3.8e-05  (8.6e-05)
  + Lowdin fold, 3 remote bands      9.9e-06  (4.5e-05)
  exact Feshbach, 3 remote bands     2.1e-06  (9.3e-06)
  resummed exact frames              4.4e-10  (5.9e-08)

Monotone on every heavy rung; the one honest wrinkle is that the eta^2 fold puts a
~1e-5 wobble on the top LIGHT shells that raw already had at 7e-6 (its correction
carries its own fixed-frame dispersion error). The Q space is pole-guarded a priori
(remote raw surfaces must stay 0.03 away from the window; nothing was actually
near it at these parameters).

**EA v2 — the resummed model with no supercell anywhere (ea_v2.py).** In the TM
pencil the Galerkin blocks over exact averaged-monolayer Bloch functions collapse:
distinct harmonics are distinct momentum cosets, so with B-orthonormal
eigenvectors the model is

    diag(E_band1(k_n)) c = lambda (I + V) c,

with V the six-neighbor interlayer hop matrix, hops n -> n - W^T h through the
layer-2 star (the delta(s) = (A2-A)s identity), elements = plain overlap sums of
monolayer eigenvectors. Exact local band data + nearest-neighbor envelope hopping;
cost is N monolayer solves. Gates: identical to the supercell-built exact-frame
Ritz at 1.2e-14 / 3.4e-14 / 6.0e-14 in lambda at (18,17)s / (32,31)s / (55,54),
and it reproduces the 51-rung (55,54) FDFD result (4.7-8.1e-8 in f) in under a
second on a laptop core.

**One real bug found and fixed on the way:** the energy-capped valley-agnostic
harmonic set must deduplicate momentum cosets (lattice copies of far valleys are
the same Bloch state), and the hop lookup must then be COSET-AWARE — a hop target
kept under a different copy's representative needs the exact re-indexing
u1(g; k+G0) = u1(g+g0; k). Without it the hops break at basin-copy seams and
inter-copy doublets fail to split (that was the anomalous flat-in-a2 worst rung).
Domain-patch results are unaffected (re-verified: gates unchanged).

**The validity frontier (c2 sweep at (18,17), full +0.148 window, all valleys,
counts closed 51 = 51 at every point; fig_frontier).** Median deviation follows a
clean a2-squared law across the whole sweep — the registry-dressing order, exactly
the next term the theory predicts:

  a2 = 0.008: med 7.6e-09 max 2.5e-08     a2 = 0.120: med 1.8e-06 max 8.6e-06
  a2 = 0.030: med 8.9e-08 max 4.0e-07     a2 = 0.200: med 5.3e-06 max 2.6e-05
  a2 = 0.060: med 3.8e-07 max 1.8e-06

At the frozen candidate material itself (a2 = 0.12) the valley-agnostic EA v2
matches ALL 51 states of the full gap window at or below 8.6e-6 in f — the
fixed-material case that section 22 said needed ~8300 cells for 40 matched rungs
with the old model, and where the old fixed frame matched 13. Section 22's
cell-count estimate and section 23's fixed-frame claim-limit protocol both stand
as statements about the OLD model generation; the resummed model supersedes them.

**Boundary convergence:** trial buffer +0.024 above the claim edge removes the
~5e-8 top-shell tails entirely (max in-domain deviation 1.8e-9 at (32,31)-scaled,
identical at +0.048) — the (55,54) 5.8e-8 median against FDFD is purely the
reference's own discretization floor (independently measured at 3.2-5.2e-8).

Answers to the open question of section 23: the envelope idea was never
structurally single-valley or fixed-frame — with exact per-momentum frames it is
valley-agnostic and its remaining error is the a2^2 registry dressing, which is
itself the next computable rung (registry-adapted k-dependent frames, the B3
option, not yet needed anywhere in this material family). Machinery committed:
order_ladder.py, fold_model in hierarchy_ladder.py, ea_v2.py, fig_order_ladder.py,
fig_frontier.py. Data local: fold_*.npz, c2_sweep_1817*.npz, c1_1817_fixed.npz.

## Section 25 (Aug 31) — The thesis crystal: the loop closed, the wall measured, and the angles brute force cannot reach

The completed envelope machinery (section 24) was ported to the thesis's own
asymmetric square bilayer — layer-1 rods r=0.20, layer-2 rods r=0.10, eps 8.9 on
background 1, TM, X carrier, band 1 — represented by its exact disk Fourier
coefficients through a Lanczos window (H=10; eps in [0.90, 9.01], gmax-converged
to 2e-7 in lambda). Every solver consumes the identical analytic coefficients.

**Angle ladder, five commensurations (m,1).** Three carry brute-force references,
two are past what this machine can solve:

  (m,1)     theta     N_cells   FDFD   MPB   EA floor f    |EA - FDFD| (floor)
  (15,1)    7.63 deg      226    yes   yes    0.381329        3.0e-04
  (29,1)    3.95 deg      842    yes    -     0.375462        6.7e-05
  (57,1)    2.01 deg     3250    yes    -     0.371813        1.0e-04
  (113,1)   1.01 deg    12770     -     -     0.370188        no reference
  (229,1)   0.50 deg    52442     -     -     0.369500        no reference

FDFD is inertia-certified per resolution and Richardson-extrapolated (fitted order
p = 2.05 at every angle); MPB at resolution 24 differs from FDFD by 1.4e-4 in f on
the same states, which is MPB's own un-extrapolated discretization error — the
envelope model's floor is already inside that at all three reference angles.

**The cost inversion, measured.** FDFD went 13 s (m=15) -> 65 s (m=29) -> 3221 s
(m=57, 3.3M unknowns at res 32); m=113 would need ~13M unknowns and did not fit.
The envelope model went the other way: its frame cache collapses as the angle
shrinks (226 -> 81 -> 25 -> 9 k-cells for m = 15/29/57/113) because pocket-bound
envelopes have a FIXED width in moire harmonic units. The 0.50 deg case — 52,442
moire cells — took 533 s, and the residual cost is the registry setup, not the
angle. Beyond about 2 deg the envelope model is the only solver in the room.

**The wall, quantified a priori.** The registry-resolved bands at X (full bilayer,
6x6 registry grid): band 0 in [2.445, 3.004], band 1 in [5.316, 7.612], band 2 in
[11.803, 16.281]. So the registry potential depth on band 1 is V = 2.296 while the
gap below is 2.312 and above 4.191 — V/gap ~ 1. This crystal is nowhere near the
weak-coupling regime of the smooth hex candidate (where the completed model is
reference-limited at 5e-8). The true ground state sits 0.42 above the registry
MINIMUM, not near the band average: the states are pocket-bound, and any model
built on the registry-averaged crystal starts 0.79 too high. This is the
quantitative form of the strong-coupling wall that sections 10 through 17 kept
hitting, now computable from monolayer data before any supercell is solved.

**Two models, one trade-off — the honest state of play.**

- Registry-adapted, k-resummed frames (ea_full): reproduces the floor at 3.0e-4 /
  6.7e-5 / 1.0e-4 in f across 7.63 / 3.95 / 2.01 deg. But it OVER-SPREADS the
  tower: at 2.01 deg its lowest 14 rungs span 0.0084 in f where FDFD's span
  0.0040 (ratio 2.1). Raising the envelope box from n_max 6 to 8 barely moved it
  (median 1.1e-2 -> 8.4e-3), so this is not trial truncation.
- The cause is symmetry, and it is diagnosed exactly. FDFD's tower is built from
  4-fold degenerate quartets — the X + X' four-fold of sections 13 through 15.
  Making the trial set C4-closed (the section-14 fix; C4 acts on envelope
  harmonics as n -> M n + n0 with M = A^T C4 A^-T, n0 = A^T (X'-X)/2pi, both
  integer, M^4 = I, verified) is NECESSARY but not sufficient here: the
  parallel-transport gauge used to smooth the adapted frames itself picks a
  direction, so the trial span is not C4-covariant and the quartets still split.
- The gauge-free model (registry-AVERAGED k-dependent frames, registry entering
  only through the interlayer hop matrix) has no gauge freedom and gives EXACTLY
  degenerate rungs (0.00000, 0.00000, 0.00008, 0.00008, ...) — symmetry perfect —
  but its floor is 1.3e-2 too high, because averaged frames cannot reach the
  pockets.

So on this crystal the accurate-floor model and the exactly-symmetric model are
currently two different models. The missing piece is a C4-covariant gauge for the
registry-adapted frames (or, equivalently, a symmetry-projected trial space in the
spirit of stage1_c4proj from section 15). That is a well-posed, bounded next step,
not a new unknown.

**What this settles.** The thesis crystal at its original parameters is a genuinely
strong-coupling system (V/gap ~ 1) — the difficulty was never an implementation
defect, and the completed envelope theory places its floor within MPB's own
discretization error while running where MPB and FDFD cannot. What it does not yet
do on THIS crystal is reproduce the symmetry-protected four-fold tower, for a
reason that is now identified precisely rather than guessed.

Machinery: thesis_port.py (material, box/C4-closed trial sets, c4_map, ea_solve
gauge-free, ea_full registry-adapted with rho/frame caches), thesis_refs.py (FDFD
with inertia census, MPB leg), fig_thesis_ladder.py. Data local: thesis_fdfd_*.npz,
thesis_mpb_15_r24.npz, thesis_ea_*.npz, thesis_eav2_57_n6.npz.
