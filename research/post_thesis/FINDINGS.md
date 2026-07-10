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
