# Post-Thesis Findings Log

## Session 2026-07-05 — infrastructure + A1 + A2 (V4 path) + gauge diagnosis

### A1 — Golden benchmark reproduced ✅

`thesis_results/T_direct_validation/plot_definitive_1deg.py` re-run from the
committed η-sweep data (`runsV3/thesis_honeycomb_K_b1_20260307_171424/
eta_sweep_20260310_191610`) + archived FDFD reference
(`fdfd_dirac_m30_n29_res40_v2.npz`):

- θ = 1.1213°, η = 0.01957, (m,n) = (30,29)
- **50/50 envelope modes uniquely Hungarian-matched to FDFD**
- mean |Δω| = 23×10⁻⁶ = **0.8% of miniband bandwidth**; max 130×10⁻⁶
- 46/50 within one mean level spacing; 49/50 within two
- Born–Huang: shifts eigenvalues by 8% of BW, improves residual 25→23×10⁻⁶
- FDFD window contains 64 modes; the 16 unmatched ones belong to other
  folded bands (correctly absent from the 2-band EA)

The headline result of Result line A exists and reproduces bit-for-bit.

### Blaze-side ingredient verification ✅

Fresh `OperatorDataExtractor.extract` on the golden crystal (honeycomb rods
ε = 11.56, r/a = 0.2, air, TM) at the true K corner (`k0_frac = (1/3, 2/3)`
for lattice vectors [[1,0],[0.5,√3/2]]):
**bands 0–1 degenerate at ω_D = 0.2744** — exactly the MPB-V3 value.
Registry landscape from the reg-64 sweep: band-0 ω ∈ [0.2436, 0.2712],
matching the V3 production run (ω_center ≈ 0.2428 at AA).

### A2 — remote-band ladder, V4-raw path (registry 64, BH on) ⚠️

Four Phase-1 registry sweeps (n_remote ∈ {0,4,8,16}, 2 retained, reg 64,
res 64, Born–Huang active — `A_triple_match/phase1_nrem*/`) completed in
~30 min total. Envelope solves at θ = 1.1213° (Ns = 64, 50 lowest modes)
Hungarian-matched against the FDFD res-40 reference
(`A_triple_match/a2_ladder_results.json`):

| n_remote | mean \|Δω\| | max \|Δω\| | within 1 spacing | BW ratio |
| --- | --- | --- | --- | --- |
| 0 | 926×10⁻⁶ | 2321×10⁻⁶ | 11/50 | 0.276 |
| 4 | 914×10⁻⁶ | 2487×10⁻⁶ | 12/50 | 0.432 |
| 8, 16 | *(rerunning — first pass failed silently behind a grep pipe)* | | | |

n_remote 0→4 shifts eigenvalues by mean 91×10⁻⁶ (max 189×10⁻⁶) — the Löwdin
dressing is a ≥4× larger effect than the golden benchmark's total residual
(23×10⁻⁶), i.e. **remote-band completeness genuinely matters at the
achieved accuracy level**, confirming the professor's point quantitatively.
The absolute match is nevertheless dominated by the gauge problem below.

**Key negative finding — gauge noise, exactly as S3 predicted:**
the V4-raw pipeline (no gauge fixing, no point-group symmetrization) is
NOT sufficient for Dirac manifolds. The off-diagonal Berry connection in
the raw ladder data spikes where the local gap closes:

- |A₀₁ₓ|: median 1.52, p95 5.5, **max 55.8**
- in the smallest-5% gap region: median 6.13 vs 1.48 elsewhere

Consequently the V4-raw envelope spectrum at θ = 1.12° is compressed
(BW ratio ≈ 0.43 vs FDFD) and the match degrades to ~10⁻³ — two orders
worse than the gauge-fixed V3 golden result. This *positively confirms*
that the V3 corrections arc (S3 parallel transport + S4b C6 symmetrization)
is a necessary ingredient, not a refinement.

**Implication for the thesis-era magic-angle sweep:** the blaze V4 sweeps
(`dirac_sweep`, `magic_angle_hunt`) ran on this same raw path (plus 0 remote
bands, BH zeroed at the time). Their sub-1° behaviour is therefore doubly
unreliable — first numerics-at-the-floor, now also unsymmetrized gauge noise
in the strongest-coupling regions.

### The definitive A2/B recipe (next session)

1. **A2-definitive:** run the remote-band ladder through the **V3-MPB
   pipeline** (gauge-fixed + S4b): copy
   `configsV3/thesis_honeycomb_K_b1.yaml` with `mpb_registry_samples: 64`
   and `n_extra_bands ∈ {8, 16}` (the golden run at reg 128 / n_extra 4 is
   the baseline rung); per rung: phase1 → phase2 → S4b → η-sweep at
   θ = 1.1213 → Hungarian vs `fdfd_dirac_m30_n29_res40_v2.npz`.
   ~30–45 min per rung, background.
2. **A3:** same protocol at the commensurate ladder
   (8,7) 4.41° / (15,14) 2.28° / (21,20) 1.61° / (30,29) 1.12° / (39,38)
   0.85° — FDFD references partly exist (`fdfd_dirac_m8_n7_res20.npz`, …);
   missing ones at 1–4 px/a are cheap. Assemble error-vs-η curve.
3. **B1/B2:** run miniband q-dispersion + v*(θ) on the **existing
   gauge-fixed golden phase-2 data** (`runsV3/thesis_honeycomb_K_b1_…`),
   not on V4-raw. `B_magic_angle/b1b2_sweep.py` contains the observable
   logic; port its k_s-loop to the V3 phase-3 solver (which supports
   moiré Bloch phases — see T03 miniband tooling).

### Assets created this session

- `lib/` — vendored V4 engine, import-compat with post-May blaze
  (`EAExtractor` → `OperatorDataExtractor`)
- `A_triple_match/` — config (golden crystal, verified K corner), ladder
  runner, Hungarian matcher; 4× reg-64 Phase-1 datasets (local)
- `B_magic_angle/b1b2_sweep.py` — bandwidth + v*(θ) observables
  (awaiting gauge-fixed input)
- Environment: blaze dev rebuilt (`maturin develop --release`), import fixed
