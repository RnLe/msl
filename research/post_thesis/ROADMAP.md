# Post-Thesis Research: Where the Thesis Stopped, and the Last Push

**Created:** 2026-07-05 · **Status:** campaigns in progress (see FINDINGS.md)

This document is the honest reconstruction of where the thesis research ended
(from the submitted `validations.typ`, the sprint reports in
`../moire_envelope/thesis_results/`, and the V4 studies in the blaze2d repo),
and the concrete program to convert the two near-miss results into clean
scientific statements.

---

## 1. Where the thesis stopped

### 1.1 Result line A — EA vs FDFD vs MPB eigenvalue agreement

**What the thesis says:** aggregate agreement (θ² bandwidth scaling, spectral
envelope, mode density); "direct eigenvalue-by-eigenvalue validation against
FDFD remains an open challenge."

**What the record actually shows:** the challenge was already essentially
solved for the right system, days before submission:

- `thesis_results/validation_summary.md` (2026-03-11), honeycomb TM Dirac
  pair at θ ≈ 1.1213° ((m,n)=(30,29), 2611 cells):
  **50/50 envelope modes uniquely matched (Hungarian) to FDFD, mean |Δω| =
  23×10⁻⁶ = 0.8% of the miniband bandwidth (relative error ~1.2×10⁻⁴),
  BW ratio 0.985, KS p = 1.0, 46/50 within one mean level spacing.**
  Reproduced 2026-07-05 from the committed η-sweep data +
  `T_direct_validation/fdfd_dirac_m30_n29_res40_v2.npz`
  (`plot_definitive_1deg.py` — figures regenerate identically).
- The residual floor was shown to be **FDFD-grid-limited, not EA-limited**:
  the EA↔FDFD residual (23–28×10⁻⁶) is stable across FDFD res 12→40 while
  FDFD's own inter-resolution drift is 2–4× larger. FDFD was capped at
  res 40 (~4.2M DOF) by CHOLMOD's 32-bit indices.
- Born–Huang (field-based implementation, activated only in the final days)
  shifts eigenvalues by 8% of BW and *improves* the match by 6%.
- At θ ≈ 4.4° the EA shows 13% bandwidth compression — the error grows with
  η exactly as the two-scale expansion predicts. **The error-vs-η scaling
  curve was never assembled.** That curve, not a single-angle match, is the
  publishable statement.

**Why the thesis under-claimed:** the write-up leaned on the *square-lattice*
comparisons, which are genuinely hard cases: the TE Γ-point sits on the
spectral positivity boundary (λ_ref = 0 → negative/NaN envelope modes), the
drift term dominates and drives indefiniteness, and the exact-operator
remainder reaches 13–39% at small angle, surviving only via cancellation
(blaze2d `phasesV4/studies` exact audits). Those failures are real physics
limits of the truncated EA in that regime — but they say nothing against the
Dirac-manifold benchmark, which is the setting the multiband EA was built for.

### 1.2 Result line B — magic angles

**What the thesis says:** 153-angle bandwidth scan of the honeycomb Dirac TM
crystal — no bandwidth collapse; θ^1.15 scaling above 1.5°; erratic plateau
below 0.4°; structural argument that the magic condition lies outside the
two-atomic-approximation validity window *for this crystal*.

**What the record actually shows — three reasons the null is not conclusive:**

1. **Incomplete Hamiltonian.** The magic-angle sweep used the Dirac doublet
   with **0 remote bands** (no Löwdin dressing of the mass tensor), and every
   saved Phase-2 run of that sprint had **Born–Huang ≡ 0**. The separate
   remote-band scan (square TE) showed key observables converge only at
   **6–8 remote bands**. The professor's observation — computing 16–24 bands
   in Phase 1 costs almost nothing — directly repairs this.
2. **Numerics at the floor.** Mini-band bandwidths in the scan (3.5–8×10⁻⁵)
   sat at the discretization floor; the moiré grid was downsampled Ns 128→48
   for cost; the registry-convergence study demands reg ≥ 48–64; the sub-1°
   bandwidth up-turn seen for hex was explicitly suspected to be a σ/grid
   artifact and never resolved.
3. **Blunt observable.** RMS bandwidth over 20 modes × 9 k-points washes out
   the actual magic-angle criterion. In Bistritzer–MacDonald physics the
   magic angle is where the **renormalized Dirac velocity v\*(θ) crosses
   zero** — an observable the EA can compute directly from miniband slopes
   (and semi-analytically from its own ingredients) but which was never
   evaluated. The earlier "θ ≈ 0.7°" claim was a Γ′-point gap minimum
   (2.97×10⁻⁶ — *below* the numerical floor), correctly retracted.

Additionally, **candidate engineering never happened**: `dirac_search`
(blaze2d) already maps (r, ε) → clean Dirac cones across a wide family;
nobody computed the coupling-to-velocity ratio α(θ) across that family to
find a crystal whose magic angle lands *inside* the EA-valid, numerically
solid window ([~0.8°, 3°]).

### 1.3 Assets already in place

- FDFD at **1 px/a** validated to reproduce the low-frequency supercell
  spectrum exactly (thesis §"lower limit") → cheap independent references at
  θ < 1°, where no other brute-force method reaches.
- All conventions debugged and documented (`GEOMETRIC_CONVENTIONS.md`):
  corner-vs-center sampling, MPB |C₁| rescale, coincidence-cell fix.
- Blaze2D `OperatorDataExtractor` (formerly `EAExtractor`): Rust-parallel
  registry sweeps with velocity/mass/Berry/Born–Huang extraction,
  checkpointing, and a native `solve_k_path`.
- Verified today: the blaze extraction reproduces the golden crystal's
  Dirac point at ω_D = 0.2744 (bands 0–1 at K) — solver-level ingredient
  agreement between the MPB-V3 and Blaze-V4 pipelines.

---

## 2. The program

### Result A — "the EA is spectrally exact where it claims to be"

| Step | What | Status |
|---|---|---|
| A1 | Reproduce the golden θ=1.1213° benchmark from archived data | ✅ reproduced bit-for-bit |
| A2 | Remote-band ladder n_remote ∈ {0,4,8,16}, Born–Huang on, registry 64 — the professor's experiment, run on the golden system | running |
| A3 | Error-vs-η curve: commensurate ladder θ ≈ 4.41° → 0.85°, EA (best A2 settings) vs FDFD; per-mode Hungarian errors + BW ratio vs η with η² reference | planned |
| A4 | Drop the FDFD floor at 1.12° (int64 Cholesky / iterative) to res ≥ 48 | stretch |

**Claim shape:** "For an isolated Dirac manifold, the multiband EA reproduces
the full Maxwell spectrum mode-by-mode to ~10⁻⁴ relative accuracy at θ ≈ 1°,
with the residual scaling as η² and bounded by the reference solver's own
grid error."

### Result B — "engineered magic angles, or a first-principles null"

| Step | What | Status |
|---|---|---|
| B1 | Repaired θ sweep (2 ret + 8 rem, BH on, reg 64, Ns 48/96 cross-check), 0.3°–3° | planned |
| B2 | v\*(θ)/v_D from miniband slopes + BM-style α(θ) from Phase-2 term norms | planned |
| B3 | Candidate engineering: α-ratio across the (r, ε) Dirac family → θ_magic prediction map → pick candidates with θ_magic ∈ [1°, 3°] | planned |
| B4 | Dense sweep on best candidate; FDFD 1 px/a cross-check at bracketing commensurate angles | planned |

**Claim shapes:** either "single-layer photonic moiré crystals exhibit a magic
angle at θ_m = X°, predicted from first principles and confirmed by an
independent supercell solver," or "for single-layer moiré, the Berry-only
coupling mechanism places all magic angles outside the envelope validity
window — a structural difference from bilayer photonic crystals," backed by
the α(θ) map instead of a noisy scan.

---

## 3. Layout

```text
post_thesis/
├── ROADMAP.md          ← this document
├── FINDINGS.md         ← quantitative results as they land
├── lib/                ← vendored V4 engine (origin: blaze2d phasesV4, see lib/__init__.py)
├── A_triple_match/     ← Result A campaigns (configs, run scripts, outputs)
└── B_magic_angle/      ← Result B campaigns
```

Data policy identical to the rest of the repo (`../DATA.md`): raw npz/h5
outputs local-only, summary tables/json + reports committed.

---

**Correction (2026-08-27, see FINDINGS section 18):** the 1.1213-degree "golden
benchmark" of section 1.1 is retired as eigenvalue-level evidence: the EA and FDFD runs
were in different supercell Bloch sectors (K folds to (1/3,2/3), the references are at
Gamma; the sectors differ by ~7 level spacings), the 23e-6 Hungarian mean carries no
alignment information against a spacing-preserving null, and the underlying geometry code
is not stable at that scale across versions. The mode-density/bandwidth-scale agreement
stands; the state-identity claim does not. Result line A is therefore open again and is
being pursued on the corrected v5 pipeline (smooth-bilayer validation ladder first).
