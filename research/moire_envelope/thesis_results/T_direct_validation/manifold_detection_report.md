# Manifold Detection for the Multi-Band Envelope Approximation

## Status

- [x] Physics bottleneck identified: the remaining failure is no longer the Berry/Hermiticity machinery itself.
- [x] Physics distinction clarified: multiband EA is designed for strong mixing inside a retained manifold, not for arbitrary coupling to omitted bands.
- [x] Implementation direction chosen: build explicit manifold-detection machinery before pushing further EA-vs-FDFD validations.
- [x] Part 1/3 started: add a dedicated Phase 1 high-symmetry sweep runner that produces reusable inputs for manifold diagnostics.
- [x] Part 2/3 implementation started: a manifest-driven diagnostics suite now consumes the Phase 1 scan outputs.
- [x] First quick square/TM scan completed at reduced resolution for Γ, X, and M.
- [x] First Part 2 diagnostics pass completed on that quick scan.
- [ ] Part 3/3: rank manifolds, produce a top-table / green-light report, and only then proceed to targeted EA validations.

## Core Physics Insight

The key distinction is **mixing within the retained manifold** versus **leakage into omitted states**, and the theory only guarantees the first.

Yes: the multiband theory is friendly to multiple bands and to band mixing, but only after the correct subspace has been chosen.

The derivation already says this, implicitly. The ansatz expands the field in a selected set of local Bloch states at a fixed carrier momentum $k_0$, with slowly varying envelopes multiplying that moving frame. The non-Abelian Berry connection is precisely there to handle mixing **inside** that chosen $N$-dimensional subspace. And the Born-Huang term is the object that measures coupling from that retained subspace into the omitted complement. So the theory is not single-band in disguise. It absolutely allows internal mixing. But it is not a theory of an arbitrary list of eigenstates. It is a theory of a smooth, approximately closed projector $P(R)$.

That is why the answer to the first question is:

- [x] We cannot arbitrarily choose the subspace.

More precisely, we cannot arbitrarily choose a set of bands and expect a low-order envelope truncation to remain controlled. The retained subspace has to satisfy a physical closure condition: as $R$ varies across the moiré cell, the states inside the subspace may rotate and mix strongly among themselves, but they should not mix comparably strongly with states outside the subspace. If they do, then the omitted sector is not perturbative, and the effective Hamiltonian is no longer a controlled truncation. In the derivation, that is exactly the regime where the Born-Huang and Löwdin-type corrections stop being corrections and start signaling that the projector itself is wrong.

So “multiband” does **not** mean “any bands you want.” It means “any physically coherent manifold you want.” Those are different.

## Why This Restriction Exists

There are three layered reasons.

### 1. Smooth moving-frame requirement

The whole construction assumes a smooth moving frame $u_n(r;R)$ over the moiré cell. If the chosen states exchange identity with outside bands as $R$ changes, then there is no globally smooth $N$-dimensional bundle to work with. One can still force a numerical basis, but then the Berry connection and effective tensors are describing a projector that is itself unstable.

### 2. The truncation is perturbative in omitted-state leakage

The effective Hamiltonian is a truncation in two senses at once:

- [x] low order in $\eta = a/L$
- [x] low order in virtual excursions outside the retained subspace

The Berry term keeps exact track of **in-subspace** mixing. The Born-Huang and mass corrections only summarize **out-of-subspace** coupling perturbatively. If outside bands come too close, the denominators in the effective-mass / Löwdin structure cease to be large, and the truncated model loses control.

### 3. The EA is local in $k$

The expansion is local in $k$. We are expanding around a chosen carrier momentum $k_0$, not building a global theory for the entire Brillouin zone. The mass tensor and drift are extracted from local derivatives near $k_0$. So the EA should be thought of as a valley-centered or extremum-centered theory, not a universal all-frequency all-$k$ theory.

That means the statement

- [x] We cannot choose arbitrary frequencies at arbitrary $k$-points with one fixed EA.

is basically correct, with one refinement: it is not forbidden in principle, but it is not what a single low-order EA is for. A given EA is tied to:

- [x] a chosen carrier momentum $k_0$
- [x] a chosen energy window
- [x] a chosen smooth manifold of local Bloch states around that $k_0$
- [x] a scale-separation assumption that the resulting envelopes vary slowly on the moiré scale

If other frequencies or another region of the band structure are needed, one usually builds another effective model around a different $k_0$, a different manifold, or a patchwork of local models.

## How to Identify the Correct Manifold

Physically, the correct manifold is the one that behaves like a closed family under the slow moiré modulation. In practice, the following criteria should be enforced.

### 1. Projector smoothness over $R$

- [ ] Overlap-based tracking between neighboring registry points should remain stable.
- [ ] The Wilson-link / Berry construction should be smooth after gauge fixing, without large non-physical jumps.
- [ ] Band identity should not repeatedly swap with omitted bands as $R$ varies.

This is exactly the logic behind the overlap-transport formulation for the non-Abelian Berry connection.

### 2. Spectral separation from omitted bands across the full moiré cell

- [ ] The relevant quantity is not the band index at one reference point.
- [ ] The relevant quantity is the **minimum outside gap over the full moiré cell**.
- [ ] That gap must be judged relative to the modulation scale and coupling scale, not just reported raw.

In the current 10° square run, the diagnostic already shows why the existing 4-band choice fails as a manifold: the isolation ratios are negative for all four retained bands, so the manifold is not spectrally closed.

### 3. Out-of-subspace leakage must remain perturbative

- [ ] Born-Huang magnitude should remain moderate compared to kinetic and potential scales.
- [ ] Enlarging the manifold should not radically redefine the low-energy modes.
- [ ] Mode content should not continually spill toward the boundary of the retained band set.

Born-Huang is the formal marker here, but the important quantity is not merely “small BH.” The meaningful question is whether omitted-band effects remain subleading compared to the in-manifold structure the EA is trying to keep exact.

### 4. The solved moiré modes should live in the intended sector

- [ ] Dominant-band weights should remain interpretable.
- [ ] Severe mode diffusion across the retained set is acceptable only if the retained set is itself isolated from the rest of the spectrum.
- [ ] If low modes are both highly mixed and poorly isolated from omitted bands, the manifold is wrong.

### 5. Stability under modest subspace enlargement

- [ ] Going from 1 band to 2 bands may improve the same target physics.
- [ ] Going from 2 bands to 4 bands should not qualitatively redefine the state family if the manifold is healthy.
- [ ] If enlargement changes the identity of the low modes, the original manifold was not closed.

This is the practical renormalization-style test. More bands are not automatically more correct.

## Implication for the Envelope Approximation

The implication is not that the EA is weak. The implication is that the EA is an **effective theory for manifolds**, not for isolated eigenvalues viewed one by one.

When the manifold is well chosen, multiband EA is stronger than single-band EA because it keeps the physically important internal mixing exactly in the retained space. But when the manifold is badly chosen, adding bands can make the model worse, not better, because one is no longer enlarging a closed subspace; one is opening a door to unresolved coupling with still more omitted states.

The correct physics slogan is:

- [x] Multiband EA handles strong **in-manifold** mixing nonperturbatively.
- [x] Multiband EA handles coupling to the **rest of the spectrum** only perturbatively.

That is why “more bands” is not automatically “more correct.”

## What the Current Square-Moiré Results Mean

- [x] The theory itself is not the issue.
- [x] The Berry/Hermiticity failure was an implementation problem and is now mostly resolved.
- [x] The remaining 4-band failure is not evidence against multiband theory.
- [x] It is evidence that the chosen $[1,2,3,4]$ family at that $k_0$ and twist is not a clean low-energy manifold for the modes being compared to FDFD.

So the right next move is not another large validation run. The right next move is to build the machinery that tells us, for a given bilayer geometry, which manifolds are physically legitimate candidates for an EA in the first place.

## Implementation Plan

### Part 1/3: Phase 1 high-symmetry manifold harvest

- [x] Create a dedicated runner that executes Phase 1 MPB sweeps for a chosen bilayer geometry at all requested high-symmetry points.
- [x] Allow the user to request a broad retained band window, rather than a single preselected target band.
- [x] Save one reusable Phase 1 dataset per high-symmetry point, with full stencil data, registry shifts, frequencies, group velocities, effective masses, and optional Bloch fields.
- [x] Save a machine-readable manifest describing the high-symmetry points, chosen bands, and produced Phase 1 files.
- [ ] Run this routinely for the first 8–12 bands of each relevant lattice / polarization combination.
- [x] First concrete quick run completed for square / TM with 12 exported bands and reduced resolution.

This step is the raw data acquisition layer. Once this exists, the expensive MPB work is organized around manifold detection rather than only around direct EA runs.

### Part 2/3: Manifold-health diagnostics suite

- [x] Enumerate candidate manifolds from the harvested Phase 1 outputs.
- [x] Measure spectral isolation over the full moiré cell, not just at one reference point.
- [x] Measure projector smoothness from overlaps / Wilson links.
- [x] Measure Born-Huang leakage for each candidate manifold.
- [x] Measure in-manifold mixing via Berry / velocity / mode-coupling summaries.
- [x] Tag extrema character at each high-symmetry point using reconstructed $v_g$ and $M^(-1)$.
- [x] Expose whether each manifold is likely healthy, marginal, or red-flagged via a first-pass heuristic health label.

This is the real physics gatekeeper. Matrix assembly here is comparatively cheap once the Phase 1 data exists.

### Part 3/3: Manifold ranking and green-light table

- [ ] Build a ranked table of candidate manifolds for a given bilayer.
- [ ] For each manifold, report:
  - [ ] high-symmetry point
  - [ ] central frequency / frequency window
  - [ ] retained bands
  - [ ] whether the point is an extremum
  - [ ] spectral isolation metrics
  - [ ] Born-Huang magnitude and leakage ratios
  - [ ] intra-band and inter-band mixing summaries
  - [ ] manifold stability under modest enlargement
  - [ ] final traffic-light classification
- [ ] Emit a top-10 table and a concise “green light” summary for the bilayer.

This is the operational endpoint: a reliable manifold report that says whether a given bilayer is ready for a controlled EA, and if so, which manifold should be used.

## Immediate Working Goal

- [x] Focus now on Parts 1–3 above, not on grand validation runs.
- [x] Start with Part 1: build the reliable Phase 1 harvesting machinery for manifold detection.
- [ ] After Part 1 is stable, add the diagnostic suite that confirms or denies the manifold-closure claims directly from the data.

## Deliverables For Part 1

- [x] Dedicated markdown report: this file.
- [x] Dedicated Phase 1 runner for manifold-oriented high-symmetry sweeps.
- [x] First example run for a concrete lattice / polarization / band window.
- [ ] Manifest format stable enough that Part 2 can consume it without ad hoc path logic.
