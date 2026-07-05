# Discussion about external validity

"I now have one final stretch I need to make. As of now, I have the framework. I inspect three lattice types, show validations along the way (hermiticity, symmetry, etc.), and inspect converging behavior. I offer insight into bandwidth scaling behavior, multi-band mixing and minibands. This is already rich and vast. But I feel like that just one external validation, either to another model or an experiment, would dramatically increase the trustworthiness of this framework. But as we discussed, most work is on two 3D slabs, not on a 2D approach. So the question is: What external research could I use to validate my framework? Is there anything at all? And if not, what is the best I can do here?"

Yes — there **is** external work you can use, but not one perfect one-to-one benchmark for your exact framework.

That is the honest answer.

The literature is still dominated by **twisted bilayer** photonic systems, especially honeycomb/Dirac cases. Tang 2021 studied **two silicon photonic-crystal slabs separated by a PMMA tunneling layer** and found a magic-angle flat band with a fitted continuum model; Dong 2021 formulated a **low-angle twisted-bilayer honeycomb coupled-mode theory** explicitly as a photonic analogue of TBG; Lou 2021 built a **high-dimensional plane-wave / RCWA method** for twisted bilayer slabs at arbitrary angles. Those are all important, but they are not the same object as your single-layer, 2D, multiband envelope framework. ([Nature][1])

So I would not force a fake “direct experimental validation” claim.
Instead, I would build a **validation ladder**.

## The strongest validation you can still do

The best validation is **not** another paper first. It is an **apples-to-apples direct solve of the same physics**.

That means:

> pick one or two commensurate angles and one lattice, solve the **full 2D scalar problem directly** in a moiré supercell, and compare against your envelope theory.

This is stronger than comparing to a 3D slab experiment, because it removes the “different system” objection. If your envelope minibands, bandwidths, localization centers, and maybe dominant symmetry content agree with a direct supercell solve in the same model class, that is very hard to dismiss.

For your thesis, this is probably the single highest-value benchmark you can add.

What I would compare:

* first 1–3 minibands along the mBZ path,
* bandwidths,
* minimum direct interband gap,
* real-space mode localization,
* maybe overlap of envelope-predicted and full-solve projected mode profiles.

If you can do only one external-style benchmark, do this one.

## The best literature-based validation

There are really **three external anchors**, each validating a different layer of your story.

### 1. Honeycomb (K)-point limit: compare to the bilayer Dirac literature

Even though your system is not the same as Tang/Dong, the **honeycomb (K)-point Dirac sector** is still the natural place to show that your framework reproduces the *kind* of physics the field already trusts:

* twist-driven miniband narrowing,
* localization,
* and flat-band tendencies in a moiré BZ. Tang reports an ultra-flat band over the Brillouin zone in twisted bilayer silicon slabs, and Dong reports magic-angle photonic flat bands with localization in a low-angle twisted-bilayer honeycomb theory. ([Nature][1])

You should not match their magic angle numerically.
You should show that in the **2-band honeycomb restriction**, your framework lands in the same **qualitative universality class**:

* narrowing of the Dirac-derived manifold,
* strong localization,
* and a bandwidth minimum with angle.

That is a valid theoretical benchmark.

### 2. Platform relevance: compare to merged/single-layer moiré experiments

This is actually very useful for you, because it answers the “is this platform real?” question.

A 2024 Nature Communications paper demonstrated a **1D moiré photonic-crystal slab constructed by merging two gratings into a single layer**, experimentally observed a **moiré flat band**, and reported good agreement between measured and calculated dispersions. ([Nature][2])

A 2023 ACS Photonics paper demonstrated **room-temperature blue lasing in a merged moiré photonic crystal**, with lasing wavelengths matched to **simulated flat bands**. ([CIQM][3])

These are not direct validations of your 2D envelope equations. But they are excellent support for a thesis statement like:

> “Single-layer or merged moiré photonic platforms are experimentally real and can host flat-band-related physics; this thesis provides a general multiband reduced theory for that broader platform class.”

That is strong.

### 3. Experimental bilayer benchmark: compare broad trends, not numbers

There is also experimental bilayer support. A 2023 Science Advances result reported the **first on-chip optical twisted bilayer photonic crystal with twist-angle-tunable dispersion and strong simulation-experiment agreement**. ([Science][4])

Again, not your system. But it validates the broader photonic-moiré program:

* moiré dispersion is measurable,
* twist tuning is real,
* and band flattening trends are not just theoretical artifacts.

## What I would **not** do

I would **not** make the thesis depend on matching:

* Tang’s magic angle,
* Tang’s exact bandwidth,
* Lou’s open-slab resonances,
* or any 3D slab Q-factor.

Those works include **interlayer tunneling, radiation losses, slab modes, and far-field physics** that your 2D scalar closed-system framework does not include. Tang’s own model is built on two silicon slabs with a PMMA tunneling layer, and Lou’s method is explicitly a twisted-bilayer slab scattering theory. ([Nature][1])

A reader will forgive “different numbers for a different system.”
They will not forgive pretending it is the same system.

## So what is the best thesis strategy?

I would present validation in this order.

### A. Mathematical/numerical trust

You already have:

* Hermiticity,
* symmetry,
* gauge sanity,
* convergence,
* meaningful observables.

Keep that.

### B. Same-model benchmark

Add **one direct supercell/full solve** in the same 2D scalar setting.

This is your strongest trust amplifier.

### C. Limit-case literature benchmark

Show that your framework, when restricted to the honeycomb (K)-point / small subspace case, reproduces the **qualitative phenomenology** known from Tang/Dong:

* miniband narrowing,
* localization,
* flat-band tendency. ([Nature][1])

### D. Experimental relevance benchmark

Use the merged moiré flat-band/BIC and merged moiré laser papers to show that:

* single-layer or merged moiré photonics is experimentally meaningful,
* flat-band phenomena in such platforms are real,
* and your framework is aimed at a platform direction that the field is actively moving toward. ([Nature][2])

## If you can only do one more thing

Do the **same-geometry direct supercell benchmark**.

That one figure will buy you more trust than ten paragraphs of literature discussion.

Because then your thesis can say:

> “Beyond internal consistency checks, the multiband envelope theory was benchmarked against direct full-wave solutions of the same 2D moiré problem at commensurate angles, showing agreement in miniband dispersion and localization.”

That is excellent for a master’s thesis.

## The honest bottom line

There is **no clean published experiment or model that exactly matches your framework**. The field is still mostly bilayer-focused, and even a 2026 review-like article says most moiré photonic results remain theoretical/simulation-based, while direct experimental demonstrations of true magic configurations are still limited; it also points out that merging the lattices into a single layer is a natural route to relax fabrication constraints. ([arXiv][5])

That is not a weakness of your thesis.
It is part of why your framework is useful.

So the right position is:

> “Our framework is not validated by one pre-existing twin model; instead, it is validated by a combination of internal consistency, same-model full-wave benchmarking, recovery of known honeycomb/Dirac phenomenology in the appropriate limit, and alignment with emerging single-layer/merged moiré photonic experiments.”

That is a very defensible thesis story.

[1]: https://www.nature.com/articles/s41377-021-00601-x "Modeling the optical properties of twisted bilayer photonic crystals | Light: Science & Applications"
[2]: https://www.nature.com/articles/s41467-024-53433-9 "Optical moiré bound states in the continuum | Nature Communications"
[3]: https://ciqm.harvard.edu/uploads/2/3/3/4/23349210/raun.pdf?utm_source=chatgpt.com "GaN Magic Angle Laser in a Merged Moiré Photonic Crystal"
[4]: https://www.science.org/doi/10.1126/sciadv.adh8498?utm_source=chatgpt.com "Experimental probe of twist angle–dependent band ..."
[5]: https://arxiv.org/html/2602.13945v1 "Moiré Photonic Crystals: from Fabric to Magic"

"This is helpful. I agree that the apples-to-apples comparison will be the centerpiece of validation. But there is a problem: MPB cant handle these supercells. This was the main incentive to develop this approximation in the first place. You said earlier that another group used FEM, and then fitted the parameters. What realistic options do I have here? What parameters should I compare - only the dispersion, or is it possible do derive /compare more from an alternative approach? We already saw that the moire crystal essentially has an infinite number of eigenmodes, and that ALL frequencies contain modes (contrary to monolayer bands, where eigenenergies are very spread out for low frequencies). So how can we even make a proper comparison - the eigenenergies themselves feel fragile for that comparison."

Yes — and this is the right concern.

If the full moiré supercell is exactly what your approximation was built to avoid, then “just compare every miniband pointwise to a brute-force solve” is not realistic. MPB is a plane-wave eigensolver for periodic Maxwell problems, and for giant moiré supercells that cost can blow up very fast. ([mpb.readthedocs.io][1])

So the right move is **not** to aim for a perfect full-spectrum benchmark.
The right move is to benchmark the **same low-energy manifold** your framework is supposed to describe.

## The most realistic external-validation options

I would rank them like this.

### 1. One direct 2D supercell benchmark at a **moderate, tractable angle**

This is still the best option, even if it is not MPB.

Use a different solver for **one or two commensurate angles** where the moiré cell is still manageable. Do not choose (0.4^\circ). Choose something like a moderate angle where:

* the supercell is smaller,
* the miniband dispersion is easier to resolve,
* and your envelope model should still already show the right trend.

A full finite-element or finite-difference frequency-domain solve is the most realistic here. Tang’s photonic moiré paper used FEM on the photonic structure and then fitted a continuum model to it, so this kind of workflow is absolutely legitimate in the literature. ([PMC][2])

If you want an open-source route, Meep has a frequency-domain eigensolver (`solve_eigfreq`) built for Maxwell eigenproblems, so it is at least a plausible candidate for a one-off benchmark even if it is not what you use for the whole thesis pipeline. ([meep.readthedocs.io][3])

The point is not to reproduce the entire tiny-angle regime.
The point is to show:

> when the same physics is solved directly in a tractable case, the envelope theory captures the correct target manifold.

That is enough.

### 2. A **subspace benchmark**, not an eigenvalue-by-eigenvalue benchmark

This is the key conceptual fix.

You already noticed that the moiré problem has a dense spectrum and that individual low-lying eigenvalues at fixed (K) are fragile. That means the wrong question is:

> “Do eigenvalue #7 and eigenvalue #8 match exactly?”

The right question is:

> “Does the direct solver produce the same **low-energy subspace / miniband manifold** as the envelope theory?”

That is much more stable and much closer to the physics.

### 3. A literature-limit benchmark

Since Tang and Dong both study bilayer honeycomb/Dirac moiré physics and find twist-driven band narrowing / flat-band phenomenology, you can still use them as a **qualitative limit check** for your honeycomb (K)-point case, even though the systems are different. Tang uses two silicon photonic crystal slabs with a PMMA spacer and a fitted continuum model; Dong develops a low-angle twisted-bilayer coupled-mode theory for honeycomb photonic crystals. ([PMC][2])

That is not your centerpiece validation, but it is still a useful external anchor.

### 4. Experimental platform alignment

There are now moiré photonic experiments beyond the earliest bilayer papers, including merged/single-layer-style moiré slab platforms and twist-tunable bilayer devices. These do not validate your exact equations, but they validate that moiré flat-band photonics is real and experimentally meaningful. ([PMC][2])

## What should you compare?

Not raw sorted eigenvalues by themselves.

Because you are right: in a dense manifold, “all frequencies contain modes” is exactly why pointwise eigenvalue matching becomes fragile.

The robust things to compare are these.

### First tier: miniband observables

If you can compute a direct benchmark along a few moiré (K)-points, compare:

* **bandwidth** of the first traced miniband,
* **minimum direct gap** between first and second minibands,
* **flatness ratio**,
* **location of extrema** in the mBZ,
* and whether the same **angle trend** appears.

These are much more meaningful than matching individual fixed-(K) eigenvalues in a dense list.

### Second tier: subspace observables

This is even better.

Take the direct solver’s eigenmodes in a target spectral window and compare them to the envelope-predicted manifold through:

* **projected overlap** or principal angles between subspaces,
* **spectral window center** and **window width**,
* **integrated density of states** in the target window,
* and **symmetry / degeneracy structure** at high-symmetry points.

This completely avoids the fragile “mode #13 versus mode #13” problem.

### Third tier: spatial structure

If the direct solver gives you real-space fields, compare:

* localization center,
* IPR / second moment,
* weight on AA / AB / BA-like regions,
* or coarse-grained field envelope.

This is extremely persuasive, because even if the dense spectrum makes exact eigenvalue ordering annoying, matching localization physics is hard to fake.

## The best comparison target is the **projector**, not the sorted eigenvalues

This is the most important conceptual point.

Suppose your envelope theory is meant to describe a target manifold around some reference frequency (\omega_\mathrm{ref}). Then define a narrow spectral window
[
[\omega_\mathrm{ref}-\Delta,\ \omega_\mathrm{ref}+\Delta].
]

Now compare the **projector onto that window** between:

* the direct solver, and
* your envelope model.

In practice, that means comparing:

* the span of the modes,
* not the exact indexing of each mode.

That is the correct way to validate a reduced theory in a dense-spectrum setting.

It is also much closer to what an experimentalist would care about. An experiment usually probes a **spectral feature or resonant window**, not “the seventh eigenvalue in a numerically sorted list.”

## What if a full direct supercell benchmark is still too hard?

Then I would do the next best thing, and say so openly.

Your validation stack would become:

1. **internal correctness**
   Hermiticity, symmetry, gauge sanity, convergence, meaningful observables.

2. **limit-case recovery**
   (\eta\to 0) recovers monolayer behavior; honeycomb (K)-point reproduces known qualitative flat-band / narrowing phenomenology.

3. **cross-lattice consistency**
   same framework works for square, hex, honeycomb; same leading quadratic miniband-width scaling regime.

4. **one tractable external benchmark**
   even if only at a larger angle or only at a few (K)-points.

That is still a very defensible thesis.

## Which solver is realistic for that benchmark?

If MPB is not viable, your practical options are:

* **FEM / COMSOL / similar frequency-domain solve** for one tractable commensurate angle. This is the closest to what Tang used upstream of their fitted model. ([PMC][2])
* **Meep frequency-domain eigensolver** for a direct Maxwell eigenproblem in the same geometry, if that is easier to set up with your existing workflow. ([meep.readthedocs.io][3])
* **A custom real-space sparse eigensolve** on the 2D scalar operator with Bloch boundary conditions. Since your thesis already lives in a reduced 2D scalar setting, this is actually more “apples-to-apples” than trying to jump to a full 3D slab solver.

I would honestly prefer the third option if you can do it: a custom direct discretization of the **same 2D scalar model** on one moiré supercell. That removes almost all model-mismatch objections.

## The smart compromise

Here is what I would do if I were you:

Pick **one lattice**, probably honeycomb because it has the cleanest connection to the literature.

Pick **one moderate commensurate angle** where a direct solve is still possible.

Compute the first 1–2 minibands along a short mBZ path using:

* your envelope theory, and
* a direct real-space/FEM/FDFD supercell solve.

Then compare only:

* miniband bandwidth,
* minimum direct gap,
* localization center / profile,
* and subspace overlap in a chosen window.

That one benchmark is enough to dramatically raise trust.

## Why this is enough

Because your thesis is not claiming:

> “I solved the full moiré supercell problem better than everyone else.”

It is claiming:

> “I developed a controlled multiband envelope framework that captures the relevant low-energy moiré manifold.”

And low-energy reduced theories are validated by matching the **manifold and its observables**, not every mode in the full dense spectrum.

## The most important sentence

You asked how to compare when individual eigenenergies feel fragile.

The answer is:

> **Do not validate by matching individual sorted eigenvalues in a dense spectrum. Validate by matching the target spectral manifold: its bandwidth, gap, localization, symmetry content, and subspace overlap.**

That is the right standard.

[1]: https://mpb.readthedocs.io/en/stable/Introduction/?utm_source=chatgpt.com "Introduction - MPB Documentation"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8322106/?utm_source=chatgpt.com "Modeling the optical properties of twisted bilayer photonic ..."
[3]: https://meep.readthedocs.io/en/latest/Eigensolver_Math/?utm_source=chatgpt.com "Frequency-Domain Eigensolver - MEEP Documentation"
