# About interpolation and extrapolation of a monolayer stencil around a chosen k_0

Yes — this is a solid approach, with one important correction:

> **Interpolation inside the stencil is good.
> Extrapolation beyond the stencil should not be part of the core thesis pipeline.**

For thesis-grade results, I would treat extrapolation as exploratory only.
If you want to support a larger angle, the clean solution is to **enlarge the 2D (k)-patch** so the full moiré-BZ excursion is actually contained in the sampled region.

## My recommendation

* **5x5**: minimal, okay for proof of concept and strict local derivatives.
* **7x7**: best default for serious research and thesis production.
* **9x9**: good as a reference / convergence patch, or if you want to push toward larger angles or use smoother polynomial fitting.

So my practical advice is:

> **Use 7x7 as the production standard.
> Use 9x9 for one or two benchmark cases to show that 7x7 is already stable.**

That gives you a strong story without exploding compute and storage.

## Why 7x7 is the sweet spot

A 7x7 patch gives you enough room to:

* fit robust 2D local polynomials,
* compute first and second derivatives much more stably than with 5x5,
* support a finite moiré-BZ excursion plus a halo,
* and still keep the MPB cost under control.

A 9x9 patch is better if:

* Berry/Born–Huang data is noisy,
* you want smoother least-squares fits,
* or you want to cover a larger (\theta_{\max}) without rebuilding phase 1.

But 9x9 as the default for everything is probably not worth the storage unless you know you need it.

## One important correction to your “10%” idea

A patch targeted at **10% of the parent (k)-path scale** is good for about **(5^\circ)**, but **not enough for (10^\circ)**.

The basic moiré excursion scale is

[
f(\theta)=2\sin(\theta/2),
]

which is approximately (\theta) in radians for small angles.

That gives:

* (1^\circ \rightarrow 1.75%)
* (2^\circ \rightarrow 3.49%)
* (5^\circ \rightarrow 8.72%)
* (10^\circ \rightarrow 17.43%)

So if you want a patch that genuinely supports minibands up to (10^\circ), you need a support radius more like

[
20%-25%
]

of the relevant parent-BZ scale, because you need:

* the moiré excursion itself,
* plus a stencil/interpolation halo.

So:

* **10% patch radius** → good for about (5^\circ)
* **20–25% patch radius** → needed if you seriously want (10^\circ)

That is the right order of magnitude.

---

# Concise report: 2D (k)-space sampling for moiré minibands

## Purpose

The current pipeline builds a multiband envelope approximation around a parent-band expansion point (k_0). For monolayer analysis, high-symmetry paths are sufficient to identify candidate points such as extrema or Dirac crossings. However, for **moire miniband calculations**, the effective Bloch momentum (K) explores a **two-dimensional moiré Brillouin zone**, and this generally does **not** align with the parent high-symmetry path.

Therefore, a 1D parent-path sampling is not sufficient input for a fully 2D moiré miniband theory. The correct phase-1 input is a **local 2D patch in parent (k)-space** around the selected candidate (k_0).

---

## Core problem

The moiré BZ is smaller than the parent BZ, but it is still a **2D region**.
If the envelope Hamiltonian is solved over the moiré BZ, then the parent data is effectively queried in a neighborhood of

[
k_0 + K,
]

where (K) ranges over the moiré BZ.

This causes two requirements:

1. The phase-1 parent data must cover a **2D neighborhood**, not only a line.
2. The patch radius must be large enough to contain the **full moiré-BZ excursion** for the intended angle range.

---

## Geometric design rule

For two identical lattices twisted by angle (\theta), the moiré reciprocal-lattice scale is

[
|g| = 2|b|\sin(\theta/2),
]

so the relevant (k)-space excursion scales as

[
f(\theta)=2\sin(\theta/2)\approx \theta_{\rm rad}.
]

This gives the approximate fraction of the parent-BZ scale explored by the moiré BZ:

* (1^\circ): (1.75%)
* (2^\circ): (3.49%)
* (5^\circ): (8.72%)
* (10^\circ): (17.43%)

The required patch radius should therefore satisfy

[
r_{\rm patch} \gtrsim K_{\max}^{\rm mBZ} + r_{\rm halo},
]

where (r_{\rm halo}) accounts for derivative stencils and interpolation safety.

A practical rule is:

* for (\theta_{\max}\le 5^\circ): patch radius (\sim 10%-12%)
* for (\theta_{\max}\le 10^\circ): patch radius (\sim 20%-25%)

relative to the relevant local parent-BZ scale.

---

## Recommended stencil size

### Production choice

[
\boxed{7\times 7}
]

This is the recommended standard for thesis-quality data.

Why:

* robust 2D derivative estimation,
* enough support for interpolation,
* manageable storage and compute,
* clearly better than the current 5x5 minimal stencil.

### Reference / validation choice

[
\boxed{9\times 9}
]

Use this for:

* one or two convergence studies,
* larger-angle support,
* noisy Berry/Born–Huang fields,
* validating that 7x7 is already stable.

### Minimal choice

[
5\times 5
]

Acceptable only as a minimal local stencil, not ideal for general miniband work.

---

## How the stencil should be used

### 1. Sample a 2D patch, not a path

The patch should be centered at (k_0) and cover the intended angular validity range.

The patch should be used to sample:

* local eigenvalues (\lambda_n(k)),
* Bloch eigenvectors (u_n(k)),
* overlaps needed for Berry/Born–Huang objects.

### 2. Use interpolation inside the patch

Interpolation inside the sampled patch is fully reasonable and should be the default.

Preferred approach:

* perform a **2D least-squares polynomial fit** or local smooth fit on the patch,
* rather than relying only on raw finite differences from the nearest few points.

This is especially helpful for noisy derivative-derived objects.

### 3. Avoid extrapolation for core claims

Extrapolation beyond the patch should not be used for the central thesis results.

It may be used only for exploratory tests, and must be explicitly labeled as such.

If a target angle requires data outside the current patch, the correct action is:

* enlarge the patch,
* or restrict the claimed angle range.

---

## Deriving the effective objects from the patch

A 2D patch allows stable extraction of all key objects needed by the envelope theory.

### Group velocity / drift term

Derived from first derivatives:

[
v_i \sim \partial_{k_i}\lambda_n(k_0).
]

These should be obtained from the 2D fit rather than one-dimensional directional differences.

### Mass tensor / kinetic term

Derived from second derivatives:

[
(M^{-1})*{ij} \sim \partial*{k_i}\partial_{k_j}\lambda_n(k_0).
]

A 2D patch is essential here, because the mixed derivative

[
\partial_{k_x}\partial_{k_y}
]

cannot be recovered reliably from a single line.

### Berry connection

Derived from the local variation of the band frame:

[
(A_i)*{mn} \sim i\langle u_m(k)\mid \partial*{k_i}u_n(k)\rangle.
]

Because this object is gauge-sensitive and derivative-based, it benefits strongly from:

* a larger 2D patch,
* smooth gauge fixing / parallel transport,
* and least-squares derivative estimation.

### Born–Huang term

Likewise derived from projected derivatives of the local basis.
This is also numerically sensitive and should be computed from smoothed 2D data, not only from a minimal stencil.

---

## Best practical workflow

### Recommended production pipeline

* Use **7x7 patch** around each chosen (k_0)
* Choose patch radius based on the largest intended validation angle
* Fit local 2D polynomials to the patch data
* Derive:

  * velocities,
  * full mass tensor,
  * Berry matrices,
  * Born–Huang matrices
* Solve minibands only for angles whose moiré BZ lies fully inside the sampled patch

### Recommended validation pipeline

* Repeat one representative case with **9x9 patch**
* Show that:

  * mass tensor changes only weakly,
  * miniband bandwidths are stable,
  * Berry/Born–Huang norms are stable,
  * final eigenmodes and gaps do not materially change

That gives a very strong numerical story.

---

## Final interpretation

This refinement does **not** change the formal envelope theory.
It improves the **numerical support** of the implementation so that the moiré BZ is sampled consistently and all derivative-based tensors are extracted from proper 2D parent data.

The key principle is:

> High-symmetry paths are for identifying candidate points.
> Moiré minibands require a 2D parent-(k) patch around those points.

---

## Bottom-line recommendation

[
\boxed{\text{Use 7x7 as default, 9x9 as benchmark/reference.}}
]

And:

[
\boxed{\text{Interpolate inside the patch, do not rely on extrapolation for thesis claims.}}
]

And:

[
\boxed{\text{Choose the patch radius from the largest intended moiré-BZ excursion, not from the monolayer plotting path.}}
]
