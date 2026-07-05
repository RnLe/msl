Yep — what you’re describing (“Bloch phase looks random”, eigenvectors “rotate wildly” under tiny geometry changes, warm-start across **R** is worse than random) is *exactly* what happens when the eigenmodes are being compared in an **inconsistent gauge / coordinate frame**, *plus* (often) some **near-degeneracy mixing**.

The good news: this doesn’t automatically mean the envelope theory is wrong. It very often means:

1. the *mathematics is gauge-covariant*, but
2. the *numerics are not enforcing a smooth gauge*, so anything involving (\partial_{\mathbf R}u), Berry connections, projections, warm starts, etc. becomes garbage.

Let’s go through your points carefully and connect them to what must be fixed.

---

## 1) “Bloch phase between cells is random” — that’s not physics, that’s gauge

For each frozen (\mathbf R), the local Bloch eigenproblem defines (u_n(\mathbf r;\mathbf R)) only up to a **unitary transformation inside the eigenspace**:

* single, isolated band: (u_n \to e^{i\theta_n(\mathbf R)}u_n)
* degenerate/near-degenerate subspace: (\mathbf u \to U(\mathbf R),\mathbf u), with (U(\mathbf R)\in U(N))

So if your solver picks (\theta(\mathbf R)) (or (U(\mathbf R))) independently at each (\mathbf R), the phase (and mixing) will look “random” even when the underlying physics varies smoothly.

**Crucial:** the *physical field* is
[
H(\mathbf r,\mathbf R)=e^{i\mathbf k_0\cdot\mathbf r}\sum_n F_n(\mathbf R),u_n(\mathbf r;\mathbf R),
]
and under a gauge change (u \to Uu), the envelope must transform as (F\to U^\dagger F). Only (H) is physical.

So the “random phase between cells” usually means:
**you are looking at (u) alone, but (u) is not a gauge-invariant object.**

---

## 2) Your warm-start observation is a smoking gun for gauge / subspace rotation

### Why warm-start across (\mathbf k) works

Moving slightly in (\mathbf k) with fixed geometry usually keeps the **same eigen-subspace** and the same numerical gauge is often preserved, so the vector is a good initial guess.

### Why warm-start across (\mathbf R) can be terrible

Changing the geometry (even slightly) can:

* reorder near-degenerate eigenvalues (mode swapping),
* rotate the eigenbasis inside a small-gap subspace (unitary mixing),
* and (most importantly for moiré shift grids) change the **coordinate representation** of what is essentially a translated pattern.

All of that makes the raw eigenvector a poor guess unless you **align** it first.

**Fix for iterative eigensolvers:** warm-start with a *subspace*, not a single vector.
Take the few lowest modes at (\mathbf R), use them as a block initial guess for (\mathbf R+\Delta\mathbf R) (LOBPCG loves this). That makes you robust to internal rotations/mode swaps.

---

## 3) “(\langle u|u\rangle) is not 1 across cells” — normalization is arbitrary, but consistency matters

### (a) Within one frozen (\mathbf R)

You *can and should* normalize your local modes at each (\mathbf R):
[
\langle u_m(\cdot;\mathbf R),u_n(\cdot;\mathbf R)\rangle_\Omega=\delta_{mn},
]
with the **correct inner product** for the operator you are using.

* For the scalar Hermitian form (-\nabla\cdot(\varepsilon^{-1}\nabla)), plain (L^2) is natural.
* For other formulations (e.g. generalized eigenproblems), the natural inner product may be weighted.

If MPB normalizes in an energy norm, that’s fine — but then *your projection* must use the same norm or you must re-normalize to the norm assumed in your derivation.

### (b) Across different (\mathbf R)

You do **not** need (\langle u(\mathbf R),u(\mathbf R')\rangle=1). The projection in the two-scale derivation is done **at fixed (\mathbf R)** over (\mathbf r\in\Omega). Cross-(\mathbf R) orthogonality is not an assumption.

**However:** when you compute (\partial_{\mathbf R}u) numerically via finite differences, inconsistent normalization will *fake* huge derivatives. So normalize consistently before taking overlaps/derivatives.

---

## 4) The big missing practical ingredient: a smooth gauge (single-band and multi-band)

Everything involving (\partial_{\mathbf R}u), Berry connection (A), Born–Huang (\Phi), etc. assumes (u(\mathbf r;\mathbf R)) is chosen **smoothly in (\mathbf R)**. Your solver won’t do that automatically.

### Single-band “parallel transport” phase fixing

For a single isolated mode, enforce:
[
\langle u(\mathbf R),u(\mathbf R+\Delta\mathbf R)\rangle_\Omega \in \mathbb{R}_+.
]
Algorithm:

1. compute overlap (s = \langle u(\mathbf R),u(\mathbf R+\Delta\mathbf R)\rangle_\Omega)
2. set (u(\mathbf R+\Delta\mathbf R) \leftarrow u(\mathbf R+\Delta\mathbf R),e^{-i\arg(s)})

That removes the “random phase”.

### Multi-band (non-Abelian) gauge fixing (this is the one you likely need)

Let (U(\mathbf R)) be an (N)-tuple of eigenmodes as columns. Define the overlap matrix
[
M(\mathbf R,\mathbf R')*{mn}=\langle u_m(\mathbf R),u_n(\mathbf R')\rangle*\Omega.
]
To align the subspace at (\mathbf R') to (\mathbf R), do the **unitary Procrustes** step:

1. SVD: (M = W\Sigma V^\dagger)
2. choose the aligning unitary: (Q = W V^\dagger)
3. rotate the new basis: (U(\mathbf R') \leftarrow U(\mathbf R'),Q^\dagger)

This makes (U(\mathbf R)) vary as smoothly as possible and stabilizes:

* numerical derivatives (\partial_{\mathbf R}u)
* Berry connection (A_j = i\langle u,\partial_{R_j}u\rangle)
* mode tracking / band indexing
* warm starts across (\mathbf R)

If you do *not* do this, then “(\partial_{\mathbf R}u)” is dominated by arbitrary basis rotations and your geometric terms will blow up.

---

## 5) A subtle but *huge* source of “wild variation”: coordinate origin / translation gauge in shift grids

You said your local approximation uses a **shift** of one atom in the unit cell, with a known (\boldsymbol\delta(\mathbf R)).

If the local permittivity is related by a translation of a motif, then **the eigenmodes are related by translations too**. In that case, what looks like a wildly changing eigenvector in *your basis representation* may physically be almost the same field, just translated.

Toy model of the effect:

* Suppose changing (\mathbf R) shifts part of the pattern by (\boldsymbol\delta(\mathbf R)).
* Then a Bloch mode transforms roughly like
  [
  H_{\mathbf k}(\mathbf r;\mathbf R) \approx H_{\mathbf k}(\mathbf r-\boldsymbol\delta(\mathbf R); \mathbf R_0).
  ]
  Writing (H=e^{i\mathbf k\cdot\mathbf r}u), you get (schematically)
  [
  u(\mathbf r;\mathbf R) \approx e^{-i\mathbf k\cdot\boldsymbol\delta(\mathbf R)},u(\mathbf r-\boldsymbol\delta(\mathbf R);\mathbf R_0).
  ]

Two consequences:

1. there is an **extra phase factor** (e^{-i\mathbf k\cdot\boldsymbol\delta(\mathbf R)}) that can vary rapidly if (\mathbf k) is not small
2. in a plane-wave / Fourier basis, the translation (\mathbf r\mapsto \mathbf r-\boldsymbol\delta) multiplies coefficients by phase ramps (e^{-i\mathbf G\cdot\boldsymbol\delta}), which can make the eigenvector look “completely different” numerically even if the real-space field is essentially shifted.

**Actionable fix:** before comparing eigenvectors across (\mathbf R), map them into a *common frame* (a “co-moving gauge”) using the known (\boldsymbol\delta(\mathbf R)). This alone can turn “random phases” into a smooth family.

This also explains your warm-start result perfectly: you were warm-starting with an eigenvector in the wrong translation gauge.

---

## 6) Does non-orthonormality “break the projection assumption”?

Only one thing is required for the projection step:

> At each fixed (\mathbf R), the chosen band functions span (approximately) the relevant fast subspace and are orthonormal under the inner product used.

If your solver gives non-orthonormal modes (common in generalized eigensolvers, or if you collect them from different runs), then yes: **you must orthonormalize them** at each (\mathbf R) (QR/Gram–Schmidt in the correct inner product). Otherwise the envelope equation becomes a **generalized** one:
[
H_{\text{eff}}(\mathbf R,\mathcal D)F = \Delta\lambda;S(\mathbf R)F,
\qquad
S_{mn}(\mathbf R)=\langle u_m,u_n\rangle_\Omega,
]
which is fine mathematically but annoying numerically.

Much easier: enforce (\langle u_m,u_n\rangle=\delta_{mn}) at every (\mathbf R).

---

## 7) “Why not add (e^{i\mathbf k\cdot\mathbf R})?” (or (e^{i\mathbf k(\mathbf R)\cdot\mathbf r}))

### Adding (e^{i\mathbf k\cdot\mathbf R}) specifically

Since (\mathbf R = \eta \mathbf r),
[
e^{i\mathbf k\cdot\mathbf R}=e^{i\eta,\mathbf k\cdot\mathbf r}.
]
That’s just a **tiny perturbation** of the fast phase — equivalent to shifting the carrier wavevector by (O(\eta)):
[
e^{i\mathbf k_0\cdot\mathbf r}e^{i\eta\mathbf q\cdot\mathbf r}
==============================================================

e^{i(\mathbf k_0+\eta\mathbf q)\cdot\mathbf r}.
]
But the whole envelope PDE already represents precisely those small (\Delta\mathbf k\sim O(\eta)) through derivatives acting on (F(\mathbf R)). So putting (e^{i\mathbf k\cdot\mathbf R}) into the ansatz is usually **redundant**; it just hard-codes one Fourier component of (F).

### The *real* generalization that sometimes matters: a slow WKB phase

If you are not at a band extremum and have drift / semiclassical motion, the natural “most general” envelope is
[
F(\mathbf R)=a(\mathbf R),e^{iS(\mathbf R)/\eta},
]
which carries a slow but potentially rapidly accumulating phase (order (1/\eta)). This is the analog of geometrical optics / WKB, and it connects to ray equations and Berry curvature.

For **cavity states**, you typically *choose* (\mathbf k_0) at an extremum (drift zero), and then you can take (F) real (up to a global phase) in a good gauge. So you usually do **not** need an extra slow phase factor.

### If what you want is to capture the shift-grid phase (e^{-i\mathbf k\cdot\boldsymbol\delta(\mathbf R)})

That should be handled as a **gauge choice** (redefining (u) and (F)), not as an extra Bloch exponential. In a clean gauge-covariant derivation, that phase is exactly what shows up inside the Berry connection (A(\mathbf R)).

---

## 8) So what’s “bigger missing”? The likely culprits (in order)

If you implement **(i)** correct normalization + **(ii)** smooth gauge/subspace alignment + **(iii)** translation co-moving frame, and things still don’t click, then the bigger misses are usually:

1. **(\mathbf k_0) is not constant**: the local extremum shifts with (\mathbf R), so expanding about a fixed (\mathbf k_0) creates a wrong “potential well”.
   → Fix: determine (\mathbf k_0(\mathbf R)) locally (or enlarge multi-band set).

2. **Not enough bands**: near-degeneracies require at least a 2×2 or 4×4 envelope system; otherwise the “cavity” can evaporate when band mixing is included.

3. **Local periodic surrogate is missing degrees of freedom**: your “two-atom shift” might capture stacking but miss local symmetry-breaking that strongly affects radiation/leakage channels (less of an issue in strict 2D, bigger in realistic slab).

But honestly, based on your symptom (“warm-start across R is worse than random” + “phase rotates wildly”), I’d bet heavily that **gauge + translation frame** is the dominant problem right now.

