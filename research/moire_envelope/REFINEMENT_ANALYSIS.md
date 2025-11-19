Let’s zoom in on the knobs we actually have, and sort them a bit:

* **A. Physics / model knobs (what terms you keep in the envelope theory)**
* **B. MPB / local-band extraction knobs (how accurately you get V, M⁻¹, v_g)**
* **C. Envelope-discretization knobs (R-grid, finite differences, solver)**

I’ll keep Meep (Phase 5) completely out of this as requested.

---

## A. Physics / model knobs

### A1. Include the group velocity term (v_g(R))

Right now (from the README and phase2/phase3 reports) you are effectively solving:

[
H_\text{EA} F(R)
= \left[
-\frac{\eta^2}{2},\nabla_R\cdot M^{-1}(R)\nabla_R + V(R)
\right] F(R)
]

You *compute* (v_g(R)) in Phase 1, but you **do not** use it in the operator (the README even mentions v_g as “optional”).

The more complete single-band envelope expansion near (k_0) has a **first-order derivative term**:

[
H_\text{EA} F(R)
= \left[
\omega_0(k_0; R) - \omega_\text{ref}

* v_g(R)\cdot(-i\eta\nabla_R)
  -\frac{\eta^2}{2},\nabla_R\cdot M^{-1}(R)\nabla_R
* \dots
  \right]F(R).
  ]

So the extra term would be:

[
H_{(1)} = v_g(R)\cdot(-i\eta\nabla_R).
]

In your case:

* At the *ideal* extremum (M-point) in a perfectly symmetric monolayer, (v_g=0).
* But with registry dependence and numerical noise, you see (|v_g|\sim 10^{-3}), i.e. small but not mathematically zero.

**What this knob does:**

* It corrects for the fact that your actual “local extremum” is a bit off from k₀ in different registries.
* It introduces a mild “drift” term in real space; modes may shift slightly in energy and position.

**How important is it?**

* Scaling-wise, the mass term enters with (\eta^2 M^{-1}), while the v_g term enters with (\eta v_g).
* If (|v_g|\ll |M^{-1} G_\text{moire}|\cdot\eta) (which it seems to be), it’s probably a small correction, but if you want “clean” theory, it is the **first missing term** to add.

Implementation-wise:

* In Phase 2, you’d add a linear-derivative operator with coefficients v_g(R).
* In discrete form this is a skew-adjoint finite-difference operator; you can write it to keep the whole operator Hermitian by symmetrizing the stencil.

So yes: **including the v_g term is a genuine accuracy knob**.
Given how small it is for your candidate, it’s probably a *second-order* correction in practice, but it’s the correct next step formally.

---

### A2. Multi-band envelope (beyond single-band EA)

Right now you are:

* Projecting onto **one band** (n_0), and ignoring coupling to other bands.
* That’s okay if:

  * the band is well isolated (Phase 0’s “spectral isolation” score), and
  * twist-induced couplings are small compared to the gaps.

To go one level up in accuracy, you could:

* Include a **second nearby band** (e.g. n=0 and n=1, or 1 and 2 depending on polarization) and build a 2×2 envelope Hamiltonian:
  [
  H_\text{EA}(R) =
  \begin{pmatrix}
  H^{(1)}*\text{EA} & U(R) \
  U^\dagger(R)      & H^{(2)}*\text{EA}
  \end{pmatrix}
  ]
  where each diagonal block is a scalar EA operator (like now), and (U(R)) encodes local band-mixing from the twist.

This is a **big** modeling upgrade (more coding and more MPB work), but physically the next systematic step if you ever hit the limits of the single-band model.

---

### A3. Higher-order in k: quartic dispersion corrections

Currently you approximate:

[
\omega(k;R) \approx
\omega_0(k_0;R) + v_g(R)\cdot(k-k_0)
+\frac12 (k-k_0)^T M^{-1}(R)(k-k_0).
]

If you suspect the band is not very parabolic out to (|k-k_0|\sim |G_\text{moire}|), you could:

* include 4th-order derivatives (\partial^4 \omega/\partial k^4) to get a quartic correction;
* or at least *test* this by fitting ω(k) to a 2D polynomial including quartic terms and see if they’re sizable.

In the envelope picture, quartic k terms map to **4th-order derivatives in R**. That’s probably overkill right now, but conceptually it’s another knob.

---

### A4. Higher-order in η (geometry / “twist gradient” corrections)

Your current model is the standard **frozen registry** approximation:

* locally, the bilayer is just two copies of the monolayer shifted by a constant δ(R);
* rotation is encoded only in δ(R), not in explicit ∇δ terms.

Next-order terms (in η) would include:

* **gradients of δ(R)** (i.e. explicit dependence on the slow variation of the stacking, not just its local value),
* possible small “geometric” / Berry-like corrections to the envelope equation.

This is again a deep modeling change, but good to keep in mind as a conceptual knob.

---

## B. MPB / local-band accuracy knobs

These are about getting cleaner (V(R), v_g(R), M^{-1}(R)) in Phase 1.

### B1. MPB resolution / basis size

Most direct:

* Increase MPB **resolution** (e.g. from 32 → 48 → 64 pixels per a).
* Check convergence of:

  * ω₀(δ),
  * v_g(δ),
  * M⁻¹(δ)
    at a few representative δ’s.

Natural convergence test:

* Pick a couple of registry points (e.g. a deep minimum and a high-symmetry point) and run MPB with increasing resolution.
* Require ω₀ to converge to within e.g. 10⁻⁴ and the principal curvatures of M⁻¹ to converge to a few percent.
* Once *that* is converged, the EA-level eigenvalues will be limited by the envelope discretization, not MPB.

### B2. Δk for derivatives (finite-difference step)

Your mass tensor and v_g come from finite differences in k:

* If Δk is too large: parabolic approximation is biased; second derivatives “average over” non-parabolic behavior.
* If Δk is too small: numerical noise (MPB eigenvalue jitter) dominates the difference.

So Δk is a **sensitive knob**:

* Give yourself a convergence study:

  * e.g. Δk = 0.005, 0.01, 0.02 (in 2π/a units).
* At fixed registry and MPB resolution, compute, for each Δk:

  * v_g(δ),
  * eigenvalues of M⁻¹(δ).
* Plot vs Δk; pick the plateau region.

This can easily reduce random fluctuations in M⁻¹(R) and v_g(R) that would otherwise look like “fake structure” in your potential/mass maps.

### B3. Higher-order finite-difference stencil in k

Right now you’re almost certainly using:

* 2-point central stencil for v_g,
* 5- or 9-point pattern for M⁻¹ (axis + diagonals).

You can upgrade to a **4th-order accurate** central difference in 1D and embed that in 2D:

* sample ω(k) on ±Δk and ±2Δk along each axis,
* fit a polynomial or use known 4th-order FD coefficients.

That gives you:

* smaller truncation error ∼O(Δk⁴) instead of O(Δk²),
* better noise rejection if MPB eigenvalues are smooth.

This is a pure “math knob” that can improve the smoothness of M⁻¹(R) and v_g(R) significantly.

### B4. Smoothing / filtering V(R) and M⁻¹(R)

Important conceptual point:

* The **envelope** is only supposed to see **slow** variations (moiré scale).
* Any grid-scale or MPB noise in V(R), M⁻¹(R) is unphysical from the EA point of view.

So one very legitimate knob is:

* Apply a small **low-pass filter** / smoothing to V(R) and M⁻¹(R) in R-space.

  * e.g. Gaussian blur over 1–2 R-grid spacings, or
  * spectral filter in reciprocal R-space that kills high-q components beyond the first moiré G’s.

This often does *more* for physical accuracy than cranking raw resolutions, because it enforces the assumption that only moiré-scale structure matters.

Given how good your plots already look, this is probably more a *stability & aesthetics* knob than a necessity, but it’s fully justifiable physically.

---

## C. Envelope-discretization knobs (Phase 2 & 3)

These are knobs *after* you’ve decided on the continuum model.

### C1. R-grid resolution (Nx, Ny)

Yes, this is a direct knob:

* You currently use 48×48 for candidate 1737.
* You can do a convergence scan: 32×32, 48×48, 64×64, maybe 96×96.

For each:

* Solve Phase 3 (Γ-point EA).
* Track:

  * lowest few eigenvalues ω₀, ω₁, …,
  * participation ratios,
  * localization length ξ.

Stop when changes in ω₀, PR, ξ are below your target tolerance (e.g. < 1% or < 10⁻³ in ω).

Rule of thumb:

* grid spacing ΔR should be **much smaller** than:

  * the localization length ξ, and
  * the characteristic length scale set by the curvature of V(R) near minima.

Given ξ ≈ 10 (for 48×48) and domain size ≈ L_moire ≈ 52, you’re at ΔR ≈ 52/48 ≈ 1.1, so ξ/ΔR ≈ 9 grid points per localization length. That’s decent. 64×64 would give you a nice refinement if you want.

### C2. Order of the spatial finite differences

Your README describes a standard 2nd-order FD for:

[
-\nabla_R\cdot M^{-1}(R)\nabla_R.
]

You could upgrade to:

* 4th-order FD Laplacian with appropriate variable-mass treatment;
* or even move to a simple finite-element discretization on a structured grid.

In practice:

* 2nd-order + moderate grid resolution is usually enough.
* If you see that eigenvalues converge *slowly* with Nx,Ny, upgrading the stencil can help a lot.

### C3. Domain size / boundary conditions in Phase 3

Phase 3 uses a finite window:

* R ∈ [−L/2, L/2] with periodic or (more likely) zero / Dirichlet boundaries (depending on your implementation).

If you use **Dirichlet** BCs for the cavity solver:

* the finite window itself acts as a weak “box potential”.
* you can reduce boundary influence by:

  * enlarging the window (multiple moiré cells),
  * or switching to periodic BCs and then localizing via V(R).

Given your ξ ≈ 10 on a domain ~52, you probably already have boundary effects well under control, but enlarging the window to e.g. 2× L_moire and checking convergence is a knob if you need very clean eigenvalues.

### C4. Solver tolerance, eigsh parameters

You already use:

* `tol = 1e-8`
* `maxiter = 2000`
* no shift-invert.

That’s fine. You can confirm:

* Re-run with even tighter tol or more iterations and check whether eigenvalues move; if not, your error is dominated by model/resolution, not solver.

---

## Which knobs matter most *right now* for you?

Given:

* v_g(R) ∼ 10⁻³ (tiny),
* effective η² M⁻¹ scale that matches your Δω₀,
* Phase 3 localized modes and Phase 4 bands looking “too good”,

my ranking of **impactful and realistic** knobs would be:

1. **MPB-side convergence**

   * do a small convergence test in resolution + Δk at a few δ’s.
   * once M⁻¹ and ω₀ are stable, you can be confident the local input is solid.

2. **R-grid resolution**

   * test 48×48 → 64×64; verify that ω₀, PR, ξ are stable.
   * if they are, 48×48 is already “good enough” for this θ, and you can keep it as default.

3. **Mild smoothing of V(R) and M⁻¹(R)**

   * especially to kill MPB noise that doesn’t have moiré symmetry.
   * this can reduce spurious shallow wells and artificial fragmentation of the cavity.

4. **Optionally: include v_g(R) in the EA operator**

   * mostly for formal completeness and to be sure that “if v_g became larger for some future candidate, the code is already ready”.

5. **Only if you hit a wall:** multi-band EA or higher-order k expansions

   * these are more work and only necessary if Phase 0 starts giving you candidates with smaller gaps or less parabolic behavior.