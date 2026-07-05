Got it — and your slides are *very* close to the standard “effective-mass / envelope” derivation for photonic crystals with a slow modulation. I’ll do two things:

1. **Sanity-check** the specific structural claims you’re making (ε(r,R), k, η-expansion, projection).
2. Explain **why the η² term + “mass tensor” appears** in a way you can re-derive reliably.

I’ll stick to your convention: **r dimensionless**, ( \mathbf r=\mathbf x/a ), ( \mathbf R=\mathbf x/L=\eta \mathbf r), ( \eta=a/L\ll1 ).

---

## 1) The gradient rule, cleanly (and why your slide drops a prefactor)

Let (u(\mathbf x)=U(\mathbf r(\mathbf x),\mathbf R(\mathbf x))) with
[
\mathbf r=\frac{\mathbf x}{a},\qquad \mathbf R=\frac{\mathbf x}{L},\qquad \eta=\frac{a}{L}.
]

Differentials:
[
d\mathbf r=\frac{1}{a}d\mathbf x,\qquad d\mathbf R=\frac{1}{L}d\mathbf x.
]

So
[
du = (\nabla_{\mathbf r}U)\cdot d\mathbf r + (\nabla_{\mathbf R}U)\cdot d\mathbf R
= \left(\frac{1}{a}\nabla_{\mathbf r}U+\frac{1}{L}\nabla_{\mathbf R}U\right)\cdot d\mathbf x.
]

Hence the physical gradient is
[
\boxed{\nabla_{\mathbf x}=\frac{1}{a}\nabla_{\mathbf r}+\frac{1}{L}\nabla_{\mathbf R}
=\frac{1}{a}\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right).}
]

✅ Your slide writes (\nabla \mapsto \nabla_{\mathbf r}+\eta\nabla_{\mathbf R}).
That is correct **if** you have implicitly scaled out the (1/a) (i.e. you are working with (a\nabla_{\mathbf x})). This is the most common convention in photonic-crystal asymptotics.

---

## 2) Can one “just” write (\varepsilon(\mathbf r)\to\varepsilon(\mathbf r,\mathbf R))?

Yes — but the *real* justification is a **regularity + frozen-cell periodicity** assumption:

* For each fixed (\mathbf R), the function (\varepsilon(\mathbf r,\mathbf R)) is **periodic in (\mathbf r)** with the *small* cell (\Omega).
* The dependence on (\mathbf R) is **smooth and slow** (varies on scale 1 in (\mathbf R), i.e. on scale (L) in (\mathbf x)).

That’s exactly what a moiré modulation is modeled as: a “locally periodic” medium whose motif (relative shift / orientation / phase between sublattices) changes slowly over the superlattice.

A very concrete moiré-style form is something like
[
\varepsilon(\mathbf r,\mathbf R)=\varepsilon_1(\mathbf r)+\varepsilon_2(\mathbf r+\boldsymbol\delta(\mathbf R)),
]
where (\boldsymbol\delta(\mathbf R)) is a slow displacement field (or slow phase).

So: ✅ yes, you can do it — but the key sentence in a thesis is:
**“Assume ε is periodic in the fast coordinate for each frozen R and varies smoothly in R.”**

---

## 3) Does **k** stay the same? Is it “invariant” under the two-scale transformation?

### What is “k” in your ansatz?

You write (e^{i\mathbf k\cdot\mathbf r}) with (\mathbf r=\mathbf x/a). So
[
e^{i\mathbf k\cdot\mathbf r}=e^{i(\mathbf k/a)\cdot\mathbf x}.
]

So:

* (\mathbf k) is a **dimensionless crystal momentum** (coordinates in the small-cell reciprocal lattice).
* The physical wavevector is (\mathbf k_{\text{phys}}=\mathbf k/a).

### Under the two-scale rewrite, nothing forces k to rescale

The two-scale chain rule changes derivatives, not the Fourier label you chose for the fast Bloch phase. So ✅ it is consistent to keep (\mathbf k=\mathbf k_0) fixed.

### The subtle point (important for moiré):

Globally, the structure is *not* exactly periodic, so (\mathbf k) is not a globally conserved quantum number. The envelope method is essentially the statement:

> The field is a narrow superposition of local Bloch waves near a chosen (\mathbf k_0); the slow spatial modulation is captured by (F(\mathbf R)).

This is why “k stays the same” in the carrier, while deviations from (\mathbf k_0) are encoded by derivatives of (F).

---

## 4) Where the η-expansion (L_0+\eta L_1+\eta^2 L_2) comes from

With your ansatz (schematically)
[
H(\mathbf r,\mathbf R)=F(\mathbf R),e^{i\mathbf k_0\cdot\mathbf r},u(\mathbf r;\mathbf R),
]
and your scaled gradient ( \nabla = \nabla_{\mathbf r}+\eta \nabla_{\mathbf R}),

every time a (\nabla_{\mathbf R}) hits something you pick up a factor **η** relative to (\nabla_{\mathbf r}). So:

* (L_0): only fast derivatives (\nabla_{\mathbf r}) (order 1)
* (L_1): exactly one slow derivative (\nabla_{\mathbf R}) (order η)
* (L_2): two slow derivatives or one slow derivative acting on something already slow (order η²)

✅ The bookkeeping on your slide (“Fast O(1), Mixed O(η), Slow O(η²)”) is exactly the right intuition.

---

## 5) The projection: what it really is doing

Your “small cell – Bloch projection”
[
\langle f,g\rangle_\Omega=\frac{1}{|\Omega|}\int_\Omega f^*(\mathbf r),g(\mathbf r),d\mathbf r
]
is a **solvability condition** / **Fredholm alternative** in disguise:

* (L_0) is the leading (fast) operator.
* At leading order, you choose (u) as an eigenfunction of (L_0) (at frozen (\mathbf R)).
* At next orders, you get an inhomogeneous equation for the correction (H_1).
* A solution exists only if the RHS is orthogonal to the nullspace of the adjoint operator — which, for a Hermitian problem, means “orthogonal to (u)”.

So when you write (\langle u, L_1\rangle) and (\langle u, L_2\rangle), you’re enforcing that solvability condition and extracting an equation for (F).

✅ This is the correct conceptual justification for “we project order-by-order”.

**One caution:** for the *vector* Maxwell curl–curl eigenproblem, the natural Hermitian inner product and normalization can depend on the chosen formulation (H vs E). Your slide uses a plain (L^2)-type inner product; that is fine for many scalar reductions and for the standard ( \nabla\times(\epsilon^{-1}\nabla\times \mathbf H)) Hermitian form with periodic BCs and real ε, but in your thesis it’s worth stating explicitly *which formulation* you use and what inner product makes the operator Hermitian.

---

## 6) Why the η² term gives the “mass tensor”

This is the heart of it.

### (A) The physical idea

An envelope (F(\mathbf R)) that varies slowly in space corresponds (Fourier-dually) to a **narrow distribution in k-space** around the carrier (\mathbf k_0):
[
H \sim \int A(\boldsymbol\kappa),e^{i(\mathbf k_0+\eta\boldsymbol\kappa)\cdot\mathbf r},u_{\mathbf k_0}(\mathbf r),d\boldsymbol\kappa,
]
with width (\Delta k \sim O(\eta)).

So you expand the local band:
[
\omega(\mathbf k_0+\eta\boldsymbol\kappa;\mathbf R)
\approx \omega_0(\mathbf R)

* \eta,(\nabla_{\mathbf k}\omega_0)\cdot\boldsymbol\kappa
* \frac{\eta^2}{2},\boldsymbol\kappa^\top \left(\partial_{\mathbf k}\partial_{\mathbf k}\omega_0\right)\boldsymbol\kappa
  +\cdots
  ]

Now map (\boldsymbol\kappa \leftrightarrow -i\nabla_{\mathbf R}) (because (\mathbf R)-space and (\boldsymbol\kappa)-space are Fourier conjugates for the envelope). That turns the quadratic form into a differential operator:
[
\boldsymbol\kappa^\top (\partial_{kk}\omega_0)\boldsymbol\kappa
\quad\leadsto\quad
-\nabla_{\mathbf R}\cdot\left(M^{-1}(\mathbf R)\nabla_{\mathbf R}\right),
]
with
[
\boxed{M^{-1}*{ij}(\mathbf R)=\left.\frac{\partial^2\omega_0(\mathbf k;\mathbf R)}{\partial k_i\partial k_j}\right|*{\mathbf k_0}.}
]

That’s why it’s called “effective mass”: it is literally the Hessian (curvature) of the dispersion surface.

### (B) Why it shows up at order η²

Because:

* a *single* (\nabla_{\mathbf R}) gives (O(\eta)), i.e. a linear-in-(\kappa) correction → the **drift / group velocity** term.
* the **curvature** of the band is quadratic in (\Delta k), and (\Delta k\sim \eta), so it is (O(\eta^2)).

✅ So “η² gives the mass tensor” is not magic — it’s just “second-order Taylor in k”.

---

## 7) Double-checking your specific final structure

### Drift term

Your envelope equation shows a drift term like
[
-i\eta,(\nabla_{\mathbf k}\omega_0(\mathbf R))\cdot\nabla_{\mathbf R}F.
]
This is the expected group-velocity term.

✅ And you correctly annotate “zero at band extremum”: if (\mathbf k_0) is chosen at a band edge, (\nabla_{\mathbf k}\omega_0|_{\mathbf k_0}=0), so the leading nontrivial dynamics starts at **η²**.

### Kinetic term

You have
[
-\frac{\eta^2}{2}\nabla_{\mathbf R}\cdot M^{-1}(\mathbf R)\nabla_{\mathbf R}F.
]
That is the canonical effective-mass operator.

**Tiny consistency note:** your projection slide naturally produces derivatives of (\omega_0^2/c^2) (because the eigenvalue is (\omega^2/c^2)). To turn that into an equation written in terms of (\omega) and (\partial_{kk}\omega), you usually linearize:
[
\omega^2 \approx \omega_0^2 + 2\omega_0,\Delta\omega,
]
and similarly relate (\partial_{kk}\omega^2) to (\partial_{kk}\omega) via
[
\partial_{kk}\omega^2 = 2(\partial_k\omega)(\partial_k\omega)^\top + 2\omega,\partial_{kk}\omega.
]
At a band extremum (\partial_k\omega=0), so (\partial_{kk}\omega^2 \approx 2\omega_0,\partial_{kk}\omega).
So your “mass tensor from (\partial_{kk}\omega)” is consistent, especially at the extremum.

### Potential term (V(\mathbf R))

Your (V(\mathbf R)=\omega_0(\mathbf R)-\omega_0^{\rm ref}) is the usual “local band-edge shift acts like a potential” statement.

✅ That matches the frozen-(\mathbf R) picture: the slowly varying medium changes the local band-edge frequency, which traps the envelope in regions where (V) forms a well.

---

## 8) The main caveat I’d explicitly address in the thesis

### Does (u) depend on (R)?

If (\varepsilon=\varepsilon(\mathbf r,\mathbf R)), then strictly the local Bloch mode should be
[
u_{n\mathbf k}(\mathbf r;\mathbf R),
]
i.e. it depends on (\mathbf R) as a parameter.

When you apply (\nabla_{\mathbf R}), it can hit (u) too, producing additional (O(\eta)) terms (often interpretable as Berry-connection / geometric terms). In many envelope derivations these can be:

* neglected under an **adiabatic** assumption,
* or removed/controlled by a **gauge choice** (e.g. parallel-transport gauge where (\langle u,\partial_{\mathbf R}u\rangle=0)),
* or shown to be small when symmetries enforce reality / no band mixing.

Your slides currently omit explicit (\nabla_{\mathbf R}u) contributions, so I’d add one sentence like:

> “We assume the dominant contribution comes from the envelope variation and the parametric change of the eigenvalue; geometric connection terms from (\partial_{\mathbf R}u) are neglected / fixed by gauge / higher order for the considered symmetry.”

That makes the derivation defensible.

---

## 9) Quick checklist (to “prove to yourself” it’s consistent)

1. **Dimensions:** with r dimensionless, remember (\nabla_{\mathbf x}=(1/a)(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R})). If you omit (1/a), you’re in scaled units.
2. **Frozen-cell periodicity:** (\varepsilon(\cdot,\mathbf R)) periodic in r for each R.
3. **Hermiticity & inner product:** state the formulation and the inner product that makes (L_0) self-adjoint.
4. **Band isolation:** single non-degenerate band; otherwise you need a multi-band envelope (matrix equation).
5. **Extremum assumption:** if you drop drift, explicitly say (\nabla_k\omega_0(\mathbf R)|_{\mathbf k_0}=0).
6. **Mass tensor symmetry:** (M^{-1}) should be symmetric (Hessian).
7. **ω vs ω²:** show the small linearization step once (especially if a reviewer is picky).

