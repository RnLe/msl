You’re right to zoom in on this — **whether (u) depends on (R)** is not a cosmetic detail. In a *locally periodic* (moire) medium it is, in principle, **inevitable** that the “cell eigenmode” changes with the slow parameter. Ignoring that is an **adiabatic / single-band** approximation. Sometimes it’s fine; sometimes it’s exactly what breaks the comparison to full-wave time-domain (Meep), especially in hex lattices where **near-degeneracies** (Dirac-like features, symmetry-protected crossings) are common.

Below is the clean, “re-derive anytime” version:

---

# 1) Is (u(\mathbf r;\mathbf R)) expected?

### Yes — generically it *must* depend on (R)

If you model the permittivity as (\epsilon(\mathbf r,\mathbf R)) (periodic in (\mathbf r) for each fixed (\mathbf R)), then the frozen-(\mathbf R) Bloch eigenproblem changes with (\mathbf R), so its eigenfunctions do too:
[
u_{n\mathbf k}(\mathbf r;\mathbf R).
]

### When can it be neglected?

Neglecting (\nabla_{\mathbf R}u) is an **adiabatic / isolated-band** assumption:

* (\eta=a/L\ll 1) (slow modulation),
* band (n) is **well-separated** from nearby bands: gap (\Delta\lambda) not tiny,
* the “driving” from (\partial_{\mathbf R}\epsilon) is modest.

If any of these fail (esp. small gaps / degeneracy), then (\partial_{\mathbf R}u) causes **interband mixing at order (\eta)** and your scalar envelope equation can go “wonky”.

---

# 2) Bloch theorem: valid or not in moiré / double-periodicity?

* **Global Bloch theorem** requires global periodicity. For a generic incommensurate moiré (or locally varying registration), global periodicity is absent ⇒ no global Bloch label (\mathbf k).
* **Local Bloch theorem** *does* hold in the “frozen-(\mathbf R)” cell problem: for each fixed (\mathbf R), (\epsilon(\cdot,\mathbf R)) is periodic in (\mathbf r), so you can use Bloch modes at that (\mathbf R).

This is the standard justification for two-scale / homogenization: you replace the globally complicated structure by a **family of periodic problems parametrized by (\mathbf R)**.

If your moiré is actually **commensurate** with a giant supercell, then Bloch theorem holds at the moiré scale. The two-scale model should then approximate the “true supercell Bloch bands” in the limit (L/a\to\infty).

---

# 3) The key calculation: what changes if (u) depends on (R)?

I’ll do this in a general “Hermitian cell operator” framework because it’s the cleanest and applies to your scalar TE-like slide and (with the right inner product) to curl–curl too.

## 3.1 Frozen-(\mathbf R) cell eigenproblem

Let ( \mathcal{L}*{\mathbf R}(\mathbf k)) be the periodic (in (\mathbf r)) Bloch operator at frozen (\mathbf R). In your scalar-style notation this is essentially
[
\mathcal{L}*{\mathbf R}(\mathbf k) = D_{\mathbf r}(\mathbf k)\cdot \epsilon^{-1}(\mathbf r,\mathbf R), D_{\mathbf r}(\mathbf k),
\qquad D_{\mathbf r}(\mathbf k)=\nabla_{\mathbf r}+i\mathbf k,
]
and the eigenproblem is
[
\mathcal{L}*{\mathbf R}(\mathbf k),u*{n}(\mathbf r;\mathbf R)=\lambda_n(\mathbf k,\mathbf R),u_{n}(\mathbf r;\mathbf R),
\qquad \lambda=\omega^2/c^2.
]
Normalize on the small cell (\Omega):
[
\langle u_m,u_n\rangle_\Omega=\delta_{mn}.
]

## 3.2 Two-scale ansatz

Take one band (n) and one carrier (\mathbf k_0):
[
H(\mathbf r,\mathbf R)=F(\mathbf R),e^{i\mathbf k_0\cdot\mathbf r},u(\mathbf r;\mathbf R).
]

Your two-scale derivative is (in the scaled convention)
[
\nabla \mapsto \nabla_{\mathbf r}+\eta\nabla_{\mathbf R}.
]

Here’s the crucial identity:
demodulate the fast Bloch phase and work with (D_{\mathbf r}(\mathbf k_0)):
[
(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R})H
= e^{i\mathbf k_0\cdot\mathbf r}\Big[
D_{\mathbf r}(\mathbf k_0),(F u)

* \eta,\nabla_{\mathbf R}(F u)
  \Big].
  ]

Now expand the slow derivative:
[
\nabla_{\mathbf R}(Fu)= (\nabla_{\mathbf R}F),u + F,(\nabla_{\mathbf R}u).
]

**This (F\nabla_{\mathbf R}u) is the new player.**

---

# 4) What projection actually does (and what it does *not* do)

Your small-cell projection
[
\langle u, \cdot \rangle_\Omega
]
only removes dependence on (\mathbf r) by taking an inner product over the unit cell.

It does **not** magically eliminate the effects of other bands.

Why? Because (\nabla_{\mathbf R}u) is not proportional to (u) in general. It has components along *all* other eigenmodes:
[
\nabla_{\mathbf R}u
= \underbrace{\langle u,\nabla_{\mathbf R}u\rangle_\Omega,u}_{\text{in-band (gauge) part}}

* \underbrace{\sum_{m\neq n} \langle u_m,\nabla_{\mathbf R}u\rangle_\Omega,u_m}_{\text{interband mixing}}.
  ]

Projection onto (u) keeps the **in-band part**, but the **interband part** feeds back at higher order (or immediately, if you do a multi-band model).

This is exactly the Born–Oppenheimer / adiabatic decomposition in QM.

---

# 5) The “geometric” terms you get: Berry connection + Born–Huang potential

Define the (vector) **Berry connection** for the band:
[
\mathbf A(\mathbf R) := i,\langle u, \nabla_{\mathbf R}u\rangle_\Omega.
]
(This is gauge-dependent; see below.)

Then you get the fundamental “covariant derivative” identity:
[
\langle u,\nabla_{\mathbf R}(F u)\rangle_\Omega
= \nabla_{\mathbf R}F + \langle u,\nabla_{\mathbf R}u\rangle_\Omega F
= \big(\nabla_{\mathbf R}-i\mathbf A(\mathbf R)\big)F.
]

So wherever your derivation produced (\nabla_{\mathbf R}F), the **correct single-band replacement** is
[
\boxed{\nabla_{\mathbf R}F ;\to; (\nabla_{\mathbf R}-i\mathbf A)F.}
]

At second order, another term appears, the **Born–Huang (scalar) potential**
[
\Phi(\mathbf R)
:= \langle \nabla_{\mathbf R}u,(1-|u\rangle\langle u|)\nabla_{\mathbf R}u\rangle_\Omega
= \sum_{m\neq n} \big|\langle u_m,\nabla_{\mathbf R}u\rangle_\Omega\big|^2.
]

This is gauge-invariant and cannot be removed by phase choices.

### What does this do to your final envelope equation?

Schematically, the “kinetic” piece becomes **minimal coupling**:
[
-\frac{\eta^2}{2},\nabla_{\mathbf R}\cdot M^{-1}(\mathbf R)\nabla_{\mathbf R}F
\quad\longrightarrow\quad
-\frac{\eta^2}{2},(\nabla_{\mathbf R}-i\mathbf A)\cdot M^{-1}(\mathbf R)(\nabla_{\mathbf R}-i\mathbf A)F
]
and the scalar potential shifts as
[
V(\mathbf R);\to; V(\mathbf R)+\eta^2,\Phi(\mathbf R)\quad(\text{up to convention-dependent prefactors}).
]

**Interpretation:**

* (\mathbf A): an effective “vector potential” (geometric phase / gauge field)
* (\Phi): an extra trapping/repulsion term from interband admixture

If your current theory ignores these, it can absolutely distort confinement frequencies and mode shapes.

---

# 6) “But we don’t have explicit (u). How can (\nabla_{\mathbf R}u) be used?”

This is the practical part: you do **not** need analytic (u). You need overlaps / matrix elements, and these can be obtained numerically.

## 6.1 Perturbation formula for (\langle u_m,\partial_{R_j}u_n\rangle)

Differentiate the frozen-(\mathbf R) eigenproblem:
[
\mathcal{L}u_n=\lambda_n u_n.
]
Take (\partial_{R_j}):
[
(\partial_{R_j}\mathcal{L})u_n+\mathcal{L}(\partial_{R_j}u_n)
= (\partial_{R_j}\lambda_n)u_n+\lambda_n(\partial_{R_j}u_n).
]

Project onto (u_m) using Hermiticity and orthonormality:

* For (m=n):
  [
  \boxed{\partial_{R_j}\lambda_n = \langle u_n,(\partial_{R_j}\mathcal{L})u_n\rangle_\Omega.}
  ]
  This is the “Hellmann–Feynman” identity.

* For (m\neq n):
  [
  \boxed{\langle u_m,\partial_{R_j}u_n\rangle_\Omega
  = \frac{\langle u_m,(\partial_{R_j}\mathcal{L})u_n\rangle_\Omega}{\lambda_n-\lambda_m}.}
  ]

This is huge: it expresses the problematic (\partial_{\mathbf R}u) in terms of:

* band gaps ((\lambda_n-\lambda_m))
* matrix elements of (\partial_{\mathbf R}\mathcal{L}), which depends on (\partial_{\mathbf R}\epsilon(\mathbf r,\mathbf R)), i.e. **known** from your geometry

So you can compute (\mathbf A) and (\Phi) with nothing “analytic” about (u), just numerical eigenmodes + overlaps.

## 6.2 Gauge handling (important!)

The diagonal connection (\mathbf A=i\langle u,\nabla_{\mathbf R}u\rangle) depends on the phase choice (u\to e^{i\theta(\mathbf R)}u).

A standard choice is **parallel transport gauge**:
[
\langle u,\partial_{R_j}u\rangle_\Omega = 0 \quad \Rightarrow\quad \mathbf A=0.
]
This removes the vector-potential term completely (for a single isolated band), but **(\Phi)** remains.

Numerically, this is implemented by fixing the phase of (u(\mathbf R+\Delta \mathbf R)) to maximize overlap with (u(\mathbf R)) (make the overlap real positive).

---

# 7) When do these terms matter enough to fix “wonky vs Meep”?

Here are the usual failure modes that match your symptoms:

### (i) Near-degeneracy ⇒ single-band envelope fails

Hex lattices often have:

* nearly degenerate bands at high-symmetry points,
* Dirac cones / symmetry-protected crossings.

Then (\lambda_n-\lambda_m) is small, so
[
\langle u_m,\partial_{\mathbf R}u_n\rangle \sim \frac{1}{\lambda_n-\lambda_m}
]
can blow up, and (\Phi) / interband mixing becomes large.

**Fix:** Use a **multi-band envelope**:
[
H=\sum_{\alpha=1}^N F_\alpha(\mathbf R)e^{i\mathbf k_0\cdot\mathbf r}u_\alpha(\mathbf r;\mathbf R),
]
leading to an (N\times N) matrix Schrödinger-like equation. This is the photonic analog of (k\cdot p) with degenerate subspaces.

### (ii) Strong local geometry variation ⇒ (\partial_{\mathbf R}\epsilon) not “small”

Even if (\eta\ll1), the *amplitude* of the slow modulation may be strong (e.g. large index contrast or sharp features shifting with (\mathbf R)). Then ((\partial_{\mathbf R}\mathcal{L})) is large, again making couplings non-negligible.

### (iii) Meep sees radiation/leakage not captured

Envelope eigen-equations often assume a closed Hermitian eigenproblem. Meep may show leakage if your “bound” state couples to continuum / out-of-plane modes / finite-size boundaries. This is a separate issue, but it often coexists with (i)/(ii).

---

# 8) Direct answers to your last questions

### “Should we genuinely include (u(\mathbf r;\mathbf R)) in the ansatz?”

**Yes**, that’s the mathematically consistent two-scale statement if (\epsilon(\mathbf r,\mathbf R)) depends on (\mathbf R).
Then you decide whether to:

* **neglect** its derivatives (adiabatic single-band),
* **keep** the geometric corrections ((\mathbf A,\Phi)),
* or go **multi-band**.

### “Does the Bloch projection eliminate other (u) functions?”

No. The projection onto a single (u) just *selects one component*. The derivatives (\nabla_{\mathbf R}u) generally contain other bands, and their influence returns as (\Phi) and/or explicit band-coupling terms unless gaps are large.

### “If we add (\nabla_{\mathbf R}u), how to handle it?”

Use the perturbation identities above. They turn (\nabla_{\mathbf R}u) into **computable overlap integrals** involving (\partial_{\mathbf R}\epsilon) and the spectral gaps.

---

# 9) What I’d do next to debug your mismatch fast

1. For each (\mathbf R) (or a path in (\mathbf R)-space), compute the frozen-cell eigenmodes (u_n(\mathbf r;\mathbf R)) and eigenvalues (\lambda_n(\mathbf k_0,\mathbf R)).

2. Check **band isolation**:
   [
   \min_{m\neq n} |\lambda_n-\lambda_m|
   ]
   If it gets small anywhere relevant, your single-band envelope is likely invalid there.

3. Measure **mode variation** via overlaps:
   [
   |\langle u(\mathbf R), u(\mathbf R+\Delta \mathbf R)\rangle_\Omega|
   ]
   If this drops significantly for small (\Delta \mathbf R), (\nabla_{\mathbf R}u) is large → include corrections or go multi-band.

4. If the band is isolated and overlaps stay ~1, choose parallel-transport gauge and compute (\Phi(\mathbf R)) (or estimate it via the perturbation sum). Add it to (V) and see if agreement improves.

