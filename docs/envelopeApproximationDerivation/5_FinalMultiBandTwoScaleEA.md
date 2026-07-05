Alright. I’m going to write this in a way that is **thesis-grade, self-contained, and reproducible** — with the *full* multi-band structure, the (\eta)-expansion, the projection, and the final “effective envelope Hamiltonian” including:

* **local band energy / potential**
* **drift (group velocity)**
* **effective mass tensor (curvature / k·p)**
* **non-Abelian Berry connection** (gauge field)
* **Born–Huang / quantum-geometric correction** (from out-of-subspace mixing)

I’ll do it for the **2D scalar** operator (your “two gradients, not two curls”), because that’s the cleanest and captures the same structure you’re using.

---

# 0) Problem and assumptions (explicit)

We consider a 2D eigenproblem of the form
[
\boxed{
\mathcal{L}*{\mathbf x},H(\mathbf x)=\lambda,H(\mathbf x),
\qquad
\lambda=\frac{\omega^2}{c^2},
}
]
with the (Hermitian) operator
[
\boxed{
\mathcal{L}*{\mathbf x} := -\nabla_{\mathbf x}\cdot\left(\varepsilon^{-1}(\mathbf x)\nabla_{\mathbf x}\right).
}
]

**Two-scale structure (locally periodic / moiré):**

* There is a **fast** lattice scale (a) and a **slow** moiré scale (L\gg a).
* Define dimensionless fast/slow coordinates
  [
  \mathbf r:=\frac{\mathbf x}{a},\qquad \mathbf R:=\frac{\mathbf x}{L},\qquad \eta:=\frac{a}{L}\ll1,
  ]
  so (\mathbf R=\eta \mathbf r).

**Two-scale permittivity:**
[
\boxed{
\varepsilon(\mathbf x)\ \leadsto\ \varepsilon(\mathbf r,\mathbf R),
}
]
with the key assumption:

* for each fixed (\mathbf R), (\varepsilon(\cdot,\mathbf R)) is **periodic in (\mathbf r)** on a small unit cell (\Omega),
* (\varepsilon) varies smoothly in (\mathbf R).

This is exactly the “zoom in → looks periodic” statement you’re using.

---

# 1) Two-scale calculus: the gradient split (with the missing prefactor handled)

Since (\mathbf r=\mathbf x/a) and (\mathbf R=\mathbf x/L),
[
\nabla_{\mathbf x}
==================

# \frac{1}{a}\nabla_{\mathbf r}+\frac{1}{L}\nabla_{\mathbf R}

\frac{1}{a}\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right).
]

It is standard to pull out the factor (1/a) and work with a **dimensionless operator**
[
\tilde{\mathcal{L}}:=a^2\mathcal{L}_{\mathbf x}.
]
Then
[
\boxed{
\tilde{\mathcal{L}}
===================

-\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right)\cdot
\left(\varepsilon^{-1}(\mathbf r,\mathbf R)\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right)\right).
}
]

This is the starting point for your (\eta)-expansion.

---

# 2) Introduce the Bloch “fast” derivative (D_{\mathbf r}(\mathbf k))

We will expand around a fixed carrier crystal momentum (\mathbf k_0) (dimensionless, in the small-cell Brillouin zone).

Define the Bloch covariant derivative
[
\boxed{
D_{\mathbf r}(\mathbf k_0):=\nabla_{\mathbf r}+i\mathbf k_0.
}
]

For any periodic function (u(\mathbf r)),
[
\nabla_{\mathbf r}\left(e^{i\mathbf k_0\cdot\mathbf r}u(\mathbf r)\right)
=========================================================================

e^{i\mathbf k_0\cdot\mathbf r}D_{\mathbf r}(\mathbf k_0)u(\mathbf r).
]

---

# 3) Frozen-(\mathbf R) local Bloch eigenproblem (the cell problem)

For each fixed (\mathbf R), define the **local Bloch operator**
[
\boxed{
\mathcal{L}*0(\mathbf R,\mathbf k)
:=
-,D*{\mathbf r}(\mathbf k)\cdot\left(\varepsilon^{-1}(\mathbf r,\mathbf R),D_{\mathbf r}(\mathbf k)\right),
}
]
acting on (\Omega)-periodic functions of (\mathbf r).

The **local band structure** is defined by
[
\boxed{
\mathcal{L}*0(\mathbf R,\mathbf k),u*{n\mathbf k}(\mathbf r;\mathbf R)
======================================================================

\lambda_n(\mathbf R,\mathbf k),u_{n\mathbf k}(\mathbf r;\mathbf R),
\qquad
\lambda_n=\frac{\omega_n^2}{c^2}.
}
]

Normalize on the small cell with
[
\boxed{
\langle f,g\rangle_\Omega :=\frac{1}{|\Omega|}\int_\Omega f^*(\mathbf r),g(\mathbf r),d\mathbf r,
\qquad
\langle u_{m\mathbf k},u_{n\mathbf k}\rangle_\Omega=\delta_{mn}.
}
]

> This is the only place where “Bloch theorem” is used: it’s applied **locally**, i.e. for frozen (\mathbf R).
> Globally the moiré medium is not periodic (unless commensurate), so global Bloch labels are not guaranteed — but the envelope method does not need them.

---

# 4) Full multi-band two-scale ansatz

Pick a set of (N) local bands near the physics of interest (single band is (N=1) as a special case). We work at the carrier (\mathbf k_0).

Define the local periodic basis functions
[
u_n(\mathbf r;\mathbf R):=u_{n\mathbf k_0}(\mathbf r;\mathbf R),\quad n=1,\dots,N.
]

**Multi-band ansatz:**
[
\boxed{
H(\mathbf r,\mathbf R)
======================

e^{i\mathbf k_0\cdot\mathbf r}\sum_{n=1}^N F_n(\mathbf R),u_n(\mathbf r;\mathbf R).
}
]

Key point: **(u_n) depends on (\mathbf R)** in general. This is where your earlier derivation becomes incomplete if you drop it.

---

# 5) Expand the operator in powers of (\eta)

Start from the dimensionless operator
[
\tilde{\mathcal{L}}
===================

-\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right)\cdot
\left(\varepsilon^{-1}(\mathbf r,\mathbf R)\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right)\right).
]

Acting on (H=e^{i\mathbf k_0\cdot\mathbf r}\psi) with (\psi(\mathbf r,\mathbf R):=\sum_n F_n(\mathbf R)u_n(\mathbf r;\mathbf R)), we use
[
\nabla_{\mathbf r}H=e^{i\mathbf k_0\cdot\mathbf r}D_{\mathbf r}(\mathbf k_0)\psi,\qquad
\nabla_{\mathbf R}H=e^{i\mathbf k_0\cdot\mathbf r}\nabla_{\mathbf R}\psi,
]
so
[
\left(\nabla_{\mathbf r}+\eta\nabla_{\mathbf R}\right)H
=======================================================

e^{i\mathbf k_0\cdot\mathbf r}\left(D_{\mathbf r}\psi+\eta\nabla_{\mathbf R}\psi\right).
]

Therefore
[
\tilde{\mathcal{L}}H
====================

e^{i\mathbf k_0\cdot\mathbf r},
\Big(
\mathcal{L}^{(0)} + \eta,\mathcal{L}^{(1)} + \eta^2,\mathcal{L}^{(2)}
\Big)\psi,
]
with the **exact** decomposition
[
\boxed{
\begin{aligned}
\mathcal{L}^{(0)} &:= -D_{\mathbf r}\cdot\left(\varepsilon^{-1}D_{\mathbf r}\right),[2mm]
\mathcal{L}^{(1)} &:= -D_{\mathbf r}\cdot\left(\varepsilon^{-1}\nabla_{\mathbf R}\right);-;\nabla_{\mathbf R}\cdot\left(\varepsilon^{-1}D_{\mathbf r}\right),[2mm]
\mathcal{L}^{(2)} &:= -\nabla_{\mathbf R}\cdot\left(\varepsilon^{-1}\nabla_{\mathbf R}\right).
\end{aligned}}
]
(Here (\varepsilon^{-1}=\varepsilon^{-1}(\mathbf r,\mathbf R)) and (D_{\mathbf r}=D_{\mathbf r}(\mathbf k_0)).)

This is the rigorous version of your slide’s (L_0,L_1,L_2).

---

# 6) Insert the multi-band ansatz into (\mathcal{L}^{(j)}): where the new terms enter

We need derivatives of
[
\psi(\mathbf r,\mathbf R)=\sum_{n}F_n(\mathbf R)u_n(\mathbf r;\mathbf R).
]

Since (F_n) depends only on (\mathbf R),
[
D_{\mathbf r}\psi=\sum_n F_n,D_{\mathbf r}u_n,
]
but
[
\nabla_{\mathbf R}\psi=\sum_n (\nabla_{\mathbf R}F_n),u_n+\sum_n F_n,(\nabla_{\mathbf R}u_n).
]

**This is the structural origin of everything you were missing:**

* ((\nabla_{\mathbf R}F_n),u_n) → the usual envelope-gradient terms
* (F_n,(\nabla_{\mathbf R}u_n)) → geometric/non-adiabatic coupling

You do *not* need closed-form (u_n) to use these; you only need the matrix elements after projection (next section).

---

# 7) Small-cell projection: derive the envelope equations

Define the projection onto the chosen (N)-band subspace by inner products on (\Omega).

Project the eigenproblem
[
\tilde{\mathcal{L}}H=\tilde{\lambda},H,
\qquad
\tilde{\lambda}:=a^2\lambda=a^2\frac{\omega^2}{c^2},
]
onto each basis state (u_m(\cdot;\mathbf R)) by multiplying by (u_m^*) and integrating over (\Omega). Because the fast factor (e^{i\mathbf k_0\cdot\mathbf r}) cancels,
[
\boxed{
\sum_{n=1}^N
\left\langle u_m,\big(\mathcal{L}^{(0)}+\eta\mathcal{L}^{(1)}+\eta^2\mathcal{L}^{(2)}\big),(F_n u_n)\right\rangle_\Omega
========================================================================================================================

\tilde{\lambda},F_m.
}
]

Now we identify the *objects* that appear when doing this systematically.

---

## 7.1 Non-Abelian Berry connection (in (\mathbf R)-space)

Define the matrix-valued connection
[
\boxed{
\big(A_j(\mathbf R)\big)*{mn}:=
i\left\langle u_m,\partial*{R_j}u_n\right\rangle_\Omega,
\qquad j=x,y.
}
]

This is a **non-Abelian gauge field** on the (N)-dimensional band subspace (for (N=1) it reduces to a scalar Berry connection).

Then one has the fundamental identity
[
\left\langle u_m,\partial_{R_j}(F_n u_n)\right\rangle_\Omega
============================================================

\partial_{R_j}F_m;-;i\sum_n (A_j)_{mn}F_n.
]

Define the **gauge-covariant derivative** on the envelope vector (F=(F_1,\dots,F_N)^\top):
[
\boxed{
\mathcal{D}*j := \partial*{R_j} - iA_j(\mathbf R),
\qquad
(\mathcal{D}*jF)*m = \partial*{R_j}F_m - i\sum_n(A_j)*{mn}F_n.
}
]

This is exactly the “( \nabla_{\mathbf R}F \to (\nabla_{\mathbf R}-i\mathbf A)F)” statement, but now **matrix-valued**.

---

## 7.2 Born–Huang / quantum-geometric correction (out-of-subspace mixing)

Let (P(\mathbf R)) be the projector onto the chosen band subspace:
[
P=\sum_{n=1}^N |u_n\rangle\langle u_n|.
]

Define the **Born–Huang (matrix) potential** as
[
\boxed{
\Phi_{mn}(\mathbf R)
:=
\sum_{j=x,y}
\left\langle \partial_{R_j}u_m,,(1-P),\partial_{R_j}u_n\right\rangle_\Omega.
}
]

* For (N=1) this reduces to the familiar scalar
  (\Phi=\sum_j|(1-P)\partial_{R_j}u|^2).
* It is **gauge invariant** (phase/gauge changes inside the subspace do not remove it).
* It measures how strongly the chosen subspace fails to be “adiabatically closed” under (\mathbf R)-variation.

This is the mathematically clean way to include (F,\nabla_{\mathbf R}u) without explicit (u)-formulas: it appears through projected matrix elements.

---

# 8) Effective envelope Hamiltonian in compact “thesis form”

At this point, the cleanest statement is:

> The projected operator on the (N)-band subspace yields an (N\times N) matrix differential operator acting on (F(\mathbf R)), of the form “local band energies + first-order drift + second-order kinetic + geometric potentials”.

I’ll write the final form first, then define each term precisely.

---

## 8.1 Final multi-band envelope eigenproblem

Choose a reference eigenvalue (\lambda_\mathrm{ref}) (e.g. a reference local band edge at some (\mathbf R_\ast)). Define (\Delta\lambda := \lambda-\lambda_\mathrm{ref}).

Then the multi-band envelope equation can be written as
[
\boxed{
\Big[
\Lambda(\mathbf R) - \lambda_\mathrm{ref},I
;+;
\eta,\hat{H}^{(1)}(\mathbf R, \mathcal{D})
;+;
\eta^2,\hat{H}^{(2)}(\mathbf R, \mathcal{D})
\Big]F(\mathbf R)
=================

\Delta\lambda,F(\mathbf R),
}
]
where (F(\mathbf R)\in\mathbb{C}^N), and:

* (\Lambda(\mathbf R)) is the diagonal matrix of local band energies at (\mathbf k_0):
  [
  \boxed{
  \Lambda_{mn}(\mathbf R):=\lambda_n(\mathbf R,\mathbf k_0),\delta_{mn}.
  }
  ]

* (\mathcal{D}=\nabla_{\mathbf R}-i\mathbf A(\mathbf R)) is the **non-Abelian covariant derivative** defined above.

* (\hat{H}^{(1)}) is the **drift / group-velocity** term (linear in (\mathcal{D})).

* (\hat{H}^{(2)}) contains the **effective mass / kinetic** term (quadratic in (\mathcal{D})) plus the **Born–Huang / geometric** potential and additional “slow coefficient” corrections.

I’ll now define (\hat{H}^{(1)}) and (\hat{H}^{(2)}) in a way that is both rigorous and computable from local cell solves.

---

# 9) Drift and effective mass: how they arise (and how to compute them)

There are two equivalent (and important) viewpoints:

### Viewpoint A: “envelope gradient = small (\Delta\mathbf k)” (dispersion Taylor expansion)

If (F(\mathbf R)) varies slowly, its Fourier components correspond to small (\mathbf q) in (\mathbf R)-space, which translate to small (\Delta\mathbf k\sim\eta\mathbf q) in the fast phase. This is the standard route to drift and mass from band derivatives.

### Viewpoint B: k·p / operator derivatives (matrix elements inside the subspace)

This is the route that remains valid and computable even with multi-band coupling.

I’ll present B (most “thesis-grade”), and point out the single-band simplification that gives your slide’s (M^{-1}=\partial_{kk}\omega).

---

## 9.1 Define the k-derivative operators (velocity and curvature operators)

For the frozen operator (\mathcal{L}_0(\mathbf R,\mathbf k)), define
[
\boxed{
\mathcal{V}*i(\mathbf R):=\left.\frac{\partial \mathcal{L}*0(\mathbf R,\mathbf k)}{\partial k_i}\right|*{\mathbf k_0},
\qquad i=x,y,
}
]
and
[
\boxed{
\mathcal{W}*{ij}(\mathbf R):=\left.\frac{\partial^2 \mathcal{L}*0(\mathbf R,\mathbf k)}{\partial k_i\partial k_j}\right|*{\mathbf k_0}.
}
]

These are explicit in terms of (D_{\mathbf r}(\mathbf k)), because
(\partial_{k_i}D_{\mathbf r}=i\hat{\mathbf e}_i).

From the definition
[
\mathcal{L}*0=-D*{\mathbf r}\cdot(\varepsilon^{-1}D_{\mathbf r}),
]
one finds (operator identities)
[
\mathcal{V}_i
=============

-i\Big[\hat{\mathbf e}*i\cdot(\varepsilon^{-1}D*{\mathbf r}) + D_{\mathbf r}\cdot(\varepsilon^{-1}\hat{\mathbf e}*i)\Big]*{\mathbf k_0},
]
and (\mathcal{W}*{ij}) is a bounded Hermitian operator built from (\varepsilon^{-1}) (for many scalar cases it essentially gives a direct (\propto \varepsilon^{-1}\delta*{ij}) term plus adjoints).

Now define their **matrix elements in the local Bloch basis**:
[
\boxed{
v^{(i)}*{mn}(\mathbf R)
:=
\langle u_m,,\mathcal{V}*i(\mathbf R),u_n\rangle*\Omega,
\qquad
w^{(ij)}*{mn}(\mathbf R)
:=
\langle u_m,,\mathcal{W}*{ij}(\mathbf R),u_n\rangle*\Omega.
}
]

These are computable from your local cell eigensolutions.

---

## 9.2 Drift term (\hat{H}^{(1)})

To leading order in slow gradients, the drift term is
[
\boxed{
\big(\hat{H}^{(1)}F\big)_m
==========================

\sum_{n=1}^N \sum_{i=x,y}
v^{(i)}_{mn}(\mathbf R),\big(-i\mathcal{D}_iF\big)_n.
}
]

* For a **single isolated band**, (v^{(i)}=\partial_{k_i}\lambda_n) (Hellmann–Feynman), so this is precisely the group-velocity drift.
* At a **band extremum**, (\nabla_{\mathbf k}\lambda_n(\mathbf R,\mathbf k_0)=0), so the diagonal drift vanishes — but off-diagonal drift can still matter in multi-band settings near degeneracy.

This is the mathematically correct version of your “drift is zero at band extremum”.

---

## 9.3 Second-order kinetic term (\hat{H}^{(2)}): effective mass + geometric terms

At second order, you get a quadratic differential operator in (\mathcal{D}). In matrix form,
[
\boxed{
\big(\hat{H}^{(2)}F\big)_m
==========================

\frac12
\sum_{n=1}^N\sum_{i,j=x,y}
\big(M^{-1}*{ij}(\mathbf R)\big)*{mn},
\big(-i\mathcal{D}*i\big)\big(-i\mathcal{D}*j\big)F_n
;+;
\sum*{n=1}^N U*{mn}(\mathbf R),F_n.
}
]

There are two pieces:

1. the **effective mass / curvature tensor** (M^{-1}_{ij}) (matrix-valued in general),
2. the **scalar (matrix) potential corrections** (U(\mathbf R)) including Born–Huang and slow-coefficient terms.

### (i) Effective mass / curvature tensor

A robust (and computable) expression for the curvature tensor inside the chosen band subspace is
[
\boxed{
\big(M^{-1}*{ij}\big)*{mn}
==========================

w^{(ij)}*{mn}
+
\sum*{\ell\notin {1,\dots,N}}
\left(
\frac{v^{(i)}*{m\ell},v^{(j)}*{\ell n}}{\lambda_n-\lambda_\ell}
+
\frac{v^{(j)}*{m\ell},v^{(i)}*{\ell n}}{\lambda_m-\lambda_\ell}
\right).
}
]
This is the standard Löwdin partitioning / second-order k·p correction: it shows explicitly how “other bands” renormalize the mass.

**Single isolated band simplification.**
If (N=1) and you ignore the sum, you recover
[
M^{-1}*{ij}(\mathbf R)\approx \partial*{k_i}\partial_{k_j}\lambda(\mathbf R,\mathbf k_0).
]
If you work in (\omega) instead of (\lambda=\omega^2/c^2), then at a band extremum (where (\partial_k\omega=0)),
[
\partial_{k_i}\partial_{k_j}\lambda
===================================

\frac{2\omega_0}{c^2},\partial_{k_i}\partial_{k_j}\omega_0,
]
so your slide’s “mass tensor from (\partial_{kk}\omega)” is consistent.

### (ii) Geometric/Born–Huang and slow-coefficient corrections

Set
[
\boxed{
U(\mathbf R) = U_{\text{BH}}(\mathbf R) + U_{\text{sc}}(\mathbf R).
}
]

* The **Born–Huang (quantum-geometric) matrix** is essentially (\Phi(\mathbf R)) (defined earlier), potentially weighted by coefficients depending on the exact operator normalization. Structurally:
  [
  \boxed{
  (U_{\text{BH}})*{mn}\ \propto\ \Phi*{mn}(\mathbf R).
  }
  ]
  This term is *exactly* the cost of “(\partial_{\mathbf R}u) has components outside the chosen subspace”.

* The **slow-coefficient correction** (U_{\text{sc}}) comes from explicit (\mathbf R)-derivatives of (\varepsilon^{-1}(\mathbf r,\mathbf R)) inside (\mathcal{L}^{(1)}) and (\mathcal{L}^{(2)}). It can be written in terms of the operator derivative
  [
  \partial_{R_j}\mathcal{L}*0(\mathbf R,\mathbf k_0)
  =
  -,D*{\mathbf r}\cdot\big((\partial_{R_j}\varepsilon^{-1})D_{\mathbf r}\big),
  ]
  and its matrix elements.
  A practical and clean form is:
  [
  \boxed{
  \big(\partial_{R_j}\Lambda\big)*{mn}
  =
  \left\langle u_m,(\partial*{R_j}\mathcal{L}*0)u_n\right\rangle*\Omega
  \quad\text{(diagonal gives Hellmann–Feynman; off-diagonal drives mixing).}
  }
  ]
  These terms can be absorbed into (U_{\text{sc}}) and into additional pieces of the connection when you cast everything in covariant form.

---

# 10) Putting it all together: the “named physics” dictionary

Your full envelope operator (matrix PDE on (F)) has this anatomy:

### **(A) Local band energy / “potential”**

[
\Lambda(\mathbf R)-\lambda_\mathrm{ref}I
]
This is the generalization of (V(\mathbf R)=\omega_0(\mathbf R)-\omega_0^\mathrm{ref}). It’s the dominant trapping physics: local band-edge shifts.

### **(B) Drift (group velocity)**

[
\eta\sum_i v^{(i)}(\mathbf R),(-i\mathcal{D}_i)
]

* In a single isolated band: (v=\nabla_{\mathbf k}\lambda) or (v_g=\nabla_{\mathbf k}\omega).
* At extrema, the diagonal drift vanishes, but multi-band off-diagonal drift can still matter.

### **(C) Effective mass / kinetic energy**

[
\frac{\eta^2}{2}\sum_{i,j} M^{-1}_{ij}(\mathbf R),(-i\mathcal{D}_i)(-i\mathcal{D}_j)
]

* Single band: curvature of dispersion (Hessian).
* Multi-band: curvature is renormalized by coupling to other bands (the Löwdin sum).

### **(D) Non-Abelian Berry connection (gauge field)**

[
\mathcal{D}=\nabla_{\mathbf R}-i\mathbf A(\mathbf R),
\qquad A_{j,mn}=i\langle u_m,\partial_{R_j}u_n\rangle
]
This is the systematic way to include (\partial_{\mathbf R}u) *within* the chosen subspace.

### **(E) Born–Huang / quantum-geometric potential**

[
U_{\text{BH}}(\mathbf R)\ \propto\ \Phi(\mathbf R)
,\qquad
\Phi_{mn}=\sum_j\langle \partial_{R_j}u_m,(1-P)\partial_{R_j}u_n\rangle
]
This captures the influence of **bands outside your chosen set** induced by slow modulation, without explicitly carrying them as envelope components.

---

# 11) What must be computed in practice (and how this stays feasible without MPB on the moiré)

You never need “global moiré bands”. You need **local frozen-cell data** at sampled (\mathbf R) points (your two-atom surrogate trick).

For each (\mathbf R) sample:

1. Compute (\lambda_n(\mathbf R,\mathbf k)) and (u_{n\mathbf k}(\mathbf r;\mathbf R)) for (n=1,\dots,N) and (\mathbf k) near (\mathbf k_0).

2. Compute drift/velocity matrices (v^{(i)}_{mn}(\mathbf R)).
   Either:

* by finite differences of (\lambda_n(\mathbf k)) for diagonal terms, and overlaps for off-diagonals, **or**
* by matrix elements of (\mathcal{V}_i) (operator derivative w.r.t. (k)).

3. Compute curvature / mass tensor:

* single band: Hessian of (\lambda(\mathbf k)) near (\mathbf k_0),
* multi-band: either directly fit the small-k (N\times N) k·p Hamiltonian, or use the Löwdin formula above.

4. Compute Berry connection matrices (A_j(\mathbf R)):
   Use overlaps between neighboring (\mathbf R) samples:
   [
   (A_j)*{mn}\approx i,\frac{\langle u_m(\mathbf R),u_n(\mathbf R+\Delta R_j)\rangle*\Omega-\delta_{mn}}{\Delta R_j}
   ]
   after a consistent gauge choice. (Parallel-transport gauge makes this stable.)

5. Compute Born–Huang / geometric term (\Phi(\mathbf R)):
   Either:

* explicitly via out-of-subspace modes (if you have them), or
* via perturbation identity (no explicit derivatives needed):
  [
  \langle u_\ell,\partial_{R_j}u_n\rangle
  =
  \frac{\langle u_\ell,(\partial_{R_j}\mathcal{L}*0)u_n\rangle}{\lambda_n-\lambda*\ell},
  \qquad \ell\notin{1,\dots,N},
  ]
  then sum (|\cdot|^2).

Finally: solve the resulting **matrix PDE eigenproblem** in (\mathbf R)-space for (F(\mathbf R)).

This is exactly the “MPB fails globally → local cell solves + envelope PDE” strategy you described — now extended to be **mathematically complete**.

---

# 12) The critical warning that often kills “predicted cavity but Meep sees none”

Even with perfect multi-band theory, a cavity requires that the envelope eigenfrequency lies in a regime where a localized mode can exist. In this framework, the most common *systematic* failure is:

### Using the wrong definition of “local band edge”

For a trapping potential, the right object is often
[
\lambda_{\mathrm{edge}}(\mathbf R)=\min_{\mathbf k}\lambda_n(\mathbf R,\mathbf k)
\quad\text{(or max, depending on which edge).}
]
If (\mathbf k_0) drifts with (\mathbf R), then “(\lambda(\mathbf R,\mathbf k_0))” can show a beautiful fake well that does not correspond to a true local spectral edge → Meep won’t find a bound state.

So in practice, your local cell sweep should also check whether the extremum is actually at the chosen (\mathbf k_0), or whether you need (\mathbf k_0(\mathbf R)) or a larger band subspace.
