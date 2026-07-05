# Repair Report for Phases 1-3

I went through your two theory files — [theoretical_background.typ](sandbox:/mnt/data/theoretical_background.typ) and [envelope_approximation.typ](sandbox:/mnt/data/envelope_approximation.typ) — together with the three pipeline stages. Bottom line:

**Your current pipeline is not yet a faithful implementation of the written Hamiltonian.**
There are several real mismatches, and at least a few of them are strong enough to fully explain “completely off” physics.

The biggest problems are these.

## 1. Your written theory and your implementation are not using the same polarization / inner-product structure

Your theory document is written in the **TE / (H_z)** formulation, with the scalar operator picture as the clean starting point. But your implementation is built around **MPB E-fields**, uses an **(\varepsilon)-weighted inner product**, and the velocity operator code is explicitly written for **TM** using the (E_z) component only. That is not a cosmetic difference — it changes the operator, the correct normalization, and the meaning of the Berry/Born–Huang objects. In code, Phase 2 explicitly states the (\varepsilon)-weighted E-field orthogonality and computes Berry using (i\langle u_m|\varepsilon|\partial_j u_n\rangle), and the velocity matrix routine says “For TM…” and extracts the (E_z) component.  

So unless you intentionally re-derived the entire envelope theory for TM in E-field language, your code is **not implementing the same theory you wrote down**.

That is a first-order blocker.

## 2. Your theory is written in (\lambda = \omega^2/c^2), but the code is assembled in (\omega)-space

Your theory Hamiltonian in `envelope_approximation.typ` is written as a **(\Delta\lambda)** problem. But Phase 1 extracts

* `omega`
* `vg = dω/dk`
* `M_inv = d²ω/dk²`

and Phase 2 builds

* `Lambda = omega - omega_ref`

rather than a (\lambda - \lambda_{\rm ref}) operator. The code itself says Phase 1 stores frequencies, group velocities, and mass tensors per band, and Phase 2 constructs the Λ potential from `omega_grid`.  

That can be fine **only if** every term was consistently transformed from the (\lambda)-theory into an (\omega)-theory. I do **not** see evidence that this was done completely.

In fact, I see the opposite:

* the field-based off-diagonal velocity divides by `2 * omega_scale`, which is clearly a (\lambda \to \omega) conversion step, 
* but the diagonal mass is still just finite differences of (\omega(k)), not a rigorously transformed (\omega)-space analogue of the mass tensor formula from the theory. 

So your Hamiltonian currently mixes a **(\lambda)-derived formalism** with a **partly ad hoc (\omega)-space implementation**.

That is another major source of wrong scaling.

## 3. The full multiband Hamiltonian from the theory is not actually assembled

Your theory includes more than just diagonal potential + drift + diagonal kinetic + Born–Huang. In particular, the full retained-space Hamiltonian needs the proper multiband tensor structure.

But your Phase 2 code explicitly says:

* **off-diagonal (M^{-1}_{mn}) is deferred**
* only diagonal mass blocks are populated
* Phase 3 “only reads diagonal blocks” of (M^{-1}). 

That means your “multiband” Hamiltonian is **not actually full multiband in the kinetic sector**.

Even if you enable off-diagonal Berry connection in Phase 3, the mass tensor is still effectively diagonal in band space. So the kinetic coupling structure is incomplete.

This is not a small omission. If your retained manifold is genuinely coupled, this changes the physics.

## 4. A term present in the theory is completely missing in the code: (U_{\mathrm{sc}})

Your theory includes the second-order slow-coefficient term (U_{\mathrm{sc}}). I do not see any implementation of that term anywhere in the uploaded Phase 1–3 pipeline.

I also do not see any placeholder or TODO for it. By contrast, Born–Huang at least has explicit placeholder / commentary in the code. 

So as of now, your implemented Hamiltonian is missing a term that your written theory says belongs at the same order as Born–Huang.

## 5. Your coordinate treatment is very likely incomplete: the moiré basis metric is basically not used in the operator assembly

This one is subtle but very important.

You define a nontrivial moiré basis (B_{\text{moire}} = (R(\theta)-I)^{-1} B_{\text{mono}}) in Phase 1. 

But in Phase 3, the actual derivative operators are built using only scalar spacings

* `dR1 = L_moire / Ns1`
* `dR2 = L_moire / Ns2`

and the comments say this is the “physical grid spacing in R-coordinates.” 

That only makes sense if your coordinate directions are orthogonal Cartesian directions with equal treatment. For a general moiré basis — especially hexagonal / triangular — that is not the full metric story.

Worse:

* `B_moire` is passed into operator builders,
* but the operator assembly does not really use the basis geometry, only `L_moire = ||B_moire[:,0]||` as a scalar length. 

So the implementation is effectively treating the slow derivatives as if the geometry were encoded by a single scalar moiré length, not by the full basis / metric tensor.

That is a very plausible reason for broken scaling and wrong anisotropy / symmetry.

## 6. Your scaling convention is internally mixed

This is the issue I flagged while working.

Phase 3 headers and docstrings still describe the Hamiltonian as
[
H = \Lambda + \eta T_{\rm drift} + \eta^2 K + \eta^2 \Phi_{\rm BH},
]
but the later comments say the explicit (\eta,\eta^2) factors were removed because derivatives are now taken with respect to physical moiré coordinates, and the implementation instead uses (1/(2\pi)) and (1/(2\pi)^2) factors.  

And indeed:

* drift builder ignores `eta` in practice and uses `coeff = 1/(2*pi)` 
* Born–Huang builder also says the explicit (\eta^2) was removed. 

This might be correct. It might also be wrong. The real problem is that the codebase currently contains **two conflicting scaling stories**:

* the old “explicit (\eta), (\eta^2)” story,
* and the newer “physical-(R)-derivative so no extra (\eta)” story.

Until you pick one convention and re-derive every term in that convention, you do **not** have a trustworthy Hamiltonian.

## 7. The registry-to-moiré interpolation contains an unexplained `+0.5` shift

This looks dangerous.

In Phase 2, when registry-grid data are interpolated back onto the moiré grid, the query points are taken as
[
\text{query} = \mathrm{mod}(\delta_{\rm frac} + 0.5, 1).
]
The same pattern shows up in the interpolation logic around Berry / Born–Huang. 

I did not find a theoretical justification for this half-cell offset in the uploaded material.

If that shift is wrong, then you are systematically sampling the universal master map at the wrong registry points. That would corrupt **all** of:

* (\Lambda)
* (A)
* (\Phi_{\rm BH})
* (v_{mn})

with one single convention error.

This is absolutely something I would test immediately.

## 8. Your “smooth retained manifold” assumption is not enforced robustly enough

Your theory needs a **smooth projector / smooth retained subspace** over the moiré cell.

What I see in the code is:

* fixed band indices coming out of MPB,
* per-band scalar gauge smoothing,
* no real subspace-tracking machinery across registry for near-degenerate manifolds.

The current gauge fix is **Abelian per band**, not a genuinely **non-Abelian subspace transport**. That is fine only when individual bands stay cleanly separated everywhere. The code itself explicitly deprecates the old non-Abelian SVD gauge because it broke (\varepsilon)-orthogonality for E-fields. 

So for coupled manifolds, the current fix is not enough to guarantee that the retained basis is the smooth physical one your theory assumes.

## 9. Born–Huang is not fully auditable from the uploaded files alone

There is a placeholder zero Born–Huang function in Phase 2, and the comments explicitly say proper Born–Huang requires Bloch-function derivatives or operator derivatives.

But the main pipeline appears to call a helper from another module (`phasesV3.bloch_fields`) to compute Born–Huang from fields. That helper file was **not uploaded here**, so I cannot verify whether your actual active Born–Huang implementation is correct.

So I can say this honestly:

* **the Born–Huang theory is present**
* **the uploaded pipeline shows the right intent**
* but **I cannot certify the actual active BH code path from the files you gave me**

That matters, because BH is one of the terms you specifically want guaranteed.

---

# What is missing to assemble the full honest Hamiltonian

If you want a Hamiltonian you can trust, you still need these pieces.

## A. Pick one formalism and enforce it everywhere

You need one of these two worlds:

### Option 1: stay in the theory’s TE / (H_z), (\lambda)-space world

Then all objects should be computed for that problem:

* local eigenvectors in the same polarization / field representation
* correct flat or operator-induced inner product for that formulation
* (v^{(i)}*{mn}), (M^{-1}*{ij,mn}), BH, (U_{\rm sc}) all in the same (\lambda)-space operator language

### Option 2: deliberately switch to TM / E-field / (\omega)-space

Then you need a **fresh complete derivation** of the envelope Hamiltonian in exactly that convention, including:

* correct inner product
* correct covariant derivative
* correct transformation of drift, mass, BH, and slow-coefficient terms from (\lambda) to (\omega)

Right now you are in between.

## B. Implement the full multiband kinetic tensor

You still need:

* off-diagonal (M^{-1}_{ij,mn}),
* and the remote-band Löwdin contributions from omitted bands.

Your own Phase 2 comments admit this is deferred. 

Without it, the Hamiltonian is incomplete.

## C. Implement (U_{\mathrm{sc}})

It is in the theory and absent in the code.

## D. Fix the coordinate / metric treatment

You need to decide whether the envelope lives in:

* fractional moiré coordinates (s), with explicit basis/metric tensors everywhere,
  or
* Cartesian physical coordinates (R), with all interpolated fields and derivatives transformed accordingly.

At the moment, the code uses fractional-grid data, scalar moiré lengths, and physical-(R) language in a way that is not cleanly closed.

## E. Add robust subspace tracking

For each registry point, you need overlap-based tracking of the retained subspace, not just fixed raw band indices.

Otherwise all subsequent geometric quantities can be discontinuous for reasons that have nothing to do with physics.

---

# What I would fix first, in order

## 1. Resolve the TE/TM mismatch first

This is the most fundamental issue.

If your thesis theory is TE/(H_z), then do **not** continue building Phase 2/3 on TM E-field formulas.

## 2. Resolve the (\lambda) vs (\omega) mismatch second

Either:

* rewrite code in (\lambda)-space,
  or
* fully re-derive the theory in (\omega)-space.

Do not keep the current hybrid state.

## 3. Remove or justify the `+0.5` interpolation shift

This is a fast, high-yield sanity test.

## 4. Fix the coordinate geometry

Make `B_moire` enter the actual differential operator, not just plotting and a scalar length extraction.

## 5. Implement full (M^{-1}*{mn,ij}) and (U*{\rm sc})

Only then do you have something close to the theory’s full Hamiltonian.

---

# How Blaze2D can close the gap cleanly

This is where your custom solver is a huge advantage.

With Blaze2D, the rigorous route is:

1. Expose the local Bloch eigenvectors in the **same formulation as the theory**.
2. Expose matrix elements of

   * (\partial_{k_i} L_0),
   * (\partial_{k_i}\partial_{k_j} L_0),
   * (\partial_{R_j} L_0) or the equivalent registry derivative.
3. Build:

   * (v^{(i)}_{mn}),
   * full Löwdin-corrected (M^{-1}_{ij,mn}),
   * Born–Huang from the perturbative projector formula,
   * (U_{\rm sc}) from the slow-coefficient derivative terms.
4. Track the retained projector by overlap singular values across registry points.

That route is much better than trying to infer everything from MPB black-box exports.

---

# My honest verdict

Here is the blunt version:

**Your current pipeline is not merely “potentially incomplete.” It is definitely incomplete relative to the written theory.**

The most critical hard mismatches are:

1. **TE/(H_z) theory vs TM/E-field implementation**
2. **(\lambda)-space theory vs (\omega)-space code**
3. **missing (U_{\rm sc})**
4. **missing off-diagonal / Löwdin-complete mass tensor**
5. **incomplete coordinate / metric handling**
6. **suspicious unexplained `+0.5` registry shift**
7. **insufficient subspace tracking for coupled manifolds**
