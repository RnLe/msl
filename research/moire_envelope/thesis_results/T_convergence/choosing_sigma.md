Yes — your “magnifying glass” picture is a good one.

And the key realization is:

> **(\sigma) is not part of the physics. It is part of how you ask the solver to show you a specific spectral window.**

So the right question is not “what is the one true sigma?”
It is:

> **Which spectral window do I want to resolve, and what scalar (\sigma) best targets it?**

That distinction matters a lot now that you are doing multiband physics.

## The first big correction

You said:

> I start from a monolayer candidate, so naturally I take that candidate frequency as sigma.

That is a **very reasonable starting choice**.
But it is not automatically the best final choice.

Why?

Because once you go from:

* one monolayer candidate band
  to
* an (N)-band local manifold
  to
* a moiré-coupled miniband problem,

the physically relevant object is no longer “the one monolayer eigenvalue.”
It is now a **whole cluster / manifold** of states derived from those target bands.

So in the multiband case, (\sigma) should usually target the **center or edge of the manifold you care about**, not one arbitrary original eigenvalue.

---

## What (\sigma) should mean in your thesis

The clean interpretation is:

> (\sigma) is the reference spectral value around which the eigensolver resolves the effective moiré Hamiltonian.

That is the thesis-safe phrasing.

Not:

* “the physical energy”
* not “the cavity frequency”
* not “the actual magic angle marker”

Just:

* the numerical target around which you inspect the spectrum.

---

## What I would do in your framework

There are really three different cases.

### 1. Isolated extremum / band edge

Suppose you pick a monolayer band extremum that is spectrally well separated.

Then the natural physics question is:

* what happens to the minibands descending from this extremum?

In that case, a good (\sigma) is near that extremal frequency.

More precisely:

* if you want the **lowest miniband sector**, choose (\sigma) slightly **below** the expected lower edge
* if you want the **upper edge**, choose it slightly above
* if you want the whole local cluster, choose something near its center

So yes: for isolated extrema, your candidate frequency is often a good starting sigma.

---

### 2. Dirac point / touching point

Here the situation is different.

A Dirac point is usually not interesting because of “lowest state” or “highest state,” but because of the **crossing structure near a specific reference frequency**.

So here the natural thing is:

> choose (\sigma) near the Dirac frequency.

And if your effective Hamiltonian is written relative to a reference
[
H_{\text{eff}} - \lambda_{\mathrm{ref}},
]
then the Dirac point may indeed sit near **0 in the shifted problem**.

That is probably the origin of your thought:

> “aren’t Dirac points usually at (\omega=0)?”

Careful:

* physically, not necessarily
* **after shifting by a reference energy**, often yes

So if in your effective theory you subtract the Dirac-point reference frequency, then (\sigma=0) is perfectly sensible.

But in the raw photonic spectrum, the Dirac frequency is generally **not** literally zero.

---

### 3. Multiband manifold

This is your current main case.

Now you no longer have one “special candidate eigenvalue.”
You have (N) local target bands.

Then the cleanest scalar sigma is usually one of these:

#### Option A: center of the target manifold

[
\sigma \approx \frac{1}{N}\sum_{n=1}^N \lambda_n(\mathbf k_0,\mathbf R_0)
]

This is often the most natural multiband choice.

It means:

* I want to inspect the miniband manifold descending from these (N) parent bands.

#### Option B: center of the local gap around the manifold

If your chosen (N)-band manifold is isolated from bands above and below, then a very good choice is the center of that spectral window.

That is often even better numerically, because it targets the whole manifold rather than one edge.

#### Option C: near the band edge of the manifold

If your actual observable is:

* lowest miniband,
* band minimum,
* first flat band,
* lowest localized mode,

then choose (\sigma) near the lower edge of the targeted manifold.

This is often the right choice for “flat-band / magic-angle” searches.

---

## So for magic-angle searches, what should you do?

This is the important part.

If you want to detect **magic-angle flattening**, the true physical quantity is not “whatever is closest to the monolayer candidate frequency.”

The true object is:

* a miniband or miniband pair,
* and its dispersion across the moiré BZ.

So for that task, I would not over-commit to one original monolayer eigenvalue.

Instead:

### For honeycomb Dirac case

Choose (\sigma) near the **center of the Dirac pair**.

That is the right object, because magic-angle physics there is about the flattening of the Dirac-derived manifold.

So something like:
[
\sigma \approx \frac{\lambda_1+\lambda_2}{2}
]
at the reference point is very sensible.

If your effective Hamiltonian is shifted so that this center is zero, then (\sigma=0) is exactly the right numerical target.

### For square / hex extrema

There, if you are tracking a band edge or an isolated extremum, then yes:
choose (\sigma) near that extremal candidate or just slightly below it.

But once again, the real target is still the **resulting miniband manifold**, not the one original eigenvalue.

---

## What is the best practical rule?

Here is the rule I would use.

### If the target is a single isolated parent band

Use that band’s reference frequency.

### If the target is a Dirac pair or symmetry-protected touching

Use the **center of the touching pair**.

### If the target is a general (N)-band manifold

Use the **center of the manifold**.

### If you specifically care about the lower edge

Bias (\sigma) slightly below that edge.

That is probably the most robust practical recipe.

---

## Very important: (\sigma) should not define the result

This is the core sanity principle.

Your physical conclusion should not depend strongly on one exact sigma choice.

Instead, once you have picked a reasonable (\sigma), you should do a **small sigma sweep**:

[
\sigma_0-\delta,\quad \sigma_0,\quad \sigma_0+\delta
]

and check whether the extracted miniband observables are stable:

* bandwidth,
* gap,
* flatness ratio,
* localization.

If those stay stable, then your results are physical.

If they change wildly, then:

* either the spectrum is too dense,
* or you are not consistently targeting the same manifold.

That is why sigma is a **targeting tool**, not a defining physical parameter.

---

## Should you still look at spectrally isolated extrema?

Yes — absolutely.

That is still a very smart strategy.

Why?

Because envelope theories work best when the chosen target manifold is reasonably distinguishable from the rest.

So extrema and isolated features are attractive because:

* they are easier to track,
* less contaminated by other bands,
* and more likely to produce interpretable minibands.

Dirac points are the other obvious class because:

* symmetry makes them special,
* their effective physics is rich,
* and Berry / multiband effects are naturally strong there.

So your current candidate strategy is good:

* isolated extrema,
* spectrally clean features,
* Dirac points.

That is exactly where I would search.

---

## One subtle but important thing

You now have a multiband theory, but `eigsh` still wants a **scalar** sigma.

That means sigma cannot encode the full multiband structure by itself.

So the right mindset is:

> the multiband theory defines the manifold,
> the scalar sigma just tells the solver where to look inside or near that manifold.

That is all.

It is not supposed to fully represent the (N)-band theory.

---