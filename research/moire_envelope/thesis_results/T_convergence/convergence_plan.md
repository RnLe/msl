These results are actually very informative. They are not telling you “the solver is broken.” They are telling you **which observables are meaningful** and which ones are not.

## What your current honeycomb results mean

### 1. `gap_01` at a single (K)-point is not a physical gap

That is the biggest conclusion.

Your own data says it clearly:

* it changes wildly with `k = n_modes`,
* it changes wildly with grid resolution,
* while (\lambda_0) itself only moves at the few-percent level.

That is exactly the signature of a **discretized continuous miniband manifold**, not of a discrete isolated mode with a robust physical gap.

So for the thesis, I would treat:

* **single-(K) gap(_{01})** = **not a physical quantity**
* **single-(K) lowest eigenvalue (\lambda_0)** = at best a rough marker / reference level
* **BZ-dependent quantities** = the real physics

That is a good outcome, because it clarifies the story.

---

### 2. The `n_modes` sweep is behaving as expected

The monotonic downward drift of (\lambda_0) with larger `k` is not convergence failure in the usual sense.

It means:

* the operator has a dense low-energy sector,
* `eigsh` with shift-invert is exposing more of it as you ask for more states,
* so absolute lowest returned values are not stable “bound state energies.”

This is fully compatible with your earlier interpretation:
you are not looking at a discrete cavity state, but at a **continuum-like miniband sector**.

So `n_modes` is **not** a physical control knob. It is an **eigensolver sampling knob**.

That means you should not ask:

> “Does (\lambda_0) converge with k?”

You should ask:

> “Do the physically extracted miniband observables converge once k is large enough to capture the target manifold?”

That is a much better question.

---

### 3. The grid / FD results are actually decent

For the single-(K) eigenvalues, you are seeing:

* (\lambda_0) stable to about (1)–(3%),
* FD order effect around (2.4%),
* symmetry preserved.

That is not perfect, but it is absolutely usable for a thesis if you frame it correctly.

It means:

* absolute eigenvalues are moderately resolution-sensitive,
* but the solver is not wildly unstable.

So this is **not** a disaster signal. It is more like:

> the operator is numerically sane, but single-(K) level spacings are not the right physics.

---

### 4. Sigma sensitivity is expected and not alarming

This is normal for shift-invert.

Different (\sigma) values target different parts of the spectrum.
So the lesson is not “this is unstable,” but:

> Fix a clear sigma protocol and keep it consistent.

For example:

* same sigma choice for all parameter sweeps,
* or sigma tracked continuously from a reference band center,
* or sigma chosen near the first miniband manifold.

But do not present sigma sensitivity as a physical instability.

---

# The main thesis consequence

Your current tests support this statement:

> Single-(K) eigenvalue ordering is not the right object for physical claims in the moiré problem. The meaningful observables are miniband quantities extracted from the full (K)-dependent moiré dispersion.

That is actually a strong and clean thesis point.

So yes:

## You should extend the convergence tests to the BZ / miniband level

That is the right next step.

Not because it is “nice to have,” but because that is where the actual observables live:

* bandwidth,
* flatness,
* inter-miniband gap,
* maybe gap-to-bandwidth ratio.

---

# What I would do next

## Priority 1: Add **(K)-sampling density** as a convergence knob

This is missing from your current plan, and for your situation it is crucial.

If your bands look perfectly flat at small angle, the first question is:

> Is the (K)-path sampling dense enough to resolve the tiny variation?

So add a sweep like:

* (N_K = 20, 40, 80, 160) points along the mBZ path

and compare for the first 1–3 minibands:
[
\mathrm{BW}*\alpha = \max_K E*\alpha(K) - \min_K E_\alpha(K).
]

This may be the most important missing convergence test.

If the bandwidth keeps growing as you refine the path, then the earlier “perfect flatness” was under-resolution.

If it stabilizes near zero, then you can honestly say:

> within numerical resolution, the miniband is ultra-flat.

That is defensible physics.

---

## Priority 2: Move convergence analysis from single-(K) to miniband observables

For a representative angle, compute the first few minibands along the mBZ path for sweeps in:

* `Ns`
* `fd_order`
* `n_modes`
* `N_K` path density

Then extract:

* **Bandwidth**
  [
  \mathrm{BW}*\alpha
  = \max_K E*\alpha(K)-\min_K E_\alpha(K)
  ]
* **Direct interband gap**
  [
  \Delta_{12}^{\min}
  = \min_K \big(E_2(K)-E_1(K)\big)
  ]
* **Flatness ratio**
  [
  F_1 = \frac{\Delta_{12}^{\min}}{\mathrm{BW}_1}
  ]
  or similar

These are the quantities that should be in your thesis tables.

---

## Priority 3: Use two representative angles, not just one

I would not do everything only at (1.1^\circ).

Use:

* one **low angle** where bands are ultra-flat / suspiciously flat, e.g. (0.4^\circ) or (0.5^\circ),
* one **intermediate angle**, e.g. (1.1^\circ),
* optionally one **higher angle**, e.g. (3^\circ).

Why?

Because you need to separate:

* “unresolved because tiny”
  from
* “bug or missing (K)-dependence.”

If the same pipeline produces visible dispersion at (3^\circ) but near-zero dispersion at (0.4^\circ), that is a very good sign.

---

## Priority 4: Add eigensolver tolerance / residual checks

If your bandwidth is tiny, say (10^{-6})–(10^{-5}), then you must know whether that is above or below solver error.

So for the first few minibands, record:

* residual norms,
* eigsh tolerance,
* maybe repeated runs with stricter tolerance.

Because otherwise a “flat” band could just mean:

> the real dispersion is smaller than the numerical error floor.

That is a crucial sanity bound.

A very good thesis sentence would be:

> For the lowest miniband at (\theta=0.4^\circ), the extracted bandwidth is below / above the eigensolver uncertainty estimated from residuals.

That is mature numerical work.

---

# Should you expand the parameter ranges?

## Slightly, yes — but selectively

### For `Ns`

Your current range is good, but I would add **one extra high point** if feasible:

* 160 or 192

Not a huge sweep. Just one more point to show whether 128 is already asymptotic.

### For `n_modes`

Do **not** waste time trying to “converge (\lambda_0)” with k.
That is the wrong target.

Instead, test only whether miniband observables stabilize, e.g.

* 20, 40, 80
* maybe 120 once

That is enough.

### For `fd_order`

2 vs 4 is enough for the thesis.
No need to go crazy.

### For sigma

Do not make sigma a whole chapter.
Just fix a protocol and do one sanity appendix plot showing that sigma changes which spectral region you target, as expected.

---

# Should you extend to the BZ and compute minibands for parameter sweeps?

## Yes. Absolutely.

That is the single clearest next step.

But do it in a **targeted** way, not as a giant combinatorial explosion.

I would do this:

### Minimal but strong convergence campaign

For honeycomb only:

At (\theta = 1.1^\circ) and (\theta = 3^\circ), compute minibands for:

* `Ns = 64, 96, 128`
* `n_modes = 20, 40, 80`
* `fd_order = 2, 4`
* `N_K = 40, 80, 160`

Then report convergence of:

* (\mathrm{BW}_1),
* (\Delta_{12}^{\min}),
* flatness ratio.

This is already thesis-grade.

Then for square and hex, do only **spot checks** at one representative setting to show the same solver behavior is not unique to honeycomb.

That saves time and still looks rigorous.

---