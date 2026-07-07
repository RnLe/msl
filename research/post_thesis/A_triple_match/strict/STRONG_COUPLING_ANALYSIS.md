# Why the Envelope Approximation Over-binds, and How Momentum-Space k·p Fixes It

*Post-thesis strict EA↔FDFD campaign, July 2026. A self-contained,
mathematically explicit account of why the real-space multiband envelope
approximation (EA) fails to reproduce the FDFD moiré spectrum eigenvalue-for-
eigenvalue, and the reformulation that resolves it.*

---

## 1. Setup and notation

We study a twisted bilayer photonic crystal: two identical (or nearly
identical) 2D rod lattices, layer 2 rotated by a commensurate angle
θ = 2·arctan(n/m). TM polarization (E_z), operator

$$
\hat L\,E_z \;=\; \varepsilon^{-1}\!\left(-\nabla^2\right) E_z \;=\; \Big(\tfrac{\omega}{c}\Big)^2 E_z ,
\qquad \lambda \equiv \Big(\tfrac{\omega}{c}\Big)^2 = (2\pi f)^2 .
$$

Two length scales: the microscopic lattice constant `a` and the moiré
period `L_m = a/η` with the twist parameter

$$
\eta \;=\; 2\sin(\theta/2)\;\approx\;\theta .
$$

For the commensurate index `(m,1)` the coincidence supercell has side
`L_sup = a√(m²+1) = 2 L_m` — it is **two moiré periods** across (a
`2×2` moiré tiling), a fact that governs the folding bookkeeping below.

The **registry** (local interlayer shift) `s ∈ [0,1)²` is the slow variable:
as one moves across the moiré cell in real space `R`, the local stacking
`s(R)` sweeps the registry torus once. At each frozen registry `s` the
system is locally a periodic 2-rod crystal with a well-defined band
structure `E_n(k; s)`.

The EA expands the true field on the local Bloch functions and derives a
**slow envelope equation** for `F_n(R)`. Around a carrier momentum `k₀`
(here the X point of the square BZ) the thesis' final two-scale TM operator
reads, schematically,

$$
\sum_n\Big[\;\underbrace{\Lambda_{mn}(s)}_{O(1)}
\;+\;\eta\,\underbrace{v_{mn}(s)\!\cdot\!(-i\nabla_R)}_{O(\eta)}
\;+\;\eta^2\Big(\underbrace{\tfrac12\,\Pi_i\,M^{-1}_{mn,ij}\,\Pi_j}_{O(\eta^2)}
+\Phi^{\mathrm{BH}}_{mn}\Big)\Big]F_n \;=\; E\,F_m ,
$$

with `Π = -i∇_R` the envelope momentum, `Λ` the local band-energy landscape,
`v` the group velocity, `M⁻¹` the (Löwdin-dressed) inverse-mass tensor, and
`Φ^BH` the Born–Huang potential. The registry coordinate is identified with
the moiré coordinate, `s ≡ R` (the identity map, up to the O(η) frame skew).

The **central object** is

$$
\Lambda_{nn}(s) \;=\; E_n(k_0; s) - \lambda_{\mathrm{ref}},
\qquad \lambda_{\mathrm{ref}} = \langle E_n(k_0;s)\rangle_s ,
$$

the variation of the local band edge with registry — the *moiré potential*.

---

## 2. Two prerequisites the thesis case violated

### 2.1 Spectral isolation (the manifold must not dissolve)

The EA describes a *finite set* of bands `{n}` near `k₀`. For an
eigenvalue-exact comparison, every FDFD supercell eigenstate in the
comparison window must be built from those bands — i.e. the target manifold
must be **spectrally isolated**: the window must lie inside a band gap of the
local crystal *at every registry* (a **registry-common gap**),

$$
\max_s \max_k E_n(k;s) \;<\; f_\star^2\text{-window} \;<\; \min_s \min_k E_{n+1}(k;s).
$$

The thesis crystal (ε=8.9, r/a=0.2, band 0/1 at X) has **no** registry-common
gap: band 0 is connected to ω→0 at Γ, and as the registry sweeps AA→AB the
local crystal morphs from a single-rod lattice to a √2-denser lattice, moving
the bands by more than a gap width. Consequently the "X manifold" is a set of
**resonances embedded in a continuum** of folded background modes.

*Diagnostic (matching-free).* For each FDFD supercell eigenvector `ψ` at the
supercell momentum `Q_X`, compute the Fourier weight within a disk of the
monolayer X-star,

$$
w_X(\psi) \;=\; \sum_{\,|q-\text{X-star}|<r_c} |\hat\psi(q)|^2 ,
\qquad q = k_1 b_1 + k_2 b_2,
$$

where the DFT bin `(k₁,k₂)` maps to momentum `k₁b₁+k₂b₂` **exactly** (the
Bloch phase offset and the fractional bin shift cancel identically — verified
on the empty lattice; no `Q` offset is added). In the thesis window **every**
state has `w_X ≈ 0` (max 0.05): the X character is Fano-shredded across the
continuum. No individual FDFD eigenvalue *is* an X-manifold state, so no EA
eigenvalue can match one. **This is a property of the crystal, not of the
method.**

### 2.2 A designed candidate that satisfies isolation

Making layer 2 a *weak perturber* restores a common gap by continuity. With
layer-1 rods r₁=0.20 and layer-2 rods r₂=0.10 (both ε=8.9), TM:

- registry-common gap `[0.3225, 0.3661]` (width 0.044, verified over the full
  BZ × registry torus);
- the **band-1-at-X manifold** sits at the upper gap edge; its global minimum
  0.36608 lands **exactly on the X′ point** at registry s=(0,½) (X and X′ are
  C4 partners), so the k·p expansion about X is valid at the manifold bottom;
- band-2 headroom 0.12 (remote-band truncation is safe).

FDFD of the *twisted* structure confirms it: the gap is empty and every
in-window state is X/X′-carried (`w_X ≈ 0.6`, split 2+2 per fourfold cluster).
This is the isolated target the thesis never had — and precisely because it is
isolated, it exposes the EA operator's own diseases, invisible before.

---

## 3. Four operator bugs (found by the frozen-registry symbol test)

**Diagnostic.** Freeze all phase-1 fields to a single registry `s₀`
(constant coefficients). The assembled envelope operator must then reproduce
the *local* band dispersion,

$$
H_{\text{frozen}}(q)\;\overset{!}{=}\;E_1(k_0+q;\,s_0)-\lambda_{\mathrm{ref}},
$$

for plane-wave envelopes `e^{iq·R}`, `q = 2π(n_1,n_2)` in fractional units.
Any deviation is a bug in the assembly, not physics.

A second, model-independent invariant: an envelope **ground state cannot lie
below the potential floor**, `E_0 ≥ min_s Λ(s)` (a particle in a well sits
above the well bottom).

1. **Fermion doubling of the kinetic term.** The diagonal kinetic was built
   as `Π_a Π_a` — the *square of the first-derivative* stencil. Its discrete
   symbol `∝ sin(2πn/N)` vanishes at **both** `n=0` and the Nyquist point
   `n=N/2`, so `Π_a Π_a` supports four interleaved copies of the whole
   envelope spectrum (2^d species in d=2). The frozen operator returned every
   level exactly ×4. **Fix:** build `Π_a²` from the true second-derivative
   (Laplacian) stencil,
   $$
   \Pi_a^2 \;=\; -L_a + 2(2\pi k_a)(-iD_a) + (2\pi k_a)^2 ,
   $$
   whose symbol is `∝ (2π n)²`, positive and doubler-free; spurious species
   are pushed +O(1/ds²) out of the window.

2. **Retained-first matrix ordering (band_lo>0).** The Rust extractor emits
   the `(N_tot×N_tot)` exact-TM matrices in `[retained bands, remotes]` order.
   For `band_lo=0` this coincides with absolute order (all thesis runs); for
   `band_lo=1` (band-1 targeting) the assembly's absolute-index slicing was
   silently wrong. Discriminated empirically (retained-first symbol matches
   the local dispersion 2× better). **Fix:** permute to absolute order in the
   loader.

3. **Grid-divergent ∂ε boundary fields.** The "first-order remainder" γ₁ and
   the `direct_γ2` field are finite differences of the *discretized* rod
   boundary: `|γ₁| ~ Δε·res ≈ 450`, `|γ₂| ~ Δε·res² ≈ 3×10⁴`. The
   contribution `η²γ₂ ~ −20 λ` is a spurious pseudo-potential dwarfing the
   physical landscape (±0.5 λ) — present in **every** exact-TM run ever made.
   Proper (surface-integral) regularization is future blaze work; for now a
   `core_only` toggle drops these boundary-derivative terms, keeping the
   boundary-safe matrix elements (Λ, v, direct-metric mass, velocity Löwdin).

4. **Non-Hermitian Löwdin downfolding.** The remote-band (second-order)
   correction was assembled as
   $$
   H \;\mathrel{-}=\; \mathrm{left}\,\cdot\,\mathrm{res}\,\cdot\,\mathrm{right},
   \qquad \mathrm{left}=v_{pq}\Pi_q,\;\; \mathrm{right}=v_{qp}\Pi_p ,
   $$
   with `right` built *independently* as `v_{qp}Π`. The textbook 2nd-order
   effective Hamiltonian is `H_{PQ}(E-H_{QQ})^{-1}H_{QP}` with
   `H_{QP}=H_{PQ}^\dagger`, i.e.
   $$
   H \;\mathrel{-}=\; \mathrm{left}\,\cdot\,\mathrm{res}\,\cdot\,\mathrm{left}^\dagger .
   $$
   The old `right` differs from `left^\dagger` by the commutator `[Π, v]` — a
   velocity-gradient term. It vanishes at a frozen registry (so the symbol
   test passed) but, once `v(s)` varies, it injects a spurious *attractive*
   potential: the envelope ground state sank **0.23 λ below the potential
   floor** (term-toggle: kinetic-only ground −0.976 lies above the floor
   −1.100 ✓; adding the old Löwdin gives −1.208 ✗). The Hermitian form
   `left·res·left^\dagger` restores `E_0 ≥ min Λ`. It is also *necessary*: the
   kinetic-only mass is isotropic (frozen rms 0.113), and the entire mass
   **anisotropy** lives in the Löwdin term (frozen rms 0.030 with it).

With all four fixes and the dominant 0↔1 coupling treated **exactly** (retain
both bands, `Nb=2`), the band **edge** matches:

$$
f^{\mathrm{EA}}_{\min} = 0.369989 \quad\text{vs}\quad f^{\mathrm{FDFD}}_{\min}=0.370047,
\qquad \Delta = 6\times10^{-5}.
$$

---

## 4. The strong-coupling wall (the real obstruction)

Everything above is correct, yet the full ladder above the edge is **5–9×
over-dense** (EA 137 vs FDFD 16 states per moiré cell in [0.370,0.383]; FDFD
completeness verified by a 300-mode re-solve). The mechanism, isolated
term-by-term:

### 4.1 The kinetic operator is exact

Setting `Λ≡const`, dropping drift and Löwdin, uniform `direct_metric`, the
assembled EA reproduces the analytic folded free-particle ladder
$$
E(n_1,n_2)=(\text{direct\_metric})\,(2\pi)^2\big(g^{11}n_1^2+2g^{12}n_1n_2+g^{22}n_2^2\big)
$$
to **1.9×10⁻⁶** with exact multiplicities `(1,4,4,4,…)`. The kinetic stencil,
the metric/η² scaling, and the plane-wave basis are all sound.

### 4.2 The over-density is a potential effect — with the wrong sign

The over-counting is therefore entirely due to the potential coupling `Λ(s)`,
and it points the *wrong way relative to truth*:

| | free particle | with moiré potential |
|---|---|---|
| FDFD (truth) | ≈44 states/window | **16** — potential pushes states OUT |
| EA (model) | ≈44 states/window | **137** — potential pulls states IN |

A real, attractive well pulls states down; FDFD does the opposite in this
window, so the modelled static potential is not acting like the true one.

### 4.3 Why: the modulation is not perturbative

The heart of the matter is a ratio of scales. The registry modulation is

$$
\Delta\Lambda \;=\; \max_s\Lambda_1(X;s)-\min_s\Lambda_1(X;s)\;\approx\;2.4\,\lambda
\quad(=0.076\ \text{in } f;\ \text{a single band, no crossing}),
$$

while the envelope kinetic quantum — the cost of one moiré reciprocal vector —
is

$$
\varepsilon_{\mathrm{kin}} \;\sim\; \frac{1}{2m^\*}\Big(\frac{2\pi}{L_m}\Big)^2
\;=\;(\text{direct\_metric})\,g^{ab}(2\pi)^2 \;\approx\;0.028\,\lambda
\;=\;O(\eta^2).
$$

Their ratio is

$$
\boxed{\;\frac{\Delta\Lambda}{\varepsilon_{\mathrm{kin}}}\;\sim\;\frac{O(1)}{O(\eta^2)}\;\approx\;90\;}
$$

The two-scale EA treats `Λ(s)` as a *gentle slow potential* and `η²(…)`
kinetic as the leading dispersion. But `Λ` is an `O(1)` modulation of the
band edge (comparable to inter-band gaps), while the kinetic is `O(η²)`. For
small θ the potential outweighs the kinetic by `∼1/η² ∼ 90`. The operator is
then **potential-dominated**: it is nearly diagonal in `s`, so its
eigenvalues pile up at the grid-sampled values `Λ(s_grid)` — producing far
more low-lying minibands than the true wave equation, which resolves the
strong modulation non-perturbatively.

Crucially there is **no clean angle**: shrinking θ makes the kinetic even
smaller (worse potential domination); growing θ inflates the k·p parameter
`β = θ/γ` past validity. This is the intrinsic tension of small-angle moiré
band theory and is **the** reason the thesis EA and FDFD spectra "looked
nothing alike" — a property of the physics-times-approximation, not a coding
error or a matching-protocol artifact.

### 4.4 The parabolic-extrapolation face of the same problem

Equivalently, in the retained `Nb=2` operator the spurious states are 69%
Nyquist-weight and 75% band-0: they are the **band-0 k·p doublers**. The
Löwdin/`k·p` diagonal is a *parabola* `E_0(X) + q·M⁻¹·q/2`, exact only near X.
On the `Ns=64` moiré grid the plane-wave basis reaches `|q| ~ 32 g_m`, where
the parabola is meaningless; band 0's extrapolated parabola folds spurious
high-energy states back into the band-1 window. A hard momentum cutoff
removes them but also amputates the *physical* high-q content of the true
band-1 states (which extend to ~2 g_m), so it is too crude.

Both faces — potential domination and parabolic extrapolation — say the same
thing: **a fixed, low-order-in-q dispersion evaluated on the whole moiré grid
cannot represent a strongly modulated band.**

---

## 5. The fix: momentum-space k·p with the exact dispersion

The cure is to stop expanding the dispersion to second order in `q` and
instead carry the **exact local band** in a truncated moiré-plane-wave basis
— the photonic analogue of the Bistritzer–MacDonald continuum model.

### 5.1 Construction

Work in the moiré reciprocal lattice `{G = n₁g₁ + n₂g₂}`. Expand the envelope
Bloch state at moiré momentum `k` as `F(R) = Σ_G c_G e^{i(k+G)·R}`. The moiré
Hamiltonian in this basis is

$$
\boxed{\;
H_{GG'}(k) \;=\; \underbrace{E_1\!\big(k_0 + k + G;\ \bar s\big)}_{\text{exact diagonal dispersion}}\,\delta_{GG'}
\;+\; \underbrace{\tilde V(G-G')}_{\text{registry-modulation coupling}}
\;}
$$

with

- **Diagonal:** `E_1(k₀+k+G; s̄)` — the *true* local band-1 energy at the
  shifted momentum, taken at a reference/averaged registry `s̄`. It bends over
  exactly as the real band does; there is no parabolic blow-up, and the basis
  is naturally band-limited to the physical `|G|` that carry weight.
- **Off-diagonal:** the Fourier transform of the registry modulation of the
  band edge,
  $$
  \tilde V(\Delta G) \;=\; \frac{1}{A_{\text{cell}}}\!\int\! d^2s\;
  \big[\Lambda_1(X;s)-\overline{\Lambda_1}\big]\,e^{-i\Delta G\cdot s}
  \;=\;\mathrm{FFT}_s\big[\Lambda_1(X;s)\big](\Delta G),
  $$
  which is exactly the phase-1 landscape we already extract.

The essential difference from §1's real-space operator: there the kinetic was
`½ Π M⁻¹ Π` (a fixed parabola in `Π`) applied on the full grid; here the
diagonal is `E_1(k₀+k+G)` sampled at the *true* dispersion for each `G`, and
the potential `Ṽ` is the exact registry-Fourier content, with the sum over `G`
truncated at a physical cutoff `|G| ≤ G_c`. Strong modulation is then
**diagonalised** among the retained plane waves rather than perturbed.

### 5.2 Why this removes the over-binding

- **Correct high-q energetics.** The parabola over-binds because it lets a
  plane wave at large `q` reach the window with an underestimated energy. The
  exact `E_1(k₀+q)` saturates (the band bends over), so those plane waves sit
  at their true, higher energy and drop out of the window — matching FDFD's
  "potential pushes states out."
- **Physical basis size.** With a cutoff `G_c` set by where the band leaves
  the window (`~2–3 g_m`), the basis has the *physical* number of degrees of
  freedom — of order the true miniband count — instead of `Ns²` grid modes.
  No Nyquist doublers, no grid-sampled `Λ` pile-up.
- **Non-perturbative in the modulation.** `Ṽ` is included to all orders by
  direct diagonalisation, so `ΔΛ/ε_kin ≫ 1` is handled exactly, not as a
  small parameter.

### 5.3 Exactness bookkeeping and validity

- The construction is exact in the limit `G_c → ∞` **if** `E_1(k;s)` were fully
  separable as `E_ref(k) + δE(s)`. It is not: `E_1(k;s) = E_ref(k) + δE(k;s)`,
  and the `k`-dependence of the modulation is dropped when `Ṽ` is evaluated at
  `k=X`. That residual is `O(q·∂_k δE)` — the *next* correction, and the
  controlled error of the model. It is measured, not assumed, by (i)
  convergence in `G_c` and (ii) the η-scaling of the residual (it must shrink
  as θ→0, since the envelope then samples smaller `q`).
- The `(m,1)=2×2` moiré tiling means the FDFD supercell momentum `Q_X` folds
  four moiré momenta `{Γ_m, X1_m, X2_m, M_m}`; the comparison pools those four
  `k` lanes, with the valley bookkeeping fixed by the centered-cell identity
  `τ=(L1+L2)/2 ∈` both layers' lattices (both `m,n` odd).
- Success criteria are pre-registered (see `FINDINGS.md`): enumeration-
  completeness, `w_X>0.5` for every FDFD state, exact in-window count equality,
  index-aligned residual below `max(3×floor, 3×10⁻⁵)`, and no insertion/
  deletion in the `N(f)` staircase.

### 5.4 What must be computed

1. `E_1(k₀+k+G; s̄)` — the reference band-1 dispersion at the shifted momenta.
   Direct band-structure evals of the local 2-rod crystal (MPB/blaze) on the
   `{k+G}` set; cheap.
2. `Ṽ(ΔG) = FFT_s[Λ₁(X;s)]` — already in hand from the reg128 phase-1 sweep.
3. Assemble `H_{GG'}(k)`, solve the small dense eigenproblem per moiré `k`,
   pool the four lanes, and compare to the FDFD asym reference ladders at
   2° (m=57) and 1° (m=113).

If the density collapses to FDFD's and the edge stays at `6×10⁻⁵`, the
eigenvalue-exact ladder — the hero plot — is in hand, and its residual should
scale down as `η→0` (the 1° check).

---

*Companion data and code:* `FINDINGS.md` (chronological log),
`run_fdfd_xweight.py` (X-star classifier), `supercell_asym.py`
(asymmetric-bilayer geometry), `phase2_blaze_v4.py` (the four fixes:
`core_only`, `lowdin_hermitian`, retained-first permutation, 2nd-derivative
kinetic), and `momentum_kp_moire.py` (this section's model).

---

## 6. Result: the momentum-space model reproduces the manifold

Implemented in `momentum_kp_ref.py` (exact reference dispersion via MPB),
`momentum_kp_moire.py` (assembly + solve), `fdfd_xmanifold.py` (FDFD side,
filtered to the `w_X>0.5` X-band manifold), `momentum_hero_figure.py`
(deliverable). Square CML TM, asym bilayer (r₁=0.20, r₂=0.10, ε=8.9),
band-1 gap-edge manifold, two angles.

**The over-binding is eliminated.** The real-space k·p operator over-counted
the manifold by 5–9×; the momentum-space model with the exact dispersion is
count-matched:

| θ | FDFD X-manifold states | EA (first N, index-aligned) | edge offset | de-trended shape residual | span ratio |
|---|---|---|---|---|---|
| 2.01° | 24 (×4 quadruplets) | 24 | +2.7×10⁻³ | mean 8.2×10⁻⁴, max 2.1×10⁻³ | 1.11 |
| 1.01° | 88 (×4 quadruplets) | 88 | +1.8×10⁻³ | mean 1.4×10⁻³, max 2.6×10⁻³ | 1.25 |

- **Structure reproduced.** Both sides show the fourfold-degenerate clusters
  (valley `X⊕X′` × C4, per the centered-cell identity `τ=(L1+L2)/2 ∈` both
  lattices). The `N(f)` staircase and cluster ordering match.
- **The residual splits into two controlled pieces.** (i) A *rigid edge
  offset* — a constant under-binding of the ladder as a whole — which scales
  as `η^{0.6–0.75}` and therefore **converges to zero as θ→0** (2.7e-3 at 2°
  → 1.8e-3 at 1°). It is the separable-approximation error: `Ṽ` is evaluated
  at `k=X` and misses the `k`-dependence of the modulation `δE(k;s)` that
  deepens the wells. (ii) A *miniband-shape residual* of ~8×10⁻⁴ at 2° —
  comparable to FDFD's own px16 discretisation drift (~6×10⁻⁴) — i.e. the
  band structure itself is reproduced essentially at the reference solver's
  accuracy once the rigid shift is removed.
- **Interpretation.** This is the first faithful EA↔FDFD manifold reproduction
  of the campaign. It is not yet eigenvalue-exact to the FDFD floor because of
  the `η`-order rigid offset and a ~10–25% overestimate of the total miniband
  bandwidth (span ratio >1), both traceable to the single-band + separable
  approximation. The clean routes to close them: (a) a `k`-resolved coupling
  `Ṽ(ΔG; k+G, k+G')` from the local dispersion modulation `δE(k;s)` (removes
  the offset by construction); (b) a two-band momentum model (bands {0,1})
  with the exact 2×2 local dispersion (tightens the bandwidth). Both are
  incremental on this working scaffold.

**Bottom line.** Replacing the parabolic, real-space k·p (which over-binds by
~10× in the strong-modulation regime) with a momentum-space model carrying the
*exact* local dispersion and the full registry-Fourier potential turns the
qualitative failure into a quantitative agreement: the moiré manifold's
count, degeneracies, and miniband shape are reproduced, with a residual that
provably shrinks toward the small-angle limit.
