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

---

## 7. The exact continuum model: Galerkin projection onto reference-Bloch states

The momentum-space model of §5 keeps the exact *dispersion* but a *scalar*
coupling (`Ṽ=FFT[Λ]`), dropping the ε⁻¹-weighted Bloch-overlap **form factors**
and the **inter-band** elements — the source of its residual offset and
bandwidth overestimate. The exact object that keeps them is a Galerkin
(Rayleigh–Ritz) projection of the true supercell operator onto a basis of
reference-Bloch states.

### 7.1 Construction

At the supercell momentum `Q_X` (the FDFD momentum), the TM problem is the
generalized eigenproblem `−∇²E = λ ε_bl E`, with `ε_bl` the full moiré
dielectric. Take the trial space spanned by reference Bloch fields at the
folded moiré momenta,

$$
E_{n,p}(\mathbf r) \;=\; e^{i\,\mathbf p\cdot\mathbf r}\,u_n(\mathbf r; \mathbf p; \bar s),
\qquad \mathbf p = \mathbf X + \tfrac12(j_1\mathbf g_1 + j_2\mathbf g_2),
$$

where `u_n(·;p;s̄)` is the periodic Bloch part of band `n` of the *local*
crystal at a reference registry `s̄`, obtained from MPB, and `g_i` are the
moiré reciprocal vectors. Because `L_sup = 2 L_m`, the supercell reciprocal
lattice is exactly `b_sup = g/2`, so the half-`g` set `{p}` is precisely the
supercell plane-wave set at `Q_X`: every `E_{n,p}` lies in the `Q_X` Bloch
sector, and `eigh(H,S)` yields the `Q_X` spectrum directly. The projected
matrices are

$$
H_{\alpha\beta} = \langle \nabla E_\alpha | \nabla E_\beta\rangle
= \sum_{\mathbf G}\hat w_\alpha^*(\mathbf G)\,|\mathbf X+\mathbf G|^2\,\hat w_\beta(\mathbf G),
\qquad
S_{\alpha\beta} = \langle E_\alpha | \varepsilon_{\mathrm{bl}} | E_\beta\rangle,
$$

with `w = e^{-iX·r}E` the (supercell-periodic) Bloch amplitude and `G` the
supercell reciprocal vectors — the kinetic must be evaluated on the periodic
part with the `X`-shift `|X+G|²`, *not* by a naive FFT of the Q_X-Bloch `E`
(which is anti-periodic; getting this wrong scrambles the spectrum). `S` is
non-orthogonal and, on a finite grid with crowded momenta, near-rank-deficient,
so we solve in the well-conditioned subspace of `S` (canonical
orthogonalization, drop eigenvalues `< tol·S_max`).

This is manifestly variational: the eigenvalues are upper bounds to the true
FDFD spectrum, decreasing monotonically as the basis (`bands`, `|G|`-cutoff)
grows, and converging to FDFD in the complete-basis limit — because the trial
space then spans the supercell Hilbert space and `H,S` become the exact
supercell operator. It contains the form factors (the `u`-overlaps inside
`S`,`H`) and inter-band coupling (off-diagonal `n≠m`) *exactly*; the §5
heuristic is its single-band, form-factor→1, `ε_bl→ε(X)` limit.

### 7.2 Validation: it converges to FDFD

On a fast large-angle cell (7,1) (θ=16.3°), against FDFD at `Q_X` (bottom-12
levels, index-aligned):

| retained bands | mean \|Δf\| | max \|Δf\| | edge Δ |
|---|---|---|---|
| 2 | 3.7×10⁻³ | 6.2×10⁻³ | +3.4×10⁻⁴ |
| 4 | 1.8×10⁻³ | 2.8×10⁻³ | +2.3×10⁻⁴ |
| 6 | 1.1×10⁻³ | 1.9×10⁻³ | +1.7×10⁻⁴ |
| 8 | 8.2×10⁻⁴ | 1.3×10⁻³ | +1.0×10⁻⁴ |

Monotone (variational) convergence toward FDFD, as the theorem demands. Two
independent sanity gates pass: single-momentum + reference-ε recovers the local
band energies to 10⁻⁴, and the free-particle limit is exact. **The exact
continuum method is correct.** The slow rate at (7,1) is the strong-coupling
regime (16° is far outside the two-scale window); many bands are needed there.

### 7.3 The small-angle regime finding

At the target small angle (57,1) (θ=2.01°), a single-reference basis is
*inefficient*: with bands {0,1} and `|G|≤4` it captures only 4 of the 24
X-manifold modes in the window. The reason is physical and important — the 24
moiré-manifold states have **registry-varying Bloch character**: near the
well-bottom registry the local band-1 function differs from the mean-registry
reference, and a single reference frame cannot span them with few bands. This
is precisely why the thesis EA uses the *registry-adapted* local Bloch basis
`u_n(r;R)` (with the Berry connection tracking its rotation), rather than a
fixed reference frame.

So the exact continuum model splits by regime:
- **Moderate/large θ (out of two-scale):** single-reference Galerkin converges
  to FDFD, but slowly (many bands) — strong-coupling.
- **Small θ (in two-scale):** the manifold is spanned efficiently only by a
  **registry-adapted** basis (local Bloch at each/several registries) — the
  fixed-reference frame is the wrong truncation. This is the same registry-
  adaptation the real-space EA performs; the momentum-space exact model
  inherits the requirement.

The practical small-angle model therefore remains the §5 momentum-space
continuum with its exact dispersion + registry-Fourier potential (count- and
structure-exact, ~10⁻³ residual), and the eigenvalue-exact ladder requires
either (a) registry-adapted reference frames (multi-reference Galerkin), or
(b) the k-resolved form-factor coupling of §6 — both well-defined extensions
of the validated scaffold.

---

## 8. The regime map: where is eigenvalue-exactness possible?

Collecting the results, the achievability of an eigenvalue-exact few-band
continuum model is governed by three conditions — two **hard walls** (failure
is categorical) and one **soft cost axis** (failure is gradual, in basis size).

### 8.1 Hard wall 1 — spectral isolation (dissolution)

If there is **no registry-common gap**, the target band manifold is a set of
resonances embedded in the folded continuum: its Bloch character is shredded
(`w_X≈0` for every supercell state, §2). No finite retained-band set contains
the in-window states, so no truncation — however large — can reproduce them.
This is categorical and crystal-determined. The thesis square case
(ε=8.9, r=0.2, bands 0/1 at X) is on the wrong side of this wall; the designed
asymmetric bilayer (r₁=0.20, r₂=0.10) is on the right side.

### 8.2 Hard wall 2 — two-scale validity (β = θ/γ)

For `β = θ/γ ≳ 1` (γ = gap-to-midgap), the local crystal changes by more than
a stop-band width over the Bragg-formation length: the "slow modulation"
picture fails, and the reference-Bloch trial space is no longer efficient — the
Galerkin still *converges* to FDFD (it is variational), but only with a basis
approaching the full plane-wave set, i.e. it degenerates into re-solving FDFD.
The (7,1) test (θ=16°, β≫1) shows exactly this: monotone convergence but slow
(mean\|Δf\| still 8×10⁻⁴ at 8 bands). Below `β≲0.1` (the 1°–2° campaign) the
two-scale picture holds and few bands *can* suffice — provided the third
condition is met.

### 8.3 Soft cost — modulation strength (V/E_kin) and registry-adaptation

Inside the valid region (isolated, β small), the *number of bands / reference
frames* needed for eigenvalue-exactness scales with the moiré coupling
strength

$$
\frac{V}{E_{\mathrm{kin}}} \;=\; \frac{\Delta\Lambda}{\tfrac12|M^{-1}_{\mathrm{eff}}|\,\eta^2}.
$$

- **Weak modulation (V/E_kin ~ 1–10):** the manifold states stay close to the
  carrier Bloch character; a single-reference, few-band model is
  eigenvalue-exact. This is the honeycomb-K Dirac regime (Λ₀₁≡0 by C6v — the
  diagonal modulation vanishes, coupling is weak/geometric), and the
  weak-perturber shallow-moiré candidates.
- **Strong modulation (V/E_kin ~ 100, the asym candidate at 2°):** the manifold
  states have strongly **registry-varying** Bloch character; a fixed reference
  frame under-populates them (4/24, §7.3). Eigenvalue-exactness then requires a
  **registry-adapted basis** — local Bloch frames at multiple registries, the
  momentum-space analogue of the thesis EA's `u_n(r;R)` + Berry-connection
  construction. The single-reference model is not wrong, just an inefficient
  truncation; the registry-adapted model is the efficient exact one.
  *Confirmed by a reference-choice sweep:* referencing the mean registry gives
  4/24 in-window modes, the well-bottom registry gives 2/24 — the deficiency is
  not curable by a better single frame, only by adapting the frame across
  registries.

### 8.4 Verdict

**Eigenvalue-exactness is achievable** for the isolated candidate — the exact
Galerkin continuum model is variational and provably converges to FDFD (§7.2).
What the campaign establishes is *the cost*: it is cheap (few bands, single
reference) only in the weak-modulation corner, and requires registry-adapted
frames in the strong-modulation corner where our deliberately-deep candidate
lives. The practical, already-delivered result — the §5 momentum-space
continuum model reproducing the manifold's count, degeneracies and shape to
~10⁻³ with a provably-vanishing offset — sits one controlled approximation
(fixed-frame form factor) away from the exact ladder.

The two clean routes to close the last ~10⁻³, both defined and both incremental
on the validated scaffold, are: **(a)** a multi-reference (registry-adapted)
Galerkin, spanning the registry-varying manifold efficiently; **(b)** the
k-resolved form-factor coupling of §6 grafted onto the §5 model. The regime map
says which is needed where: (a) for strong modulation, (b) suffices for weak.

---

## 9. Registry-adapted Galerkin: convergence, and the position-locking plateau

Implemented the registry-adapted basis via the memory-robust reciprocal-space
engine (`galerkin_recip.py`: sparse plane-wave coefficients + per-basis FFT
convolution for the ε-coupling; <1 GB at px16, aliasing-free, validated against
the real-space engine on (7,1) — bottom 2.6× closer to FDFD, no undersampling).
Basis: local Bloch frames from a grid of reference registries `{s_k}` × moiré
plane waves, all bands {0,1}. (57,1) 2°, in-window [0.365,0.385] vs the FDFD
X-manifold (24 states, bottom 0.37005):

| basis | in-window count | bottom | Δ vs FDFD |
|---|---|---|---|
| single reference | 1 | 0.36824* | — (non-manifold) |
| 4 frames (½-grid), G_c=4 | 7 | 0.37941 | +9.4×10⁻³ |
| 4 frames (½-grid), G_c=6 | 10 | 0.37749 | +7.4×10⁻³ |
| 9 frames (⅓-grid), G_c=4 | 10 | 0.37680 | +6.8×10⁻³ |
| 16 frames (¼-grid), G_c=4 | 10 | 0.37654 | +6.5×10⁻³ |

Two facts stand out. **(i) Registry-adaptation works** — going from 1 to 9
reference frames lifts the captured manifold from 1 to 10 states: adding frames
*does* enrich the trial space, exactly as §8.3 predicted. **(ii) But it
plateaus** — 9→16 frames barely moves (10 states, bottom 0.3765; the S-rank
saturates, 2684→2875 of 6498→11552, so the extra frames are largely redundant),
and pushing the plane-wave cutoff G_c 4→6 improves the bottom by only ~2×10⁻³.
The bottom stalls at **+6–7×10⁻³ above the true FDFD ground state**, regardless
of frame count or cutoff.

**Why (the fundamental reason).** The true moiré ground state is
`Ψ(r) ≈ Σ_n F_n(R) u_n(r; s(R))` — the local Bloch character is *locked to the
moiré position* through the registry map `s(R)`. The momentum-space basis
`e^{i p·r} u_n(r; s_k)` carries a *fixed* registry `s_k` delocalized by plane
waves; to synthesize the position-locked character it must build, for each
frame, an envelope `F_k(R)` sharply peaked on the sub-region where `s(R)≈s_k`,
which requires very high plane-wave content. A coarse frame grid × a modest
cutoff cannot resolve this correlation — hence the plateau. This is precisely
why the thesis EA is formulated in **real space** with the *continuously*
registry-adapted local frame `u_n(r; R)` and the Berry connection tracking its
rotation: the position-registry locking is built in from the start, not
synthesized from fixed frames.

**Conclusion for eigenvalue-exactness.** For the deep strong-coupling regime
(V/E_kin≈86, the deliberately-hard candidate), the eigenvalue-exact ladder is
*reachable in principle* (the Galerkin is variational and complete-basis
convergent — §7.2) but *not efficiently* from a momentum-space fixed-frame
basis: it plateaus ~10⁻² short. The efficient exact vehicle is the **real-space
continuously-registry-adapted envelope** — the thesis EA's structure, but
carrying the *exact* local dispersion (full Bloch functions, all orders in q)
and the *exact* ε-weighted coupling, rather than the O(η²) k·p truncation that
caused the original over-binding (§4). That is the single well-defined
construction that unifies the two threads of this campaign: it has the
registry-adaptation the momentum-space model lacks, and the exact dispersion
the thesis operator lacked. Building and validating it is the natural next
program; the reciprocal-space and FDFD-X-manifold machinery here is the
scaffold for it.

*The practical continuum model (§5) remains the delivered result — count- and
structure-exact, ~10⁻³ residual with a provably-vanishing offset — and the
above maps exactly what closing the last 10⁻³ requires and why.*

> **Correction (see §10).** The "~10⁻³ residual with a provably-vanishing
> (∝η^0.7) offset" quoted above is **superseded**. A momentum-model coupling bug
> (a transposed Ṽ axis pairing) plus FDFD sub-pixel under-resolution together
> accounted for ~96 % of that residual. Corrected, the model's ground **energy**
> is exact to ~2×10⁻⁵; the residual "offset" is θ-independent (FDFD-resolution,
> not separable-approximation), and the genuine non-exactness is **structural**
> (a symmetry-protected-degeneracy break), not a rigid offset. Read §10 for the
> re-analysis.

---

## 10. Weak-coupling test, a coupling-bug correction, and the structural verdict

**Goal.** §8.3 predicted that *weakening the moiré coupling* (V/E_kin ↓) should
make a few-parameter continuum model eigenvalue-exact "cheaply." §10 tests that
prediction with two controlled coupling sweeps at fixed geometry (m=57, θ=2°),
and — through an adversarial audit of the result — corrects a coupling bug that
had inflated the §5/§6 residuals threefold. The corrected verdict is sharper
than either the prediction or its naïve refutation.

### 10.1 Two coupling knobs, and a resolution trap

At fixed angle the moiré depth ΔΛ is dialed by the weak (layer-2) rods. Two
independent knobs:

- **r₂ (rod size).** `scan_common_gap2.py` confirms the clean single-variable
  structure: shrinking r₂ from 0.10→0.03 drops ΔΛ 2.44→0.21 λ (V/E_kin 86→8)
  while the registry-common gap *widens* (0.044→0.113) and β=θ/γ *falls*
  (0.28→0.12) — isolation improves as coupling weakens. All four finalists
  (r₂∈{0.070,0.054,0.040,0.031}) are isolated, manifold bottom on the X-star,
  band-2 headroom ~0.11 (`refine_candidate.py`).
- **ε₂ (dielectric contrast) at fixed r₂=0.10.** ΔΛ 2.44→1.13 λ as ε₂ 8.9→2.0.

**The r₂ knob is unsafe.** Weak coupling ⟺ *tiny rods*: at px16 the layer-2 rod
radius is r₂·px px — **1.6 px at r₂=0.10 but 0.64 px at r₂=0.040** (sub-pixel).
The FDFD ground truth is then discretization-limited: at r₂=0.040 a Richardson
step moves the FDFD ground 0.42995 (px16) → 0.43129 (px32), *straddling* the
model. A naïve r₂ sweep therefore shows a **spurious** ground-residual dip
(≈2×10⁻⁵ at r₂=0.070) that is a coincidental crossing of an under-resolved FDFD
with the model, not exactness. The ε₂ knob (rods fixed at 1.6 px) removes the
confound and is the clean probe. (Lesson: the layer-2 rods must be resolved by
*both* solvers — MPB res-64 and FDFD px — before any residual is meaningful.)

### 10.2 The coupling bug (found by adversarial audit)

A three-way adversarial verification of the "clean ε₂" result (independent
audits of model construction, comparison methodology, and salvage/sub-pixel)
surfaced a genuine **transpose bug** in
`momentum_kp_moire.py`. The registry landscape is stored as `Λ₁[sy, sx]`, so
`Ṽ = FFT₂(Λ₁)` has its **first** index the s_y-harmonic (weak, |Ṽ|≈0.012) and
its **second** the s_x-harmonic (strong, |Ṽ|≈0.58). A first-principles FFT of
the real-space moiré potential V(**r**)=Λ(s(**r**)) shows the strong s_x
modulation drives the **g₁** moiré vector — so the coupling for a g₁-difference
must use the s_x (axis-1) harmonic. The code paired them the other way
(`Ṽ[(n₁−m₁), (n₂−m₂)]`), swapping the strong and weak harmonics between the two
reciprocal axes. Fix: `Ṽ[(n₂−m₂), (n₁−m₁)]` (equivalently transpose Λ₁ before
the FFT). **This bug was present in the §5/§6 delivered result too** — reproducing
the anchor did not catch it because the bug is present at every ε₂.

**Impact — the reported residual was almost entirely artifact.** Decomposing the
§6 headline (2° ground residual +2.74×10⁻³) against a Richardson-extrapolated
FDFD ground (res16/32/48 → 0.370907; the res16 value used before is itself
0.86×10⁻³ too low):

| contribution | value |
|---|---|
| transpose bug (buggy − fixed model) | +1.86×10⁻³ |
| FDFD sub-pixel (res16 − px→∞) | −0.86×10⁻³ |
| **TRUE residual (fixed model − extrapolated FDFD)** | **+1.8×10⁻⁵** |

The corrected model's ground **energy is exact to ~2×10⁻⁵** — i.e. below the
FDFD px16 floor. The "offset ∝η^0.7 → 0 as θ→0" claim of §6 is **retracted**:
corrected, the offset is θ-independent (η^−0.01: +8.8×10⁻⁴ at both 2° and 1°),
because it is dominated by the (angle-independent, rod-resolution-set) FDFD
sub-pixel floor, not a separable-approximation term.

### 10.3 The structural verdict (correction-invariant)

Fixing the bug does **not** make the model eigenvalue-exact — it exposes *why*
it cannot be. The single-band, **X-only** (no X′=(0,π) carrier) model cannot
represent the X⊕X′-mixed manifold, and two correction-invariant signatures make
this quantitative (immune to the DC/reference, transpose, and sub-pixel
corrections above):

1. **It breaks the symmetry-protected 4-fold ground degeneracy.** FDFD holds the
   ground quadruplet degenerate to 1.7×10⁻¹⁰ (a C4×valley symmetry); the
   corrected model splits it by **1.17×10⁻⁴ at 2°** — a ~10⁶× symmetry violation.
   Tellingly the split **→0 as θ→0** (8.9×10⁻⁷ at 1°): the valley mixing the
   model omits is an O(η) effect, so the structural error is θ-suppressed but
   nonzero at any finite angle. *(The buggy model accidentally preserved the
   degeneracy, masking this.)*
2. **It over-splits the miniband fine-structure.** FDFD's lowest 8 states form a
   near-degenerate cluster (X⊕X′×C4, spread ~1.9×10⁻⁴); the model spreads them
   ~13× wider. The low-8 span ratio is 7.7×–22.9× across the ε₂ sweep; the
   inter-quadruplet over-split is 2×–14×.

**Does weakening the coupling help?** Mixed, and *not* the clean "→exact" §8.3
predicted. As ε₂ falls (V/E_kin 86→40) the miniband over-split *shrinks*
(inter-quad 11×→2×) but the degeneracy-break *grows* (1.17→3.35×10⁻⁴); the
ground-energy residual grows (dominated by the removable reference-registry DC
term E_ref(X;s̄)−⟨Λ₁⟩, which drifts as s̄ leaves the registry mean). No single
metric approaches exactness. The one clean limit is **θ→0**, where the valley
error is suppressed — consistent with the small-angle validity zone (§8.2), not
with weak modulation.

### 10.4 Verdict

- The compact momentum-space model is **energy-accurate** — its ground-state
  frequency matches resolution-converged FDFD to ~2×10⁻⁵, far better than the
  §6 figure once the coupling bug and FDFD resolution are removed.
- It is **not eigenvalue-exact**: as a single-valley reduction it breaks the
  X⊕X′ symmetry (degeneracy split ~10⁻⁴ at 2°) and over-splits the minibands
  (~10×). These are **structural**, not offsets, and are not removed by
  weakening the coupling; only θ→0 suppresses them.
- **§8.3's "weak-coupling → cheap few-parameter exactness" is not borne out**
  for this square-X photonic manifold, for either few-parameter vehicle: the
  scalar momentum model (this section) *and* the fixed-frame Galerkin (§9,
  which under-populates for the *same* registry/valley reason, coupling-
  independently). Eigenvalue-exactness requires the X⊕X′ valley-coupled
  (form-factor) continuum operator — the §8.4 route (b) grafted with the §9
  registry adaptation — or the full solve. The exactness the campaign *can*
  claim is the §7 Galerkin's variational convergence with the **full** basis,
  not a few-parameter model.

*Methodological residue worth keeping: (i) the ground-residual-average is a poor
exactness proxy here — it conflated a code bug, FDFD under-resolution, and a
reference-registry DC term; the degeneracy-break and over-split are the robust
diagnostics. (ii) FDFD ground truth must be Richardson-extrapolated even at
1.6 px before any model error is quoted. (iii) Adversarial cross-checking was
load-bearing: it found the transpose bug that reproducing the validated anchor
did not.* Deliverables: `fig_weak_verdict.{png,pdf}` (decomposition + structural
limit + fine-structure), `eps2_crossover.py`/`scan_common_gap2.py`/
`richardson_analysis.py` machinery, all r₂/ε₂ ladders in the `*_e*`/`*_r2_*` npz.

---

## 11. The two-valley (X⊕X′) completion — the missing ingredient, and a correction to §9

§10 pinned the non-exactness on the model being **single-valley** (X-only). §11
tests the fix — a second carrier X′=(0,π) — and, in doing so, **corrects §9's
central conclusion**. Every number below is reproduced from the committed npz and
was independently adversarially audited (verdict: the
improvement + §9-correction *hold*, exactness *not* achieved — see §11.5).

### 11.1 The premise: the manifold is X⊕X′ valley-mixed (measured)

Direct FDFD measurement (`run_asym_carrier.py` → `valley_composition_2deg.npz`):
the 2° ground quadruplet (0.370047, 4-fold to **1.7×10⁻¹⁰**) decomposes as
**2-at-X + 2-at-X′** — states 1,3 have w_X′≈0.61 (Fourier peak (0,π)), states 2,4
have w_X≈0.61 (peak (π,0)); w_M=0.000, w_Γ≈0.004. The 4-fold is C4×valley
(C4 maps X↔X′). **A single-X-carrier model must miss half of it** — hence §10.3's
degeneracy break, and (§11.4) the §9 plateau.

### 11.2 The single-valley limitation, and why X′ is unreachable by cutoff

All builders (`momentum_kp_*`, `galerkin_*`) carry only X=(π,0). X′ folds to the
same supercell momentum Q_X — X′−X=(−π,π) is a *supercell* reciprocal vector for
odd m ((−28,29)·b_sup at m=57) — but it is a *half-integer* in moiré units, so it
sits ~14 moiré cells (≈40 half-g steps) from X: **unreachable at any feasible
cutoff at 2°**. (At (7,1)/16° it is only ~2 cells away, reached at gcut≥4 — see
§11.3.) The exact **Galerkin** absorbs X′ as a near drop-in: the momenta list
gains {X′+½(j₁g₁+j₂g₂)}; the kinetic |X+G|², `basis_coeffs`, and ε-coupling are
all valley-agnostic (X′−X is an integer supercell-G shift). Flag: `--two-valley`.

### 11.3 Result: adding X′ lifts the §9 plateau and recovers the full count

Registry-adapted (nref=9), gcut=4, m=57 px16, vs the FDFD X-manifold (24 states,
bottom 0.370047):

| basis | in-window | bottom | Δ vs FDFD | S-rank |
|---|---|---|---|---|
| single-valley, 9 frames (§9) | 10 | 0.37680 | +6.8×10⁻³ | 2684/6498 |
| single-valley, 16 frames (§9) | 10 | 0.37654 | +6.5×10⁻³ | 2875/11552 |
| **two-valley, 9 frames** (band_lo=0) | **24** | **0.37154** | **+1.5×10⁻³** | **4921/12996** |
| two-valley, 9 frames (band_lo=1, clean) | 24 | 0.37248 | +2.4×10⁻³ | 6557/12996 |

Adding X′ takes the captured manifold **10→24 (= the FDFD count)** and lifts the
ground **4.5× (band_lo=0) / 2.8× (band_lo=1, band-0-free control)**. The (7,1)/16°
mechanism check (real-space, `galerkin_moire`): as gcut 3→4 lets the basis reach
the X′-star, the ground 4-cluster split converges **2.09×10⁻⁴ → 1.48×10⁻⁴** toward
FDFD's physical 1.25×10⁻⁴ (saturating at gcut 4→5). Figure: `fig_two_valley`.

### 11.4 It is the valley, not basis size (the isolation)

The single-valley basis **saturates**: 9→16 frames adds +5054 functions for only
+191 rank (+7%) and −2.6×10⁻⁴ of bottom. Adding the X′ block instead (+6498
functions) adds **+2237 rank (+83%, ~9× more rank-efficient) and −5.3×10⁻³ of
bottom (~16× more per function)**. So *more basis of the same (X, registry) kind
does not help*; the X′ subspace opens genuinely independent directions the X-basis
cannot reach. The physics closes it: the FDFD ground has weight only on X and X′
(w_M=0), so X′ is the *only* subspace with weight to add. (Caveat: no explicit
M-valley null-control was run; the case rests on the saturation control + w_M=0.)

### 11.5 Verdict — and the correction to §9

**Demonstrated:** a specific, **valley-attributed** 2.8–4.5× lift of the §9
plateau, recovery of the full 24-state count, and a variational method (§7) that
provably converges in the complete-basis limit. **This corrects §9.** §9 attributed
the +6–7×10⁻³ plateau to "position-registry locking" that stalls *"regardless of
frame count or cutoff"* and *"fundamentally"* needs the real-space continuously-
adapted vehicle. That is over-generalized: the plateau stalls against more
*X-valley* frames/cutoff, but is largely lifted by a **different momentum-space
carrier (X′) in the same reciprocal vehicle**. The plateau was substantially a
**missing-X′-valley truncation, not proof that only real-space adaptation works.**

**Not demonstrated — eigenvalue-exactness.** Even with X′ the two-valley bottom is
+1.5×10⁻³ above the floor and is a **singlet**, where FDFD's ground is a
1.7×10⁻¹⁰ four-fold; the recovered 24-state manifold is singlets + a few
near-degenerate pairs (~10⁻⁵), never FDFD's clean 4-folds. **Adding an *uncoupled*
X′ block recovers the COUNT but not the symmetry-protected DEGENERACY** — exactly
§10.3's structural point. Restoring the 4-folds and closing the residual requires
the **X↔X′ valley-COUPLED (form-factor) operator** (or the full basis), not a
concatenated second block. The two-valley Galerkin here is the **diagnostic that
isolates the valley**, not the exact vehicle.

**The path to eigenvalue-exactness is therefore now concrete and staged:**
(1) two-valley completion — *done*, lifts the plateau and fixes the count;
(2) X↔X′ valley coupling with the ε-weighted form factors (§8.4 route b + §10.3) —
the remaining step that restores the degeneracy and drives the bottom to the floor;
optionally carried by the real-space continuously-registry-adapted envelope (§9)
for efficiency at small angle. The earlier verdict "no efficient few-parameter
exact model exists" (§8.4) is softened: the obstruction was a *diagnosed, fixable
omission* (the X′ carrier), not a fundamental wall — consistent with there being
no dissolution/two-scale obstruction at 2° (§8.1–8.2).

*Metric hygiene (audit): the FDFD reference is shift-inverted (not a guaranteed
global floor) and near-rank-deficient S can emit spurious sub-floor states at
nref=1; the nref=9 comparisons are trustworthy because they return exactly 24
states with a clean gap and zero sub-floor states. (7,1) validates the method's
monotone convergence, not 2° exactness (16° is a different, strong-coupling
regime). Deliverables: `galerkin_recip.py --two-valley`, `grecip_2deg_2v_mr3*.npz`,
`valley_composition_2deg.npz`, `fig_two_valley.{png,pdf}`.*

---

## 12. Convergence of the two-valley Galerkin — exact at (7,1), conditioning-limited at 2°

§11 established that the X′ valley lifts the §9 plateau to +1.5×10⁻³. §12 asks the
next question directly: **does the two-valley Galerkin *converge* to the FDFD
floor?** The variational theorem (§7) says it must, in the complete-basis limit.
Testing it required breaking the memory wall and cleaning the metric.

### 12.1 Enabling machinery (memory-lean eigensolve + band-1 filter)

The dense `eigh(S)` (zheevd, O(Nb²) workspace) + dense `H` were the OOM driver
(peaked 15 GB at Nb=13k; nref/gcut pushes → 35–40 GB). Replaced (`galerkin_recip.py`,
audit-designed) with: canonical orthogonalization via `eigh(S, subset_by_value,
driver="evr")` (O(Nb) workspace, returns only the kept subspace) + a **matrix-free**
`Hp = Vpᵀ·(dA·Cᵀ·diag(kin)·C)·Vp` from the sparse `C` (dense only in n_kept²). Gate:
reproduces the committed gcut=4 result to **1.4×10⁻¹¹**. Added a **band-1-weight
classifier** (per eigenstate, from its basis coefficients) — the Galerkin analogue
of the FDFD w_X filter — to separate the band-1 manifold from band-0 active-band
pollution and to flag spurious sub-floor states.

### 12.2 (7,1): the method reaches eigenvalue-exactness

Valley-complete (gcut=4 spans X′) band ladder at (7,1)/16.3° (`galerkin_moire`),
edge offset vs FDFD:

| N_b | 2 | 4 | 8 | 12 | **16** |
|---|---|---|---|---|---|
| edge \|Δf\| | 3.4×10⁻⁴ | 2.3×10⁻⁴ | 8.5×10⁻⁵ | 4.4×10⁻⁵ | **3.4×10⁻⁵** |

Monotone, still descending — **the two-valley Galerkin converges to the FDFD floor**
(+3.4×10⁻⁵ ≪ the px-scale floor) at a tractable cell. This is the direct proof that
the residual is basis incompleteness, not a wall: with the valley present and the
basis complete, the method is eigenvalue-exact.

### 12.3 2°: gcut converges *per rank*, but the fixed-frame basis is conditioning-limited

Pushing the plane-wave cutoff at 2° (nref=3, two-valley), sweeping the
canonical-orthogonalization tolerance `s_tol` (which sets the well-conditioned rank):

| gcut | s_tol | rank | band-1 bottom Δ | note |
|---|---|---|---|---|
| 4 | 1e-4 | 1804 | +8.6×10⁻³ | clean |
| 4 | 1e-5 | 3165 | +4.5×10⁻³ | clean |
| 4 | 1e-6 | 4921 | **+1.5×10⁻³** | clean (best) |
| 5 | 1e-4 | 2675 | +2.0×10⁻³ | clean |
| 5 | 1e-5 | 4653 | −2.1×10⁻³ | **spurious** (sub-floor) |

Two facts. **(i) gcut genuinely converges:** at *matched* rank, gcut=5 beats gcut=4
(rank 2675 → +2.0×10⁻³ vs gcut=4's ~+5.8×10⁻³ interpolated there) — more envelope
resolution lowers the bound, as §7 requires. **(ii) But the fixed-frame reciprocal
basis is conditioning-limited:** the well-conditioned rank caps ~4900 (the extra
gcut-5 plane waves are near-linearly-dependent), and beyond it the near-singular
`S` emits **spurious sub-floor states** (variational-principle-violating, Δ<0) that
a tighter `s_tol` only removes by discarding usable content. So the best **clean**
2° bottom is +1.5×10⁻³ (gcut=4), a **conditioning floor of this formulation** — not a
fundamental completeness wall (the (7,1) convergence + the matched-rank gcut trend
both show the energy is convergent).

### 12.4 Verdict on the path to 2° eigenvalue-exactness

- The compact/exact continuum model **can** reproduce FDFD eigenvalue-for-eigenvalue:
  demonstrated to +3.4×10⁻⁵ at (7,1) once the **X⊕X′ valley** is present.
- At 2° the valley lifts the §9 plateau 4.5× (to +1.5×10⁻³); the residual is **not**
  a wall — it is (a) finite-basis convergence (gcut helps per rank) throttled by
  (b) the **ill-conditioning of the fixed-frame reciprocal (plane-wave) basis**,
  whose redundant high-cutoff modes cannot be cleanly retained.
- Therefore the efficient route to 2° exactness is a **better-conditioned two-valley
  basis**: the real-space *continuously-registry-adapted* envelope (§9's program)
  now carrying **both valleys** — it represents the position-locked envelope without
  the redundant plane waves that ill-condition the reciprocal basis, and (per §11.4)
  with the valley that §9 lacked. This unifies the three threads: exact local
  dispersion (§4/§5), continuous registry adaptation (§9), and the X⊕X′ valley
  (§11). Building it is the defined next program; the exact 4-fold degeneracy would
  follow from a C4-symmetric (fundamental-domain + rotation) construction of that
  basis (audit-scoped, deferred).

*Deliverables: `galerkin_recip.py` (evr + matrix-free H + `band1_weight`),
`fig_exactness_ladder.{png,pdf}`, the (7,1) `galerkin_m7_g4nb{10,12,16}` and 2°
`grecip_2deg_2v_g5` ladders.*


---

## 13. Foundations for the exactness program — floor reconciliation and the symmetry of the 4-fold

Before building a symmetry-adapted basis to fix the 4-fold, a critical design review
flagged two foundational issues. Both are now resolved with
direct computation, and they change the physical picture.

### 13.1 The residual was measured against the wrong operator (floor reconciliation)

The variational Galerkin is a **spectral** method (exact `|X+G|²` kinetic), so its variational
floor is the **continuum** ground of `−∇²E = λ ε_bl E`, approached **from above**. But every prior
residual — and `galerkin_recip.py`'s `--fdfd` comparison — was quoted against the **res16
finite-difference** FDFD ground `0.370047`, which converges to the continuum **from below**. The
two straddle the continuum and differ by the whole size of the remaining gap.

The frozen-candidate (m=57, r1=0.20, r2=0.10, ε=8.9) res-ladder (`floor_reconciliation.py`) is
**O(1/px²)** convergent (the 1/px² slopes are collinear to 2.7%; 1/px is not), giving

| px | ground quad mean | 4-fold split |
|---|---|---|
| 16 | 0.370047 | 1.7×10⁻¹⁰ |
| 32 | 0.370696 | 8.1×10⁻¹¹ |
| 48 | 0.370813 | 2.3×10⁻¹⁰ |

Richardson (1/px², px32→px48): **continuum floor = 0.370907 ± 5.7×10⁻⁶** (the px16→px32 pair gives
0.370912 — agreement to 5×10⁻⁶). Re-baselining against this **spectral-consistent** floor (shift
+8.6×10⁻⁴ off res16):

| result | bottom | old (vs res16) | **new (vs continuum)** |
|---|---|---|---|
| two-valley 9fr band_lo=0 (best clean) | 0.37154 | +1.5×10⁻³ | **+6.3×10⁻⁴** |
| two-valley 9fr band_lo=1 | 0.37248 | +2.4×10⁻³ | +1.6×10⁻³ |
| gcut5 spurious (sub-floor) | 0.36599 | −4.1×10⁻³ | **−4.9×10⁻³** (still sub-floor) |

So the true 2° gap is **~1.4× smaller** than reported, and the gcut5 state stays clearly
variational-violating (not rescued). *Action for Stage 2: repoint the comparison/sub-floor
threshold to 0.370907.* (Caveat for sub-1e-4 claims: the Galerkin's own mass matrix samples ε_bl at
px16/Nsub=8 and reference fields at MPB res=64, an ~1×10⁻⁴ discretization floor of its *own*
operator — so the falsifiable Stage-2 target is ~3×10⁻⁴ unless that resolution is also extrapolated.)

### 13.2 The space group is p4 (chiral), and the exact even-grid C4 is a roll, not `np.rot90`

`stage0c_symmetry.py` tests candidate operations on ε_bl directly (max|ε − gε|/max):

- **C4 about the origin is exact** via the roll-corrected permutation `c4(A) = A[:,(−arange N)%N].T`
  (0.0e+00). **`np.rot90` is NOT a symmetry** (0.89) — it is a half-pixel off the even-grid C4
  centre and would cap any degeneracy fix at O(1/N). (Corrects the earlier "rot90 is exact" note.)
- C4 is also exact about the cell centre τ=(½,½)L; but {C4|(I−C4)τ} = {C4|L1} is C4@origin composed
  with a **full** lattice translation — no extra content.
- **No mirrors** (σ along the axes or diagonals all 0.89): the twist (layer-2 = R(θ)·layer-1,
  r1≠r2) is **chiral**. In 2D every {C4|t}/{C2|t} is a rotation about a shifted centre (no genuine
  screws) and glides require mirrors — so the space group is **symmorphic p4** (point group C4).

### 13.3 The ground 4-fold is the regular rep of C4, and it is EMERGENT (not p4-protected)

`stage0b_characters.py`/`stage0b_analyze.py` solve the 2° FDFD (saving the 4 ground eigenvectors),
reconstruct the periodic parts `u = e^{−iQ·r}·x/√ε` (the eigenvector is `x=√ε·E_full`, Bloch phase
in the stencil), Löwdin-orthonormalise in the ε-metric (ARPACK returns the correct degenerate
subspace but a 0.09-non-orthonormal internal basis across the 1.7×10⁻¹⁰ cluster), and build the
little-group representation `D(g)_{ab}=⟨u_a|Ŝ_g u_b⟩_ε` with the correct symmetry operator
`(Ŝ_g u)(r)=e^{+iG₀·r} u(g⁻¹r)`, `G₀=R_cart Q − Q` (all D(g) unitary to 10⁻¹⁵; C4@quarter correctly
non-unitary as a control). Character table of the 4-fold:

| op | χ | eigenphases | reading |
|---|---|---|---|
| E | 4 | {0,0,0,0} | |
| C4 | 0 | {−90°, 0°, +90°, 180°} = {−i, 1, i, −1} | **all four C4 eigenvalues, one each** |
| C2 | 0 | {−1,−1,+1,+1} | = (C4)² ✓ |

The 4-fold is the **regular representation of C4** = A ⊕ B ⊕ ¹E ⊕ ²E. C4 maps X↔X′ (the valley
structure *is* the C4 action; the C4 eigenstates are 50/50 X/X′ combinations of the measured
2-at-X + 2-at-X′).

**The 4-fold is TWO exact 2-folds that merge only as θ→0.** Resolving the fine structure by C4 irrep
(`stage1_finestructure.py`, at both the 16° and 2° manifolds — the identical C4 structure appears in
*every* moiré manifold, X-carried or not) shows the four levels split by the **C2 = C4² eigenvalue**
into two pairs, each **rigorously degenerate at all angles**:

- **{¹E, ²E}** (C2 = −1): degenerate to ≤10⁻¹⁵. **T-protected** — the T-representation in the 4-fold
  is exactly `T:¹E↔²E`, `T:A→A`, `T:B→B` (`stage0b_analyze` T-test), so T glues the {i,−i} pair.
- **{A, B}** (C2 = +1, C4 = +1 and −1): degenerate to ≤10⁻¹⁵ **but NOT via p4 or T** — T fixes A and
  B individually, and no p4 operation maps the C4=+1 eigenvalue to −1. So A≡B is protected by a
  **hidden / additional symmetry** (candidate: an emergent-exact valley operation; the physical
  A,B = (X1±X′1)/√2 are degenerate iff Re⟨X1|H|X′1⟩=0). **Its origin is a precise open question** —
  the character machinery is in place to resolve it (test antiunitary {g|τ}·T candidates).

The **emergent** quantity is the split *between* the two 2-folds — the {A,B}↔{¹E,²E} (inter-C2-sector)
gap that closes as θ→0 and forms the full 4-fold:

| θ | inter-sector split (→ 4-fold) | gap to 5th |
|---|---|---|
| 16.26° | 1.25×10⁻⁴ | 7.9×10⁻² |
| 2.01° | 1.7×10⁻¹⁰ | 1.9×10⁻⁴ |
| 1.00° | 2.1×10⁻¹¹ | 1.7×10⁻⁶ |

It drops ~6 orders from 16°→2° (the quadruplet staying a well-separated cluster), while the *intra*-
2-fold splits stay ≤10⁻¹⁵ at every angle. So the picture is: two rigorous 2-folds (one T-protected,
one hidden-symmetry-protected) that an **emergent small-angle symmetry** fuses into the 4-fold as
θ→0 — the answer to the earlier question "missing physics vs numerical gauge?" is a third one: an emergent
symmetry (BM-type valley physics), on top of two rigorous 2-folds.

### 13.4 Consequences for the program

- **Stage 1 (fix the 4-fold) splits in three.** A C4+T-closed trial basis restores exact C4 quantum
  numbers and the **rigorous ¹E,²E 2-fold** exactly, and removes the numerical valley-mixing
  artifact. The **A≡B 2-fold** needs the basis to also respect the hidden A≡B symmetry — to be
  measured; if the model splits A,B, that identifies the missing symmetry. The **inter-sector merge**
  into the 4-fold is *emergent* (θ-suppressed) — a convergence target, not enforced by symmetrization.
  This three-way split is the honest statement of "completely fix the 4-fold."
- **Stage 4 (plain-EA 1/2 vs 1/4) is predicted 1/2, on firmer footing.** C2@origin is a *rigorous*
  p4 symmetry fixing X ((π,0)→(−π,0)≡(π,0)); single-carrier-X EA is C2-invariant, so it spans **both**
  X-valley states → recovers **2 of 4 = 1/2** of each quadruplet, split by the removed C4-link to X′.
  "1/4" would require an extra C4-irrep projection that "plain" does not perform. (Test empirically.)

*Deliverables: `floor_reconciliation.py` (+`.npz`), `stage0c_symmetry.py`, `stage0b_characters.py`
(+`.npz` with the 4 ground eigenvectors), `stage0b_analyze.py`, `stage1_finestructure.py`. Continuum
floor **0.370907**; space group **p4**; 4-fold = **regular rep of C4 = two rigorous 2-folds
(T-protected ¹E²E + hidden-symmetry A≡B) fused emergently as θ→0**. Note: the (7,1)/16° manifold that
§12 validated (f≈0.067) has w_X=0 — at β≫1 the band-1-at-X manifold has dissolved (§8.2); it shares
the C4 fine-structure but is a different band, so it is a symmetry testbed, not an X-manifold proxy.*

---

## 14. Fixing the 4-fold — the C4-irrep-projected Galerkin

§13 labelled the target. §14 supplies the fix and tests it.

### 14.1 Naive C4-closure is not enough; explicit irrep projection is

A first attempt (`stage1_c4basis.py`) generates the X′ block as the exact C4-image of the X block —
on the sparse supercell plane-wave coeffs, C4 is the index permutation `n → nG₀ + C4·n`
(nG₀ = Bᵀ(X′−X)/2π = ((1−m)/2, (1+m)/2); C4·(n1,n2)=(−n2,n1); value unchanged; derivation
M_C4 = Bᵀ·C4·B⁻ᵀ = C4, an integer matrix, and the G₀=X′−X shift realises the Bloch gauge as an index
translation). The self-check is exact: the C4-image populates **12544/12544** of the independent-
extraction X′ indices. **But this alone does not restore the 2-fold** (min ground gap 2.3×10⁻⁵, same
as independent extraction): the canonical orthogonalisation + generalised eigensolve do not enforce
the symmetry, and the MPB-gauge X block is only *approximately* C4-closed (C4²·X ≈ X only up to the
per-k gauge + the nbands truncation).

### 14.2 The fix: project each seed onto the C4 irreps

`stage1_c4proj.py` symmetry-adapts. Using the exact grid permutation P (n → nG₀ + C4·n), it projects
every X-seed onto the four C4 irreps
  v_χ(b) = ¼ Σ_{k=0}^{3} χ̄ᵏ · Pᵏ C[:,b],   χ ∈ {A:1, B:−1, ¹E:i, ²E:−i},
assembles H,S **within each irrep block**, and solves. Because ¹E and ²E carry conjugate characters,
their blocks are exact time-reversal images → **identical spectra by construction**, so the {¹E,²E}
2-fold is machine-exact independent of energy convergence. A, B are the C2=+1 sector.

### 14.3 Result: the rigorous 2-fold is restored exactly; A≡B converges

| cell | basis | max\|f(¹E)−f(²E)\| | A–B split | inter-sector split |
|---|---|---|---|---|
| m=7 (16°) | gcut3, nb2 | **9.2×10⁻¹²** | 2.7×10⁻⁵ | 1.11×10⁻⁴ |
| m=7 (16°) | gcut4, nb3 | **1.4×10⁻¹¹** | **5.9×10⁻⁶** | 1.04×10⁻⁴ |
| m=57 (2°) | gcut4, nb2 | **7.0×10⁻¹²** (over 634 levels) | 2.0×10⁻⁶ | 1.32×10⁻⁴ |

Three facts. **(i)** The **rigorous ¹E≡²E 2-fold is restored to ~10⁻¹¹** at both the 16° testbed and
the 2° target (vs ~2×10⁻⁵ un-projected) — the symmetry-adapted basis reproduces the T-protected
degeneracy by construction. **(ii)** The **A≡B split converges** (2.7×10⁻⁵ → 5.9×10⁻⁶ as gcut3,nb2 →
gcut4,nb3): the hidden A≡B symmetry is respected in the complete-basis limit, so A≡B is a *convergence
target*, not a hard obstruction the basis breaks. **(iii)** The **emergent inter-C2-sector split**
(¹E²E vs A,B) comes out 1.0–1.3×10⁻⁴, matching FDFD's physical 1.25×10⁻⁴ at 16° — the model
reproduces the emergent θ-physics.

*(The per-irrep grounds shown at 2° (f≈0.251) are band-0 states — a single reference frame (nref=1)
under-populates the 0.370 band-1-at-X manifold (§8.3); the ¹E≡²E exactness is a universal statement
over all 634 levels, so it holds for the manifold. Resolving the manifold's own quadruplet needs the
registry-adapted basis of Stage 2.)*

### 14.4 Verdict

"Completely fixing the 4-fold" resolves into: **the rigorous ¹E≡²E 2-fold is fixed exactly by
C4-irrep projection** (the load-bearing, symmetry-protected part), **A≡B is fixed convergently** (the
hidden-symmetry part, respected as the basis completes), and **the 4-fold merge is the emergent θ→0
physics** (a convergence/angle quantity, not a symmetry to impose). The vehicle is a symmetry-adapted
(C4-irrep-block) Galerkin — which additionally **block-diagonalises H,S into four smaller, better-
conditioned blocks**, a free win for the Stage-2 conditioning problem. Open: the exact identity of the
hidden A≡B symmetry (not p4, not T, not any {g|τ}·T; its convergent restoration says the complete
basis carries it — a precise question for a follow-up).

*Deliverables: `stage1_c4basis.py` (C4-remap + self-check + naive-closure null result),
`stage1_c4proj.py` (C4-irrep projection; `stage1_c4proj_m{7,57}.npz`). The 4-fold fix = **C4-irrep
projection: ¹E≡²E exact to 10⁻¹¹, A≡B convergent, emergent merge reproduced.***
