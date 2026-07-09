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
