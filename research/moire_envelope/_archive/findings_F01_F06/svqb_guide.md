# SVQB B-Orthonormalization: Implementation Guide

## What SVQB Does

Given `p` vectors that may be nearly linearly dependent under the B-inner product, SVQB produces a well-conditioned B-orthonormal basis for their span. It replaces Gram-Schmidt, which fails catastrophically near degeneracies.

**Inputs:** `vectors[0..p]`, `mass_vectors[0..p]` (precomputed `B * vectors[i]`), a `SvqbConfig` with `drop_tol`.

**Outputs:** The first `rank` slots of `vectors` / `mass_vectors` are overwritten with B-orthonormal vectors. Excess slots are zeroed. Returns `SvqbResult` with rank info.

## Algorithm Steps

```
1.  Pre-normalize each column to unit B-norm               ← essential, see Pitfall #1
2.  Form Gram matrix  G = X^H · (BX)     via GEMM          ← O(n·p²), all in f64
3.  Eigendecompose     G = Q Λ Q^H        (Hermitian)       ← O(p³), sorted descending
4.  Rank-reveal:       drop λ_i / λ_max < drop_tol          ← default 1e-12
5.  Build transform    T = Q_kept · Λ_kept^{-1/2}           ← p × rank
6.  Apply              X_new = X_old · T   via GEMM          ← O(n·p·rank), f64 accum
                       (BX)_new = (BX)_old · T
7.  Zero excess slots  [rank..p]
```

The key invariant after SVQB: `X_new^H · B · X_new = I_rank`.

## Pitfalls & Lessons Learned

### Pitfall #1: Pre-normalization Is Mandatory

**Problem:** Without pre-normalizing columns, a tiny residual vector (e.g., `||w|| ~ 1e-8`) gets a Gram eigenvalue `~ 1e-16`, which falls below `drop_tol * λ_max` and is discarded — even though it carries an essential search direction.

**Fix (line ~300):** Before forming the Gram matrix, every column is normalized to unit B-norm with a very permissive tolerance (`1e-30`). This decouples "how big is the vector" from "is it linearly independent of the others."

### Pitfall #2: Conjugation in `project_out`

**Problem:** The `backend.dot(x, y)` computes `x^H · y` (conjugate-linear in the first argument). To project `v` along `u`, you need coefficient `α = ⟨u, v⟩_B = u^H · Bv`. But `dot(v, Bu)` gives `v^H · Bu` — that's `⟨v, u⟩_B`, which is the conjugate.

**Fix (line ~153):** `let coeff = backend.dot(vector, basis_mass).conj();` — the `.conj()` converts from `⟨v,u⟩_B` to `⟨u,v⟩_B`.

### Pitfall #3: Mixed-Precision Accumulation

All Gram matrix entries and the GEMM transformation are computed in **f64**, regardless of whether the storage type is f32 (mixed-precision mode). The final write-back downcasts to the storage precision. Without this, the Gram matrix accumulates roundoff errors that corrupt the eigendecomposition.

### Pitfall #4: TE vs TM Mass Operator

The mass operator `B` depends on polarization:
- **TE:** `B = I` (identity) → `⟨x,y⟩_B = x^H y` (standard dot product)
- **TM:** `B = diag(ε(r))` → `⟨x,y⟩_B = x^H · ε · y` (ε-weighted)

If you're orthonormalizing Bloch functions from MPB, you need the correct `B` for your polarization. The mass vectors `Bx` must be precomputed before calling SVQB.

### Pitfall #5: Eigenvalue Sorting

faer returns eigenvalues in **ascending** order; the code reverses them to **descending**. nalgebra returns them unsorted; the code sorts explicitly. Both paths produce `eigenvalues[0] = λ_max`. The drop threshold is `drop_tol * λ_max`, so getting this wrong silently drops the wrong vectors.

## How the LOBPCG Eigensolver Uses SVQB

Each iteration, the search subspace `[X, P, W]` is B-orthonormalized:

```
X  (m vectors) — current Ritz vectors
P  (m vectors) — conjugate directions from previous iteration
W  (m vectors) — preconditioned residuals
```

1. The mass vectors `B·X` are precomputed and **reused** (saves `m` operator applications).
2. `B·P` and `B·W` are computed fresh via `batch_apply_mass`.
3. SVQB orthonormalizes the concatenated `[X, P, W]` block (up to `3m` vectors).
4. If rank < 3m, vectors are dropped. `SvqbResult::compute_block_drops()` tracks which block (X, P, or W) lost vectors. Losing X vectors is critical; losing W vectors is expected near convergence.

## How to Apply This for Envelope Theory

**Goal:** B-orthonormalize the Bloch functions `{u_n(R, r)}` extracted from MPB at each shift-grid point `R`, so they satisfy `⟨u_m, u_n⟩_Ω = δ_mn`.

### Step-by-Step

1. **Define your B operator.** For the Bloch cell-periodic inner product:
   - **TM polarization:** `⟨u_m, u_n⟩ = ∫_Ω u_m*(r) ε(r) u_n(r) dr` → `B = diag(ε)` in reciprocal space, applied pointwise in real space.
   - **TE polarization:** `⟨u_m, u_n⟩ = ∫_Ω u_m*(r) u_n(r) dr` → `B = I`.

2. **Extract ε(r) from MPB.** MPB stores the dielectric on the same grid. Export it and load into your `Dielectric2D`. The `ThetaOperator::apply_mass()` already implements this correctly.

3. **Prepare the inputs.**
   ```
   vectors[i]      = u_i(R, ·)    in plane-wave coefficients  (length N)
   mass_vectors[i]  = B · u_i(R, ·) precomputed
   ```

4. **Call SVQB.**
   ```rust
   let config = SvqbConfig { drop_tol: 1e-12 };
   let result = svqb_orthonormalize(backend, &mut vectors, &mut mass_vectors, &config);
   ```

5. **Check the result.**
   - `result.output_rank` should equal `N_bands`. If less, you lost bands — either your MPB extraction included near-degenerate modes that collapsed, or ε was inconsistent.
   - `result.had_rank_deficiency()` → `true` is a warning.

6. **Use the orthonormalized `u_n`.** The first `result.output_rank` entries in `vectors` are now B-orthonormal. Feed them into your Berry connection / velocity matrix element computations.

### Why Not Just Use Gram-Schmidt?

At high-symmetry k-points (Γ, X, M), bands become degenerate. Two Bloch functions spanning the same 2D eigenspace will have nearly parallel components. Gram-Schmidt subtracts one from the other, leaving a vector with norm `~ ε_machine` — catastrophic cancellation. SVQB handles this via the SVD/eigendecomposition of the Gram matrix, rotating within the degenerate subspace instead of subtracting.

### Integration Checklist

- [ ] Dielectric `ε(r)` on the same grid as the Bloch functions
- [ ] Mass vectors `B·u_i` precomputed (one `apply_mass` per band)
- [ ] `drop_tol` set to `1e-12` (default) — may need tuning if ε contrast is extreme
- [ ] Rank check after SVQB: `assert_eq!(result.output_rank, n_bands)`
- [ ] If using mixed-precision: storage is f32, accumulation is f64 — SVQB handles this internally
