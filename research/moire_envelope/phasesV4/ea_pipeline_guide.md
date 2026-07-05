# EA Hamiltonian Extraction — Developer Guide

> **Audience:** Researchers using Blaze2D's Python API to extract all
> envelope-approximation (EA) Hamiltonian ingredients for moiré photonic
> crystals, including registry sweeps over a 2-atomic unit cell and
> k-stencil sweeps around high-symmetry points.
>
> **Package:** `blaze2d ≥ 0.5.1` on PyPI. Import as `from blaze import EAExtractor`.

---

## 1. Physical background

At each registry point **R** (the relative sliding offset of two
photonic-crystal layers) and carrier momentum **k₀** (a high-symmetry
point in the moiré Brillouin zone), Blaze2D solves the local Maxwell
eigenproblem of the small unit cell and extracts the matrix elements
that enter the multi-band envelope Hamiltonian:

$$
H_{\mathrm{EA}} = \Lambda + \sum_i v^{(i)} q_i
+ \tfrac{1}{2} \sum_{ij} M^{-1}_{ij}\, q_i q_j
+ \Phi_{\mathrm{BH}}
$$

where $q = k - k_0$ is the crystal-momentum deviation.

The ingredients extracted at each (R, k₀) are:

| # | Quantity | Symbol | Shape |
|---|----------|--------|-------|
| 1 | Eigenvalues | $\lambda_n$ | `n_total` |
| 2 | Eigenvectors | $u_n(\mathbf{G})$ | `n_total × grid_size` |
| 3 | Velocity matrices | $v^{(i)}_{mn} = \langle u_m | \partial L_0 / \partial k_i | u_n \rangle$ | `n_retained × n_total` |
| 4 | Second-derivative matrices | $w^{(ij)}_{mn} = \langle u_m | \partial^2 L_0 / \partial k_i \partial k_j | u_n \rangle$ | `n_retained × n_retained` |
| 5 | Löwdin-corrected inverse mass tensor | $M^{-1}_{ij,mn}$ | `n_retained × n_retained` |
| 6 | R-derivative matrices | $\langle u_m | \partial L_0 / \partial R_j | u_n \rangle$ | `n_retained × n_total` |
| 7 | Born–Huang potential | $\Phi_{mn}(\mathbf{R})$ | `n_retained × n_retained` |
| 8 | Overlap matrix | $S_{mn} = \langle u_m(\mathbf{R}) | u_n(\mathbf{R'}) \rangle$ | `n_retained × n_retained` |

Here `n_total = n_retained + n_remote`. The remote bands enter only
through the Löwdin perturbation sums in the mass tensor.

---

## 2. Units and coordinate conventions

| Quantity | Units | Convention |
|----------|-------|------------|
| Lattice vectors `a₁, a₂` | length (arbitrary, call it *a*) | Cartesian 2D, row vectors |
| Atom positions | **fractional** (of `a₁, a₂`) | `[f₁, f₂]` so that real-space pos = f₁·a₁ + f₂·a₂ |
| Atom radius | **fractional** | fraction of lattice constant |
| k₀ (carrier momentum) | **Cartesian reciprocal space**, units of 2π/a | not fractional of b₁, b₂ |
| Eigenvalues λ | (2π/a)² | = (ω/c)² in natural units |
| Velocity matrices | (2π/a) | eigenvalue per k-unit |
| w, mass tensor | dimensionless | eigenvalue per k²-unit |
| R-derivative matrices | (2π/a)² / fractional | eigenvalue per fractional-R-unit |
| Born–Huang Φ | (2π/a)² | same as eigenvalue |
| Registry R | **fractional** | metadata label; atoms must be pre-shifted |
| FD step | **fractional** | finite-difference step for ∂ε/∂R |

**Key point:** The `registry` parameter is **metadata only**. It does
*not* shift the atoms for you. You must shift the atom positions in
the `atoms` list yourself before calling `extract()`. The `registry`
value is stored in the output dict for bookkeeping.

---

## 3. Single-point extraction

```python
from blaze import EAExtractor

result = EAExtractor.extract(
    lattice_vectors=[[1.0, 0.0], [0.5, 0.866025]],  # hexagonal
    atoms=[
        {"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 1.0},
        {"pos": [1/3, 1/3], "radius": 0.2, "eps_inside": 1.0},
    ],
    eps_bg=12.0,
    k0=[0.0, 0.0],          # Γ point
    polarization="TE",
    resolution=32,
    n_retained=4,
    n_remote=8,
    compute_born_huang=True, # optional, default False
    compute_overlap=False,   # optional, default False
    compute_r_derivatives=True,
    atom_index=0,            # which atom for ∂/∂R
    fd_step=0.001,
    registry=[0.0, 0.0],    # metadata only
    tolerance=1e-8,
    max_iterations=300,
    smoothing=True,
)
```

### Result dictionary keys

**Metadata:**
- `polarization` (str), `inner_product` (str), `k0` (tuple), `registry` (tuple)
- `n_retained` (int), `n_remote` (int), `grid_dims` (tuple)
- `n_iterations` (int), `converged` (bool)
- `solve_time_seconds` (float), `extract_time_seconds` (float)

**Eigensystem:**
- `eigenvalues` — list of `n_total` floats
- `eigenvectors` — list of `n_total` lists of `(re, im)` tuples

**Velocity matrices** (shape: `n_retained × n_total`, row-major):
- `velocity_matrices_x`, `velocity_matrices_y` — list of `(re, im)`
- `velocity_matrix_rows`, `velocity_matrix_cols`

**W matrices** (shape: `n_retained × n_retained`):
- `w_matrices_xx`, `w_matrices_xy`, `w_matrices_yx`, `w_matrices_yy`
- `w_matrix_size`

**Inverse mass tensor** (shape: `n_retained × n_retained`):
- `mass_tensor_inv_xx`, `mass_tensor_inv_xy`, `mass_tensor_inv_yx`, `mass_tensor_inv_yy`

**R-derivative matrices** (shape: `n_retained × n_total`, optional):
- `r_derivative_matrices_x`, `r_derivative_matrices_y`
- `has_r_derivatives` (bool)

**Born–Huang** (shape: `n_retained × n_retained`, optional):
- `born_huang` — list of `(re, im)`
- `has_born_huang` (bool)

**Overlap** (shape: `n_retained × n_retained`, optional):
- `overlap_matrix` — list of `(re, im)`
- `has_overlap` (bool)

### Reshaping to NumPy

All matrix data is returned as flat row-major lists of `(re, im)` tuples.
To get proper complex NumPy arrays:

```python
import numpy as np

n_ret = result["n_retained"]
n_tot = n_ret + result["n_remote"]

def to_matrix(data, rows, cols):
    arr = np.array([(re + 1j * im) for re, im in data])
    return arr.reshape(rows, cols)

v_x = to_matrix(result["velocity_matrices_x"], n_ret, n_tot)
v_y = to_matrix(result["velocity_matrices_y"], n_ret, n_tot)

M_inv_xx = to_matrix(result["mass_tensor_inv_xx"], n_ret, n_ret)
# ... etc.
```

---

## 4. k-Stencil sweep

For band-structure verification or k·p finite-difference checks, use
`extract_k_stencil` to solve on a grid of k-points around k₀.

The method:
1. **Builds the dielectric once** (shared across all k-points).
2. **Solves at center k₀** first (cold start).
3. **Warm-starts all neighbors** from the center eigenvectors → fast convergence.

```python
stencil = EAExtractor.extract_k_stencil(
    lattice_vectors=[[1.0, 0.0], [0.5, 0.866025]],
    atoms=[
        {"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 1.0},
        {"pos": [1/3, 1/3], "radius": 0.2, "eps_inside": 1.0},
    ],
    eps_bg=12.0,
    k0=[0.0, 0.0],
    polarization="TE",
    resolution=32,
    n_stencil=3,       # 3×3 grid → 9 k-points (1 center + 8 neighbors)
    delta_k=0.01,      # max displacement in each direction (2π/a)
    n_retained=4,
    n_remote=8,
)
```

### Stencil geometry

`n_stencil` is the number of points **per axis** (must be odd).

| `n_stencil` | Grid | Total points | Neighbors |
|-------------|------|--------------|-----------|
| 1 | 1×1 | 1 | 0 |
| 3 | 3×3 | 9 | 8 |
| 5 | 5×5 | 25 | 24 |
| 7 | 7×7 | 49 | 48 |

The spacing between stencil points is `delta_k / ((n_stencil - 1) / 2)`.
For `n_stencil=3, delta_k=0.01`: spacing = 0.01, points at k₀ ± 0.01
along each axis.

### Stencil result dictionary

```python
center = stencil["center"]           # dict, same format as extract()
neighbors = stencil["neighbors"]     # list of dicts
k_points = stencil["neighbor_k_points"]  # list of (kx, ky) tuples
n_stencil = stencil["n_stencil"]     # int
delta_k = stencil["delta_k"]         # float
```

### Example: verify group velocity via finite differences

```python
import numpy as np

center = stencil["center"]
lam_center = np.array(center["eigenvalues"][:4])  # retained bands

# Find the +x neighbor (δkx = +delta_k, δky = 0)
for i, kp in enumerate(stencil["neighbor_k_points"]):
    if abs(kp[0] - center["k0"][0] - stencil["delta_k"]) < 1e-12 \
       and abs(kp[1] - center["k0"][1]) < 1e-12:
        lam_plus_x = np.array(stencil["neighbors"][i]["eigenvalues"][:4])
        break

# Find the -x neighbor
for i, kp in enumerate(stencil["neighbor_k_points"]):
    if abs(kp[0] - center["k0"][0] + stencil["delta_k"]) < 1e-12 \
       and abs(kp[1] - center["k0"][1]) < 1e-12:
        lam_minus_x = np.array(stencil["neighbors"][i]["eigenvalues"][:4])
        break

# Central FD group velocity
dk = stencil["delta_k"]
v_fd_x = (lam_plus_x - lam_minus_x) / (2 * dk)

# Analytic velocity (diagonal elements of v_x)
n_ret = center["n_retained"]
n_tot = n_ret + center["n_remote"]
v_x = np.array([(re + 1j*im) for re, im in center["velocity_matrices_x"]])
v_x = v_x.reshape(n_ret, n_tot)
v_analytic_x = np.diag(v_x[:n_ret, :n_ret]).real

print("FD group velocity:", v_fd_x)
print("Analytic velocity:", v_analytic_x)
```

---

## 5. Registry sweep — 2-atomic unit cell

In a moiré bilayer, the local stacking configuration varies smoothly
across the supercell. This is parameterized by the **registry** R —
the relative displacement of one layer's atom(s) within the unit cell.

For a **2-atom basis** (e.g., honeycomb), you sweep R by physically
shifting one atom in fractional coordinates. Blaze2D does not do this
automatically; you construct the shifted geometry for each R-point.

### Strategy

```
For each R-point on an Nᴿ × Nᴿ grid:
    1. Shift atom positions to the registry R
    2. Call EAExtractor.extract(..., registry=[Rx, Ry])
    3. Store the result indexed by R
```

### Complete registry sweep example

```python
import numpy as np
from blaze import EAExtractor

# --- Geometry definition ---
a1 = [1.0, 0.0]
a2 = [0.5, np.sqrt(3)/2]
eps_bg = 12.0

# Base atom positions (fractional) — 2-atom honeycomb
atom_A_base = [0.0, 0.0]
atom_B_base = [1/3, 1/3]
radius = 0.2
eps_rod = 1.0

# --- Sweep parameters ---
k0 = [0.0, 0.0]               # Γ point (or K, M, ...)
pol = "TE"
resolution = 32
n_retained = 4
n_remote = 8
N_R = 11                       # 11×11 registry grid

# --- Run sweep ---
R_grid = np.linspace(0, 1, N_R, endpoint=False)
results = {}

for i, Rx in enumerate(R_grid):
    for j, Ry in enumerate(R_grid):
        # Shift atom B by the registry offset (atom A stays fixed)
        atom_B_shifted = [
            (atom_B_base[0] + Rx) % 1.0,
            (atom_B_base[1] + Ry) % 1.0,
        ]
        atoms = [
            {"pos": atom_A_base, "radius": radius, "eps_inside": eps_rod},
            {"pos": atom_B_shifted, "radius": radius, "eps_inside": eps_rod},
        ]

        result = EAExtractor.extract(
            lattice_vectors=[a1, a2],
            atoms=atoms,
            eps_bg=eps_bg,
            k0=k0,
            polarization=pol,
            resolution=resolution,
            n_retained=n_retained,
            n_remote=n_remote,
            compute_born_huang=True,
            compute_r_derivatives=True,
            atom_index=1,            # differentiate w.r.t. atom B
            fd_step=0.001,
            registry=[Rx, Ry],       # metadata tag
        )

        results[(i, j)] = result
        print(f"R=({Rx:.2f}, {Ry:.2f}): {result['n_iterations']} iters, "
              f"converged={result['converged']}")

# --- Post-process: build Λ(R) landscape ---
eigenvalue_map = np.zeros((N_R, N_R, n_retained))
for (i, j), res in results.items():
    eigenvalue_map[i, j, :] = res["eigenvalues"][:n_retained]
```

### Overlap matrix for gauge transport

When sweeping over R, the eigenvector gauge is generally discontinuous
between neighboring R-points. The overlap matrix $S_{mn}(\mathbf{R}
\to \mathbf{R'})$ enables parallel transport.

To compute overlaps, you need to run at two neighboring R-points and
pass one's eigenvectors as the reference:

> **Note:** As of v0.5.1, the overlap matrix is computed internally by
> the Rust solver when `compute_overlap=True`, using the reference
> eigenvectors from a neighboring R-point. This requires running
> `run_with_reference()` at the Rust driver level. The Python API
> currently computes the overlap at each individual R-point (self-overlap),
> which is the identity matrix and not useful for gauge transport.
>
> For proper gauge transport, compute the overlap manually in Python:
>
> ```python
> # After extracting at R and R':
> eigvecs_R  = [np.array([re + 1j*im for re, im in ev])
>               for ev in result_R["eigenvectors"][:n_retained]]
> eigvecs_Rp = [np.array([re + 1j*im for re, im in ev])
>               for ev in result_Rp["eigenvectors"][:n_retained]]
>
> # Overlap: S_mn = <u_m(R) | u_n(R')>
> S = np.zeros((n_retained, n_retained), dtype=complex)
> for m in range(n_retained):
>     for n in range(n_retained):
>         S[m, n] = np.vdot(eigvecs_R[m], eigvecs_Rp[n]) / len(eigvecs_R[m])
> ```

---

## 6. Combined sweep: R-grid × k-stencil

For the full EA Hamiltonian construction, you need ingredients at many
R-points. At each R-point, you can optionally run a k-stencil for
verification. A practical pattern:

```python
import numpy as np
from blaze import EAExtractor

a1 = [1.0, 0.0]
a2 = [0.5, np.sqrt(3)/2]

N_R = 11
R_grid = np.linspace(0, 1, N_R, endpoint=False)

all_results = {}

for i, Rx in enumerate(R_grid):
    for j, Ry in enumerate(R_grid):
        atoms = [
            {"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 1.0},
            {"pos": [(1/3 + Rx) % 1.0, (1/3 + Ry) % 1.0],
             "radius": 0.2, "eps_inside": 1.0},
        ]

        stencil = EAExtractor.extract_k_stencil(
            lattice_vectors=[a1, a2],
            atoms=atoms,
            eps_bg=12.0,
            k0=[0.0, 0.0],
            polarization="TE",
            resolution=32,
            n_stencil=3,
            delta_k=0.01,
            n_retained=4,
            n_remote=8,
            compute_born_huang=True,
            registry=[Rx, Ry],
            atom_index=1,
        )

        all_results[(i, j)] = stencil
```

---

## 7. Inner product conventions

Understanding which inner product is used is critical for interpreting
matrix elements.

### TE mode

The eigenproblem is standard: $L_0 u = \lambda u$.

$$L_0 = -\nabla \cdot \varepsilon^{-1}(\mathbf{r})\, \nabla$$

Inner product: standard $\langle u | v \rangle = \frac{1}{N} \sum_{\mathbf{G}} u^*(\mathbf{G})\, v(\mathbf{G})$.

Mass matrix $B = I$ (identity). Eigenvectors satisfy $\langle u_m | u_n \rangle = \delta_{mn}$.

### TM mode

The eigenproblem is generalized: $A u = \lambda B u$.

$$A = -|k+G|^2, \qquad B = \varepsilon(\mathbf{r})$$

Inner product: $B$-weighted $\langle u | v \rangle_B = \frac{1}{N} \sum_{\mathbf{r}} u^*(\mathbf{r})\, \varepsilon(\mathbf{r})\, v(\mathbf{r})$.

Eigenvectors satisfy $\langle u_m | B | u_n \rangle = \delta_{mn}$.

**Both polarizations:** The velocity and w matrices are computed with
the standard `dot()` (without explicit mass-matrix application), which
is correct because:
- TE: mass is identity
- TM: $B$ is k-independent, so $\partial B / \partial k = 0$, and for
  $B$-orthonormal eigenvectors, $v^{(i)}_{mn} = \langle u_m | \partial A / \partial k_i | u_n \rangle_{\mathrm{std}}$

The `inner_product` key in the result dict tells you which convention
was used (`"standard"` for TE, `"B_weighted"` for TM).

---

## 8. Polarization-specific operator derivatives

For reference, the operator derivatives applied internally:

### TE

$$\frac{\partial L_0}{\partial k_x}\bigg|_{\mathbf{G}} = -\varepsilon^{-1}(\mathbf{G})\, \bigl[2(k_x + G_x)\bigr]$$

$$\frac{\partial^2 L_0}{\partial k_x^2}\bigg|_{\mathbf{G}} = -2\,\varepsilon^{-1}(\mathbf{G})$$

### TM

$$\frac{\partial A}{\partial k_x}\bigg|_{\mathbf{G}} = -2(k_x + G_x)$$

$$\frac{\partial^2 A}{\partial k_x^2} = -2$$

(Pure k-space multipliers; $B = \varepsilon(\mathbf{r})$ is k-independent.)

### R-derivatives

Computed via **central finite differences** on the dielectric:

$$\frac{\partial \varepsilon}{\partial R_j} \approx \frac{\varepsilon(\mathbf{R} + \delta_j \hat{e}_j) - \varepsilon(\mathbf{R} - \delta_j \hat{e}_j)}{2\delta_j}$$

The `atom_index` parameter selects which atom is shifted. The `fd_step`
parameter controls $\delta_j$ (in fractional coordinates).

---

## 9. Löwdin mass tensor

The inverse effective mass tensor is:

$$M^{-1}_{ij,mn} = w^{(ij)}_{mn} + \sum_{l \in \mathrm{remote}} \left(\frac{v^{(i)}_{ml}\, v^{(j)}_{ln}}{\lambda_m - \lambda_l} + \frac{v^{(j)}_{ml}\, v^{(i)}_{ln}}{\lambda_m - \lambda_l}\right)$$

This is already computed and available as `mass_tensor_inv_xx` etc.
The sum runs over the `n_remote` bands. Increasing `n_remote` improves
the mass tensor accuracy at the cost of solving for more bands.

**Typical values:** `n_retained=2–6`, `n_remote=8–20`. Convergence of
the mass tensor with `n_remote` should be checked.

---

## 10. Practical tips

### Resolution

- `resolution=32` is a good starting point for development/testing.
- `resolution=64` for publication-quality results.
- `resolution=128` if you need high accuracy on gap sizes.
- Runtime scales as $O(N^2 \log N)$ where N = resolution².

### Convergence

- Check `result["converged"]` — if `False`, increase `max_iterations`
  or `resolution`.
- Typical iteration counts: 30–100 for well-conditioned problems.
- The k-stencil warm-start typically halves the iteration count for
  neighbor points.

### Atom naming

The `atom_index` parameter indexes into the `atoms` list. For a
2-atom basis:
- `atom_index=0` → derivatives with respect to atom A
- `atom_index=1` → derivatives with respect to atom B

For a moiré bilayer where one layer slides, you typically differentiate
with respect to the sliding atom (e.g., atom B, `atom_index=1`).

### Band ordering

Eigenvalues are returned sorted in ascending order. Band crossings
at different R-points will cause the band indices to permute. For
smooth interpolation across R, use the overlap matrix or eigenvector
continuity to track bands.

### Dielectric smoothing

Enabled by default (`smoothing=True`). This applies subpixel smoothing
at dielectric interfaces, which significantly improves convergence
with resolution. Disable for debugging or comparison with unsmoothed
solvers.

---

## 11. API reference summary

### `EAExtractor.extract(...)`

Single (R, k₀) point extraction. Returns a dict with all EA ingredients.

**Required:** `lattice_vectors`, `atoms`, `eps_bg`, `k0`, `polarization`, `resolution`

**Optional (with defaults):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_retained` | 4 | Active subspace bands |
| `n_remote` | 8 | Remote bands for Löwdin sums |
| `compute_born_huang` | False | Compute Born–Huang potential |
| `compute_overlap` | False | Compute overlap matrix |
| `compute_r_derivatives` | True | Compute ∂L₀/∂R matrix elements |
| `atom_index` | 0 | Atom for R-derivatives |
| `fd_step` | 0.001 | FD step for ∂ε/∂R (fractional) |
| `registry` | [0, 0] | Metadata label |
| `tolerance` | 1e-8 | Eigensolver convergence |
| `max_iterations` | 300 | Max LOBPCG iterations |
| `smoothing` | True | Subpixel dielectric smoothing |

### `EAExtractor.extract_k_stencil(...)`

k-stencil sweep around k₀. Same parameters as `extract()`, plus:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `n_stencil` | yes | Points per axis (odd, ≥ 1) |
| `delta_k` | yes | Max displacement (2π/a) |

Returns dict with `center`, `neighbors`, `neighbor_k_points`, `n_stencil`, `delta_k`.

---

## 12. Changelog

- **v0.5.1** — Added `extract_k_stencil()`, exposed `compute_overlap` parameter.
- **v0.5.0** — Initial EA extraction API (`extract()`), published to PyPI.
