# Geometric Conventions for Moiré Photonic Crystal Simulations

This document summarizes the coordinate, grid, and lattice conventions used across FDFD, MPB, and the Envelope Approximation (EA). Established during epsilon-grid forensics (March 2026) to prevent future alignment bugs.

---

## 1. Grid Sampling Convention

| Solver | Grid point formula | Domain | Origin |
|--------|-------------------|--------|--------|
| **FDFD** | `s = i/N`, `i = 0..N-1` | [0, 1) | Cell **corner** |
| **EA** | `s = i/N`, `i = 0..N-1` | [0, 1) | Cell **corner** |
| **MPB** | `s = (i+0.5)/N`, `i = 0..N-1` | (0, 1) | Cell **center** (mapped to [-0.5, 0.5)) |

**Consequence:** FDFD and EA grids are natively compatible. MPB grids require a **half-cell roll** to align:

```python
eps_mpb = np.roll(eps_mpb, (Nx // 2, Ny // 2), axis=(0, 1))
```

The half-pixel offset (`1/(2N)`) between FDFD corners and MPB centers is irrelevant for bulk statistics but causes ~5-16% of **boundary pixels** to disagree in pixelwise comparisons. This is an intrinsic grid-convention artifact, not a bug.

---

## 2. Physical Coordinates

All three solvers map fractional coordinates `(s₁, s₂)` to physical coordinates via:

$$\mathbf{r} = s_1 \, \mathbf{L}_1 + s_2 \, \mathbf{L}_2$$

where `L₁`, `L₂` are the lattice vectors of the computational cell (monolayer unit cell or moiré supercell).

---

## 3. Honeycomb Monolayer Basis

Lattice constant `a`, primitive vectors:

$$\mathbf{a}_1 = a \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \qquad
\mathbf{a}_2 = a \begin{pmatrix} 1/2 \\ \sqrt{3}/2 \end{pmatrix}$$

Sublattice positions (fractional): `(0, 0)` and `(1/3, 1/3)`.

Rod radius: `r = r/a · a` (physical units), where `r/a = 0.2` is the standard value.

---

## 4. Coincidence (Commensurate) Supercell

### 4.1 Correct Convention

For a twisted bilayer with commensurate indices `(m, n)`, the **coincidence supercell** vectors are:

$$\mathbf{C}_1 = n \, \mathbf{a}_1 + m \, \mathbf{a}_2, \qquad
\mathbf{C}_2 = -m \, \mathbf{a}_1 + (m+n) \, \mathbf{a}_2$$

These are integer linear combinations of **both** the unrotated `{a₁, a₂}` and the rotated `{R(θ)a₁, R(θ)a₂}` bases. This ensures periodicity in both layers simultaneously.

**Number of unit cells per supercell:**

$$N_\text{cells} = m^2 + mn + n^2$$

**Twist angle:**

$$\cos\theta = \frac{m^2 + 4mn + n^2}{2(m^2 + mn + n^2)}$$

### 4.2 Known Issue: `commensurate_utils.build_supercell_vectors`

The function in `commensurate_utils.py` uses the **old** convention:

```python
L1 = m * a1 + n * a2    # ← NOT the coincidence vector
L2 = -n * a1 + (m+n) * a2
```

This is periodic for the unrotated layer but **not** the rotated layer. For `m ≈ n` (e.g., (30,29)), the error is small because `L1 ≈ C1`. For `m ≠ n`, this produces incorrect rod enumeration in the rotated layer.

**Use `build_coincidence_supercell()` from `eps_forensics.py` instead**, which implements the correct `C1 = n·a1 + m·a2`.

---

## 5. MPB-Specific Conventions

### 5.1 Lattice Setup

For a cell with physical lattice vectors `L₁`, `L₂`:

```python
lattice = mp.Lattice(
    size=mp.Vector3(1, 1, 0),           # Always (1, 1, 0) for 2D
    basis1=mp.Vector3(L1[0], L1[1], 0), # Raw (non-normalized) vectors
    basis2=mp.Vector3(L2[0], L2[1], 0),
)
```

The actual lattice vector used by MPB is `real_Lᵢ = sizeᵢ × basisᵢ`. With `size=(1,1)`, the basis vectors ARE the lattice vectors.

### 5.2 Radius

With `size=(1,1)` and raw basis vectors, MPB interprets the cylinder `radius` in units of `|basisᵢ|`. Therefore:

```python
radius_mpb = r_physical / |L1|
```

For a honeycomb monolayer (`|a₁| = a`, `size=(1,1)`):
```python
radius = r_over_a  # e.g. 0.2, since |basis1| = a
```

For a supercell (`|C₁| = √(N_cells) · a` approximately):
```python
radius = r_over_a * a / np.linalg.norm(C1)
```

### 5.3 Rod Centers

Rod centers are given in **fractional** coordinates within [-0.5, 0.5). Values outside this range are wrapped by MPB automatically.

**Unified convention (post-March-2026):** Specify rod centers in FDFD fractional coordinates [0, 1). MPB wraps them internally. Then apply `np.roll(eps, (N//2, N//2))` to the extracted grid. This makes the pattern identical for monolayer and supercell.

```python
# Monolayer: standard sublattice positions
center = mp.Vector3(0, 0, 0)       # origin → MPB center → rolled to pixel 0
center = mp.Vector3(1/3, 1/3, 0)   # sublattice B

# Supercell: fractional coords from B_coinc_inv @ pos_cart, wrapped to [0, 1)
center = mp.Vector3(f_wrapped[0], f_wrapped[1], 0)  # NO "-0.5" shift
```

### 5.4 Resolution

With `size=(1,1)`, the MPB grid is `resolution × resolution` pixels. To match an FDFD grid with `res` pixels per monolayer unit cell length:

```python
mpb_resolution = int(round(res * np.linalg.norm(C1)))
```

This ensures the same physical pixel density.

### 5.5 Frequency Units (CRITICAL)

MPB reports dimensionless frequencies $\tilde{\omega} = \omega a_{\text{cell}} / (2\pi c)$, where $a_{\text{cell}}$ is the length of the first lattice vector. With `size=(1,1)` and `basis1 = C1`, the effective lattice constant is $a_{\text{cell}} = |\mathbf{C}_1|$, **not** the monolayer lattice constant $a$.

To convert MPB supercell frequencies to the standard $\omega a / (2\pi c)$ units used by FDFD and the EA:

```python
freqs_ca = freqs_mpb / np.linalg.norm(C1)  # = freqs_mpb * a / |C1|
```

For a monolayer with `basis1 = a1` and `|a1| = a = 1`, the conversion factor is 1 (no rescaling needed).

**Verification (Phase A, (4,3) supercell):** MPB band 20 = 1.14453 → rescaled = 1.14453 / 6.083 = 0.18816. FDFD band 20 = 0.18813. Match to 4 decimal places.

### 5.6 Grid Extraction and Alignment

```python
eps = np.array(ms.get_epsilon(), dtype=np.float64)
if eps.ndim == 3:
    eps = eps[:, :, 0]  # 2D slice

# Convert from MPB center-origin to FDFD corner-origin
Nx, Ny = eps.shape
eps = np.roll(eps, (Nx // 2, Ny // 2), axis=(0, 1))
```

---

## 6. Subpixel Smoothing

### 6.1 Pixel Geometry

A pixel at fractional index `(i, j)` has:
- **Corner** (FDFD): `s₁ = i/N`, `s₂ = j/N`
- **Center** (smoothing sub-grid): `s₁ = (i + 0.5)/N`, `s₂ = (j + 0.5)/N`

### 6.2 Bilayer Overlap Rule

When multiple rods overlap at a pixel boundary, use `max()` to combine smoothed values:

```python
eps_smoothed[i, j] = max(eps_smoothed[i, j], eps_eff)
```

This prevents a neighboring rod's boundary smoothing from degrading a pixel that is fully inside another rod.

---

## 7. Quick Reference: Converting Between Conventions

| From → To | Operation |
|-----------|-----------|
| **FDFD → EA** | None (same convention) |
| **MPB → FDFD** | `np.roll(eps, (Nx//2, Ny//2), axis=(0,1))` |
| **FDFD frac → MPB center** | Pass directly; MPB wraps [0,1) to [-0.5,0.5) |
| **Physical → FDFD frac** | `f = B_inv @ r_physical`, then `f_wrapped = f - floor(f)` |

---

## 8. Validation Metrics (Reference Values)

Results from `eps_forensics.py` with (4,3) supercell at resolution 32 (= 196×196 grid):

### Monolayer (64×64)

| Comparison | Mean |Δε| | Max |Δε| | Pixels different |
|------------|---------|---------|------------------|
| Smoothed vs MPB | **0.0015** | 0.21 | 5.76% |
| Binary vs MPB | 0.110 | 4.95 | 5.76% |

### Bilayer Supercell (196×196)

| Comparison | Mean |Δε| | Max |Δε| | Pixels different |
|------------|---------|---------|------------------|
| Smoothed vs MPB | **0.159** | 5.41 | 15.93% |
| Binary vs MPB | 0.318 | 8.21 | 15.92% |

The higher supercell boundary fraction (~16% vs ~6%) reflects the denser rod packing in the bilayer. The harmonic mean ε (relevant for FDFD's 1/ε operator) matches closely: **Smoothed 1.982 vs MPB 1.976**.
