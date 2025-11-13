1. **Global architecture + directory layout**
2. **Phase 0: Candidate search & scoring (optimization front-end)**
3. **Phase 1: Local Bloch problems at frozen registry**
4. **Phase 2: Envelope operator assembly**
5. **Phase 3: Envelope eigenproblem**
6. **Phase 4: EA vs full moiré validation**
7. **Phase 5: Meep cavity / Q-factor validation**
8. **Optimization loop + visualization strategy**

Throughout, assume Python >= 3.10, `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py` or `zarr`, `meep`/`mpb`, and possibly `optuna` for optimization.

---

## 1. Global architecture

### 1.1 Folder layout

```text
moire_envelope/
  common/
    geometry.py
    moire_utils.py
    mpb_utils.py
    ea_utils.py
    io_utils.py
    scoring.py
    plotting.py
  phases/
    phase0_candidate_search.py
    phase1_local_bloch.py
    phase2_ea_operator.py
    phase3_ea_solver.py
    phase4_validation.py
    phase5_meep_qfactor.py
  configs/
    base_config.yaml
    search_config_square.yaml
    search_config_hex.yaml
  runs/
    run_YYYYMMDD_HHMMSS/
      phase0_candidates.csv
      candidate_0001/
        phase0_meta.json
        phase1_band_data.h5
        phase2_operator.npz
        phase3_eigenpairs.h5
        phase4_validation.csv
        phase5_q.csv
  optimize_driver.py
  README.md
```

Each phase script:

* Reads **only** well-defined input files.
* Writes outputs in a candidate-specific folder.
* Can be run standalone on a subset of candidates.

### 1.2 Core parameter space (for Phase 0)

Global search variables:

* Twist angle: (\theta \in [0.5^\circ, 3^\circ])
* Lattice type: `square`, `hex`, `rect`
* Hole radius: (r/a) in some range (e.g. 0.15–0.45)
* Background ε: (\varepsilon_\text{bg} \in [2, 12]) (for example)

Fixed assumptions:

* Air holes: (\varepsilon_\text{hole} = 1)
* 2D slab or 2D TE/TM approximation (depending on MPB configuration).

**High-sym k-points** per lattice:

* `square`: Γ, X, M
* `hex`: Γ, M, K
* `rect`: Γ, X, Y, M

Target band index: say `n0` (e.g. 3–10 depending on pattern) – configurable.

For each combination of `(lattice_type, theta, r, eps_bg, k_point_label, band_index)` we can define a **candidate**.

---

## 2. Phase 0 – Candidate search & scoring

Goal: Explore design space and output **K best candidates** to feed into EA pipeline.

### 2.1 Physics / math in Phase 0

For *untwisted* or reference bilayer cell:

1. Define transversely periodic 2D PC with lattice type, radius, ε_bg.

2. Solve band structure along full high-symmetry path.

3. For each high-symmetry point (k_0) on that path and band (n_0):

   * Extract frequency (\omega_{0}(k_0)).
   * Approximate curvature and group velocity around (k_0) by finite differences.
   * Compute local **parabolic validity radius** (k_\text{parab}) where quadratic fit stays within tolerance (e.g. 1–2 %).
   * Compute **spectral gaps** above and below:
     (\Delta_+ = \min_{\text{bands>n0}} (\omega_{n}-\omega_{n_0})) at that k,
     (\Delta_- = \min_{\text{bands<n0}} (\omega_{n_0}-\omega_{n})).

4. For each candidate, estimate **moiré reciprocal magnitude**

   [
   |G_m| \approx \frac{4\pi}{\sqrt{3}a} \sin(\theta/2) \approx \frac{2\pi}{a}\theta
   ]

   and impose validity condition

   [
   |G_m| \le \alpha k_\text{parab},
   ]

   with (\alpha\sim 0.3)–0.5 (configurable).

Candidates that fail this are thrown out or heavily penalized.

### 2.2 Scoring metrics

For each candidate (c):

1. **Band flatness / effective mass**:

   * For isotropic approximation in 2D:
     [
     \kappa = \tfrac12\mathrm{Tr},\partial_{k_i}\partial_{k_j}\omega(k_0)
     ]
   * Define flatness score (S_\text{flat} = 1 / (1 + \kappa/\kappa_0))
     (κ₀ = reference curvature, say curvature of a typical band; smaller κ → flatter → higher score).

2. **Spectral isolation**:

   * ( \Delta = \min(\Delta_+, \Delta_-)).
   * Normalize: (S_\text{gap} = \Delta / (\Delta + \Delta_0)).

3. **Parabolic validity**:

   * (S_\text{parab} = \min\left(1, k_\text{parab} / (\beta |G_m|)\right))
     where β>1 enforces a safety margin.

4. **Group velocity** at k₀:

   * (v_g = |\nabla_k\omega(k_0)|).
   * (S_\text{vg} = 1/(1 + v_g/v_0)).

5. Optional: **Dielectric contrast** weight:

   * (S_\text{contrast} = (\varepsilon_\text{bg}-1)/(\varepsilon_\text{bg,max}-1)).

Total score:

[
S_\text{tot} = w_\text{flat}S_\text{flat}

* w_\text{gap}S_\text{gap}
* w_\text{parab}S_\text{parab}
* w_\text{vg}S_\text{vg}
* w_\text{contrast}S_\text{contrast}
  ]

with weights satisfying (\sum w_i = 1). For cavity-friendly band edges, one natural choice:

* `w_flat=0.35, w_gap=0.25, w_parab=0.2, w_vg=0.15, w_contrast=0.05`.

### 2.3 Data structures and outputs

**Output file:** `phase0_candidates.csv`

Columns (for each candidate row):

* `candidate_id` (int)
* `lattice_type` (`square|hex|rect`)
* `theta_deg`
* `a` (lattice constant)
* `r_over_a`
* `eps_bg`
* `band_index`
* `k_label` (`Γ`, `M`, etc.)
* `k0_x`, `k0_y` (cartesian, units of 2π/a)
* `omega0`
* `curvature_xx`, `curvature_xy`, `curvature_yy`
* `curvature_trace`, `curvature_det`
* `vg_x`, `vg_y`, `vg_norm`
* `k_parab`
* `gap_above`, `gap_below`, `gap_min`
* `G_magnitude`
* `S_flat`, `S_gap`, `S_parab`, `S_vg`, `S_contrast`, `S_total`
* `valid_ea_flag` (bool)

**Optional**: per-candidate JSON with all MPB config.

### 2.4 Skeleton code-style for Phase 0

```python
# phases/phase0_candidate_search.py
import numpy as np
import pandas as pd
from common.geometry import build_lattice
from common.mpb_utils import compute_bandstructure, fit_local_dispersion
from common.scoring import score_candidate
from common.io_utils import ensure_run_dir

def run_phase0(config_path: str):
    config = load_yaml(config_path)
    run_dir = ensure_run_dir(config)

    rows = []
    for lattice_type in config["lattice_types"]:
        for theta in np.linspace(0.5, 3.0, config["n_theta"]):
            for r_over_a in config["r_over_a_list"]:
                for eps_bg in config["eps_bg_list"]:
                    geom = build_lattice(lattice_type, r_over_a, eps_bg)
                    bands = compute_bandstructure(geom, config)

                    for k_label, k_vec in high_symmetry_points(lattice_type):
                        for band_index in config["target_bands"]:
                            metrics = fit_local_dispersion(bands, k_label, band_index)
                            row = assemble_candidate_row(
                                lattice_type, theta, r_over_a, eps_bg,
                                band_index, k_label, k_vec, metrics
                            )
                            row.update(score_candidate(row, config))
                            rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values("S_total", ascending=False, inplace=True)
    df.to_csv(run_dir / "phase0_candidates.csv", index=False)

if __name__ == "__main__":
    import sys
    run_phase0(sys.argv[1])
```

---

## 3. Phase 1 – Local Bloch problems at frozen registry

Now Phase 1 no longer sweeps everything; it takes a **selected list of candidates**.

### 3.1 Inputs

* `phase0_candidates.csv`
* `config` specifying:

  * `K_candidates` (top K to process)
  * `R_grid` resolution (Nx, Ny)
  * `delta_grid` resolution (usually same as R)
* Monolayer lattice vectors `a1`, `a2`.
* For each candidate: `theta`, `tau` (stacking gauge), `eta`.

### 3.2 Geometry & registry map

Use the `compute_registry_map` function (very good starting point):

```python
# common/moire_utils.py
def compute_registry_map(R_grid, a1, a2, theta, tau, eta):
    """
    Compute the local registry map δ(R).
    R_grid: array [Nx, Ny, 2]
    Returns delta_grid: [Nx, Ny, 2] fractional shifts
    """
    Nx, Ny, _ = R_grid.shape
    delta_grid = np.zeros((Nx, Ny, 2))

    R_rot = rotation_matrix_2d(theta)
    I = np.eye(2)
    lattice_mat = np.column_stack([a1[:2], a2[:2]])

    for i in range(Nx):
        for j in range(Ny):
            R_vec = R_grid[i, j, :]
            delta_physical = (R_rot - I) @ R_vec / eta + tau
            delta_frac = fractional_coordinates(delta_physical, lattice_mat)
            delta_grid[i, j, :] = delta_frac

    return delta_grid
```

At each R (i.e. local δ) the bilayer geometry is “frozen” with that stacking shift.

### 3.3 Physics / math

For each candidate and each registry point (\mathbf R):

1. Build MPB geometry for bilayer with local shift δ(R).
2. Compute band (n_0) at k₀ and a few nearby k-points for derivs:

   * k₀ ± Δk eₓ, k₀ ± Δk e_y.
3. Extract:

   * `omega0(R)` = ω₀(k₀; R).
   * `v_g(R)` from central finite difference in k.
   * Hessian (curvature tensor) → (M^{-1}(R)).

The math:

[
v_i(R) \approx \frac{\omega(k_0 + \Delta k,e_i; R) - \omega(k_0 - \Delta k,e_i; R)}{2\Delta k}
]

[
\partial_{k_i}\partial_{k_j}\omega(k_0;R)
\approx \frac{
\omega(k_0+\Delta k e_i+\Delta k e_j)
-\omega(k_0+\Delta k e_i-\Delta k e_j)
-\omega(k_0-\Delta k e_i+\Delta k e_j)
+\omega(k_0-\Delta k e_i-\Delta k e_j)
}{4(\Delta k)^2}
]

Then define:

[
M^{-1}*{ij}(R) = \partial*{k_i}\partial_{k_j}\omega_0(k_0;R).
]

Also define potential:

[
V(R) = \omega_0(k_0;R) - \omega_0^{\text{ref}}
]

where (\omega_0^{\text{ref}}) can be the average over R or the minimum (configurable).

### 3.4 Outputs

For each candidate `candidate_XXXX`:

* `phase1_band_data.h5` (HDF5 or Zarr)

  * `R_grid` : [Nx, Ny, 2], slow coordinates
  * `delta_grid`: [Nx, Ny, 2]
  * `omega0`: [Nx, Ny]
  * `vg`: [Nx, Ny, 2]
  * `M_inv`: [Nx, Ny, 2, 2]
  * `V`: [Nx, Ny]
  * `omega_ref`: scalar
  * `eta`, `theta`, `k0`, `band_index`

* `phase1_band_visualization.png`:

  * heatmaps of V(R), |v_g(R)|, eigenvalues of M^{-1}(R) over moiré cell.

* `phase1_reference_band.csv`:

  * 1D band structure of reference registry (e.g. δ=τ) around k₀ for validation.

### 3.5 Skeleton Phase 1 script

```python
# phases/phase1_local_bloch.py
import h5py
import numpy as np
import pandas as pd
from common.moire_utils import build_R_grid, compute_registry_map
from common.mpb_utils import compute_local_band_data
from common.io_utils import candidate_dir

def run_phase1(run_dir, config_path):
    config = load_yaml(config_path)
    candidates = pd.read_csv(run_dir / "phase0_candidates.csv")
    top = candidates.head(config["K_candidates"])

    for _, row in top.iterrows():
        cid = int(row["candidate_id"])
        cdir = candidate_dir(run_dir, cid)
        cdir.mkdir(parents=True, exist_ok=True)

        R_grid = build_R_grid(config["Nx"], config["Ny"], row, config)
        delta_grid = compute_registry_map(
            R_grid, a1_from_row(row), a2_from_row(row),
            np.deg2rad(row["theta_deg"]), config["tau"], row["eta"]
        )

        omega0, vg, M_inv = compute_local_band_data(
            R_grid, delta_grid, row, config
        )

        omega_ref = choose_reference_frequency(omega0, config)
        V = omega0 - omega_ref

        with h5py.File(cdir / "phase1_band_data.h5", "w") as hf:
            hf.create_dataset("R_grid", data=R_grid)
            hf.create_dataset("delta_grid", data=delta_grid)
            hf.create_dataset("omega0", data=omega0)
            hf.create_dataset("vg", data=vg)
            hf.create_dataset("M_inv", data=M_inv)
            hf.create_dataset("V", data=V)
            hf.attrs["omega_ref"] = omega_ref
            hf.attrs["eta"] = row["eta"]
            hf.attrs["theta_deg"] = row["theta_deg"]
            hf.attrs["band_index"] = row["band_index"]

        make_phase1_plots(cdir, R_grid, V, vg, M_inv)

if __name__ == "__main__":
    import sys
    run_phase1(Path(sys.argv[1]), sys.argv[2])
```

---

## 4. Phase 2 – Envelope operator assembly

Now the EA Hamiltonian is constructed on the R-grid.

### 4.1 Physics / math

For band extrema where v_g ≈ 0:

[
H_\text{EA} = -\frac{\eta^2}{2}\nabla_R\cdot M^{-1}(R)\nabla_R + V(R).
]

Discretize on a rectangular grid with lattice vectors (A_1, A_2) for the moiré cell.

Use finite differences with variable coefficients:

For each grid point (i,j):

[
(H F)*{i,j} = -\frac{\eta^2}{2}\sum*{\alpha=x,y}
\frac{1}{\Delta R_\alpha}
\left[
M^{-1}*{\alpha\beta}(R*{i+1/2,j}) \frac{F_{i+1,j}-F_{i,j}}{\Delta R_\beta}
--------------------------------------------------------------------------

M^{-1}*{\alpha\beta}(R*{i-1/2,j}) \frac{F_{i,j}-F_{i-1,j}}{\Delta R_\beta}
\right]

* V_{i,j} F_{i,j}
  ]

plus similar stencil in y. This yields a sparse Hermitian matrix.

Periodic BCs: indices wrap modulo Nx, Ny.

In practice: treat the operator as a 2D generalization of variable-mass Schrödinger Hamiltonian.

### 4.2 Inputs

For each candidate:

* `phase1_band_data.h5` (R_grid, M_inv, V, eta).
* Optional: whether to include v_g term (for non-extremal k₀).

### 4.3 Outputs

Per candidate:

* `phase2_operator.npz`: SciPy sparse matrix in CSR format

  * arrays `data`, `indices`, `indptr`, plus shape.
* `phase2_operator_info.csv`:

  * `candidate_id`
  * `Nx`, `Ny`
  * `nnz`
  * `min_V`, `max_V`, `mean_V`
  * distribution of eigenvalues of M^{-1} (min, max, average).
* `phase2_fields_visualization.png`:

  * V(R) map
  * eigenvalues of M^{-1} (heatmaps)

### 4.4 Skeleton code

```python
# phases/phase2_ea_operator.py
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, save_npz
from common.io_utils import candidate_dir

def assemble_ea_operator(R_grid, M_inv, V, eta):
    Nx, Ny, _ = R_grid.shape
    N = Nx * Ny
    H = lil_matrix((N, N), dtype=np.float64)

    dx = R_grid[1,0,0] - R_grid[0,0,0]
    dy = R_grid[0,1,1] - R_grid[0,0,1]

    def idx(i, j):
        return (i % Nx) * Ny + (j % Ny)

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            # On-site potential
            H[p, p] += V[i, j]

            # Kinetic terms (simplified isotropic case as starting point)
            # Extension to full tensor M_inv can be implemented later.

            # x-direction
            mx = 0.5 * (M_inv[i, j, 0, 0] + M_inv[(i+1) % Nx, j, 0, 0])
            coeff_x = -0.5 * eta**2 * mx / dx**2
            H[p, idx(i+1, j)] += coeff_x
            H[p, p] += -coeff_x  # symmetric counterpart

            mx = 0.5 * (M_inv[i, j, 0, 0] + M_inv[(i-1) % Nx, j, 0, 0])
            coeff_x = -0.5 * eta**2 * mx / dx**2
            H[p, idx(i-1, j)] += coeff_x
            H[p, p] += -coeff_x

            # y-direction (analogous)
            my = 0.5 * (M_inv[i, j, 1, 1] + M_inv[i, (j+1) % Ny, 1, 1])
            coeff_y = -0.5 * eta**2 * my / dy**2
            H[p, idx(i, j+1)] += coeff_y
            H[p, p] += -coeff_y

            my = 0.5 * (M_inv[i, j, 1, 1] + M_inv[i, (j-1) % Ny, 1, 1])
            coeff_y = -0.5 * eta**2 * my / dy**2
            H[p, idx(i, j-1)] += coeff_y
            H[p, p] += -coeff_y

    return H.tocsr()

def run_phase2(run_dir, config_path):
    config = load_yaml(config_path)
    candidates = pd.read_csv(run_dir / "phase0_candidates.csv")
    top = candidates.head(config["K_candidates"])

    for _, row in top.iterrows():
        cid = int(row["candidate_id"])
        cdir = candidate_dir(run_dir, cid)
        with h5py.File(cdir / "phase1_band_data.h5", "r") as hf:
            R_grid = hf["R_grid"][:]
            V = hf["V"][:]
            M_inv = hf["M_inv"][:]
            eta = hf.attrs["eta"]

        H = assemble_ea_operator(R_grid, M_inv, V, eta)
        save_npz(cdir / "phase2_operator.npz", H)

        write_phase2_info(cdir, H, V, M_inv)

if __name__ == "__main__":
    import sys
    run_phase2(Path(sys.argv[1]), sys.argv[2])
```

---

## 5. Phase 3 – Envelope eigenvalue solver

### 5.1 Physics / math

Solve:

[
H_\text{EA} F_n = \Delta\omega_n F_n
]

for lowest few eigenvalues.

Use `scipy.sparse.linalg.eigsh` for hermitian matrices.

Cavity frequencies:

[
\omega_n^\text{cavity} = \omega_\text{ref} + \Delta\omega_n.
]

### 5.2 Inputs

* For each candidate:

  * `phase2_operator.npz`
  * `phase1_band_data.h5` (for `omega_ref`, R_grid shape).

### 5.3 Outputs

Per candidate:

* `phase3_eigenvalues.csv`:

  * `candidate_id`, `n`, `Delta_omega`, `omega_cav`, `norm`, localization metrics:

    * participation ratio, entropy, etc.

* `phase3_eigenstates.h5`:

  * `F`: [N_modes, Nx, Ny] real fields
  * `R_grid`, `omega_ref`.

* `phase3_cavity_modes.png`:

  * Contour plots of |F_n(R)|² for lowest modes.

* `phase3_spectrum.png`:

  * Eigenvalues near bottom (Δω_n) vs index.

### 5.4 Skeleton

```python
# phases/phase3_ea_solver.py
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh
from common.io_utils import candidate_dir
from common.plotting import plot_envelope_modes

def run_phase3(run_dir, config_path):
    config = load_yaml(config_path)
    candidates = pd.read_csv(run_dir / "phase0_candidates.csv")
    top = candidates.head(config["K_candidates"])

    for _, row in top.iterrows():
        cid = int(row["candidate_id"])
        cdir = candidate_dir(run_dir, cid)

        H = load_npz(cdir / "phase2_operator.npz")

        with h5py.File(cdir / "phase1_band_data.h5", "r") as hf:
            R_grid = hf["R_grid"][:]
            omega_ref = hf.attrs["omega_ref"]

        n_modes = config["ea_n_modes"]
        # Smallest eigenvalues
        vals, vecs = eigsh(H, k=n_modes, which="SA")

        Nx, Ny, _ = R_grid.shape
        F = vecs.T.reshape((n_modes, Nx, Ny))

        # Store
        with h5py.File(cdir / "phase3_eigenstates.h5", "w") as hf:
            hf.create_dataset("F", data=F)
            hf.create_dataset("R_grid", data=R_grid)
            hf.attrs["omega_ref"] = omega_ref
            hf.attrs["Delta_omega"] = vals

        rows = []
        for n, dE in enumerate(vals):
            omega_cav = omega_ref + dE
            pr = participation_ratio(F[n])
            rows.append({
                "candidate_id": cid,
                "mode_index": n,
                "Delta_omega": float(dE),
                "omega_cavity": float(omega_cav),
                "participation_ratio": float(pr),
            })
        pd.DataFrame(rows).to_csv(cdir / "phase3_eigenvalues.csv", index=False)

        plot_envelope_modes(cdir, R_grid, F, vals)

if __name__ == "__main__":
    import sys
    run_phase3(Path(sys.argv[1]), sys.argv[2])
```

---

## 6. Phase 4 – Validation vs full moiré band structure

### 6.1 Purpose

Check whether EA is quantitatively accurate for a subset of promising candidates:

1. Build full moiré supercell geometry for selected θ and lattice.

2. Use MPB to compute:

   * Bloch bands along a small k-path in the moiré Brillouin zone near Γ.
   * Possibly localized defect/cavity modes in that supercell.

3. Compare:

   * EA minibands vs full bands (Δω vs k).
   * Envelopes |F(R)|² vs coarse-grained field intensity of full modes.

### 6.2 Inputs

Per candidate:

* `phase3_eigenvalues.csv`
* `phase3_eigenstates.h5`
* `phase1_band_data.h5`
* Phase 0 geometry params (θ, r, eps_bg, lattice type).

### 6.3 Outputs

Per candidate:

* `phase4_validation_report.csv`:

  For each compared mode:

  * `mode_index`, `omega_EA`, `omega_full`, `rel_error`,
    `overlap_envelope`, `k_error`, etc.

* `phase4_comparison_plot.png`:

  * EA spectra vs full spectra on same plot.

* `phase4_error_analysis.csv`:

  * summary metrics: RMS error, max error, etc.

### 6.4 Skeleton outline

```python
# phases/phase4_validation.py
import pandas as pd
from common.mpb_utils import compute_moire_bandstructure
from common.ea_utils import interpolate_ea_bands
from common.io_utils import candidate_dir

def run_phase4(run_dir, config_path):
    config = load_yaml(config_path)
    candidates = pd.read_csv(run_dir / "phase0_candidates.csv")
    top = candidates.head(config["K_validate"])

    for _, row in top.iterrows():
        cid = int(row["candidate_id"])
        cdir = candidate_dir(run_dir, cid)

        ea_eigs = pd.read_csv(cdir / "phase3_eigenvalues.csv")
        # For band comparison: treat EA as k≈0; optional extension to K≠0.

        moire_bands = compute_moire_bandstructure(row, config)
        comparison = compare_ea_full(ea_eigs, moire_bands, config)
        comparison.to_csv(cdir / "phase4_validation_report.csv", index=False)

        plot_phase4_comparison(cdir, ea_eigs, moire_bands, comparison)

if __name__ == "__main__":
    import sys
    run_phase4(Path(sys.argv[1]), sys.argv[2])
```

The actual comparison can be as simple as matching lowest few frequencies at Γ and computing `rel_error = |ω_full − ω_EA| / ω_full`.

---

## 7. Phase 5 – Meep Q-factor analysis

### 7.1 Physics

Full Maxwell TD simulation in Meep of the best cavity candidate:

1. Build 3D or 2D Yee grid of moiré cavity geometry.

2. Place Gaussian or dipole source near EA predicted frequency.

3. Run simulation, record fields.

4. Use Harminv to extract resonant frequencies and decay rates:

   [
   Q = \pi f / \gamma.
   ]

5. Compare with EA frequency prediction.

### 7.2 Inputs

Per candidate:

* Best cavity mode index from Phase 3 and frequency `omega_cavity`.
* Envelope peak location (R_\text{peak}) → map to physical coordinate for source placement.

### 7.3 Outputs

* `phase5_q_factor_results.csv`:

  * `candidate_id`, `mode_index`, `omega_EA`, `omega_meep`, `Q`, `rel_freq_error`.

* `phase5_field_animation.gif` and field snapshot PNGs.

* `phase5_harminv_spectrum.png`.

### 7.4 Skeleton outline

```python
# phases/phase5_meep_qfactor.py
import pandas as pd
from common.meep_utils import run_meep_cavity_sim
from common.io_utils import candidate_dir

def run_phase5(run_dir, config_path):
    config = load_yaml(config_path)
    candidates = pd.read_csv(run_dir / "phase0_candidates.csv")
    top = candidates.head(config["K_meep"])

    rows = []
    for _, row in top.iterrows():
        cid = int(row["candidate_id"])
        cdir = candidate_dir(run_dir, cid)

        eigs = pd.read_csv(cdir / "phase3_eigenvalues.csv")
        best = select_best_mode(eigs, config)  # e.g. lowest PR or smallest Δω

        sim_result = run_meep_cavity_sim(row, best, cdir, config)

        rows.append({
            "candidate_id": cid,
            "mode_index": int(best["mode_index"]),
            "omega_EA": float(best["omega_cavity"]),
            "omega_meep": float(sim_result.omega),
            "Q": float(sim_result.Q),
            "rel_freq_error": abs(sim_result.omega - best["omega_cavity"]) / sim_result.omega,
        })

    pd.DataFrame(rows).to_csv(run_dir / "phase5_q_factor_results.csv", index=False)

if __name__ == "__main__":
    import sys
    run_phase5(Path(sys.argv[1]), sys.argv[2])
```

---

## 8. Optimization loop & visualization

### 8.1 Optimization driver

`optimize_driver.py` is the main entry point. For a given configuration:

1. Run Phase 0 to generate candidate table.
2. Choose top K₀ candidates.
3. For each candidate, run Phases 1–3.
4. Compute **EA-level score** that combines:

   * Original Phase-0 score.
   * Envelope localization metrics (participation ratio, depth of V, etc.).
5. Optionally run Phases 4–5 on top K₁ candidates.
6. Rank everything by a composite **global score**:

   [
   S_\text{global} =
   \alpha S_\text{Phase0} +
   \beta S_\text{EA} +
   \gamma S_\text{Q}
   ]

   where `S_Q` can be a function of Q and freq errors, if available.

This driver can either:

* Use a **grid search**, or
* Wrap Phase 0 scoring in a `def objective(params)` and run e.g. `optuna`.

### 8.2 Visualization strategy

For each **run**:

* Global dashboard notebook or script that reads:

  * `phase0_candidates.csv` → scatter plots of S_total vs θ, vs r, vs ε_bg.
  * From candidate folders → heatmaps of V(R), F(R), etc.
  * `phase5_q_factor_results.csv` → Q vs design parameters.

* Per phase:

  * Phase 0:

    * `phase0_summary.png`: panel showing candidate scores vs θ, etc.

  * Phase 1:

    * For each candidate, 3×1 panels: V(R), |v_g(R)|, eigenvalues of M^{-1}(R).

  * Phase 3:

    * Cavity envelopes, color-coded by frequency, annotated with score.

  * Phase 4:

    * Two curves: EA vs full moiré bands; display relative error text.

---

## 9. Packages / tooling summary

* **Core numerics:** `numpy`, `scipy`, `pandas`, `h5py` or `zarr`, `matplotlib`.
* **Bandstructure:** `meep.mpb` (Python interface to MPB).
* **Full-wave sim:** `meep` (Python).
* **Optimization (optional):** `optuna` or `scikit-optimize`.
* **Parallelization:** `mpi4py` or `joblib` to distribute candidates across cores/nodes.