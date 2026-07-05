# EA Comparison Runs — Data Index

## Crystal

- **Lattice**: Square
- **Dielectric profile**: Rods in vacuum, $r/a = 0.2$, $\varepsilon_\text{rod} = 8.9$, $\varepsilon_\text{bg} = 1.0$
- **TM expansion point**: M = (0.5, 0.5) in fractional reciprocal coordinates
- **TE expansion point**: $\Gamma$ = (0, 0) in fractional reciprocal coordinates

## Benchmark Angles

| Label | Twist angle | Supercell (M,N) | η | L_m / a |
|-------|-------------|-----------------|---|---------|
| `8deg` | 8.17° | (14, 1) | 0.1425 | 7.02 |
| `3deg` | 3.01° | (38, 1) | 0.0525 | 19.04 |
| `1deg` | 1.005° | (114, 1) | 0.0175 | 57.01 |

## Phase 1 Parameters (common)

- Registry grid: 32 × 32 (1024 shift samples)
- Blaze FDFD resolution: 64 px/cell
- Born–Huang: computed
- Registry derivatives: computed

## Phase 2 Parameters (common)

- Moiré grid: Ns = 32
- FD order: 4 (9-point stencil)
- k-point: Γ only (k_s = 0)
- TM shift: $\sigma = 0.02$ in envelope units $\Delta\lambda$
- TE shift: `sigma_omega = 0.02` in direct-solver units $f$ [c/a], converted to absolute $\lambda = (2\pi f)^2 = 0.015791367...$ and then to the Phase 2 shift used by eigsh

---

## TM Runs — `ea_comparison_output/`

Single configuration: **4 retained bands, 4 remote bands**.

### Phase 1

| Directory | File | Polarization | N_ret | N_rem |
|-----------|------|-------------|-------|-------|
| `phase1/` | `square_tm_phase1.h5` | TM | 4 | 4 |

### Phase 2

| File | θ | Modes | Freq range (c/a) |
|------|---|-------|------------------|
| `ea_gamma_modes_8deg.npz` | 8.17° | 30 | 0.2886 – 0.3048 |
| `ea_gamma_modes_3deg.npz` | 3.01° | 50 | 0.2951 – 0.3002 |
| `ea_gamma_modes_1deg.npz` | 1.005° | 50 | 0.2975 – 0.2997 |

Full H5 outputs with eigenvectors are in `phase2_{8,3,1}deg/square_tm_phase2.h5`.

---

## TE Runs — `ea_comparison_output_te/`

Four types exploring the effect of retained/remote bands on the envelope spectrum.

### Phase 1 (3 configurations)

| Directory | Polarization | N_ret | N_rem | Slow coeff |
|-----------|-------------|-------|-------|------------|
| `phase1_1ret_0rem/` | TE | 1 | 0 | yes |
| `phase1_1ret_5rem/` | TE | 1 | 5 | yes |
| `phase1_4ret_0rem/` | TE | 4 | 0 | yes |

Each contains `square_gamma_te_phase1.h5` and `square_gamma_te_phase1.npz`.
These corrected TE runs are carrier-centered at $\Gamma$, not M.

### Phase 2 (4 types × 3 angles = 12 solves)

| Type | Phase 1 used | N_ret | Löwdin remote | What is solved |
|------|-------------|-------|---------------|----------------|
| **type1** `1ret_0rem_1band` | `1ret_0rem` | 1 | none | Lowest band only |
| **type2** `1ret_5rem_1band` | `1ret_5rem` | 1 | 5 bands | Lowest band only |
| **type3** `4ret_0rem_1band` | `4ret_0rem` | 4 | none | Lowest band only |
| **type4** `4ret_0rem_4band` | `4ret_0rem` | 4 | none | Lowest 4 bands |

### NPZ file naming

```
ea_te_{type}_{angle}.npz
```

Example: `ea_te_type2_1ret_5rem_1band_3deg.npz`

Each `.npz` contains:
- `eigenvalues` — raw EA eigenvalues (relative to λ_ref)
- `frequencies` — converted to physical frequency (c/a)
- `lambda_ref` — reference eigenvalue
- `theta_deg` — twist angle
- `Ns` — moiré grid size (32)
- `n_modes` — number of modes solved

### Full H5 outputs

```
phase2_{type}/{angle}/square_gamma_te_phase2.h5
```

Example: `phase2_type3_4ret_0rem_1band/3deg/square_gamma_te_phase2.h5`

These contain the full Hamiltonian, eigenvectors, and mode statistics.

### Results Summary

| Type | Description | 8.17° BW | 3.01° BW | 1.005° BW |
|------|-------------|----------|----------|-----------|
| 1 | 1 ret, 0 rem → 1 band | `nan` | 0.10860 | `nan` |
| 2 | 1 ret, 5 rem → 1 band | `nan` | `nan` | 0.03508 |
| 3 | 4 ret, 0 rem → 1 band | `nan` | `nan` | `nan` |
| 4 | 4 ret, 0 rem → 4 bands | `nan` | `nan` | `nan` |

Bandwidth = max(freq) − min(freq) over all solved modes at Γ after converting from $\lambda = \lambda_\mathrm{ref} + \Delta\lambda$ to $f = \sqrt{\lambda}/2\pi$. `nan` indicates at least one solved mode had negative total $\lambda$.

### Negative-Mode Check

- Total corrected TE runs inspected: 12
- Total negative-total-$\lambda$ modes: 35
- Total `nan` frequencies: 35
- Smallest negative values are tiny roundoff-scale drifts, e.g. $-1.9\times 10^{-8}$
- The larger negative values occur in the multiband TE runs, up to about $-5.29$

Interpretation: at $\Gamma$ for TE, the lowest retained local band has $\lambda_\mathrm{ref} = 0$, so the envelope solve is sitting directly on the positivity boundary. The EA Hamiltonian is Hermitian, but it is not constructed to be manifestly positive-semidefinite. Small negative modes near zero are therefore expected numerically; larger negative excursions in the multiband cases indicate that the low-frequency $\Gamma$ TE test is probing a regime where the truncated EA model is not positivity-preserving and may not be quantitatively reliable without additional constraints or a different formulation around the zero mode.

---

## Driver Scripts

| Script | What it runs |
|--------|-------------|
| `run_ea_comparison.py` | TM: 1 Phase 1 + 3 Phase 2 |
| `run_ea_comparison_te.py` | TE: 3 Phase 1 + 12 Phase 2 |
