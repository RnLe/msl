# Envelope Approximation Pipeline for Moiré Photonic Crystals

## Overview

This pipeline implements an envelope approximation (EA) method for computing localized cavity modes in twisted bilayer photonic crystals. The approach treats the slowly-varying moiré superlattice potential as a perturbation to local photonic crystal band structures.

**Status:** ✅ **All 4 phases complete!**

## Pipeline Architecture

### Phase 0: Moiré Lattice Setup
**File:** `phase0_lattice_setup.py`

Creates the moiré superlattice from twisted bilayer configuration:
- Computes moiré vectors: **A₁**, **A₂** from reciprocal space difference
- Calculates registry map: δ(**R**) = stacking shift across moiré cell
- Generates visualization of overlapped lattices and stacking shift

**Key Parameters:**
- Lattice type: Square
- Twist angle: θ = 1.1° (adjustable 0.5-2°)
- Moiré period: L ≈ 52.09a
- Grid resolution: 64×64 (subsampled to 8×8 for calculations)

**Outputs:**
- `phase0_parameters.csv` - Moiré parameters (L, η, vectors)
- `phase0_lattice_overlapped.png` - Bilayer visualization
- `phase0_moire_lattice.png` - Moiré cell
- `phase0_stacking_shift_detailed.png` - Registry map δ(**R**)

---

### Phase 1: Local Bloch Problems
**File:** `phase1_local_bloch.py`

Solves photonic band structure at frozen registries using MPB:
- Computes local frequencies: ω₀(**R**)
- Calculates effective mass tensor: **M**⁻¹(**R**)
- Evaluates group velocity: **v**ᵍ(**R**) via numerical differentiation

**Key Parameters:**
- Polarization: TM
- Cylinder radius: r = 0.48a
- Dielectric: ε = 4.64 (cylinders), ε = 1.0 (background)
- Target: M point, band 1 (index 0)
- MPB resolution: 32

**Outputs:**
- `phase1_data.pkl` - Pickled field data (ω₀, **M**⁻¹, **v**ᵍ on 8×8 grid)
- `phase1_band_data.csv` - Band structure at all points
- `phase1_reference.csv` - Reference frequency ω_ref = 0.1048
- `phase1_band_visualization.png` - Spatial variation plots

**Notable Finding:** |**v**ᵍ| ≈ 3.2 >> 0, indicating M point is not at band extremum for this geometry. This necessitates including a drift term in the EA operator.

---

### Phase 2: Envelope Operator Assembly
**File:** `phase2_ea_operator.py`

Constructs the envelope approximation Hamiltonian:

```
H_EA = T + D + V
```

Where:
- **T**: Kinetic term = -(η²/2)∇·**M**⁻¹(**R**)·∇ (variable-coefficient Laplacian)
- **D**: Drift term = -η·**v**ᵍ(**R**)·∇ (first-order, needed due to large |**v**ᵍ|)
- **V**: Potential term = ω₀(**R**) - ω_ref (diagonal, local frequency variation)

**Numerical Details:**
- Grid: 8×8 = 64 points
- Boundary conditions: Periodic (moiré cell)
- Stencils: 5-point (Laplacian), 9-point (variable-coefficient)
- Sparsity: 576 non-zeros / 4096 elements (14% density)

**Outputs:**
- `phase2_operator.npz` - Sparse Hamiltonian matrix (4.5K)
- `phase2_operator_info.csv` - Matrix statistics
- `phase2_fields_visualization.png` - V(**R**), |**v**ᵍ|, **M**⁻¹ components

**Operator Properties:**
- Size: 64×64
- η = 0.0192
- V range: [-0.012, 0]
- σ(V) = 0.0022 (2.1% of ω_ref)

---

### Phase 3: Envelope Eigenvalue Solver
**File:** `phase3_envelope_solver.py`

Solves the eigenvalue problem:

```
H_EA · F(**R**) = ΔE · F(**R**)
```

to find envelope wavefunctions **F**ₙ(**R**) and energy detunings ΔE from reference frequency.

**Cavity Frequencies:** ω_cavity = ω_ref + ΔE

**Method:**
- Solver: `scipy.sparse.linalg.eigs` (non-Hermitian due to drift)
- Modes computed: 10 lowest energy states
- Shift-invert: Enabled for interior eigenvalues

**Outputs:**
- `phase3_eigenstates.npz` - Full eigenvector data (6.5K)
- `phase3_eigenvalues.csv` - ΔE and ω_cavity for each mode
- `phase3_mode_analysis.csv` - IPR, localization, peak positions
- `phase3_cavity_modes.png` - Spatial profiles |F(**R**)|
- `phase3_spectrum.png` - Energy spectrum

**Results:**
- **10 modes found**
- **Frequency range:** ω ∈ [0.096054, 0.097237]
- **All modes below ω_ref** (negative detunings)
- **Inverse participation ratio:** IPR ≈ 0.028-0.042 (moderately localized)
- **Imaginary parts:** max ~5×10⁻⁵ (negligible, from drift term)

---

### Phase 4: Validation and Analysis
**File:** `phase4_validation.py`

Validates EA results and analyzes approximation accuracy:

**Analyses Performed:**
1. **Local Variation Metrics**
   - Potential variation: σ(V)/ω_ref = 2.1%
   - Group velocity: ⟨|**v**ᵍ|⟩ = 2.93
   - EA parameter: η·ξ₀ = 2.55

2. **Perturbation Theory Comparison**
   - Compare full EA eigenvalues with 1st-order PT: ΔE⁽¹⁾ = ⟨F|V|F⟩
   - Mean absolute error: 0.00085
   - RMS error: 0.00088
   - Relative error: ~10.6%

3. **Mode Character Analysis**
   - Localization (IPR)
   - Node counting
   - Symmetry breaking

4. **Spectral Gap Analysis**
   - Largest gap: 0.000391 at ω ≈ 0.0964
   - Mean mode spacing: 0.000131

**Outputs:**
- `phase4_validation_summary.csv` - All metrics (816 bytes)
- `phase4_perturbation_theory_comparison.csv` - Mode-by-mode PT vs EA
- `phase4_mode_character.csv` - Localization properties
- `phase4_convergence_info.csv` - Grid resolution data
- `phase4_validation_comprehensive.png` - Multi-panel visualization (197K)

---

## Key Findings

### ✅ Successes

1. **Pipeline Functional:** All phases execute successfully and produce consistent results
2. **Perturbation Theory Agreement:** 90% accuracy vs 1st-order PT indicates EA captures dominant physics
3. **Slow Variation Confirmed:** σ(V)/ω_ref = 2.1% << 1 validates envelope assumption
4. **Localized Modes Found:** IPR ≈ 0.03-0.04 indicates non-trivial spatial structure

### ⚠️ Limitations

1. **Marginal EA Regime:** η·ξ₀ = 2.55 > 1
   - **Interpretation:** Modes are relatively extended (ξ₀ ≈ 21 vs L ≈ 52)
   - **Ideal EA requires:** ξ << L, i.e., η·ξ << 1
   - **Current state:** ξ ≈ 0.4L (marginal, not ideal)

2. **Not at Band Extremum:** |**v**ᵍ| ≈ 3.2 >> 0
   - **Implication:** M point not optimal for this geometry (r=0.48, ε=4.64)
   - **Solution:** Drift term included, but increases complexity
   - **Future:** Could search for better k-point or geometry parameters

3. **Coarse Grid:** 8×8 resolution
   - **Tradeoff:** Computational cost vs accuracy
   - **Impact:** Modes may have discretization artifacts
   - **Recommendation:** Convergence study at higher resolution (16×16)

### 🎯 Physical Interpretation

The envelope approximation **qualitatively captures** the physics of moiré cavity modes:
- Modes localize to favorable stacking regions
- Energy spectrum shows splitting due to moiré potential
- Agreement with PT indicates potential V(**R**) is dominant effect

However, the **marginal validity regime** (η·ξ ≈ 2.5) suggests:
- Results should be treated as **semi-quantitative**
- Kinetic energy (∇² term) is non-negligible
- Full numerical simulation (Meep) would provide validation

---

## Usage

### Run Entire Pipeline

```bash
# Phase 0: Lattice setup
mamba run -n msl python phase0_lattice_setup.py

# Phase 1: MPB band calculations (~10 min)
mamba run -n msl python phase1_local_bloch.py

# Phase 2: Operator assembly
mamba run -n msl python phase2_ea_operator.py

# Phase 3: Eigenvalue solver
mamba run -n msl python phase3_envelope_solver.py

# Phase 4: Validation
mamba run -n msl python phase4_validation.py
```

### View Results

All outputs saved in `outputs/` directory:
- CSV files: Numerical data, importable to analysis tools
- PNG files: Visualizations at publication quality (150 DPI)
- NPZ/PKL files: Binary data for further processing

Key result files:
- `phase3_cavity_modes.png` - Main result: envelope mode profiles
- `phase4_validation_comprehensive.png` - Validation summary
- `phase3_eigenvalues.csv` - Cavity frequencies

---

## Parameter Tuning

### Geometry Parameters
Edit in `phase1_local_bloch.py`:
```python
GeometryConfig(
    eps_hi=4.64,        # Dielectric constant
    eps_lo=1.0,
    cylinder_radius=0.48,  # r/a ratio
    polarization='TM'   # or 'TE'
)
```

### Twist Angle
Edit in `phase0_lattice_setup.py`:
```python
LatticeConfig(
    twist_angle_deg=1.1,  # Valid: 0.5-2.0°
    ...
)
```

**Note:** Changing parameters requires re-running all phases.

---

## Dependencies

- **Python:** 3.8+
- **Core:** numpy, scipy, matplotlib, pandas
- **Photonics:** meep, mpb (MIT Photonic Bands)
- **Rust bindings:** moire_lattice_py (custom)
- **Environment:** msl (mamba environment)

Install:
```bash
mamba env create -f ../environment.yml
mamba activate msl
```

---

## Theory Reference

See `EA_Moire_Photonics.md` for mathematical derivation and theory.

**Key Equation:**
```
[-η²/2 ∇·M⁻¹(R)·∇ - η·v_g(R)·∇ + (ω₀(R) - ω_ref)] F(R) = ΔE·F(R)
```

Where:
- η = 2π/L (moiré momentum scale)
- ω₀(**R**) = local band frequency
- **M**⁻¹(**R**) = effective mass tensor
- **v**ᵍ(**R**) = group velocity
- F(**R**) = envelope wavefunction
- ΔE = energy detuning from ω_ref

---

## Future Work

1. **Higher Resolution:** Convergence study at 16×16, 32×32
2. **Geometry Optimization:** Scan (r, ε) to find band extremum
3. **Different k-points:** Try Γ point or high-symmetry edges
4. **Full Meep Validation:** Direct eigenmode simulation
5. **Q-factor Analysis:** Radiative losses and confinement
6. **Multi-band Effects:** Coupling to other bands
7. **Nonlinear Extension:** χ⁽³⁾ effects in localized modes

---

## References

1. **Envelope Approximation:** Wannier (1960), Kohn-Luttinger (1955)
2. **Moiré Photonics:** Hu et al. (2021), Wang et al. (2023)
3. **MPB Documentation:** ab-initio.mit.edu/wiki/index.php/MIT_Photonic_Bands

---

## Authors

Pipeline developed for moiré photonic crystal research (October 2024).

For questions or issues, see `pipeline_checklist.md` for implementation details.
