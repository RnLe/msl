#!/usr/bin/env python3
"""
3-way supercell comparison: MPB vs FDFD vs EA
===============================================
Square lattice, (m,n)=(11,1) commensurate, θ≈10.39°, N_cells=122.
TM band 3 at M-point (ω₀≈0.6846 c/a).

Strategy:
  1. Build the bilayer supercell epsilon grid, sanity-check
  2. MPB supercell at Γ (where M folds) — ~450 bands
  3. FDFD supercell at q=0 — shift-invert near ω₀²
  4. EA: mini Phase-1 registry (no Bloch fields) → envelope Hamiltonian → diag
  5. Side-by-side comparison plot

Usage:
    python square_supercell_3way.py                  # run all
    python square_supercell_3way.py --skip-mpb       # skip MPB (load from file)
    python square_supercell_3way.py --skip-fdfd      # skip FDFD
    python square_supercell_3way.py --skip-ea        # skip EA
    python square_supercell_3way.py --plot-only       # just plot from saved data
"""

import sys, os

# CRITICAL: Set threading env vars BEFORE importing numpy/scipy/mpb.
# Single-threaded MPB is actually faster (avoids lock contention).
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import argparse, time, json, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(THESIS_DIR))      # for T_direct_validation.* package imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# ═══════════════════════════════════════════════════════════════
#  Physical parameters
# ═══════════════════════════════════════════════════════════════

A         = 1.0
R_OVER_A  = 0.2
EPS_ROD   = 11.56
EPS_BG    = 1.0
M_IDX, N_IDX = 11, 1             # commensurate indices
N_CELLS   = M_IDX**2 + N_IDX**2  # = 122
THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
THETA_DEG = np.degrees(THETA_RAD)

# Supercell vectors: L1 = m*a1 + n*a2, L2 = -n*a1 + m*a2
L1 = np.array([M_IDX, N_IDX], dtype=float) * A    # (11, 1)
L2 = np.array([-N_IDX, M_IDX], dtype=float) * A   # (-1, 11)
L_SUPER = np.sqrt(L1 @ L1)  # = sqrt(122) ≈ 11.045
B_SUPER = np.column_stack([L1, L2])

# Monolayer basis (square: a1=(1,0), a2=(0,1))
B_MONO = np.eye(2) * A

# Target frequency (TM band 3 at M, from MPB res=128)
OMEGA0    = 0.68457    # ωa/2πc
SIGMA     = (2 * np.pi * OMEGA0)**2   # ≈ 18.50

# Band configuration
TARGET_BAND = 3   # 0-indexed monolayer band
N_BANDS_MPB = 8   # monolayer bands we consider
N_MODES     = 50  # unified: all three methods return this many eigenvalues

# Resolutions — all expressed PER MONOLAYER UNIT CELL
RES_PER_CELL_MPB  = 64    # MPB supercell: 64 px per monolayer cell
                          # → total = round(11.045 × 64) = 707
                          # Memory ~20 GB for 549 bands (fits in 53 GB RAM)
RES_PER_CELL_FDFD = 128   # FDFD supercell: 128 px per monolayer cell
                          # → Nx = round(11.045 × 128) = 1414, DOF ≈ 2.0M
                          # Uses CHOLMOD shift-invert → only 50 modes needed
REGISTRY_NR    = 32       # EA registry sampling (32×32)
NS_EA          = 128      # EA moiré spatial grid for envelope Hamiltonian

# Derived resolutions
RES_MPB_SUPER = int(round(L_SUPER * RES_PER_CELL_MPB))   # 353
NX_FDFD       = int(round(L_SUPER * RES_PER_CELL_FDFD))  # 1414
DOF_FDFD      = NX_FDFD ** 2

# Output
OUTDIR = SCRIPT_DIR / "square_3way"

# ═══════════════════════════════════════════════════════════════
#  1. EPSILON SANITY CHECK
# ═══════════════════════════════════════════════════════════════

def build_and_check_epsilon():
    """Build bilayer supercell epsilon grid via FDFD code, sanity-check."""
    from T_direct_validation.supercell_geometry import build_supercell_eps

    t0 = time.time()
    eps, info = build_supercell_eps(
        'square', M_IDX, N_IDX, a=A,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=NX_FDFD, Ny=NX_FDFD,
    )
    dt = time.time() - t0

    rod_frac = (eps > (EPS_BG + 0.1)).mean()
    expected_rod_frac = 2 * np.pi * R_OVER_A**2  # ≈ 0.2513 (two layers)
    print(f"  ε grid: {eps.shape}, build time={dt:.1f}s")
    print(f"  ε range: [{eps.min():.2f}, {eps.max():.2f}]")
    print(f"  Rod fraction: {rod_frac:.4f} (expected ≈ {expected_rod_frac:.4f})")
    print(f"  θ={info['theta_deg']:.4f}°, N_cells={info['N_cells']}")

    # Plot epsilon
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(eps.T, origin='lower', cmap='RdBu_r',
              extent=[0, 1, 0, 1])
    ax.set_title(f'ε(s₁,s₂) — square ({M_IDX},{N_IDX}), θ={THETA_DEG:.2f}°')
    ax.set_xlabel('s₁'); ax.set_ylabel('s₂')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'eps_supercell.png', dpi=150)
    plt.close()
    print(f"  Saved epsilon plot")

    return eps, info


# ═══════════════════════════════════════════════════════════════
#  2. MPB SUPERCELL AT Γ
# ═══════════════════════════════════════════════════════════════

def enumerate_rods_supercell():
    """Enumerate all rod centers from both layers in supercell fractional coords."""
    B_inv = np.linalg.inv(B_SUPER)
    R_mat = np.array([
        [np.cos(THETA_RAD), -np.sin(THETA_RAD)],
        [np.sin(THETA_RAD),  np.cos(THETA_RAD)],
    ])

    rods = []  # list of (f1, f2, layer)
    margin = max(M_IDX, N_IDX) + 2

    # Layer 1: unrotated
    seen = set()
    for p in range(-margin, margin + 1):
        for q in range(-margin, margin + 1):
            pos = np.array([p, q], dtype=float) * A  # physical
            f = B_inv @ pos
            f = f - np.floor(f)  # wrap to [0, 1)
            key = (round(f[0] * N_CELLS) % N_CELLS, round(f[1] * N_CELLS) % N_CELLS)
            if key not in seen:
                seen.add(key)
                rods.append((f[0], f[1], 1))

    # Layer 2: rotated
    seen2 = set()
    for p in range(-margin, margin + 1):
        for q in range(-margin, margin + 1):
            pos = R_mat @ (np.array([p, q], dtype=float) * A)
            f = B_inv @ pos
            f = f - np.floor(f)
            key = (round(f[0] * N_CELLS) % N_CELLS, round(f[1] * N_CELLS) % N_CELLS)
            if key not in seen2:
                seen2.add(key)
                rods.append((f[0], f[1], 2))

    print(f"  Rods: {len(rods)} total ({len(seen)} layer 1, {len(seen2)} layer 2)")
    assert len(seen) == N_CELLS, f"Layer 1: expected {N_CELLS}, got {len(seen)}"
    assert len(seen2) == N_CELLS, f"Layer 2: expected {N_CELLS}, got {len(seen2)}"
    return rods


def run_mpb_supercell():
    """Run MPB on the bilayer supercell at Γ. Returns frequencies in c/a units."""
    import meep as mp
    from meep import mpb

    print(f"\n  MPB supercell: resolution={RES_MPB_SUPER}, "
          f"|C₁|={L_SUPER:.3f}, θ={THETA_DEG:.2f}°")

    # Enumerate rods
    rods = enumerate_rods_supercell()
    # MPB radius in supercell units: r = r_physical / |C1|
    # (because MPB uses size=(1,1) with basis = C1, C2)
    r_mpb = R_OVER_A * A / L_SUPER

    # Build MPB geometry
    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(L1[0], L1[1], 0),
        basis2=mp.Vector3(L2[0], L2[1], 0),
    )

    geometry = []
    for f1, f2, layer in rods:
        geometry.append(mp.Cylinder(
            radius=r_mpb,
            center=mp.Vector3(f1, f2, 0),
            material=mp.Medium(epsilon=EPS_ROD),
        ))

    # How many bands do we need?
    # MPB computes bottom-up (no shift-invert).
    # Band 3 (0-indexed) × N_cells = 3 × 122 = 366 folded bands below.
    # Plus ~122 for band 3 itself, plus margin. Round to 500.
    n_mpb_bands = (TARGET_BAND + 1) * N_CELLS + N_CELLS // 2  # = 549
    print(f"  Computing {n_mpb_bands} bands at Γ (MPB bottom-up, ")
    print(f"    need ≥{TARGET_BAND * N_CELLS} to reach band {TARGET_BAND})")
    print(f"  Resolution: {RES_MPB_SUPER} total = "
          f"{RES_PER_CELL_MPB} px/cell")

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=n_mpb_bands,
        resolution=RES_MPB_SUPER,
        k_points=[mp.Vector3(0, 0, 0)],  # Γ
    )

    t0 = time.time()
    mp.verbosity(0)
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    ms.run_tm()
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)
    dt = time.time() - t0

    # MPB reports in c/|C1| units → convert to c/a
    freqs_raw = np.array(ms.all_freqs[0])  # (n_bands,)
    freqs_ca = freqs_raw / L_SUPER
    print(f"  MPB done in {dt:.0f}s ({dt/60:.1f}min)")
    print(f"  Frequency range: [{freqs_ca[0]:.6f}, {freqs_ca[-1]:.6f}] (c/a)")

    # Find bands near ω₀
    near_mask = np.abs(freqs_ca - OMEGA0) < 0.10
    near_idx = np.where(near_mask)[0]
    print(f"  Bands within ±0.10 of ω₀={OMEGA0:.5f}: {len(near_idx)} "
          f"(indices {near_idx[0]}..{near_idx[-1]})")

    # Also extract epsilon for sanity check
    mpb_eps = np.array(ms.get_epsilon(), dtype=np.float64)
    if mpb_eps.ndim == 3:
        mpb_eps = mpb_eps[:, :, 0]
    # Roll from MPB center convention to FDFD corner convention
    Nx, Ny = mpb_eps.shape
    mpb_eps = np.roll(mpb_eps, (Nx // 2, Ny // 2), axis=(0, 1))

    return freqs_ca, freqs_raw, dt, mpb_eps


# ═══════════════════════════════════════════════════════════════
#  3. FDFD SUPERCELL AT Γ
# ═══════════════════════════════════════════════════════════════

def run_fdfd_supercell(eps_grid, info):
    """Run FDFD eigensolver on supercell at q=0, shift-invert near ω₀."""
    import scipy.sparse as sp
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    from scipy.sparse.linalg import eigsh

    n_modes = N_MODES
    q_vec = np.zeros(2)  # M folds to Γ exactly

    print(f"\n  FDFD supercell: Nx={NX_FDFD} ({RES_PER_CELL_FDFD} px/cell), "
          f"DOF={DOF_FDFD:,}, σ={SIGMA:.4f}, {n_modes} modes")

    t0 = time.time()
    L_op = build_fdfd_operator(eps_grid, info, q_vec=q_vec, polarization='tm')
    t_build = time.time() - t0
    print(f"  Operator: nnz={L_op.nnz:,}, dtype={L_op.dtype}, "
          f"build time={t_build:.1f}s")

    # Check if we can use CHOLMOD
    try:
        from sksparse.cholmod import cholesky
        t0 = time.time()
        shifted = L_op - SIGMA * sp.eye(L_op.shape[0], format='csc',
                                         dtype=L_op.dtype)
        factor = cholesky(shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        print(f"  CHOLMOD LDLᵀ factorization: {t_factor:.1f}s")

        from scipy.sparse.linalg import LinearOperator
        OPinv = LinearOperator(L_op.shape,
                               matvec=lambda x: factor(x),
                               dtype=L_op.dtype)
        t0 = time.time()
        evals, evecs = eigsh(L_op, k=n_modes, sigma=SIGMA, which='LM',
                             OPinv=OPinv, maxiter=10000, tol=1e-10)
        t_eig = time.time() - t0
    except ImportError:
        print("  CHOLMOD not available, using scipy LU fallback")
        t0 = time.time()
        evals, evecs = eigsh(L_op, k=n_modes, sigma=SIGMA, which='LM',
                             maxiter=10000, tol=1e-10)
        t_eig = time.time() - t0

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Convert to frequencies: ω = √λ / (2π)
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    print(f"  eigsh: {t_eig:.1f}s")
    print(f"  ω range: [{freqs[0]:.6f}, {freqs[-1]:.6f}] (c/a)")

    near_mask = np.abs(freqs - OMEGA0) < 0.10
    n_near = near_mask.sum()
    print(f"  Modes within ±0.10 of ω₀: {n_near}")

    return freqs, evals, t_build + t_eig


# ═══════════════════════════════════════════════════════════════
#  4. ENVELOPE APPROXIMATION
# ═══════════════════════════════════════════════════════════════

def run_ea_phase1():
    """Quick Phase 1: registry sweep at reduced resolution, no Bloch fields."""
    from multiprocessing import Pool
    from phase1_mpb_v3 import _compute_single_registry_point

    print(f"\n  EA Phase 1: {REGISTRY_NR}×{REGISTRY_NR} registry, "
          f"MPB res={128}, 16 workers")

    params = {
        'lattice_type': 'square',
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_ROD,
        'k0': [0.5, 0.5],   # M-point
        'dk': 0.06,
        'all_bands': list(range(N_BANDS_MPB)),
        'polarization': 'TM',
        'fd_order': 6,
        'resolution': 128,
        'max_band': N_BANDS_MPB,
        'export_bloch_fields': False,
    }

    step = 1.0 / REGISTRY_NR
    work = []
    for ix in range(REGISTRY_NR):
        for iy in range(REGISTRY_NR):
            delta_frac = np.array([ix * step, iy * step])
            work.append((ix, iy, delta_frac, params))

    t0 = time.time()
    n_stencil = 7  # fd_order=6 → 7×7 stencil
    omega0_reg = np.full((REGISTRY_NR, REGISTRY_NR, N_BANDS_MPB), np.nan)
    vg_reg = np.full((REGISTRY_NR, REGISTRY_NR, N_BANDS_MPB, 2), np.nan)
    Minv_reg = np.full((REGISTRY_NR, REGISTRY_NR, N_BANDS_MPB, 2, 2), np.nan)
    stencil_omega_reg = np.full(
        (REGISTRY_NR, REGISTRY_NR, N_BANDS_MPB, n_stencil, n_stencil), np.nan)

    with Pool(processes=16) as pool:
        for ix, iy, result in pool.imap_unordered(
                _compute_single_registry_point, work, chunksize=4):
            omega0_reg[ix, iy] = result['omega0']
            vg_reg[ix, iy] = result['vg']
            Minv_reg[ix, iy] = result['M_inv']
            stencil_omega_reg[ix, iy] = result['omega_stencil']
            done = np.isfinite(omega0_reg[:, :, 0]).sum()
            if done % 100 == 0:
                print(f"    {done}/{REGISTRY_NR**2} points done", flush=True)

    dt = time.time() - t0
    print(f"  Registry sweep done in {dt:.0f}s ({dt/60:.1f}min)")

    return omega0_reg, vg_reg, Minv_reg, stencil_omega_reg


def run_ea_solve(omega0_reg, Minv_reg):
    """Build single-band envelope Hamiltonian and diagonalize."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from scipy.interpolate import RegularGridInterpolator

    print(f"\n  EA solving: Ns={NS_EA}×{NS_EA}, band {TARGET_BAND}")

    # ── 1. Map moiré grid to registry shifts ──
    Ns = NS_EA
    B_moire = B_SUPER.copy()  # moiré cell = supercell
    B_moire_inv = np.linalg.inv(B_moire)
    R_mat = np.array([
        [np.cos(THETA_RAD), -np.sin(THETA_RAD)],
        [np.sin(THETA_RAD),  np.cos(THETA_RAD)],
    ])
    B_mono_inv = np.linalg.inv(B_MONO)

    # Moiré fractional grid
    s1 = np.arange(Ns) / Ns
    s2 = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    # Physical positions
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]

    # Registry shift: δ(R) = B_mono^{-1} @ [(R(θ) - I) @ R_phys] mod 1
    # For small θ: R(θ) @ r - r ≈ θ × (-y, x) + O(θ²)
    # But exact: compute displacement of layer 2 relative to layer 1
    pos = np.stack([X, Y], axis=-1)  # (Ns, Ns, 2)
    pos_flat = pos.reshape(-1, 2)

    # Layer 2 position at moiré location R: R_rot @ R vs R (unrotated)
    # Displacement: (R_rot - I) @ R
    disp = ((R_mat - np.eye(2)) @ pos_flat.T).T  # (Ns², 2)

    # Convert to monolayer fractional coords and wrap
    delta_frac = (B_mono_inv @ disp.T).T  # (Ns², 2)
    delta_frac = delta_frac - np.floor(delta_frac)  # wrap to [0, 1)
    delta_frac = delta_frac.reshape(Ns, Ns, 2)

    # ── 2. Interpolate registry data to moiré grid ──
    reg_ax = np.linspace(0, 1, REGISTRY_NR, endpoint=False)

    # Band TARGET_BAND only
    omega0_b = omega0_reg[:, :, TARGET_BAND]
    Minv_b = Minv_reg[:, :, TARGET_BAND, :, :]  # (NR, NR, 2, 2)

    # Wrap-pad for periodic interpolation
    def pad_periodic(arr):
        """Pad a periodic 2D array for smooth interpolation at boundaries."""
        # Add one extra row/col that wraps around
        padded = np.concatenate([arr, arr[:1, :]], axis=0)
        padded = np.concatenate([padded, padded[:, :1]], axis=1)
        return padded

    reg_ax_ext = np.concatenate([reg_ax, [1.0]])

    omega0_padded = pad_periodic(omega0_b)
    interp_omega = RegularGridInterpolator(
        (reg_ax_ext, reg_ax_ext), omega0_padded,
        method='linear', bounds_error=False, fill_value=None)

    omega_moire = interp_omega(delta_frac.reshape(-1, 2)).reshape(Ns, Ns)
    omega_ref = np.mean(omega_moire)
    V_moire = omega_moire - omega_ref

    # Interpolate M_inv components
    Minv_moire = np.zeros((Ns, Ns, 2, 2))
    for i in range(2):
        for j in range(2):
            m_padded = pad_periodic(Minv_b[:, :, i, j])
            interp_m = RegularGridInterpolator(
                (reg_ax_ext, reg_ax_ext), m_padded,
                method='linear', bounds_error=False, fill_value=None)
            Minv_moire[:, :, i, j] = interp_m(
                delta_frac.reshape(-1, 2)).reshape(Ns, Ns)

    print(f"  ω_ref = {omega_ref:.6f}")
    print(f"  V range: [{V_moire.min():.6f}, {V_moire.max():.6f}]")
    print(f"  M⁻¹_11 range: [{Minv_moire[:,:,0,0].min():.4f}, "
          f"{Minv_moire[:,:,0,0].max():.4f}]")

    # ── 3. Build Hamiltonian ──
    # H = V(s) + (1/2) × 1/(2π)² × Σ_ij M⁻¹_ij d²/dR̃_i dR̃_j
    # where R̃ = physical position in units of a
    # derivatives with respect to R̃, built with physical spacing dR
    N_total = Ns * Ns
    dR = L_SUPER / Ns  # physical spacing (same in both directions since L1⊥L2)

    # 4th-order periodic Laplacian and derivative matrices (1D)
    from phase3_mpb_v3 import build_periodic_laplacian_matrix as build_lap
    from phase3_mpb_v3 import build_periodic_derivative_matrix as build_deriv

    L1_mat = build_lap(Ns, dR, order=4)     # d²/dR₁²
    L2_mat = build_lap(Ns, dR, order=4)     # d²/dR₂²
    D1_mat = build_deriv(Ns, dR, order=4)   # d/dR₁
    D2_mat = build_deriv(Ns, dR, order=4)   # d/dR₂

    I_Ns = sp.eye(Ns, format='csr')

    # Full 2D operators
    L1_2d = sp.kron(L1_mat, I_Ns, format='csr')
    L2_2d = sp.kron(I_Ns, L2_mat, format='csr')
    D1_2d = sp.kron(D1_mat, I_Ns, format='csr')
    D2_2d = sp.kron(I_Ns, D2_mat, format='csr')

    # Diagonal M_inv components
    M11 = sp.diags(Minv_moire[:, :, 0, 0].ravel(), format='csr')
    M22 = sp.diags(Minv_moire[:, :, 1, 1].ravel(), format='csr')
    M12 = sp.diags(Minv_moire[:, :, 0, 1].ravel(), format='csr')

    prefactor = 0.5 / (2 * np.pi)**2

    K_op = -prefactor * (M11 @ L1_2d + M22 @ L2_2d + 2 * M12 @ (D1_2d @ D2_2d))
    K_op = 0.5 * (K_op + K_op.conj().T)  # Hermitize

    V_op = sp.diags(V_moire.ravel(), format='csr')

    H = V_op + K_op

    # ── 4. Diagonalize ──
    n_modes = N_MODES
    sigma_ea = V_moire.min()
    evals_ea, evecs_ea = eigsh(H, k=n_modes, sigma=sigma_ea, which='LM',
                                maxiter=10000, tol=1e-10)
    idx = np.argsort(evals_ea)
    evals_ea = evals_ea[idx]

    # Physical frequencies: ω = ω_ref + E
    freqs_ea = omega_ref + evals_ea
    print(f"  EA eigenvalues (first 10 detunings): {evals_ea[:10]}")
    print(f"  EA frequencies (first 10): {freqs_ea[:10]}")

    return freqs_ea, evals_ea, omega_ref, V_moire


# ═══════════════════════════════════════════════════════════════
#  5. COMPARISON PLOT
# ═══════════════════════════════════════════════════════════════

def plot_comparison(freqs_mpb=None, freqs_fdfd=None, freqs_ea=None,
                    omega_ref_ea=None, V_moire=None):
    """Side-by-side eigenvalue comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Level diagram ──
    ax = axes[0]
    window = 0.06  # ±0.06 around ω₀
    y_positions = {'MPB': 0.2, 'FDFD': 0.5, 'EA': 0.8}

    for label, freqs, color in [
            ('MPB', freqs_mpb, 'blue'),
            ('FDFD', freqs_fdfd, 'red'),
            ('EA', freqs_ea, 'green')]:
        if freqs is None:
            continue
        mask = np.abs(freqs - OMEGA0) < window
        f = freqs[mask]
        y = y_positions[label]
        ax.hlines(f, y - 0.08, y + 0.08, color=color, lw=0.8)
        ax.text(y, OMEGA0 + window * 0.9, label, ha='center',
                fontsize=10, color=color, fontweight='bold')

    ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5, label=f'ω₀={OMEGA0:.5f}')
    ax.set_ylabel(r'$\omega\, a / 2\pi c$')
    ax.set_title('Eigenvalue Level Diagram')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(OMEGA0 - window, OMEGA0 + window)

    # ── Panel 2: Eigenvalue differences ──
    ax = axes[1]
    if freqs_mpb is not None and freqs_fdfd is not None:
        mask_mpb = np.abs(freqs_mpb - OMEGA0) < window
        mask_fdfd = np.abs(freqs_fdfd - OMEGA0) < window
        f_mpb = np.sort(freqs_mpb[mask_mpb])
        f_fdfd = np.sort(freqs_fdfd[mask_fdfd])
        n_compare = min(len(f_mpb), len(f_fdfd))
        if n_compare > 0:
            diff = f_fdfd[:n_compare] - f_mpb[:n_compare]
            ax.plot(range(n_compare), diff * 1000, 'ko-', ms=3,
                    label='FDFD − MPB')

    if freqs_mpb is not None and freqs_ea is not None:
        mask_mpb = np.abs(freqs_mpb - OMEGA0) < window
        mask_ea = np.abs(freqs_ea - OMEGA0) < window
        f_mpb = np.sort(freqs_mpb[mask_mpb])
        f_ea = np.sort(freqs_ea[mask_ea])
        n_compare = min(len(f_mpb), len(f_ea))
        if n_compare > 0:
            diff = f_ea[:n_compare] - f_mpb[:n_compare]
            ax.plot(range(n_compare), diff * 1000, 'gs-', ms=3,
                    label='EA − MPB')

    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel('Eigenvalue index (near ω₀)')
    ax.set_ylabel(r'$\Delta\omega \times 10^3$')
    ax.set_title('Eigenvalue Differences')
    ax.legend()

    # ── Panel 3: Moiré potential (if available) ──
    ax = axes[2]
    if V_moire is not None:
        im = ax.imshow(V_moire.T * 1000, origin='lower', cmap='coolwarm',
                       extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax, label=r'$V \times 10^3$')
        ax.set_title(r'Moiré potential $V(R)$')
        ax.set_xlabel('s₁'); ax.set_ylabel('s₂')
    else:
        ax.text(0.5, 0.5, 'No EA data', ha='center', va='center',
                transform=ax.transAxes)

    fig.suptitle(f'Square Lattice ({M_IDX},{N_IDX}): '
                 f'θ={THETA_DEG:.2f}°, N={N_CELLS}, '
                 f'TM band {TARGET_BAND} at M',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig_3way_comparison.png', dpi=200)
    print(f"  Saved comparison plot")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-mpb', action='store_true')
    parser.add_argument('--skip-fdfd', action='store_true')
    parser.add_argument('--skip-ea', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("=" * 70)
    print("  3-Way Supercell Comparison: MPB vs FDFD vs EA")
    print(f"  Square lattice ({M_IDX},{N_IDX}): θ={THETA_DEG:.2f}°, "
          f"N_cells={N_CELLS}")
    print(f"  Target: TM band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}")
    print(f"  MPB: {RES_PER_CELL_MPB} px/cell (total={RES_MPB_SUPER}), "
          f"{(TARGET_BAND+1)*N_CELLS + N_CELLS//2} bands")
    print(f"  FDFD: {RES_PER_CELL_FDFD} px/cell (Nx={NX_FDFD}), "
          f"{N_MODES} modes near σ")
    print(f"  EA: Ns={NS_EA}, registry={REGISTRY_NR}×{REGISTRY_NR}, "
          f"{N_MODES} modes")
    print(f"  Output: {OUTDIR}")
    print("=" * 70)

    freqs_mpb = freqs_fdfd = freqs_ea = None
    omega_ref_ea = None
    V_moire = None

    npz_file = OUTDIR / 'supercell_3way_data.npz'

    if args.plot_only:
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)
            freqs_mpb = data.get('freqs_mpb')
            freqs_fdfd = data.get('freqs_fdfd')
            freqs_ea = data.get('freqs_ea')
            V_moire = data.get('V_moire')
            omega_ref_ea = float(data.get('omega_ref_ea', OMEGA0))
            plot_comparison(freqs_mpb, freqs_fdfd, freqs_ea,
                            omega_ref_ea, V_moire)
        else:
            print(f"  No data file found: {npz_file}")
        return

    # ── 1. Epsilon check ──
    print("\n1. Building supercell epsilon grid...")
    eps_grid, info = build_and_check_epsilon()

    # ── 2. MPB ──
    if not args.skip_mpb:
        print("\n2. Running MPB supercell...")
        freqs_mpb, freqs_mpb_raw, dt_mpb, mpb_eps = run_mpb_supercell()
        np.savez(OUTDIR / 'mpb_supercell.npz',
                 freqs_ca=freqs_mpb, freqs_raw=freqs_mpb_raw,
                 wall_time=dt_mpb)
        print(f"  Saved MPB results")
    else:
        f = OUTDIR / 'mpb_supercell.npz'
        if f.exists():
            d = np.load(f)
            freqs_mpb = d['freqs_ca']
            print(f"\n2. Loaded MPB data: {len(freqs_mpb)} bands")
        else:
            print("\n2. MPB skipped (no saved data)")

    # ── 3. FDFD ──
    if not args.skip_fdfd:
        print("\n3. Running FDFD supercell...")
        freqs_fdfd, evals_fdfd, dt_fdfd = run_fdfd_supercell(eps_grid, info)
        np.savez(OUTDIR / 'fdfd_supercell.npz',
                 freqs=freqs_fdfd, evals=evals_fdfd, wall_time=dt_fdfd)
        print(f"  Saved FDFD results")
    else:
        f = OUTDIR / 'fdfd_supercell.npz'
        if f.exists():
            d = np.load(f)
            freqs_fdfd = d['freqs']
            print(f"\n3. Loaded FDFD data: {len(freqs_fdfd)} bands")
        else:
            print("\n3. FDFD skipped (no saved data)")

    # ── 4. EA ──
    if not args.skip_ea:
        print("\n4. Running Envelope Approximation...")
        reg_file = OUTDIR / 'ea_registry.npz'
        if reg_file.exists():
            print("  Loading saved registry data...")
            d = np.load(reg_file)
            omega0_reg = d['omega0']
            Minv_reg = d['M_inv']
            stencil_omega_reg = d.get('stencil_omega', None)
        else:
            omega0_reg, vg_reg, Minv_reg, stencil_omega_reg = run_ea_phase1()
            np.savez(reg_file, omega0=omega0_reg, vg=vg_reg, M_inv=Minv_reg,
                     stencil_omega=stencil_omega_reg)
            print(f"  Saved registry data")

        freqs_ea, evals_ea, omega_ref_ea, V_moire = run_ea_solve(
            omega0_reg, Minv_reg)
        np.savez(OUTDIR / 'ea_supercell.npz',
                 freqs=freqs_ea, evals=evals_ea,
                 omega_ref=omega_ref_ea, V_moire=V_moire)
        print(f"  Saved EA results")
    else:
        f = OUTDIR / 'ea_supercell.npz'
        if f.exists():
            d = np.load(f)
            freqs_ea = d['freqs']
            V_moire = d.get('V_moire')
            omega_ref_ea = float(d.get('omega_ref', OMEGA0))
            print(f"\n4. Loaded EA data: {len(freqs_ea)} modes")
        else:
            print("\n4. EA skipped (no saved data)")

    # ── 5. Comparison ──
    print("\n5. Generating comparison plot...")
    plot_comparison(freqs_mpb, freqs_fdfd, freqs_ea, omega_ref_ea, V_moire)

    # Save combined data
    save_dict = {}
    if freqs_mpb is not None: save_dict['freqs_mpb'] = freqs_mpb
    if freqs_fdfd is not None: save_dict['freqs_fdfd'] = freqs_fdfd
    if freqs_ea is not None: save_dict['freqs_ea'] = freqs_ea
    if V_moire is not None: save_dict['V_moire'] = V_moire
    if omega_ref_ea is not None: save_dict['omega_ref_ea'] = omega_ref_ea
    np.savez(npz_file, **save_dict)

    # ── Summary table ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: eigenvalues within ±0.03 of ω₀={OMEGA0:.5f}")
    print(f"{'='*70}")
    w = 0.03
    for label, freqs in [('MPB', freqs_mpb), ('FDFD', freqs_fdfd),
                          ('EA', freqs_ea)]:
        if freqs is not None:
            mask = np.abs(freqs - OMEGA0) < w
            f = np.sort(freqs[mask])
            print(f"\n  {label}: {len(f)} eigenvalues")
            for i, ff in enumerate(f[:20]):
                print(f"    {i:3d}: ω = {ff:.6f}  (Δ = {ff-OMEGA0:+.6f})")

    dt_total = time.time() - t_total
    print(f"\nTotal time: {dt_total:.0f}s ({dt_total/60:.1f}min)")


if __name__ == '__main__':
    main()
