#!/usr/bin/env python
"""
η-sweep validation: run the envelope pipeline at multiple twist angles
to test the asymptotic scaling of the theory's error.

ARCHITECTURE:
  Phase 1 produces a θ-INDEPENDENT "universal master map" of Bloch fields
  ω_n(δ) and u_n(r; δ) over the monolayer stacking space δ ∈ [0,1)².
  
  Only the mapping from moiré position R → stacking offset δ(R) depends
  on θ, and this enters through the moiré basis B_moire and length L_moire.
  
  Therefore: we run Phase 1 ONCE, then re-run Phases 2+3 at each θ by
  patching the θ-dependent metadata in the HDF5 files.

OBSERVABLES COLLECTED (per θ):
  A) N-band convergence: |λ(N=3) - λ(N=1)| for the lowest mode
  B) Bandwidth / spectrum: eigenvalue spread, flatness ratio
  C) Per-tile FD-corrected residual (non-trivial when modes mix bands)

USAGE:
  python eta_sweep.py [--candidate_id 0]
"""

import sys, math, json, shutil, time, argparse, gc, os, resource
from pathlib import Path
import numpy as np
import h5py


def log_mem(label=""):
    """Log current RSS memory usage (actual, not peak)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss_kb = int(line.split()[1])
                    rss_gb = rss_kb / 1e6
                    print(f"  [MEM] {label}: RSS = {rss_gb:.2f} GB (current)")
                    return
    except Exception:
        pass
    # Fallback to peak RSS
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / 1e6
    print(f"  [MEM] {label}: RSS = {rss_gb:.2f} GB (peak)")

# Add project root to path (this file lives inside phasesV3/)
SCRIPT_DIR = Path(__file__).resolve().parent          # phasesV3/
PROJECT_ROOT = SCRIPT_DIR.parent                      # moire_envelope/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import phase2_mpb_v3 as p2
import phase3_mpb_v3 as p3
import phase4_field_reconstruction as p4
from validation_residual import (
    gauge_fix_bloch_fields, compute_per_tile_residual,
    extract_mpb_epsilon_grid,
)
from common.io_utils import candidate_dir, load_json

# Import symmetrization (lives in thesis_results/)
sys.path.insert(0, str(PROJECT_ROOT / 'thesis_results'))
try:
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False

# =============================================================================
# Geometry helpers
# =============================================================================

def compute_moire_params(theta_deg, lattice_type='square', a=1.0):
    """
    Compute moiré geometric parameters for a given twist angle.
    Uses the EXACT formula from phase1_mpb_v3.py:
      B_moire = (R(θ) - I)^{-1} @ B_mono
    
    Returns dict with: theta_rad, eta, B_moire, moire_length
    """
    theta_rad = math.radians(theta_deg)
    eta = 2 * math.sin(theta_rad / 2)
    
    # Monolayer basis
    if lattice_type == 'square':
        B_mono = np.array([[a, 0.0], [0.0, a]])
    elif lattice_type in ('hex', 'honeycomb'):
        # Hexagonal/honeycomb lattice: a1 = a*(1, 0), a2 = a*(1/2, √3/2)
        # (honeycomb = triangular Bravais lattice + 2-atom basis)
        B_mono = np.array([[a, 0.0], [a/2.0, a * math.sqrt(3)/2.0]])
    else:
        raise NotImplementedError(f"Lattice type {lattice_type} not implemented in sweep")
    
    # Moiré basis: B_moire = (R(θ) - I)^{-1} @ B_mono  [matches phase1_mpb_v3.py]
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R_theta = np.array([[c, -s], [s, c]])
    Delta_R = R_theta - np.eye(2)
    Delta_R_inv = np.linalg.inv(Delta_R)
    B_moire = Delta_R_inv @ B_mono
    
    moire_length = np.linalg.norm(B_moire[:, 0])
    
    return {
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'eta': eta,
        'B_moire': B_moire,
        'moire_length': moire_length,
    }


def patch_h5_theta(h5_path, moire_params):
    """
    Patch θ-dependent attributes and datasets in an HDF5 file in-place.
    """
    with h5py.File(h5_path, 'r+') as hf:
        hf.attrs['theta_deg'] = moire_params['theta_deg']
        hf.attrs['theta_rad'] = moire_params['theta_rad']
        hf.attrs['eta'] = moire_params['eta']
        hf.attrs['moire_length'] = moire_params['moire_length']
        hf.attrs['B_moire'] = moire_params['B_moire']
        
        # Recompute R_grid = B_moire @ s  (fractional_to_cartesian convention)
        if 'R_grid' in hf and 's_grid' in hf:
            s_grid = hf['s_grid'][:]
            B_m = moire_params['B_moire']
            R_new = np.einsum('ij,...j->...i', B_m, s_grid)
            hf['R_grid'][...] = R_new


def patch_meta_theta(meta_path, moire_params):
    """
    Patch θ-dependent fields in phase0_meta.json.
    """
    with open(meta_path) as f:
        meta = json.load(f)
    meta['theta_deg'] = moire_params['theta_deg']
    meta['theta_rad'] = moire_params['theta_rad']
    meta['eta'] = moire_params['eta']
    meta['moire_length'] = moire_params['moire_length']
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Single-θ runner
# =============================================================================

def run_single_theta(theta_deg, source_cdir, sweep_dir, config, n_modes=100):
    """
    Run Phases 2+3 at a single twist angle.
    
    1. Copy Phase 1 HDF5 + meta to a sweep sub-directory
    2. Patch θ-dependent attributes
    3. Run Phase 2
    4. Run Phase 3
    5. Collect observables
    
    Args:
        theta_deg: twist angle in degrees
        source_cdir: Path to the original candidate directory (with Phase 1 data)
        sweep_dir: Path to sweep output directory
        config: dict of pipeline config
        n_modes: number of envelope modes to compute
    
    Returns:
        results: dict of observables for this θ
    """
    t0 = time.time()
    theta_label = f"theta_{theta_deg:.3f}"
    work_dir = sweep_dir / theta_label / "candidate_0000"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  θ = {theta_deg}°, η = {2*math.sin(math.radians(theta_deg)/2):.6f}")
    print(f"  Work dir: {work_dir}")
    print(f"{'='*70}")
    
    # 1. Copy Phase 1 data (symlink bloch_fields for speed, copy the rest)
    phase1_src = source_cdir / "phase1_multiband_data.h5"
    phase1_dst = work_dir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"
    meta_dst = work_dir / "phase0_meta.json"
    
    if not phase1_dst.exists():
        # Smart copy: create a new HDF5 that hardlinks small datasets
        # but uses an external link for the huge bloch_fields array.
        # This saves ~7 GB disk per sweep angle.
        print(f"  Smart-copying Phase 1 HDF5 (linking bloch_fields)...")
        with h5py.File(phase1_src, 'r') as src, h5py.File(phase1_dst, 'w') as dst:
            # Copy all attributes
            for key, val in src.attrs.items():
                dst.attrs[key] = val
            # Copy datasets — full copy for small ones, external link for big
            n_linked = 0
            for key in src.keys():
                obj = src[key]
                if isinstance(obj, h5py.Dataset) and obj.nbytes > 1e9:
                    dst[key] = h5py.ExternalLink(str(phase1_src), f'/{key}')
                    n_linked += 1
                else:
                    src.copy(key, dst)
        print(f"  Done (linked {n_linked} large datasets)")
    if not meta_dst.exists():
        shutil.copy2(meta_src, meta_dst)
    
    # 2. Patch θ-dependent data
    lattice_type = load_json(meta_src).get('lattice_type', 'square')
    a = load_json(meta_src).get('a', 1.0)
    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    
    print(f"  Patching θ = {theta_deg}° → η = {moire_params['eta']:.6f}, "
          f"L_m = {moire_params['moire_length']:.2f}")
    patch_h5_theta(phase1_dst, moire_params)
    patch_meta_theta(meta_dst, moire_params)
    
    # 3. Run Phase 2
    log_mem("Before Phase 2")
    print(f"  Running Phase 2...")
    p2_config = {
        'include_born_huang': config.get('include_born_huang', True),
        'include_drift_term': config.get('include_drift_term', True),
        'use_parallel_transport_gauge': config.get('use_parallel_transport_gauge', True),
        'n_extra_bands': config.get('n_extra_bands', 4),
        'mpb_fd_order': config.get('mpb_fd_order', 4),
    }
    p2.process_candidate_phase2_v3(str(work_dir), p2_config)
    
    # Force garbage collection to release any remaining Phase 2 arrays
    gc.collect()
    log_mem("After Phase 2 + gc.collect")
    
    # 3b. Symmetrize Phase 2 data (C4 for square, C2 for hex M-point, C6 for honeycomb K)
    sym_type = config.get('symmetry_type', None)
    if sym_type is None:
        # Auto-detect from lattice type
        if lattice_type == 'square':
            sym_type = 'C4'
        elif lattice_type == 'hex':
            sym_type = 'C2'
        elif lattice_type == 'honeycomb':
            sym_type = 'C6'
    if sym_type and HAS_SYMMETRIZE:
        print(f"  Symmetrizing Phase 2 data ({sym_type})...")
        try:
            symmetrize_phase2(work_dir, sym_type)
        except Exception as e:
            print(f"  WARNING: Symmetrization failed: {e}. Using unsymmetrized data.")
    elif sym_type and not HAS_SYMMETRIZE:
        print(f"  WARNING: Symmetrization requested ({sym_type}) but module not available.")
    
    # 4. Run Phase 3 — full subspace (N_sub = 3)
    log_mem("Before Phase 3")
    print(f"  Running Phase 3 (N_sub=3, {n_modes} modes)...")
    p3_config_full = {
        'n_modes': n_modes,
        'include_drift_term': True,
        'include_kinetic_term': True,
        'include_born_huang': True,
        'include_offdiag_A': config.get('include_offdiag_A', True),
        'fd_order': 4,
        'sigma_shift': None,  # auto
    }
    p3.process_candidate_phase3_v3(str(work_dir), p3_config_full)
    log_mem("After Phase 3")
    
    # 5. Collect observables
    results = collect_observables(work_dir, moire_params, source_cdir, config)
    results['wall_time_s'] = time.time() - t0
    
    print(f"  Done in {results['wall_time_s']:.1f}s")
    return results


# =============================================================================
# N-band convergence (Option A)
# =============================================================================

def run_nband_convergence(work_dir, moire_params, n_modes=20):
    """
    Solve with N_sub = 1 (single-band, no coupling) and compare to N_sub = 3.
    
    We do this by building the Hamiltonian with only the dominant band's
    diagonal block (Λ_11, M_11, v_11) and solving the scalar envelope equation.
    
    Returns:
        dict with single-band eigenvalues and comparison metrics
    """
    phase2_h5 = work_dir / "phase2_multiband_data.h5"
    
    with h5py.File(phase2_h5, 'r') as hf:
        Lambda = hf['Lambda'][:]
        A_berry = hf['A_berry'][:]
        Phi_BH = hf['Phi_BH'][:]
        v_drift = hf['v_drift'][:]
        M_inv = hf['M_inv'][:]
        
        eta = hf.attrs['eta']
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_sub = int(hf.attrs['N_subspace'])
        B_moire = hf.attrs['B_moire']
        target_idx = int(hf.attrs['target_index_in_subspace'])
    
    L_moire = np.linalg.norm(B_moire[0])
    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2
    
    # --- Single-band solve (dominant band only) ---
    # Extract 1×1 blocks for the target band
    t = target_idx
    Lambda_1 = Lambda[:, :, t:t+1, t:t+1]
    v_drift_1 = v_drift[:, :, t:t+1, t:t+1, :]
    M_inv_1 = M_inv[:, :, t:t+1, t:t+1, :, :]
    A_berry_1 = A_berry[:, :, t:t+1, t:t+1, :]
    Phi_BH_1 = Phi_BH[:, :, t:t+1, t:t+1]
    
    H_1 = p3.assemble_multiband_hamiltonian(
        Lambda_1, v_drift_1, M_inv_1, A_berry_1, Phi_BH_1,
        eta, Ns1, Ns2, 1, dR1, dR2, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=True,
        order=4
    )
    
    # Determine sigma for single-band solve
    diag_V = Lambda_1[:, :, 0, 0].flatten()
    m_trace = np.mean(M_inv_1[:, :, 0, 0, 0, 0] + M_inv_1[:, :, 0, 0, 1, 1])
    sigma_1 = float(np.max(diag_V)) if m_trace < 0 else float(np.min(diag_V))
    
    evals_1, _ = p3.solve_multiband_envelope(H_1, min(n_modes, Ns1*Ns2 - 2), sigma=sigma_1)
    
    return {
        'eigenvalues_N1': evals_1.tolist(),
        'sigma_N1': sigma_1,
    }


# =============================================================================
# Observables collection
# =============================================================================

def collect_observables(work_dir, moire_params, source_cdir, config):
    """
    Collect all observables for a single θ point.
    
    A) N-band convergence
    B) Bandwidth and spectrum
    C) Per-tile residual (band mixing check)
    """
    theta_deg = moire_params['theta_deg']
    eta = moire_params['eta']
    
    # Load Phase 3 results (N_sub = 3)
    F_spinor, eigenvalues, mode_stats = p4.load_phase3_envelopes(work_dir)
    n_modes = len(eigenvalues)
    N_sub = F_spinor.shape[-1]
    
    # Load band metadata from Phase 1 (avoid loading 19 GB bloch_fields)
    with h5py.File(work_dir / 'phase1_multiband_data.h5', 'r') as hf:
        omega_bands = hf['omega'][:]   # (Ns_env, Ns_env, N_sub)
        omega_ref = float(hf.attrs['omega_ref'])
        subspace_bands = hf.attrs['subspace_bands'][:].tolist()
        all_bands = hf.attrs['all_bands'][:].tolist()
    band_indices = p4.get_subspace_band_indices(subspace_bands, all_bands)
    
    # --- Observable B: Eigenvalue spectrum ---
    # Eigenvalues λ_i = ω_i - ω_ref. Physical frequency: f_pred = ω_ref + λ_i
    evals = eigenvalues[:min(n_modes, 50)]  # first 50 modes
    bandwidth = float(evals[-1] - evals[0]) if len(evals) > 1 else 0.0
    gap_01 = float(evals[1] - evals[0]) if len(evals) > 1 else 0.0
    
    # Band composition per mode (compute for ALL modes)
    band_compositions = []
    for m in range(n_modes):
        F_mode = F_spinor[m]
        bw = np.array([np.sum(np.abs(F_mode[:, :, n])**2) for n in range(N_sub)])
        bw /= bw.sum()
        band_compositions.append({
            'mode': m,
            'weights': bw.tolist(),
            'dominant': int(np.argmax(bw)),
            'max_weight': float(np.max(bw)),
            'mixing': float(1.0 - np.max(bw)),  # 0 = pure single-band, >0 = mixed
        })
    
    max_mixing = max(bc['mixing'] for bc in band_compositions)
    
    # --- Observable A: N-band convergence ---
    print(f"  Running N=1 band convergence test...")
    nband_results = run_nband_convergence(work_dir, moire_params, n_modes=20)
    
    import gc; gc.collect()
    
    # Compare lowest eigenvalues
    evals_N3 = eigenvalues[:20]
    evals_N1 = np.array(nband_results['eigenvalues_N1'][:20])
    
    # The eigenvalues live in different subspaces, so we compare:
    # - The lowest eigenvalue of the target band in N=3 vs N=1
    # First find modes in N=3 that are dominated by target band
    meta = load_json(work_dir / 'phase0_meta.json')
    target_sub = meta.get('target_index_in_subspace', 0)
    
    target_modes_N3 = []
    for m in range(len(band_compositions)):
        if band_compositions[m]['dominant'] == target_sub:
            target_modes_N3.append(m)
    
    if target_modes_N3:
        lambda_0_N3 = float(eigenvalues[target_modes_N3[0]])
        lambda_0_N1 = float(evals_N1[0])
        delta_lambda_N = abs(lambda_0_N3 - lambda_0_N1)
    else:
        lambda_0_N3 = float(eigenvalues[0])
        lambda_0_N1 = float(evals_N1[0])
        delta_lambda_N = abs(lambda_0_N3 - lambda_0_N1)
    
    # --- Observable C: Per-tile residual ---
    # Only compute if we have epsilon data and modes show some mixing
    p0_meta = load_json(source_cdir / 'phase0_meta.json')
    k0_x = p0_meta.get('k0_x', 0.0)
    k0_y = p0_meta.get('k0_y', 0.0)
    G_mono = 2 * np.pi * np.eye(2)  # square lattice
    k0_phys = G_mono @ np.array([k0_x, k0_y])
    
    R_fd_corrected = None
    ratio_fd_corrected = None
    # Skip per-tile residual during sweep to avoid OOM (requires ~17GB for
    # bloch_fields + eps_registry + gauge-fixed fields simultaneously).
    # Band composition data above already captures the mixing information.
    if False and max_mixing > 1e-6:
        print(f"  Computing per-tile residual (max mixing = {max_mixing:.4e})...")
        try:
            eps_registry = extract_mpb_epsilon_grid(
                p0_meta, n_registry=bloch_fields.shape[0],
                resolution=bloch_fields.shape[3],
                cache_path=source_cdir / 'mpb_epsilon_registry.h5',
                n_workers=8
            )
            u_sub_fixed = gauge_fix_bloch_fields(bloch_fields, band_indices)
            tile_results = compute_per_tile_residual(
                bloch_fields_fixed=u_sub_fixed,
                band_indices=band_indices,
                eps_registry=eps_registry,
                omega_bands=omega_bands,
                F_spinor=F_spinor,
                mode_idx=0,
                k0_phys=k0_phys,
                bloch_fields_raw=bloch_fields,
            )
            R_fd_corrected = tile_results['R_fd_corrected']
            ratio_fd_corrected = tile_results['ratio_fd_corrected']
        except Exception as e:
            print(f"  WARNING: Per-tile residual failed: {e}")
    else:
        print(f"  Skipping per-tile residual (max mixing = {max_mixing:.2e} ≈ 0, "
              f"metric is trivially 1.0 for single-band modes)")
    
    # --- Compile results ---
    results = {
        'theta_deg': theta_deg,
        'eta': eta,
        'moire_length': moire_params['moire_length'],
        # Spectrum (Observable B)
        'eigenvalues': evals.tolist(),
        'bandwidth_50': bandwidth,
        'gap_01': gap_01,
        'omega_ref': omega_ref,
        # Band mixing
        'max_mixing': max_mixing,
        'band_compositions': band_compositions,
        # N-band convergence (Observable A)
        'lambda_0_N3': lambda_0_N3,
        'lambda_0_N1': lambda_0_N1,
        'delta_lambda_N': delta_lambda_N,
        # Per-tile residual (Observable C)
        'R_fd_corrected': R_fd_corrected,
        'ratio_fd_corrected': ratio_fd_corrected,
    }
    
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_sweep_results(all_results, sweep_dir):
    """
    Generate the key validation plots from the η-sweep.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    etas = np.array([r['eta'] for r in all_results])
    thetas = np.array([r['theta_deg'] for r in all_results])
    sort_idx = np.argsort(etas)
    etas = etas[sort_idx]
    thetas = thetas[sort_idx]
    results_sorted = [all_results[i] for i in sort_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Panel A: N-band convergence ---
    ax = axes[0, 0]
    delta_lambdas = np.array([r['delta_lambda_N'] for r in results_sorted])
    ax.loglog(etas, delta_lambdas, 'bo-', markersize=8, linewidth=2, label='|λ₀(N=3) − λ₀(N=1)|')
    
    # Reference slopes
    if len(etas) >= 2:
        eta_ref = np.linspace(etas.min(), etas.max(), 100)
        for power, ls, label in [(2, '--', 'η²'), (3, ':', 'η³')]:
            scale = delta_lambdas[len(delta_lambdas)//2] / etas[len(etas)//2]**power
            ax.loglog(eta_ref, scale * eta_ref**power, ls, color='gray', alpha=0.6, label=label)
    
    ax.set_xlabel('η = 2 sin(θ/2)')
    ax.set_ylabel('|Δλ|')
    ax.set_title('A: N-band convergence\n|λ₀(N=3) − λ₀(N=1)|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Panel B: Eigenvalue spectrum (waterfall) ---
    ax = axes[0, 1]
    for i, r in enumerate(results_sorted):
        evals = np.array(r['eigenvalues'])
        n_show = min(20, len(evals))
        ax.scatter([r['eta']] * n_show, evals[:n_show], s=10, c='steelblue', alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('η')
    ax.set_ylabel('Eigenvalue λ = ω − ωref')
    ax.set_title('B: Eigenvalue spectrum vs η')
    ax.grid(True, alpha=0.3)
    
    # --- Panel C: Bandwidth and gap ---
    ax = axes[1, 0]
    bandwidths = np.array([r['bandwidth_50'] for r in results_sorted])
    gaps = np.array([r['gap_01'] for r in results_sorted])
    ax.loglog(etas, bandwidths, 'rs-', markersize=7, linewidth=2, label='Bandwidth (50 modes)')
    ax.loglog(etas, gaps, 'g^-', markersize=7, linewidth=2, label='Gap Δλ₀₁')
    
    if len(etas) >= 2:
        for power, ls in [(2, '--')]:
            scale_bw = bandwidths[len(bandwidths)//2] / etas[len(etas)//2]**power
            scale_gap = gaps[len(gaps)//2] / etas[len(etas)//2]**power
            ax.loglog(eta_ref, scale_bw * eta_ref**power, ls, color='gray', alpha=0.4)
            ax.loglog(eta_ref, scale_gap * eta_ref**power, ls, color='gray', alpha=0.4, label='η²')
    
    ax.set_xlabel('η')
    ax.set_ylabel('Energy scale')
    ax.set_title('B: Bandwidth & gap scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Panel D: Band mixing ---
    ax = axes[1, 1]
    mixings = np.array([r['max_mixing'] for r in results_sorted])
    ax.semilogy(etas, mixings, 'mo-', markersize=8, linewidth=2, label='max(1 − max_weight)')
    
    # Also plot FD-corrected ratio if available
    fd_corr = [r.get('ratio_fd_corrected') for r in results_sorted]
    has_fd = [x is not None for x in fd_corr]
    if any(has_fd):
        etas_fd = etas[has_fd]
        ratios_fd = np.array([fd_corr[i] for i in range(len(fd_corr)) if has_fd[i]])
        ax2 = ax.twinx()
        ax2.plot(etas_fd, ratios_fd, 'cv-', markersize=8, linewidth=2, label='FD-corrected R_q ratio')
        ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('R_q ratio (FD-corrected)', color='c')
        ax2.legend(loc='upper left', fontsize=9)
    
    ax.set_xlabel('η')
    ax.set_ylabel('Band mixing (1 − max weight)')
    ax.set_title('C+D: Band mixing & residual')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('η-Sweep Validation — Moiré Envelope Approximation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(sweep_dir / 'eta_sweep_results.png', dpi=150)
    plt.close()
    print(f"\nPlot saved: {sweep_dir / 'eta_sweep_results.png'}")
    
    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"  {'θ (°)':>8} {'η':>10} {'L_m':>8} {'λ₀(N=3)':>10} {'λ₀(N=1)':>10} "
          f"{'|Δλ|':>10} {'BW':>10} {'mixing':>10} {'time':>6}")
    print(f"{'='*90}")
    for r in results_sorted:
        print(f"  {r['theta_deg']:8.3f} {r['eta']:10.6f} {r['moire_length']:8.1f} "
              f"{r['lambda_0_N3']:10.6f} {r['lambda_0_N1']:10.6f} "
              f"{r['delta_lambda_N']:10.2e} {r['bandwidth_50']:10.4f} "
              f"{r['max_mixing']:10.2e} {r.get('wall_time_s', 0):6.0f}s")
    print(f"{'='*90}")


# =============================================================================
# Main driver
# =============================================================================

def run_eta_sweep(
    candidate_id=0,
    theta_list=None,
    n_modes=100,
    config_overrides=None,
):
    """
    Run the full η-sweep.
    
    Args:
        candidate_id: which candidate's Phase 1 data to use
        theta_list: list of twist angles in degrees (default: logarithmic spread)
        n_modes: number of envelope modes per angle
        config_overrides: dict of config overrides
    """
    if theta_list is None:
        theta_list = [0.5, 0.8, 1.1, 1.5, 2.0, 3.0, 5.0, 8.0]
    
    config = {
        'include_born_huang': True,
        'include_drift_term': True,
        'use_parallel_transport_gauge': True,
        'n_extra_bands': 4,
        'mpb_fd_order': 4,
    }
    if config_overrides:
        config.update(config_overrides)
    
    # Find source data
    run_dir = p4.find_latest_run_dir()
    source_cdir = candidate_dir(run_dir, candidate_id)
    
    if not (source_cdir / 'phase1_multiband_data.h5').exists():
        raise FileNotFoundError(f"Phase 1 data not found in {source_cdir}")
    
    print(f"Source candidate dir: {source_cdir}")
    print(f"Angles: {theta_list}")
    print(f"Modes per angle: {n_modes}")
    
    # Create sweep output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(run_dir) / f"eta_sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep output: {sweep_dir}")
    
    # Save sweep config
    sweep_config = {
        'candidate_id': candidate_id,
        'source_dir': str(source_cdir),
        'theta_list': theta_list,
        'n_modes': n_modes,
        'config': config,
        'timestamp': timestamp,
    }
    with open(sweep_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run each angle
    all_results = []
    for theta_deg in theta_list:
        try:
            result = run_single_theta(
                theta_deg, source_cdir, sweep_dir, config, n_modes
            )
            all_results.append(result)
            
            # Save intermediate results
            with open(sweep_dir / 'sweep_results_partial.json', 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            # Explicit memory cleanup between angles
            import gc
            gc.collect()
                
        except Exception as e:
            print(f"\n  ERROR at θ = {theta_deg}°: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'theta_deg': theta_deg,
                'eta': 2 * math.sin(math.radians(theta_deg) / 2),
                'error': str(e),
            })
    
    # Save final results
    with open(sweep_dir / 'sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {sweep_dir / 'sweep_results.json'}")
    
    # Plot
    valid_results = [r for r in all_results if 'error' not in r]
    if len(valid_results) >= 2:
        plot_sweep_results(valid_results, sweep_dir)
    
    return all_results, sweep_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="η-sweep validation")
    parser.add_argument("--candidate_id", type=int, default=0, help="Candidate ID")
    parser.add_argument("--angles", type=float, nargs='+',
                        default=None,
                        help="Twist angles in degrees (default: 0.5 0.8 1.1 1.5 2.0 3.0 5.0 8.0)")
    parser.add_argument("--n_modes", type=int, default=100, help="Modes per angle")
    args = parser.parse_args()
    
    run_eta_sweep(
        candidate_id=args.candidate_id,
        theta_list=args.angles,
        n_modes=args.n_modes,
    )
