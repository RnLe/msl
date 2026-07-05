#!/usr/bin/env python
"""
Convergence analysis for the moiré envelope approximation pipeline.

Tests three independent convergence axes at a fixed twist angle:
  A) n_modes (eigsh Krylov truncation):  k ∈ {10, 20, 50, 100}
  B) Grid resolution (Ns):              Ns ∈ {32, 48, 64, 96, 128}
  C) FD order:                          {2, 4}

For each lattice type (honeycomb, hex, square), runs Phase 2 at the chosen
angle in a temp directory, then sweeps the convergence parameters using only
Phase 3 (Hamiltonian assembly + eigsh).  Phase 2 is computed ONCE per lattice.

Usage:
    python thesis_results/T_convergence/convergence_test.py
    python thesis_results/T_convergence/convergence_test.py --theta 1.1
    python thesis_results/T_convergence/convergence_test.py --only honeycomb
    python thesis_results/T_convergence/convergence_test.py --skip_ns  # skip Ns sweep
"""

import sys, math, json, time, gc, argparse, shutil, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import eigsh

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase2_mpb_v3 as p2
import phase3_mpb_v3 as p3
from common.io_utils import load_json

try:
    sys.path.insert(0, str(THESIS_DIR))
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False

# ── lattice configurations ──
LATTICE_CONFIGS = {
    'honeycomb': {
        'run_dir_pattern': 'thesis_honeycomb_K_b1_2026*',
        'run_dir_exclude': 'TE',
        'lattice_type': 'honeycomb',
        'symmetry': 'C6',
        'a': 1.0,
    },
    'hex': {
        'run_dir_pattern': 'thesis_hex_M_b1_2026*',
        'run_dir_exclude': None,
        'lattice_type': 'hex',
        'symmetry': 'C2',
        'a': 1.0,
    },
    'square': {
        'run_dir_pattern': 'thesis_square_M_b3_2026*',
        'run_dir_exclude': None,
        'lattice_type': 'square',
        'symmetry': 'C4',
        'a': 1.0,
    },
}

OUTPUT_DIR = SCRIPT_DIR


# ── geometry ──

def compute_moire_params(theta_deg, lattice_type='honeycomb', a=1.0):
    theta_rad = math.radians(theta_deg)
    eta = 2 * math.sin(theta_rad / 2)
    if lattice_type == 'square':
        B_mono = np.array([[a, 0.0], [0.0, a]])
    else:
        B_mono = np.array([[a, 0.0], [a / 2.0, a * math.sqrt(3) / 2.0]])
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R_theta = np.array([[c, -s], [s, c]])
    B_moire = np.linalg.inv(R_theta - np.eye(2)) @ B_mono
    moire_length = np.linalg.norm(B_moire[:, 0])
    return {
        'theta_deg': theta_deg, 'theta_rad': theta_rad,
        'eta': eta, 'B_moire': B_moire, 'moire_length': moire_length,
    }


def patch_h5_theta(h5_path, moire_params):
    with h5py.File(h5_path, 'r+') as hf:
        hf.attrs['theta_deg'] = moire_params['theta_deg']
        hf.attrs['theta_rad'] = moire_params['theta_rad']
        hf.attrs['eta'] = moire_params['eta']
        hf.attrs['moire_length'] = moire_params['moire_length']
        hf.attrs['B_moire'] = moire_params['B_moire']
        if 'R_grid' in hf and 's_grid' in hf:
            s_grid = hf['s_grid'][:]
            R_new = np.einsum('ij,...j->...i', moire_params['B_moire'], s_grid)
            hf['R_grid'][...] = R_new


def patch_meta_theta(meta_path, moire_params):
    with open(meta_path) as f:
        meta = json.load(f)
    meta['theta_deg'] = moire_params['theta_deg']
    meta['theta_rad'] = moire_params['theta_rad']
    meta['eta'] = moire_params['eta']
    meta['moire_length'] = moire_params['moire_length']
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ── find run directories ──

def find_source_dir(lattice_name):
    """Find the Phase 1 run directory for a lattice candidate."""
    cfg = LATTICE_CONFIGS[lattice_name]
    base = PROJECT_ROOT / "runsV3"
    candidates = sorted(base.glob(cfg['run_dir_pattern']))
    if cfg['run_dir_exclude']:
        candidates = [c for c in candidates if cfg['run_dir_exclude'] not in c.name]
    if not candidates:
        return None
    return candidates[-1] / "candidate_0000"


# ── Phase 2 computation ──

def compute_phase2_at_angle(theta_deg, source_cdir, lattice_name):
    """
    Run Phase 2 at the given angle.  Returns the temp directory path (caller
    must clean up) and the path to the Phase 2 HDF5.
    """
    cfg = LATTICE_CONFIGS[lattice_name]
    lattice_type = cfg['lattice_type']
    symmetry = cfg['symmetry']
    a = cfg['a']

    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    phase1_src = source_cdir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"

    tmp_base = tempfile.mkdtemp(prefix=f"conv_{lattice_name}_{theta_deg:.2f}_")
    work_dir = Path(tmp_base) / "candidate_0000"
    work_dir.mkdir(parents=True)

    # Smart-copy Phase 1
    phase1_dst = work_dir / "phase1_multiband_data.h5"
    with h5py.File(phase1_src, 'r') as src, h5py.File(phase1_dst, 'w') as dst:
        for key, val in src.attrs.items():
            dst.attrs[key] = val
        for key in src.keys():
            obj = src[key]
            if isinstance(obj, h5py.Dataset) and obj.nbytes > 1e9:
                dst[key] = h5py.ExternalLink(str(phase1_src), f'/{key}')
            else:
                src.copy(key, dst)

    shutil.copy2(meta_src, work_dir / "phase0_meta.json")
    patch_h5_theta(phase1_dst, moire_params)
    patch_meta_theta(work_dir / "phase0_meta.json", moire_params)

    # Run Phase 2
    p2_config = {
        'include_born_huang': False,
        'include_drift_term': True,
        'use_parallel_transport_gauge': True,
        'n_extra_bands': 4,
        'mpb_fd_order': 4,
    }
    p2.process_candidate_phase2_v3(str(work_dir), p2_config)

    # Symmetrize
    if HAS_SYMMETRIZE and symmetry:
        try:
            symmetrize_phase2(work_dir, symmetry)
        except Exception as e:
            print(f"  WARNING: Symmetrization ({symmetry}) failed: {e}")

    gc.collect()
    return tmp_base, work_dir


# ── Phase 2 data loading ──

def load_phase2_data(work_dir):
    """Load Phase 2 data from HDF5.  Returns a dict of arrays + metadata."""
    phase2_h5 = work_dir / "phase2_multiband_data.h5"
    with h5py.File(phase2_h5, 'r') as hf:
        data = {
            's_grid': hf['s_grid'][:],
            'R_grid': hf['R_grid'][:],
            'Lambda': hf['Lambda'][:],
            'A_berry': hf['A_berry'][:],
            'Phi_BH': hf['Phi_BH'][:],
            'v_drift': hf['v_drift'][:],
            'M_inv': hf['M_inv'][:],
            'omega_grid': hf['omega'][:],
            'V_grid': hf['V'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta': float(hf.attrs['eta']),
            'theta_rad': float(hf.attrs['theta_rad']),
            'Ns1': int(hf.attrs['Ns1']),
            'Ns2': int(hf.attrs['Ns2']),
            'N_subspace': int(hf.attrs['N_subspace']),
            'target_idx': int(hf.attrs['target_index_in_subspace']),
            'B_moire': hf.attrs['B_moire'][:],
            'B_mono': hf.attrs['B_mono'][:],
            'subspace_bands': hf.attrs['subspace_bands'][:].tolist(),
        }
    return data


# ── Downsampling ──

def downsample_phase2(data, Ns_target):
    """
    Downsample Phase 2 tensor fields from (Ns1, Ns2) to (Ns_target, Ns_target)
    using bilinear interpolation on the periodic s-grid.
    """
    Ns_orig = data['Ns1']
    if Ns_target == Ns_orig:
        return data  # no-op

    # Original s-coordinates (periodic [0, 1))
    s1_orig = np.linspace(0, 1, Ns_orig, endpoint=False)
    s2_orig = np.linspace(0, 1, Ns_orig, endpoint=False)
    s1_new = np.linspace(0, 1, Ns_target, endpoint=False)
    s2_new = np.linspace(0, 1, Ns_target, endpoint=False)

    out = dict(data)
    out['Ns1'] = Ns_target
    out['Ns2'] = Ns_target

    # Fields to interpolate and their trailing shapes
    fields = {
        'Lambda': data['Lambda'],      # (Ns, Ns, Nb, Nb)
        'A_berry': data['A_berry'],     # (Ns, Ns, Nb, Nb, 2)
        'Phi_BH': data['Phi_BH'],       # (Ns, Ns, Nb, Nb)
        'v_drift': data['v_drift'],     # (Ns, Ns, Nb, Nb, 2)
        'M_inv': data['M_inv'],          # (Ns, Ns, Nb, Nb, 2, 2)
    }

    for name, arr in fields.items():
        orig_shape = arr.shape
        trailing = orig_shape[2:]
        # Flatten trailing dims
        flat = arr.reshape(Ns_orig, Ns_orig, -1)
        n_comp = flat.shape[2]

        new_flat = np.empty((Ns_target, Ns_target, n_comp), dtype=arr.dtype)
        for c in range(n_comp):
            interp = RegularGridInterpolator(
                (s1_orig, s2_orig), flat[:, :, c],
                method='linear', bounds_error=False, fill_value=None,
            )
            # Create meshgrid of new coordinates
            s1g, s2g = np.meshgrid(s1_new, s2_new, indexing='ij')
            pts = np.stack([s1g.ravel(), s2g.ravel()], axis=-1)
            new_flat[:, :, c] = interp(pts).reshape(Ns_target, Ns_target)

        out[name] = new_flat.reshape((Ns_target, Ns_target) + trailing)

    # Recompute s_grid and R_grid
    s1g, s2g = np.meshgrid(s1_new, s2_new, indexing='ij')
    out['s_grid'] = np.stack([s1g, s2g], axis=-1)
    out['R_grid'] = np.einsum('ij,...j->...i', data['B_moire'], out['s_grid'])

    return out


# ── Hamiltonian assembly + eigsh (uses phase3 API directly) ──

def assemble_and_solve(data, n_modes, fd_order=4, sigma=None, candidate_type=None):
    """
    Assemble the multiband Hamiltonian and solve for n_modes eigenvalues.
    Returns (eigenvalues, sigma_used, wall_time_s).
    """
    Ns1, Ns2 = data['Ns1'], data['Ns2']
    Nb = data['N_subspace']
    eta = data['eta']
    B_moire = data['B_moire']
    target_idx = data['target_idx']

    L_moire = np.linalg.norm(B_moire[0])
    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2

    t0 = time.time()

    H = p3.assemble_multiband_hamiltonian(
        data['Lambda'], data['v_drift'], data['M_inv'],
        data['A_berry'], data['Phi_BH'],
        eta, Ns1, Ns2, Nb, dR1, dR2, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
        order=fd_order, include_offdiag_A=True,
    )

    # Auto-select sigma via compute_sigma()
    if sigma is None:
        sigma, sigma_info = p3.compute_sigma(
            data['Lambda'], data['M_inv'], target_idx,
            candidate_type=candidate_type,
        )

    eigenvalues, eigenvectors = p3.solve_multiband_envelope(H, n_modes, sigma=sigma)
    wall_time = time.time() - t0

    return eigenvalues, sigma, wall_time


# ── Test A: n_modes convergence ──

def test_nmodes_convergence(data, lattice_name, k_values=(10, 20, 50, 100)):
    """
    Assemble H once at full resolution, solve for varying k = n_modes.
    """
    print(f"\n{'='*70}")
    print(f"  TEST A: n_modes convergence — {lattice_name}")
    print(f"  k values: {k_values}")
    print(f"  Grid: {data['Ns1']}×{data['Ns2']}, Nb={data['N_subspace']}")
    N_total = data['Ns1'] * data['Ns2'] * data['N_subspace']
    print(f"  Matrix size: {N_total}×{N_total}")
    print(f"{'='*70}")

    results = []
    sigma_fixed = None

    for k in k_values:
        print(f"\n  → k = {k} ...", flush=True)
        evals, sigma_used, wt = assemble_and_solve(data, n_modes=k, sigma=sigma_fixed)
        if sigma_fixed is None:
            sigma_fixed = sigma_used  # lock sigma for fair comparison
        print(f"    λ₀ = {evals[0]:.8e}, gap₀₁ = {evals[1]-evals[0]:.4e}, "
              f"σ = {sigma_used:.4f}, t = {wt:.1f}s")
        results.append({
            'k': k,
            'eigenvalues': evals.tolist(),
            'sigma': sigma_used,
            'wall_time_s': wt,
        })

    # Analysis
    ref = results[-1]  # highest k as reference
    ref_evals = np.array(ref['eigenvalues'])
    print(f"\n  ── Analysis (reference: k={ref['k']}) ──")
    for r in results:
        evals = np.array(r['eigenvalues'])
        k = r['k']
        lam0 = evals[0]
        gap01 = evals[1] - evals[0]
        ref_gap01 = ref_evals[1] - ref_evals[0]

        # Compare first min(k, k_ref) eigenvalues
        n_common = min(len(evals), len(ref_evals))
        common_evals = evals[:n_common]
        common_ref = ref_evals[:n_common]
        max_abs_diff = float(np.max(np.abs(common_evals - common_ref)))
        max_rel_diff = float(np.max(np.abs(common_evals - common_ref) /
                                     (np.abs(common_ref) + 1e-20)))

        print(f"  k={k:4d}: λ₀={lam0:.8e}  gap₀₁={gap01:.4e}  "
              f"Δgap/gap={abs(gap01-ref_gap01)/(abs(ref_gap01)+1e-20):.2e}  "
              f"max|Δλ|={max_abs_diff:.2e}  max|Δλ/λ|={max_rel_diff:.2e}")

    return results


# ── Test B: Grid resolution (Ns) convergence ──

def test_ns_convergence(data_full, lattice_name, ns_values=(32, 48, 64, 96, 128)):
    """
    Downsample Phase 2 data to varying Ns, assemble H and solve at fixed k=20.
    """
    print(f"\n{'='*70}")
    print(f"  TEST B: Grid resolution convergence — {lattice_name}")
    print(f"  Ns values: {ns_values}")
    print(f"{'='*70}")

    k_fixed = 20
    sigma_fixed = None
    results = []

    for ns in ns_values:
        print(f"\n  → Ns = {ns} ...", flush=True)
        data_ds = downsample_phase2(data_full, ns)
        N_total = ns * ns * data_ds['N_subspace']
        print(f"    Matrix: {N_total}×{N_total}")

        evals, sigma_used, wt = assemble_and_solve(
            data_ds, n_modes=k_fixed, sigma=sigma_fixed)

        if sigma_fixed is None:
            sigma_fixed = sigma_used
        print(f"    λ₀ = {evals[0]:.8e}, gap₀₁ = {evals[1]-evals[0]:.4e}, "
              f"t = {wt:.1f}s")
        results.append({
            'Ns': ns,
            'eigenvalues': evals.tolist(),
            'sigma': sigma_used,
            'wall_time_s': wt,
        })

    # Analysis (reference = highest Ns)
    ref = results[-1]
    ref_evals = np.array(ref['eigenvalues'])
    print(f"\n  ── Analysis (reference: Ns={ref['Ns']}) ──")
    for r in results:
        evals = np.array(r['eigenvalues'])
        ns = r['Ns']
        lam0 = evals[0]
        gap01 = evals[1] - evals[0]
        ref_gap01 = ref_evals[1] - ref_evals[0]

        n_common = min(len(evals), len(ref_evals))
        max_abs_diff = float(np.max(np.abs(evals[:n_common] - ref_evals[:n_common])))
        max_rel_diff = float(np.max(np.abs(evals[:n_common] - ref_evals[:n_common]) /
                                     (np.abs(ref_evals[:n_common]) + 1e-20)))

        print(f"  Ns={ns:4d}: λ₀={lam0:.8e}  gap₀₁={gap01:.4e}  "
              f"Δgap/gap={abs(gap01-ref_gap01)/(abs(ref_gap01)+1e-20):.2e}  "
              f"max|Δλ|={max_abs_diff:.2e}  max|Δλ/λ|={max_rel_diff:.2e}")

    return results


# ── Test C: FD order comparison ──

def test_fd_order(data, lattice_name, orders=(2, 4)):
    """
    Assemble H with different FD orders, compare eigenvalues.
    """
    print(f"\n{'='*70}")
    print(f"  TEST C: FD order convergence — {lattice_name}")
    print(f"  Orders: {orders}")
    print(f"{'='*70}")

    k_fixed = 20
    sigma_fixed = None
    results = []

    for order in orders:
        print(f"\n  → FD order = {order} ...", flush=True)
        evals, sigma_used, wt = assemble_and_solve(
            data, n_modes=k_fixed, fd_order=order, sigma=sigma_fixed)
        if sigma_fixed is None:
            sigma_fixed = sigma_used
        print(f"    λ₀ = {evals[0]:.8e}, gap₀₁ = {evals[1]-evals[0]:.4e}, "
              f"t = {wt:.1f}s")
        results.append({
            'fd_order': order,
            'eigenvalues': evals.tolist(),
            'sigma': sigma_used,
            'wall_time_s': wt,
        })

    # Compare
    if len(results) >= 2:
        e2 = np.array(results[0]['eigenvalues'])
        e4 = np.array(results[-1]['eigenvalues'])
        n_common = min(len(e2), len(e4))
        max_abs = float(np.max(np.abs(e2[:n_common] - e4[:n_common])))
        max_rel = float(np.max(np.abs(e2[:n_common] - e4[:n_common]) /
                                (np.abs(e4[:n_common]) + 1e-20)))
        gap2 = e2[1] - e2[0]
        gap4 = e4[1] - e4[0]
        print(f"\n  ── order=2 vs order=4 ──")
        print(f"  max|Δλ|     = {max_abs:.4e}")
        print(f"  max|Δλ/λ|   = {max_rel:.4e}")
        print(f"  Δgap/gap    = {abs(gap2-gap4)/(abs(gap4)+1e-20):.4e}")

    return results


# ── Test D: Sigma sensitivity ──

def test_sigma_sensitivity(data, lattice_name):
    """
    Run eigsh with three different sigma values to verify eigenvalue independence.
    """
    print(f"\n{'='*70}")
    print(f"  TEST D: Sigma sensitivity — {lattice_name}")
    print(f"{'='*70}")

    k_fixed = 20
    target_idx = data['target_idx']
    V_target = data['Lambda'][..., target_idx, target_idx]
    V_min = float(np.min(V_target))
    V_max = float(np.max(V_target))
    V_mean = float(np.mean(V_target))

    # Determine band type
    M_inv = data['M_inv']
    tr = M_inv[..., target_idx, target_idx, 0, 0] + M_inv[..., target_idx, target_idx, 1, 1]
    is_hole = float(np.mean(tr)) < 0
    # Auto sigma
    sigma_auto = V_max if is_hole else V_min

    sigmas = {
        'V_min': V_min,
        'V_mean': V_mean,
        'V_max': V_max,
        'auto': sigma_auto,
        'auto-0.01': sigma_auto - 0.01,
    }

    results = []
    for label, sig in sigmas.items():
        print(f"\n  → σ = {sig:.6f} ({label}) ...", flush=True)
        try:
            evals, _, wt = assemble_and_solve(data, n_modes=k_fixed, sigma=sig)
            evals_sorted = np.sort(evals)
            print(f"    λ₀ = {evals_sorted[0]:.8e}, gap₀₁ = {evals_sorted[1]-evals_sorted[0]:.4e}, "
                  f"t = {wt:.1f}s")
            results.append({
                'sigma_label': label,
                'sigma': sig,
                'eigenvalues': evals_sorted.tolist(),
                'wall_time_s': wt,
            })
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'sigma_label': label, 'sigma': sig, 'error': str(e),
            })

    # Compare all successful runs against 'auto'
    auto_result = next((r for r in results if r['sigma_label'] == 'auto' and 'eigenvalues' in r), None)
    if auto_result:
        ref_evals = np.array(auto_result['eigenvalues'])
        print(f"\n  ── Comparison vs auto (σ={auto_result['sigma']:.4f}) ──")
        for r in results:
            if 'eigenvalues' not in r:
                continue
            evals = np.array(r['eigenvalues'])
            n_c = min(len(evals), len(ref_evals))
            max_diff = float(np.max(np.abs(evals[:n_c] - ref_evals[:n_c])))
            print(f"  {r['sigma_label']:12s}: max|Δλ| = {max_diff:.4e}")

    return results


# ── Plotting ──

def generate_convergence_plots(all_results, output_dir):
    """Generate thesis-quality convergence plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lattice_colors = {'honeycomb': '#1f77b4', 'hex': '#ff7f0e', 'square': '#2ca02c'}
    lattice_markers = {'honeycomb': 'o', 'hex': 's', 'square': '^'}

    # ── Plot 1: n_modes convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title("Ground state λ₀ vs n_modes", fontsize=12, fontweight='bold')
    for lname, ldata in all_results.items():
        if 'nmodes' not in ldata:
            continue
        ks = [r['k'] for r in ldata['nmodes']]
        lam0s = [r['eigenvalues'][0] for r in ldata['nmodes']]
        ax.plot(ks, lam0s, f'-{lattice_markers[lname]}',
                color=lattice_colors[lname], label=lname, markersize=8)
    ax.set_xlabel("n_modes (k)")
    ax.set_ylabel("λ₀")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Gap₀₁ vs n_modes", fontsize=12, fontweight='bold')
    for lname, ldata in all_results.items():
        if 'nmodes' not in ldata:
            continue
        ks = [r['k'] for r in ldata['nmodes']]
        gaps = [r['eigenvalues'][1] - r['eigenvalues'][0] for r in ldata['nmodes']]
        ax.semilogy(ks, gaps, f'-{lattice_markers[lname]}',
                     color=lattice_colors[lname], label=lname, markersize=8)
    ax.set_xlabel("n_modes (k)")
    ax.set_ylabel("Gap₀₁")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_nmodes.pdf", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "convergence_nmodes.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: convergence_nmodes.pdf/png")

    # ── Plot 2: Ns convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title("Ground state λ₀ vs grid size Ns", fontsize=12, fontweight='bold')
    for lname, ldata in all_results.items():
        if 'ns' not in ldata:
            continue
        nss = [r['Ns'] for r in ldata['ns']]
        lam0s = [r['eigenvalues'][0] for r in ldata['ns']]
        ax.plot(nss, lam0s, f'-{lattice_markers[lname]}',
                color=lattice_colors[lname], label=lname, markersize=8)
    ax.set_xlabel("Ns (grid points per axis)")
    ax.set_ylabel("λ₀")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("Gap₀₁ vs grid size Ns", fontsize=12, fontweight='bold')
    for lname, ldata in all_results.items():
        if 'ns' not in ldata:
            continue
        nss = [r['Ns'] for r in ldata['ns']]
        gaps = [r['eigenvalues'][1] - r['eigenvalues'][0] for r in ldata['ns']]
        ax.semilogy(nss, gaps, f'-{lattice_markers[lname]}',
                     color=lattice_colors[lname], label=lname, markersize=8)
    ax.set_xlabel("Ns")
    ax.set_ylabel("Gap₀₁")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_ns.pdf", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "convergence_ns.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: convergence_ns.pdf/png")

    # ── Plot 3: Combined summary ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Convergence Summary", fontsize=14, fontweight='bold')

    # (0,0): λ₀ vs k
    ax = axes[0, 0]
    for lname, ldata in all_results.items():
        if 'nmodes' not in ldata:
            continue
        ks = [r['k'] for r in ldata['nmodes']]
        lam0s = [r['eigenvalues'][0] for r in ldata['nmodes']]
        ax.plot(ks, lam0s, f'-{lattice_markers[lname]}',
                color=lattice_colors[lname], label=lname, markersize=6)
    ax.set_xlabel("n_modes")
    ax.set_ylabel("λ₀")
    ax.set_title("λ₀ vs n_modes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1): gap vs k
    ax = axes[0, 1]
    for lname, ldata in all_results.items():
        if 'nmodes' not in ldata:
            continue
        ks = [r['k'] for r in ldata['nmodes']]
        gaps = [r['eigenvalues'][1] - r['eigenvalues'][0] for r in ldata['nmodes']]
        ax.semilogy(ks, gaps, f'-{lattice_markers[lname]}',
                     color=lattice_colors[lname], label=lname, markersize=6)
    ax.set_xlabel("n_modes")
    ax.set_ylabel("Gap₀₁")
    ax.set_title("Gap₀₁ vs n_modes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0): λ₀ vs Ns
    ax = axes[1, 0]
    for lname, ldata in all_results.items():
        if 'ns' not in ldata:
            continue
        nss = [r['Ns'] for r in ldata['ns']]
        lam0s = [r['eigenvalues'][0] for r in ldata['ns']]
        ax.plot(nss, lam0s, f'-{lattice_markers[lname]}',
                color=lattice_colors[lname], label=lname, markersize=6)
    ax.set_xlabel("Ns")
    ax.set_ylabel("λ₀")
    ax.set_title("λ₀ vs Ns")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1): gap vs Ns
    ax = axes[1, 1]
    for lname, ldata in all_results.items():
        if 'ns' not in ldata:
            continue
        nss = [r['Ns'] for r in ldata['ns']]
        gaps = [r['eigenvalues'][1] - r['eigenvalues'][0] for r in ldata['ns']]
        ax.semilogy(nss, gaps, f'-{lattice_markers[lname]}',
                     color=lattice_colors[lname], label=lname, markersize=6)
    ax.set_xlabel("Ns")
    ax.set_ylabel("Gap₀₁")
    ax.set_title("Gap₀₁ vs Ns")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_summary.pdf", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / "convergence_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: convergence_summary.pdf/png")


# ── Report generation ──

def generate_report(all_results, theta_deg, output_dir):
    """Generate a plain-text convergence report."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"  CONVERGENCE REPORT — θ = {theta_deg}°")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    for lname, ldata in all_results.items():
        lines.append(f"\n{'─'*72}")
        lines.append(f"  Lattice: {lname}")
        lines.append(f"  Nb = {ldata.get('N_subspace', '?')}, η = {ldata.get('eta', '?'):.4f}")
        lines.append(f"{'─'*72}")

        # n_modes
        if 'nmodes' in ldata:
            lines.append(f"\n  A) n_modes convergence (Ns={ldata.get('Ns_full', '?')}):")
            ref = ldata['nmodes'][-1]
            ref_gap = ref['eigenvalues'][1] - ref['eigenvalues'][0]
            lines.append(f"     Reference: k={ref['k']}")
            lines.append(f"     {'k':>5s}  {'λ₀':>14s}  {'gap₀₁':>12s}  {'Δgap/gap':>10s}  {'max|Δλ/λ|':>10s}")
            for r in ldata['nmodes']:
                evals = np.array(r['eigenvalues'])
                ref_evals = np.array(ref['eigenvalues'])
                gap = evals[1] - evals[0]
                n_c = min(len(evals), len(ref_evals))
                max_rel = float(np.max(np.abs(evals[:n_c] - ref_evals[:n_c]) /
                                       (np.abs(ref_evals[:n_c]) + 1e-20)))
                dgap = abs(gap - ref_gap) / (abs(ref_gap) + 1e-20)
                lines.append(f"     {r['k']:5d}  {evals[0]:14.8e}  {gap:12.4e}  {dgap:10.2e}  {max_rel:10.2e}")

        # Ns
        if 'ns' in ldata:
            lines.append(f"\n  B) Grid resolution convergence (k=20, fd_order=4):")
            ref = ldata['ns'][-1]
            ref_gap = ref['eigenvalues'][1] - ref['eigenvalues'][0]
            lines.append(f"     Reference: Ns={ref['Ns']}")
            lines.append(f"     {'Ns':>5s}  {'λ₀':>14s}  {'gap₀₁':>12s}  {'Δgap/gap':>10s}  {'max|Δλ/λ|':>10s}")
            for r in ldata['ns']:
                evals = np.array(r['eigenvalues'])
                ref_evals = np.array(ref['eigenvalues'])
                gap = evals[1] - evals[0]
                n_c = min(len(evals), len(ref_evals))
                max_rel = float(np.max(np.abs(evals[:n_c] - ref_evals[:n_c]) /
                                       (np.abs(ref_evals[:n_c]) + 1e-20)))
                dgap = abs(gap - ref_gap) / (abs(ref_gap) + 1e-20)
                lines.append(f"     {r['Ns']:5d}  {evals[0]:14.8e}  {gap:12.4e}  {dgap:10.2e}  {max_rel:10.2e}")

        # FD order
        if 'fd_order' in ldata:
            lines.append(f"\n  C) FD order comparison (Ns={ldata.get('Ns_full', '?')}, k=20):")
            for r in ldata['fd_order']:
                evals = np.array(r['eigenvalues'])
                lines.append(f"     order={r['fd_order']}: λ₀={evals[0]:.8e}  gap₀₁={evals[1]-evals[0]:.4e}")
            if len(ldata['fd_order']) >= 2:
                e2 = np.array(ldata['fd_order'][0]['eigenvalues'])
                e4 = np.array(ldata['fd_order'][-1]['eigenvalues'])
                n_c = min(len(e2), len(e4))
                max_rel = float(np.max(np.abs(e2[:n_c] - e4[:n_c]) /
                                       (np.abs(e4[:n_c]) + 1e-20)))
                lines.append(f"     max|Δλ/λ| (order 2 vs 4) = {max_rel:.4e}")

        # Sigma
        if 'sigma' in ldata:
            lines.append(f"\n  D) Sigma sensitivity (Ns={ldata.get('Ns_full', '?')}, k=20):")
            for r in ldata['sigma']:
                if 'eigenvalues' in r:
                    evals = np.array(r['eigenvalues'])
                    lines.append(f"     σ={r['sigma']:.6f} ({r['sigma_label']:12s}): "
                                 f"λ₀={evals[0]:.8e}  gap₀₁={evals[1]-evals[0]:.4e}")
                else:
                    lines.append(f"     σ={r['sigma']:.6f} ({r['sigma_label']:12s}): FAILED")

    # ── Classification ──
    lines.append(f"\n{'='*72}")
    lines.append("  CONVERGENCE CLASSIFICATION")
    lines.append(f"{'='*72}")

    for lname, ldata in all_results.items():
        lines.append(f"\n  {lname}:")

        # n_modes classification
        if 'nmodes' in ldata and len(ldata['nmodes']) >= 2:
            ref = ldata['nmodes'][-1]
            ref_gap = ref['eigenvalues'][1] - ref['eigenvalues'][0]
            # Compare k=20 vs k=100
            k20 = next((r for r in ldata['nmodes'] if r['k'] == 20), None)
            if k20:
                gap20 = k20['eigenvalues'][1] - k20['eigenvalues'][0]
                dgap = abs(gap20 - ref_gap) / (abs(ref_gap) + 1e-20)
                if dgap < 1e-3:
                    verdict = "CONVERGED (Δgap/gap < 0.1%)"
                elif dgap < 1e-2:
                    verdict = "ADEQUATE (Δgap/gap < 1%)"
                else:
                    verdict = f"SENSITIVE (Δgap/gap = {dgap:.1%})"
                lines.append(f"    n_modes: {verdict}")

        # Ns classification
        if 'ns' in ldata and len(ldata['ns']) >= 2:
            ref = ldata['ns'][-1]
            ref_gap = ref['eigenvalues'][1] - ref['eigenvalues'][0]
            ns64 = next((r for r in ldata['ns'] if r['Ns'] == 64), None)
            if ns64:
                gap64 = ns64['eigenvalues'][1] - ns64['eigenvalues'][0]
                dgap = abs(gap64 - ref_gap) / (abs(ref_gap) + 1e-20)
                if dgap < 1e-3:
                    verdict = "CONVERGED at Ns=64 (Δgap/gap < 0.1%)"
                elif dgap < 1e-2:
                    verdict = "ADEQUATE at Ns=64 (Δgap/gap < 1%)"
                else:
                    verdict = f"NOT CONVERGED at Ns=64 (Δgap/gap = {dgap:.1%})"
                lines.append(f"    Ns:      {verdict}")

    report = "\n".join(lines)
    report_path = output_dir / "CONVERGENCE_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")
    print(report)
    return report


# ── main ──

def main():
    parser = argparse.ArgumentParser(description="Convergence analysis for moiré envelope pipeline")
    parser.add_argument("--theta", type=float, default=1.1, help="Twist angle (degrees)")
    parser.add_argument("--only", type=str, default=None,
                        choices=['honeycomb', 'hex', 'square'],
                        help="Run only one lattice type")
    parser.add_argument("--skip_ns", action='store_true', help="Skip Ns sweep (Test B)")
    parser.add_argument("--skip_nmodes", action='store_true', help="Skip n_modes sweep (Test A)")
    parser.add_argument("--skip_fd", action='store_true', help="Skip FD order test (Test C)")
    parser.add_argument("--skip_sigma", action='store_true', help="Skip sigma sensitivity test (Test D)")
    parser.add_argument("--k_values", type=str, default="10,20,50,100",
                        help="Comma-separated k values for n_modes test")
    parser.add_argument("--ns_values", type=str, default="32,48,64,96,128",
                        help="Comma-separated Ns values for grid test")
    parser.add_argument("--use_existing_phase2", action='store_true',
                        help="Load Phase 2 data from run directory instead of recomputing. "
                             "Requires that Phase 2 data exists at the stored angle.")
    args = parser.parse_args()

    theta_deg = args.theta
    k_values = tuple(int(x) for x in args.k_values.split(','))
    ns_values = tuple(int(x) for x in args.ns_values.split(','))

    lattices = [args.only] if args.only else ['honeycomb', 'hex', 'square']

    print(f"\n{'#'*72}")
    print(f"  CONVERGENCE ANALYSIS")
    print(f"  θ = {theta_deg}°, lattices: {lattices}")
    print(f"  k values: {k_values}")
    print(f"  Ns values: {ns_values}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*72}\n")

    all_results = {}

    for lname in lattices:
        source_cdir = find_source_dir(lname)
        if source_cdir is None:
            print(f"\n  SKIP {lname}: no run directory found")
            continue

        print(f"\n{'*'*72}")
        print(f"  LATTICE: {lname}")
        print(f"  Source: {source_cdir}")
        print(f"{'*'*72}")

        # Load Phase 2 data
        tmp_base = None
        if args.use_existing_phase2:
            # Load directly from run directory (avoids 25+ GB memory for Phase 2 recompute)
            print(f"\n  Loading existing Phase 2 data from run directory...")
            phase2_path = source_cdir / "phase2_multiband_data.h5"
            if not phase2_path.exists():
                print(f"  ERROR: No Phase 2 data at {phase2_path}")
                continue
            data = load_phase2_data(source_cdir)
            stored_theta = math.degrees(data['theta_rad'])
            print(f"  Loaded Phase 2: θ={stored_theta:.2f}°, Ns={data['Ns1']}×{data['Ns2']}, Nb={data['N_subspace']}")
            if abs(stored_theta - theta_deg) > 0.05:
                print(f"  WARNING: Stored θ={stored_theta:.2f}° differs from requested {theta_deg}°")
        else:
            print(f"\n  Computing Phase 2 at θ={theta_deg}°...")
            t0 = time.time()
            tmp_base, work_dir = compute_phase2_at_angle(theta_deg, source_cdir, lname)
            print(f"  Phase 2 done in {time.time() - t0:.1f}s")
            data = load_phase2_data(work_dir)

        try:
            lresults = {
                'N_subspace': data['N_subspace'],
                'eta': data['eta'],
                'Ns_full': data['Ns1'],
            }

            # Test A: n_modes
            if not args.skip_nmodes:
                lresults['nmodes'] = test_nmodes_convergence(data, lname, k_values)
                gc.collect()

            # Test B: Ns
            if not args.skip_ns:
                lresults['ns'] = test_ns_convergence(data, lname, ns_values)
                gc.collect()

            # Test C: FD order
            if not args.skip_fd:
                lresults['fd_order'] = test_fd_order(data, lname)
                gc.collect()

            # Test D: Sigma sensitivity
            if not args.skip_sigma:
                lresults['sigma'] = test_sigma_sensitivity(data, lname)
                gc.collect()

            all_results[lname] = lresults

        finally:
            if tmp_base:
                shutil.rmtree(tmp_base, ignore_errors=True)
            gc.collect()

    # Save raw JSON results (merge with existing if present)
    json_path = OUTPUT_DIR / "convergence_results.json"
    # Convert numpy types for JSON serialization
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    # Merge with existing results from previous runs
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
        existing_results = existing.get('results', {})
        # New results override existing for same lattice
        existing_results.update(sanitize(all_results))
        all_results_merged = existing_results
    else:
        all_results_merged = sanitize(all_results)

    with open(json_path, 'w') as f:
        json.dump(sanitize({
            'theta_deg': theta_deg,
            'timestamp': datetime.now().isoformat(),
            'results': all_results_merged,
        }), f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # Use merged results for plots and report
    all_results = all_results_merged

    # Generate plots
    print(f"\n  Generating plots...")
    generate_convergence_plots(all_results, OUTPUT_DIR)

    # Generate report
    generate_report(all_results, theta_deg, OUTPUT_DIR)


if __name__ == '__main__':
    main()
