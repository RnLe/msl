#!/usr/bin/env python3
"""
Phase A Diagnostic: Isolate EA bandwidth compression root cause.
================================================================

Test 1: Single-band EA (SUBSPACE_BANDS=[3], N_bands=1)
  - No tracking ambiguity (single band, no permutation)
  - If BW matches FDFD → tracking confirmed as root cause

Test 2: Multi-band EA with energy-sort tracking (instead of overlap-Hungarian)
  - Same config as overnight, but bands sorted by ω at each registry point
  - If BW improves → overlap tracking is producing bad permutations

Test 3: Multi-band EA with NO tracking (raw MPB band order)
  - Baseline: what does the raw MPB ordering give us?

All tests compare to existing FDFD data from the overnight run.
Uses 24×24 registry for speed (overnight used 48×48).
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import gc
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Path setup ──
SCRIPT_DIR = Path(__file__).resolve().parent
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT))
sys.path.insert(0, str(MOIRE_ROOT / 'phasesV3'))
sys.path.insert(0, str(SCRIPT_DIR.parent))  # thesis_results/ for T_direct_validation.*

from common.io_utils import candidate_dir, choose_reference_frequency, save_json
from phasesV3.bloch_fields import save_bloch_fields
from phase1_mpb_v3 import (
    build_fractional_grid,
    build_monolayer_basis,
    compute_eta_physics,
    extract_multiband_data_from_mpb_v3,
    fractional_to_cartesian,
    run_mpb_registry_sweep,
)
from phase2_mpb_v3 import process_candidate_phase2_v3
from phase3_mpb_v3 import process_candidate_phase3_v3

# ═══════════════════════════════════════════════════════════════
#  Physical constants (same as overnight run)
# ═══════════════════════════════════════════════════════════════
A = 1.0
R_OVER_A = 0.2
EPS_BG = 1.0
EPS_HOLE = 11.56
OMEGA0 = 0.68457
TARGET_BAND = 3
N_WORKERS = 16

# Test case: 10° (fastest angle, existing FDFD data)
TEST_CASE = {
    'name': '10deg',
    'm': 11, 'n': 1,
    'fdfd_res': 64,
    'ea_registry': 24,   # Reduced from 48 for speed
    'ea_ns': 128,
    'mpb_resolution': 64,
    'n_modes': 80,
}

# Overnight run directory (for FDFD reference data)
OVERNIGHT_DIR = SCRIPT_DIR / 'overnight_validation' / 'run_20260313_004032'

# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════
_log_file = None

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    if _log_file:
        _log_file.write(line + '\n')
        _log_file.flush()


# ═══════════════════════════════════════════════════════════════
#  Geometry helpers (same as overnight)
# ═══════════════════════════════════════════════════════════════

def theta_from_mn(m_idx, n_idx):
    theta_rad = 2.0 * math.atan2(n_idx, m_idx)
    return theta_rad, math.degrees(theta_rad)


def build_candidate(case, subspace_bands, all_bands):
    """Build candidate dict with specified band configuration."""
    m_idx, n_idx = case['m'], case['n']
    theta_rad, theta_deg = theta_from_mn(m_idx, n_idx)
    l1 = np.array([m_idx, n_idx], dtype=float) * A
    l2 = np.array([-n_idx, m_idx], dtype=float) * A
    B_super = np.column_stack([l1, l2])
    moire_length = float(np.linalg.norm(l1))
    eta = compute_eta_physics(theta_rad)
    return {
        'candidate_id': 0,
        'lattice_type': 'square',
        'a': A,
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_HOLE,
        'band_index': TARGET_BAND,
        'k_label': 'M',
        'k0_x': 0.5,
        'k0_y': 0.5,
        'omega0': OMEGA0,
        'polarization': 'TM',
        'dominant_polarization': 'TM',
        'local_polarization': 'TM',
        'n_subspace_bands': len(subspace_bands),
        'subspace_bands': subspace_bands,
        'all_bands': all_bands,
        'target_index_in_subspace': 0,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'moire_length': moire_length,
        'eta': eta,
        'B_super': B_super.tolist(),
        'commensurate_m': m_idx,
        'commensurate_n': n_idx,
    }


def build_config(case, candidate, mt_value, subspace_bands):
    return {
        'phase1_Ns1': case['ea_ns'],
        'phase1_Ns2': case['ea_ns'],
        'mpb_resolution': case['mpb_resolution'],
        'mpb_registry_samples': case['ea_registry'],
        'mpb_dk': 0.06,
        'mpb_fd_order': 6,
        'mpb_polarization': 'TM',
        'export_bloch_fields': True,
        'mpb_n_workers': N_WORKERS,
        'tau': [0.0, 0.0],
        'default_theta_deg': candidate['theta_deg'],
        'ref_frequency_mode': 'mean',
        'include_born_huang': True,
        'include_drift_term': True,
        'include_kinetic_term': True,
        'include_offdiag_A': len(subspace_bands) > 1,  # only for multi-band
        'fd_order': 4,
        'n_modes': case['n_modes'],
        'candidate_type': 'band_minimum',
        'M_inv_max_trace': mt_value,
    }


def compute_exact_delta_frac_grid(candidate, ns):
    theta_rad = candidate['theta_rad']
    B_super = np.array(candidate['B_super'], dtype=float)
    B_mono = build_monolayer_basis('square', A)
    B_mono_inv = np.linalg.inv(B_mono)
    R_mat = np.array([
        [math.cos(theta_rad), -math.sin(theta_rad)],
        [math.sin(theta_rad), math.cos(theta_rad)],
    ])
    s_grid = build_fractional_grid(ns, ns)
    R_grid = fractional_to_cartesian(s_grid, B_super)
    disp = np.einsum('ij,...j->...i', R_mat - np.eye(2), R_grid)
    delta_frac = np.einsum('ij,...j->...i', B_mono_inv, disp)
    delta_frac = np.mod(delta_frac, 1.0)
    return s_grid, R_grid, delta_frac, B_super, B_mono


# ═══════════════════════════════════════════════════════════════
#  Band tracking variants
# ═══════════════════════════════════════════════════════════════

def compute_overlap_matrix(fields_ref, fields_cur, epsilon):
    """ε-weighted overlap matrix between band sets."""
    n_bands = fields_ref.shape[0]
    weights = np.repeat(epsilon[..., None], fields_ref.shape[-1], axis=2).reshape(-1)
    ref_flat = fields_ref.reshape(n_bands, -1)
    cur_flat = fields_cur.reshape(n_bands, -1)
    norms_ref = np.sqrt(np.sum(weights[None, :] * np.abs(ref_flat) ** 2, axis=1))
    norms_cur = np.sqrt(np.sum(weights[None, :] * np.abs(cur_flat) ** 2, axis=1))
    overlaps = (ref_flat.conj() * weights[None, :]) @ cur_flat.T
    return np.abs(overlaps) / (norms_ref[:, None] * norms_cur[None, :] + 1e-15)


def track_energy_sort(registry_data):
    """Energy-sort tracking: reorder bands by ascending ω at each registry point."""
    omega = registry_data['registry_omega0'].copy()
    vg = registry_data['registry_vg'].copy()
    m_inv = registry_data['registry_M_inv'].copy()
    stencil = registry_data['stencil_omega'].copy()
    bloch = registry_data['bloch_fields'].copy() if 'bloch_fields' in registry_data else None

    n_reg1, n_reg2, n_bands = omega.shape
    n_changed = 0

    for ix in range(n_reg1):
        for iy in range(n_reg2):
            perm = np.argsort(omega[ix, iy])
            if not np.array_equal(perm, np.arange(n_bands)):
                n_changed += 1
            omega[ix, iy] = omega[ix, iy, perm]
            vg[ix, iy] = vg[ix, iy, perm]
            m_inv[ix, iy] = m_inv[ix, iy, perm]
            stencil[ix, iy] = stencil[ix, iy, perm]
            if bloch is not None:
                bloch[ix, iy] = bloch[ix, iy, perm]

    result = dict(registry_data)
    result['registry_omega0'] = omega
    result['registry_vg'] = vg
    result['registry_M_inv'] = m_inv
    result['stencil_omega'] = stencil
    if bloch is not None:
        result['bloch_fields'] = bloch

    diag = {
        'method': 'energy_sort',
        'n_points_changed': int(n_changed),
        'fraction_changed': float(n_changed / (n_reg1 * n_reg2)),
    }
    return result, diag


def track_overlap_hungarian(registry_data):
    """Original raster-scan Hungarian tracking (from overnight script)."""
    if 'bloch_fields' not in registry_data or 'epsilon' not in registry_data:
        return registry_data, {'enabled': False}

    omega = registry_data['registry_omega0']
    vg = registry_data['registry_vg']
    m_inv = registry_data['registry_M_inv']
    stencil = registry_data['stencil_omega']
    bloch = registry_data['bloch_fields']
    epsilon = registry_data['epsilon']

    n_reg1, n_reg2, n_bands = bloch.shape[:3]
    tracked_omega = np.empty_like(omega)
    tracked_vg = np.empty_like(vg)
    tracked_m_inv = np.empty_like(m_inv)
    tracked_stencil = np.empty_like(stencil)
    tracked_bloch = np.empty_like(bloch)
    identity = np.arange(n_bands)
    match_scores = []
    n_changed = 0

    for ix in range(n_reg1):
        for iy in range(n_reg2):
            if ix == 0 and iy == 0:
                perm = identity
            else:
                score = np.zeros((n_bands, n_bands))
                contributors = 0
                if iy > 0:
                    score += compute_overlap_matrix(
                        tracked_bloch[ix, iy - 1], bloch[ix, iy], epsilon[ix, iy])
                    contributors += 1
                if ix > 0:
                    score += compute_overlap_matrix(
                        tracked_bloch[ix - 1, iy], bloch[ix, iy], epsilon[ix, iy])
                    contributors += 1
                if contributors == 0:
                    perm = identity
                else:
                    rows, cols = linear_sum_assignment(-score)
                    perm = np.empty(n_bands, dtype=int)
                    perm[rows] = cols
                if not np.array_equal(perm, identity):
                    n_changed += 1
                diag_scores = np.diag(score[:, perm]) if score.ndim == 2 else np.ones(n_bands)
                match_scores.extend(diag_scores.tolist())

            tracked_omega[ix, iy] = omega[ix, iy, perm]
            tracked_vg[ix, iy] = vg[ix, iy, perm]
            tracked_m_inv[ix, iy] = m_inv[ix, iy, perm]
            tracked_stencil[ix, iy] = stencil[ix, iy, perm]
            tracked_bloch[ix, iy] = bloch[ix, iy, perm]

    result = dict(registry_data)
    result['registry_omega0'] = tracked_omega
    result['registry_vg'] = tracked_vg
    result['registry_M_inv'] = tracked_m_inv
    result['stencil_omega'] = tracked_stencil
    result['bloch_fields'] = tracked_bloch

    diag = {
        'method': 'overlap_hungarian',
        'n_points_changed': int(n_changed),
        'fraction_changed': float(n_changed / (n_reg1 * n_reg2)),
        'match_score_min': float(np.min(match_scores)) if match_scores else 1.0,
        'match_score_mean': float(np.mean(match_scores)) if match_scores else 1.0,
    }
    return result, diag


def track_none(registry_data):
    """No tracking: use raw MPB band ordering."""
    return registry_data, {'method': 'none', 'n_points_changed': 0}


# ═══════════════════════════════════════════════════════════════
#  EA pipeline runner
# ═══════════════════════════════════════════════════════════════

def run_ea_test(test_name, case, subspace_bands, all_bands, mt_value,
                tracking_fn, run_dir):
    """
    Run a complete EA pipeline (Phase 1 → Phase 2 → Phase 3) and return frequencies.
    """
    import phase1_mpb_v3 as p1
    p1._log_fn = log

    candidate = build_candidate(case, subspace_bands, all_bands)
    config = build_config(case, candidate, mt_value, subspace_bands)

    theta_deg = candidate['theta_deg']
    eta = candidate['eta']
    n_cells = case['m'] ** 2 + case['n'] ** 2
    n_sub = len(subspace_bands)

    test_dir = run_dir / test_name
    cdir = candidate_dir(test_dir, 0)
    cdir.mkdir(parents=True, exist_ok=True)

    log(f'\n{"="*60}')
    log(f'  TEST: {test_name}')
    log(f'  θ={theta_deg:.2f}°, η={eta:.4f}, N={n_cells}')
    log(f'  subspace_bands={subspace_bands}, all_bands={all_bands}')
    log(f'  registry={case["ea_registry"]}×{case["ea_registry"]}, ns={case["ea_ns"]}')
    log(f'  n_modes={case["n_modes"]}, mt={mt_value}')
    log(f'  tracking: {tracking_fn.__name__}')
    log(f'{"="*60}')

    save_json(candidate, cdir / 'phase0_meta.json')
    save_json(config, test_dir / 'config.json')

    # ── Phase 1: MPB registry sweep ──
    t0 = time.time()
    log(f'  Phase 1: MPB registry sweep ({case["ea_registry"]}×{case["ea_registry"]}, {N_WORKERS} workers)...')
    registry_data = run_mpb_registry_sweep(
        candidate, config, config['mpb_registry_samples'], all_bands, subspace_bands
    )
    t_phase1_mpb = time.time() - t0
    log(f'  Phase 1 MPB sweep: {t_phase1_mpb:.1f}s')

    # ── Band tracking ──
    t0_track = time.time()
    registry_data, tracking_diag = tracking_fn(registry_data)
    t_track = time.time() - t0_track
    log(f'  Tracking ({tracking_diag.get("method", "?")}): '
        f'{tracking_diag.get("n_points_changed", 0)} points reordered '
        f'({t_track:.1f}s)')
    if 'match_score_min' in tracking_diag:
        log(f'    min_score={tracking_diag["match_score_min"]:.4f}, '
            f'mean_score={tracking_diag["match_score_mean"]:.4f}')

    # ── Interpolate to moiré grid ──
    ns = config['phase1_Ns1']
    s_grid, R_grid, delta_frac, B_super, B_mono = compute_exact_delta_frac_grid(candidate, ns)
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_mpb_v3(
        registry_data, delta_frac, all_bands, subspace_bands
    )
    target_idx = candidate['target_index_in_subspace']
    omega_ref = choose_reference_frequency(omega_grid[:, :, target_idx], config)
    V_grid = omega_grid - omega_ref

    # ── Save Phase 1 ──
    h5_path = cdir / 'phase1_multiband_data.h5'
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset('s_grid', data=s_grid, compression='gzip')
        hf.create_dataset('R_grid', data=R_grid, compression='gzip')
        hf.create_dataset('delta_frac', data=delta_frac, compression='gzip')
        hf.create_dataset('omega', data=omega_grid, compression='gzip')
        hf.create_dataset('vg', data=vg_grid, compression='gzip')
        hf.create_dataset('M_inv', data=M_inv_grid, compression='gzip')
        hf.create_dataset('V', data=V_grid, compression='gzip')
        stencil_grp = hf.create_group('stencil')
        stencil_grp.create_dataset('omega_all', data=stencil_info['stencil_omega_all'], compression='gzip')
        stencil_grp.create_dataset('registry_omega_all', data=stencil_info['registry_omega_all'], compression='gzip')
        stencil_grp.create_dataset('offsets', data=stencil_info['offsets'])
        stencil_grp.attrs['dk'] = stencil_info['dk']
        stencil_grp.attrs['fd_order'] = stencil_info['fd_order']
        stencil_grp.attrs['n_registry'] = stencil_info['n_registry']

        if 'bloch_fields' in registry_data and registry_data['bloch_fields'] is not None:
            save_bloch_fields(hf, registry_data['bloch_fields'], {
                'resolution': config['mpb_resolution'],
                'polarization': 'TM',
            })
        if 'epsilon' in registry_data and registry_data['epsilon'] is not None:
            hf.create_dataset(
                'epsilon', data=registry_data['epsilon'],
                compression='lzf',
                chunks=(1, 1, registry_data['epsilon'].shape[2], registry_data['epsilon'].shape[3]),
            )

        hf.attrs['omega_ref'] = omega_ref
        hf.attrs['eta'] = eta
        hf.attrs['theta_deg'] = theta_deg
        hf.attrs['theta_rad'] = candidate['theta_rad']
        hf.attrs['target_band_index'] = TARGET_BAND
        hf.attrs['target_index_in_subspace'] = target_idx
        hf.attrs['k0_x'] = 0.5
        hf.attrs['k0_y'] = 0.5
        hf.attrs['lattice_type'] = 'square'
        hf.attrs['r_over_a'] = R_OVER_A
        hf.attrs['eps_bg'] = EPS_BG
        hf.attrs['a'] = A
        hf.attrs['moire_length'] = candidate['moire_length']
        hf.attrs['Ns1'] = ns
        hf.attrs['Ns2'] = ns
        hf.attrs['N_subspace'] = n_sub
        hf.attrs['B_moire'] = np.array(candidate['B_super'], dtype=float)
        hf.attrs['B_mono'] = B_mono
        hf.attrs['subspace_bands'] = np.array(subspace_bands)
        hf.attrs['all_bands'] = np.array(all_bands)
        hf.attrs['solver'] = 'mpb'
        hf.attrs['pipeline_version'] = 'V3-diagnostic'
        hf.attrs['coordinate_system'] = 'fractional'

    save_json(tracking_diag, cdir / 'tracking_diagnostics.json')
    del registry_data
    gc.collect()

    # ── Phase 2: Berry connection + Born-Huang ──
    t0 = time.time()
    log(f'  Phase 2: Berry connection + Born-Huang...')
    process_candidate_phase2_v3(cdir, config)
    t_phase2 = time.time() - t0
    log(f'  Phase 2: {t_phase2:.1f}s')

    # ── Phase 3: Envelope solve ──
    t0 = time.time()
    with h5py.File(cdir / 'phase2_multiband_data.h5', 'r') as hf:
        omega_ref_p2 = float(hf.attrs['omega_ref'])
    config['sigma_shift'] = OMEGA0 - omega_ref_p2
    log(f'  Phase 3: Envelope solve (sigma_shift={config["sigma_shift"]:.6f})...')
    process_candidate_phase3_v3(cdir, config)
    t_phase3 = time.time() - t0
    log(f'  Phase 3: {t_phase3:.1f}s')

    # ── Load results ──
    with h5py.File(cdir / 'phase3_multiband_modes.h5', 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        omega_ref_p3 = float(hf.attrs['omega_ref'])
    freqs_ea = np.sort(omega_ref_p3 + eigenvalues)
    bw_ea = float(freqs_ea[-1] - freqs_ea[0])

    np.savez(test_dir / 'ea_freqs.npz', freqs=freqs_ea, omega_ref=omega_ref_p3)

    log(f'  Result: BW_EA = {bw_ea:.6f}')
    log(f'  EA freq range: [{freqs_ea[0]:.6f}, {freqs_ea[-1]:.6f}]')
    log(f'  omega_ref = {omega_ref_p3:.6f}')

    return {
        'test_name': test_name,
        'freqs': freqs_ea,
        'bw': bw_ea,
        'omega_ref': omega_ref_p3,
        'tracking_diag': tracking_diag,
        't_phase1': t_phase1_mpb,
        't_phase2': t_phase2,
        't_phase3': t_phase3,
    }


# ═══════════════════════════════════════════════════════════════
#  Load FDFD reference
# ═══════════════════════════════════════════════════════════════

def load_fdfd_reference(case_name, fdfd_res, n_modes):
    """Load FDFD frequencies from overnight run."""
    fdfd_path = OVERNIGHT_DIR / case_name / f'fdfd_res{fdfd_res}_k{n_modes}.npz'
    if not fdfd_path.exists():
        log(f'  WARNING: FDFD data not found at {fdfd_path}')
        return None

    data = np.load(fdfd_path)
    freqs = np.sort(data['freqs'])
    bw = float(freqs[-1] - freqs[0])
    log(f'  FDFD reference: {len(freqs)} modes, BW = {bw:.6f}')
    log(f'  FDFD freq range: [{freqs[0]:.6f}, {freqs[-1]:.6f}]')
    return {'freqs': freqs, 'bw': bw}


# ═══════════════════════════════════════════════════════════════
#  Summary & comparison plot
# ═══════════════════════════════════════════════════════════════

def plot_comparison(results, fdfd, run_dir):
    """Plot spectral comparison: EA tests vs FDFD."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Superimposed spectra
    ax = axes[0]
    n_fdfd = len(fdfd['freqs'])
    ax.plot(range(n_fdfd), fdfd['freqs'], 'k-', label=f'FDFD (BW={fdfd["bw"]:.4f})', lw=2)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, r in enumerate(results):
        n_ea = len(r['freqs'])
        ax.plot(range(n_ea), r['freqs'], '--', color=colors[i % len(colors)],
                label=f'{r["test_name"]} (BW={r["bw"]:.4f})', lw=1.5)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Frequency (c/a)')
    ax.set_title('Eigenvalue spectra')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: BW ratios
    ax = axes[1]
    names = [r['test_name'] for r in results]
    bw_ratios = [r['bw'] / fdfd['bw'] for r in results]
    bars = ax.bar(range(len(names)), bw_ratios, color=colors[:len(names)])
    ax.axhline(y=1.0, color='k', ls='--', alpha=0.5, label='Perfect match')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('BW_EA / BW_FDFD')
    ax.set_title('Bandwidth ratio')
    for i, (b, r) in enumerate(zip(bars, bw_ratios)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{r:.3f}', ha='center', va='bottom', fontsize=9)
    ax.legend()

    # Panel 3: DOS comparison
    ax = axes[2]
    f_min = min(fdfd['freqs'].min(), min(r['freqs'].min() for r in results))
    f_max = max(fdfd['freqs'].max(), max(r['freqs'].max() for r in results))
    margin = 0.1 * (f_max - f_min)
    bins = np.linspace(f_min - margin, f_max + margin, 80)
    ax.hist(fdfd['freqs'], bins=bins, alpha=0.4, color='black', label='FDFD', density=True)
    for i, r in enumerate(results):
        ax.hist(r['freqs'], bins=bins, alpha=0.3, color=colors[i % len(colors)],
                label=r['test_name'], density=True)
    ax.set_xlabel('Frequency (c/a)')
    ax.set_ylabel('DOS (normalized)')
    ax.set_title('Density of States')
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(run_dir / 'phase_a_comparison.png', dpi=150)
    plt.close(fig)
    log(f'  Saved comparison plot to {run_dir / "phase_a_comparison.png"}')


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    global _log_file

    run_dir = SCRIPT_DIR / 'phase_a_diagnostic' / f'run_{datetime.now():%Y%m%d_%H%M%S}'
    run_dir.mkdir(parents=True, exist_ok=True)

    _log_file = open(run_dir / 'diagnostic.log', 'w')

    log('='*60)
    log('  Phase A Diagnostic: EA Bandwidth Root Cause')
    log('='*60)
    log(f'  Run directory: {run_dir}')
    log(f'  FDFD reference: {OVERNIGHT_DIR}')
    log('')

    # ── Load FDFD reference ──
    fdfd = load_fdfd_reference('10deg', TEST_CASE['fdfd_res'], TEST_CASE['n_modes'])
    if fdfd is None:
        log('FATAL: No FDFD reference data found. Cannot proceed.')
        return

    results = []

    # ══════════════════════════════════════════════════════════
    #  Test 1: Single-band EA (N_bands=1, no tracking)
    # ══════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('  TEST 1: Single-band EA (SUBSPACE=[3], no tracking)')
    log('='*60)

    r1 = run_ea_test(
        test_name='single_band',
        case=TEST_CASE,
        subspace_bands=[3],
        all_bands=list(range(10)),
        mt_value=None,  # No M_inv clamping for single band
        tracking_fn=track_none,
        run_dir=run_dir,
    )
    results.append(r1)
    log(f'\n  >> TEST 1 RESULT: BW_EA/BW_FDFD = {r1["bw"]/fdfd["bw"]:.4f}')

    # ══════════════════════════════════════════════════════════
    #  Test 2: Multi-band with energy-sort tracking
    # ══════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('  TEST 2: Multi-band EA with energy-sort tracking')
    log('='*60)

    r2 = run_ea_test(
        test_name='energy_sort',
        case=TEST_CASE,
        subspace_bands=[3, 4, 5, 6],
        all_bands=list(range(10)),
        mt_value=2.0,
        tracking_fn=track_energy_sort,
        run_dir=run_dir,
    )
    results.append(r2)
    log(f'\n  >> TEST 2 RESULT: BW_EA/BW_FDFD = {r2["bw"]/fdfd["bw"]:.4f}')

    # ══════════════════════════════════════════════════════════
    #  Test 3: Multi-band with Hungarian overlap tracking
    # ══════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('  TEST 3: Multi-band EA with overlap-Hungarian tracking (baseline)')
    log('='*60)

    r3 = run_ea_test(
        test_name='overlap_hungarian',
        case=TEST_CASE,
        subspace_bands=[3, 4, 5, 6],
        all_bands=list(range(10)),
        mt_value=2.0,
        tracking_fn=track_overlap_hungarian,
        run_dir=run_dir,
    )
    results.append(r3)
    log(f'\n  >> TEST 3 RESULT: BW_EA/BW_FDFD = {r3["bw"]/fdfd["bw"]:.4f}')

    # ══════════════════════════════════════════════════════════
    #  Test 4: Multi-band with NO tracking (raw MPB order)
    # ══════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('  TEST 4: Multi-band EA with NO tracking (raw MPB order)')
    log('='*60)

    r4 = run_ea_test(
        test_name='no_tracking',
        case=TEST_CASE,
        subspace_bands=[3, 4, 5, 6],
        all_bands=list(range(10)),
        mt_value=2.0,
        tracking_fn=track_none,
        run_dir=run_dir,
    )
    results.append(r4)
    log(f'\n  >> TEST 4 RESULT: BW_EA/BW_FDFD = {r4["bw"]/fdfd["bw"]:.4f}')

    # ══════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════
    log('\n' + '='*60)
    log('  PHASE A SUMMARY')
    log('='*60)
    log(f'  FDFD BW = {fdfd["bw"]:.6f}')
    log('')
    for r in results:
        ratio = r['bw'] / fdfd['bw']
        log(f'  {r["test_name"]:25s}  BW={r["bw"]:.6f}  ratio={ratio:.4f}  '
            f't={r["t_phase1"]+r["t_phase2"]+r["t_phase3"]:.0f}s')
    log('')

    # Diagnostic interpretation
    r1_ratio = r1['bw'] / fdfd['bw']
    r3_ratio = r3['bw'] / fdfd['bw']
    if r1_ratio > 0.7:
        log('  DIAGNOSIS: Single-band BW matches FDFD → band tracking IS the root cause')
    elif r1_ratio < 0.5:
        log('  DIAGNOSIS: Single-band BW also compressed → problem is NOT just tracking')
        log('           → Check scale factors, coordinate systems, Hamiltonian structure')
    else:
        log('  DIAGNOSIS: Inconclusive — single-band BW partially matches FDFD')

    # Save summary
    summary = {
        'fdfd_bw': fdfd['bw'],
        'fdfd_n_modes': len(fdfd['freqs']),
        'tests': [{
            'name': r['test_name'],
            'bw': r['bw'],
            'bw_ratio': r['bw'] / fdfd['bw'],
            'omega_ref': r['omega_ref'],
            'tracking': r['tracking_diag'],
            'timing': {
                'phase1': r['t_phase1'],
                'phase2': r['t_phase2'],
                'phase3': r['t_phase3'],
            },
        } for r in results],
    }
    save_json(summary, run_dir / 'phase_a_summary.json')

    # Plot comparison
    plot_comparison(results, fdfd, run_dir)

    log(f'\n  All results saved to {run_dir}')
    _log_file.close()


if __name__ == '__main__':
    main()
