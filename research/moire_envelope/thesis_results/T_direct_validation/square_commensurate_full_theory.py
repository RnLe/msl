#!/usr/bin/env python3
"""
Full-theory square commensurate validation runner.

This script runs the generic V3 pipeline with Bloch fields enabled so the
square direct-validation path uses the same Berry/Born-Huang physics as the
theory chapter.

Usage:
    python square_commensurate_full_theory.py --case 10deg
    python square_commensurate_full_theory.py --case 2deg
"""

import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

SCRIPT_DIR = Path(__file__).resolve().parent
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT))
sys.path.insert(0, str(MOIRE_ROOT / 'phasesV3'))

from common.io_utils import candidate_dir, choose_reference_frequency, save_json
from phasesV3.bloch_fields import save_bloch_fields
from phase1_mpb_v3 import (
    build_fractional_grid,
    build_monolayer_basis,
    compute_eta_physics,
    compute_eta_geometric,
    extract_multiband_data_from_mpb_v3,
    fractional_to_cartesian,
    run_mpb_registry_sweep,
)
from phase2_mpb_v3 import process_candidate_phase2_v3
from phase3_mpb_v3 import process_candidate_phase3_v3


CASES = {
    '10deg': {
        'm': 11,
        'n': 1,
        'fdfd_npz': SCRIPT_DIR / 'square_3way' / 'fdfd_supercell.npz',
        'label': '10.39 deg',
    },
    '7deg': {
        'm': 17,
        'n': 1,
        'fdfd_npz': None,
        'label': '6.73 deg',
    },
    '4deg': {
        'm': 29,
        'n': 1,
        'fdfd_npz': None,
        'label': '3.95 deg',
    },
    '2deg': {
        'm': 57,
        'n': 1,
        'fdfd_npz': SCRIPT_DIR / 'square_2deg' / 'fdfd_supercell_2deg.npz',
        'label': '2.01 deg',
    },
}

OUTROOT = SCRIPT_DIR / 'square_full_theory'

A = 1.0
R_OVER_A = 0.2
EPS_BG = 1.0
EPS_HOLE = 11.56
OMEGA0 = 0.68457
TARGET_BAND = 3
SUBSPACE_BANDS = [3, 4, 5, 6]
ALL_BANDS = list(range(10))


def log(message):
    print(message, flush=True)


def theta_from_mn(m_idx, n_idx):
    theta_rad = 2.0 * math.atan2(n_idx, m_idx)
    return theta_rad, math.degrees(theta_rad)


def build_candidate(case_name):
    case = CASES[case_name]
    theta_rad, theta_deg = theta_from_mn(case['m'], case['n'])
    l1 = np.array([case['m'], case['n']], dtype=float) * A
    l2 = np.array([-case['n'], case['m']], dtype=float) * A
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
        'n_subspace_bands': len(SUBSPACE_BANDS),
        'subspace_bands': SUBSPACE_BANDS,
        'all_bands': ALL_BANDS,
        'target_index_in_subspace': 0,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'moire_length': moire_length,
        'eta': eta,
        'B_super': B_super.tolist(),
        'commensurate_m': case['m'],
        'commensurate_n': case['n'],
    }


def build_config(candidate, args):
    return {
        'phase1_Ns1': args.ns,
        'phase1_Ns2': args.ns,
        'mpb_resolution': args.mpb_resolution,
        'mpb_registry_samples': args.registry,
        'mpb_dk': args.dk,
        'mpb_fd_order': args.fd_order,
        'mpb_polarization': 'TM',
        'export_bloch_fields': True,
        'mpb_n_workers': args.workers,
        'tau': [0.0, 0.0],
        'default_theta_deg': candidate['theta_deg'],
        'ref_frequency_mode': 'mean',
        'include_born_huang': True,
        'include_drift_term': True,
        'include_kinetic_term': True,
        'include_offdiag_A': True,
        'fd_order': 4,
        'n_modes': args.n_modes,
        'candidate_type': 'band_minimum',
        'M_inv_max_trace': args.m_inv_max_trace,
    }


def update_phase1_metadata(phase1_path, candidate, ns):
    theta_rad = candidate['theta_rad']
    theta_deg = candidate['theta_deg']
    B_mono = build_monolayer_basis('square', A)
    B_moire = np.array(candidate['B_super'], dtype=float)
    s_grid = build_fractional_grid(ns, ns)
    R_grid = fractional_to_cartesian(s_grid, B_moire)

    with h5py.File(phase1_path, 'a') as hf:
        if 'R_grid' in hf:
            del hf['R_grid']
        hf.create_dataset('R_grid', data=R_grid, compression='gzip')
        hf.attrs['theta_deg'] = theta_deg
        hf.attrs['theta_rad'] = theta_rad
        hf.attrs['eta'] = candidate['eta']
        hf.attrs['moire_length'] = candidate['moire_length']
        hf.attrs['B_moire'] = B_moire
        hf.attrs['B_mono'] = B_mono


def build_commensurate_basis(candidate):
    return np.array(candidate['B_super'], dtype=float)


def compute_exact_delta_frac_grid(candidate, ns):
    theta_rad = candidate['theta_rad']
    B_super = build_commensurate_basis(candidate)
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


def compute_overlap_matrix(fields_ref, fields_cur, epsilon):
    n_bands = fields_ref.shape[0]
    weights = np.repeat(epsilon[..., None], fields_ref.shape[-1], axis=2).reshape(-1)
    ref_flat = fields_ref.reshape(n_bands, -1)
    cur_flat = fields_cur.reshape(n_bands, -1)
    norms_ref = np.sqrt(np.sum(weights[None, :] * np.abs(ref_flat) ** 2, axis=1))
    norms_cur = np.sqrt(np.sum(weights[None, :] * np.abs(cur_flat) ** 2, axis=1))
    overlaps = (ref_flat.conj() * weights[None, :]) @ cur_flat.T
    return np.abs(overlaps) / (norms_ref[:, None] * norms_cur[None, :] + 1e-15)


def reorder_registry_data_by_overlap(registry_data):
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
    perms = np.zeros((n_reg1, n_reg2, n_bands), dtype=int)
    match_scores = []
    n_changed = 0

    identity = np.arange(n_bands)

    for ix in range(n_reg1):
        for iy in range(n_reg2):
            if ix == 0 and iy == 0:
                perm = identity
                score = np.eye(n_bands)
            else:
                score = np.zeros((n_bands, n_bands))
                contributors = 0
                if iy > 0:
                    score += compute_overlap_matrix(
                        tracked_bloch[ix, iy - 1], bloch[ix, iy], epsilon[ix, iy]
                    )
                    contributors += 1
                if ix > 0:
                    score += compute_overlap_matrix(
                        tracked_bloch[ix - 1, iy], bloch[ix, iy], epsilon[ix, iy]
                    )
                    contributors += 1
                if contributors == 0:
                    perm = identity
                else:
                    rows, cols = linear_sum_assignment(-score)
                    perm = np.empty(n_bands, dtype=int)
                    perm[rows] = cols
                if not np.array_equal(perm, identity):
                    n_changed += 1

            perms[ix, iy] = perm
            tracked_omega[ix, iy] = omega[ix, iy, perm]
            tracked_vg[ix, iy] = vg[ix, iy, perm]
            tracked_m_inv[ix, iy] = m_inv[ix, iy, perm]
            tracked_stencil[ix, iy] = stencil[ix, iy, perm]
            tracked_bloch[ix, iy] = bloch[ix, iy, perm]
            diag_scores = np.diag(score[:, perm]) if score.ndim == 2 else np.ones(n_bands)
            match_scores.extend(diag_scores.tolist())

    tracked = dict(registry_data)
    tracked['registry_omega0'] = tracked_omega
    tracked['registry_vg'] = tracked_vg
    tracked['registry_M_inv'] = tracked_m_inv
    tracked['stencil_omega'] = tracked_stencil
    tracked['bloch_fields'] = tracked_bloch

    diag = {
        'enabled': True,
        'n_points_changed': int(n_changed),
        'fraction_points_changed': float(n_changed / (n_reg1 * n_reg2)),
        'match_score_min': float(np.min(match_scores)),
        'match_score_mean': float(np.mean(match_scores)),
        'match_score_p05': float(np.quantile(match_scores, 0.05)),
    }
    return tracked, diag


def save_phase1_commensurate(run_dir, candidate, config, tracking_diag=None):
    cdir = candidate_dir(run_dir, candidate['candidate_id'])
    cdir.mkdir(parents=True, exist_ok=True)
    save_json(candidate, cdir / 'phase0_meta.json')

    registry_data = run_mpb_registry_sweep(
        candidate, config, config['mpb_registry_samples'], ALL_BANDS, SUBSPACE_BANDS
    )
    registry_data, tracking_diag = reorder_registry_data_by_overlap(registry_data)

    ns = config['phase1_Ns1']
    s_grid, R_grid, delta_frac, B_super, B_mono = compute_exact_delta_frac_grid(candidate, ns)
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_mpb_v3(
        registry_data, delta_frac, ALL_BANDS, SUBSPACE_BANDS
    )
    target_idx = candidate['target_index_in_subspace']
    omega_ref = choose_reference_frequency(omega_grid[:, :, target_idx], config)
    V_grid = omega_grid - omega_ref

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

        save_bloch_fields(hf, registry_data['bloch_fields'], {
            'resolution': config['mpb_resolution'],
            'polarization': 'TM',
        })
        hf.create_dataset(
            'epsilon',
            data=registry_data['epsilon'],
            compression='lzf',
            chunks=(1, 1, registry_data['epsilon'].shape[2], registry_data['epsilon'].shape[3]),
        )

        hf.attrs['omega_ref'] = omega_ref
        hf.attrs['eta'] = candidate['eta']
        hf.attrs['theta_deg'] = candidate['theta_deg']
        hf.attrs['theta_rad'] = candidate['theta_rad']
        hf.attrs['target_band_index'] = candidate['band_index']
        hf.attrs['target_index_in_subspace'] = target_idx
        hf.attrs['k0_x'] = candidate['k0_x']
        hf.attrs['k0_y'] = candidate['k0_y']
        hf.attrs['lattice_type'] = candidate['lattice_type']
        hf.attrs['r_over_a'] = candidate['r_over_a']
        hf.attrs['eps_bg'] = candidate['eps_bg']
        hf.attrs['a'] = candidate['a']
        hf.attrs['moire_length'] = candidate['moire_length']
        hf.attrs['Ns1'] = ns
        hf.attrs['Ns2'] = ns
        hf.attrs['N_subspace'] = len(SUBSPACE_BANDS)
        hf.attrs['B_moire'] = B_super
        hf.attrs['B_mono'] = B_mono
        hf.attrs['subspace_bands'] = np.array(SUBSPACE_BANDS)
        hf.attrs['all_bands'] = np.array(ALL_BANDS)
        hf.attrs['solver'] = 'mpb'
        hf.attrs['pipeline_version'] = 'V3-commensurate'
        hf.attrs['coordinate_system'] = 'fractional'

    if tracking_diag is not None:
        save_json(tracking_diag, cdir / 'phase1_tracking_diagnostics.json')

    return cdir


def clone_phase1_from(source_candidate_dir, target_candidate_dir, candidate, ns):
    source_candidate_dir = Path(source_candidate_dir)
    target_candidate_dir.mkdir(parents=True, exist_ok=True)

    source_h5 = source_candidate_dir / 'phase1_multiband_data.h5'
    target_h5 = target_candidate_dir / 'phase1_multiband_data.h5'
    if not source_h5.exists():
        raise FileNotFoundError(f'Missing Phase 1 source: {source_h5}')

    shutil.copy2(source_h5, target_h5)
    update_phase1_metadata(target_h5, candidate, ns)
    save_json(candidate, target_candidate_dir / 'phase0_meta.json')


def load_freqs_from_phase3(candidate_dir_path):
    with h5py.File(Path(candidate_dir_path) / 'phase3_multiband_modes.h5', 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        omega_ref = float(hf.attrs['omega_ref'])
    return np.sort(omega_ref + eigenvalues)


def load_fdfd_freqs(case_name):
    path = CASES[case_name]['fdfd_npz']
    if not path.exists():
        raise FileNotFoundError(f'Missing FDFD reference: {path}')
    data = np.load(path)
    return np.sort(data['freqs'])


def load_fdfd_freqs_from_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing FDFD reference: {path}')
    data = np.load(path)
    return np.sort(data['freqs'])


def plot_level_comparison(run_dir, case_name, freqs_ea, freqs_fdfd, target_omega):
    run_dir = Path(run_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    window = max(0.03, 1.15 * max(np.max(np.abs(freqs_ea - target_omega)), np.max(np.abs(freqs_fdfd - target_omega))))
    positions = {'FDFD': 0.30, 'EA': 0.70}
    for label, freqs, color in [('FDFD', freqs_fdfd, 'tab:red'), ('EA', freqs_ea, 'tab:green')]:
        mask = np.abs(freqs - target_omega) < window
        windowed = freqs[mask]
        ax.hlines(windowed, positions[label] - 0.12, positions[label] + 0.12, color=color, lw=0.9)
        ax.text(positions[label], target_omega + 0.93 * window, label, ha='center', va='center', color=color, fontweight='bold')
    ax.axhline(target_omega, color='0.4', ls='--', lw=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_ylim(target_omega - window, target_omega + window)
    ax.set_ylabel(r'$\omega a / 2\pi c$')
    ax.set_title('Level Diagram')

    ax = axes[1]
    n_compare = min(len(freqs_ea), len(freqs_fdfd))
    indices = np.arange(n_compare)
    ax.plot(indices, freqs_fdfd[:n_compare], 'o-', ms=3, lw=1.0, color='tab:red', label='FDFD')
    ax.plot(indices, freqs_ea[:n_compare], 'o-', ms=3, lw=1.0, color='tab:green', label='EA')
    ax.axhline(target_omega, color='0.4', ls='--', lw=0.8)
    ax.set_xlabel('Sorted mode index')
    ax.set_ylabel(r'$\omega a / 2\pi c$')
    ax.set_title('Sorted Eigenvalues')
    ax.legend()

    fig.suptitle(f'Square full-theory comparison: {case_name} at $\\omega_0={target_omega:.5f}$')
    fig.tight_layout()
    plot_path = run_dir / 'comparison_levels.png'
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def summarize(case_name, run_dir, candidate_dir_path, fdfd_npz=None):
    freqs_ea = load_freqs_from_phase3(candidate_dir_path)
    if fdfd_npz is None:
        freqs_fdfd = load_fdfd_freqs(case_name)
        fdfd_source = str(CASES[case_name]['fdfd_npz'])
    else:
        freqs_fdfd = load_fdfd_freqs_from_path(fdfd_npz)
        fdfd_source = str(Path(fdfd_npz))
    n_compare = min(len(freqs_ea), len(freqs_fdfd))
    diff = freqs_ea[:n_compare] - freqs_fdfd[:n_compare]
    plot_path = plot_level_comparison(run_dir, case_name, freqs_ea, freqs_fdfd, OMEGA0)
    summary = {
        'case': case_name,
        'theta_deg': build_candidate(case_name)['theta_deg'],
        'n_compare': int(n_compare),
        'fdfd_source': fdfd_source,
        'plot_path': str(plot_path),
        'ea_min': float(freqs_ea[0]),
        'ea_max': float(freqs_ea[n_compare - 1]),
        'fdfd_min': float(freqs_fdfd[0]),
        'fdfd_max': float(freqs_fdfd[n_compare - 1]),
        'rms_abs': float(np.sqrt(np.mean(diff ** 2))),
        'max_abs': float(np.max(np.abs(diff))),
        'mean_abs': float(np.mean(np.abs(diff))),
        'diff_first10': diff[:10].tolist(),
        'freqs_ea_first10': freqs_ea[:10].tolist(),
        'freqs_fdfd_first10': freqs_fdfd[:10].tolist(),
    }
    save_json(summary, Path(run_dir) / 'comparison_summary.json')
    np.savez(
        Path(run_dir) / 'comparison_data.npz',
        freqs_ea=freqs_ea,
        freqs_fdfd=freqs_fdfd,
        diff=diff,
    )
    return summary


def ensure_run_dir(case_name):
    OUTROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = OUTROOT / f'{case_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=sorted(CASES), required=True)
    parser.add_argument('--reuse-phase1-from', type=Path, default=None)
    parser.add_argument('--fdfd-npz', type=Path, default=None)
    parser.add_argument('--registry', type=int, default=32)
    parser.add_argument('--ns', type=int, default=128)
    parser.add_argument('--mpb-resolution', type=int, default=32)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dk', type=float, default=0.06)
    parser.add_argument('--fd-order', type=int, default=6)
    parser.add_argument('--n-modes', type=int, default=50)
    parser.add_argument('--m-inv-max-trace', type=float, default=2.0)
    parser.add_argument('--target-omega', type=float, default=OMEGA0)
    args = parser.parse_args()

    if args.reuse_phase1_from is not None:
        raise ValueError(
            '--reuse-phase1-from is disabled: the stored Phase 1 HDF5 already contains '\
            'geometry-specific extracted omega/V/M_inv data and cannot be safely reused '\
            'across commensurate cases without a fresh resampling step.'
        )

    candidate = build_candidate(args.case)
    config = build_config(candidate, args)
    run_dir = ensure_run_dir(args.case)
    cdir = candidate_dir(run_dir, candidate['candidate_id'])

    log('=' * 72)
    log(f"Square full-theory commensurate run: {args.case} ({CASES[args.case]['label']})")
    log(f"Output: {run_dir}")
    if args.reuse_phase1_from is not None:
        log(f"Reusing Phase 1 from: {args.reuse_phase1_from}")
    else:
        log(f"Phase 1: registry={args.registry}, mpb_res={args.mpb_resolution}, workers={args.workers}")
        log("Phase 1 geometry: exact commensurate supercell map + overlap-based band tracking")
    log(f"Phase 3: n_modes={args.n_modes}, M_inv_max_trace={args.m_inv_max_trace}")
    log('=' * 72)

    save_json(candidate, run_dir / 'candidate.json')
    save_json(config, run_dir / 'config.json')

    if args.reuse_phase1_from is None:
        cdir = save_phase1_commensurate(run_dir, candidate.copy(), config)
    else:
        clone_phase1_from(args.reuse_phase1_from, cdir, candidate, args.ns)

    process_candidate_phase2_v3(cdir, config)
    with h5py.File(cdir / 'phase2_multiband_data.h5', 'r') as hf:
        omega_ref = float(hf.attrs['omega_ref'])
    config['sigma_shift'] = args.target_omega - omega_ref
    log(f"Phase 3 sigma target: omega_target={args.target_omega:.6f}, sigma_shift={config['sigma_shift']:.6f}")
    process_candidate_phase3_v3(cdir, config)

    summary = summarize(args.case, run_dir, cdir, fdfd_npz=args.fdfd_npz)
    log('')
    log('Comparison summary:')
    log(f"  RMS |EA-FDFD| = {summary['rms_abs']:.6e}")
    log(f"  Max |EA-FDFD| = {summary['max_abs']:.6e}")
    log(f"  Mean |EA-FDFD| = {summary['mean_abs']:.6e}")
    log(f"  EA range        = [{summary['ea_min']:.6f}, {summary['ea_max']:.6f}]")
    log(f"  FDFD range      = [{summary['fdfd_min']:.6f}, {summary['fdfd_max']:.6f}]")
    log(f"  Level plot      = {summary['plot_path']}")


if __name__ == '__main__':
    main()