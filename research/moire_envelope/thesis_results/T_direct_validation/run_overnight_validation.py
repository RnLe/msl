#!/usr/bin/env python3
"""
Overnight multi-angle EA vs FDFD validation pipeline.
=====================================================

Runs the full validation for 4 commensurate square-lattice twist angles:
  10.39° (m=11,n=1), 6.73° (m=17,n=1), 3.95° (m=29,n=1), 2.01° (m=57,n=1)

Pipeline steps (all sequential):
  0. Quick MPB single-thread verification
  1. Phase 1: EA registry sweep + Phase 2 + Phase 3 per angle
  2. FDFD reference solves per angle (float64 at Gamma)
  3. Spectral comparison + multi-angle summary plots + report

Usage:
    python run_overnight_validation.py           # full pipeline
    python run_overnight_validation.py --skip-ea # skip EA, only FDFD + comparison
    python run_overnight_validation.py --skip-fdfd # skip FDFD, only EA + comparison
    python run_overnight_validation.py --comparison-only  # only comparison
"""

# ═══════════════════════════════════════════════════════════════
# CRITICAL: Set single-thread env BEFORE any numerical imports.
# MPB has internal threading that thrashes the CPU.
# The Python multiprocessing workers in the registry sweep each
# run single-threaded, but we launch 16 of them in parallel.
# ═══════════════════════════════════════════════════════════════
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde, ks_2samp

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
    compute_moire_basis,
    compute_eta_physics,
    extract_multiband_data_from_mpb_v3,
    fractional_to_cartesian,
    run_mpb_registry_sweep,
)
from phase2_mpb_v3 import apply_abelian_gauge_2d, apply_svqb_to_bloch_fields, process_candidate_phase2_v3
from phase3_mpb_v3 import process_candidate_phase3_v3
from subspace_tracking import analyze_registry_subspace_tracking
from T_direct_validation.commensurate_utils import commensurate_twist_angle


# ═══════════════════════════════════════════════════════════════
#  Physical constants
# ═══════════════════════════════════════════════════════════════

A = 1.0
R_OVER_A = 0.2
EPS_BG = 1.0
EPS_HOLE = 11.56
OMEGA0 = 0.0
LATTICE_TYPE = 'square'
TARGET_BAND = 1
SUBSPACE_BANDS = [1, 2, 3, 4]
ALL_BANDS = list(range(10))
OMEGA_REF_MODE = 'zero'
TARGET_FREQUENCY_MODE = 'mean'
INCLUDE_DRIFT_TERM = True
INCLUDE_OFFDIAG_A = True
INCLUDE_BORN_HUANG = True
INCLUDE_OFFDIAG_V_DRIFT = True
MPB_FD_ORDER = 6

# Band configurations for validation
BAND_CONFIGS = [
    {'name': 'multi4', 'subspace_bands': [1, 2, 3, 4], 'mt_values': [None, 2.0, 0.5]},
    {'name': 'pair2', 'subspace_bands': [1, 2], 'mt_values': [None]},
    {'name': 'single', 'subspace_bands': [1], 'mt_values': [None]},
]

# ═══════════════════════════════════════════════════════════════
#  Angle configurations
# ═══════════════════════════════════════════════════════════════

ANGLE_CASES = [
    {
        'name': '10deg',
        'm': 11, 'n': 1,
        'fdfd_res': 64,
        'ea_registry': 48,
        'ea_ns': 128,
        'mpb_resolution': 64,
        'n_modes': 80,
        'mt_values': [None, 2.0, 0.5],
    },
    {
        'name': '7deg',
        'm': 17, 'n': 1,
        'fdfd_res': 56,
        'ea_registry': 48,
        'ea_ns': 128,
        'mpb_resolution': 64,
        'n_modes': 80,
        'mt_values': [None, 2.0, 0.5],
    },
    {
        'name': '4deg',
        'm': 29, 'n': 1,
        'fdfd_res': 48,
        'ea_registry': 48,
        'ea_ns': 128,
        'mpb_resolution': 64,
        'n_modes': 80,
        'mt_values': [None, 2.0, 0.5],
    },
    {
        'name': '2deg',
        'm': 57, 'n': 1,
        'fdfd_res': 48,
        'ea_registry': 48,
        'ea_ns': 128,
        'mpb_resolution': 64,
        'n_modes': 80,
        'mt_values': [None, 2.0, 0.5],
    },
    {
        'name': '1p1deg_hc',
        'm': 30, 'n': 29,
        'fdfd_res': 40,
        'ea_registry': 32,
        'ea_ns': 64,
        'mpb_resolution': 32,
        'n_modes': 50,
        'mt_values': [None],
    },
]

N_WORKERS = 16  # multiprocessing workers for MPB registry sweep

# ═══════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════

_LOG_FILE = None


def log(message, level='INFO'):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] [{level}] {message}'
    print(line, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, 'a') as f:
            f.write(line + '\n')


def log_section(title):
    log('=' * 72)
    log(title)
    log('=' * 72)


# ═══════════════════════════════════════════════════════════════
#  Step 0: MPB single-thread verification
# ═══════════════════════════════════════════════════════════════

def verify_single_thread():
    """Quick test that OMP_NUM_THREADS=1 is effective for MPB."""
    log_section('STEP 0: MPB single-thread verification')

    # Check environment
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        val = os.environ.get(var, 'NOT SET')
        log(f'  {var} = {val}')
        if val != '1':
            raise RuntimeError(f'{var} must be "1", got "{val}"')

    # Quick single-point MPB call to verify it stays single-core
    log('Running single MPB call to verify threading...')
    import meep as mp
    from meep import mpb

    geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
    geometry = [mp.Cylinder(R_OVER_A, material=mp.Medium(epsilon=EPS_HOLE))]

    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        resolution=16,
        k_points=[mp.Vector3(0.5, 0.5)],
        num_bands=5,
    )

    pid = os.getpid()
    log(f'  PID of this process: {pid}')
    # Run the solver (single k-point, fast)
    ms.run_tm()
    freqs = ms.all_freqs[-1]
    log(f'  MPB test: 5 TM bands at M = {[f"{f:.4f}" for f in freqs]}')
    log(f'  Band 1 = {freqs[1]:.6f}')
    log('Single-thread verification PASSED')
    return True


# ═══════════════════════════════════════════════════════════════
#  Geometry helpers (from square_commensurate_full_theory.py)
# ═══════════════════════════════════════════════════════════════

def theta_from_mn(m_idx, n_idx):
    theta_rad = float(commensurate_twist_angle(LATTICE_TYPE, m_idx, n_idx))
    return theta_rad, math.degrees(theta_rad)


def get_target_k_info(lattice_type):
    if lattice_type in ('honeycomb', 'hex', 'hexagonal', 'triangular'):
        return 'K', 2.0 / 3.0, 1.0 / 3.0
    return 'M', 0.5, 0.5


def get_lattice_display_name(lattice_type):
    if lattice_type == 'square':
        return 'Square'
    if lattice_type == 'honeycomb':
        return 'Honeycomb'
    if lattice_type in ('hex', 'hexagonal', 'triangular'):
        return 'Hexagonal'
    return lattice_type


def compute_case_n_cells(case):
    if LATTICE_TYPE == 'square':
        return case['m'] ** 2 + case['n'] ** 2
    return case['m'] ** 2 + case['m'] * case['n'] + case['n'] ** 2


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def choose_band_frequency(omega_band_grid, m_inv_band_grid, mode):
    omega_band_grid = np.asarray(omega_band_grid, dtype=float)

    if mode in {'mean', 'min', 'max', 'median'}:
        value = choose_reference_frequency(omega_band_grid, {'ref_frequency_mode': mode})
        info = {
            'mode': mode,
            'selection': mode,
            'band_character': 'n/a',
            'trace_mean': None,
        }
        return float(value), info

    if mode != 'target_extremum':
        raise ValueError(f'Unsupported frequency mode: {mode}')

    trace_grid = m_inv_band_grid[..., 0, 0] + m_inv_band_grid[..., 1, 1]
    trace_mean = float(np.mean(trace_grid))
    if trace_mean < 0.0:
        return float(np.max(omega_band_grid)), {
            'mode': mode,
            'selection': 'max',
            'band_character': 'hole',
            'trace_mean': trace_mean,
        }
    return float(np.min(omega_band_grid)), {
        'mode': mode,
        'selection': 'min',
        'band_character': 'electron',
        'trace_mean': trace_mean,
    }


def choose_interior_manifold_extremum(omega_grid, m_inv_grid, subspace_bands):
    nb = omega_grid.shape[2]
    manifold_center = 0.5 * (float(np.min(omega_grid)) + float(np.max(omega_grid)))
    candidate_indices = list(range(1, nb - 1)) if nb > 2 else list(range(nb))
    if not candidate_indices:
        candidate_indices = [0]

    scored = []
    for sub_idx in candidate_indices:
        value, info = choose_band_frequency(
            omega_grid[:, :, sub_idx],
            m_inv_grid[:, :, sub_idx],
            'target_extremum',
        )
        scored.append((abs(value - manifold_center), sub_idx, value, info))

    scored.sort(key=lambda item: (item[0], abs(item[1] - 0.5 * (nb - 1))))
    _, sub_idx, value, info = scored[0]
    return value, {
        'mode': 'interior_extremum',
        'selection': info['selection'],
        'band_character': info['band_character'],
        'trace_mean': info['trace_mean'],
        'selected_subspace_index': int(sub_idx),
        'selected_band': int(subspace_bands[sub_idx]),
        'manifold_center': float(manifold_center),
        'distance_to_center': float(abs(value - manifold_center)),
    }


def choose_validation_frequencies(omega_grid, m_inv_grid, candidate, config):
    target_idx = candidate['target_index_in_subspace']
    target_band = int(candidate['subspace_bands'][target_idx])
    target_mode = config.get('target_frequency_mode', TARGET_FREQUENCY_MODE)
    omega_ref_mode = config.get('omega_ref_mode', OMEGA_REF_MODE)

    if target_mode == 'interior_extremum':
        omega_target_abs, target_info = choose_interior_manifold_extremum(
            omega_grid,
            m_inv_grid,
            candidate['subspace_bands'],
        )
    else:
        omega_target_abs, target_info = choose_band_frequency(
            omega_grid[:, :, target_idx],
            m_inv_grid[:, :, target_idx],
            target_mode,
        )

    ref_info = {
        'target_band': target_band,
        'omega_ref_mode': omega_ref_mode,
        'target_frequency_mode': target_mode,
        'target_frequency_info': target_info,
    }

    if omega_ref_mode == 'zero':
        omega_ref = float(OMEGA0)
        ref_info['omega_ref_info'] = {
            'mode': 'zero',
            'selection': 'constant',
            'band_character': 'n/a',
            'trace_mean': None,
        }
        return omega_ref, omega_target_abs, ref_info

    if omega_ref_mode == 'interior_extremum':
        omega_ref, omega_ref_info = choose_interior_manifold_extremum(
            omega_grid,
            m_inv_grid,
            candidate['subspace_bands'],
        )
    else:
        omega_ref, omega_ref_info = choose_band_frequency(
            omega_grid[:, :, target_idx],
            m_inv_grid[:, :, target_idx],
            omega_ref_mode,
        )
    ref_info['omega_ref_info'] = omega_ref_info
    return omega_ref, omega_ref, ref_info


def build_candidate(case, subspace_bands=None):
    if subspace_bands is None:
        subspace_bands = SUBSPACE_BANDS
    m_idx, n_idx = case['m'], case['n']
    theta_rad, theta_deg = theta_from_mn(m_idx, n_idx)
    k_label, k0_x, k0_y = get_target_k_info(LATTICE_TYPE)
    B_mono = build_monolayer_basis(LATTICE_TYPE, A)
    B_super = compute_moire_basis(B_mono, theta_rad)
    moire_length = float(np.linalg.norm(B_super[:, 0]))
    eta = compute_eta_physics(theta_rad)
    target_idx = subspace_bands.index(TARGET_BAND) if TARGET_BAND in subspace_bands else 0
    return {
        'candidate_id': 0,
        'lattice_type': LATTICE_TYPE,
        'a': A,
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_HOLE,
        'band_index': TARGET_BAND,
        'k_label': k_label,
        'k0_x': k0_x,
        'k0_y': k0_y,
        'omega0': OMEGA0,
        'polarization': 'TM',
        'dominant_polarization': 'TM',
        'local_polarization': 'TM',
        'n_subspace_bands': len(subspace_bands),
        'subspace_bands': subspace_bands,
        'all_bands': ALL_BANDS,
        'target_index_in_subspace': target_idx,
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'moire_length': moire_length,
        'eta': eta,
        'B_super': B_super.tolist(),
        'domain_type': 'single_moire_cell',
        'commensurate_m': m_idx,
        'commensurate_n': n_idx,
    }


def build_config(case, candidate, mt_value):
    return {
        'phase1_Ns1': case['ea_ns'],
        'phase1_Ns2': case['ea_ns'],
        'mpb_resolution': case['mpb_resolution'],
        'mpb_registry_samples': case['ea_registry'],
        'mpb_dk': 0.06,
        'mpb_fd_order': MPB_FD_ORDER,
        'mpb_polarization': 'TM',
        'export_bloch_fields': True,
        'mpb_n_workers': N_WORKERS,
        'tau': [0.0, 0.0],
        'default_theta_deg': candidate['theta_deg'],
        'ref_frequency_mode': 'mean',
        'omega_ref_mode': OMEGA_REF_MODE,
        'target_frequency_mode': TARGET_FREQUENCY_MODE,
        'include_born_huang': INCLUDE_BORN_HUANG,
        'include_drift_term': INCLUDE_DRIFT_TERM,
        'include_kinetic_term': True,
        'include_offdiag_A': INCLUDE_OFFDIAG_A,
        'include_offdiag_v_drift': INCLUDE_OFFDIAG_V_DRIFT,
        'fd_order': 4,
        'n_modes': case['n_modes'],
        'candidate_type': 'band_minimum',
        'M_inv_max_trace': mt_value,
    }


def compute_exact_delta_frac_grid(candidate, ns):
    theta_rad = candidate['theta_rad']
    B_super = np.array(candidate['B_super'], dtype=float)
    B_mono = build_monolayer_basis(LATTICE_TYPE, A)
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


def load_phase1_absolute_target_frequency(shared_phase1_cdir):
    h5_path = Path(shared_phase1_cdir) / 'phase1_multiband_data.h5'
    with h5py.File(h5_path, 'r') as hf:
        if 'omega_target_abs' in hf.attrs:
            return float(hf.attrs['omega_target_abs'])
        return float(hf.attrs['omega_ref'])


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


def compute_subspace_tracking_diagnostic(registry_data, subspace_bands, all_bands):
    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return value

    if 'bloch_fields' not in registry_data or 'epsilon' not in registry_data:
        return {'enabled': False, 'reason': 'missing_bloch_fields_or_epsilon'}

    bloch_fields = np.array(registry_data['bloch_fields'], copy=True)
    epsilon = registry_data['epsilon']
    bloch_fields, gauge_diag = apply_abelian_gauge_2d(bloch_fields)
    bloch_fields, svqb_stats = apply_svqb_to_bloch_fields(bloch_fields, epsilon)
    subspace_indices = [all_bands.index(band) for band in subspace_bands]
    diagnostic = analyze_registry_subspace_tracking(
        bloch_fields,
        epsilon,
        subspace_indices,
        seed=(0, 0),
        periodic=True,
    )
    return {
        'enabled': True,
        'gauge_diagnostics': _jsonable(gauge_diag),
        'svqb_stats': _jsonable(svqb_stats),
        'diagnostic': _jsonable(diagnostic),
    }


def create_subset_phase1_h5(src_h5_path, dst_h5_path, band_indices, subspace_bands):
    """Create a Phase 1 HDF5 with a subset of bands extracted from a multi-band Phase 1.

    Args:
        src_h5_path: Path to full multi-band Phase 1 HDF5
        dst_h5_path: Path for the new subset HDF5
        band_indices: indices into the subspace axis to keep (e.g. [0] for first band)
        subspace_bands: new subspace band list (e.g. [3])
    """
    dst_h5_path = Path(dst_h5_path)
    if dst_h5_path.exists():
        return

    with h5py.File(src_h5_path, 'r') as src, h5py.File(dst_h5_path, 'w') as dst:
        # Copy with band subsetting: omega, vg, M_inv, V have band axis at dim 2
        for key in ['omega', 'vg', 'M_inv', 'V']:
            dst.create_dataset(key, data=src[key][:][:, :, band_indices], compression='gzip')

        # Copy unchanged datasets
        for key in ['s_grid', 'R_grid', 'delta_frac', 'epsilon']:
            src.copy(key, dst)

        # Subset bloch_fields (axis 2 = band)
        bf = src['bloch_fields'][:, :, band_indices]
        dst.create_dataset('bloch_fields', data=bf, compression='lzf',
                           chunks=(1, 1, len(band_indices), bf.shape[3], bf.shape[4], 3),
                           dtype=np.complex64)
        for attr_name in src['bloch_fields'].attrs:
            dst['bloch_fields'].attrs[attr_name] = src['bloch_fields'].attrs[attr_name]

        # Copy stencil group unchanged
        src.copy('stencil', dst)

        # Copy scalar attributes, skipping ones we override below
        override_keys = {'all_bands', 'N_subspace', 'subspace_bands', 'target_index_in_subspace'}
        for attr_name in src.attrs:
            if attr_name not in override_keys:
                dst.attrs[attr_name] = src.attrs[attr_name]
        dst.attrs['N_subspace'] = len(subspace_bands)
        dst.attrs['subspace_bands'] = np.array(subspace_bands)
        dst.attrs['all_bands'] = np.array(subspace_bands)
        target_idx = subspace_bands.index(TARGET_BAND) if TARGET_BAND in subspace_bands else 0
        dst.attrs['target_index_in_subspace'] = target_idx

    log(f'  Created subset Phase 1: {dst_h5_path.name} with bands {subspace_bands}')


# ═══════════════════════════════════════════════════════════════
#  Step 1: EA pipeline (Phase 1 + Phase 2 + Phase 3)
# ═══════════════════════════════════════════════════════════════

def run_ea_phase1(case, run_dir):
    """Run Phase 1 (MPB registry sweep) once per angle. Results are shared across mt values."""
    import phase1_mpb_v3 as p1
    p1._log_fn = log  # redirect phase1 logging to our file+stdout logger

    case_name = case['name']
    shared_dir = run_dir / case_name / 'shared_phase1'
    cdir = candidate_dir(shared_dir, 0)
    cdir.mkdir(parents=True, exist_ok=True)

    h5_path = cdir / 'phase1_multiband_data.h5'

    # Resumability: skip if Phase 1 already done
    if h5_path.exists():
        log(f'  Phase 1 {case_name}: SKIPPED (already exists at {h5_path})')
        return str(cdir)

    candidate = build_candidate(case)
    # Phase 1 does not depend on mt_value, so use None
    config = build_config(case, candidate, mt_value=None)

    theta_deg = candidate['theta_deg']
    n_cells = compute_case_n_cells(case)
    eta = candidate['eta']

    log(f'  Phase 1 {case_name}: θ={theta_deg:.2f}°, N={n_cells}, η={eta:.4f}')
    log(f'  registry={case["ea_registry"]}, ns={case["ea_ns"]}, mpb_res={case["mpb_resolution"]}')

    save_json(candidate, cdir / 'phase0_meta.json')
    save_json(config, shared_dir / 'config.json')

    t0 = time.time()
    log(f'  Phase 1: Running MPB registry sweep ({case["ea_registry"]}×{case["ea_registry"]}, {N_WORKERS} workers)...')
    registry_data = run_mpb_registry_sweep(
        candidate, config, config['mpb_registry_samples'], ALL_BANDS, SUBSPACE_BANDS
    )
    t_phase1_mpb = time.time() - t0
    log(f'  Phase 1 MPB sweep: {t_phase1_mpb:.1f}s')

    # Band tracking
    registry_data, tracking_diag = reorder_registry_data_by_overlap(registry_data)
    log(f'  Band tracking: {tracking_diag.get("n_points_changed", 0)} points reordered, '
        f'min_score={tracking_diag.get("match_score_min", 1.0):.4f}')
    subspace_tracking_diag = compute_subspace_tracking_diagnostic(registry_data, SUBSPACE_BANDS, ALL_BANDS)
    if subspace_tracking_diag.get('enabled'):
        diag_core = subspace_tracking_diag['diagnostic']
        raw_min = diag_core['raw_subspace_fidelity_min_singular_value']['min']
        edge_min = diag_core['transport_edge_min_singular_value']['min']
        path_min = diag_core['path_consistency_min_singular_value']['min']
        log(
            '  Subspace transport: '
            f'edge_min_sv={edge_min if edge_min is not None else float("nan"):.4f}, '
            f'raw_fidelity_min_sv={raw_min if raw_min is not None else float("nan"):.4f}, '
            f'path_min_sv={path_min if path_min is not None else float("nan"):.4f}'
        )

    # Interpolate to moiré grid
    ns = config['phase1_Ns1']
    s_grid, R_grid, delta_frac, B_super, B_mono = compute_exact_delta_frac_grid(candidate, ns)
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_mpb_v3(
        registry_data, delta_frac, ALL_BANDS, SUBSPACE_BANDS
    )
    target_idx = candidate['target_index_in_subspace']
    omega_ref, omega_target_abs, omega_ref_info = choose_validation_frequencies(
        omega_grid,
        M_inv_grid,
        candidate,
        config,
    )
    V_grid = omega_grid - omega_ref
    log(
        f'  Using validation reference frequency ω_ref={omega_ref:.6f} '
        f'({omega_ref_info["omega_ref_mode"]}); '
        f'physical target for FDFD sigma is ω_target={omega_target_abs:.6f} '
        f'({omega_ref_info["target_frequency_mode"]})'
    )

    # Save Phase 1
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
            'epsilon', data=registry_data['epsilon'],
            compression='lzf',
            chunks=(1, 1, registry_data['epsilon'].shape[2], registry_data['epsilon'].shape[3]),
        )
        hf.attrs['omega_ref'] = omega_ref
        hf.attrs['omega_target_abs'] = omega_target_abs
        hf.attrs['eta'] = eta
        hf.attrs['theta_deg'] = theta_deg
        hf.attrs['theta_rad'] = candidate['theta_rad']
        hf.attrs['target_band_index'] = TARGET_BAND
        hf.attrs['target_index_in_subspace'] = target_idx
        hf.attrs['k0_x'] = candidate['k0_x']
        hf.attrs['k0_y'] = candidate['k0_y']
        hf.attrs['lattice_type'] = LATTICE_TYPE
        hf.attrs['r_over_a'] = R_OVER_A
        hf.attrs['eps_bg'] = EPS_BG
        hf.attrs['a'] = A
        hf.attrs['moire_length'] = candidate['moire_length']
        hf.attrs['Ns1'] = ns
        hf.attrs['Ns2'] = ns
        hf.attrs['N_subspace'] = len(SUBSPACE_BANDS)
        hf.attrs['B_moire'] = np.array(candidate['B_super'], dtype=float)
        hf.attrs['B_mono'] = B_mono
        hf.attrs['subspace_bands'] = np.array(SUBSPACE_BANDS)
        hf.attrs['all_bands'] = np.array(ALL_BANDS)
        hf.attrs['solver'] = 'mpb'
        hf.attrs['pipeline_version'] = 'V3-moire'
        hf.attrs['coordinate_system'] = 'fractional'
    save_json(tracking_diag, cdir / 'phase1_tracking_diagnostics.json')
    save_json(subspace_tracking_diag, cdir / 'phase1_subspace_tracking_diagnostics.json')
    save_json(omega_ref_info, cdir / 'phase1_reference_selection.json')

    t_phase1 = time.time() - t0
    log(f'  Phase 1 complete: {t_phase1:.1f}s')

    # Free large arrays
    del registry_data
    gc.collect()

    return str(cdir)


def run_ea_phase23(case, run_dir, mt_value, shared_phase1_cdir,
                   band_config_name='multi4', subspace_bands=None):
    """Run Phase 2+3 for one (angle, mt_value, band_config), reusing shared Phase 1 data."""
    if subspace_bands is None:
        subspace_bands = SUBSPACE_BANDS
    case_name = case['name']
    mt_str = f'mt{mt_value}' if mt_value is not None else 'mt_raw'
    config_prefix = f'{band_config_name}/' if band_config_name != 'multi4' else ''
    angle_dir = run_dir / case_name / f'{config_prefix}{mt_str}'
    cdir = candidate_dir(angle_dir, 0)
    cdir.mkdir(parents=True, exist_ok=True)

    label = f'{case_name}/{config_prefix}{mt_str}'

    # Resumability: skip if final output exists
    ea_freqs_path = angle_dir / 'ea_freqs.npz'
    if ea_freqs_path.exists():
        log(f'  EA {label}: SKIPPED (ea_freqs.npz exists)')
        data = np.load(ea_freqs_path)
        freqs_ea = np.sort(data['freqs'])
        candidate = build_candidate(case, subspace_bands)
        return {
            'case': case_name,
            'band_config': band_config_name,
            'mt_value': mt_value,
            'mt_str': mt_str,
            'theta_deg': candidate['theta_deg'],
            'eta': candidate['eta'],
            'n_cells': compute_case_n_cells(case),
            'freqs_ea': freqs_ea,
            'cdir': str(cdir),
            'run_dir': str(angle_dir),
            't_phase1': 0, 't_phase2': 0, 't_phase3': 0,
            'bw_ea': float(freqs_ea[-1] - freqs_ea[0]),
            'omega_ref': float(data['omega_ref']),
            'resumed': True,
        }

    candidate = build_candidate(case, subspace_bands)
    config = build_config(case, candidate, mt_value)
    theta_deg = candidate['theta_deg']
    n_cells = compute_case_n_cells(case)
    eta = candidate['eta']

    log(f'  EA {label}: θ={theta_deg:.2f}°, N={n_cells}, η={eta:.4f}, bands={subspace_bands}')
    log(f'  n_modes={case["n_modes"]}, M_inv_max_trace={mt_value}')

    save_json(candidate, cdir / 'phase0_meta.json')
    save_json(config, angle_dir / 'config.json')

    # Copy shared Phase 1 data (use band-config-specific H5 if not multi4)
    if band_config_name != 'multi4':
        # For single-band etc., we build a subset H5 from the multi-band Phase 1
        multi_h5 = Path(shared_phase1_cdir) / 'phase1_multiband_data.h5'
        subset_h5 = Path(shared_phase1_cdir) / f'phase1_{band_config_name}.h5'
        # Find which indices in multi-band subspace correspond to our requested bands
        multi_subspace = SUBSPACE_BANDS
        band_indices = [multi_subspace.index(b) for b in subspace_bands]
        create_subset_phase1_h5(multi_h5, subset_h5, band_indices, subspace_bands)
        shared_h5 = subset_h5
    else:
        shared_h5 = Path(shared_phase1_cdir) / 'phase1_multiband_data.h5'

    local_h5 = cdir / 'phase1_multiband_data.h5'
    if not local_h5.exists():
        shutil.copy2(shared_h5, local_h5)
        log(f'  Copied Phase 1 data ({band_config_name})')

    # ── Phase 2: Berry connection + Born-Huang ──
    t0 = time.time()
    log(f'  Phase 2: Computing Berry connection and Born-Huang terms...')
    process_candidate_phase2_v3(cdir, config)
    t_phase2 = time.time() - t0
    log(f'  Phase 2 complete: {t_phase2:.1f}s')

    # ── Phase 3: Envelope solve ──
    t0 = time.time()
    with h5py.File(cdir / 'phase2_multiband_data.h5', 'r') as hf:
        omega_ref = float(hf.attrs['omega_ref'])
    config['sigma_shift'] = OMEGA0
    log(f'  Phase 3: Envelope solve (n_modes={case["n_modes"]}, sigma_shift={config["sigma_shift"]:.6f})...')
    process_candidate_phase3_v3(cdir, config)
    t_phase3 = time.time() - t0
    log(f'  Phase 3 complete: {t_phase3:.1f}s')

    # ── Load results ──
    with h5py.File(cdir / 'phase3_multiband_modes.h5', 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        omega_ref_p3 = float(hf.attrs['omega_ref'])
    freqs_ea = np.sort(omega_ref_p3 + eigenvalues)
    freqs_ea_shifted = np.sort(eigenvalues)

    result = {
        'case': case_name,
        'band_config': band_config_name,
        'mt_value': mt_value,
        'mt_str': mt_str,
        'theta_deg': theta_deg,
        'eta': eta,
        'n_cells': n_cells,
        'freqs_ea': freqs_ea,
        'cdir': str(cdir),
        'run_dir': str(angle_dir),
        't_phase1': 0,
        't_phase2': t_phase2,
        't_phase3': t_phase3,
        'bw_ea': float(freqs_ea[-1] - freqs_ea[0]),
        'omega_ref': omega_ref_p3,
        'freqs_ea_shifted': freqs_ea_shifted,
    }
    save_json({k: v for k, v in result.items() if k not in {'freqs_ea', 'freqs_ea_shifted'}},
              angle_dir / 'ea_summary.json')
    np.savez(
        angle_dir / 'ea_freqs.npz',
        freqs=freqs_ea,
        freqs_shifted=freqs_ea_shifted,
        omega_ref=omega_ref_p3,
        comparison_center=omega_ref_p3,
    )

    return result


# ═══════════════════════════════════════════════════════════════
#  Step 2: FDFD reference solves
# ═══════════════════════════════════════════════════════════════

def run_fdfd_solve(case, run_dir, target_freq_abs):
    """Run FDFD on a single moire cell using float64."""
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    from T_direct_validation.supercell_geometry import build_moire_supercell_eps, build_supercell_eps
    import scipy.sparse as sp
    from scipy.sparse.linalg import LinearOperator, eigsh

    case_name = case['name']
    m_idx, n_idx = case['m'], case['n']
    resolution = case['fdfd_res']
    n_modes = case['n_modes']
    theta_rad = float(commensurate_twist_angle(LATTICE_TYPE, m_idx, n_idx))
    theta_deg = math.degrees(theta_rad)
    if LATTICE_TYPE == 'square':
        n_cells = m_idx**2 + n_idx**2
    else:
        n_cells = m_idx**2 + m_idx * n_idx + n_idx**2
    l_super = A / (2.0 * math.sin(theta_rad / 2.0))
    nx = int(round(np.sqrt(n_cells) * resolution))
    dof = nx * nx

    output_path = run_dir / case_name / f'fdfd_moire_res{resolution}_k{n_modes}.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        log(f'  FDFD {case_name}: output exists, loading {output_path}')
        data = np.load(output_path)
        return {
            'case': case_name,
            'freqs': np.sort(data['freqs']),
            'freqs_shifted': np.sort(data['freqs_shifted']) if 'freqs_shifted' in data else np.sort(data['freqs']) - target_freq_abs,
            'comparison_center': float(data['comparison_center']) if 'comparison_center' in data else target_freq_abs,
            'output_path': str(output_path),
            'reused': True,
        }

    sigma = (2.0 * np.pi * target_freq_abs) ** 2

    domain_label = 'moire cell' if LATTICE_TYPE == 'square' else 'commensurate supercell'
    log_section(f'FDFD solve ({domain_label}): {case_name}')
    log(f'  (m,n)=({m_idx},{n_idx}), θ={theta_deg:.2f}°, N_cells={n_cells}')
    log(f'  L_moire={l_super:.4f}a, res={resolution} px/a, Nx={nx}, DOF={dof:,}')
    log(f'  sigma={sigma:.6f}, target ω_ref={target_freq_abs:.6f}, n_modes={n_modes}')
    log(f'  output={output_path}')

    # Build epsilon grid on the validation domain
    t0 = time.time()
    if LATTICE_TYPE == 'square':
        eps, info = build_moire_supercell_eps(
            LATTICE_TYPE, theta_rad=theta_rad, a=A,
            r_over_a=R_OVER_A, eps_rod=EPS_HOLE, eps_bg=EPS_BG,
            Nx=nx, Ny=nx,
        )
    else:
        eps, info = build_supercell_eps(
            LATTICE_TYPE, m=m_idx, n=n_idx, a=A,
            r_over_a=R_OVER_A, eps_rod=EPS_HOLE, eps_bg=EPS_BG,
            Nx=nx, Ny=nx,
        )
    t_eps = time.time() - t0
    log(f'  Built epsilon grid: {eps.shape}, '
        f'range=[{eps.min():.2f}, {eps.max():.2f}], t={t_eps:.1f}s')

    # Build operator (float64 at q=0 in the moire Brillouin zone)
    t0 = time.time()
    operator = build_fdfd_operator(eps, info, q_vec=np.zeros(2), polarization='tm')
    t_op = time.time() - t0
    log(f'  Built operator: nnz={operator.nnz:,}, dtype={operator.dtype}, t={t_op:.1f}s')

    del eps
    gc.collect()

    # Shift-invert with CHOLMOD
    shifted = operator - sigma * sp.eye(dof, format='csc', dtype=operator.dtype)

    try:
        from sksparse.cholmod import cholesky
        t0 = time.time()
        factor = cholesky(shifted.tocsc(), beta=0, mode='simplicial')
        t_factor = time.time() - t0
        log(f'  CHOLMOD factorization: t={t_factor:.1f}s')

        op_inv = LinearOperator(operator.shape, matvec=lambda vec: factor(vec), dtype=operator.dtype)
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=sigma, which='LM',
                         OPinv=op_inv, maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0
    except ImportError:
        log('  CHOLMOD unavailable, falling back to scipy shift-invert', level='WARN')
        t_factor = 0.0
        t0 = time.time()
        evals, _ = eigsh(operator, k=n_modes, sigma=sigma, which='LM',
                         maxiter=10000, tol=1e-10)
        t_solve = time.time() - t0

    del operator, shifted
    gc.collect()

    evals = np.sort(evals)
    freqs = np.sqrt(np.maximum(evals, 0.0)) / (2.0 * np.pi)
    freqs_shifted = freqs - target_freq_abs
    log(f'  Eigensolve: t={t_solve:.1f}s, freq range=[{freqs.min():.6f}, {freqs.max():.6f}]')
    log(f'  Bandwidth: {freqs.max() - freqs.min():.6e}')

    np.savez(
        output_path,
        freqs=freqs, evals=evals,
        freqs_shifted=freqs_shifted,
        case=case_name, m=m_idx, n=n_idx,
        theta_deg=theta_deg, omega0=OMEGA0, omega_target_abs=target_freq_abs,
        comparison_center=target_freq_abs, sigma=sigma,
        N_cells=n_cells, res_per_cell=resolution, Nx=nx, n_modes=n_modes,
        t_eps=t_eps, t_op=t_op, t_factor=t_factor, t_solve=t_solve,
    )
    log(f'  Saved {output_path}')

    return {
        'case': case_name,
        'freqs': freqs,
        'freqs_shifted': freqs_shifted,
        'comparison_center': target_freq_abs,
        'output_path': str(output_path),
        'reused': False,
        't_eps': t_eps,
        't_op': t_op,
        't_factor': t_factor,
        't_solve': t_solve,
    }


# ═══════════════════════════════════════════════════════════════
#  Step 3: Spectral comparison + plots
# ═══════════════════════════════════════════════════════════════

def find_gaps(freqs, min_gap_ratio=2.0):
    sorted_f = np.sort(freqs)
    spacings = np.diff(sorted_f)
    if len(spacings) == 0:
        return []
    median_sp = np.median(spacings)
    gaps = []
    for i, s in enumerate(spacings):
        if s > min_gap_ratio * median_sp:
            gaps.append({
                'below': float(sorted_f[i]),
                'above': float(sorted_f[i + 1]),
                'gap': float(s),
                'center': float(0.5 * (sorted_f[i] + sorted_f[i + 1])),
                'index': i,
            })
    return gaps


def compare_spectra(freqs_ea, freqs_fdfd, case_name, theta_deg, eta, plot_dir, center_freq_abs):
    """Compute spectral comparison metrics and generate per-angle plots."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    freqs_ea = np.sort(freqs_ea)
    freqs_fdfd = np.sort(freqs_fdfd)
    freqs_ea_shifted = freqs_ea - center_freq_abs
    freqs_fdfd_shifted = freqs_fdfd - center_freq_abs

    bw_ea = float(freqs_ea[-1] - freqs_ea[0])
    bw_fdfd = float(freqs_fdfd[-1] - freqs_fdfd[0])
    bw_ratio = bw_ea / bw_fdfd if bw_fdfd > 0 else float('nan')

    # Spectral moments
    mean_ea, mean_fdfd = float(np.mean(freqs_ea_shifted)), float(np.mean(freqs_fdfd_shifted))
    std_ea, std_fdfd = float(np.std(freqs_ea_shifted)), float(np.std(freqs_fdfd_shifted))

    # KS test
    ks_stat, ks_pval = ks_2samp(freqs_ea_shifted, freqs_fdfd_shifted)

    # Histogram DOS correlation
    window_lo = min(freqs_ea_shifted.min(), freqs_fdfd_shifted.min()) - 0.001
    window_hi = max(freqs_ea_shifted.max(), freqs_fdfd_shifted.max()) + 0.001
    bins = np.linspace(window_lo, window_hi, 60)
    hist_ea, _ = np.histogram(freqs_ea_shifted, bins=bins)
    hist_fdfd, _ = np.histogram(freqs_fdfd_shifted, bins=bins)
    hist_ea_norm = hist_ea / max(len(freqs_ea), 1)
    hist_fdfd_norm = hist_fdfd / max(len(freqs_fdfd), 1)
    if np.std(hist_ea_norm) > 0 and np.std(hist_fdfd_norm) > 0:
        dos_corr = float(np.corrcoef(hist_ea_norm, hist_fdfd_norm)[0, 1])
    else:
        dos_corr = 0.0

    # Gap structure
    gaps_ea = find_gaps(freqs_ea_shifted)
    gaps_fdfd = find_gaps(freqs_fdfd_shifted)

    # Per-eigenvalue comparison (sorted)
    n_compare = min(len(freqs_ea_shifted), len(freqs_fdfd_shifted))
    diff = freqs_ea_shifted[:n_compare] - freqs_fdfd_shifted[:n_compare]
    rms = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    mean_err = float(np.mean(np.abs(diff)))

    metrics = {
        'case': case_name,
        'theta_deg': theta_deg,
        'eta': eta,
        'bw_ea': bw_ea,
        'bw_fdfd': bw_fdfd,
        'bw_ratio': bw_ratio,
        'mean_ea': mean_ea,
        'mean_fdfd': mean_fdfd,
        'std_ea': std_ea,
        'std_fdfd': std_fdfd,
        'ks_stat': float(ks_stat),
        'ks_pval': float(ks_pval),
        'dos_corr': dos_corr,
        'n_gaps_ea': len(gaps_ea),
        'n_gaps_fdfd': len(gaps_fdfd),
        'rms': rms,
        'max_err': max_err,
        'mean_err': mean_err,
        'n_compare': n_compare,
        'comparison_center_abs': center_freq_abs,
    }

    # ── Per-angle 3-panel figure ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Level diagram
    ax = axes[0]
    window = max(0.015, 1.15 * max(
        np.max(np.abs(freqs_ea_shifted)),
        np.max(np.abs(freqs_fdfd_shifted))
    ))
    for label, freqs, color, xpos in [
        ('FDFD', freqs_fdfd_shifted, 'tab:red', 0.30),
        ('EA', freqs_ea_shifted, 'tab:green', 0.70),
    ]:
        mask = np.abs(freqs) < window
        ax.hlines(freqs[mask], xpos - 0.12, xpos + 0.12, color=color, lw=0.9)
        ax.text(xpos, 0.93 * window, label, ha='center', va='center',
                color=color, fontweight='bold', fontsize=10)
    ax.axhline(0.0, color='0.4', ls='--', lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylim(-window, window)
    ax.set_ylabel(r'$(\omega - \omega_{\mathrm{ref}}) a / 2\pi c$')
    ax.set_title('Level diagram')

    # (b) KDE DOS
    ax = axes[1]
    f_grid = np.linspace(window_lo, window_hi, 500)
    try:
        kde_ea = gaussian_kde(freqs_ea_shifted, bw_method=0.01)
        kde_fdfd = gaussian_kde(freqs_fdfd_shifted, bw_method=0.01)
        ax.plot(f_grid, kde_ea(f_grid), color='tab:green', lw=1.5, label='EA')
        ax.plot(f_grid, kde_fdfd(f_grid), color='tab:red', lw=1.5, label='FDFD')
    except Exception:
        ax.hist(freqs_ea_shifted, bins=40, alpha=0.5, color='tab:green', label='EA', density=True)
        ax.hist(freqs_fdfd_shifted, bins=40, alpha=0.5, color='tab:red', label='FDFD', density=True)
    ax.axvline(0.0, color='0.4', ls='--', lw=0.8)
    ax.set_xlabel(r'$(\omega - \omega_{\mathrm{ref}}) a / 2\pi c$')
    ax.set_ylabel('Density')
    ax.set_title(f'DOS (KS={ks_stat:.3f}, corr={dos_corr:.3f})')
    ax.legend(fontsize=9)

    # (c) CDF
    ax = axes[2]
    ea_sorted = np.sort(freqs_ea_shifted)
    fdfd_sorted = np.sort(freqs_fdfd_shifted)
    ax.step(ea_sorted, np.arange(1, len(ea_sorted) + 1) / len(ea_sorted),
            where='post', color='tab:green', lw=1.5, label='EA')
    ax.step(fdfd_sorted, np.arange(1, len(fdfd_sorted) + 1) / len(fdfd_sorted),
            where='post', color='tab:red', lw=1.5, label='FDFD')
    ax.axvline(0.0, color='0.4', ls='--', lw=0.8)
    ax.set_xlabel(r'$(\omega - \omega_{\mathrm{ref}}) a / 2\pi c$')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('CDF')
    ax.legend(fontsize=9)

    fig.suptitle(f'{case_name}: θ={theta_deg:.2f}°, η={eta:.4f}, BW ratio={bw_ratio:.3f}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = plot_dir / f'comparison_{case_name}.png'
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    metrics['plot_path'] = str(fig_path)

    save_json(metrics, plot_dir / f'metrics_{case_name}.json')
    return metrics


def generate_multi_angle_summary(all_metrics, run_dir):
    """Generate combined multi-angle summary figure and report."""
    run_dir = Path(run_dir)

    if len(all_metrics) < 2:
        log('Not enough angles for multi-angle summary', level='WARN')
        return

    # Sort by angle (descending)
    all_metrics.sort(key=lambda m: m['theta_deg'], reverse=True)

    thetas = [m['theta_deg'] for m in all_metrics]
    etas = [m['eta'] for m in all_metrics]
    bw_ratios = [m['bw_ratio'] for m in all_metrics]
    ks_stats = [m['ks_stat'] for m in all_metrics]
    dos_corrs = [m['dos_corr'] for m in all_metrics]
    bw_ea = [m['bw_ea'] for m in all_metrics]
    bw_fdfd = [m['bw_fdfd'] for m in all_metrics]
    rms_vals = [m['rms'] for m in all_metrics]

    # ── Figure: 6-panel multi-angle summary ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) BW ratio vs theta
    ax = axes[0, 0]
    ax.plot(thetas, bw_ratios, 'ko-', ms=8, lw=2)
    ax.axhline(1.0, color='0.5', ls='--', lw=0.8)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('BW_EA / BW_FDFD')
    ax.set_title('(a) Bandwidth ratio')
    ax.set_ylim(0, max(2.0, max(bw_ratios) * 1.1))

    # (b) KS distance vs theta
    ax = axes[0, 1]
    ax.plot(thetas, ks_stats, 'rs-', ms=8, lw=2)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('KS statistic')
    ax.set_title('(b) KS distance')

    # (c) DOS correlation vs theta
    ax = axes[0, 2]
    ax.plot(thetas, dos_corrs, 'b^-', ms=8, lw=2)
    ax.axhline(1.0, color='0.5', ls='--', lw=0.8)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('Pearson r')
    ax.set_title('(c) DOS correlation')

    # (d) BW scaling law
    ax = axes[1, 0]
    ax.loglog(etas, bw_ea, 'go-', ms=8, lw=2, label='EA')
    ax.loglog(etas, bw_fdfd, 'rs-', ms=8, lw=2, label='FDFD')
    # η² reference line
    eta_ref = np.array(etas)
    bw_ref = bw_fdfd[0] * (eta_ref / etas[0]) ** 2
    ax.loglog(eta_ref, bw_ref, 'k--', lw=1, alpha=0.5, label='η² reference')
    ax.set_xlabel('η')
    ax.set_ylabel('Bandwidth')
    ax.set_title('(d) BW scaling law')
    ax.legend(fontsize=9)

    # (e) RMS error vs theta
    ax = axes[1, 1]
    ax.semilogy(thetas, rms_vals, 'mD-', ms=8, lw=2)
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('RMS |EA - FDFD|')
    ax.set_title('(e) RMS error')

    # (f) Per-angle mini level diagrams
    ax = axes[1, 2]
    n_angles = len(all_metrics)
    x_positions = np.linspace(0.1, 0.9, n_angles)
    for i, m in enumerate(all_metrics):
        bc = m.get('band_config', 'multi4')
        ea_file = run_dir / m['case'] / f'ea_freqs_{bc}_best.npz'
        if not ea_file.exists():
            ea_file = run_dir / m['case'] / 'ea_freqs_best.npz'
        ea_npz = np.load(ea_file)
        fdfd_npz = np.load(run_dir / m['case'] / 'fdfd_freqs.npz')
        freqs_ea_i = ea_npz['freqs_shifted'] if 'freqs_shifted' in ea_npz else ea_npz['freqs']
        freqs_fdfd_i = fdfd_npz['freqs_shifted'] if 'freqs_shifted' in fdfd_npz else fdfd_npz['freqs']
        ax.hlines(freqs_ea_i, x_positions[i] - 0.03, x_positions[i], color='tab:green', lw=0.5, alpha=0.7)
        ax.hlines(freqs_fdfd_i, x_positions[i], x_positions[i] + 0.03, color='tab:red', lw=0.5, alpha=0.7)
        ax.text(x_positions[i], -0.025, f'{m["theta_deg"]:.1f}°',
                ha='center', va='top', fontsize=8)
    ax.axhline(0.0, color='0.4', ls='--', lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel(r'$(\omega - \omega_{\mathrm{ref}}) a / 2\pi c$')
    ax.set_title('(f) Level diagrams')

    k_label, _, _ = get_target_k_info(LATTICE_TYPE)
    lattice_name = get_lattice_display_name(LATTICE_TYPE)
    fig.suptitle(f'EA vs FDFD: Multi-Angle Spectral Validation ({lattice_name}, Band {TARGET_BAND} at {k_label})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    summary_fig = run_dir / 'multi_angle_summary.png'
    fig.savefig(summary_fig, dpi=200)
    plt.close(fig)
    log(f'Saved multi-angle summary: {summary_fig}')

    # ── Text report ──
    report_lines = [
        '# EA vs FDFD Multi-Angle Validation Report',
        f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Parameters',
        f'- Lattice: {lattice_name}, a={A}, r/a={R_OVER_A}, ε_rod={EPS_HOLE}, ε_bg={EPS_BG}, TM',
        f'- Target: Band {TARGET_BAND} at {k_label}-point',
        f'- Envelope reference mode: {OMEGA_REF_MODE}',
        f'- Physical target-frequency mode: {TARGET_FREQUENCY_MODE}',
        f'- Subspace bands: {SUBSPACE_BANDS}',
        f'- All bands: {list(ALL_BANDS)}',
        '- Comparison frame: frequencies plotted as ω - ω_ref from the shared Phase 1 run',
        '',
        '## Results',
        '',
        '| θ (deg) | η | N_cells | BW_EA | BW_FDFD | BW ratio | KS stat | DOS corr | RMS |',
        '|---------|---|---------|-------|---------|----------|---------|----------|-----|',
    ]
    for m in all_metrics:
        report_lines.append(
            f'| {m["theta_deg"]:.2f} | {m["eta"]:.4f} | '
            f'{m.get("n_cells", "?")} | '
            f'{m["bw_ea"]:.2e} | {m["bw_fdfd"]:.2e} | '
            f'{m["bw_ratio"]:.3f} | {m["ks_stat"]:.3f} | '
            f'{m["dos_corr"]:.3f} | {m["rms"]:.2e} |'
        )
    report_lines.extend([
        '',
        '## Key Observations',
        '',
        f'- BW ratio convergence: {bw_ratios[-1]:.3f} at θ={thetas[-1]:.1f}° → {bw_ratios[0]:.3f} at θ={thetas[0]:.1f}°',
        f'- KS stat range: {min(ks_stats):.3f} – {max(ks_stats):.3f}',
        f'- DOS corr range: {min(dos_corrs):.3f} – {max(dos_corrs):.3f}',
        '',
        '## Files',
        f'- Summary figure: {summary_fig}',
    ])
    for m in all_metrics:
        report_lines.append(f'- {m["case"]}: {m.get("plot_path", "N/A")}')

    report_path = run_dir / 'validation_report.md'
    report_path.write_text('\n'.join(report_lines))
    log(f'Saved validation report: {report_path}')

    # Save aggregated metrics
    save_json(all_metrics, run_dir / 'all_metrics.json')
    return summary_fig


# ═══════════════════════════════════════════════════════════════
#  Subspace gap diagnostic
# ═══════════════════════════════════════════════════════════════

def run_subspace_diagnostic(ea_result, run_dir):
    """Check if subspace bands remain isolated across registry sweep."""
    cdir = Path(ea_result['cdir'])
    h5_path = cdir / 'phase1_multiband_data.h5'
    if not h5_path.exists():
        log('  Skipping subspace diagnostic: no Phase 1 data', level='WARN')
        return None

    with h5py.File(h5_path, 'r') as hf:
        stencil_grp = hf['stencil']
        registry_omega_all = stencil_grp['registry_omega_all'][:]

    # registry_omega_all shape: (n_reg, n_reg, N_all_bands)
    n_reg1, n_reg2, n_all = registry_omega_all.shape

    # For each subspace band, compute min gap to nearest external band
    subspace_set = set(SUBSPACE_BANDS)
    all_set = set(ALL_BANDS)
    external_bands = sorted(all_set - subspace_set)

    results = {}
    for band_idx in SUBSPACE_BANDS:
        omega_band = registry_omega_all[:, :, band_idx]
        max_omega = float(np.max(omega_band))
        min_omega = float(np.min(omega_band))
        modulation_range = max_omega - min_omega

        # Min gap to adjacent bands
        min_gap_below = float('inf')
        min_gap_above = float('inf')
        if band_idx > 0 and band_idx - 1 in all_set:
            gap_below = omega_band - registry_omega_all[:, :, band_idx - 1]
            min_gap_below = float(np.min(gap_below))
        if band_idx < n_all - 1 and band_idx + 1 in all_set:
            gap_above = registry_omega_all[:, :, band_idx + 1] - omega_band
            min_gap_above = float(np.min(gap_above))

        min_gap = min(min_gap_below, min_gap_above)
        isolation = min_gap / modulation_range if modulation_range > 0 else float('inf')

        results[f'band_{band_idx}'] = {
            'min_omega': min_omega,
            'max_omega': max_omega,
            'modulation_range': modulation_range,
            'min_gap_below': min_gap_below,
            'min_gap_above': min_gap_above,
            'min_gap': min_gap,
            'isolation_ratio': isolation,
        }

        status = 'OK' if isolation > 1.0 else ('WARN' if isolation > 0.5 else 'DANGER')
        log(f'  Band {band_idx}: ω∈[{min_omega:.5f}, {max_omega:.5f}], '
            f'modulation={modulation_range:.4e}, min_gap={min_gap:.4e}, '
            f'isolation={isolation:.2f} [{status}]')

    save_json(results, Path(run_dir) / ea_result['case'] / 'subspace_diagnostic.json')
    return results


# ═══════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Overnight multi-angle EA vs FDFD validation')
    parser.add_argument('--skip-ea', action='store_true', help='Skip EA runs')
    parser.add_argument('--skip-fdfd', action='store_true', help='Skip FDFD runs')
    parser.add_argument('--comparison-only', action='store_true', help='Only comparison step')
    parser.add_argument('--skip-thread-test', action='store_true', help='Skip MPB thread test')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing run directory (e.g. run_20260313_002139)')
    parser.add_argument('--cases', nargs='+', default=None,
                        help='Subset of cases to run (e.g. 10deg 2deg)')
    parser.add_argument('--mt', type=float, default=2.0,
                        help='Default M_inv_max_trace for comparison (default: 2.0)')
    parser.add_argument('--subspace-bands', type=str, default=None,
                        help='Comma-separated subspace band list (e.g. 0,1,2,3)')
    parser.add_argument('--all-bands', type=str, default=None,
                        help='Comma-separated all-band list used in Phase 1 (e.g. 0,1,2,3,4,5,6,7,8,9,10,11)')
    parser.add_argument('--target-band', type=int, default=None,
                        help='Target band inside the chosen subspace')
    parser.add_argument('--only-multi4', action='store_true',
                        help='Only run the primary multi-band configuration')
    parser.add_argument('--only-raw-mt', action='store_true',
                        help='Only run the unregularized multi-band solve (mt_raw)')
    parser.add_argument('--omega-ref-mode', type=str,
                        choices=['zero', 'mean', 'min', 'max', 'median', 'target_extremum', 'interior_extremum'],
                        default='zero',
                        help='How to choose the envelope reference frequency ω_ref')
    parser.add_argument('--target-frequency-mode', type=str,
                        choices=['mean', 'min', 'max', 'median', 'target_extremum', 'interior_extremum'],
                        default='mean',
                        help='How to choose the physical target frequency used for FDFD sigma when ω_ref=0')
    parser.add_argument('--mpb-resolution', type=int, default=None,
                        help='Override MPB resolution for selected cases')
    parser.add_argument('--ea-registry', type=int, default=None,
                        help='Override MPB registry sample count for selected cases')
    parser.add_argument('--ea-ns', type=int, default=None,
                        help='Override Phase 1/2/3 moire grid size for selected cases')
    parser.add_argument('--fdfd-res', type=int, default=None,
                        help='Override FDFD spatial resolution (pixels per a) for selected cases')
    parser.add_argument('--n-modes', type=int, default=None,
                        help='Override the number of EA/FDFD modes to compute')
    parser.add_argument('--disable-drift', action='store_true',
                        help='Disable the Phase 2/3 drift term')
    parser.add_argument('--disable-offdiag-A', action='store_true',
                        help='Disable off-diagonal Berry-connection transport terms')
    parser.add_argument('--disable-born-huang', action='store_true',
                        help='Disable the Born-Huang term in Phase 2/3')
    parser.add_argument('--disable-offdiag-v-drift', action='store_true',
                        help='Disable off-diagonal drift velocity matrix elements from Bloch fields')
    parser.add_argument('--mpb-fd-order', type=int, choices=[2, 4, 6], default=None,
                        help='Override the Phase 1 MPB stencil finite-difference order')
    parser.add_argument('--r-over-a', type=float, default=None,
                        help='Override the cylinder radius r/a used in Phase 1 and FDFD')
    parser.add_argument('--eps-bg', type=float, default=None,
                        help='Override the background dielectric constant')
    parser.add_argument('--eps-hole', type=float, default=None,
                        help='Override the rod dielectric constant')
    parser.add_argument('--lattice-type', type=str, choices=['square', 'honeycomb'], default='square',
                        help='Underlying monolayer lattice type for Phase 1 and FDFD')
    args = parser.parse_args()

    if args.comparison_only:
        args.skip_ea = True
        args.skip_fdfd = True

    global SUBSPACE_BANDS, ALL_BANDS, TARGET_BAND, OMEGA_REF_MODE, TARGET_FREQUENCY_MODE
    global INCLUDE_DRIFT_TERM, INCLUDE_OFFDIAG_A, BAND_CONFIGS, R_OVER_A, EPS_BG, EPS_HOLE, LATTICE_TYPE
    global INCLUDE_BORN_HUANG, INCLUDE_OFFDIAG_V_DRIFT, MPB_FD_ORDER

    if args.subspace_bands is not None:
        SUBSPACE_BANDS = parse_int_list(args.subspace_bands)
    LATTICE_TYPE = args.lattice_type
    if args.all_bands is not None:
        ALL_BANDS = parse_int_list(args.all_bands)
    else:
        max_band = max(SUBSPACE_BANDS) if SUBSPACE_BANDS else 0
        ALL_BANDS = list(range(max(max_band + 1, len(ALL_BANDS))))
    if args.target_band is not None:
        TARGET_BAND = int(args.target_band)
    OMEGA_REF_MODE = args.omega_ref_mode
    TARGET_FREQUENCY_MODE = args.target_frequency_mode
    INCLUDE_DRIFT_TERM = not args.disable_drift
    INCLUDE_OFFDIAG_A = not args.disable_offdiag_A
    INCLUDE_BORN_HUANG = not args.disable_born_huang
    INCLUDE_OFFDIAG_V_DRIFT = not args.disable_offdiag_v_drift
    if args.mpb_fd_order is not None:
        MPB_FD_ORDER = int(args.mpb_fd_order)
    if args.r_over_a is not None:
        R_OVER_A = float(args.r_over_a)
    if args.eps_bg is not None:
        EPS_BG = float(args.eps_bg)
    if args.eps_hole is not None:
        EPS_HOLE = float(args.eps_hole)

    if TARGET_BAND not in SUBSPACE_BANDS:
        print(f'ERROR: target band {TARGET_BAND} is not in subspace {SUBSPACE_BANDS}')
        sys.exit(1)
    missing = sorted(set(SUBSPACE_BANDS) - set(ALL_BANDS))
    if missing:
        print(f'ERROR: subspace bands missing from all-band list: {missing}')
        sys.exit(1)

    if args.only_multi4:
        BAND_CONFIGS = [{
            'name': 'multi4',
            'subspace_bands': list(SUBSPACE_BANDS),
            'mt_values': [None] if args.only_raw_mt else [None, 2.0, 0.5],
        }]
    else:
        BAND_CONFIGS = [
            {'name': 'multi4', 'subspace_bands': list(SUBSPACE_BANDS), 'mt_values': [None, 2.0, 0.5]},
            {'name': 'pair2', 'subspace_bands': list(SUBSPACE_BANDS[:2]), 'mt_values': [None]},
            {'name': 'single', 'subspace_bands': [SUBSPACE_BANDS[0]], 'mt_values': [None]},
        ]

    # Select cases
    cases = [dict(case) for case in ANGLE_CASES]
    if args.cases is not None:
        valid_names = {c['name'] for c in ANGLE_CASES}
        for name in args.cases:
            if name not in valid_names:
                print(f'ERROR: Unknown case "{name}". Valid: {sorted(valid_names)}')
                sys.exit(1)
        cases = [c for c in ANGLE_CASES if c['name'] in args.cases]
        cases = [dict(case) for case in cases]

    for case in cases:
        if args.mpb_resolution is not None:
            case['mpb_resolution'] = args.mpb_resolution
        if args.ea_registry is not None:
            case['ea_registry'] = args.ea_registry
        if args.ea_ns is not None:
            case['ea_ns'] = args.ea_ns
        if args.fdfd_res is not None:
            case['fdfd_res'] = args.fdfd_res
        if args.n_modes is not None:
            case['n_modes'] = args.n_modes

    # Create or resume run directory
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = SCRIPT_DIR / 'overnight_validation' / resume_path
        if not resume_path.exists():
            print(f'ERROR: Resume directory not found: {resume_path}')
            sys.exit(1)
        run_dir = resume_path
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = SCRIPT_DIR / 'overnight_validation' / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=True)

    global _LOG_FILE
    _LOG_FILE = run_dir / 'pipeline.log'

    log_section('OVERNIGHT MULTI-ANGLE EA vs FDFD VALIDATION')
    log(f'Run directory: {run_dir}')
    if args.resume:
        log(f'RESUMING from existing run directory')
    log(f'Cases: {[c["name"] for c in cases]}')
    log(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Available RAM: ~40-45 GB, CPUs: 32')

    # Save configuration
    save_json({
        'cases': cases,
        'subspace_bands': SUBSPACE_BANDS,
        'all_bands': list(ALL_BANDS),
        'r_over_a': R_OVER_A,
        'eps_bg': EPS_BG,
        'eps_hole': EPS_HOLE,
        'omega0': OMEGA0,
        'omega_ref_mode': OMEGA_REF_MODE,
        'target_frequency_mode': TARGET_FREQUENCY_MODE,
        'include_drift_term': INCLUDE_DRIFT_TERM,
        'include_offdiag_A': INCLUDE_OFFDIAG_A,
        'include_born_huang': INCLUDE_BORN_HUANG,
        'include_offdiag_v_drift': INCLUDE_OFFDIAG_V_DRIFT,
        'mpb_fd_order': MPB_FD_ORDER,
        'target_band': TARGET_BAND,
        'n_workers': N_WORKERS,
        'domain_type': 'single_moire_cell',
    }, run_dir / 'pipeline_config.json')

    t_total_start = time.time()

    # ── Step 0: MPB thread test ──
    if not args.skip_thread_test:
        try:
            verify_single_thread()
        except Exception as e:
            log(f'MPB thread test FAILED: {e}', level='ERROR')
            traceback.print_exc()
            sys.exit(1)

    # ── Step 1: EA pipeline ──
    # Phase 1 (MPB sweep) is run ONCE per angle with full subspace, then shared.
    # Phase 2+3 are run per (angle, band_config, mt_value).
    ea_results = {}
    if not args.skip_ea:
        log_section('STEP 1: EA PIPELINE')
        for case in cases:
            case_name = case['name']

            # Phase 1: run once per angle (shared across band configs and mt values)
            log_section(f'Phase 1: {case_name} (shared across band configs)')
            try:
                shared_phase1_cdir = run_ea_phase1(case, run_dir)
            except Exception as e:
                log(f'  Phase 1 {case_name} FAILED: {e}', level='ERROR')
                traceback.print_exc()
                continue  # skip all configs for this angle

            # Phase 2+3: run per (band_config, mt_value)
            for bconf in BAND_CONFIGS:
                bc_name = bconf['name']
                bc_bands = bconf['subspace_bands']
                bc_mt_values = bconf['mt_values']
                log_section(f'EA {case_name} / {bc_name} (bands={bc_bands})')
                for mt_value in bc_mt_values:
                    mt_str = f'mt{mt_value}' if mt_value is not None else 'mt_raw'
                    config_prefix = f'{bc_name}/' if bc_name != 'multi4' else ''
                    label = f'{case_name}/{config_prefix}{mt_str}'
                    log(f'')
                    try:
                        t0 = time.time()
                        result = run_ea_phase23(
                            case, run_dir, mt_value, shared_phase1_cdir,
                            band_config_name=bc_name, subspace_bands=bc_bands,
                        )
                        t_total = time.time() - t0
                        resumed = result.get('resumed', False)
                        status = 'RESUMED' if resumed else f'complete in {t_total:.1f}s'
                        log(f'  EA {label} {status}: BW={result["bw_ea"]:.4e}')
                        ea_results[(case_name, bc_name, mt_str)] = result
                    except Exception as e:
                        log(f'  EA {label} FAILED: {e}', level='ERROR')
                        traceback.print_exc()

            # Subspace diagnostic (using shared Phase 1)
            run_subspace_diagnostic({'cdir': shared_phase1_cdir, 'case': case_name}, run_dir)

    # ── Step 2: FDFD solves ──
    fdfd_results = {}
    if not args.skip_fdfd:
        log_section('STEP 2: FDFD REFERENCE SOLVES')
        for case in cases:
            case_name = case['name']
            try:
                shared_phase1_cdir = run_dir / case_name / 'shared_phase1' / 'candidate_0000'
                target_freq_abs = load_phase1_absolute_target_frequency(shared_phase1_cdir)
                t0 = time.time()
                result = run_fdfd_solve(case, run_dir, target_freq_abs)
                t_total = time.time() - t0
                fdfd_results[case_name] = result
                bw = float(result['freqs'][-1] - result['freqs'][0])
                log(f'  FDFD {case_name}: BW={bw:.4e}, total={t_total:.1f}s')
                # Save a copy for the summary plotter
                np.savez(
                    run_dir / case_name / 'fdfd_freqs.npz',
                    freqs=result['freqs'],
                    freqs_shifted=result['freqs_shifted'],
                    comparison_center=result['comparison_center'],
                )
            except Exception as e:
                log(f'  FDFD {case_name} FAILED: {e}', level='ERROR')
                traceback.print_exc()
    else:
        # Try to load existing FDFD results
        for case in cases:
            case_name = case['name']
            # Try commensurate, then moiré cell, then old naming
            fdfd_path = run_dir / case_name / f'fdfd_commensurate_res{case["fdfd_res"]}_k{case["n_modes"]}.npz'
            if not fdfd_path.exists():
                fdfd_path = run_dir / case_name / f'fdfd_moire_res{case["fdfd_res"]}_k{case["n_modes"]}.npz'
            if not fdfd_path.exists():
                fdfd_path = run_dir / case_name / f'fdfd_res{case["fdfd_res"]}_k{case["n_modes"]}.npz'
            if fdfd_path.exists():
                data = np.load(fdfd_path)
                fdfd_results[case_name] = {
                    'case': case_name,
                    'freqs': np.sort(data['freqs']),
                    'freqs_shifted': np.sort(data['freqs_shifted']) if 'freqs_shifted' in data else None,
                    'comparison_center': float(data['comparison_center']) if 'comparison_center' in data else float(data['omega_target_abs']) if 'omega_target_abs' in data else 0.0,
                    'output_path': str(fdfd_path),
                    'reused': True,
                }

    # ── Step 3: Spectral comparison ──
    log_section('STEP 3: SPECTRAL COMPARISON')
    all_metrics = []
    mt_comparison_str = 'mt_raw' if args.only_raw_mt else (f'mt{args.mt}' if args.mt is not None else 'mt_raw')

    for bconf in BAND_CONFIGS:
        bc_name = bconf['name']
        bc_mt_str = mt_comparison_str if bc_name == 'multi4' else 'mt_raw'
        config_prefix = f'{bc_name}/' if bc_name != 'multi4' else ''

        log(f'')
        log(f'── Band config: {bc_name} (bands={bconf["subspace_bands"]}) ──')

        for case in cases:
            case_name = case['name']
            ea_key = (case_name, bc_name, bc_mt_str)
            comparison_label = f'{case_name}/{config_prefix}{bc_mt_str}'

            # Load EA results
            if ea_key in ea_results:
                freqs_ea = ea_results[ea_key]['freqs_ea']
                center_freq_abs = ea_results[ea_key]['omega_ref']
                theta_deg = ea_results[ea_key]['theta_deg']
                eta = ea_results[ea_key]['eta']
                n_cells = ea_results[ea_key].get('n_cells', compute_case_n_cells(case))
            else:
                ea_npz = run_dir / case_name / f'{config_prefix}{bc_mt_str}' / 'ea_freqs.npz'
                if ea_npz.exists():
                    ea_data = np.load(ea_npz)
                    freqs_ea = np.sort(ea_data['freqs'])
                    center_freq_abs = float(ea_data['comparison_center']) if 'comparison_center' in ea_data else float(ea_data['omega_ref'])
                    candidate = build_candidate(case, bconf['subspace_bands'])
                    theta_deg = candidate['theta_deg']
                    eta = candidate['eta']
                    n_cells = compute_case_n_cells(case)
                else:
                    log(f'  SKIP {comparison_label}: no EA data', level='WARN')
                    continue

            # Load FDFD results
            if case_name in fdfd_results:
                freqs_fdfd = fdfd_results[case_name]['freqs']
            else:
                fdfd_npz = run_dir / case_name / 'fdfd_freqs.npz'
                if fdfd_npz.exists():
                    freqs_fdfd = np.sort(np.load(fdfd_npz)['freqs'])
                else:
                    log(f'  SKIP {comparison_label}: no FDFD data', level='WARN')
                    continue

            # Save for summary plotter
            plot_label = f'{case_name}_{bc_name}'
            np.savez(
                run_dir / case_name / f'ea_freqs_{bc_name}_best.npz',
                freqs=freqs_ea,
                freqs_shifted=freqs_ea - center_freq_abs,
                comparison_center=center_freq_abs,
            )
            if not (run_dir / case_name / 'fdfd_freqs.npz').exists():
                np.savez(
                    run_dir / case_name / 'fdfd_freqs.npz',
                    freqs=freqs_fdfd,
                    freqs_shifted=freqs_fdfd - center_freq_abs,
                    comparison_center=center_freq_abs,
                )
            # Backward compat: also save ea_freqs_best.npz for multi4
            if bc_name == 'multi4':
                np.savez(
                    run_dir / case_name / 'ea_freqs_best.npz',
                    freqs=freqs_ea,
                    freqs_shifted=freqs_ea - center_freq_abs,
                    comparison_center=center_freq_abs,
                )

            try:
                metrics = compare_spectra(
                    freqs_ea, freqs_fdfd, plot_label,
                    theta_deg, eta, run_dir / case_name, center_freq_abs
                )
                metrics['n_cells'] = n_cells
                metrics['band_config'] = bc_name
                metrics['case'] = case_name
                all_metrics.append(metrics)
                log(f'  {comparison_label}: BW_ratio={metrics["bw_ratio"]:.3f}, '
                    f'KS={metrics["ks_stat"]:.3f}, DOS_corr={metrics["dos_corr"]:.3f}, '
                    f'RMS={metrics["rms"]:.2e}')
            except Exception as e:
                log(f'  Comparison {comparison_label} FAILED: {e}', level='ERROR')
                traceback.print_exc()

    # ── Step 3b: Multi-config comparison per angle ──
    log('')
    log('Per-angle band-config and regularization sensitivity:')
    for case in cases:
        case_name = case['name']
        if case_name not in fdfd_results:
            continue
        freqs_fdfd = fdfd_results[case_name]['freqs']
        center_freq_abs = float(fdfd_results[case_name].get('comparison_center', 0.0))
        bw_fdfd = float(freqs_fdfd[-1] - freqs_fdfd[0])
        sensitivity_results = []
        for bconf in BAND_CONFIGS:
            bc_name = bconf['name']
            config_prefix = f'{bc_name}/' if bc_name != 'multi4' else ''
            for mt_value in bconf['mt_values']:
                mt_str = f'mt{mt_value}' if mt_value is not None else 'mt_raw'
                ea_key = (case_name, bc_name, mt_str)
                if ea_key in ea_results:
                    freqs_ea = ea_results[ea_key]['freqs_ea']
                else:
                    ea_npz = run_dir / case_name / f'{config_prefix}{mt_str}' / 'ea_freqs.npz'
                    if ea_npz.exists():
                        freqs_ea = np.sort(np.load(ea_npz)['freqs'])
                    else:
                        continue
                bw_ea = float(freqs_ea[-1] - freqs_ea[0])
                n_cmp = min(len(freqs_ea), len(freqs_fdfd))
                rms = float(np.sqrt(np.mean(((freqs_ea[:n_cmp] - center_freq_abs) - (freqs_fdfd[:n_cmp] - center_freq_abs)) ** 2)))
                sensitivity_results.append({
                    'band_config': bc_name,
                    'mt': mt_value,
                    'bw_ea': bw_ea,
                    'bw_ratio': bw_ea / bw_fdfd if bw_fdfd > 0 else float('nan'),
                    'rms': rms,
                })
                log(f'  {case_name} {bc_name}/{mt_str}: BW_EA={bw_ea:.4e}, ratio={bw_ea/bw_fdfd:.3f}, RMS={rms:.2e}')
        if sensitivity_results:
            save_json(sensitivity_results, run_dir / case_name / 'sensitivity.json')

    # ── Multi-angle summary (per band config) ──
    for bconf in BAND_CONFIGS:
        bc_name = bconf['name']
        bc_metrics = [m for m in all_metrics if m.get('band_config') == bc_name]
        if len(bc_metrics) >= 2:
            try:
                log(f'Generating multi-angle summary for {bc_name}...')
                generate_multi_angle_summary(bc_metrics, run_dir)
            except Exception as e:
                log(f'Multi-angle summary ({bc_name}) FAILED: {e}', level='ERROR')
                traceback.print_exc()

    t_total = time.time() - t_total_start
    log_section('PIPELINE COMPLETE')
    log(f'Total wall time: {t_total:.1f}s ({t_total/3600:.2f}h)')
    log(f'Results directory: {run_dir}')
    log(f'End time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
