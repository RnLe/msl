#!/usr/bin/env python
"""
Miniband convergence tests — BZ-resolved observables.

Sweeps convergence knobs (N_K, Ns, n_modes, fd_order) and measures
miniband bandwidth, inter-band gap, and flatness ratio across the
moiré Brillouin zone.

Usage:
    python convergence_miniband.py --only honeycomb
    python convergence_miniband.py --only honeycomb --theta 1.1
    python convergence_miniband.py --only honeycomb --angles 0.5,1.1,3.0
"""

import sys, math, json, time, gc, argparse, shutil, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))
sys.path.insert(0, str(PROJECT_ROOT / "results_bands"))
sys.path.insert(0, str(PROJECT_ROOT / "thesis_results"))

import phase2_mpb_v3 as p2
from phase3_mpb_v3 import (
    compute_sigma,
    _regularize_M_inv,
)
from miniband_sweep import (
    load_phase2_from_h5,
    prepare_data,
    assemble_bloch_hamiltonian,
    sweep_k_points,
    LATTICE_CONFIGS,
    M_INV_MAX_TRACE,
)
from compute_miniband_structure import (
    get_moire_bz_path,
    extract_miniband_metrics,
)

try:
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = SCRIPT_DIR / "miniband_convergence"

# Convergence sweep defaults
DEFAULT_NK_VALUES = (13, 25, 49)       # n_per_segment = (5, 9, 17)
DEFAULT_NS_VALUES_HC = (16, 32, 64, 128)  # honeycomb (must divide Ns_full=128)
DEFAULT_NS_VALUES = (16, 32, 64)       # hex/square (OOM above 64)
DEFAULT_NMODES_VALUES = (6, 10, 20)
DEFAULT_FD_ORDERS = (2, 4)
DEFAULT_ANGLES = (0.5, 1.1, 3.0)

# Reference/default values for sweeps
REF_NK = 25        # n_per_segment = 9
REF_NS = 64
REF_NMODES = 10
REF_FD_ORDER = 4

LATTICE_META = {
    'honeycomb': {'lattice_type': 'honeycomb', 'symmetry': 'C6', 'a': 1.0,
                  'run_dir_pattern': 'thesis_honeycomb_K_b1_2026*',
                  'run_dir_exclude': 'TE'},
    'hex':       {'lattice_type': 'hex', 'symmetry': 'C2', 'a': 1.0,
                  'run_dir_pattern': 'thesis_hex_M_b1_2026*',
                  'run_dir_exclude': None},
    'square':    {'lattice_type': 'square', 'symmetry': 'C4', 'a': 1.0,
                  'run_dir_pattern': 'thesis_square_M_b3_2026*',
                  'run_dir_exclude': None},
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def n_per_segment_from_nk(nk):
    """Invert N_K = 3*(n_per_segment-1)+1 for 3-segment BZ path."""
    return (nk - 1) // 3 + 1


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


def find_source_dir(lattice_name):
    meta = LATTICE_META[lattice_name]
    base = PROJECT_ROOT / "runsV3"
    candidates = sorted(base.glob(meta['run_dir_pattern']))
    if meta['run_dir_exclude']:
        candidates = [c for c in candidates if meta['run_dir_exclude'] not in c.name]
    if not candidates:
        return None
    return candidates[-1] / "candidate_0000"


def compute_phase2_at_angle(theta_deg, source_cdir, lattice_name):
    meta = LATTICE_META[lattice_name]
    lattice_type = meta['lattice_type']
    symmetry = meta['symmetry']
    a = meta['a']

    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    phase1_src = source_cdir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"

    tmp_base = tempfile.mkdtemp(prefix=f"mbconv_{lattice_name}_{theta_deg:.2f}_")
    work_dir = Path(tmp_base) / "candidate_0000"
    work_dir.mkdir(parents=True)

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

    p2_config = {
        'include_born_huang': False,
        'include_drift_term': True,
        'use_parallel_transport_gauge': True,
        'n_extra_bands': 4,
        'mpb_fd_order': 4,
    }
    p2.process_candidate_phase2_v3(str(work_dir), p2_config)

    if HAS_SYMMETRIZE and symmetry:
        try:
            symmetrize_phase2(work_dir, symmetry)
        except Exception as e:
            print(f"  WARNING: Symmetrization ({symmetry}) failed: {e}")

    gc.collect()
    return tmp_base, work_dir


def sanitize(obj):
    """Convert numpy types for JSON serialization."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Core: single K-sweep with specific parameters → miniband metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_single_sweep(data_full, lattice_type, ns, nk, n_modes, fd_order,
                     sigma, label=""):
    """
    Run one K-sweep with given parameters and return miniband metrics.

    Returns dict with bandwidths, gaps, flatness, eigenvalues, wall_time.
    """
    # Downsample to target Ns
    data = prepare_data(data_full, ns)
    # Pre-regularize M_inv
    data['M_inv_reg'] = _regularize_M_inv(data['M_inv'].copy(), M_INV_MAX_TRACE)

    # Build BZ path
    n_per_seg = n_per_segment_from_nk(nk)
    q_points, q_dist, tick_pos, tick_labels = get_moire_bz_path(
        data['B_moire'], n_per_segment=n_per_seg, lattice_type=lattice_type
    )
    actual_nk = len(q_points)

    t0 = time.time()
    all_evals = sweep_k_points(
        data, q_points, n_modes,
        include_offdiag_A=True, label=label,
        sigma=sigma, order=fd_order,
    )
    wall_time = time.time() - t0

    metrics = extract_miniband_metrics(all_evals, q_dist)

    # Extract key observables for first 3 bands
    bandwidths = [m['bandwidth'] for m in metrics[:3]]
    gaps = [m.get('gap_above') for m in metrics[:3]]
    flatness = [m.get('flatness') for m in metrics[:3]]

    return {
        'Ns': ns, 'N_K': actual_nk, 'n_per_seg': n_per_seg,
        'n_modes': n_modes, 'fd_order': fd_order,
        'sigma': float(sigma),
        'bandwidths': bandwidths,
        'gaps': gaps,
        'flatness': flatness,
        'metrics': metrics,
        'wall_time_s': wall_time,
        'all_evals': all_evals.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sweep functions
# ─────────────────────────────────────────────────────────────────────────────

def sweep_NK(data_full, lattice_type, sigma, nk_values, ns=REF_NS,
             n_modes=REF_NMODES, fd_order=REF_FD_ORDER):
    """Sweep K-path density at fixed Ns, n_modes, fd_order."""
    print(f"\n  {'─'*60}")
    print(f"  SWEEP: N_K  (Ns={ns}, n_modes={n_modes}, fd_order={fd_order})")
    print(f"  N_K values: {nk_values}")
    print(f"  {'─'*60}")

    results = []
    for nk in nk_values:
        label = f"NK={nk}"
        r = run_single_sweep(data_full, lattice_type, ns, nk, n_modes,
                             fd_order, sigma, label)
        bw = r['bandwidths']
        print(f"    N_K={r['N_K']:3d}: BW₀={bw[0]:.4e}  "
              f"BW₁={bw[1]:.4e}  gap₀₁={r['gaps'][0]:.4e}  "
              f"t={r['wall_time_s']:.1f}s")
        results.append(r)
    return results


def sweep_Ns(data_full, lattice_type, sigma, ns_values, nk=REF_NK,
             n_modes=REF_NMODES, fd_order=REF_FD_ORDER):
    """Sweep grid resolution at fixed N_K, n_modes, fd_order."""
    print(f"\n  {'─'*60}")
    print(f"  SWEEP: Ns  (N_K≈{nk}, n_modes={n_modes}, fd_order={fd_order})")
    print(f"  Ns values: {ns_values}")
    print(f"  {'─'*60}")

    results = []
    for ns in ns_values:
        if data_full['Ns1'] % ns != 0:
            print(f"    Ns={ns}: SKIP (not divisor of {data_full['Ns1']})")
            continue
        label = f"Ns={ns}"
        r = run_single_sweep(data_full, lattice_type, ns, nk, n_modes,
                             fd_order, sigma, label)
        bw = r['bandwidths']
        print(f"    Ns={ns:3d}: BW₀={bw[0]:.4e}  "
              f"BW₁={bw[1]:.4e}  gap₀₁={r['gaps'][0]:.4e}  "
              f"t={r['wall_time_s']:.1f}s")
        results.append(r)
    return results


def sweep_nmodes(data_full, lattice_type, sigma, nmodes_values, ns=REF_NS,
                 nk=REF_NK, fd_order=REF_FD_ORDER):
    """Sweep n_modes at fixed Ns, N_K, fd_order."""
    print(f"\n  {'─'*60}")
    print(f"  SWEEP: n_modes  (Ns={ns}, N_K≈{nk}, fd_order={fd_order})")
    print(f"  n_modes values: {nmodes_values}")
    print(f"  {'─'*60}")

    results = []
    for nm in nmodes_values:
        label = f"nm={nm}"
        r = run_single_sweep(data_full, lattice_type, ns, nk, nm,
                             fd_order, sigma, label)
        bw = r['bandwidths']
        print(f"    n_modes={nm:3d}: BW₀={bw[0]:.4e}  "
              f"BW₁={bw[1]:.4e}  gap₀₁={r['gaps'][0]:.4e}  "
              f"t={r['wall_time_s']:.1f}s")
        results.append(r)
    return results


def sweep_fd_order(data_full, lattice_type, sigma, orders, ns=REF_NS,
                   nk=REF_NK, n_modes=REF_NMODES):
    """Sweep FD order at fixed Ns, N_K, n_modes."""
    print(f"\n  {'─'*60}")
    print(f"  SWEEP: fd_order  (Ns={ns}, N_K≈{nk}, n_modes={n_modes})")
    print(f"  Orders: {orders}")
    print(f"  {'─'*60}")

    results = []
    for order in orders:
        label = f"fd={order}"
        r = run_single_sweep(data_full, lattice_type, ns, nk, n_modes,
                             order, sigma, label)
        bw = r['bandwidths']
        print(f"    order={order}: BW₀={bw[0]:.4e}  "
              f"BW₁={bw[1]:.4e}  gap₀₁={r['gaps'][0]:.4e}  "
              f"t={r['wall_time_s']:.1f}s")
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — regenerated after each sweep
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(results, lattice_name, output_dir):
    """Generate all convergence plots from results dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for theta_key, angle_data in results.items():
        theta = float(theta_key)

        # Per-sweep plots
        for sweep_name, sweep_key, x_key, x_label in [
            ('NK', 'NK_sweep', 'N_K', '$N_K$ (K-path points)'),
            ('Ns', 'Ns_sweep', 'Ns', '$N_s$ (grid points)'),
            ('nmodes', 'nmodes_sweep', 'n_modes', '$n_{modes}$'),
        ]:
            if sweep_key not in angle_data:
                continue
            sweep = angle_data[sweep_key]
            if not sweep:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.suptitle(f'{lattice_name}  θ={theta}°  —  {sweep_name} sweep',
                         fontsize=13)

            x_vals = [s[x_key] for s in sweep]

            # (a) Bandwidths
            ax = axes[0]
            for bn in range(min(3, len(sweep[0]['bandwidths']))):
                bw = [s['bandwidths'][bn] for s in sweep]
                ax.plot(x_vals, bw, 'o-', label=f'Band {bn}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Bandwidth')
            ax.set_title('Miniband bandwidth')
            ax.legend()
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

            # (b) Inter-band gaps
            ax = axes[1]
            for bn in range(min(3, len(sweep[0]['gaps']))):
                gaps = [s['gaps'][bn] if s['gaps'][bn] is not None else np.nan
                        for s in sweep]
                ax.plot(x_vals, gaps, 's-', label=f'Gap {bn}→{bn+1}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Inter-band gap')
            ax.set_title('Inter-band gap')
            ax.legend()
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

            # (c) Flatness
            ax = axes[2]
            for bn in range(min(3, len(sweep[0]['flatness']))):
                flat = [s['flatness'][bn] if s['flatness'][bn] is not None
                        else np.nan for s in sweep]
                ax.plot(x_vals, flat, 'D-', label=f'Band {bn}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Flatness (gap/BW)')
            ax.set_title('Flatness ratio')
            ax.legend()

            plt.tight_layout()
            fname = f"{lattice_name}_{theta_key}_{sweep_name}_sweep"
            fig.savefig(output_dir / f"{fname}.png", dpi=150)
            fig.savefig(output_dir / f"{fname}.pdf")
            plt.close(fig)
            print(f"    Saved: {fname}.png/pdf")

        # FD order sweep (bar chart)
        if 'fd_order_sweep' in angle_data and angle_data['fd_order_sweep']:
            sweep = angle_data['fd_order_sweep']
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            fig.suptitle(f'{lattice_name}  θ={theta}°  —  FD order comparison',
                         fontsize=13)

            orders = [s['fd_order'] for s in sweep]
            x_pos = np.arange(len(orders))

            for panel, (ax, ylabel, key) in enumerate(zip(
                axes, ['Bandwidth', 'Inter-band gap', 'Flatness'],
                ['bandwidths', 'gaps', 'flatness']
            )):
                for bn in range(min(3, len(sweep[0][key]))):
                    vals = [s[key][bn] if s[key][bn] is not None else 0
                            for s in sweep]
                    ax.bar(x_pos + bn * 0.25, vals, 0.2,
                           label=f'Band {bn}')
                ax.set_xticks(x_pos + 0.25)
                ax.set_xticklabels([str(o) for o in orders])
                ax.set_xlabel('FD order')
                ax.set_ylabel(ylabel)
                ax.legend()

            plt.tight_layout()
            fname = f"{lattice_name}_{theta_key}_fd_order_sweep"
            fig.savefig(output_dir / f"{fname}.png", dpi=150)
            fig.savefig(output_dir / f"{fname}.pdf")
            plt.close(fig)
            print(f"    Saved: {fname}.png/pdf")

    # Multi-angle summary (if multiple angles)
    all_thetas = sorted(results.keys(), key=float)
    if len(all_thetas) > 1:
        _plot_multi_angle_summary(results, lattice_name, all_thetas, output_dir)


def _plot_multi_angle_summary(results, lattice_name, all_thetas, output_dir):
    """2×2 summary: BW₀ vs each knob, one line per angle."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{lattice_name} — Convergence Overview (Band 0 BW)',
                 fontsize=14)

    sweep_configs = [
        ('NK_sweep', 'N_K', '$N_K$', axes[0, 0]),
        ('Ns_sweep', 'Ns', '$N_s$', axes[0, 1]),
        ('nmodes_sweep', 'n_modes', '$n_{modes}$', axes[1, 0]),
    ]

    for sweep_key, x_key, xlabel, ax in sweep_configs:
        for theta_key in all_thetas:
            if sweep_key not in results[theta_key]:
                continue
            sweep = results[theta_key][sweep_key]
            if not sweep:
                continue
            x = [s[x_key] for s in sweep]
            bw0 = [s['bandwidths'][0] for s in sweep]
            ax.plot(x, bw0, 'o-', label=f'θ={theta_key}°')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('BW₀')
        ax.legend()
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    # FD order panel
    ax = axes[1, 1]
    for theta_key in all_thetas:
        if 'fd_order_sweep' not in results[theta_key]:
            continue
        sweep = results[theta_key]['fd_order_sweep']
        if not sweep:
            continue
        orders = [s['fd_order'] for s in sweep]
        bw0 = [s['bandwidths'][0] for s in sweep]
        ax.plot(orders, bw0, 'o-', label=f'θ={theta_key}°')
    ax.set_xlabel('FD order')
    ax.set_ylabel('BW₀')
    ax.legend()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))

    plt.tight_layout()
    fname = f"{lattice_name}_convergence_overview"
    fig.savefig(output_dir / f"{fname}.png", dpi=150)
    fig.savefig(output_dir / f"{fname}.pdf")
    plt.close(fig)
    print(f"    Saved: {fname}.png/pdf")


# ─────────────────────────────────────────────────────────────────────────────
# JSON persistence (incremental)
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_results, lattice_name, output_dir):
    """Save results and merge with existing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"convergence_miniband_{lattice_name}.json"

    save_data = sanitize({
        'lattice': lattice_name,
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    })

    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
        existing_results = existing.get('results', {})
        for theta_key, angle_data in all_results.items():
            if theta_key not in existing_results:
                existing_results[theta_key] = {}
            existing_results[theta_key].update(sanitize(angle_data))
        save_data['results'] = existing_results

    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Miniband convergence tests (BZ-resolved)"
    )
    parser.add_argument("--only", type=str, default=None,
                        choices=['honeycomb', 'hex', 'square'])
    parser.add_argument("--angles", type=str, default="0.5,1.1,3.0",
                        help="Comma-separated angles (degrees)")
    parser.add_argument("--theta", type=float, default=None,
                        help="Single angle (overrides --angles)")
    parser.add_argument("--skip_NK", action='store_true')
    parser.add_argument("--skip_Ns", action='store_true')
    parser.add_argument("--skip_nmodes", action='store_true')
    parser.add_argument("--skip_fd", action='store_true')
    parser.add_argument("--nk_values", type=str, default="13,25,49")
    parser.add_argument("--ns_values", type=str, default=None,
                        help="Comma-separated Ns values (default: auto)")
    parser.add_argument("--nmodes_values", type=str, default="6,10,20")
    parser.add_argument("--fd_orders", type=str, default="2,4")
    parser.add_argument("--use_existing_phase2", action='store_true')
    args = parser.parse_args()

    if args.theta is not None:
        angles = [args.theta]
    else:
        angles = [float(x) for x in args.angles.split(',')]

    nk_values = tuple(int(x) for x in args.nk_values.split(','))
    nmodes_values = tuple(int(x) for x in args.nmodes_values.split(','))
    fd_orders = tuple(int(x) for x in args.fd_orders.split(','))

    lattices = [args.only] if args.only else ['honeycomb', 'hex', 'square']

    print(f"\n{'#'*72}")
    print(f"  MINIBAND CONVERGENCE ANALYSIS")
    print(f"  Angles: {angles}")
    print(f"  Lattices: {lattices}")
    print(f"  N_K values: {nk_values}")
    print(f"  n_modes values: {nmodes_values}")
    print(f"  FD orders: {fd_orders}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*72}\n")

    for lattice_name in lattices:
        meta = LATTICE_META[lattice_name]
        lattice_type = meta['lattice_type']

        # Ns values: auto or user
        if args.ns_values:
            ns_values = tuple(int(x) for x in args.ns_values.split(','))
        elif lattice_name == 'honeycomb':
            ns_values = DEFAULT_NS_VALUES_HC
        else:
            ns_values = DEFAULT_NS_VALUES

        source_cdir = find_source_dir(lattice_name)
        if source_cdir is None:
            print(f"\n  SKIP {lattice_name}: no run directory found")
            continue

        print(f"\n{'*'*72}")
        print(f"  LATTICE: {lattice_name}")
        print(f"  Source: {source_cdir}")
        print(f"  Ns values: {ns_values}")
        print(f"{'*'*72}")

        all_results = {}

        for theta in angles:
            print(f"\n{'='*72}")
            print(f"  θ = {theta}° — {lattice_name}")
            print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*72}")

            theta_key = str(theta)
            angle_results = {}

            # Compute or load Phase 2
            tmp_base = None
            if args.use_existing_phase2:
                phase2_path = source_cdir / "phase2_multiband_data.h5"
                if not phase2_path.exists():
                    print(f"  ERROR: No Phase 2 data at {phase2_path}")
                    continue
                data_full = load_phase2_from_h5(phase2_path)
                print(f"  Loaded existing Phase 2: Ns={data_full['Ns1']}, "
                      f"Nb={data_full['Nb']}")
            else:
                print(f"  Computing Phase 2 at θ={theta}°...")
                t0 = time.time()
                tmp_base, work_dir = compute_phase2_at_angle(
                    theta, source_cdir, lattice_name)
                print(f"  Phase 2 done in {time.time() - t0:.1f}s")
                data_full = load_phase2_from_h5(
                    work_dir / "phase2_multiband_data.h5")

            try:
                # Compute sigma once for this (lattice, angle)
                sigma, sigma_info = compute_sigma(
                    data_full['Lambda'], data_full['M_inv'],
                    data_full['target_idx'],
                )
                print(f"  σ = {sigma:.6f} ({sigma_info['method']})")

                angle_results['sigma'] = float(sigma)
                angle_results['sigma_method'] = sigma_info['method']
                angle_results['eta'] = float(data_full['eta'])
                angle_results['Ns_full'] = data_full['Ns1']
                angle_results['Nb'] = data_full['Nb']

                # 1. N_K sweep
                if not args.skip_NK:
                    angle_results['NK_sweep'] = sweep_NK(
                        data_full, lattice_type, sigma, nk_values)
                    gc.collect()

                # 2. Ns sweep
                if not args.skip_Ns:
                    angle_results['Ns_sweep'] = sweep_Ns(
                        data_full, lattice_type, sigma, ns_values)
                    gc.collect()

                # 3. n_modes sweep
                if not args.skip_nmodes:
                    angle_results['nmodes_sweep'] = sweep_nmodes(
                        data_full, lattice_type, sigma, nmodes_values)
                    gc.collect()

                # 4. FD order sweep
                if not args.skip_fd:
                    angle_results['fd_order_sweep'] = sweep_fd_order(
                        data_full, lattice_type, sigma, fd_orders)
                    gc.collect()

            finally:
                if tmp_base:
                    shutil.rmtree(tmp_base, ignore_errors=True)
                gc.collect()

            all_results[theta_key] = angle_results

            # Save + regenerate plots after each angle
            save_results(all_results, lattice_name, OUTPUT_DIR)
            print(f"\n  Generating plots...")
            generate_plots(all_results, lattice_name, OUTPUT_DIR)

        # Final save
        save_results(all_results, lattice_name, OUTPUT_DIR)
        print(f"\n  Final plots...")
        generate_plots(all_results, lattice_name, OUTPUT_DIR)

    print(f"\n{'#'*72}")
    print(f"  ALL DONE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*72}")


if __name__ == '__main__':
    main()
