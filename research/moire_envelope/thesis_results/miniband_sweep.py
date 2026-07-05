#!/usr/bin/env python
"""
Miniband θ×K sweep — true bandwidth and flatness ratio for all lattices.

For each twist angle θ and lattice type:
  1. Run Phase 2 ONCE (K-independent data: Λ, A_berry, M_inv, v_drift, Φ_BH)
  2. Sweep Bloch wavevector K along moiré BZ high-symmetry path:
     - hex/honeycomb: Γ → K → M → Γ
     - square: Γ → X → M → Γ
  3. At each K-point, assemble H(K) with Bloch-phase FD operators and solve eigsh
  4. Extract per-band: bandwidth W_n = max_K E_n - min_K E_n, gap Δ_n, flatness Δ_n/W_n
  5. Save eigenvalues(K) and metrics to JSON, update plots

Modes:
  [initial]  12 angles per lattice, 5 K-pts/segment → 13 K-points total
  [bisect]   θ-bisection at existing K resolution (default after initial grid)
  [refine]   Double K-path density at specified angle range

Phase 1 is θ-INDEPENDENT → stored once, reused via external HDF5 links.
Phase 2 is K-INDEPENDENT → computed once per θ, reused for all K-points.

Usage:
    python thesis_results/miniband_sweep.py
    python thesis_results/miniband_sweep.py --only honeycomb
    python thesis_results/miniband_sweep.py --theta_min 0.6 --theta_max 1.0
    python thesis_results/miniband_sweep.py --refine_k --theta_min 0.8 --theta_max 1.0
    python thesis_results/miniband_sweep.py --ns_target 64   # faster (downsampled)

    Ctrl+C to stop gracefully (results saved after each angle).
"""

import sys, math, json, time, signal, gc, argparse, shutil, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase2_mpb_v3 as p2
from phase3_mpb_v3 import (
    _regularize_M_inv,
    _build_band_block_diagonal,
    build_multiband_potential_operator,
)
from common.io_utils import load_json

try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False

# Import Bloch-phase operators from the miniband module
sys.path.insert(0, str(PROJECT_ROOT / "results_bands"))
from compute_miniband_structure import (
    build_bloch_derivative_matrix,
    build_bloch_laplacian_matrix,
    build_bloch_drift_operator,
    build_bloch_kinetic_operator,
    get_moire_bz_path,
    extract_miniband_metrics,
)
from scipy.sparse.linalg import eigsh
import signal


# ─────────────────────────────────────────────────────────────────────────────
# Lattice configurations
# ─────────────────────────────────────────────────────────────────────────────

LATTICE_CONFIGS = {
    "honeycomb": {
        "name": "honeycomb_K_b1",
        "label": "Honeycomb K-point",
        "run_dir_pattern": "thesis_honeycomb_K_b1_2026*",
        "symmetry": "C6",
        "lattice_type": "honeycomb",
        "bz_type": "hex",
        "output_subdir": "T_miniband_sweep/honeycomb",
    },
    "hex": {
        "name": "hex_M_b1",
        "label": "Hexagonal M-point (C1)",
        "run_dir_pattern": "thesis_hex_M_b1_2026*",
        "symmetry": "C2",
        "lattice_type": "hex",
        "bz_type": "hex",
        "output_subdir": "T_miniband_sweep/hex",
    },
    "square": {
        "name": "square_M_b3",
        "label": "Square M-point (C3)",
        "run_dir_pattern": "thesis_square_M_b3_2026*",
        "symmetry": "C4",
        "lattice_type": "square",
        "bz_type": "square",
        "output_subdir": "T_miniband_sweep/square",
    },
}

# Global defaults
M_INV_MAX_TRACE = 20.0
N_MODES = 20


# ─────────────────────────────────────────────────────────────────────────────
# Geometry (same as bisection_sweep_multi.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moire_params(theta_deg, lattice_type, a=1.0):
    theta_rad = math.radians(theta_deg)
    eta = 2 * math.sin(theta_rad / 2)

    if lattice_type == 'square':
        B_mono = np.array([[a, 0.0], [0.0, a]])
    elif lattice_type in ('hex', 'honeycomb'):
        B_mono = np.array([[a, 0.0], [a / 2.0, a * math.sqrt(3) / 2.0]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_phase2_from_h5(h5_path):
    """Load Phase 2 HDF5 into dict (same format as compute_miniband_structure.py)."""
    with h5py.File(h5_path, 'r') as hf:
        data = {
            'Lambda':    hf['Lambda'][:],
            'A_berry':   hf['A_berry'][:],
            'Phi_BH':    hf['Phi_BH'][:],
            'v_drift':   hf['v_drift'][:],
            'M_inv':     hf['M_inv'][:],
            'omega':     hf['omega'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta':       float(hf.attrs['eta']),
            'Ns1':       int(hf.attrs['Ns1']),
            'Ns2':       int(hf.attrs['Ns2']),
            'Nb':        int(hf.attrs['N_subspace']),
            'B_moire':   hf.attrs['B_moire'][:],
            'target_idx': int(hf.attrs['target_index_in_subspace']),
        }
    return data


def downsample_field(field, factor):
    """Downsample a field on (Ns, Ns, ...) grid by integer factor via block averaging."""
    Ns_new = field.shape[0] // factor
    shape_extra = field.shape[2:]
    result = np.zeros((Ns_new, Ns_new) + shape_extra, dtype=field.dtype)
    for i in range(factor):
        for j in range(factor):
            result += field[i::factor, j::factor, ...]
    result /= factor**2
    return result


def prepare_data(data, ns_target):
    """Downsample all Phase 2 fields to target grid size."""
    ns_orig = data['Ns1']
    factor = ns_orig // ns_target
    if factor <= 1:
        return data.copy()
    out = {}
    for key in ['Lambda', 'A_berry', 'Phi_BH', 'v_drift', 'M_inv', 'omega']:
        out[key] = downsample_field(data[key], factor)
    for key in ['omega_ref', 'eta', 'Nb', 'B_moire', 'target_idx']:
        out[key] = data[key]
    out['Ns1'] = out['Ns2'] = ns_target
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bloch Hamiltonian assembly (uses corrected q_phase from compute_miniband)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_bloch_hamiltonian(data, q_vec, include_offdiag_A=True, order=4):
    """
    Assemble H(K) using Bloch-phase FD operators.

    Corrected for non-orthogonal lattices:
      q_phase_j = dot(q, a_j)  where a_j = B_moire[j]
      dR_j = |a_j| / Ns_j
    """
    Ns1 = data['Ns1']
    Ns2 = data['Ns2']
    Nb = data['Nb']
    B_moire = data['B_moire']

    L1 = np.linalg.norm(B_moire[0])
    L2 = np.linalg.norm(B_moire[1])
    dR1 = L1 / Ns1
    dR2 = L2 / Ns2

    q_phase1 = np.dot(q_vec, B_moire[0])
    q_phase2 = np.dot(q_vec, B_moire[1])

    M_inv = data.get('M_inv_reg')
    if M_inv is None:
        M_inv = _regularize_M_inv(data['M_inv'].copy(), M_INV_MAX_TRACE)

    # (1) Potential (q-independent)
    H = build_multiband_potential_operator(data['Lambda'], B_moire)

    # (2) Drift
    T_drift = build_bloch_drift_operator(
        data['v_drift'], Ns1, Ns2, Nb, dR1, dR2, q_phase1, q_phase2, order
    )
    H = H + T_drift

    # (3) Kinetic
    K_op = build_bloch_kinetic_operator(
        M_inv, data['A_berry'], Ns1, Ns2, Nb, dR1, dR2, B_moire,
        q_phase1, q_phase2, order, include_offdiag_A=include_offdiag_A
    )
    H = H + K_op

    # (4) Born-Huang
    if np.max(np.abs(data['Phi_BH'])) > 1e-15:
        U_BH = build_multiband_potential_operator(data['Phi_BH'], None)
        H = H + U_BH

    return H.tocsr()


# ─────────────────────────────────────────────────────────────────────────────
# K-sweep solver
# ─────────────────────────────────────────────────────────────────────────────

def sweep_k_points(data, q_points, n_modes, include_offdiag_A=True, label="",
                   sigma=None, candidate_type=None, order=4):
    """
    Solve H(K) at each K-point and collect eigenvalues.

    Returns:
        all_evals: (N_q, n_modes) array of eigenvalues, sorted ascending per K
    """
    N_q = len(q_points)
    all_evals = np.full((N_q, n_modes), np.nan)

    # Determine sigma via compute_sigma()
    if sigma is None:
        from phasesV3.phase3_mpb_v3 import compute_sigma as _compute_sigma
        sigma, sigma_info = _compute_sigma(
            data['Lambda'], data['M_inv'], data['target_idx'],
            candidate_type=candidate_type,
        )
        print(f"    [{label}] σ = {sigma:.6f} ({sigma_info['method']})")
    else:
        print(f"    [{label}] σ = {sigma:.6f} (user-provided)")

    print(f"    Sweeping {N_q} K-points, {n_modes} modes each")

    EIGSH_TIMEOUT = 300  # seconds per K-point

    def _eigsh_alarm(signum, frame):
        raise TimeoutError("eigsh exceeded per-K-point timeout")

    t0 = time.time()
    for iq, q in enumerate(q_points):
        t_q = time.time()
        H = assemble_bloch_hamiltonian(data, q, include_offdiag_A=include_offdiag_A, order=order)

        try:
            old_handler = signal.signal(signal.SIGALRM, _eigsh_alarm)
            signal.alarm(EIGSH_TIMEOUT)
            evals, _ = eigsh(H, k=n_modes, sigma=sigma, which='LM',
                             maxiter=5000, tol=1e-10)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            all_evals[iq] = np.sort(evals)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(f"      K[{iq}] TIMEOUT after {EIGSH_TIMEOUT}s")
            continue
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            print(f"      K[{iq}] FAILED: {e}")
            continue

        dt = time.time() - t_q
        if iq == 0 or (iq + 1) % 5 == 0 or iq == N_q - 1:
            print(f"      K[{iq:2d}] = ({q[0]:+.5f}, {q[1]:+.5f})  "
                  f"E0={all_evals[iq, 0]:.6f}  dt={dt:.1f}s")

    dt_total = time.time() - t0
    print(f"    Total K-sweep: {dt_total:.1f}s ({dt_total/N_q:.1f}s/K-point)")

    return all_evals


# ─────────────────────────────────────────────────────────────────────────────
# Single-angle miniband solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_miniband_at_angle(theta_deg, source_cdir, n_modes, lattice_type,
                            symmetry, bz_type, n_per_segment, ns_target, a=1.0):
    """
    Full pipeline for one twist angle: Phase 2 → K-sweep → metrics.

    Returns dict with eigenvalues(K), bandwidth, flatness, etc.
    """
    t0 = time.time()

    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    B_moire = moire_params['B_moire']

    phase1_src = source_cdir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"

    tmp_base = tempfile.mkdtemp(prefix=f"mb_{lattice_type}_{theta_deg:.4f}_")
    work_dir = Path(tmp_base) / "candidate_0000"
    work_dir.mkdir(parents=True)

    try:
        # ── Copy Phase 1 (external links for large datasets) ──
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

        # ── Phase 2 (K-independent, runs once per angle) ──
        print(f"    Phase 2...", flush=True)
        t_p2 = time.time()
        p2_config = {
            'include_born_huang': False,
            'include_drift_term': True,
            'use_parallel_transport_gauge': True,
            'n_extra_bands': 4,
            'mpb_fd_order': 4,
        }
        p2.process_candidate_phase2_v3(str(work_dir), p2_config)
        dt_p2 = time.time() - t_p2

        # ── Symmetrize ──
        if HAS_SYMMETRIZE and symmetry:
            try:
                symmetrize_phase2(work_dir, symmetry)
            except Exception as e:
                print(f"    WARNING: Symmetrization ({symmetry}) failed: {e}")

        gc.collect()

        # ── Load Phase 2 data ──
        phase2_h5 = work_dir / "phase2_multiband_data.h5"
        # Check for symmetrized version
        sym_suffix = f"_{symmetry.lower()}sym" if symmetry else ""
        sym_h5 = work_dir / f"phase2_multiband_data{sym_suffix}.h5"
        load_h5 = sym_h5 if sym_h5.exists() else phase2_h5

        raw_data = load_phase2_from_h5(load_h5)
        data = prepare_data(raw_data, ns_target)

        # Pre-regularize M_inv once (avoids noisy per-K-point messages)
        data['M_inv_reg'] = _regularize_M_inv(data['M_inv'].copy(), M_INV_MAX_TRACE)

        Ns = data['Ns1']
        Nb = data['Nb']
        N_total = Ns * Ns * Nb
        print(f"    Grid: {Ns}×{Ns}×{Nb} = {N_total} (Phase 2: {dt_p2:.1f}s)")

        # ── Build K-path ──
        q_points, q_dist, tick_pos, tick_labels = get_moire_bz_path(
            B_moire, n_per_segment=n_per_segment, lattice_type=bz_type
        )
        N_q = len(q_points)

        # ── K-sweep ──
        print(f"    K-sweep: {N_q} points along "
              f"{'Γ-K-M-Γ' if bz_type in ('hex','honeycomb') else 'Γ-X-M-Γ'}")
        t_ks = time.time()
        all_evals = sweep_k_points(
            data, q_points, n_modes, include_offdiag_A=True,
            label=f"{lattice_type} θ={theta_deg:.2f}°"
        )
        dt_ks = time.time() - t_ks

        # ── Extract metrics ──
        metrics = extract_miniband_metrics(all_evals, q_dist)

        wall_time = time.time() - t0

        # Per-band bandwidth and flatness
        bandwidths = [m['bandwidth'] for m in metrics]
        flatness_ratios = [m['flatness'] for m in metrics if m['flatness'] is not None]

        # Gap between first two bands
        gap_01 = metrics[0]['gap_above'] if metrics and metrics[0]['gap_above'] is not None else 0.0

        # Gamma eigenvalues (first K-point = Γ)
        gamma_evals = all_evals[0].tolist() if not np.isnan(all_evals[0, 0]) else []

        return {
            "theta_deg": theta_deg,
            "eta": moire_params['eta'],
            "L_moire": moire_params['moire_length'],
            "Ns": Ns,
            "Nb": Nb,
            "n_modes": n_modes,
            "N_q": N_q,
            "n_per_segment": n_per_segment,
            # K-path geometry
            "q_points": q_points.tolist(),
            "q_dist": q_dist.tolist(),
            "tick_positions": tick_pos,
            "tick_labels": tick_labels,
            # Eigenvalues at all K-points: shape (N_q, n_modes)
            "eigenvalues_K": all_evals.tolist(),
            # Gamma eigenvalues (for comparison with bisection sweep)
            "eigenvalues_gamma": gamma_evals,
            # Per-band metrics
            "metrics": metrics,
            # Summary scalars
            "gap_01_true": gap_01,
            "bandwidth_0": bandwidths[0] if bandwidths else 0.0,
            "flatness_0": flatness_ratios[0] if flatness_ratios else None,
            # Timing
            "wall_time_s": wall_time,
            "phase2_time_s": dt_p2,
            "ksweep_time_s": dt_ks,
        }

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Bisection scheduler
# ─────────────────────────────────────────────────────────────────────────────

def pick_next_angle(computed_angles, theta_min, theta_max):
    angles = sorted(set(computed_angles) | {theta_min, theta_max})
    if len(angles) < 2:
        return (theta_min + theta_max) / 2.0
    max_gap = 0
    best_mid = None
    for i in range(len(angles) - 1):
        gap = angles[i + 1] - angles[i]
        if gap > max_gap:
            max_gap = gap
            best_mid = (angles[i] + angles[i + 1]) / 2.0
    if max_gap < 0.001:
        return None
    return best_mid


# ─────────────────────────────────────────────────────────────────────────────
# Data management
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_results(result_file):
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def save_results(result_file, results, lattice_key, n_modes, theta_min, theta_max,
                 n_per_segment, ns_target):
    data = {
        "metadata": {
            "script": "miniband_sweep.py",
            "lattice": lattice_key,
            "n_modes": n_modes,
            "n_per_segment": n_per_segment,
            "ns_target": ns_target,
            "theta_range": [theta_min, theta_max],
            "last_updated": datetime.now().isoformat(),
            "n_angles": len(results),
        },
        "results": results,
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_miniband_summary(results, output_dir, lattice_label):
    """Plot bandwidth(θ) and flatness(θ) for one lattice type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(results) < 1:
        return

    results_sorted = sorted(results, key=lambda r: r["theta_deg"])
    thetas = [r["theta_deg"] for r in results_sorted]

    # Extract bandwidth and flatness for first 5 bands
    n_bands_show = 5
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    colors = plt.cm.viridis(np.linspace(0, 0.85, n_bands_show))

    # Panel 1: Bandwidth vs θ
    ax = axes[0]
    for n in range(n_bands_show):
        bws = []
        ts = []
        for r in results_sorted:
            m = r.get("metrics", [])
            if n < len(m):
                bws.append(m[n]["bandwidth"])
                ts.append(r["theta_deg"])
        if bws:
            ax.semilogy(ts, bws, 'o-', color=colors[n], markersize=4,
                        linewidth=1, label=f'Band {n}', alpha=0.8)
    ax.set_ylabel("Bandwidth $W_n$", fontsize=12)
    ax.set_title(f"{lattice_label} — Miniband Properties ({len(results)} angles)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Flatness ratio vs θ
    ax = axes[1]
    for n in range(n_bands_show):
        flats = []
        ts = []
        for r in results_sorted:
            m = r.get("metrics", [])
            if n < len(m) and m[n].get("flatness") is not None:
                flats.append(m[n]["flatness"])
                ts.append(r["theta_deg"])
        if flats:
            ax.semilogy(ts, flats, 'o-', color=colors[n], markersize=4,
                        linewidth=1, label=f'Band {n}', alpha=0.8)
    ax.set_xlabel("Twist angle θ (°)", fontsize=12)
    ax.set_ylabel("Flatness ratio Δ/W", fontsize=12)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3, label="Δ/W = 1")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"miniband_summary.{ext}", dpi=150)
    plt.close()


def plot_miniband_dispersion_single(result, output_dir):
    """Plot E_n(K) miniband diagram for a single angle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals_K = np.array(result["eigenvalues_K"])
    q_dist = np.array(result["q_dist"])
    tick_pos = result["tick_positions"]
    tick_labels = result["tick_labels"]
    theta = result["theta_deg"]

    if np.all(np.isnan(evals_K)):
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    n_modes = evals_K.shape[1]
    n_show = min(10, n_modes)

    for n in range(n_show):
        ax.plot(q_dist, evals_K[:, n], '-', linewidth=1.2, alpha=0.8)

    for tp in tick_pos:
        ax.axvline(tp, color='gray', linewidth=0.5, alpha=0.5)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("$E_n$ [$\\omega \\cdot a / 2\\pi c$]", fontsize=12)
    ax.set_title(f"θ = {theta:.3f}°  (first {n_show} bands)", fontsize=12)
    ax.set_xlim(q_dist[0], q_dist[-1])
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    fname = f"dispersion_{theta:.4f}".replace('.', 'p')
    for ext in ("png",):
        plt.savefig(output_dir / f"{fname}.{ext}", dpi=120)
    plt.close()


def plot_combined_summary(all_states, summary_dir):
    """Combined figure: bandwidth + flatness for all lattices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    active = {k: s for k, s in all_states.items() if len(s.results) > 0}
    if not active:
        return

    n_lattices = len(active)
    fig, axes = plt.subplots(2, n_lattices, figsize=(6 * n_lattices, 8),
                             squeeze=False, sharex=True)

    for col, (key, state) in enumerate(sorted(active.items())):
        results_sorted = sorted(state.results, key=lambda r: r["theta_deg"])

        # Top: bandwidth
        ax = axes[0, col]
        for n in range(3):
            ts, bws = [], []
            for r in results_sorted:
                m = r.get("metrics", [])
                if n < len(m):
                    ts.append(r["theta_deg"])
                    bws.append(m[n]["bandwidth"])
            if bws:
                ax.semilogy(ts, bws, 'o-', markersize=3, linewidth=1,
                            label=f'n={n}', alpha=0.8)
        ax.set_title(state.config["label"], fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Bandwidth $W_n$", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: flatness
        ax = axes[1, col]
        for n in range(3):
            ts, flats = [], []
            for r in results_sorted:
                m = r.get("metrics", [])
                if n < len(m) and m[n].get("flatness") is not None:
                    ts.append(r["theta_deg"])
                    flats.append(m[n]["flatness"])
            if flats:
                ax.semilogy(ts, flats, 'o-', markersize=3, linewidth=1,
                            label=f'n={n}', alpha=0.8)
        ax.set_xlabel("θ (°)", fontsize=11)
        if col == 0:
            ax.set_ylabel("Flatness Δ/W", fontsize=11)
        ax.axhline(y=1, color='gray', ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Moiré Miniband Sweep — True Bandwidth & Flatness",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(summary_dir / f"combined_miniband_summary.{ext}",
                    dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Lattice state manager
# ─────────────────────────────────────────────────────────────────────────────

class LatticeState:
    """Manages the miniband sweep state for one lattice configuration."""

    def __init__(self, key, config, n_modes, n_per_segment, ns_target,
                 theta_min, theta_max):
        self.key = key
        self.config = config
        self.n_modes = n_modes
        self.n_per_segment = n_per_segment
        self.ns_target = ns_target
        self.theta_min = theta_min
        self.theta_max = theta_max

        # Find run directory
        base = PROJECT_ROOT / "runsV3"
        candidates = sorted(base.glob(config["run_dir_pattern"]))
        # Filter out TE-labelled dirs (honeycomb-specific)
        candidates = [c for c in candidates if "TE" not in c.name]
        if not candidates:
            raise FileNotFoundError(
                f"No run directory for {key}: {config['run_dir_pattern']}")
        self.run_dir = candidates[-1]
        self.source_cdir = self.run_dir / "candidate_0000"

        # Read lattice info from meta
        meta = load_json(self.source_cdir / "phase0_meta.json")
        self.lattice_type = config["lattice_type"]
        self.bz_type = config["bz_type"]
        self.a = meta.get('a', 1.0)
        self.symmetry = config["symmetry"]

        # Output
        self.output_dir = SCRIPT_DIR / config["output_subdir"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.result_file = self.output_dir / "miniband_results.json"

        # Load existing results
        self.results = load_existing_results(self.result_file)
        self.total_compute_time = 0

    def save(self):
        save_results(self.result_file, self.results, self.key,
                     self.n_modes, self.theta_min, self.theta_max,
                     self.n_per_segment, self.ns_target)

    def plot(self, all_states=None):
        try:
            plot_miniband_summary(self.results, self.output_dir, self.config["label"])
        except Exception as e:
            print(f"    WARNING: Summary plot failed for {self.key}: {e}")

    def has_angle(self, theta):
        return any(abs(theta - r["theta_deg"]) < 1e-5 for r in self.results)

    def get_next_angle(self, initial_grid=None):
        """Get next angle: from initial grid first, then bisection."""
        computed = {round(r["theta_deg"], 5) for r in self.results}

        # First fill any missing initial grid points
        if initial_grid is not None:
            for theta in initial_grid:
                if round(theta, 5) not in computed:
                    return theta

        # Then bisect
        return pick_next_angle(
            [r["theta_deg"] for r in self.results],
            self.theta_min, self.theta_max
        )

    def compute_one(self, theta=None, initial_grid=None):
        """Compute miniband structure at one angle. Returns True if work done."""
        if theta is None:
            theta = self.get_next_angle(initial_grid)
        if theta is None:
            return False
        if self.has_angle(theta):
            return True

        n_pts = len(self.results)
        print(f"\n  [{self.key}] θ = {theta:.4f}° | {n_pts} existing angles")

        result = solve_miniband_at_angle(
            theta, self.source_cdir, self.n_modes,
            self.lattice_type, self.symmetry, self.bz_type,
            self.n_per_segment, self.ns_target, self.a
        )
        self.results.append(result)
        self.total_compute_time += result["wall_time_s"]

        bw0 = result["bandwidth_0"]
        f0 = result["flatness_0"]
        gap = result["gap_01_true"]
        wt = result["wall_time_s"]
        f0_str = f"{f0:.2f}" if f0 is not None else "N/A"
        print(f"    → BW₀={bw0:.2e}, Δ/W₀={f0_str}, "
              f"gap₀₁={gap:.2e}, time={wt:.1f}s")

        self.save()
        try:
            plot_miniband_dispersion_single(result, self.plots_dir)
        except Exception as e:
            print(f"    WARNING: Dispersion plot failed: {e}")
        self.plot()
        gc.collect()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

GRACEFUL_EXIT = False


def signal_handler(sig, frame):
    global GRACEFUL_EXIT
    print("\n\n  >>> Ctrl+C received. Finishing current angle, then exiting... <<<\n")
    GRACEFUL_EXIT = True


def main():
    parser = argparse.ArgumentParser(
        description="Miniband θ×K sweep — true bandwidth and flatness ratio")
    parser.add_argument("--n_modes", type=int, default=N_MODES)
    parser.add_argument("--n_per_segment", type=int, default=5,
                        help="K-points per BZ segment (default 5 → 13 total)")
    parser.add_argument("--ns_target", type=int, default=None,
                        help="Downsample grid to this size (default: full grid)")
    parser.add_argument("--theta_min", type=float, default=0.4)
    parser.add_argument("--theta_max", type=float, default=1.5)
    parser.add_argument("--only", type=str, default=None,
                        choices=list(LATTICE_CONFIGS.keys()),
                        help="Run only one lattice type")
    parser.add_argument("--n_initial", type=int, default=12,
                        help="Number of initial angles (evenly spaced)")
    parser.add_argument("--refine_k", action="store_true",
                        help="Double K-path density at existing angles in range")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Determine which lattices to run
    if args.only:
        lattice_keys = [args.only]
    else:
        lattice_keys = list(LATTICE_CONFIGS.keys())

    # Initial angle grid
    initial_grid = np.linspace(args.theta_min, args.theta_max, args.n_initial).tolist()

    print(f"{'=' * 70}")
    print(f"  MINIBAND θ×K SWEEP — True Bandwidth & Flatness")
    print(f"  Lattices: {', '.join(lattice_keys)}")
    print(f"  Modes: {args.n_modes}, K-pts/segment: {args.n_per_segment}")
    print(f"  Grid: {'full' if args.ns_target is None else f'{args.ns_target}×{args.ns_target}'}")
    print(f"  Range: [{args.theta_min}°, {args.theta_max}°], {args.n_initial} initial angles")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"{'=' * 70}")

    # Initialize lattice states
    states = {}
    for key in lattice_keys:
        try:
            states[key] = LatticeState(
                key, LATTICE_CONFIGS[key], args.n_modes, args.n_per_segment,
                args.ns_target, args.theta_min, args.theta_max
            )
            s = states[key]
            print(f"\n  {key}: {s.config['label']}")
            print(f"    Run dir:  {s.run_dir.name}")
            print(f"    Lattice:  {s.lattice_type}, BZ: {s.bz_type}, a={s.a}")
            print(f"    Symmetry: {s.symmetry}")
            print(f"    Existing: {len(s.results)} angles")
        except FileNotFoundError as e:
            print(f"\n  WARNING: Skipping {key}: {e}")

    if not states:
        print("ERROR: No lattice configurations available!")
        sys.exit(1)

    # Summary directory for combined plots
    summary_dir = SCRIPT_DIR / "T_miniband_sweep" / "summary_plots"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # ── K-refine mode ──
    if args.refine_k:
        print(f"\n{'─' * 50}")
        print(f"  K-PATH REFINEMENT MODE  (doubling K density)")
        print(f"  Angle range: [{args.theta_min}°, {args.theta_max}°]")
        print(f"{'─' * 50}")

        new_n_per_segment = args.n_per_segment * 2
        for key, state in states.items():
            angles_to_refine = [
                r["theta_deg"] for r in state.results
                if args.theta_min <= r["theta_deg"] <= args.theta_max
                and r.get("n_per_segment", 0) < new_n_per_segment
            ]
            if not angles_to_refine:
                print(f"  {key}: No angles need refinement in range")
                continue

            print(f"  {key}: Refining {len(angles_to_refine)} angles "
                  f"({args.n_per_segment} → {new_n_per_segment} pts/segment)")

            # Remove old results for these angles then recompute
            state.results = [
                r for r in state.results
                if not (args.theta_min <= r["theta_deg"] <= args.theta_max
                        and r.get("n_per_segment", 0) < new_n_per_segment)
            ]
            state.n_per_segment = new_n_per_segment

            for theta in sorted(angles_to_refine):
                if GRACEFUL_EXIT:
                    break
                state.compute_one(theta=theta)

            state.save()
            state.plot()

        plot_combined_summary(states, summary_dir)
        return

    # ── Normal sweep mode ──
    print(f"\n{'=' * 70}")
    print(f"  Starting alternating sweep (initial grid + bisection)...")
    print(f"{'=' * 70}")

    # Generate initial plots
    for s in states.values():
        s.plot()

    # Main alternation loop
    iteration = 0
    active_keys = list(states.keys())

    while not GRACEFUL_EXIT and active_keys:
        iteration += 1

        # Round-robin through active lattices
        key = active_keys[(iteration - 1) % len(active_keys)]
        state = states[key]

        print(f"\n{'─' * 50}")
        print(f"  Iteration {iteration} — {state.config['label']}")
        print(f"{'─' * 50}")

        try:
            ok = state.compute_one(initial_grid=initial_grid)
            if not ok:
                print(f"  {key}: Resolution limit reached. Removing from rotation.")
                active_keys.remove(key)
        except Exception as e:
            print(f"  ERROR in {key} at current angle: {e}")
            import traceback
            traceback.print_exc()

        # Update combined plot periodically
        if iteration % len(states) == 0:
            try:
                plot_combined_summary(states, summary_dir)
            except Exception as e:
                print(f"  WARNING: Combined plot failed: {e}")

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE")
    print(f"{'=' * 70}")
    for key, state in states.items():
        print(f"  {key}: {len(state.results)} angles, "
              f"total compute: {state.total_compute_time:.0f}s")
    plot_combined_summary(states, summary_dir)


if __name__ == "__main__":
    main()
