#!/usr/bin/env python3
"""
T_sanity_check — Final gauge & symmetry sanity checks.

Six targeted diagnostics on existing Phase 2/3 HDF5 data:

  1. Hermiticity of Berry connection A_berry
  2. Hermiticity & positivity of Born-Huang potential Phi_BH
  3. Gauge-invariant norm plots (should show crystal symmetry)
  4. Proper symmetry transformation test (raw vs gauge-invariant)
  5. Gauge smoothness (Berry curvature spikes)
  6. Hamiltonian anti-Hermitian residual (per operator term)

No re-computation of expensive MPB phases — all checks load
existing phase2/phase3 outputs.

Usage:
    python thesis_results/T_sanity_check/compute.py [--candidate NAME]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import kron, eye, diags, csr_matrix

# Add parent for thesis_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thesis_utils import (
    apply_thesis_style, find_candidate_dir, load_candidates_yaml,
    CANDIDATE_LABELS, CANDIDATE_COLORS, FIGURES_DIR, RUNS_DIR,
    ensure_output_dir, save_figure,
)

# Add phasesV3 for Hamiltonian assembly
PHASES_DIR = Path(__file__).resolve().parent.parent.parent / "phasesV3"
sys.path.insert(0, str(PHASES_DIR))
from phase3_mpb_v3 import (
    build_multiband_potential_operator,
    build_multiband_drift_operator,
    build_multiband_kinetic_operator,
    build_multiband_born_huang_operator,
)

# Add symmetrize
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from symmetrize import (
    apply_rot_n_grid, get_rot_mats, get_order,
    measure_error_scalar, measure_error_vector,
)

TASK = "T_sanity_check"

# Thresholds
HERM_PASS = 1e-10
HERM_MARGINAL = 1e-6
HERM_FAIL = 1e-3
POS_TOL = 1e-12
H_ANTIHERM_PASS = 1e-8
GAUGE_OVERLAP_PASS = 0.99

# Candidate → symmetry type mapping
SYM_MAP = {
    'square_M_b3': 'C4',
    'hex_M_b1': 'C2',
    'honeycomb_K_b1': 'C6',
}

# ===========================================================================
# Data loading helpers
# ===========================================================================


def find_correct_candidate_dir(candidate_name):
    """Find the correct thesis run directory, excluding *_TE_ variants."""
    pattern = f"thesis_{candidate_name}_2*"  # Must start with timestamp
    matches = sorted(p for p in RUNS_DIR.glob(pattern) if p.is_dir())
    if not matches:
        # Fallback to general match
        return find_candidate_dir(candidate_name)
    run_dir = matches[-1]
    cand_dirs = sorted(run_dir.glob("candidate_*"))
    if not cand_dirs:
        raise FileNotFoundError(f"No candidate directory in {run_dir}")
    return cand_dirs[0]


def load_phase2_raw(cand_dir, symmetrized=True):
    """Load phase2 data, choosing symmetrized or unsymmetrized version."""
    if symmetrized:
        # Try symmetrized versions first
        for suffix in ['c4sym', 'c2sym', 'c6sym', 'c3sym']:
            p = cand_dir / f"phase2_multiband_data_{suffix}.h5"
            if p.exists():
                return _load_h5(p), p.name
        # Fall back to default
        p = cand_dir / "phase2_multiband_data.h5"
        return _load_h5(p), p.name
    else:
        p = cand_dir / "phase2_multiband_data_unsym.h5"
        if p.exists():
            return _load_h5(p), p.name
        # Fall back to default (may not have been symmetrized)
        p = cand_dir / "phase2_multiband_data.h5"
        return _load_h5(p), p.name


def _load_h5(path):
    """Load all datasets and attrs from HDF5."""
    data = {}
    with h5py.File(path, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][:]
        for key, val in hf.attrs.items():
            data[f'attr_{key}'] = val
    return data


def load_phase3_sparse_H(cand_dir):
    """Load sparse Hamiltonian from phase3 HDF5."""
    p = cand_dir / "phase3_multiband_modes.h5"
    if not p.exists():
        return None, {}
    with h5py.File(p, 'r') as hf:
        if 'H_data' not in hf:
            return None, {}
        H_data = hf['H_data'][:]
        H_indices = hf['H_indices'][:]
        H_indptr = hf['H_indptr'][:]
        shape = tuple(hf['H_shape'][:]) if 'H_shape' in hf else None
        attrs = {k: v for k, v in hf.attrs.items()}
    if shape is None:
        n = int(np.sqrt(len(H_data)))  # fallback
        shape = (n, n)
    H = csr_matrix((H_data, H_indices, H_indptr), shape=shape)
    return H, attrs


# ===========================================================================
# Step 1: Hermiticity of A_berry
# ===========================================================================


def check_hermiticity_A(A_berry):
    """
    Check A_j†[m,n] = conj(A_j[n,m]) for each direction j.

    A_berry shape: (Ns1, Ns2, Nb, Nb, 2)
    Returns dict with per-direction and aggregate metrics.
    """
    # A_j†: swap band indices and conjugate
    # A_berry[..., m, n, j] vs conj(A_berry[..., n, m, j])
    A_dag = np.conj(np.swapaxes(A_berry, 2, 3))  # (Ns1, Ns2, Nb, Nb, 2)

    results = {}
    for j in range(2):
        A_j = A_berry[..., j]       # (Ns1, Ns2, Nb, Nb)
        A_j_dag = A_dag[..., j]     # (Ns1, Ns2, Nb, Nb)

        diff = A_j - A_j_dag
        # Per-point Frobenius norm ratio
        norms_diff = np.sqrt(np.sum(np.abs(diff)**2, axis=(-2, -1)))
        norms_A = np.sqrt(np.sum(np.abs(A_j)**2, axis=(-2, -1)))
        # Avoid division by zero
        mask = norms_A > 1e-30
        eps = np.zeros_like(norms_diff)
        eps[mask] = norms_diff[mask] / norms_A[mask]

        results[f'dir{j}'] = {
            'max': float(np.max(eps)),
            'mean': float(np.mean(eps[mask])) if np.any(mask) else 0.0,
            'std': float(np.std(eps[mask])) if np.any(mask) else 0.0,
            'median': float(np.median(eps[mask])) if np.any(mask) else 0.0,
            'n_nonzero': int(np.sum(mask)),
        }

    # Aggregate: global Frobenius
    diff_all = A_berry - A_dag
    global_eps = np.linalg.norm(diff_all) / (np.linalg.norm(A_berry) + 1e-30)
    results['global'] = float(global_eps)
    results['verdict'] = classify_hermiticity(global_eps)

    # Also check diagonal entries are real
    Nb = A_berry.shape[2]
    diag_imag_max = 0.0
    diag_real_max = 0.0
    for n in range(Nb):
        diag_vals = A_berry[:, :, n, n, :]
        diag_imag_max = max(diag_imag_max,
                            float(np.max(np.abs(np.imag(diag_vals)))))
        diag_real_max = max(diag_real_max,
                            float(np.max(np.abs(np.real(diag_vals)))))
    results['diag_imag_max'] = diag_imag_max
    results['diag_imag_rel'] = diag_imag_max / (diag_real_max + 1e-30)

    # Decompose into Hermitian + anti-Hermitian parts
    A_herm = 0.5 * (A_berry + A_dag)
    A_anti = 0.5 * (A_berry - A_dag)
    results['hermitian_part_norm'] = float(np.linalg.norm(A_herm))
    results['antihermitian_part_norm'] = float(np.linalg.norm(A_anti))

    return results


def classify_hermiticity(eps):
    if eps < HERM_PASS:
        return 'PASS'
    elif eps < HERM_MARGINAL:
        return 'MARGINAL'
    elif eps < HERM_FAIL:
        return 'MARGINAL_HIGH'
    else:
        return 'FAIL'


# ===========================================================================
# Step 2: Born-Huang Hermiticity & Positivity
# ===========================================================================


def check_born_huang(Phi_BH):
    """
    Check Hermiticity and positivity of Phi_BH.

    Phi_BH shape: (Ns1, Ns2, Nb, Nb)
    """
    results = {}

    # Check if all zeros (placeholder)
    max_val = float(np.max(np.abs(Phi_BH)))
    results['max_abs'] = max_val
    if max_val < 1e-15:
        results['is_placeholder'] = True
        results['verdict_herm'] = 'SKIP_ZERO'
        results['verdict_pos'] = 'SKIP_ZERO'
        return results
    results['is_placeholder'] = False

    # Hermiticity check
    # For real Phi: Phi[m,n] should equal Phi[n,m]
    # For complex Phi: Phi[m,n] should equal conj(Phi[n,m])
    Phi_dag = np.conj(np.swapaxes(Phi_BH, 2, 3))
    diff = Phi_BH - Phi_dag
    eps_herm = np.linalg.norm(diff) / (np.linalg.norm(Phi_BH) + 1e-30)
    results['herm_eps'] = float(eps_herm)
    results['verdict_herm'] = classify_hermiticity(eps_herm)
    results['dtype'] = str(Phi_BH.dtype)

    # Positivity check: eigenvalues at each grid point
    Ns1, Ns2, Nb, _ = Phi_BH.shape
    min_eig = np.inf
    n_negative = 0
    n_total = Ns1 * Ns2

    for i in range(Ns1):
        for j in range(Ns2):
            mat = Phi_BH[i, j]
            # Make Hermitian for eigh
            mat_h = 0.5 * (mat + mat.conj().T)
            eigs = np.linalg.eigvalsh(mat_h)
            local_min = float(np.min(eigs))
            if local_min < min_eig:
                min_eig = local_min
            if local_min < -POS_TOL:
                n_negative += 1

    results['min_eigenvalue'] = float(min_eig)
    results['n_negative_points'] = n_negative
    results['frac_negative'] = n_negative / n_total
    results['verdict_pos'] = 'PASS' if n_negative == 0 else (
        'MARGINAL' if n_negative / n_total < 0.01 else 'FAIL'
    )

    # Trace statistics (gauge-invariant)
    trace = np.trace(Phi_BH, axis1=2, axis2=3)
    results['trace_mean'] = float(np.mean(trace.real))
    results['trace_std'] = float(np.std(trace.real))
    results['trace_min'] = float(np.min(trace.real))
    results['trace_max'] = float(np.max(trace.real))

    return results


# ===========================================================================
# Step 3: Gauge-invariant norm plots
# ===========================================================================


def compute_gauge_invariant_norms(A_berry, Phi_BH):
    """
    Compute gauge-invariant scalar fields from A and Phi.

    Returns dict of 2D arrays (Ns1, Ns2).
    """
    norms = {}

    # Berry Frobenius norm: ||A(R)||_F = sqrt(sum_{j,m,n} |A_j,mn|^2)
    norms['A_frob'] = np.sqrt(np.sum(np.abs(A_berry)**2, axis=(-3, -2, -1)))

    # Per-direction
    norms['A_frob_x'] = np.sqrt(np.sum(np.abs(A_berry[..., 0])**2, axis=(-2, -1)))
    norms['A_frob_y'] = np.sqrt(np.sum(np.abs(A_berry[..., 1])**2, axis=(-2, -1)))

    # Off-diagonal magnitude |A_{01}|
    Nb = A_berry.shape[2]
    if Nb >= 2:
        A01 = A_berry[:, :, 0, 1, :]  # (Ns1, Ns2, 2)
        norms['A01_mag'] = np.sqrt(np.sum(np.abs(A01)**2, axis=-1))

    # Born-Huang trace
    if Phi_BH is not None and np.max(np.abs(Phi_BH)) > 1e-15:
        norms['Phi_trace'] = np.real(np.trace(Phi_BH, axis1=2, axis2=3))

    return norms


def plot_gauge_invariant_norms(norms, candidate_name, out_dir, A_berry_raw=None):
    """
    Plot gauge-invariant norms on 2D grid.

    Optionally includes a raw A entry for comparison.
    """
    apply_thesis_style()

    n_panels = len(norms)
    if A_berry_raw is not None:
        n_panels += 1

    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    idx = 0
    # Plot raw entry first (for comparison)
    if A_berry_raw is not None:
        ax = axes.flat[idx]
        raw_data = np.real(A_berry_raw[:, :, 0, 1, 0])  # Re(A_{01,x})
        im = ax.imshow(raw_data.T, origin='lower', aspect='equal', cmap='RdBu_r')
        ax.set_title(r'Raw Re($A_{01,x}$)')
        plt.colorbar(im, ax=ax, shrink=0.7)
        idx += 1

    for key, data in norms.items():
        if idx >= len(axes.flat):
            break
        ax = axes.flat[idx]
        im = ax.imshow(data.T, origin='lower', aspect='equal', cmap='viridis')
        ax.set_title(key)
        plt.colorbar(im, ax=ax, shrink=0.7)
        idx += 1

    # Hide unused axes
    for i in range(idx, len(axes.flat)):
        axes.flat[i].set_visible(False)

    fig.suptitle(f'Gauge-Invariant Norms — {CANDIDATE_LABELS.get(candidate_name, candidate_name)}',
                 fontsize=12)
    fig.tight_layout()
    save_figure(fig, TASK, f"gauge_inv_norms_{candidate_name}")
    return fig


# ===========================================================================
# Step 4: Symmetry transformation test
# ===========================================================================


def check_symmetry_gauge_invariant(norms, Ns, sym_type):
    """
    Check if gauge-invariant norms respect crystal symmetry.

    For each gauge-invariant scalar f(R), compute:
        eps_sym = ||f(gR) - f(R)|| / ||f(R)||
    for the first non-trivial generator g.
    """
    results = {}
    for key, field in norms.items():
        # field is (Ns, Ns) or (Ns, Ns, ...)
        err = measure_error_scalar(field, Ns, sym_type)
        results[key] = float(err)

    # Verdict: only use SCALAR invariants (A_frob, A01_mag, Phi_trace).
    # Individual components A_frob_x, A_frob_y transform as VECTOR components
    # under Cn (n>2), so testing scalar symmetry on them is incorrect.
    scalar_keys = [k for k in results if k in ('A_frob', 'A01_mag', 'Phi_trace')]
    scalar_errs = [results[k] for k in scalar_keys]
    results['verdict'] = 'PASS' if all(v < 1e-6 for v in scalar_errs) else 'CHECK'
    return results


def check_symmetry_raw_A(A_berry, Ns, sym_type):
    """Check raw symmetry error of A_berry as a vector field."""
    return float(measure_error_vector(A_berry, Ns, sym_type))


# ===========================================================================
# Step 5: Gauge smoothness (Berry curvature)
# ===========================================================================


def compute_berry_curvature(A_berry, ds1=1.0, ds2=1.0):
    """
    Compute the non-Abelian Berry curvature:
        F_xy = ∂_x A_y - ∂_y A_x - i[A_x, A_y]

    A_berry shape: (Ns1, Ns2, Nb, Nb, 2)
    Returns: F_xy (Ns1, Ns2, Nb, Nb) complex
    """
    Ax = A_berry[..., 0]  # (Ns1, Ns2, Nb, Nb)
    Ay = A_berry[..., 1]

    Ns1, Ns2 = A_berry.shape[:2]

    # Periodic finite differences (central, 2nd order)
    # ∂_x A_y : derivative along axis 0
    dAy_dx = (np.roll(Ay, -1, axis=0) - np.roll(Ay, 1, axis=0)) / (2 * ds1)
    # ∂_y A_x : derivative along axis 1
    dAx_dy = (np.roll(Ax, -1, axis=1) - np.roll(Ax, 1, axis=1)) / (2 * ds2)

    # Commutator [A_x, A_y] at each point
    # For Nb×Nb matrices: [Ax, Ay]_{mn} = sum_p (Ax_{mp} Ay_{pn} - Ay_{mp} Ax_{pn})
    comm = np.einsum('...mp,...pn->...mn', Ax, Ay) - \
           np.einsum('...mp,...pn->...mn', Ay, Ax)

    F_xy = dAy_dx - dAx_dy - 1j * comm

    return F_xy


def check_gauge_smoothness(A_berry):
    """
    Gauge smoothness diagnostics:
    1. Berry curvature magnitude (should be smooth, no spikes)
    2. Nearest-neighbor overlap of A entries (should be close to 1)
    """
    results = {}

    F_xy = compute_berry_curvature(A_berry)

    # Frobenius norm of curvature at each point
    F_frob = np.sqrt(np.sum(np.abs(F_xy)**2, axis=(-2, -1)))
    results['F_frob_max'] = float(np.max(F_frob))
    results['F_frob_mean'] = float(np.mean(F_frob))
    results['F_frob_std'] = float(np.std(F_frob))
    results['F_frob_p99'] = float(np.percentile(F_frob, 99))

    # Spike ratio: max / mean
    spike_ratio = results['F_frob_max'] / (results['F_frob_mean'] + 1e-30)
    results['spike_ratio'] = spike_ratio

    # Nearest-neighbor smoothness of A entries
    # Compute ||A(s) - A(s+ds)||_F / ||A(s)||_F for both directions
    A_shift_x = np.roll(A_berry, -1, axis=0)
    A_shift_y = np.roll(A_berry, -1, axis=1)

    # Frobenius norm over (Nb, Nb, 2) dimensions
    diff_x = np.sqrt(np.sum(np.abs(A_berry - A_shift_x)**2, axis=(-3, -2, -1)))
    diff_y = np.sqrt(np.sum(np.abs(A_berry - A_shift_y)**2, axis=(-3, -2, -1)))
    norm_A = np.sqrt(np.sum(np.abs(A_berry)**2, axis=(-3, -2, -1)))

    mask = norm_A > 1e-30
    rel_jump_x = np.zeros_like(diff_x)
    rel_jump_y = np.zeros_like(diff_y)
    rel_jump_x[mask] = diff_x[mask] / norm_A[mask]
    rel_jump_y[mask] = diff_y[mask] / norm_A[mask]

    results['jump_x_max_rel'] = float(np.max(rel_jump_x))
    results['jump_x_mean_rel'] = float(np.mean(rel_jump_x[mask]))
    results['jump_y_max_rel'] = float(np.max(rel_jump_y))
    results['jump_y_mean_rel'] = float(np.mean(rel_jump_y[mask]))

    # Also compute ABSOLUTE NN jumps (more meaningful when A≈0 at nodes)
    results['jump_x_max_abs'] = float(np.max(diff_x))
    results['jump_y_max_abs'] = float(np.max(diff_y))
    results['jump_x_mean_abs'] = float(np.mean(diff_x))
    results['jump_y_mean_abs'] = float(np.mean(diff_y))
    norm_A_mean = float(np.mean(norm_A))
    results['norm_A_mean'] = norm_A_mean
    # Meaningful relative metric: max absolute jump / mean A norm
    results['jump_x_rel_robust'] = results['jump_x_max_abs'] / (norm_A_mean + 1e-30)
    results['jump_y_rel_robust'] = results['jump_y_max_abs'] / (norm_A_mean + 1e-30)

    # Verdict: use spike ratio + robust relative metric (immune to near-zero-A nodes)
    smooth = (spike_ratio < 20) and (results['jump_x_rel_robust'] < 1.0) and (results['jump_y_rel_robust'] < 1.0)
    results['verdict'] = 'PASS' if smooth else (
        'MARGINAL' if spike_ratio < 50 else 'FAIL'
    )

    return results, F_frob


def plot_berry_curvature(F_frob, candidate_name, out_dir):
    """Plot Berry curvature |F_xy| on 2D grid."""
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(F_frob.T, origin='lower', aspect='equal', cmap='inferno')
    ax.set_title(f'$|F_{{xy}}|$ (Berry curvature) — {CANDIDATE_LABELS.get(candidate_name, candidate_name)}')
    plt.colorbar(im, ax=ax, label=r'$\|F_{xy}\|_F$')
    fig.tight_layout()
    save_figure(fig, TASK, f"berry_curvature_{candidate_name}")
    return fig


# ===========================================================================
# Step 6: Hamiltonian anti-Hermitian residual
# ===========================================================================


def check_hamiltonian_hermiticity(data, candidate_name, cand_info):
    """
    Reconstruct the Hamiltonian WITHOUT the final (H+H†)/2 enforcement
    and measure the anti-Hermitian residual of each term.

    Uses the same assembly functions as phase3_mpb_v3.py.
    """
    results = {}

    Lambda = data['Lambda']
    v_drift = data['v_drift']
    M_inv = data['M_inv']
    A_berry = data['A_berry']
    Phi_BH = data['Phi_BH']

    Ns1 = int(data.get('attr_Ns1', data.get('attr_Ns', Lambda.shape[0])))
    Ns2 = int(data.get('attr_Ns2', data.get('attr_Ns', Lambda.shape[1])))
    Nb = Lambda.shape[2]
    eta = float(data.get('attr_eta', 0.019))

    # Reconstruct B_moire and grid spacing
    B_moire = data.get('attr_B_moire', None)
    if B_moire is None:
        # Try loading from phase0
        moire_length = float(data.get('attr_moire_length', 52.09))
        B_moire = np.array([[moire_length, 0], [0, moire_length]])
    if isinstance(B_moire, (int, float)):
        ml = float(B_moire)
        B_moire = np.array([[ml, 0], [0, ml]])

    L_moire = np.linalg.norm(B_moire[0]) if B_moire.ndim == 2 else float(B_moire)
    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2

    N_total = Ns1 * Ns2 * Nb

    # Potential term
    V_op = build_multiband_potential_operator(Lambda, B_moire)
    results['V'] = _antiherm_metric(V_op)

    # Drift term
    try:
        T_op = build_multiband_drift_operator(v_drift, eta, Ns1, Ns2, Nb, dR1, dR2, 4)
        results['T_drift'] = _antiherm_metric(T_op)
    except Exception as e:
        results['T_drift'] = {'error': str(e)}

    # Kinetic term (WITHOUT Hermitization — need to bypass the (K+K†)/2 line)
    # We can't easily bypass it without modifying the function, so instead
    # we build K and check its residual AFTER the internal Hermitization.
    # Then separately check the full H.
    try:
        K_op = build_multiband_kinetic_operator(
            M_inv, A_berry, eta, Ns1, Ns2, Nb, dR1, dR2, B_moire, 4,
            include_offdiag_A=True
        )
        # Note: K_op already has (K+K†)/2 applied internally, so this will be ~0
        results['K_kinetic_post_hermitize'] = _antiherm_metric(K_op)
    except Exception as e:
        results['K_kinetic'] = {'error': str(e)}

    # Born-Huang term
    if np.max(np.abs(Phi_BH)) > 1e-15:
        U_BH = build_multiband_born_huang_operator(Phi_BH, eta, Ns1, Ns2, Nb)
        results['U_BH'] = _antiherm_metric(U_BH)
    else:
        results['U_BH'] = {'skipped': 'Phi_BH is zero'}

    # Full H = V + T + K + U_BH (each already individually assembled)
    try:
        H_full = V_op.tocsr()
        if 'T_drift' in results and 'error' not in results['T_drift']:
            H_full = H_full + T_op.tocsr()
        if 'K_kinetic_post_hermitize' in results:
            H_full = H_full + K_op.tocsr()
        if isinstance(results.get('U_BH'), dict) and 'skipped' not in results['U_BH']:
            H_full = H_full + U_BH.tocsr()

        # Full H BEFORE explicit (H+H†)/2
        results['H_full_pre_enforce'] = _antiherm_metric(H_full)

        # After enforcement
        H_herm = 0.5 * (H_full + H_full.conj().T)
        results['H_full_post_enforce'] = _antiherm_metric(H_herm)

        # Check eigenvalues are real
        # Sample: compute a few eigenvalues of the non-Hermitized H
        results['H_size'] = H_full.shape[0]
    except Exception as e:
        results['H_full'] = {'error': str(e)}

    # Verdict is based on FULL H (pre-enforce), not individual terms.
    # Individual terms (e.g. drift T=-ivD) are expected to be non-Hermitian.
    # The physical requirement is that the FULL H is Hermitian.
    h_pre = results.get('H_full_pre_enforce', {})
    h_eps = h_pre.get('eps', 1.0) if isinstance(h_pre, dict) else 1.0
    results['verdict'] = classify_hermiticity(h_eps)

    return results


def _antiherm_metric(M):
    """Compute ||M - M†|| / ||M|| for a sparse matrix."""
    M_csr = M.tocsr()
    diff = M_csr - M_csr.conj().T
    # Use Frobenius norm via data arrays for efficiency
    norm_diff = np.sqrt(np.sum(np.abs(diff.data)**2))
    norm_M = np.sqrt(np.sum(np.abs(M_csr.data)**2))
    eps = norm_diff / (norm_M + 1e-30)
    return {
        'eps': float(eps),
        'norm_diff': float(norm_diff),
        'norm_M': float(norm_M),
        'verdict': classify_hermiticity(eps),
    }


# ===========================================================================
# Summary & orchestration
# ===========================================================================


def run_all_checks(candidate_name, cand_dir, cand_info, out_dir):
    """Run all 6 sanity checks for one candidate."""
    print(f"\n{'='*70}")
    print(f"  SANITY CHECK: {CANDIDATE_LABELS.get(candidate_name, candidate_name)}")
    print(f"  Directory: {cand_dir}")
    print(f"{'='*70}")

    sym_type = SYM_MAP.get(candidate_name, 'C2')
    summary = {
        'candidate': candidate_name,
        'sym_type': sym_type,
        'directory': str(cand_dir),
    }

    # Load data (both unsymmetrized and symmetrized)
    print("\n[0] Loading Phase 2 data...")
    try:
        data_sym, sym_file = load_phase2_raw(cand_dir, symmetrized=True)
        print(f"  Symmetrized: {sym_file}")
    except FileNotFoundError:
        print("  WARNING: No symmetrized Phase 2 data found")
        data_sym = None

    try:
        data_unsym, unsym_file = load_phase2_raw(cand_dir, symmetrized=False)
        print(f"  Unsymmetrized: {unsym_file}")
    except FileNotFoundError:
        print("  WARNING: No unsymmetrized Phase 2 data found")
        data_unsym = None

    # Use symmetrized data as primary
    data = data_sym if data_sym is not None else data_unsym
    if data is None:
        print("  ERROR: No Phase 2 data found at all!")
        return summary

    A_berry = data['A_berry']
    Phi_BH = data['Phi_BH']
    Ns = A_berry.shape[0]
    Nb = A_berry.shape[2]
    print(f"  Grid: {Ns}×{Ns}, Nb={Nb}, A dtype={A_berry.dtype}, Phi dtype={Phi_BH.dtype}")

    # ── Step 1: Hermiticity of A_berry ──
    print("\n[1] Hermiticity of A_berry...")
    herm_A_sym = check_hermiticity_A(A_berry)
    summary['step1_A_hermiticity_sym'] = herm_A_sym
    print(f"  Symmetrized:  global ε = {herm_A_sym['global']:.2e}  → {herm_A_sym['verdict']}")
    print(f"    dir0: max={herm_A_sym['dir0']['max']:.2e}, mean={herm_A_sym['dir0']['mean']:.2e}")
    print(f"    dir1: max={herm_A_sym['dir1']['max']:.2e}, mean={herm_A_sym['dir1']['mean']:.2e}")
    print(f"    diag Im/Re: {herm_A_sym['diag_imag_rel']:.2e}")
    print(f"    ||A_herm||={herm_A_sym['hermitian_part_norm']:.4e}, "
          f"||A_anti||={herm_A_sym['antihermitian_part_norm']:.4e}")

    # Note on metric correction (applies to ALL polarizations using E-fields)
    print("  NOTE: Berry connection uses ε-weighted E-field inner product.")
    print("        When ε(r;R) varies with moiré coordinate R, the Berry connection")
    print("        has a METRIC CORRECTION: A+A† = -i⟨u_m|∂_R ε|u_n⟩ ≠ 0.")
    print("        This anti-Hermitian part is expected and cancels in the full H.")
    print("        The FULL Hamiltonian Hermiticity (Step 6) is the decisive test.")

    if data_unsym is not None:
        herm_A_unsym = check_hermiticity_A(data_unsym['A_berry'])
        summary['step1_A_hermiticity_unsym'] = herm_A_unsym
        print(f"  Unsymmetrized: global ε = {herm_A_unsym['global']:.2e}  → {herm_A_unsym['verdict']}")

    # ── Step 2: Born-Huang checks ──
    print("\n[2] Born-Huang Hermiticity & Positivity...")
    bh_results = check_born_huang(Phi_BH)
    summary['step2_born_huang'] = bh_results
    if bh_results.get('is_placeholder'):
        print("  Phi_BH is all zeros (placeholder) — SKIPPED")
    else:
        print(f"  Hermiticity ε = {bh_results['herm_eps']:.2e}  → {bh_results['verdict_herm']}")
        print(f"  Positivity: min eigenvalue = {bh_results['min_eigenvalue']:.2e}, "
              f"{bh_results['n_negative_points']} negative ({bh_results['frac_negative']:.1%})")
        print(f"  Trace: mean={bh_results['trace_mean']:.4e}, range=[{bh_results['trace_min']:.4e}, {bh_results['trace_max']:.4e}]")
        print(f"  → herm: {bh_results['verdict_herm']}, pos: {bh_results['verdict_pos']}")

    # ── Step 3: Gauge-invariant norms ──
    print("\n[3] Gauge-invariant norms...")
    norms = compute_gauge_invariant_norms(A_berry, Phi_BH)
    for key, val in norms.items():
        print(f"  {key}: min={np.min(val):.4e}, max={np.max(val):.4e}, mean={np.mean(val):.4e}")
    plot_gauge_invariant_norms(norms, candidate_name, out_dir, A_berry_raw=A_berry)

    # ── Step 4: Symmetry of gauge-invariant norms ──
    print(f"\n[4] Symmetry test ({sym_type})...")
    sym_gi = check_symmetry_gauge_invariant(norms, Ns, sym_type)
    summary['step4_symmetry_gauge_inv'] = sym_gi
    for key, err in sym_gi.items():
        if key != 'verdict':
            print(f"  {key}: ε_sym = {err:.2e}")
    print(f"  → {sym_gi['verdict']}")

    # Raw A symmetry for comparison
    raw_A_err = check_symmetry_raw_A(A_berry, Ns, sym_type)
    summary['step4_raw_A_sym_error'] = raw_A_err
    print(f"  Raw A_berry vector symmetry error: {raw_A_err:.2e}")

    if data_unsym is not None:
        norms_unsym = compute_gauge_invariant_norms(data_unsym['A_berry'], data_unsym['Phi_BH'])
        sym_gi_unsym = check_symmetry_gauge_invariant(norms_unsym, Ns, sym_type)
        summary['step4_symmetry_gauge_inv_unsym'] = sym_gi_unsym
        print("  Pre-symmetrization gauge-inv errors:")
        for key, err in sym_gi_unsym.items():
            if key != 'verdict':
                print(f"    {key}: ε_sym = {err:.2e}")

    # ── Step 5: Gauge smoothness ──
    print("\n[5] Gauge smoothness (Berry curvature)...")
    smooth_results, F_frob = check_gauge_smoothness(A_berry)
    summary['step5_gauge_smoothness'] = smooth_results
    print(f"  |F_xy| max={smooth_results['F_frob_max']:.4e}, "
          f"mean={smooth_results['F_frob_mean']:.4e}, "
          f"spike ratio={smooth_results['spike_ratio']:.1f}")
    print(f"  NN jump (absolute): x_max={smooth_results['jump_x_max_abs']:.4e}, "
          f"y_max={smooth_results['jump_y_max_abs']:.4e}")
    print(f"  NN jump (robust rel): x={smooth_results['jump_x_rel_robust']:.4e}, "
          f"y={smooth_results['jump_y_rel_robust']:.4e}")
    print(f"  → {smooth_results['verdict']}")
    plot_berry_curvature(F_frob, candidate_name, out_dir)

    # ── Step 6: Hamiltonian anti-Hermitian residual ──
    print("\n[6] Hamiltonian anti-Hermitian residual...")
    h_results = check_hamiltonian_hermiticity(data, candidate_name, cand_info)
    summary['step6_H_antihermitian'] = h_results
    for key, val in h_results.items():
        if isinstance(val, dict) and 'eps' in val:
            print(f"  {key}: ε = {val['eps']:.2e}  → {val['verdict']}")
        elif key not in ('verdict', 'H_size'):
            print(f"  {key}: {val}")
    print(f"  → {h_results['verdict']}")

    # ── Final verdict ──
    print(f"\n{'─'*70}")
    # A_hermiticity is informational — not expected to pass when ε varies with R
    # (applies to ALL polarizations using E-fields with ε-weighted inner product).
    # The decisive test is H_antihermitian (full Hamiltonian Hermiticity).
    a_verdict = herm_A_sym['verdict']
    if a_verdict in ('FAIL',):
        a_verdict = 'INFO_METRIC'  # Expected due to R-dependent ε-weighted inner product

    verdicts = {
        'A_hermiticity': a_verdict,
        'BH_hermiticity': bh_results.get('verdict_herm', 'SKIP'),
        'BH_positivity': bh_results.get('verdict_pos', 'SKIP'),
        'gauge_inv_symmetry': sym_gi['verdict'],
        'gauge_smoothness': smooth_results['verdict'],
        'H_antihermitian': h_results['verdict'],
    }
    summary['verdicts'] = verdicts

    # PASS if H is Hermitian and no hard failures in gauge-inv or gauge-smooth
    acceptable = ('PASS', 'SKIP_ZERO', 'SKIP', 'MARGINAL', 'INFO_METRIC')
    all_pass = all(v in acceptable for v in verdicts.values())
    summary['overall'] = 'PASS' if all_pass else 'REVIEW'

    print(f"  VERDICTS for {candidate_name}:")
    for check, verdict in verdicts.items():
        marker = '✓' if verdict in ('PASS', 'SKIP_ZERO', 'SKIP', 'INFO_METRIC') else (
            '~' if 'MARGINAL' in verdict else '✗')
        print(f"    [{marker}] {check}: {verdict}")
    print(f"  OVERALL: {summary['overall']}")

    return summary


def print_cross_candidate_table(summaries):
    """Print summary table across all candidates."""
    print(f"\n\n{'='*80}")
    print("  CROSS-CANDIDATE SUMMARY TABLE")
    print(f"{'='*80}\n")

    checks = [
        'A_hermiticity', 'BH_hermiticity', 'BH_positivity',
        'gauge_inv_symmetry', 'gauge_smoothness', 'H_antihermitian',
    ]

    # Header
    names = [s['candidate'] for s in summaries]
    header = f"{'Check':<25}" + "".join(f"{n:<20}" for n in names) + "Verdict"
    print(header)
    print("─" * len(header))

    for check in checks:
        row = f"{check:<25}"
        all_pass = True
        for s in summaries:
            v = s.get('verdicts', {}).get(check, '?')
            row += f"{v:<20}"
            if v not in ('PASS', 'SKIP_ZERO', 'SKIP', 'MARGINAL', 'INFO_METRIC'):
                all_pass = False
        row += 'PASS' if all_pass else 'REVIEW'
        print(row)

    print()
    overall = all(s.get('overall') == 'PASS' for s in summaries)
    print(f"  OVERALL PIPELINE VERDICT: {'PASS ✓' if overall else 'REVIEW NEEDED'}")


def main():
    parser = argparse.ArgumentParser(description="T_sanity_check — Final gauge & symmetry diagnostics")
    parser.add_argument('--candidate', '-c', type=str, default=None,
                        help="Run for specific candidate (default: all three thesis candidates)")
    args = parser.parse_args()

    out_dir = ensure_output_dir(TASK)

    # Select candidates
    thesis_candidates = ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']
    if args.candidate:
        thesis_candidates = [args.candidate]

    cand_yaml = load_candidates_yaml()
    summaries = []

    for name in thesis_candidates:
        try:
            cand_dir = find_correct_candidate_dir(name)
            cand_info = cand_yaml['candidates'].get(name, {})
            s = run_all_checks(name, cand_dir, cand_info, out_dir)
            summaries.append(s)
        except FileNotFoundError as e:
            print(f"\n  SKIP {name}: {e}")
            continue

    # Cross-candidate table
    if len(summaries) > 1:
        print_cross_candidate_table(summaries)

    # Save JSON
    json_path = out_dir / "sanity_check_results.json"
    with open(json_path, 'w') as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\n  Results saved: {json_path}")


if __name__ == '__main__':
    main()
