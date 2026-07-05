#!/usr/bin/env python3
"""
Moiré Miniband Structure Analysis
===================================

Computes the moiré miniband dispersion E_n(q) by sweeping Bloch wavevector q
along the high-symmetry path Γ → X → M → Γ of the moiré Brillouin zone.

At each q-point, Bloch phases e^{iq·L} are injected into the periodic
finite-difference operators (derivative & Laplacian) at the wrap-around
boundary elements. This turns the periodic eigenvalue problem into a
Bloch eigenvalue problem without modifying the Hamiltonian assembly logic.

Physics:
  The envelope Hamiltonian H(R) acts on F(R) with periodic boundary conditions
  on the moiré unit cell. The Bloch ansatz F(R) = e^{iq·R} f(R), with f periodic,
  transforms ∂/∂R → ∂/∂R + iq in the FD stencils. Equivalently, the periodic
  wrap-around elements in the FD matrices acquire phases e^{±iqL}.

  The resulting eigenvalues E_n(q) form the moiré miniband structure.
  Flat minibands (small bandwidth W) with large gaps Δ indicate strong
  photon confinement and slow group velocity — the photonic analogue of
  magic-angle flat bands in twisted bilayer graphene.

Data source:
  C4-symmetrized Phase 2 data at θ=1.1°, square lattice, 5-band subspace [5-9].
  Downsampled to Ns=64 for speed (~1s per q-point solve).

Output:
  - Miniband dispersion plot (4 panels)
  - Miniband metrics (bandwidths, gaps, flatness ratios)
  - JSON data file for further analysis

Author: Moiré envelope pipeline
Date: 2026-02-17
"""

import sys
import os
import json
import time
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import (csr_matrix, lil_matrix, kron, eye, diags)
from scipy.sparse.linalg import eigsh

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "phasesV3"))

from phase3_mpb_v3 import (
    _regularize_M_inv,
    _build_band_block_diagonal,
    build_multiband_potential_operator,
)

RUN_DIR = REPO / "runsV3" / "phase0_mpb_v3_candidate_search_b20_20260217_132202"
CAND    = RUN_DIR / "candidate_214201"
H5_SYM  = CAND / "phase2_multiband_data_c4sym.h5"
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────
NS_TARGET = 64          # Downsample from 128 to 64 for speed
N_MODES = 20            # Number of eigenvalues per q-point
N_PER_SEGMENT = 6       # Points per high-symmetry segment (including endpoints)
FD_ORDER = 4            # Finite-difference order
M_INV_MAX_TRACE = 20.0  # Regularization clamp for M_inv


# ===========================================================================
#  Data Loading & Downsampling
# ===========================================================================

def load_phase2(h5_path):
    """Load Phase 2 HDF5 data and metadata."""
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
    Ns = field.shape[0]
    Ns_new = Ns // factor
    shape_extra = field.shape[2:]
    result = np.zeros((Ns_new, Ns_new) + shape_extra, dtype=field.dtype)
    for i in range(factor):
        for j in range(factor):
            result += field[i::factor, j::factor, ...]
    result /= factor**2
    return result


def prepare_data(data, Ns_target):
    """Downsample all Phase 2 fields to target grid size."""
    Ns_orig = data['Ns1']
    factor = Ns_orig // Ns_target
    if factor == 1:
        return data.copy()

    out = {}
    for key in ['Lambda', 'A_berry', 'Phi_BH', 'v_drift', 'M_inv', 'omega']:
        out[key] = downsample_field(data[key], factor)
    for key in ['omega_ref', 'eta', 'Nb', 'B_moire', 'target_idx']:
        out[key] = data[key]
    out['Ns1'] = out['Ns2'] = Ns_target
    return out


# ===========================================================================
#  Bloch-Phase Finite-Difference Operators
# ===========================================================================

def build_bloch_derivative_matrix(N, ds, q_phase, order=4):
    """
    Build periodic FD derivative matrix with Bloch phase twist.
    
    At wrap-around boundaries, stencil coefficients are multiplied by
    e^{±i·q_phase} where q_phase = q_component * L (L = N*ds).
    
    For wrap from the right end to the left (forward wrap, offset > 0):
      coeff * e^{+i·q_phase}
    For wrap from the left end to the right (backward wrap, offset < 0):
      coeff * e^{-i·q_phase}
      
    When q_phase=0, this reduces to the standard periodic derivative.
    
    Args:
        N: number of grid points
        ds: grid spacing
        q_phase: Bloch phase = q_j * L_j (dimensionless)
        order: FD order (2 or 4)
        
    Returns:
        D: (N, N) sparse derivative matrix (complex)
    """
    if order == 4:
        coeffs = np.array([1, -8, 0, 8, -1]) / (12 * ds)
        offsets = [-2, -1, 0, 1, 2]
    else:
        coeffs = np.array([-0.5, 0, 0.5]) / ds
        offsets = [-1, 0, 1]

    # Build bulk diagonals (real)
    diagonals = [np.full(N, c, dtype=complex) for c in coeffs]
    D = lil_matrix((N, N), dtype=complex)

    # Fill bulk
    for diag, offset in zip(diagonals, offsets):
        for i in range(N):
            j = (i + offset) % N
            # Only set bulk (non-wrapping) elements here
            if 0 <= i + offset < N:
                D[i, j] = diag[i]

    # Fill wrap-around elements with Bloch phases
    phase_fwd = np.exp(1j * q_phase)   # forward wrap: index goes past N-1 → 0
    phase_bwd = np.exp(-1j * q_phase)  # backward wrap: index goes below 0 → N-1

    for coeff, offset in zip(coeffs, offsets):
        if offset < 0:
            # Backward wrap: rows 0..|offset|-1 reference cols N+offset..N-1
            for i in range(-offset):
                D[i, N + offset + i] = coeff * phase_bwd
        elif offset > 0:
            # Forward wrap: rows N-offset..N-1 reference cols 0..offset-1
            for i in range(offset):
                D[N - offset + i, i] = coeff * phase_fwd

    return D.tocsr()


def build_bloch_laplacian_matrix(N, ds, q_phase, order=4):
    """
    Build periodic FD Laplacian matrix with Bloch phase twist.
    
    Same phase convention as build_bloch_derivative_matrix.
    
    Args:
        N: number of grid points
        ds: grid spacing
        q_phase: Bloch phase = q_j * L_j (dimensionless)
        order: FD order (2 or 4)
        
    Returns:
        L: (N, N) sparse Laplacian matrix (complex)
    """
    if order == 4:
        coeffs = np.array([-1, 16, -30, 16, -1]) / (12 * ds**2)
        offsets = [-2, -1, 0, 1, 2]
    else:
        coeffs = np.array([1, -2, 1]) / ds**2
        offsets = [-1, 0, 1]

    D = lil_matrix((N, N), dtype=complex)

    # Fill bulk (non-wrapping)
    for coeff, offset in zip(coeffs, offsets):
        for i in range(N):
            if 0 <= i + offset < N:
                D[i, i + offset] = coeff

    # Fill wrap-around with Bloch phases
    phase_fwd = np.exp(1j * q_phase)
    phase_bwd = np.exp(-1j * q_phase)

    for coeff, offset in zip(coeffs, offsets):
        if offset < 0:
            for i in range(-offset):
                D[i, N + offset + i] = coeff * phase_bwd
        elif offset > 0:
            for i in range(offset):
                D[N - offset + i, i] = coeff * phase_fwd

    return D.tocsr()


# ===========================================================================
#  Bloch Hamiltonian Assembly
# ===========================================================================

def build_bloch_drift_operator(v_drift, Ns1, Ns2, N_bands, dR1, dR2,
                               q_phase1, q_phase2, order=4):
    """
    Build drift operator with Bloch-phase derivatives.
    
    T_drift = -i/(2π) * (V1 @ D1 + V2 @ D2)
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands

    # Velocity multiplication operators (same as periodic case)
    v_flat = v_drift.reshape(N_s, N_bands, N_bands, 2)
    k_grid = np.arange(N_s)

    rows1, cols1, data1 = [], [], []
    rows2, cols2, data2 = [], [], []
    for m in range(N_bands):
        for n in range(N_bands):
            vals1 = v_flat[:, m, n, 0]
            mask1 = np.abs(vals1) > 1e-15
            if np.any(mask1):
                k = k_grid[mask1]
                rows1.append(k * N_bands + m)
                cols1.append(k * N_bands + n)
                data1.append(vals1[mask1])
            vals2 = v_flat[:, m, n, 1]
            mask2 = np.abs(vals2) > 1e-15
            if np.any(mask2):
                k = k_grid[mask2]
                rows2.append(k * N_bands + m)
                cols2.append(k * N_bands + n)
                data2.append(vals2[mask2])

    V1_op = csr_matrix((N_total, N_total), dtype=complex)
    V2_op = csr_matrix((N_total, N_total), dtype=complex)
    if rows1:
        V1_op = csr_matrix((np.concatenate(data1),
                            (np.concatenate(rows1), np.concatenate(cols1))),
                           shape=(N_total, N_total))
    if rows2:
        V2_op = csr_matrix((np.concatenate(data2),
                            (np.concatenate(rows2), np.concatenate(cols2))),
                           shape=(N_total, N_total))

    # Bloch-phase derivative operators
    D1_base = build_bloch_derivative_matrix(Ns1, dR1, q_phase1, order)
    D2_base = build_bloch_derivative_matrix(Ns2, dR2, q_phase2, order)
    D1_full = kron(D1_base, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2_base, eye(N_bands)), format='csr')

    coeff = 1.0 / (2 * np.pi)
    T_drift = -1j * coeff * (V1_op @ D1_full + V2_op @ D2_full)
    return T_drift


def build_bloch_kinetic_operator(M_inv, A_berry, Ns1, Ns2, N_bands, dR1, dR2,
                                  B_moire, q_phase1, q_phase2, order=4,
                                  include_offdiag_A=False):
    """
    Build kinetic operator with Bloch-phase FD operators.
    
    K = prefactor * [-M11 L1 - M22 L2 - 2 M12 D1 D2 + Berry terms]
    Hermitized: K → (K + K†)/2
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    scale_factor = 1.0 / (2 * np.pi)**2
    prefactor = 0.5 * scale_factor

    # Bloch-phase base operators
    L1 = build_bloch_laplacian_matrix(Ns1, dR1, q_phase1, order)
    L2 = build_bloch_laplacian_matrix(Ns2, dR2, q_phase2, order)
    D1 = build_bloch_derivative_matrix(Ns1, dR1, q_phase1, order)
    D2 = build_bloch_derivative_matrix(Ns2, dR2, q_phase2, order)

    # Flatten M_inv to (N_total, 2, 2) — diagonal in bands
    M_inv_flat = np.zeros((N_total, 2, 2), dtype=complex)
    M_inv_reshaped = M_inv.reshape(N_s, N_bands, N_bands, 2, 2)
    for n in range(N_bands):
        indices = np.arange(N_s) * N_bands + n
        M_inv_flat[indices] = M_inv_reshaped[:, n, n, :, :]

    M11_diag = diags(M_inv_flat[:, 0, 0], format='csr')
    M22_diag = diags(M_inv_flat[:, 1, 1], format='csr')
    M12_diag = diags(M_inv_flat[:, 0, 1], format='csr')

    # Full-space operators
    L1_full = kron(L1, eye(Ns2 * N_bands), format='csr')
    L2_full = kron(eye(Ns1), kron(L2, eye(N_bands)), format='csr')
    D1_full = kron(D1, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2, eye(N_bands)), format='csr')

    K_op = -(M11_diag @ L1_full + M22_diag @ L2_full)
    if np.max(np.abs(M_inv_flat[:, 0, 1])) > 1e-15:
        K_op = K_op - 2 * M12_diag @ (D1_full @ D2_full)
    K_op = prefactor * K_op

    # Berry connection terms
    A_berry_reshaped = A_berry.reshape(N_s, N_bands, N_bands, 2)

    if include_offdiag_A:
        M_diag = np.zeros((N_s, N_bands, 2, 2), dtype=complex)
        for p in range(N_bands):
            M_diag[:, p] = M_inv_reshaped[:, p, p]

        # Diamagnetic A² term
        B_ma = np.einsum('kpij,kpnj->kpni', M_diag, A_berry_reshaped)
        A2_val = np.einsum('kmpi,kpni->kmn', A_berry_reshaped, B_ma)
        if np.max(np.abs(A2_val)) > 1e-15:
            A2_op = _build_band_block_diagonal(A2_val, N_s, N_bands, N_total)
            K_op = K_op + prefactor * A2_op

        # Paramagnetic cross-terms
        VA = np.einsum('kmij,kmni->kmnj', M_diag, A_berry_reshaped)
        if np.max(np.abs(VA)) > 1e-15:
            VA_mat_0 = _build_band_block_diagonal(VA[:, :, :, 0], N_s, N_bands, N_total)
            VA_mat_1 = _build_band_block_diagonal(VA[:, :, :, 1], N_s, N_bands, N_total)
            para_op = -1j * prefactor * (VA_mat_0 @ D1_full + VA_mat_1 @ D2_full)
            K_op = K_op + para_op
    else:
        # Legacy diagonal-only Berry
        A_berry_flat = np.zeros((N_total, 2), dtype=complex)
        for n in range(N_bands):
            indices = np.arange(N_s) * N_bands + n
            A_berry_flat[indices] = A_berry_reshaped[:, n, n, :]
        A1 = A_berry_flat[:, 0]
        A2 = A_berry_flat[:, 1]
        M11 = M_inv_flat[:, 0, 0]
        M22 = M_inv_flat[:, 1, 1]
        M12 = M_inv_flat[:, 0, 1]
        A_sq_val = (M11 * np.abs(A1)**2 + M22 * np.abs(A2)**2 +
                    2 * M12 * np.real(A1 * np.conj(A2)))
        if np.max(np.abs(A_sq_val)) > 1e-15:
            K_op = K_op + diags(prefactor * A_sq_val, format='csr')

    # Hermitize
    K_op = (K_op + K_op.T.conj()) / 2
    return K_op


def assemble_bloch_hamiltonian(data, q_vec, include_offdiag_A=False, order=4):
    """
    Assemble the full multi-band envelope Hamiltonian at Bloch wavevector q.
    
    Args:
        data: dict with downsampled Phase 2 fields and metadata
        q_vec: (2,) Bloch wavevector in physical reciprocal space (1/a units)
        include_offdiag_A: use full off-diagonal Berry connection
        order: FD order
        
    Returns:
        H: sparse Hermitian Hamiltonian (CSR)
    """
    Ns1 = data['Ns1']
    Ns2 = data['Ns2']
    Nb = data['Nb']
    eta = data['eta']
    B_moire = data['B_moire']

    # Physical grid spacing: dR_j = |a_j| / Ns_j where a_j = B_moire[j]
    L1 = np.linalg.norm(B_moire[0])
    L2 = np.linalg.norm(B_moire[1])
    dR1 = L1 / Ns1
    dR2 = L2 / Ns2

    # Bloch phases: q_phase_j = q · a_j  (dot product with j-th lattice vector)
    # This is correct for both orthogonal (square) and non-orthogonal (hex) lattices.
    # q_vec is in physical reciprocal space (1/a units).
    q_phase1 = np.dot(q_vec, B_moire[0])
    q_phase2 = np.dot(q_vec, B_moire[1])

    Lambda = data['Lambda']
    v_drift = data['v_drift']
    M_inv = _regularize_M_inv(data['M_inv'].copy(), M_INV_MAX_TRACE)
    A_berry = data['A_berry']
    Phi_BH = data['Phi_BH']

    # (1) Potential (q-independent)
    H = build_multiband_potential_operator(Lambda, B_moire)

    # (2) Drift (uses Bloch derivatives)
    T_drift = build_bloch_drift_operator(
        v_drift, Ns1, Ns2, Nb, dR1, dR2, q_phase1, q_phase2, order
    )
    H = H + T_drift

    # (3) Kinetic (uses Bloch derivatives + Laplacians)
    K_op = build_bloch_kinetic_operator(
        M_inv, A_berry, Ns1, Ns2, Nb, dR1, dR2, B_moire,
        q_phase1, q_phase2, order, include_offdiag_A=include_offdiag_A
    )
    H = H + K_op

    # (4) Born-Huang potential (q-independent, scalar)
    if np.max(np.abs(Phi_BH)) > 1e-15:
        U_BH = build_multiband_potential_operator(Phi_BH, None)
        H = H + U_BH

    H = H.tocsr()
    return H


# ===========================================================================
#  Moiré Brillouin Zone Path
# ===========================================================================

def get_moire_bz_path(B_moire, n_per_segment=N_PER_SEGMENT, lattice_type='square'):
    """
    Build the high-symmetry path in the moiré BZ.
    
    For square lattice:  Γ → X → M → Γ
      Γ = (0,0),  X = (1/2, 0),  M = (1/2, 1/2)
      
    For hex/honeycomb:  Γ → K → M → Γ
      Γ = (0,0),  K = (1/3, 2/3),  M = (1/2, 0)
      
    Fractional coordinates converted to physical q via:
      q = frac[0]*G1 + frac[1]*G2  where G = 2π * B_moire^{-T}
    
    Returns:
        q_points: (N_q, 2) array of q-vectors in physical 1/a units
        q_dist: (N_q,) cumulative path distance for plotting
        tick_positions: list of tick positions at high-symmetry points
        tick_labels: list of labels
    """
    # Moiré reciprocal lattice vectors
    # B_moire rows are the real-space moiré lattice vectors
    # G = 2π * B_moire^{-T}
    G = 2 * np.pi * np.linalg.inv(B_moire).T  # (2, 2), rows = G1, G2

    # High-symmetry points in fractional coordinates
    if lattice_type in ('hex', 'honeycomb'):
        hs_points = [
            ('Γ', np.array([0.0, 0.0])),
            ('K', np.array([1/3, 2/3])),
            ('M', np.array([0.5, 0.0])),
            ('Γ', np.array([0.0, 0.0])),
        ]
    else:
        hs_points = [
            ('Γ', np.array([0.0, 0.0])),
            ('X', np.array([0.5, 0.0])),
            ('M', np.array([0.5, 0.5])),
            ('Γ', np.array([0.0, 0.0])),
        ]

    q_points = []
    q_dist = []
    tick_positions = []
    tick_labels = []
    cumulative_dist = 0.0

    for seg_idx in range(len(hs_points) - 1):
        label_start, frac_start = hs_points[seg_idx]
        label_end, frac_end = hs_points[seg_idx + 1]

        q_start = frac_start[0] * G[0] + frac_start[1] * G[1]
        q_end = frac_end[0] * G[0] + frac_end[1] * G[1]

        # n_per_segment points including start, excluding end (except last segment)
        is_last = (seg_idx == len(hs_points) - 2)
        n_pts = n_per_segment if is_last else n_per_segment - 1

        for i in range(n_pts if is_last else n_per_segment - 1):
            t = i / (n_per_segment - 1)
            q = q_start + t * (q_end - q_start)
            dist = cumulative_dist + t * np.linalg.norm(q_end - q_start)
            q_points.append(q)
            q_dist.append(dist)

        # Record tick at segment start
        if seg_idx == 0:
            tick_positions.append(cumulative_dist)
            tick_labels.append(label_start)

        cumulative_dist += np.linalg.norm(q_end - q_start)

        # Record tick at segment end
        tick_positions.append(cumulative_dist)
        tick_labels.append(label_end)

    # Add the final point (Γ at end)
    q_points.append(hs_points[-1][1][0] * G[0] + hs_points[-1][1][1] * G[1])
    q_dist.append(cumulative_dist)

    q_points = np.array(q_points)
    q_dist = np.array(q_dist)

    # Deduplicate (remove duplicate points from segment boundaries)
    _, unique_idx = np.unique(np.round(q_dist, decimals=10), return_index=True)
    unique_idx = np.sort(unique_idx)
    q_points = q_points[unique_idx]
    q_dist = q_dist[unique_idx]

    return q_points, q_dist, tick_positions, tick_labels


# ===========================================================================
#  q-Sweep Solver
# ===========================================================================

def sweep_q_points(data, q_points, n_modes, include_offdiag_A=False, label="",
                   sigma=None, candidate_type=None):
    """
    Solve the Bloch Hamiltonian at each q-point and collect eigenvalues.
    
    Args:
        data: downsampled Phase 2 data dict
        q_points: (N_q, 2) array of q-vectors
        n_modes: number of eigenvalues per q-point
        include_offdiag_A: use full Berry connection
        label: description for progress printing
        
    Returns:
        all_evals: (N_q, n_modes) array of eigenvalues, sorted ascending per q
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
        print(f"\n  [{label}] σ = {sigma:.6f} ({sigma_info['method']})")
    else:
        print(f"\n  [{label}] σ = {sigma:.6f} (user-provided)")

    print(f"  Sweeping {N_q} q-points, {n_modes} modes each")
    
    t0 = time.time()
    hermiticity_errors = []

    for iq, q in enumerate(q_points):
        t_q = time.time()
        H = assemble_bloch_hamiltonian(data, q, include_offdiag_A=include_offdiag_A)

        # Check Hermiticity
        nh = np.linalg.norm((H - H.T.conj()).data) / np.linalg.norm(H.data) if H.nnz > 0 else 0
        hermiticity_errors.append(nh)

        try:
            evals, _ = eigsh(H, k=n_modes, sigma=sigma, which='LM',
                             maxiter=10000, tol=1e-10)
            all_evals[iq] = np.sort(evals)
        except Exception as e:
            print(f"    q[{iq}] FAILED: {e}")
            continue

        dt = time.time() - t_q
        if iq == 0 or (iq + 1) % 5 == 0 or iq == N_q - 1:
            print(f"    q[{iq:2d}] = ({q[0]:+.4f}, {q[1]:+.4f})  "
                  f"E_min={all_evals[iq, 0]:.6f}  E_max={all_evals[iq, -1]:.6f}  "
                  f"nh={nh:.1e}  dt={dt:.2f}s")

    dt_total = time.time() - t0
    print(f"  Total: {dt_total:.1f}s  ({dt_total/N_q:.2f}s/q-point)")
    print(f"  Max Hermiticity error: {max(hermiticity_errors):.2e}")

    return all_evals


# ===========================================================================
#  Miniband Analysis
# ===========================================================================

def extract_miniband_metrics(all_evals, q_dist):
    """
    Extract miniband widths, gaps, and flatness ratios from E_n(q).
    
    Minibands are identified as the n-th eigenvalue track across q-points.
    
    Returns:
        metrics: list of dicts, one per miniband
    """
    N_q, n_modes = all_evals.shape
    metrics = []

    for n in range(n_modes):
        band = all_evals[:, n]
        valid = ~np.isnan(band)
        if np.sum(valid) < 2:
            continue

        W = np.max(band[valid]) - np.min(band[valid])
        E_mean = np.mean(band[valid])
        E_min = np.min(band[valid])
        E_max = np.max(band[valid])

        # Gap to next band
        gap_above = np.nan
        if n + 1 < n_modes:
            next_band = all_evals[:, n + 1]
            both_valid = valid & ~np.isnan(next_band)
            if np.sum(both_valid) > 0:
                gap_above = np.min(next_band[both_valid]) - np.max(band[both_valid])

        # Flatness = gap / bandwidth
        flatness = gap_above / W if (W > 1e-15 and not np.isnan(gap_above)) else np.nan

        metrics.append({
            'band_index': n,
            'E_min': float(E_min),
            'E_max': float(E_max),
            'E_mean': float(E_mean),
            'bandwidth': float(W),
            'gap_above': float(gap_above) if not np.isnan(gap_above) else None,
            'flatness': float(flatness) if not np.isnan(flatness) else None,
        })

    return metrics


# ===========================================================================
#  Plotting
# ===========================================================================

def plot_miniband_structure(q_dist, ticks_pos, ticks_labels,
                            evals_diag, evals_full,
                            metrics_diag, metrics_full,
                            data, save_dir):
    """
    4-panel miniband structure plot.
    
    (a) Full band structure E_n(q) — both Berry variants
    (b) Zoom into the lowest few minibands
    (c) Bandwidth W_n for each miniband (bar chart)
    (d) Flatness ratio Δ_n / W_n (bar chart)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Moiré Miniband Structure — θ={np.degrees(2*np.arcsin(data['eta']/2)):.1f}°, "
        f"Ns={data['Ns1']}, {data['Nb']}-band subspace, {N_MODES} modes",
        fontsize=13, fontweight='bold'
    )

    # Colors for the two variants
    c_diag = 'C0'
    c_full = 'C3'

    # ── Panel (a): Full band structure ──────────────────────────────────
    ax = axes[0, 0]
    for n in range(N_MODES):
        ax.plot(q_dist, evals_diag[:, n], '-', color=c_diag, lw=0.8,
                alpha=0.7, label='diag A' if n == 0 else None)
        ax.plot(q_dist, evals_full[:, n], '--', color=c_full, lw=0.8,
                alpha=0.7, label='full A' if n == 0 else None)

    for tp in ticks_pos:
        ax.axvline(tp, color='gray', lw=0.5, ls=':')
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_labels)
    ax.set_ylabel('E (ω − ω_ref)')
    ax.set_title('(a) Moiré miniband dispersion')
    ax.legend(fontsize=8, loc='upper right')

    # ── Panel (b): Zoom into lowest few minibands ──────────────────────
    ax = axes[0, 1]
    n_zoom = min(8, N_MODES)
    # Use pastel colormap for individual bands
    cmap = plt.cm.tab10
    for n in range(n_zoom):
        color = cmap(n % 10)
        ax.plot(q_dist, evals_diag[:, n], '-', color=color, lw=1.2,
                label=f'n={n}' if n < 6 else None)
        ax.plot(q_dist, evals_full[:, n], '--', color=color, lw=1.0, alpha=0.6)

    for tp in ticks_pos:
        ax.axvline(tp, color='gray', lw=0.5, ls=':')
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_labels)
    ax.set_ylabel('E (ω − ω_ref)')
    ax.set_title(f'(b) Lowest {n_zoom} minibands (solid=diag, dashed=full A)')
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    # ── Panel (c): Miniband bandwidths ─────────────────────────────────
    ax = axes[1, 0]
    n_show = min(12, len(metrics_diag), len(metrics_full))
    x_idx = np.arange(n_show)
    bw_diag = [metrics_diag[i]['bandwidth'] for i in range(n_show)]
    bw_full = [metrics_full[i]['bandwidth'] for i in range(n_show)]

    bar_width = 0.35
    ax.bar(x_idx - bar_width/2, bw_diag, bar_width, color=c_diag, alpha=0.8, label='diag A')
    ax.bar(x_idx + bar_width/2, bw_full, bar_width, color=c_full, alpha=0.8, label='full A')
    ax.set_xlabel('Miniband index n')
    ax.set_ylabel('Bandwidth W_n')
    ax.set_title('(c) Miniband bandwidths')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.set_xticks(x_idx)

    # ── Panel (d): Flatness ratios ─────────────────────────────────────
    ax = axes[1, 1]
    flat_diag = [metrics_diag[i]['flatness'] if metrics_diag[i]['flatness'] is not None else 0
                 for i in range(n_show)]
    flat_full = [metrics_full[i]['flatness'] if metrics_full[i]['flatness'] is not None else 0
                 for i in range(n_show)]

    ax.bar(x_idx - bar_width/2, flat_diag, bar_width, color=c_diag, alpha=0.8, label='diag A')
    ax.bar(x_idx + bar_width/2, flat_full, bar_width, color=c_full, alpha=0.8, label='full A')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='Δ = W (gap = bandwidth)')
    ax.set_xlabel('Miniband index n')
    ax.set_ylabel('Flatness Δ_n / W_n')
    ax.set_title('(d) Flatness ratio (Δ/W > 1 = isolated flat band)')
    ax.legend(fontsize=8)
    ax.set_xticks(x_idx)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = save_dir / f"miniband_structure.{ext}"
        plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved miniband_structure.png/pdf to {save_dir}/")
    plt.close()


# ===========================================================================
#  Verification
# ===========================================================================

def verify_gamma_consistency(data, evals_at_gamma):
    """Check that q=0 eigenvalues match a standalone Γ-point solve."""
    print("\n  Verification: Γ-point consistency")
    q_zero = np.array([0.0, 0.0])
    H = assemble_bloch_hamiltonian(data, q_zero, include_offdiag_A=False)

    target = data['target_idx']
    V_target = data['Lambda'][:, :, target, target].real
    sigma = float(np.mean([data['Lambda'][:, :, n, n].real for n in range(data['Nb'])]))

    evals_check, _ = eigsh(H, k=N_MODES, sigma=sigma, which='LM',
                           maxiter=10000, tol=1e-10)
    evals_check = np.sort(evals_check)

    diff = np.abs(evals_at_gamma - evals_check)
    print(f"    Max |E_sweep(Γ) - E_check(Γ)| = {np.max(diff):.2e}")
    print(f"    {'✓ PASS' if np.max(diff) < 1e-8 else '⚠ DISCREPANCY'}")


def verify_periodicity(evals_all, q_dist, ticks_pos):
    """Check that E(Γ_start) ≈ E(Γ_end)."""
    print("\n  Verification: BZ periodicity")
    diff = np.abs(evals_all[0] - evals_all[-1])
    print(f"    Max |E(Γ_start) - E(Γ_end)| = {np.max(diff):.2e}")
    print(f"    {'✓ PASS' if np.max(diff) < 1e-6 else '⚠ DISCREPANCY'}")


# ===========================================================================
#  Main
# ===========================================================================

def main(h5_path=None, plot_dir=None, ns_target=NS_TARGET, n_modes=N_MODES,
         n_per_segment=N_PER_SEGMENT):
    """Run miniband structure analysis.
    
    Args:
        h5_path: Path to C4-symmetrized Phase 2 HDF5. If None, uses default.
        plot_dir: Output directory for plots. If None, uses default.
        ns_target: Grid size to downsample to.
        n_modes: Number of eigenvalues per q-point.
        n_per_segment: q-points per high-symmetry segment.
    """
    if h5_path is None:
        h5_path = H5_SYM
    else:
        h5_path = Path(h5_path)
    if plot_dir is None:
        plot_dir = PLOT_DIR
    else:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  MOIRÉ MINIBAND STRUCTURE ANALYSIS")
    print("=" * 72)

    # ── [1] Load and downsample data ──────────────────────────────────
    print(f"\n[1] Loading C4-symmetrized Phase 2 data from:\n    {h5_path}")
    raw = load_phase2(h5_path)
    print(f"    Original grid: {raw['Ns1']}×{raw['Ns2']}, {raw['Nb']} bands")
    print(f"    η = {raw['eta']:.6f}, θ ≈ {np.degrees(2*np.arcsin(raw['eta']/2)):.2f}°")

    data = prepare_data(raw, ns_target)
    Ns = data['Ns1']
    Nb = data['Nb']
    eta = data['eta']
    L_moire = 1.0 / eta
    dR = L_moire / Ns
    print(f"    Downsampled to: {Ns}×{Ns}, L_moire={L_moire:.2f}a, dR={dR:.4f}a")

    # ── [2] Build moiré BZ path ──────────────────────────────────────
    print(f"\n[2] Building moiré BZ path: Γ → X → M → Γ")
    B_moire = data['B_moire']
    G = 2 * np.pi * np.linalg.inv(B_moire).T
    print(f"    B_moire = {B_moire}")
    print(f"    G_moire = {G}")
    print(f"    |G1| = {np.linalg.norm(G[0]):.4f}, |G2| = {np.linalg.norm(G[1]):.4f}")

    q_points, q_dist, ticks_pos, ticks_labels = get_moire_bz_path(B_moire, n_per_segment)
    N_q = len(q_points)
    print(f"    {N_q} q-points along path")
    print(f"    Path length: {q_dist[-1]:.4f} (1/a)")

    # ── [3] Solve with diagonal-only Berry connection ────────────────
    print(f"\n[3] Solving moiré band structure (diagonal Berry connection)...")
    evals_diag = sweep_q_points(data, q_points, n_modes,
                                include_offdiag_A=False, label="diag-A")

    # ── [4] Solve with full off-diagonal Berry connection ────────────
    print(f"\n[4] Solving moiré band structure (full off-diagonal Berry)...")
    evals_full = sweep_q_points(data, q_points, n_modes,
                                include_offdiag_A=True, label="full-A")

    # ── [5] Verification ─────────────────────────────────────────────
    print(f"\n[5] Verification checks...")
    verify_gamma_consistency(data, evals_diag[0])
    verify_periodicity(evals_diag, q_dist, ticks_pos)
    verify_periodicity(evals_full, q_dist, ticks_pos)

    # ── [6] Extract miniband metrics ─────────────────────────────────
    print(f"\n[6] Miniband metrics...")
    metrics_diag = extract_miniband_metrics(evals_diag, q_dist)
    metrics_full = extract_miniband_metrics(evals_full, q_dist)

    print(f"\n  {'':4s}  {'--- Diagonal A ---':^36s}  {'--- Full A ---':^36s}")
    print(f"  {'n':>3s}  {'W_n':>10s}  {'Δ_n':>10s}  {'Δ/W':>8s}  {'E_mean':>10s}"
          f"  {'W_n':>10s}  {'Δ_n':>10s}  {'Δ/W':>8s}")
    print(f"  {'---':>3s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}  {'----------':>10s}"
          f"  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for i in range(min(n_modes, len(metrics_diag), len(metrics_full))):
        md, mf = metrics_diag[i], metrics_full[i]
        gap_d = f"{md['gap_above']:.2e}" if md['gap_above'] is not None else "—"
        gap_f = f"{mf['gap_above']:.2e}" if mf['gap_above'] is not None else "—"
        flat_d = f"{md['flatness']:.2f}" if md['flatness'] is not None else "—"
        flat_f = f"{mf['flatness']:.2f}" if mf['flatness'] is not None else "—"
        print(f"  {i:>3d}  {md['bandwidth']:.2e}  {gap_d:>10s}  {flat_d:>8s}  {md['E_mean']:.6f}"
              f"  {mf['bandwidth']:.2e}  {gap_f:>10s}  {flat_f:>8s}")

    # ── [7] Miniband group analysis ──────────────────────────────────
    print(f"\n[7] Miniband group analysis...")
    # Group minibands by looking for large gaps
    all_bw_diag = [m['bandwidth'] for m in metrics_diag]
    all_gaps_diag = [m['gap_above'] for m in metrics_diag if m['gap_above'] is not None]
    if all_gaps_diag:
        median_gap = np.median([g for g in all_gaps_diag if g > 0])
        print(f"    Median gap (diag A): {median_gap:.2e}")
        print(f"    Mean bandwidth (diag A): {np.mean(all_bw_diag):.2e}")
        
        # Find miniband groups separated by large gaps
        threshold = 3 * median_gap if median_gap > 0 else 1e-4
        groups = []
        current_group = [0]
        for i in range(len(metrics_diag) - 1):
            gap = metrics_diag[i].get('gap_above')
            if gap is not None and gap > threshold:
                groups.append(current_group)
                current_group = [i + 1]
            else:
                current_group.append(i + 1)
        groups.append(current_group)

        print(f"\n    Miniband groups (gap threshold = {threshold:.2e}):")
        for gi, group in enumerate(groups):
            E_lo = metrics_diag[group[0]]['E_min']
            E_hi = metrics_diag[group[-1]]['E_max']
            total_bw = E_hi - E_lo
            print(f"      Group {gi}: bands {group[0]}–{group[-1]} "
                  f"({len(group)} bands), E=[{E_lo:.6f}, {E_hi:.6f}], "
                  f"total BW={total_bw:.4e}")

    # ── [8] Plot ─────────────────────────────────────────────────────
    print(f"\n[8] Generating plots...")
    plot_miniband_structure(q_dist, ticks_pos, ticks_labels,
                            evals_diag, evals_full,
                            metrics_diag, metrics_full,
                            data, plot_dir)

    # ── [9] Save data ────────────────────────────────────────────────
    print(f"\n[9] Saving data...")
    results = {
        'theta_deg': float(np.degrees(2 * np.arcsin(eta / 2))),
        'eta': float(eta),
        'Ns': Ns,
        'Nb': Nb,
        'N_modes': n_modes,
        'N_q': N_q,
        'n_per_segment': n_per_segment,
        'L_moire': float(L_moire),
        'q_dist': q_dist.tolist(),
        'q_points': q_points.tolist(),
        'tick_positions': ticks_pos,
        'tick_labels': ticks_labels,
        'evals_diag': evals_diag.tolist(),
        'evals_full': evals_full.tolist(),
        'metrics_diag': metrics_diag,
        'metrics_full': metrics_full,
    }
    json_path = plot_dir / "miniband_data.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"    Saved {json_path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  SUMMARY")
    print(f"{'=' * 72}")
    print(f"  θ = {np.degrees(2*np.arcsin(eta/2)):.2f}°, η = {eta:.5f}")
    print(f"  Grid: {Ns}×{Ns}×{Nb} = {Ns*Ns*Nb} DoF")
    print(f"  q-path: {N_q} points along Γ→X→M→Γ")
    print(f"  {n_modes} minibands computed")

    if metrics_diag:
        total_bw = metrics_diag[-1]['E_max'] - metrics_diag[0]['E_min']
        mean_bw = np.mean([m['bandwidth'] for m in metrics_diag])
        valid_flat = [m['flatness'] for m in metrics_diag
                      if m['flatness'] is not None and m['flatness'] > 0]
        print(f"\n  --- Diagonal Berry ---")
        print(f"  Total eigenvalue range: {total_bw:.4e}")
        print(f"  Mean miniband width: {mean_bw:.4e}")
        if valid_flat:
            print(f"  Max flatness Δ/W: {max(valid_flat):.2f}")
            n_flat = sum(1 for f in valid_flat if f > 1)
            print(f"  Bands with Δ/W > 1: {n_flat}/{len(valid_flat)}")

    if metrics_full:
        total_bw = metrics_full[-1]['E_max'] - metrics_full[0]['E_min']
        mean_bw = np.mean([m['bandwidth'] for m in metrics_full])
        valid_flat = [m['flatness'] for m in metrics_full
                      if m['flatness'] is not None and m['flatness'] > 0]
        print(f"\n  --- Full Berry ---")
        print(f"  Total eigenvalue range: {total_bw:.4e}")
        print(f"  Mean miniband width: {mean_bw:.4e}")
        if valid_flat:
            print(f"  Max flatness Δ/W: {max(valid_flat):.2f}")
            n_flat = sum(1 for f in valid_flat if f > 1)
            print(f"  Bands with Δ/W > 1: {n_flat}/{len(valid_flat)}")

    print(f"\n{'=' * 72}")
    print(f"  DONE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Moiré miniband structure analysis")
    parser.add_argument("--h5", type=str, default=None,
                        help="Path to C4-symmetrized Phase 2 HDF5 (default: built-in old candidate)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots and data (default: results_bands/plots/)")
    parser.add_argument("--ns", type=int, default=NS_TARGET,
                        help=f"Grid size to downsample to (default: {NS_TARGET})")
    parser.add_argument("--nmodes", type=int, default=N_MODES,
                        help=f"Number of eigenvalues per q-point (default: {N_MODES})")
    parser.add_argument("--nq", type=int, default=N_PER_SEGMENT,
                        help=f"q-points per BZ segment (default: {N_PER_SEGMENT})")
    args = parser.parse_args()
    main(h5_path=args.h5, plot_dir=args.outdir, ns_target=args.ns,
         n_modes=args.nmodes, n_per_segment=args.nq)
