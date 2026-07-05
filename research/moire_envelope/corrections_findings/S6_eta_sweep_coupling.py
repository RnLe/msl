#!/usr/bin/env python3
"""
S6: Twist angle sweep + off-diagonal coupling analysis.
========================================================

Two main analyses:

Part A — Off-diagonal coupling survey:
  - Measure off-diagonal A_berry, Φ_BH, Λ, v_drift, M_inv in Phase 2 data
  - Identify which coupling channels exist vs are missing
  - Build H with off-diagonal A via a CORRECTED kinetic operator
  - Compare band mixing: diagonal-only vs full off-diagonal A

Part B — Twist angle (η) sweep:
  - Same Phase 2 data, vary η (= 2 sin(θ/2))
  - Track V_depth/E_kin ratio, eigenvalue spread, mode localization
  - Find the θ range where the EA produces well-separated bound states
  - Detailed analysis at the optimal θ

Uses: phase2_multiband_data_c4sym.h5 (C4-symmetrized Phase 2 data from S4b)
"""

import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.sparse import (csr_matrix, lil_matrix, kron, eye, diags,
                           block_diag as sp_block_diag)

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "phasesV3"))

from phase3_mpb_v3 import (
    assemble_multiband_hamiltonian,
    _regularize_M_inv,
    build_multiband_potential_operator,
    build_multiband_kinetic_operator,
    build_multiband_drift_operator,
    build_multiband_born_huang_operator,
    build_periodic_laplacian_matrix,
    build_periodic_derivative_matrix,
)

RUN_DIR = REPO / "runsV3" / "phase0_mpb_v3_20260206_152443"
CAND    = RUN_DIR / "candidate_0000"
H5_SYM  = CAND / "phase2_multiband_data_c4sym.h5"
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def load_phase2(h5_path):
    """Load Phase 2 HDF5 data and metadata."""
    with h5py.File(h5_path, 'r') as hf:
        data = {
            'Lambda':   hf['Lambda'][:],
            'A_berry':  hf['A_berry'][:],
            'Phi_BH':   hf['Phi_BH'][:],
            'v_drift':  hf['v_drift'][:],
            'M_inv':    hf['M_inv'][:],
            'omega':    hf['omega'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta':       float(hf.attrs['eta']),
            'Ns1':       int(hf.attrs['Ns1']),
            'Ns2':       int(hf.attrs['Ns2']),
            'Nb':        int(hf.attrs['N_subspace']),
            'B_moire':   hf.attrs['B_moire'][:],
            'target_idx': int(hf.attrs['target_index_in_subspace']),
        }
    Ns = data['Ns1']
    data['Ns'] = Ns
    return data


def downsample_field(field, factor):
    """Downsample via block averaging."""
    Ns = field.shape[0]
    Ns_new = Ns // factor
    shape_extra = field.shape[2:]
    result = np.zeros((Ns_new, Ns_new) + shape_extra, dtype=field.dtype)
    for i in range(factor):
        for j in range(factor):
            result += field[i::factor, j::factor, ...]
    result /= factor**2
    return result


def build_kinetic_with_offdiag_A(
    M_inv, A_berry, eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire, order=4
):
    """
    Build kinetic operator that USES off-diagonal Berry connection.
    
    The standard Phase 3 kinetic operator only uses diagonal A_{nn}.
    This version implements the full covariant derivative:
    
    K_{mn} = (prefactor) * Σ_ij M^{-1}_{ij,nn} δ_{mn} * (-i∂_i)(-i∂_j)
           + (prefactor) * Σ_ij M^{-1}_{ij,nn} * [A_i A_j]_{mn}
           - (prefactor) * Σ_ij M^{-1}_{ij,nn} * (-i)(A_{i,mn} ∂_j + ∂_i A_{j,mn})
    
    Note: M_inv is still diagonal in bands. The off-diagonal coupling comes
    entirely from the Berry connection covariant derivative terms.
    
    This implements: K = (1/2) * M^{-1} * D_cov * D_cov where
    D_cov = -i∂ - A (matrix-valued gauge field)
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    
    scale_factor = 1.0 / (2 * np.pi)**2
    prefactor = 0.5 * scale_factor
    
    # Base FD operators
    L1 = build_periodic_laplacian_matrix(Ns1, dR1, order)
    L2 = build_periodic_laplacian_matrix(Ns2, dR2, order)
    D1 = build_periodic_derivative_matrix(Ns1, dR1, order)
    D2 = build_periodic_derivative_matrix(Ns2, dR2, order)
    
    # Full-space derivative operators
    L1_full = kron(L1, eye(Ns2 * N_bands), format='csr')
    L2_full = kron(eye(Ns1), kron(L2, eye(N_bands)), format='csr')
    D1_full = kron(D1, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2, eye(N_bands)), format='csr')
    
    # ── Term 1: Diagonal kinetic (same as standard) ──────────────────────
    # K_diag = -prefactor * (M11 * L1 + M22 * L2 + 2*M12 * D1*D2)
    M_inv_reshaped = M_inv.reshape(N_s, N_bands, N_bands, 2, 2)
    M_inv_flat = np.zeros((N_total, 2, 2), dtype=complex)
    for n in range(N_bands):
        indices = np.arange(N_s) * N_bands + n
        M_inv_flat[indices] = M_inv_reshaped[:, n, n, :, :]
    
    M11_diag = diags(M_inv_flat[:, 0, 0], format='csr')
    M22_diag = diags(M_inv_flat[:, 1, 1], format='csr')
    M12_diag = diags(M_inv_flat[:, 0, 1], format='csr')
    
    K_op = -prefactor * (M11_diag @ L1_full + M22_diag @ L2_full)
    if np.max(np.abs(M_inv_flat[:, 0, 1])) > 1e-15:
        K_op = K_op - 2 * prefactor * M12_diag @ (D1_full @ D2_full)
    
    # ── Term 2: |A|² diamagnetic (FULL matrix, not just diagonal) ─────────
    # For each spatial point k, bands m,n:
    # [A²]_{mn} = Σ_ij M_{ij,mm} δ_mm * Σ_p A_{i,mp} A_{j,pn}
    # But with M diagonal in bands, for band m:
    # val_{mn}(k) = Σ_ij M_{ij,mm} Σ_p A_{i,mp}(k) A_{j,pn}(k)
    # 
    # This creates off-diagonal terms in the Hamiltonian!
    
    A_reshaped = A_berry.reshape(N_s, N_bands, N_bands, 2)
    
    # Build A² matrix for each spatial point
    rows_a2, cols_a2, data_a2 = [], [], []
    for k_s in range(N_s):
        A_k = A_reshaped[k_s]  # (Nb, Nb, 2)
        for m in range(N_bands):
            M_mm = M_inv_reshaped[k_s, m, m]  # (2, 2)
            for n in range(N_bands):
                # [M A A]_{mn} = Σ_{ij} M_{ij,mm} * Σ_p A_{i,mp} A_{j,pn}
                val = 0.0 + 0.0j
                for i_c in range(2):
                    for j_c in range(2):
                        # Σ_p A[m,p,i] * A[p,n,j]
                        a_prod = np.sum(A_k[m, :, i_c] * A_k[:, n, j_c])
                        val += M_mm[i_c, j_c] * a_prod
                if abs(val) > 1e-18:
                    rows_a2.append(k_s * N_bands + m)
                    cols_a2.append(k_s * N_bands + n)
                    data_a2.append(val)
    
    if rows_a2:
        A2_op = csr_matrix((np.array(data_a2), (np.array(rows_a2), np.array(cols_a2))),
                           shape=(N_total, N_total))
        K_op = K_op + prefactor * A2_op
    
    # ── Term 3: Paramagnetic cross-terms −i(M·A·∂ + ∂·M·A) ──────────────
    # [-i M A D]_{mn} at spatial point k:
    # val = -i Σ_ij M_{ij,mm} A_{i,mn}(k) * (D_j F_n)(k)
    # This is: -i * [Σ_j (Σ_i M_{ij,mm} A_{i,mn}) * D_j] operator
    # 
    # Build as: V_A1_mn(k) = Σ_i M_{ii or ij,mm} A_{i,mn}(k)
    # Then operator = -i * V_A1_mn * D1 + -i * V_A2_mn * D2
    
    # For each (m,n) band pair, build the spatial diagonal operator
    # VA1_{mn}(k) = M_{00,mm}*A_{0,mn} + M_{01,mm}*A_{1,mn}
    # VA2_{mn}(k) = M_{10,mm}*A_{0,mn} + M_{11,mm}*A_{1,mn}
    
    para_rows1, para_cols1, para_data1 = [], [], []
    para_rows2, para_cols2, para_data2 = [], [], []
    
    for m in range(N_bands):
        for n in range(N_bands):
            # VA1[k] = M[k,m,m,0,0]*A[k,m,n,0] + M[k,m,m,0,1]*A[k,m,n,1]
            va1 = (M_inv_reshaped[:, m, m, 0, 0] * A_reshaped[:, m, n, 0] +
                   M_inv_reshaped[:, m, m, 0, 1] * A_reshaped[:, m, n, 1])
            va2 = (M_inv_reshaped[:, m, m, 1, 0] * A_reshaped[:, m, n, 0] +
                   M_inv_reshaped[:, m, m, 1, 1] * A_reshaped[:, m, n, 1])
            
            mask1 = np.abs(va1) > 1e-18
            if np.any(mask1):
                k_idx = np.where(mask1)[0]
                para_rows1.append(k_idx * N_bands + m)
                para_cols1.append(k_idx * N_bands + n)
                para_data1.append(va1[mask1])
            
            mask2 = np.abs(va2) > 1e-18
            if np.any(mask2):
                k_idx = np.where(mask2)[0]
                para_rows2.append(k_idx * N_bands + m)
                para_cols2.append(k_idx * N_bands + n)
                para_data2.append(va2[mask2])
    
    if para_rows1:
        VA1_op = csr_matrix((np.concatenate(para_data1),
                             (np.concatenate(para_rows1), np.concatenate(para_cols1))),
                            shape=(N_total, N_total))
    else:
        VA1_op = csr_matrix((N_total, N_total), dtype=complex)
    
    if para_rows2:
        VA2_op = csr_matrix((np.concatenate(para_data2),
                             (np.concatenate(para_rows2), np.concatenate(para_cols2))),
                            shape=(N_total, N_total))
    else:
        VA2_op = csr_matrix((N_total, N_total), dtype=complex)
    
    # Paramagnetic term: -i * prefactor * (VA1 @ D1 + VA2 @ D2) + h.c.
    para_op = -1j * prefactor * (VA1_op @ D1_full + VA2_op @ D2_full)
    K_op = K_op + para_op
    
    # ── Hermitize ─────────────────────────────────────────────────────────
    K_op = (K_op + K_op.T.conj()) / 2
    
    return K_op


def compute_band_weights(eigvec, Ns, Nb):
    """Fractional weight in each subspace band."""
    n_modes = eigvec.shape[1]
    weights = np.zeros((n_modes, Nb))
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        for n in range(Nb):
            weights[m, n] = np.sum(np.abs(F[:, :, n])**2)
        weights[m] /= np.sum(weights[m])
    return weights


def compute_ipr(eigvec, Ns, Nb):
    n_modes = eigvec.shape[1]
    ipr = np.zeros(n_modes)
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        rho = np.sum(np.abs(F)**2, axis=2)
        ipr[m] = np.sum(rho**2) / np.sum(rho)**2
    return ipr


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  S6: TWIST ANGLE SWEEP + OFF-DIAGONAL COUPLING")
    print("=" * 70)
    
    # ── [1] Load data ────────────────────────────────────────────────────
    print(f"\n[1] Loading C4-symmetrized Phase 2 data...")
    d = load_phase2(H5_SYM)
    Ns, Nb = d['Ns'], d['Nb']
    eta_orig = d['eta']
    target_idx = d['target_idx']
    omega_ref = d['omega_ref']
    B_moire = d['B_moire']
    Lambda = d['Lambda']
    A_berry = d['A_berry']
    Phi_BH = d['Phi_BH']
    v_drift = d['v_drift']
    M_inv = d['M_inv']
    
    print(f"  Grid: {Ns}×{Ns}, N_bands={Nb}, η_orig={eta_orig:.6f}")
    
    # ══════════════════════════════════════════════════════════════════════
    # PART A: Off-diagonal coupling survey
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PART A: OFF-DIAGONAL COUPLING SURVEY")
    print(f"{'='*70}")
    
    print(f"\n[A1] Off-diagonal magnitudes in Phase 2 data:")
    
    # Lambda
    Lambda_diag_norm = np.sqrt(np.sum([np.sum(np.abs(Lambda[:,:,n,n])**2) for n in range(Nb)]))
    Lambda_offdiag_norm = 0.0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                Lambda_offdiag_norm += np.sum(np.abs(Lambda[:,:,m,n])**2)
    Lambda_offdiag_norm = np.sqrt(Lambda_offdiag_norm)
    Lambda_ratio = Lambda_offdiag_norm / Lambda_diag_norm if Lambda_diag_norm > 0 else 0
    print(f"  Λ diagonal ||:     {Lambda_diag_norm:.6e}")
    print(f"  Λ off-diagonal ||: {Lambda_offdiag_norm:.6e}  (ratio: {Lambda_ratio:.4e})")
    
    # A_berry
    A_diag_norm = np.sqrt(np.sum([np.sum(np.abs(A_berry[:,:,n,n,:])**2) for n in range(Nb)]))
    A_offdiag_norm = 0.0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                A_offdiag_norm += np.sum(np.abs(A_berry[:,:,m,n,:])**2)
    A_offdiag_norm = np.sqrt(A_offdiag_norm)
    A_ratio = A_offdiag_norm / A_diag_norm if A_diag_norm > 0 else 0
    print(f"  A diagonal ||:     {A_diag_norm:.6e}")
    print(f"  A off-diagonal ||: {A_offdiag_norm:.6e}  (ratio: {A_ratio:.4e})")
    
    # Phi_BH
    Phi_diag_norm = np.sqrt(np.sum([np.sum(np.abs(Phi_BH[:,:,n,n])**2) for n in range(Nb)]))
    Phi_offdiag_norm = 0.0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                Phi_offdiag_norm += np.sum(np.abs(Phi_BH[:,:,m,n])**2)
    Phi_offdiag_norm = np.sqrt(Phi_offdiag_norm)
    Phi_ratio = Phi_offdiag_norm / Phi_diag_norm if Phi_diag_norm > 0 else 0
    print(f"  Φ_BH diagonal ||:     {Phi_diag_norm:.6e}")
    print(f"  Φ_BH off-diagonal ||: {Phi_offdiag_norm:.6e}  (ratio: {Phi_ratio:.4e})")
    
    # v_drift
    v_diag_norm = np.sqrt(np.sum([np.sum(np.abs(v_drift[:,:,n,n,:])**2) for n in range(Nb)]))
    v_offdiag_norm = 0.0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                v_offdiag_norm += np.sum(np.abs(v_drift[:,:,m,n,:])**2)
    v_offdiag_norm = np.sqrt(v_offdiag_norm)
    v_ratio = v_offdiag_norm / v_diag_norm if v_diag_norm > 0 else 0
    print(f"  v_drift diagonal ||:     {v_diag_norm:.6e}")
    print(f"  v_drift off-diagonal ||: {v_offdiag_norm:.6e}  (ratio: {v_ratio:.4e})")
    
    # M_inv
    M_diag_norm = np.sqrt(np.sum([np.sum(np.abs(M_inv[:,:,n,n,:,:])**2) for n in range(Nb)]))
    M_offdiag_norm = 0.0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                M_offdiag_norm += np.sum(np.abs(M_inv[:,:,m,n,:,:])**2)
    M_offdiag_norm = np.sqrt(M_offdiag_norm)
    M_ratio = M_offdiag_norm / M_diag_norm if M_diag_norm > 0 else 0
    print(f"  M_inv diagonal ||:     {M_diag_norm:.6e}")
    print(f"  M_inv off-diagonal ||: {M_offdiag_norm:.6e}  (ratio: {M_ratio:.4e})")
    
    # Per-pair off-diagonal A_berry
    print(f"\n  Off-diagonal A_berry per band pair (max |A_{m,n}|):")
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                amax = np.max(np.abs(A_berry[:,:,m,n,:]))
                amean = np.mean(np.abs(A_berry[:,:,m,n,:]))
                print(f"    A[{m},{n}]: max={amax:.4e}  mean={amean:.4e}")
    
    # ── [A2] Phase 3 coupling analysis ───────────────────────────────────
    print(f"\n[A2] How Phase 3 uses these terms:")
    print(f"  Potential Λ:   iterates over ALL (m,n) ← would use off-diag if ≠ 0")
    print(f"  Drift v:       iterates over ALL (m,n) ← would use off-diag if ≠ 0")
    print(f"  Born-Huang Φ:  iterates over ALL (m,n) ← would use off-diag if ≠ 0")
    print(f"  Kinetic K:     uses ONLY diagonal M[n,n] and A[n,n]  ← IGNORES off-diag!")
    print(f"")
    print(f"  ⇒ Only A_berry has off-diagonal data, but the kinetic operator drops it.")
    print(f"  ⇒ To enable interband coupling, we need a CORRECTED kinetic operator")
    print(f"     that uses off-diagonal A in the covariant derivative.")
    
    # ── [A3] Build H with off-diagonal A (corrected kinetic) ─────────────
    print(f"\n[A3] Building H with FULL off-diagonal Berry connection...")
    print(f"  Using corrected kinetic operator with covariant derivative...")
    
    eta = eta_orig
    L_moire = 1.0 / eta
    dR = L_moire / Ns
    
    # Regularize M_inv
    M_inv_reg = _regularize_M_inv(M_inv.copy(), 20.0)
    
    # Standard H (diagonal A only — current Phase 3 behavior)
    print(f"\n  --- Standard H (Phase 3: diagonal A only) ---")
    H_std = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, A_berry, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
    )
    H_std = 0.5 * (H_std + H_std.conj().T)
    
    V_target = Lambda[:, :, target_idx, target_idx].real
    M_target = M_inv[:, :, target_idx, target_idx, :, :]
    mean_trace = np.mean(M_target[:, :, 0, 0] + M_target[:, :, 1, 1])
    sigma = float(np.max(V_target)) if mean_trace < 0 else float(np.min(V_target))
    
    n_modes = 12
    print(f"  Solving {n_modes} modes (sigma={sigma:.6f})...")
    ev_std, evec_std = eigsh(H_std, k=n_modes, sigma=sigma, which='LM',
                              maxiter=10000, tol=1e-10)
    order = np.argsort(ev_std)
    ev_std = ev_std[order]
    evec_std = evec_std[:, order]
    
    w_std = compute_band_weights(evec_std, Ns, Nb)
    max_mixing_std = 1.0 - np.max(w_std, axis=1)
    
    # Corrected H (full off-diagonal A in covariant derivative)
    print(f"\n  --- Corrected H (full off-diagonal A) ---")
    print(f"  Building corrected kinetic operator (may take a moment)...")
    
    # Build the full H manually: V + drift + K_corrected
    V_op = build_multiband_potential_operator(Lambda, B_moire)
    T_op = build_multiband_drift_operator(v_drift, eta, Ns, Ns, Nb, dR, dR)
    K_corr = build_kinetic_with_offdiag_A(M_inv_reg, A_berry, eta, Ns, Ns, Nb, dR, dR, B_moire)
    
    H_corr = V_op + T_op + K_corr
    H_corr = 0.5 * (H_corr + H_corr.conj().T)
    
    print(f"  H_corr nnz = {H_corr.nnz} (vs standard {H_std.nnz})")
    print(f"  Solving {n_modes} modes...")
    ev_corr, evec_corr = eigsh(H_corr, k=n_modes, sigma=sigma, which='LM',
                                maxiter=10000, tol=1e-10)
    order = np.argsort(ev_corr)
    ev_corr = ev_corr[order]
    evec_corr = evec_corr[:, order]
    
    w_corr = compute_band_weights(evec_corr, Ns, Nb)
    max_mixing_corr = 1.0 - np.max(w_corr, axis=1)
    
    # Also build A=0 reference
    print(f"\n  --- A=0 reference ---")
    H_a0 = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, np.zeros_like(A_berry), np.zeros_like(Phi_BH),
        eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
    )
    H_a0 = 0.5 * (H_a0 + H_a0.conj().T)
    ev_a0, evec_a0 = eigsh(H_a0, k=n_modes, sigma=sigma, which='LM',
                            maxiter=10000, tol=1e-10)
    order = np.argsort(ev_a0)
    ev_a0 = ev_a0[order]
    evec_a0 = evec_a0[:, order]
    w_a0 = compute_band_weights(evec_a0, Ns, Nb)
    max_mixing_a0 = 1.0 - np.max(w_a0, axis=1)
    
    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n[A4] Comparison: band mixing across configurations")
    print(f"  {'Mode':>4s}  {'A=0 mix':>8s}  {'Std mix':>8s}  {'Corr mix':>9s}  "
          f"{'A=0 dom':>7s}  {'Std dom':>7s}  {'Corr dom':>8s}")
    print(f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'---------':>9s}  "
          f"{'-------':>7s}  {'-------':>7s}  {'--------':>8s}")
    for i in range(n_modes):
        dom_a0 = np.argmax(w_a0[i])
        dom_std = np.argmax(w_std[i])
        dom_corr = np.argmax(w_corr[i])
        print(f"  {i:>4d}  {max_mixing_a0[i]:>8.4f}  {max_mixing_std[i]:>8.4f}  "
              f"{max_mixing_corr[i]:>9.4f}  "
              f"B{dom_a0:>5d}  B{dom_std:>5d}  B{dom_corr:>6d}")
    
    print(f"\n  Eigenvalue comparison:")
    print(f"  {'Mode':>4s}  {'A=0':>14s}  {'Std(diag A)':>14s}  {'Corr(full A)':>14s}  {'Δ(corr-std)':>12s}")
    print(f"  {'----':>4s}  {'-'*14:>14s}  {'-'*14:>14s}  {'-'*14:>14s}  {'-'*12:>12s}")
    for i in range(n_modes):
        diff = ev_corr[i] - ev_std[i]
        print(f"  {i:>4d}  {ev_a0[i]:>+14.8e}  {ev_std[i]:>+14.8e}  {ev_corr[i]:>+14.8e}  {diff:>+12.4e}")
    
    mean_mix_std = np.mean(max_mixing_std)
    mean_mix_corr = np.mean(max_mixing_corr)
    print(f"\n  Mean mixing (1 - max_weight):")
    print(f"    A=0:            {np.mean(max_mixing_a0):.6f}")
    print(f"    Standard:       {mean_mix_std:.6f}")
    print(f"    Corrected:      {mean_mix_corr:.6f}")
    print(f"    Improvement:    {mean_mix_corr/mean_mix_std:.2f}× " 
          f"{'(MORE mixing)' if mean_mix_corr > mean_mix_std else '(less mixing)'}"
          if mean_mix_std > 1e-6 else "    (both effectively zero)")
    
    # ══════════════════════════════════════════════════════════════════════
    # PART B: Twist angle (η) sweep
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  PART B: TWIST ANGLE SWEEP")
    print(f"{'='*70}")
    
    # Use Ns=64 for speed
    Ns_sweep = 64
    factor = Ns // Ns_sweep
    Lambda_ds = downsample_field(Lambda, factor)
    M_inv_ds = downsample_field(M_inv, factor)
    v_drift_ds = downsample_field(v_drift, factor)
    A_berry_ds = np.zeros((Ns_sweep, Ns_sweep, Nb, Nb, 2))  # A=0 for clean comparison
    Phi_BH_ds = np.zeros((Ns_sweep, Ns_sweep, Nb, Nb))
    
    V_target_ds = Lambda_ds[:, :, target_idx, target_idx].real
    M_eff = np.mean(0.5 * (M_inv_ds[:, :, target_idx, target_idx, 0, 0] +
                           M_inv_ds[:, :, target_idx, target_idx, 1, 1]))
    V_depth = np.max(V_target_ds) - np.min(V_target_ds)
    
    # θ-values to sweep
    theta_deg = np.array([1.1, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0])
    eta_vals = 2 * np.sin(np.deg2rad(theta_deg) / 2)
    
    print(f"\n[B1] Energy scale predictions:")
    print(f"  V_depth = {V_depth:.6f}")
    print(f"  M_eff = {M_eff:.4f} ({'hole' if M_eff < 0 else 'electron'})")
    print(f"")
    print(f"  {'θ (°)':>6s}  {'η':>8s}  {'L_moire':>8s}  {'E_kin':>10s}  "
          f"{'V/E_kin':>8s}  {'N_bound':>7s}  {'δε_est':>10s}")
    print(f"  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}  "
          f"{'--------':>8s}  {'-------':>7s}  {'----------':>10s}")
    for theta, eta in zip(theta_deg, eta_vals):
        L = 1.0 / eta
        E_kin = 0.5 * abs(M_eff) * eta**2
        ratio = V_depth / E_kin
        N_bound = V_depth * abs(M_eff) * L**2 / (2 * np.pi)
        delta_eps = V_depth / N_bound if N_bound > 0 else 0
        print(f"  {theta:>6.1f}  {eta:>8.4f}  {L:>8.2f}  {E_kin:>10.4e}  "
              f"{ratio:>8.1f}  {N_bound:>7.0f}  {delta_eps:>10.4e}")
    
    # Solve at each θ
    print(f"\n[B2] Eigensolve sweep (Ns={Ns_sweep}, A=0)...")
    n_sweep = 12
    sweep_results = {}
    
    for theta, eta in zip(theta_deg, eta_vals):
        L = 1.0 / eta
        dR_sweep = L / Ns_sweep
        
        # Regularize M_inv
        M_inv_reg_ds = _regularize_M_inv(M_inv_ds.copy(), 20.0)
        
        # Build H
        H = assemble_multiband_hamiltonian(
            Lambda_ds, v_drift_ds, M_inv_reg_ds, A_berry_ds, Phi_BH_ds,
            eta, Ns_sweep, Ns_sweep, Nb, dR_sweep, dR_sweep, B_moire,
            include_drift=True, include_kinetic=True, include_born_huang=False,
        )
        H = 0.5 * (H + H.conj().T)
        
        # Sigma
        sigma_s = float(np.max(V_target_ds)) if M_eff < 0 else float(np.min(V_target_ds))
        
        try:
            ev, evec = eigsh(H, k=n_sweep, sigma=sigma_s, which='LM',
                              maxiter=10000, tol=1e-10)
            order = np.argsort(ev)
            ev = ev[order]
            evec = evec[:, order]
            
            ipr = compute_ipr(evec, Ns_sweep, Nb)
            weights = compute_band_weights(evec, Ns_sweep, Nb)
            
            sweep_results[theta] = {
                'ev': ev, 'evec': evec, 'ipr': ipr, 'weights': weights,
                'eta': eta, 'sigma': sigma_s,
            }
            
            spread = ev[-1] - ev[0]
            ipr_ext = 1.0 / (Ns_sweep * Ns_sweep)
            n_loc = np.sum(ipr > 10 * ipr_ext)
            max_mix = np.max(1.0 - np.max(weights, axis=1))
            
            print(f"  θ={theta:5.1f}°: spread={spread:.4e}  "
                  f"loc={n_loc}/{n_sweep}  max_mix={max_mix:.4f}  "
                  f"ε₀={ev[0]:+.6e}")
        except Exception as e:
            print(f"  θ={theta:5.1f}°: ERROR: {e}")
            sweep_results[theta] = {'error': str(e)}
    
    # ── [B3] Detailed analysis at interesting θ ──────────────────────────
    # Find the θ with best eigenvalue spacing relative to spread
    print(f"\n[B3] Eigenvalue spectra comparison:")
    print(f"  {'Mode':>4s}", end="")
    for theta in theta_deg:
        print(f"  {'θ='+str(theta)+'°':>12s}", end="")
    print()
    print(f"  {'----':>4s}" + "  " + "-"*12 * len(theta_deg))
    
    for i in range(min(10, n_sweep)):
        row = f"  {i:>4d}"
        for theta in theta_deg:
            if theta in sweep_results and 'ev' in sweep_results[theta]:
                ev = sweep_results[theta]['ev']
                if i < len(ev):
                    row += f"  {ev[i] - np.max(V_target_ds):>+12.4e}"
                else:
                    row += f"  {'—':>12s}"
            else:
                row += f"  {'ERR':>12s}"
        print(row)
    
    # ── [B4] Summary metrics per θ ────────────────────────────────────────
    print(f"\n[B4] Summary metrics:")
    print(f"  {'θ (°)':>6s}  {'E_kin':>10s}  {'V/E_kin':>8s}  {'Spread':>10s}  "
          f"{'δε_mean':>10s}  {'Loc/12':>6s}  {'Max mix':>8s}")
    print(f"  {'------':>6s}  {'----------':>10s}  {'--------':>8s}  {'----------':>10s}  "
          f"{'----------':>10s}  {'------':>6s}  {'--------':>8s}")
    for theta in theta_deg:
        eta = sweep_results.get(theta, {}).get('eta', 0)
        if theta in sweep_results and 'ev' in sweep_results[theta]:
            ev = sweep_results[theta]['ev']
            ipr = sweep_results[theta]['ipr']
            weights = sweep_results[theta]['weights']
            E_kin = 0.5 * abs(M_eff) * eta**2
            ratio = V_depth / E_kin
            spread = ev[-1] - ev[0]
            spacing = np.mean(np.diff(ev))
            ipr_ext = 1.0 / (Ns_sweep * Ns_sweep)
            n_loc = np.sum(ipr > 10 * ipr_ext)
            max_mix = np.max(1.0 - np.max(weights, axis=1))
            print(f"  {theta:>6.1f}  {E_kin:>10.4e}  {ratio:>8.1f}  {spread:>10.4e}  "
                  f"{spacing:>10.4e}  {n_loc:>6d}  {max_mix:>8.4f}")
        else:
            print(f"  {theta:>6.1f}  {'ERROR':>10s}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Plots
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[C] Generating plots...")
    
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("S6: Twist Angle Sweep + Off-diagonal Coupling", fontsize=14, fontweight='bold')
    
    # Panel 1: V/E_kin ratio vs θ
    ax1 = fig.add_subplot(2, 3, 1)
    theta_ok = [t for t in theta_deg if t in sweep_results and 'ev' in sweep_results[t]]
    ratios = [V_depth / (0.5 * abs(M_eff) * sweep_results[t]['eta']**2) for t in theta_ok]
    ax1.semilogy(theta_ok, ratios, 'o-', color='steelblue', markersize=8)
    ax1.axhspan(1, 10, alpha=0.2, color='green', label='Optimal (1-10)')
    ax1.axhspan(10, 100, alpha=0.1, color='yellow', label='Acceptable (10-100)')
    ax1.set_xlabel('θ (degrees)')
    ax1.set_ylabel('V_depth / E_kin')
    ax1.set_title('Energy Scale Ratio vs Twist Angle')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Eigenvalue spread vs θ
    ax2 = fig.add_subplot(2, 3, 2)
    spreads = []
    spacings = []
    for t in theta_ok:
        ev = sweep_results[t]['ev']
        spreads.append(ev[-1] - ev[0])
        spacings.append(np.mean(np.diff(ev)))
    ax2.semilogy(theta_ok, spreads, 'o-', color='darkorange', label='Spread (ε_max - ε_min)')
    ax2.semilogy(theta_ok, spacings, 's-', color='forestgreen', label='Mean spacing')
    ax2.set_xlabel('θ (degrees)')
    ax2.set_ylabel('Energy')
    ax2.set_title('Eigenvalue Spread vs θ')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Eigenvalue spectra at selected θ
    ax3 = fig.add_subplot(2, 3, 3)
    V_max_ds = np.max(V_target_ds)
    for t in theta_ok:
        ev = sweep_results[t]['ev']
        y = ev - V_max_ds
        ax3.plot(range(len(ev)), y, 'o-', markersize=4, label=f'θ={t}°')
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5, label='V_max')
    ax3.set_xlabel('Mode index')
    ax3.set_ylabel('ε − V_max')
    ax3.set_title('Eigenvalue Spectra')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Band mixing comparison (Part A)
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(n_modes)
    width = 0.3
    ax4.bar(x - width, max_mixing_a0, width, label='A=0', color='gray', alpha=0.7)
    ax4.bar(x, max_mixing_std, width, label='Std (diag A)', color='steelblue', alpha=0.7)
    ax4.bar(x + width, max_mixing_corr, width, label='Corr (full A)', color='darkorange', alpha=0.7)
    ax4.set_xlabel('Mode index')
    ax4.set_ylabel('Band mixing (1 - max_weight)')
    ax4.set_title('Interband Mixing (θ=1.1°, Ns=128)')
    ax4.legend(fontsize=8)
    
    # Panel 5: IPR vs θ  
    ax5 = fig.add_subplot(2, 3, 5)
    for t in theta_ok:
        ipr = sweep_results[t]['ipr']
        ax5.semilogy(range(len(ipr)), ipr, 'o-', markersize=4, label=f'θ={t}°')
    ipr_ext = 1.0 / (Ns_sweep**2)
    ax5.axhline(ipr_ext, color='gray', linestyle='--', label='extended')
    ax5.set_xlabel('Mode index')
    ax5.set_ylabel('IPR')
    ax5.set_title('Localization vs θ')
    ax5.legend(fontsize=7, ncol=2)
    
    # Panel 6: Off-diagonal coupling magnitude summary
    ax6 = fig.add_subplot(2, 3, 6)
    quantities = ['Λ', 'A_berry', 'Φ_BH', 'v_drift', 'M_inv']
    diag_norms = [Lambda_diag_norm, A_diag_norm, Phi_diag_norm, v_diag_norm, M_diag_norm]
    offdiag_norms = [Lambda_offdiag_norm, A_offdiag_norm, Phi_offdiag_norm, v_offdiag_norm, M_offdiag_norm]
    
    x_pos = np.arange(len(quantities))
    bars1 = ax6.bar(x_pos - 0.2, [max(d, 1e-20) for d in diag_norms], 0.35, 
                     label='Diagonal', color='steelblue')
    bars2 = ax6.bar(x_pos + 0.2, [max(d, 1e-20) for d in offdiag_norms], 0.35,
                     label='Off-diagonal', color='darkorange')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(quantities)
    ax6.set_yscale('log')
    ax6.set_ylabel('||·||_F')
    ax6.set_title('Off-diagonal Coupling Magnitudes')
    ax6.legend(fontsize=8)
    
    plt.tight_layout()
    plot_path = PLOT_DIR / "S6_eta_sweep_coupling.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {plot_path.name}")
    plt.close()
    
    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  S6 SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  Part A — Off-diagonal coupling:")
    print(f"    Only A_berry has off-diagonal data (ratio = {A_ratio:.4f})")
    print(f"    All other quantities are band-diagonal")
    print(f"    Phase 3 kinetic operator IGNORES off-diagonal A")
    print(f"    Corrected kinetic op mean mixing = {mean_mix_corr:.6f}")
    print(f"    Standard kinetic op mean mixing  = {mean_mix_std:.6f}")
    if mean_mix_corr > 1e-4:
        print(f"    → Off-diagonal A provides {'significant' if mean_mix_corr > 0.01 else 'modest'} interband coupling")
    else:
        print(f"    → Off-diagonal A provides NEGLIGIBLE interband coupling")
        print(f"    → Root cause: V/E_kin = {V_depth/(0.5*abs(M_eff)*eta_orig**2):.0f} too large")
        print(f"       Potential dominates → modes are band-pure regardless of coupling")
    
    print(f"\n  Part B — Twist angle sweep:")
    print(f"    V_depth = {V_depth:.6f}, M_eff = {M_eff:.4f}")
    
    # Find optimal θ
    best_theta = None
    best_ratio = float('inf')
    for t in theta_ok:
        eta_t = sweep_results[t]['eta']
        r = V_depth / (0.5 * abs(M_eff) * eta_t**2)
        if abs(r - 10) < abs(best_ratio - 10):
            best_ratio = r
            best_theta = t
    
    if best_theta:
        print(f"    Closest to V/E_kin = 10: θ = {best_theta}° (ratio = {best_ratio:.1f})")
        print(f"    For V/E_kin = 10: need η = {np.sqrt(V_depth/(5*abs(M_eff))):.4f} "
              f"→ θ = {np.degrees(2*np.arcsin(np.sqrt(V_depth/(5*abs(M_eff)))/2)):.1f}°")
    
    print(f"\n    θ for V/E_kin = 1:   "
          f"η = {np.sqrt(V_depth/(0.5*abs(M_eff))):.4f} → "
          f"θ = {np.degrees(2*np.arcsin(np.sqrt(V_depth/(0.5*abs(M_eff)))/2)):.1f}°")
    print(f"    θ for V/E_kin = 100: "
          f"η = {np.sqrt(V_depth/(50*abs(M_eff))):.4f} → "
          f"θ = {np.degrees(2*np.arcsin(np.sqrt(V_depth/(50*abs(M_eff)))/2)):.1f}°")
    
    print(f"\n    ⚠  The EA requires η ≪ 1 (θ ≲ 5-10°).")
    print(f"    ⚠  Optimal V/E_kin requires θ ≈ {np.degrees(2*np.arcsin(np.sqrt(V_depth/(5*abs(M_eff)))/2)):.0f}° → EA may be invalid.")
    print(f"    → This candidate is PATHOLOGICAL for the EA: the only way to get")
    print(f"      reasonable V/E_kin requires θ beyond the EA validity range.")
    print(f"    → A different candidate with LARGER |M_eff| or SMALLER V_depth is needed.")
    
    print(f"\n  Plots saved to {PLOT_DIR}/")
    print(f"\n{'='*70}")
    print(f"  S6 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
