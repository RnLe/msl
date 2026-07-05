#!/usr/bin/env python3
"""
S4b: C4-symmetrize all Phase 2 operator data and rebuild Hamiltonian.

Phase A of the post-S4 correction plan.

Strategy:
  1. Load raw Phase 2 data (Lambda, A_berry, M_inv, Phi_BH, v_drift, omega)
  2. C4-symmetrize each quantity using tensor-rank-dependent transformation:
       Q_sym(R) = (1/4) Σ_{n=0}^{3} T_{C4^n}[ Q(C4^{-n} R) ]
     - Scalars (Lambda, Phi_BH, omega): direct average
     - Vectors (A_berry, v_drift):      rotate vector components
     - 2-tensors (M_inv):               rotate tensor components
  3. Report before/after C4 error metrics
  4. Rebuild Hamiltonian from symmetrized data
  5. Verify [H, C4] = 0 (should be machine precision)
  6. Solve eigenmodes and classify by C4 irreps (1, i, -1, -i)
  7. Compare to unsymmetrized results

Also runs Phase B test: A_berry=0 (since paramagnetic terms are missing anyway,
setting A=0 is more self-consistent than keeping a gauge-broken A).

C4 grid convention (square lattice, C4 about center):
  Forward C4:  (ix, iy) → ((Ns - iy) % Ns, ix)
  Inverse C4:  (ix, iy) → (iy, (Ns - ix) % Ns)

C4 rotation matrix (2D):
  R   = [[ 0, -1],     R^2 = [[-1,  0],     R^3 = [[ 0,  1],
         [ 1,  0]]            [ 0, -1]]            [-1,  0]]

On vectors: (Qx, Qy) → (-Qy, Qx) → (-Qx, -Qy) → (Qy, -Qx)
"""

import numpy as np
import h5py
import sys
import os
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add phasesV3 to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phasesV3"))
from phase3_mpb_v3 import (
    build_multiband_potential_operator,
    build_multiband_drift_operator,
    build_multiband_kinetic_operator,
    build_multiband_born_huang_operator,
    assemble_multiband_hamiltonian,
    _regularize_M_inv,
)

# ====================================================================================
# Paths
# ====================================================================================
CAND = Path("/home/renlephy/msl/research/moire_envelope/runsV3/"
            "phase0_mpb_v3_20260206_152443/candidate_0000")
PHASE2_H5 = CAND / "phase2_multiband_data.h5"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ====================================================================================
# C4 grid operations
# ====================================================================================

def apply_C4_inv_grid(Q, Ns):
    """
    Pull-back by C4^{-1}: Q_pulled[ix, iy, ...] = Q[iy, (Ns-ix)%Ns, ...].
    
    This computes Q(C4^{-1} R) — the value at the inverse-rotated point.
    Works for any trailing dimensions.
    """
    # C4^{-1}(ix, iy) = (iy, (Ns-ix)%Ns)
    ix_src = np.arange(Ns)[:, None]  # (Ns, 1)
    iy_src = np.arange(Ns)[None, :]  # (1, Ns)
    
    # Source indices for C4^{-1}
    ix_from = iy_src.ravel()            # iy → new ix
    iy_from = (Ns - ix_src) % Ns        # (Ns-ix)%Ns → new iy
    iy_from = iy_from.ravel()
    
    # Build full index arrays matching the 2D grid unrolled
    # We need Q[iy, (Ns-ix)%Ns, ...] for every (ix, iy)
    Q_pulled = np.empty_like(Q)
    for ix in range(Ns):
        for iy in range(Ns):
            Q_pulled[ix, iy] = Q[iy, (Ns - ix) % Ns]
    return Q_pulled


def apply_C4_inv_n_grid(Q, n_rot, Ns):
    """
    Pull-back by C4^{-n}: apply C4^{-1} n times.
    Q_pulled = Q(C4^{-n} R).
    """
    result = Q.copy()
    for _ in range(n_rot % 4):
        result = apply_C4_inv_grid(result, Ns)
    return result


# ====================================================================================
# C4 rotation matrices for vectors and tensors
# ====================================================================================

# C4^n rotation matrices (2D)
C4_MATS = [
    np.eye(2),                             # C4^0 = I
    np.array([[0., -1.], [1., 0.]]),        # C4^1
    np.array([[-1., 0.], [0., -1.]]),       # C4^2
    np.array([[0., 1.], [-1., 0.]]),        # C4^3
]


# ====================================================================================
# Symmetrization functions
# ====================================================================================

def symmetrize_scalar(Q, Ns):
    """
    C4-symmetrize a scalar field Q(R) with shape (Ns, Ns, ...).
    
    Q_sym(R) = (1/4) Σ_n Q(C4^{-n} R)
    """
    result = np.zeros_like(Q)
    for n_rot in range(4):
        result += apply_C4_inv_n_grid(Q, n_rot, Ns)
    return result / 4.0


def symmetrize_vector(Q, Ns):
    """
    C4-symmetrize a vector field Q(R) with shape (Ns, Ns, ..., 2).
    Last axis is the 2D vector component.
    
    Q_sym_i(R) = (1/4) Σ_n (C4^n)_{ij} Q_j(C4^{-n} R)
    """
    result = np.zeros_like(Q)
    for n_rot in range(4):
        # Pull-back: Q at C4^{-n}R
        Q_pulled = apply_C4_inv_n_grid(Q, n_rot, Ns)
        # Rotate vector components by C4^n
        R_mat = C4_MATS[n_rot]  # (2, 2)
        # Q_pulled[..., j] → R_mat[i, j] * Q_pulled[..., j]
        # result[..., i] += Σ_j R_mat[i,j] * Q_pulled[..., j]
        for i in range(2):
            for j in range(2):
                result[..., i] += R_mat[i, j] * Q_pulled[..., j]
    return result / 4.0


def symmetrize_2tensor(Q, Ns):
    """
    C4-symmetrize a 2-tensor field Q(R) with shape (Ns, Ns, ..., 2, 2).
    Last two axes are the 2D tensor components.
    
    Q_sym_{ij}(R) = (1/4) Σ_n (C4^n)_{ik} (C4^n)_{jl} Q_{kl}(C4^{-n} R)
    
    i.e., M → R @ M @ R^T
    """
    result = np.zeros_like(Q)
    for n_rot in range(4):
        Q_pulled = apply_C4_inv_n_grid(Q, n_rot, Ns)
        R_mat = C4_MATS[n_rot]  # (2, 2)
        # R @ M @ R^T for each grid point
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        result[..., i, j] += R_mat[i, k] * R_mat[j, l] * Q_pulled[..., k, l]
    return result / 4.0


# ====================================================================================
# C4 error measurement (from S4)
# ====================================================================================

def measure_C4_error_scalar(Q, Ns, label=""):
    """Measure C4 error for a scalar field: ||Q(R) - Q(C4^{-1}R)|| / ||Q||."""
    Q_rot = apply_C4_inv_grid(Q, Ns)
    err = np.linalg.norm(Q_rot - Q) / (np.linalg.norm(Q) + 1e-30)
    return err


def measure_C4_error_vector(Q, Ns, label=""):
    """Measure C4 error for a vector field: ||Q(R) - R_C4 Q(C4^{-1}R)|| / ||Q||."""
    Q_pulled = apply_C4_inv_grid(Q, Ns)
    R_mat = C4_MATS[1]  # C4^1
    Q_expected = np.zeros_like(Q)
    for i in range(2):
        for j in range(2):
            Q_expected[..., i] += R_mat[i, j] * Q_pulled[..., j]
    err = np.linalg.norm(Q_expected - Q) / (np.linalg.norm(Q) + 1e-30)
    return err


def measure_C4_error_2tensor(Q, Ns, label=""):
    """Measure C4 error for 2-tensor: ||Q(R) - R M(C4^{-1}R) R^T|| / ||Q||."""
    Q_pulled = apply_C4_inv_grid(Q, Ns)
    R_mat = C4_MATS[1]
    Q_expected = np.zeros_like(Q)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    Q_expected[..., i, j] += R_mat[i, k] * R_mat[j, l] * Q_pulled[..., k, l]
    err = np.linalg.norm(Q_expected - Q) / (np.linalg.norm(Q) + 1e-30)
    return err


# ====================================================================================
# C4 permutation matrix and commutator (from S4)
# ====================================================================================

def build_C4_permutation_matrix(Ns, Nb):
    """Build sparse C4 permutation in flat index space."""
    N_total = Ns * Ns * Nb
    rows, cols = [], []
    for ix in range(Ns):
        for iy in range(Ns):
            for n in range(Nb):
                idx_target = (ix * Ns + iy) * Nb + n
                ix_src = iy
                iy_src = (Ns - ix) % Ns
                idx_source = (ix_src * Ns + iy_src) * Nb + n
                rows.append(idx_target)
                cols.append(idx_source)
    data = np.ones(len(rows))
    P = coo_matrix((data, (rows, cols)), shape=(N_total, N_total)).tocsr()
    return P


def check_H_C4_commutator(H, Ns, Nb, label=""):
    """Compute ||[H, C4]||_F / ||H||_F."""
    P = build_C4_permutation_matrix(Ns, Nb)
    HP = H @ P
    PH = P @ H
    commutator = HP - PH
    from scipy.sparse.linalg import norm as sp_norm
    comm_norm = sp_norm(commutator, 'fro')
    H_norm = sp_norm(H, 'fro')
    relative = comm_norm / (H_norm + 1e-30)
    print(f"    [H, C4] test ({label}):")
    print(f"      ||[H, C4]||_F = {comm_norm:.6e}")
    print(f"      ||H||_F       = {H_norm:.6e}")
    print(f"      Relative      = {relative:.6e}")
    return relative


# ====================================================================================
# C4 eigenmode analysis
# ====================================================================================

def apply_C4_flat(v, Ns, Nb):
    """Apply C4 to a flat eigenvector. v has length Ns*Ns*Nb."""
    F = v.reshape(Ns, Ns, Nb)
    F_rot = np.empty_like(F)
    for ix in range(Ns):
        for iy in range(Ns):
            F_rot[ix, iy, :] = F[iy, (Ns - ix) % Ns, :]
    return F_rot.ravel()


def classify_C4_modes(eigenvalues, eigenvectors, Ns, Nb, n_modes=20, degen_tol=1e-6):
    """
    Classify eigenmodes by C4 irreps.
    
    For non-degenerate modes: compute <ψ|C4|ψ> → should be 1, i, -1, or -i.
    For degenerate groups: compute C4 representation matrix and diagonalize.
    
    Returns list of dicts with classification info.
    """
    n_modes = min(n_modes, len(eigenvalues))
    results = []
    used = set()
    
    for i in range(n_modes):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, n_modes):
            if j in used:
                continue
            if abs(eigenvalues[j] - eigenvalues[i]) < degen_tol:
                group.append(j)
                used.add(j)
        
        n_deg = len(group)
        V_sub = eigenvectors[:, group]
        
        # C4 representation matrix within the degenerate subspace
        C4_mat = np.zeros((n_deg, n_deg), dtype=complex)
        for a, idx_a in enumerate(group):
            v_rot = apply_C4_flat(eigenvectors[:, idx_a], Ns, Nb)
            for b, idx_b in enumerate(group):
                C4_mat[a, b] = np.vdot(eigenvectors[:, idx_b], v_rot)
        
        # Subspace closure quality
        closure_errors = []
        for a, idx_a in enumerate(group):
            v_rot = apply_C4_flat(eigenvectors[:, idx_a], Ns, Nb)
            v_proj = V_sub @ (V_sub.conj().T @ v_rot)
            closure = np.linalg.norm(v_proj) / (np.linalg.norm(v_rot) + 1e-30)
            closure_errors.append(closure)
        
        result = {
            'indices': group,
            'eigenvalue': eigenvalues[group[0]],
            'degeneracy': n_deg,
            'c4_closure': min(closure_errors),
        }
        
        if n_deg == 1:
            c4_eig = C4_mat[0, 0]
            expected = [1.0, 1j, -1.0, -1j]
            labels_c4 = ['1 (A)', 'i (E+)', '-1 (B)', '-i (E-)']
            dists = [abs(c4_eig - e) for e in expected]
            best = np.argmin(dists)
            result['c4_eigenvalue'] = c4_eig
            result['c4_irrep'] = labels_c4[best]
            result['c4_error'] = dists[best]
        else:
            # Diagonalize C4 within the degenerate subspace
            c4_eigs = np.linalg.eigvals(C4_mat)
            result['c4_eigenvalues'] = c4_eigs
            result['c4_trace'] = np.trace(C4_mat)
            unitarity = np.linalg.norm(C4_mat @ C4_mat.conj().T - np.eye(n_deg))
            result['c4_unitarity_error'] = unitarity
            # Classify each eigenvalue
            irreps = []
            expected = [1.0, 1j, -1.0, -1j]
            labels_c4 = ['A', 'E+', 'B', 'E-']
            for ev in c4_eigs:
                dists = [abs(ev - e) for e in expected]
                best = np.argmin(dists)
                irreps.append(labels_c4[best])
            result['c4_irreps'] = irreps
        
        results.append(result)
    
    return results


def print_classification(results, label=""):
    """Pretty-print C4 classification results."""
    print(f"\n{'='*70}")
    print(f"  C4 classification: {label}")
    print(f"{'='*70}")
    
    for r in results:
        idx_str = ','.join(str(i) for i in r['indices'])
        ev = r['eigenvalue']
        deg = r['degeneracy']
        closure = r['c4_closure']
        
        status = "✓" if closure > 0.999 else ("~" if closure > 0.99 else "✗")
        
        if deg == 1:
            irrep = r.get('c4_irrep', '?')
            err = r.get('c4_error', 999)
            c4_ev = r.get('c4_eigenvalue', 0)
            print(f"  [{status}] Mode {idx_str:>3s}: ε={ev:+.8e}  "
                  f"closure={closure:.6f}  irrep={irrep}  "
                  f"<ψ|C4|ψ>={c4_ev:.4f}  err={err:.2e}")
        else:
            irreps = r.get('c4_irreps', [])
            irrep_str = '+'.join(irreps) if irreps else '?'
            unit_err = r.get('c4_unitarity_error', 999)
            print(f"  [{status}] Modes {idx_str:>6s}: ε={ev:+.8e}  deg={deg}  "
                  f"closure={closure:.6f}  irreps={irrep_str}  "
                  f"unitarity_err={unit_err:.2e}")
    
    n_pass = sum(1 for r in results if r['c4_closure'] > 0.999)
    n_total = len(results)
    print(f"\n  Summary: {n_pass}/{n_total} mode groups have C4 closure > 0.999")


# ====================================================================================
# Main
# ====================================================================================

def main():
    print("=" * 70)
    print("  S4b: C4-SYMMETRIZE PHASE 2 DATA & REBUILD HAMILTONIAN")
    print("=" * 70)
    
    # ================================================================
    # [1] Load Phase 2 data
    # ================================================================
    print("\n[1] Loading Phase 2 data...")
    with h5py.File(PHASE2_H5, 'r') as hf:
        Lambda = hf['Lambda'][:]              # (Ns, Ns, Nb, Nb)
        A_berry = hf['A_berry'][:]            # (Ns, Ns, Nb, Nb, 2) complex
        Phi_BH = hf['Phi_BH'][:]             # (Ns, Ns, Nb, Nb) complex
        v_drift = hf['v_drift'][:]            # (Ns, Ns, Nb, Nb, 2) float
        M_inv = hf['M_inv'][:]               # (Ns, Ns, Nb, Nb, 2, 2) float
        omega = hf['omega'][:]               # (Ns, Ns, Nb)
        
        omega_ref = float(hf.attrs['omega_ref'])
        eta = float(hf.attrs['eta'])
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        Nb = int(hf.attrs['N_subspace'])
        B_moire = hf.attrs['B_moire'][:]
        target_idx = int(hf.attrs['target_index_in_subspace'])
    
    assert Ns1 == Ns2, "Require square grid"
    Ns = Ns1
    L_moire = np.linalg.norm(B_moire[0])
    dR = L_moire / Ns
    N_total = Ns * Ns * Nb
    
    print(f"  Grid: {Ns}×{Ns}, N_bands={Nb}, N_total={N_total}")
    print(f"  η = {eta:.6f}, L_moire = {L_moire:.4f} a")
    print(f"  dR = {dR:.6f} a")
    print(f"  ω_ref = {omega_ref:.6f}")
    print(f"  Target band index: {target_idx}")
    print(f"  Shapes: Λ={Lambda.shape}, A={A_berry.shape}, M={M_inv.shape}, "
          f"Φ={Phi_BH.shape}, v={v_drift.shape}, ω={omega.shape}")
    
    # ================================================================
    # [2] Measure C4 errors BEFORE symmetrization
    # ================================================================
    print(f"\n[2] C4 errors BEFORE symmetrization:")
    
    # Lambda: scalar, (Ns, Ns, Nb, Nb)
    err_Lambda_before = measure_C4_error_scalar(Lambda, Ns)
    print(f"  Λ (scalar):   {err_Lambda_before:.6e}")
    
    # Phi_BH: scalar, (Ns, Ns, Nb, Nb)
    err_Phi_before = measure_C4_error_scalar(Phi_BH, Ns)
    print(f"  Φ_BH (scalar): {err_Phi_before:.6e}")
    
    # omega: scalar, (Ns, Ns, Nb)
    err_omega_before = measure_C4_error_scalar(omega, Ns)
    print(f"  ω (scalar):   {err_omega_before:.6e}")
    
    # A_berry: vector in last axis, (Ns, Ns, Nb, Nb, 2)
    err_A_before = measure_C4_error_vector(A_berry, Ns)
    print(f"  A_berry (vector): {err_A_before:.6e}")
    
    # v_drift: vector in last axis, (Ns, Ns, Nb, Nb, 2)
    err_v_before = measure_C4_error_vector(v_drift, Ns)
    print(f"  v_drift (vector): {err_v_before:.6e}")
    
    # M_inv: 2-tensor, (Ns, Ns, Nb, Nb, 2, 2)
    err_M_before = measure_C4_error_2tensor(M_inv, Ns)
    print(f"  M_inv (2-tensor): {err_M_before:.6e}")
    
    # Per-band M_inv errors
    print(f"  Per-band M_inv C4 errors:")
    M_inv_band_errs_before = []
    for n in range(Nb):
        M_nn = M_inv[:, :, n, n, :, :]
        err = measure_C4_error_2tensor(M_nn, Ns)
        M_inv_band_errs_before.append(err)
        print(f"    Band {n}: {err:.6e}")
    
    # Per-band A_berry errors
    print(f"  Per-band A_berry C4 errors:")
    A_berry_band_errs_before = []
    for n in range(Nb):
        A_nn = A_berry[:, :, n, n, :]
        err = measure_C4_error_vector(A_nn, Ns)
        A_berry_band_errs_before.append(err)
        print(f"    Band {n}: {err:.6e}")
    
    # ================================================================
    # [3] C4-symmetrize all quantities
    # ================================================================
    print(f"\n[3] C4-symmetrizing all Phase 2 quantities...")
    
    Lambda_sym = symmetrize_scalar(Lambda, Ns)
    print(f"  ✓ Lambda symmetrized")
    
    Phi_BH_sym = symmetrize_scalar(Phi_BH, Ns)
    print(f"  ✓ Phi_BH symmetrized")
    
    omega_sym = symmetrize_scalar(omega, Ns)
    print(f"  ✓ omega symmetrized")
    
    A_berry_sym = symmetrize_vector(A_berry, Ns)
    print(f"  ✓ A_berry symmetrized (vector)")
    
    v_drift_sym = symmetrize_vector(v_drift, Ns)
    print(f"  ✓ v_drift symmetrized (vector)")
    
    M_inv_sym = symmetrize_2tensor(M_inv, Ns)
    print(f"  ✓ M_inv symmetrized (2-tensor)")
    
    # ================================================================
    # [4] Measure C4 errors AFTER symmetrization
    # ================================================================
    print(f"\n[4] C4 errors AFTER symmetrization:")
    
    err_Lambda_after = measure_C4_error_scalar(Lambda_sym, Ns)
    print(f"  Λ (scalar):      {err_Lambda_after:.6e}  (was {err_Lambda_before:.6e})")
    
    err_Phi_after = measure_C4_error_scalar(Phi_BH_sym, Ns)
    print(f"  Φ_BH (scalar):   {err_Phi_after:.6e}  (was {err_Phi_before:.6e})")
    
    err_omega_after = measure_C4_error_scalar(omega_sym, Ns)
    print(f"  ω (scalar):      {err_omega_after:.6e}  (was {err_omega_before:.6e})")
    
    err_A_after = measure_C4_error_vector(A_berry_sym, Ns)
    print(f"  A_berry (vector): {err_A_after:.6e}  (was {err_A_before:.6e})")
    
    err_v_after = measure_C4_error_vector(v_drift_sym, Ns)
    print(f"  v_drift (vector): {err_v_after:.6e}  (was {err_v_before:.6e})")
    
    err_M_after = measure_C4_error_2tensor(M_inv_sym, Ns)
    print(f"  M_inv (2-tensor): {err_M_after:.6e}  (was {err_M_before:.6e})")
    
    # Per-band after
    print(f"  Per-band M_inv C4 errors after:")
    for n in range(Nb):
        M_nn = M_inv_sym[:, :, n, n, :, :]
        err = measure_C4_error_2tensor(M_nn, Ns)
        print(f"    Band {n}: {err:.6e}  (was {M_inv_band_errs_before[n]:.6e})")
    
    print(f"  Per-band A_berry C4 errors after:")
    for n in range(Nb):
        A_nn = A_berry_sym[:, :, n, n, :]
        err = measure_C4_error_vector(A_nn, Ns)
        print(f"    Band {n}: {err:.6e}  (was {A_berry_band_errs_before[n]:.6e})")
    
    # ================================================================
    # [5] Build Hamiltonians: original, symmetrized, symmetrized+A=0
    # ================================================================
    print(f"\n[5] Building Hamiltonians...")
    
    M_inv_max_trace = 20.0
    n_modes_solve = 12  # reduced from 20 for faster eigensolve
    
    # Determine sigma (shift for shift-invert eigensolve)
    V_target = Lambda[:, :, target_idx, target_idx]
    M_target = M_inv[:, :, target_idx, target_idx, :, :]
    mean_trace = np.mean(M_target[:, :, 0, 0] + M_target[:, :, 1, 1])
    if mean_trace < 0:  # hole band → target V_max
        sigma = float(np.max(V_target))
    else:
        sigma = float(np.min(V_target))
    print(f"  Mean M_inv trace: {mean_trace:.4f} ({'hole' if mean_trace < 0 else 'electron'} band)")
    print(f"  sigma = {sigma:.6f}")
    
    configs = [
        {
            'label':    'Original (unsymmetrized)',
            'short':    'Original',
            'Lambda':   Lambda,
            'v_drift':  v_drift,
            'M_inv':    M_inv,
            'A_berry':  A_berry,
            'Phi_BH':   Phi_BH,
        },
        {
            'label':    'C4-symmetrized (full)',
            'short':    'C4-sym',
            'Lambda':   Lambda_sym,
            'v_drift':  v_drift_sym,
            'M_inv':    M_inv_sym,
            'A_berry':  A_berry_sym,
            'Phi_BH':   Phi_BH_sym,
        },
        {
            'label':    'C4-symmetrized, A=0 (Phase B)',
            'short':    'C4-sym A=0',
            'Lambda':   Lambda_sym,
            'v_drift':  v_drift_sym,
            'M_inv':    M_inv_sym,
            'A_berry':  np.zeros_like(A_berry),
            'Phi_BH':   np.zeros_like(Phi_BH),  # Phi_BH also gauge-dependent
        },
        {
            'label':    'C4-sym, Λ + K(no A) only',
            'short':    'C4-sym Λ+K',
            'Lambda':   Lambda_sym,
            'v_drift':  np.zeros_like(v_drift),
            'M_inv':    M_inv_sym,
            'A_berry':  np.zeros_like(A_berry),
            'Phi_BH':   np.zeros_like(Phi_BH),
        },
    ]
    
    all_results = {}
    
    for cfg in configs:
        label = cfg['label']
        print(f"\n  --- {label} ---")
        
        # Regularize M_inv
        M_use = _regularize_M_inv(cfg['M_inv'].copy(), M_inv_max_trace)
        
        # Build H
        H = assemble_multiband_hamiltonian(
            cfg['Lambda'], cfg['v_drift'], M_use, cfg['A_berry'], cfg['Phi_BH'],
            eta, Ns, Ns, Nb, dR, dR, B_moire,
            include_drift=np.any(np.abs(cfg['v_drift']) > 1e-15),
            include_kinetic=True,
            include_born_huang=np.any(np.abs(cfg['Phi_BH']) > 1e-15),
        )
        
        # Enforce Hermiticity
        H = 0.5 * (H + H.conj().T)
        
        # Check [H, C4]
        comm_rel = check_H_C4_commutator(H, Ns, Nb, label)
        
        # Solve eigenvalues
        print(f"    Solving for {n_modes_solve} modes with sigma={sigma:.6f}...")
        try:
            eigenvalues, eigenvectors = eigsh(
                H, k=n_modes_solve, sigma=sigma, which='LM',
                maxiter=10000, tol=1e-10
            )
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            print(f"    Eigenvalues (first 10):")
            for i in range(min(10, len(eigenvalues))):
                print(f"      mode {i}: ε = {eigenvalues[i]:+.8e}  "
                      f"(ω = {omega_ref + eigenvalues[i]:.6f})")
            
            # C4 classification
            c4_results = classify_C4_modes(eigenvalues, eigenvectors, Ns, Nb,
                                           n_modes=n_modes_solve, degen_tol=1e-6)
            print_classification(c4_results, label)
            
            all_results[cfg['short']] = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'c4_results': c4_results,
                'commutator': comm_rel,
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[cfg['short']] = {'error': str(e), 'commutator': comm_rel}
    
    # ================================================================
    # [6] Summary comparison
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Before vs After C4-symmetrization")
    print(f"{'='*70}")
    
    print(f"\n  C4 error reduction in operator data:")
    print(f"  {'Quantity':<20s}  {'Before':>12s}  {'After':>12s}  {'Reduction':>10s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*10}")
    for name, before, after in [
        ('Λ', err_Lambda_before, err_Lambda_after),
        ('Φ_BH', err_Phi_before, err_Phi_after),
        ('ω', err_omega_before, err_omega_after),
        ('A_berry', err_A_before, err_A_after),
        ('v_drift', err_v_before, err_v_after),
        ('M_inv', err_M_before, err_M_after),
    ]:
        if before > 1e-30:
            reduction = f"{before/max(after, 1e-16):.0e}×"
        else:
            reduction = "—"
        print(f"  {name:<20s}  {before:>12.4e}  {after:>12.4e}  {reduction:>10s}")
    
    print(f"\n  Hamiltonian [H, C4] commutator:")
    print(f"  {'Configuration':<30s}  {'||[H,C4]||/||H||':>18s}")
    print(f"  {'-'*30}  {'-'*18}")
    for cfg in configs:
        short = cfg['short']
        if short in all_results:
            comm = all_results[short]['commutator']
            print(f"  {cfg['label']:<30s}  {comm:>18.4e}")
    
    print(f"\n  Eigenvalue comparison (first 10 modes):")
    header = f"  {'mode':>4s}"
    for cfg in configs:
        header += f"  {cfg['short']:>14s}"
    print(header)
    print(f"  {'----':>4s}" + "  ".join(['-'*14]*len(configs)))
    
    for i in range(10):
        row = f"  {i:>4d}"
        for cfg in configs:
            short = cfg['short']
            if short in all_results and 'eigenvalues' in all_results[short]:
                ev = all_results[short]['eigenvalues']
                if i < len(ev):
                    row += f"  {ev[i]:>+14.6e}"
                else:
                    row += f"  {'—':>14s}"
            else:
                row += f"  {'ERR':>14s}"
        print(row)
    
    # ================================================================
    # [7] Plots
    # ================================================================
    print(f"\n[7] Generating plots...")
    plot_results(all_results, configs, Lambda, Lambda_sym, M_inv, M_inv_sym,
                 A_berry, A_berry_sym, Ns, Nb, target_idx, omega_ref)
    
    # ================================================================
    # [8] Save symmetrized data
    # ================================================================
    out_h5 = CAND / "phase2_multiband_data_c4sym.h5"
    print(f"\n[8] Saving symmetrized data to {out_h5}...")
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('Lambda', data=Lambda_sym)
        hf.create_dataset('A_berry', data=A_berry_sym)
        hf.create_dataset('Phi_BH', data=Phi_BH_sym)
        hf.create_dataset('v_drift', data=v_drift_sym)
        hf.create_dataset('M_inv', data=M_inv_sym)
        hf.create_dataset('omega', data=omega_sym)
        
        # Copy original metadata
        with h5py.File(PHASE2_H5, 'r') as hf_orig:
            for key, val in hf_orig.attrs.items():
                hf.attrs[key] = val
            # Copy any other datasets not symmetrized
            for key in hf_orig.keys():
                if key not in hf.keys():
                    hf.create_dataset(key, data=hf_orig[key][:])
        
        hf.attrs['c4_symmetrized'] = True
        hf.attrs['c4_symmetrized_date'] = '2025-02-09'
    print(f"  ✓ Saved ({os.path.getsize(out_h5) / 1e6:.1f} MB)")
    
    print(f"\n{'='*70}")
    print(f"  S4b COMPLETE")
    print(f"{'='*70}")


# ====================================================================================
# Plotting
# ====================================================================================

def plot_results(all_results, configs, Lambda, Lambda_sym, M_inv, M_inv_sym,
                 A_berry, A_berry_sym, Ns, Nb, target_idx, omega_ref):
    """Generate comprehensive diagnostic plots."""
    
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle("S4b: C4-symmetrization of Phase 2 operator data", 
                 fontsize=14, fontweight='bold')
    
    # --- Row 1: Operator fields before/after ---
    # Panel 1,1: Lambda diagonal (target band)
    ax = fig.add_subplot(4, 4, 1)
    im = ax.imshow(Lambda[:, :, target_idx, target_idx].T, origin='lower', cmap='viridis')
    ax.set_title(f'Λ[{target_idx},{target_idx}] original')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = fig.add_subplot(4, 4, 2)
    im = ax.imshow(Lambda_sym[:, :, target_idx, target_idx].T, origin='lower', cmap='viridis')
    ax.set_title(f'Λ[{target_idx},{target_idx}] C4-sym')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Panel 1,3-4: M_inv[target,target,0,0] before/after
    ax = fig.add_subplot(4, 4, 3)
    M_show = M_inv[:, :, target_idx, target_idx, 0, 0]
    vmax = np.percentile(np.abs(M_show), 95)
    im = ax.imshow(M_show.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'M⁻¹[{target_idx},{target_idx},0,0] orig')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = fig.add_subplot(4, 4, 4)
    M_show_sym = M_inv_sym[:, :, target_idx, target_idx, 0, 0]
    im = ax.imshow(M_show_sym.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'M⁻¹[{target_idx},{target_idx},0,0] C4-sym')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # --- Row 2: A_berry and difference maps ---
    ax = fig.add_subplot(4, 4, 5)
    A_mag_orig = np.sqrt(np.abs(A_berry[:,:,target_idx,target_idx,0])**2 +
                         np.abs(A_berry[:,:,target_idx,target_idx,1])**2)
    im = ax.imshow(A_mag_orig.T, origin='lower', cmap='hot')
    ax.set_title(f'|A|[{target_idx},{target_idx}] orig')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = fig.add_subplot(4, 4, 6)
    A_mag_sym = np.sqrt(np.abs(A_berry_sym[:,:,target_idx,target_idx,0])**2 +
                        np.abs(A_berry_sym[:,:,target_idx,target_idx,1])**2)
    im = ax.imshow(A_mag_sym.T, origin='lower', cmap='hot')
    ax.set_title(f'|A|[{target_idx},{target_idx}] C4-sym')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Panel 2,3-4: Λ and M_inv difference (before - after) 
    ax = fig.add_subplot(4, 4, 7)
    diff_Lambda = Lambda[:,:,target_idx,target_idx] - Lambda_sym[:,:,target_idx,target_idx]
    vmax_d = np.max(np.abs(diff_Lambda))
    if vmax_d > 0:
        im = ax.imshow(diff_Lambda.T, origin='lower', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    else:
        im = ax.imshow(diff_Lambda.T, origin='lower', cmap='RdBu_r')
    ax.set_title('ΔΛ (orig−sym)')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    ax = fig.add_subplot(4, 4, 8)
    diff_M = M_inv[:,:,target_idx,target_idx,0,0] - M_inv_sym[:,:,target_idx,target_idx,0,0]
    vmax_d = np.max(np.abs(diff_M))
    if vmax_d > 0:
        im = ax.imshow(diff_M.T, origin='lower', cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
    else:
        im = ax.imshow(diff_M.T, origin='lower', cmap='RdBu_r')
    ax.set_title('ΔM⁻¹₀₀ (orig−sym)')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # --- Row 3: Eigenvalue spectra and commutator ---
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    
    ax = fig.add_subplot(4, 4, 9)
    for idx, cfg in enumerate(configs):
        short = cfg['short']
        if short in all_results and 'eigenvalues' in all_results[short]:
            ev = all_results[short]['eigenvalues']
            ax.plot(range(len(ev)), ev, 'o-', markersize=4,
                    label=short, color=colors[idx], alpha=0.8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue ε")
    ax.set_title("Eigenvalue spectra")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(4, 4, 10)
    comm_vals = []
    comm_labels = []
    for idx, cfg in enumerate(configs):
        short = cfg['short']
        if short in all_results:
            comm_vals.append(all_results[short]['commutator'])
            comm_labels.append(short)
    ax.bar(range(len(comm_vals)), comm_vals, color=colors[:len(comm_vals)])
    ax.set_xticks(range(len(comm_labels)))
    ax.set_xticklabels(comm_labels, rotation=30, ha='right', fontsize=7)
    ax.set_yscale('log')
    ax.set_ylabel("||[H, C4]|| / ||H||")
    ax.set_title("[H, C4] commutator")
    ax.grid(True, alpha=0.3)
    
    # Panel 3,3: C4 closure per mode for symmetrized configs
    ax = fig.add_subplot(4, 4, 11)
    for idx, cfg in enumerate(configs):
        short = cfg['short']
        if short in all_results and 'c4_results' in all_results[short]:
            c4r = all_results[short]['c4_results']
            mode_ids = []
            closures = []
            for r in c4r:
                for mi in r['indices']:
                    mode_ids.append(mi)
                    closures.append(r['c4_closure'])
            ax.plot(mode_ids, closures, 'o', markersize=5, label=short,
                    color=colors[idx], alpha=0.7)
    ax.axhline(y=0.999, color='k', linestyle='--', alpha=0.5, label='0.999')
    ax.set_xlabel("Mode index")
    ax.set_ylabel("C4 closure")
    ax.set_title("C4 quality per mode")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    # Panel 3,4: C4 irrep distribution for best config
    ax = fig.add_subplot(4, 4, 12)
    best_key = 'C4-sym A=0'
    if best_key not in all_results or 'c4_results' not in all_results[best_key]:
        best_key = 'C4-sym'
    if best_key in all_results and 'c4_results' in all_results[best_key]:
        irrep_counts = {'A': 0, 'E+': 0, 'B': 0, 'E-': 0, '?': 0}
        for r in all_results[best_key]['c4_results']:
            if 'c4_irrep' in r:
                label_ir = r['c4_irrep'].split('(')[-1].rstrip(')')
                if label_ir in irrep_counts:
                    irrep_counts[label_ir] += 1
                else:
                    irrep_counts['?'] += 1
            elif 'c4_irreps' in r:
                for ir in r['c4_irreps']:
                    if ir in irrep_counts:
                        irrep_counts[ir] += 1
                    else:
                        irrep_counts['?'] += 1
        labels_ir = list(irrep_counts.keys())
        vals_ir = list(irrep_counts.values())
        bar_colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6']
        ax.bar(labels_ir, vals_ir, color=bar_colors[:len(labels_ir)])
        ax.set_xlabel("C4 irrep")
        ax.set_ylabel("Count")
        ax.set_title(f"C4 irrep distribution ({best_key})")
    ax.grid(True, alpha=0.3)
    
    # --- Row 4: Mode profiles for symmetrized configs ---
    for col, mode_idx in enumerate(range(4)):
        ax = fig.add_subplot(4, 4, 13 + col)
        best_key2 = 'C4-sym A=0'
        if best_key2 not in all_results or 'eigenvectors' not in all_results[best_key2]:
            best_key2 = 'C4-sym'
        if best_key2 in all_results and 'eigenvectors' in all_results[best_key2]:
            evecs = all_results[best_key2]['eigenvectors']
            evals = all_results[best_key2]['eigenvalues']
            if mode_idx < evecs.shape[1]:
                F = evecs[:, mode_idx].reshape(Ns, Ns, Nb)
                prob = np.sum(np.abs(F)**2, axis=2)
                im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
                
                # Get irrep label
                irrep_label = '?'
                for r in all_results[best_key2]['c4_results']:
                    if mode_idx in r['indices']:
                        if 'c4_irrep' in r:
                            irrep_label = r['c4_irrep']
                        elif 'c4_irreps' in r:
                            irrep_label = '+'.join(r['c4_irreps'])
                        break
                
                ax.set_title(f"Mode {mode_idx}: ε={evals[mode_idx]:+.4e}\n{irrep_label}",
                           fontsize=9)
                plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_xlabel("ix")
        if col == 0:
            ax.set_ylabel("iy")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4b_c4_symmetrization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved S4b_c4_symmetrization.png")
    
    # === Second figure: detailed mode gallery comparing original vs C4-sym ===
    fig2, axes2 = plt.subplots(4, 5, figsize=(20, 16))
    fig2.suptitle("S4b: Mode gallery — Original vs C4-symmetrized", 
                  fontsize=14, fontweight='bold')
    
    row_configs = ['Original', 'C4-sym', 'C4-sym A=0', 'C4-sym Λ+K']
    for row_idx, key in enumerate(row_configs):
        for col_idx in range(5):
            ax = axes2[row_idx, col_idx]
            if key in all_results and 'eigenvectors' in all_results[key]:
                evecs = all_results[key]['eigenvectors']
                evals = all_results[key]['eigenvalues']
                if col_idx < evecs.shape[1]:
                    F = evecs[:, col_idx].reshape(Ns, Ns, Nb)
                    prob = np.sum(np.abs(F)**2, axis=2)
                    im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
                    
                    # Get C4 info
                    c4_info = ''
                    for r in all_results[key]['c4_results']:
                        if col_idx in r['indices']:
                            if 'c4_irrep' in r:
                                c4_info = r['c4_irrep']
                            break
                    
                    ax.set_title(f"ε={evals[col_idx]:+.4e}\n{c4_info}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center', 
                       transform=ax.transAxes)
            
            if col_idx == 0:
                ax.set_ylabel(key, fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4b_mode_gallery.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved S4b_mode_gallery.png")


if __name__ == "__main__":
    main()
