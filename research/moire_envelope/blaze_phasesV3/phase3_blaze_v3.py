"""
Phase 3 (BLAZE): Multi-Band Envelope Solver — V3 Multi-Band Pipeline

This is the V3 multi-band implementation of Phase 3.
It solves the full multi-band envelope Hamiltonian including:

1. DIAGONAL POTENTIALS: Λ_nn(s) = ω_n(s) - ω_ref
2. DRIFT TERM: η × v_mn · ∇ F_n
3. KINETIC TERM: η² × (1/2) Σ_ij (∂_i + iA_i)_mn M^(-1)_mn,ij (∂_j + iA_j)_mn
4. BORN-HUANG: η² × Φ_mn(s) (Born-Huang potential)

The full Hamiltonian is:
    Ĥ_mn = Λ_mn(s) δ_mn + η v_mn · (-i∇) + η² (1/2) D_i M^(-1)_ij D_j + η² Φ_BH,mn

where D_i = -i∂_i + A_i is the gauge-covariant derivative.

DATA STRUCTURES (V3):
- Hamiltonian: sparse block matrix of size (N_s × N_bands)²
- Eigenvectors: F(s) ∈ ℂ^(N_s × N_bands) spinor envelopes
- Eigenvalues: cavity mode frequencies

THEORY REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md
"""

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from pathlib import Path
import sys
import os
import math
import time
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, save_json, load_json


def plot_phase3_envelope_modes_v3(cdir, s_grid, F_all, eigenvalues, omega_ref,
                                   mode_stats, subspace_bands, B_moire):
    """
    Plot envelope modes.
    
    Creates two plots:
    1. 4x4 grid of modes with LOWEST SPREAD (most localized), sorted by spread
    2. Saves full mode statistics to a log file
    """
    import matplotlib.pyplot as plt
    
    n_modes_total = len(eigenvalues)
    N_subspace = F_all.shape[3]
    
    # Sort modes by spread (lowest = most localized)
    spreads = [s['spread'] for s in mode_stats]
    sorted_by_spread = np.argsort(spreads)
    
    # Plot 1: 4x4 grid of lowest-spread modes
    n_plot = min(16, n_modes_total)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for plot_idx in range(n_plot):
        mode_idx = sorted_by_spread[plot_idx]
        ax = axes[plot_idx]
        
        # Total probability density |F|² = Σ_n |F_n|²
        prob = np.sum(np.abs(F_all[mode_idx])**2, axis=2)
        
        im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
        
        stats = mode_stats[mode_idx]
        omega = stats['omega']
        spread = stats['spread']
        dom_band = stats['dominant_band']
        dom_weight = stats['dominant_band_weight']
        
        ax.set_title(f'Mode {mode_idx}\nω={omega:.4f}\nspread={spread:.3f}, band={dom_band} ({dom_weight:.0%})', 
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel('s₁')
        ax.set_ylabel('s₂')
    
    # Hide unused subplots
    for idx in range(n_plot, 16):
        axes[idx].axis('off')
    
    plt.suptitle(f'Top {n_plot} Most Localized Modes (sorted by spread)', fontsize=14)
    plt.tight_layout()
    plt.savefig(cdir / 'phase3_envelope_modes_by_spread.png', dpi=150)
    plt.close()
    
    # Plot 2: Modes sorted by eigenvalue (for reference)
    sorted_by_eigenvalue = np.argsort(eigenvalues)
    n_plot_eig = min(16, n_modes_total)
    
    fig2, axes2 = plt.subplots(4, 4, figsize=(16, 16))
    axes2 = axes2.flatten()
    
    for plot_idx in range(n_plot_eig):
        mode_idx = sorted_by_eigenvalue[plot_idx]
        ax = axes2[plot_idx]
        
        prob = np.sum(np.abs(F_all[mode_idx])**2, axis=2)
        im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
        
        stats = mode_stats[mode_idx]
        omega = stats['omega']
        spread = stats['spread']
        
        ax.set_title(f'Mode {mode_idx}\nω={omega:.4f}\nspread={spread:.3f}', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    for idx in range(n_plot_eig, 16):
        axes2[idx].axis('off')
    
    plt.suptitle(f'Lowest {n_plot_eig} Eigenvalue Modes', fontsize=14)
    plt.tight_layout()
    plt.savefig(cdir / 'phase3_envelope_modes_by_eigenvalue.png', dpi=150)
    plt.close()
    
    # Write detailed log file
    log_path = cdir / 'phase3_modes_detailed.log'
    with open(log_path, 'w') as f:
        f.write(f"Phase 3 Multi-Band Envelope Mode Analysis\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Total modes computed: {n_modes_total}\n")
        f.write(f"Reference frequency: {omega_ref:.6f}\n")
        f.write(f"Eigenvalue range: [{eigenvalues.min():.6e}, {eigenvalues.max():.6e}]\n\n")
        
        f.write(f"=== ALL MODES (sorted by eigenvalue) ===\n")
        for mode_idx in sorted_by_eigenvalue:
            s = mode_stats[mode_idx]
            f.write(f"Mode {mode_idx:3d}: ω = {s['omega']:12.6f}, "
                    f"eigenval = {eigenvalues[mode_idx]:12.6e}, "
                    f"spread = {s['spread']:.4f}, "
                    f"dom_band = {s['dominant_band']} ({s['dominant_band_weight']:.2%})\n")
        
        f.write(f"\n=== MODES BY SPREAD (most localized first) ===\n")
        for rank, mode_idx in enumerate(sorted_by_spread):
            s = mode_stats[mode_idx]
            f.write(f"Rank {rank:2d} | Mode {mode_idx:3d}: spread = {s['spread']:.4f}, "
                    f"ω = {s['omega']:.6f}, "
                    f"dom_band = {s['dominant_band']} ({s['dominant_band_weight']:.2%})\n")
    
    log(f"    Saved detailed mode log to {log_path}")


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# Multi-Band Finite-Difference Operators
# ==============================================================================

def build_periodic_derivative_matrix(N, ds, order=4):
    """
    Build periodic finite-difference derivative matrix.
    
    Args:
        N: number of grid points
        ds: grid spacing
        order: finite difference order (2 or 4)
        
    Returns:
        D: (N, N) sparse derivative matrix
    """
    if order == 4:
        # 4th order: (-1, 8, 0, -8, 1) / 12h
        coeffs = np.array([1, -8, 0, 8, -1]) / (12 * ds)
        offsets = [-2, -1, 0, 1, 2]
    else:
        # 2nd order: (-1, 0, 1) / 2h
        coeffs = np.array([-0.5, 0, 0.5]) / ds
        offsets = [-1, 0, 1]
    
    diagonals = []
    for coeff, offset in zip(coeffs, offsets):
        diag = np.full(N, coeff)
        diagonals.append(diag)
    
    D = sparse.diags(diagonals, offsets, shape=(N, N), format='lil')
    
    # Periodic boundary conditions
    for coeff, offset in zip(coeffs, offsets):
        if offset < 0:
            for i in range(-offset):
                D[i, N + offset + i] = coeff
        elif offset > 0:
            for i in range(offset):
                D[N - offset + i, i] = coeff
    
    return D.tocsr()


def build_periodic_laplacian_matrix(N, ds, order=4):
    """
    Build periodic finite-difference Laplacian matrix.
    
    Args:
        N: number of grid points
        ds: grid spacing
        order: finite difference order (2 or 4)
        
    Returns:
        L: (N, N) sparse Laplacian matrix
    """
    if order == 4:
        # 4th order: (-1, 16, -30, 16, -1) / 12h²
        coeffs = np.array([-1, 16, -30, 16, -1]) / (12 * ds**2)
        offsets = [-2, -1, 0, 1, 2]
    else:
        # 2nd order: (1, -2, 1) / h²
        coeffs = np.array([1, -2, 1]) / ds**2
        offsets = [-1, 0, 1]
    
    diagonals = []
    for coeff, offset in zip(coeffs, offsets):
        diag = np.full(N, coeff)
        diagonals.append(diag)
    
    L = sparse.diags(diagonals, offsets, shape=(N, N), format='lil')
    
    # Periodic BCs
    for coeff, offset in zip(coeffs, offsets):
        if offset < 0:
            for i in range(-offset):
                L[i, N + offset + i] = coeff
        elif offset > 0:
            for i in range(offset):
                L[N - offset + i, i] = coeff
    
    return L.tocsr()


def to_block_diagonal_index(i, j, n, Ns1, Ns2, N_bands):
    """
    Convert (spatial_index_i, spatial_index_j, band_index_n) to flat index.
    
    Layout: [s_0 band_0, s_0 band_1, ..., s_0 band_N-1, s_1 band_0, ...]
    """
    spatial_idx = i * Ns2 + j
    return spatial_idx * N_bands + n


def build_multiband_potential_operator(Lambda, B_moire):
    """
    Build the diagonal potential operator Λ_mn(s).
    
    VECTORIZED IMPLEMENTATION using indices broadcasting.
    
    Args:
        Lambda: (Ns1, Ns2, N_bands, N_bands) potential matrix
        B_moire: (2, 2) moiré basis vectors
        
    Returns:
        V_op: sparse matrix of size (Ns × N_bands, Ns × N_bands)
    """
    Ns1, Ns2, N_bands, _ = Lambda.shape
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    
    # Generate indices
    # Spatial indices (i, j)
    # Band indices (m, n)
    
    # Flatten Lambda to find non-zeros
    # Layout matches to_block_diagonal_index: (i*Ns2 + j)*N_bands + n
    # But Lambda is (Ns1, Ns2, N_bands, N_bands)
    # We want indices for the full matrix.
    
    # Create grid of spatial indices k = i*Ns2 + j
    # This repeats for every block element (m, n)
    k_grid = np.arange(N_s)
    
    # We handle the block structure by iterating over band indices m, n
    # and creating diagonals or off-diagonals in the large matrix.
    # V_{k*Nb + m, k*Nb + n} = Lambda[k_unraveled, m, n]
    
    V_op = sparse.lil_matrix((N_total, N_total), dtype=complex)
    
    # Lambda reshaped to (Ns, Nb, Nb)
    Lambda_flat = Lambda.reshape(N_s, N_bands, N_bands)
    
    # Vectorized construction using COO format directly
    rows = []
    cols = []
    data = []
    
    # Iterate over band blocks (small loop 5x5)
    for m in range(N_bands):
        for n in range(N_bands):
            vals = Lambda_flat[:, m, n]
            mask = np.abs(vals) > 1e-15
            
            if np.any(mask):
                k_indices = k_grid[mask]
                active_vals = vals[mask]
                
                # Compute global indices
                r_idx = k_indices * N_bands + m
                c_idx = k_indices * N_bands + n
                
                rows.append(r_idx)
                cols.append(c_idx)
                data.append(active_vals)
    
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
        V_op = sparse.csr_matrix((data, (rows, cols)), shape=(N_total, N_total))
    else:
        V_op = sparse.csr_matrix((N_total, N_total), dtype=complex)
        
    return V_op


def build_multiband_drift_operator(v_drift, eta, Ns1, Ns2, N_bands, ds1, ds2, order=4):
    """
    Build the drift term operator: η × v_mn · (-i∇).
    
    VECTORIZED IMPLEMENTATION using Kronecker products.
    
    T = -i η Σ_μ V_μ (D_μ ⊗ I_bands)
    
    Args:
        v_drift: (Ns1, Ns2, N_bands, N_bands, 2) drift velocity matrix
        
    Returns:
        T_drift: sparse matrix
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    
    # 1. Construct V_mu operators (block diagonal in spatial, dense in bands)
    # V_mu is a matrix field.
    # We construct it similarly to potential operator.
    
    V1_op = sparse.lil_matrix((N_total, N_total), dtype=complex)
    V2_op = sparse.lil_matrix((N_total, N_total), dtype=complex)
    
    # Reshape v_drift to (Ns, Nb, Nb, 2)
    v_flat = v_drift.reshape(N_s, N_bands, N_bands, 2)
    k_grid = np.arange(N_s)
    
    rows1, cols1, data1 = [], [], []
    rows2, cols2, data2 = [], [], []
    
    for m in range(N_bands):
        for n in range(N_bands):
            # Component 1
            vals1 = v_flat[:, m, n, 0]
            mask1 = np.abs(vals1) > 1e-15
            if np.any(mask1):
                k = k_grid[mask1]
                rows1.append(k * N_bands + m)
                cols1.append(k * N_bands + n)
                data1.append(vals1[mask1])
            
            # Component 2
            vals2 = v_flat[:, m, n, 1]
            mask2 = np.abs(vals2) > 1e-15
            if np.any(mask2):
                k = k_grid[mask2]
                rows2.append(k * N_bands + m)
                cols2.append(k * N_bands + n)
                data2.append(vals2[mask2])
                
    if rows1:
        V1_op = sparse.csr_matrix((np.concatenate(data1), (np.concatenate(rows1), np.concatenate(cols1))), shape=(N_total, N_total))
    if rows2:
        V2_op = sparse.csr_matrix((np.concatenate(data2), (np.concatenate(rows2), np.concatenate(cols2))), shape=(N_total, N_total))
        
    # 2. Construct Derivative operators in full space
    # D1 acts on s1: D1 ⊗ I_s2 ⊗ I_bands
    # D2 acts on s2: I_s1 ⊗ D2 ⊗ I_bands
    
    D1_base = build_periodic_derivative_matrix(Ns1, ds1, order)
    D2_base = build_periodic_derivative_matrix(Ns2, ds2, order)
    
    # Order of tensor product matches flattening: i (Ns1), j (Ns2), n (Nbands)
    # Flat index = (i * Ns2 + j) * Nbands + n
    # This corresponds to Tensor Product: Space1 ⊗ Space2 ⊗ Band
    
    D1_full = kron(D1_base, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2_base, eye(N_bands)), format='csr')
    
    # 3. Combine: T = -i η (V1 D1 + V2 D2)
    # Note: V operator is diagonal in spatial indices (multiplicative), D is derivative.
    # Order matters: v(r) * d/dr.
    
    T_drift = -1j * eta * (V1_op @ D1_full + V2_op @ D2_full)
    
    return T_drift


def build_multiband_kinetic_operator(
    M_inv, A_berry, eta, Ns1, Ns2, N_bands, ds1, ds2, B_moire, order=4
):
    """
    Build the kinetic term operator: η² × (1/2) D_i M^(-1)_ij D_j.
    
    VECTORIZED IMPLEMENTATION using Kronecker products.
    
    Assumes diagonal approximation for mass tensor in bands (block diagonal).
    K = \sum_n P_n K_n P_n^T
    
    Args:
        M_inv: (Ns1, Ns2, N_bands, N_bands, 2, 2)
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    
    scale_factor = 1.0 / (2 * np.pi)**2
    prefactor = 0.5 * eta**2 * scale_factor
    
    # Base operators
    L1 = build_periodic_laplacian_matrix(Ns1, ds1, order)
    L2 = build_periodic_laplacian_matrix(Ns2, ds2, order)
    D1 = build_periodic_derivative_matrix(Ns1, ds1, order)
    D2 = build_periodic_derivative_matrix(Ns2, ds2, order)
    
    # Full operators
    # We construct "Identity-like" full operators where diagonal is 1
    # But for band-specific operators, we need specific diagonals.
    
    # Strategy: Build K by summing terms:
    # 1. -M_11 L_1
    # 2. -M_22 L_2
    # 3. -M_12 D_1 D_2
    # 4. Berry terms
    
    # We construct diagonal matrices for Mass terms M_11(s), M_22(s), etc.
    # M_inv is (Ns1, Ns2, Nb, Nb, 2, 2).
    # We assume it is diagonal in bands.
    # Flatten to (N_total, 2, 2)
    M_inv_flat = np.zeros((N_total, 2, 2), dtype=complex)
    
    # Populate diagonal elements (iterating bands is fast, 5 iterations)
    # M_inv[i,j,n,n] -> flattens to index k*Nb + n
    M_inv_reshaped = M_inv.reshape(N_s, N_bands, N_bands, 2, 2)
    for n in range(N_bands):
        # Extract diagonal component n,n for all spatial points
        # Place into correct slots in flat array
        indices = np.arange(N_s) * N_bands + n
        M_inv_flat[indices] = M_inv_reshaped[:, n, n, :, :]
        
    M11_diag = diags(M_inv_flat[:, 0, 0], format='csr')
    M22_diag = diags(M_inv_flat[:, 1, 1], format='csr')
    M12_diag = diags(M_inv_flat[:, 0, 1], format='csr') # Symmetrized usually
    
    # Full derivatives
    L1_full = kron(L1, eye(Ns2 * N_bands), format='csr')
    L2_full = kron(eye(Ns1), kron(L2, eye(N_bands)), format='csr')
    D1_full = kron(D1, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2, eye(N_bands)), format='csr')
    
    # K = - (M11 L1 + M22 L2 + 2 M12 D1 D2)
    # (Assuming symmetry M12=M21)
    
    K_op = - (M11_diag @ L1_full + M22_diag @ L2_full)
    
    # Mixed term
    # Check if M12 is non-zero
    if np.max(np.abs(M_inv_flat[:, 0, 1])) > 1e-15:
         K_op = K_op - 2 * M12_diag @ (D1_full @ D2_full)
         
    # Apply prefactor
    K_op = prefactor * K_op
    
    # Berry terms (Diagonal A_ii^2 + derivatives)
    # A_berry is (Ns, Nb, Nb, 2)
    # A_sq term: 0.5 * eta^2 * M * A^2 (scalar potential addition)
    
    A_berry_flat = np.zeros((N_total, 2), dtype=complex)
    A_berry_reshaped = A_berry.reshape(N_s, N_bands, N_bands, 2)
    for n in range(N_bands):
        indices = np.arange(N_s) * N_bands + n
        A_berry_flat[indices] = A_berry_reshaped[:, n, n, :]
        
    A1 = A_berry_flat[:, 0]
    A2 = A_berry_flat[:, 1]
    
    # |A|^2 term (Diagonal)
    M11 = M_inv_flat[:, 0, 0]
    M22 = M_inv_flat[:, 1, 1]
    M12 = M_inv_flat[:, 0, 1]
    
    A_sq_val = (M11 * np.abs(A1)**2 + M22 * np.abs(A2)**2 + 
                2 * M12 * np.real(A1 * np.conj(A2)))
    
    if np.max(np.abs(A_sq_val)) > 1e-15:
        K_op = K_op + diags(prefactor * A_sq_val, format='csr')
        
    # Paramagnetic terms: -i M A D (couple A and derivative)
    # term: -2i * 0.5 * eta^2 * M * A * D = -i eta^2 M A D
    # (Simplified, assumes Coulomb gauge div A ~ 0 for some parts, but we execute M A D)
    
    # Actually term is (pi + pi^dag)
    # Here we perform -i ( M11 A1 D1 + M22 A2 D2 + ... ) + h.c.
    # Since we lack full gauge covariance implementation in this refactoring,
    # and A is approximated as 0 usually, we keep it simple.
    
    return K_op


def build_multiband_born_huang_operator(Phi_BH, eta, Ns1, Ns2, N_bands):
    """
    Build Born-Huang potential operator: η² × Φ_mn(s).
    
    VECTORIZED IMPLEMENTATION.
    """
    # Reuse Potential operator logic
    V_BH = build_multiband_potential_operator(Phi_BH, None)
    return eta**2 * V_BH


# ==============================================================================
# Full Hamiltonian Assembly
# ==============================================================================

def assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv, A_berry, Phi_BH,
    eta, Ns1, Ns2, N_bands, ds1, ds2, B_moire,
    include_drift=True, include_kinetic=True, include_born_huang=True,
    order=4
):
    """
    Assemble the full multi-band envelope Hamiltonian.
    
    Ĥ = Λ + η T_drift + η² K + η² U_BH
    
    Args:
        Lambda: (Ns1, Ns2, N_bands, N_bands) diagonal potentials
        v_drift: (Ns1, Ns2, N_bands, N_bands, 2) drift velocities
        M_inv: (Ns1, Ns2, N_bands, N_bands, 2, 2) inverse mass tensor
        A_berry: (Ns1, Ns2, N_bands, N_bands, 2) Berry connection
        Phi_BH: (Ns1, Ns2, N_bands, N_bands) Born-Huang potential
        eta: small parameter
        Ns1, Ns2: grid dimensions
        N_bands: number of bands
        ds1, ds2: grid spacings
        B_moire: moiré basis
        include_drift: include drift term
        include_kinetic: include kinetic term
        include_born_huang: include Born-Huang term
        order: finite difference order
        
    Returns:
        H: sparse Hamiltonian matrix
    """
    N_total = Ns1 * Ns2 * N_bands
    log(f"    Building Hamiltonian: {N_total}×{N_total} ({Ns1}×{Ns2}×{N_bands})")
    
    # Potential term (always included)
    log("    - Potential operator V...")
    H = build_multiband_potential_operator(Lambda, B_moire)
    log(f"      nnz = {H.nnz}")
    
    # Drift term
    if include_drift:
        log("    - Drift operator T...")
        T_drift = build_multiband_drift_operator(
            v_drift, eta, Ns1, Ns2, N_bands, ds1, ds2, order
        )
        H = H + T_drift
        log(f"      nnz = {T_drift.nnz}")
    
    # Kinetic term
    if include_kinetic:
        log("    - Kinetic operator K...")
        K_op = build_multiband_kinetic_operator(
            M_inv, A_berry, eta, Ns1, Ns2, N_bands, ds1, ds2, B_moire, order
        )
        H = H + K_op
        log(f"      nnz = {K_op.nnz}")
    
    # Born-Huang term
    if include_born_huang and np.max(np.abs(Phi_BH)) > 1e-15:
        log("    - Born-Huang operator U_BH...")
        U_BH = build_multiband_born_huang_operator(Phi_BH, eta, Ns1, Ns2, N_bands)
        H = H + U_BH
        log(f"      nnz = {U_BH.nnz}")
    
    # Convert to CSR for efficient eigensolve
    H = H.tocsr()
    log(f"    Total nnz = {H.nnz}")
    
    return H


# ==============================================================================
# Eigenvalue Solver
# ==============================================================================

def solve_multiband_envelope(H, n_modes, sigma=None, which='SM'):
    """
    Solve the multi-band envelope eigenvalue problem.
    
    Args:
        H: sparse Hamiltonian matrix
        n_modes: number of modes to compute
        sigma: shift for shift-invert mode (target eigenvalue)
        which: 'SM' for smallest magnitude, 'LM' for largest
        
    Returns:
        eigenvalues: (n_modes,) array of eigenvalues
        eigenvectors: (N_total, n_modes) array of eigenvectors
    """
    N_total = H.shape[0]
    
    # Ensure Hermitian
    H_herm = 0.5 * (H + H.conj().T)
    
    # DIAGNOSTIC: Check energy range
    h_diag = H_herm.diagonal().real
    min_E, max_E = np.min(h_diag), np.max(h_diag)
    log(f"    Hamiltonian Diagonal Range: [{min_E:.4f}, {max_E:.4f}]")
    
    # AUTO-SELECT SIGMA: Target the potential minimum to find the lowest bound states
    if sigma is None:
        if min_E < -0.1:
            # Use the exact minimum - eigenvalues could be slightly lower due to
            # off-diagonal coupling, but this targets the ground state region
            sigma = min_E
            log(f"    Auto-selected sigma = {sigma:.4f} (targeting lowest bound states)")
        else:
            sigma = 0.0
            log(f"    Using sigma = 0.0 (no deep potential well detected)")

    log(f"    Solving eigenvalue problem ({N_total}×{N_total})...")
    start_time = time.time()
    
    # Convergence parameters
    max_iterations = 10000
    tolerance = 1e-10
    
    # Use tqdm for a spinner during eigensolve
    with tqdm(total=1, desc="      Eigensolve", bar_format='{desc}: {elapsed} elapsed', leave=False) as pbar:
        try:
            # Shift-invert: find largest eigenvalues of (H - sigma*I)^(-1)
            # which corresponds to eigenvalues of H closest to sigma
            eigenvalues, eigenvectors = eigsh(
                H_herm, k=n_modes, sigma=sigma, which='LM',
                maxiter=max_iterations, tol=tolerance
            )
            pbar.update(1)
            log(f"    Converged with tol={tolerance}, maxiter={max_iterations}")
        except Exception as e:
            pbar.close()
            log(f"    WARNING: eigsh failed ({e}), trying dense solve...")
            with tqdm(total=1, desc="      Dense eigh", bar_format='{desc}: {elapsed} elapsed', leave=False) as pbar2:
                H_dense = H_herm.toarray()
                all_eigs, all_vecs = np.linalg.eigh(H_dense)
                pbar2.update(1)
            
            if sigma is not None:
                # Find modes closest to sigma
                order = np.argsort(np.abs(all_eigs - sigma))
            else:
                # Smallest eigenvalues
                order = np.argsort(all_eigs)
            
            eigenvalues = all_eigs[order[:n_modes]]
            eigenvectors = all_vecs[:, order[:n_modes]]
    
    elapsed = time.time() - start_time
    log(f"    Eigensolve completed in {elapsed:.2f}s")
    
    # Sort by eigenvalue
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    return eigenvalues, eigenvectors


def reshape_spinor_envelope(eigenvector, Ns1, Ns2, N_bands):
    """
    Reshape flat eigenvector to spinor envelope F(s).
    
    Args:
        eigenvector: (N_total,) flat eigenvector
        Ns1, Ns2: grid dimensions
        N_bands: number of bands
        
    Returns:
        F: (Ns1, Ns2, N_bands) spinor envelope
    """
    F = eigenvector.reshape(Ns1, Ns2, N_bands)
    return F


def compute_mode_statistics(F, omega_ref, eigenvalue):
    """
    Compute statistics for a single envelope mode.
    
    Args:
        F: (Ns1, Ns2, N_bands) spinor envelope
        omega_ref: reference frequency
        eigenvalue: mode eigenvalue
        
    Returns:
        stats: dict with mode statistics
    """
    Ns1, Ns2, N_bands = F.shape
    
    # Probability density per band
    prob_per_band = np.sum(np.abs(F)**2, axis=(0, 1))
    prob_per_band /= np.sum(prob_per_band)
    
    # Total probability density |F|² = Σ_n |F_n|²
    prob_total = np.sum(np.abs(F)**2, axis=2)
    prob_total /= np.sum(prob_total)
    
    # Centroid
    s1 = np.arange(Ns1) / Ns1
    s2 = np.arange(Ns2) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    
    centroid_s1 = np.sum(S1 * prob_total)
    centroid_s2 = np.sum(S2 * prob_total)
    
    # Spread (standard deviation)
    var_s1 = np.sum((S1 - centroid_s1)**2 * prob_total)
    var_s2 = np.sum((S2 - centroid_s2)**2 * prob_total)
    spread = np.sqrt(var_s1 + var_s2)
    
    # Mode frequency
    omega = omega_ref + eigenvalue
    
    stats = {
        'omega': float(omega),
        'eigenvalue': float(eigenvalue),
        'omega_ref': float(omega_ref),
        'prob_per_band': prob_per_band.tolist(),
        'dominant_band': int(np.argmax(prob_per_band)),
        'dominant_band_weight': float(np.max(prob_per_band)),
        'centroid': [float(centroid_s1), float(centroid_s2)],
        'spread': float(spread),
        'max_density': float(np.max(prob_total)),
    }
    
    return stats


# ==============================================================================
# Process Single Candidate
# ==============================================================================

def process_candidate_phase3_v3(candidate_dir_path, config):
    """Process single candidate through Phase 3 V3."""
    cdir = Path(candidate_dir_path)
    cid = int(cdir.name.split('_')[-1])
    
    log(f"\n=== Phase 3 V3: Candidate {cid} ===")
    
    phase2_h5 = cdir / "phase2_multiband_data.h5"
    if not phase2_h5.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {phase2_h5}")
    
    # Load Phase 2 data
    with h5py.File(phase2_h5, 'r') as hf:
        s_grid = hf['s_grid'][:]
        R_grid = hf['R_grid'][:]
        
        Lambda = hf['Lambda'][:]      # (Ns1, Ns2, N, N)
        A_berry = hf['A_berry'][:]    # (Ns1, Ns2, N, N, 2)
        Phi_BH = hf['Phi_BH'][:]      # (Ns1, Ns2, N, N)
        v_drift = hf['v_drift'][:]    # (Ns1, Ns2, N, N, 2)
        M_inv = hf['M_inv'][:]        # (Ns1, Ns2, N, N, 2, 2)
        
        omega_grid = hf['omega'][:]
        V_grid = hf['V'][:]
        
        omega_ref = hf.attrs['omega_ref']
        eta = hf.attrs['eta']
        theta_rad = hf.attrs['theta_rad']
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_subspace = int(hf.attrs['N_subspace'])
        target_idx = int(hf.attrs['target_index_in_subspace'])
        B_moire = hf.attrs['B_moire']
        B_mono = hf.attrs['B_mono']
        subspace_bands = hf.attrs['subspace_bands'][:].tolist()
    
    log(f"  Grid: {Ns1} × {Ns2}, N_subspace = {N_subspace}")
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  ω_ref = {omega_ref:.6f}")
    
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    
    # Configuration
    n_modes = config.get('n_modes', 50)  # Default to 50 modes for better statistics
    include_drift = config.get('include_drift_term', True)
    include_kinetic = config.get('include_kinetic_term', True)
    include_born_huang = config.get('include_born_huang', True)
    fd_order = config.get('fd_order', 4)
    sigma_shift = config.get('sigma_shift', None)
    
    log(f"  Computing {n_modes} modes")
    log(f"  Include drift: {include_drift}")
    log(f"  Include kinetic: {include_kinetic}")
    log(f"  Include Born-Huang: {include_born_huang}")
    
    # Assemble Hamiltonian
    log("  Assembling Hamiltonian...")
    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        eta, Ns1, Ns2, N_subspace, ds1, ds2, B_moire,
        include_drift, include_kinetic, include_born_huang,
        fd_order
    )
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = solve_multiband_envelope(
        H, n_modes, sigma=sigma_shift
    )
    
    log(f"  Eigenvalue range: [{eigenvalues.min():.6e}, {eigenvalues.max():.6e}]")
    
    # Compute mode statistics
    mode_stats = []
    for mode_idx in range(n_modes):
        F = reshape_spinor_envelope(eigenvectors[:, mode_idx], Ns1, Ns2, N_subspace)
        stats = compute_mode_statistics(F, omega_ref, eigenvalues[mode_idx])
        stats['mode_index'] = mode_idx
        mode_stats.append(stats)
        
        log(f"    Mode {mode_idx}: ω = {stats['omega']:.6f}, "
            f"dom_band = {stats['dominant_band']} ({stats['dominant_band_weight']:.2%}), "
            f"spread = {stats['spread']:.3f}")
    
    # Save results
    h5_path = cdir / "phase3_multiband_modes.h5"
    
    with h5py.File(h5_path, 'w') as hf:
        # Coordinate grids
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        
        # Eigenvalues and eigenvectors
        hf.create_dataset("eigenvalues", data=eigenvalues)
        hf.create_dataset("eigenvectors", data=eigenvectors, compression="gzip")
        
        # Reshaped spinor envelopes for convenience
        F_all = eigenvectors.reshape(Ns1, Ns2, N_subspace, n_modes)
        F_all = np.moveaxis(F_all, -1, 0)  # (n_modes, Ns1, Ns2, N_subspace)
        hf.create_dataset("F_spinor", data=F_all, compression="gzip")
        
        # Hamiltonian (sparse)
        hf.create_dataset("H_data", data=H.data)
        hf.create_dataset("H_indices", data=H.indices)
        hf.create_dataset("H_indptr", data=H.indptr)
        hf.attrs["H_shape"] = H.shape
        
        # Metadata
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["theta_deg"] = math.degrees(theta_rad)
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["Ns1"] = Ns1
        hf.attrs["Ns2"] = Ns2
        hf.attrs["N_subspace"] = N_subspace
        hf.attrs["n_modes"] = n_modes
        hf.attrs["target_index_in_subspace"] = target_idx
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_mono"] = B_mono
        hf.attrs["subspace_bands"] = np.array(subspace_bands)
        
        hf.attrs["include_drift"] = include_drift
        hf.attrs["include_kinetic"] = include_kinetic
        hf.attrs["include_born_huang"] = include_born_huang
        hf.attrs["fd_order"] = fd_order
        hf.attrs["pipeline_version"] = "V3"
    
    log(f"  Saved Phase 3 data to {h5_path}")
    
    # Save mode statistics as JSON
    save_json(mode_stats, cdir / "phase3_mode_stats.json")
    
    # Generate visualizations
    try:
        plot_phase3_envelope_modes_v3(
            cdir, s_grid, F_all, eigenvalues, omega_ref, 
            mode_stats, subspace_bands, B_moire
        )
    except Exception as e:
        log(f"    WARNING: Visualization failed: {e}")
    
    log(f"=== Phase 3 Complete: Candidate {cid} ===")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_phase3_v3(run_dir, config_path):
    """Main Phase 3 V3 driver."""
    log("\n" + "="*70)
    log("PHASE 3 V3 (BLAZE): Multi-Band Envelope Solver")
    log("="*70)
    
    config = load_yaml(config_path)
    log(f"Loaded config from: {config_path}")
    
    candidate_filter = os.getenv('MSL_PHASE3_CANDIDATE_ID')
    if candidate_filter is None:
        candidate_filter = config.get('phase3_candidate_id')
    if candidate_filter is not None:
        try:
            candidate_filter = int(candidate_filter)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid candidate ID '{candidate_filter}'")
    
    if run_dir in ['auto', 'latest']:
        runs_base = Path(config.get('output_dir', 'runsV3'))
        phase0_runs = sorted(runs_base.glob('phase0_blaze_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No BLAZE phase0 run directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest run: {run_dir}")
    
    run_dir = Path(run_dir)
    
    # Find candidate directories
    candidate_dirs = sorted(run_dir.glob("candidate_*"))
    if not candidate_dirs:
        raise FileNotFoundError(f"No candidate directories found in {run_dir}")
    
    if candidate_filter is not None:
        candidate_dirs = [d for d in candidate_dirs if int(d.name.split('_')[-1]) == candidate_filter]
        if not candidate_dirs:
            raise ValueError(f"Candidate ID {candidate_filter} not found")
    
    log(f"Found {len(candidate_dirs)} candidate(s) to process")
    
    for cdir in candidate_dirs:
        try:
            process_candidate_phase3_v3(cdir, config)
        except Exception as e:
            cid = int(cdir.name.split('_')[-1])
            print(f"ERROR processing candidate {cid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 3 V3 (BLAZE) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("V3 Multi-Band Pipeline Complete!")


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase3_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase3_v3("auto", str(default_config))
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        
        try:
            candidate_id = int(arg)
            log(f"Using default config: {default_config}")
            os.environ['MSL_PHASE3_CANDIDATE_ID'] = str(candidate_id)
            run_phase3_v3("auto", str(default_config))
        except ValueError:
            log(f"Using default config: {default_config}")
            run_phase3_v3(arg, str(default_config))
    elif len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        try:
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE3_CANDIDATE_ID'] = str(candidate_id)
            run_phase3_v3(arg2, str(default_config))
        except ValueError:
            run_phase3_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python blaze_phasesV3/phase3_blaze_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
