"""
Phase 3 (MPB): Multi-Band Envelope Solver — V3 Multi-Band Pipeline

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
from scipy.ndimage import map_coordinates
from pathlib import Path
import sys
import os
import math
import time
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, save_json, load_json
from phasesV3.sanity_checks import log_sanity_block, phase3_sanity_report


def _plot_real_space_tiled(cdir, F_all, eigenvalues, mode_stats, B_moire, Ns1, Ns2, 
                           n_plot=64, filename_suffix="real_space_tiled"):
    """
    Plot the first n_plot modes (sorted by frequency) in tiled real space (Hex only).
    Uses B_moire to map real coords to s-grid and interpolates.
    """
    import matplotlib.pyplot as plt
    
    n_modes = len(F_all)
    n_plot = min(n_plot, n_modes)
    n_rows = int(np.ceil(np.sqrt(n_plot)))
    n_cols = int(np.ceil(n_plot / n_rows))
    
    # 4 inches per subplot might be too large for 8x8, reducing to 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    # Pre-compute coordinate mapping
    # Viewport: 3.0 moire periods (similar to phase 1 interactive plot big view)
    Lm = np.linalg.norm(B_moire[:, 0])
    R_span = 3.0 * Lm
    N_pts = 800 # Resolution for plot (increased 4x from 200)
    
    r_vec = np.linspace(-R_span/2, R_span/2, N_pts)
    X, Y = np.meshgrid(r_vec, r_vec)
    
    B_inv = np.linalg.inv(B_moire)
    
    # s = B_inv @ r
    s1_map = B_inv[0,0]*X + B_inv[0,1]*Y
    s2_map = B_inv[1,0]*X + B_inv[1,1]*Y
    
    s1_map = np.mod(s1_map, 1.0)
    s2_map = np.mod(s2_map, 1.0)
    
    # Map to array indices [0, Ns-1]
    coords = np.stack([s1_map * (Ns1-1), s2_map * (Ns2-1)])
    
    sorted_indices = np.argsort(eigenvalues)
    
    for i in range(n_plot):
        mode_idx = sorted_indices[i]
        ax = axes[i]
        
        # |F|^2 summed over bands -> (Ns1, Ns2)
        prob = np.sum(np.abs(F_all[mode_idx])**2, axis=2)
        
        # Interpolate
        real_space_prob = map_coordinates(prob, coords, order=1, mode='wrap', prefilter=False)
        
        # Draw holes overlay? 
        # For simplicity, just plot the potential density
        im = ax.imshow(real_space_prob, origin='lower', cmap='hot', 
                       extent=[-R_span/2, R_span/2, -R_span/2, R_span/2])
        
        omega = mode_stats[mode_idx]['omega']
        
        # Add scale bar or circle? 
        # Just simple title
        ax.set_title(f'M{mode_idx} $\\omega$={omega:.5f}')
        ax.axis('off')
        
    for i in range(n_plot, len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(f'Top {n_plot} Modes in Real Space (Tiled)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = cdir / f'phase3_envelope_modes_{filename_suffix}.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    log(f"    Saved real space plot: {plot_path}")


def plot_phase3_envelope_modes_v3(cdir, s_grid, F_all, eigenvalues, omega_ref,
                                   mode_stats, subspace_bands, B_moire, lattice_type='hex'):
    """
    Plot envelope modes.
    
    Creates FOUR plots:
    1. sorted by EIGENVALUE (Frequency)
    2. sorted by SPREAD (Ascending - most localized spatial variance)
    3. sorted by IPR (Descending - most localized peak density)
    4. sorted by CONFINEMENT EFFICIENCY (Descending - most energy in center)
    
    PLUS (if hex):
    5. Real Space Tiled Plot (sorted by Frequency)
    
    Grid size: 10x10 (100 modes max)
    """
    import matplotlib.pyplot as plt
    
    n_modes_total = len(eigenvalues)
    
    # Common plotting parameters
    n_rows, n_cols = 10, 10
    n_plot = min(n_rows * n_cols, n_modes_total)
    
    def create_sorted_plot(sort_indices, sort_name, filename_suffix):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
        axes = axes.flatten()
        
        for plot_idx in range(n_plot):
            mode_idx = sort_indices[plot_idx]
            ax = axes[plot_idx]
            
            # Total probability density |F|²
            prob = np.sum(np.abs(F_all[mode_idx])**2, axis=2)
            
            im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
            
            stats = mode_stats[mode_idx]
            omega = stats['omega']
            spread = stats['spread']
            ipr = stats.get('ipr', 0.0)
            conf = stats.get('confinement_efficiency', 0.0)
            dom_band = stats['dominant_band']
            dom_weight = stats['dominant_band_weight']
            
            # Updated Title Format
            ax.set_title(f'M{mode_idx} ω={omega:.5f}\n'
                         f'σ={spread:.2f} IPR={ipr:.1e} η={conf:.1%}', 
                         fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(n_plot, n_rows * n_cols):
            axes[idx].axis('off')
        
        plt.suptitle(f'Top {n_plot} Modes sorted by {sort_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
        plt.savefig(cdir / f'phase3_envelope_modes_{filename_suffix}.png', dpi=150)
        plt.close()

    # 1. Sort by Eigenvalue (Frequency) - Method 'SM' or 'LM' results are usually sorted by eigsh, but good to ensure
    # Note: eigenvalues passed in are already sorted by magnitude in main function, but let's be explicit
    sorted_by_eigenvalue = np.argsort(eigenvalues)
    create_sorted_plot(sorted_by_eigenvalue, "Frequency (Eigenvalue)", "by_frequency")
    
    # 2. Sort by Spread (Ascending) -> Smaller is more localized
    spreads = [s['spread'] for s in mode_stats]
    sorted_by_spread = np.argsort(spreads)
    create_sorted_plot(sorted_by_spread, "Spatial Spread (Ascending)", "by_spread")
    
    # 3. Sort by IPR (Descending) -> Larger is more localized/peaked
    iprs = [s.get('ipr', 0) for s in mode_stats]
    sorted_by_ipr = np.argsort(iprs)[::-1] # Descending
    create_sorted_plot(sorted_by_ipr, "Inverse Participation Ratio (Descending)", "by_ipr")
    
    # 4. Sort by Confinement Efficiency (Descending) -> Higher is better cavity
    confs = [s.get('confinement_efficiency', 0) for s in mode_stats]
    sorted_by_conf = np.argsort(confs)[::-1] # Descending
    create_sorted_plot(sorted_by_conf, "Confinement Efficiency (Descending)", "by_confinement")
    
    # 5. Real Space Tiled (Hex Only)
    if lattice_type == 'hex':
        Ns1, Ns2 = s_grid.shape[0], s_grid.shape[1]
        try:
            _plot_real_space_tiled(cdir, F_all, eigenvalues, mode_stats, B_moire, Ns1, Ns2)
        except Exception as e:
            print(f"Failed to generate real space plot: {e}")

    # Write detailed log file (preserving existing functionality)
    log_path = cdir / 'phase3_modes_detailed.log'
    with open(log_path, 'w') as f:
        f.write(f"Phase 3 Multi-Band Envelope Mode Analysis\n")
        f.write(f"Total modes: {n_modes_total}\n\n")
        f.write(f"=== TOP 20 BY CONFINEMENT EFFICIENCY ===\n")
        for rank, mode_idx in enumerate(sorted_by_conf[:20]):
             s = mode_stats[mode_idx]
             f.write(f"Rank {rank:2d} | Mode {mode_idx:3d}: Conf={s['confinement_efficiency']:.2%}, "
                     f"IPR={s['ipr']:.2e}, ω={s['omega']:.6f}\n")
    
    log(f"    Saved 4 visualization plots to {cdir}")



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


def build_multiband_drift_operator(v_drift, eta, Ns1, Ns2, N_bands, dR1, dR2, order=4):
    """
    Build the drift term operator: η × v_mn · (-i∇).
    
    VECTORIZED IMPLEMENTATION using Kronecker products.
    
    T = -i η Σ_μ V_μ (D_μ ⊗ I_bands)
    
    Args:
        v_drift: (Ns1, Ns2, N_bands, N_bands, 2) drift velocity matrix
        dR1, dR2: PHYSICAL grid spacings in R-coordinates (units of a)
        
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
        
    # 2. Construct Derivative operators in full space (in PHYSICAL R-coordinates)
    # D1 acts on R1: D1 ⊗ I_R2 ⊗ I_bands
    # D2 acts on R2: I_R1 ⊗ D2 ⊗ I_bands
    
    D1_base = build_periodic_derivative_matrix(Ns1, dR1, order)
    D2_base = build_periodic_derivative_matrix(Ns2, dR2, order)
    
    # Order of tensor product matches flattening: i (Ns1), j (Ns2), n (Nbands)
    # Flat index = (i * Ns2 + j) * Nbands + n
    # This corresponds to Tensor Product: Space1 ⊗ Space2 ⊗ Band
    
    D1_full = kron(D1_base, eye(Ns2 * N_bands), format='csr')
    D2_full = kron(eye(Ns1), kron(D2_base, eye(N_bands)), format='csr')
    
    # 3. Combine: T = -i η (V1 D1 + V2 D2)
    # Note: V operator is diagonal in spatial indices (multiplicative), D is derivative.
    # Order matters: v(r) * d/dr.
    
    # Correction: Use 1/(2pi) instead of eta to convert physical D to MPB k-units
    # D_MPB = (a / 2pi) * D_phys. With a=1, coeff is 1/(2pi)
    coeff = 1.0 / (2 * np.pi)
    
    T_drift = -1j * coeff * (V1_op @ D1_full + V2_op @ D2_full)
    
    # Hermitize (Weyl ordering): T → (T + T†)/2
    # Necessary because v(R)·D is not self-adjoint for position-dependent v.
    T_drift = (T_drift + T_drift.conj().T) / 2
    
    return T_drift


def _build_band_block_diagonal(vals, N_s, N_bands, N_total):
    """
    Build a sparse block-diagonal-in-space matrix from per-point band matrices.

    Args:
        vals: (N_s, N_bands, N_bands) complex array — one Nb×Nb block per spatial point
        N_s: number of spatial points
        N_bands: number of bands
        N_total: N_s * N_bands

    Returns:
        Sparse CSR matrix of shape (N_total, N_total)
    """
    k_offsets = np.arange(N_s) * N_bands
    m_idx, n_idx = np.meshgrid(np.arange(N_bands), np.arange(N_bands), indexing='ij')
    rows = (k_offsets[:, None, None] + m_idx[None, :, :]).ravel()
    cols = (k_offsets[:, None, None] + n_idx[None, :, :]).ravel()
    data = vals.ravel()
    return csr_matrix((data, (rows, cols)), shape=(N_total, N_total))


def build_multiband_kinetic_operator(
    M_inv, A_berry, eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire, order=4,
    include_offdiag_A=False
):
    """
    Build the kinetic term operator: (1/2) M^(-1)_ij D_i D_j (Hermitized).

    VECTORIZED IMPLEMENTATION using Kronecker products.

    When include_offdiag_A=False (default, legacy behaviour):
        Uses only diagonal Berry connection A_{nn} and diagonal M_inv_{nn}.
        The kinetic term is block-diagonal in bands.

    When include_offdiag_A=True:
        Uses the full Berry connection matrix A_{mn} in the covariant
        derivative D_i = -i∂_i - A_i.  This adds:
          • Diamagnetic:  (1/2) Σ_{ij,p} M_{ij,pp} A_{i,mp} A_{j,pn}   [off-diag in bands]
          • Paramagnetic: -i Σ_{ij} M_{ij,mm} A_{j,mn} ∂_i + h.c.      [off-diag in bands]
        The Hermitization K → (K+K†)/2 ensures self-adjointness.

    The operator M(x)*L is NOT self-adjoint when M varies in space.
    We symmetrize: K -> (K + K^dag)/2 to enforce Hermiticity (F03 fix).

    Args:
        M_inv: (Ns1, Ns2, N_bands, N_bands, 2, 2) inverse effective mass tensor
        A_berry: (Ns1, Ns2, N_bands, N_bands, 2) Berry connection
        dR1, dR2: PHYSICAL grid spacings in R-coordinates (units of a).
                  The kinetic term uses d/dR, so these must be L_moire/N.
        include_offdiag_A: if True, use full off-diagonal A_berry in covariant derivative
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands
    
    scale_factor = 1.0 / (2 * np.pi)**2
    # Correction: removed eta**2. The scale_factor 1/(2pi)^2 correctly converts D_phys^2 to k_MPB^2
    prefactor = 0.5 * scale_factor
    
    # Base operators (using PHYSICAL grid spacing dR, not dimensionless ds)
    # This ensures the Laplacian eigenvalues are correct: -(2πn/L_moire)² not -(2πn)²
    L1 = build_periodic_laplacian_matrix(Ns1, dR1, order)
    L2 = build_periodic_laplacian_matrix(Ns2, dR2, order)
    D1 = build_periodic_derivative_matrix(Ns1, dR1, order)
    D2 = build_periodic_derivative_matrix(Ns2, dR2, order)
    
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
    
    # ── Berry connection terms ────────────────────────────────────────
    A_berry_reshaped = A_berry.reshape(N_s, N_bands, N_bands, 2)

    if include_offdiag_A:
        # ── FULL covariant derivative with off-diagonal A ────────────
        # Uses ALL A_{mn} components and sums over intermediate bands p.
        #
        # M_diag[k, p, i, j] = M_inv[k, p, p, i, j]  (diagonal in bands)
        M_diag = np.zeros((N_s, N_bands, 2, 2), dtype=complex)
        for p in range(N_bands):
            M_diag[:, p] = M_inv_reshaped[:, p, p]

        # -- Diamagnetic A² term --
        # A²[k,m,n] = Σ_{ij,p} M_inv_{ij,pp}(k) · A_{i,mp}(k) · A_{j,pn}(k)
        #
        # Step 1: B[k,p,n,i] = Σ_j M[k,p,i,j] · A[k,p,n,j]
        B_ma = np.einsum('kpij,kpnj->kpni', M_diag, A_berry_reshaped)
        # Step 2: A²[k,m,n] = Σ_{p,i} A[k,m,p,i] · B[k,p,n,i]
        A2_val = np.einsum('kmpi,kpni->kmn', A_berry_reshaped, B_ma)

        if np.max(np.abs(A2_val)) > 1e-15:
            A2_op = _build_band_block_diagonal(A2_val, N_s, N_bands, N_total)
            K_op = K_op + prefactor * A2_op

        # -- Paramagnetic cross-terms --
        # VA[k,m,n,j] = Σ_i M_diag[k,m,i,j] · A[k,m,n,i]
        # para = -i · prefactor · Σ_j blkdiag(VA[:,:,:,j]) @ D_j_full
        # Hermitization adds the adjoint (D† @ blkdiag(VA†)) automatically.
        VA = np.einsum('kmij,kmni->kmnj', M_diag, A_berry_reshaped)

        va_max = np.max(np.abs(VA))
        if va_max > 1e-15:
            VA_mat_0 = _build_band_block_diagonal(VA[:, :, :, 0], N_s, N_bands, N_total)
            VA_mat_1 = _build_band_block_diagonal(VA[:, :, :, 1], N_s, N_bands, N_total)
            para_op = -1j * prefactor * (VA_mat_0 @ D1_full + VA_mat_1 @ D2_full)
            K_op = K_op + para_op

    else:
        # ── LEGACY: diagonal-only Berry terms ───────────────────────
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

    # --- Hermiticity fix (F03) ---
    # The operator M(x)·L is NOT self-adjoint when M varies in space.
    # The theory's kinetic operator comes from the self-adjoint parent
    # L^(2) = -∇·(ε^{-1}∇), so the projected form must be Hermitian.
    # The notation M^{-1}(-iD)(-iD) is implicitly Weyl-ordered.
    # Fix: symmetrize K → (K + K†)/2.
    K_op = (K_op + K_op.T.conj()) / 2
    
    return K_op


def build_multiband_born_huang_operator(Phi_BH, eta, Ns1, Ns2, N_bands):
    """
    Build Born-Huang potential operator: Φ_mn(s).
    
    VECTORIZED IMPLEMENTATION.
    """
    # Reuse Potential operator logic
    V_BH = build_multiband_potential_operator(Phi_BH, None)
    # Correction: removed eta**2. Phi_BH is computed with physical derivatives,
    # so it already has correct 1/L^2 scaling via dR
    return V_BH


# ==============================================================================
# M_inv Regularization (F03)
# ==============================================================================

def _regularize_M_inv(M_inv, max_trace):
    """
    Regularize the inverse effective mass tensor by clamping eigenvalues.
    
    At grid points near band degeneracies, M_inv diverges (∝ 1/Δ_gap).
    This clamps the 2×2 mass tensor eigenvalues to [-max_trace, max_trace]
    while preserving the eigenvector structure.
    
    Args:
        M_inv: (Ns1, Ns2, N_bands, N_bands, 2, 2) mass tensor
        max_trace: maximum allowed absolute eigenvalue
        
    Returns:
        M_inv_reg: regularized mass tensor (same shape)
    """
    M_reg = M_inv.copy()
    Ns1, Ns2, Nb, _, _, _ = M_inv.shape
    n_clamped = 0
    for n in range(Nb):
        for i in range(Ns1):
            for j in range(Ns2):
                M = M_reg[i, j, n, n, :, :]
                eigs, vecs = np.linalg.eigh(M)
                if np.max(np.abs(eigs)) > max_trace:
                    eigs_clamped = np.clip(eigs, -max_trace, max_trace)
                    M_reg[i, j, n, n, :, :] = vecs @ np.diag(eigs_clamped) @ vecs.T
                    n_clamped += 1
    if n_clamped > 0:
        total = Ns1 * Ns2 * Nb
        log(f"      M_inv regularized: clamped {n_clamped}/{total} points "
            f"({100*n_clamped/total:.1f}%) to max |eig|={max_trace}")
    return M_reg


# ==============================================================================
# Full Hamiltonian Assembly
# ==============================================================================

def assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv, A_berry, Phi_BH,
    eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
    include_drift=True, include_kinetic=True, include_born_huang=True,
    order=4, M_inv_max_trace=None, include_offdiag_A=False
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
        eta: small parameter (a / L_moire)
        Ns1, Ns2: grid dimensions
        N_bands: number of bands
        dR1, dR2: PHYSICAL grid spacings in R-coordinates (units of a).
                  Must be L_moire/N, NOT 1/N! The envelope equation uses
                  derivatives w.r.t. physical coordinates R, not dimensionless s.
        B_moire: moiré basis vectors
        include_drift: include drift term
        include_kinetic: include kinetic term
        include_born_huang: include Born-Huang term
        order: finite difference order
        M_inv_max_trace: if set, clamp |Tr(M_inv)| at each grid point to this
                         value. This regularizes hot spots near band degeneracies
                         where M_inv diverges due to 1/Δ_gap (F03).
        include_offdiag_A: if True, use full off-diagonal Berry connection
                           A_{mn} in the covariant derivative (enables interband
                           coupling through diamagnetic + paramagnetic terms).
        
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
            v_drift, eta, Ns1, Ns2, N_bands, dR1, dR2, order
        )
        H = H + T_drift
        log(f"      nnz = {T_drift.nnz}")
    
    # Kinetic term
    if include_kinetic:
        log("    - Kinetic operator K...")
        M_inv_use = M_inv
        if M_inv_max_trace is not None:
            # Regularize M_inv: clamp eigenvalues of M_inv tensor at each point
            # This handles divergent mass at near-degeneracy hot spots (F03)
            M_inv_use = _regularize_M_inv(M_inv, M_inv_max_trace)
        K_op = build_multiband_kinetic_operator(
            M_inv_use, A_berry, eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire, order,
            include_offdiag_A=include_offdiag_A
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
# Sigma (shift-invert target) Selection
# ==============================================================================

def compute_sigma(Lambda, M_inv, target_idx, candidate_type=None):
    """
    Compute the optimal sigma (shift-invert target) for eigsh.

    Rules (from CHOOSING_SIGMA.md):
    - Dirac pair (candidate_type='dirac_cone' or auto-detected Nb=2):
      σ = spatial mean of manifold center = <(1/Nb) Σ_n Λ_nn(R)>_R
    - Isolated extremum (candidate_type='band_minimum' or Nb>2):
      σ = V_max of target band if HOLE, V_min if ELECTRON
      (based on sign of Tr(M⁻¹) for the target band)

    Args:
        Lambda: (Ns1, Ns2, Nb, Nb) potential matrix
        M_inv: (Ns1, Ns2, Nb, Nb, 2, 2) inverse mass tensor
        target_idx: target band index within subspace
        candidate_type: 'dirac_cone', 'band_minimum', or None (auto-detect)

    Returns:
        (sigma, info): sigma value and dict with diagnostic info
    """
    Nb = Lambda.shape[2]

    # Auto-detect candidate type from Nb if not provided
    if candidate_type is None:
        candidate_type = 'dirac_cone' if Nb == 2 else 'band_minimum'

    # Per-band analysis (always computed for diagnostics)
    per_band_info = []
    for n in range(Nb):
        tr = M_inv[..., n, n, 0, 0] + M_inv[..., n, n, 1, 1]
        mean_tr = float(np.mean(tr))
        V_n = Lambda[..., n, n].real
        V_min_n, V_max_n = float(np.min(V_n)), float(np.max(V_n))
        band_type = 'HOLE' if mean_tr < 0 else 'ELECTRON'
        per_band_info.append({
            'mean_trace': mean_tr, 'type': band_type,
            'V_min': V_min_n, 'V_max': V_max_n,
        })

    info = {
        'candidate_type': candidate_type,
        'Nb': Nb,
        'target_idx': target_idx,
        'per_band': per_band_info,
    }

    if candidate_type == 'dirac_cone':
        # Dirac pair: target the center of the manifold
        # σ = spatial mean of (1/Nb) Σ_n Λ_nn(R)
        manifold_center = np.mean([Lambda[..., n, n].real for n in range(Nb)], axis=0)
        sigma = float(np.mean(manifold_center))
        info['method'] = 'manifold_center'
        info['manifold_center_spatial_mean'] = sigma
    else:
        # Isolated extremum: use target band's edge based on mass character
        target_info = per_band_info[target_idx]
        if target_info['type'] == 'HOLE':
            sigma = target_info['V_max']
            info['method'] = f"V_max of target band {target_idx} [HOLE]"
        else:
            sigma = target_info['V_min']
            info['method'] = f"V_min of target band {target_idx} [ELECTRON]"

    info['sigma'] = sigma
    return sigma, info


def log_sigma_info(info, log_fn=None):
    """Log diagnostic info from compute_sigma()."""
    if log_fn is None:
        log_fn = log
    log_fn(f"  Sigma Selection ({info['candidate_type']}, Nb={info['Nb']}):")
    for n, b in enumerate(info['per_band']):
        log_fn(f"    Band {n}: Tr(M⁻¹)={b['mean_trace']:+.4f} ({b['type']})  "
               f"V∈[{b['V_min']:.6f}, {b['V_max']:.6f}]")
    log_fn(f"    -> σ = {info['sigma']:.6f} ({info['method']})")


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

    # --- CAVITY METRICS ---
    # 1. Inverse Participation Ratio (IPR)
    #    IPR = ∫ |ψ|^4 dV / (∫ |ψ|^2 dV)^2
    #    For localized states, IPR is large (order 1/volume_of_localization).
    #    For delocalized states, IPR is small (order 1/volume_total).
    #    prob_total is already |ψ|^2 normalized to sum to 1.
    #    So IPR = sum(prob_total^2).
    ipr = np.sum(prob_total**2)

    # 2. Confinement Efficiency (eta_conf)
    #    Fraction of energy in the central 25% area of the supercell.
    #    Region: [N/4, 3N/4] along both axes.
    #    This metric identifies modes that look like point-defect cavities.
    
    # Handle wrapping/periodic boundaries for centroid?
    # For now, just assume center is roughly (Ns1/2, Ns2/2) based on potential.
    # We measure confinement relative to the geometric center of the cell.
    
    n_start_1, n_end_1 = Ns1 // 4, 3 * Ns1 // 4
    n_start_2, n_end_2 = Ns2 // 4, 3 * Ns2 // 4
    
    # Sum probability in the central box
    conf_efficiency = np.sum(prob_total[n_start_1:n_end_1, n_start_2:n_end_2])

    
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
        'ipr': float(ipr),
        'confinement_efficiency': float(conf_efficiency)
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
        moire_length = float(hf.attrs['moire_length']) if 'moire_length' in hf.attrs else None
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_subspace = int(hf.attrs['N_subspace'])
        target_idx = int(hf.attrs['target_index_in_subspace'])
        B_moire = hf.attrs['B_moire']
        B_mono = hf.attrs['B_mono']
        subspace_bands = hf.attrs['subspace_bands'][:].tolist()
        
    # Load Meta for lattice type
    meta_path = cdir / "phase0_meta.json"
    lattice_type = 'hex' # default
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            lattice_type = meta.get('lattice_type', 'hex')
        except Exception:
            pass
    
    log(f"  Grid: {Ns1} × {Ns2}, N_subspace = {N_subspace}")
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  Lattice: {lattice_type}")
    log(f"  ω_ref = {omega_ref:.6f}")
    
    # =========================================================================
    # CRITICAL: Grid spacing in PHYSICAL (R) coordinates, not dimensionless (s)
    # =========================================================================
    # The envelope Hamiltonian uses derivatives w.r.t physical R (units of a).
    # dR = L_moire / N
    #
    # SCALING FACTORS:
    # 1. Kinetic: (1/2) D_R M^-1 D_R
    #    - Requires NO extra eta^2 factor if D_R is used directly.
    #    - Code implementation applies `prefactor = 0.5 * (1/2pi)^2` to convert
    #      D_phys to MPB k-units (where band curvature is defined).
    #
    # 2. Drift: v_g * D_R
    #    - Use coefficient 1/(2pi) to align units.
    #
    # 3. Born-Huang: Phi_BH
    #    - Phi_BH computed via physical derivatives is already energy-dimensioned.
    #    - No extra factors needed.
    # =========================================================================
    
    # Prefer the length stored by Phase 1/2. Recomputing it from the basis can
    # be error-prone if the wrong axis is used or the basis is not orthogonal.
    if moire_length is None:
        L_moire = np.linalg.norm(B_moire[:, 0])
    else:
        L_moire = moire_length
    
    # Physical grid spacing in R-coordinates (units of monolayer lattice constant a)
    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2
    
    log(f"  L_moire = {L_moire:.4f} a")
    log(f"  Physical grid spacing: dR = {dR1:.4f} a")
    
    # Configuration
    n_modes = config.get('n_modes', 100)  # UPDATED DEFAULT: 100 modes
    include_drift = config.get('include_drift_term', True)
    include_kinetic = config.get('include_kinetic_term', True)
    include_born_huang = config.get('include_born_huang', True)
    include_offdiag_A = config.get('include_offdiag_A', False)
    fd_order = config.get('fd_order', 4)
    M_inv_max_trace = config.get('M_inv_max_trace', None)
    sigma_shift = config.get('sigma_shift', None)
    candidate_type = config.get('candidate_type', None)
    
    # --- SIGMA SELECTION via compute_sigma() ---
    if sigma_shift is None:
        sigma_shift, sigma_info = compute_sigma(
            Lambda, M_inv, target_idx, candidate_type=candidate_type
        )
        log_sigma_info(sigma_info)
    else:
        log(f"    -> Using user-configured sigma: {sigma_shift}")
    
    log(f"  Computing {n_modes} modes")
    log(f"  Include drift: {include_drift}")
    log(f"  Include kinetic: {include_kinetic}")
    log(f"  Include Born-Huang: {include_born_huang}")
    log(f"  Include off-diagonal A: {include_offdiag_A}")
    if M_inv_max_trace is not None:
        log(f"  M_inv trace clamp: {M_inv_max_trace}")
    
    # Assemble Hamiltonian
    log("  Assembling Hamiltonian...")
    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        eta, Ns1, Ns2, N_subspace, dR1, dR2, B_moire,
        include_drift, include_kinetic, include_born_huang,
        fd_order, M_inv_max_trace=M_inv_max_trace,
        include_offdiag_A=include_offdiag_A
    )

    phase3_report = phase3_sanity_report(H)
    log_sanity_block(log, 'Phase 3 sanity checks', phase3_report)
    save_json(phase3_report, cdir / 'phase3_sanity_checks.json')
    
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
            f"spread = {stats['spread']:.3f}, "
            f"IPR = {stats['ipr']:.2e}, "
            f"ConfEff = {stats['confinement_efficiency']:.2%}")
    
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
        hf.attrs["include_offdiag_A"] = include_offdiag_A
        if M_inv_max_trace is not None:
            hf.attrs["M_inv_max_trace"] = M_inv_max_trace
        hf.attrs["fd_order"] = fd_order
        hf.attrs["pipeline_version"] = "V3"
    
    log(f"  Saved Phase 3 data to {h5_path}")
    
    # Save mode statistics as JSON
    save_json(mode_stats, cdir / "phase3_mode_stats.json")
    
    # Generate visualizations
    try:
        plot_phase3_envelope_modes_v3(
            cdir, s_grid, F_all, eigenvalues, omega_ref, 
            mode_stats, subspace_bands, B_moire, lattice_type
        )
    except Exception as e:
        log(f"    WARNING: Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    log(f"=== Phase 3 Complete: Candidate {cid} ===")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_phase3_v3(run_dir, config_path):
    """Main Phase 3 V3 driver."""
    log("\n" + "="*70)
    log("PHASE 3 V3 (MPB): Multi-Band Envelope Solver")
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
        phase0_runs = sorted(runs_base.glob('phase0_mpb_*'))
        if not phase0_runs:
            raise FileNotFoundError(f"No MPB phase0 run directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        log(f"Auto-selected latest run: {run_dir}")
    else:
        # Check if path exists as provided
        p_run = Path(run_dir)
        if not p_run.exists():
            # Check if it's a name inside output_dir
            runs_base = Path(config.get('output_dir', 'runsV3'))
            if (runs_base / run_dir).exists():
                run_dir = runs_base / run_dir
                log(f"Found run directory in output folder: {run_dir}")
    
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
    log("PHASE 3 V3 (MPB) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("V3 Multi-Band Pipeline Complete!")


# ==============================================================================
# Moiré Band Structure: H(K) solved at each K-point
# ==============================================================================

def solve_moire_band_structure(
    K_points,
    stencil_data,
    delta_frac_grid,
    A_berry, Phi_BH,
    eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
    omega_ref,
    n_modes=10,
    include_drift=True, include_kinetic=True, include_born_huang=True,
    order=4, include_offdiag_A=False,
    fit_order='quadratic',
    sigma=None,
):
    """
    Compute moiré band structure by solving H(K) at each K-point on a path.
    
    For each K-point, the band data (ω, vg, M_inv) is re-interpolated from the
    raw stencil using polynomial fit, giving K-dependent effective parameters.
    Then a new Hamiltonian H(K) is assembled and diagonalized.
    
    Args:
        K_points: (N_K, 2) array of moiré K-points in MPB reciprocal units (2π/a)
        stencil_data: dict with keys:
            - stencil_omega: (n_reg, n_reg, N_all, n_stencil, n_stencil)
            - registry_omega0, registry_vg, registry_M_inv: center values
            - offsets, dk, n_registry
            - all_bands, subspace_bands
        delta_frac_grid: (Ns1, Ns2, 2) fractional registry coordinates
        A_berry: (Ns1, Ns2, N_bands, N_bands, 2) Berry connection
        Phi_BH: (Ns1, Ns2, N_bands, N_bands) Born-Huang potential
        eta: small parameter (a/L_moire)
        Ns1, Ns2: spatial grid dimensions
        N_bands: number of subspace bands
        dR1, dR2: physical grid spacings
        B_moire: moiré basis vectors
        omega_ref: reference frequency
        n_modes: number of eigenvalues to compute at each K
        fit_order: 'quadratic' or 'quartic' for stencil polynomial fit
        sigma: shift-invert target (None for auto)
        
    Returns:
        K_points: (N_K, 2) K-points used
        band_energies: (N_K, n_modes) eigenvalues at each K
    """
    from phasesV3.stencil_interpolation import (
        fit_stencil_polynomials, interpolate_band_data_at_K,
    )
    
    N_K = len(K_points)
    band_energies = np.full((N_K, n_modes), np.nan)
    
    # Pre-fit polynomials once (expensive, but amortized over all K)
    log(f"  Pre-fitting stencil polynomials ({fit_order})...")
    poly_coeffs, rms_residuals = fit_stencil_polynomials(
        stencil_data['stencil_omega'],
        stencil_data['offsets'],
        stencil_data['dk'],
        fit_order=fit_order,
    )
    max_rms = np.nanmax(rms_residuals)
    mean_rms = np.nanmean(rms_residuals)
    log(f"    Polynomial fit RMS: mean={mean_rms:.2e}, max={max_rms:.2e}")
    
    all_bands = stencil_data['all_bands']
    subspace_bands = stencil_data['subspace_bands']
    
    log(f"  Solving H(K) at {N_K} K-points for {n_modes} modes each...")
    pbar = tqdm(total=N_K, desc="  H(K) solve", unit="K-pt")
    
    for ik, K in enumerate(K_points):
        # 1. Interpolate band data at this K
        omega_K, vg_K, M_inv_K = interpolate_band_data_at_K(
            stencil_data['stencil_omega'],
            stencil_data['registry_omega0'],
            stencil_data['registry_vg'],
            stencil_data['registry_M_inv'],
            stencil_data['offsets'],
            stencil_data['dk'],
            stencil_data['n_registry'],
            delta_frac_grid,
            K,
            all_bands, subspace_bands,
            fit_order=fit_order,
            poly_coeffs=poly_coeffs,
        )
        
        # 2. Build Lambda_K = ω_n(K, s) - omega_ref (K-dependent potential)
        Lambda_K = np.zeros((Ns1, Ns2, N_bands, N_bands))
        for n in range(N_bands):
            Lambda_K[:, :, n, n] = omega_K[:, :, n] - omega_ref
        
        # 3. Build v_drift_K (off-diagonal stays from Berry/Phase 2, diagonal from stencil)
        v_drift_K = np.zeros((Ns1, Ns2, N_bands, N_bands, 2))
        for n in range(N_bands):
            v_drift_K[:, :, n, n, :] = vg_K[:, :, n, :]
        
        # 4. Build M_inv_K (diagonal blocks from stencil)
        M_inv_full_K = np.zeros((Ns1, Ns2, N_bands, N_bands, 2, 2))
        for n in range(N_bands):
            M_inv_full_K[:, :, n, n, :, :] = M_inv_K[:, :, n, :, :]
        
        # 5. Assemble H(K)
        H_K = assemble_multiband_hamiltonian(
            Lambda_K, v_drift_K, M_inv_full_K, A_berry, Phi_BH,
            eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
            include_drift=include_drift, include_kinetic=include_kinetic,
            include_born_huang=include_born_huang, order=order,
            include_offdiag_A=include_offdiag_A,
        )
        
        # 6. Solve
        eigs, _ = solve_multiband_envelope(H_K, n_modes, sigma=sigma)
        band_energies[ik, :len(eigs)] = eigs
        
        pbar.update(1)
    
    pbar.close()
    log(f"  Moiré band structure complete: {N_K} K-points × {n_modes} modes")
    
    return K_points, band_energies


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase3_mpb.yaml"


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
            # Case 1: [candidate_id] [run_dir]
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE3_CANDIDATE_ID'] = str(candidate_id)
            run_phase3_v3(arg2, str(default_config))
        except ValueError:
            # Case 2: [run_dir] [candidate_id]
            try:
                candidate_id = int(arg2)
                default_config = get_default_config_path()
                if not default_config.exists():
                    raise SystemExit(f"Default config not found: {default_config}")
                os.environ['MSL_PHASE3_CANDIDATE_ID'] = str(candidate_id)
                run_phase3_v3(arg1, str(default_config))
            except ValueError:
                # Case 3: [run_dir] [config_path]
                run_phase3_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python phasesV3/phase3_mpb_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
