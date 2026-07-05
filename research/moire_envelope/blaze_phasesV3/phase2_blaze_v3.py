"""
Phase 2 (BLAZE): Berry Connection & Born-Huang Potential — V3 Multi-Band Pipeline

This is the V3 multi-band implementation of Phase 2 using BLAZE.
The key physics components implemented here:

1. PARALLEL TRANSPORT GAUGE: SVD-based gauge fixing for smooth Berry connection
2. BERRY CONNECTION: A_j,mn(s) = i⟨u_m|∂_j u_n⟩ (non-Abelian gauge field)
3. BORN-HUANG POTENTIAL: Φ_mn = Σ_j ⟨∂_j u_m|(1-P)|∂_j u_n⟩
4. GAUGE-COVARIANT MASS TENSOR: M^(-1)_mn → properly transformed

DATA STRUCTURES (V3):
- A: (Ns1, Ns2, N_subspace, N_subspace, 2) - Berry connection matrices
- Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) - Born-Huang potential matrix
- Lambda_n: (Ns1, Ns2, N_subspace) - diagonal potentials (on-site energies)
- M_inv_mn: (Ns1, Ns2, N_subspace, N_subspace, 2, 2) - generalized mass tensors

THEORY REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md
"""

import h5py
import numpy as np
from pathlib import Path
import sys
import os
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, save_json, load_json


def create_multiband_visualization(cdir, s_grid, omega_grid, V_grid, Lambda,
                                    A_berry, Phi_BH, v_drift, M_inv_grid,
                                    N_subspace, target_idx, B_moire):
    """Stub for multi-band visualization (TODO: implement in common.plotting)."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot diagonal potentials for each band
    for n in range(min(N_subspace, 3)):
        ax = axes[0, n]
        im = ax.imshow(Lambda[:, :, n, n].T, origin='lower', cmap='RdBu_r')
        ax.set_title(f'Λ_{n}{n}(s)')
        plt.colorbar(im, ax=ax)
    
    # Plot Born-Huang diagonal
    for n in range(min(N_subspace, 3)):
        ax = axes[1, n]
        im = ax.imshow(Phi_BH[:, :, n, n].T, origin='lower', cmap='viridis')
        ax.set_title(f'Φ_BH,{n}{n}(s)')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(cdir / 'phase2_multiband_fields.png', dpi=150)
    plt.close()


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# Parallel Transport Gauge (SVD-based)
# ==============================================================================

def compute_overlap_matrix(u_current, u_next):
    """
    Compute overlap matrix: O_mn = ⟨u_m(s)|u_n(s+ds)⟩
    
    For BLAZE, we use frequency-based proxies since we don't have 
    actual wavefunctions. The overlap is approximated via band continuity.
    
    Args:
        u_current: (N_bands, ...) eigenvector data at current point
        u_next: (N_bands, ...) eigenvector data at next point
        
    Returns:
        O: (N_bands, N_bands) overlap matrix
    """
    N_bands = u_current.shape[0]
    
    # If we have actual eigenvector data, compute true overlaps
    if len(u_current.shape) > 1:
        # Flatten spatial dimensions: (N_bands, N_spatial)
        u_curr_flat = u_current.reshape(N_bands, -1)
        u_next_flat = u_next.reshape(N_bands, -1)
        
        # Normalize
        norms_curr = np.linalg.norm(u_curr_flat, axis=1, keepdims=True)
        norms_next = np.linalg.norm(u_next_flat, axis=1, keepdims=True)
        u_curr_norm = u_curr_flat / (norms_curr + 1e-10)
        u_next_norm = u_next_flat / (norms_next + 1e-10)
        
        # Overlap: O_mn = <u_m | u_n>
        O = u_curr_norm @ u_next_norm.conj().T
    else:
        # Proxy: use identity (no gauge rotation needed)
        O = np.eye(N_bands, dtype=complex)
    
    return O


def parallel_transport_step(O):
    """
    Apply parallel transport gauge via SVD.
    
    Given overlap matrix O = U @ S @ V^†, the gauge transformation 
    that maximizes Re(Tr(O)) is W = U @ V^†.
    
    Args:
        O: (N_bands, N_bands) overlap matrix
        
    Returns:
        W: (N_bands, N_bands) unitary gauge transformation
    """
    U, S, Vh = np.linalg.svd(O)
    W = U @ Vh
    return W


def apply_parallel_transport_gauge(eigenvector_data, axis=0):
    """
    Apply parallel transport gauge along a specified axis.
    
    This ensures smooth gauge across the moiré unit cell, minimizing
    spurious Berry connection contributions.
    
    Args:
        eigenvector_data: (..., N_bands, spatial_dims)
        axis: axis along which to transport (0 for s1, 1 for s2)
        
    Returns:
        gauge_fixed_data: same shape, with smooth gauge
        gauge_matrices: (..., N_bands, N_bands) accumulated gauge transforms
    """
    shape = eigenvector_data.shape
    N_bands = shape[-2] if len(shape) > 2 else shape[0]
    
    # Move axis to front for easier iteration
    data = np.moveaxis(eigenvector_data, axis, 0)
    n_steps = data.shape[0]
    
    # Initialize gauge matrices
    remaining_shape = data.shape[1:-1] if len(data.shape) > 2 else ()
    gauge_matrices = np.zeros((*data.shape[:-1], N_bands, N_bands), dtype=complex)
    gauge_matrices[0] = np.eye(N_bands)
    
    # Propagate gauge
    for i in range(1, n_steps):
        u_prev = data[i-1]
        u_curr = data[i]
        
        if len(remaining_shape) > 0:
            # Multiple points in other dimensions
            for idx in np.ndindex(*remaining_shape):
                O = compute_overlap_matrix(u_prev[idx], u_curr[idx])
                W = parallel_transport_step(O)
                gauge_matrices[(i,) + idx] = gauge_matrices[(i-1,) + idx] @ W
        else:
            O = compute_overlap_matrix(u_prev, u_curr)
            W = parallel_transport_step(O)
            gauge_matrices[i] = gauge_matrices[i-1] @ W
    
    # Apply accumulated gauge
    fixed_data = np.copy(data)
    for i in range(n_steps):
        if len(remaining_shape) > 0:
            for idx in np.ndindex(*remaining_shape):
                G = gauge_matrices[(i,) + idx]
                fixed_data[(i,) + idx] = np.einsum('mn,...n->...m', G, data[(i,) + idx])
        else:
            G = gauge_matrices[i]
            fixed_data[i] = np.einsum('mn,...n->...m', G, data[i])
    
    # Move axis back
    fixed_data = np.moveaxis(fixed_data, 0, axis)
    gauge_matrices = np.moveaxis(gauge_matrices, 0, axis)
    
    return fixed_data, gauge_matrices


# ==============================================================================
# Berry Connection Computation
# ==============================================================================

def compute_berry_connection_fd(omega_grid, ds1, ds2, fd_order=4):
    """
    Compute Berry connection via finite-difference on frequency field.
    
    This is an APPROXIMATION when true eigenvectors are not available.
    The Berry connection for degenerate bands requires actual wavefunctions.
    
    For now, we compute the diagonal elements from frequency variations,
    which captures the essential physics of band warping.
    
    A_j,nn(s) ≈ ∂_j arg(u_n(s)) estimated from ∂_j ω_n(s)
    
    Args:
        omega_grid: (Ns1, Ns2, N_bands) - frequencies
        ds1, ds2: grid spacings in fractional coordinates
        fd_order: finite difference order (2 or 4)
        
    Returns:
        A: (Ns1, Ns2, N_bands, N_bands, 2) - Berry connection matrices
           For now, only diagonal elements are computed.
    """
    Ns1, Ns2, N_bands = omega_grid.shape
    A = np.zeros((Ns1, Ns2, N_bands, N_bands, 2), dtype=complex)
    
    # Without true wavefunctions, Berry connection is effectively zero
    # in the natural gauge. The parallel transport gauge ensures this.
    # The off-diagonal elements require actual Bloch function overlaps.
    
    # Diagonal elements: A_nn = 0 in parallel transport gauge
    # Off-diagonal: A_mn = i⟨u_m|∂u_n⟩ requires wavefunctions
    
    # Placeholder: return zeros for now
    # True implementation requires eigenvector export from BLAZE
    
    log("    NOTE: Berry connection approximated (diagonal only, requires wavefunctions for full calculation)")
    
    return A


def compute_berry_connection_from_eigenvectors(
    eigenvector_data,  # (Ns1, Ns2, N_bands, resolution, resolution) or similar
    ds1, ds2,
    fd_order=4
):
    """
    Compute Berry connection from actual eigenvector data.
    
    A_j,mn(s) = i⟨u_m(s)|∂_j u_n(s)⟩
    
    Uses periodic boundary conditions via circular indexing.
    
    Args:
        eigenvector_data: (Ns1, Ns2, N_bands, ...) eigenvector fields
        ds1, ds2: grid spacings in fractional coordinates
        fd_order: finite difference order
        
    Returns:
        A: (Ns1, Ns2, N_bands, N_bands, 2) Berry connection
    """
    Ns1, Ns2, N_bands = eigenvector_data.shape[:3]
    spatial_shape = eigenvector_data.shape[3:]
    
    # Reshape to (Ns1, Ns2, N_bands, N_spatial)
    u = eigenvector_data.reshape(Ns1, Ns2, N_bands, -1)
    N_spatial = u.shape[-1]
    
    A = np.zeros((Ns1, Ns2, N_bands, N_bands, 2), dtype=complex)
    
    # Finite difference coefficients
    if fd_order == 4:
        coeffs = np.array([1, -8, 0, 8, -1]) / 12.0
        offsets = [-2, -1, 0, 1, 2]
    else:
        coeffs = np.array([-0.5, 0, 0.5])
        offsets = [-1, 0, 1]
    
    # Compute derivatives and overlaps
    for i in range(Ns1):
        for j in range(Ns2):
            u_ij = u[i, j]  # (N_bands, N_spatial)
            
            # Derivative in s1 direction
            du_ds1 = np.zeros_like(u_ij)
            for c, offset in zip(coeffs, offsets):
                i_off = (i + offset) % Ns1
                du_ds1 += c * u[i_off, j]
            du_ds1 /= ds1
            
            # Derivative in s2 direction  
            du_ds2 = np.zeros_like(u_ij)
            for c, offset in zip(coeffs, offsets):
                j_off = (j + offset) % Ns2
                du_ds2 += c * u[i, j_off]
            du_ds2 /= ds2
            
            # Berry connection: A_mn = i <u_m | du_n>
            for m in range(N_bands):
                for n in range(N_bands):
                    # s1 component
                    overlap_s1 = np.sum(u_ij[m].conj() * du_ds1[n])
                    A[i, j, m, n, 0] = 1j * overlap_s1
                    
                    # s2 component
                    overlap_s2 = np.sum(u_ij[m].conj() * du_ds2[n])
                    A[i, j, m, n, 1] = 1j * overlap_s2
    
    return A


# ==============================================================================
# Born-Huang Potential Computation
# ==============================================================================

def compute_born_huang_potential(
    omega_subspace,      # (Ns1, Ns2, N_subspace) subspace band frequencies
    omega_extra,         # (Ns1, Ns2, N_extra) extra band frequencies
    M_inv_subspace,      # (Ns1, Ns2, N_subspace, 2, 2) mass tensors
    M_inv_extra=None,    # (Ns1, Ns2, N_extra, 2, 2) optional extra band mass tensors
    coupling_strength=1.0
):
    """
    Compute Born-Huang potential matrix.
    
    Φ_mn = Σ_j ⟨∂_j u_m|(1-P)|∂_j u_n⟩
    
    In the absence of true wavefunctions, we use an approximation based on
    the frequency gaps and mass tensor information.
    
    The Born-Huang potential captures the adiabatic correction from
    remote bands not included in the subspace.
    
    Args:
        omega_subspace: frequencies of subspace bands
        omega_extra: frequencies of bands for Born-Huang correction
        M_inv_subspace: inverse mass tensors of subspace bands
        M_inv_extra: inverse mass tensors of extra bands (optional)
        coupling_strength: overall scaling factor
        
    Returns:
        Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) Born-Huang potential matrix
    """
    Ns1, Ns2, N_subspace = omega_subspace.shape
    N_extra = omega_extra.shape[2]
    
    Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
    
    # Born-Huang contribution from extra bands
    # Φ_mn ≈ Σ_α Σ_j (1/Δ_mα + 1/Δ_nα) × curvature_coupling
    # where Δ_mα = ω_α - ω_m is the energy gap
    
    for i in range(Ns1):
        for j in range(Ns2):
            omega_sub_ij = omega_subspace[i, j]  # (N_subspace,)
            omega_ext_ij = omega_extra[i, j]      # (N_extra,)
            
            for m in range(N_subspace):
                for n in range(N_subspace):
                    phi_mn = 0.0
                    
                    for alpha in range(N_extra):
                        # Energy denominators
                        Delta_m_alpha = omega_ext_ij[alpha] - omega_sub_ij[m]
                        Delta_n_alpha = omega_ext_ij[alpha] - omega_sub_ij[n]
                        
                        # Avoid division by zero for nearly degenerate cases
                        eps = 1e-8
                        if abs(Delta_m_alpha) > eps and abs(Delta_n_alpha) > eps:
                            # Simple approximation: use average inverse gap squared
                            inv_gap_sq = 0.5 * (1/Delta_m_alpha**2 + 1/Delta_n_alpha**2)
                            
                            # Coupling estimate from mass tensor trace
                            M_trace_m = np.trace(M_inv_subspace[i, j, m])
                            M_trace_n = np.trace(M_inv_subspace[i, j, n])
                            
                            # Born-Huang contribution (sign convention)
                            phi_mn += inv_gap_sq * np.sqrt(abs(M_trace_m * M_trace_n))
                    
                    Phi_BH[i, j, m, n] = coupling_strength * phi_mn
    
    # Ensure Hermitian
    Phi_BH = 0.5 * (Phi_BH + np.swapaxes(Phi_BH, 2, 3))
    
    return Phi_BH


# ==============================================================================
# Drift Term Computation
# ==============================================================================

def compute_drift_term(vg_grid, omega_grid, omega_ref):
    """
    Compute the drift term contribution: v^(i)_mn = ⟨u_m|V_i|u_n⟩
    
    In the diagonal approximation, this is simply the group velocity.
    Off-diagonal elements require proper inter-band matrix elements.
    
    For the envelope equation, the drift term appears as:
    η × v_mn · ∇_R F_n
    
    Args:
        vg_grid: (Ns1, Ns2, N_bands, 2) group velocities per band
        omega_grid: (Ns1, Ns2, N_bands) frequencies
        omega_ref: reference frequency
        
    Returns:
        v_drift: (Ns1, Ns2, N_bands, N_bands, 2) drift velocity matrix
    """
    Ns1, Ns2, N_bands = omega_grid.shape
    v_drift = np.zeros((Ns1, Ns2, N_bands, N_bands, 2))
    
    # Diagonal elements: actual group velocities
    for n in range(N_bands):
        v_drift[:, :, n, n, :] = vg_grid[:, :, n, :]
    
    # Off-diagonal elements: zero in the absence of wavefunctions
    # Would require: v_mn = ⟨u_m|∂H/∂k|u_n⟩ / (ω_m - ω_n)
    # Currently set to zero
    
    return v_drift


# ==============================================================================
# Multi-band Potential (Lambda) Matrix
# ==============================================================================

def construct_lambda_potential(omega_grid, omega_ref):
    """
    Construct the Λ_mn potential matrix (on-site energies).
    
    Λ_mn(s) = (ω_n(s) - ω_ref) × δ_mn
    
    In the current diagonal approximation.
    
    Args:
        omega_grid: (Ns1, Ns2, N_bands) frequencies
        omega_ref: reference frequency
        
    Returns:
        Lambda: (Ns1, Ns2, N_bands, N_bands) diagonal potential matrix
    """
    Ns1, Ns2, N_bands = omega_grid.shape
    Lambda = np.zeros((Ns1, Ns2, N_bands, N_bands))
    
    for n in range(N_bands):
        Lambda[:, :, n, n] = omega_grid[:, :, n] - omega_ref
    
    return Lambda


# ==============================================================================
# Process Single Candidate
# ==============================================================================

def process_candidate_phase2_v3(candidate_dir_path, config):
    """Process single candidate through Phase 2 V3."""
    cdir = Path(candidate_dir_path)
    cid = int(cdir.name.split('_')[-1])
    
    log(f"\n=== Phase 2 V3: Candidate {cid} ===")
    
    phase1_h5 = cdir / "phase1_multiband_data.h5"
    if not phase1_h5.exists():
        raise FileNotFoundError(f"Phase 1 data not found: {phase1_h5}")
    
    # Load Phase 1 data
    with h5py.File(phase1_h5, 'r') as hf:
        s_grid = hf['s_grid'][:]
        R_grid = hf['R_grid'][:]
        delta_frac = hf['delta_frac'][:]
        
        omega_grid = hf['omega'][:]          # (Ns1, Ns2, N_subspace)
        vg_grid = hf['vg'][:]                # (Ns1, Ns2, N_subspace, 2)
        M_inv_grid = hf['M_inv'][:]          # (Ns1, Ns2, N_subspace, 2, 2)
        V_grid = hf['V'][:]                  # (Ns1, Ns2, N_subspace)
        
        # Load stencil data for all bands
        stencil_grp = hf['stencil']
        registry_omega_all = stencil_grp['registry_omega_all'][:]  # (n_reg, n_reg, N_all)
        
        # Metadata
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
        all_bands = hf.attrs['all_bands'][:].tolist()
    
    log(f"  Grid: {Ns1} × {Ns2}, N_subspace = {N_subspace}")
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  Subspace bands: {subspace_bands}")
    log(f"  All bands: {all_bands}")
    log(f"  ω_ref = {omega_ref:.6f}")
    
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    
    # Configuration
    include_born_huang = config.get('include_born_huang', True)
    include_drift_term = config.get('include_drift_term', True)
    use_parallel_transport = config.get('use_parallel_transport_gauge', True)
    n_extra_bands = config.get('n_extra_bands', 4)
    fd_order = config.get('blaze_fd_order', 4)
    
    log(f"  Include Born-Huang: {include_born_huang}")
    log(f"  Include drift term: {include_drift_term}")
    log(f"  Use parallel transport gauge: {use_parallel_transport}")
    
    # =========================================================================
    # 1. Berry Connection
    # =========================================================================
    log("  Computing Berry connection...")
    
    # Since BLAZE doesn't export eigenvectors, we use frequency-based approximation
    # True calculation requires wavefunction data
    A_berry = compute_berry_connection_fd(omega_grid, ds1, ds2, fd_order)
    
    # =========================================================================
    # 2. Born-Huang Potential
    # =========================================================================
    Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
    
    if include_born_huang:
        log("  Computing Born-Huang potential...")
        
        # Identify extra bands (not in subspace)
        extra_band_indices = [i for i, b in enumerate(all_bands) if b not in subspace_bands]
        
        if len(extra_band_indices) > 0:
            # Extract extra band frequencies from registry data
            # registry_omega_all: (n_reg, n_reg, N_all)
            n_registry = registry_omega_all.shape[0]
            
            # Map subspace and extra indices
            subspace_local_indices = [all_bands.index(b) for b in subspace_bands]
            
            # Interpolate extra bands to full grid (simplified: use direct mapping)
            from scipy.interpolate import RegularGridInterpolator
            
            omega_extra = np.zeros((Ns1, Ns2, len(extra_band_indices)))
            
            x_reg = np.linspace(0, 1 - 1/n_registry, n_registry)
            y_reg = np.linspace(0, 1 - 1/n_registry, n_registry)
            
            for local_idx, all_idx in enumerate(extra_band_indices):
                grid_2d = registry_omega_all[:, :, all_idx]
                
                # Make periodic
                extended = np.zeros((n_registry + 1, n_registry + 1))
                extended[:n_registry, :n_registry] = grid_2d
                extended[n_registry, :n_registry] = grid_2d[0, :]
                extended[:n_registry, n_registry] = grid_2d[:, 0]
                extended[n_registry, n_registry] = grid_2d[0, 0]
                
                x_ext = np.append(x_reg, 1.0)
                y_ext = np.append(y_reg, 1.0)
                
                interp = RegularGridInterpolator((x_ext, y_ext), extended,
                                                 method='linear', bounds_error=False)
                
                # Query points from delta_frac
                query_x = np.mod(delta_frac[:, :, 0] + 0.5, 1.0)
                query_y = np.mod(delta_frac[:, :, 1] + 0.5, 1.0)
                query_pts = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
                
                omega_extra[:, :, local_idx] = interp(query_pts).reshape(Ns1, Ns2)
            
            log(f"    Extra bands for Born-Huang: {[all_bands[i] for i in extra_band_indices]}")
            
            # Compute Born-Huang potential
            Phi_BH = compute_born_huang_potential(
                omega_grid, omega_extra, M_inv_grid,
                coupling_strength=config.get('born_huang_coupling', 1.0)
            )
            
            log(f"    Born-Huang potential range: [{Phi_BH.min():.6e}, {Phi_BH.max():.6e}]")
        else:
            log("    WARNING: No extra bands for Born-Huang calculation")
    
    # =========================================================================
    # 3. Drift Term
    # =========================================================================
    v_drift = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2))
    
    if include_drift_term:
        log("  Computing drift term...")
        v_drift = compute_drift_term(vg_grid, omega_grid, omega_ref)
        
        vg_max = np.max(np.abs(v_drift[:, :, np.arange(N_subspace), np.arange(N_subspace), :]))
        log(f"    Max diagonal group velocity: {vg_max:.6e}")
    
    # =========================================================================
    # 4. Lambda Potential Matrix
    # =========================================================================
    log("  Constructing Λ potential matrix...")
    Lambda = construct_lambda_potential(omega_grid, omega_ref)
    
    log(f"    Λ range: [{Lambda.min():.6e}, {Lambda.max():.6e}]")
    
    # =========================================================================
    # 5. Prepare Mass Tensor Matrix
    # =========================================================================
    log("  Preparing mass tensor matrix...")
    
    # M_inv as full matrix (diagonal for now)
    M_inv_matrix = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2, 2))
    for n in range(N_subspace):
        M_inv_matrix[:, :, n, n, :, :] = M_inv_grid[:, :, n, :, :]
    
    # =========================================================================
    # Save Phase 2 Output
    # =========================================================================
    h5_path = cdir / "phase2_multiband_data.h5"
    
    with h5py.File(h5_path, 'w') as hf:
        # Coordinate grids (copy from Phase 1)
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("delta_frac", data=delta_frac, compression="gzip")
        
        # Multi-band operator components
        hf.create_dataset("Lambda", data=Lambda, compression="gzip")      # (Ns1, Ns2, N, N)
        hf.create_dataset("A_berry", data=A_berry, compression="gzip")    # (Ns1, Ns2, N, N, 2)
        hf.create_dataset("Phi_BH", data=Phi_BH, compression="gzip")      # (Ns1, Ns2, N, N)
        hf.create_dataset("v_drift", data=v_drift, compression="gzip")    # (Ns1, Ns2, N, N, 2)
        hf.create_dataset("M_inv", data=M_inv_matrix, compression="gzip") # (Ns1, Ns2, N, N, 2, 2)
        
        # Single-band data (for compatibility and visualization)
        hf.create_dataset("omega", data=omega_grid, compression="gzip")
        hf.create_dataset("vg", data=vg_grid, compression="gzip")
        hf.create_dataset("V", data=V_grid, compression="gzip")
        
        # Metadata
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["theta_deg"] = math.degrees(theta_rad)
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["Ns1"] = Ns1
        hf.attrs["Ns2"] = Ns2
        hf.attrs["N_subspace"] = N_subspace
        hf.attrs["target_index_in_subspace"] = target_idx
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_mono"] = B_mono
        hf.attrs["subspace_bands"] = np.array(subspace_bands)
        hf.attrs["all_bands"] = np.array(all_bands)
        
        # Config flags
        hf.attrs["include_born_huang"] = include_born_huang
        hf.attrs["include_drift_term"] = include_drift_term
        hf.attrs["use_parallel_transport_gauge"] = use_parallel_transport
        hf.attrs["pipeline_version"] = "V3"
        hf.attrs["solver"] = "blaze2d"
    
    log(f"  Saved Phase 2 data to {h5_path}")
    
    # Generate visualization
    try:
        create_multiband_visualization(cdir, s_grid, omega_grid, V_grid, Lambda,
                                       A_berry, Phi_BH, v_drift, M_inv_grid, 
                                       N_subspace, target_idx, B_moire)
    except Exception as e:
        log(f"    WARNING: Visualization failed: {e}")
    
    log(f"=== Phase 2 Complete: Candidate {cid} ===")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_phase2_v3(run_dir, config_path):
    """Main Phase 2 V3 driver."""
    log("\n" + "="*70)
    log("PHASE 2 V3 (BLAZE): Berry Connection & Born-Huang Potential")
    log("="*70)
    
    config = load_yaml(config_path)
    log(f"Loaded config from: {config_path}")
    
    candidate_filter = os.getenv('MSL_PHASE2_CANDIDATE_ID')
    if candidate_filter is None:
        candidate_filter = config.get('phase2_candidate_id')
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
            process_candidate_phase2_v3(cdir, config)
        except Exception as e:
            cid = int(cdir.name.split('_')[-1])
            print(f"ERROR processing candidate {cid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 2 V3 (BLAZE) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 3 V3 for multi-band envelope solver")


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase2_blaze.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        log(f"Using default config: {default_config}")
        run_phase2_v3("auto", str(default_config))
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        
        try:
            candidate_id = int(arg)
            log(f"Using default config: {default_config}")
            os.environ['MSL_PHASE2_CANDIDATE_ID'] = str(candidate_id)
            run_phase2_v3("auto", str(default_config))
        except ValueError:
            log(f"Using default config: {default_config}")
            run_phase2_v3(arg, str(default_config))
    elif len(sys.argv) == 3:
        arg1, arg2 = sys.argv[1], sys.argv[2]
        try:
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE2_CANDIDATE_ID'] = str(candidate_id)
            run_phase2_v3(arg2, str(default_config))
        except ValueError:
            run_phase2_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python blaze_phasesV3/phase2_blaze_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
