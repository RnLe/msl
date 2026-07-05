"""
Bloch Field Storage and Born-Huang Computation Framework

This module handles the extraction, storage, and processing of Bloch functions
u_n(r; R) from MPB for proper Born-Huang potential computation.

The Born-Huang potential is defined as (from theory doc section 7.2):
    Φ_mn(R) = Σ_j ⟨∂_{R_j} u_m | (1-P) | ∂_{R_j} u_n⟩_Ω

where:
- u_n(r; R) is the periodic Bloch function at moiré position R
- P is the projector onto the chosen N-band subspace
- ∂_{R_j} denotes derivative with respect to slow coordinate R_j
- ⟨·|·⟩_Ω is the inner product over the small unit cell

IMPLEMENTATION APPROACH:
1. Phase 1: Extract and store Bloch functions u_n(r; R) at each registry point
2. Phase 2: Compute ∂u_n/∂R via finite differences, then compute Φ_mn

STORAGE FORMAT:
- HDF5 dataset: 'bloch_fields' with shape (Ns1, Ns2, N_bands, Nx, Ny, N_components)
- Complex64 data type
- N_components = 3 for full 3D field, but typically 1 for 2D TM (Ez only)

REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os

try:
    import meep as mp
    from meep import mpb
except ImportError:
    mp = None
    mpb = None


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# Phase 1: Bloch Field Extraction from MPB
# ==============================================================================

def extract_bloch_field_from_mpb(
    ms,  # MPB ModeSolver after running
    band_index: int,
    polarization: str = 'TM',
    normalize: bool = True
) -> np.ndarray:
    """
    Extract the periodic Bloch function u_n(r) from MPB.
    
    Uses bloch_phase=False to get the periodic envelope, not e^{ikr} u_n(r).
    
    Args:
        ms: MPB ModeSolver instance after running
        band_index: 1-based band index for MPB
        polarization: 'TM' or 'TE'
        normalize: if True, normalize so ⟨u|u⟩_Ω = 1
        
    Returns:
        u_field: complex numpy array of shape (Nx, Ny, N_components)
                 For 2D TM: only z-component is non-zero
                 For 2D TE: x,y components are in-plane
    """
    # Get the electric field without Bloch phase
    # This gives us the periodic part u_n(r)
    efield = ms.get_efield(band_index, bloch_phase=False)
    
    # efield is an MPBArray with shape (Nx, Ny, Nz=1, 3) for 2D
    # Components are (Ex, Ey, Ez) stored as complex numbers
    
    # For 2D, squeeze out the z-dimension
    if efield.ndim == 4 and efield.shape[2] == 1:
        efield = efield[:, :, 0, :]  # Shape: (Nx, Ny, 3)
    
    efield = np.array(efield, dtype=np.complex64)
    
    # Normalize: ⟨u|u⟩_Ω = (1/|Ω|) ∫ |u|² dr = 1
    # For discrete grid: sum |u|² / N_pixels = 1
    if normalize:
        Nx, Ny = efield.shape[:2]
        norm_sq = np.sum(np.abs(efield)**2) / (Nx * Ny)
        if norm_sq > 1e-12:
            efield = efield / np.sqrt(norm_sq)
    
    return efield


def extract_all_bloch_fields(
    ms,
    band_indices: List[int],
    polarization: str = 'TM'
) -> np.ndarray:
    """
    Extract Bloch fields for all specified bands.
    
    Args:
        ms: MPB ModeSolver after running
        band_indices: list of 0-based band indices
        polarization: 'TM' or 'TE'
        
    Returns:
        u_fields: array of shape (N_bands, Nx, Ny, 3)
    """
    fields = []
    for band_idx in band_indices:
        # MPB uses 1-based indexing
        u = extract_bloch_field_from_mpb(ms, band_idx + 1, polarization)
        fields.append(u)
    
    return np.stack(fields, axis=0)


def compute_single_point_with_fields(args):
    """
    Worker function that computes bands AND extracts Bloch fields at one registry point.
    
    This is a modified version of _compute_single_registry_point that also
    extracts the Bloch function data needed for Born-Huang.
    
    Args:
        args: tuple of (ix, iy, delta_frac, params_dict)
              params_dict must include 'export_bloch_fields': True
              
    Returns:
        tuple of (ix, iy, result_dict)
        result_dict includes 'bloch_fields' if export_bloch_fields is True
    """
    ix, iy, delta_frac, params = args
    
    # Import here to avoid circular imports
    from phasesV3.phase1_mpb_v3 import create_mpb_geometry, create_mpb_solver
    
    # Create geometry with this registry shift
    geometry, lattice, bg_eps = create_mpb_geometry(
        params['lattice_type'], params['r_over_a'], params['eps_bg'], 
        eps_hole=params.get('eps_hole', 1.0), delta_frac=delta_frac
    )
    
    # Create solver
    ms = create_mpb_solver(
        geometry, lattice, bg_eps, params['max_band'], 
        params['resolution'], params['polarization']
    )
    
    # Run at the central k-point only first to get fields
    k0 = params['k0']
    ms.k_points = [mp.Vector3(k0[0], k0[1], 0)]
    
    # Suppress C-level MPB output
    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        if params['polarization'] == 'TM':
            ms.run_tm()
        else:
            ms.run_te()
    finally:
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull_fd)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
    
    # Extract Bloch fields if requested
    bloch_fields = None
    if params.get('export_bloch_fields', False):
        bloch_fields = extract_all_bloch_fields(
            ms, params['all_bands'], params['polarization']
        )
    
    # Now compute the full k-stencil for derivatives (reuses compute_bands_at_k_stencil logic)
    from phasesV3.phase1_mpb_v3 import compute_bands_at_k_stencil
    result = compute_bands_at_k_stencil(
        ms, k0, params['dk'], params['all_bands'], 
        params['polarization'], params['fd_order']
    )
    
    if bloch_fields is not None:
        result['bloch_fields'] = bloch_fields
    
    return (ix, iy, result)


# ==============================================================================
# Phase 2: Born-Huang Computation from Bloch Fields
# ==============================================================================

def compute_born_huang_from_fields(
    bloch_fields: np.ndarray,
    dR: Tuple[float, float],
    subspace_indices: List[int],
    extra_indices: Optional[List[int]] = None,
    epsilon: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the Born-Huang potential matrix from stored Bloch fields.
    
    The Born-Huang potential is:
        Φ_mn(R) = Σ_j ⟨∂_{R_j} u_m | (1-P) | ∂_{R_j} u_n⟩_ε
    
    where (1-P) projects onto bands OUTSIDE the subspace, and the inner
    product is ε-weighted: ⟨f|ε|g⟩ = ∫ ε(r) f*(r)·g(r) d²r.
    
    Args:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex Bloch functions
        dR: (dR1, dR2) physical grid spacing in R coordinates (units of a)
        subspace_indices: indices of bands in the subspace (0-based into N_bands)
        extra_indices: indices of bands outside subspace for (1-P) projection
                       If None, uses complement of subspace_indices
        epsilon: (Ns1, Ns2, Nx, Ny) dielectric function. If provided,
                 uses ε-weighted inner product. If None, uses flat inner product.
        
    Returns:
        Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) Born-Huang potential matrix
    """
    Ns1, Ns2, N_bands, Nx, Ny, N_comp = bloch_fields.shape
    N_subspace = len(subspace_indices)
    dR1, dR2 = dR
    
    # Determine extra band indices for out-of-subspace projection
    if extra_indices is None:
        all_indices = set(range(N_bands))
        extra_indices = list(all_indices - set(subspace_indices))
    N_extra = len(extra_indices)
    
    log(f"    Computing Born-Huang: {N_subspace} subspace bands, {N_extra} extra bands")
    log(f"    Grid: {Ns1}×{Ns2}, unit cell: {Nx}×{Ny}, dR = ({dR1:.4f}, {dR2:.4f})")
    if epsilon is not None:
        log(f"    Using ε-weighted inner product for Born-Huang")
    else:
        log(f"    WARNING: No ε data — using flat inner product for Born-Huang")
    
    Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace), dtype=np.float64)
    
    # Compute ∂u/∂R using central differences with periodic boundary
    # ∂u_n/∂R_1 ≈ (u[i+1,j] - u[i-1,j]) / (2*dR1)
    # ∂u_n/∂R_2 ≈ (u[i,j+1] - u[i,j-1]) / (2*dR2)
    
    def compute_du_dR(fields, direction):
        """Compute finite difference derivative with periodic BC."""
        if direction == 0:  # R1
            du = (np.roll(fields, -1, axis=0) - np.roll(fields, 1, axis=0)) / (2 * dR1)
        else:  # R2
            du = (np.roll(fields, -1, axis=1) - np.roll(fields, 1, axis=1)) / (2 * dR2)
        return du
    
    def inner_product_cell(f, g, eps_2d=None):
        """
        Compute ε-weighted inner product over unit cell:
            ⟨f|ε|g⟩_Ω = (1/|Omega|) ∫ ε(r) f*(r) · g(r) d²r
        For discrete grid: sum over pixels, normalized by number of pixels.
        If eps_2d is None, uses flat inner product (weight = 1).
        """
        # f, g have shape (Nx, Ny, N_comp)
        if eps_2d is not None:
            # ε is (Nx, Ny), broadcast to (Nx, Ny, 1) for component multiplication
            return np.sum(eps_2d[:, :, np.newaxis] * np.conj(f) * g) / (Nx * Ny)
        else:
            return np.sum(np.conj(f) * g) / (Nx * Ny)
    
    # For each moiré position (i, j)
    for i in range(Ns1):
        for j in range(Ns2):
            # Get ε at this registry point (if available)
            eps_ij = epsilon[i, j] if epsilon is not None else None
            
            # Extract fields at this position: (N_bands, Nx, Ny, 3)
            u_all = bloch_fields[i, j]
            
            for m_idx, m_band in enumerate(subspace_indices):
                for n_idx, n_band in enumerate(subspace_indices):
                    phi_mn = 0.0
                    
                    for direction in [0, 1]:  # R1, R2
                        # Get u at neighboring points for derivative
                        if direction == 0:
                            ip1 = (i + 1) % Ns1
                            im1 = (i - 1) % Ns1
                            u_m_plus = bloch_fields[ip1, j, m_band]
                            u_m_minus = bloch_fields[im1, j, m_band]
                            u_n_plus = bloch_fields[ip1, j, n_band]
                            u_n_minus = bloch_fields[im1, j, n_band]
                            dR_dir = dR1
                        else:
                            jp1 = (j + 1) % Ns2
                            jm1 = (j - 1) % Ns2
                            u_m_plus = bloch_fields[i, jp1, m_band]
                            u_m_minus = bloch_fields[i, jm1, m_band]
                            u_n_plus = bloch_fields[i, jp1, n_band]
                            u_n_minus = bloch_fields[i, jm1, n_band]
                            dR_dir = dR2
                        
                        # Compute derivatives: ∂u/∂R ≈ (u_plus - u_minus) / (2*dR)
                        du_m = (u_m_plus - u_m_minus) / (2 * dR_dir)
                        du_n = (u_n_plus - u_n_minus) / (2 * dR_dir)
                        
                        # Project onto out-of-subspace: (1-P)|∂u_n⟩
                        # (1-P)|v⟩ = |v⟩ - Σ_{k∈subspace} |u_k⟩⟨u_k|v⟩
                        
                        # First compute projection onto subspace
                        P_du_m = np.zeros_like(du_m)
                        P_du_n = np.zeros_like(du_n)
                        
                        for k_band in subspace_indices:
                            u_k = bloch_fields[i, j, k_band]
                            
                            # ⟨u_k|ε|∂u_m⟩
                            overlap_m = inner_product_cell(u_k, du_m, eps_ij)
                            P_du_m += overlap_m * u_k
                            
                            # ⟨u_k|ε|∂u_n⟩
                            overlap_n = inner_product_cell(u_k, du_n, eps_ij)
                            P_du_n += overlap_n * u_k
                        
                        # (1-P)|∂u⟩
                        one_minus_P_du_m = du_m - P_du_m
                        one_minus_P_du_n = du_n - P_du_n
                        
                        # ⟨∂u_m|ε|(1-P)|∂u_n⟩
                        contribution = inner_product_cell(du_m, one_minus_P_du_n, eps_ij)
                        phi_mn += np.real(contribution)
                    
                    Phi_BH[i, j, m_idx, n_idx] = phi_mn
    
    # Ensure Hermitian (should already be, but enforce for numerical stability)
    Phi_BH = 0.5 * (Phi_BH + np.swapaxes(Phi_BH, 2, 3))
    
    return Phi_BH


# ==============================================================================
# Storage Utilities
# ==============================================================================

def save_bloch_fields(
    h5file: h5py.File,
    bloch_fields: np.ndarray,
    metadata: Dict
):
    """
    Save Bloch fields to HDF5 file.
    
    Args:
        h5file: open HDF5 file handle
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array
        metadata: dict with 'resolution', 'polarization', etc.
    """
    # Use compression for potentially large field data
    if 'bloch_fields' in h5file:
        del h5file['bloch_fields']
    
    h5file.create_dataset(
        'bloch_fields',
        data=bloch_fields,
        compression='lzf',
        chunks=(1, 1, bloch_fields.shape[2], bloch_fields.shape[3], bloch_fields.shape[4], 3),
        dtype=np.complex64
    )
    
    # Save metadata
    h5file['bloch_fields'].attrs['resolution'] = metadata.get('resolution', 32)
    h5file['bloch_fields'].attrs['polarization'] = metadata.get('polarization', 'TM')
    h5file['bloch_fields'].attrs['description'] = (
        'Periodic Bloch functions u_n(r; R) for Born-Huang computation. '
        'Shape: (Ns1, Ns2, N_bands, Nx, Ny, 3) where 3 is (Ex, Ey, Ez).'
    )


def load_bloch_fields(h5file: h5py.File) -> Tuple[np.ndarray, Dict]:
    """
    Load Bloch fields from HDF5 file.
    
    Args:
        h5file: open HDF5 file handle
        
    Returns:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array
        metadata: dict with attributes
    """
    bloch_fields = h5file['bloch_fields'][:]
    
    metadata = {
        'resolution': h5file['bloch_fields'].attrs.get('resolution', 32),
        'polarization': h5file['bloch_fields'].attrs.get('polarization', 'TM'),
    }
    
    return bloch_fields, metadata


def estimate_bloch_field_storage(
    Ns1: int, Ns2: int, N_bands: int, resolution: int
) -> str:
    """
    Estimate storage requirements for Bloch fields.
    
    Args:
        Ns1, Ns2: moiré grid dimensions
        N_bands: number of bands to store
        resolution: MPB resolution (determines unit cell grid)
        
    Returns:
        Human-readable string with storage estimate
    """
    # For 2D, grid is roughly resolution × resolution
    Nx = Ny = resolution
    N_components = 3  # Ex, Ey, Ez
    
    # Complex64 = 8 bytes per element
    bytes_per_element = 8
    total_elements = Ns1 * Ns2 * N_bands * Nx * Ny * N_components
    total_bytes = total_elements * bytes_per_element
    
    # With gzip compression, typically ~2-4x reduction
    compressed_bytes = total_bytes / 3
    
    if total_bytes < 1024**2:
        size_str = f"{total_bytes / 1024:.1f} KB"
        compressed_str = f"{compressed_bytes / 1024:.1f} KB"
    elif total_bytes < 1024**3:
        size_str = f"{total_bytes / 1024**2:.1f} MB"
        compressed_str = f"{compressed_bytes / 1024**2:.1f} MB"
    else:
        size_str = f"{total_bytes / 1024**3:.2f} GB"
        compressed_str = f"{compressed_bytes / 1024**3:.2f} GB"
    
    return f"Uncompressed: {size_str}, Compressed (est.): {compressed_str}"


# ==============================================================================
# Validation and Diagnostics
# ==============================================================================

def validate_bloch_field_normalization(bloch_fields: np.ndarray) -> Dict:
    """
    Check normalization of Bloch fields.
    
    MPB normalizes fields such that ∫ ε|E|² dV = 1.
    For our purposes, we care about ⟨u|u⟩_Ω ~ 1.
    
    Args:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array
        
    Returns:
        dict with normalization statistics
    """
    Ns1, Ns2, N_bands, Nx, Ny, _ = bloch_fields.shape
    
    norms = np.zeros((Ns1, Ns2, N_bands))
    for i in range(Ns1):
        for j in range(Ns2):
            for n in range(N_bands):
                u = bloch_fields[i, j, n]
                # ||u||² = Σ |u|² / (Nx*Ny)
                norms[i, j, n] = np.sum(np.abs(u)**2) / (Nx * Ny)
    
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'min_norm': np.min(norms),
        'max_norm': np.max(norms),
        'norms': norms
    }


def diagnose_born_huang_values(Phi_BH: np.ndarray) -> Dict:
    """
    Provide diagnostic information about Born-Huang potential values.
    
    Args:
        Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) Born-Huang matrix
        
    Returns:
        dict with diagnostic info
    """
    N_subspace = Phi_BH.shape[2]
    
    # Diagonal elements
    diag_values = np.array([Phi_BH[:, :, n, n] for n in range(N_subspace)])
    
    # Off-diagonal Frobenius norm at each point
    offdiag_norm = np.zeros(Phi_BH.shape[:2])
    for i in range(Phi_BH.shape[0]):
        for j in range(Phi_BH.shape[1]):
            mat = Phi_BH[i, j]
            offdiag = mat - np.diag(np.diag(mat))
            offdiag_norm[i, j] = np.linalg.norm(offdiag, 'fro')
    
    return {
        'diagonal_range': [(diag_values[n].min(), diag_values[n].max()) 
                           for n in range(N_subspace)],
        'diagonal_mean': [diag_values[n].mean() for n in range(N_subspace)],
        'offdiag_max': offdiag_norm.max(),
        'offdiag_mean': offdiag_norm.mean(),
        'total_max': np.abs(Phi_BH).max(),
    }
