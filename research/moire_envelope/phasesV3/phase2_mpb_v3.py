"""
Phase 2 (MPB): Berry Connection & Born-Huang Potential — V3 Multi-Band Pipeline

This is the V3 multi-band implementation of Phase 2 using MPB.
The key physics components implemented here:

1. ABELIAN GAUGE FIX: Per-band scalar phase alignment (BFS from center + Zak ramp)
2. SVQB B-ORTHONORMALIZATION: ε-weighted orthonormalization via eigendecomposition
3. BERRY CONNECTION: A_j,mn(s) = i⟨u_m|ε|∂_j u_n⟩ (ε-weighted non-Abelian gauge field)
4. BORN-HUANG POTENTIAL: Φ_mn = Σ_j ⟨∂_j u_m|(1-P)|∂_j u_n⟩_ε

DATA STRUCTURES (V3):
- A: (Ns1, Ns2, N_subspace, N_subspace, 2) - Berry connection matrices
- Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) - Born-Huang potential matrix
- Lambda_n: (Ns1, Ns2, N_subspace) - diagonal potentials (on-site energies)
- M_inv_mn: (Ns1, Ns2, N_subspace, N_subspace, 2, 2) - generalized mass tensors

INNER PRODUCT: For E-fields from MPB, the correct orthogonality is
    ⟨u_m|ε|u_n⟩ = ∫ ε(r) u_m*(r)·u_n(r) d²r = δ_mn
All overlaps use B = diag(ε) weighting. SVQB ensures machine-precision orthonormality.

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
from phasesV3.sanity_checks import log_sanity_block, phase2_sanity_report


def create_multiband_visualization(cdir, s_grid, omega_grid, V_grid, Lambda,
                                    A_berry, Phi_BH, v_drift, M_inv_grid,
                                    N_subspace, target_idx, B_moire):
    """Stub for multi-band visualization (TODO: implement in common.plotting)."""
    import matplotlib.pyplot as plt
    
    # Dynamic layout based on number of subspace bands
    fig, axes = plt.subplots(2, N_subspace, figsize=(4 * N_subspace, 8), squeeze=False)
    
    # Plot diagonal potentials for each band
    for n in range(N_subspace):
        ax = axes[0, n]
        im = ax.imshow(Lambda[:, :, n, n].T, origin='lower', cmap='RdBu_r')
        ax.set_title(f'$\\Lambda_{{{n}{n}}}(s)$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        if n == target_idx:
            ax.set_xlabel("(Target Band)", color='red', fontweight='bold')
    
    # Plot Born-Huang diagonal
    for n in range(N_subspace):
        ax = axes[1, n]
        im = ax.imshow(Phi_BH[:, :, n, n].T, origin='lower', cmap='viridis')
        ax.set_title(f'$\\Phi_{{BH,{n}{n}}}(s)$')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(cdir / 'phase2_multiband_fields.png', dpi=150)
    plt.close()


def log(message):
    """Print message with flush."""
    print(message, flush=True)


# ==============================================================================
# Parallel Transport Gauge (SVD-based) — DEPRECATED
# ==============================================================================
# NOTE: These three functions use flat inner products and the non-Abelian SVD gauge,
# which destroys ε-orthogonality for E-fields. They are kept for backward
# compatibility but are NO LONGER CALLED by the main pipeline.
# Use apply_abelian_gauge_2d() + apply_svqb_to_bloch_fields() instead.

def compute_overlap_matrix(u_current, u_next):
    """
    Compute overlap matrix: O_mn = ⟨u_m(s)|u_n(s+ds)⟩
    
    For MPB, we use frequency-based proxies since we don't have 
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
    Apply parallel transport gauge to bloch_fields.
    
    Args:
        eigenvector_data: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex fields
        axis: 0 (along s1) or 1 (along s2)
        
    Returns:
        fixed_data: same shape, with locally smooth gauge
        gauge_matrices: None (simplify return)
    """
    # Fixed implementation for 6D bloch_fields array
    fixed_data = np.copy(eigenvector_data)
    Ns1, Ns2, N_bands = fixed_data.shape[:3]
    
    log(f"  Applying parallel transport gauge along axis {axis}...")
    
    if axis == 0:
        # For each column j, transport along rows i
        for j in range(Ns2):
            for i in range(Ns1 - 1):
                # u_curr: (N_bands, Nx, Ny, 3)
                u_curr = fixed_data[i, j]
                u_next_raw = fixed_data[i+1, j]
                
                # Compute overlap O_mn = <u_m(i) | u_n(i+1)>
                O_code = compute_overlap_matrix(u_curr, u_next_raw)
                
                # SVD: O = U S V†, parallel transport matrix W = U V†
                W = parallel_transport_step(O_code)
                
                # Apply W† to u_next_raw to align with u_curr.
                # The gauge rotation that maximizes Re Tr(<u_curr | u'_next>)
                # is u'_n = Σ_k (W†)_nk u_k = Σ_k conj(W_kn) u_k.
                # Verified empirically: W†@u gives 0° phase residual, W@u gives ~57°.
                fixed_data[i+1, j] = np.einsum('mn,n...->m...', W.conj().T, u_next_raw)
                
        # Handle Periodic Boundary if needed? 
        # For now, just linear chain. The gap between N-1 and 0 will remain discontinuous
        # unless we explicitly close the loop. 
        # For derivatives, we need smoothness at the boundary if using periodic diffs.
        # But let's fix the bulk first.
        
    elif axis == 1:
        # For each row i, transport along cols j
        for i in range(Ns1):
            for j in range(Ns2 - 1):
                u_curr = fixed_data[i, j]
                u_next_raw = fixed_data[i, j+1]
                
                O_code = compute_overlap_matrix(u_curr, u_next_raw)
                W = parallel_transport_step(O_code)
                fixed_data[i, j+1] = np.einsum('mn,n...->m...', W.conj().T, u_next_raw)
                
    return fixed_data, None


# ==============================================================================
# Abelian Gauge Fix — BFS from center + Zak phase distribution
# ==============================================================================

def apply_abelian_gauge_2d(bloch_fields):
    """
    Abelian (per-band scalar) gauge fix on 2D registry grid.

    For each band independently, align the complex phase between neighbors
    so that ⟨u_n(s)|u_n(s+δs)⟩ is real and positive.

    Algorithm (BFS from center + Zak phase ramp):
      1) Seed from grid center (Ns1//2, Ns2//2).
      2) BFS expanding to 4-connected neighbors. Each visited point gets
         its phase aligned to its parent via u *= conj(ov)/|ov|.
         This is *isotropic* — treats s1 and s2 symmetrically, preserving
         any C4/C6 lattice symmetry of the underlying fields.
      3) After BFS (open boundary), measure average residual boundary phase:
           φ1 = mean_j angle(⟨u(Ns1-1,j)|u(0,j)⟩)
           φ2 = mean_i angle(⟨u(i,Ns2-1)|u(i,0)⟩)
         Apply linear ramp u(i,j) *= exp(-i·(φ1·i/Ns1 + φ2·j/Ns2)) to
         distribute the Zak phase uniformly, making FD periodic boundaries
         smooth.

    This is Abelian (no band mixing) so it:
      ✓ Preserves orthogonality between bands
      ✓ Preserves normalization
      ✓ Avoids the ε-inner-product problem (non-Abelian SVD mixing destroys them)
      ✓ Treats s1 and s2 symmetrically (C4-preserving)

    Args:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array

    Returns:
        fixed: same shape, with smooth per-band phase
        diagnostics: dict of per-band phase-unwinding info
    """
    from collections import deque

    Ns1, Ns2, N_bands = bloch_fields.shape[:3]
    # Work in-place to avoid doubling memory usage (array can be 7+ GB)
    fixed = bloch_fields
    diag = {}

    center_i, center_j = Ns1 // 2, Ns2 // 2
    # 4-connected neighbor offsets (isotropic expansion)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for n in range(N_bands):
        min_ov = 1.0
        n_singular = 0
        n_aligned = 0

        # --- Step 1: BFS from center ---
        visited = np.zeros((Ns1, Ns2), dtype=bool)
        visited[center_i, center_j] = True
        queue = deque()
        # Seed BFS from center to all 4 neighbors
        for di, dj in neighbors:
            ni, nj = center_i + di, center_j + dj
            if 0 <= ni < Ns1 and 0 <= nj < Ns2:
                queue.append((center_i, center_j, ni, nj))

        while queue:
            ci, cj, ni, nj = queue.popleft()
            if visited[ni, nj]:
                continue
            visited[ni, nj] = True

            # Align u(ni,nj) to u(ci,cj)
            u_parent = fixed[ci, cj, n].ravel()
            u_child = fixed[ni, nj, n].ravel()
            np_norm = np.linalg.norm(u_parent)
            nc_norm = np.linalg.norm(u_child)
            if np_norm > 1e-10 and nc_norm > 1e-10:
                ov = np.dot(np.conj(u_parent / np_norm), u_child / nc_norm)
                mag = abs(ov)
                min_ov = min(min_ov, mag)
                if mag > 1e-10:
                    fixed[ni, nj, n] *= np.conj(ov) / mag
                    n_aligned += 1
                else:
                    n_singular += 1

            # Enqueue unvisited neighbors of (ni, nj)
            for di, dj in neighbors:
                nni, nnj = ni + di, nj + dj
                if 0 <= nni < Ns1 and 0 <= nnj < Ns2 and not visited[nni, nnj]:
                    queue.append((ni, nj, nni, nnj))

        # --- Step 2: Zak phase distribution (periodic boundary smoothness) ---
        # Measure average boundary phase jump along s1 (wrap Ns1-1 → 0)
        phases_s1 = []
        for j in range(Ns2):
            u_last = fixed[Ns1 - 1, j, n].ravel()
            u_first = fixed[0, j, n].ravel()
            nl = np.linalg.norm(u_last)
            nf = np.linalg.norm(u_first)
            if nl > 1e-10 and nf > 1e-10:
                ov = np.dot(np.conj(u_last / nl), u_first / nf)
                phases_s1.append(np.angle(ov))
        phi1 = float(np.mean(phases_s1)) if phases_s1 else 0.0

        # Measure average boundary phase jump along s2 (wrap Ns2-1 → 0)
        phases_s2 = []
        for i in range(Ns1):
            u_last = fixed[i, Ns2 - 1, n].ravel()
            u_first = fixed[i, 0, n].ravel()
            nl = np.linalg.norm(u_last)
            nf = np.linalg.norm(u_first)
            if nl > 1e-10 and nf > 1e-10:
                ov = np.dot(np.conj(u_last / nl), u_first / nf)
                phases_s2.append(np.angle(ov))
        phi2 = float(np.mean(phases_s2)) if phases_s2 else 0.0

        # Apply linear ramp to distribute Zak phase across all links
        # After ramp: boundary jump ≈ 0 (smooth periodic)
        if abs(phi1) > 1e-12 or abs(phi2) > 1e-12:
            for i in range(Ns1):
                for j in range(Ns2):
                    ramp_phase = -(phi1 * i / Ns1 + phi2 * j / Ns2)
                    fixed[i, j, n] *= np.exp(1j * ramp_phase)

        # --- Step 3: Post-ramp boundary diagnostic ---
        post_phases_s1 = []
        for j in range(Ns2):
            u_last = fixed[Ns1 - 1, j, n].ravel()
            u_first = fixed[0, j, n].ravel()
            nl = np.linalg.norm(u_last)
            nf = np.linalg.norm(u_first)
            if nl > 1e-10 and nf > 1e-10:
                ov = np.dot(np.conj(u_last / nl), u_first / nf)
                post_phases_s1.append(np.angle(ov))

        post_phases_s2 = []
        for i in range(Ns1):
            u_last = fixed[i, Ns2 - 1, n].ravel()
            u_first = fixed[i, 0, n].ravel()
            nl = np.linalg.norm(u_last)
            nf = np.linalg.norm(u_first)
            if nl > 1e-10 and nf > 1e-10:
                ov = np.dot(np.conj(u_last / nl), u_first / nf)
                post_phases_s2.append(np.angle(ov))

        post_std_s1 = float(np.std(post_phases_s1)) if post_phases_s1 else 0.0
        post_std_s2 = float(np.std(post_phases_s2)) if post_phases_s2 else 0.0

        diag[n] = {
            'min_ov': min_ov,
            'n_singular': n_singular,
            'n_aligned': n_aligned,
            'zak_phi1': phi1,
            'zak_phi2': phi2,
            'post_boundary_std_s1': post_std_s1,
            'post_boundary_std_s2': post_std_s2,
        }
        log(f"      Band {n}: min|ov|={min_ov:.4f}, singular={n_singular}, "
            f"Zak=({phi1:.3f},{phi2:.3f}) rad, "
            f"boundary σ=({post_std_s1:.3f},{post_std_s2:.3f}) rad")

    return fixed, diag


# ==============================================================================
# SVQB B-Orthonormalization
# ==============================================================================

def apply_B_operator(u_flat, eps_flat):
    """
    Apply B = diag(ε) to a flattened Bloch field.

    For E-fields with 3 components at each (x,y) pixel:
      (Bu)_{x,y,c} = ε(x,y) · u_{x,y,c}

    Args:
        u_flat: (Nx*Ny*3,) complex vector
        eps_flat: (Nx*Ny*3,) real ε values (already repeated for 3 components)

    Returns:
        Bu_flat: (Nx*Ny*3,) complex vector
    """
    return eps_flat * u_flat


def svqb_orthonormalize(vectors, mass_vectors, drop_tol=1e-12):
    """
    SVQB B-orthonormalization.

    Given p vectors and their mass-operator images (B·vectors), produce
    a B-orthonormal basis for their span.

    Algorithm (following svqb_guide.md):
      1. Pre-normalize each column to unit B-norm
      2. Form Gram matrix G = X^H · (BX)
      3. Eigendecompose G = Q Λ Q^H (Hermitian)
      4. Rank-reveal: drop λ_i / λ_max < drop_tol
      5. Build transform T = Q_kept · Λ_kept^{-1/2}
      6. Apply X_new = X_old · T, (BX)_new = (BX)_old · T

    Args:
        vectors: list of p arrays, each shape (N,) complex128
        mass_vectors: list of p arrays, each shape (N,) complex128 (B · vectors[i])
        drop_tol: threshold for rank-revealing (default 1e-12)

    Returns:
        dict with:
          'vectors': list of rank arrays (B-orthonormal)
          'mass_vectors': list of rank arrays (B · orthonormalized vectors)
          'rank': int, number of linearly independent vectors found
          'eigenvalues': array of Gram eigenvalues before dropping
          'dropped': number of vectors dropped
    """
    p = len(vectors)
    if p == 0:
        return {'vectors': [], 'mass_vectors': [], 'rank': 0,
                'eigenvalues': np.array([]), 'dropped': 0}

    N = vectors[0].shape[0]

    # Work in float64 throughout (Pitfall #3: mixed-precision accumulation)
    X = np.column_stack([v.astype(np.complex128) for v in vectors])      # (N, p)
    BX = np.column_stack([mv.astype(np.complex128) for mv in mass_vectors])  # (N, p)

    # Step 1: Pre-normalize each column to unit B-norm (Pitfall #1)
    for j in range(p):
        b_norm_sq = np.real(np.dot(np.conj(X[:, j]), BX[:, j]))
        if b_norm_sq > 1e-60:
            scale = 1.0 / np.sqrt(b_norm_sq)
            X[:, j] *= scale
            BX[:, j] *= scale

    # Step 2: Form Gram matrix G = X^H · (BX)   [O(N·p²)]
    G = X.conj().T @ BX   # (p, p) complex128

    # Symmetrize (should be Hermitian, but enforce it)
    G = 0.5 * (G + G.conj().T)

    # Step 3: Eigendecompose G = Q Λ Q^H (Hermitian)
    eigenvalues, Q = np.linalg.eigh(G)
    # eigh returns ascending order → reverse to descending (Pitfall #5)
    eigenvalues = eigenvalues[::-1]
    Q = Q[:, ::-1]

    # Step 4: Rank-reveal
    lambda_max = eigenvalues[0]
    if lambda_max <= 0:
        return {'vectors': [], 'mass_vectors': [], 'rank': 0,
                'eigenvalues': eigenvalues, 'dropped': p}

    rank = 0
    for i in range(p):
        if eigenvalues[i] / lambda_max >= drop_tol:
            rank += 1
        else:
            break

    if rank == 0:
        return {'vectors': [], 'mass_vectors': [], 'rank': 0,
                'eigenvalues': eigenvalues, 'dropped': p}

    # Step 5: Build transform T = Q_kept · Λ_kept^{-1/2}   (p × rank)
    Q_kept = Q[:, :rank]
    lambda_kept = eigenvalues[:rank]
    T = Q_kept * (1.0 / np.sqrt(lambda_kept))[np.newaxis, :]  # (p, rank)

    # Step 6: Apply X_new = X_old · T    [O(N·p·rank)]
    X_new = X @ T      # (N, rank)
    BX_new = BX @ T    # (N, rank)

    # Convert back to list
    out_vectors = [X_new[:, j].copy() for j in range(rank)]
    out_mass_vectors = [BX_new[:, j].copy() for j in range(rank)]

    return {
        'vectors': out_vectors,
        'mass_vectors': out_mass_vectors,
        'rank': rank,
        'eigenvalues': eigenvalues,
        'dropped': p - rank,
    }


def apply_svqb_to_bloch_fields(bloch_fields, epsilon):
    """
    Apply SVQB B-orthonormalization to Bloch fields at every registry point.

    At each (ix, iy), takes the N_bands Bloch vectors (shape Nx*Ny*3 each),
    builds B = diag(ε), and runs SVQB to achieve:
        ⟨u_m|ε|u_n⟩ = δ_mn   (to machine precision)

    Args:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array
        epsilon: (Ns1, Ns2, Nx, Ny) real dielectric function

    Returns:
        bf_ortho: same shape, B-orthonormalized
        stats: dict with rank_loss_count, max_gram_cond, mean_gram_cond
    """
    Ns1, Ns2, N_bands, Nx, Ny, Nc = bloch_fields.shape
    # Work in-place to avoid doubling memory usage (array can be 7+ GB).
    # SVQB computes in float64 per-point but writes back to original dtype.
    N_flat = Nx * Ny * Nc

    rank_loss_count = 0
    gram_conds = []

    for ix in range(Ns1):
        if ix % 16 == 0:
            log(f"    SVQB progress: row {ix}/{Ns1}")
        for iy in range(Ns2):
            # Build ε-flat: repeat ε(x,y) for 3 components
            eps_2d = epsilon[ix, iy]  # (Nx, Ny)
            eps_flat = np.repeat(eps_2d[:, :, np.newaxis], Nc, axis=2).ravel()  # (Nx*Ny*3,)

            # Build vectors and mass vectors (one point at a time, in float64)
            vectors = []
            mass_vectors = []
            for n in range(N_bands):
                u_flat = bloch_fields[ix, iy, n].ravel().astype(np.complex128)
                Bu_flat = apply_B_operator(u_flat, eps_flat)
                vectors.append(u_flat)
                mass_vectors.append(Bu_flat)

            # Run SVQB
            result = svqb_orthonormalize(vectors, mass_vectors)

            if result['rank'] < N_bands:
                rank_loss_count += 1

            # Gram condition number
            eigs = result['eigenvalues']
            if len(eigs) > 0 and eigs[-1] > 0:
                gram_conds.append(eigs[0] / eigs[-1])

            # Write back orthonormalized vectors in-place (cast back to original dtype)
            for n in range(result['rank']):
                bloch_fields[ix, iy, n] = result['vectors'][n].reshape(Nx, Ny, Nc).astype(bloch_fields.dtype)

    stats = {
        'rank_loss_count': rank_loss_count,
        'max_gram_cond': max(gram_conds) if gram_conds else 0.0,
        'mean_gram_cond': np.mean(gram_conds) if gram_conds else 0.0,
        'total_points': Ns1 * Ns2,
    }

    return bloch_fields, stats


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
    # True implementation requires eigenvector export from MPB
    
    log("    NOTE: Berry connection approximated (diagonal only, requires wavefunctions for full calculation)")
    
    return A


def compute_berry_connection_from_eigenvectors(
    eigenvector_data,  # (Ns1, Ns2, N_bands, resolution, resolution) or similar
    ds1, ds2,
    fd_order=4,
    epsilon=None,
    return_diagnostics=False,
):
    """
    Compute Berry connection from actual eigenvector data.
    
    A_j,mn(s) = i⟨u_m(s)|ε|∂_j u_n(s)⟩
    
    Uses periodic boundary conditions via circular indexing.
    Uses ε-weighted inner product when epsilon is provided.
    
    Args:
        eigenvector_data: (Ns1, Ns2, N_bands, ...) eigenvector fields
        ds1, ds2: grid spacings in fractional coordinates
        fd_order: finite difference order
        epsilon: (Ns1, Ns2, Nx, Ny) dielectric function. If provided,
                 uses ε-weighted inner product ⟨u_m|ε|∂u_n⟩.
        
    Returns:
        A: (Ns1, Ns2, N_bands, N_bands, 2) Berry connection
    """
    Ns1, Ns2, N_bands = eigenvector_data.shape[:3]
    spatial_shape = eigenvector_data.shape[3:]
    
    # Reshape to (Ns1, Ns2, N_bands, N_spatial)
    u = eigenvector_data.reshape(Ns1, Ns2, N_bands, -1)
    N_spatial = u.shape[-1]
    
    # Build ε-weight arrays if provided
    if epsilon is not None:
        # epsilon: (Ns1, Ns2, Nx, Ny) → expand to (Ns1, Ns2, Nx*Ny*Nc)
        Nx, Ny = epsilon.shape[2:]
        Nc = spatial_shape[-1] if len(spatial_shape) > 2 else 1
        # Repeat ε for each vector component
        eps_expanded = np.repeat(epsilon[:, :, :, :, np.newaxis], Nc, axis=4)  # (Ns1, Ns2, Nx, Ny, Nc)
        eps_flat = eps_expanded.reshape(Ns1, Ns2, -1)  # (Ns1, Ns2, N_spatial)
    
    overlap_s1_all = np.zeros((Ns1, Ns2, N_bands, N_bands), dtype=complex)
    overlap_s2_all = np.zeros((Ns1, Ns2, N_bands, N_bands), dtype=complex)
    
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
            
            # Berry connection: A_mn = i <u_m | ε | du_n>
            # Get ε-weight for this point (or use flat weight = 1)
            if epsilon is not None:
                eps_ij = eps_flat[i, j]  # (N_spatial,)
            
            for m in range(N_bands):
                for n in range(N_bands):
                    if epsilon is not None:
                        # ε-weighted overlap
                        overlap_s1 = np.sum(eps_ij * u_ij[m].conj() * du_ds1[n])
                        overlap_s2 = np.sum(eps_ij * u_ij[m].conj() * du_ds2[n])
                    else:
                        # Flat overlap (backward compat)
                        overlap_s1 = np.sum(u_ij[m].conj() * du_ds1[n])
                        overlap_s2 = np.sum(u_ij[m].conj() * du_ds2[n])
                    
                    overlap_s1_all[i, j, m, n] = overlap_s1
                    overlap_s2_all[i, j, m, n] = overlap_s2

    # Exact continuum identity gives O + O^† = 0 for O_mn = <u_m|∂u_n>.
    # Use the manifestly Hermitian discretization A = (i/2)(O - O^†), which
    # matches iO in the continuum limit but does not inject non-physical
    # anti-Hermitian contamination from finite-difference / gauge noise.
    A = np.zeros((Ns1, Ns2, N_bands, N_bands, 2), dtype=complex)
    A_raw_s1 = 1j * overlap_s1_all
    A_raw_s2 = 1j * overlap_s2_all
    A[..., 0] = 0.5j * (overlap_s1_all - np.swapaxes(np.conj(overlap_s1_all), 2, 3))
    A[..., 1] = 0.5j * (overlap_s2_all - np.swapaxes(np.conj(overlap_s2_all), 2, 3))

    if return_diagnostics:
        raw_residual_0 = A_raw_s1 - np.swapaxes(np.conj(A_raw_s1), 2, 3)
        raw_residual_1 = A_raw_s2 - np.swapaxes(np.conj(A_raw_s2), 2, 3)
        diagnostics = {
            'raw_hermiticity_max_abs': float(max(np.max(np.abs(raw_residual_0)), np.max(np.abs(raw_residual_1)))),
            'raw_component_max_abs': [
                float(np.max(np.abs(raw_residual_0))),
                float(np.max(np.abs(raw_residual_1))),
            ],
            'projected_component_max_abs': [
                float(np.max(np.abs(A[..., 0]))),
                float(np.max(np.abs(A[..., 1]))),
            ],
        }
        return A, diagnostics

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
    Compute Born-Huang potential matrix placeholder.
    
    The correct Born-Huang formula from theory is:
        Φ_mn = Σ_j ⟨∂_{R_j} u_m|(1-P)|∂_{R_j} u_n⟩_Ω
    
    This requires computing ∂u/∂R, the derivative of Bloch functions with
    respect to the slow moiré coordinate R. This can be done via:
    
    1. Finite differences: ∂u_n/∂R ≈ (u_n(R+dR) - u_n(R-dR))/(2*dR)
       - Requires storing Bloch functions u_n(r; R) at each R sample
    
    2. Perturbation identity (from theory doc section 11):
       ⟨u_ℓ|∂_{R_j}u_n⟩ = ⟨u_ℓ|(∂_{R_j}L₀)|u_n⟩/(λ_n - λ_ℓ) for ℓ ∉ subspace
       - Requires matrix elements of the operator derivative
    
    CURRENT STATUS: PLACEHOLDER
    ============================
    Without access to actual Bloch functions or operator derivatives,
    we cannot correctly compute the Born-Huang potential.
    
    We set Φ_BH = 0 as a placeholder. This is an O(η²) correction that
    is typically smaller than the kinetic term for well-isolated bands.
    
    TODO: Implement proper Born-Huang computation when Bloch function
    data becomes available from MPB field exports.
    
    Args:
        omega_subspace: frequencies of subspace bands (unused - placeholder)
        omega_extra: frequencies of bands for Born-Huang correction (unused)
        M_inv_subspace: inverse mass tensors of subspace bands (unused)
        M_inv_extra: inverse mass tensors of extra bands (unused)
        coupling_strength: overall scaling factor (unused)
        
    Returns:
        Phi_BH: (Ns1, Ns2, N_subspace, N_subspace) ZERO matrix (placeholder)
    """
    Ns1, Ns2, N_subspace = omega_subspace.shape
    
    # PLACEHOLDER: Return zero matrix
    # The Born-Huang correction requires Bloch function derivatives ∂u/∂R
    # which we currently don't compute. This is a known limitation.
    Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
    
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
    
    # Off-diagonal elements: filled by compute_velocity_matrix_from_fields()
    # when Bloch fields are available. Otherwise remain zero.
    
    return v_drift


# ==============================================================================
# Velocity Matrix from Bloch Fields  (off-diagonal v_mn)
# ==============================================================================

def compute_velocity_matrix_from_fields(
    bloch_fields,     # (Ns1, Ns2, N_bands, Nx, Ny, 3) complex E-field envelopes
    k0_mpb,           # (2,) k-point in MPB reciprocal lattice coords, e.g. [0.5, 0.5]
    omega_scale,      # physical frequency scale (MPB units)
    a=1.0,            # lattice constant
    polarization='TM',
):
    """
    Compute velocity matrix v_mn from Bloch fields at k0.

    For TM, the operator derivative ∂Θ/∂k_i = -2i(∂_i+ik_i)/ε.
    The ε-weighted matrix element simplifies to a FLAT inner product:

        Π_mn^(i) = ⟨u_m|ε·∂_{k_i}Θ|u_n⟩ = -2i⟨u_m|(∂_i+ik_i)|u_n⟩_flat

    The velocity in ω-space (for the envelope Hamiltonian) is:

        v_mn^(i) = Π_mn^(i) / (2·ω_phys)

    Diagonal v_nn should match the stencil-computed group velocity.

    Args:
        bloch_fields: periodic Bloch E-field envelopes u_n(r;δ)
        k0_mpb: k-point in MPB units (reciprocal lattice coordinates)
        omega_scale: physical frequency scale for ω-space conversion
        a: lattice constant
        polarization: 'TM' (uses Ez) or 'TE' (not implemented)

    Returns:
        v_matrix: (Ns1, Ns2, N_bands, N_bands, 2) complex velocity matrix
        diagnostics: dict with consistency checks
    """
    if polarization != 'TM':
        raise NotImplementedError("Only TM polarization implemented for velocity matrix")
    if abs(omega_scale) < 1e-12:
        raise ValueError(
            "Velocity matrix requires a non-zero physical frequency scale; "
            "use the absolute target frequency rather than a shifted envelope reference."
        )

    Ns1, Ns2, N_bands = bloch_fields.shape[:3]
    Nx, Ny = bloch_fields.shape[3], bloch_fields.shape[4]

    # Physical k-point (radians/length)
    k0x_phys = 2 * np.pi * k0_mpb[0] / a
    k0y_phys = 2 * np.pi * k0_mpb[1] / a

    # FFT wave-vectors on the unit cell grid
    # np.fft.fftfreq(N, d=dx) gives cycles/length; multiply by 2π for radians
    dx = a / Nx
    dy = a / Ny
    Gx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)   # (Nx,)
    Gy = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)   # (Ny,)
    GX, GY = np.meshgrid(Gx, Gy, indexing='ij')  # (Nx, Ny)

    # Combined q = G + k0
    qx = GX + k0x_phys   # (Nx, Ny)
    qy = GY + k0y_phys

    # Extract Ez component for TM
    comp = 2
    u_all = bloch_fields[:, :, :, :, :, comp]  # (Ns1, Ns2, N_bands, Nx, Ny) complex

    v_matrix = np.zeros((Ns1, Ns2, N_bands, N_bands, 2), dtype=complex)

    for i1 in range(Ns1):
        for i2 in range(Ns2):
            u = u_all[i1, i2]  # (N_bands, Nx, Ny) complex

            for n in range(N_bands):
                # D_x u_n = IFFT2[i·q_x · FFT2(u_n)]
                u_fft = np.fft.fft2(u[n])
                Dx_un = np.fft.ifft2(1j * qx * u_fft)
                Dy_un = np.fft.ifft2(1j * qy * u_fft)

                for m in range(N_bands):
                    # Π_mn = -2i · Σ_r u_m*(r) · D_i u_n(r)
                    Pi_x = -2j * np.sum(u[m].conj() * Dx_un)
                    Pi_y = -2j * np.sum(u[m].conj() * Dy_un)

                    # v_mn = Π_mn / (2 ω_phys). The energy zero is arbitrary, so
                    # this scale must remain tied to the physical carrier frequency.
                    v_matrix[i1, i2, m, n, 0] = Pi_x / (2 * omega_scale)
                    v_matrix[i1, i2, m, n, 1] = Pi_y / (2 * omega_scale)

    # Diagnostics: check diagonal vs stencil, Hermiticity
    v_diag_x = np.array([v_matrix[:, :, n, n, 0].real for n in range(N_bands)])
    v_diag_y = np.array([v_matrix[:, :, n, n, 1].real for n in range(N_bands)])
    offdiag_max = 0.0
    hermiticity_err = 0.0
    for m in range(N_bands):
        for n in range(m + 1, N_bands):
            offdiag_max = max(offdiag_max,
                              np.max(np.abs(v_matrix[:, :, m, n, :])))
            # v should be Hermitian: v_mn = v_nm*
            hermiticity_err = max(hermiticity_err,
                                  np.max(np.abs(v_matrix[:, :, m, n, :] -
                                                v_matrix[:, :, n, m, :].conj())))

    diagnostics = {
        'v_diag_x_range': [float(v_diag_x.min()), float(v_diag_x.max())],
        'v_diag_y_range': [float(v_diag_y.min()), float(v_diag_y.max())],
        'offdiag_max': float(offdiag_max),
        'hermiticity_error': float(hermiticity_err),
    }

    return v_matrix, diagnostics


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
    bloch_fields = None
    epsilon = None
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
        n_registry = int(stencil_grp.attrs['n_registry'])
        
        # Load Bloch fields if available (for proper Born-Huang computation)
        if 'bloch_fields' in hf:
            bloch_fields = hf['bloch_fields'][:]
            log(f"  Loaded Bloch fields: shape {bloch_fields.shape}")
        
        # Load dielectric function ε(r; δ) for B-orthonormalization
        if 'epsilon' in hf:
            epsilon = hf['epsilon'][:]
            log(f"  Loaded ε(r; δ): shape {epsilon.shape}, "
                f"range [{epsilon.min():.2f}, {epsilon.max():.2f}]")
        
        if bloch_fields is not None:
            # --- ABELIAN GAUGE FIX + SVQB B-ORTHONORMALIZATION ---
            # 
            # Pipeline order (validated in F06):
            #   1. Abelian gauge fix: per-band scalar phase alignment
            #      (makes each band's phase smooth, no band mixing)
            #   2. SVQB B-orthonormalization with B = diag(ε)
            #      (ensures ⟨u_m|ε|u_n⟩ = δ_mn to machine precision)
            #
            # This replaces the old flat normalization + non-Abelian SVD gauge,
            # which was broken for E-fields (SVD uses flat inner product,
            # destroying ε-orthogonality).
            # ---------------------------------------------------------
            
            # Step 1: Abelian gauge fix (per-band scalar phase alignment)
            log("  Step 1: Abelian gauge fix (per-band scalar phase alignment)...")
            bloch_fields, gauge_diag = apply_abelian_gauge_2d(bloch_fields)
            log("    Abelian gauge fix complete.")
            
            # Step 2: SVQB B-orthonormalization
            if epsilon is not None:
                log("  Step 2: SVQB B-orthonormalization (B = diag(ε))...")
                bloch_fields, svqb_stats = apply_svqb_to_bloch_fields(bloch_fields, epsilon)
                log(f"    SVQB complete: rank_loss={svqb_stats['rank_loss_count']}/{svqb_stats['total_points']}, "
                    f"Gram κ mean={svqb_stats['mean_gram_cond']:.4f}, max={svqb_stats['max_gram_cond']:.4f}")
                if svqb_stats['rank_loss_count'] > 0:
                    log(f"    WARNING: {svqb_stats['rank_loss_count']} points lost rank — check for near-degeneracies")
            else:
                log("  WARNING: No ε data available for SVQB — falling back to flat normalization")
                log("    (Re-run Phase 1 with export_bloch_fields=True to generate ε data)")
                norms = np.sqrt(np.sum(np.abs(bloch_fields)**2, axis=(-3, -2, -1), keepdims=True))
                log(f"    Flat-normalizing Bloch fields. Mean norm before: {np.mean(norms**2):.4f}")
                bloch_fields = bloch_fields / (norms + 1e-15)
        
        # Metadata
        omega_ref = hf.attrs['omega_ref']
        omega_target_abs = hf.attrs.get('omega_target_abs', omega_ref)
        eta = hf.attrs['eta']
        theta_rad = hf.attrs['theta_rad']
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_subspace = int(hf.attrs['N_subspace'])
        target_idx = int(hf.attrs['target_index_in_subspace'])
        moire_length = float(hf.attrs['moire_length'])
        k0_x = float(hf.attrs.get('k0_x', 0.5))
        k0_y = float(hf.attrs.get('k0_y', 0.5))
        
        B_moire = hf.attrs['B_moire']
        B_mono = hf.attrs['B_mono']
        subspace_bands = hf.attrs['subspace_bands'][:].tolist()
        all_bands = hf.attrs['all_bands'][:].tolist()
    
    log(f"  Grid: {Ns1} × {Ns2}, N_subspace = {N_subspace}")
    log(f"  η = {eta:.4f}, θ = {math.degrees(theta_rad):.4f}°")
    log(f"  Subspace bands: {subspace_bands}")
    log(f"  All bands: {all_bands}")
    log(f"  ω_ref = {omega_ref:.6f}")
    log(f"  ω_phys(scale) = {omega_target_abs:.6f}")
    log(f"  L_moire = {moire_length:.4f}")
    
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    
    # Physical grid spacing (for Born-Huang derivatives)
    # Registry grid covers [0,1)^2 in fractional coords, physical size is L_moire
    dR_registry = moire_length / n_registry
    log(f"  Registry grid spacing: dR = {dR_registry:.4f} (physical units)")
    
    # Configuration
    include_born_huang = config.get('include_born_huang', True)
    include_drift_term = config.get('include_drift_term', True)
    use_parallel_transport = config.get('use_parallel_transport_gauge', True)
    n_extra_bands = config.get('n_extra_bands', 4)
    fd_order = config.get('mpb_fd_order', 4)
    
    log(f"  Include Born-Huang: {include_born_huang}")
    log(f"  Include drift term: {include_drift_term}")
    log(f"  Use parallel transport gauge: {use_parallel_transport}")
    
    # =========================================================================
    # 1. Berry Connection
    # =========================================================================
    log("  Computing Berry connection...")
    
    # Initialize A_berry
    A_berry = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2), dtype=complex)
    
    # Calculate A from Bloch fields if available (BEST METHOD)
    if bloch_fields is not None:
        log("    Using Bloch fields to compute Berry connection A = i⟨u|ε|du⟩...")
        if epsilon is not None:
            log("    Using ε-weighted inner product for Berry connection")
        else:
            log("    WARNING: No ε data — using flat inner product for Berry connection")
        
        # Map subspace bands to indices
        subspace_indices = [all_bands.index(b) for b in subspace_bands]
        
        # Extract subspace fields: (Ns1, Ns2, N_sub, Nx, Ny, 3)
        bloch_sub = bloch_fields[:, :, subspace_indices, ...]
        
        # Also extract subspace epsilon if available
        eps_sub = epsilon  # epsilon is already (n_reg, n_reg, Nx, Ny), no band axis
        
        # Compute A for subspace
        # bloch_sub is on the REGISTRY grid (n_registry x n_registry).
        step_reg = 1.0 / n_registry
        
        A_frac, berry_raw_diag = compute_berry_connection_from_eigenvectors(
            bloch_sub, step_reg, step_reg, fd_order, epsilon=eps_sub,
            return_diagnostics=True,
        )
        
        # Convert to physical units (1/a)
        # A_phys = A_frac / L_moire.

        A_berry_registry = A_frac / moire_length

        # Interpolate A_berry from registry grid (n_reg x n_reg) to moire grid (Ns1 x Ns2)
        log(f"    Interpolating Berry connection from {n_registry}×{n_registry} to {Ns1}×{Ns2}...")
        
        # Reuse interpolation logic from BH section
        # We need to interpolate each component (m, n, d)
        # A_berry_registry shape: (n_reg, n_reg, N_sub, N_sub, 2)
        
        # Define interpolator factory (needs to be redefined or moved up scope if used twice)
        # Or just cut-paste logic since variables x_reg/y_reg defined below in BH section.
        # Let's define x_reg/y_reg here.
        
        step_reg_val = 1.0 / n_registry # careful with variable naming
        x_reg_AX = np.linspace(0, 1 - step_reg_val, n_registry)
        y_reg_AX = np.linspace(0, 1 - step_reg_val, n_registry)
        
        # Helper to make periodic interpolator
        from scipy.interpolate import RegularGridInterpolator
        def make_periodic_interp_A(grid_2d):
            extended = np.zeros((n_registry + 1, n_registry + 1), dtype=complex)
            extended[:n_registry, :n_registry] = grid_2d
            extended[n_registry, :n_registry] = grid_2d[0, :]
            extended[:n_registry, n_registry] = grid_2d[:, 0]
            extended[n_registry, n_registry] = grid_2d[0, 0]
            x_ext = np.append(x_reg_AX, 1.0)
            y_ext = np.append(y_reg_AX, 1.0)
            return RegularGridInterpolator((x_ext, y_ext), extended,
                                           method='linear', bounds_error=False, fill_value=None)
        
        # Query points (needs delta_frac)
        # delta_frac corresponds to the s-grid mapping to registry
        # We use the same query points as BH
        delta_frac_x = delta_frac[:, :, 0]
        delta_frac_y = delta_frac[:, :, 1]
        query_x = np.mod(delta_frac_x + 0.5, 1.0)
        query_y = np.mod(delta_frac_y + 0.5, 1.0)
        query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
        
        A_berry = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2), dtype=complex)
        
        # Interpolate all components
        for m in range(N_subspace):
            for n in range(N_subspace):
                for d in range(2):
                    interp = make_periodic_interp_A(A_berry_registry[:, :, m, n, d])
                    A_berry[:, :, m, n, d] = interp(query_points).reshape(Ns1, Ns2)

        log(f"    Computed Berry connection from fields. Max value: {np.max(np.abs(A_berry)):.6e} (physical units)")
        log(f"    Raw Berry non-Hermiticity before projection: {berry_raw_diag['raw_hermiticity_max_abs']:.6e}")
        
    else:
        # Fallback to approximation (usually zeros)
        log("    WARNING: No Bloch fields available - using placeholder/approximate Berry connection")
        A_berry = compute_berry_connection_fd(omega_grid, ds1, ds2, fd_order)

    # =========================================================================
    # 2. Born-Huang Potential
    # =========================================================================
    
    # NOTE: Born-Huang Potential Magnitude
    # With proper normalization, Phi_BH should be small (perturbation).
    # It represents the energy cost of the Bloch function "twisting" in place.
    # If unnormalized, this term blows up and destroys the envelope physics.
    
    Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
    born_huang_method = "none"
    
    if include_born_huang:
        log("  Computing Born-Huang potential...")
        
        # Check if we have Bloch fields for proper computation
        if bloch_fields is not None:
            # Use proper Born-Huang computation from Bloch function derivatives
            log("    Using Bloch field method (proper theory implementation)")
            born_huang_method = "bloch_fields"
            
            from phasesV3.bloch_fields import compute_born_huang_from_fields, diagnose_born_huang_values
            
            # Physical grid spacing for derivatives
            # bloch_fields shape: (n_registry, n_registry, N_bands, Nx, Ny, 3)
            dR = (dR_registry, dR_registry)  # Assuming isotropic grid
            
            # Map subspace bands to indices in bloch_fields
            subspace_local_indices = [all_bands.index(b) for b in subspace_bands]
            extra_band_indices = [i for i, b in enumerate(all_bands) if b not in subspace_bands]
            
            # Compute Born-Huang from Bloch fields (on registry grid)
            Phi_BH_registry = compute_born_huang_from_fields(
                bloch_fields, dR, subspace_local_indices, extra_band_indices,
                epsilon=epsilon
            )
            
            # Interpolate from registry grid to full moiré grid
            log(f"    Interpolating Born-Huang from {n_registry}×{n_registry} to {Ns1}×{Ns2}...")
            from scipy.interpolate import RegularGridInterpolator
            
            # Registry grid coordinates
            step_reg = 1.0 / n_registry
            x_reg = np.linspace(0, 1 - step_reg, n_registry)
            y_reg = np.linspace(0, 1 - step_reg, n_registry)
            
            def make_periodic_interp(grid_2d):
                """Create interpolator with periodic boundary handling."""
                extended = np.zeros((n_registry + 1, n_registry + 1))
                extended[:n_registry, :n_registry] = grid_2d
                extended[n_registry, :n_registry] = grid_2d[0, :]
                extended[:n_registry, n_registry] = grid_2d[:, 0]
                extended[n_registry, n_registry] = grid_2d[0, 0]
                x_ext = np.append(x_reg, 1.0)
                y_ext = np.append(y_reg, 1.0)
                return RegularGridInterpolator((x_ext, y_ext), extended,
                                               method='linear', bounds_error=False, fill_value=None)
            
            # Query points from delta_frac grid (same as Phase 1)
            delta_frac_x = delta_frac[:, :, 0]
            delta_frac_y = delta_frac[:, :, 1]
            query_x = np.mod(delta_frac_x + 0.5, 1.0)
            query_y = np.mod(delta_frac_y + 0.5, 1.0)
            query_points = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
            
            # Interpolate each (m, n) component of Phi_BH
            Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
            for m in range(N_subspace):
                for n in range(N_subspace):
                    interp_BH = make_periodic_interp(Phi_BH_registry[:, :, m, n])
                    Phi_BH[:, :, m, n] = interp_BH(query_points).reshape(Ns1, Ns2)
            
            log(f"    Interpolated Born-Huang: shape {Phi_BH.shape}")
            
            # Diagnose values
            bh_diag = diagnose_born_huang_values(Phi_BH)
            log(f"    Born-Huang diagonal ranges: {bh_diag['diagonal_range']}")
            log(f"    Born-Huang off-diag max: {bh_diag['offdiag_max']:.6e}")
        else:
            # No Bloch fields - set Born-Huang to zero (documented placeholder)
            log("    WARNING: No Bloch fields available - Born-Huang set to ZERO")
            log("    (Enable export_bloch_fields in config to compute proper Born-Huang)")
            born_huang_method = "placeholder_zero"
            Phi_BH = np.zeros((Ns1, Ns2, N_subspace, N_subspace))
        
        log(f"    Born-Huang potential range: [{Phi_BH.min():.6e}, {Phi_BH.max():.6e}]")
    
    # =========================================================================
    # 2b. Velocity Matrix from Bloch Fields (off-diagonal v_mn)
    # =========================================================================
    # Must compute BEFORE freeing bloch_fields.
    v_mn_fields = None
    v_mn_diagnostics = None
    
    include_offdiag_v_drift = config.get('include_offdiag_v_drift', True)

    if bloch_fields is not None and include_drift_term:
        log("  Computing velocity matrix v_mn from Bloch fields...")
        
        subspace_indices = [all_bands.index(b) for b in subspace_bands]
        bloch_sub = bloch_fields[:, :, subspace_indices, ...]
        
        k0_mpb = np.array([k0_x, k0_y], dtype=float)
        log(f"    Velocity matrix carrier k0 = ({k0_x:.6f}, {k0_y:.6f}) in MPB coords")
        
        # Compute on the registry grid (n_registry × n_registry)
        v_mn_registry, v_mn_diagnostics = compute_velocity_matrix_from_fields(
            bloch_sub, k0_mpb, omega_target_abs, a=1.0,
            polarization='TM',
        )
        
        log(f"    v_mn diagonal x range: {v_mn_diagnostics['v_diag_x_range']}")
        log(f"    v_mn diagonal y range: {v_mn_diagnostics['v_diag_y_range']}")
        log(f"    v_mn off-diagonal max: {v_mn_diagnostics['offdiag_max']:.6e}")
        log(f"    v_mn Hermiticity error: {v_mn_diagnostics['hermiticity_error']:.6e}")
        
        # Interpolate from registry grid to moiré grid
        log(f"    Interpolating v_mn from {n_registry}×{n_registry} to {Ns1}×{Ns2}...")
        from scipy.interpolate import RegularGridInterpolator
        
        step_reg_vm = 1.0 / n_registry
        x_reg_vm = np.linspace(0, 1 - step_reg_vm, n_registry)
        y_reg_vm = np.linspace(0, 1 - step_reg_vm, n_registry)
        
        def _make_periodic_interp_complex(grid_2d):
            """Create periodic interpolator for complex-valued 2D field."""
            extended = np.zeros((n_registry + 1, n_registry + 1), dtype=complex)
            extended[:n_registry, :n_registry] = grid_2d
            extended[n_registry, :n_registry] = grid_2d[0, :]
            extended[:n_registry, n_registry] = grid_2d[:, 0]
            extended[n_registry, n_registry] = grid_2d[0, 0]
            x_ext = np.append(x_reg_vm, 1.0)
            y_ext = np.append(y_reg_vm, 1.0)
            return RegularGridInterpolator((x_ext, y_ext), extended,
                                           method='linear', bounds_error=False, fill_value=None)
        
        # Use same query points as Berry connection
        delta_frac_x = delta_frac[:, :, 0]
        delta_frac_y = delta_frac[:, :, 1]
        query_x = np.mod(delta_frac_x + 0.5, 1.0)
        query_y = np.mod(delta_frac_y + 0.5, 1.0)
        query_points_vm = np.stack([query_x.ravel(), query_y.ravel()], axis=-1)
        
        v_mn_fields = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2), dtype=complex)
        for m in range(N_subspace):
            for n in range(N_subspace):
                for d in range(2):
                    interp = _make_periodic_interp_complex(v_mn_registry[:, :, m, n, d])
                    v_mn_fields[:, :, m, n, d] = interp(query_points_vm).reshape(Ns1, Ns2)
        
        log(f"    Interpolated v_mn: max off-diag = {np.max(np.abs(v_mn_fields - np.einsum('ijmnk,mn->ijmnk', v_mn_fields, np.eye(N_subspace)))):.6e}")
    
    # Free large arrays no longer needed (bloch_fields ~19 GB, epsilon ~0.5 GB)
    # All downstream computations (drift, Lambda, M_inv, save) don't use them.
    _freed_info = []
    if bloch_fields is not None:
        _freed_info.append(f"bloch_fields({bloch_fields.nbytes/1e9:.1f}GB)")
        del bloch_fields
    if epsilon is not None:
        _freed_info.append(f"epsilon({epsilon.nbytes/1e9:.1f}GB)")
        del epsilon
    import gc; gc.collect()
    log(f"  Freed large arrays: {', '.join(_freed_info) if _freed_info else 'none'}")
    
    # =========================================================================
    # 3. Drift Term
    # =========================================================================
    v_drift = np.zeros((Ns1, Ns2, N_subspace, N_subspace, 2), dtype=complex)
    
    if include_drift_term:
        log("  Computing drift term...")
        # Start with stencil-computed diagonal (real-valued, most accurate for v_nn)
        v_drift_diag = compute_drift_term(vg_grid, omega_grid, omega_ref)
        v_drift[:] = v_drift_diag
        
        vg_max = np.max(np.abs(v_drift[:, :, np.arange(N_subspace), np.arange(N_subspace), :]))
        log(f"    Max diagonal group velocity (stencil): {vg_max:.6e}")
        
        # Merge off-diagonal v_mn from Bloch fields if enabled
        if v_mn_fields is not None and include_offdiag_v_drift:
            # Use field-computed off-diagonal elements
            for m in range(N_subspace):
                for n in range(N_subspace):
                    if m != n:
                        v_drift[:, :, m, n, :] = v_mn_fields[:, :, m, n, :]
            
            # Diagnostic: compare field vs stencil on diagonal
            for n in range(N_subspace):
                stencil_vn = v_drift_diag[:, :, n, n, :]   # real
                field_vn = v_mn_fields[:, :, n, n, :].real  # take real part
                rms_diff = np.sqrt(np.mean((stencil_vn - field_vn)**2))
                rms_stencil = np.sqrt(np.mean(stencil_vn**2))
                rel_err = rms_diff / (rms_stencil + 1e-15)
                log(f"    Band {subspace_bands[n]}: stencil vs field v_g "
                    f"RMS diff = {rms_diff:.4e}, rel = {rel_err:.2%}")
            
            offdiag_rms = np.sqrt(np.mean(np.abs(v_drift[:, :, ~np.eye(N_subspace, dtype=bool), :])**2))
            diag_rms = np.sqrt(np.mean(np.abs(v_drift[:, :, np.eye(N_subspace, dtype=bool), :])**2))
            log(f"    Off-diag/diag RMS ratio: {offdiag_rms/(diag_rms+1e-15):.4f}")
        elif v_mn_fields is not None:
            log("    Off-diagonal v_mn computed but disabled by config; using diagonal stencil drift only")
        else:
            log("    No Bloch fields — off-diagonal v_mn = 0")

        if not np.all(np.isfinite(v_drift)):
            bad = int(np.size(v_drift) - np.count_nonzero(np.isfinite(v_drift)))
            log(f"    WARNING: Replacing {bad} non-finite v_drift entries with zero")
            v_drift = np.nan_to_num(v_drift, nan=0.0, posinf=0.0, neginf=0.0)
    
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
    
    # M_inv as full matrix (diagonal only — off-diagonal M_inv_mn requires
    # second-order k-perturbation theory: ⟨u_m|∂²Θ/∂k_i∂k_j|u_n⟩ + sum terms.
    # The Phase 3 kinetic operator also only reads diagonal blocks.
    # Off-diagonal M_inv is O(η²) — deferred.)
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
        hf.attrs["moire_length"] = moire_length
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
        hf.attrs["include_offdiag_v_drift"] = include_offdiag_v_drift
        hf.attrs["use_parallel_transport_gauge"] = use_parallel_transport
        hf.attrs["born_huang_method"] = born_huang_method  # Track how BH was computed
        hf.attrs["pipeline_version"] = "V3"
        hf.attrs["solver"] = "mpb"

    phase2_report = phase2_sanity_report(
        A_berry=A_berry,
        Phi_BH=Phi_BH,
        v_drift=v_drift,
        M_inv=M_inv_matrix,
        Lambda=Lambda,
        berry_raw_diagnostics=locals().get('berry_raw_diag', None),
    )
    save_json(phase2_report, cdir / 'phase2_sanity_checks.json')
    log_sanity_block(log, 'Phase 2 sanity checks', phase2_report)
    
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
    log("PHASE 2 V3 (MPB): Berry Connection & Born-Huang Potential")
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
            process_candidate_phase2_v3(cdir, config)
        except Exception as e:
            cid = int(cdir.name.split('_')[-1])
            print(f"ERROR processing candidate {cid}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log("\n" + "="*70)
    log("PHASE 2 V3 (MPB) COMPLETE")
    log("="*70)
    log(f"\nOutputs saved to candidate directories in: {run_dir}")
    log("Next step: Run Phase 3 V3 for multi-band envelope solver")


def get_default_config_path() -> Path:
    return PROJECT_ROOT / "configsV3" / "phase2_mpb.yaml"


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
            # Case 1: [candidate_id] [run_dir]
            candidate_id = int(arg1)
            default_config = get_default_config_path()
            if not default_config.exists():
                raise SystemExit(f"Default config not found: {default_config}")
            os.environ['MSL_PHASE2_CANDIDATE_ID'] = str(candidate_id)
            run_phase2_v3(arg2, str(default_config))
        except ValueError:
            # Case 2: [run_dir] [candidate_id]
            try:
                candidate_id = int(arg2)
                default_config = get_default_config_path()
                if not default_config.exists():
                    raise SystemExit(f"Default config not found: {default_config}")
                os.environ['MSL_PHASE2_CANDIDATE_ID'] = str(candidate_id)
                run_phase2_v3(arg1, str(default_config))
            except ValueError:
                # Case 3: [run_dir] [config_path]
                run_phase2_v3(arg1, arg2)
    else:
        raise SystemExit(
            "Usage: python phasesV3/phase2_mpb_v3.py [candidate_id] [run_dir|auto] [config.yaml]"
        )
