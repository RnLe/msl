#!/usr/bin/env python3
"""
F06 — Part 3: SVQB B-Orthonormalization of Bloch Fields

Implements the SVQB algorithm (from Blaze) to B-orthonormalize the Bloch
E-fields at each registry point, where B = diag(ε(r)).

SVQB is more stable than Gram-Schmidt near band degeneracies. It:
  1) Pre-normalizes each vector to unit B-norm
  2) Forms the Gram matrix G = X^H · (BX)
  3) Eigendecomposes G = Q Λ Q^H
  4) Rank-reveals by dropping λ_i / λ_max < drop_tol
  5) Applies T = Q_kept · Λ_kept^{-1/2} to produce B-orthonormal vectors

After SVQB: X_new^H · B · X_new = I_rank.

Comparison:
  - Raw MPB:             ⟨u_m|ε|u_n⟩ ≈ δ_mn (from eigensolver, ~0.02 residual)
  - Simple ε-normalize:  divide by √⟨u|ε|u⟩, no cross-band correction
  - SVQB B-orthonorm:    exact B-orthonormality via eigendecomposition

Reference: findings/svqb_guide.md
"""

import numpy as np
import h5py
import sys
import time

sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'
THETA = '2.000'
CDIR = f'{SWEEP}/theta_{THETA}/candidate_0000'
N_SUB = 3


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


def apply_B_operator(u_flat, eps_flat):
    """
    Apply B = diag(ε) to a flattened Bloch field.
    
    For E-fields with 3 components at each (x,y) pixel:
      (Bu)_{x,y,c} = ε(x,y) · u_{x,y,c}
    
    Args:
        u_flat: (Nx*Ny*3,) complex vector  
        eps_flat: (Nx*Ny,) real ε values, repeated for 3 components
    
    Returns:
        Bu_flat: (Nx*Ny*3,) complex vector
    """
    return eps_flat * u_flat


def compute_B_overlap_matrix(vectors, mass_vectors, rank):
    """
    Compute the B-overlap matrix: O_{mn} = ⟨v_m | B | v_n⟩ = v_m^H · (Bv_n)
    
    Returns:
        O: (rank, rank) complex matrix (should be identity if B-orthonormal)
    """
    O = np.zeros((rank, rank), dtype=np.complex128)
    for m in range(rank):
        for n in range(rank):
            O[m, n] = np.dot(np.conj(vectors[m]), mass_vectors[n])
    return O


def run():
    t0 = time.time()
    print("=" * 70)
    print("F06 Part 3 — SVQB B-Orthonormalization of Bloch Fields")
    print("=" * 70)
    
    # =========================================================================
    # Load data
    # =========================================================================
    h5path = f'{CDIR}/phase1_multiband_data.h5'
    with h5py.File(h5path, 'r') as hf:
        bf_raw = hf['bloch_fields'][:, :, :N_SUB, :, :, :]
    
    Ns1, Ns2, _, Nx, Ny, Nc = bf_raw.shape
    Npix = Nx * Ny
    N_flat = Nx * Ny * Nc  # length of flattened vector
    print(f"  Bloch fields: ({Ns1},{Ns2},{N_SUB},{Nx},{Ny},{Nc})")
    print(f"  Flattened vector length: {N_flat}")
    
    # Load epsilon grid (from F06_epsilon_orthogonality.py output)
    eps_data = np.load(f'{FINDINGS}/F06_epsilon_data.npz')
    eps_grid = eps_data['eps_grid']  # (Ns1, Ns2, Nx, Ny)
    print(f"  ε grid shape: {eps_grid.shape}")
    print(f"  ε range: [{eps_grid.min():.3f}, {eps_grid.max():.3f}]")
    
    # Also load the raw ε-weighted orthogonality for comparison
    ortho_eps_raw = eps_data['ortho_eps_raw']  # from simple normalization
    offdiag_eps_raw = eps_data['offdiag_eps_raw']
    
    # =========================================================================
    # Method 1: Simple ε-normalization (divide by √⟨u|ε|u⟩, no cross-band fix)
    # Already computed in F06_epsilon_orthogonality.py — load results
    # =========================================================================
    ortho_simple = eps_data['ortho_eps_normed']
    offdiag_simple = eps_data['offdiag_eps_normed']
    print(f"\n  Simple ε-normalize (from previous run):")
    print(f"    max|⟨u_m|ε|u_n⟩| (m≠n): mean={offdiag_simple.mean():.6f}, max={offdiag_simple.max():.6f}")
    
    # =========================================================================
    # Method 2: SVQB B-orthonormalization
    # =========================================================================
    print(f"\n  SVQB B-orthonormalization at all {Ns1}×{Ns2} registry points...")
    
    # Store results
    bf_svqb = np.zeros_like(bf_raw, dtype=np.complex128)
    ortho_svqb = np.zeros((Ns1, Ns2, N_SUB, N_SUB))
    offdiag_svqb = np.zeros((Ns1, Ns2))
    gram_eigenvalues = np.zeros((Ns1, Ns2, N_SUB))
    rank_map = np.zeros((Ns1, Ns2), dtype=int)
    
    n_rank_loss = 0
    max_offdiag_svqb = 0.0
    
    t1 = time.time()
    for ix in range(Ns1):
        for iy in range(Ns2):
            eps_2d = eps_grid[ix, iy].astype(np.float64)  # (Nx, Ny)
            
            # Build eps repeated for 3 components: (Nx, Ny) → (Nx, Ny, 3) → flatten
            eps_3d = np.repeat(eps_2d[:, :, np.newaxis], Nc, axis=2)  # (Nx, Ny, 3)
            eps_flat = eps_3d.ravel()  # (N_flat,)
            
            # Prepare vectors and mass vectors
            vectors = []
            mass_vectors = []
            for n in range(N_SUB):
                u = bf_raw[ix, iy, n].astype(np.complex128).ravel()  # (N_flat,)
                Bu = apply_B_operator(u, eps_flat)  # (N_flat,)
                vectors.append(u)
                mass_vectors.append(Bu)
            
            # SVQB
            result = svqb_orthonormalize(vectors, mass_vectors, drop_tol=1e-12)
            
            rank = result['rank']
            rank_map[ix, iy] = rank
            if rank < N_SUB:
                n_rank_loss += 1
            
            # Store eigenvalues
            evals = result['eigenvalues']
            gram_eigenvalues[ix, iy, :len(evals)] = evals[:N_SUB]
            
            # Store B-orthonormalized vectors
            for n in range(min(rank, N_SUB)):
                bf_svqb[ix, iy, n] = result['vectors'][n].reshape(Nx, Ny, Nc)
            
            # Verify B-orthonormality: compute O = X^H B X (should be I)
            if rank > 0:
                O = compute_B_overlap_matrix(result['vectors'], result['mass_vectors'], rank)
                for m in range(rank):
                    for n in range(rank):
                        ortho_svqb[ix, iy, m, n] = np.abs(O[m, n])
                
                for m in range(rank):
                    for n in range(rank):
                        if m != n:
                            offdiag_svqb[ix, iy] = max(offdiag_svqb[ix, iy], 
                                                        np.abs(O[m, n]))
                max_offdiag_svqb = max(max_offdiag_svqb, offdiag_svqb[ix, iy])
        
        if ix % 16 == 0:
            elapsed = time.time() - t1
            print(f"    Row {ix}/{Ns1} ({elapsed:.1f}s)")
    
    t_svqb = time.time() - t1
    
    # =========================================================================
    # Results
    # =========================================================================
    print(f"\n  SVQB complete in {t_svqb:.1f}s")
    print(f"  Rank deficiency: {n_rank_loss}/{Ns1*Ns2} points lost bands")
    
    print(f"\n  SVQB B-orthonormality verification:")
    for n in range(N_SUB):
        diag = ortho_svqb[:, :, n, n].ravel()
        print(f"    ⟨u_{n}|ε|u_{n}⟩: mean={diag.mean():.10f}, "
              f"std={diag.std():.2e}, "
              f"max|1-diag|={np.abs(diag - 1.0).max():.2e}")
    
    print(f"\n  Off-diagonal (should be ~machine epsilon):")
    print(f"    max|⟨u_m|ε|u_n⟩| (m≠n): mean={offdiag_svqb.mean():.2e}, max={offdiag_svqb.max():.2e}")
    for m in range(N_SUB):
        for n in range(m + 1, N_SUB):
            vals = ortho_svqb[:, :, m, n].ravel()
            print(f"    |⟨u_{m}|ε|u_{n}⟩|: mean={vals.mean():.2e}, max={vals.max():.2e}")
    
    # =========================================================================
    # Comparison table
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Raw vs Simple-ε-norm vs SVQB")
    print(f"{'='*70}")
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    print(f"\n  {'Method':<25s}  {'(0,1) mean':>10s}  {'(0,1) max':>10s}  "
          f"{'(0,2) mean':>10s}  {'(0,2) max':>10s}  "
          f"{'(1,2) mean':>10s}  {'(1,2) max':>10s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    
    for label, data in [
        ("Raw ε-weighted", ortho_eps_raw),
        ("Simple ε-norm", ortho_simple),
        ("SVQB B-orthonorm", ortho_svqb),
    ]:
        vals = []
        for m, n in pairs:
            d = data[:, :, m, n].ravel()
            vals.extend([d.mean(), d.max()])
        print(f"  {label:<25s}  {vals[0]:10.6f}  {vals[1]:10.6f}  "
              f"{vals[2]:10.6f}  {vals[3]:10.6f}  "
              f"{vals[4]:10.6f}  {vals[5]:10.6f}")
    
    # =========================================================================
    # Gram eigenvalue statistics (condition of the Gram matrix)
    # =========================================================================
    print(f"\n  Gram matrix eigenvalue statistics:")
    for n in range(N_SUB):
        ev = gram_eigenvalues[:, :, n].ravel()
        print(f"    λ_{n}: mean={ev.mean():.6f}, min={ev.min():.6f}, max={ev.max():.6f}")
    
    # Condition number = λ_max / λ_min
    cond = gram_eigenvalues[:, :, 0] / np.maximum(gram_eigenvalues[:, :, N_SUB-1], 1e-30)
    print(f"    Condition (λ_0/λ_{N_SUB-1}): mean={cond.mean():.2f}, max={cond.max():.2f}")
    
    # =========================================================================
    # Also check: how much did SVQB change the fields (vs raw)?
    # =========================================================================
    print(f"\n  Field change from SVQB (vs raw):")
    for n in range(N_SUB):
        # Compute ||u_svqb - u_raw|| / ||u_raw|| at each point
        diff_norms = []
        for ix in range(Ns1):
            for iy in range(Ns2):
                u_raw = bf_raw[ix, iy, n].ravel().astype(np.complex128)
                u_svqb = bf_svqb[ix, iy, n].ravel()
                
                # Phase-align (SVQB may introduce a global phase)
                ov = np.dot(np.conj(u_svqb), u_raw)
                if abs(ov) > 1e-10:
                    phase = np.conj(ov) / abs(ov)
                    u_svqb_aligned = u_svqb * phase
                else:
                    u_svqb_aligned = u_svqb
                
                raw_norm = np.linalg.norm(u_raw)
                if raw_norm > 1e-10:
                    d = np.linalg.norm(u_svqb_aligned - u_raw) / raw_norm
                    diff_norms.append(d)
        
        diff_norms = np.array(diff_norms)
        print(f"    Band {n}: ||Δu||/||u|| mean={diff_norms.mean():.6f}, max={diff_norms.max():.6f}")
    
    # =========================================================================
    # Save
    # =========================================================================
    np.savez_compressed(
        f'{FINDINGS}/F06_svqb_data.npz',
        bf_svqb=bf_svqb.astype(np.complex64),  # save space
        ortho_svqb=ortho_svqb,
        offdiag_svqb=offdiag_svqb,
        gram_eigenvalues=gram_eigenvalues,
        rank_map=rank_map,
        Ns1=Ns1, Ns2=Ns2, N_sub=N_SUB, Nx=Nx, Ny=Ny,
    )
    print(f"\n  Saved: {FINDINGS}/F06_svqb_data.npz")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    
    print(f"\n  Key result:")
    print(f"    Simple ε-norm max offdiag: {offdiag_simple.max():.6f}")
    print(f"    SVQB max offdiag:          {offdiag_svqb.max():.2e}")
    improvement = offdiag_simple.max() / max(offdiag_svqb.max(), 1e-30)
    print(f"    → SVQB improvement:        {improvement:.0f}×")


if __name__ == '__main__':
    run()
