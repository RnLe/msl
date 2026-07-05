#!/usr/bin/env python3
"""
S3d: Definitive C4 test — EQUIVARIANCE vs INVARIANCE
=====================================================
KEY INSIGHT: All prior S1-S3 diagnostics used the WRONG C4 test for 
generic (non-fixed) points!

INVARIANCE test (what we did):  Is the subspace at R self-C4 invariant?
   → Only correct at C4 fixed points (0,0) and (0.5,0.5)
   → At generic R, C4 maps R→C4R, so invariance is NOT required.

EQUIVARIANCE test (what we need): Does C4-rotating the subspace at R 
   give the subspace at C4·R?
   → P(C4R) = C4·P(R)·C4⁻¹  (projector equivariance)
   → Correct for ALL points

This script runs BOTH tests to see if the "98.3% C4 broken" finding
was partially a test artifact.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

OUTDIR = Path(__file__).resolve().parent / "plots"
OUTDIR.mkdir(exist_ok=True)

BASE  = Path(__file__).resolve().parent.parent
CAND  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
P1    = CAND / "phase1_multiband_data.h5"

SUBSPACE = [5, 6, 7, 8, 9]
N_SUB = 5
N_ALL = 18
Nx = 32


# ══════════════════════════════════════════════════════════════════════════
# C4 actions
# ══════════════════════════════════════════════════════════════════════════

def c4_registry(ix, iy, Nr):
    return (Nr - iy) % Nr, ix

def rotate_field_c4(field):
    """C4 (90° CCW): u'(C4r) = R·u(r) with R = [[0,-1],[1,0],[0,0,1]]"""
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated


# ══════════════════════════════════════════════════════════════════════════
# Inner products 
# ══════════════════════════════════════════════════════════════════════════

def eps_inner(u1, u2, eps):
    return np.sum(eps[:, :, None] * np.conj(u1) * u2) / (Nx * Nx)


# ══════════════════════════════════════════════════════════════════════════
# Two different C4 metrics
# ══════════════════════════════════════════════════════════════════════════

def c4_invariance_test(states, eps):
    """INVARIANCE: rotate states at R, compare with SAME states at R.
    Only meaningful at C4 fixed points."""
    Nb = len(states)
    rotated = [rotate_field_c4(u) for u in states]
    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            M[m, n] = eps_inner(states[m], rotated[n], eps)
    # Normalize
    norms = np.sqrt(np.abs(np.diag(M)))
    M_n = M / (np.outer(norms, norms) + 1e-30)
    return np.linalg.svd(M_n, compute_uv=False).min()

def c4_equivariance_test(states_R, states_C4R, eps_C4R):
    """EQUIVARIANCE: rotate states at R, compare with states at C4·R.
    Correct for ALL points.
    Returns min singular value of overlap matrix."""
    Nb = len(states_R)
    rotated_R = [rotate_field_c4(u) for u in states_R]
    
    # Overlap: <C4·u_m(R) | ε(C4R) | u_n(C4R)>
    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            M[m, n] = eps_inner(rotated_R[m], states_C4R[n], eps_C4R)
    # Normalize rows and columns
    norm_rot = [np.sqrt(np.abs(eps_inner(u, u, eps_C4R))) for u in rotated_R]
    norm_c4r = [np.sqrt(np.abs(eps_inner(u, u, eps_C4R))) for u in states_C4R]
    M_n = M / (np.outer(norm_rot, norm_c4r) + 1e-30)
    return np.linalg.svd(M_n, compute_uv=False).min()


# ══════════════════════════════════════════════════════════════════════════
# BFS parallel transport
# ══════════════════════════════════════════════════════════════════════════

def parallel_transport_step(parent_states, child_all_bands, eps_parent):
    O = np.zeros((N_SUB, N_ALL), dtype=complex)
    for m in range(N_SUB):
        for b in range(N_ALL):
            O[m, b] = eps_inner(parent_states[m], child_all_bands[b], eps_parent)
    U, sigma, Vt = np.linalg.svd(O, full_matrices=False)
    V = Vt.conj().T
    M_mix = V @ U.conj().T
    transported = []
    for m in range(N_SUB):
        state = np.zeros_like(child_all_bands[0])
        for b in range(N_ALL):
            state += M_mix[b, m] * child_all_bands[b]
        transported.append(state)
    return transported, sigma.min()


def bfs_transport(bf, eps):
    Nr = bf.shape[0]
    transported = np.zeros((Nr, Nr, N_SUB, Nx, Nx, 3), dtype=np.complex64)
    quality = np.zeros((Nr, Nr))
    visited = np.zeros((Nr, Nr), dtype=bool)

    for m, b in enumerate(SUBSPACE):
        transported[0, 0, m] = bf[0, 0, b]
    quality[0, 0] = 1.0
    visited[0, 0] = True

    queue = deque()
    for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nix, niy = dix % Nr, diy % Nr
        queue.append((nix, niy, 0, 0))

    while queue:
        ix, iy, pix, piy = queue.popleft()
        if visited[ix, iy]:
            continue
        visited[ix, iy] = True
        parent = [transported[pix, piy, m] for m in range(N_SUB)]
        child = [bf[ix, iy, b] for b in range(N_ALL)]
        new_states, q = parallel_transport_step(parent, child, eps[pix, piy])
        for m in range(N_SUB):
            transported[ix, iy, m] = new_states[m].astype(np.complex64)
        quality[ix, iy] = q
        for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nix = (ix + dix) % Nr
            niy = (iy + diy) % Nr
            if not visited[nix, niy]:
                queue.append((nix, niy, ix, iy))

    return transported, quality


# ══════════════════════════════════════════════════════════════════════════
# Smoothness (adjacent overlap via subspace projection)
# ══════════════════════════════════════════════════════════════════════════

def subspace_adjacent_overlap(states_grid, eps_grid, Nr):
    """Min singular value of overlap between states at adjacent points.
    Returns array of shape (Nr, Nr, 2) for x- and y-neighbors."""
    adj = np.zeros((Nr, Nr, 2))
    for ix in range(Nr):
        for iy in range(Nr):
            for d, (dix, diy) in enumerate([(1, 0), (0, 1)]):
                nix = (ix + dix) % Nr
                niy = (iy + diy) % Nr
                M = np.zeros((N_SUB, N_SUB), dtype=complex)
                for m in range(N_SUB):
                    for n in range(N_SUB):
                        M[m, n] = eps_inner(states_grid[ix, iy, m],
                                            states_grid[nix, niy, n],
                                            eps_grid[ix, iy])
                # Normalize
                norm1 = [np.sqrt(np.abs(eps_inner(states_grid[ix,iy,m],
                         states_grid[ix,iy,m], eps_grid[ix,iy]))) for m in range(N_SUB)]
                norm2 = [np.sqrt(np.abs(eps_inner(states_grid[nix,niy,n],
                         states_grid[nix,niy,n], eps_grid[ix,iy]))) for n in range(N_SUB)]
                M_n = M / (np.outer(norm1, norm2) + 1e-30)
                adj[ix, iy, d] = np.linalg.svd(M_n, compute_uv=False).min()
    return adj.min(axis=2)  # worst of x- and y-neighbor


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("S3d: DEFINITIVE C4 EQUIVARIANCE vs INVARIANCE TEST")
    print("="*70)

    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]
        eps = f['epsilon'][:]
    Nr = bf.shape[0]
    print(f"Grid: {Nr}×{Nr}, {N_ALL} bands, unit cell {Nx}×{Nx}")

    # ── Identify C4 structure ──
    fixed_pts = []
    for ix in range(Nr):
        for iy in range(Nr):
            jx, jy = c4_registry(ix, iy, Nr)
            if jx == ix and jy == iy:
                fixed_pts.append((ix, iy))
    print(f"C4 fixed points: {fixed_pts}")

    # ── Test 1: Original bands [5-9] ──
    print(f"\n{'='*70}")
    print("Test 1: Original bands [5-9] — EQUIVARIANCE vs INVARIANCE")
    print("="*70)

    inv_orig = np.zeros((Nr, Nr))
    eqv_orig = np.zeros((Nr, Nr))

    for ix in range(Nr):
        for iy in range(Nr):
            states_R = [bf[ix, iy, b] for b in SUBSPACE]
            inv_orig[ix, iy] = c4_invariance_test(states_R, eps[ix, iy])

            jx, jy = c4_registry(ix, iy, Nr)
            states_C4R = [bf[jx, jy, b] for b in SUBSPACE]
            eqv_orig[ix, iy] = c4_equivariance_test(states_R, states_C4R, eps[jx, jy])

    print(f"  INVARIANCE  (same-pt): >0.9: {np.sum(inv_orig>0.9)/(Nr*Nr):.1%}, "
          f"mean={inv_orig.mean():.4f}")
    print(f"  EQUIVARIANCE (cross):  >0.9: {np.sum(eqv_orig>0.9)/(Nr*Nr):.1%}, "
          f"mean={eqv_orig.mean():.4f}")

    for ix, iy in fixed_pts:
        s = f"({ix},{iy})"
        print(f"    Fixed pt {s:>10}: inv={inv_orig[ix,iy]:.4f}, eqv={eqv_orig[ix,iy]:.4f}")

    # ── Test 2: BFS parallel transport ──
    print(f"\n{'='*70}")
    print("Test 2: BFS parallel transport — EQUIVARIANCE vs INVARIANCE")
    print("="*70)
    
    print("  Running BFS transport...")
    transported, quality = bfs_transport(bf, eps)
    print(f"  Transport quality: min={quality.min():.4f}, mean={quality.mean():.4f}")

    inv_bfs = np.zeros((Nr, Nr))
    eqv_bfs = np.zeros((Nr, Nr))

    for ix in range(Nr):
        for iy in range(Nr):
            states_R = [transported[ix, iy, m] for m in range(N_SUB)]
            inv_bfs[ix, iy] = c4_invariance_test(states_R, eps[ix, iy])

            jx, jy = c4_registry(ix, iy, Nr)
            states_C4R = [transported[jx, jy, m] for m in range(N_SUB)]
            eqv_bfs[ix, iy] = c4_equivariance_test(states_R, states_C4R, eps[jx, jy])

    print(f"  INVARIANCE:   >0.9: {np.sum(inv_bfs>0.9)/(Nr*Nr):.1%}, "
          f"mean={inv_bfs.mean():.4f}")
    print(f"  EQUIVARIANCE: >0.9: {np.sum(eqv_bfs>0.9)/(Nr*Nr):.1%}, "
          f"mean={eqv_bfs.mean():.4f}")

    for ix, iy in fixed_pts:
        s = f"({ix},{iy})"
        print(f"    Fixed pt {s:>10}: inv={inv_bfs[ix,iy]:.4f}, eqv={eqv_bfs[ix,iy]:.4f}")


    # ── Compare at key non-fixed points ──
    print(f"\n{'='*70}")
    print("Key NON-fixed points")
    print("="*70)

    key_pts = [
        ("(16,16)", 16, 16, "→ C4: (48,16)"),
        ("(10,5)",  10,  5, "→ C4: (59,10)"),
        ("(32,0)",  32,  0, "→ C4: (0,32)"),
        ("(16,0)",  16,  0, "→ C4: (0,16)"),
        ("(48,16)", 48, 16, "→ C4: (48,48)"),
    ]
    for label, ix, iy, arrow in key_pts:
        jx, jy = c4_registry(ix, iy, Nr)
        print(f"  {label} {arrow}:")
        print(f"    ORIG:  inv={inv_orig[ix,iy]:.4f}, eqv={eqv_orig[ix,iy]:.4f}")
        print(f"    BFS:   inv={inv_bfs[ix,iy]:.4f},  eqv={eqv_bfs[ix,iy]:.4f}")

    # ── Smoothness of BFS transported ──
    print(f"\n{'='*70}")
    print("Smoothness: adjacent-point subspace overlap (BFS)")
    print("="*70)

    adj_bfs = subspace_adjacent_overlap(transported, eps, Nr)
    print(f"  min={adj_bfs.min():.4f}, mean={adj_bfs.mean():.4f}, "
          f"median={np.median(adj_bfs):.4f}")
    print(f"  >0.90: {np.sum(adj_bfs>0.9)/(Nr*Nr):.1%}")
    print(f"  >0.50: {np.sum(adj_bfs>0.5)/(Nr*Nr):.1%}")

    # ── Combined verdict ──
    print(f"\n{'='*70}")
    print("COMBINED VERDICT")
    print("="*70)
    
    # Good = equivariance > 0.5 AND smoothness > 0.5
    good_eqv = eqv_bfs > 0.5
    good_smooth = adj_bfs > 0.5
    good_both = good_eqv & good_smooth
    
    print(f"  BFS transport:")
    print(f"    C4-equivariant (>0.5): {np.sum(good_eqv)/(Nr*Nr):.1%}")
    print(f"    Smooth adj (>0.5):     {np.sum(good_smooth)/(Nr*Nr):.1%}")
    print(f"    Both:                  {np.sum(good_both)/(Nr*Nr):.1%}")
    
    # Same with higher threshold
    good_eqv90 = eqv_bfs > 0.9
    good_smooth90 = adj_bfs > 0.9
    good_both90 = good_eqv90 & good_smooth90
    print(f"\n    C4-equivariant (>0.9): {np.sum(good_eqv90)/(Nr*Nr):.1%}")
    print(f"    Smooth adj (>0.9):     {np.sum(good_smooth90)/(Nr*Nr):.1%}")
    print(f"    Both:                  {np.sum(good_both90)/(Nr*Nr):.1%}")

    if np.sum(eqv_bfs > 0.9) / (Nr*Nr) > 0.8:
        print(f"\n  ✓ BFS transport is C4-EQUIVARIANT — prior test was wrong!")
        print(f"    The subspace transforms correctly under C4.")
        print(f"    The invariance test fails are just non-fixed points (expected).")
    elif np.sum(eqv_bfs > 0.5) / (Nr*Nr) > 0.5:
        print(f"\n  ⚠ BFS transport partially C4-equivariant")
        print(f"    Some points transform correctly, others don't.")
    else:
        print(f"\n  ❌ BFS transport genuinely fails C4 equivariance")
        print(f"    The BFS path dependence produces incompatible subspaces")
        print(f"    at C4-related points. This is a real obstacle.")

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    for col, (name, inv_data, eqv_data) in enumerate([
        ("Original [5-9]", inv_orig, eqv_orig),
        ("BFS transported", inv_bfs, eqv_bfs),
    ]):
        im1 = axes[0, col].pcolormesh(sr, sr, inv_data.T, cmap='RdYlGn',
                                       shading='auto', vmin=0, vmax=1)
        axes[0, col].set_title(f'INVARIANCE — {name}')
        axes[0, col].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, col])

        im2 = axes[1, col].pcolormesh(sr, sr, eqv_data.T, cmap='RdYlGn',
                                       shading='auto', vmin=0, vmax=1)
        axes[1, col].set_title(f'EQUIVARIANCE — {name}')
        axes[1, col].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1, col])

    # Smoothness
    im5 = axes[0, 2].pcolormesh(sr, sr, adj_bfs.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[0, 2].set_title('Smoothness — BFS transported')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(im5, ax=axes[0, 2])

    # Equivariance - Invariance diff
    diff = eqv_bfs - inv_bfs
    im6 = axes[1, 2].pcolormesh(sr, sr, diff.T, cmap='RdBu',
                                 shading='auto', vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title('Eqv - Inv (BFS) [blue=eqv better]')
    axes[1, 2].set_aspect('equal')
    plt.colorbar(im6, ax=axes[1, 2])

    # Mark fixed points
    for row in range(2):
        for col in range(3):
            for ix, iy in fixed_pts:
                axes[row, col].plot(sr[ix], sr[iy], 'k*', markersize=8)

    fig.suptitle('S3d: C4 EQUIVARIANCE vs INVARIANCE — definitive test', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3d_equivariance.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved S3d_equivariance.png")


if __name__ == '__main__':
    main()
