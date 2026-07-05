#!/usr/bin/env python3
"""
S3e: Subspace smoothness test for original bands [5-9]
=======================================================
S3d showed that [5-9] is 99.4% C4-equivariant. Now test whether 
the SUBSPACE is also smooth (adjacent-point overlap), which is 
required for the within-subspace gauge fix to produce good results.

If the subspace is smooth AND equivariant, then:
1. The original index-based approach was correct
2. The BFS gauge fix WITHIN [5-9] is sufficient  
3. The (32,32) failure is a localized defect, not a global problem
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "plots"
OUTDIR.mkdir(exist_ok=True)
BASE   = Path(__file__).resolve().parent.parent
CAND   = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
P1     = CAND / "phase1_multiband_data.h5"

SUBSPACE = [5, 6, 7, 8, 9]
N_SUB = 5
N_ALL = 18
Nx = 32

def eps_inner(u1, u2, eps):
    return np.sum(eps[:, :, None] * np.conj(u1) * u2) / (Nx * Nx)

def c4_registry(ix, iy, Nr):
    return (Nr - iy) % Nr, ix

def rotate_field_c4(field):
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated


def subspace_overlap(states1, states2, eps):
    """Subspace overlap between two sets of states (phase-independent).
    Returns min singular value of normalized overlap matrix."""
    N = len(states1)
    M = np.zeros((N, N), dtype=complex)
    for m in range(N):
        for n in range(N):
            M[m, n] = eps_inner(states1[m], states2[n], eps)
    norm1 = [np.sqrt(np.abs(eps_inner(u, u, eps))) for u in states1]
    norm2 = [np.sqrt(np.abs(eps_inner(u, u, eps))) for u in states2]
    M_n = M / (np.outer(norm1, norm2) + 1e-30)
    return np.linalg.svd(M_n, compute_uv=False).min()


def main():
    print("="*70)
    print("S3e: Original [5-9] subspace smoothness & equivariance")
    print("="*70)

    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]
        eps = f['epsilon'][:]
        omega = f['omega'][:]
    Nr = bf.shape[0]
    print(f"Grid: {Nr}×{Nr}, {N_ALL} bands, unit cell {Nx}×{Nx}")
    print(f"omega shape: {omega.shape}")

    # ── 1. Subspace smoothness ──
    print(f"\n{'='*70}")
    print("1. SUBSPACE smoothness: adjacent-point overlap for [5-9]")
    print("="*70)

    adj_x = np.zeros((Nr, Nr))
    adj_y = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            s1 = [bf[ix, iy, b] for b in SUBSPACE]
            # x-neighbor
            nix = (ix + 1) % Nr
            s2x = [bf[nix, iy, b] for b in SUBSPACE]
            adj_x[ix, iy] = subspace_overlap(s1, s2x, eps[ix, iy])
            # y-neighbor
            niy = (iy + 1) % Nr
            s2y = [bf[ix, niy, b] for b in SUBSPACE]
            adj_y[ix, iy] = subspace_overlap(s1, s2y, eps[ix, iy])

    adj_min = np.minimum(adj_x, adj_y)
    print(f"  Overall: min={adj_min.min():.4f}, mean={adj_min.mean():.4f}, "
          f"median={np.median(adj_min):.4f}")
    print(f"  >0.99: {np.sum(adj_min>0.99)/(Nr*Nr):.1%}")
    print(f"  >0.95: {np.sum(adj_min>0.95)/(Nr*Nr):.1%}")
    print(f"  >0.90: {np.sum(adj_min>0.90)/(Nr*Nr):.1%}")
    print(f"  >0.50: {np.sum(adj_min>0.50)/(Nr*Nr):.1%}")
    print(f"  <0.50: {np.sum(adj_min<0.50)/(Nr*Nr):.1%}")

    # Find worst points
    bad_mask = adj_min < 0.5
    bad_pts = np.argwhere(bad_mask)
    print(f"\n  Points with subspace overlap < 0.5:")
    for pt in bad_pts[:10]:
        ix, iy = pt
        print(f"    ({ix},{iy}): adj_x={adj_x[ix,iy]:.4f}, adj_y={adj_y[ix,iy]:.4f}")

    # ── 2. Band gaps at boundary ──
    print(f"\n{'='*70}")
    print("2. Band gap at subspace boundary: Δω(4↔5) and Δω(9↔10)")
    print("="*70)

    if omega.shape[2] >= N_ALL:
        # omega is (Nr, Nr, N_bands) or similar
        omega_grid = omega
    elif omega.shape[0] > Nr:
        # omega might be on finer grid; subsample
        step = omega.shape[0] // Nr
        omega_grid = omega[::step, ::step]
    else:
        omega_grid = omega

    # We have omega(Nr_om, Nr_om, 5) for the 5 bands [5-9]
    Nr_om = omega_grid.shape[0]
    print(f"  omega grid: {omega_grid.shape}")

    # The omega file has 5 bands = [5-9] in the subspace
    # We need the FULL frequency spectrum to check gaps at boundaries
    # Let's compute from the raw MPB data if available
    # For now, just report the gap between bands within [5-9]
    
    if omega_grid.shape[2] == 5:
        print(f"  (omega grid has only 5 bands — subspace [5-9] only)")
        print(f"  Min gap within subspace: Δω(5↔6) to Δω(8↔9)")
        for pair in range(4):
            gap = omega_grid[:, :, pair+1] - omega_grid[:, :, pair]
            min_gap = gap.min()
            min_loc = np.unravel_index(gap.argmin(), gap.shape)
            print(f"    band {5+pair}↔{6+pair}: min gap = {min_gap:.6f} "
                  f"at ({min_loc[0]},{min_loc[1]})")
    else:
        print(f"  Full spectrum available")

    # ── 3. C4 equivariance (confirmed) ──
    print(f"\n{'='*70}")
    print("3. C4 equivariance recap for [5-9]")
    print("="*70)

    eqv = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            s_R = [bf[ix, iy, b] for b in SUBSPACE]
            jx, jy = c4_registry(ix, iy, Nr)
            s_C4R = [bf[jx, jy, b] for b in SUBSPACE]
            # C4-rotate states at R
            rot_R = [rotate_field_c4(u) for u in s_R]
            M = np.zeros((N_SUB, N_SUB), dtype=complex)
            for m in range(N_SUB):
                for n in range(N_SUB):
                    M[m, n] = eps_inner(rot_R[m], s_C4R[n], eps[jx, jy])
            norm1 = [np.sqrt(np.abs(eps_inner(u, u, eps[jx,jy]))) for u in rot_R]
            norm2 = [np.sqrt(np.abs(eps_inner(u, u, eps[jx,jy]))) for u in s_C4R]
            M_n = M / (np.outer(norm1, norm2) + 1e-30)
            eqv[ix, iy] = np.linalg.svd(M_n, compute_uv=False).min()

    print(f"  >0.99: {np.sum(eqv>0.99)/(Nr*Nr):.1%}")
    print(f"  >0.95: {np.sum(eqv>0.95)/(Nr*Nr):.1%}")
    print(f"  >0.90: {np.sum(eqv>0.90)/(Nr*Nr):.1%}")

    # ── 4. Combined map ──
    print(f"\n{'='*70}")
    print("4. Combined: smooth AND equivariant")
    print("="*70)

    both_90 = (adj_min > 0.9) & (eqv > 0.9)
    both_50 = (adj_min > 0.5) & (eqv > 0.5)
    print(f"  Both > 0.9: {np.sum(both_90)/(Nr*Nr):.1%}")
    print(f"  Both > 0.5: {np.sum(both_50)/(Nr*Nr):.1%}")

    # Failure classification
    fail_eqv = eqv < 0.5
    fail_smooth = adj_min < 0.5
    fail_both = fail_eqv & fail_smooth
    fail_eqv_only = fail_eqv & ~fail_smooth
    fail_smooth_only = fail_smooth & ~fail_eqv
    
    print(f"\n  Failure classification (threshold 0.5):")
    print(f"    C4 equivariance only:  {np.sum(fail_eqv_only)}/{Nr*Nr} = "
          f"{np.sum(fail_eqv_only)/(Nr*Nr):.2%}")
    print(f"    Smoothness only:       {np.sum(fail_smooth_only)}/{Nr*Nr} = "
          f"{np.sum(fail_smooth_only)/(Nr*Nr):.2%}")
    print(f"    Both fail:             {np.sum(fail_both)}/{Nr*Nr} = "
          f"{np.sum(fail_both)/(Nr*Nr):.2%}")
    print(f"    Neither fail:          {np.sum(~fail_eqv & ~fail_smooth)}/{Nr*Nr} = "
          f"{np.sum(~fail_eqv & ~fail_smooth)/(Nr*Nr):.1%}")

    # ── 5. Radial distribution of failures around (32,32) ──
    print(f"\n{'='*70}")
    print("5. Radial distribution of failures around δ=(0.5,0.5)")
    print("="*70)

    center = (32, 32)
    for r_max in [2, 3, 5, 8, 10, 16]:
        in_circle_eqv = []
        in_circle_smooth = []
        for ix in range(Nr):
            for iy in range(Nr):
                dx = min(abs(ix - center[0]), Nr - abs(ix - center[0]))
                dy = min(abs(iy - center[1]), Nr - abs(iy - center[1]))
                if dx*dx + dy*dy <= r_max*r_max:
                    in_circle_eqv.append(eqv[ix, iy])
                    in_circle_smooth.append(adj_min[ix, iy])
        in_circle_eqv = np.array(in_circle_eqv)
        in_circle_smooth = np.array(in_circle_smooth)
        n = len(in_circle_eqv)
        eqv_bad = np.sum(in_circle_eqv < 0.5)
        smo_bad = np.sum(in_circle_smooth < 0.5)
        print(f"    r≤{r_max:2d}: {n:4d} pts, eqv<0.5: {eqv_bad:3d}, "
              f"smooth<0.5: {smo_bad:3d}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    pct_both = np.sum(both_90) / (Nr*Nr)
    if pct_both > 0.95:
        print(f"✓ Original [5-9] subspace is {pct_both:.1%} smooth+equivariant")
        print(f"  → Within-subspace gauge fix is sufficient")
        print(f"  → Handle {Nr*Nr - np.sum(both_90)} defect points near (32,32)")
        print(f"  → No Wannier/band-tracking needed!")
    elif pct_both > 0.80:
        print(f"⚠ Original [5-9] is {pct_both:.1%} smooth+equivariant")
        print(f"  → Mostly viable, some problem regions need attention")
    else:
        print(f"❌ Original [5-9] only {pct_both:.1%} smooth+equivariant")
        print(f"  → Significant issues remain")
    print("="*70)

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    im0 = axes[0, 0].pcolormesh(sr, sr, adj_min.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('Subspace smoothness (min adj overlap)')
    axes[0, 0].set_aspect('equal'); plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].pcolormesh(sr, sr, eqv.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('C4 equivariance')
    axes[0, 1].set_aspect('equal'); plt.colorbar(im1, ax=axes[0, 1])

    # Combined: min of both
    combined = np.minimum(adj_min, eqv)
    im2 = axes[0, 2].pcolormesh(sr, sr, combined.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[0, 2].set_title('Combined (min of smooth, eqv)')
    axes[0, 2].set_aspect('equal'); plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].pcolormesh(sr, sr, adj_x.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('X-neighbor overlap')
    axes[1, 0].set_aspect('equal'); plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(sr, sr, adj_y.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('Y-neighbor overlap')
    axes[1, 1].set_aspect('equal'); plt.colorbar(im4, ax=axes[1, 1])

    # Zoom near (0.5, 0.5)
    mask_zoom = (sr >= 0.3) & (sr <= 0.7)
    ix_zoom = np.where(mask_zoom)[0]
    sr_zoom = sr[ix_zoom]
    comb_zoom = combined[np.ix_(ix_zoom, ix_zoom)]
    im5 = axes[1, 2].pcolormesh(sr_zoom, sr_zoom, comb_zoom.T, cmap='RdYlGn',
                                 shading='auto', vmin=0, vmax=1)
    axes[1, 2].set_title('Combined (zoom 0.3-0.7)')
    axes[1, 2].set_aspect('equal'); plt.colorbar(im5, ax=axes[1, 2])

    # Mark fixed points and failure zones
    for row in range(2):
        for col in range(3):
            axes[row, col].plot(sr[0], sr[0], 'k*', markersize=8)
            axes[row, col].plot(sr[32], sr[32], 'k*', markersize=8)

    fig.suptitle('S3e: Original [5-9] — smoothness + equivariance', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3e_subspace_quality.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved S3e_subspace_quality.png")


if __name__ == '__main__':
    main()
