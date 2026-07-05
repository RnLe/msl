#!/usr/bin/env python3
"""
S3c: C4-symmetrized parallel transport
========================================
S3b showed that BFS parallel transport gives a SMOOTH subspace (local
overlap >0.95) but NOT C4-symmetric (BFS path dependence → holonomy drift).

Fix: symmetrize the transported subspace using C4 orbit averaging.
At each R, combine transported states from R and its 3 C4 images,
then SVD to extract the 5D intersection → C4-symmetric by construction.

Also test: fundamental-domain transport (transport only in 1/4 cell,
C4-extend the rest).
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
    """C4 (90° CCW) on registry grid."""
    return (Nr - iy) % Nr, ix

def c4_inv_registry(ix, iy, Nr):
    """C4⁻¹ (90° CW) on registry grid."""
    return iy, (Nr - ix) % Nr

def c4_orbit(ix, iy, Nr):
    """Return 4 members of the C4 orbit: [R, C4R, C4²R, C4³R]."""
    pts = [(ix, iy)]
    cx, cy = ix, iy
    for _ in range(3):
        cx, cy = c4_registry(cx, cy, Nr)
        pts.append((cx, cy))
    return pts

def rotate_field_c4(field):
    """Apply C4 (90° CCW) to a vector field on the unit cell.
    Maps u(r) → u'(r) where u'(C4r) = R·u(r), R = [[-1,0],[1,0],[0,0,1]].
    Equivalently: u'(jx,jy)_x = -u(ix,iy)_y, u'(jx,jy)_y = u(ix,iy)_x
    with jx=(Nx-iy)%Nx, jy=ix."""
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated

def rotate_field_c4_inv(field):
    """Apply C4⁻¹ (90° CW) to a vector field."""
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = iy
            jy = (Nx - ix) % Nx
            rotated[jx, jy, 0] =  field[ix, iy, 1]
            rotated[jx, jy, 1] = -field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated

def rotate_field_c4_n(field, n):
    """Apply C4^n to field (n=0,1,2,3)."""
    u = field.copy()
    for _ in range(n % 4):
        u = rotate_field_c4(u)
    return u

def rotate_field_c4_inv_n(field, n):
    """Apply C4^{-n} to field (n=0,1,2,3)."""
    u = field.copy()
    for _ in range(n % 4):
        u = rotate_field_c4_inv(u)
    return u


# ══════════════════════════════════════════════════════════════════════════
# Inner products and metrics
# ══════════════════════════════════════════════════════════════════════════

def eps_inner(u1, u2, eps):
    return np.sum(eps[:, :, None] * np.conj(u1) * u2) / (Nx * Nx)

def flat_inner(u1, u2):
    return np.sum(np.conj(u1) * u2) / (Nx * Nx)

def c4_closure_metric(states, eps=None):
    Nb = len(states)
    rotated = [rotate_field_c4(u) for u in states]
    if eps is not None:
        norms = [np.sqrt(np.abs(eps_inner(u, u, eps))) for u in states]
    else:
        norms = [np.sqrt(np.abs(flat_inner(u, u))) for u in states]

    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            if eps is not None:
                M[m, n] = eps_inner(states[m], rotated[n], eps)
            else:
                M[m, n] = flat_inner(states[m], rotated[n])
    M_norm = M / np.outer(norms, norms)
    sv = np.linalg.svd(M_norm, compute_uv=False)
    return np.abs(np.linalg.det(M_norm)), sv.min()


# ══════════════════════════════════════════════════════════════════════════
# BFS parallel transport (same as S3b)
# ══════════════════════════════════════════════════════════════════════════

def parallel_transport_step(parent_states, child_all_bands, eps_parent):
    """Non-Abelian parallel transport: project parent subspace onto child bands."""
    O = np.zeros((N_SUB, N_ALL), dtype=complex)
    for m in range(N_SUB):
        for b in range(N_ALL):
            O[m, b] = eps_inner(parent_states[m], child_all_bands[b], eps_parent)

    U, sigma, Vt = np.linalg.svd(O, full_matrices=False)
    V = Vt.conj().T
    M_mix = V @ U.conj().T  # (N_ALL, N_SUB)

    transported = []
    for m in range(N_SUB):
        state = np.zeros_like(child_all_bands[0])
        for b in range(N_ALL):
            state += M_mix[b, m] * child_all_bands[b]
        transported.append(state)

    return transported, sigma.min()


def bfs_transport(bf, eps, seed_ix=0, seed_iy=0, seed_bands=None):
    Nr = bf.shape[0]
    if seed_bands is None:
        seed_bands = SUBSPACE

    transported = np.zeros((Nr, Nr, N_SUB, Nx, Nx, 3), dtype=np.complex64)
    quality = np.zeros((Nr, Nr))
    visited = np.zeros((Nr, Nr), dtype=bool)

    for m, b in enumerate(seed_bands):
        transported[seed_ix, seed_iy, m] = bf[seed_ix, seed_iy, b]
    quality[seed_ix, seed_iy] = 1.0
    visited[seed_ix, seed_iy] = True

    queue = deque()
    for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nix = (seed_ix + dix) % Nr
        niy = (seed_iy + diy) % Nr
        queue.append((nix, niy, seed_ix, seed_iy))

    while queue:
        ix, iy, pix, piy = queue.popleft()
        if visited[ix, iy]:
            continue
        visited[ix, iy] = True

        parent_states = [transported[pix, piy, m] for m in range(N_SUB)]
        child_all = [bf[ix, iy, b] for b in range(N_ALL)]

        new_states, q = parallel_transport_step(parent_states, child_all, eps[pix, piy])
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
# C4 symmetrization of transported subspace
# ══════════════════════════════════════════════════════════════════════════

def symmetrize_subspace(transported, Nr):
    """
    At each point R, combine transported states from R and its 3 C4 images:
      S₁ = {u_m(r; R)}                     (direct)
      S₂ = {C4 · u_m(C4⁻¹r; C4⁻¹R)}       (from C4⁻¹R, rotated forward)
      S₃ = {C4² · u_m(C4⁻²r; C4⁻²R)}      (from C4⁻²R, rotated forward 2x)
      S₄ = {C4³ · u_m(C4⁻³r; C4⁻³R)}      (from C4⁻³R, rotated forward 3x)
    
    Stack 20 states, SVD → top 5 span the C4-symmetrized subspace.
    """
    symm = np.zeros_like(transported)
    gap_ratio = np.zeros((Nr, Nr))  # σ₅/σ₆ — how well-defined is the 5D subspace?

    for ix in range(Nr):
        for iy in range(Nr):
            # Collect 4x5 = 20 states at point (ix,iy)
            all_states = []

            # Direct: states at (ix,iy)
            for m in range(N_SUB):
                all_states.append(transported[ix, iy, m].flatten())

            # From C4⁻¹R: pre-image under one C4 step
            for power in range(1, 4):
                # Pre-image: apply C4^{-power} to (ix,iy)
                pix, piy = ix, iy
                for _ in range(power):
                    pix, piy = c4_inv_registry(pix, piy, Nr)

                # Get transported states at pre-image, then C4^power rotate
                for m in range(N_SUB):
                    u = transported[pix, piy, m].copy()
                    u_rot = rotate_field_c4_n(u, power)
                    all_states.append(u_rot.flatten())

            # Stack into matrix: (dim, 20) where dim = Nx*Nx*3
            A = np.array(all_states).T  # (dim, 20)

            # SVD
            U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
            # Take top N_SUB left singular vectors
            for m in range(N_SUB):
                symm[ix, iy, m] = U[:, m].reshape(Nx, Nx, 3)

            # Gap ratio: σ₅/σ₆ (if ≫ 1, the 5D subspace is well-separated)
            if len(sigma) > N_SUB:
                gap_ratio[ix, iy] = sigma[N_SUB - 1] / (sigma[N_SUB] + 1e-30)
            else:
                gap_ratio[ix, iy] = sigma[N_SUB - 1] / 1e-30

    return symm, gap_ratio


# ══════════════════════════════════════════════════════════════════════════
# Fundamental domain approach
# ══════════════════════════════════════════════════════════════════════════

def fundamental_domain_transport(bf, eps, seed_ix=0, seed_iy=0, seed_bands=None):
    """
    Transport only within the C4 fundamental domain.
    For each C4 orbit, pick one representative (first visited by BFS).
    Extend to rest of grid by C4 rotation.
    """
    Nr = bf.shape[0]
    if seed_bands is None:
        seed_bands = SUBSPACE

    # Step 1: Identify C4 orbits and representatives
    rep_of = {}  # (ix,iy) → representative (rx,ry)
    power_of = {}  # (ix,iy) → which C4^n maps rep to this point
    orbits = []

    assigned = set()
    for ix in range(Nr):
        for iy in range(Nr):
            if (ix, iy) in assigned:
                continue
            orbit = c4_orbit(ix, iy, Nr)
            # Remove duplicates (fixed points have orbit size < 4)
            seen = set()
            unique_orbit = []
            for p in orbit:
                if p not in seen:
                    seen.add(p)
                    unique_orbit.append(p)

            rep = unique_orbit[0]  # (ix,iy) itself is the representative
            orbits.append(unique_orbit)
            for k, p in enumerate(unique_orbit):
                rep_of[p] = rep
                power_of[p] = k
                assigned.add(p)

    n_reps = len(orbits)
    print(f"    {n_reps} C4 orbits ({Nr*Nr} points)")

    # Step 2: BFS transport only to representatives
    transported_reps = {}  # (rx,ry) → list of N_SUB states
    quality_reps = {}

    for m, b in enumerate(seed_bands):
        if (seed_ix, seed_iy) not in transported_reps:
            transported_reps[(seed_ix, seed_iy)] = []
        transported_reps[(seed_ix, seed_iy)].append(bf[seed_ix, seed_iy, b].copy())
    quality_reps[(seed_ix, seed_iy)] = 1.0

    queue = deque()
    visited_reps = {(seed_ix, seed_iy)}

    for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nix = (seed_ix + dix) % Nr
        niy = (seed_iy + diy) % Nr
        nrep = rep_of[(nix, niy)]
        if nrep not in visited_reps:
            queue.append((nrep, seed_ix, seed_iy))

    while queue:
        (rix, riy), pix, piy = queue.popleft()
        if (rix, riy) in visited_reps:
            continue
        visited_reps.add((rix, riy))

        parent_states = transported_reps[(pix, piy)]
        child_all = [bf[rix, riy, b] for b in range(N_ALL)]

        new_states, q = parallel_transport_step(parent_states, child_all, eps[pix, piy])
        transported_reps[(rix, riy)] = [s.copy() for s in new_states]
        quality_reps[(rix, riy)] = q

        # Enqueue neighbors' representatives
        for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nix = (rix + dix) % Nr
            niy = (riy + diy) % Nr
            nrep = rep_of[(nix, niy)]
            if nrep not in visited_reps:
                queue.append((nrep, rix, riy))

    print(f"    Transported {len(visited_reps)} representatives")

    # Step 3: Extend to full grid by C4 rotation
    transported = np.zeros((Nr, Nr, N_SUB, Nx, Nx, 3), dtype=np.complex64)
    quality = np.zeros((Nr, Nr))

    for ix in range(Nr):
        for iy in range(Nr):
            rep = rep_of[(ix, iy)]
            power = power_of[(ix, iy)]

            if rep in transported_reps:
                rep_states = transported_reps[rep]
                for m in range(N_SUB):
                    u = rotate_field_c4_n(rep_states[m], power)
                    transported[ix, iy, m] = u.astype(np.complex64)
                quality[ix, iy] = quality_reps.get(rep, 0.0)
            else:
                quality[ix, iy] = 0.0

    return transported, quality


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("S3c: C4-symmetrized parallel transport")
    print("="*70)

    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]
        eps = f['epsilon'][:]
    Nr = bf.shape[0]

    # ── Method 1: BFS transport + C4 symmetrization ──
    print(f"\n{'='*70}")
    print("Method 1: BFS transport + post-hoc C4 symmetrization")
    print("="*70)

    print("  BFS transport...")
    transported, quality = bfs_transport(bf, eps)
    print(f"  Transport quality: min={quality.min():.4f}, mean={quality.mean():.4f}")

    print("  C4 symmetrization...")
    symm, gap_ratio = symmetrize_subspace(transported, Nr)

    print(f"  SVD gap ratio σ₅/σ₆: min={gap_ratio.min():.2f}, "
          f"mean={gap_ratio.mean():.2f}, max={gap_ratio.max():.2f}")
    print(f"  Fraction with σ₅/σ₆ > 5:  {np.sum(gap_ratio > 5)/(Nr*Nr):.1%}")
    print(f"  Fraction with σ₅/σ₆ > 2:  {np.sum(gap_ratio > 2)/(Nr*Nr):.1%}")
    print(f"  Fraction with σ₅/σ₆ > 1.5: {np.sum(gap_ratio > 1.5)/(Nr*Nr):.1%}")

    # C4 closure test
    print("  C4 closure test...")
    c4_orig = np.zeros((Nr, Nr))
    c4_symm = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            _, c4_orig[ix, iy] = c4_closure_metric([bf[ix, iy, b] for b in SUBSPACE])
            _, c4_symm[ix, iy] = c4_closure_metric([symm[ix, iy, m] for m in range(N_SUB)])

    pct_symm = np.sum(c4_symm > 0.9) / (Nr*Nr)
    print(f"\n  ORIGINAL: min_sv>0.9: {np.sum(c4_orig>0.9)/(Nr*Nr):.1%}, mean={c4_orig.mean():.4f}")
    print(f"  SYMMETRIZED: min_sv>0.9: {pct_symm:.1%}, mean={c4_symm.mean():.4f}")

    # ── Method 2: Fundamental domain transport ──
    print(f"\n{'='*70}")
    print("Method 2: Fundamental domain transport (C4-extend)")
    print("="*70)

    transported_fd, quality_fd = fundamental_domain_transport(bf, eps)

    # C4 closure test
    print("  C4 closure test...")
    c4_fd = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            _, c4_fd[ix, iy] = c4_closure_metric(
                [transported_fd[ix, iy, m] for m in range(N_SUB)])

    pct_fd = np.sum(c4_fd > 0.9) / (Nr*Nr)
    print(f"  FD-TRANSPORTED: min_sv>0.9: {pct_fd:.1%}, mean={c4_fd.mean():.4f}")
    print(f"    min_sv>0.5: {np.sum(c4_fd>0.5)/(Nr*Nr):.1%}")
    print(f"    mean(min_sv): {c4_fd.mean():.4f}")
    print(f"    min(min_sv):  {c4_fd.min():.4f}")

    # ── Key points detail ──
    print(f"\n{'='*70}")
    print("Key registry points comparison")
    print("="*70)

    key_points = [
        ("δ=(0,0)",       0,  0),
        ("δ=(0.25,0.25)",16, 16),
        ("δ=(0.5,0)",    32,  0),
        ("δ=(0.5,0.5)",  32, 32),
    ]

    for label, ix, iy in key_points:
        print(f"  {label}:")
        print(f"    original={c4_orig[ix,iy]:.4f}, symm={c4_symm[ix,iy]:.4f}, "
              f"fd={c4_fd[ix,iy]:.4f}, gap_ratio={gap_ratio[ix,iy]:.2f}")

    # ── FD continuity check ──
    print(f"\n{'='*70}")
    print("Continuity check: FD-transported overlap between adjacent points")
    print("="*70)

    adj_ovs = []
    for ix in range(Nr):
        for iy in range(Nr):
            nix = (ix + 1) % Nr
            for m in range(N_SUB):
                u1 = transported_fd[ix, iy, m]
                u2 = transported_fd[nix, iy, m]
                n1 = np.sqrt(np.sum(np.abs(u1)**2) / (Nx*Nx))
                n2 = np.sqrt(np.sum(np.abs(u2)**2) / (Nx*Nx))
                ov = np.abs(flat_inner(u1, u2)) / (n1 * n2 + 1e-30)
                adj_ovs.append(ov)

    adj_ovs = np.array(adj_ovs)
    print(f"  min={adj_ovs.min():.4f}, mean={adj_ovs.mean():.4f}, "
          f"median={np.median(adj_ovs):.4f}")
    print(f"  >0.99: {np.sum(adj_ovs>0.99)/len(adj_ovs):.1%}")
    print(f"  >0.95: {np.sum(adj_ovs>0.95)/len(adj_ovs):.1%}")
    print(f"  >0.90: {np.sum(adj_ovs>0.90)/len(adj_ovs):.1%}")
    print(f"  <0.50: {np.sum(adj_ovs<0.50)/len(adj_ovs):.1%}")

    # ── Verdict ──
    best_method = "symmetrized" if pct_symm >= pct_fd else "fundamental domain"
    best_pct = max(pct_symm, pct_fd)
    best_c4 = c4_symm if best_method == "symmetrized" else c4_fd

    print(f"\n{'='*70}")
    print(f"Best method: {best_method} ({best_pct:.1%} C4-closed)")
    if best_pct > 0.9:
        print(f"✓ C4-symmetric Wannier-like subspace construction WORKS")
    elif best_pct > 0.5:
        print(f"⚠ Partial success — may need refinement")
    else:
        print(f"❌ Both methods fail — deeper topological obstruction")
        print(f"  The 5-band subspace may be genuinely C4-incompatible")
    print("="*70)

    # ── Plots ──
    print(f"\nGenerating plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    for ax_idx, (name, data) in enumerate([
        ("Original [5-9]", c4_orig),
        ("BFS + symmetrize", c4_symm),
        ("Fundamental domain", c4_fd),
    ]):
        im = axes[0, ax_idx].pcolormesh(sr, sr, data.T, cmap='RdYlGn',
                                         shading='auto', vmin=0, vmax=1)
        axes[0, ax_idx].set_title(f'C4 min_sv — {name}')
        axes[0, ax_idx].set_aspect('equal')
        plt.colorbar(im, ax=axes[0, ax_idx])

    im3 = axes[1, 0].pcolormesh(sr, sr, np.log10(gap_ratio.T + 0.01),
                                 cmap='RdYlGn', shading='auto', vmin=-1, vmax=2)
    axes[1, 0].set_title('log₁₀(σ₅/σ₆) — symmetrization quality')
    axes[1, 0].set_aspect('equal'); plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(sr, sr, quality.T, cmap='RdYlGn',
                                 shading='auto', vmin=0.5, vmax=1.0)
    axes[1, 1].set_title('BFS transport quality')
    axes[1, 1].set_aspect('equal'); plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(sr, sr, quality_fd.T, cmap='RdYlGn',
                                 shading='auto', vmin=0.5, vmax=1.0)
    axes[1, 2].set_title('FD transport quality')
    axes[1, 2].set_aspect('equal'); plt.colorbar(im5, ax=axes[1, 2])

    fig.suptitle('S3c: C4-symmetrized subspace construction', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3c_symmetrized.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S3c_symmetrized.png")


if __name__ == '__main__':
    main()
