#!/usr/bin/env python3
"""
S3b: Local parallel transport — BFS subspace continuation
============================================================
Global overlap tracking fails (S3 Outcome C) because anti-crossings make
intermediate states 50/50 superpositions of two band characters.

But LOCAL continuation should work: adjacent δ-points (Δδ=1/64) have
near-unity overlap. We BFS-walk from δ=(0,0) and at each step:
  1. Take the parent's 5-state subspace
  2. Compute 5×18 ε-weighted overlap with ALL bands at the child point
  3. SVD to find the best 5-band span of the parent subspace
  4. Procrustes-rotate to align smoothly

This is non-Abelian parallel transport in registry space — the core
building block of Wannier90, adapted from k-space to R-space.

If this succeeds (C4 closure > 90%), we have a working Wannier-like
subspace construction that can be integrated into the pipeline.
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
N_SUB = len(SUBSPACE)
N_ALL = 18
Nx = 32


# ══════════════════════════════════════════════════════════════════════════
# Core: ε-weighted overlap and parallel transport
# ══════════════════════════════════════════════════════════════════════════

def eps_inner(u1, u2, eps):
    """⟨u1|ε|u2⟩ / (Nx²)."""
    return np.sum(eps[:, :, None] * np.conj(u1) * u2) / (Nx * Nx)


def overlap_matrix_eps(states_parent, states_child, eps_parent, eps_child):
    """Compute N_parent × N_child overlap matrix using average ε.
    O_mn = ⟨parent_m | ε_avg | child_n⟩
    We use the parent's ε for the inner product (subspace-defining).
    """
    Np = len(states_parent)
    Nc = len(states_child)
    O = np.zeros((Np, Nc), dtype=complex)
    for m in range(Np):
        for n in range(Nc):
            O[m, n] = eps_inner(states_parent[m], states_child[n], eps_parent)
    return O


def parallel_transport_step(subspace_parent, all_bands_child, eps_parent, eps_child):
    """
    Given N_SUB parent states and N_ALL child bands, find the optimal
    N_SUB-dimensional subspace at the child point that best continues
    the parent subspace, and align it via Procrustes.

    Returns: (transported_states, quality_metrics)
      transported_states: N_SUB states at child, Procrustes-aligned
      quality_metrics: dict with overlap info
    """
    # Step 1: Compute N_SUB × N_ALL overlap matrix
    O = overlap_matrix_eps(subspace_parent, all_bands_child, eps_parent, eps_child)
    # O[m, b] = ⟨parent_m | ε | child_b⟩

    # Step 2: SVD to find best N_SUB-dimensional subspace
    # The columns of O with the largest singular values span the
    # projection of the parent subspace onto the child bands.
    # But we want to SELECT N_SUB bands from N_ALL, not mix them.
    #
    # Actually, for parallel transport, we SHOULD mix: the transported
    # states are linear combinations of child bands that best approximate
    # the parent subspace. This is the Wannier approach.
    #
    # Method: Use the N_SUB × N_ALL overlap matrix O.
    # The best N_SUB-dim subspace at child is spanned by O† O's
    # top N_SUB eigenvectors, but it's cleaner to use SVD directly:
    #
    # O = U Σ V†  (U: N_SUB×N_SUB, Σ: N_SUB×N_SUB, V: N_ALL×N_SUB)
    # The transported states are: |new_m⟩ = Σ_b (V Q)_{b,m} |child_b⟩
    # where Q = U (the Procrustes rotation to align with parent ordering)
    #
    # Simpler: let W = V (the first N_SUB right singular vectors).
    # Then W†W = I, and {Σ_b W_{b,m} |child_b⟩} spans the best subspace.
    # Apply Procrustes: rotate by (U V†)† to align with parent ordering.

    U, sigma, Vt = np.linalg.svd(O, full_matrices=False)
    # U: (N_SUB, N_SUB), sigma: (N_SUB,), Vt: (N_SUB, N_ALL)
    # V = Vt.T: (N_ALL, N_SUB)

    # Quality: singular values should all be close to 1
    # (meaning parent subspace fully embedded in child bands)
    min_sv = sigma.min()
    mean_sv = sigma.mean()

    # Procrustes rotation: Q = U V† maps child back to parent ordering
    # But we need the transported states, not just the rotation.
    # The optimal transported subspace basis:
    #   |new_m⟩ = Σ_b (V · Q)_{b,m} |child_b⟩
    # where Q aligns the ordering. For Procrustes: Q = U† (since V†·O = Σ U†)
    # Actually: |new_m⟩ = Σ_b Σ_n (V_{b,n} (U†)_{n,m}) |child_b⟩
    # This is: new = child_allbands @ V @ U†.conj()

    # Mixing matrix: M = V @ U†.conj()  (N_ALL × N_SUB)
    # Actually Vt is (N_SUB, N_ALL), so V = Vt.conj().T is (N_ALL, N_SUB)
    V = Vt.conj().T  # (N_ALL, N_SUB)
    M_mix = V @ U.conj().T  # (N_ALL, N_SUB) — mixing coefficients

    # Construct transported states
    # new_states[m] = Σ_b M_mix[b, m] * child_bands[b]
    transported = []
    for m in range(N_SUB):
        state = np.zeros_like(all_bands_child[0])
        for b in range(N_ALL):
            state += M_mix[b, m] * all_bands_child[b]
        transported.append(state)

    metrics = {
        'min_sv': min_sv,
        'mean_sv': mean_sv,
        'sigma': sigma,
    }

    return transported, metrics


def rotate_90_origin(field):
    """C4 rotation (90° CCW) about origin on periodic grid."""
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated


def c4_closure_metric(states, eps=None):
    """Compute C4 closure: (|det|, min_sv)."""
    Nb = len(states)
    rotated = [rotate_90_origin(u) for u in states]

    if eps is not None:
        norms = [np.sqrt(np.abs(eps_inner(u, u, eps))) for u in states]
    else:
        norms = [np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx)) for u in states]

    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            if eps is not None:
                M[m, n] = eps_inner(states[m], rotated[n], eps)
            else:
                M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx*Nx)

    M_norm = M / np.outer(norms, norms)
    sv = np.linalg.svd(M_norm, compute_uv=False)
    return np.abs(np.linalg.det(M_norm)), sv.min()


# ══════════════════════════════════════════════════════════════════════════
# BFS parallel transport across the 2D registry grid
# ══════════════════════════════════════════════════════════════════════════

def bfs_parallel_transport(bf, eps, seed_ix=0, seed_iy=0, seed_bands=None):
    """
    BFS walk from seed, transporting a N_SUB-dimensional subspace.
    At each step, the parent's subspace is projected onto the child's
    full band space and Procrustes-aligned.

    Returns:
      transported_fields: (Nr, Nr, N_SUB, Nx, Ny, 3) complex
      quality_map: (Nr, Nr) float — min singular value at each transport step
    """
    Nr = bf.shape[0]
    if seed_bands is None:
        seed_bands = SUBSPACE

    # Storage: transported subspace at each point
    transported = np.zeros((Nr, Nr, N_SUB, Nx, Nx, 3), dtype=np.complex64)
    quality_map = np.zeros((Nr, Nr))
    visited = np.zeros((Nr, Nr), dtype=bool)

    # Initialize seed
    for m, b in enumerate(seed_bands):
        transported[seed_ix, seed_iy, m] = bf[seed_ix, seed_iy, b]
    quality_map[seed_ix, seed_iy] = 1.0
    visited[seed_ix, seed_iy] = True

    # BFS queue
    queue = deque()
    # Add neighbors of seed
    for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
        nix = (seed_ix + dix) % Nr
        niy = (seed_iy + diy) % Nr
        queue.append((nix, niy, seed_ix, seed_iy))

    n_processed = 1
    min_quality = 1.0

    while queue:
        ix, iy, pix, piy = queue.popleft()

        if visited[ix, iy]:
            continue
        visited[ix, iy] = True
        n_processed += 1

        if n_processed % 500 == 0:
            print(f"    BFS: {n_processed}/{Nr*Nr} processed, "
                  f"min_quality so far: {min_quality:.4f}")

        # Parent's transported subspace
        parent_states = [transported[pix, piy, m] for m in range(N_SUB)]
        parent_eps = eps[pix, piy]

        # Child's full band space
        child_all = [bf[ix, iy, b] for b in range(N_ALL)]
        child_eps = eps[ix, iy]

        # Parallel transport step
        new_states, metrics = parallel_transport_step(
            parent_states, child_all, parent_eps, child_eps
        )

        # Store
        for m in range(N_SUB):
            transported[ix, iy, m] = new_states[m].astype(np.complex64)
        quality_map[ix, iy] = metrics['min_sv']
        min_quality = min(min_quality, metrics['min_sv'])

        # Enqueue neighbors
        for dix, diy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nix = (ix + dix) % Nr
            niy = (iy + diy) % Nr
            if not visited[nix, niy]:
                queue.append((nix, niy, ix, iy))

    assert n_processed == Nr * Nr, f"BFS incomplete: {n_processed}/{Nr*Nr}"
    return transported, quality_map


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("S3b: Local parallel transport — BFS subspace continuation")
    print("="*70)

    # Load data
    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]  # (64,64,18,32,32,3)
        eps = f['epsilon'][:]       # (64,64,32,32)
    Nr = bf.shape[0]
    print(f"Data: {bf.shape}, Nr={Nr}")

    # ── Step 1: BFS parallel transport from δ=(0,0) ──
    print(f"\n{'='*70}")
    print(f"Step 1: BFS parallel transport from δ=(0,0)")
    print(f"{'='*70}")
    transported, quality_map = bfs_parallel_transport(bf, eps,
                                                       seed_ix=0, seed_iy=0,
                                                       seed_bands=SUBSPACE)

    print(f"\n  Transport quality (min singular value per step):")
    print(f"    min={quality_map.min():.6f}, mean={quality_map.mean():.6f}, "
          f"max={quality_map.max():.6f}")
    print(f"    fraction > 0.9: {np.sum(quality_map > 0.9)/(Nr*Nr):.1%}")
    print(f"    fraction > 0.5: {np.sum(quality_map > 0.5)/(Nr*Nr):.1%}")
    print(f"    fraction < 0.3: {np.sum(quality_map < 0.3)/(Nr*Nr):.1%}")

    # ── Step 2: C4 closure test on transported subspace ──
    print(f"\n{'='*70}")
    print(f"Step 2: C4 closure test on transported subspace")
    print(f"{'='*70}")

    c4_minsv_orig = np.zeros((Nr, Nr))
    c4_minsv_trans = np.zeros((Nr, Nr))

    for ix in range(Nr):
        if ix % 8 == 0:
            print(f"  C4 scan row {ix}/{Nr}...")
        for iy in range(Nr):
            # Original
            states_orig = [bf[ix, iy, b] for b in SUBSPACE]
            _, msv_o = c4_closure_metric(states_orig)
            c4_minsv_orig[ix, iy] = msv_o

            # Transported
            states_trans = [transported[ix, iy, m] for m in range(N_SUB)]
            _, msv_t = c4_closure_metric(states_trans)
            c4_minsv_trans[ix, iy] = msv_t

    pct_orig = np.sum(c4_minsv_orig > 0.9) / (Nr * Nr)
    pct_trans = np.sum(c4_minsv_trans > 0.9) / (Nr * Nr)

    print(f"\n  ORIGINAL [5-9]:")
    print(f"    min_sv > 0.9: {pct_orig:.1%}")
    print(f"    min_sv > 0.5: {np.sum(c4_minsv_orig > 0.5)/(Nr*Nr):.1%}")
    print(f"    mean(min_sv): {c4_minsv_orig.mean():.4f}")

    print(f"\n  TRANSPORTED (BFS parallel transport):")
    print(f"    min_sv > 0.9: {pct_trans:.1%}")
    print(f"    min_sv > 0.5: {np.sum(c4_minsv_trans > 0.5)/(Nr*Nr):.1%}")
    print(f"    mean(min_sv): {c4_minsv_trans.mean():.4f}")
    print(f"    min(min_sv):  {c4_minsv_trans.min():.4f}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    if pct_trans > 0.9:
        print(f"VERDICT: ✓ Parallel transport works! ({pct_trans:.1%} C4-closed)")
        print(f"  → The Wannier-like subspace is well-defined")
    elif pct_trans > 0.5:
        print(f"VERDICT: ⚠ Partial success ({pct_trans:.1%} C4-closed)")
        print(f"  → Transport mostly works but some regions fail")
    else:
        print(f"VERDICT: ❌ Parallel transport also fails ({pct_trans:.1%} C4-closed)")
        print(f"  → The 5-band subspace may be topologically obstructed")
    print(f"{'='*70}")

    # ── Step 3: ε-orthonormality of transported states ──
    print(f"\n{'='*70}")
    print(f"Step 3: ε-orthonormality of transported states")
    print(f"{'='*70}")

    max_offdiag = np.zeros((Nr, Nr))
    max_diag_err = np.zeros((Nr, Nr))

    for ix in range(Nr):
        for iy in range(Nr):
            e = eps[ix, iy]
            G = np.zeros((N_SUB, N_SUB), dtype=complex)
            for m in range(N_SUB):
                for n in range(N_SUB):
                    G[m, n] = eps_inner(transported[ix, iy, m],
                                        transported[ix, iy, n], e)
            max_offdiag[ix, iy] = np.abs(G - np.diag(np.diag(G))).max()
            max_diag_err[ix, iy] = np.max(np.abs(np.diag(G).real - 1.0))

    print(f"  Max |off-diag|: min={max_offdiag.min():.4e}, "
          f"mean={max_offdiag.mean():.4e}, max={max_offdiag.max():.4e}")
    print(f"  Max |diag-1|:   min={max_diag_err.min():.4e}, "
          f"mean={max_diag_err.mean():.4e}, max={max_diag_err.max():.4e}")

    # ── Step 4: Check at key points ──
    print(f"\n{'='*70}")
    print(f"Step 4: Key registry points")
    print(f"{'='*70}")

    key_points = [
        ("δ=(0,0)",       0,  0),
        ("δ=(0.25,0)",   16,  0),
        ("δ=(0,0.25)",    0, 16),
        ("δ=(0.25,0.25)",16, 16),
        ("δ=(0.5,0)",    32,  0),
        ("δ=(0,0.5)",     0, 32),
        ("δ=(0.5,0.5)",  32, 32),
    ]

    for label, ix, iy in key_points:
        msv_o = c4_minsv_orig[ix, iy]
        msv_t = c4_minsv_trans[ix, iy]
        q = quality_map[ix, iy]
        od = max_offdiag[ix, iy]

        # Check which original bands the transported states most overlap with
        band_desc = []
        for m in range(N_SUB):
            best_ov = 0
            best_b = -1
            for b in range(N_ALL):
                ov = np.abs(np.sum(np.conj(transported[ix, iy, m]) *
                                   bf[ix, iy, b]) / (Nx*Nx))
                norm_t = np.sqrt(np.sum(np.abs(transported[ix, iy, m])**2) / (Nx*Nx))
                norm_b = np.sqrt(np.sum(np.abs(bf[ix, iy, b])**2) / (Nx*Nx))
                fid = ov / (norm_t * norm_b + 1e-30)
                if fid > best_ov:
                    best_ov = fid
                    best_b = b
            band_desc.append(f"{best_b}({best_ov:.2f})")

        print(f"  {label:20s}: C4 {msv_o:.4f}→{msv_t:.4f}, "
              f"q={q:.4f}, |off-diag|={od:.2e}")
        print(f"    {'':20s}  closest bands: {', '.join(band_desc)}")

    # ── Step 5: Continuity check — overlap between adjacent points ──
    print(f"\n{'='*70}")
    print(f"Step 5: Continuity (overlap between adjacent transported states)")
    print(f"{'='*70}")

    adj_overlaps = []
    for ix in range(Nr):
        for iy in range(Nr):
            nix = (ix + 1) % Nr
            for m in range(N_SUB):
                ov = np.abs(np.sum(np.conj(transported[ix, iy, m]) *
                                   transported[nix, iy, m]) / (Nx*Nx))
                norm1 = np.sqrt(np.sum(np.abs(transported[ix, iy, m])**2) / (Nx*Nx))
                norm2 = np.sqrt(np.sum(np.abs(transported[nix, iy, m])**2) / (Nx*Nx))
                adj_overlaps.append(ov / (norm1 * norm2 + 1e-30))

    adj_overlaps = np.array(adj_overlaps)
    print(f"  Adjacent-point overlap (s1 direction):")
    print(f"    min={adj_overlaps.min():.6f}, mean={adj_overlaps.mean():.6f}, "
          f"median={np.median(adj_overlaps):.6f}")
    print(f"    fraction > 0.99: {np.sum(adj_overlaps > 0.99)/len(adj_overlaps):.1%}")
    print(f"    fraction > 0.95: {np.sum(adj_overlaps > 0.95)/len(adj_overlaps):.1%}")
    print(f"    fraction > 0.90: {np.sum(adj_overlaps > 0.90)/len(adj_overlaps):.1%}")

    # ── Step 6: Plots ──
    print(f"\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    im00 = axes[0, 0].pcolormesh(sr, sr, c4_minsv_orig.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('C4 min_sv — ORIGINAL [5-9]')
    axes[0, 0].set_aspect('equal'); plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].pcolormesh(sr, sr, c4_minsv_trans.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('C4 min_sv — TRANSPORTED')
    axes[0, 1].set_aspect('equal'); plt.colorbar(im01, ax=axes[0, 1])

    improvement = c4_minsv_trans - c4_minsv_orig
    im02 = axes[0, 2].pcolormesh(sr, sr, improvement.T, cmap='RdBu',
                                  shading='auto', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('Improvement (trans - orig)')
    axes[0, 2].set_aspect('equal'); plt.colorbar(im02, ax=axes[0, 2])

    im10 = axes[1, 0].pcolormesh(sr, sr, quality_map.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Transport quality (min σ per step)')
    axes[1, 0].set_aspect('equal'); plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].pcolormesh(sr, sr, np.log10(max_offdiag.T + 1e-15),
                                  cmap='hot_r', shading='auto')
    axes[1, 1].set_title('log₁₀(max |off-diag| of ε-Gram)')
    axes[1, 1].set_aspect('equal'); plt.colorbar(im11, ax=axes[1, 1])

    im12 = axes[1, 2].pcolormesh(sr, sr, max_diag_err.T, cmap='hot_r',
                                  shading='auto')
    axes[1, 2].set_title('max |diag(ε-Gram) - 1|')
    axes[1, 2].set_aspect('equal'); plt.colorbar(im12, ax=axes[1, 2])

    fig.suptitle('S3b: BFS parallel transport — C4 closure test',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3b_parallel_transport.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S3b_parallel_transport.png")

    # ── Step 7: If transport works, check if transported states
    #    give better M_inv (less divergence) ──
    if pct_trans > 0.5:
        print(f"\n{'='*70}")
        print(f"Step 7: Would M_inv improve? (checking field smoothness)")
        print(f"{'='*70}")

        # Compute finite-difference "derivative norm" as proxy for field smoothness
        deriv_rms_orig = np.zeros((Nr, Nr))
        deriv_rms_trans = np.zeros((Nr, Nr))
        for ix in range(Nr):
            for iy in range(Nr):
                nix = (ix + 1) % Nr
                for m in range(N_SUB):
                    diff_o = bf[nix, iy, SUBSPACE[m]] - bf[ix, iy, SUBSPACE[m]]
                    diff_t = transported[nix, iy, m] - transported[ix, iy, m]
                    deriv_rms_orig[ix, iy] += np.sum(np.abs(diff_o)**2)
                    deriv_rms_trans[ix, iy] += np.sum(np.abs(diff_t)**2)

        deriv_rms_orig = np.sqrt(deriv_rms_orig / (N_SUB * Nx * Nx))
        deriv_rms_trans = np.sqrt(deriv_rms_trans / (N_SUB * Nx * Nx))

        print(f"  ∂u/∂s₁ RMS (original):    mean={deriv_rms_orig.mean():.4f}, "
              f"max={deriv_rms_orig.max():.4f}")
        print(f"  ∂u/∂s₁ RMS (transported): mean={deriv_rms_trans.mean():.4f}, "
              f"max={deriv_rms_trans.max():.4f}")
        ratio = deriv_rms_orig.mean() / (deriv_rms_trans.mean() + 1e-30)
        print(f"  Smoothness ratio (orig/trans): {ratio:.2f}×")


if __name__ == '__main__':
    main()
