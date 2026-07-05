#!/usr/bin/env python3
"""
S3: Overlap-based band reordering experiment
==============================================
The decisive test: can we fix the subspace by tracking physical states
via ε-weighted overlap with reference states at δ=(0,0)?

Algorithm:
  1. Load all 18 bands' Bloch fields from Phase 1 HDF5
  2. Define reference = 5 subspace bands at δ=(0,0) (C4-verified correct)
  3. At every registry point: compute ε-weighted overlap matrix (5×18),
     solve optimal assignment (Hungarian), pick the 5 best-matching bands
  4. Re-run C4 closure test on the reordered subspace
  5. Compare with the original index-based subspace

Outcome A (C4 closure >90% of R): band tracking works → fix pipeline
Outcome B (50-90%):               partial fix needed
Outcome C (<50%):                 pivot to Wannier
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment

OUTDIR = Path(__file__).resolve().parent / "plots"
OUTDIR.mkdir(exist_ok=True)

BASE  = Path(__file__).resolve().parent.parent
CAND  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
P1    = CAND / "phase1_multiband_data.h5"

SUBSPACE = [5, 6, 7, 8, 9]
N_SUB = len(SUBSPACE)
N_ALL = 18
Nx = 32  # MPB resolution per cell


# ══════════════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════════════

def eps_overlap(u1, u2, eps):
    """ε-weighted overlap ⟨u1|ε|u2⟩ / (Nx*Ny).
    u1, u2: (Nx,Ny,3), eps: (Nx,Ny)."""
    return np.sum(eps[:, :, None] * np.conj(u1) * u2) / (Nx * Nx)


def eps_norm(u, eps):
    """√⟨u|ε|u⟩."""
    return np.sqrt(np.abs(eps_overlap(u, u, eps)))


def rotate_90_origin(field):
    """C4 rotation (90° CCW) about origin on periodic grid.
    field: (Nx,Ny,3) → (Nx,Ny,3)."""
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
    """Compute C4 closure metrics for a set of states.
    Returns (|det(M)|, min_sv(M), singular_values)."""
    Nb = len(states)
    rotated = [rotate_90_origin(u) for u in states]

    if eps is not None:
        norms = [eps_norm(u, eps) for u in states]
    else:
        norms = [np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx)) for u in states]

    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            if eps is not None:
                M[m, n] = eps_overlap(states[m], rotated[n], eps)
            else:
                M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx*Nx)

    M_norm = M / np.outer(norms, norms)
    sv = np.linalg.svd(M_norm, compute_uv=False)
    return np.abs(np.linalg.det(M_norm)), sv.min(), sv


# ══════════════════════════════════════════════════════════════════════════
# Main algorithm
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("S3: Overlap-based band reordering experiment")
    print("="*70)

    # ── Load data ──
    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]   # (64,64,18,32,32,3)
        eps = f['epsilon'][:]        # (64,64,32,32)
        omega_all = f['omega'][:]    # (128,128,5) — only subspace freqs on moiré grid

    Nr = bf.shape[0]  # 64
    print(f"Data: bloch_fields {bf.shape}, epsilon {eps.shape}, Nr={Nr}")

    # ── Define reference states at δ=(0,0) ──
    ref_ix, ref_iy = 0, 0
    ref_states = [bf[ref_ix, ref_iy, b] for b in SUBSPACE]  # 5 states, each (32,32,3)
    ref_eps    = eps[ref_ix, ref_iy]                          # (32,32)
    ref_norms  = [eps_norm(u, ref_eps) for u in ref_states]

    print(f"\nReference point: δ=({ref_ix},{ref_iy})")
    print(f"Reference band norms (ε-weighted): {[f'{n:.4f}' for n in ref_norms]}")

    # Verify reference C4 closure
    det0, minsv0, _ = c4_closure_metric(ref_states, ref_eps)
    print(f"Reference C4 closure: |det|={det0:.6f}, min_sv={minsv0:.6f}")

    # ── Step 1: Compute overlap-based reordering at every R ──
    print(f"\nComputing overlap-based reordering across {Nr}×{Nr} grid...")

    reorder_map   = np.zeros((Nr, Nr, N_SUB), dtype=int)   # new band indices
    match_quality = np.zeros((Nr, Nr))                       # mean overlap
    min_overlap   = np.zeros((Nr, Nr))                       # worst single-band overlap

    for ix in range(Nr):
        if ix % 8 == 0:
            print(f"  row {ix}/{Nr}...")
        for iy in range(Nr):
            e = eps[ix, iy]  # (32,32)

            # Compute ε-weighted overlap matrix: O[m, b] = |⟨ref_m|ε_ref|u_b(δ)⟩|
            # Note: we use the reference ε for consistency (ε varies with δ,
            # but the reference defines the subspace identity)
            # Actually, let's use the LOCAL ε since that's what defines the
            # inner product at this δ. Both are reasonable; local is more physical.
            O = np.zeros((N_SUB, N_ALL))
            for m in range(N_SUB):
                for b in range(N_ALL):
                    # Use ε at the reference point for the inner product
                    # (this is the subspace-defining inner product)
                    ov = np.abs(eps_overlap(ref_states[m], bf[ix, iy, b], ref_eps))
                    O[m, b] = ov

            # Hungarian algorithm: maximize total overlap
            # linear_sum_assignment minimizes cost, so negate
            cost = -O
            row_ind, col_ind = linear_sum_assignment(cost)

            # col_ind[m] = which band at (ix,iy) matches reference band m
            reorder_map[ix, iy, :] = col_ind
            overlaps = O[row_ind, col_ind]
            match_quality[ix, iy] = overlaps.mean()
            min_overlap[ix, iy] = overlaps.min()

    print(f"  Done.")

    # ── Step 2: Diagnostics on the reordering ──
    print(f"\n{'='*70}")
    print(f"Reordering diagnostics")
    print(f"{'='*70}")

    # How often does each reference band map to a different index?
    for m in range(N_SUB):
        orig_band = SUBSPACE[m]
        same = np.sum(reorder_map[:, :, m] == orig_band)
        frac = same / (Nr * Nr)
        # Most common reassignment
        vals, counts = np.unique(reorder_map[:, :, m], return_counts=True)
        top3 = np.argsort(counts)[::-1][:3]
        desc = ", ".join([f"band {vals[t]}: {counts[t]/(Nr*Nr):.1%}" for t in top3])
        print(f"  Ref band {orig_band}: same index {frac:.1%} | top targets: {desc}")

    print(f"\n  Match quality (mean overlap):")
    print(f"    min={match_quality.min():.4f}, mean={match_quality.mean():.4f}, "
          f"max={match_quality.max():.4f}")
    print(f"  Min single-band overlap:")
    print(f"    min={min_overlap.min():.4f}, mean={min_overlap.mean():.4f}, "
          f"max={min_overlap.max():.4f}")
    print(f"  Fraction with mean overlap > 0.9: {np.sum(match_quality > 0.9)/(Nr*Nr):.1%}")
    print(f"  Fraction with mean overlap > 0.5: {np.sum(match_quality > 0.5)/(Nr*Nr):.1%}")
    print(f"  Fraction with min overlap > 0.5:  {np.sum(min_overlap > 0.5)/(Nr*Nr):.1%}")
    print(f"  Fraction with min overlap > 0.3:  {np.sum(min_overlap > 0.3)/(Nr*Nr):.1%}")

    # ── Step 3: C4 closure test — reordered vs original ──
    print(f"\n{'='*70}")
    print(f"C4 closure comparison: original vs reordered")
    print(f"{'='*70}")

    c4_det_orig    = np.zeros((Nr, Nr))
    c4_minsv_orig  = np.zeros((Nr, Nr))
    c4_det_reord   = np.zeros((Nr, Nr))
    c4_minsv_reord = np.zeros((Nr, Nr))

    for ix in range(Nr):
        if ix % 8 == 0:
            print(f"  C4 scan row {ix}/{Nr}...")
        for iy in range(Nr):
            # Original subspace
            states_orig = [bf[ix, iy, b] for b in SUBSPACE]
            det_o, msv_o, _ = c4_closure_metric(states_orig)
            c4_det_orig[ix, iy] = det_o
            c4_minsv_orig[ix, iy] = msv_o

            # Reordered subspace
            new_bands = reorder_map[ix, iy]
            states_reord = [bf[ix, iy, b] for b in new_bands]
            det_r, msv_r, _ = c4_closure_metric(states_reord)
            c4_det_reord[ix, iy] = det_r
            c4_minsv_reord[ix, iy] = msv_r

    # Report
    for name, det_map, msv_map in [("ORIGINAL", c4_det_orig, c4_minsv_orig),
                                    ("REORDERED", c4_det_reord, c4_minsv_reord)]:
        print(f"\n  {name}:")
        print(f"    min_sv > 0.9: {np.sum(msv_map > 0.9)/(Nr*Nr):.1%}")
        print(f"    min_sv > 0.5: {np.sum(msv_map > 0.5)/(Nr*Nr):.1%}")
        print(f"    min_sv > 0.3: {np.sum(msv_map > 0.3)/(Nr*Nr):.1%}")
        print(f"    mean(min_sv): {msv_map.mean():.4f}")
        print(f"    min(min_sv):  {msv_map.min():.4f}")
        print(f"    |det| > 0.5:  {np.sum(det_map > 0.5)/(Nr*Nr):.1%}")

    # ── Verdict ──
    pct_good = np.sum(c4_minsv_reord > 0.9) / (Nr * Nr)
    print(f"\n{'='*70}")
    if pct_good > 0.9:
        print(f"VERDICT: ✓ OUTCOME A — Overlap tracking works! ({pct_good:.1%} C4-closed)")
        print(f"  → Proceed with band reordering fix in the pipeline")
    elif pct_good > 0.5:
        print(f"VERDICT: ⚠ OUTCOME B — Partial success ({pct_good:.1%} C4-closed)")
        print(f"  → Band tracking helps but doesn't fully solve the problem")
        print(f"  → May need larger subspace or hybrid approach")
    else:
        print(f"VERDICT: ❌ OUTCOME C — Tracking fails ({pct_good:.1%} C4-closed)")
        print(f"  → Pivot to Wannier-style smooth subspace construction")
    print(f"{'='*70}")

    # ── Step 4: Detailed analysis at key points ──
    print(f"\n{'='*70}")
    print(f"Detailed reordering at key registry points")
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
        new_bands = reorder_map[ix, iy]
        mq = match_quality[ix, iy]
        mo = min_overlap[ix, iy]
        msv_o = c4_minsv_orig[ix, iy]
        msv_r = c4_minsv_reord[ix, iy]

        old_str = str(SUBSPACE)
        new_str = str(list(new_bands))
        changed = "CHANGED" if list(new_bands) != SUBSPACE else "same"

        print(f"  {label:20s}: {old_str} → {new_str} ({changed})")
        print(f"    {'':20s}  mean_ov={mq:.4f}, min_ov={mo:.4f}, "
              f"C4: {msv_o:.4f}→{msv_r:.4f}")

    # ── Step 5: Try with ε-weighted C4 ──
    print(f"\n{'='*70}")
    print(f"Bonus: C4 closure with ε-weighted overlaps")
    print(f"{'='*70}")

    c4_minsv_reord_eps = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            new_bands = reorder_map[ix, iy]
            states = [bf[ix, iy, b] for b in new_bands]
            _, msv, _ = c4_closure_metric(states, eps[ix, iy])
            c4_minsv_reord_eps[ix, iy] = msv

    pct_eps = np.sum(c4_minsv_reord_eps > 0.9) / (Nr * Nr)
    print(f"  Reordered + ε-weighted C4: min_sv>0.9 at {pct_eps:.1%}")
    print(f"  mean(min_sv) = {c4_minsv_reord_eps.mean():.4f}")

    # ── Step 6: Plots ──
    print(f"\nGenerating plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    # Row 1: C4 min_sv comparison
    im00 = axes[0, 0].pcolormesh(sr, sr, c4_minsv_orig.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('C4 min_sv — ORIGINAL [5-9]')
    axes[0, 0].set_aspect('equal'); plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].pcolormesh(sr, sr, c4_minsv_reord.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[0, 1].set_title('C4 min_sv — REORDERED')
    axes[0, 1].set_aspect('equal'); plt.colorbar(im01, ax=axes[0, 1])

    improvement = c4_minsv_reord - c4_minsv_orig
    im02 = axes[0, 2].pcolormesh(sr, sr, improvement.T, cmap='RdBu',
                                  shading='auto', vmin=-0.5, vmax=0.5)
    axes[0, 2].set_title('Improvement (reord - orig)')
    axes[0, 2].set_aspect('equal'); plt.colorbar(im02, ax=axes[0, 2])

    # Row 2: overlap quality + reorder map
    im10 = axes[1, 0].pcolormesh(sr, sr, match_quality.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('Mean overlap with ref')
    axes[1, 0].set_aspect('equal'); plt.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].pcolormesh(sr, sr, min_overlap.T, cmap='RdYlGn',
                                  shading='auto', vmin=0, vmax=1)
    axes[1, 1].set_title('Min single-band overlap')
    axes[1, 1].set_aspect('equal'); plt.colorbar(im11, ax=axes[1, 1])

    # Number of bands that changed from original
    n_changed = np.sum(reorder_map != np.array(SUBSPACE)[None, None, :], axis=2)
    im12 = axes[1, 2].pcolormesh(sr, sr, n_changed.T, cmap='hot_r',
                                  shading='auto', vmin=0, vmax=5)
    axes[1, 2].set_title('# bands changed from [5-9]')
    axes[1, 2].set_aspect('equal'); plt.colorbar(im12, ax=axes[1, 2])

    fig.suptitle('S3: Overlap-based band reordering — C4 closure test',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3_overlap_reorder.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S3_overlap_reorder.png")

    # Per-band reorder map
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    for m in range(N_SUB):
        im = axes[m].pcolormesh(sr, sr, reorder_map[:, :, m].T, cmap='tab20',
                                shading='auto', vmin=0, vmax=17)
        axes[m].set_title(f'Ref band {SUBSPACE[m]} → ?')
        axes[m].set_aspect('equal')
        plt.colorbar(im, ax=axes[m], shrink=0.8)
    fig.suptitle('S3: Band index after overlap tracking',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S3_band_mapping.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S3_band_mapping.png")

    # ── Step 7: Also try EXPANDED subspace (track best 7 bands, not 5) ──
    print(f"\n{'='*70}")
    print(f"Bonus: Expanded reference — track 7 bands [4-10] from ref")
    print(f"{'='*70}")

    EXP_SUB = [4, 5, 6, 7, 8, 9, 10]
    N_EXP = len(EXP_SUB)
    ref_states_exp = [bf[ref_ix, ref_iy, b] for b in EXP_SUB]

    c4_minsv_exp = np.zeros((Nr, Nr))
    for ix in range(Nr):
        for iy in range(Nr):
            O = np.zeros((N_EXP, N_ALL))
            for m in range(N_EXP):
                for b in range(N_ALL):
                    O[m, b] = np.abs(eps_overlap(ref_states_exp[m], bf[ix, iy, b], ref_eps))

            cost = -O
            row_ind, col_ind = linear_sum_assignment(cost)

            states_exp = [bf[ix, iy, b] for b in col_ind]
            _, msv, _ = c4_closure_metric(states_exp)
            c4_minsv_exp[ix, iy] = msv

    pct_exp = np.sum(c4_minsv_exp > 0.9) / (Nr * Nr)
    pct_exp5 = np.sum(c4_minsv_exp > 0.5) / (Nr * Nr)
    print(f"  7-band expanded: min_sv>0.9 at {pct_exp:.1%}, >0.5 at {pct_exp5:.1%}")
    print(f"  mean(min_sv) = {c4_minsv_exp.mean():.4f}")


if __name__ == '__main__':
    main()
