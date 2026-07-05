#!/usr/bin/env python3
"""
S2b: Smaller subspace closure tests
=====================================
Check if 1-band, 2-band, or 3-band subspaces are C4-closed
where the 5-band subspace fails catastrophically.
Also check gaps to bands OUTSIDE the 5-band subspace.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent / "plots"
OUTDIR.mkdir(exist_ok=True)

BASE  = Path(__file__).resolve().parent.parent
CAND  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
P1    = CAND / "phase1_multiband_data.h5"

ALL_BANDS = list(range(18))
SUBSPACE  = [5, 6, 7, 8, 9]


def rotate_90_origin(field, Nx):
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


def subspace_closure(bf, band_indices, ix, iy, Nx=32):
    """Compute C4 closure metric for a given subspace at one R-point."""
    Nb = len(band_indices)
    states  = [bf[ix, iy, b] for b in band_indices]
    rotated = [rotate_90_origin(u, Nx) for u in states]
    norms   = [np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx)) for u in states]

    M = np.zeros((Nb, Nb), dtype=complex)
    for m in range(Nb):
        for n in range(Nb):
            M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx*Nx)

    M_norm = M / np.outer(norms, norms)
    sv = np.linalg.svd(M_norm, compute_uv=False)
    return np.abs(np.linalg.det(M_norm)), sv.min(), sv


def main():
    with h5py.File(P1, 'r') as f:
        bf = f['bloch_fields'][:]  # (64,64,18,32,32,3)

    Nr, Nx = 64, 32

    # ══════════════════════════════════════════════════════════════════
    # Test different subspace sizes across the registry grid
    # ══════════════════════════════════════════════════════════════════
    subspaces = {
        "band 7 alone":     [7],
        "bands 7,8":        [7, 8],
        "bands 6,7,8":      [6, 7, 8],
        "bands 5,6,7,8,9":  [5, 6, 7, 8, 9],
        "bands 4-9 (6)":    [4, 5, 6, 7, 8, 9],
        "bands 4-10 (7)":   [4, 5, 6, 7, 8, 9, 10],
        "bands 3-10 (8)":   [3, 4, 5, 6, 7, 8, 9, 10],
    }

    results = {}
    for name, bands in subspaces.items():
        Nb = len(bands)
        det_map = np.zeros((Nr, Nr))
        minsv_map = np.zeros((Nr, Nr))

        for ix in range(Nr):
            for iy in range(Nr):
                det_val, minsv_val, _ = subspace_closure(bf, bands, ix, iy, Nx)
                det_map[ix, iy] = det_val
                minsv_map[ix, iy] = minsv_val

        pct_closed = np.sum(minsv_map > 0.5) / (Nr*Nr)
        pct_good   = np.sum(minsv_map > 0.9) / (Nr*Nr)
        results[name] = (det_map, minsv_map)

        print(f"  {name:25s}: "
              f"min_sv>0.5: {pct_closed:6.1%}, "
              f"min_sv>0.9: {pct_good:6.1%}, "
              f"mean(min_sv)={minsv_map.mean():.4f}, "
              f"min(min_sv)={minsv_map.min():.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Plot comparison
    # ══════════════════════════════════════════════════════════════════
    n_sub = len(subspaces)
    fig, axes = plt.subplots(2, n_sub, figsize=(4*n_sub, 8))
    sr = np.linspace(0, 1, Nr, endpoint=False)

    for idx, (name, (det_map, minsv_map)) in enumerate(results.items()):
        im0 = axes[0, idx].pcolormesh(sr, sr, det_map.T, cmap='RdYlGn',
                                       shading='auto', vmin=0, vmax=1)
        axes[0, idx].set_title(f'|det| {name}', fontsize=8)
        axes[0, idx].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0, idx], shrink=0.6)

        im1 = axes[1, idx].pcolormesh(sr, sr, minsv_map.T, cmap='RdYlGn',
                                       shading='auto', vmin=0, vmax=1)
        axes[1, idx].set_title(f'min_sv {name}', fontsize=8)
        axes[1, idx].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1, idx], shrink=0.6)

    fig.suptitle('S2b: C4 closure for different subspace sizes',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S2b_subspace_sizes.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved S2b_subspace_sizes.png")

    # ══════════════════════════════════════════════════════════════════
    # Detailed analysis at key points
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Detailed C4 overlap at δ=(0.5,0.5) for extended subspaces")
    print(f"{'='*70}")

    ix, iy = 32, 32

    for name, bands in [("bands 4-10 (7)", [4,5,6,7,8,9,10]),
                         ("bands 3-10 (8)", [3,4,5,6,7,8,9,10])]:
        Nb = len(bands)
        states  = [bf[ix, iy, b] for b in bands]
        rotated = [rotate_90_origin(u, Nx) for u in states]
        norms   = [np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx)) for u in states]

        M = np.zeros((Nb, Nb), dtype=complex)
        for m in range(Nb):
            for n in range(Nb):
                M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx*Nx)

        M_norm = M / np.outer(norms, norms)
        sv = np.linalg.svd(M_norm, compute_uv=False)
        evals = np.linalg.eigvals(M_norm)

        print(f"\n  {name}:")
        print(f"    |M| matrix:")
        for m in range(Nb):
            row = "      "
            for n in range(Nb):
                row += f"{np.abs(M_norm[m,n]):7.3f}"
            print(row)
        print(f"    Singular values: {np.sort(sv)[::-1]}")
        print(f"    |det| = {np.abs(np.linalg.det(M_norm)):.6f}")
        print(f"    Eigenvalue mags: {np.sort(np.abs(evals))[::-1]}")
        print(f"    Eigenvalue phases/π: {np.sort(np.angle(evals)/np.pi)}")

    # ══════════════════════════════════════════════════════════════════
    # What does the C4 image of each band look like at δ=(0.5,0.5)?
    # Find which bands outside [5-9] the rotation maps to
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Where does C4 map each band at δ=(0.5,0.5)?")
    print(f"{'='*70}")

    ix, iy = 32, 32
    for n_band in range(18):
        u = bf[ix, iy, n_band]
        u_rot = rotate_90_origin(u, Nx)
        norm_u = np.sqrt(np.sum(np.abs(u)**2) / (Nx*Nx))
        if norm_u < 1e-10:
            continue

        overlaps = []
        for m_band in range(18):
            um = bf[ix, iy, m_band]
            norm_um = np.sqrt(np.sum(np.abs(um)**2) / (Nx*Nx))
            if norm_um < 1e-10:
                ov = 0.0
            else:
                ov = np.abs(np.sum(np.conj(um) * u_rot) / (Nx*Nx)) / (norm_um * norm_u)
            overlaps.append(ov)

        # Show top 3 target bands
        top3 = np.argsort(overlaps)[::-1][:3]
        desc = ", ".join([f"→band {t}: {overlaps[t]:.3f}" for t in top3])
        in_sub = "✓" if n_band in SUBSPACE else " "
        print(f"  Band {n_band:2d} {in_sub}: {desc}")


if __name__ == '__main__':
    main()
