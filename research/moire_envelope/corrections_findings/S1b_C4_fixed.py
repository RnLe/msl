#!/usr/bin/env python3
"""
S1b: Fix C4 symmetry test and investigate degeneracies.

The S1 test wrongly rotated about grid center (Nx/2, Ny/2).
For MPB's cell-periodic Bloch functions, C4 is about r=(0,0),
which on a periodic grid [0,a) maps as (ix,iy) → (Nx-iy mod Nx, ix).
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

SUBSPACE = [5, 6, 7, 8, 9]


def rotate_90_origin(field, Nx):
    """
    C4 rotation (90° CCW) about origin on periodic [0,a) grid.
    Position map: (x,y) → (-y,x) → ((Nx-iy) mod Nx, ix) in index space.
    Vector rotation: (Ex,Ey,Ez) → (-Ey,Ex,Ez).
    """
    rotated = np.zeros_like(field)
    for ix in range(Nx):
        for iy in range(Nx):
            jx = (Nx - iy) % Nx
            jy = ix
            rotated[jx, jy, 0] = -field[ix, iy, 1]
            rotated[jx, jy, 1] =  field[ix, iy, 0]
            rotated[jx, jy, 2] =  field[ix, iy, 2]
    return rotated


def c4_fidelity(u, u_rot, Nx):
    """Compute |⟨u|R₄u⟩|/(||u|| ||R₄u||)."""
    ov = np.sum(np.conj(u) * u_rot) / (Nx * Nx)
    norm_u = np.sqrt(np.sum(np.abs(u)**2) / (Nx * Nx))
    norm_r = np.sqrt(np.sum(np.abs(u_rot)**2) / (Nx * Nx))
    return np.abs(ov) / (norm_u * norm_r + 1e-30), np.angle(ov)


def main():
    with h5py.File(P1, 'r') as f:
        bf  = f['bloch_fields'][:]  # (64,64,18,32,32,3)
        eps = f['epsilon'][:]       # (64,64,32,32)

    Nx = 32

    # ── Fixed C4 test at C4-symmetric registry points ──
    symm_points = [
        ("δ=(0,0)",       (0,  0)),
        ("δ=(0.5,0.5)",   (32, 32)),
    ]

    for label, (ix, iy) in symm_points:
        print(f"\n{'='*60}")
        print(f"C4 test at {label}  (registry ix,iy=({ix},{iy}))")
        print(f"{'='*60}")

        for n_sub, n_band in enumerate(SUBSPACE):
            u = bf[ix, iy, n_band]  # (32,32,3)
            u_rot = rotate_90_origin(u, Nx)

            fid, phase = c4_fidelity(u, u_rot, Nx)

            # Intensity check
            I_orig = np.sum(np.abs(u)**2, axis=2)
            I_rot  = np.sum(np.abs(u_rot)**2, axis=2)
            I_rel  = np.max(np.abs(I_orig - I_rot)) / (I_orig.max() + 1e-30)

            status = "✓" if fid > 0.99 else ("⚠" if fid > 0.9 else "❌")
            print(f"  Band {n_band}: fidelity={fid:.6f}, "
                  f"phase={phase/np.pi:.3f}π, I_diff={I_rel:.4e} {status}")

        # ── Degenerate pair check for bands 7,8 ──
        print(f"\n  Degenerate pair (bands 7,8) joint C4 check:")
        u7 = bf[ix, iy, 7]
        u8 = bf[ix, iy, 8]
        u7r = rotate_90_origin(u7, Nx)
        u8r = rotate_90_origin(u8, Nx)

        # Build 2×2 overlap matrix: M_mn = ⟨u_m | C4 u_n⟩ for m,n ∈ {7,8}
        states = [u7, u8]
        rotated = [u7r, u8r]
        M = np.zeros((2, 2), dtype=complex)
        for m in range(2):
            for n in range(2):
                M[m, n] = np.sum(np.conj(states[m]) * rotated[n]) / (Nx * Nx)

        norms = np.array([np.sqrt(np.sum(np.abs(s)**2) / (Nx * Nx)) for s in states])
        M_norm = M / np.outer(norms, norms)

        print(f"    Overlap matrix (normalized):")
        print(f"      [[{M_norm[0,0]:.4f}, {M_norm[0,1]:.4f}],")
        print(f"       [{M_norm[1,0]:.4f}, {M_norm[1,1]:.4f}]]")
        print(f"    |det| = {np.abs(np.linalg.det(M_norm)):.6f}")

        # If bands form a 2D E-rep, M should be unitary with |det|≈1
        # Eigenvalues of C4 in E-rep should be exp(±iπ/2) = ±i
        evals = np.linalg.eigvals(M_norm)
        print(f"    Eigenvalues: {evals[0]:.4f}, {evals[1]:.4f}")
        print(f"    |eigenvalues|: {np.abs(evals[0]):.4f}, {np.abs(evals[1]):.4f}")
        print(f"    Phases/π: {np.angle(evals[0])/np.pi:.4f}, {np.angle(evals[1])/np.pi:.4f}")

        if np.abs(np.abs(np.linalg.det(M_norm)) - 1) < 0.1:
            print(f"    ✓ Pair behaves as 2D representation (|det|≈1)")
        else:
            print(f"    ❌ Pair does NOT form valid 2D representation")

        # ── Also check ALL 5 bands as a joint subspace ──
        print(f"\n  Full 5-band subspace C4 overlap matrix:")
        all_u  = [bf[ix, iy, b] for b in SUBSPACE]
        all_ur = [rotate_90_origin(u, Nx) for u in all_u]
        M5 = np.zeros((5, 5), dtype=complex)
        norms5 = np.zeros(5)
        for m in range(5):
            norms5[m] = np.sqrt(np.sum(np.abs(all_u[m])**2) / (Nx * Nx))
        for m in range(5):
            for n in range(5):
                M5[m, n] = np.sum(np.conj(all_u[m]) * all_ur[n]) / (Nx * Nx)

        M5_norm = M5 / np.outer(norms5, norms5)
        print(f"    |M5_mn|:")
        for m in range(5):
            row = "    "
            for n in range(5):
                row += f"{np.abs(M5_norm[m,n]):8.4f}"
            print(row)

        evals5 = np.linalg.eigvals(M5_norm)
        print(f"    Eigenvalue magnitudes: {np.sort(np.abs(evals5))[::-1]}")
        print(f"    Eigenvalue phases/π:   {np.sort(np.angle(evals5)/np.pi)}")

    # ── Also check: does epsilon itself have C4 symmetry? ──
    print(f"\n{'='*60}")
    print("Epsilon C4 symmetry check")
    print(f"{'='*60}")

    for label, (ix, iy) in symm_points:
        e = eps[ix, iy]  # (32,32)
        # Rotate epsilon: scalar field, no vector rotation
        e_rot = np.zeros_like(e)
        for i in range(Nx):
            for j in range(Nx):
                ji = (Nx - j) % Nx
                jj = i
                e_rot[ji, jj] = e[i, j]
        diff = np.max(np.abs(e - e_rot)) / e.max()
        print(f"  {label}: max|ε - C4ε|/max(ε) = {diff:.6e}")
        if diff < 1e-10:
            print(f"    ✓ ε is C4-symmetric")
        else:
            print(f"    ❌ ε is NOT C4-symmetric!")

    # ── Visualize fields at δ=(0,0) ──
    print(f"\n{'='*60}")
    print("Generating field visualization at δ=(0,0)")
    print(f"{'='*60}")

    ix, iy = 0, 0
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for n_sub, n_band in enumerate(SUBSPACE):
        u = bf[ix, iy, n_band]  # (32,32,3)
        I = np.sum(np.abs(u)**2, axis=2)  # intensity

        u_rot = rotate_90_origin(u, Nx)
        I_rot = np.sum(np.abs(u_rot)**2, axis=2)

        im0 = axes[0, n_sub].imshow(I.T, origin='lower', cmap='hot')
        axes[0, n_sub].set_title(f'Band {n_band} |u|²')
        plt.colorbar(im0, ax=axes[0, n_sub], shrink=0.6)

        im1 = axes[1, n_sub].imshow((I - I_rot).T, origin='lower',
                                     cmap='RdBu_r', vmin=-I.max()*0.5, vmax=I.max()*0.5)
        axes[1, n_sub].set_title(f'Band {n_band} |u|²-|C₄u|²')
        plt.colorbar(im1, ax=axes[1, n_sub], shrink=0.6)

    fig.suptitle('Bloch functions at δ=(0,0) — C4 symmetry test', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S1b_C4_fields_delta00.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S1b_C4_fields_delta00.png")

    # ── Visualize fields at δ=(0.5,0.5) ──
    ix, iy = 32, 32
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for n_sub, n_band in enumerate(SUBSPACE):
        u = bf[ix, iy, n_band]
        I = np.sum(np.abs(u)**2, axis=2)

        u_rot = rotate_90_origin(u, Nx)
        I_rot = np.sum(np.abs(u_rot)**2, axis=2)

        im0 = axes[0, n_sub].imshow(I.T, origin='lower', cmap='hot')
        axes[0, n_sub].set_title(f'Band {n_band} |u|²')
        plt.colorbar(im0, ax=axes[0, n_sub], shrink=0.6)

        im1 = axes[1, n_sub].imshow((I - I_rot).T, origin='lower',
                                     cmap='RdBu_r', vmin=-I.max()*0.5, vmax=I.max()*0.5)
        axes[1, n_sub].set_title(f'Band {n_band} |u|²-|C₄u|²')
        plt.colorbar(im1, ax=axes[1, n_sub], shrink=0.6)

    fig.suptitle('Bloch functions at δ=(0.5,0.5) — C4 symmetry test', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / "S1b_C4_fields_delta05.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S1b_C4_fields_delta05.png")

    # ── Visualize ε at both points ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, (label, (ix, iy)) in enumerate(symm_points):
        e = eps[ix, iy]
        im = axes[idx].imshow(e.T, origin='lower', cmap='viridis')
        axes[idx].set_title(f'ε at {label}')
        plt.colorbar(im, ax=axes[idx], shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "S1b_epsilon.png", dpi=150)
    plt.close(fig)
    print(f"  Saved S1b_epsilon.png")


if __name__ == '__main__':
    main()
