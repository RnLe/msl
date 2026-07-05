#!/usr/bin/env python3
"""
F06 — Part 2: ε-Weighted Orthogonality Verification

KEY INSIGHT:
  Phase 1 extracts E-field Bloch functions via ms.get_efield().
  E-fields satisfy the generalized eigenvalue problem:
      ∇×∇× E_n = ε(r) (ω_n/c)² E_n
  
  The correct orthogonality relation is:
      ⟨u_m | ε | u_n⟩ = ∫ ε(r) u_m*(r) · u_n(r) d²r = δ_mn
  
  NOT the flat inner product ⟨u_m|u_n⟩ = Σ u_m* · u_n = δ_mn.
  
  This script:
    1) Extracts ε(r) from MPB at representative registry points
    2) Verifies ε-weighted orthogonality (should be perfect)
    3) Computes ε-weighted orthogonality across the full grid
    4) Produces data for additional F06 plot rows

APPROACH:
  For each registry point, we need ε(r) on the MPB grid. Since ε depends
  on the stacking shift δ (the second cylinder moves), ε(r;δ) varies.
  
  We extract ε from MPB at all 64×64 points to be rigorous.
  Each extraction is fast (~10ms, no eigensolve needed), so 64×64 ≈ 40s.
"""

import numpy as np
import h5py
import json
import sys
import time
import math
import io
import contextlib

sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')

import meep as mp
from meep import mpb

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'
THETA = '2.000'
CDIR = f'{SWEEP}/theta_{THETA}/candidate_0000'

N_SUB = 3
# Crystal parameters (from phase0_meta.json)
LATTICE_TYPE = 'square'
R_OVER_A = 0.29
EPS_BG = 7.9
EPS_HOLE = 1.0
RESOLUTION = 32  # MPB resolution → Nx=Ny=32 for square lattice... 
# Wait — bloch fields are 64×64. Let me check.
# Actually the bloch fields shape is (64, 64, 10, 64, 64, 3)
# so Nx=Ny=64 meaning resolution=64. Let me verify from the data.


def extract_epsilon_from_mpb(delta_frac, resolution):
    """
    Extract ε(r) from MPB for a given registry shift.
    
    Uses MPB's subpixel averaging — the returned ε includes all
    the sophisticated boundary averaging that MPB uses internally.
    
    Args:
        delta_frac: [dx, dy] fractional stacking shift
        resolution: MPB resolution (pixels per lattice constant)
    
    Returns:
        eps_array: (Nx, Ny) real array of ε values on MPB grid
    """
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    
    r = R_OVER_A * 1.0
    cyl1 = mp.Cylinder(
        radius=r,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=EPS_HOLE)
    )
    geometry = [cyl1]
    
    if delta_frac is not None:
        cyl2 = mp.Cylinder(
            radius=r,
            center=mp.Vector3(delta_frac[0], delta_frac[1], 0),
            material=mp.Medium(epsilon=EPS_HOLE)
        )
        geometry.append(cyl2)
    
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=1,
        resolution=resolution,
        k_points=[mp.Vector3(0.5, 0, 0)],  # X point
    )
    
    # Suppress MPB output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ms.init_params(mp.NO_PARITY, False)
    
    # Get epsilon — returns MPBArray
    eps = ms.get_epsilon()
    
    # eps shape for 2D: (Nx, Ny, 1) or (Nx, Ny, 1, 1)
    eps_np = np.array(eps, dtype=np.float64)
    if eps_np.ndim == 3 and eps_np.shape[2] == 1:
        eps_np = eps_np[:, :, 0]
    elif eps_np.ndim == 4:
        eps_np = eps_np[:, :, 0, 0]
    
    return eps_np


def compute_eps_weighted_overlap(u_m, u_n, eps):
    """
    Compute ε-weighted overlap: ⟨u_m|ε|u_n⟩ / sqrt(⟨u_m|ε|u_m⟩ ⟨u_n|ε|u_n⟩)
    
    Args:
        u_m: (Nx, Ny, 3) complex field
        u_n: (Nx, Ny, 3) complex field
        eps: (Nx, Ny) real dielectric
    
    Returns:
        complex overlap (normalized)
    """
    # ε-weighted dot: Σ_ij ε(i,j) * u_m*(i,j,c) · u_n(i,j,c) 
    eps_3d = eps[:, :, np.newaxis]  # (Nx, Ny, 1)
    overlap = np.sum(eps_3d * np.conj(u_m) * u_n)
    norm_m = np.sqrt(np.abs(np.sum(eps_3d * np.conj(u_m) * u_m)))
    norm_n = np.sqrt(np.abs(np.sum(eps_3d * np.conj(u_n) * u_n)))
    if norm_m > 1e-30 and norm_n > 1e-30:
        return overlap / (norm_m * norm_n)
    return 0.0 + 0j


def run():
    t0 = time.time()
    print("=" * 70)
    print("F06 Part 2 — ε-Weighted Orthogonality Verification")
    print("=" * 70)
    
    # =========================================================================
    # STEP 0: Determine resolution from stored bloch fields
    # =========================================================================
    h5path = f'{CDIR}/phase1_multiband_data.h5'
    with h5py.File(h5path, 'r') as hf:
        bf_shape = hf['bloch_fields'].shape
        print(f"  Bloch fields shape: {bf_shape}")
        # (Ns1, Ns2, N_all, Nx, Ny, 3) 
        Ns1_bf, Ns2_bf, N_all, Nx, Ny, Nc = bf_shape
    
    # Ns1_bf is the bloch field grid (64), but the registry grid may be larger (128)
    # Check s_grid to get the actual registry sampling
    with h5py.File(h5path, 'r') as hf:
        s_grid = hf['s_grid'][:]  # (Ns1, Ns2, 2) — this is the FULL registry grid
        Ns1_full = s_grid.shape[0]
    
    # The bloch fields are on a SUBSAMPLED grid if Ns1_bf != Ns1_full
    print(f"  Registry grid: {Ns1_full}×{Ns1_full}, Bloch field grid: {Ns1_bf}×{Ns2_bf}")
    print(f"  Unit cell pixels: {Nx}×{Ny}×{Nc}")
    
    # MPB resolution = Nx (for square lattice with a=1)
    resolution = Nx
    print(f"  Inferred MPB resolution: {resolution}")
    
    # =========================================================================
    # STEP 1: Quick verification at a few registry points
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Verification at representative registry points")
    print("=" * 70)
    
    # Load subspace bloch fields
    with h5py.File(h5path, 'r') as hf:
        bf_raw = hf['bloch_fields'][:, :, :N_SUB, :, :, :]
    
    test_points = [(0, 0), (0, 32), (32, 0), (32, 32), (16, 48)]
    step = 1.0 / Ns1_bf
    
    print(f"\n  Testing at {len(test_points)} representative points...")
    print(f"  {'Point':>10s}  {'delta':>15s}  {'|<0|ε|1>|':>10s}  {'|<0|ε|2>|':>10s}  "
          f"{'|<1|ε|2>|':>10s}  {'|<0|1>|flat':>12s}  {'|<0|2>|flat':>12s}  {'|<1|2>|flat':>12s}")
    print("  " + "─" * 100)
    
    for ix, iy in test_points:
        delta = np.array([ix * step, iy * step])
        eps = extract_epsilon_from_mpb(delta, resolution)
        
        u = bf_raw[ix, iy]  # (N_sub, Nx, Ny, 3)
        
        # ε-weighted overlaps
        ov_01_eps = compute_eps_weighted_overlap(u[0], u[1], eps)
        ov_02_eps = compute_eps_weighted_overlap(u[0], u[2], eps)
        ov_12_eps = compute_eps_weighted_overlap(u[1], u[2], eps)
        
        # Flat overlaps (for comparison)
        def flat_ov(a, b):
            a_flat = a.ravel()
            b_flat = b.ravel()
            na = np.linalg.norm(a_flat)
            nb = np.linalg.norm(b_flat)
            if na > 1e-10 and nb > 1e-10:
                return np.abs(np.dot(np.conj(a_flat / na), b_flat / nb))
            return 0.0
        
        ov_01_flat = flat_ov(u[0], u[1])
        ov_02_flat = flat_ov(u[0], u[2])
        ov_12_flat = flat_ov(u[1], u[2])
        
        print(f"  ({ix:2d},{iy:2d})  ({delta[0]:.3f},{delta[1]:.3f})  "
              f"{abs(ov_01_eps):10.6f}  {abs(ov_02_eps):10.6f}  {abs(ov_12_eps):10.6f}  "
              f"{ov_01_flat:12.6f}  {ov_02_flat:12.6f}  {ov_12_flat:12.6f}")
    
    # =========================================================================
    # STEP 2: Extract ε at ALL registry points and compute full-grid overlaps
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Full-grid ε extraction and orthogonality")
    print("=" * 70)
    
    # Store ε for all registry points
    eps_grid = np.zeros((Ns1_bf, Ns2_bf, Nx, Ny), dtype=np.float32)
    
    print(f"  Extracting ε from MPB at {Ns1_bf}×{Ns2_bf} = {Ns1_bf*Ns2_bf} registry points...")
    t1 = time.time()
    
    for ix in range(Ns1_bf):
        if ix % 8 == 0:
            elapsed = time.time() - t1
            if ix > 0:
                rate = elapsed / ix
                remaining = rate * (Ns1_bf - ix)
                print(f"    Row {ix}/{Ns1_bf} ({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)")
            else:
                print(f"    Row {ix}/{Ns1_bf}...")
        for iy in range(Ns2_bf):
            delta = np.array([ix * step, iy * step])
            eps = extract_epsilon_from_mpb(delta, resolution)
            eps_grid[ix, iy] = eps.astype(np.float32)
    
    t_eps = time.time() - t1
    print(f"  ε extraction complete in {t_eps:.1f}s")
    print(f"  ε range: [{eps_grid.min():.3f}, {eps_grid.max():.3f}]")
    
    # =========================================================================
    # STEP 3: Compute ε-weighted orthogonality for full grid
    # =========================================================================
    print("\n  Computing ε-weighted orthogonality for all registry points...")
    
    ortho_eps = np.zeros((Ns1_bf, Ns2_bf, N_SUB, N_SUB))
    offdiag_eps = np.zeros((Ns1_bf, Ns2_bf))
    
    for ix in range(Ns1_bf):
        for iy in range(Ns2_bf):
            eps = eps_grid[ix, iy]
            u = bf_raw[ix, iy]  # (N_sub, Nx, Ny, 3)
            
            for m in range(N_SUB):
                for n in range(N_SUB):
                    ov = compute_eps_weighted_overlap(u[m], u[n], eps)
                    ortho_eps[ix, iy, m, n] = np.abs(ov)
            
            # Max off-diagonal
            for m in range(N_SUB):
                for n in range(N_SUB):
                    if m != n:
                        offdiag_eps[ix, iy] = max(offdiag_eps[ix, iy], 
                                                   ortho_eps[ix, iy, m, n])
    
    print(f"\n  ε-weighted orthogonality results:")
    print(f"    max offdiag |⟨u_m|ε|u_n⟩|: mean={offdiag_eps.mean():.6f}, "
          f"max={offdiag_eps.max():.6f}")
    for m in range(N_SUB):
        for n in range(m + 1, N_SUB):
            vals = ortho_eps[:, :, m, n].ravel()
            print(f"    |⟨u_{m}|ε|u_{n}⟩|: mean={vals.mean():.6f}, "
                  f"max={vals.max():.6f}")

    # Also compute diagonal (should be ≈ 1.0 when properly normalized)
    for n in range(N_SUB):
        diag = ortho_eps[:, :, n, n].ravel()
        print(f"    ⟨u_{n}|ε|u_{n}⟩: mean={diag.mean():.6f}, std={diag.std():.6f}")
    
    # =========================================================================
    # STEP 4: ε-normalize and show that orthogonality is restored
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: After ε-normalization: ⟨u|ε|u⟩ = NxNy")
    print("=" * 70)
    
    # ε-normalize each field: u → u / sqrt(⟨u|ε|u⟩/(NxNy))
    bf_eps_normed = bf_raw.copy()
    Npix = Nx * Ny
    
    for ix in range(Ns1_bf):
        for iy in range(Ns2_bf):
            eps = eps_grid[ix, iy]
            eps_3d = eps[:, :, np.newaxis]  # (Nx, Ny, 1)
            for n in range(N_SUB):
                u = bf_eps_normed[ix, iy, n]  # (Nx, Ny, 3)
                eps_norm_sq = np.sum(eps_3d * np.abs(u)**2) / Npix
                if eps_norm_sq > 1e-30:
                    bf_eps_normed[ix, iy, n] = u / np.sqrt(eps_norm_sq)
    
    # Verify
    ortho_eps_after = np.zeros((Ns1_bf, Ns2_bf, N_SUB, N_SUB))
    for ix in range(Ns1_bf):
        for iy in range(Ns2_bf):
            eps = eps_grid[ix, iy]
            u = bf_eps_normed[ix, iy]
            for m in range(N_SUB):
                for n in range(N_SUB):
                    ov = compute_eps_weighted_overlap(u[m], u[n], eps)
                    ortho_eps_after[ix, iy, m, n] = np.abs(ov)
    
    offdiag_eps_after = np.zeros((Ns1_bf, Ns2_bf))
    for m in range(N_SUB):
        for n in range(N_SUB):
            if m != n:
                offdiag_eps_after = np.maximum(offdiag_eps_after, ortho_eps_after[:, :, m, n])
    
    print(f"\n  After ε-normalization:")
    for n in range(N_SUB):
        diag = ortho_eps_after[:, :, n, n].ravel()
        print(f"    ⟨u_{n}|ε|u_{n}⟩_Ω: mean={diag.mean():.6f}, std={diag.std():.2e}")
    print(f"    max offdiag |⟨u_m|ε|u_n⟩|: mean={offdiag_eps_after.mean():.6f}, "
          f"max={offdiag_eps_after.max():.6f}")
    for m in range(N_SUB):
        for n in range(m + 1, N_SUB):
            vals = ortho_eps_after[:, :, m, n].ravel()
            print(f"    |⟨u_{m}|ε|u_{n}⟩|: mean={vals.mean():.6f}, max={vals.max():.6f}")
    
    # =========================================================================
    # STEP 5: Also compute FLAT orthogonality after gauge fix (for comparison)  
    # =========================================================================
    # Load F06_after_data to get gauge-fixed flat ortho
    after_data = np.load(f'{FINDINGS}/F06_after_data.npz')
    ortho_flat_gf = after_data['ortho']  # (Ns1, Ns2, N_sub, N_sub) — flat, gauge-fixed
    
    # =========================================================================
    # SAVE
    # =========================================================================
    np.savez_compressed(
        f'{FINDINGS}/F06_epsilon_data.npz',
        eps_grid=eps_grid,
        ortho_eps_raw=ortho_eps,
        offdiag_eps_raw=offdiag_eps,
        ortho_eps_normed=ortho_eps_after,
        offdiag_eps_normed=offdiag_eps_after,
        ortho_flat_gf=ortho_flat_gf,
        Ns1=Ns1_bf, Ns2=Ns2_bf, N_sub=N_SUB, Nx=Nx, Ny=Ny,
    )
    print(f"\n  Saved: {FINDINGS}/F06_epsilon_data.npz")
    
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  Key finding:")
    print(f"    Flat |⟨u_0|u_2⟩|:   mean={ortho_flat_gf[:,:,0,2].mean():.4f}, max={ortho_flat_gf[:,:,0,2].max():.4f}")
    print(f"    ε-wt |⟨u_0|ε|u_2⟩|: mean={ortho_eps[:,:,0,2].mean():.6f}, max={ortho_eps[:,:,0,2].max():.6f}")
    print(f"    → ε-weighted orthogonality is {'PERFECT' if offdiag_eps.max() < 0.01 else 'IMPERFECT — investigate!'}")


if __name__ == '__main__':
    run()
