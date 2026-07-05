#!/usr/bin/env python3
"""
F06 — Bloch Field Gauge & Normalization Diagnostic

PURPOSE:
  Visualize and quantify the two problems with raw MPB Bloch fields:
    1) Normalization is inconsistent (MPB uses ε-weighted norm, not ⟨u|u⟩_Ω = 1)
    2) Phase/gauge is arbitrary at each registry point (random gauge)

  Then apply fixes (proper normalization + SVD parallel-transport gauge)
  and show the repaired state.

DATASETS PRODUCED:
  F06_before_data.npz — raw MPB state (norms, phases, orthogonality)
  F06_after_data.npz  — after normalization + gauge fixing

REFERENCE:
  findings/FixingBlochAndGauge.md — theoretical analysis
  phasesV3/phase2_mpb_v3.py — existing gauge-fixing code
  phasesV3/bloch_fields.py — normalization conventions
"""

import numpy as np
import h5py
import json
import sys
import time
import gc

sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
# We do NOT use apply_parallel_transport_gauge — it's broken for these fields.
# See Section B below for the correct Abelian gauge fix.

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'
# Use theta=2.0 as representative (Bloch fields are theta-independent)
THETA = '2.000'
CDIR = f'{SWEEP}/theta_{THETA}/candidate_0000'

N_SUB = 3  # subspace bands for X-point crystal


def apply_abelian_gauge_2d(bloch_fields):
    """
    Abelian (per-band scalar) gauge fix on 2D registry grid.

    For each band independently, align the complex phase between neighbors
    so that ⟨u_n(s)|u_n(s+δs)⟩ is real and positive.

    Algorithm (seed-row + columns):
      1) Fix reference row (i=0) along s2: align j → j+1
      2) For each column j, fix along s1: align i → i+1, seeded from row 0

    This is Abelian (no band mixing) so it:
      ✓ Preserves orthogonality between bands
      ✓ Preserves normalization
      ✓ Avoids the ε-inner-product problem (MPB fields are NOT orthogonal
        under flat inner product — non-Abelian SVD mixing destroys them)

    Limitation: At topological singularities (|⟨u(s)|u(s+ds)⟩| ≈ 0),
    the phase is undefined. These points get whatever phase the path assigns.

    Args:
        bloch_fields: (Ns1, Ns2, N_bands, Nx, Ny, 3) complex array

    Returns:
        fixed: same shape, with smooth per-band phase
        diagnostics: dict of per-band phase-unwinding info
    """
    Ns1, Ns2, N_bands = bloch_fields.shape[:3]
    fixed = np.copy(bloch_fields)
    diag = {}

    for n in range(N_bands):
        min_ov = 1.0
        n_singular = 0

        # Step 1: Fix row 0 along s2
        for j in range(Ns2 - 1):
            u_curr = fixed[0, j, n].ravel()  # (Nx*Ny*3,)
            u_next = fixed[0, j + 1, n].ravel()
            # Normalize for overlap computation
            nc = np.linalg.norm(u_curr)
            nn = np.linalg.norm(u_next)
            if nc > 1e-10 and nn > 1e-10:
                ov = np.dot(np.conj(u_curr / nc), u_next / nn)
                mag = abs(ov)
                min_ov = min(min_ov, mag)
                if mag > 1e-10:
                    # Rotate u_next by exp(-i arg(ov)) = conj(ov)/|ov|
                    fixed[0, j + 1, n] *= np.conj(ov) / mag
                else:
                    n_singular += 1

        # Step 2: Fix each column j along s1, seeded from gauge-fixed row 0
        for j in range(Ns2):
            for i in range(Ns1 - 1):
                u_curr = fixed[i, j, n].ravel()
                u_next = fixed[i + 1, j, n].ravel()
                nc = np.linalg.norm(u_curr)
                nn = np.linalg.norm(u_next)
                if nc > 1e-10 and nn > 1e-10:
                    ov = np.dot(np.conj(u_curr / nc), u_next / nn)
                    mag = abs(ov)
                    min_ov = min(min_ov, mag)
                    if mag > 1e-10:
                        fixed[i + 1, j, n] *= np.conj(ov) / mag
                    else:
                        n_singular += 1

        diag[n] = {'min_ov': min_ov, 'n_singular': n_singular}
        print(f"      Band {n}: min|ov|={min_ov:.4f}, singular_pts={n_singular}")

    return fixed, diag


def compute_diagnostics(bloch_fields_sub, label=""):
    """
    Compute normalization, phase, and orthogonality diagnostics.

    Args:
        bloch_fields_sub: (Ns1, Ns2, N_sub, Nx, Ny, 3) complex array
        label: string for print messages

    Returns:
        dict with diagnostic arrays
    """
    Ns1, Ns2, N_sub, Nx, Ny, Nc = bloch_fields_sub.shape
    Npix = Nx * Ny  # number of spatial pixels per tile

    print(f"\n  Computing diagnostics ({label})...")
    print(f"    Grid: {Ns1}×{Ns2}, Bands: {N_sub}, Unit cell: {Nx}×{Ny}×{Nc}")

    # --- 1. Normalization ---
    # Flat L2 norm² = sum |u|² over (Nx, Ny, 3)
    norm_sq_flat = np.sum(
        np.abs(bloch_fields_sub)**2, axis=(-3, -2, -1)
    )  # (Ns1, Ns2, N_sub)

    # Cell-averaged norm² = sum |u|² / (Nx*Ny)
    norm_sq_avg = norm_sq_flat / Npix  # (Ns1, Ns2, N_sub)

    for n in range(N_sub):
        nf = norm_sq_flat[:, :, n]
        na = norm_sq_avg[:, :, n]
        print(f"    Band {n}: ||u||²_flat = {nf.mean():.4f} ± {nf.std():.4f} "
              f"[{nf.min():.4f}, {nf.max():.4f}]")
        print(f"             ||u||²_avg  = {na.mean():.6f} ± {na.std():.6f} "
              f"[{na.min():.6f}, {na.max():.6f}]")

    # --- 2. Neighbor overlap phase ---
    # Normalize each tile for overlap computation (to isolate phase from norm)
    norms = np.sqrt(norm_sq_flat)[..., np.newaxis, np.newaxis, np.newaxis]
    norms = np.maximum(norms, 1e-30)
    bf_normed = bloch_fields_sub / norms  # each tile has ||u||=1

    # Overlap along s1: <u(i,j)|u(i+1,j)>
    bf_shifted_s1 = np.roll(bf_normed, -1, axis=0)
    bf_shifted_s2 = np.roll(bf_normed, -1, axis=1)

    ov_s1 = np.sum(
        np.conj(bf_normed) * bf_shifted_s1, axis=(-3, -2, -1)
    )  # (Ns1, Ns2, N_sub)
    ov_s2 = np.sum(
        np.conj(bf_normed) * bf_shifted_s2, axis=(-3, -2, -1)
    )  # (Ns1, Ns2, N_sub)

    phase_s1 = np.angle(ov_s1)  # (Ns1, Ns2, N_sub)
    phase_s2 = np.angle(ov_s2)
    mag_s1 = np.abs(ov_s1)
    mag_s2 = np.abs(ov_s2)

    for n in range(N_sub):
        ps1 = phase_s1[:, :, n].ravel()
        ps2 = phase_s2[:, :, n].ravel()
        ms1 = mag_s1[:, :, n].ravel()
        ms2 = mag_s2[:, :, n].ravel()
        frac_good = float(np.mean((ms1 > 0.99) & (ms2 > 0.99)))
        print(f"    Band {n}: phase_std=({np.std(ps1):.4f}, {np.std(ps2):.4f}) rad, "
              f"|ov|_min=({ms1.min():.4f}, {ms2.min():.4f}), "
              f"frac(|ov|>0.99)={frac_good:.3f}")

    # --- 3. Orthogonality ---
    # For each tile, compute |<u_m|u_n>| for m≠n (using normalized fields)
    ortho = np.zeros((Ns1, Ns2, N_sub, N_sub))
    for m in range(N_sub):
        for n in range(N_sub):
            # <u_m|u_n> at each tile
            ov = np.sum(
                np.conj(bf_normed[:, :, m]) * bf_normed[:, :, n],
                axis=(-3, -2, -1)
            )
            ortho[:, :, m, n] = np.abs(ov)

    # Max off-diagonal overlap per tile
    offdiag_max = np.zeros((Ns1, Ns2))
    for m in range(N_sub):
        for n in range(N_sub):
            if m != n:
                offdiag_max = np.maximum(offdiag_max, ortho[:, :, m, n])

    print(f"    Orthogonality: max|<u_m|u_n>| (m≠n) = {offdiag_max.max():.4f}, "
          f"mean = {offdiag_max.mean():.4f}")
    for m in range(N_sub):
        for n in range(m + 1, N_sub):
            vals = ortho[:, :, m, n].ravel()
            print(f"      |<u_{m}|u_{n}>|: mean={vals.mean():.4f}, "
                  f"max={vals.max():.4f}, "
                  f"frac(<0.01)={float(np.mean(vals < 0.01)):.3f}")

    return {
        'norm_sq_flat': norm_sq_flat,      # (Ns1, Ns2, N_sub)
        'norm_sq_avg': norm_sq_avg,        # (Ns1, Ns2, N_sub)
        'phase_s1': phase_s1,              # (Ns1, Ns2, N_sub)
        'phase_s2': phase_s2,              # (Ns1, Ns2, N_sub)
        'mag_s1': mag_s1,                  # (Ns1, Ns2, N_sub)
        'mag_s2': mag_s2,                  # (Ns1, Ns2, N_sub)
        'ortho': ortho,                    # (Ns1, Ns2, N_sub, N_sub)
        'offdiag_max': offdiag_max,        # (Ns1, Ns2)
    }


def run():
    t0 = time.time()
    print("=" * 70)
    print("F06 — Bloch Field Gauge & Normalization Diagnostic")
    print("=" * 70)
    print(f"  Data: {CDIR}")
    print(f"  Subspace bands: 0..{N_SUB - 1}")

    # =========================================================================
    # LOAD RAW BLOCH FIELDS
    # =========================================================================
    print("\n  Loading raw Bloch fields from Phase 1 HDF5...")
    h5path = f'{CDIR}/phase1_multiband_data.h5'
    with h5py.File(h5path, 'r') as hf:
        bf_shape = hf['bloch_fields'].shape
        print(f"    Full bloch_fields shape: {bf_shape}")
        # (Ns1, Ns2, N_all, Nx, Ny, 3)
        # Load only subspace bands to save memory
        bf_raw = hf['bloch_fields'][:, :, :N_SUB, :, :, :]
        print(f"    Loaded subspace slice: {bf_raw.shape}, dtype={bf_raw.dtype}")

    Ns1, Ns2, _, Nx, Ny, Nc = bf_raw.shape
    Npix = Nx * Ny

    # =========================================================================
    # SECTION A: "BEFORE" — Raw MPB state
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SECTION A: RAW MPB STATE (before fixes)")
    print("=" * 70)

    before = compute_diagnostics(bf_raw, label="RAW")

    np.savez_compressed(
        f'{FINDINGS}/F06_before_data.npz',
        norm_sq_flat=before['norm_sq_flat'],
        norm_sq_avg=before['norm_sq_avg'],
        phase_s1=before['phase_s1'],
        phase_s2=before['phase_s2'],
        mag_s1=before['mag_s1'],
        mag_s2=before['mag_s2'],
        ortho=before['ortho'],
        offdiag_max=before['offdiag_max'],
        Ns1=Ns1, Ns2=Ns2, N_sub=N_SUB, Nx=Nx, Ny=Ny,
    )
    print(f"\n  Saved: {FINDINGS}/F06_before_data.npz")

    # =========================================================================
    # SECTION B: "AFTER" — Normalized + gauge-fixed
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SECTION B: AFTER NORMALIZATION + SVD GAUGE FIX")
    print("=" * 70)

    bf_fixed = bf_raw.copy()

    # --- Fix 1: Normalize to ⟨u|u⟩_Ω = sum|u|²/(Nx*Ny) = 1 ---
    print("\n  Fix 1: Cell-averaged normalization (⟨u|u⟩_Ω = 1)...")
    norm_sq = np.sum(np.abs(bf_fixed)**2, axis=(-3, -2, -1), keepdims=True)
    norm_sq_per_pixel = norm_sq / Npix  # cell-averaged
    bf_fixed = bf_fixed / np.sqrt(np.maximum(norm_sq_per_pixel, 1e-30))

    # Verify
    check = np.sum(np.abs(bf_fixed[:, :, 0])**2, axis=(-3, -2, -1)) / Npix
    print(f"    Verification: ⟨u_0|u_0⟩_Ω = {check.mean():.6f} ± {check.std():.2e}")

    # --- Fix 2: Abelian (per-band) scalar gauge alignment ---
    print("\n  Fix 2: Abelian (per-band) 2D gauge fix...")
    print("    Algorithm: seed row 0 along s2, then fix each column along s1")
    print("    Note: Non-Abelian SVD gauge FAILS because MPB fields are NOT")
    print("    orthogonal under flat inner product (they use ε-weighted ortho).")
    print("    Abelian gauge avoids cross-band mixing → safe.\n")

    bf_fixed, gauge_diag = apply_abelian_gauge_2d(bf_fixed)
    print("    Gauge fixing complete.")

    after = compute_diagnostics(bf_fixed, label="FIXED")

    np.savez_compressed(
        f'{FINDINGS}/F06_after_data.npz',
        norm_sq_flat=after['norm_sq_flat'],
        norm_sq_avg=after['norm_sq_avg'],
        phase_s1=after['phase_s1'],
        phase_s2=after['phase_s2'],
        mag_s1=after['mag_s1'],
        mag_s2=after['mag_s2'],
        ortho=after['ortho'],
        offdiag_max=after['offdiag_max'],
        Ns1=Ns1, Ns2=Ns2, N_sub=N_SUB, Nx=Nx, Ny=Ny,
    )
    print(f"\n  Saved: {FINDINGS}/F06_after_data.npz")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  F06 DIAGNOSTIC COMPLETE — {elapsed:.1f} s")
    print("=" * 70)

    print("\n  Summary of key changes:")
    for n in range(N_SUB):
        b_phase = np.std(before['phase_s1'][:, :, n])
        a_phase = np.std(after['phase_s1'][:, :, n])
        b_norm = before['norm_sq_avg'][:, :, n].std()
        a_norm = after['norm_sq_avg'][:, :, n].std()
        print(f"    Band {n}: phase_std {b_phase:.3f} → {a_phase:.3f} rad, "
              f"norm_std {b_norm:.4f} → {a_norm:.2e}")

    b_orth = before['offdiag_max'].mean()
    a_orth = after['offdiag_max'].mean()
    print(f"    Orthogonality (mean max|<u_m|u_n>|): {b_orth:.4f} → {a_orth:.4f}")

    print(f"\n  Next: python findings/make_F06_plot.py")


if __name__ == '__main__':
    run()
