#!/usr/bin/env python3
"""
R04: Full Field Reconstruction & Mode Volume
==============================================
Reconstruct the full electromagnetic field from envelope + Bloch functions.
Compute mode volume V_eff, mode area A_eff, LDOS enhancement.

Requires bloch_fields in Phase 1 HDF5 (only available for candidate_0000, θ=1.1°).

Output: R04_data.json + R04_fields.npz
"""

import numpy as np
import h5py
import json
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PHASES_DIR = BASE / "phasesV3"
CAND_DIR = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
OUTDIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PHASES_DIR))
from phase4_field_reconstruction import (
    reconstruct_full_field_single_mode,
    load_phase1_bloch_fields,
    get_subspace_band_indices,
)

N_MODES_RECONSTRUCT = 10  # First 10 modes


def compute_mode_volume_2d(field_intensity, epsilon, dx, dy):
    """
    Compute 2D effective mode area A_eff.

    A_eff = [∫ ε|E|² dA]² / ∫ (ε|E|²)² dA

    For 2D photonic crystals this is the mode area (units of a²).
    The 3D mode volume V_eff ≈ A_eff × (effective out-of-plane extent).
    """
    eE2 = epsilon * field_intensity
    dA = dx * dy

    numerator = (np.sum(eE2) * dA) ** 2
    denominator = np.sum(eE2**2) * dA
    if denominator > 0:
        return numerator / denominator
    return float('inf')


def compute_mode_volume_standard(field_intensity, epsilon, dx, dy):
    """
    Standard mode volume definition:
    V_eff = ∫ ε|E|² dA / max(ε|E|²)
    """
    eE2 = epsilon * field_intensity
    dA = dx * dy
    integral = np.sum(eE2) * dA
    peak = np.max(eE2)
    if peak > 0:
        return integral / peak
    return float('inf')


def main():
    print("="*70)
    print("R04: Field Reconstruction & Mode Volume")
    print("="*70)

    # ── Check bloch_fields availability ────────────────────────────────────
    p1_file = CAND_DIR / "phase1_multiband_data.h5"
    print(f"Loading Bloch fields from {p1_file}")

    with h5py.File(p1_file, 'r') as hf:
        if 'bloch_fields' not in hf:
            print("ERROR: bloch_fields not found in Phase 1 data!")
            print("Re-run Phase 1 with export_bloch_fields: true")
            sys.exit(1)
        bloch_fields = hf['bloch_fields'][:]  # (Ns1_b, Ns2_b, N_all, Nx, Ny, 3)
        subspace_bands = hf.attrs.get('subspace_bands', None)
        all_bands = hf.attrs.get('all_bands', None)
        epsilon = hf['epsilon'][:]  # (Ns1_b, Ns2_b, Nx, Ny) if available

    print(f"  bloch_fields shape: {bloch_fields.shape}")
    print(f"  subspace_bands: {subspace_bands}")
    Ns1_b, Ns2_b, N_all, Nx, Ny, _ = bloch_fields.shape

    # Get band indices mapping
    if subspace_bands is not None and all_bands is not None:
        band_indices = get_subspace_band_indices(subspace_bands, all_bands)
    else:
        # Fallback: assume subspace_bands = [5,6,7,8,9], all_bands = 0..17
        band_indices = [5, 6, 7, 8, 9]
    print(f"  band_indices: {band_indices}")

    # ── Load Phase 3 envelope modes ────────────────────────────────────────
    p3_file = CAND_DIR / "phase3_multiband_modes.h5"
    with h5py.File(p3_file, 'r') as hf:
        F_spinor = hf['F_spinor'][:]     # (n_modes, Ns1, Ns2, N_sub)
        eigenvalues = hf['eigenvalues'][:]
        R_grid = hf['R_grid'][:]
        omega_ref = float(hf.attrs['omega_ref'])
        eta = float(hf.attrs['eta'])
        theta_deg = float(hf.attrs['theta_deg'])
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_sub = int(hf.attrs['N_subspace'])
        B_moire = hf.attrs['B_moire']

    L_moire = float(np.linalg.norm(B_moire[:, 0]))
    print(f"  Envelope grid: {Ns1}×{Ns2}×{N_sub}, {len(eigenvalues)} modes")
    print(f"  L_moire={L_moire:.2f}a, θ={theta_deg}°, η={eta:.5f}")

    n_modes = min(N_MODES_RECONSTRUCT, len(eigenvalues))

    # ── Build average epsilon for the moiré cell ───────────────────────────
    # Use the epsilon from Phase 1 (Ns1_b × Ns2_b × Nx × Ny)
    # For mode volume, we need epsilon at the full tiled resolution
    # Average epsilon per unit cell → tile across moiré cell
    eps_avg = np.mean(epsilon, axis=(0, 1))  # (Nx, Ny) average over registry
    # Tile to full grid
    eps_tiled = np.tile(eps_avg, (Ns1, Ns2))  # (Ns1*Nx, Ns2*Ny)
    print(f"  ε tiled shape: {eps_tiled.shape}, mean ε = {eps_avg.mean():.2f}")

    # Physical grid spacing on the tiled grid
    dx = L_moire / (Ns1 * Nx)  # in units of a
    dy = L_moire / (Ns2 * Ny)

    # ── Reconstruct fields and compute metrics ─────────────────────────────
    mode_results = []
    field_samples = {}  # Store a few fields for plotting

    bloch_cache = {}

    for mi in range(n_modes):
        t0 = time.time()
        print(f"\n  Mode {mi} (λ={eigenvalues[mi]:.6f})...")

        # Reconstruct full field — for TE, E = (Ex, Ey, 0)
        # We need both in-plane components
        H_full_x = reconstruct_full_field_single_mode(
            mode_idx=mi,
            F_spinor=F_spinor,
            bloch_fields=bloch_fields,
            band_indices=band_indices,
            component=0,           # Ex for TE
            include_bloch_phase=False,
            normalize_bloch=True,
            bloch_interp_cache=bloch_cache,
        )
        H_full_y = reconstruct_full_field_single_mode(
            mode_idx=mi,
            F_spinor=F_spinor,
            bloch_fields=bloch_fields,
            band_indices=band_indices,
            component=1,           # Ey for TE
            include_bloch_phase=False,
            normalize_bloch=True,
            bloch_interp_cache=bloch_cache,
        )

        # |E|² = |Ex|² + |Ey|²
        intensity = np.abs(H_full_x)**2 + np.abs(H_full_y)**2
        # Normalize
        norm = np.sum(intensity) * dx * dy
        if norm > 0:
            intensity_normed = intensity / norm
        else:
            intensity_normed = intensity

        # ε|E|²
        eE2 = eps_tiled * intensity

        # Mode area (IPR-based)
        A_eff_ipr = compute_mode_volume_2d(intensity, eps_tiled, dx, dy)

        # Mode area (standard: ∫/max)
        A_eff_std = compute_mode_volume_standard(intensity, eps_tiled, dx, dy)

        # LDOS enhancement: max(ε|E|²) / <ε|E|²>
        if np.sum(eE2) > 0:
            ldos_enhancement = float(np.max(eE2) / np.mean(eE2))
        else:
            ldos_enhancement = 0.0

        # Envelope-only density for comparison
        W_env = np.sum(np.abs(F_spinor[mi])**2, axis=-1)  # (Ns1, Ns2)
        W_env_norm = W_env / max(np.sum(W_env), 1e-20)

        # Envelope-to-full correlation
        # Downsample full field to envelope resolution
        intensity_coarse = intensity.reshape(Ns1, Nx, Ns2, Ny).mean(axis=(1, 3))
        ic_norm = intensity_coarse / max(np.sum(intensity_coarse), 1e-20)
        correlation = float(np.sum(np.sqrt(W_env_norm * ic_norm)))

        elapsed = time.time() - t0

        result = {
            'mode_index': mi,
            'eigenvalue': float(eigenvalues[mi]),
            'omega': float(eigenvalues[mi] + omega_ref),
            'A_eff_ipr': float(A_eff_ipr),
            'A_eff_std': float(A_eff_std),
            'A_eff_over_a2': float(A_eff_std),  # already in units of a²
            'A_eff_over_Lm2': float(A_eff_std / L_moire**2),
            'ldos_enhancement': ldos_enhancement,
            'envelope_correlation': correlation,
            'peak_intensity': float(np.max(intensity)),
            'reconstruction_time_s': elapsed,
        }
        mode_results.append(result)

        print(f"    A_eff(std) = {A_eff_std:.1f} a² = {A_eff_std/L_moire**2:.4f} L_m²")
        print(f"    A_eff(IPR) = {A_eff_ipr:.1f} a²")
        print(f"    LDOS enhancement = {ldos_enhancement:.1f}×")
        print(f"    Envelope correlation = {correlation:.4f}")
        print(f"    [{elapsed:.1f}s]")

        # Store fields for first 5 modes
        if mi < 5:
            # Downsample for storage (full field is huge)
            # Store every 4th pixel
            stride = max(1, Nx // 8)
            field_samples[f"field_{mi}"] = intensity[::stride, ::stride].astype(np.float32)
            field_samples[f"envelope_{mi}"] = W_env.astype(np.float32)

    # ── Save ──────────────────────────────────────────────────────────────
    outfile_json = OUTDIR / "R04_data.json"
    meta = {
        'theta_deg': theta_deg,
        'eta': eta,
        'omega_ref': omega_ref,
        'L_moire': L_moire,
        'Ns_env': Ns1,
        'Ns_bloch': Ns1_b,
        'Nx': Nx, 'Ny': Ny,
        'full_grid': [Ns1 * Nx, Ns2 * Ny],
        'dx': dx, 'dy': dy,
        'modes': mode_results,
    }
    with open(outfile_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata to {outfile_json}")

    outfile_npz = OUTDIR / "R04_fields.npz"
    np.savez_compressed(outfile_npz, **field_samples)
    print(f"Saved field samples to {outfile_npz}")

    print("\nR04 compute complete.")


if __name__ == '__main__':
    main()
