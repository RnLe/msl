#!/usr/bin/env python3
"""
R06 Compute: Disorder Robustness
==================================
Monte Carlo study of how eigenvalues and localization respond to disorder.

Part A — On-site potential noise:
    Add δV(R) ~ N(0, σ²_V) to Lambda diagonal entries and re-solve.
    σ_V = fraction × ΔV (fractional depth of the potential landscape).

Part B — Geometric noise (registry perturbation):
    Perturb the local registry by a random displacement
    δs(R) ~ N(0, σ²_s), effectively representing fabrication imperfections.
    This is implemented as a smooth random perturbation of the Lambda field.

For each noise level and realization, we compute:
    - Eigenvalue shift σ_ω = std(ω - ω_clean)
    - IPR change
    - Band mixing change

Output: R06_data.json, R06_disorder.npz
"""

import sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pathlib import Path

OUTDIR   = Path(__file__).resolve().parent
BASE_RUN = OUTDIR.parent / "runsV3" / "phase0_mpb_v3_20260206_152443"
CAND     = BASE_RUN / "candidate_0000"

sys.path.insert(0, str(OUTDIR.parent / "phasesV3"))
from phase3_mpb_v3 import (
    assemble_multiband_hamiltonian,
    solve_multiband_envelope,
)

import h5py

# ── Configuration ─────────────────────────────────────────────────────────
N_REALIZATIONS = 50
NOISE_LEVELS_ONSITE = [0.001, 0.003, 0.01, 0.03, 0.1]   # fraction of V depth
NOISE_LEVELS_GEOM   = [0.001, 0.003, 0.01, 0.03]          # registry noise σ_s (frac of L_m)
N_MODES = 20
M_INV_MAX = 20.0


def build_smooth_noise(Ns, correlation_length=5, rng=None):
    """Generate spatially correlated noise via Gaussian convolution."""
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.standard_normal((Ns, Ns))
    # Smooth via FFT convolution with Gaussian kernel
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(raw, sigma=correlation_length, mode='wrap')
    # Normalize to unit std
    smoothed /= (smoothed.std() + 1e-30)
    return smoothed


def main():
    print("="*70)
    print("R06 Compute: Disorder Robustness")
    print("="*70)

    # ── Load clean Phase 2 data ───────────────────────────────────────────
    p2_file = CAND / "phase2_multiband_data.h5"
    with h5py.File(p2_file, 'r') as hf:
        Lambda_clean  = hf['Lambda'][:]
        v_drift       = hf['v_drift'][:]
        M_inv         = hf['M_inv'][:]
        A_berry       = hf['A_berry'][:]
        Phi_BH        = hf['Phi_BH'][:]
        eta           = float(hf.attrs['eta'])
        Ns1           = int(hf.attrs['Ns1'])
        Ns2           = int(hf.attrs.get('Ns2', Ns1))
        B_moire       = np.array(hf.attrs['B_moire'])

    N_bands = Lambda_clean.shape[2]

    p0_file = CAND / "phase0_meta.json"
    with open(p0_file) as f:
        p0 = json.load(f)
    L_moire = p0['L_moire']
    theta_deg = p0['theta_deg']

    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2

    print(f"Crystal: θ={theta_deg}°, η={eta:.5f}, L_m={L_moire:.2f}a")
    print(f"Grid: {Ns1}×{Ns2}, {N_bands} bands")
    print(f"dR = {dR1:.4f}a")

    # ── Clean reference solve ─────────────────────────────────────────────
    print("\nSolving clean Hamiltonian...")
    t0 = time.time()
    H_clean = assemble_multiband_hamiltonian(
        Lambda_clean, v_drift, M_inv, A_berry, Phi_BH,
        eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
        M_inv_max_trace=M_INV_MAX
    )
    evals_clean, evecs_clean = solve_multiband_envelope(H_clean, N_MODES)
    t_clean = time.time() - t0
    print(f"  Clean eigenvalues: [{evals_clean[0]:.8e}, ..., {evals_clean[-1]:.8e}]")
    print(f"  Clean solve time: {t_clean:.1f}s")

    # Compute clean IPR and band composition
    N_total = Ns1 * Ns2 * N_bands
    ipr_clean = np.zeros(N_MODES)
    dom_weight_clean = np.zeros(N_MODES)
    for m in range(N_MODES):
        psi = evecs_clean[:, m].reshape(Ns1, Ns2, N_bands)
        dens = np.sum(np.abs(psi)**2, axis=2)  # (Ns1, Ns2)
        dens_norm = dens / (dens.sum() + 1e-30)
        ipr_clean[m] = np.sum(dens_norm**2)
        band_weights = np.array([np.sum(np.abs(psi[:,:,n])**2) for n in range(N_bands)])
        band_weights /= (band_weights.sum() + 1e-30)
        dom_weight_clean[m] = band_weights.max()

    # ── Potential depth for noise calibration ─────────────────────────────
    V_depths = np.zeros(N_bands)
    for n in range(N_bands):
        V = Lambda_clean[:, :, n, n].real
        V_depths[n] = V.max() - V.min()
    V_depth_mean = V_depths.mean()
    print(f"  V depths: {V_depths}")
    print(f"  Mean V depth: {V_depth_mean:.6f}")

    rng = np.random.default_rng(42)

    # ══════════════════════════════════════════════════════════════════════
    # Part A: On-site potential noise
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Part A: On-site potential noise")
    print("="*50)

    n_noise_A = len(NOISE_LEVELS_ONSITE)
    evals_shift_A = np.zeros((n_noise_A, N_REALIZATIONS, N_MODES))
    ipr_disorder_A = np.zeros((n_noise_A, N_REALIZATIONS, N_MODES))
    dom_weight_disorder_A = np.zeros((n_noise_A, N_REALIZATIONS, N_MODES))

    for ni, frac in enumerate(NOISE_LEVELS_ONSITE):
        sigma_V = frac * V_depth_mean
        print(f"\n  Noise level {ni+1}/{n_noise_A}: σ/ΔV = {frac}, σ_V = {sigma_V:.6e}")
        t0 = time.time()

        for r in range(N_REALIZATIONS):
            # Add uncorrelated noise to diagonal Lambda entries
            Lambda_noisy = Lambda_clean.copy()
            for n in range(N_bands):
                noise = rng.normal(0, sigma_V, size=(Ns1, Ns2))
                Lambda_noisy[:, :, n, n] += noise

            H_noisy = assemble_multiband_hamiltonian(
                Lambda_noisy, v_drift, M_inv, A_berry, Phi_BH,
                eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
                M_inv_max_trace=M_INV_MAX
            )
            evals_r, evecs_r = solve_multiband_envelope(H_noisy, N_MODES)

            evals_shift_A[ni, r, :] = evals_r - evals_clean

            for m in range(N_MODES):
                psi = evecs_r[:, m].reshape(Ns1, Ns2, N_bands)
                dens = np.sum(np.abs(psi)**2, axis=2)
                dens_norm = dens / (dens.sum() + 1e-30)
                ipr_disorder_A[ni, r, m] = np.sum(dens_norm**2)
                bw = np.array([np.sum(np.abs(psi[:,:,n])**2) for n in range(N_bands)])
                bw /= (bw.sum() + 1e-30)
                dom_weight_disorder_A[ni, r, m] = bw.max()

        dt = time.time() - t0
        sigma_omega = np.std(evals_shift_A[ni, :, 0])
        print(f"    σ_ω(mode 0) = {sigma_omega:.4e}, time = {dt:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Part B: Geometric (spatially correlated) noise
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*50)
    print("Part B: Geometric noise (correlated)")
    print("="*50)

    n_noise_B = len(NOISE_LEVELS_GEOM)
    evals_shift_B = np.zeros((n_noise_B, N_REALIZATIONS, N_MODES))
    ipr_disorder_B = np.zeros((n_noise_B, N_REALIZATIONS, N_MODES))

    # Correlated noise: smooth random field modulating the potential
    # This simulates registry perturbations: the potential landscape
    # is shifted/deformed, approximated at first order as
    # δV ≈ ∇V · δs, where δs is the correlated registry noise.
    # We approximate this by gradient-weighted correlated noise.

    # Pre-compute gradient of Lambda diagonal
    grad_V = np.zeros((Ns1, Ns2, N_bands, 2))
    for n in range(N_bands):
        V = Lambda_clean[:, :, n, n].real
        # Periodic gradient
        grad_V[:, :, n, 0] = np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)
        grad_V[:, :, n, 1] = np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)
    grad_V /= (2.0 * dR1)  # central difference

    for ni, sigma_s in enumerate(NOISE_LEVELS_GEOM):
        sigma_phys = sigma_s * L_moire  # physical displacement noise
        print(f"\n  Noise level {ni+1}/{n_noise_B}: σ_s/L_m = {sigma_s}, σ_phys = {sigma_phys:.4f}a")
        # Correlation length = 5 grid points ≈ 5*dR
        corr_len = max(3, int(Ns1 * 0.04))
        t0 = time.time()

        for r in range(N_REALIZATIONS):
            # Generate 2 correlated displacement fields
            ds1 = build_smooth_noise(Ns1, corr_len, rng) * sigma_phys
            ds2 = build_smooth_noise(Ns2, corr_len, rng) * sigma_phys

            # δV_n(R) = ∇V_n · δs
            Lambda_noisy = Lambda_clean.copy()
            for n in range(N_bands):
                dV = grad_V[:, :, n, 0] * ds1 + grad_V[:, :, n, 1] * ds2
                Lambda_noisy[:, :, n, n] += dV

            H_noisy = assemble_multiband_hamiltonian(
                Lambda_noisy, v_drift, M_inv, A_berry, Phi_BH,
                eta, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
                M_inv_max_trace=M_INV_MAX
            )
            evals_r, evecs_r = solve_multiband_envelope(H_noisy, N_MODES)
            evals_shift_B[ni, r, :] = evals_r - evals_clean

            for m in range(N_MODES):
                psi = evecs_r[:, m].reshape(Ns1, Ns2, N_bands)
                dens = np.sum(np.abs(psi)**2, axis=2)
                dens_norm = dens / (dens.sum() + 1e-30)
                ipr_disorder_B[ni, r, m] = np.sum(dens_norm**2)

        dt = time.time() - t0
        sigma_omega_B = np.std(evals_shift_B[ni, :, 0])
        print(f"    σ_ω(mode 0) = {sigma_omega_B:.4e}, time = {dt:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(OUTDIR / "R06_disorder.npz",
             evals_clean=evals_clean,
             ipr_clean=ipr_clean,
             dom_weight_clean=dom_weight_clean,
             # Part A
             noise_levels_onsite=np.array(NOISE_LEVELS_ONSITE),
             evals_shift_A=evals_shift_A,
             ipr_disorder_A=ipr_disorder_A,
             dom_weight_disorder_A=dom_weight_disorder_A,
             # Part B
             noise_levels_geom=np.array(NOISE_LEVELS_GEOM),
             evals_shift_B=evals_shift_B,
             ipr_disorder_B=ipr_disorder_B)
    print(f"\nSaved {OUTDIR / 'R06_disorder.npz'}")

    # ── Summary statistics ────────────────────────────────────────────────
    summary = {
        'theta_deg': theta_deg,
        'eta': eta,
        'L_moire': L_moire,
        'N_modes': N_MODES,
        'N_realizations': N_REALIZATIONS,
        'V_depth_mean': float(V_depth_mean),
        'evals_clean': evals_clean.tolist(),
        'ipr_clean': ipr_clean.tolist(),
        'part_A': {
            'noise_levels': NOISE_LEVELS_ONSITE,
            'sigma_omega_mode0': [float(np.std(evals_shift_A[ni, :, 0]))
                                   for ni in range(n_noise_A)],
            'sigma_omega_mean': [float(np.mean([np.std(evals_shift_A[ni, :, m])
                                                 for m in range(N_MODES)]))
                                  for ni in range(n_noise_A)],
            'ipr_change_mean': [float(np.mean(ipr_disorder_A[ni, :, 0]) - ipr_clean[0])
                                 for ni in range(n_noise_A)],
            'dom_weight_change': [float(np.mean(dom_weight_disorder_A[ni, :, 0]) - dom_weight_clean[0])
                                   for ni in range(n_noise_A)],
        },
        'part_B': {
            'noise_levels': NOISE_LEVELS_GEOM,
            'sigma_omega_mode0': [float(np.std(evals_shift_B[ni, :, 0]))
                                   for ni in range(n_noise_B)],
            'sigma_omega_mean': [float(np.mean([np.std(evals_shift_B[ni, :, m])
                                                 for m in range(N_MODES)]))
                                  for ni in range(n_noise_B)],
        },
    }
    with open(OUTDIR / "R06_data.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {OUTDIR / 'R06_data.json'}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("Part A summary:")
    for ni, frac in enumerate(NOISE_LEVELS_ONSITE):
        s = summary['part_A']
        print(f"  σ/ΔV={frac:7.3f}: σ_ω(m0)={s['sigma_omega_mode0'][ni]:.4e}, "
              f"σ_ω(mean)={s['sigma_omega_mean'][ni]:.4e}, "
              f"ΔIPR(m0)={s['ipr_change_mean'][ni]:+.4e}")

    print("\nPart B summary:")
    for ni, sigma_s in enumerate(NOISE_LEVELS_GEOM):
        s = summary['part_B']
        print(f"  σ_s/L_m={sigma_s:7.3f}: σ_ω(m0)={s['sigma_omega_mode0'][ni]:.4e}, "
              f"σ_ω(mean)={s['sigma_omega_mean'][ni]:.4e}")
    print("="*70)

    print("\nR06 compute complete.")


if __name__ == '__main__':
    main()
