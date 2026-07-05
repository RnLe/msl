#!/usr/bin/env python3
"""
R03: Envelope Mode Gallery & Spatial Structure
================================================
Load Phase 3 envelope spinors for multiple twist angles.
Compute total energy density, per-band decomposition, localization metrics.

Output: R03_data.json + R03_data.npz
"""

import numpy as np
import h5py
import json
from pathlib import Path
from scipy.optimize import curve_fit

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
SWEEP_DIR = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "eta_sweep_20260206_173808"
CAND_DIR  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
OUTDIR = Path(__file__).resolve().parent

THETAS = ['1.100', '3.000', '8.000']
N_MODES_SHOW = 20


def gaussian_2d(r, A, xi):
    """Gaussian fit model: A * exp(-r²/(2ξ²))."""
    return A * np.exp(-r**2 / (2 * xi**2))


def compute_localization_length(density, R_grid, centroid):
    """Fit a Gaussian to the total density to extract localization length ξ."""
    Ns1, Ns2 = density.shape
    R = R_grid  # (Ns1, Ns2, 2)

    # Distance from centroid
    dR = R - centroid[np.newaxis, np.newaxis, :]
    r = np.sqrt(dR[:, :, 0]**2 + dR[:, :, 1]**2)

    r_flat = r.flatten()
    d_flat = density.flatten()

    # Sort by distance and bin
    idx = np.argsort(r_flat)
    r_sorted = r_flat[idx]
    d_sorted = d_flat[idx]

    # Radial average (20 bins)
    n_bins = 20
    r_max = r_sorted.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r_binned = []
    d_binned = []
    for i in range(n_bins):
        mask = (r_sorted >= bin_edges[i]) & (r_sorted < bin_edges[i+1])
        if np.sum(mask) > 0:
            r_binned.append(0.5 * (bin_edges[i] + bin_edges[i+1]))
            d_binned.append(np.mean(d_sorted[mask]))

    r_binned = np.array(r_binned)
    d_binned = np.array(d_binned)

    if len(r_binned) < 3:
        return float('nan')

    try:
        popt, _ = curve_fit(gaussian_2d, r_binned, d_binned,
                            p0=[d_binned.max(), r_max / 4],
                            bounds=([0, 0], [np.inf, r_max]),
                            maxfev=5000)
        return float(popt[1])
    except (RuntimeError, ValueError):
        return float('nan')


def load_phase3_data(theta_str):
    """Load Phase 3 data for given theta."""
    if theta_str == '1.100':
        cdir = CAND_DIR
    else:
        cdir = SWEEP_DIR / f"theta_{theta_str}" / "candidate_0000"

    p3file = cdir / "phase3_multiband_modes.h5"
    with h5py.File(p3file, 'r') as hf:
        F_spinor = hf['F_spinor'][:]       # (n_modes, Ns1, Ns2, N_sub)
        eigenvalues = hf['eigenvalues'][:]  # (n_modes,)
        R_grid = hf['R_grid'][:]           # (Ns1, Ns2, 2)
        s_grid = hf['s_grid'][:]
        eta = float(hf.attrs['eta'])
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_sub = int(hf.attrs['N_subspace'])
        L_moire = float(np.linalg.norm(hf.attrs['B_moire'][:, 0]))
        theta_deg = float(hf.attrs['theta_deg'])
        omega_ref = float(hf.attrs['omega_ref'])

    return {
        'F_spinor': F_spinor,
        'eigenvalues': eigenvalues,
        'R_grid': R_grid,
        's_grid': s_grid,
        'eta': eta, 'Ns1': Ns1, 'Ns2': Ns2, 'N_sub': N_sub,
        'L_moire': L_moire, 'theta_deg': theta_deg, 'omega_ref': omega_ref,
    }


def main():
    print("="*70)
    print("R03: Envelope Mode Gallery & Spatial Structure")
    print("="*70)

    all_results = {}
    all_densities = {}

    for theta_str in THETAS:
        print(f"\n{'─'*60}")
        print(f"θ = {theta_str}°")
        print(f"{'─'*60}")

        d = load_phase3_data(theta_str)
        F = d['F_spinor']  # (n_modes, Ns1, Ns2, N_sub)
        evals = d['eigenvalues']
        R_grid = d['R_grid']
        Ns1, Ns2, N_sub = d['Ns1'], d['Ns2'], d['N_sub']
        L_moire = d['L_moire']

        n_modes = min(N_MODES_SHOW, len(evals))
        print(f"  Grid: {Ns1}×{Ns2}×{N_sub}, {n_modes} modes, L_moire={L_moire:.1f}a")

        # Total density W(R) = Σ_n |F_n(R)|²
        W = np.sum(np.abs(F[:n_modes])**2, axis=-1)  # (n_modes, Ns1, Ns2)

        mode_data = []
        for mi in range(n_modes):
            Wm = W[mi]  # (Ns1, Ns2)

            # Per-band weights
            band_weights = []
            for nb in range(N_sub):
                bw = float(np.sum(np.abs(F[mi, :, :, nb])**2))
                band_weights.append(bw)
            total_w = sum(band_weights)
            if total_w > 0:
                band_weights = [w / total_w for w in band_weights]
            dominant_band = int(np.argmax(band_weights))
            dominant_weight = float(max(band_weights))

            # Centroid
            norm = np.sum(Wm)
            if norm > 1e-20:
                cx = float(np.sum(R_grid[:, :, 0] * Wm) / norm)
                cy = float(np.sum(R_grid[:, :, 1] * Wm) / norm)
            else:
                cx, cy = 0.0, 0.0

            # Spread (RMS radius)
            dR = R_grid - np.array([cx, cy])
            r2 = dR[:, :, 0]**2 + dR[:, :, 1]**2
            spread = float(np.sqrt(np.sum(r2 * Wm) / max(norm, 1e-20)))

            # IPR (inverse participation ratio)
            ipr = float(np.sum(Wm**2) / max(norm**2, 1e-40))

            # Participation number = 1/IPR * N_sites
            PN = 1.0 / max(ipr, 1e-20) if ipr > 0 else 0.0

            # Localization length (Gaussian fit)
            xi = compute_localization_length(Wm, R_grid, np.array([cx, cy]))

            mode_data.append({
                'mode_index': mi,
                'eigenvalue': float(evals[mi]),
                'omega': float(evals[mi] + d['omega_ref']),
                'band_weights': band_weights,
                'dominant_band': dominant_band,
                'dominant_weight': dominant_weight,
                'centroid': [cx, cy],
                'spread': spread,
                'spread_over_Lm': spread / L_moire,
                'ipr': ipr,
                'participation_number': PN,
                'localization_length': xi,
                'xi_over_Lm': xi / L_moire if not np.isnan(xi) else None,
            })

            if mi < 5:
                print(f"  Mode {mi}: λ={evals[mi]:.6f}, dom=B{dominant_band}"
                      f"({dominant_weight:.0%}), σ/L={spread/L_moire:.3f}, PN={PN:.0f}")

        all_results[theta_str] = {
            'theta_deg': float(theta_str),
            'eta': d['eta'],
            'L_moire': L_moire,
            'omega_ref': d['omega_ref'],
            'Ns': Ns1,
            'N_sub': N_sub,
            'modes': mode_data,
        }

        # Store densities for plotting
        all_densities[f"W_{theta_str}"] = W[:n_modes]  # (n_modes, Ns1, Ns2)
        all_densities[f"s_grid_{theta_str}"] = d['s_grid']

    # ── Save ──────────────────────────────────────────────────────────────
    outfile_npz = OUTDIR / "R03_data.npz"
    np.savez_compressed(outfile_npz, **all_densities)
    print(f"\nSaved arrays to {outfile_npz}")

    outfile_json = OUTDIR / "R03_data.json"
    with open(outfile_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved metadata to {outfile_json}")
    print("\nR03 compute complete.")


if __name__ == '__main__':
    main()
