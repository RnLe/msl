#!/usr/bin/env python3
"""
F04 — Full Validation Suite: Options A, B, and C

This script performs all three validation tests from the original plan:

Option A: Eigenvalue convergence with N_bands (N=1 vs N=3)
  - At each η, solve with N_sub=1 (single-band, per band) and N_sub=3 (coupled).
  - |λ(N=3) − λ(N=1)| should decrease as η → 0.

Option B: Bandwidth / potential-depth ratio scaling
  - BW₂₀ vs η power law for each band.
  - Already computed in F03 resweep; we integrate those results here.

Option C: Per-tile Maxwell residual for modes with band mixing
  - Reconstruct the full field E(r) = Σ_n F_n(R) u_n(r;R) e^{ik₀·r}
  - Compute Rayleigh quotient R_q per tile: ∫|curl_k E|² / ∫ε|E|²
  - Compare to ω²_n(R) from Phase 1.
  - Track residual vs η.

Uses the CORRECTED Hamiltonian (Hermitized kinetic + M_inv regularization).
"""

import numpy as np
import h5py
import json
import sys
import time
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808'
FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'

thetas = ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']
n_modes = 20
M_INV_MAX = 20  # regularization cap
Ns_solve = 128  # grid resolution for solve

# ============================================================================
# OPTION A: N_bands convergence (N=1 vs N=3)
# ============================================================================
def run_option_A():
    """Compare single-band (N=1) vs 3-band coupled (N=3) eigenvalues."""
    print("\n" + "="*70)
    print("OPTION A: Eigenvalue convergence with N_bands")
    print("="*70)

    results = []

    for theta_str in thetas:
        cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'

        with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
            Lambda = hf['Lambda'][:]
            M_inv = hf['M_inv'][:]
            A_berry = hf['A_berry'][:]
            Phi_BH = hf['Phi_BH'][:]
            v_drift = hf['v_drift'][:]
            eta = float(hf.attrs['eta'])
            Ns1 = int(hf.attrs['Ns1'])
            B_moire = hf.attrs['B_moire']
            omega_ref = float(hf.attrs['omega_ref'])
            if isinstance(B_moire, str):
                B_moire = np.array(eval(B_moire.replace('\n', ',')))

        L_moire = np.linalg.norm(B_moire[0])
        dR = L_moire / Ns1
        N_bands_full = Lambda.shape[2]  # = 3

        print(f'\n{"="*60}')
        print(f'theta={theta_str} deg, eta={eta:.5f}, L={L_moire:.2f}')
        print(f'{"="*60}')

        entry = {
            'theta_deg': float(theta_str),
            'eta': eta,
            'omega_ref': omega_ref,
            'L_moire': L_moire,
            'bands': {}
        }

        # === N=1: single-band solve for each band ===
        t0 = time.time()
        for t in range(N_bands_full):
            Lb = Lambda[:, :, t:t+1, t:t+1]
            Mb = M_inv[:, :, t:t+1, t:t+1, :, :]
            vb = v_drift[:, :, t:t+1, t:t+1, :]
            Ab = A_berry[:, :, t:t+1, t:t+1, :]
            Pb = Phi_BH[:, :, t:t+1, t:t+1]

            V_min = float(Lb.min())
            V_max = float(Lb.max())
            M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
            mean_mass = float(np.mean(M_trace))
            band_type = 'hole' if mean_mass < 0 else 'electron'
            sigma = V_max if mean_mass < 0 else V_min

            H1 = p3.assemble_multiband_hamiltonian(
                Lb, vb, Mb, Ab, Pb, eta, Ns1, Ns1, 1, dR, dR, B_moire,
                include_drift=True, include_kinetic=True, include_born_huang=True,
                order=4, M_inv_max_trace=M_INV_MAX
            )
            evals_1, evecs_1 = p3.solve_multiband_envelope(H1, n_modes, sigma=sigma)
            evals_1 = np.sort(np.real(evals_1))

            entry['bands'][t] = {
                'type': band_type,
                'V_min': V_min,
                'V_max': V_max,
                'mean_mass_trace': mean_mass,
                'N1_eigenvalues': evals_1.tolist(),
            }

        # === N=3: coupled 3-band solve ===
        H3 = p3.assemble_multiband_hamiltonian(
            Lambda, v_drift, M_inv, A_berry, Phi_BH, eta, Ns1, Ns1, N_bands_full, dR, dR, B_moire,
            include_drift=True, include_kinetic=True, include_born_huang=True,
            order=4, M_inv_max_trace=M_INV_MAX
        )

        # For N=3, solve near each band's spectral region
        N_s = Ns1 * Ns1
        for t in range(N_bands_full):
            bd = entry['bands'][t]
            sigma = bd['V_max'] if bd['type'] == 'hole' else bd['V_min']
            evals_3, evecs_3 = p3.solve_multiband_envelope(H3, n_modes, sigma=sigma)
            evals_3_sorted = np.sort(np.real(evals_3))

            # Identify which eigenvalues are dominated by band t
            band_weights = np.zeros((len(evals_3), N_bands_full))
            for m in range(len(evals_3)):
                v = evecs_3[:, m].reshape(N_s, N_bands_full)
                for b in range(N_bands_full):
                    band_weights[m, b] = np.sum(np.abs(v[:, b])**2)
                band_weights[m] /= band_weights[m].sum()

            # Filter eigenvalues dominated by band t (> 50% weight)
            dom_mask = band_weights[:, t] > 0.5
            evals_t = np.sort(np.real(evals_3[dom_mask]))

            # Also record the inter-band mixing: max weight of OTHER bands
            mixing_weights = []
            for m in range(len(evals_3)):
                if dom_mask[m]:
                    other_weight = 1.0 - band_weights[m, t]
                    mixing_weights.append(float(other_weight))

            n_dom = len(evals_t)
            n_match = min(n_dom, len(bd['N1_eigenvalues']))

            bd['N3_eigenvalues'] = evals_t.tolist()
            bd['N3_band_weights'] = band_weights.tolist()
            bd['N3_mixing_weights'] = mixing_weights

            if n_match > 0:
                e1 = np.array(bd['N1_eigenvalues'][:n_match])
                e3 = evals_t[:n_match]
                diff = np.abs(e3 - e1)
                bd['delta_N1_N3'] = {
                    'max_abs': float(np.max(diff)),
                    'mean_abs': float(np.mean(diff)),
                    'rel_E0': float(diff[0] / max(abs(e1[0]), 1e-10)),
                    'n_matched': n_match,
                    'max_mixing': float(max(mixing_weights)) if mixing_weights else 0.0,
                    'mean_mixing': float(np.mean(mixing_weights)) if mixing_weights else 0.0,
                }
                print(f'  Band {t} ({bd["type"]}): N1 E0={e1[0]:.6f}, '
                      f'N3 E0={evals_t[0]:.6f}, '
                      f'|ΔE0|={diff[0]:.2e}, '
                      f'rel={bd["delta_N1_N3"]["rel_E0"]:.2e}, '
                      f'max_mix={bd["delta_N1_N3"]["max_mixing"]:.4f}')
            else:
                bd['delta_N1_N3'] = None
                print(f'  Band {t} ({bd["type"]}): no N=3 modes dominated by this band')

        dt = time.time() - t0
        print(f'  Time: {dt:.1f}s')
        results.append(entry)

    return results


# ============================================================================
# OPTION B: Bandwidth scaling (integrate F03 corrected sweep results)
# ============================================================================
def run_option_B():
    """Load and process the F03 corrected sweep results for BW scaling."""
    print("\n" + "="*70)
    print("OPTION B: Bandwidth / potential-depth ratio scaling")
    print("="*70)

    with open(f'{FINDINGS}/sweep_results_F03_corrected.json') as f:
        data = json.load(f)

    results = {}
    N_bands = len(data[0]['per_band'])
    for t in range(N_bands):
        M_trace = data[0]['per_band'][t]['mean_mass_trace']
        band_type = 'hole' if M_trace < 0 else 'electron'
        name = f'Band {t} ({band_type})'
        etas = np.array([d['eta'] for d in data])
        thetas_deg = np.array([d['theta_deg'] for d in data])
        bw20 = np.array([d['per_band'][t]['bandwidth_20'] for d in data])
        delta_rel = np.array([d['per_band'][t]['delta_shallow_rel'] for d in data])
        V_range = np.array([d['per_band'][t]['V_max'] - d['per_band'][t]['V_min'] for d in data])
        E0 = np.array([d['per_band'][t]['eigenvalues'][0] for d in data])

        # Power-law fit for valid regime (θ ≤ 3°)
        mask = thetas_deg <= 3.0
        valid = mask & (bw20 > 0)
        if valid.sum() >= 2:
            p = np.polyfit(np.log(etas[valid]), np.log(bw20[valid]), 1)
            alpha = p[0]
            A = np.exp(p[1])
            residuals = np.log(bw20[valid]) - (p[0]*np.log(etas[valid]) + p[1])
            r2 = 1 - np.var(residuals)/np.var(np.log(bw20[valid]))
        else:
            alpha, A, r2 = np.nan, np.nan, np.nan

        results[t] = {
            'name': name,
            'etas': etas.tolist(),
            'bw20': bw20.tolist(),
            'delta_rel': delta_rel.tolist(),
            'V_range': V_range.tolist(),
            'E0': E0.tolist(),
            'power_law_alpha': float(alpha),
            'power_law_A': float(A),
            'power_law_R2': float(r2),
        }

        print(f'\n{name}:')
        print(f'  BW ∝ η^{alpha:.3f} (R² = {r2:.4f}, fit θ ≤ 3°)')
        print(f'  BW range: {bw20[0]:.6f} → {bw20[-1]:.6f}')
        print(f'  δ/V range: {delta_rel[0]:.4f} → {delta_rel[-1]:.4f}')

    return results


# ============================================================================
# OPTION C: Per-tile Maxwell residual vs η
# ============================================================================
def run_option_C():
    """
    Compute per-tile Maxwell residual at each angle.

    For each θ, we:
    1. Re-solve Phase 3 with corrected H (N=3 coupled) to get F_spinor
    2. Reconstruct E(r) = Σ_n F_n(R) u_n(r;R) per tile
    3. Compute Rayleigh quotient per tile: R_q = ∫|curl_k E|² / ∫ε|E|²
    4. Compare to ω²_n(R) from Phase 1 bands
    5. Report |F|²-weighted residual
    """
    print("\n" + "="*70)
    print("OPTION C: Per-tile Maxwell residual vs η")
    print("="*70)

    results = []

    for theta_str in thetas:
        cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'

        # Load Phase 0 meta for k0
        with open(f'{cdir}/phase0_meta.json') as f:
            meta = json.load(f)
        k0_x = meta.get('k0_x', 0.5)
        k0_y = meta.get('k0_y', 0.0)
        a = meta.get('a', 1.0)
        # For square lattice: k_phys = 2π * (k0_x, k0_y) / a
        k0_phys = 2 * np.pi * np.array([k0_x, k0_y]) / a

        # Load Phase 1 data (Bloch fields + band frequencies)
        with h5py.File(f'{cdir}/phase1_multiband_data.h5', 'r') as hf:
            bloch_fields = hf['bloch_fields'][:]    # (Ns_b, Ns_b, 10, 64, 64, 3)
            omega_bands = hf['omega'][:]            # (Ns_env, Ns_env, 3)

        # Load Phase 2 data
        with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
            Lambda = hf['Lambda'][:]
            M_inv = hf['M_inv'][:]
            A_berry = hf['A_berry'][:]
            Phi_BH = hf['Phi_BH'][:]
            v_drift = hf['v_drift'][:]
            eta = float(hf.attrs['eta'])
            Ns1 = int(hf.attrs['Ns1'])
            B_moire = hf.attrs['B_moire']
            omega_ref = float(hf.attrs['omega_ref'])
            if isinstance(B_moire, str):
                B_moire = np.array(eval(B_moire.replace('\n', ',')))

        L_moire = np.linalg.norm(B_moire[0])
        dR = L_moire / Ns1
        N_bands_full = Lambda.shape[2]  # = 3
        Ns_b = bloch_fields.shape[0]    # = 64

        print(f'\n{"="*60}')
        print(f'theta={theta_str} deg, eta={eta:.5f}')
        print(f'  Bloch registry: {Ns_b}×{Ns_b}, micro: {bloch_fields.shape[3]}×{bloch_fields.shape[4]}')
        print(f'  k0_phys = ({k0_phys[0]:.4f}, {k0_phys[1]:.4f})')
        print(f'{"="*60}')

        t0 = time.time()

        # Re-solve Phase 3 with corrected H (N=3 coupled)
        H3 = p3.assemble_multiband_hamiltonian(
            Lambda, v_drift, M_inv, A_berry, Phi_BH, eta, Ns1, Ns1, N_bands_full, dR, dR, B_moire,
            include_drift=True, include_kinetic=True, include_born_huang=True,
            order=4, M_inv_max_trace=M_INV_MAX
        )

        entry = {
            'theta_deg': float(theta_str),
            'eta': eta,
            'omega_ref': omega_ref,
            'L_moire': L_moire,
            'per_band_residuals': [],
        }

        # Solve near each band's spectral region
        N_s = Ns1 * Ns1
        for t in range(N_bands_full):
            V_min = float(Lambda[:,:,t,t].min())
            V_max = float(Lambda[:,:,t,t].max())
            M_trace = M_inv[:, :, t, t, 0, 0] + M_inv[:, :, t, t, 1, 1]
            mean_mass = float(np.mean(M_trace))
            band_type = 'hole' if mean_mass < 0 else 'electron'
            sigma = V_max if band_type == 'hole' else V_min

            evals_3, evecs_3 = p3.solve_multiband_envelope(H3, n_modes, sigma=sigma)

            # Find modes dominated by band t
            band_weights = np.zeros((len(evals_3), N_bands_full))
            for m in range(len(evals_3)):
                v = evecs_3[:, m].reshape(N_s, N_bands_full)
                for b in range(N_bands_full):
                    band_weights[m, b] = np.sum(np.abs(v[:, b])**2)
                band_weights[m] /= band_weights[m].sum()

            dom_mask = band_weights[:, t] > 0.5
            dom_indices = np.where(dom_mask)[0]

            if len(dom_indices) == 0:
                print(f'  Band {t} ({band_type}): no dominated modes, skipping residual')
                entry['per_band_residuals'].append({
                    'band_index': t,
                    'type': band_type,
                    'residual': None,
                    'reason': 'no dominated modes'
                })
                continue

            # Take the first dominated mode (ground state of this band)
            mode_idx_global = dom_indices[0]
            mode_eval = float(np.real(evals_3[mode_idx_global]))
            mode_mixing = float(1.0 - band_weights[mode_idx_global, t])

            # Build F_spinor for this mode: reshape eigenvector
            F_mode = evecs_3[:, mode_idx_global].reshape(Ns1, Ns1, N_bands_full)

            # Compute per-tile Rayleigh quotient
            # Within-tile FD (periodic boundary on unit cell)
            Nx_micro = bloch_fields.shape[3]
            d = 1.0 / Nx_micro
            subspace_bands = list(range(N_bands_full))

            # Prepare envelope on Bloch registry grid
            from scipy.ndimage import zoom
            if Ns1 != Ns_b:
                F_reg = np.zeros((Ns_b, Ns_b, N_bands_full), dtype=complex)
                for b in range(N_bands_full):
                    F_reg[:,:,b] = zoom(F_mode[:,:,b].real, Ns_b/Ns1, order=1, mode='wrap') + \
                                   1j*zoom(F_mode[:,:,b].imag, Ns_b/Ns1, order=1, mode='wrap')
            else:
                F_reg = F_mode

            # Compute |F|² per tile (total over bands)
            F_sq = np.sum(np.abs(F_reg)**2, axis=-1)  # (Ns_b, Ns_b)

            # Per-tile Rayleigh quotient
            Rq_tiles = np.zeros((Ns_b, Ns_b))
            omega2_tiles = np.zeros((Ns_b, Ns_b))

            # Prepare omega_bands on registry grid
            if omega_bands.shape[0] != Ns_b:
                omega_bands_reg = np.zeros((Ns_b, Ns_b, omega_bands.shape[2]))
                for n in range(omega_bands.shape[2]):
                    omega_bands_reg[:,:,n] = zoom(omega_bands[:,:,n], Ns_b/omega_bands.shape[0], order=1, mode='wrap')
            else:
                omega_bands_reg = omega_bands

            for ti in range(Ns_b):
                for tj in range(Ns_b):
                    # Build E-field at this tile: E = Σ_n F_n * u_n
                    Ex = np.zeros((Nx_micro, Nx_micro), dtype=complex)
                    Ey = np.zeros((Nx_micro, Nx_micro), dtype=complex)
                    for n in range(N_bands_full):
                        u_x = bloch_fields[ti, tj, subspace_bands[n], :, :, 0]
                        u_y = bloch_fields[ti, tj, subspace_bands[n], :, :, 1]
                        Ex += F_reg[ti, tj, n] * u_x
                        Ey += F_reg[ti, tj, n] * u_y

                    # k-modified curl: (∂/∂x + ik₀ₓ)Ey - (∂/∂y + ik₀ᵧ)Ex
                    dEx_dy = (np.roll(Ex, -1, 1) - np.roll(Ex, 1, 1)) / (2*d)
                    dEy_dx = (np.roll(Ey, -1, 0) - np.roll(Ey, 1, 0)) / (2*d)
                    curl_k = (dEy_dx + 1j*k0_phys[0]*Ey) - (dEx_dy + 1j*k0_phys[1]*Ex)

                    num = np.sum(np.abs(curl_k)**2)
                    # We need ε for this tile — but we don't have ε registry
                    # precomputed. We'll use a simpler approach: compute R_q
                    # and compare the SINGLE-EIGENSTATE R_q to calibrate FD error.
                    # For now, use ε=1 metric (∫|curl E|² vs ∫(2πf)²|E|²)
                    # Actually, for the correct metric we need ε.
                    # Use |E|² only (no ε weighting) — this gives R_q ≈ ω²_eff × <ε>
                    den = np.sum(np.abs(Ex)**2 + np.abs(Ey)**2)

                    if den > 1e-30:
                        Rq_tiles[ti, tj] = num / den

                    # Expected ω² for this tile (dominant band)
                    omega2_tiles[ti, tj] = (2*np.pi * omega_bands_reg[ti, tj, t])**2

            # Also compute single-eigenstate baseline (FD calibration)
            Rq_eigen = np.zeros((Ns_b, Ns_b))
            for ti in range(Ns_b):
                for tj in range(Ns_b):
                    u_x = bloch_fields[ti, tj, subspace_bands[t], :, :, 0]
                    u_y = bloch_fields[ti, tj, subspace_bands[t], :, :, 1]
                    dux_dy = (np.roll(u_x, -1, 1) - np.roll(u_x, 1, 1)) / (2*d)
                    duy_dx = (np.roll(u_y, -1, 0) - np.roll(u_y, 1, 0)) / (2*d)
                    curl_eigen = (duy_dx + 1j*k0_phys[0]*u_y) - (dux_dy + 1j*k0_phys[1]*u_x)
                    num_e = np.sum(np.abs(curl_eigen)**2)
                    den_e = np.sum(np.abs(u_x)**2 + np.abs(u_y)**2)
                    if den_e > 1e-30:
                        Rq_eigen[ti, tj] = num_e / den_e

            # FD-corrected residual: ratio of moiré R_q to eigenstate R_q
            # This cancels the FD error (which is ~40% at res=64)
            w = F_sq
            w_sum = np.sum(w)

            # Method 1: weighted R_q / weighted ω²
            Rq_weighted = float(np.sum(w * Rq_tiles) / w_sum) if w_sum > 0 else 0
            omega2_weighted = float(np.sum(w * omega2_tiles) / w_sum) if w_sum > 0 else 0
            ratio_raw = Rq_weighted / omega2_weighted if omega2_weighted > 0 else 0

            # Method 2: FD-corrected ratio (moiré / eigenstate)
            ratio_corrected_map = np.where(Rq_eigen > 0, Rq_tiles / Rq_eigen, 1.0)
            ratio_corrected = float(np.sum(w * ratio_corrected_map) / w_sum) if w_sum > 0 else 0

            # Residual = RMS deviation of corrected ratio from 1
            R_fd_corrected = float(np.sqrt(np.sum(w * (ratio_corrected_map - 1.0)**2) / w_sum)) if w_sum > 0 else 0

            br = {
                'band_index': t,
                'type': band_type,
                'eigenvalue': mode_eval,
                'mode_mixing': mode_mixing,
                'Rq_weighted': Rq_weighted,
                'omega2_weighted': omega2_weighted,
                'ratio_raw': ratio_raw,
                'ratio_fd_corrected': ratio_corrected,
                'R_fd_corrected': R_fd_corrected,
                'n_dominated': len(dom_indices),
            }
            entry['per_band_residuals'].append(br)

            print(f'  Band {t} ({band_type}): '
                  f'E0={mode_eval:.6f}, mix={mode_mixing:.4f}, '
                  f'R_q/ω²={ratio_raw:.4f}, '
                  f'FD-corr ratio={ratio_corrected:.4f}, '
                  f'R_fd={R_fd_corrected:.4e}')

        dt = time.time() - t0
        print(f'  Time: {dt:.1f}s')
        results.append(entry)

    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    t_start = time.time()

    # Option A
    print("\n" + "#"*70)
    print("# RUNNING OPTION A: N_bands convergence")
    print("#"*70)
    optA = run_option_A()

    # Option B
    print("\n" + "#"*70)
    print("# RUNNING OPTION B: BW scaling")
    print("#"*70)
    optB = run_option_B()

    # Option C
    print("\n" + "#"*70)
    print("# RUNNING OPTION C: Maxwell residual")
    print("#"*70)
    optC = run_option_C()

    # Save all results
    all_results = {
        'option_A': optA,
        'option_B': optB,
        'option_C': optC,
        'meta': {
            'n_modes': n_modes,
            'M_inv_max_trace': M_INV_MAX,
            'Ns_solve': Ns_solve,
            'thetas': thetas,
        }
    }

    outfile = f'{FINDINGS}/F04_validation_data.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\nAll results saved to {outfile}')

    # === PRINT FINAL SUMMARY ===
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    # Option A summary
    N_bands_report = len(optA[0]['bands'])
    print("\n--- OPTION A: |λ(N_full) − λ(N=1)| vs η ---")
    header = f'{"θ":>6s} {"η":>8s}'
    for t in range(N_bands_report):
        header += f' | {f"Band {t} |ΔE₀|":>14s} {"mix%":>6s}'
    print(header)
    for entry in optA:
        parts = [f'{entry["theta_deg"]:6.1f} {entry["eta"]:8.5f}']
        for t in range(N_bands_report):
            bd = entry['bands'][t]
            d = bd.get('delta_N1_N3')
            if d:
                parts.append(f'{d["max_abs"]:14.2e} {d["mean_mixing"]*100:5.2f}%')
            else:
                parts.append(f'{"N/A":>14s} {"N/A":>6s}')
        print(' | '.join(parts))

    # Option B summary
    print("\n--- OPTION B: Power-law BW ∝ η^α ---")
    for t_key in sorted(optB.keys(), key=lambda x: int(x)):
        b = optB[t_key]
        print(f'  {b["name"]}: α = {b["power_law_alpha"]:.3f}, R² = {b["power_law_R2"]:.4f}')

    # Option C summary
    print("\n--- OPTION C: Maxwell residual vs η ---")
    for entry in optC:
        parts = [f'{entry["theta_deg"]:6.1f} {entry["eta"]:8.5f}']
        for br in entry['per_band_residuals']:
            if br.get('R_fd_corrected') is not None:
                parts.append(f'B{br["band_index"]}: R={br.get("ratio_raw", 0):.4f} FD={br.get("ratio_fd_corrected", 0):.4f}')
            else:
                parts.append(f'B{br["band_index"]}: N/A')
        print(' | '.join(parts))

    t_total = time.time() - t_start
    print(f'\nTotal runtime: {t_total:.0f}s')
