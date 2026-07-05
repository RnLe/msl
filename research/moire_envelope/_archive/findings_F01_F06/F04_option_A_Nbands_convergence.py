#!/usr/bin/env python3
"""
F04 — Option A: Eigenvalue convergence with N_bands.

At each η, solve the envelope eigenvalue problem with N_sub = 1 and 3 bands.
The change |λ(N=3) − λ(N=1)| should decrease as η → 0, because inter-band
coupling scales with η.

Uses the CORRECTED Hamiltonian (Hermitized kinetic + M_inv regularization).
"""

import numpy as np
import h5py
import json
import sys
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'

thetas = ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']
n_modes = 20
M_INV_MAX = 20  # regularization

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

    L_moire = np.linalg.norm(B_moire[0])
    dR = L_moire / Ns1
    N_bands_full = Lambda.shape[2]  # = 3

    print(f'\n{"="*70}')
    print(f'theta={theta_str} deg, eta={eta:.5f}, L={L_moire:.2f}')
    print(f'{"="*70}')

    entry = {
        'theta_deg': float(theta_str),
        'eta': eta,
        'omega_ref': omega_ref,
        'L_moire': L_moire,
        'bands': {}
    }

    # === Solve N=1 (single-band) for each band ===
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
        evals_1, _ = p3.solve_multiband_envelope(H1, n_modes, sigma=sigma)
        evals_1 = np.sort(np.real(evals_1))

        entry['bands'][t] = {
            'type': band_type,
            'V_min': V_min,
            'V_max': V_max,
            'mean_mass_trace': mean_mass,
            'N1_eigenvalues': evals_1.tolist(),
        }

    # === Solve N=3 (coupled 3-band) ===
    # Use ALL 3 bands together (the full Phase 2 data)
    H3 = p3.assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH, eta, Ns1, Ns1, N_bands_full, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=True,
        order=4, M_inv_max_trace=M_INV_MAX
    )

    # For N=3, we need to target multiple spectral regions.
    # Solve with different sigma values to cover all bands.
    all_N3_evals = {}
    for t in range(N_bands_full):
        bd = entry['bands'][t]
        sigma = bd['V_max'] if bd['type'] == 'hole' else bd['V_min']
        # For the 3-band case, modes near each band's potential extremum
        evals_3, evecs_3 = p3.solve_multiband_envelope(H3, n_modes, sigma=sigma)
        evals_3 = np.sort(np.real(evals_3))

        # Identify which band each eigenvalue belongs to by checking
        # the eigenvector band composition
        # Each eigenvector has shape (Ns1*Ns2*3,). Reshape to (Ns1*Ns2, 3)
        # and check weight per band.
        N_s = Ns1 * Ns1
        band_weights = np.zeros((len(evals_3), N_bands_full))
        for m in range(len(evals_3)):
            v = evecs_3[:, m].reshape(N_s, N_bands_full)
            for b in range(N_bands_full):
                band_weights[m, b] = np.sum(np.abs(v[:, b])**2)
            band_weights[m] /= band_weights[m].sum()

        # Find eigenvalues dominated by band t
        dom_mask = band_weights[:, t] > 0.5
        evals_t = evals_3[dom_mask]
        evals_t = np.sort(evals_t)

        bd['N3_eigenvalues_all'] = evals_3.tolist()
        bd['N3_eigenvalues_band_filtered'] = evals_t.tolist()
        bd['N3_band_weights'] = band_weights.tolist()

        n_dom = len(evals_t)
        n_match = min(n_dom, len(bd['N1_eigenvalues']))

        if n_match > 0:
            diff = np.abs(evals_t[:n_match] - np.array(bd['N1_eigenvalues'][:n_match]))
            bd['N1_vs_N3_max_diff'] = float(np.max(diff))
            bd['N1_vs_N3_mean_diff'] = float(np.mean(diff))
            bd['N1_vs_N3_rel_diff'] = float(np.max(diff) / max(abs(evals_t[0]), 1e-10))
            print(f'  Band {t} ({bd["type"]}): N=1 E0={bd["N1_eigenvalues"][0]:.6f}, '
                  f'N=3 E0={evals_t[0]:.6f}, '
                  f'|ΔE|_max={bd["N1_vs_N3_max_diff"]:.2e}, '
                  f'rel={bd["N1_vs_N3_rel_diff"]:.2e}, '
                  f'n_dom={n_dom}/{n_modes}')
        else:
            bd['N1_vs_N3_max_diff'] = None
            bd['N1_vs_N3_mean_diff'] = None
            bd['N1_vs_N3_rel_diff'] = None
            print(f'  Band {t} ({bd["type"]}): no N=3 modes dominated by this band')

    results.append(entry)

# Save
outfile = '/home/renlephy/msl/research/moire_envelope/findings/F04_option_A_data.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved to {outfile}')

# === Print summary ===
print(f'\n{"="*70}')
print('OPTION A SUMMARY: |λ(N=3) - λ(N=1)| vs η')
print(f'{"="*70}')
print(f'{"theta":>6s} {"eta":>8s} | {"Band 0 (hole)":>20s} | {"Band 1 (elec)":>20s} | {"Band 2 (hole)":>20s}')
print(f'{"":>6s} {"":>8s} | {"max|ΔE|":>10s} {"rel":>8s} | {"max|ΔE|":>10s} {"rel":>8s} | {"max|ΔE|":>10s} {"rel":>8s}')
print('-' * 85)
for entry in results:
    parts = [f'{entry["theta_deg"]:6.1f} {entry["eta"]:8.5f}']
    for t in range(3):
        bd = entry['bands'][str(t)] if str(t) in entry['bands'] else entry['bands'][t]
        d = bd.get('N1_vs_N3_max_diff')
        r = bd.get('N1_vs_N3_rel_diff')
        if d is not None:
            parts.append(f'{d:10.2e} {r:8.2e}')
        else:
            parts.append(f'{"N/A":>10s} {"N/A":>8s}')
    print(' | '.join(parts))
