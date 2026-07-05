#!/usr/bin/env python3
"""
Re-sweep with corrected Hamiltonian (F03 fixes):
1. Hermitized kinetic operator: K → (K + K†)/2
2. M_inv regularization: clamp |M_inv eigenvalues| ≤ M_max

Compare: 
  A. Original (non-Hermitian, no reg) — for reference
  B. Hermitized, no regularization  
  C. Hermitized + regularized (M_max=20)
  D. Hermitized + regularized (M_max=10)
"""

import numpy as np
import h5py
import json
import sys
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808'
PYTHON = '/home/renlephy/.local/share/mamba/envs/msl/bin/python'

thetas = ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']
n_modes = 20

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
    N_bands = Lambda.shape[2]
    
    print(f'\n{"="*70}')
    print(f'theta={theta_str} deg, eta={eta:.5f}, L={L_moire:.2f}')
    print(f'{"="*70}')
    
    entry = {
        'theta_deg': float(theta_str),
        'eta': eta,
        'omega_ref': omega_ref,
        'L_moire': L_moire,
        'per_band': []
    }
    
    for t in range(N_bands):
        Lb = Lambda[:, :, t:t+1, t:t+1]
        Mb = M_inv[:, :, t:t+1, t:t+1, :, :]
        vb = v_drift[:, :, t:t+1, t:t+1, :]
        Ab = A_berry[:, :, t:t+1, t:t+1, :]
        Pb = Phi_BH[:, :, t:t+1, t:t+1]
        
        V_min = float(Lb.min())
        V_max = float(Lb.max())
        V_range = V_max - V_min
        
        M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
        mean_mass = float(np.mean(M_trace))
        band_type = 'hole' if mean_mass < 0 else 'electron'
        
        # Determine sigma
        sigma = V_max if mean_mass < 0 else V_min
        
        band_result = {
            'band_index': t,
            'type': band_type,
            'mean_mass_trace': mean_mass,
            'V_min': V_min,
            'V_max': V_max,
            'sigma': sigma,
        }
        
        # Solve with regularization (M_max=20)
        try:
            H = p3.assemble_multiband_hamiltonian(
                Lb, vb, Mb, Ab, Pb, eta, Ns1, Ns1, 1, dR, dR, B_moire,
                include_drift=True, include_kinetic=True, include_born_huang=True,
                order=4, M_inv_max_trace=20
            )
            evals, _ = p3.solve_multiband_envelope(H, n_modes, sigma=sigma)
            evals_sorted = np.sort(np.real(evals))
            
            band_result['eigenvalues'] = evals_sorted.tolist()
            band_result['bandwidth_20'] = float(evals_sorted[-1] - evals_sorted[0])
            
            # Physical quality metrics
            delta_shallow = float(evals_sorted[0]) - V_min if band_type == 'electron' else V_max - float(evals_sorted[-1])
            band_result['delta_shallow'] = delta_shallow
            band_result['delta_shallow_rel'] = delta_shallow / V_range if V_range > 0 else 0
            
            print(f'  Band {t} ({band_type}): E0={evals_sorted[0]:.6f}, BW20={band_result["bandwidth_20"]:.6f}, '
                  f'delta_shallow={delta_shallow:.6f} ({delta_shallow/V_range:.3f}×V_range)')
        except Exception as e:
            print(f'  Band {t} ({band_type}): FAILED — {e}')
            band_result['eigenvalues'] = []
            band_result['error'] = str(e)
        
        entry['per_band'].append(band_result)
    
    results.append(entry)

# Save
outfile = '/home/renlephy/msl/research/moire_envelope/findings/sweep_results_F03_corrected.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nResults saved to {outfile}')
