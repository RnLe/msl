#!/usr/bin/env python
"""
Re-solve the η-sweep Hamiltonians with CORRECT per-band sigma targeting.

The original Phase 3 auto-selection used avg_mass_trace over all bands,
which for mixed electron/hole subspaces led to sigma=V_max(band 2),
finding scattering states of band 1 instead of bound states of band 0.

This script:
1. Loads the stored Hamiltonian from each angle's Phase 3 HDF5
2. Re-solves with sigma = V_max(band 0) = 0.091 for the target band
3. Also solves per-band blocks for comparison
4. Saves corrected results

No Phase 2 re-computation needed — we re-use the stored H matrix.
"""

import sys, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from pathlib import Path
import h5py

# Find the sweep directory
RUNS_BASE = Path('/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337')
SWEEP_DIR = sorted(RUNS_BASE.glob('eta_sweep_*'))[-1]

print(f"Sweep dir: {SWEEP_DIR}")

N_MODES = 50

def analyze_hamiltonian(p3_path):
    """Load Hamiltonian and re-solve with correct sigma targeting."""
    with h5py.File(p3_path, 'r') as f:
        H_data = f['H_data'][:]
        H_indices = f['H_indices'][:]
        H_indptr = f['H_indptr'][:]
        H_shape = f.attrs['H_shape']
        theta = float(f.attrs['theta_deg'])
        eta = float(f.attrs['eta'])
        N_sub = int(f.attrs['N_subspace'])
        omega_ref = float(f.attrs['omega_ref'])
        F_spinor = f['F_spinor'][:]
        evals_old = f['eigenvalues'][:]
    
    H = sp.csr_matrix((H_data, H_indices, H_indptr), shape=tuple(H_shape))
    N_total = int(H_shape[0])
    Ns = N_total // N_sub
    Ns_side = int(np.sqrt(Ns))
    
    # Also load Phase 2 data for potential ranges
    cand_dir = p3_path.parent
    with h5py.File(cand_dir / 'phase2_multiband_data.h5', 'r') as f2:
        Lambda = f2['Lambda'][:]
        M_inv = f2['M_inv'][:]
    
    result = {
        'theta_deg': theta,
        'eta': eta,
        'omega_ref': omega_ref,
        'old_eigenvalues': evals_old[:10].tolist(),
    }
    
    # ─── Per-band analysis ───
    per_band = []
    for n in range(N_sub):
        tr = M_inv[:, :, n, n, 0, 0] + M_inv[:, :, n, n, 1, 1]
        mean_tr = float(np.mean(tr))
        V_n = Lambda[:, :, n, n]
        V_min_n, V_max_n = float(np.min(V_n)), float(np.max(V_n))
        band_type = 'hole' if mean_tr < 0 else 'electron'
        
        # Determine correct sigma for this band
        sigma_n = V_max_n if band_type == 'hole' else V_min_n
        
        # Extract per-band block from the full Hamiltonian
        idx = np.arange(Ns) * N_sub + n
        H_block = H[np.ix_(idx, idx)]
        
        # Solve
        k = min(N_MODES, Ns - 2)
        try:
            ev_block, evec_block = eigsh(H_block.tocsc(), k=k, sigma=sigma_n, which='LM')
            # Sort by proximity to sigma
            order = np.argsort(np.abs(ev_block - sigma_n))
            ev_block = ev_block[order]
            evec_block = evec_block[:, order]
            
            E_bind = float(np.abs(ev_block[0] - sigma_n))
            bw_block = float(np.max(ev_block[:min(20, len(ev_block))]) - np.min(ev_block[:min(20, len(ev_block))]))
        except Exception as e:
            ev_block = np.array([])
            E_bind = float('nan')
            bw_block = float('nan')
            print(f"    Band {n} solve failed: {e}")
        
        per_band.append({
            'band_index': n,
            'type': band_type,
            'mean_mass_trace': mean_tr,
            'V_min': V_min_n,
            'V_max': V_max_n,
            'sigma': sigma_n,
            'lambda_0': float(ev_block[0]) if len(ev_block) > 0 else float('nan'),
            'E_bind': E_bind,
            'eigenvalues': ev_block[:20].tolist() if len(ev_block) > 0 else [],
            'bandwidth_20': bw_block,
        })
    
    result['per_band'] = per_band
    
    # ─── Full 3-band re-solve with TARGET BAND sigma ───
    # Target band is band 0 (hole-like, V_max = 0.091)
    target_idx = 0
    target_info = per_band[target_idx]
    sigma_target = target_info['sigma']  # V_max for band 0
    
    try:
        k = min(N_MODES, N_total - 2)
        ev_full, evec_full = eigsh(H.tocsc(), k=k, sigma=sigma_target, which='LM')
        # Sort by proximity to sigma
        order = np.argsort(np.abs(ev_full - sigma_target))
        ev_full = ev_full[order]
        evec_full = evec_full[:, order]
        
        # Compute band decomposition for re-solved modes
        compositions = []
        for m in range(min(len(ev_full), 20)):
            psi = evec_full[:, m].reshape(Ns_side, Ns_side, N_sub)
            weights = []
            for n in range(N_sub):
                weights.append(float(np.sum(np.abs(psi[:, :, n])**2)))
            total = sum(weights)
            weights = [w/total for w in weights]
            compositions.append({
                'mode': m,
                'weights': weights,
                'dominant': int(np.argmax(weights)),
                'max_weight': max(weights),
            })
        
        result['corrected_eigenvalues'] = ev_full[:50].tolist()
        result['corrected_sigma'] = sigma_target,
        result['corrected_compositions'] = compositions
        result['corrected_lambda_0'] = float(ev_full[0])
        result['corrected_E_bind'] = float(np.abs(ev_full[0] - sigma_target))
        result['corrected_bandwidth_50'] = float(ev_full[min(49, len(ev_full)-1)] - ev_full[0])
        result['corrected_gap_01'] = float(ev_full[1] - ev_full[0]) if len(ev_full) > 1 else 0.0
        
        # Max mixing among first 20 modes
        max_mixing = max(1.0 - c['max_weight'] for c in compositions)
        result['corrected_max_mixing'] = max_mixing
        
    except Exception as e:
        print(f"    Full re-solve failed: {e}")
        result['corrected_eigenvalues'] = []
    
    return result


def main():
    t0 = time.time()
    results = []
    
    for theta_dir in sorted(SWEEP_DIR.glob('theta_*')):
        p3 = theta_dir / 'candidate_0000' / 'phase3_multiband_modes.h5'
        if not p3.exists():
            print(f"  SKIP {theta_dir.name}: no Phase 3 data")
            continue
        
        print(f"\n{'='*60}")
        print(f"  {theta_dir.name}")
        r = analyze_hamiltonian(p3)
        results.append(r)
        
        # Summary
        tb = r['per_band'][0]
        print(f"  η={r['eta']:.5f}")
        print(f"  OLD  λ₀ = {r['old_eigenvalues'][0]:.6f} (σ=V_max_all=0.351)")
        print(f"  NEW  λ₀ = {r.get('corrected_lambda_0', 'N/A'):.6f} (σ=V_max[band0]={tb['sigma']:.4f})")
        print(f"  Band 0 block λ₀ = {tb['lambda_0']:.6f}, E_bind={tb['E_bind']:.6f}")
        
        if 'corrected_compositions' in r:
            c0 = r['corrected_compositions'][0]
            print(f"  Corrected mode 0: dominant={c0['dominant']}, weights={[f'{w:.4f}' for w in c0['weights']]}")
    
    # Save
    out_path = SWEEP_DIR / 'sweep_results_corrected.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s. Saved: {out_path}")
    
    # Print summary table
    print(f"\n{'θ°':>5s}  {'η':>8s}  {'OLD λ₀':>10s}  {'NEW λ₀':>10s}  {'E_bind':>10s}  {'E/η²':>8s}  {'dom':>4s}")
    print("-" * 65)
    for r in sorted(results, key=lambda x: x['eta']):
        old_l0 = r['old_eigenvalues'][0]
        new_l0 = r.get('corrected_lambda_0', float('nan'))
        eb = r.get('corrected_E_bind', float('nan'))
        e_ratio = eb / r['eta']**2 if not np.isnan(eb) else float('nan')
        dom = r['corrected_compositions'][0]['dominant'] if 'corrected_compositions' in r else '?'
        print(f"{r['theta_deg']:5.1f}  {r['eta']:8.5f}  {old_l0:10.6f}  {new_l0:10.6f}  {eb:10.6f}  {e_ratio:8.2f}  {dom:>4}")


if __name__ == '__main__':
    main()
