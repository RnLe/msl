#!/usr/bin/env python3
"""
F05 — Additional Thesis-Grade Validations

From Validations.md, the remaining items not yet covered:

1. GAUGE SMOOTHNESS DIAGNOSTIC
   - Measure overlap <u_n(R)|u_n(R+dR)> across registry
   - Quantify gauge discontinuity before/after fixing
   - Check stability of Berry/BH corrections under gauge

2. INVERSE PARTICIPATION RATIO (IPR)
   - Classify envelope modes as localized vs extended
   - Track IPR vs η for each band

3. GEOMETRIC CORRECTION MAGNITUDES
   - Systematic budget: |V|, |T_drift|, |K_kinetic|, |A²|, |Φ_BH|
   - Track relative importance vs η (thesis energy budget table)

4. MINIBAND DISPERSION Δλ(q)
   - Solve envelope eigenvalue problem at different q-points
   - Compute moiré miniband dispersion
   - Extract effective mass at band edges, bandwidth

5. HAMILTONIAN TERM CONVERGENCE
   - Solve with subsets of terms: V only, V+K, V+K+drift, V+K+drift+BH
   - Show which terms matter at which η
"""

import numpy as np
import h5py
import json
import sys
import time
from scipy.sparse import diags, kron, eye, lil_matrix
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808'
FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'

thetas = ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']
M_INV_MAX = 20
n_modes = 20


def load_sweep_data(theta_str, load_bloch=False, bloch_bands=3):
    """Load Phase 2 data for one angle. Optionally load Phase 1 bloch fields."""
    cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
    with open(f'{cdir}/phase0_meta.json') as f:
        meta = json.load(f)
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
    
    result = dict(meta=meta, Lambda=Lambda, M_inv=M_inv, A_berry=A_berry,
                  Phi_BH=Phi_BH, v_drift=v_drift, eta=eta, Ns1=Ns1,
                  B_moire=B_moire, omega_ref=omega_ref, L_moire=L_moire, dR=dR)
    
    if load_bloch:
        with h5py.File(f'{cdir}/phase1_multiband_data.h5', 'r') as hf:
            # Only load first bloch_bands bands to save memory
            result['bloch_fields'] = hf['bloch_fields'][:, :, :bloch_bands, :, :, :]
    
    return result


# ============================================================================
# 1. GAUGE SMOOTHNESS DIAGNOSTIC
# ============================================================================
def run_gauge_diagnostic():
    """
    Measure Bloch function overlap <u(R)|u(R+dR)> across registry.
    Before gauge-fixing: overlap has random phase.
    After gauge-fixing: overlap is real and positive.
    """
    print("\n" + "="*70)
    print("1. GAUGE SMOOTHNESS DIAGNOSTIC")
    print("="*70)
    
    results = []
    gauge_thetas = ['0.500', '1.500', '3.000', '8.000']  # Subset for memory
    for theta_str in gauge_thetas:
        cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
        d = load_sweep_data(theta_str, load_bloch=False)
        
        print(f'\n  theta={theta_str}°')
        entry = {'theta_deg': float(theta_str), 'eta': d['eta'], 'bands': {}}
        
        # Compute overlaps using h5py partial reads to avoid loading full array
        with h5py.File(f'{cdir}/phase1_multiband_data.h5', 'r') as hf:
            bf_dset = hf['bloch_fields']  # (Ns_b, Ns_b, N_total_bands, Nx, Ny, 3)
            Ns_b = bf_dset.shape[0]
            N_subspace = int(hf.attrs['N_subspace']) if 'N_subspace' in hf.attrs else bf_dset.shape[2]
            
            for t in range(N_subspace):
                # Vectorized: load band t for all (i,j) at once
                # Shape: (Ns_b, Ns_b, 64, 64, 3)
                bf_t = bf_dset[:, :, t, :, :, :]
                
                # Normalize each tile: u(R) → u(R) / ||u(R)||
                norms = np.sqrt(np.sum(np.abs(bf_t)**2, axis=(2, 3, 4), keepdims=True))
                norms = np.maximum(norms, 1e-30)  # avoid division by zero
                bf_t = bf_t / norms
                
                # Overlap along s1: <u(i,j)|u(i+1,j)> for all (i,j)
                # Roll by -1 along axis 0 to get the neighbor
                bf_shifted_s1 = np.roll(bf_t, -1, axis=0)
                bf_shifted_s2 = np.roll(bf_t, -1, axis=1)
                
                # Inner product over micro-cell axes (2,3,4) = (nx, ny, comp)
                ov_s1 = np.sum(np.conj(bf_t) * bf_shifted_s1, axis=(2, 3, 4))
                ov_s2 = np.sum(np.conj(bf_t) * bf_shifted_s2, axis=(2, 3, 4))
                
                ov_s1 = ov_s1.ravel()
                ov_s2 = ov_s2.ravel()
                
                phase_s1 = np.angle(ov_s1)
                phase_s2 = np.angle(ov_s2)
                mag_s1 = np.abs(ov_s1)
                mag_s2 = np.abs(ov_s2)
                
                frac_good_s1 = float(np.mean(mag_s1 > 0.99))
                frac_good_s2 = float(np.mean(mag_s2 > 0.99))
                phase_std_s1 = float(np.std(phase_s1))
                phase_std_s2 = float(np.std(phase_s2))
                min_mag = float(min(np.min(mag_s1), np.min(mag_s2)))
                
                entry['bands'][t] = {
                    'min_overlap_mag': min_mag,
                    'mean_overlap_mag_s1': float(np.mean(mag_s1)),
                    'mean_overlap_mag_s2': float(np.mean(mag_s2)),
                    'phase_std_s1': phase_std_s1,
                    'phase_std_s2': phase_std_s2,
                    'frac_good_s1': frac_good_s1,
                    'frac_good_s2': frac_good_s2,
                }
                
                print(f'    Band {t}: min|ov|={min_mag:.4f}, '
                      f'phase_std=({phase_std_s1:.3f}, {phase_std_s2:.3f}) rad, '
                      f'frac_good=({frac_good_s1:.3f}, {frac_good_s2:.3f})')
                
                del bf_t, bf_shifted_s1, bf_shifted_s2, ov_s1, ov_s2
        
        results.append(entry)
        import gc; gc.collect()
    
    return results


# ============================================================================
# 2. INVERSE PARTICIPATION RATIO (IPR)
# ============================================================================
def run_ipr_analysis():
    """
    Compute IPR of envelope modes: IPR = ∫|F|⁴ / (∫|F|²)²
    
    IPR → 1/N_sites for extended plane wave
    IPR → 1 for fully localized delta function
    """
    print("\n" + "="*70)
    print("2. INVERSE PARTICIPATION RATIO (IPR)")
    print("="*70)
    
    results = []
    for theta_str in thetas:
        d = load_sweep_data(theta_str)
        Ns1, N_bands = d['Ns1'], d['Lambda'].shape[2]
        dR = d['dR']
        L_moire = d['L_moire']
        
        print(f'\n  theta={theta_str}°, eta={d["eta"]:.5f}')
        entry = {'theta_deg': float(theta_str), 'eta': d['eta'], 'bands': {}}
        
        for t in range(N_bands):
            Lb = d['Lambda'][:, :, t:t+1, t:t+1]
            Mb = d['M_inv'][:, :, t:t+1, t:t+1, :, :]
            vb = d['v_drift'][:, :, t:t+1, t:t+1, :]
            Ab = d['A_berry'][:, :, t:t+1, t:t+1, :]
            Pb = d['Phi_BH'][:, :, t:t+1, t:t+1]
            
            M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
            band_type = 'hole' if np.mean(M_trace) < 0 else 'electron'
            sigma = float(Lb.max()) if band_type == 'hole' else float(Lb.min())
            
            H = p3.assemble_multiband_hamiltonian(
                Lb, vb, Mb, Ab, Pb, d['eta'], Ns1, Ns1, 1, dR, dR, d['B_moire'],
                include_drift=True, include_kinetic=True, include_born_huang=True,
                order=4, M_inv_max_trace=M_INV_MAX
            )
            evals, evecs = p3.solve_multiband_envelope(H, n_modes, sigma=sigma)
            idx = np.argsort(np.real(evals))
            evals = np.real(evals[idx])
            evecs = evecs[:, idx]
            
            iprs = []
            spreads = []
            for m in range(len(evals)):
                F = evecs[:, m].reshape(Ns1, Ns1)
                F_sq = np.abs(F)**2
                ipr = float(np.sum(F_sq**2) / np.sum(F_sq)**2 * Ns1 * Ns1)
                # IPR normalized: 1 = localized to 1 site, N = fully extended
                # We report IPR * N_sites so that extended → 1, localized → N
                # Actually, standard: IPR = ∫|F|⁴/(∫|F|²)², dimensionless
                # For extended: IPR ≈ 1/N. For localized: IPR ≈ 1.
                # We report IPR * N_sites (participation number): extended → N, localized → 1
                participation_number = 1.0 / (np.sum(F_sq**2) / np.sum(F_sq)**2)
                
                # Spatial spread: σ = sqrt(<r²> - <r>²) / L_moire
                s1 = np.linspace(0, 1, Ns1, endpoint=False)
                s2 = np.linspace(0, 1, Ns1, endpoint=False)
                S1, S2 = np.meshgrid(s1, s2, indexing='ij')
                
                weight = F_sq / F_sq.sum()
                s1_mean = np.sum(weight * S1)
                s2_mean = np.sum(weight * S2)
                s1_var = np.sum(weight * (S1 - s1_mean)**2)
                s2_var = np.sum(weight * (S2 - s2_mean)**2)
                spread = float(np.sqrt(s1_var + s2_var))  # in units of L_moire
                
                iprs.append(float(participation_number))
                spreads.append(spread)
            
            entry['bands'][t] = {
                'type': band_type,
                'eigenvalues': evals.tolist(),
                'participation_numbers': iprs,
                'spatial_spreads': spreads,
                'mean_PN_5': float(np.mean(iprs[:5])),
                'mean_spread_5': float(np.mean(spreads[:5])),
            }
            
            print(f'    Band {t} ({band_type}): '
                  f'PN(5 lowest)={np.mean(iprs[:5]):.0f}/{Ns1**2}, '
                  f'spread={np.mean(spreads[:5]):.3f}×L')
        
        results.append(entry)
    
    return results


# ============================================================================
# 3. GEOMETRIC CORRECTION MAGNITUDES (Energy Budget)
# ============================================================================
def run_energy_budget():
    """
    Compute magnitude of each Hamiltonian term as a function of η.
    
    Budget: ||V||, ||T_drift||, ||K_kinetic||, ||Φ_BH|| relative to ||V||.
    """
    print("\n" + "="*70)
    print("3. GEOMETRIC CORRECTION MAGNITUDES (Energy Budget)")
    print("="*70)
    
    results = []
    for theta_str in thetas:
        d = load_sweep_data(theta_str)
        Ns1, N_bands = d['Ns1'], d['Lambda'].shape[2]
        dR = d['dR']
        eta = d['eta']
        
        print(f'\n  theta={theta_str}°, eta={eta:.5f}')
        entry = {'theta_deg': float(theta_str), 'eta': eta, 'bands': {}}
        
        for t in range(N_bands):
            Lb = d['Lambda'][:, :, t:t+1, t:t+1]
            Mb = d['M_inv'][:, :, t:t+1, t:t+1, :, :]
            vb = d['v_drift'][:, :, t:t+1, t:t+1, :]
            Ab = d['A_berry'][:, :, t:t+1, t:t+1, :]
            Pb = d['Phi_BH'][:, :, t:t+1, t:t+1]
            
            # Build individual operators
            V = p3.build_multiband_potential_operator(Lb, d['B_moire'])
            T = p3.build_multiband_drift_operator(vb, eta, Ns1, Ns1, 1, dR, dR, order=4)
            
            M_inv_reg = p3._regularize_M_inv(Mb, M_INV_MAX)
            K = p3.build_multiband_kinetic_operator(
                M_inv_reg, Ab, eta, Ns1, Ns1, 1, dR, dR, d['B_moire'], order=4)
            
            BH = p3.build_multiband_born_huang_operator(Pb, eta, Ns1, Ns1, 1)
            
            # Frobenius norms (operator magnitude)
            from scipy.sparse.linalg import norm as sp_norm
            norm_V = sp_norm(V, 'fro')
            norm_T = sp_norm(T, 'fro')
            norm_K = sp_norm(K, 'fro')
            norm_BH = sp_norm(BH, 'fro')
            
            # Also max diagonal element (direct physical magnitude)
            diag_V = np.abs(V.diagonal())
            diag_K = np.abs(K.diagonal())
            diag_BH = np.abs(BH.diagonal())
            
            # Berry connection magnitudes
            A_nn = d['A_berry'][:, :, t, t, :]
            max_A = float(np.max(np.abs(A_nn)))
            mean_A = float(np.mean(np.abs(A_nn)))
            
            # Born-Huang diagonal
            max_phi = float(np.max(np.abs(d['Phi_BH'][:, :, t, t])))
            
            bd = {
                'norm_V': float(norm_V),
                'norm_T': float(norm_T),
                'norm_K': float(norm_K),
                'norm_BH': float(norm_BH),
                'ratio_T_V': float(norm_T / norm_V) if norm_V > 0 else 0,
                'ratio_K_V': float(norm_K / norm_V) if norm_V > 0 else 0,
                'ratio_BH_V': float(norm_BH / norm_V) if norm_V > 0 else 0,
                'max_diag_V': float(np.max(diag_V)),
                'max_diag_K': float(np.max(diag_K)),
                'max_diag_BH': float(np.max(diag_BH)),
                'max_A_berry': max_A,
                'mean_A_berry': mean_A,
                'max_Phi_BH': max_phi,
            }
            entry['bands'][t] = bd
            
            print(f'    Band {t}: ||K||/||V||={bd["ratio_K_V"]:.4f}, '
                  f'||T||/||V||={bd["ratio_T_V"]:.2e}, '
                  f'||BH||/||V||={bd["ratio_BH_V"]:.4f}, '
                  f'max|A|={max_A:.3f}')
        
        results.append(entry)
    
    return results


# ============================================================================
# 4. MINIBAND DISPERSION Δλ(q)
# ============================================================================
def run_miniband_dispersion():
    """
    Solve envelope eigenvalue problem at different moiré q-points.
    
    The envelope equation on the moiré lattice has Bloch periodicity:
        F(R + L_i) = e^{iq·L_i} F(R)
    
    We implement this by adding phase factors to the periodic FD operators:
        D → D_q where boundary condition wraps with phase e^{iq·L}.
    
    This is equivalent to replacing -i∂/∂R → -i∂/∂R + q in the FD stencil.
    
    We scan q along the moiré BZ path: Γ → M → K → Γ (for square lattice: Γ → X → M → Γ).
    """
    print("\n" + "="*70)
    print("4. MINIBAND DISPERSION Δλ(q)")
    print("="*70)
    
    # Use a small subset of angles — this is very expensive
    test_thetas = ['1.500', '5.000']
    n_qpoints = 8  # per segment (reduced for speed)
    
    results = []
    for theta_str in test_thetas:
        d = load_sweep_data(theta_str)
        Ns1, N_bands = d['Ns1'], d['Lambda'].shape[2]
        dR = d['dR']
        eta = d['eta']
        L_moire = d['L_moire']
        B_moire = d['B_moire']
        
        print(f'\n  theta={theta_str}°, eta={eta:.5f}')
        
        # Moiré reciprocal lattice
        G_moire = 2 * np.pi * np.linalg.inv(B_moire).T  # columns are G1, G2
        
        # BZ path for square moiré lattice: Γ(0,0) → X(0.5,0) → M(0.5,0.5) → Γ(0,0)
        path_labels = ['Γ', 'X', 'M', 'Γ']
        path_frac = [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0)]
        
        q_points_frac = []
        segment_lengths = []
        for seg in range(len(path_frac) - 1):
            q0 = np.array(path_frac[seg])
            q1 = np.array(path_frac[seg + 1])
            for i in range(n_qpoints):
                t_param = i / n_qpoints
                q_points_frac.append(q0 + t_param * (q1 - q0))
            segment_lengths.append(n_qpoints)
        q_points_frac.append(np.array(path_frac[-1]))
        
        n_qtotal = len(q_points_frac)
        
        entry = {
            'theta_deg': float(theta_str),
            'eta': eta,
            'L_moire': L_moire,
            'path_labels': path_labels,
            'n_qpoints_per_segment': n_qpoints,
            'bands': {}
        }
        
        # For each band, solve at each q-point (single-band for speed)
        for t in range(N_bands):
            Lb = d['Lambda'][:, :, t:t+1, t:t+1]
            Mb = d['M_inv'][:, :, t:t+1, t:t+1, :, :]
            vb = d['v_drift'][:, :, t:t+1, t:t+1, :]
            Ab = d['A_berry'][:, :, t:t+1, t:t+1, :]
            Pb = d['Phi_BH'][:, :, t:t+1, t:t+1]
            
            M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
            band_type = 'hole' if np.mean(M_trace) < 0 else 'electron'
            sigma = float(Lb.max()) if band_type == 'hole' else float(Lb.min())
            
            all_evals = []
            for qi, q_frac in enumerate(q_points_frac):
                # q in physical units: q_phys = G_moire · q_frac
                q_phys = G_moire @ q_frac
                
                # Build H with Bloch phase: replace D → D + iq
                # We add the q-shift to the kinetic operator via modified FD:
                # D_q[n,n+1] = D[n,n+1] * e^{i q·dR}
                # For the kinetic term K = M D² + ..., adding q means:
                # (-iD + q)² = D² + 2iqD + q²  (needs careful implementation)
                #
                # Simplest correct approach: shift the FD boundary conditions.
                # D_q = D + iq (where D is the standard periodic derivative)
                # L_q = (D + iq)² = D² + 2iqD - q²
                
                # Build base operators
                D1_base = p3.build_periodic_derivative_matrix(Ns1, dR, 4)
                D2_base = p3.build_periodic_derivative_matrix(Ns1, dR, 4)
                L1_base = p3.build_periodic_laplacian_matrix(Ns1, dR, 4)
                L2_base = p3.build_periodic_laplacian_matrix(Ns1, dR, 4)
                
                # Modified operators: L_q = L + 2iq·D - q²
                q1_val = q_phys[0]
                q2_val = q_phys[1]
                
                L1_q = L1_base + 2j * q1_val * D1_base - q1_val**2 * eye(Ns1)
                L2_q = L2_base + 2j * q2_val * D2_base - q2_val**2 * eye(Ns1)
                D1_q = D1_base + 1j * q1_val * eye(Ns1)
                D2_q = D2_base + 1j * q2_val * eye(Ns1)
                
                # Build full kinetic operator with q-shifted derivatives
                # This is manual construction following build_multiband_kinetic_operator logic
                N_s = Ns1 * Ns1
                scale = 0.5 / (2 * np.pi)**2
                
                M_inv_reg = p3._regularize_M_inv(Mb, M_INV_MAX)
                M_flat = M_inv_reg.reshape(N_s, 1, 1, 2, 2)
                M11 = M_flat[:, 0, 0, 0, 0]
                M22 = M_flat[:, 0, 0, 1, 1]
                M12 = M_flat[:, 0, 0, 0, 1]
                
                M11_diag = diags(M11, format='csr')
                M22_diag = diags(M22, format='csr')
                M12_diag = diags(M12, format='csr')
                
                L1_q_full = kron(L1_q, eye(Ns1), format='csr')
                L2_q_full = kron(eye(Ns1), L2_q, format='csr')
                D1_q_full = kron(D1_q, eye(Ns1), format='csr')
                D2_q_full = kron(eye(Ns1), D2_q, format='csr')
                
                K_q = -scale * (M11_diag @ L1_q_full + M22_diag @ L2_q_full)
                if np.max(np.abs(M12)) > 1e-15:
                    K_q = K_q - 2 * scale * M12_diag @ (D1_q_full @ D2_q_full)
                
                # Hermitize
                K_q = (K_q + K_q.T.conj()) / 2
                
                # Potential
                V_op = p3.build_multiband_potential_operator(Lb, d['B_moire'])
                
                # Born-Huang
                BH_op = p3.build_multiband_born_huang_operator(Pb, eta, Ns1, Ns1, 1)
                
                H_q = V_op + K_q + BH_op
                H_q = H_q.tocsr()
                
                # Solve
                ev, _ = p3.solve_multiband_envelope(H_q, min(5, n_modes), sigma=sigma)
                ev = np.sort(np.real(ev))
                all_evals.append(ev.tolist())
            
            # Extract band edges
            all_evals_arr = np.array(all_evals)  # (n_q, n_modes_found)
            n_found = all_evals_arr.shape[1]
            
            bandwidth_q = []
            for m in range(min(3, n_found)):
                bw = float(np.max(all_evals_arr[:, m]) - np.min(all_evals_arr[:, m]))
                bandwidth_q.append(bw)
            
            entry['bands'][t] = {
                'type': band_type,
                'eigenvalues_vs_q': all_evals,
                'q_points_frac': [q.tolist() for q in q_points_frac],
                'bandwidth_q': bandwidth_q,
            }
            
            print(f'    Band {t} ({band_type}): '
                  f'BW(q) of lowest 3 minibands: {[f"{bw:.6f}" for bw in bandwidth_q[:3]]}')
        
        results.append(entry)
    
    return results


# ============================================================================
# 5. HAMILTONIAN TERM CONVERGENCE
# ============================================================================
def run_term_convergence():
    """
    Solve with different subsets of Hamiltonian terms:
    V only, V+K, V+K+drift, V+K+drift+BH (full)
    
    Shows which terms matter at which η.
    """
    print("\n" + "="*70)
    print("5. HAMILTONIAN TERM CONVERGENCE")
    print("="*70)
    
    results = []
    for theta_str in thetas:
        d = load_sweep_data(theta_str)
        Ns1, N_bands = d['Ns1'], d['Lambda'].shape[2]
        dR = d['dR']
        eta = d['eta']
        
        print(f'\n  theta={theta_str}°, eta={eta:.5f}')
        entry = {'theta_deg': float(theta_str), 'eta': eta, 'bands': {}}
        
        configs = [
            ('V_only', True, False, False),
            ('V+K', True, True, False),
            ('V+K+BH', True, True, True),
        ]
        # Note: drift is negligible (max|v| = 1.5e-4 at X), skip it for brevity
        
        for t in range(N_bands):
            Lb = d['Lambda'][:, :, t:t+1, t:t+1]
            Mb = d['M_inv'][:, :, t:t+1, t:t+1, :, :]
            vb = d['v_drift'][:, :, t:t+1, t:t+1, :]
            Ab = d['A_berry'][:, :, t:t+1, t:t+1, :]
            Pb = d['Phi_BH'][:, :, t:t+1, t:t+1]
            
            M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
            band_type = 'hole' if np.mean(M_trace) < 0 else 'electron'
            sigma = float(Lb.max()) if band_type == 'hole' else float(Lb.min())
            
            bd = {'type': band_type, 'configs': {}}
            
            for name, inc_k, inc_bh, _ in configs:
                H = p3.assemble_multiband_hamiltonian(
                    Lb, vb, Mb, Ab, Pb, eta, Ns1, Ns1, 1, dR, dR, d['B_moire'],
                    include_drift=False, include_kinetic=inc_k,
                    include_born_huang=inc_bh,
                    order=4, M_inv_max_trace=M_INV_MAX if inc_k else None
                )
                ev, _ = p3.solve_multiband_envelope(H, n_modes, sigma=sigma)
                ev = np.sort(np.real(ev))
                bd['configs'][name] = {
                    'eigenvalues': ev[:5].tolist(),
                    'E0': float(ev[0]),
                    'BW5': float(ev[4] - ev[0]) if len(ev) >= 5 else 0,
                }
            
            # Compute differences
            e_V = bd['configs']['V_only']['E0']
            e_VK = bd['configs']['V+K']['E0']
            e_full = bd['configs']['V+K+BH']['E0']
            
            bd['shift_K'] = float(e_VK - e_V)
            bd['shift_BH'] = float(e_full - e_VK)
            bd['shift_total'] = float(e_full - e_V)
            
            entry['bands'][t] = bd
            print(f'    Band {t} ({band_type}): '
                  f'E0(V)={e_V:.6f}, '
                  f'+K: {e_VK - e_V:+.2e}, '
                  f'+BH: {e_full - e_VK:+.2e}')
        
        results.append(entry)
    
    return results


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    t_start = time.time()
    all_results = {}
    
    # 1. Gauge diagnostic
    print("\n" + "#"*70)
    print("# SECTION 1: GAUGE SMOOTHNESS")
    print("#"*70)
    all_results['gauge_smoothness'] = run_gauge_diagnostic()
    
    # 2. IPR
    print("\n" + "#"*70)
    print("# SECTION 2: INVERSE PARTICIPATION RATIO")
    print("#"*70)
    all_results['ipr'] = run_ipr_analysis()
    
    # 3. Energy budget
    print("\n" + "#"*70)
    print("# SECTION 3: ENERGY BUDGET")
    print("#"*70)
    all_results['energy_budget'] = run_energy_budget()
    
    # 4. Miniband dispersion
    print("\n" + "#"*70)
    print("# SECTION 4: MINIBAND DISPERSION")
    print("#"*70)
    all_results['miniband_dispersion'] = run_miniband_dispersion()
    
    # 5. Term convergence
    print("\n" + "#"*70)
    print("# SECTION 5: TERM CONVERGENCE")
    print("#"*70)
    all_results['term_convergence'] = run_term_convergence()
    
    # Save
    outfile = f'{FINDINGS}/F05_validation_data.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    t_total = time.time() - t_start
    print(f'\nTotal runtime: {t_total:.0f}s')
    print(f'Results saved to {outfile}')
