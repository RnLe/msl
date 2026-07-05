#!/usr/bin/env python3
"""
F03: Definitive numerical test of the eta^2 question.

The analytical derivation gives two conflicting conclusions:
1. The eta^2 from theory cancels against L^2 = (a/eta)^2 from converting
   dimensionless partial_R to physical partial_x. This gives prefactor = 1/(8pi^2).
2. But numerically, WITHOUT eta^2, eigenvalues are wildly unphysical.

RESOLUTION APPROACH: Use a simple analytically-solvable 1D problem to verify.
Particle in a cosine potential with known mass: compare exact vs code.
"""

import numpy as np
import sys, json
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3
from scipy import sparse
import h5py


def test_full_diag_small_grid():
    """
    Compare full diag vs eigsh, with and without eta^2, on a small grid.
    This is the DEFINITIVE test.
    """
    print("="*80)
    print("TEST: Full diagonalization on small Ns grid, with vs without eta^2")
    print("="*80)
    
    SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
    
    results = {}
    
    for theta_str in ['0.500', '1.500', '5.000', '8.000']:
        cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
        
        with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
            Lambda_full = hf['Lambda'][:]
            M_inv_full = hf['M_inv'][:]
            A_berry_full = hf['A_berry'][:]
            Phi_BH_full = hf['Phi_BH'][:]
            v_drift_full = hf['v_drift'][:]
            eta = float(hf.attrs['eta'])
            Ns1 = int(hf.attrs['Ns1'])
            B_moire = hf.attrs['B_moire']
        
        # Subsample to Ns=16 for full diag (16^2 = 256 eigenvalues)
        Ns_small = 16
        stride = Ns1 // Ns_small
        
        t = 1  # band 1 (electron)
        Lb = Lambda_full[::stride, ::stride, t:t+1, t:t+1]
        Mb = M_inv_full[::stride, ::stride, t:t+1, t:t+1, :, :]
        vb = v_drift_full[::stride, ::stride, t:t+1, t:t+1, :]
        Ab = A_berry_full[::stride, ::stride, t:t+1, t:t+1, :]
        Pb = Phi_BH_full[::stride, ::stride, t:t+1, t:t+1]
        
        V_min = float(np.min(Lb))
        V_max = float(np.max(Lb))
        V_range = V_max - V_min
        
        L_moire = np.linalg.norm(B_moire[0])
        dR = L_moire / Ns_small
        
        print(f"\ntheta={theta_str} deg, eta={eta:.5f}, L={L_moire:.2f}, Ns={Ns_small}")
        print(f"  V = [{V_min:.6f}, {V_max:.6f}], V_range = {V_range:.6f}")
        
        # --- Case A: Potential only ---
        H_V = p3.assemble_multiband_hamiltonian(
            Lb, vb, Mb, Ab, Pb, eta, Ns_small, Ns_small, 1, dR, dR, B_moire,
            include_drift=False, include_kinetic=False, include_born_huang=False, order=4
        )
        evals_V = np.sort(np.real(np.linalg.eigvalsh(H_V.toarray())))
        
        # --- Case B: No eta^2 (current code) ---
        H_no = p3.assemble_multiband_hamiltonian(
            Lb, vb, Mb, Ab, Pb, eta, Ns_small, Ns_small, 1, dR, dR, B_moire,
            include_drift=False, include_kinetic=True, include_born_huang=False, order=4
        )
        evals_no = np.sort(np.real(np.linalg.eigvalsh(H_no.toarray())))
        
        # --- Case C: WITH eta^2 ---
        Mb_eta2 = Mb * eta**2
        H_yes = p3.assemble_multiband_hamiltonian(
            Lb, vb, Mb_eta2, Ab, Pb, eta, Ns_small, Ns_small, 1, dR, dR, B_moire,
            include_drift=False, include_kinetic=True, include_born_huang=False, order=4
        )
        evals_yes = np.sort(np.real(np.linalg.eigvalsh(H_yes.toarray())))
        
        n_below_Vmin_no = int((evals_no < V_min).sum())
        n_below_Vmin_yes = int((evals_yes < V_min).sum())
        
        print(f"  Potential only:  E_min={evals_V[0]:.6f}, E_max={evals_V[-1]:.6f}")
        print(f"  No eta^2:        E_min={evals_no[0]:.6f}, E_max={evals_no[-1]:.6f}, below V_min: {n_below_Vmin_no}/{len(evals_no)}")
        print(f"  With eta^2:      E_min={evals_yes[0]:.6f}, E_max={evals_yes[-1]:.6f}, below V_min: {n_below_Vmin_yes}/{len(evals_yes)}")
        
        # Kinetic contribution at first moire K (n=1):
        q1 = 2*np.pi / L_moire
        T1_no = 0.5 / (2*np.pi)**2 * float(np.mean(Mb[:,:,0,0,0,0] + Mb[:,:,0,0,1,1])) * q1**2
        T1_yes = T1_no * eta**2
        
        print(f"  T(q1) no eta^2:  {T1_no:.6f} ({T1_no/V_range:.2f}x V_range)")
        print(f"  T(q1) with eta^2: {T1_yes:.2e} ({T1_yes/V_range:.4f}x V_range)")
        
        # For positive-definite kinetic (M>0 everywhere), eigenvalues
        # should be >= V_min. Having eigenvalues below V_min means M_inv < 0 somewhere
        # OR the kinetic is too large and numerical issues arise.
        
        results[theta_str] = {
            'eta': eta,
            'V_min': V_min, 'V_max': V_max,
            'evals_V_min': float(evals_V[0]),
            'evals_no_min': float(evals_no[0]), 'evals_no_max': float(evals_no[-1]),
            'evals_yes_min': float(evals_yes[0]), 'evals_yes_max': float(evals_yes[-1]),
            'n_below_Vmin_no': n_below_Vmin_no,
            'n_below_Vmin_yes': n_below_Vmin_yes,
            'T1_no_eta2': T1_no,
            'T1_with_eta2': T1_yes,
        }
    
    # ====================================================================
    # CRITICAL CHECK: Is M_inv positive definite everywhere?
    # If not, eigenvalues CAN go below V_min even with correct prefactor.
    # ====================================================================
    print("\n" + "="*80)
    print("CHECK: M_inv eigenvalue spectrum for band 1")
    print("="*80)
    
    cdir = f'{SWEEP}/theta_5.000/candidate_0000'
    with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
        M_inv = hf['M_inv'][:]
    
    # Band 1 mass tensor eigenvalues at each grid point
    M_eig_min = np.zeros((128, 128))
    M_eig_max = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            M = M_inv[i, j, 1, 1, :, :]  # band 1
            eigs = np.linalg.eigvalsh(M)
            M_eig_min[i,j] = eigs[0]
            M_eig_max[i,j] = eigs[1]
    
    print(f"  M_inv tensor eigenvalue range:")
    print(f"    Min eigenvalue: [{M_eig_min.min():.4f}, {M_eig_min.max():.4f}], mean={M_eig_min.mean():.4f}")
    print(f"    Max eigenvalue: [{M_eig_max.min():.4f}, {M_eig_max.max():.4f}], mean={M_eig_max.mean():.4f}")
    print(f"    Fraction with negative eigenvalue: {(M_eig_min < 0).mean()*100:.1f}%")
    
    if (M_eig_min < 0).any():
        print(f"    >>> M_inv has NEGATIVE eigenvalues at {(M_eig_min<0).sum()} points!")
        print(f"    >>> This means the kinetic operator is NOT positive-definite!")
        print(f"    >>> Eigenvalues CAN go below V_min regardless of eta^2!")
    else:
        print(f"    >>> M_inv is positive-definite everywhere")
        print(f"    >>> If eigenvalues go below V_min, the kinetic is too strong")
    
    return results


def test_dimensionless_vs_physical():
    """
    The ULTIMATE test: build the Hamiltonian in DIMENSIONLESS coordinates
    (as the theory uses) and compare.
    
    Theory: [Lambda(s) + eta^2 * (1/2) * M^{-1}_theory * (-i d/ds)^2] F = E F
    where s in [0, 1)^2, ds = 1/Ns
    
    This DIRECTLY implements the theory without any coordinate conversion.
    """
    print("\n" + "="*80)
    print("TEST: Dimensionless coordinates (theory's own formulation)")
    print("="*80)
    
    SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
    
    for theta_str in ['5.000']:
        cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
        
        with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
            Lambda_full = hf['Lambda'][:]
            M_inv_full = hf['M_inv'][:]
            eta = float(hf.attrs['eta'])
            Ns1 = int(hf.attrs['Ns1'])
            B_moire = hf.attrs['B_moire']
        
        Ns = 16
        stride = Ns1 // Ns
        t = 1
        
        L_moire = np.linalg.norm(B_moire[0])
        f0 = 0.2276  # reference frequency
        
        # Potential in f-units
        Lb_f = Lambda_full[::stride, ::stride, t, t]  # (Ns, Ns) — f - f_ref
        V_min_f = float(Lb_f.min())
        V_max_f = float(Lb_f.max())
        
        # M_inv in MPB units: d^2f / dk^2_MPB
        M_inv_MPB = M_inv_full[::stride, ::stride, t, t, :, :]  # (Ns, Ns, 2, 2)
        
        # Convert to THEORY units:
        # Lambda_theory = 4*pi^2 * f^2 -> linearized: 8*pi^2*f0*(f-f0)
        # M^{-1}_theory = 2*f0 * M^{-1}_MPB (at band extremum)
        
        # Build H in DIMENSIONLESS s-coordinates with ds = 1/Ns
        ds = 1.0 / Ns
        
        # Dimensionless Laplacian (eigenvalues: -(2*pi*n)^2)
        Lap_s = p3.build_periodic_laplacian_matrix(Ns, ds, order=4)
        
        # Kinetic prefactor in f-eigenvalue units:
        # eta^2 * (1/2) * M^{-1}_theory / (8*pi^2*f0)
        # = eta^2 * (1/2) * 2*f0 * M_MPB / (8*pi^2*f0)
        # = eta^2 * M_MPB / (8*pi^2)
        # = eta^2 / (2*(2*pi)^2) * M_MPB
        
        M_mean = float(np.mean(M_inv_MPB[:,:,0,0] + M_inv_MPB[:,:,1,1]))
        
        alpha_theory = eta**2 / (2 * (2*np.pi)**2)  # times M_MPB
        
        # Build 2D Hamiltonian manually (Ns x Ns = 256 dim)
        N_total = Ns * Ns
        
        # Potential: diagonal
        V_flat = Lb_f.ravel()
        H_V = sparse.diags(V_flat, format='csr')
        
        # Kinetic: -alpha * M * Laplacian (using UNIFORM M for simplicity)
        # 2D: Lap = Lap_x kron I + I kron Lap_y
        I_s = sparse.eye(Ns, format='csr')
        Lap_2d = sparse.kron(Lap_s, I_s) + sparse.kron(I_s, Lap_s)
        
        # Case A: Theory formula (with eta^2, dimensionless coords)
        H_theory = H_V - alpha_theory * M_mean * Lap_2d
        evals_theory = np.sort(np.real(np.linalg.eigvalsh(H_theory.toarray())))
        
        # Case B: Code formula (no eta^2, physical coords dR = L/Ns)
        dR = L_moire / Ns
        Lap_phys = p3.build_periodic_laplacian_matrix(Ns, dR, order=4)
        Lap_2d_phys = sparse.kron(Lap_phys, I_s) + sparse.kron(I_s, Lap_phys)
        
        alpha_code = 0.5 / (2*np.pi)**2
        H_code = H_V - alpha_code * M_mean * Lap_2d_phys
        evals_code = np.sort(np.real(np.linalg.eigvalsh(H_code.toarray())))
        
        # Case C: Code formula WITH eta^2, physical coords
        H_code_eta2 = H_V - alpha_code * eta**2 * M_mean * Lap_2d_phys
        evals_code_eta2 = np.sort(np.real(np.linalg.eigvalsh(H_code_eta2.toarray())))
        
        print(f"\n  theta={theta_str}, eta={eta:.5f}, L={L_moire:.2f}")
        print(f"  V_range = [{V_min_f:.6f}, {V_max_f:.6f}]")
        print(f"  M_mean = {M_mean:.4f}")
        print()
        print(f"  Theory (dim'less s, with eta^2):   E_min={evals_theory[0]:.6f}, E_max={evals_theory[-1]:.6f}")
        print(f"  Code   (physical x, no eta^2):     E_min={evals_code[0]:.6f}, E_max={evals_code[-1]:.6f}")
        print(f"  Code   (physical x, WITH eta^2):   E_min={evals_code_eta2[0]:.6f}, E_max={evals_code_eta2[-1]:.6f}")
        print()
        
        # Check: theory should equal code (if eta^2 cancels)
        diff_theory_code = np.max(np.abs(evals_theory - evals_code))
        diff_theory_code_eta2 = np.max(np.abs(evals_theory - evals_code_eta2))
        
        print(f"  |Theory - Code(no eta^2)| max   = {diff_theory_code:.2e}")
        print(f"  |Theory - Code(WITH eta^2)| max = {diff_theory_code_eta2:.2e}")
        print()
        
        if diff_theory_code < 1e-10:
            print(f"  >>> MATCH: Theory == Code(no eta^2)")
            print(f"  >>> The eta^2 DOES cancel. Code is CORRECT.")
        elif diff_theory_code_eta2 < 1e-10:
            print(f"  >>> MATCH: Theory == Code(WITH eta^2)")
            print(f"  >>> The eta^2 does NOT cancel. Need eta^2 in code!")
        else:
            print(f"  >>> Neither matches exactly. Need closer inspection.")
            print(f"  >>> 5 lowest eigenvalues:")
            print(f"      Theory:         {evals_theory[:5]}")
            print(f"      Code(no eta^2): {evals_code[:5]}")
            print(f"      Code(+eta^2):   {evals_code_eta2[:5]}")


if __name__ == '__main__':
    results = test_full_diag_small_grid()
    test_dimensionless_vs_physical()
    
    # Save results
    with open('/home/renlephy/msl/research/moire_envelope/findings/F03_data.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to findings/F03_data.json")
