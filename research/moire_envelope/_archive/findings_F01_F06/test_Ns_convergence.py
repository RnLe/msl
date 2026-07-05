#!/usr/bin/env python3
"""
Ns-convergence test: Do the Phase 3 eigenvalues converge with grid resolution?

We take Band 1 at theta=0.5 deg and rebuild the Hamiltonian with subsampled grids
(Ns = 16, 32, 64, 128) to test convergence.

The potential Lambda is the SAME physical function sampled on each grid.
The kinetic operator has Nyquist energy ~ Ns^2.
If the eigenvalues converge, the kinetic coupling is a benign resolution effect.
If they DON'T converge, we have a fundamental problem.
"""
import h5py
import numpy as np
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../phasesV3')
from phasesV3 import phase3_mpb_v3 as p3
from scipy.sparse.linalg import eigsh

SWEEP = '../runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
cdir = f'{SWEEP}/theta_0.500/candidate_0000'

with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
    Lambda_full = hf['Lambda'][:]
    A_berry_full = hf['A_berry'][:]
    Phi_BH_full = hf['Phi_BH'][:]
    v_drift_full = hf['v_drift'][:]
    M_inv_full = hf['M_inv'][:]
    eta = float(hf.attrs['eta'])
    Ns1_full = int(hf.attrs['Ns1'])
    Ns2_full = int(hf.attrs['Ns2'])
    B_moire = hf.attrs['B_moire']

L_moire = np.linalg.norm(B_moire[0])
band = 1

print(f"eta = {eta:.6f}")
print(f"L_moire = {L_moire:.4f}")
print(f"Full grid: Ns = {Ns1_full}")
print()

# Test different subsampled Ns values
# We subsample by taking every step-th point
Ns_values = [16, 32, 64, 128]

print(f"{'Ns':>4}  {'N_total':>8}  {'dR':>10}  {'T_Nyq':>10}  {'lambda_0':>12}  {'delta':>12}  {'BW20':>12}  {'BW5':>12}")

for Ns in Ns_values:
    step = Ns1_full // Ns
    if step * Ns != Ns1_full:
        print(f"Ns={Ns}: cannot evenly subsample {Ns1_full}")
        continue
    
    # Subsample all fields
    Lambda_sub = Lambda_full[::step, ::step, :, :]
    A_berry_sub = A_berry_full[::step, ::step, :, :, :]
    Phi_BH_sub = Phi_BH_full[::step, ::step, :, :]
    v_drift_sub = v_drift_full[::step, ::step, :, :, :]
    M_inv_sub = M_inv_full[::step, ::step, :, :, :, :]
    
    # Single band extraction
    Lb = Lambda_sub[:, :, band:band+1, band:band+1]
    vd = v_drift_sub[:, :, band:band+1, band:band+1, :]
    Mi = M_inv_sub[:, :, band:band+1, band:band+1, :, :]
    Ab = A_berry_sub[:, :, band:band+1, band:band+1, :]
    Ph = Phi_BH_sub[:, :, band:band+1, band:band+1]
    
    dR = L_moire / Ns
    V_min = float(np.min(Lb[:, :, 0, 0]))
    
    # Build full Hamiltonian
    H = p3.assemble_multiband_hamiltonian(
        Lb, vd, Mi, Ab, Ph,
        eta, Ns, Ns, 1, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=True, order=4
    )
    
    # Kinetic Nyquist energy estimate
    M_trace_mean = np.mean(np.abs(np.trace(Mi[:, :, 0, 0, :, :], axis1=-2, axis2=-1).real))
    T_nyq = 0.5 / (2*np.pi)**2 * M_trace_mean * (np.pi/dR)**2
    
    # Solve
    k_eig = min(20, Ns*Ns - 2)
    evals, _ = eigsh(H, k=k_eig, sigma=V_min, which='LM')
    evals = np.sort(evals.real)
    
    delta = evals[0] - V_min
    bw20 = evals[-1] - evals[0]
    bw5 = evals[min(4, len(evals)-1)] - evals[0]
    
    print(f"{Ns:4d}  {Ns*Ns:8d}  {dR:10.6f}  {T_nyq:10.4f}  {evals[0]:12.8f}  {delta:12.4e}  {bw20:12.4e}  {bw5:12.4e}")

print()
print("If lambda_0 converges as Ns increases, the discretization is correct.")
print("If lambda_0 grows with Ns, the kinetic coupling is non-convergent (problem).")
