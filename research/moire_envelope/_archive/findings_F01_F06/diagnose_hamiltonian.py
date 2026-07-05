#!/usr/bin/env python3
"""Diagnose why eigsh eigenvalues differ so much from diagonal Lambda values."""
import h5py
import numpy as np
import sys
import scipy.sparse as sp
sys.path.insert(0, '..')
sys.path.insert(0, '../phasesV3')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '../runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'
cdir = f'{SWEEP}/theta_0.500/candidate_0000'
with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
    Lambda = hf['Lambda'][:]
    A_berry = hf['A_berry'][:]
    Phi_BH = hf['Phi_BH'][:]
    v_drift = hf['v_drift'][:]
    M_inv = hf['M_inv'][:]
    eta = float(hf.attrs['eta'])
    Ns1 = int(hf.attrs['Ns1'])
    Ns2 = int(hf.attrs['Ns2'])
    B_moire = hf.attrs['B_moire']

L_moire = np.linalg.norm(B_moire[0])
dR1, dR2 = L_moire / Ns1, L_moire / Ns2

band = 1
Lambda_b = Lambda[:, :, band:band+1, band:band+1]
v_drift_b = v_drift[:, :, band:band+1, band:band+1, :]
M_inv_b = M_inv[:, :, band:band+1, band:band+1, :, :]
A_berry_b = A_berry[:, :, band:band+1, band:band+1, :]
Phi_BH_b = Phi_BH[:, :, band:band+1, band:band+1]

print(f"eta = {eta:.6f}")
print(f"Ns1 = {Ns1}, Ns2 = {Ns2}")
print(f"L_moire = {L_moire:.4f}")
print(f"dR1 = {dR1:.6f}, dR2 = {dR2:.6f}")
print(f"B_moire[0] = {B_moire[0]}")
print(f"B_moire[1] = {B_moire[1]}")
print()

# Build H step by step to see each contribution
print("=== Building Hamiltonian piece by piece ===")

# Potential only
H_pot = p3.assemble_multiband_hamiltonian(
    Lambda_b, v_drift_b, M_inv_b, A_berry_b, Phi_BH_b,
    eta, Ns1, Ns2, 1, dR1, dR2, B_moire,
    include_drift=False, include_kinetic=False, include_born_huang=False, order=4)

# Potential + drift
H_pd = p3.assemble_multiband_hamiltonian(
    Lambda_b, v_drift_b, M_inv_b, A_berry_b, Phi_BH_b,
    eta, Ns1, Ns2, 1, dR1, dR2, B_moire,
    include_drift=True, include_kinetic=False, include_born_huang=False, order=4)

# Potential + kinetic
H_pk = p3.assemble_multiband_hamiltonian(
    Lambda_b, v_drift_b, M_inv_b, A_berry_b, Phi_BH_b,
    eta, Ns1, Ns2, 1, dR1, dR2, B_moire,
    include_drift=False, include_kinetic=True, include_born_huang=False, order=4)

# Full
H_full = p3.assemble_multiband_hamiltonian(
    Lambda_b, v_drift_b, M_inv_b, A_berry_b, Phi_BH_b,
    eta, Ns1, Ns2, 1, dR1, dR2, B_moire,
    include_drift=True, include_kinetic=True, include_born_huang=True, order=4)

print(f"\nH_pot  shape: {H_pot.shape}, nnz: {H_pot.nnz}")
print(f"H_full shape: {H_full.shape}, nnz: {H_full.nnz}")

# Check norms of each part
H_drift = H_pd - H_pot
H_kin = H_pk - H_pot
H_bh = H_full - H_pk - H_drift

# Frobenius norms
print(f"\nFrobenius norms:")
print(f"  ||Potential||     = {sp.linalg.norm(H_pot):.6e}")
print(f"  ||Drift||         = {sp.linalg.norm(H_drift):.6e}")
print(f"  ||Kinetic||       = {sp.linalg.norm(H_kin):.6e}")
print(f"  ||Born-Huang||    = {sp.linalg.norm(H_bh):.6e}")
print(f"  ||Full||          = {sp.linalg.norm(H_full):.6e}")

# Also max absolute element
print(f"\nMax absolute element:")
print(f"  max|Potential|    = {abs(H_pot).max():.6e}")
print(f"  max|Drift|        = {abs(H_drift).max():.6e}")
print(f"  max|Kinetic|      = {abs(H_kin).max():.6e}")

# Check the DIAGONAL of the kinetic part 
H_kin_dense_diag = H_kin.diagonal()
print(f"\nKinetic diagonal: min={H_kin_dense_diag.real.min():.6e}, max={H_kin_dense_diag.real.max():.6e}")
print(f"Kinetic diagonal mean={H_kin_dense_diag.real.mean():.6e}")

# Compare: potential diagonal vs full diagonal
pot_diag = H_pot.diagonal().real
full_diag = H_full.diagonal().real
print(f"\nPotential diagonal: min={pot_diag.min():.6e}, max={pot_diag.max():.6e}")
print(f"Full diagonal:      min={full_diag.min():.6e}, max={full_diag.max():.6e}")

# Eigenvalues of potential-only (just sorted diagonal)
pot_evals = np.sort(pot_diag)
print(f"\n20 smallest potential-only eigenvalues (diagonal):")
for i in range(10):
    print(f"  [{i:2d}] {pot_evals[i]:.10f}")

# Now solve eigsh for each version
from scipy.sparse.linalg import eigsh

V_min = float(np.min(Lambda_b[:,:,0,0]))

print(f"\n=== Eigsh with k=20, sigma={V_min:.6f} ===")

for label, H in [("Pot only", H_pot), ("Pot+Kin", H_pk), ("Full", H_full)]:
    try:
        evals, _ = eigsh(H, k=20, sigma=V_min, which='LM')
        evals = np.sort(evals.real)
        print(f"\n{label}:")
        for i in range(5):
            print(f"  [{i:2d}] {evals[i]:.10f}  (delta = {evals[i]-V_min:.4e})")
        print(f"  ...")
        print(f"  [{len(evals)-1:2d}] {evals[-1]:.10f}")
    except Exception as e:
        print(f"\n{label}: FAILED - {e}")
