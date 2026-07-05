#!/usr/bin/env python3
"""Check the scale of different terms in the envelope Hamiltonian."""
import h5py
import numpy as np

with h5py.File("runsV3/phase0_mpb_v3_20260203_084230/candidate_0000/phase2_multiband_data.h5", "r") as f:
    Lambda = f["Lambda"][:]
    M_inv = f["M_inv"][:]
    eta = float(f.attrs["eta"])
    B_moire = f.attrs["B_moire"]

Ns1, Ns2 = Lambda.shape[:2]
ds1 = 1.0 / Ns1

print("="*60)
print("SCALE ANALYSIS FOR ENVELOPE HAMILTONIAN")
print("="*60)
print()
print(f"Grid: {Ns1} x {Ns2}")
print(f"ds1 = {ds1:.6f}")
print(f"eta = {eta:.6f}")
print(f"eta^2 = {eta**2:.8f}")
print()
print("B_moire:")
print(B_moire)
L_moire = np.linalg.norm(B_moire, axis=1)
print(f"L_moire magnitudes: {L_moire}")
print()

print("="*60)
print("EXPECTED TERM MAGNITUDES (from theory)")
print("="*60)
print()
print("Theory from 5_FinalMultiBandTwoScaleEA.md:")
print("  H = Lambda(R) + eta*v*D + eta^2*(1/2)*D_i*M_ij*D_j + eta^2*Phi_BH")
print()
print("Where D = -i*partial_R (in physical Cartesian coordinates)")
print()

pot_scale = np.abs(Lambda[:,:,2,2]).mean()
M_inv_scale = np.abs(M_inv[:,:,2,2,:,:]).max()
print(f"  Potential Lambda ~", pot_scale)
print(f"  M_inv max ~", M_inv_scale)
print()

# The problem
print("="*60)
print("THE COORDINATE TRANSFORMATION ISSUE")
print("="*60)
print()
print("We work in fractional coordinates s in [0,1]^2")
print("Physical coordinates: R = B_moire @ s")
print()
print("Gradient transformation:")
print("  partial_R_i = (B_moire_inv)_ij * partial_s_j")
print("  partial_s_i = B_moire_ij * partial_R_j")
print()
print(f"  ||B_moire|| ~ L_moire ~ {L_moire[0]:.2f} (in units of a)")
print(f"  This means: partial_s ~ L_moire * partial_R")
print(f"              partial_s^2 ~ L_moire^2 * partial_R^2")
print()

print("="*60)
print("WHAT THE CODE DOES vs WHAT IT SHOULD DO")
print("="*60)
print()
print("KINETIC TERM:")
print("-" * 40)
print("Code computes: K = eta^2 * (1/2) * M_inv * d^2/ds^2")
print()
print("But d/ds = B_moire @ d/dR, so d^2/ds^2 ~ L_moire^2 * d^2/dR^2")
print()
print(f"  L_moire ~ 1/eta ~ {1/eta:.2f}")
print(f"  L_moire^2 ~ 1/eta^2 ~ {1/eta**2:.2f}")
print()
print("So effective kinetic:")
print(f"  K_eff = eta^2 * L^2 * M * d^2/dR^2")
print(f"        = eta^2 * (1/eta^2) * M * d^2/dR^2")
print(f"        = M * d^2/dR^2  (eta^2 cancels!)")
print()
print(f"Expected kinetic: K = eta^2 * M * d^2/dR^2  (small)")
print()
print(f"RATIO: code/expected = 1/eta^2 = {1/eta**2:.2f}")
print()
print("This explains why eigenvalues are ~2700x too large!")
print()

print("="*60)
print("NUMERICAL CHECK")
print("="*60)
print()

# What eigenvalues should look like
# If Lambda ~ 0.05 is the dominant term, eigenvalues should be O(0.01-0.1)
# With kinetic correction ~ eta^2 * M * k^2 ~ 0.0004 * 50 * (2*pi)^2 ~ 0.8
# Total: eigenvalue ~ 0.1 + 0.8 = O(1) at most

# But code gives eigenvalue ~ -21 which is:
# Lambda ~ 0.05
# Kinetic (wrong) ~ M * k^2 ~ 50 * 40 ~ 2000 scaled by ds^2 factor...

# Actually, let me compute what the kinetic term contributes
# d^2/ds^2 with ds = 1/128 gives factor (128)^2 = 16384
# Divided by 2*pi^2 or something similar from periodic BC

# With periodic BC, momentum modes are k = 2*pi*n (in fractional units)
# <k^2> ~ (2*pi)^2 * <n^2>
# For localized state, <n^2> ~ (width in k-space)^2 ~ (1/width_real)^2

print("Expected eigenvalue magnitude:")
print(f"  Lambda contribution: ~{pot_scale:.4f}")
print(f"  Kinetic (correct): eta^2 * M * k^2 ~ {eta**2 * M_inv_scale * (2*np.pi)**2:.4f}")
print(f"  Total expected: O(0.01 - 1)")
print()
print("Observed eigenvalue: ~21")
print(f"Ratio: 21 / 0.1 ~ 200 (order of 1/eta^2 = {1/eta**2:.0f})")
print()

print("="*60)
print("FIX REQUIRED")
print("="*60)
print()
print("Option 1: Scale the kinetic term by eta^2")
print("  Currently: prefactor = 0.5 * eta^2")
print("  Should be: prefactor = 0.5 * eta^4")
print("  (extra eta^2 to convert from s-derivatives to R-derivatives)")
print()
print("Option 2: Use B_moire metric tensor")
print("  Properly transform the Laplacian from R to s coordinates")
print("  -partial_R^2 = -B_moire_inv_ij * B_moire_inv_kl * partial_si * partial_sk")
print()
print("Option 3: Work directly in physical R coordinates")
print("  (requires irregular grid or different discretization)")
