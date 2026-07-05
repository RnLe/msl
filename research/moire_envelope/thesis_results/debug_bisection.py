#!/usr/bin/env python
"""
Debug script: Compare bisection-style solve vs eta_sweep at reference angle.
Tests whether "load Phase 2 once + reuse" gives the same result as
loading from the per-angle Phase 2 HDF5.
"""
import sys, math, numpy as np, h5py, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phasesV3"))
import phase3_mpb_v3 as p3

RUN_DIR = Path(__file__).resolve().parent.parent / (
    "runsV3/thesis_honeycomb_K_b1_20260307_171424"
)

# ─── Load Phase 2 data (the STORED version at theta_ref=1.1°) ────────────
h5_path = RUN_DIR / "candidate_0000" / "phase2_multiband_data.h5"
with h5py.File(h5_path, 'r') as hf:
    Lambda = hf['Lambda'][:]
    A_berry = hf['A_berry'][:]
    Phi_BH = hf['Phi_BH'][:]
    v_drift = hf['v_drift'][:]
    M_inv = hf['M_inv'][:]
    N_sub = int(hf.attrs['N_subspace'])
    eta_ref = float(hf.attrs['eta'])
    B_moire_ref = hf.attrs['B_moire']
    Ns = int(hf.attrs['Ns1'])

L_moire_ref = np.linalg.norm(B_moire_ref[0])
dR_ref = L_moire_ref / Ns
print(f"Phase 2 stored: theta=1.1°, eta={eta_ref:.6f}, L_moire={L_moire_ref:.4f}")
print(f"  Ns={Ns}, N_sub={N_sub}, dR={dR_ref:.4f}")
print(f"  B_moire_ref = {B_moire_ref}")

# ─── Method A: Direct assembly at reference angle (should match seed) ─────
print("\n=== METHOD A: Direct Phase 3 at reference angle θ=1.1° ===")
t0 = time.time()
H_A = p3.assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv, A_berry, Phi_BH,
    eta_ref, Ns, Ns, N_sub, dR_ref, dR_ref, B_moire_ref,
    include_drift=True, include_kinetic=True, include_born_huang=False,
    order=4, include_offdiag_A=True,
)
print(f"  H built in {time.time()-t0:.1f}s, shape={H_A.shape}, nnz={H_A.nnz}")
print(f"  H diagonal range: [{H_A.diagonal().min():.6f}, {H_A.diagonal().max():.6f}]")
evals_A, _ = p3.solve_multiband_envelope(H_A, 20, sigma=None)
gap_A = float(evals_A[1] - evals_A[0])
bw_A = float(evals_A[-1] - evals_A[0])
print(f"  gap={gap_A:.6e}, BW={bw_A:.6e}")
print(f"  evals[:5] = {evals_A[:5]}")

# ─── Method B: Load from sweep theta=1.1 (the "ground truth") ────────────
# Find the per-angle Phase 2/3 from the sweep that produced the seed data
sweep_dirs = sorted(RUN_DIR.glob("eta_sweep_*"))
theta_match = None
for sd in sweep_dirs:
    theta_dir = sd / "theta_1.100" / "candidate_0000" / "phase2_multiband_data.h5"
    if theta_dir.exists():
        theta_match = theta_dir
        break
    # Also try theta_1.1000
    theta_dir2 = sd / "theta_1.1000" / "candidate_0000" / "phase2_multiband_data.h5" 
    if theta_dir2.exists():
        theta_match = theta_dir2
        break

if theta_match:
    print(f"\n=== METHOD B: From sweep HDF5: {theta_match} ===")
    with h5py.File(theta_match, 'r') as hf:
        Lambda_B = hf['Lambda'][:]
        A_berry_B = hf['A_berry'][:]
        Phi_BH_B = hf['Phi_BH'][:]
        v_drift_B = hf['v_drift'][:]
        M_inv_B = hf['M_inv'][:]
        N_sub_B = int(hf.attrs['N_subspace'])
        eta_B = float(hf.attrs['eta'])
        B_moire_B = hf.attrs['B_moire']
        Ns_B = int(hf.attrs['Ns1'])
    L_moire_B = np.linalg.norm(B_moire_B[0])
    dR_B = L_moire_B / Ns_B
    print(f"  eta={eta_B:.6f}, L_moire={L_moire_B:.4f}, Ns={Ns_B}")
    print(f"  B_moire = {B_moire_B}")
    
    H_B = p3.assemble_multiband_hamiltonian(
        Lambda_B, v_drift_B, M_inv_B, A_berry_B, Phi_BH_B,
        eta_B, Ns_B, Ns_B, N_sub_B, dR_B, dR_B, B_moire_B,
        include_drift=True, include_kinetic=True, include_born_huang=False,
        order=4, include_offdiag_A=True,
    )
    evals_B, _ = p3.solve_multiband_envelope(H_B, 20, sigma=None)
    gap_B = float(evals_B[1] - evals_B[0])
    bw_B = float(evals_B[-1] - evals_B[0])
    print(f"  gap={gap_B:.6e}, BW={bw_B:.6e}")
    print(f"  evals[:5] = {evals_B[:5]}")
    
    # Compare data
    print(f"\n=== COMPARISON ===")
    print(f"  B_moire match: {np.allclose(B_moire_ref, B_moire_B, atol=1e-4)}")
    print(f"  Lambda match: {np.allclose(Lambda, Lambda_B, atol=1e-6)}")
    print(f"  A_berry match: {np.allclose(A_berry, A_berry_B, atol=1e-6)}")
    print(f"  Lambda max diff: {np.max(np.abs(Lambda - Lambda_B)):.6e}")
    print(f"  A_berry max diff: {np.max(np.abs(A_berry - A_berry_B)):.6e}")
    print(f"  v_drift max diff: {np.max(np.abs(v_drift - v_drift_B)):.6e}")
    print(f"  M_inv max diff: {np.max(np.abs(M_inv - M_inv_B)):.6e}")
else:
    print("\nNo per-angle sweep data found at theta=1.1 for comparison")

# ─── Also check what the SEED says for theta=1.1 ─────────────────────────
import json
seed_data = {}
for sd in sweep_dirs:
    res_file = sd / "sweep_results.json"
    if res_file.exists():
        with open(res_file) as f:
            data = json.load(f)
        for pt in data.get("results", []):
            theta = pt.get("theta_deg", 0)
            if abs(theta - 1.1) < 0.01:
                seed_data = pt
                break

if seed_data:
    print(f"\n=== SEED DATA at theta=1.1° ===")
    print(f"  gap_01: {seed_data.get('gap_01', 'N/A')}")
    print(f"  bandwidth: {seed_data.get('bandwidth', 'N/A')}")
    evals_seed = seed_data.get('eigenvalues', [])
    if evals_seed:
        print(f"  evals[:5]: {evals_seed[:5]}")
