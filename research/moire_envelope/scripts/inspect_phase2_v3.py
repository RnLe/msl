
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def log(msg):
    print(msg)

def inspect_phase2(h5_path):
    log(f"=== Inspecting Phase 2 Data: {h5_path} ===")
    
    if not os.path.exists(h5_path):
        log(f"ERROR: File not found {h5_path}")
        return

    with h5py.File(h5_path, 'r') as hf:
        # 1. Born-Huang
        # Shape (Ns1, Ns2, N_sub, N_sub)
        if 'Phi_BH' in hf:
            Phi_BH = hf['Phi_BH'][:]
            log("\n--- Born-Huang Potential ---")
            log(f"Shape: {Phi_BH.shape}")
            
            # Diagonal elements
            Phi_diag = np.diagonal(Phi_BH, axis1=2, axis2=3) # (Ns1, Ns2, N_sub)
            Phi_diag = np.moveaxis(Phi_diag, 2, 0) # (N_sub, Ns1, Ns2)
            
            for n in range(min(4, Phi_diag.shape[0])):
                P = Phi_diag[n]
                log(f"Band {n} BH range: [{np.min(P):.6e}, {np.max(P):.6e}]")
                
                # Check Smoothness
                diff = np.abs(np.roll(P, -1, axis=0) - P)
                log(f"Band {n} BH Max Adjacent Diff: {np.max(diff):.6e}")
                
                # Check Symmetry (Rot90)
                P_rot = np.rot90(P, k=-1)
                sym_err = np.mean(np.abs(P - P_rot))
                log(f"Band {n} BH Symmetry Error: {sym_err:.6e}")
        else:
            log("No Phi_BH found.")

        # 2. Berry Connection
        if 'A_berry' in hf:
            A = hf['A_berry'][:] # (Ns1, Ns2, N, N, 2)
            log("\n--- Berry Connection ---")
            # Diagonal A is Im<u|du>. Real part should be 0 or small?
            # A_berry_nn is usually Real if u is normalized? No, A is Hermitian, diagonal is Real.
            # Im(i <u|du>) -> Re <u|du> = 0. So A should be Real.
            
            A_real_mag = np.mean(np.abs(np.real(A)))
            A_imag_mag = np.mean(np.abs(np.imag(A)))
            log(f"Mean |Re(A)|: {A_real_mag:.6e}")
            log(f"Mean |Im(A)|: {A_imag_mag:.6e} (Should be small/zero for diagonal)")
            
            # Check smoothness of A_diagonal
            A_diag = A[:,:,0,0,0] # Band 0, comp 0 (Ax)
            diff = np.abs(np.roll(A_diag, -1, axis=0) - A_diag)
            log(f"A_x (Band 0) Max Adjacent Diff: {np.max(diff):.6e}")
            
            log(f"A range: [{np.min(np.real(A)):.6e}, {np.max(np.real(A)):.6e}]")

        # 3. Mass Tensor again (just to verify it corresponds to Phi phase)
        M_inv = hf['M_inv'][:]
        # Check if M_inv is diagonal in bands? (Code assumed block diagonal usually)
        
        # 4. Drift V
        if 'v_drift' in hf:
            v_drift = hf['v_drift'][:]
            log(f"Drift Max: {np.max(np.abs(v_drift)):.6e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_phase2.py <h5_file>")
        sys.exit(1)
    
    inspect_phase2(sys.argv[1])
