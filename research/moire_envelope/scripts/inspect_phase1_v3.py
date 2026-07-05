
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def log(msg):
    print(msg)

def inspect_phase1(h5_path):
    log(f"=== Inspecting Phase 1 Data: {h5_path} ===")
    
    if not os.path.exists(h5_path):
        log(f"ERROR: File not found {h5_path}")
        return

    with h5py.File(h5_path, 'r') as hf:
        # 1. Metadata
        log("\n--- Metadata ---")
        for key in ['eta', 'theta_deg', 'Ns1', 'Ns2', 'N_subspace', 'omega_ref', 'a', 'moire_length']:
            if key in hf.attrs:
                log(f"{key}: {hf.attrs[key]}")
        
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        N_subspace = int(hf.attrs['N_subspace'])

        # 2. Data Ranges
        log("\n--- Data Ranges and Statistics ---")
        
        # Omega / Potential
        omega = hf['omega'][:]
        V = hf['V'][:]
        log(f"Omega shape: {omega.shape}")
        log(f"Omega range: [{np.min(omega):.6e}, {np.max(omega):.6e}]")
        log(f"V (Potential) range: [{np.min(V):.6e}, {np.max(V):.6e}]")
        
        # Check smoothness of V
        diff_V_1 = np.abs(np.roll(V, -1, axis=0) - V)
        diff_V_2 = np.abs(np.roll(V, -1, axis=1) - V)
        max_diff_V = max(np.max(diff_V_1), np.max(diff_V_2))
        log(f"Max adjacent difference in V: {max_diff_V:.6e}")
        
        # Symmetry of V
        # Square lattice check (90 deg rotation)
        # V[i, j] should be close to V[N-1-j, i] (approx)
        V0 = V[:,:,0] # 1st band
        V0_rot = np.rot90(V0, k=-1)
        sym_error = np.mean(np.abs(V0 - V0_rot))
        log(f"C4 Symmetry Error (mean abs diff): {sym_error:.6e}")
        
        # Velocity
        vg = hf['vg'][:]
        log(f"Group Velocity range: [{np.min(vg):.6e}, {np.max(vg):.6e}]")
        
        # Mass
        M_inv = hf['M_inv'][:] # (Ns1, Ns2, N, 2, 2)
        log(f"Mass Tensor (inv) range: [{np.min(M_inv):.6e}, {np.max(M_inv):.6e}]")
        
        # Check if Mass is positive definite (for band minima)
        # Trace and Det
        tr = M_inv[..., 0, 0] + M_inv[..., 1, 1]
        det = M_inv[..., 0, 0]*M_inv[..., 1, 1] - M_inv[..., 0, 1]*M_inv[..., 1, 0]
        log(f"Mass Trace range: [{np.min(tr):.6e}, {np.max(tr):.6e}]")
        log(f"Mass Det range: [{np.min(det):.6e}, {np.max(det):.6e}]")
        
        # 3. Bloch Fields Analysis (The "Noise" Suspect)
        if 'bloch_fields' in hf:
            log("\n--- Bloch Fields Analysis (Phase 1 Raw) ---")
            fields = hf['bloch_fields'][:]
            # Shape (N_reg, N_reg, N_bands, Nx, Ny, 3) probably
            # Wait, phase1_mpb_v3 saves it as (n_reg, n_reg, N_bands, N_cells_x, N_cells_y, 3) ?
            # Let's check shape
            log(f"Bloch Fields shape: {fields.shape}")
            
            # Use mid-band
            b_idx = 0
            # Spatial inner product at each R
            # field[i, j, b] is (Nx, Ny, 3)
            
            # Check phase correlations between R=(0,0) and R=(0,1)
            u00 = fields[0, 0, b_idx].flatten()
            u01 = fields[0, 1, b_idx].flatten()
            
            norm00 = np.vdot(u00, u00).real
            norm01 = np.vdot(u01, u01).real
            log(f"Norm at (0,0): {norm00:.6f}")
            log(f"Norm at (0,1): {norm01:.6f}")
            
            overlap = np.vdot(u00, u01)
            phase_jump = np.angle(overlap)
            log(f"Overlap <u(0,0)|u(0,1)>: {overlap:.6e}")
            log(f"Phase jump (rad): {phase_jump:.6f}")
            log(f"Abs(Overlap)/Norm: {np.abs(overlap)/np.sqrt(norm00*norm01):.6f}")
            
            # Global Phase Smoothness metric
            # Compute phase jumps along a line
            phases = []
            n_reg = fields.shape[0]
            for i in range(n_reg-1):
                u_curr = fields[i, 0, b_idx].flatten()
                u_next = fields[i+1, 0, b_idx].flatten()
                ov = np.vdot(u_curr, u_next)
                phases.append(np.angle(ov))
                
            phases = np.array(phases)
            log(f"Phase jumps statistic: Mean={np.mean(phases):.4f}, Std={np.std(phases):.4f}")
            log(f"Max phase jump: {np.max(np.abs(phases)):.4f}")
            if np.std(phases) > 1.0:
                log("!!! WARNING: High phase noise detected (Expected in Raw Phase 1) !!!")
        else:
            log("\nNo Bloch fields stored in Phase 1 file.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_phase1.py <h5_file>")
        sys.exit(1)
    
    inspect_phase1(sys.argv[1])
