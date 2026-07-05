
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def log(msg):
    print(msg)

def inspect_phase3(h5_path):
    log(f"=== Inspecting Phase 3 Data: {h5_path} ===")
    
    if not os.path.exists(h5_path):
        log(f"ERROR: File not found {h5_path}")
        return

    with h5py.File(h5_path, 'r') as hf:
        log(f"Keys: {list(hf.keys())}")
        
        psi_key = 'eigenvectors' if 'eigenvectors' in hf else 'envelope_functions'
        if psi_key not in hf:
            log("No eigenvectors found.")
            return

        psi = hf[psi_key][:] # (N_modes, Ns1, Ns2, N_bands) or flattened?
        eigenvalues = hf['eigenvalues'][:]
        
        log(f"Modes shape: {psi.shape}")
        log(f"Eigenvalues: {eigenvalues}")
        
        # If shape is flattened (N_modes, Total_DOF), reshape
        if len(psi.shape) == 2:
             # Assume Total_DOF = Ns1 * Ns2 * N_bands
             # We need Ns1, Ns2, N_bands from metadata or Phase 1
             # Guessing 128x128x4 from Phase 1 inspection
             Ns1, Ns2, Nb = 128, 128, 4
             N_modes = psi.shape[0]
             psi = psi.reshape(N_modes, Ns1, Ns2, Nb)
        
        # Analyze Mode 0
        psi0 = psi[0]
        prob = np.sum(np.abs(psi0)**2, axis=-1) # Sum over bands (Ns1, Ns2)
        norm = np.sum(prob)
        prob = prob / norm
        
        # IPR (Spread measure)
        # IPR = Sum p_i^2. High IPR = Localized. Low IPR = Delocalized.
        # Number of participating pixels = 1/IPR
        ipr = np.sum(prob**2)
        participation = 1.0 / ipr
        
        log(f"\n--- Mode 0 Analysis ---")
        log(f"Eigenvalue: {eigenvalues[0]:.6e}")
        log(f"Participation Ratio (pixels): {participation:.2f} / {prob.size}")
        
        # Check "Sprinkled Dots" (multi-peak?)
        # Count local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(prob, size=3) == prob
        num_peaks = np.sum(local_max & (prob > 0.01 * np.max(prob)))
        log(f"Number of significant peaks: {num_peaks}")
        
        # Check Kinetic Scale vs Potential Scale
        # We can't easily reconstruct the operator here without code, 
        # but we can check if the eigenvalue is close to V_min.
        
        # Load Potential from Phase 2 (passed through metadata or we assume we have it)
        # It's not in Phase 3 h5 usually.
        # We assume V min is around -0.13 (from Phase 1 inspection).
        # Eigenvalue should be > V_min.
        expected_min = -0.135
        log(f"E - V_min_est: {eigenvalues[0] - expected_min:.6e}")
        
        # If E is very close to V_min, it means Kinetic energy is tiny -> Localization.
        # If E is far above, Kinetic is large.
        
        if eigenvalues[0] < expected_min:
             log("!!! WARNING: Eigenvalue BELOW potential minimum? (Could be large negative mass effect?) !!!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_phase3.py <h5_file>")
        sys.exit(1)
    
    inspect_phase3(sys.argv[1])
