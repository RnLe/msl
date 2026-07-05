
import h5py
import numpy as np
import sys

def check_norm(h5_path):
    print(f"Checking {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        if 'bloch_fields' not in f:
            print("No bloch_fields found.")
            return
            
        # Shape: (Ns1, Ns2, N_bands, Nx, Ny, 3)
        bf = f['bloch_fields'][:]
        print(f"Shape: {bf.shape}")
        
        # Norm over spatial grid (Nx, Ny)
        # Assuming discrete norm sum |u|^2 = 1? Or integral?
        # usually sum |u|^2 * dV = 1.
        # Here we just check the sum of squares.
        
        norm_sq = np.sum(np.abs(bf)**2, axis=(3, 4, 5))
        print(f"Norm Sq Range: [{np.min(norm_sq):.6e}, {np.max(norm_sq):.6e}]")
        print(f"Mean Norm Sq: {np.mean(norm_sq):.6e}")
        
        if np.abs(np.mean(norm_sq) - 1.0) > 0.1:
            print("WARNING: Fields do not appear to be normalized to 1.")

if __name__ == "__main__":
    check_norm(sys.argv[1])
