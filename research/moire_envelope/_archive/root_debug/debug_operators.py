
import sys
import os
import numpy as np
from scipy import sparse

# Add path to find the module
# hardcoded for safety
sys.path.append('/home/renlephy/msl/research/moire_envelope/blaze_phasesV3')

from phase3_blaze_v3 import (
    build_multiband_drift_operator,
    build_multiband_kinetic_operator,
    build_multiband_potential_operator,
    build_periodic_derivative_matrix,
    build_periodic_laplacian_matrix
)

def test_operators():
    Ns1, Ns2 = 3, 3
    N_bands = 2
    ds1, ds2 = 1.0, 1.0
    eta = 0.1
    
    print("Testing Operators on 3x3 grid, 2 bands...")
    
    # 1. Potential
    # random potential
    Lambda = np.random.rand(Ns1, Ns2, N_bands, N_bands) + 1j*0.0
    # Make it hermitian in band indices
    for i in range(Ns1):
        for j in range(Ns2):
            Lambda[i,j] = 0.5*(Lambda[i,j] + Lambda[i,j].conj().T)
            
    V_op = build_multiband_potential_operator(Lambda, None)
    print(f"Potential Hermitian? {np.allclose(V_op.toarray(), V_op.toarray().conj().T)}")
    print(f"Potential Diagonal in Space? (Checked visually or nnz)")
    # It should be block diagonal size (Nb x Nb) blocks.
    # Total size Ns*Nb.
    # Check off-diagonal blocks between spatial sites
    V_dense = V_op.toarray()
    # Check if V_dense[r, c] is 0 if spatial index differs.
    # r = s*Nb + b. s = r // Nb.
    is_local = True
    rows, cols = V_op.nonzero()
    for r, c in zip(rows, cols):
        s_r = r // N_bands
        s_c = c // N_bands
        if s_r != s_c:
            is_local = False
            break
    print(f"Potential Local in Space? {is_local}")
    
    # 2. Drift
    # v_drift (Ns1, Ns2, Nb, Nb, 2)
    # Make it constant for simplicity check
    v_drift = np.zeros((Ns1, Ns2, N_bands, N_bands, 2))
    # Band 0 has velocity in x (dim 0)
    v_drift[:,:,0,0,0] = 1.0
    
    T_op = build_multiband_drift_operator(v_drift, eta, Ns1, Ns2, N_bands, ds1, ds2)
    
    # Check Hermiticity (Drift is usually hermitian if v is constant)
    # T = -i v d/dx.
    # (v p)^dag = p v = v p (constant).
    # -i d/dx is hermitian.
    # So T should be hermitian.
    T_dense = T_op.toarray()
    diff = np.max(np.abs(T_dense - T_dense.conj().T))
    print(f"Drift Hermitian Error: {diff}")
    
    # Check sparsity pattern.
    # Should couple s to s+1, s-1 (periodic).
    # Band 0 only.
    rows, cols = T_op.nonzero()
    # Check band mixing
    # v is diagonal in bands => T should be diagonal in bands
    is_band_diag = True
    for r, c in zip(rows, cols):
        b_r = r % N_bands
        b_c = c % N_bands
        if b_r != b_c:
            is_band_diag = False
    print(f"Drift Band Diagonal? {is_band_diag}")
    
    # 3. Kinetic
    # M_inv diagonal in bands, identity
    M_inv = np.zeros((Ns1, Ns2, N_bands, N_bands, 2, 2))
    for i in range(Ns1):
        for j in range(Ns2):
            for n in range(N_bands):
                M_inv[i,j,n,n,0,0] = 1.0
                M_inv[i,j,n,n,1,1] = 1.0
    
    A_berry = np.zeros((Ns1, Ns2, N_bands, N_bands, 2))
    
    K_op = build_multiband_kinetic_operator(M_inv, A_berry, eta, Ns1, Ns2, N_bands, ds1, ds2, None)
    
    K_dense = K_op.toarray()
    diffK = np.max(np.abs(K_dense - K_dense.conj().T))
    print(f"Kinetic Hermitian Error: {diffK}")
    
    # Check if it looks like a Laplacian
    # internal logic check
    print("Test Complete")

if __name__ == "__main__":
    test_operators()
