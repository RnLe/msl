#!/usr/bin/env python3
"""
S4: Term-by-term Hamiltonian build — diagnose which term breaks C4.

Strategy:
  S4.1: H = Λ only          → must have C4 (Λ is C4-symmetric)
  S4.2: H = Λ + drift       → drift ≈ 0 at Γ, should be same as S4.1
  S4.3: H = Λ + K (no A)    → tests bare M_inv kinetic operator
  S4.4: H = Λ + K (with A)  → tests Berry gauge contribution
  S4.5: H = full             → include Born-Huang

For each configuration we:
  1. Build H with the specified terms
  2. Solve for lowest ~20 eigenmodes
  3. Test C4 symmetry of each mode
  4. Report C4 quality and eigenvalue structure

C4 testing of envelope modes:
  - C4 acts on the spatial grid: (ix,iy) → ((N-iy)%N, ix)
  - Band indices are NOT mixed (energy-ordered, C4-invariant Λ diagonal)
  - For non-degenerate eigenvalues: C4·F must be proportional to F
  - For degenerate eigenvalues: C4 must map within the degenerate subspace
  - C4 eigenvalues: 1, i, -1, -i  (since C4^4 = 1)
"""

import numpy as np
import h5py
import sys, os
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add phasesV3 to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phasesV3"))
from phase3_mpb_v3 import (
    build_multiband_potential_operator,
    build_multiband_drift_operator,
    build_multiband_kinetic_operator,
    build_multiband_born_huang_operator,
    assemble_multiband_hamiltonian,
    _regularize_M_inv,
)

# ====================================================================================
# Paths  
# ====================================================================================
CAND = Path("/home/renlephy/msl/research/moire_envelope/runsV3/"
            "phase0_mpb_v3_20260206_152443/candidate_0000")
PHASE2_H5 = CAND / "phase2_multiband_data.h5"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ====================================================================================
# C4 rotation on the envelope grid
# ====================================================================================

def rotate_C4_envelope(F, Ns):
    """
    Apply C4 rotation to a multiband envelope spinor F.
    
    F has shape (Ns, Ns, N_bands).
    C4: (ix, iy) → ((Ns - iy) % Ns, ix)
    
    Band indices are NOT mixed — energy-ordered bands are C4 scalars.
    """
    Ns1, Ns2, Nb = F.shape
    assert Ns1 == Ns2 == Ns, "Only square grids"
    F_rot = np.zeros_like(F)
    for ix in range(Ns):
        for iy in range(Ns):
            # After C4 rotation, the point that was at (ix, iy) moves to ((Ns-iy)%Ns, ix)
            # So F_rot at ((Ns-iy)%Ns, ix) = F at (ix, iy)
            # Equivalently: F_rot(ix, iy) = F(iy, (Ns-ix)%Ns)  [inverse rotation]
            ix_src = iy
            iy_src = (Ns - ix) % Ns
            F_rot[ix, iy, :] = F[ix_src, iy_src, :]
    return F_rot


def apply_C4_flat(v, Ns, Nb):
    """Apply C4 to a flat eigenvector. v has length Ns*Ns*Nb."""
    F = v.reshape(Ns, Ns, Nb)
    F_rot = rotate_C4_envelope(F, Ns)
    return F_rot.ravel()


def test_C4_eigenmodes(eigenvalues, eigenvectors, Ns, Nb, n_modes=20, 
                       degen_tol=1e-6, label=""):
    """
    Test C4 symmetry of eigenmodes.
    
    For each eigenmode:
      1. Apply C4 rotation
      2. Project C4·F onto the eigenspace
      3. Check if C4·F lies entirely within the eigenspace
    
    For near-degenerate eigenvalues (|Δε| < degen_tol), group them and
    test C4 closure of the subspace.
    
    Returns:
      results: list of dicts with C4 quality per mode/group
    """
    n_modes = min(n_modes, len(eigenvalues))
    results = []
    
    # Group modes by degeneracy
    groups = []
    used = set()
    for i in range(n_modes):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, n_modes):
            if j in used:
                continue
            if abs(eigenvalues[j] - eigenvalues[i]) < degen_tol:
                group.append(j)
                used.add(j)
        groups.append(group)
    
    for group in groups:
        # Build subspace basis from this degenerate group
        V_sub = eigenvectors[:, group]  # (N_total, n_deg)
        n_deg = len(group)
        
        # For each mode in the group, apply C4 and project onto the group subspace
        overlaps = []
        c4_eigenvalues = []
        
        for idx in group:
            v = eigenvectors[:, idx]
            v_rot = apply_C4_flat(v, Ns, Nb)
            
            # Project onto the degenerate subspace: P = V_sub @ V_sub†
            # projection coefficients: c = V_sub† @ v_rot
            c = V_sub.conj().T @ v_rot
            v_proj = V_sub @ c
            
            # Quality: how much of C4·v lies in the subspace
            overlap = np.abs(np.vdot(v_proj, v_rot)) / (np.linalg.norm(v_proj) * np.linalg.norm(v_rot) + 1e-30)
            overlaps.append(overlap)
            
            # For non-degenerate modes, C4 eigenvalue = <v|C4v>
            if n_deg == 1:
                c4_eig = np.vdot(v, v_rot)
                c4_eigenvalues.append(c4_eig)
        
        mean_overlap = np.mean(overlaps)
        min_overlap = np.min(overlaps)
        
        result = {
            'indices': group,
            'eigenvalue': eigenvalues[group[0]],
            'degeneracy': n_deg,
            'c4_subspace_overlap': min_overlap,
            'c4_mean_overlap': mean_overlap,
        }
        
        if n_deg == 1:
            c4_eig = c4_eigenvalues[0]
            # C4 eigenvalue should be one of: 1, i, -1, -i
            # Check which one
            expected = [1, 1j, -1, -1j]
            labels_c4 = ['1 (A)', 'i (E+)', '-1 (B)', '-i (E-)']
            dists = [abs(c4_eig - e) for e in expected]
            best = np.argmin(dists)
            result['c4_eigenvalue'] = c4_eig
            result['c4_irrep'] = labels_c4[best]
            result['c4_eigenvalue_error'] = dists[best]
        else:
            # For degenerate groups, compute the C4 representation matrix
            C4_mat = np.zeros((n_deg, n_deg), dtype=complex)
            for a, idx_a in enumerate(group):
                v_rot = apply_C4_flat(eigenvectors[:, idx_a], Ns, Nb)
                for b, idx_b in enumerate(group):
                    C4_mat[a, b] = np.vdot(eigenvectors[:, idx_b], v_rot)
            # Trace gives character
            result['c4_trace'] = np.trace(C4_mat)
            # The representation matrix C4_mat should be unitary within the subspace
            unitarity = np.linalg.norm(C4_mat @ C4_mat.conj().T - np.eye(n_deg))
            result['c4_unitarity_error'] = unitarity
        
        results.append(result)
    
    return results


def print_c4_results(results, label):
    """Pretty-print C4 test results."""
    print(f"\n{'='*70}")
    print(f"  C4 symmetry test: {label}")
    print(f"{'='*70}")
    
    for r in results:
        idx_str = ','.join(str(i) for i in r['indices'])
        ev = r['eigenvalue']
        deg = r['degeneracy']
        c4_ov = r['c4_subspace_overlap']
        
        status = "✓" if c4_ov > 0.99 else ("~" if c4_ov > 0.9 else "✗")
        
        if deg == 1:
            irrep = r.get('c4_irrep', '?')
            c4_err = r.get('c4_eigenvalue_error', 999)
            print(f"  [{status}] Mode {idx_str:>3s}: ε={ev:+.6e}  "
                  f"C4 overlap={c4_ov:.4f}  irrep={irrep}  err={c4_err:.2e}")
        else:
            trace = r.get('c4_trace', 0)
            unit_err = r.get('c4_unitarity_error', 999)
            print(f"  [{status}] Modes {idx_str:>6s}: ε={ev:+.6e}  deg={deg}  "
                  f"C4 overlap={c4_ov:.4f}  Tr(C4)={trace:.3f}  "
                  f"unitarity_err={unit_err:.2e}")
    
    # Summary
    all_overlaps = [r['c4_subspace_overlap'] for r in results]
    n_pass = sum(1 for o in all_overlaps if o > 0.99)
    n_total = len(all_overlaps)
    mean_ov = np.mean(all_overlaps)
    min_ov = np.min(all_overlaps)
    print(f"\n  Summary: {n_pass}/{n_total} mode groups pass C4 (overlap > 0.99)")
    print(f"           mean overlap = {mean_ov:.4f}, min = {min_ov:.4f}")
    
    return mean_ov, min_ov


# ====================================================================================
# Check if H commutes with C4 directly
# ====================================================================================

def build_C4_permutation_matrix(Ns, Nb):
    """
    Build the C4 permutation matrix P_C4 such that (P_C4 @ v) applies
    C4 rotation to the flat eigenvector v.
    
    Flat index = (ix * Ns + iy) * Nb + n
    C4: (ix, iy) → ((Ns - iy) % Ns, ix)
    Band index n unchanged.
    """
    N_total = Ns * Ns * Nb
    rows = []
    cols = []
    
    for ix in range(Ns):
        for iy in range(Ns):
            for n in range(Nb):
                # Target flat index
                idx_target = (ix * Ns + iy) * Nb + n
                # Source: C4^{-1}(ix,iy) = (iy, (Ns-ix)%Ns)
                ix_src = iy
                iy_src = (Ns - ix) % Ns
                idx_source = (ix_src * Ns + iy_src) * Nb + n
                rows.append(idx_target)
                cols.append(idx_source)
    
    from scipy.sparse import coo_matrix
    data = np.ones(len(rows))
    P = coo_matrix((data, (rows, cols)), shape=(N_total, N_total)).tocsr()
    return P


def check_H_C4_commutator(H, Ns, Nb, label=""):
    """
    Compute || [H, C4] || / ||H|| to check if H commutes with C4.
    
    This is the definitive test: if [H,C4] = 0, then all eigenmodes
    must be C4 eigenstates.
    """
    P = build_C4_permutation_matrix(Ns, Nb)
    
    # [H, P] = H @ P - P @ H
    HP = H @ P
    PH = P @ H
    commutator = HP - PH
    
    # Frobenius norm
    from scipy.sparse.linalg import norm as sp_norm
    comm_norm = sp_norm(commutator, 'fro')
    H_norm = sp_norm(H, 'fro')
    
    relative = comm_norm / (H_norm + 1e-30)
    
    print(f"\n  [H, C4] test ({label}):")
    print(f"    ||[H, C4]||_F = {comm_norm:.6e}")
    print(f"    ||H||_F       = {H_norm:.6e}")
    print(f"    Relative      = {relative:.6e}")
    
    return relative


# ====================================================================================
# Additional theory check: verify prefactors
# ====================================================================================

def check_prefactors(eta, L_moire, Ns, dR):
    """
    Verify the relationship between dimensionless and physical coordinates.
    
    Theory:
      R = s · L_moire  (physical)
      dR = L_moire / Ns
      η = a / L_moire  (so L_moire = a/η)
      
    The kinetic term in the theory is:
      (η²/2) M^{-1}_{ij} (-i D_i)(-i D_j)
    where D_i = ∂/∂R_i - i A_i
    
    V3 code uses:
      prefactor = 0.5 / (2π)²
    with derivatives in physical dR.
    
    The (2π)² factor converts MPB k-units to physical:
      k_MPB = k_phys * a / (2π)
      ∂²/∂k² = (a/(2π))² ∂²/∂k_phys²
      
    But wait — the derivative is w.r.t. R, not k. Let's trace it:
      - FD Laplacian gives ∂²/∂R² with spacing dR
      - This has eigenvalues -(2πn/L)² for Fourier mode n
      - M_inv from MPB is in k-units: M_inv_phys = M_inv_MPB * (2π/a)²
      - So: K = 0.5 * M_inv_MPB * (2π/a)² * ∂²/∂R²  ... no, this isn't right
    
    Actually, let me trace more carefully:
      - Theory eigenvalue: ω_n(k₀ + Δk) ≈ ω_n(k₀) + v·Δk + ½ M_inv·Δk²
      - In MPB, k is in units of 2π/a, so M_inv_MPB = ∂²ω/∂k_MPB²
      - Physical k: k_phys = k_MPB · 2π/a
      - So: M_inv_phys = M_inv_MPB · (a/(2π))²
      
    The envelope equation with physical R:
      ½ M_inv_phys · (-i ∂/∂R)² F = ½ M_inv_MPB · (a/(2π))² · (-i ∂/∂R)² F
      
    But the envelope "momentum" q relates to Δk via: Δk = η·q (in 1/a units)
    And q = -i∂/∂R with R in units of a.
    
    Hmm, this needs very careful unit tracking. Let me check numerically.
    """
    print(f"\n  === Prefactor verification ===")
    print(f"  η = {eta:.6f}")
    print(f"  L_moire = {L_moire:.4f} a")
    print(f"  Ns = {Ns}")
    print(f"  dR = {dR:.6f} a")
    print(f"  a/η = {1.0/eta:.4f} (should ≈ L_moire)")
    print(f"  L_moire * η = {L_moire * eta:.6f} (should ≈ a = 1)")
    
    # The V3 code kinetic prefactor
    v3_prefactor = 0.5 / (2 * np.pi)**2
    print(f"\n  V3 kinetic prefactor = 0.5/(2π)² = {v3_prefactor:.6e}")
    
    # The Laplacian eigenvalue for lowest non-trivial Fourier mode:
    # -(2π/L)² = -(2π/(Ns·dR))² 
    lap_eig_lowest = -(2*np.pi / (Ns * dR))**2
    print(f"  Lowest Laplacian eigenvalue = {lap_eig_lowest:.6e}")
    
    # What the kinetic energy of the lowest mode should be:
    # E_kin = η²/(2) · M_inv · q², where q = 2π/L_moire (lowest Fourier mode)
    # In MPB units: q_MPB = q · a/(2π) = (2π/L_moire) · a/(2π) = 1/L_moire = η/a = η
    # So E_kin = 0.5 · M_inv_MPB · η² (for single-band isotropic mass)
    
    # In V3 code: E_kin = prefactor * M_inv * lap_eig = 0.5/(2π)² * M_inv * (2π/L)²
    # = 0.5 * M_inv / L² = 0.5 * M_inv / (Ns·dR)²
    # We need this to equal 0.5 * M_inv_MPB * η² for q=1 mode
    # → 0.5 * M_inv / L² vs 0.5 * M_inv * η²/a²? No...
    
    # Actually let's just compute it:
    # V3: K·F ~ prefactor * M_inv * L_eig * F = 0.5/(2π)² * M_inv * (-(2π·n/(Ns·dR))²) * F
    # = 0.5/(2π)² * M_inv * (-(2πn)²/(Ns·dR)²) * F
    # = -0.5 * M_inv * n²/(Ns·dR)² * F
    # = -0.5 * M_inv * n²/L_moire² * F   (since Ns·dR = L_moire)
    
    # Theory: E_kin = 0.5 * M_inv_phys * |q|²
    # q = 2πn/L_moire (physical wave vector in 1/a units)
    # M_inv_phys = M_inv_MPB * (a/(2π))²  [converting from MPB k-units]
    # E_kin = 0.5 * M_inv_MPB * (1/(2π))² * (2πn/L_moire)²
    # = 0.5 * M_inv_MPB * n²/L_moire²
    
    # ✓ This matches! V3 code correctly gives 0.5 * M_inv * n²/L_moire²
    print(f"\n  Kinetic energy of lowest Fourier mode (n=1):")
    print(f"    Code: 0.5/(2π)² × M_inv × (2π/L)² = 0.5 × M_inv / L² = 0.5 × M_inv × {1/L_moire**2:.6e}")
    print(f"    Theory: 0.5 × M_inv_MPB × (q·a/(2π))² = 0.5 × M_inv × η² = 0.5 × M_inv × {eta**2:.6e}")
    print(f"    η² = {eta**2:.6e}")
    print(f"    1/L² = {1/L_moire**2:.6e}")
    print(f"    Match: η² ≈ 1/L²? {abs(eta**2 - 1/L_moire**2)/eta**2:.2e} relative error")
    # Since L = a/η and a=1: 1/L² = η²/a² = η². ✓
    

# ====================================================================================
# Main diagnostic
# ====================================================================================

def main():
    print("=" * 70)
    print("  S4: TERM-BY-TERM HAMILTONIAN DIAGNOSTIC")
    print("=" * 70)
    
    # Load Phase 2 data
    print("\n[1] Loading Phase 2 data...")
    with h5py.File(PHASE2_H5, 'r') as hf:
        Lambda = hf['Lambda'][:]
        A_berry = hf['A_berry'][:]
        Phi_BH = hf['Phi_BH'][:]
        v_drift = hf['v_drift'][:]
        M_inv = hf['M_inv'][:]
        omega = hf['omega'][:]
        V = hf['V'][:]
        
        omega_ref = float(hf.attrs['omega_ref'])
        eta = float(hf.attrs['eta'])
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        Nb = int(hf.attrs['N_subspace'])
        B_moire = hf.attrs['B_moire'][:]
        target_idx = int(hf.attrs['target_index_in_subspace'])
    
    assert Ns1 == Ns2, "Require square grid"
    Ns = Ns1
    L_moire = np.linalg.norm(B_moire[0])
    dR = L_moire / Ns
    N_total = Ns * Ns * Nb
    
    print(f"  Grid: {Ns}×{Ns}, N_bands={Nb}, N_total={N_total}")
    print(f"  η = {eta:.6f}, L_moire = {L_moire:.4f} a")
    print(f"  dR = {dR:.6f} a")
    print(f"  ω_ref = {omega_ref:.6f}")
    print(f"  Target band index: {target_idx}")
    
    # Check prefactors
    check_prefactors(eta, L_moire, Ns, dR)
    
    # ================================================================
    # Check C4 symmetry of input data
    # ================================================================
    print(f"\n[2] Checking C4 symmetry of input operators...")
    
    # Λ C4 symmetry
    Lambda_C4_errs = []
    for m in range(Nb):
        for n in range(Nb):
            L_mn = Lambda[:, :, m, n]
            L_rot = np.zeros_like(L_mn)
            for ix in range(Ns):
                for iy in range(Ns):
                    ix_src = iy
                    iy_src = (Ns - ix) % Ns
                    L_rot[ix, iy] = L_mn[ix_src, iy_src]
            err = np.max(np.abs(L_rot - L_mn)) / (np.max(np.abs(L_mn)) + 1e-30)
            Lambda_C4_errs.append(err)
    print(f"  Λ C4 error (max over mn): {max(Lambda_C4_errs):.6e}")
    
    # M_inv C4 symmetry (tensor must transform as M(C4·R) = C4·M(R)·C4ᵀ)
    # For 2D: C4 rotation matrix is [[0,-1],[1,0]]
    # M'(R) = R_C4 · M(C4⁻¹R) · R_C4ᵀ
    R_C4 = np.array([[0, -1], [1, 0]], dtype=float)
    M_inv_C4_errs = []
    for n in range(Nb):
        M_nn = M_inv[:, :, n, n, :, :]  # (Ns, Ns, 2, 2)
        M_rot = np.zeros_like(M_nn)
        for ix in range(Ns):
            for iy in range(Ns):
                ix_src = iy
                iy_src = (Ns - ix) % Ns
                # M at rotated point, transformed
                M_rot[ix, iy] = R_C4 @ M_nn[ix_src, iy_src] @ R_C4.T
        err = np.max(np.abs(M_rot - M_nn)) / (np.max(np.abs(M_nn)) + 1e-30)
        M_inv_C4_errs.append(err)
    print(f"  M_inv C4 tensor error (max over bands): {max(M_inv_C4_errs):.6e}")
    for n in range(Nb):
        print(f"    Band {n}: max relative error = {M_inv_C4_errs[n]:.6e}")
    
    # A_berry C4 symmetry (vector: A(C4·R) = C4·A(R))
    A_berry_C4_errs = []
    for n in range(Nb):
        A_nn = A_berry[:, :, n, n, :]  # (Ns, Ns, 2) 
        A_rot = np.zeros_like(A_nn)
        for ix in range(Ns):
            for iy in range(Ns):
                ix_src = iy
                iy_src = (Ns - ix) % Ns
                # A is a vector: A(C4R) should = C4 · A(R)
                # C4 · (Ax, Ay) = (-Ay, Ax)
                A_rot[ix, iy, 0] = -A_nn[ix_src, iy_src, 1]
                A_rot[ix, iy, 1] = A_nn[ix_src, iy_src, 0]
        err = np.max(np.abs(A_rot - A_nn)) / (np.max(np.abs(A_nn)) + 1e-30)
        A_berry_C4_errs.append(err)
    print(f"  A_berry C4 vector error (max over bands): {max(A_berry_C4_errs):.6e}")
    for n in range(Nb):
        print(f"    Band {n}: max relative error = {A_berry_C4_errs[n]:.6e}")
    
    # v_drift C4 symmetry 
    v_drift_max = np.max(np.abs(v_drift))
    print(f"  v_drift max |value|: {v_drift_max:.6e}  (should be ~0 at Γ-point)")
    
    # Phi_BH C4 symmetry
    Phi_C4_errs = []
    for m in range(Nb):
        for n in range(Nb):
            P_mn = Phi_BH[:, :, m, n]
            P_rot = np.zeros_like(P_mn)
            for ix in range(Ns):
                for iy in range(Ns):
                    ix_src = iy
                    iy_src = (Ns - ix) % Ns
                    P_rot[ix, iy] = P_mn[ix_src, iy_src]
            err = np.max(np.abs(P_rot - P_mn)) / (np.max(np.abs(P_mn)) + 1e-30)
            Phi_C4_errs.append(err)
    print(f"  Φ_BH C4 error (max over mn): {max(Phi_C4_errs):.6e}")
    
    # ================================================================
    # Magnitude analysis of each term
    # ================================================================
    print(f"\n[3] Magnitude analysis of Hamiltonian terms...")
    
    # Potential
    Lambda_diag_range = [np.min(Lambda[:,:,n,n]) for n in range(Nb)], [np.max(Lambda[:,:,n,n]) for n in range(Nb)]
    print(f"  Λ diagonal:")
    for n in range(Nb):
        print(f"    Band {n}: [{np.min(Lambda[:,:,n,n]):.6f}, {np.max(Lambda[:,:,n,n]):.6f}]")
    
    # Off-diagonal Λ
    offdiag_max = 0
    for m in range(Nb):
        for n in range(Nb):
            if m != n:
                offdiag_max = max(offdiag_max, np.max(np.abs(Lambda[:,:,m,n])))
    print(f"  Λ off-diagonal max: {offdiag_max:.6e}")
    
    # Kinetic scale: 0.5/(2π)² * M_inv * (2π/L)² ≈ 0.5 * M_inv / L²
    typical_M = np.median(np.abs(M_inv[:,:,:,:,0,0]))
    kinetic_scale = 0.5 * typical_M / L_moire**2
    print(f"  Typical |M_inv|: {typical_M:.4f}")
    print(f"  Kinetic scale ~ 0.5*M/L² = {kinetic_scale:.6e}")
    print(f"  Potential depth ~ {np.max(Lambda[:,:,target_idx,target_idx]) - np.min(Lambda[:,:,target_idx,target_idx]):.6e}")
    print(f"  Ratio (kinetic/potential): {kinetic_scale/(np.max(Lambda[:,:,target_idx,target_idx]) - np.min(Lambda[:,:,target_idx,target_idx])):.4f}")
    
    # Berry |A|² scale
    A_diag = A_berry[:,:,:,:,:]
    A_mag = np.sqrt(np.abs(A_berry[:,:,target_idx,target_idx,0])**2 + 
                    np.abs(A_berry[:,:,target_idx,target_idx,1])**2)
    print(f"  |A_berry| for target band: median={np.median(A_mag):.4e}, max={np.max(A_mag):.4e}")
    A_sq_scale = 0.5 / (2*np.pi)**2 * typical_M * np.median(A_mag)**2
    print(f"  |A|² energy scale: {A_sq_scale:.6e}")
    
    # Born-Huang
    print(f"  Φ_BH scale: [{np.min(Phi_BH):.6e}, {np.max(Phi_BH):.6e}]")
    
    # ================================================================
    # S4.1 – S4.5: Build and test Hamiltonians
    # ================================================================
    
    configs = [
        ("S4.1: H = Λ only",           True, False, False, False),
        ("S4.2: H = Λ + drift",        True, True,  False, False),
        ("S4.3: H = Λ + K (no A)",     True, False, True,  False),  # A zeroed
        ("S4.4: H = Λ + K (with A)",   True, False, True,  True),   # A included
        ("S4.5: H = full",             True, True,  True,  True),
    ]
    # Columns: (label, include_potential, include_drift, include_kinetic, include_A)
    
    n_modes_solve = 20
    all_results = {}
    
    for label, inc_pot, inc_drift, inc_kin, inc_A in configs:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        
        # For S4.3: zero out A_berry
        if inc_kin and not inc_A:
            A_use = np.zeros_like(A_berry)
            print("  [Berry connection A set to ZERO]")
        else:
            A_use = A_berry
        
        # Build H using the assembly function
        # We use include_born_huang=True for S4.5 only
        inc_bh = (label == "S4.5: H = full")
        
        # For "Λ only", we manually build just the potential
        if not inc_drift and not inc_kin:
            H = build_multiband_potential_operator(Lambda, B_moire)
            if inc_bh:
                H = H + build_multiband_born_huang_operator(Phi_BH, eta, Ns, Ns, Nb)
            H = H.tocsr()
        else:
            # Use M_inv regularization for kinetic terms
            M_inv_use = M_inv
            if inc_kin:
                M_inv_use = _regularize_M_inv(M_inv.copy(), max_trace=20.0)
            
            H = assemble_multiband_hamiltonian(
                Lambda, v_drift if inc_drift else np.zeros_like(v_drift),
                M_inv_use if inc_kin else np.zeros_like(M_inv),
                A_use,
                Phi_BH if inc_bh else np.zeros_like(Phi_BH),
                eta, Ns, Ns, Nb, dR, dR, B_moire,
                include_drift=inc_drift,
                include_kinetic=inc_kin,
                include_born_huang=inc_bh,
            )
        
        # Enforce Hermiticity
        H = 0.5 * (H + H.conj().T)
        
        # Check [H, C4] commutator
        comm_rel = check_H_C4_commutator(H, Ns, Nb, label)
        
        # Determine sigma for shift-invert
        # Target band potential
        V_target = Lambda[:, :, target_idx, target_idx]
        target_info_M = M_inv[:, :, target_idx, target_idx, :, :]
        mean_trace = np.mean(target_info_M[:,:,0,0] + target_info_M[:,:,1,1])
        
        if mean_trace < 0:  # hole band
            sigma = float(np.max(V_target))
        else:
            sigma = float(np.min(V_target))
        
        print(f"  Solving for {n_modes_solve} modes with sigma={sigma:.6f}...")
        
        try:
            eigenvalues, eigenvectors = eigsh(H, k=n_modes_solve, sigma=sigma, 
                                               which='LM', maxiter=10000, tol=1e-10)
            # Sort
            order = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            
            print(f"  Eigenvalues (first 10):")
            for i in range(min(10, len(eigenvalues))):
                print(f"    mode {i}: ε = {eigenvalues[i]:+.8e}  (ω = {omega_ref + eigenvalues[i]:.6f})")
            
            # Test C4 of eigenmodes
            results = test_C4_eigenmodes(eigenvalues, eigenvectors, Ns, Nb,
                                          n_modes=n_modes_solve, degen_tol=1e-6, label=label)
            mean_ov, min_ov = print_c4_results(results, label)
            
            all_results[label] = {
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'c4_results': results,
                'c4_mean_overlap': mean_ov,
                'c4_min_overlap': min_ov,
                'commutator_relative': comm_rel,
            }  
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[label] = {'error': str(e)}
    
    # ================================================================
    # Summary comparison
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: C4 quality across configurations")
    print(f"{'='*70}")
    print(f"  {'Configuration':<30s}  {'[H,C4] rel':>12s}  {'C4 mean':>8s}  {'C4 min':>8s}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*8}  {'-'*8}")
    
    for label, _, _, _, _ in configs:
        if label in all_results and 'error' not in all_results[label]:
            r = all_results[label]
            print(f"  {label:<30s}  {r['commutator_relative']:>12.4e}  "
                  f"{r['c4_mean_overlap']:>8.4f}  {r['c4_min_overlap']:>8.4f}")
        else:
            err = all_results.get(label, {}).get('error', 'unknown')
            print(f"  {label:<30s}  {'ERROR':>12s}  {'—':>8s}  {'—':>8s}")
    
    # ================================================================
    # Eigenvalue comparison across configurations
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Eigenvalue comparison (first 10 modes)")
    print(f"{'='*70}")
    
    # Header
    short_labels = ["Λ only", "Λ+drift", "Λ+K(0)", "Λ+K(A)", "full"]
    header = f"  {'mode':>4s}"
    for sl in short_labels:
        header += f"  {sl:>12s}"
    print(header)
    print(f"  {'----':>4s}" + "  " + "  ".join(['-'*12]*len(short_labels)))
    
    for i in range(10):
        row = f"  {i:>4d}"
        for label, _, _, _, _ in configs:
            if label in all_results and 'error' not in all_results[label]:
                ev = all_results[label]['eigenvalues']
                if i < len(ev):
                    row += f"  {ev[i]:>+12.6e}"
                else:
                    row += f"  {'—':>12s}"
            else:
                row += f"  {'ERR':>12s}"
        print(row)
    
    # ================================================================
    # Plot results
    # ================================================================
    plot_results(all_results, configs, Ns, Nb, omega_ref)
    
    print(f"\n  Plots saved to {PLOT_DIR}/")
    print(f"\n{'='*70}")
    print(f"  S4 DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


# ====================================================================================
# Plotting
# ====================================================================================

def plot_results(all_results, configs, Ns, Nb, omega_ref):
    """Generate diagnostic plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("S4: Term-by-term Hamiltonian diagnostic", fontsize=14, fontweight='bold')
    
    short_labels = ["Λ only", "Λ+drift", "Λ+K(0)", "Λ+K(A)", "full"]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    # --- Panel 1: Eigenvalue spectrum ---
    ax = axes[0, 0]
    for idx, (label, _, _, _, _) in enumerate(configs):
        if label in all_results and 'error' not in all_results[label]:
            ev = all_results[label]['eigenvalues']
            ax.plot(range(len(ev)), ev, 'o-', markersize=4, label=short_labels[idx],
                    color=colors[idx], alpha=0.8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue ε")
    ax.set_title("Eigenvalue spectra")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: [H, C4] commutator ---
    ax = axes[0, 1]
    comm_vals = []
    comm_labels = []
    for idx, (label, _, _, _, _) in enumerate(configs):
        if label in all_results and 'error' not in all_results[label]:
            comm_vals.append(all_results[label]['commutator_relative'])
            comm_labels.append(short_labels[idx])
    ax.bar(range(len(comm_vals)), comm_vals, color=colors[:len(comm_vals)])
    ax.set_xticks(range(len(comm_labels)))
    ax.set_xticklabels(comm_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel("||[H, C4]|| / ||H||")
    ax.set_title("[H, C4] commutator (should be 0)")
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: C4 overlap per mode ---
    ax = axes[0, 2]
    for idx, (label, _, _, _, _) in enumerate(configs):
        if label in all_results and 'error' not in all_results[label]:
            results = all_results[label]['c4_results']
            mode_indices = []
            overlaps = []
            for r in results:
                for mi in r['indices']:
                    mode_indices.append(mi)
                    overlaps.append(r['c4_subspace_overlap'])
            ax.plot(mode_indices, overlaps, 'o', markersize=5, label=short_labels[idx],
                    color=colors[idx], alpha=0.7)
    ax.axhline(y=0.99, color='k', linestyle='--', alpha=0.5, label='0.99 threshold')
    ax.set_xlabel("Mode index")
    ax.set_ylabel("C4 subspace overlap")
    ax.set_title("C4 quality per eigenmode")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: Mode profiles for Λ only ---
    ax = axes[1, 0]
    label_pot = configs[0][0]
    if label_pot in all_results and 'error' not in all_results[label_pot]:
        ev = all_results[label_pot]['eigenvectors']
        for mode_idx in range(min(4, ev.shape[1])):
            F = ev[:, mode_idx].reshape(Ns, Ns, Nb)
            prob = np.sum(np.abs(F)**2, axis=2)
            if mode_idx == 0:
                im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
                ax.set_title(f"Λ only: mode 0  |F|²")
                plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    
    # --- Panel 5: Mode profiles for K(no A) ---
    ax = axes[1, 1]
    label_k = configs[2][0]
    if label_k in all_results and 'error' not in all_results[label_k]:
        ev = all_results[label_k]['eigenvectors']
        F = ev[:, 0].reshape(Ns, Ns, Nb)
        prob = np.sum(np.abs(F)**2, axis=2)
        im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
        ax.set_title(f"Λ+K(no A): mode 0  |F|²")
        plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    
    # --- Panel 6: Mode profiles for full ---
    ax = axes[1, 2]
    label_full = configs[4][0]
    if label_full in all_results and 'error' not in all_results[label_full]:
        ev = all_results[label_full]['eigenvectors']
        F = ev[:, 0].reshape(Ns, Ns, Nb)
        prob = np.sum(np.abs(F)**2, axis=2)
        im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
        ax.set_title(f"Full H: mode 0  |F|²")
        plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4_hamiltonian_termbyterm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # === Second figure: detailed mode gallery ===
    fig2, axes2 = plt.subplots(5, 4, figsize=(16, 20))
    fig2.suptitle("S4: Ground state |F|² per configuration (4 lowest modes)", fontsize=14, fontweight='bold')
    
    for row_idx, (label, _, _, _, _) in enumerate(configs):
        for col_idx in range(4):
            ax = axes2[row_idx, col_idx]
            if label in all_results and 'error' not in all_results[label]:
                ev_data = all_results[label]
                evecs = ev_data['eigenvalues']
                F = ev_data['eigenvectors'][:, col_idx].reshape(Ns, Ns, Nb)
                prob = np.sum(np.abs(F)**2, axis=2)
                im = ax.imshow(prob.T, origin='lower', cmap='hot', aspect='equal')
                
                # Get C4 quality for this mode
                c4_ov = 0
                for r in ev_data['c4_results']:
                    if col_idx in r['indices']:
                        c4_ov = r['c4_subspace_overlap']
                        break
                
                c4_str = f"C4={c4_ov:.3f}"
                irrep_str = ""
                for r in ev_data['c4_results']:
                    if col_idx in r['indices'] and 'c4_irrep' in r:
                        irrep_str = f" ({r['c4_irrep']})"
                
                ax.set_title(f"ε={evecs[col_idx]:+.4e}\n{c4_str}{irrep_str}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "ERROR", ha='center', va='center', transform=ax.transAxes)
            
            if col_idx == 0:
                ax.set_ylabel(short_labels[row_idx], fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "S4_mode_gallery.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved S4_hamiltonian_termbyterm.png")
    print(f"  Saved S4_mode_gallery.png")


if __name__ == "__main__":
    main()
