#!/usr/bin/env python3
"""
S5: Mode analysis & eigenvalue clustering diagnostic.
=====================================================

Now that S4b confirmed C4-symmetrization works (all modes carry clean C4 irreps),
we ask: are these modes *physically meaningful*?

Questions addressed:
  1. Mode localization: IPR, real-space |F(R)|², localization length
  2. Band composition: which subspace bands contribute to each mode?
  3. Energy scales: V_depth vs E_kin — why eigenvalues cluster
  4. Free-particle analytical cross-check (Λ=const, M=const, A=0)
  5. Grid convergence: Ns=128 vs 64 vs 32

Uses: phase2_multiband_data_c4sym.h5 (C4-symmetrized Phase 2 data from S4b)
"""

import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse.linalg import eigsh
from scipy.signal import fftconvolve

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "phasesV3"))

from phase3_mpb_v3 import (
    assemble_multiband_hamiltonian,
    _regularize_M_inv,
    build_multiband_potential_operator,
    build_multiband_kinetic_operator,
)

RUN_DIR = REPO / "runsV3" / "phase0_mpb_v3_20260206_152443"
CAND    = RUN_DIR / "candidate_0000"
H5_SYM  = CAND / "phase2_multiband_data_c4sym.h5"
H5_ORIG = CAND / "phase2_multiband_data.h5"
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════════

def load_phase2(h5_path):
    """Load Phase 2 HDF5 data and metadata."""
    with h5py.File(h5_path, 'r') as hf:
        data = {
            'Lambda':   hf['Lambda'][:],
            'A_berry':  hf['A_berry'][:],
            'Phi_BH':   hf['Phi_BH'][:],
            'v_drift':  hf['v_drift'][:],
            'M_inv':    hf['M_inv'][:],
            'omega':    hf['omega'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta':       float(hf.attrs['eta']),
            'Ns1':       int(hf.attrs['Ns1']),
            'Ns2':       int(hf.attrs['Ns2']),
            'Nb':        int(hf.attrs['N_subspace']),
            'B_moire':   hf.attrs['B_moire'][:],
            'target_idx': int(hf.attrs['target_index_in_subspace']),
        }
    Ns = data['Ns1']
    L_moire = 1.0 / data['eta']   # a / η
    dR = L_moire / Ns
    data['L_moire'] = L_moire
    data['dR'] = dR
    data['Ns'] = Ns
    return data


def downsample_field(field, factor):
    """Downsample a field on (Ns, Ns, ...) grid by integer factor via block averaging."""
    Ns = field.shape[0]
    Ns_new = Ns // factor
    shape_extra = field.shape[2:]
    result = np.zeros((Ns_new, Ns_new) + shape_extra, dtype=field.dtype)
    for i in range(factor):
        for j in range(factor):
            result += field[i::factor, j::factor, ...]
    result /= factor**2
    return result


def build_H_at_resolution(data, Ns_target, include_A=False):
    """Build Hamiltonian at a given resolution, downsampling if needed."""
    Ns = data['Ns']
    factor = Ns // Ns_target
    eta = data['eta']
    Nb = data['Nb']
    
    if factor == 1:
        Lambda = data['Lambda']
        v_drift = data['v_drift']
        M_inv = data['M_inv']
        A_berry = data['A_berry'] if include_A else np.zeros_like(data['A_berry'])
        Phi_BH = np.zeros_like(data['Phi_BH'])  # skip BH for simplicity
    else:
        Lambda = downsample_field(data['Lambda'], factor)
        v_drift = downsample_field(data['v_drift'], factor)
        M_inv = downsample_field(data['M_inv'], factor)
        A_berry = downsample_field(data['A_berry'], factor) if include_A else np.zeros((Ns_target, Ns_target, Nb, Nb, 2))
        Phi_BH = np.zeros((Ns_target, Ns_target, Nb, Nb))
    
    L_moire = data['L_moire']
    dR = L_moire / Ns_target
    B_moire = data['B_moire']
    
    M_inv_reg = _regularize_M_inv(M_inv.copy(), 20.0)
    
    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, A_berry, Phi_BH,
        eta, Ns_target, Ns_target, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
    )
    H = 0.5 * (H + H.conj().T)
    return H


def compute_ipr(eigvec, Ns, Nb):
    """
    Compute Inverse Participation Ratio for each eigenmode.
    IPR = sum |F|^4 / (sum |F|^2)^2
    IPR → 1/N_sites for fully extended, → 1 for fully localized.
    """
    n_modes = eigvec.shape[1]
    ipr = np.zeros(n_modes)
    for m in range(n_modes):
        psi = eigvec[:, m]
        # Reshape to (Ns, Ns, Nb)
        F = psi.reshape(Ns, Ns, Nb)
        # Probability density on spatial grid (sum over bands)
        rho = np.sum(np.abs(F)**2, axis=2)  # (Ns, Ns)
        ipr[m] = np.sum(rho**2) / np.sum(rho)**2
    return ipr


def compute_band_weights(eigvec, Ns, Nb):
    """
    For each eigenmode, compute fractional weight in each subspace band.
    Returns (n_modes, Nb) array.
    """
    n_modes = eigvec.shape[1]
    weights = np.zeros((n_modes, Nb))
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        for n in range(Nb):
            weights[m, n] = np.sum(np.abs(F[:, :, n])**2)
        weights[m] /= np.sum(weights[m])
    return weights


def compute_localization_length(eigvec, Ns, Nb, dR):
    """
    RMS spread of each eigenmode around its center of mass.
    ξ = sqrt(<r²> - <r>²), in units of L_moire.
    """
    n_modes = eigvec.shape[1]
    L = Ns * dR
    xi = np.zeros(n_modes)
    
    # Grid coordinates
    x = np.arange(Ns) * dR
    y = np.arange(Ns) * dR
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        rho = np.sum(np.abs(F)**2, axis=2)  # (Ns, Ns)
        rho /= np.sum(rho) * dR**2  # normalize
        
        # Center of mass (periodic handling via circular mean)
        # Use angle method for periodic coordinates
        theta_x = 2 * np.pi * X / L
        theta_y = 2 * np.pi * Y / L
        
        cos_x = np.sum(rho * np.cos(theta_x)) * dR**2
        sin_x = np.sum(rho * np.sin(theta_x)) * dR**2
        cos_y = np.sum(rho * np.cos(theta_y)) * dR**2
        sin_y = np.sum(rho * np.sin(theta_y)) * dR**2
        
        x0 = L * np.arctan2(sin_x, cos_x) / (2 * np.pi)
        y0 = L * np.arctan2(sin_y, cos_y) / (2 * np.pi)
        
        # RMS spread with periodic wrapping
        dx = X - x0
        dy = Y - y0
        # Minimum image
        dx = dx - L * np.round(dx / L)
        dy = dy - L * np.round(dy / L)
        
        r2 = dx**2 + dy**2
        mean_r2 = np.sum(rho * r2) * dR**2
        xi[m] = np.sqrt(mean_r2) / L  # in units of L_moire
    
    return xi


def free_particle_eigenvalues(Ns, Nb, L_moire, M_inv_scalar, V_offset, n_modes):
    """
    Analytical eigenvalues for free-particle limit:
    H = V_offset * I + 0.5 * M_inv * |q|² per band
    
    q = 2π * (n1, n2) / L with n1, n2 integers
    """
    q_vals = np.fft.fftfreq(Ns, d=L_moire/Ns) * 2 * np.pi  # in 1/a
    eigenvalues = []
    for n1 in range(Ns):
        for n2 in range(Ns):
            q2 = q_vals[n1]**2 + q_vals[n2]**2
            for b in range(Nb):
                E = V_offset + 0.5 * M_inv_scalar[b] * q2
                eigenvalues.append(E)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues[:n_modes]


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  S5: MODE ANALYSIS & EIGENVALUE CLUSTERING")
    print("=" * 70)
    
    # ── [1] Load data ────────────────────────────────────────────────────
    print(f"\n[1] Loading C4-symmetrized Phase 2 data...")
    d = load_phase2(H5_SYM)
    Ns, Nb, eta = d['Ns'], d['Nb'], d['eta']
    L_moire, dR = d['L_moire'], d['dR']
    target_idx = d['target_idx']
    omega_ref = d['omega_ref']
    
    print(f"  Grid: {Ns}×{Ns}, N_bands={Nb}")
    print(f"  η = {eta:.6f}, L_moire = {L_moire:.4f} a")
    print(f"  dR = {dR:.6f} a")
    
    # ── [2] Energy scale analysis ────────────────────────────────────────
    print(f"\n[2] Energy scale analysis...")
    
    Lambda = d['Lambda']
    M_inv = d['M_inv']
    
    # Potential depth per band
    print(f"  Potential landscape Λ (per band):")
    for n in range(Nb):
        Vn = Lambda[:, :, n, n].real
        print(f"    Band {n}: min={np.min(Vn):.6f}  max={np.max(Vn):.6f}  "
              f"depth={np.max(Vn)-np.min(Vn):.6f}")
    
    V_target = Lambda[:, :, target_idx, target_idx].real
    V_depth = np.max(V_target) - np.min(V_target)
    V_max = np.max(V_target)
    V_min = np.min(V_target)
    
    # Mass tensor per band (spatial average of diagonal elements)
    print(f"\n  Effective mass (spatially averaged, diagonal bands):")
    M_eff = np.zeros(Nb)
    for n in range(Nb):
        mxx = np.mean(M_inv[:, :, n, n, 0, 0])
        myy = np.mean(M_inv[:, :, n, n, 1, 1])
        M_eff[n] = 0.5 * (mxx + myy)
        mxy = np.mean(M_inv[:, :, n, n, 0, 1])
        print(f"    Band {n}: M_xx={mxx:.4f}  M_yy={myy:.4f}  M_xy={mxy:.4f}  "
              f"<M>={M_eff[n]:.4f}  ({'hole' if M_eff[n] < 0 else 'electron'})")
    
    # Kinetic energy scale
    E_kin = np.abs(0.5 * M_eff[target_idx]) / L_moire**2
    ratio = V_depth / E_kin if E_kin > 1e-15 else float('inf')
    
    print(f"\n  Target band ({target_idx}): M_eff = {M_eff[target_idx]:.4f}")
    print(f"  V_depth = {V_depth:.6f}")
    print(f"  E_kin = 0.5 * |M| / L² = 0.5 * {np.abs(M_eff[target_idx]):.4f} / {L_moire:.2f}² = {E_kin:.6e}")
    print(f"  Ratio V_depth / E_kin = {ratio:.1f}")
    print(f"  → {'DEEP TRAPPING: many clustered modes expected' if ratio > 100 else 'MODERATE trapping: few resolved modes' if ratio > 1 else 'WEAK trapping: free-particle-like'}")
    
    # How many bound states expected?
    # In 2D box of size L with depth V and mass M:
    # N_bound ~ (V * M * L²) / (2π)
    N_bound_est = V_depth * np.abs(M_eff[target_idx]) * L_moire**2 / (2 * np.pi)
    print(f"  Estimated # of bound states per band: ~ {N_bound_est:.0f}")
    
    # Eigenvalue spacing expected:
    delta_E = V_depth / N_bound_est if N_bound_est > 0 else 0
    print(f"  Expected level spacing: ~ {delta_E:.6e}")
    
    # ── [3] Solve at full resolution ─────────────────────────────────────
    print(f"\n[3] Solving at Ns={Ns} (full resolution)...")
    
    n_modes = 20
    sigma = float(V_max) if M_eff[target_idx] < 0 else float(V_min)
    print(f"  sigma = {sigma:.6f}  ({'hole band, target V_max' if M_eff[target_idx] < 0 else 'electron band, target V_min'})")
    
    H = build_H_at_resolution(d, Ns, include_A=False)
    print(f"  H built: {H.shape[0]}×{H.shape[1]}, nnz={H.nnz}")
    print(f"  Solving for {n_modes} eigenvalues...")
    eigenvalues, eigenvectors = eigsh(H, k=n_modes, sigma=sigma, which='LM',
                                       maxiter=10000, tol=1e-10)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    print(f"  Eigenvalues (all {n_modes}):")
    for i in range(n_modes):
        delta_from_vmax = eigenvalues[i] - V_max
        print(f"    mode {i:2d}: ε = {eigenvalues[i]:+.8e}  "
              f"Δ(V_max) = {delta_from_vmax:+.4e}  "
              f"ω = {omega_ref + eigenvalues[i]:.6f}")
    
    # ── [4] Mode localization ────────────────────────────────────────────
    print(f"\n[4] Mode localization analysis...")
    
    ipr = compute_ipr(eigenvectors, Ns, Nb)
    ipr_extended = 1.0 / (Ns * Ns)  # IPR of a perfectly extended mode
    ipr_localized = 1.0  # IPR of a perfectly localized mode
    
    xi = compute_localization_length(eigenvectors, Ns, Nb, dR)
    xi_extended = 1.0 / (2 * np.sqrt(3))  # RMS of uniform distribution on [0,L]: L/(2√3) → ξ/L ≈ 0.289
    
    print(f"  {'Mode':>4s}  {'IPR':>10s}  {'IPR/ext':>8s}  {'ξ/L':>8s}  {'Character':>12s}")
    print(f"  {'----':>4s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}  {'------------':>12s}")
    for i in range(n_modes):
        ipr_ratio = ipr[i] / ipr_extended
        char = "LOCALIZED" if ipr_ratio > 10 else ("extended" if ipr_ratio < 2 else "intermediate")
        print(f"  {i:>4d}  {ipr[i]:>10.4e}  {ipr_ratio:>8.1f}×  {xi[i]:>8.4f}  {char:>12s}")
    
    print(f"\n  Reference: IPR(extended) = 1/N_sites = {ipr_extended:.4e}")
    print(f"  Reference: ξ/L(extended) ≈ 0.289 (uniform)")
    
    # ── [5] Band composition ─────────────────────────────────────────────
    print(f"\n[5] Band composition of eigenmodes...")
    
    weights = compute_band_weights(eigenvectors, Ns, Nb)
    
    print(f"  {'Mode':>4s}", end="")
    for n in range(Nb):
        print(f"  {'B'+str(n):>6s}", end="")
    print(f"  {'Dominant':>10s}")
    print(f"  {'----':>4s}" + "  ------" * Nb + f"  {'----------':>10s}")
    for i in range(n_modes):
        row = f"  {i:>4d}"
        for n in range(Nb):
            row += f"  {weights[i, n]:>6.3f}"
        dom = np.argmax(weights[i])
        row += f"  Band {dom}"
        if weights[i, dom] < 0.5:
            row += " (mixed)"
        print(row)
    
    # ── [6] Free-particle cross-check ────────────────────────────────────
    print(f"\n[6] Free-particle cross-check...")
    print(f"  Building constant-Λ, constant-M, A=0 Hamiltonian...")
    
    # Use spatially uniform Λ = V_max (for hole band) and spatially averaged M
    d_fp = {k: v for k, v in d.items()}
    Lambda_const = np.zeros_like(Lambda)
    for n in range(Nb):
        Lambda_const[:, :, n, n] = np.mean(Lambda[:, :, n, n].real)
    d_fp['Lambda'] = Lambda_const
    
    # Zero out A, v_drift, Phi_BH
    d_fp['A_berry'] = np.zeros_like(d['A_berry'])
    d_fp['v_drift'] = np.zeros_like(d['v_drift'])
    d_fp['Phi_BH'] = np.zeros_like(d['Phi_BH'])
    
    # Use constant M_inv (spatial average)
    M_const = np.zeros_like(M_inv)
    for n in range(Nb):
        for i_c in range(2):
            for j_c in range(2):
                M_const[:, :, n, n, i_c, j_c] = np.mean(M_inv[:, :, n, n, i_c, j_c])
    d_fp['M_inv'] = M_const
    
    H_fp = build_H_at_resolution(d_fp, Ns, include_A=False)
    print(f"  H_fp built: {H_fp.shape[0]}×{H_fp.shape[1]}, nnz={H_fp.nnz}")
    
    sigma_fp = float(np.max(Lambda_const[:, :, target_idx, target_idx])) if M_eff[target_idx] < 0 else float(np.min(Lambda_const[:, :, target_idx, target_idx]))
    
    print(f"  Solving for {n_modes} eigenvalues (sigma={sigma_fp:.6f})...")
    ev_fp, _ = eigsh(H_fp, k=n_modes, sigma=sigma_fp, which='LM',
                      maxiter=10000, tol=1e-10)
    ev_fp = np.sort(ev_fp)
    
    # Analytical formula
    print(f"\n  Analytical eigenvalues (first {n_modes}):")
    V_offsets = np.array([np.mean(Lambda[:, :, n, n].real) for n in range(Nb)])
    ev_analytical = free_particle_eigenvalues(Ns, Nb, L_moire, M_eff, 0, n_modes * 10)
    # Offset by V_offsets — need to include per-band offset
    # Actually the analytical formula already includes V_offset per band
    # Redo more carefully:
    q_vals = np.fft.fftfreq(Ns, d=L_moire/Ns) * 2 * np.pi
    ev_list = []
    for n1 in range(Ns):
        for n2 in range(Ns):
            q2 = q_vals[n1]**2 + q_vals[n2]**2
            for b in range(Nb):
                E = V_offsets[b] + 0.5 * M_eff[b] * q2
                ev_list.append(E)
    ev_list = np.sort(ev_list)
    
    # Find the n_modes closest to sigma
    ev_near_sigma = ev_list[np.argsort(np.abs(ev_list - sigma_fp))[:n_modes]]
    ev_near_sigma = np.sort(ev_near_sigma)
    
    print(f"  {'Mode':>4s}  {'Numerical':>14s}  {'Analytical':>14s}  {'Diff':>12s}")
    print(f"  {'----':>4s}  {'-'*14:>14s}  {'-'*14:>14s}  {'-'*12:>12s}")
    for i in range(min(n_modes, len(ev_near_sigma))):
        diff = ev_fp[i] - ev_near_sigma[i]
        print(f"  {i:>4d}  {ev_fp[i]:>+14.8e}  {ev_near_sigma[i]:>+14.8e}  {diff:>+12.4e}")
    
    max_diff = np.max(np.abs(ev_fp[:min(n_modes, len(ev_near_sigma))] - ev_near_sigma[:min(n_modes, len(ev_near_sigma))]))
    print(f"\n  Max |numerical - analytical| = {max_diff:.4e}")
    print(f"  {'✓ FREE-PARTICLE LIMIT PASSES' if max_diff < 1e-8 else '⚠ FREE-PARTICLE LIMIT HAS DISCREPANCY: ' + f'{max_diff:.2e}'}")
    
    # ── [7] Grid convergence ─────────────────────────────────────────────
    print(f"\n[7] Grid convergence test (C4-sym, A=0)...")
    
    grid_sizes = [32, 64, 128]
    n_conv = 12
    conv_results = {}
    
    for Ns_test in grid_sizes:
        print(f"\n  --- Ns = {Ns_test} ---")
        factor = Ns // Ns_test
        
        if Ns_test == Ns:
            # Reuse full result
            ev_conv = eigenvalues[:n_conv]
        else:
            H_test = build_H_at_resolution(d, Ns_test, include_A=False)
            N_total = Ns_test * Ns_test * Nb
            
            # Adjust sigma for downsampled V
            Lambda_ds = downsample_field(Lambda, factor)
            V_target_ds = Lambda_ds[:, :, target_idx, target_idx].real
            sigma_ds = float(np.max(V_target_ds)) if M_eff[target_idx] < 0 else float(np.min(V_target_ds))
            
            print(f"    H: {N_total}×{N_total}, nnz={H_test.nnz}")
            print(f"    sigma = {sigma_ds:.6f}")
            n_solve = min(n_conv, N_total - 2)
            ev_test, _ = eigsh(H_test, k=n_solve, sigma=sigma_ds, which='LM',
                                maxiter=10000, tol=1e-10)
            ev_conv = np.sort(ev_test)[:n_conv]
        
        conv_results[Ns_test] = ev_conv
        print(f"    First 10 eigenvalues:")
        for i in range(min(10, len(ev_conv))):
            print(f"      mode {i}: ε = {ev_conv[i]:+.8e}")
    
    # Convergence comparison
    print(f"\n  Grid convergence comparison:")
    print(f"  {'Mode':>4s}", end="")
    for Ns_test in grid_sizes:
        print(f"  {'Ns='+str(Ns_test):>14s}", end="")
    print(f"  {'Δ(64-128)':>12s}  {'Δ(32-64)':>12s}")
    print(f"  {'----':>4s}" + "  " + "-"*14 * len(grid_sizes) + "  " + "-"*12 + "  " + "-"*12)
    
    n_compare = min(n_conv, *(len(conv_results[g]) for g in grid_sizes))
    for i in range(min(10, n_compare)):
        row = f"  {i:>4d}"
        for Ns_test in grid_sizes:
            if i < len(conv_results[Ns_test]):
                row += f"  {conv_results[Ns_test][i]:>+14.8e}"
            else:
                row += f"  {'—':>14s}"
        if 64 in conv_results and 128 in conv_results and i < len(conv_results[64]) and i < len(conv_results[128]):
            row += f"  {conv_results[64][i] - conv_results[128][i]:>+12.4e}"
        else:
            row += f"  {'—':>12s}"
        if 32 in conv_results and 64 in conv_results and i < len(conv_results[32]) and i < len(conv_results[64]):
            row += f"  {conv_results[32][i] - conv_results[64][i]:>+12.4e}"
        else:
            row += f"  {'—':>12s}"
        print(row)
    
    # ── [8] Potential landscape analysis ─────────────────────────────────
    print(f"\n[8] Potential landscape of target band...")
    
    V = V_target
    V_well_area = np.sum(V > (V_max - V_depth * 0.1)) / (Ns * Ns)  # fraction above 90% of V_max
    V_well_area_50 = np.sum(V > (V_max - V_depth * 0.5)) / (Ns * Ns)  # fraction above 50%
    
    print(f"  V_max = {V_max:.6f}")
    print(f"  V_min = {V_min:.6f}")
    print(f"  V_depth = {V_depth:.6f}")
    print(f"  Area above 90% of V_max: {V_well_area*100:.1f}% of moiré cell")
    print(f"  Area above 50% of V_max: {V_well_area_50*100:.1f}% of moiré cell")
    
    # For hole band: "well" is at V_max (inverted → bound states at top)
    # Modes at V_max are trapped in the HILL of V (inverted potential)
    print(f"\n  For hole band: potential HILL traps modes")
    print(f"  Modes sit near V_max = {V_max:.6f}")
    print(f"  All {n_modes} eigenvalues within [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")
    print(f"  Eigenvalue range = {eigenvalues[-1] - eigenvalues[0]:.6e}")
    print(f"  Distance from V_max: [{eigenvalues[0] - V_max:.4e}, {eigenvalues[-1] - V_max:.4e}]")
    
    # ── [9] Plots ────────────────────────────────────────────────────────
    print(f"\n[9] Generating plots...")
    
    fig = plt.figure(figsize=(28, 24))
    fig.suptitle("S5: Mode Analysis & Eigenvalue Clustering", fontsize=16, fontweight='bold')
    
    # ---- Panel 1: Potential landscape ----
    ax1 = fig.add_subplot(3, 4, 1)
    im = ax1.imshow(V_target.T, origin='lower', cmap='RdBu_r')
    plt.colorbar(im, ax=ax1, label='Λ')
    ax1.set_title(f'Potential Λ (band {target_idx})\nV_depth={V_depth:.4f}')
    ax1.set_xlabel('ix')
    ax1.set_ylabel('iy')
    
    # ---- Panel 2: Eigenvalue spectrum ----
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.barh(range(n_modes), eigenvalues - V_max, height=0.8, color='steelblue')
    ax2.axvline(0, color='red', linestyle='--', label='V_max')
    ax2.set_xlabel('ε − V_max')
    ax2.set_ylabel('Mode index')
    ax2.set_title(f'Eigenvalues relative to V_max\nspread = {eigenvalues[-1]-eigenvalues[0]:.2e}')
    ax2.legend()
    ax2.invert_yaxis()
    
    # ---- Panel 3: IPR ----
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.semilogy(range(n_modes), ipr, 'o-', color='darkorange')
    ax3.axhline(ipr_extended, color='gray', linestyle='--', label=f'extended (1/N={ipr_extended:.1e})')
    ax3.set_xlabel('Mode index')
    ax3.set_ylabel('IPR')
    ax3.set_title('Inverse Participation Ratio')
    ax3.legend(fontsize=8)
    
    # ---- Panel 4: Localization length ----
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.plot(range(n_modes), xi, 'o-', color='forestgreen')
    ax4.axhline(xi_extended, color='gray', linestyle='--', label=f'extended (~{xi_extended:.3f})')
    ax4.set_xlabel('Mode index')
    ax4.set_ylabel('ξ / L_moire')
    ax4.set_title('Localization Length')
    ax4.legend(fontsize=8)
    
    # ---- Panels 5-8: Mode profiles (first 4 modes) ----
    for i in range(4):
        ax = fig.add_subplot(3, 4, 5 + i)
        F = eigenvectors[:, i].reshape(Ns, Ns, Nb)
        rho = np.sum(np.abs(F)**2, axis=2)
        im = ax.imshow(rho.T, origin='lower', cmap='inferno')
        plt.colorbar(im, ax=ax, label='|F|²')
        ax.set_title(f'Mode {i}: ε={eigenvalues[i]:.6f}\nIPR={ipr[i]:.2e}, ξ/L={xi[i]:.3f}')
        ax.set_xlabel('ix')
        ax.set_ylabel('iy')
    
    # ---- Panel 9: Band weights ----
    ax9 = fig.add_subplot(3, 4, 9)
    bottom = np.zeros(n_modes)
    colors = plt.cm.tab10(range(Nb))
    for n in range(Nb):
        ax9.bar(range(n_modes), weights[:, n], bottom=bottom, color=colors[n],
                label=f'Band {n}', width=0.8)
        bottom += weights[:, n]
    ax9.set_xlabel('Mode index')
    ax9.set_ylabel('Band weight fraction')
    ax9.set_title('Band Composition')
    ax9.legend(fontsize=7, ncol=Nb)
    
    # ---- Panel 10: Grid convergence ----
    ax10 = fig.add_subplot(3, 4, 10)
    for gs in grid_sizes:
        ev = conv_results[gs]
        ax10.plot(range(len(ev)), ev - V_max, 'o-', markersize=4, label=f'Ns={gs}')
    ax10.set_xlabel('Mode index')
    ax10.set_ylabel('ε − V_max')
    ax10.set_title('Grid Convergence')
    ax10.legend(fontsize=8)
    
    # ---- Panel 11: Convergence differences ----
    ax11 = fig.add_subplot(3, 4, 11)
    if all(g in conv_results for g in [32, 64, 128]):
        n_c = min(len(conv_results[32]), len(conv_results[64]), len(conv_results[128]))
        diff_64_128 = conv_results[64][:n_c] - conv_results[128][:n_c]
        diff_32_64 = conv_results[32][:n_c] - conv_results[64][:n_c]
        ax11.semilogy(range(n_c), np.abs(diff_64_128), 'o-', label='|Ns64 - Ns128|', markersize=4)
        ax11.semilogy(range(n_c), np.abs(diff_32_64), 's-', label='|Ns32 - Ns64|', markersize=4)
        ax11.set_xlabel('Mode index')
        ax11.set_ylabel('|Δε|')
        ax11.set_title('Convergence Rate')
        ax11.legend(fontsize=8)
    
    # ---- Panel 12: Energy scale comparison ----
    ax12 = fig.add_subplot(3, 4, 12)
    scales = {
        'V_depth': V_depth,
        'E_kin\n(0.5M/L²)': E_kin,
        'Mode\nspread': eigenvalues[-1] - eigenvalues[0],
        'Level\nspacing': np.mean(np.diff(eigenvalues)) if len(eigenvalues) > 1 else 0,
        'η²': eta**2,
    }
    bars = ax12.bar(range(len(scales)), list(scales.values()), color='teal')
    ax12.set_xticks(range(len(scales)))
    ax12.set_xticklabels(list(scales.keys()), fontsize=8)
    ax12.set_yscale('log')
    ax12.set_ylabel('Energy scale')
    ax12.set_title('Energy Scale Hierarchy')
    for bar, val in zip(bars, scales.values()):
        ax12.text(bar.get_x() + bar.get_width()/2, val * 1.5, f'{val:.1e}',
                  ha='center', fontsize=7)
    
    plt.tight_layout()
    plot_path = PLOT_DIR / "S5_mode_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {plot_path.name}")
    plt.close()
    
    # ── Mode gallery (extended) ──────────────────────────────────────────
    n_gallery = min(12, n_modes)
    fig2, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig2.suptitle("S5: Mode Gallery (C4-sym, A=0, Ns=128)", fontsize=14, fontweight='bold')
    
    for i in range(n_gallery):
        ax = axes[i // 4, i % 4]
        F = eigenvectors[:, i].reshape(Ns, Ns, Nb)
        rho = np.sum(np.abs(F)**2, axis=2)
        im = ax.imshow(rho.T, origin='lower', cmap='inferno')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Mode {i}: ε={eigenvalues[i]:.6f}\nIPR={ipr[i]:.2e}', fontsize=9)
    
    for i in range(n_gallery, 12):
        axes[i // 4, i % 4].axis('off')
    
    plt.tight_layout()
    plot_path2 = PLOT_DIR / "S5_mode_gallery.png"
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {plot_path2.name}")
    plt.close()
    
    # ── [10] Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  S5 SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n  Energy scales:")
    print(f"    V_depth            = {V_depth:.6f}")
    print(f"    E_kin (0.5·M/L²)   = {E_kin:.6e}")
    print(f"    V_depth / E_kin    = {ratio:.1f}")
    print(f"    η²                 = {eta**2:.6e}")
    print(f"    Eigenvalue spread  = {eigenvalues[-1] - eigenvalues[0]:.6e}")
    print(f"    Mean level spacing = {np.mean(np.diff(eigenvalues)):.6e}")
    
    print(f"\n  Mode character:")
    n_localized = np.sum(ipr > 10 * ipr_extended)
    n_extended = np.sum(ipr < 2 * ipr_extended)
    print(f"    Localized (IPR > 10×extended): {n_localized}/{n_modes}")
    print(f"    Extended  (IPR < 2×extended):  {n_extended}/{n_modes}")
    print(f"    Mean IPR = {np.mean(ipr):.4e}  (extended = {ipr_extended:.4e})")
    print(f"    Mean ξ/L = {np.mean(xi):.4f}  (extended ≈ 0.289)")
    
    print(f"\n  Band composition:")
    dom_band = np.argmax(weights, axis=1)
    for n in range(Nb):
        count = np.sum(dom_band == n)
        mean_w = np.mean(weights[dom_band == n, n]) if count > 0 else 0
        print(f"    Band {n}: dominant in {count}/{n_modes} modes (mean weight = {mean_w:.3f})")
    
    print(f"\n  Grid convergence:")
    if all(g in conv_results for g in [32, 64, 128]):
        n_c = min(len(conv_results[32]), len(conv_results[64]), len(conv_results[128]))
        diff_64_128 = np.abs(conv_results[64][:n_c] - conv_results[128][:n_c])
        diff_32_64 = np.abs(conv_results[32][:n_c] - conv_results[64][:n_c])
        print(f"    |Ns=64 - Ns=128|: mean={np.mean(diff_64_128):.4e}, max={np.max(diff_64_128):.4e}")
        print(f"    |Ns=32 - Ns=64|:  mean={np.mean(diff_32_64):.4e}, max={np.max(diff_32_64):.4e}")
        ratio_conv = np.mean(diff_32_64) / np.mean(diff_64_128) if np.mean(diff_64_128) > 1e-15 else float('inf')
        print(f"    Convergence ratio (32→64)/(64→128) = {ratio_conv:.1f}")
        if ratio_conv > 3:
            print(f"    → Converging (ratio > 3 suggests ~2nd order)")
        else:
            print(f"    → {'Slow or not converging' if ratio_conv < 1.5 else 'Marginal convergence'}")
    
    print(f"\n  Free-particle cross-check: max diff = {max_diff:.4e}")
    
    print(f"\n  Plots saved to {PLOT_DIR}/")
    print(f"\n{'='*70}")
    print(f"  S5 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
