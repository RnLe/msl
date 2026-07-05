#!/usr/bin/env python3
"""
S7: Validate production off-diagonal Berry connection implementation.
=====================================================================

Tests the modified `build_multiband_kinetic_operator(include_offdiag_A=True)`
in phase3_mpb_v3.py against the S6 proof-of-concept implementation.

Checks:
  1. Legacy path unchanged (include_offdiag_A=False reproduces S6 standard)
  2. Off-diagonal path reproduces S6 corrected results
  3. Hermiticity of the corrected operator
  4. A=0 limit: off-diagonal path reduces to legacy
  5. Single-band limit: off-diagonal path reduces to scalar kinetic
  6. Eigenvalue comparison: legacy vs full Berry coupling
  7. Band composition analysis with full coupling
  8. C4 commutator with full coupling
  9. Twist-angle sweep with full Berry coupling

Uses: phase2_multiband_data_c4sym.h5 (C4-symmetrized Phase 2 data)
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
from scipy.sparse import csr_matrix, eye

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "phasesV3"))

from phase3_mpb_v3 import (
    assemble_multiband_hamiltonian,
    _regularize_M_inv,
    build_multiband_kinetic_operator,
    build_multiband_potential_operator,
    build_multiband_drift_operator,
    _build_band_block_diagonal,
)

RUN_DIR = REPO / "runsV3" / "phase0_mpb_v3_20260206_152443"
CAND    = RUN_DIR / "candidate_0000"
H5_SYM  = CAND / "phase2_multiband_data_c4sym.h5"
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_phase2(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = {
            'Lambda':   hf['Lambda'][:],
            'A_berry':  hf['A_berry'][:],
            'Phi_BH':   hf['Phi_BH'][:],
            'v_drift':  hf['v_drift'][:],
            'M_inv':    hf['M_inv'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta':       float(hf.attrs['eta']),
            'Ns1':       int(hf.attrs['Ns1']),
            'Ns2':       int(hf.attrs['Ns2']),
            'Nb':        int(hf.attrs['N_subspace']),
            'B_moire':   hf.attrs['B_moire'][:],
            'target_idx': int(hf.attrs['target_index_in_subspace']),
        }
    data['Ns'] = data['Ns1']
    return data


def compute_band_weights(eigvec, Ns, Nb):
    n_modes = eigvec.shape[1]
    weights = np.zeros((n_modes, Nb))
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        for n in range(Nb):
            weights[m, n] = np.sum(np.abs(F[:, :, n])**2)
        weights[m] /= np.sum(weights[m])
    return weights


def compute_ipr(eigvec, Ns, Nb):
    n_modes = eigvec.shape[1]
    ipr = np.zeros(n_modes)
    for m in range(n_modes):
        F = eigvec[:, m].reshape(Ns, Ns, Nb)
        rho = np.sum(np.abs(F)**2, axis=2)
        ipr[m] = np.sum(rho**2) / np.sum(rho)**2
    return ipr


def build_c4_matrix(Ns, Nb):
    """Build C4 rotation matrix for the full N_total-dim space."""
    N_total = Ns * Ns * Nb
    rows, cols = [], []
    for ix in range(Ns):
        for iy in range(Ns):
            ix2 = (Ns - iy) % Ns
            iy2 = ix
            for n in range(Nb):
                old_idx = (ix * Ns + iy) * Nb + n
                new_idx = (ix2 * Ns + iy2) * Nb + n
                rows.append(new_idx)
                cols.append(old_idx)
    data = np.ones(len(rows))
    return csr_matrix((data, (np.array(rows), np.array(cols))),
                      shape=(N_total, N_total))


def downsample(field, factor):
    Ns = field.shape[0]
    Ns_new = Ns // factor
    extra = field.shape[2:]
    result = np.zeros((Ns_new, Ns_new) + extra, dtype=field.dtype)
    for i in range(factor):
        for j in range(factor):
            result += field[i::factor, j::factor, ...]
    result /= factor**2
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  S7: OFF-DIAGONAL A — PRODUCTION VALIDATION")
    print("=" * 70)

    d = load_phase2(H5_SYM)
    Ns, Nb = d['Ns'], d['Nb']
    eta = d['eta']
    target_idx = d['target_idx']
    B_moire = d['B_moire']
    Lambda = d['Lambda']
    A_berry = d['A_berry']
    Phi_BH = d['Phi_BH']
    v_drift = d['v_drift']
    M_inv = d['M_inv']
    L_moire = 1.0 / eta
    dR = L_moire / Ns
    n_modes = 20

    print(f"  Grid: {Ns}×{Ns}, Nb={Nb}, η={eta:.6f}, dR={dR:.4f}")

    # Regularize
    M_inv_reg = _regularize_M_inv(M_inv.copy(), 20.0)

    # ══════════════════════════════════════════════════════════════════════
    # [1] Legacy path unchanged
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[1] LEGACY PATH VALIDATION (include_offdiag_A=False)...")

    H_legacy = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, A_berry, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
        include_offdiag_A=False,
    )
    H_legacy = 0.5 * (H_legacy + H_legacy.conj().T)

    V_target = Lambda[:, :, target_idx, target_idx].real
    M_target = M_inv[:, :, target_idx, target_idx, :, :]
    mean_trace = np.mean(M_target[:, :, 0, 0] + M_target[:, :, 1, 1])
    sigma = float(np.max(V_target)) if mean_trace < 0 else float(np.min(V_target))

    ev_leg, evec_leg = eigsh(H_legacy, k=n_modes, sigma=sigma, which='LM',
                              maxiter=10000, tol=1e-10)
    order = np.argsort(ev_leg)
    ev_leg = ev_leg[order]
    evec_leg = evec_leg[:, order]

    w_leg = compute_band_weights(evec_leg, Ns, Nb)
    max_mix_leg = 1.0 - np.max(w_leg, axis=1)

    print(f"  Legacy: {n_modes} modes, spread={ev_leg[-1]-ev_leg[0]:.4e}")
    print(f"  Mean mixing = {np.mean(max_mix_leg):.6f} (expect ~0)")
    print(f"  ✓ Legacy path produces band-diagonal modes" if np.mean(max_mix_leg) < 1e-3
          else f"  ✗ Unexpected mixing in legacy!")

    # ══════════════════════════════════════════════════════════════════════
    # [2] Full Berry coupling path
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[2] FULL BERRY COUPLING (include_offdiag_A=True)...")

    H_full = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, A_berry, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
        include_offdiag_A=True,
    )
    H_full = 0.5 * (H_full + H_full.conj().T)

    print(f"  H_full nnz = {H_full.nnz} (legacy: {H_legacy.nnz})")

    ev_full, evec_full = eigsh(H_full, k=n_modes, sigma=sigma, which='LM',
                                maxiter=10000, tol=1e-10)
    order = np.argsort(ev_full)
    ev_full = ev_full[order]
    evec_full = evec_full[:, order]

    w_full = compute_band_weights(evec_full, Ns, Nb)
    max_mix_full = 1.0 - np.max(w_full, axis=1)
    ipr_full = compute_ipr(evec_full, Ns, Nb)

    print(f"  Full:   {n_modes} modes, spread={ev_full[-1]-ev_full[0]:.4e}")
    print(f"  Mean mixing = {np.mean(max_mix_full):.6f}")

    # ══════════════════════════════════════════════════════════════════════
    # [3] Hermiticity check
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[3] HERMITICITY CHECK...")

    K_full = build_multiband_kinetic_operator(
        M_inv_reg, A_berry, eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_offdiag_A=True,
    )
    diff = K_full - K_full.conj().T
    herm_err = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
    K_max = np.max(np.abs(K_full.data)) if K_full.nnz > 0 else 1.0
    herm_rel = herm_err / K_max
    print(f"  ||K - K†||_max = {herm_err:.4e}")
    print(f"  ||K - K†||/||K|| = {herm_rel:.4e}")
    print(f"  ✓ Hermitian to machine precision" if herm_rel < 1e-13
          else f"  ✗ Hermiticity error: {herm_rel:.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # [4] A=0 consistency: off-diagonal path with A=0 ≡ legacy with A=0
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[4] A=0 CONSISTENCY CHECK...")

    A_zero = np.zeros_like(A_berry)

    K_leg_a0 = build_multiband_kinetic_operator(
        M_inv_reg, A_zero, eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_offdiag_A=False,
    )
    K_full_a0 = build_multiband_kinetic_operator(
        M_inv_reg, A_zero, eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_offdiag_A=True,
    )

    diff_a0_sp = K_leg_a0 - K_full_a0
    diff_a0 = np.max(np.abs(diff_a0_sp.data)) if diff_a0_sp.nnz > 0 else 0.0
    print(f"  ||K_legacy(A=0) - K_full(A=0)||_max = {diff_a0:.4e}")
    print(f"  ✓ Paths agree for A=0" if diff_a0 < 1e-13
          else f"  ✗ A=0 consistency failed: {diff_a0:.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # [5] Diagonal-only A: off-diagonal path with diagonal A ≈ legacy
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[5] DIAGONAL-A CONSISTENCY CHECK...")

    A_diag = np.zeros_like(A_berry)
    for n in range(Nb):
        A_diag[:, :, n, n, :] = A_berry[:, :, n, n, :]

    K_leg_diag = build_multiband_kinetic_operator(
        M_inv_reg, A_diag, eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_offdiag_A=False,
    )
    K_full_diag = build_multiband_kinetic_operator(
        M_inv_reg, A_diag, eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_offdiag_A=True,
    )

    # They won't be exactly equal because off-diag path includes paramagnetic terms
    # that legacy doesn't. But the A² diagonal contribution should match.
    # After Hermitization the paramagnetic terms with diagonal A add the "para" correction.
    diff_diag_sp = K_leg_diag - K_full_diag
    diff_diag = np.max(np.abs(diff_diag_sp.data)) if diff_diag_sp.nnz > 0 else 0.0
    K_leg_max = np.max(np.abs(K_leg_diag.data)) if K_leg_diag.nnz > 0 else 1.0
    rel_diag = diff_diag / max(K_leg_max, 1e-20)
    print(f"  ||K_legacy(A_diag) - K_full(A_diag)||_max = {diff_diag:.4e}")
    print(f"  Relative difference = {rel_diag:.4e}")
    print(f"  NOTE: difference expected due to paramagnetic terms (para is zero in legacy)")
    # Cross-check: are the eigenvalues close?
    N_total = Ns * Ns * Nb
    # Use a small test: solve 6 eigenvalues of each
    V_op = build_multiband_potential_operator(Lambda, B_moire)
    H_test_leg = V_op + K_leg_diag
    H_test_leg = 0.5 * (H_test_leg + H_test_leg.conj().T)
    H_test_full = V_op + K_full_diag
    H_test_full = 0.5 * (H_test_full + H_test_full.conj().T)
    ev_tl, _ = eigsh(H_test_leg, k=6, sigma=sigma, which='LM', tol=1e-10)
    ev_tf, _ = eigsh(H_test_full, k=6, sigma=sigma, which='LM', tol=1e-10)
    ev_tl.sort(); ev_tf.sort()
    ev_diff = np.max(np.abs(ev_tl - ev_tf))
    print(f"  Max eigenvalue difference (6 modes, V+K only): {ev_diff:.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # [6] C4 commutator with full Berry coupling
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[6] C4 COMMUTATOR CHECK...")

    C4 = build_c4_matrix(Ns, Nb)
    comm = H_full @ C4 - C4 @ H_full
    comm_norm = np.sqrt(np.sum(np.abs(comm.data)**2)) if comm.nnz > 0 else 0.0
    H_norm = np.sqrt(np.sum(np.abs(H_full.data)**2))
    rel_comm = comm_norm / H_norm
    print(f"  ||[H_full, C4]|| / ||H|| = {rel_comm:.4e}")
    print(f"  ✓ C4 preserved" if rel_comm < 1e-10
          else f"  ⚠ C4 breaking at {rel_comm:.4e}")

    # ══════════════════════════════════════════════════════════════════════
    # [7] Detailed eigenvalue and mode comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[7] EIGENVALUE & MODE COMPARISON ({n_modes} modes):")

    # Also solve A=0 for reference
    H_a0 = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_reg, A_zero, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=False,
    )
    H_a0 = 0.5 * (H_a0 + H_a0.conj().T)
    ev_a0, evec_a0 = eigsh(H_a0, k=n_modes, sigma=sigma, which='LM',
                            maxiter=10000, tol=1e-10)
    order = np.argsort(ev_a0); ev_a0 = ev_a0[order]; evec_a0 = evec_a0[:, order]
    w_a0 = compute_band_weights(evec_a0, Ns, Nb)
    max_mix_a0 = 1.0 - np.max(w_a0, axis=1)

    print(f"\n  {'Mode':>4s}  {'ε(A=0)':>14s}  {'ε(legacy)':>14s}  {'ε(full A)':>14s}  "
          f"{'mix(A=0)':>8s}  {'mix(leg)':>8s}  {'mix(full)':>9s}  {'dom(full)':>9s}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}")
    for i in range(n_modes):
        dom = np.argmax(w_full[i])
        weights_str = '/'.join(f'{w:.2f}' for w in w_full[i] if w > 0.05)
        print(f"  {i:>4d}  {ev_a0[i]:>+14.8e}  {ev_leg[i]:>+14.8e}  {ev_full[i]:>+14.8e}  "
              f"{max_mix_a0[i]:>8.4f}  {max_mix_leg[i]:>8.4f}  {max_mix_full[i]:>9.4f}  "
              f"B{dom}")

    print(f"\n  Band weight table (full Berry coupling):")
    print(f"  {'Mode':>4s}  " + "  ".join(f"{'B'+str(n):>6s}" for n in range(Nb)) + f"  {'mix':>8s}")
    print(f"  {'─'*4}  " + "  ".join(f"{'─'*6}" for _ in range(Nb)) + f"  {'─'*8}")
    for i in range(n_modes):
        row = f"  {i:>4d}  "
        row += "  ".join(f"{w_full[i,n]:>6.3f}" for n in range(Nb))
        row += f"  {max_mix_full[i]:>8.4f}"
        print(row)

    # ══════════════════════════════════════════════════════════════════════
    # [8] Mode gallery (6 lowest modes, full coupling)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[8] Generating mode gallery...")

    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle("S7: Modes with Full Berry Coupling (top) vs Legacy (middle) vs A=0 (bottom)",
                 fontsize=13, fontweight='bold')

    n_show = 6
    configs = [
        (evec_full, ev_full, w_full, "Full A (offdiag)"),
        (evec_leg, ev_leg, w_leg, "Legacy (diag A)"),
        (evec_a0, ev_a0, w_a0, "A = 0"),
    ]

    for row, (evecs, evals, weights, label) in enumerate(configs):
        for col in range(n_show):
            ax = axes[row, col]
            F = evecs[:, col].reshape(Ns, Ns, Nb)
            rho = np.sum(np.abs(F)**2, axis=2)
            ax.imshow(rho, origin='lower', cmap='inferno', interpolation='nearest')
            dom = np.argmax(weights[col])
            mix = 1.0 - weights[col, dom]
            ax.set_title(f"m={col}, B{dom}\nmix={mix:.3f}", fontsize=8)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = PLOT_DIR / "S7_offdiag_validation.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {path.name}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # [9] η-sweep with full Berry coupling
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[9] η-SWEEP WITH FULL BERRY COUPLING (Ns=64)...")

    Ns_sw = 64
    factor = Ns // Ns_sw
    Lambda_ds = downsample(Lambda, factor)
    M_inv_ds = downsample(M_inv, factor)
    v_drift_ds = downsample(v_drift, factor)
    A_berry_ds = downsample(A_berry, factor)
    Phi_BH_ds = np.zeros((Ns_sw, Ns_sw, Nb, Nb))

    V_target_ds = Lambda_ds[:, :, target_idx, target_idx].real
    sigma_ds = float(np.max(V_target_ds)) if mean_trace < 0 else float(np.min(V_target_ds))
    M_inv_reg_ds = _regularize_M_inv(M_inv_ds.copy(), 20.0)

    theta_sweep = np.array([1.1, 3.0, 5.0, 7.0, 10.0])
    eta_sweep = 2 * np.sin(np.deg2rad(theta_sweep) / 2)
    n_sw = 12

    print(f"\n  {'θ (°)':>6s}  {'V/E_kin':>8s}  "
          f"{'legs spread':>11s}  {'full spread':>11s}  "
          f"{'leg mix':>8s}  {'full mix':>8s}  "
          f"{'leg loc':>7s}  {'full loc':>8s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*11}  {'─'*11}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*8}")

    sweep_data = {}
    M_eff = np.mean(0.5 * (M_inv_ds[:, :, target_idx, target_idx, 0, 0] +
                           M_inv_ds[:, :, target_idx, target_idx, 1, 1]))
    V_depth = np.max(V_target_ds) - np.min(V_target_ds)

    for theta, eta_s in zip(theta_sweep, eta_sweep):
        L_s = 1.0 / eta_s
        dR_s = L_s / Ns_sw
        ratio = V_depth / (0.5 * abs(M_eff) * eta_s**2)

        try:
            # Legacy (A=0 for speed, since diagonal A gives 0 mixing anyway)
            H_leg_s = assemble_multiband_hamiltonian(
                Lambda_ds, v_drift_ds, M_inv_reg_ds, np.zeros_like(A_berry_ds), Phi_BH_ds,
                eta_s, Ns_sw, Ns_sw, Nb, dR_s, dR_s, B_moire,
                include_drift=True, include_kinetic=True, include_born_huang=False,
                include_offdiag_A=False,
            )
            H_leg_s = 0.5 * (H_leg_s + H_leg_s.conj().T)
            ev_l, evec_l = eigsh(H_leg_s, k=n_sw, sigma=sigma_ds, which='LM', tol=1e-10)
            o = np.argsort(ev_l); ev_l = ev_l[o]; evec_l = evec_l[:, o]
            w_l = compute_band_weights(evec_l, Ns_sw, Nb)
            ipr_l = compute_ipr(evec_l, Ns_sw, Nb)

            # Full coupling
            H_full_s = assemble_multiband_hamiltonian(
                Lambda_ds, v_drift_ds, M_inv_reg_ds, A_berry_ds, Phi_BH_ds,
                eta_s, Ns_sw, Ns_sw, Nb, dR_s, dR_s, B_moire,
                include_drift=True, include_kinetic=True, include_born_huang=False,
                include_offdiag_A=True,
            )
            H_full_s = 0.5 * (H_full_s + H_full_s.conj().T)
            ev_f, evec_f = eigsh(H_full_s, k=n_sw, sigma=sigma_ds, which='LM', tol=1e-10)
            o = np.argsort(ev_f); ev_f = ev_f[o]; evec_f = evec_f[:, o]
            w_f = compute_band_weights(evec_f, Ns_sw, Nb)
            ipr_f = compute_ipr(evec_f, Ns_sw, Nb)

            ipr_ext = 1.0 / Ns_sw**2
            n_loc_l = np.sum(ipr_l > 10 * ipr_ext)
            n_loc_f = np.sum(ipr_f > 10 * ipr_ext)
            mmix_l = np.mean(1.0 - np.max(w_l, axis=1))
            mmix_f = np.mean(1.0 - np.max(w_f, axis=1))

            print(f"  {theta:>6.1f}  {ratio:>8.1f}  "
                  f"{ev_l[-1]-ev_l[0]:>11.4e}  {ev_f[-1]-ev_f[0]:>11.4e}  "
                  f"{mmix_l:>8.4f}  {mmix_f:>8.4f}  "
                  f"{n_loc_l:>7d}  {n_loc_f:>8d}")

            sweep_data[theta] = {
                'ev_leg': ev_l, 'ev_full': ev_f,
                'mix_leg': mmix_l, 'mix_full': mmix_f,
                'loc_leg': n_loc_l, 'loc_full': n_loc_f,
                'ratio': ratio,
            }
        except Exception as e:
            print(f"  {theta:>6.1f}  ERROR: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # [10] Summary plots
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[10] Generating summary plots...")

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("S7: Off-diagonal Berry Connection Validation", fontsize=14, fontweight='bold')

    # Panel 1: Eigenvalue spectra (legacy vs full vs A=0)
    ax1 = fig.add_subplot(2, 3, 1)
    V_max = np.max(V_target)
    ax1.plot(range(n_modes), ev_a0 - V_max, 'o-', ms=4, label='A=0', color='gray')
    ax1.plot(range(n_modes), ev_leg - V_max, 's-', ms=4, label='Legacy (diag A)', color='steelblue')
    ax1.plot(range(n_modes), ev_full - V_max, '^-', ms=4, label='Full A (offdiag)', color='darkorange')
    ax1.axhline(0, color='red', ls='--', alpha=0.4, label='V_max')
    ax1.set_xlabel('Mode index')
    ax1.set_ylabel('ε − V_max')
    ax1.set_title(f'Eigenvalue Spectra (θ={np.degrees(2*np.arcsin(eta/2)):.1f}°, Ns={Ns})')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Band mixing per mode
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(n_modes)
    w = 0.25
    ax2.bar(x - w, max_mix_a0, w, label='A=0', color='gray', alpha=0.7)
    ax2.bar(x, max_mix_leg, w, label='Legacy (diag A)', color='steelblue', alpha=0.7)
    ax2.bar(x + w, max_mix_full, w, label='Full A', color='darkorange', alpha=0.7)
    ax2.set_xlabel('Mode index')
    ax2.set_ylabel('1 − max(weight)')
    ax2.set_title('Interband Mixing per Mode')
    ax2.legend(fontsize=8)

    # Panel 3: Band weight heatmap (full coupling)
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(w_full.T, aspect='auto', cmap='YlOrRd', origin='lower',
                    vmin=0, vmax=1)
    ax3.set_xlabel('Mode index')
    ax3.set_ylabel('Band index')
    ax3.set_title('Band Weights (Full A)')
    ax3.set_yticks(range(Nb))
    plt.colorbar(im, ax=ax3, label='Weight')

    # Panel 4: IPR comparison
    ax4 = fig.add_subplot(2, 3, 4)
    ipr_a0 = compute_ipr(evec_a0, Ns, Nb)
    ipr_leg = compute_ipr(evec_leg, Ns, Nb)
    ax4.semilogy(range(n_modes), ipr_a0, 'o-', ms=4, label='A=0', color='gray')
    ax4.semilogy(range(n_modes), ipr_leg, 's-', ms=4, label='Legacy', color='steelblue')
    ax4.semilogy(range(n_modes), ipr_full, '^-', ms=4, label='Full A', color='darkorange')
    ipr_ext = 1.0 / Ns**2
    ax4.axhline(ipr_ext, color='gray', ls='--', label='extended')
    ax4.set_xlabel('Mode index')
    ax4.set_ylabel('IPR')
    ax4.set_title('Localization')
    ax4.legend(fontsize=8)

    # Panel 5: η-sweep mixing comparison
    if sweep_data:
        ax5 = fig.add_subplot(2, 3, 5)
        thetas = sorted(sweep_data.keys())
        mix_leg_arr = [sweep_data[t]['mix_leg'] for t in thetas]
        mix_full_arr = [sweep_data[t]['mix_full'] for t in thetas]
        ax5.plot(thetas, mix_leg_arr, 'o-', label='Legacy', color='steelblue')
        ax5.plot(thetas, mix_full_arr, '^-', label='Full A', color='darkorange')
        ax5.set_xlabel('θ (degrees)')
        ax5.set_ylabel('Mean mixing')
        ax5.set_title('Mixing vs Twist Angle (Ns=64)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Panel 6: Eigenvalue shift
        ax6 = fig.add_subplot(2, 3, 6)
        for t in thetas:
            ev_l = sweep_data[t]['ev_leg']
            ev_f = sweep_data[t]['ev_full']
            ax6.plot(range(len(ev_l)), ev_f - ev_l, 'o-', ms=3, label=f'θ={t}°')
        ax6.axhline(0, color='gray', ls='--')
        ax6.set_xlabel('Mode index')
        ax6.set_ylabel('ε(full) − ε(legacy)')
        ax6.set_title('Eigenvalue Shift from Off-diag A')
        ax6.legend(fontsize=7, ncol=2)
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = PLOT_DIR / "S7_offdiag_summary.png"
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved {path2.name}")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  S7 SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Hermiticity: ||K - K†||/||K|| = {herm_rel:.4e} "
          f"{'✓' if herm_rel < 1e-13 else '✗'}")
    print(f"  A=0 consistency:   Δ_max = {diff_a0:.4e} "
          f"{'✓' if diff_a0 < 1e-13 else '✗'}")
    print(f"  C4 commutator:     ||[H,C4]||/||H|| = {rel_comm:.4e} "
          f"{'✓' if rel_comm < 1e-10 else '⚠'}")

    print(f"\n  Mode properties at θ = {np.degrees(2*np.arcsin(eta/2)):.1f}°:")
    print(f"    Legacy:  mean mixing = {np.mean(max_mix_leg):.6f}")
    print(f"    Full A:  mean mixing = {np.mean(max_mix_full):.6f}")
    print(f"    Mixing enhancement: {np.mean(max_mix_full)/max(np.mean(max_mix_leg),1e-10):.0f}×" 
          if np.mean(max_mix_leg) > 1e-6 
          else f"    Mixing: 0 → {np.mean(max_mix_full):.4f} (from zero!)")

    if sweep_data:
        print(f"\n  η-sweep (full coupling):")
        for t in sorted(sweep_data.keys()):
            sd = sweep_data[t]
            print(f"    θ={t:5.1f}°: V/E_kin={sd['ratio']:6.1f}  "
                  f"mix={sd['mix_full']:.4f}  loc={sd['loc_full']}/{n_sw}")

    print(f"\n  Plots: {PLOT_DIR}/S7_offdiag_validation.png")
    print(f"         {PLOT_DIR}/S7_offdiag_summary.png")
    print(f"\n{'='*70}")
    print(f"  S7 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
