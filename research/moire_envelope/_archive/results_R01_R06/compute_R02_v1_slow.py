#!/usr/bin/env python3
"""
R02: Multi-Band Miniband Dispersion
=====================================
Solve the full 5-band coupled envelope Hamiltonian at finite Bloch wavevector q
along the moiré BZ high-symmetry path Γ → X → M → Γ.

OPTIMIZED: Precompute q-independent operators once per angle, then assemble
H(q) = H₀ + iq₁·C₁ + iq₂·C₂ + q₁²·Q₁₁ + q₂²·Q₂₂ + q₁q₂·Q₁₂
which makes each q-point O(nnz) instead of rebuilding everything.

Output: R02_data.json + R02_data.npz
"""

import numpy as np
import h5py
import json
import sys
import time
from pathlib import Path
from scipy import sparse
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.sparse.linalg import eigsh

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PHASES_DIR = BASE / "phasesV3"
SWEEP_DIR = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "eta_sweep_20260206_173808"
CAND_DIR  = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
OUTDIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PHASES_DIR))
import phase3_mpb_v3 as p3

# ── Config ─────────────────────────────────────────────────────────────────────
THETAS = ['1.100', '3.000', '5.000', '8.000']
N_QPOINTS_PER_SEG = 30
N_MODES = 20
M_INV_MAX = 20

PATH_LABELS = ['Γ', 'X', 'M', 'Γ']
PATH_FRAC = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]


def precompute_q_operators(Lambda, v_drift, M_inv, A_berry, Phi_BH,
                            Ns1, Ns2, N_bands, dR1, dR2, B_moire, order=4):
    """
    Precompute all q-independent operators so that H(q) assembly is O(nnz).

    Returns H₀ (q=0 Hamiltonian) and correction operators:
        H(q) = H₀ + iq₁·C₁ + iq₂·C₂ + q₁²·Q₁₁ + q₂²·Q₂₂ + q₁q₂·Q₁₂
    """
    N_s = Ns1 * Ns2
    N_total = N_s * N_bands

    # ── H₀ (full Hamiltonian at q=0) ──────────────────────────────────────
    H0 = p3.assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        None, Ns1, Ns2, N_bands, dR1, dR2, B_moire,
        M_inv_max_trace=M_INV_MAX
    )

    # ── Build base FD operators ───────────────────────────────────────────
    D1 = p3.build_periodic_derivative_matrix(Ns1, dR1, order)
    D2 = p3.build_periodic_derivative_matrix(Ns2, dR2, order)
    I1, I2, Ib = eye(Ns1), eye(Ns2), eye(N_bands)

    # Full-space FD operators
    D1_full = kron(D1, kron(I2, Ib, format='csr'), format='csr')
    D2_full = kron(I1, kron(D2, Ib, format='csr'), format='csr')

    # ── Drift q-correction: T(q) - T(0) = -i·coeff · (V1·(iq₁I) + V2·(iq₂I))
    # Only the iq₁, iq₂ parts that modify the derivative
    coeff_drift = 1.0 / (2 * np.pi)
    v_flat = v_drift.reshape(N_s, N_bands, N_bands, 2)
    k_grid = np.arange(N_s)

    def build_V_diag(comp):
        rows, cols, data = [], [], []
        for m in range(N_bands):
            for n in range(N_bands):
                vals = v_flat[:, m, n, comp]
                mask = np.abs(vals) > 1e-15
                if np.any(mask):
                    k = k_grid[mask]
                    rows.append(k * N_bands + m)
                    cols.append(k * N_bands + n)
                    data.append(vals[mask])
        if not rows:
            return csr_matrix((N_total, N_total), dtype=complex)
        return csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N_total, N_total))

    V1_op = build_V_diag(0)
    V2_op = build_V_diag(1)

    # Drift q-correction: ΔT = -i·coeff·(V1·(iq₁·I_full) + V2·(iq₂·I_full))
    # = coeff·(q₁·V1 + q₂·V2)  (the i and -i cancel)
    # Actually: D_j → D_j + iq_j, so T(q)-T(0) = -i·coeff·(V1·iq₁·I + V2·iq₂·I)
    # = coeff·(q₁·V1 + q₂·V2)
    C1_drift = coeff_drift * V1_op   # coefficient of q₁ (real part)
    C2_drift = coeff_drift * V2_op   # coefficient of q₂

    # ── Kinetic q-corrections ─────────────────────────────────────────────
    scale = 0.5 / (2 * np.pi)**2

    M_inv_flat = np.zeros((N_total, 2, 2), dtype=complex)
    M_inv_reshaped = M_inv.reshape(N_s, N_bands, N_bands, 2, 2)
    for nb in range(N_bands):
        indices = np.arange(N_s) * N_bands + nb
        M_inv_flat[indices] = M_inv_reshaped[:, nb, nb, :, :]

    M11 = diags(M_inv_flat[:, 0, 0], format='csr')
    M22 = diags(M_inv_flat[:, 1, 1], format='csr')
    M12 = diags(M_inv_flat[:, 0, 1], format='csr')

    # Berry connection
    A_berry_flat = np.zeros((N_total, 2), dtype=complex)
    A_berry_reshaped = A_berry.reshape(N_s, N_bands, N_bands, 2)
    for nb in range(N_bands):
        indices = np.arange(N_s) * N_bands + nb
        A_berry_flat[indices] = A_berry_reshaped[:, nb, nb, :]

    A1_diag = diags(A_berry_flat[:, 0], format='csr')
    A2_diag = diags(A_berry_flat[:, 1], format='csr')

    # Kinetic: K(q) uses L_q = L + 2iq·D - q², so
    # K(q) - K(0) = -scale * [
    #   M11 · (2iq₁·D1 - q₁²·I)
    #   + M22 · (2iq₂·D2 - q₂²·I)
    #   + 2·M12 · (iq₁·D2 + iq₂·D1 - q₁q₂... cross terms)
    #   + Berry A terms with q
    # ]
    # Actually it's cleaner to define total operators for each q power:

    # C₁: coefficient of iq₁ in H(q)-H(0)
    # From drift: coeff_drift·V1·I (but we already have that above with q₁ not iq₁)
    # From kinetic L_q₁ term: M11·(2iq₁·D1) → -scale·M11·2i·D1
    # Wait, let me be more careful. Let me define:
    #   H(q) = H₀ + q₁·A₁ + q₂·A₂ + q₁²·B₁₁ + q₂²·B₂₂ + q₁q₂·B₁₂

    # Linear in q₁ (from drift + kinetic cross terms):
    # Drift: -i·coeff·V1·(iq₁·I) = coeff·q₁·V1
    # Kinetic D → D+iq: contributes 2i·q₁ terms from Laplacian 2iq₁D₁ in M11 term
    #   -scale · M11 · 2iq₁·D1_full → -scale·2i·M11·D1_full · q₁
    #   -scale · 2·M12 · iq₁·D2_full → -scale·2i·M12·D2_full · q₁
    # Berry: coupling of A and q is more complex, but for diagonal approximation:
    #   -scale · (M11·(2iq₁·A₁ - 2A₁²·(iq₁ term)...)) — actually Berry contributes
    #   through the gauge-covariant derivative (D-iA), so the cross terms are
    #   -2i·M·A·q + ... These are typically small; let me include them.

    # For simplicity and correctness, let's compute H(q) = H₀ + dH where
    # dH encodes only the modifications from D→D+iq in the kinetic operator and drift

    # The cleanest approach: compute C1, C2 (linear) and Q11, Q22, Q12 (quadratic)
    # from the kinetic operator structure.

    # Kinetic at q=0: K₀ = -scale * (M11·L1 + M22·L2 + 2·M12·D1·D2) + Berry terms
    # Kinetic at q: K_q = -scale * (M11·(L1 + 2iq₁D1 - q₁²I)
    #                              + M22·(L2 + 2iq₂D2 - q₂²I)
    #                              + 2·M12·(D1+iq₁I)·(D2+iq₂I))
    #                    + Berry with (D-iA+iq) terms

    # K_q - K₀ = -scale * (
    #   M11·(2iq₁·D1 - q₁²·I)
    #   + M22·(2iq₂·D2 - q₂²·I)
    #   + 2·M12·(iq₁·D2 + iq₂·D1 + iq₁·iq₂·I - D1·D2 + D1·D2)  [cross term expansion]
    #   + 2·M12·(iq₁·D2 + iq₂·D1 - q₁q₂·I)
    # )
    # = -scale * (
    #   2iq₁ · M11·D1  +  2iq₂ · M22·D2
    #   - q₁² · M11    -  q₂² · M22
    #   + 2iq₁ · M12·D2  +  2iq₂ · M12·D1
    #   - 2q₁q₂ · M12
    # )

    # Linear terms (coefficient of iq₁ and iq₂):
    C1_kin = -scale * 2 * (M11 @ D1_full + M12 @ D2_full)
    C2_kin = -scale * 2 * (M22 @ D2_full + M12 @ D1_full)

    # Combined linear (multiply by iq):
    #   drift contributes coeff·qⱼ·Vⱼ = -i·(iq₁)·coeff·V1 etc
    # Actually let's define: H(q) = H₀ + iq₁·Ĉ₁ + iq₂·Ĉ₂ + q₁²·Q₁₁ + q₂²·Q₂₂ + q₁q₂·Q₁₂
    # where Ĉⱼ means the coefficient of (iqⱼ)

    # Drift: T(q)-T(0) = -i·coeff·Vⱼ·(iqⱼ·I) summed over j
    #   Coefficient of iq₁: -i·coeff·V1
    #   BUT this is already multiplied by the iq₁ from the derivative
    # Let me redefine more carefully:
    # Drift uses D_j + iq_j. The drift operator is -i·coeff·(V1·D1 + V2·D2).
    # At q: -i·coeff·(V1·(D1+iq₁I) + V2·(D2+iq₂I))
    #      = -i·coeff·(V1·D1 + V2·D2) + -i·coeff·(V1·iq₁I + V2·iq₂I)
    #      = T₀ + coeff·(q₁·V1 + q₂·V2)
    # So drift correction is REAL: q₁·coeff·V1 + q₂·coeff·V2

    # Total linear in q₁: coeff·V1 + i·C1_kin  (since C1_kin multiplied by iq₁)
    # Actually let's just compute directly.

    # Define: dH(q) = sum of:
    #   q₁ · L1_op   (from drift)
    #   q₂ · L2_op   (from drift)
    #   iq₁ · iL1_op (from kinetic)
    #   iq₂ · iL2_op (from kinetic)
    #   -q₁² · Q11_op
    #   -q₂² · Q22_op
    #   -q₁q₂ · Q12_op

    # Quadratic terms:
    Q11 = scale * M11    # coefficient of -q₁² (with negative absorbed)
    Q22 = scale * M22
    Q12 = 2 * scale * M12

    # Hermitize the linear operators
    C1_kin_h = (C1_kin + C1_kin.conj().T) / 2
    C2_kin_h = (C2_kin + C2_kin.conj().T) / 2

    # Total: precompute these sparse matrices
    ops = {
        'H0': H0,
        'C1_drift': C1_drift,  # real: multiply by q₁
        'C2_drift': C2_drift,  # real: multiply by q₂
        'C1_kin': C1_kin_h,    # imaginary: multiply by iq₁
        'C2_kin': C2_kin_h,    # imaginary: multiply by iq₂
        'Q11': Q11,            # multiply by -q₁²
        'Q22': Q22,            # multiply by -q₂²
        'Q12': Q12,            # multiply by -q₁q₂
    }
    return ops


def assemble_H_at_q(ops, q_phys):
    """Fast assembly: H(q) = H₀ + linear + quadratic terms."""
    q1, q2 = q_phys[0], q_phys[1]
    H = ops['H0'].copy()
    # Drift correction (linear, real)
    if abs(q1) > 1e-15:
        H = H + q1 * ops['C1_drift']
    if abs(q2) > 1e-15:
        H = H + q2 * ops['C2_drift']
    # Kinetic correction (linear, from 2iq·D terms)
    if abs(q1) > 1e-15:
        H = H + (1j * q1) * ops['C1_kin']
    if abs(q2) > 1e-15:
        H = H + (1j * q2) * ops['C2_kin']
    # Quadratic terms
    if abs(q1) > 1e-15:
        H = H - q1**2 * ops['Q11']
    if abs(q2) > 1e-15:
        H = H - q2**2 * ops['Q22']
    if abs(q1 * q2) > 1e-15:
        H = H - q1 * q2 * ops['Q12']
    # Enforce Hermiticity
    H = 0.5 * (H + H.conj().T)
    return H


def load_phase2_data(theta_str):
    """Load Phase 2 data for a given twist angle."""
    if theta_str == '1.100':
        cdir = CAND_DIR
    else:
        cdir = SWEEP_DIR / f"theta_{theta_str}" / "candidate_0000"

    p2file = cdir / "phase2_multiband_data.h5"
    if not p2file.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {p2file}")

    with h5py.File(p2file, 'r') as hf:
        d = {
            'Lambda': hf['Lambda'][:],
            'M_inv': hf['M_inv'][:],
            'A_berry': hf['A_berry'][:],
            'v_drift': hf['v_drift'][:],
            'Phi_BH': hf['Phi_BH'][:],
            'eta': float(hf.attrs['eta']),
            'Ns1': int(hf.attrs['Ns1']),
            'Ns2': int(hf.attrs['Ns2']),
            'N_sub': int(hf.attrs['N_subspace']),
            'B_moire': hf.attrs['B_moire'],
            'L_moire': float(hf.attrs['moire_length']),
            'theta_deg': float(hf.attrs['theta_deg']),
            'omega_ref': float(hf.attrs['omega_ref']),
        }
    return d


def compute_q_path(B_moire, n_per_seg):
    """Compute q-points along Γ→X→M→Γ in physical units."""
    G_moire = 2 * np.pi * np.linalg.inv(B_moire).T

    q_points_frac = []
    q_distances = []
    tick_positions = []
    cumulative_dist = 0.0

    for seg in range(len(PATH_FRAC) - 1):
        q0 = np.array(PATH_FRAC[seg])
        q1 = np.array(PATH_FRAC[seg + 1])
        dq_frac = q1 - q0
        dq_phys = G_moire @ dq_frac
        seg_len = np.linalg.norm(dq_phys)

        tick_positions.append(cumulative_dist)

        for i in range(n_per_seg):
            t = i / n_per_seg
            qf = q0 + t * dq_frac
            q_points_frac.append(qf)
            q_distances.append(cumulative_dist + t * seg_len)

        cumulative_dist += seg_len

    # Final point
    q_points_frac.append(np.array(PATH_FRAC[-1]))
    q_distances.append(cumulative_dist)
    tick_positions.append(cumulative_dist)

    q_points_phys = [G_moire @ qf for qf in q_points_frac]

    return q_points_frac, q_points_phys, np.array(q_distances), tick_positions


def main():
    print("="*70)
    print("R02: Multi-Band Miniband Dispersion")
    print("="*70)

    all_results = {}

    for theta_str in THETAS:
        print(f"\n{'─'*60}")
        print(f"θ = {theta_str}°")
        print(f"{'─'*60}")

        d = load_phase2_data(theta_str)
        Ns1, Ns2, N_bands = d['Ns1'], d['Ns2'], d['N_sub']
        L_moire = d['L_moire']
        B_moire = d['B_moire']
        eta = d['eta']

        dR1 = L_moire / Ns1
        dR2 = L_moire / Ns2

        # Regularize M_inv
        M_inv_reg = p3._regularize_M_inv(d['M_inv'], M_INV_MAX)

        # Compute q-path
        q_frac, q_phys, q_dists, ticks = compute_q_path(B_moire, N_QPOINTS_PER_SEG)
        n_q = len(q_frac)
        print(f"  Grid: {Ns1}×{Ns2}×{N_bands}, η={eta:.5f}, L={L_moire:.1f}a")
        print(f"  q-path: {n_q} points, {N_MODES} eigenvalues each")

        # Precompute q-independent operators (one-time cost)
        t_pre = time.time()
        ops = precompute_q_operators(
            d['Lambda'], d['v_drift'], M_inv_reg, d['A_berry'], d['Phi_BH'],
            Ns1, Ns2, N_bands, dR1, dR2, B_moire
        )
        print(f"  Precomputed operators in {time.time()-t_pre:.1f}s")

        # Determine sigma from q=0
        h_diag = ops['H0'].diagonal().real
        sigma = float(np.min(h_diag))
        print(f"  Shift-invert σ = {sigma:.6f}")

        # Solve at each q-point (now fast: just sparse add + eigsh)
        all_evals = np.zeros((n_q, N_MODES))
        all_weights = np.zeros((n_q, N_MODES, N_bands))

        t_start = time.time()
        for qi, (qf, qp) in enumerate(zip(q_frac, q_phys)):
            H_q = assemble_H_at_q(ops, qp)

            try:
                evals, evecs = eigsh(H_q, k=N_MODES, sigma=sigma, which='LM',
                                     maxiter=5000, tol=1e-8)
            except Exception as e:
                print(f"  Warning: eigsh failed at q={qi}: {e}")
                evals = np.full(N_MODES, np.nan)
                evecs = np.zeros((H_q.shape[0], N_MODES))

            # Sort by eigenvalue
            idx = np.argsort(evals.real)
            evals = evals[idx].real
            evecs = evecs[:, idx]

            all_evals[qi] = evals

            # Compute band character for each mode
            N_s = Ns1 * Ns2
            for mi in range(N_MODES):
                v = evecs[:, mi]
                for nb in range(N_bands):
                    # Band nb occupies indices k*N_bands + nb for k=0..N_s-1
                    indices = np.arange(N_s) * N_bands + nb
                    all_weights[qi, mi, nb] = np.sum(np.abs(v[indices])**2)

            if (qi + 1) % 10 == 0 or qi == 0:
                elapsed = time.time() - t_start
                eta_time = elapsed / (qi + 1) * (n_q - qi - 1)
                print(f"  q={qi+1}/{n_q} done [{elapsed:.0f}s elapsed, ~{eta_time:.0f}s remaining]")

        elapsed = time.time() - t_start
        print(f"  Completed in {elapsed:.1f}s")

        # Compute group velocity (finite difference along path)
        vg = np.zeros((n_q, N_MODES))
        for mi in range(N_MODES):
            vg[1:-1, mi] = (all_evals[2:, mi] - all_evals[:-2, mi]) / (q_dists[2:] - q_dists[:-2])
        vg[0] = vg[1]
        vg[-1] = vg[-2]

        # Store results
        all_results[theta_str] = {
            'theta_deg': float(theta_str),
            'eta': eta,
            'L_moire': L_moire,
            'omega_ref': d['omega_ref'],
            'eigenvalues': all_evals,       # (n_q, N_MODES)
            'band_weights': all_weights,     # (n_q, N_MODES, N_bands)
            'q_distances': q_dists,          # (n_q,)
            'q_frac': [qf.tolist() for qf in q_frac],
            'tick_positions': ticks,
            'group_velocity': vg,            # (n_q, N_MODES)
        }

        # Print summary
        bw = all_evals.max(axis=0) - all_evals.min(axis=0)
        print(f"  Bandwidths (first 5): {['%.2e' % b for b in bw[:5]]}")

    # ── Save ──────────────────────────────────────────────────────────────
    # NPZ for arrays
    npz_data = {}
    json_data = {'path_labels': PATH_LABELS, 'n_qpoints_per_segment': N_QPOINTS_PER_SEG,
                 'n_modes': N_MODES, 'thetas': THETAS, 'per_theta': {}}

    for theta_str, res in all_results.items():
        prefix = f"t{theta_str}"
        npz_data[f"{prefix}_eigenvalues"] = res['eigenvalues']
        npz_data[f"{prefix}_band_weights"] = res['band_weights']
        npz_data[f"{prefix}_q_distances"] = res['q_distances']
        npz_data[f"{prefix}_group_velocity"] = res['group_velocity']

        # JSON-serializable summary
        evals = res['eigenvalues']
        bw = evals.max(axis=0) - evals.min(axis=0)
        json_data['per_theta'][theta_str] = {
            'theta_deg': res['theta_deg'],
            'eta': res['eta'],
            'L_moire': res['L_moire'],
            'omega_ref': res['omega_ref'],
            'tick_positions': res['tick_positions'],
            'bandwidths': bw.tolist(),
            'E_min': float(evals.min()),
            'E_max': float(evals.max()),
        }

    outfile_npz = OUTDIR / "R02_data.npz"
    np.savez_compressed(outfile_npz, **npz_data)
    print(f"\nSaved arrays to {outfile_npz}")

    outfile_json = OUTDIR / "R02_data.json"
    with open(outfile_json, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved metadata to {outfile_json}")
    print("\nR02 compute complete.")


if __name__ == '__main__':
    main()
