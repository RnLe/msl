#!/usr/bin/env python3
"""
R02: Multi-Band Miniband Dispersion
=====================================
Solve the full 5-band coupled envelope Hamiltonian at finite Bloch wavevector q
along the moiré BZ high-symmetry path Γ → X → M → Γ.

SPEED OPTIMIZATION:
  Phase 2 data lives on a 128×128 grid → 81,920-dim Hamiltonian → eigsh ~3 min/q.
  We DOWNSAMPLE to NS_Q×NS_Q (default 32) → 5,120-dim → eigsh ~1s/q.
  The moiré potential is slowly-varying, so 32×32 still captures band structure.
  Precompute q-independent operators so H(q) assembly is O(nnz).

Output: R02_data.json + R02_data.npz
"""

import numpy as np
import h5py
import json
import sys
import time
from pathlib import Path
from scipy import sparse
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
NS_Q = 32                  # downsample grid (32→5120-dim, ~1s/eigsh)
N_QPOINTS_PER_SEG = 12     # 37 total q-points
N_MODES = 10                # eigenvalues per q-point
M_INV_MAX = 20
THETAS = ['1.100', '3.000', '5.000', '8.000']

PATH_LABELS = ['Γ', 'X', 'M', 'Γ']
PATH_FRAC = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]


# ── Fourier downsampling ──────────────────────────────────────────────────────

def downsample_field(field, Ns_orig, Ns_new):
    """
    Downsample a periodic field via Fourier truncation.
    Preserves smoothness better than interpolation for periodic data.
    field: (Ns_orig, Ns_orig, *trailing)
    """
    if Ns_orig == Ns_new:
        return field
    trailing = field.shape[2:]
    flat = field.reshape(Ns_orig, Ns_orig, -1)
    n_ch = flat.shape[2]
    out = np.zeros((Ns_new, Ns_new, n_ch), dtype=flat.dtype)
    half = Ns_new // 2
    for ch in range(n_ch):
        F = np.fft.fft2(flat[:, :, ch])
        Ft = np.zeros((Ns_new, Ns_new), dtype=complex)
        Ft[:half, :half] = F[:half, :half]
        Ft[:half, -half:] = F[:half, -half:]
        Ft[-half:, :half] = F[-half:, :half]
        Ft[-half:, -half:] = F[-half:, -half:]
        result = np.fft.ifft2(Ft) * (Ns_new / Ns_orig)**2
        out[:, :, ch] = result if np.iscomplexobj(flat) else result.real
    return out.reshape((Ns_new, Ns_new) + trailing)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_phase2_data(theta_str, Ns_target):
    """Load Phase 2 data, downsampled to Ns_target."""
    cdir = CAND_DIR if theta_str == '1.100' else (
        SWEEP_DIR / f"theta_{theta_str}" / "candidate_0000")
    p2 = cdir / "phase2_multiband_data.h5"
    if not p2.exists():
        raise FileNotFoundError(p2)

    with h5py.File(p2, 'r') as hf:
        Ns_orig = int(hf.attrs['Ns1'])
        fields = {k: hf[k][:] for k in ['Lambda', 'M_inv', 'A_berry', 'v_drift', 'Phi_BH']}
        meta = {k: hf.attrs[k] for k in ['eta', 'N_subspace', 'moire_length', 'theta_deg', 'omega_ref']}
        B_moire = hf.attrs['B_moire']

    if Ns_target != Ns_orig:
        print(f"  Downsampling {Ns_orig}→{Ns_target}...", end=" ", flush=True)
        t0 = time.time()
        for k in fields:
            fields[k] = downsample_field(fields[k], Ns_orig, Ns_target)
        print(f"{time.time()-t0:.1f}s")

    return {
        **fields,
        'Ns': Ns_target, 'N_sub': int(meta['N_subspace']),
        'eta': float(meta['eta']), 'L_moire': float(meta['moire_length']),
        'theta_deg': float(meta['theta_deg']), 'omega_ref': float(meta['omega_ref']),
        'B_moire': B_moire,
    }


# ── q-path ────────────────────────────────────────────────────────────────────

def compute_q_path(B_moire, n_per_seg):
    """q-points along Γ→X→M→Γ in physical units."""
    G = 2 * np.pi * np.linalg.inv(B_moire).T
    q_frac, q_phys, q_dist = [], [], []
    ticks = []
    cum = 0.0
    for seg in range(len(PATH_FRAC) - 1):
        q0, q1 = np.array(PATH_FRAC[seg]), np.array(PATH_FRAC[seg + 1])
        seg_len = np.linalg.norm(G @ (q1 - q0))
        ticks.append(cum)
        for i in range(n_per_seg):
            t = i / n_per_seg
            qf = q0 + t * (q1 - q0)
            q_frac.append(qf)
            q_phys.append(G @ qf)
            q_dist.append(cum + t * seg_len)
        cum += seg_len
    q_frac.append(np.array(PATH_FRAC[-1]))
    q_phys.append(G @ np.array(PATH_FRAC[-1]))
    q_dist.append(cum)
    ticks.append(cum)
    return q_frac, q_phys, np.array(q_dist), ticks


# ── Precompute q-operators ────────────────────────────────────────────────────

def build_q_operators(d):
    """
    Precompute q-independent parts. H(q) = H₀ + q₁L₁ + q₂L₂ - q₁²Q₁₁ - q₂²Q₂₂ - q₁q₂Q₁₂.
    """
    Ns = d['Ns']
    Nb = d['N_sub']
    L = d['L_moire']
    dR = L / Ns
    eta = d['eta']
    N_s = Ns * Ns
    N_total = N_s * Nb

    M_inv_reg = p3._regularize_M_inv(d['M_inv'], M_INV_MAX)

    # H₀ at q=0
    H0 = p3.assemble_multiband_hamiltonian(
        d['Lambda'], d['v_drift'], M_inv_reg, d['A_berry'], d['Phi_BH'],
        eta, Ns, Ns, Nb, dR, dR, d['B_moire'], M_inv_max_trace=M_INV_MAX)

    # Base 1D FD operators
    D1 = p3.build_periodic_derivative_matrix(Ns, dR, 4)
    D2 = p3.build_periodic_derivative_matrix(Ns, dR, 4)
    I1, I2, Ib = sparse.eye(Ns), sparse.eye(Ns), sparse.eye(Nb)
    D1f = sparse.kron(D1, sparse.kron(I2, Ib, 'csr'), 'csr')
    D2f = sparse.kron(I1, sparse.kron(D2, Ib, 'csr'), 'csr')

    # Block-diagonal mass and drift operators
    def build_block_diag(field, trailing_shape):
        """Build sparse block-diagonal from (Ns, Ns, Nb, Nb, ...) field."""
        flat = field.reshape(N_s, Nb, Nb, *trailing_shape)
        rows, cols, data = [], [], []
        for m in range(Nb):
            for n in range(Nb):
                if len(trailing_shape) == 0:
                    vals = flat[:, m, n]
                else:
                    vals = flat[:, m, n]  # returns (N_s, *trailing_shape)
                    return None  # handled separately for vector fields
                mask = np.abs(vals) > 1e-15
                if mask.any():
                    k = np.arange(N_s)[mask]
                    rows.append(k * Nb + m)
                    cols.append(k * Nb + n)
                    data.append(vals[mask])
        if not rows:
            return sparse.csr_matrix((N_total, N_total), dtype=complex)
        return sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N_total, N_total))

    def build_block_diag_comp(field, comp_indices):
        """Build sparse from (Ns, Ns, Nb, Nb, 2, 2) selecting specific components."""
        flat = field.reshape(N_s, Nb, Nb, *field.shape[4:])
        rows, cols, data = [], [], []
        for m in range(Nb):
            for n in range(Nb):
                vals = flat[:, m, n]
                for idx in comp_indices:
                    vals = vals[:, idx] if isinstance(idx, int) else vals
                # Now vals should be (N_s,)
                mask = np.abs(vals) > 1e-15
                if mask.any():
                    k = np.arange(N_s)[mask]
                    rows.append(k * Nb + m)
                    cols.append(k * Nb + n)
                    data.append(vals[mask])
        if not rows:
            return sparse.csr_matrix((N_total, N_total), dtype=complex)
        return sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N_total, N_total))

    # Mass tensor components M_{ij}
    M_flat = M_inv_reg.reshape(N_s, Nb, Nb, 2, 2)

    def build_Mij(i, j):
        rows, cols, data = [], [], []
        for m in range(Nb):
            for n in range(Nb):
                vals = M_flat[:, m, n, i, j]
                mask = np.abs(vals) > 1e-15
                if mask.any():
                    k = np.arange(N_s)[mask]
                    rows.append(k * Nb + m)
                    cols.append(k * Nb + n)
                    data.append(vals[mask])
        if not rows:
            return sparse.csr_matrix((N_total, N_total), dtype=complex)
        return sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N_total, N_total))

    M11, M22, M12 = build_Mij(0, 0), build_Mij(1, 1), build_Mij(0, 1)

    # Drift velocity V_j
    v_flat = d['v_drift'].reshape(N_s, Nb, Nb, 2)

    def build_Vj(comp):
        rows, cols, data = [], [], []
        for m in range(Nb):
            for n in range(Nb):
                vals = v_flat[:, m, n, comp]
                mask = np.abs(vals) > 1e-15
                if mask.any():
                    k = np.arange(N_s)[mask]
                    rows.append(k * Nb + m)
                    cols.append(k * Nb + n)
                    data.append(vals[mask])
        if not rows:
            return sparse.csr_matrix((N_total, N_total), dtype=complex)
        return sparse.csr_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(N_total, N_total))

    V1, V2 = build_Vj(0), build_Vj(1)

    # Correction coefficients
    c_drift = eta / (2 * np.pi)
    c_kin = 0.5 * eta**2 / (2 * np.pi)**2

    # Linear: L_j = c_drift·V_j - 2i·c_kin·(M_jj·D_j + M_j_other·D_other)
    L1 = c_drift * V1 + c_kin * (-2j) * (M11 @ D1f + M12 @ D2f)
    L2 = c_drift * V2 + c_kin * (-2j) * (M22 @ D2f + M12 @ D1f)

    # Quadratic
    Q11 = c_kin * M11
    Q22 = c_kin * M22
    Q12 = 2 * c_kin * M12

    return {'H0': H0.tocsr(), 'L1': L1.tocsr(), 'L2': L2.tocsr(),
            'Q11': Q11.tocsr(), 'Q22': Q22.tocsr(), 'Q12': Q12.tocsr()}


def assemble_H_q(ops, q):
    """H(q) = H₀ + q₁L₁ + q₂L₂ - q₁²Q₁₁ - q₂²Q₂₂ - q₁q₂Q₁₂."""
    q1, q2 = q
    H = ops['H0'].copy()
    if abs(q1) > 1e-15:
        H += q1 * ops['L1'] - q1**2 * ops['Q11']
    if abs(q2) > 1e-15:
        H += q2 * ops['L2'] - q2**2 * ops['Q22']
    if abs(q1 * q2) > 1e-15:
        H -= q1 * q2 * ops['Q12']
    return 0.5 * (H + H.conj().T)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R02: Multi-Band Miniband Dispersion (fast)")
    print(f"  Grid: {NS_Q}×{NS_Q}, q/seg: {N_QPOINTS_PER_SEG}, modes: {N_MODES}")
    print(f"  Angles: {THETAS}")
    print("=" * 70)

    all_results = {}

    for theta_str in THETAS:
        print(f"\n{'─'*60}")
        print(f"θ = {theta_str}°")
        print(f"{'─'*60}")

        d = load_phase2_data(theta_str, NS_Q)
        Ns, Nb = d['Ns'], d['N_sub']
        N_s = Ns * Ns
        mat_dim = N_s * Nb
        print(f"  Hamiltonian: {mat_dim}×{mat_dim}, η={d['eta']:.5f}")

        t0 = time.time()
        ops = build_q_operators(d)
        print(f"  Built operators in {time.time()-t0:.1f}s")

        q_frac, q_phys, q_dist, ticks = compute_q_path(d['B_moire'], N_QPOINTS_PER_SEG)
        n_q = len(q_frac)
        print(f"  q-path: {n_q} points")

        sigma = float(np.min(ops['H0'].diagonal().real))
        evals_all = np.zeros((n_q, N_MODES))
        weights_all = np.zeros((n_q, N_MODES, Nb))

        t_start = time.time()
        for qi in range(n_q):
            H_q = assemble_H_q(ops, q_phys[qi])
            try:
                evals, evecs = eigsh(H_q, k=N_MODES, sigma=sigma, which='LM',
                                     maxiter=3000, tol=1e-6)
            except Exception as e:
                print(f"  eigsh failed at q={qi}: {e}")
                evals = np.full(N_MODES, np.nan)
                evecs = np.zeros((mat_dim, N_MODES))

            idx = np.argsort(evals.real)
            evals_all[qi] = evals[idx].real
            evecs = evecs[:, idx]

            for mi in range(N_MODES):
                v = evecs[:, mi]
                for nb in range(Nb):
                    weights_all[qi, mi, nb] = np.sum(np.abs(v[np.arange(N_s)*Nb+nb])**2)

            if qi % 5 == 0 or qi == n_q - 1:
                el = time.time() - t_start
                rem = el / (qi+1) * (n_q-qi-1) if qi > 0 else 0
                print(f"  q={qi+1}/{n_q} [{el:.0f}s, ~{rem:.0f}s left]")

        total = time.time() - t_start
        print(f"  Done in {total:.1f}s ({total/n_q:.2f}s/q)")

        # Group velocity
        vg = np.zeros_like(evals_all)
        for mi in range(N_MODES):
            dq = np.diff(q_dist)
            de = np.diff(evals_all[:, mi])
            vg_mid = de / np.where(dq > 1e-15, dq, 1e-15)
            vg[0, mi] = vg_mid[0]
            vg[-1, mi] = vg_mid[-1]
            vg[1:-1, mi] = 0.5 * (vg_mid[:-1] + vg_mid[1:])

        all_results[theta_str] = {
            'theta_deg': d['theta_deg'], 'eta': d['eta'],
            'L_moire': d['L_moire'], 'omega_ref': d['omega_ref'],
            'eigenvalues': evals_all, 'band_weights': weights_all,
            'q_distances': q_dist, 'q_frac': [qf.tolist() for qf in q_frac],
            'tick_positions': ticks, 'group_velocity': vg,
        }
        bw = evals_all.max(0) - evals_all.min(0)
        print(f"  BW (5 lowest): {['%.2e' % b for b in bw[:5]]}")

    # ── Save ──
    npz, jd = {}, {
        'path_labels': PATH_LABELS, 'n_qpoints_per_segment': N_QPOINTS_PER_SEG,
        'n_modes': N_MODES, 'Ns_q': NS_Q, 'thetas': THETAS, 'per_theta': {},
    }
    for ts, r in all_results.items():
        p = f"t{ts}"
        npz[f"{p}_eigenvalues"] = r['eigenvalues']
        npz[f"{p}_band_weights"] = r['band_weights']
        npz[f"{p}_q_distances"] = r['q_distances']
        npz[f"{p}_group_velocity"] = r['group_velocity']
        bw = r['eigenvalues'].max(0) - r['eigenvalues'].min(0)
        jd['per_theta'][ts] = {
            'theta_deg': r['theta_deg'], 'eta': r['eta'],
            'L_moire': r['L_moire'], 'omega_ref': r['omega_ref'],
            'tick_positions': r['tick_positions'], 'bandwidths': bw.tolist(),
            'E_min': float(r['eigenvalues'].min()),
            'E_max': float(r['eigenvalues'].max()),
        }

    np.savez_compressed(OUTDIR / "R02_data.npz", **npz)
    print(f"\nSaved R02_data.npz")
    with open(OUTDIR / "R02_data.json", 'w') as f:
        json.dump(jd, f, indent=2)
    print("Saved R02_data.json")
    print("\nR02 complete.")


if __name__ == '__main__':
    main()
