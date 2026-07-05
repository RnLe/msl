#!/usr/bin/env python3
"""
R01: Effective Hamiltonian Landscape
====================================
Extract all Hamiltonian parameter fields from Phase 2 data and compute
derived quantities for visualization.

Output: R01_data.npz (compressed numpy arrays)
"""

import numpy as np
import h5py
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CDIR = BASE / "runsV3" / "phase0_mpb_v3_20260206_152443" / "candidate_0000"
OUTDIR = Path(__file__).resolve().parent

def main():
    print("="*70)
    print("R01: Effective Hamiltonian Landscape")
    print("="*70)

    # ── Load Phase 2 data ──────────────────────────────────────────────────
    p2_file = CDIR / "phase2_multiband_data.h5"
    print(f"Loading {p2_file}")
    with h5py.File(p2_file, 'r') as hf:
        Lambda   = hf['Lambda'][:]        # (Ns1, Ns2, Nb, Nb)
        V        = hf['V'][:]             # (Ns1, Ns2, Nb)
        A_berry  = hf['A_berry'][:]       # (Ns1, Ns2, Nb, Nb, 2)
        M_inv    = hf['M_inv'][:]         # (Ns1, Ns2, Nb, Nb, 2, 2)
        Phi_BH   = hf['Phi_BH'][:]       # (Ns1, Ns2, Nb, Nb)
        v_drift  = hf['v_drift'][:]      # (Ns1, Ns2, Nb, Nb, 2)
        omega    = hf['omega'][:]         # (Ns1, Ns2, Nb)
        R_grid   = hf['R_grid'][:]        # (Ns1, Ns2, 2)
        s_grid   = hf['s_grid'][:]        # (Ns1, Ns2, 2)
        # Attrs
        eta       = float(hf.attrs['eta'])
        omega_ref = float(hf.attrs['omega_ref'])
        theta_deg = float(hf.attrs['theta_deg'])
        Ns1       = int(hf.attrs['Ns1'])
        Ns2       = int(hf.attrs['Ns2'])
        N_sub     = int(hf.attrs['N_subspace'])
        L_moire   = float(hf.attrs['moire_length'])
        B_moire   = hf.attrs['B_moire']
        target_idx = int(hf.attrs['target_index_in_subspace'])
        sub_bands = hf.attrs['subspace_bands']

    print(f"  Grid: {Ns1}×{Ns2}, N_sub={N_sub}, eta={eta:.5f}, theta={theta_deg}°")
    print(f"  L_moire={L_moire:.2f} a, omega_ref={omega_ref:.6f}")

    # ── Compute derived quantities ─────────────────────────────────────────
    print("Computing derived quantities...")

    # 1) Diagonal V per band: V_n(s) = Lambda_nn(s) = omega_n - omega_ref
    V_diag = np.zeros((Ns1, Ns2, N_sub))
    for n in range(N_sub):
        V_diag[:, :, n] = Lambda[:, :, n, n].real

    # 2) Berry connection magnitude (diagonal): |A_nn(s)|
    A_diag_mag = np.zeros((Ns1, Ns2, N_sub))
    for n in range(N_sub):
        A_diag_mag[:, :, n] = np.sqrt(
            np.abs(A_berry[:, :, n, n, 0])**2 +
            np.abs(A_berry[:, :, n, n, 1])**2
        )

    # 3) Trace of inverse mass tensor (diagonal): Tr[M^{-1}_nn]
    M_trace = np.zeros((Ns1, Ns2, N_sub))
    for n in range(N_sub):
        M_trace[:, :, n] = M_inv[:, :, n, n, 0, 0].real + M_inv[:, :, n, n, 1, 1].real

    # 4) Mass anisotropy: |M11 - M22| / (|M11| + |M22|)
    M_aniso = np.zeros((Ns1, Ns2, N_sub))
    for n in range(N_sub):
        m11 = np.abs(M_inv[:, :, n, n, 0, 0].real)
        m22 = np.abs(M_inv[:, :, n, n, 1, 1].real)
        denom = m11 + m22
        mask = denom > 1e-10
        M_aniso[mask, n] = np.abs(m11[mask] - m22[mask]) / denom[mask]

    # 5) Born-Huang diagonal: Phi_BH_nn(s)
    Phi_diag = np.zeros((Ns1, Ns2, N_sub))
    for n in range(N_sub):
        Phi_diag[:, :, n] = Phi_BH[:, :, n, n].real

    # 6) Off-diagonal coupling magnitudes
    #    Key pairs: nearest bands (0-1, 1-2, 2-3, 3-4) and target with all others
    off_diag_pairs = []
    for m in range(N_sub):
        for n in range(m+1, N_sub):
            Lambda_mn = np.abs(Lambda[:, :, m, n])
            A_mn = np.sqrt(np.abs(A_berry[:, :, m, n, 0])**2 +
                           np.abs(A_berry[:, :, m, n, 1])**2)
            v_mn = np.sqrt(np.abs(v_drift[:, :, m, n, 0])**2 +
                           np.abs(v_drift[:, :, m, n, 1])**2)
            Phi_mn = np.abs(Phi_BH[:, :, m, n])
            off_diag_pairs.append({
                'pair': (m, n),
                'Lambda_mn_max': float(Lambda_mn.max()),
                'Lambda_mn_mean': float(Lambda_mn.mean()),
                'A_mn_max': float(A_mn.max()),
                'A_mn_mean': float(A_mn.mean()),
                'v_mn_max': float(v_mn.max()),
                'v_mn_mean': float(v_mn.mean()),
                'Phi_mn_max': float(Phi_mn.max()),
                'Phi_mn_mean': float(Phi_mn.mean()),
            })
    
    # 7) Per-band summary statistics
    band_stats = []
    for n in range(N_sub):
        Vn = V_diag[:, :, n]
        band_type = 'hole' if np.mean(M_trace[:, :, n]) < 0 else 'electron'
        stats = {
            'band_index': int(sub_bands[n]),
            'subspace_index': n,
            'type': band_type,
            'V_min': float(Vn.min()),
            'V_max': float(Vn.max()),
            'V_range': float(Vn.max() - Vn.min()),
            'V_mean': float(Vn.mean()),
            'M_trace_mean': float(np.mean(M_trace[:, :, n])),
            'M_trace_min': float(np.min(M_trace[:, :, n])),
            'M_trace_max': float(np.max(M_trace[:, :, n])),
            'M_aniso_mean': float(np.mean(M_aniso[:, :, n])),
            'A_mean': float(np.mean(A_diag_mag[:, :, n])),
            'A_max': float(np.max(A_diag_mag[:, :, n])),
            'Phi_BH_mean': float(np.mean(np.abs(Phi_diag[:, :, n]))),
            'Phi_BH_max': float(np.max(np.abs(Phi_diag[:, :, n]))),
        }
        band_stats.append(stats)
        print(f"  Band {n} ({band_type}): V=[{stats['V_min']:.4f}, {stats['V_max']:.4f}], "
              f"Tr(M⁻¹)={stats['M_trace_mean']:.2f}, |A|_max={stats['A_max']:.4f}")

    # ── Save results ───────────────────────────────────────────────────────
    outfile_npz = OUTDIR / "R01_data.npz"
    np.savez_compressed(
        outfile_npz,
        V_diag=V_diag,
        A_diag_mag=A_diag_mag,
        M_trace=M_trace,
        M_aniso=M_aniso,
        Phi_diag=Phi_diag,
        s_grid=s_grid,
        R_grid=R_grid,
    )
    print(f"\nSaved arrays to {outfile_npz}")

    outfile_json = OUTDIR / "R01_data.json"
    meta = {
        'theta_deg': theta_deg,
        'eta': eta,
        'omega_ref': omega_ref,
        'L_moire': L_moire,
        'Ns1': Ns1, 'Ns2': Ns2,
        'N_subspace': N_sub,
        'target_index_in_subspace': target_idx,
        'subspace_bands': sub_bands.tolist(),
        'band_stats': band_stats,
        'off_diag_coupling': [
            {**d, 'pair': list(d['pair'])} for d in off_diag_pairs
        ],
    }
    with open(outfile_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {outfile_json}")

    print("\nR01 compute complete.")


if __name__ == '__main__':
    main()
