"""
Symmetrize Phase 2 data — generalized C4/C2/C3/C6 symmetrization.

Adapted from corrections_findings/S4b_c4_symmetrize.py to work with any
candidate directory and support C4 (square), C2 (hex M-point), C3, and C6
(honeycomb K-point on hexagonal lattice).

Usage:
    python thesis_results/symmetrize.py <candidate_dir> --sym C4
    python thesis_results/symmetrize.py <candidate_dir> --sym C2
    python thesis_results/symmetrize.py <candidate_dir> --sym C3
    python thesis_results/symmetrize.py <candidate_dir> --sym C6
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# ===========================================================================
# Grid operations
# ===========================================================================


def apply_C4_inv_grid(Q, Ns):
    """Pull-back by C4^{-1}: Q_pulled[ix, iy, ...] = Q[iy, (Ns-ix)%Ns, ...]."""
    Q_pulled = np.empty_like(Q)
    for ix in range(Ns):
        for iy in range(Ns):
            Q_pulled[ix, iy] = Q[iy, (Ns - ix) % Ns]
    return Q_pulled


def apply_C2_inv_grid(Q, Ns):
    """Pull-back by C2^{-1} = C2: Q_pulled[ix, iy, ...] = Q[(Ns-ix)%Ns, (Ns-iy)%Ns, ...]."""
    Q_pulled = np.empty_like(Q)
    for ix in range(Ns):
        for iy in range(Ns):
            Q_pulled[ix, iy] = Q[(Ns - ix) % Ns, (Ns - iy) % Ns]
    return Q_pulled


def apply_C6_inv_grid(Q, Ns):
    """Pull-back by C6^{-1} on hexagonal lattice coordinates.

    For a triangular lattice with basis vectors a₁=(1,0), a₂=(1/2,√3/2):
    C6 acts on lattice coords as S=[[0,-1],[1,1]], so
    C6^{-1} = S^5 = [[1,1],[-1,0]]:  (s₁,s₂) → (s₁+s₂, -s₁)

    Pull-back: Q_pulled[ix, iy] = Q[(ix+iy)%Ns, (Ns-ix)%Ns]
    """
    Q_pulled = np.empty_like(Q)
    for ix in range(Ns):
        for iy in range(Ns):
            Q_pulled[ix, iy] = Q[(ix + iy) % Ns, (Ns - ix) % Ns]
    return Q_pulled


def apply_C3_inv_grid(Q, Ns):
    """Pull-back by C3^{-1} on hexagonal lattice coordinates.

    C3 = C6², so C3^{-1} = S^4 = [[0,1],[-1,-1]]:  (s₁,s₂) → (s₂, -s₁-s₂)

    Pull-back: Q_pulled[ix, iy] = Q[iy, (Ns-ix-iy)%Ns]
    """
    Q_pulled = np.empty_like(Q)
    for ix in range(Ns):
        for iy in range(Ns):
            Q_pulled[ix, iy] = Q[iy, (Ns - ix - iy) % Ns]
    return Q_pulled


def apply_rot_n_grid(Q, n_rot, Ns, sym_type):
    """Apply rotation n times."""
    result = Q.copy()
    order = get_order(sym_type)
    if sym_type == "C4":
        for _ in range(n_rot % order):
            result = apply_C4_inv_grid(result, Ns)
    elif sym_type == "C2":
        for _ in range(n_rot % order):
            result = apply_C2_inv_grid(result, Ns)
    elif sym_type == "C6":
        for _ in range(n_rot % order):
            result = apply_C6_inv_grid(result, Ns)
    elif sym_type == "C3":
        for _ in range(n_rot % order):
            result = apply_C3_inv_grid(result, Ns)
    return result


# Rotation matrices (Cartesian)
C4_MATS = [
    np.eye(2),
    np.array([[0., -1.], [1., 0.]]),
    np.array([[-1., 0.], [0., -1.]]),
    np.array([[0., 1.], [-1., 0.]]),
]

C2_MATS = [
    np.eye(2),
    np.array([[-1., 0.], [0., -1.]]),
]

# C6 rotation matrices: R(n*60°) in Cartesian, n=0,...,5
_c60 = np.cos(np.pi / 3)   # 1/2
_s60 = np.sin(np.pi / 3)   # √3/2
C6_MATS = [np.array([[np.cos(n * np.pi / 3), -np.sin(n * np.pi / 3)],
                      [np.sin(n * np.pi / 3),  np.cos(n * np.pi / 3)]])
           for n in range(6)]

# C3 rotation matrices: R(n*120°) in Cartesian, n=0,1,2
C3_MATS = [np.array([[np.cos(n * 2 * np.pi / 3), -np.sin(n * 2 * np.pi / 3)],
                      [np.sin(n * 2 * np.pi / 3),  np.cos(n * 2 * np.pi / 3)]])
           for n in range(3)]


def get_rot_mats(sym_type):
    if sym_type == "C4":
        return C4_MATS
    elif sym_type == "C2":
        return C2_MATS
    elif sym_type == "C6":
        return C6_MATS
    elif sym_type == "C3":
        return C3_MATS
    else:
        raise ValueError(f"Unknown sym_type: {sym_type}")


def get_order(sym_type):
    orders = {"C2": 2, "C3": 3, "C4": 4, "C6": 6}
    if sym_type not in orders:
        raise ValueError(f"Unknown sym_type: {sym_type}")
    return orders[sym_type]


# ===========================================================================
# Symmetrization functions
# ===========================================================================


def symmetrize_scalar(Q, Ns, sym_type):
    """Symmetrize scalar field Q(R) with shape (Ns, Ns, ...)."""
    order = get_order(sym_type)
    result = np.zeros_like(Q)
    for n_rot in range(order):
        result += apply_rot_n_grid(Q, n_rot, Ns, sym_type)
    return result / order


def symmetrize_vector(Q, Ns, sym_type):
    """Symmetrize vector field Q(R) with shape (Ns, Ns, ..., 2)."""
    order = get_order(sym_type)
    rot_mats = get_rot_mats(sym_type)
    result = np.zeros_like(Q)
    for n_rot in range(order):
        Q_pulled = apply_rot_n_grid(Q, n_rot, Ns, sym_type)
        R_mat = rot_mats[n_rot]
        for i in range(2):
            for j in range(2):
                result[..., i] += R_mat[i, j] * Q_pulled[..., j]
    return result / order


def symmetrize_2tensor(Q, Ns, sym_type):
    """Symmetrize 2-tensor field Q(R) with shape (Ns, Ns, ..., 2, 2)."""
    order = get_order(sym_type)
    rot_mats = get_rot_mats(sym_type)
    result = np.zeros_like(Q)
    for n_rot in range(order):
        Q_pulled = apply_rot_n_grid(Q, n_rot, Ns, sym_type)
        R_mat = rot_mats[n_rot]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        result[..., i, j] += R_mat[i, k] * R_mat[j, l] * Q_pulled[..., k, l]
    return result / order


# ===========================================================================
# Error measurement
# ===========================================================================


def measure_error_scalar(Q, Ns, sym_type):
    """Measure symmetry error for scalar field."""
    Q_rot = apply_rot_n_grid(Q, 1, Ns, sym_type)
    return np.linalg.norm(Q_rot - Q) / (np.linalg.norm(Q) + 1e-30)


def measure_error_vector(Q, Ns, sym_type):
    """Measure symmetry error for vector field."""
    rot_mats = get_rot_mats(sym_type)
    Q_pulled = apply_rot_n_grid(Q, 1, Ns, sym_type)
    R_mat = rot_mats[1]  # First non-trivial rotation
    Q_expected = np.zeros_like(Q)
    for i in range(2):
        for j in range(2):
            Q_expected[..., i] += R_mat[i, j] * Q_pulled[..., j]
    return np.linalg.norm(Q_expected - Q) / (np.linalg.norm(Q) + 1e-30)


def measure_error_2tensor(Q, Ns, sym_type):
    """Measure symmetry error for 2-tensor field."""
    rot_mats = get_rot_mats(sym_type)
    Q_pulled = apply_rot_n_grid(Q, 1, Ns, sym_type)
    R_mat = rot_mats[1]
    Q_expected = np.zeros_like(Q)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    Q_expected[..., i, j] += R_mat[i, k] * R_mat[j, l] * Q_pulled[..., k, l]
    return np.linalg.norm(Q_expected - Q) / (np.linalg.norm(Q) + 1e-30)


# ===========================================================================
# Main symmetrization
# ===========================================================================


def symmetrize_phase2(cand_dir: Path, sym_type: str):
    """
    Symmetrize Phase 2 data in-place.

    Args:
        cand_dir: Path to candidate directory (e.g., runsV3/thesis_.../candidate_0000)
        sym_type: 'C4' or 'C2'
    """
    phase2_h5 = cand_dir / "phase2_multiband_data.h5"
    if not phase2_h5.exists():
        raise FileNotFoundError(f"Phase 2 data not found: {phase2_h5}")

    print(f"{'='*70}")
    print(f"  {sym_type}-SYMMETRIZE Phase 2 data")
    print(f"  Candidate: {cand_dir}")
    print(f"{'='*70}")

    # Load
    print("\n[1] Loading Phase 2 data...")
    with h5py.File(phase2_h5, 'r') as hf:
        Lambda = hf['Lambda'][:]
        A_berry = hf['A_berry'][:]
        Phi_BH = hf['Phi_BH'][:]
        v_drift = hf['v_drift'][:]
        M_inv = hf['M_inv'][:]
        omega = hf['omega'][:]
        Ns1 = int(hf.attrs['Ns1'])
        Ns2 = int(hf.attrs['Ns2'])
        Nb = int(hf.attrs['N_subspace'])

    assert Ns1 == Ns2, f"Require square grid, got {Ns1}×{Ns2}"
    Ns = Ns1
    print(f"  Grid: {Ns}×{Ns}, N_bands={Nb}")

    # Measure errors before
    print(f"\n[2] {sym_type} errors BEFORE symmetrization:")
    err_L = measure_error_scalar(Lambda, Ns, sym_type)
    err_A = measure_error_vector(A_berry, Ns, sym_type)
    err_M = measure_error_2tensor(M_inv, Ns, sym_type)
    err_Phi = measure_error_scalar(Phi_BH, Ns, sym_type)
    err_v = measure_error_vector(v_drift, Ns, sym_type)
    print(f"  Λ:     {err_L:.6e}")
    print(f"  A:     {err_A:.6e}")
    print(f"  M⁻¹:  {err_M:.6e}")
    print(f"  Φ_BH:  {err_Phi:.6e}")
    print(f"  v_dr:  {err_v:.6e}")

    # Symmetrize
    print(f"\n[3] Applying {sym_type} symmetrization...")
    Lambda_sym = symmetrize_scalar(Lambda, Ns, sym_type)
    omega_sym = symmetrize_scalar(omega, Ns, sym_type)
    Phi_BH_sym = symmetrize_scalar(Phi_BH, Ns, sym_type)
    A_berry_sym = symmetrize_vector(A_berry, Ns, sym_type)
    v_drift_sym = symmetrize_vector(v_drift, Ns, sym_type)
    M_inv_sym = symmetrize_2tensor(M_inv, Ns, sym_type)
    print("  Done.")

    # Measure errors after
    print(f"\n[4] {sym_type} errors AFTER symmetrization:")
    err_L2 = measure_error_scalar(Lambda_sym, Ns, sym_type)
    err_A2 = measure_error_vector(A_berry_sym, Ns, sym_type)
    err_M2 = measure_error_2tensor(M_inv_sym, Ns, sym_type)
    err_Phi2 = measure_error_scalar(Phi_BH_sym, Ns, sym_type)
    err_v2 = measure_error_vector(v_drift_sym, Ns, sym_type)
    print(f"  Λ:     {err_L2:.6e}  (was {err_L:.2e})")
    print(f"  A:     {err_A2:.6e}  (was {err_A:.2e})")
    print(f"  M⁻¹:  {err_M2:.6e}  (was {err_M:.2e})")
    print(f"  Φ_BH:  {err_Phi2:.6e}  (was {err_Phi:.2e})")
    print(f"  v_dr:  {err_v2:.6e}  (was {err_v:.2e})")

    # Save
    out_h5 = cand_dir / f"phase2_multiband_data_{sym_type.lower()}sym.h5"
    print(f"\n[5] Saving to {out_h5}...")
    with h5py.File(out_h5, 'w') as hf:
        hf.create_dataset('Lambda', data=Lambda_sym)
        hf.create_dataset('A_berry', data=A_berry_sym)
        hf.create_dataset('Phi_BH', data=Phi_BH_sym)
        hf.create_dataset('v_drift', data=v_drift_sym)
        hf.create_dataset('M_inv', data=M_inv_sym)
        hf.create_dataset('omega', data=omega_sym)

        # Copy metadata from original
        with h5py.File(phase2_h5, 'r') as hf_orig:
            for key, val in hf_orig.attrs.items():
                hf.attrs[key] = val
            for key in hf_orig.keys():
                if key not in hf.keys():
                    hf.create_dataset(key, data=hf_orig[key][:])

        hf.attrs[f'{sym_type.lower()}_symmetrized'] = True

    sz_mb = os.path.getsize(out_h5) / 1e6
    print(f"  Saved ({sz_mb:.1f} MB)")

    # Also update the original phase2 file to point to symmetrized version
    # (Phase 3 will look for phase2_multiband_data.h5 by default)
    backup_h5 = cand_dir / "phase2_multiband_data_unsym.h5"
    if not backup_h5.exists():
        print(f"\n[6] Backing up original → {backup_h5.name}")
        import shutil
        shutil.copy2(phase2_h5, backup_h5)
        print(f"  Replacing phase2_multiband_data.h5 with symmetrized version")
        shutil.copy2(out_h5, phase2_h5)
    else:
        print(f"\n[6] Backup already exists, replacing phase2_multiband_data.h5")
        import shutil
        shutil.copy2(out_h5, phase2_h5)

    print(f"\n{'='*70}")
    print(f"  {sym_type}-SYMMETRIZATION COMPLETE")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Symmetrize Phase 2 data")
    parser.add_argument("candidate_dir", type=str,
                        help="Path to candidate directory")
    parser.add_argument("--sym", choices=["C2", "C3", "C4", "C6"], required=True,
                        help="Symmetry type: C4 (square), C2 (hex M), C3 (hex K), C6 (honeycomb K)")
    args = parser.parse_args()

    cand_dir = Path(args.candidate_dir)
    if not cand_dir.exists():
        raise FileNotFoundError(f"Candidate directory not found: {cand_dir}")

    symmetrize_phase2(cand_dir, args.sym)


if __name__ == "__main__":
    main()
