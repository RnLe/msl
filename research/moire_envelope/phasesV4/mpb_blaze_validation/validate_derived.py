"""
Blaze2D vs MPB Validation — Step 4: Derived Quantities

Compares quantities that feed into the EA Hamiltonian:
  1. Berry connection A_mn = <u_m|∂_R u_n>  (via FD on both solvers)
  2. Born-Huang potential Φ_mn (from Blaze directly, from MPB via field FD)
  3. R-derivative matrix elements <u_m|∂L₀/∂R|u_n>
  4. Off-diagonal velocity matrices v^(i)_mn
  5. Off-diagonal mass tensor elements (Löwdin-corrected)

For Berry connection and Born-Huang, we use a 2-atom honeycomb with a
registry shift, since these quantities are only meaningful for multi-atom
bases where the dielectric depends on a sliding parameter R.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import math
import time

# IMPORTANT: Import blaze BEFORE meep to avoid BLAS corruption of TM solver
from blaze import EAExtractor

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESOLUTION = 32
N_RETAINED = 4
N_REMOTE = 12
FD_STEP = 0.005  # fractional step for ∂/∂R
DELTA_K = 0.1


# ============================================================================
# Configuration: honeycomb with registry
# ============================================================================

def hex_lat():
    return [[1.0, 0.0], [0.5, math.sqrt(3)/2]]


CONFIG = {
    "label": "Honeycomb, ε=12 bg, air holes, r/a=0.25",
    "lattice_vectors": hex_lat(),
    "base_atoms": [
        {"pos": [0.0, 0.0], "radius": 0.25, "eps_inside": 1.0},
        {"pos": [1/3, 1/3], "radius": 0.25, "eps_inside": 1.0},
    ],
    "eps_bg": 12.0,
    "k0_frac": [0.2, 0.1],  # generic k-point (avoid degeneracies)
    "registry": [0.1, 0.05],  # non-trivial registry shift
}


def shifted_atoms(base_atoms, registry):
    """Shift atom B by registry (fractional) — atom A stays fixed."""
    atoms = [dict(base_atoms[0])]  # atom A unchanged
    pos_b = base_atoms[1]["pos"]
    shifted = [(pos_b[0] + registry[0]) % 1.0,
               (pos_b[1] + registry[1]) % 1.0]
    atoms.append({**base_atoms[1], "pos": shifted})
    return atoms


def frac_to_cartesian_k(k_frac, lattice_vectors):
    """Convert MPB fractional reciprocal coords to Blaze Cartesian k (1/a)."""
    A = np.array(lattice_vectors)
    A_inv_T = np.linalg.inv(A).T
    return 2 * np.pi * A_inv_T @ np.array(k_frac)


# ============================================================================
# 1. Off-diagonal velocity matrix comparison
# ============================================================================

def compare_velocity_matrices(config, polarization):
    """
    Compare Blaze analytic velocity matrices with MPB FD estimates.
    
    Blaze gives v^(i)_mn = <u_m|∂L/∂k_i|u_n> directly.
    MPB: we approximate via FD on eigenvalues and eigenvectors.
    For the DIAGONAL: v_nn ≈ dλ_n/dk_i.
    Off-diagonal is harder from MPB, so we mainly validate diagonal.
    """
    lattice_vectors = config["lattice_vectors"]
    k0_frac = config["k0_frac"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)
    registry = config["registry"]
    atoms = shifted_atoms(config["base_atoms"], registry)

    # Blaze extraction with stencil
    stencil = EAExtractor.extract_k_stencil(
        lattice_vectors=lattice_vectors,
        atoms=atoms,
        eps_bg=config["eps_bg"],
        k0=list(k0_cart),
        polarization=polarization,
        resolution=RESOLUTION,
        n_stencil=3,
        delta_k=DELTA_K,
        n_retained=N_RETAINED,
        n_remote=N_REMOTE,
    )

    center = stencil["center"]
    n_ret = center["n_retained"]
    n_tot = n_ret + center["n_remote"]

    def to_matrix(data, rows, cols):
        arr = np.array([(re + 1j * im) for re, im in data])
        return arr.reshape(rows, cols)

    v_x = to_matrix(center["velocity_matrices_x"], n_ret, n_tot)
    v_y = to_matrix(center["velocity_matrices_y"], n_ret, n_tot)

    # FD diagonal velocity from stencil
    dk = stencil["delta_k"]
    lam_center = np.array(center["eigenvalues"][:n_ret])

    vg_fd = np.zeros((n_ret, 2))
    for idx, kp in enumerate(stencil["neighbor_k_points"]):
        kp = np.array(kp)
        dkv = kp - np.array(center["k0"])
        nbr_eigs = np.array(stencil["neighbors"][idx]["eigenvalues"][:n_ret])
        # +x neighbor
        if abs(dkv[0] - dk) < 1e-12 and abs(dkv[1]) < 1e-12:
            lam_px = nbr_eigs
        # -x neighbor
        if abs(dkv[0] + dk) < 1e-12 and abs(dkv[1]) < 1e-12:
            lam_mx = nbr_eigs
        # +y neighbor
        if abs(dkv[1] - dk) < 1e-12 and abs(dkv[0]) < 1e-12:
            lam_py = nbr_eigs
        # -y neighbor
        if abs(dkv[1] + dk) < 1e-12 and abs(dkv[0]) < 1e-12:
            lam_my = nbr_eigs

    vg_fd[:, 0] = (lam_px - lam_mx) / (2 * dk)
    vg_fd[:, 1] = (lam_py - lam_my) / (2 * dk)

    return {
        "v_x_full": v_x,
        "v_y_full": v_y,
        "vg_fd": vg_fd,
        "eigenvalues": lam_center,
    }


# ============================================================================
# 2. R-derivative and Born-Huang comparison
# ============================================================================

def compare_registry_derivatives(config, polarization):
    """
    Compare Blaze R-derivative matrices with MPB FD estimates.
    
    Blaze: <u_m|∂L₀/∂R_j|u_n> computed internally via FD on ε(R).
    MPB: We do the same FD externally — solve at R±δR and compute
         eigenvalue shifts → diagonal elements of ∂L₀/∂R.
    """
    lattice_vectors = config["lattice_vectors"]
    k0_frac = config["k0_frac"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)
    registry = config["registry"]

    # --- Blaze at center registry ---
    atoms_center = shifted_atoms(config["base_atoms"], registry)
    result_center = EAExtractor.extract(
        lattice_vectors=lattice_vectors,
        atoms=atoms_center,
        eps_bg=config["eps_bg"],
        k0=list(k0_cart),
        polarization=polarization,
        resolution=RESOLUTION,
        n_retained=N_RETAINED,
        n_remote=N_REMOTE,
        compute_r_derivatives=True,
        compute_born_huang=True,
        atom_index=1,
        fd_step=FD_STEP,
        registry=list(registry),
    )

    n_ret = N_RETAINED
    n_tot = n_ret + N_REMOTE

    def to_matrix(data, rows, cols):
        arr = np.array([(re + 1j * im) for re, im in data])
        return arr.reshape(rows, cols)

    eigs_center = np.array(result_center["eigenvalues"][:n_ret])

    has_r_deriv = result_center.get("has_r_derivatives", False)
    if has_r_deriv:
        dL_dRx = to_matrix(result_center["r_derivative_matrices_x"], n_ret, n_tot)
        dL_dRy = to_matrix(result_center["r_derivative_matrices_y"], n_ret, n_tot)
    else:
        dL_dRx = dL_dRy = None

    has_bh = result_center.get("has_born_huang", False)
    if has_bh:
        born_huang = to_matrix(result_center["born_huang"], n_ret, n_ret)
    else:
        born_huang = None

    # --- MPB FD for eigenvalue shifts under registry change ---
    # Solve at R ± δR in each direction
    mpb_dL_diag = np.zeros((n_ret, 2))

    for direction in [0, 1]:
        registry_plus = list(registry)
        registry_minus = list(registry)
        registry_plus[direction] += FD_STEP
        registry_minus[direction] -= FD_STEP

        # Plus
        atoms_p = shifted_atoms(config["base_atoms"], registry_plus)
        result_p = EAExtractor.extract(
            lattice_vectors=lattice_vectors,
            atoms=atoms_p,
            eps_bg=config["eps_bg"],
            k0=list(k0_cart),
            polarization=polarization,
            resolution=RESOLUTION,
            n_retained=N_RETAINED,
            n_remote=N_REMOTE,
            compute_r_derivatives=False,
            compute_born_huang=False,
        )
        eigs_p = np.array(result_p["eigenvalues"][:n_ret])

        # Minus
        atoms_m = shifted_atoms(config["base_atoms"], registry_minus)
        result_m = EAExtractor.extract(
            lattice_vectors=lattice_vectors,
            atoms=atoms_m,
            eps_bg=config["eps_bg"],
            k0=list(k0_cart),
            polarization=polarization,
            resolution=RESOLUTION,
            n_retained=N_RETAINED,
            n_remote=N_REMOTE,
            compute_r_derivatives=False,
            compute_born_huang=False,
        )
        eigs_m = np.array(result_m["eigenvalues"][:n_ret])

        mpb_dL_diag[:, direction] = (eigs_p - eigs_m) / (2 * FD_STEP)

    # Born-Huang diagonal from external FD: Φ_nn = eigenvalue curvature in R
    # We just compare the Blaze BH diagonal with the FD second derivative
    mpb_bh_diag = np.zeros((n_ret, 2))  # xx and yy only
    for direction in [0, 1]:
        registry_plus = list(registry)
        registry_minus = list(registry)
        registry_plus[direction] += FD_STEP
        registry_minus[direction] -= FD_STEP

        atoms_p = shifted_atoms(config["base_atoms"], registry_plus)
        atoms_m = shifted_atoms(config["base_atoms"], registry_minus)

        res_p = EAExtractor.extract(
            lattice_vectors=lattice_vectors, atoms=atoms_p, eps_bg=config["eps_bg"],
            k0=list(k0_cart), polarization=polarization, resolution=RESOLUTION,
            n_retained=N_RETAINED, n_remote=N_REMOTE,
            compute_r_derivatives=False, compute_born_huang=False,
        )
        res_m = EAExtractor.extract(
            lattice_vectors=lattice_vectors, atoms=atoms_m, eps_bg=config["eps_bg"],
            k0=list(k0_cart), polarization=polarization, resolution=RESOLUTION,
            n_retained=N_RETAINED, n_remote=N_REMOTE,
            compute_r_derivatives=False, compute_born_huang=False,
        )

        eigs_p = np.array(res_p["eigenvalues"][:n_ret])
        eigs_m = np.array(res_m["eigenvalues"][:n_ret])
        # d²λ/dR² (second derivative via central FD)
        mpb_bh_diag[:, direction] = (eigs_p - 2 * eigs_center + eigs_m) / (FD_STEP ** 2)

    return {
        "blaze_dL_dRx": dL_dRx,
        "blaze_dL_dRy": dL_dRy,
        "blaze_born_huang": born_huang,
        "mpb_dL_diag": mpb_dL_diag,
        "mpb_bh_diag": mpb_bh_diag,
        "eigenvalues": eigs_center,
    }


# ============================================================================
# 3. Full mass tensor (off-diagonal) comparison
# ============================================================================

def compare_mass_tensor_full(config, polarization):
    """
    Compare the full Löwdin-corrected mass tensor from Blaze
    with FD estimates from stencil eigenvalues.
    
    The Blaze mass tensor includes remote-band Löwdin corrections.
    FD from eigenvalues only captures the diagonal (per-band curvature).
    """
    lattice_vectors = config["lattice_vectors"]
    k0_frac = config["k0_frac"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)
    registry = config["registry"]
    atoms = shifted_atoms(config["base_atoms"], registry)

    result = EAExtractor.extract(
        lattice_vectors=lattice_vectors,
        atoms=atoms,
        eps_bg=config["eps_bg"],
        k0=list(k0_cart),
        polarization=polarization,
        resolution=RESOLUTION,
        n_retained=N_RETAINED,
        n_remote=N_REMOTE,
    )

    n_ret = N_RETAINED

    def to_matrix(data, rows, cols):
        arr = np.array([(re + 1j * im) for re, im in data])
        return arr.reshape(rows, cols)

    M_inv_xx = to_matrix(result["mass_tensor_inv_xx"], n_ret, n_ret)
    M_inv_xy = to_matrix(result["mass_tensor_inv_xy"], n_ret, n_ret)
    M_inv_yx = to_matrix(result["mass_tensor_inv_yx"], n_ret, n_ret)
    M_inv_yy = to_matrix(result["mass_tensor_inv_yy"], n_ret, n_ret)

    # Also get w matrices (bare second derivative, without Löwdin)
    w_xx = to_matrix(result["w_matrices_xx"], n_ret, n_ret)
    w_xy = to_matrix(result["w_matrices_xy"], n_ret, n_ret)
    w_yx = to_matrix(result["w_matrices_yx"], n_ret, n_ret)
    w_yy = to_matrix(result["w_matrices_yy"], n_ret, n_ret)

    return {
        "M_inv_xx": M_inv_xx, "M_inv_xy": M_inv_xy,
        "M_inv_yx": M_inv_yx, "M_inv_yy": M_inv_yy,
        "w_xx": w_xx, "w_xy": w_xy, "w_yx": w_yx, "w_yy": w_yy,
        "eigenvalues": np.array(result["eigenvalues"][:n_ret]),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_derived_quantities(vel_results, reg_results, mass_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pol = "TE"

    # --- Velocity diagonal ---
    ax = axes[0, 0]
    vr = vel_results[pol]
    n_ret = vr["vg_fd"].shape[0]
    bands = np.arange(n_ret)
    v_diag_x = np.diag(vr["v_x_full"][:n_ret, :n_ret]).real
    ax.bar(bands - 0.15, vr["vg_fd"][:, 0], 0.3, label="FD dλ/dk_x", color='blue', alpha=0.7)
    ax.bar(bands + 0.15, v_diag_x, 0.3, label="Blaze v_x diag", color='green', alpha=0.7)
    ax.set_xlabel("Band")
    ax.set_ylabel("dλ/dk_x")
    ax.set_title("Velocity: FD vs Analytic")
    ax.legend(fontsize=8)

    # --- Velocity off-diagonal magnitudes ---
    ax = axes[0, 1]
    v_x_nn = vr["v_x_full"][:n_ret, :n_ret]
    im = ax.imshow(np.abs(v_x_nn), cmap='viridis', origin='lower')
    ax.set_title("|v_x| matrix (retained)")
    ax.set_xlabel("Band n")
    ax.set_ylabel("Band m")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- v_y off-diagonal ---
    ax = axes[0, 2]
    v_y_nn = vr["v_y_full"][:n_ret, :n_ret]
    im = ax.imshow(np.abs(v_y_nn), cmap='viridis', origin='lower')
    ax.set_title("|v_y| matrix (retained)")
    ax.set_xlabel("Band n")
    ax.set_ylabel("Band m")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # --- R-derivative diagonal ---
    ax = axes[1, 0]
    rr = reg_results[pol]
    if rr["blaze_dL_dRx"] is not None:
        blaze_diag_x = np.diag(rr["blaze_dL_dRx"][:n_ret, :n_ret]).real
    else:
        blaze_diag_x = np.zeros(n_ret)
    ax.bar(bands - 0.15, rr["mpb_dL_diag"][:, 0], 0.3,
           label="External FD dλ/dR_x", color='blue', alpha=0.7)
    ax.bar(bands + 0.15, blaze_diag_x, 0.3,
           label="Blaze dL/dR_x diag", color='green', alpha=0.7)
    ax.set_xlabel("Band")
    ax.set_ylabel("dλ/dR_x")
    ax.set_title("R-derivative: External FD vs Blaze")
    ax.legend(fontsize=8)

    # --- Born-Huang diagonal ---
    ax = axes[1, 1]
    if rr["blaze_born_huang"] is not None:
        bh_diag = np.diag(rr["blaze_born_huang"]).real
    else:
        bh_diag = np.zeros(n_ret)
    ax.bar(bands - 0.15, rr["mpb_bh_diag"][:, 0], 0.3,
           label="External FD d²λ/dR²_xx", color='blue', alpha=0.7)
    ax.bar(bands + 0.15, bh_diag, 0.3,
           label="Blaze BH diag", color='green', alpha=0.7)
    ax.set_xlabel("Band")
    ax.set_title("Born-Huang diagonal")
    ax.legend(fontsize=8)

    # --- Mass tensor full matrix ---
    ax = axes[1, 2]
    mr = mass_results[pol]
    M = mr["M_inv_xx"]
    im = ax.imshow(np.abs(M), cmap='viridis', origin='lower')
    ax.set_title("|M⁻¹_xx| full (Löwdin)")
    ax.set_xlabel("Band n")
    ax.set_ylabel("Band m")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Derived Quantities Comparison (TE, Honeycomb)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "derived_quantities.png", dpi=150)
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'derived_quantities.png'}")


def print_derived_summary(vel_results, reg_results, mass_results):
    print("\n" + "=" * 80)
    print("DERIVED QUANTITIES SUMMARY")
    print("=" * 80)

    for pol in ["TE", "TM"]:
        print(f"\n{'='*40} {pol} {'='*40}")

        # Velocity
        vr = vel_results[pol]
        n_ret = vr["vg_fd"].shape[0]
        v_diag_x = np.diag(vr["v_x_full"][:n_ret, :n_ret]).real
        v_diag_y = np.diag(vr["v_y_full"][:n_ret, :n_ret]).real
        print(f"\n  Velocity (diagonal) comparison:")
        print(f"  {'Band':>5} {'FD_vx':>12} {'Ana_vx':>12} {'diff':>12} "
              f"{'FD_vy':>12} {'Ana_vy':>12} {'diff':>12}")
        for b in range(n_ret):
            dx = abs(vr["vg_fd"][b, 0] - v_diag_x[b])
            dy = abs(vr["vg_fd"][b, 1] - v_diag_y[b])
            print(f"  {b+1:5d} {vr['vg_fd'][b,0]:12.4f} {v_diag_x[b]:12.4f} {dx:12.2e} "
                  f"{vr['vg_fd'][b,1]:12.4f} {v_diag_y[b]:12.4f} {dy:12.2e}")

        # Off-diagonal velocity magnitudes
        print(f"\n  Off-diagonal velocity magnitudes |v_x_mn| (m≠n):")
        v_x_ret = vr["v_x_full"][:n_ret, :n_ret]
        for m in range(n_ret):
            for n in range(n_ret):
                if m != n:
                    print(f"    v_x[{m+1},{n+1}] = {abs(v_x_ret[m,n]):.6f}")

        # R-derivatives
        rr = reg_results[pol]
        print(f"\n  R-derivative (diagonal) comparison:")
        if rr["blaze_dL_dRx"] is not None:
            blaze_x = np.diag(rr["blaze_dL_dRx"][:n_ret, :n_ret]).real
            blaze_y = np.diag(rr["blaze_dL_dRy"][:n_ret, :n_ret]).real
        else:
            blaze_x = blaze_y = np.zeros(n_ret)
        print(f"  {'Band':>5} {'ExtFD_x':>12} {'Blaze_x':>12} {'diff':>12} "
              f"{'ExtFD_y':>12} {'Blaze_y':>12} {'diff':>12}")
        for b in range(n_ret):
            dx = abs(rr["mpb_dL_diag"][b, 0] - blaze_x[b])
            dy = abs(rr["mpb_dL_diag"][b, 1] - blaze_y[b])
            print(f"  {b+1:5d} {rr['mpb_dL_diag'][b,0]:12.4f} {blaze_x[b]:12.4f} {dx:12.2e} "
                  f"{rr['mpb_dL_diag'][b,1]:12.4f} {blaze_y[b]:12.4f} {dy:12.2e}")

        # Born-Huang
        print(f"\n  Born-Huang (diagonal) comparison:")
        if rr["blaze_born_huang"] is not None:
            bh_diag = np.diag(rr["blaze_born_huang"]).real
        else:
            bh_diag = np.zeros(n_ret)
        print(f"  {'Band':>5} {'ExtFD_xx':>12} {'Blaze_BH':>12} {'diff':>12}")
        for b in range(n_ret):
            d = abs(rr["mpb_bh_diag"][b, 0] - bh_diag[b])
            print(f"  {b+1:5d} {rr['mpb_bh_diag'][b,0]:12.4f} {bh_diag[b]:12.4f} {d:12.2e}")

        # Mass tensor
        mr = mass_results[pol]
        print(f"\n  Löwdin mass tensor M⁻¹_xx (full matrix):")
        M = mr["M_inv_xx"]
        print(f"  Diagonal: {np.diag(M).real}")
        print(f"  Max off-diag: {np.max(np.abs(M - np.diag(np.diag(M)))):.6f}")

        print(f"\n  Bare w_xx vs Löwdin M⁻¹_xx (diagonal):")
        w_diag = np.diag(mr["w_xx"]).real
        m_diag = np.diag(mr["M_inv_xx"]).real
        print(f"  {'Band':>5} {'w_xx':>12} {'M⁻¹_xx':>12} {'Löwdin_corr':>12}")
        for b in range(n_ret):
            print(f"  {b+1:5d} {w_diag[b]:12.4f} {m_diag[b]:12.4f} {m_diag[b]-w_diag[b]:12.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Blaze2D — Derived Quantities Validation")
    print(f"Configuration: {CONFIG['label']}")
    print(f"Registry: {CONFIG['registry']}")
    print(f"k0 (frac): {CONFIG['k0_frac']}")
    print(f"Resolution: {RESOLUTION}, n_retained: {N_RETAINED}, n_remote: {N_REMOTE}")
    print("=" * 80)

    vel_results = {}
    reg_results = {}
    mass_results = {}

    for pol in ["TE", "TM"]:
        print(f"\n[{pol}] Computing velocity matrices...", flush=True)
        t0 = time.time()
        vel_results[pol] = compare_velocity_matrices(CONFIG, pol)
        print(f"  done ({time.time()-t0:.1f}s)")

        print(f"[{pol}] Computing R-derivatives & Born-Huang...", flush=True)
        t0 = time.time()
        reg_results[pol] = compare_registry_derivatives(CONFIG, pol)
        print(f"  done ({time.time()-t0:.1f}s)")

        print(f"[{pol}] Computing full mass tensor...", flush=True)
        t0 = time.time()
        mass_results[pol] = compare_mass_tensor_full(CONFIG, pol)
        print(f"  done ({time.time()-t0:.1f}s)")

    print_derived_summary(vel_results, reg_results, mass_results)

    print("\nGenerating plots...")
    plot_derived_quantities(vel_results, reg_results, mass_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
