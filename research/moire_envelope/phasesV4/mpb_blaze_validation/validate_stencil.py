"""
Blaze2D vs MPB Validation — Step 2: K-stencil, Group Velocity & Mass Tensor

For each canonical configuration, we pick a high-symmetry k-point and compare:
  1. K-stencil eigenvalue landscapes
  2. Group velocity (dω/dk diagonal) via finite differences
  3. Inverse mass tensor (d²ω/dk²) via finite differences
  4. Blaze2D analytic velocity and mass tensor from operator derivatives

This validates that the Blaze2D operator-derivative matrices (v, w, M^{-1})
agree with finite-difference estimates from eigenvalue stencils, and that both
agree with MPB finite-difference results.
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
N_REMOTE = 16
N_STENCIL = 5       # 5×5 stencil
DELTA_K = 0.1        # stencil extent in Cartesian 1/a
MPB_DK = 0.02        # MPB finite-difference step (1/a)


# ============================================================================
# Configuration definitions (same lattices as Step 1)
# ============================================================================

def sq_lattice_vectors():
    return [[1.0, 0.0], [0.0, 1.0]]

def hex_lattice_vectors():
    return [[1.0, 0.0], [0.5, math.sqrt(3)/2]]


CONFIGS = {
    "square_rods": {
        "label": "Square, ε=8.9 rods, r/a=0.2",
        "lattice_vectors": sq_lattice_vectors(),
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 8.9}],
        "eps_bg": 1.0,
        # k0 to probe: M point (fractional: 0.5, 0.5)
        "k0_frac": [0.5, 0.5],
        "k0_label": "M",
        "mpb_lattice_basis": None,
        "mpb_cylinders": [{"radius": 0.2, "eps": 8.9, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 1.0,
    },
    "hex_holes": {
        "label": "Hex, ε=13 bg, air holes, r/a=0.48",
        "lattice_vectors": hex_lattice_vectors(),
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.48, "eps_inside": 1.0}],
        "eps_bg": 13.0,
        # k0 to probe: K point (fractional: 1/3, 1/3)
        "k0_frac": [1/3, 1/3],
        "k0_label": "K",
        "mpb_lattice_basis": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "mpb_cylinders": [{"radius": 0.48, "eps": 1.0, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 13.0,
    },
    "honeycomb_holes": {
        "label": "Honeycomb, ε=12 bg, air holes, r/a=0.25",
        "lattice_vectors": hex_lattice_vectors(),
        "atoms": [
            {"pos": [0.0, 0.0], "radius": 0.25, "eps_inside": 1.0},
            {"pos": [1/3, 1/3], "radius": 0.25, "eps_inside": 1.0},
        ],
        "eps_bg": 12.0,
        # k0 to probe: K point (fractional: 1/3, 1/3)
        "k0_frac": [1/3, 1/3],
        "k0_label": "K",
        "mpb_lattice_basis": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "mpb_cylinders": [
            {"radius": 0.25, "eps": 1.0, "center": [0.0, 0.0]},
            {"radius": 0.25, "eps": 1.0, "center": [1/3, 1/3]},
        ],
        "mpb_eps_bg": 12.0,
    },
}


# ============================================================================
# Coordinate conversions
# ============================================================================

def frac_to_cartesian_k(k_frac, lattice_vectors):
    """Convert k from MPB fractional reciprocal coords to Blaze Cartesian k (1/a)."""
    A = np.array(lattice_vectors)
    A_inv_T = np.linalg.inv(A).T
    return 2 * np.pi * A_inv_T @ np.array(k_frac)


def cartesian_to_frac_k(k_cart, lattice_vectors):
    """Convert Blaze Cartesian k (1/a) back to MPB fractional reciprocal coords."""
    A = np.array(lattice_vectors)
    return A.T @ np.array(k_cart) / (2 * np.pi)


# ============================================================================
# MPB stencil computation
# ============================================================================

def compute_mpb_stencil(config, polarization, k0_frac, dk, n_stencil, num_bands):
    """
    Compute eigenvalues on a Cartesian k-grid around k0.
    dk and grid are in Cartesian reciprocal space; we convert to MPB fractional.
    """
    import meep as mp
    from meep import mpb

    basis = config.get("mpb_lattice_basis")
    if basis is None:
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    else:
        lattice = mp.Lattice(
            size=mp.Vector3(1, 1, 0),
            basis1=mp.Vector3(basis[0][0], basis[0][1], 0),
            basis2=mp.Vector3(basis[1][0], basis[1][1], 0),
        )
    geometry = []
    for cyl in config["mpb_cylinders"]:
        geometry.append(mp.Cylinder(
            radius=cyl["radius"],
            material=mp.Medium(epsilon=cyl["eps"]),
            center=mp.Vector3(cyl["center"][0], cyl["center"][1], 0),
        ))

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=config["mpb_eps_bg"]),
        num_bands=num_bands,
        resolution=RESOLUTION,
    )

    lattice_vectors = config["lattice_vectors"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)

    half = (n_stencil - 1) // 2
    spacing = dk / half if half > 0 else dk

    freqs_grid = np.zeros((n_stencil, n_stencil, num_bands))
    k_cart_grid = np.zeros((n_stencil, n_stencil, 2))

    for ix in range(n_stencil):
        for iy in range(n_stencil):
            dkx = (ix - half) * spacing
            dky = (iy - half) * spacing
            k_cart = k0_cart + np.array([dkx, dky])
            k_frac = cartesian_to_frac_k(k_cart, lattice_vectors)

            ms.k_points = [mp.Vector3(k_frac[0], k_frac[1], 0)]

            mp.verbosity(0)
            devnull = os.open(os.devnull, os.O_WRONLY)
            old1, old2 = os.dup(1), os.dup(2)
            try:
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                if polarization == "TE":
                    ms.run_te()
                else:
                    ms.run_tm()
            finally:
                os.dup2(old1, 1)
                os.dup2(old2, 2)
                os.close(devnull)
                os.close(old1)
                os.close(old2)

            freqs_grid[ix, iy, :] = np.array(ms.all_freqs[0])
            k_cart_grid[ix, iy, :] = k_cart

    return freqs_grid, k_cart_grid, spacing


def fd_derivatives_from_stencil(freqs_grid, spacing, center_idx):
    """
    Compute group velocity and mass tensor from a stencil grid via
    central finite differences (2nd order).
    
    freqs_grid: (n_stencil, n_stencil, n_bands)
    Returns vg (n_bands, 2), M_inv (n_bands, 2, 2) in freq-units/(k-unit).
    """
    c = center_idx
    n_bands = freqs_grid.shape[2]

    vg = np.zeros((n_bands, 2))
    M_inv = np.zeros((n_bands, 2, 2))

    # dω/dk_x via central 2nd-order FD
    vg[:, 0] = (freqs_grid[c + 1, c, :] - freqs_grid[c - 1, c, :]) / (2 * spacing)
    vg[:, 1] = (freqs_grid[c, c + 1, :] - freqs_grid[c, c - 1, :]) / (2 * spacing)

    # d²ω/dk²
    M_inv[:, 0, 0] = (freqs_grid[c + 1, c, :] - 2 * freqs_grid[c, c, :] +
                       freqs_grid[c - 1, c, :]) / (spacing**2)
    M_inv[:, 1, 1] = (freqs_grid[c, c + 1, :] - 2 * freqs_grid[c, c, :] +
                       freqs_grid[c, c - 1, :]) / (spacing**2)
    M_inv[:, 0, 1] = (freqs_grid[c + 1, c + 1, :] - freqs_grid[c + 1, c - 1, :] -
                       freqs_grid[c - 1, c + 1, :] + freqs_grid[c - 1, c - 1, :]) / (4 * spacing**2)
    M_inv[:, 1, 0] = M_inv[:, 0, 1]

    return vg, M_inv


# ============================================================================
# Blaze stencil computation
# ============================================================================

def compute_blaze_stencil(config, polarization, k0_frac):
    """Use Blaze EAExtractor.extract_k_stencil for the stencil."""
    lattice_vectors = config["lattice_vectors"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)

    stencil = EAExtractor.extract_k_stencil(
        lattice_vectors=lattice_vectors,
        atoms=config["atoms"],
        eps_bg=config["eps_bg"],
        k0=list(k0_cart),
        polarization=polarization,
        resolution=RESOLUTION,
        n_stencil=N_STENCIL,
        delta_k=DELTA_K,
        n_retained=N_RETAINED,
        n_remote=N_REMOTE,
    )

    return stencil


def extract_blaze_analytic(stencil):
    """
    Extract analytic velocity and mass tensor from Blaze stencil center.
    
    Blaze returns everything in λ = (ω/c)² space, (2π/a) units.
    Velocity: dλ/dk (diagonal of v_x, v_y)
    Mass tensor: M^{-1}_{ij} in eigenvalue space (d²λ/dk²)
    
    To compare with MPB (which returns freq = ωa/(2πc)):
    freq = sqrt(λ) / (2π)
    dfreq/dk = (dλ/dk) / (2 * 2π * sqrt(λ)) = v_λ / (4π sqrt(λ))
    d²freq/dk² = [d²λ/dk² * 2sqrt(λ) - (dλ/dk)² / sqrt(λ)] / (4 * (2π)² * λ)
              ... this gets complicated. Better to compare in λ-space.
    
    Actually, let's just convert MPB freqs to λ-space:
    λ = (2π freq)² in (2π/a)² units
    Then all comparisons are in eigenvalue (λ) space.
    """
    center = stencil["center"]
    n_ret = center["n_retained"]
    n_tot = n_ret + center["n_remote"]

    # Eigenvalues
    eigenvalues = np.array(center["eigenvalues"][:n_ret])

    # Velocity matrices (n_ret × n_tot)
    def to_matrix(data, rows, cols):
        arr = np.array([(re + 1j * im) for re, im in data])
        return arr.reshape(rows, cols)

    v_x = to_matrix(center["velocity_matrices_x"], n_ret, n_tot)
    v_y = to_matrix(center["velocity_matrices_y"], n_ret, n_tot)

    # Diagonal velocity = group velocity in λ-space
    vg_lambda = np.zeros((n_ret, 2))
    vg_lambda[:, 0] = np.diag(v_x[:, :n_ret]).real
    vg_lambda[:, 1] = np.diag(v_y[:, :n_ret]).real

    # Inverse mass tensor (n_ret × n_ret)
    M_inv_xx = to_matrix(center["mass_tensor_inv_xx"], n_ret, n_ret)
    M_inv_xy = to_matrix(center["mass_tensor_inv_xy"], n_ret, n_ret)
    M_inv_yx = to_matrix(center["mass_tensor_inv_yx"], n_ret, n_ret)
    M_inv_yy = to_matrix(center["mass_tensor_inv_yy"], n_ret, n_ret)

    # Diagonal elements of mass tensor
    mass_diag = np.zeros((n_ret, 2, 2))
    mass_diag[:, 0, 0] = np.diag(M_inv_xx).real
    mass_diag[:, 0, 1] = np.diag(M_inv_xy).real
    mass_diag[:, 1, 0] = np.diag(M_inv_yx).real
    mass_diag[:, 1, 1] = np.diag(M_inv_yy).real

    return {
        "eigenvalues": eigenvalues,
        "vg_lambda": vg_lambda,
        "mass_diag_lambda": mass_diag,
        "v_x_full": v_x,
        "v_y_full": v_y,
        "M_inv_xx": M_inv_xx,
        "M_inv_xy": M_inv_xy,
        "M_inv_yx": M_inv_yx,
        "M_inv_yy": M_inv_yy,
    }


def blaze_stencil_fd(stencil, n_ret):
    """
    Compute FD derivatives from Blaze stencil eigenvalues (in λ-space).
    This gives us dλ/dk and d²λ/dk² from the stencil neighbors.
    """
    center = stencil["center"]
    delta_k = stencil["delta_k"]
    n_stencil = stencil["n_stencil"]
    half = (n_stencil - 1) // 2
    spacing = delta_k / half if half > 0 else delta_k

    k0 = np.array(center["k0"])
    lam_center = np.array(center["eigenvalues"][:n_ret])

    # Build eigenvalue grid
    lam_grid = np.full((n_stencil, n_stencil, n_ret), np.nan)
    lam_grid[half, half, :] = lam_center

    for idx, kp in enumerate(stencil["neighbor_k_points"]):
        kp = np.array(kp)
        dk = kp - k0
        # Determine grid indices
        ix = round(dk[0] / spacing) + half
        iy = round(dk[1] / spacing) + half
        if 0 <= ix < n_stencil and 0 <= iy < n_stencil:
            nbr = stencil["neighbors"][idx]
            lam_grid[int(ix), int(iy), :] = np.array(nbr["eigenvalues"][:n_ret])

    # FD derivatives in λ-space
    vg_fd = np.zeros((n_ret, 2))
    M_inv_fd = np.zeros((n_ret, 2, 2))
    c = half

    vg_fd[:, 0] = (lam_grid[c + 1, c, :] - lam_grid[c - 1, c, :]) / (2 * spacing)
    vg_fd[:, 1] = (lam_grid[c, c + 1, :] - lam_grid[c, c - 1, :]) / (2 * spacing)

    M_inv_fd[:, 0, 0] = (lam_grid[c + 1, c, :] - 2 * lam_grid[c, c, :] +
                          lam_grid[c - 1, c, :]) / spacing**2
    M_inv_fd[:, 1, 1] = (lam_grid[c, c + 1, :] - 2 * lam_grid[c, c, :] +
                          lam_grid[c, c - 1, :]) / spacing**2
    M_inv_fd[:, 0, 1] = (lam_grid[c + 1, c + 1, :] - lam_grid[c + 1, c - 1, :] -
                          lam_grid[c - 1, c + 1, :] + lam_grid[c - 1, c - 1, :]) / (4 * spacing**2)
    M_inv_fd[:, 1, 0] = M_inv_fd[:, 0, 1]

    return vg_fd, M_inv_fd, lam_grid


# ============================================================================
# Plotting
# ============================================================================

def plot_stencil_comparison(all_results):
    """Plot velocity and mass tensor comparison."""
    n_configs = len(all_results)
    fig, axes = plt.subplots(2, n_configs, figsize=(6 * n_configs, 10))
    if n_configs == 1:
        axes = axes[:, np.newaxis]

    config_names = list(all_results.keys())
    pol = "TE"  # focus on TE

    for col, cname in enumerate(config_names):
        res = all_results[cname][pol]

        # -- Group velocity comparison --
        ax = axes[0, col]
        n_bands = min(N_RETAINED, res["mpb_vg_lambda"].shape[0])
        band_idx = np.arange(n_bands)
        width = 0.2

        ax.bar(band_idx - width, res["mpb_vg_lambda"][:n_bands, 0],
               width, label="MPB FD vx", color='blue', alpha=0.7)
        ax.bar(band_idx, res["blaze_fd_vg"][:n_bands, 0],
               width, label="Blaze FD vx", color='red', alpha=0.7)
        ax.bar(band_idx + width, res["blaze_analytic_vg"][:n_bands, 0],
               width, label="Blaze analytic vx", color='green', alpha=0.7)

        ax.set_xlabel("Band index")
        ax.set_ylabel("dλ/dk_x")
        ax.set_title(f"{CONFIGS[cname]['label']}\nGroup velocity (x)")
        ax.legend(fontsize=7)
        ax.set_xticks(band_idx)

        # -- Mass tensor comparison --
        ax = axes[1, col]
        ax.bar(band_idx - width, res["mpb_mass_lambda"][:n_bands, 0, 0],
               width, label="MPB FD M⁻¹_xx", color='blue', alpha=0.7)
        ax.bar(band_idx, res["blaze_fd_mass"][:n_bands, 0, 0],
               width, label="Blaze FD M⁻¹_xx", color='red', alpha=0.7)
        ax.bar(band_idx + width, res["blaze_analytic_mass"][:n_bands, 0, 0],
               width, label="Blaze analytic M⁻¹_xx", color='green', alpha=0.7)

        ax.set_xlabel("Band index")
        ax.set_ylabel("d²λ/dk_x²")
        ax.set_title("Mass tensor (xx)")
        ax.legend(fontsize=7)
        ax.set_xticks(band_idx)

    fig.suptitle(f"K-stencil: Group Velocity & Mass Tensor (TE, res={RESOLUTION})", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stencil_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'stencil_comparison.png'}")


def print_stencil_comparison(all_results):
    """Printed numerical comparison of stencil-derived quantities."""
    print("\n" + "=" * 80)
    print("K-STENCIL COMPARISON: Group Velocity & Mass Tensor")
    print("=" * 80)

    for cname, res_pols in all_results.items():
        print(f"\n--- {CONFIGS[cname]['label']} ---")
        for pol in ["TE", "TM"]:
            if pol not in res_pols:
                continue
            res = res_pols[pol]
            n_bands = min(N_RETAINED, res["mpb_vg_lambda"].shape[0])

            print(f"\n  {pol} polarization at {CONFIGS[cname]['k0_label']}:")

            # Group velocity comparison
            print(f"\n  Group velocity dλ/dk (retained bands):")
            print(f"  {'Band':>5} {'MPB_FD_vx':>12} {'Blz_FD_vx':>12} {'Blz_ana_vx':>12} "
                  f"{'MPB_FD_vy':>12} {'Blz_FD_vy':>12} {'Blz_ana_vy':>12}")
            for b in range(n_bands):
                mvx = res["mpb_vg_lambda"][b, 0]
                bfvx = res["blaze_fd_vg"][b, 0]
                bavx = res["blaze_analytic_vg"][b, 0]
                mvy = res["mpb_vg_lambda"][b, 1]
                bfvy = res["blaze_fd_vg"][b, 1]
                bavy = res["blaze_analytic_vg"][b, 1]
                print(f"  {b+1:5d} {mvx:12.6f} {bfvx:12.6f} {bavx:12.6f} "
                      f"{mvy:12.6f} {bfvy:12.6f} {bavy:12.6f}")

            # Mass tensor comparison
            print(f"\n  Mass tensor d²λ/dk² (diagonal, retained bands):")
            print(f"  {'Band':>5} {'MPB_xx':>12} {'Blz_FD_xx':>12} {'Blz_ana_xx':>12} "
                  f"{'MPB_yy':>12} {'Blz_FD_yy':>12} {'Blz_ana_yy':>12}")
            for b in range(n_bands):
                mxx = res["mpb_mass_lambda"][b, 0, 0]
                bfxx = res["blaze_fd_mass"][b, 0, 0]
                baxx = res["blaze_analytic_mass"][b, 0, 0]
                myy = res["mpb_mass_lambda"][b, 1, 1]
                bfyy = res["blaze_fd_mass"][b, 1, 1]
                bayy = res["blaze_analytic_mass"][b, 1, 1]
                print(f"  {b+1:5d} {mxx:12.4f} {bfxx:12.4f} {baxx:12.4f} "
                      f"{myy:12.4f} {bfyy:12.4f} {bayy:12.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Blaze2D vs MPB — K-stencil / Velocity / Mass Tensor Validation")
    print(f"Resolution: {RESOLUTION}, n_retained: {N_RETAINED}, n_remote: {N_REMOTE}")
    print(f"Blaze stencil: {N_STENCIL}×{N_STENCIL}, delta_k={DELTA_K}")
    print("=" * 80)

    all_results = {}
    num_bands_mpb = N_RETAINED + N_REMOTE

    # -- Phase 1: Compute ALL Blaze stencils BEFORE importing meep --
    # (Importing meep corrupts Blaze's TM generalized eigensolver)
    print("\n--- Phase 1: Computing all Blaze stencils (before meep import) ---")
    blaze_cache = {}
    for cname, config in CONFIGS.items():
        blaze_cache[cname] = {}
        k0_frac = config["k0_frac"]
        for pol in ["TE", "TM"]:
            print(f"  [{cname}/{pol}] Blaze stencil... ", end="", flush=True)
            t0 = time.time()
            blaze_stencil = compute_blaze_stencil(config, pol, k0_frac)
            t_blaze = time.time() - t0
            print(f"done ({t_blaze:.1f}s)")

            blaze_analytic = extract_blaze_analytic(blaze_stencil)
            blaze_fd_vg, blaze_fd_mass, blaze_lam_grid = blaze_stencil_fd(
                blaze_stencil, N_RETAINED
            )
            blaze_cache[cname][pol] = {
                "blaze_fd_vg": blaze_fd_vg,
                "blaze_fd_mass": blaze_fd_mass,
                "blaze_analytic_vg": blaze_analytic["vg_lambda"],
                "blaze_analytic_mass": blaze_analytic["mass_diag_lambda"],
                "blaze_eigenvalues": blaze_analytic["eigenvalues"],
            }

    # -- Phase 2: Compute ALL MPB stencils --
    print("\n--- Phase 2: Computing all MPB stencils ---")
    for cname, config in CONFIGS.items():
        all_results[cname] = {}
        k0_frac = config["k0_frac"]
        lattice_vectors = config["lattice_vectors"]

        for pol in ["TE", "TM"]:
            print(f"  [{cname}/{pol}] MPB stencil ({N_STENCIL}×{N_STENCIL})... ",
                  end="", flush=True)
            t0 = time.time()
            mpb_freqs, mpb_kcart, mpb_spacing = compute_mpb_stencil(
                config, pol, k0_frac, DELTA_K, N_STENCIL, num_bands_mpb
            )
            t_mpb = time.time() - t0
            print(f"done ({t_mpb:.1f}s)")

            mpb_lambda = (2 * np.pi * mpb_freqs) ** 2
            center_idx = (N_STENCIL - 1) // 2
            mpb_vg_lambda, mpb_mass_lambda = fd_derivatives_from_stencil(
                mpb_lambda, mpb_spacing, center_idx
            )

            bc = blaze_cache[cname][pol]
            all_results[cname][pol] = {
                "mpb_vg_lambda": mpb_vg_lambda[:N_RETAINED],
                "mpb_mass_lambda": mpb_mass_lambda[:N_RETAINED],
                "mpb_eigenvalues": mpb_lambda[center_idx, center_idx, :N_RETAINED],
                **bc,
            }

    print_stencil_comparison(all_results)

    print("\nGenerating plots...")
    plot_stencil_comparison(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
