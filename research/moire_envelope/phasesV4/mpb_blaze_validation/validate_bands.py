"""
Blaze2D vs MPB Validation — Step 1: Band Diagram Comparison

Computes TE and TM band diagrams for three canonical photonic crystal
configurations using both MPB and Blaze2D, then compares numerically
and produces overlay plots.

Canonical configurations:
  (A) Square lattice, air background (eps=1), eps=8.9 rods, r/a=0.2
  (B) Hexagonal lattice, eps=13 background, air holes (eps=1), r/a=0.48
  (C) Honeycomb lattice, eps=12 background, air holes (eps=1), r/a=0.25

Total bands: 20 (combined TE+TM → 10 each, or all 20 per polarization).
We compute 20 bands for EACH polarization → 20 TE + 20 TM.

Units:
  - MPB returns freq = ω a / (2π c)  (dimensionless)
  - Blaze2D returns eigenvalue λ = (ω/c)² in (2π/a)² units
    → freq = sqrt(λ) / (2π)  in c/a units
  - k-points: MPB uses reciprocal-lattice fractional coords;
    Blaze uses Cartesian reciprocal space in 2π/a units.
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

# IMPORTANT: Blaze TM must be imported and used BEFORE meep.
# Importing meep corrupts Blaze's generalized eigensolver (TM mode)
# due to a shared BLAS/LAPACK library conflict.
from blaze import EAExtractor

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_BANDS = 20
RESOLUTION = 32
N_KPTS_PER_SEGMENT = 15  # k-points per segment of the BZ path


# ============================================================================
# Lattice / geometry definitions
# ============================================================================

CONFIGS = {
    "square_rods": {
        "label": "Square, ε=8.9 rods, r/a=0.2",
        "lattice_type": "square",
        # Blaze: row-vector lattice vectors
        "lattice_vectors": [[1.0, 0.0], [0.0, 1.0]],
        # Blaze atoms: fractional coords, eps_inside = rod epsilon
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 8.9}],
        "eps_bg": 1.0,
        # MPB geometry (built lazily after meep import)
        "mpb_lattice_basis": None,  # None = square
        "mpb_cylinders": [{"radius": 0.2, "eps": 8.9, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 1.0,
        # BZ path in fractional reciprocal coords (MPB convention)
        # Square: Γ → X → M → Γ
        "kpath_labels": ["Γ", "X", "M", "Γ"],
        "kpath_frac": [
            [0.0, 0.0],   # Γ
            [0.5, 0.0],   # X
            [0.5, 0.5],   # M
            [0.0, 0.0],   # Γ
        ],
    },
    "hex_holes": {
        "label": "Hex, ε=13 bg, air holes, r/a=0.48",
        "lattice_type": "hex",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.48, "eps_inside": 1.0}],
        "eps_bg": 13.0,
        "mpb_lattice_basis": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "mpb_cylinders": [{"radius": 0.48, "eps": 1.0, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 13.0,
        # Hex BZ: Γ → M → K → Γ
        "kpath_labels": ["Γ", "M", "K", "Γ"],
        "kpath_frac": [
            [0.0, 0.0],       # Γ
            [0.5, 0.0],       # M
            [1/3, 1/3],       # K
            [0.0, 0.0],       # Γ
        ],
    },
    "honeycomb_holes": {
        "label": "Honeycomb, ε=12 bg, air holes, r/a=0.25",
        "lattice_type": "honeycomb",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "atoms": [
            {"pos": [0.0, 0.0], "radius": 0.25, "eps_inside": 1.0},
            {"pos": [1/3, 1/3], "radius": 0.25, "eps_inside": 1.0},
        ],
        "eps_bg": 12.0,
        "mpb_lattice_basis": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "mpb_cylinders": [
            {"radius": 0.25, "eps": 1.0, "center": [0.0, 0.0]},
            {"radius": 0.25, "eps": 1.0, "center": [1/3, 1/3]},
        ],
        "mpb_eps_bg": 12.0,
        # Honeycomb shares the triangular BZ: Γ → M → K → Γ
        "kpath_labels": ["Γ", "M", "K", "Γ"],
        "kpath_frac": [
            [0.0, 0.0],
            [0.5, 0.0],
            [1/3, 1/3],
            [0.0, 0.0],
        ],
    },
}


# ============================================================================
# k-path utilities
# ============================================================================

def build_kpath_frac(kpath_frac, n_per_seg):
    """
    Build a list of k-points (fractional) along the BZ path segments,
    matching meep's mp.interpolate(n_per_seg, k_points) convention:
    n_per_seg interior points are inserted between each pair, giving
    (n_per_seg + 2) points per segment (including both endpoints).
    """
    kpts = []
    n_sub = n_per_seg + 1  # number of sub-intervals per segment
    for i in range(len(kpath_frac) - 1):
        k_start = np.array(kpath_frac[i])
        k_end = np.array(kpath_frac[i + 1])
        for j in range(n_sub + 1):  # 0 to n_sub inclusive
            if i > 0 and j == 0:
                continue  # skip duplicate boundary point
            t = j / n_sub
            kpts.append(k_start + t * (k_end - k_start))
    return np.array(kpts)


def frac_to_cartesian_k(k_frac, lattice_vectors):
    """
    Convert MPB fractional reciprocal coords to Blaze Cartesian k (1/a units).

    MPB k_frac: coordinates in the basis of reciprocal lattice vectors
      G_j defined by R_i · G_j = 2π δ_ij.
      So k_physical = k_frac[0]*G1 + k_frac[1]*G2 = 2π * inv(A).T @ k_frac

    Blaze k0: Cartesian reciprocal space in 1/a units (NOT 2π/a as the guide
    incorrectly states). Empirically verified via free-photon tests.

    Eigenvalues are in (1/a)² and freq = sqrt(λ)/(2π) in c/a units.
    """
    A = np.array(lattice_vectors)  # rows = lattice vectors
    A_inv_T = np.linalg.inv(A).T
    return 2 * np.pi * A_inv_T @ np.array(k_frac)


def kpath_distances(kpts_cart):
    """Compute cumulative path distance along k-path in Cartesian coords."""
    dists = [0.0]
    for i in range(1, len(kpts_cart)):
        d = np.linalg.norm(kpts_cart[i] - kpts_cart[i - 1])
        dists.append(dists[-1] + d)
    return np.array(dists)


# ============================================================================
# MPB band computation
# ============================================================================

def compute_mpb_bands(config, polarization, num_bands, resolution, n_per_seg):
    """Run MPB along the BZ path and return frequencies."""
    import meep as mp
    from meep import mpb

    # Build lattice
    basis = config.get("mpb_lattice_basis")
    if basis is None:
        lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    else:
        lattice = mp.Lattice(
            size=mp.Vector3(1, 1, 0),
            basis1=mp.Vector3(basis[0][0], basis[0][1], 0),
            basis2=mp.Vector3(basis[1][0], basis[1][1], 0),
        )

    # Build geometry
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
        resolution=resolution,
    )

    # Build k-path using MPB interpolation
    kpath_frac = config["kpath_frac"]
    k_points_mp = [mp.Vector3(k[0], k[1], 0) for k in kpath_frac]
    ms.k_points = mp.interpolate(n_per_seg, k_points_mp)

    # Run
    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        if polarization == "TE":
            ms.run_te()
        else:
            ms.run_tm()
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull_fd)
        os.close(old_stdout)
        os.close(old_stderr)

    # Extract frequencies: ms.all_freqs is list of lists
    freqs = np.array(ms.all_freqs)  # shape: (n_kpts, num_bands)

    # Extract actual k-points used (in fractional coords)
    kpts_frac = np.array([[k.x, k.y] for k in ms.k_points])

    return freqs, kpts_frac


# ============================================================================
# Blaze2D band computation
# ============================================================================

def compute_blaze_bands(config, polarization, num_bands, resolution, kpts_frac):
    """
    Compute bands using Blaze2D at each k-point.

    kpts_frac: array of fractional reciprocal coordinates (MPB convention).
    Blaze needs Cartesian k in units of 2π/a.
    """
    lattice_vectors = config["lattice_vectors"]
    atoms = config["atoms"]
    eps_bg = config["eps_bg"]

    n_kpts = len(kpts_frac)
    freqs_blaze = np.zeros((n_kpts, num_bands))

    # We use n_retained=num_bands and n_remote=0 since we only need eigenvalues
    for i, kf in enumerate(kpts_frac):
        # Convert fractional k to Cartesian Blaze k
        k_cart = frac_to_cartesian_k(kf, lattice_vectors)

        result = EAExtractor.extract(
            lattice_vectors=lattice_vectors,
            atoms=atoms,
            eps_bg=eps_bg,
            k0=list(k_cart),
            polarization=polarization,
            resolution=resolution,
            n_retained=num_bands,
            n_remote=0,
            compute_born_huang=False,
            compute_r_derivatives=False,
        )

        # Blaze eigenvalues: λ_n in (2π/a)² units → freq = sqrt(λ) / (2π)
        eigenvalues = np.array(result["eigenvalues"][:num_bands])
        # Clamp any slightly negative eigenvalues (numerical noise near Γ)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        freqs_blaze[i, :] = np.sqrt(eigenvalues) / (2 * np.pi)

    return freqs_blaze


# ============================================================================
# Plotting
# ============================================================================

def plot_comparison(all_results):
    """Create a 3-column figure with side-by-side band diagrams."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=False)

    config_names = list(all_results.keys())

    for col, cname in enumerate(config_names):
        res = all_results[cname]
        for row, pol in enumerate(["TE", "TM"]):
            ax = axes[row, col]
            pdata = res[pol]

            x = pdata["k_dist"]
            x_norm = x / x[-1]  # normalize to [0, 1]

            # Plot MPB bands
            for b in range(pdata["freqs_mpb"].shape[1]):
                ax.plot(x_norm, pdata["freqs_mpb"][:, b], 'b-', lw=1.2,
                        alpha=0.8, label="MPB" if b == 0 else None)

            # Plot Blaze bands
            for b in range(pdata["freqs_blaze"].shape[1]):
                ax.plot(x_norm, pdata["freqs_blaze"][:, b], 'r--', lw=1.0,
                        alpha=0.8, label="Blaze" if b == 0 else None)

            # High-symmetry labels
            labels = CONFIGS[cname]["kpath_labels"]
            n_seg = len(labels) - 1
            n_per = N_KPTS_PER_SEGMENT
            tick_positions = []
            for s in range(n_seg + 1):
                idx = min(s * n_per, len(x_norm) - 1)
                tick_positions.append(x_norm[idx])
            # For MPB-interpolated paths, tick positions come from segment endpoints
            # Recompute from the actual k_dist
            kpath_frac = CONFIGS[cname]["kpath_frac"]
            lattice_vectors = CONFIGS[cname]["lattice_vectors"]
            seg_endpoints = []
            for kf in kpath_frac:
                kc = frac_to_cartesian_k(np.array(kf), lattice_vectors)
                seg_endpoints.append(kc)
            seg_dist = [0.0]
            for s in range(1, len(seg_endpoints)):
                d = np.linalg.norm(np.array(seg_endpoints[s]) - np.array(seg_endpoints[s - 1]))
                seg_dist.append(seg_dist[-1] + d)
            seg_dist = np.array(seg_dist)
            total = seg_dist[-1]
            if total > 0:
                tick_norm = seg_dist / total
            else:
                tick_norm = np.linspace(0, 1, len(labels))

            ax.set_xticks(tick_norm)
            ax.set_xticklabels(labels)
            for tp in tick_norm:
                ax.axvline(tp, color='gray', lw=0.5, ls=':')

            ax.set_ylabel(f"Frequency (c/a)")
            ax.legend(loc='upper left', fontsize=8)
            if row == 0:
                ax.set_title(f"{CONFIGS[cname]['label']}\n{pol}", fontsize=10)
            else:
                ax.set_title(f"{pol}", fontsize=10)
            ax.set_xlim(0, 1)

    fig.suptitle("Blaze2D vs MPB: Band Diagram Comparison (resolution=32)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "band_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'band_comparison.png'}")


def print_numerical_comparison(all_results):
    """Print band-by-band max/rms errors."""
    print("\n" + "=" * 80)
    print("NUMERICAL COMPARISON: Blaze2D vs MPB")
    print("=" * 80)

    for cname, res in all_results.items():
        print(f"\n--- {CONFIGS[cname]['label']} ---")
        for pol in ["TE", "TM"]:
            pdata = res[pol]
            diff = pdata["freqs_blaze"] - pdata["freqs_mpb"]
            rms = np.sqrt(np.mean(diff**2, axis=0))
            max_err = np.max(np.abs(diff), axis=0)
            rel_err = np.abs(diff) / (np.abs(pdata["freqs_mpb"]) + 1e-15)
            max_rel = np.max(rel_err, axis=0)

            print(f"\n  {pol} polarization:")
            print(f"  {'Band':>5} {'RMS err':>12} {'Max err':>12} {'Max rel%':>12}")
            for b in range(min(10, len(rms))):  # print first 10 bands
                print(f"  {b+1:5d} {rms[b]:12.6f} {max_err[b]:12.6f} {max_rel[b]*100:11.4f}%")
            # Summary
            overall_rms = np.sqrt(np.mean(diff**2))
            overall_max = np.max(np.abs(diff))
            print(f"  {'Overall':>5} {overall_rms:12.6f} {overall_max:12.6f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Blaze2D vs MPB — Band Diagram Validation")
    print(f"Resolution: {RESOLUTION}, Bands: {NUM_BANDS}, K-pts/segment: {N_KPTS_PER_SEGMENT}")
    print("=" * 80)

    all_results = {}

    # ---------------------------------------------------------------
    # Phase 1: Compute ALL Blaze bands BEFORE importing meep.
    # Importing meep corrupts Blaze's TM generalized eigensolver
    # (shared BLAS/LAPACK library conflict).
    # ---------------------------------------------------------------
    print("\n--- Phase 1: Computing all Blaze bands (before meep import) ---")
    blaze_cache = {}
    for cname, config in CONFIGS.items():
        blaze_cache[cname] = {}
        for pol in ["TE", "TM"]:
            # Build k-path in fractional coords (same as MPB will use)
            kpath_frac = config["kpath_frac"]
            kpts_frac = build_kpath_frac(kpath_frac, N_KPTS_PER_SEGMENT)

            print(f"  [{cname}/{pol}] Running Blaze2D ({len(kpts_frac)} k-pts)... ",
                  end="", flush=True)
            t0 = time.time()
            freqs_blaze = compute_blaze_bands(
                config, pol, NUM_BANDS, RESOLUTION, kpts_frac
            )
            t_blaze = time.time() - t0
            print(f"done ({t_blaze:.1f}s)")

            blaze_cache[cname][pol] = {
                "freqs_blaze": freqs_blaze,
                "kpts_frac": kpts_frac,
            }

    # ---------------------------------------------------------------
    # Phase 2: Import meep and compute ALL MPB bands.
    # ---------------------------------------------------------------
    print("\n--- Phase 2: Computing all MPB bands ---")
    for cname, config in CONFIGS.items():
        all_results[cname] = {}
        for pol in ["TE", "TM"]:
            print(f"  [{cname}/{pol}] Running MPB... ", end="", flush=True)
            t0 = time.time()
            freqs_mpb, kpts_frac_mpb = compute_mpb_bands(
                config, pol, NUM_BANDS, RESOLUTION, N_KPTS_PER_SEGMENT
            )
            t_mpb = time.time() - t0
            print(f"done ({t_mpb:.1f}s, {len(kpts_frac_mpb)} k-pts)")

            # Use the Blaze k-path (which matches MPB interpolation for
            # uniform segments, but may differ slightly). For a proper
            # comparison we need the same k-points.
            bc = blaze_cache[cname][pol]

            # MPB uses mp.interpolate which may yield different k-points
            # than our build_kpath_frac. We need to match them up.
            # Re-run Blaze at MPB's exact k-points if they differ.
            kpts_blaze = bc["kpts_frac"]
            if len(kpts_frac_mpb) != len(kpts_blaze) or \
               not np.allclose(kpts_frac_mpb, kpts_blaze, atol=1e-10):
                print(f"    Warning: MPB has {len(kpts_frac_mpb)} k-pts vs "
                      f"Blaze {len(kpts_blaze)}. Using Blaze's own k-path.")

            # Compute k-path distances for plotting (using Blaze's k-points)
            lattice_vectors = config["lattice_vectors"]
            kpts_cart = np.array([
                frac_to_cartesian_k(kf, lattice_vectors) for kf in kpts_blaze
            ])
            k_dist = kpath_distances(kpts_cart)

            # Trim to common length
            n_common = min(len(bc["freqs_blaze"]), len(freqs_mpb))
            all_results[cname][pol] = {
                "freqs_mpb": freqs_mpb[:n_common],
                "freqs_blaze": bc["freqs_blaze"][:n_common],
                "kpts_frac": kpts_blaze[:n_common],
                "k_dist": k_dist[:n_common],
            }

    # Numerical comparison
    print_numerical_comparison(all_results)

    # Plots
    print("\nGenerating plots...")
    plot_comparison(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
