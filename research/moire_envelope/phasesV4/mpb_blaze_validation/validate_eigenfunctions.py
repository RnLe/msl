"""
Blaze2D vs MPB Validation — Step 3: Eigenfunction Comparison

Compares the Bloch eigenvectors from Blaze2D and MPB via:
  1. Scalar products |<u_m^MPB | u_n^Blaze>|² (overlap matrix)
  2. Per-band overlap magnitudes (should be ~1 for same band, ~0 otherwise)
  3. Field profile comparison plots (Hz for TE, Ez for TM)

The comparison is done in real space on the unit cell grid.
Key subtlety: eigenvectors have arbitrary phase (gauge freedom),
so we compare |overlap|², not raw complex overlaps.
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
N_BANDS = 6   # compare first 6 bands
N_RETAINED = 4
N_REMOTE = 6


# ============================================================================
# Lattice configurations (reused from validate_bands.py)
# ============================================================================

CONFIGS = {
    "square_rods": {
        "label": "Square, ε=8.9 rods",
        "lattice_vectors": [[1.0, 0.0], [0.0, 1.0]],
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 8.9}],
        "eps_bg": 1.0,
        "k0_frac": [0.25, 0.0],  # not at high-symmetry to avoid degeneracies
        "k0_label": "Δ",
        "mpb_lattice_basis": None,
        "mpb_cylinders": [{"radius": 0.2, "eps": 8.9, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 1.0,
    },
    "hex_holes": {
        "label": "Hex, ε=13 bg, air holes",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "atoms": [{"pos": [0.0, 0.0], "radius": 0.48, "eps_inside": 1.0}],
        "eps_bg": 13.0,
        "k0_frac": [0.2, 0.1],  # generic point
        "k0_label": "generic",
        "mpb_lattice_basis": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "mpb_cylinders": [{"radius": 0.48, "eps": 1.0, "center": [0.0, 0.0]}],
        "mpb_eps_bg": 13.0,
    },
    "honeycomb_holes": {
        "label": "Honeycomb, ε=12 bg",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3)/2]],
        "atoms": [
            {"pos": [0.0, 0.0], "radius": 0.25, "eps_inside": 1.0},
            {"pos": [1/3, 1/3], "radius": 0.25, "eps_inside": 1.0},
        ],
        "eps_bg": 12.0,
        "k0_frac": [0.2, 0.1],  # generic point
        "k0_label": "generic",
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
    """Convert MPB fractional reciprocal coords to Blaze Cartesian k (1/a)."""
    A = np.array(lattice_vectors)
    A_inv_T = np.linalg.inv(A).T
    return 2 * np.pi * A_inv_T @ np.array(k_frac)


# ============================================================================
# MPB eigenfunction extraction
# ============================================================================

def get_mpb_eigenvectors(config, polarization, k0_frac, n_bands):
    """
    Extract MPB eigenvectors at a single k-point.
    Returns: eigenvectors as complex arrays on the MPB grid,
             eigenfrequencies, and epsilon grid.
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
        num_bands=n_bands,
        resolution=RESOLUTION,
    )

    ms.k_points = [mp.Vector3(k0_frac[0], k0_frac[1], 0)]

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

    freqs = np.array(ms.all_freqs[0])
    eps = np.array(ms.get_epsilon())
    if eps.ndim == 3:
        eps = eps[:, :, 0]

    # Extract fields for each band
    eigenvectors = []
    for band in range(1, n_bands + 1):
        if polarization == "TE":
            # TE: relevant field is Hz (z-component of H)
            ms.get_hfield(band)
            field = ms.get_hfield(band, bloch_phase=False)
            # field is shape (Nx, Ny, Nz, 3) complex
            field = np.array(field)
            if field.ndim == 4:
                # z-component
                hz = field[:, :, 0, 2]
            else:
                hz = field[:, :, 2]
            eigenvectors.append(hz.copy())
        else:
            # TM: relevant field is Ez (z-component of E)
            field = ms.get_efield(band, bloch_phase=False)
            field = np.array(field)
            if field.ndim == 4:
                ez = field[:, :, 0, 2]
            else:
                ez = field[:, :, 2]
            eigenvectors.append(ez.copy())

    return eigenvectors, freqs, eps


# ============================================================================
# Blaze eigenfunction extraction
# ============================================================================

def get_blaze_eigenvectors(config, polarization, k0_frac, n_bands):
    """
    Extract Blaze2D eigenvectors at a single k-point.
    
    Blaze returns eigenvectors in Fourier (G-space) as flat arrays.
    We need to inverse FFT to get real-space fields for comparison with MPB.
    """
    lattice_vectors = config["lattice_vectors"]
    k0_cart = frac_to_cartesian_k(k0_frac, lattice_vectors)

    result = EAExtractor.extract(
        lattice_vectors=lattice_vectors,
        atoms=config["atoms"],
        eps_bg=config["eps_bg"],
        k0=list(k0_cart),
        polarization=polarization,
        resolution=RESOLUTION,
        n_retained=n_bands,
        n_remote=0,
        compute_born_huang=False,
        compute_r_derivatives=False,
    )

    # Grid dimensions
    grid_dims = result["grid_dims"]  # (Nx, Ny)
    Nx, Ny = grid_dims

    eigenvalues = np.array(result["eigenvalues"][:n_bands])
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2 * np.pi)

    # Eigenvectors: list of n_total vectors, each a flat list of (re, im) tuples
    eigenvectors_fourier = []
    for b in range(n_bands):
        ev_data = result["eigenvectors"][b]
        ev = np.array([re + 1j * im for re, im in ev_data])
        eigenvectors_fourier.append(ev.reshape(Nx, Ny))

    # Convert from G-space to real-space via 2D IFFT
    eigenvectors_real = []
    for ev_g in eigenvectors_fourier:
        ev_r = np.fft.ifft2(ev_g) * (Nx * Ny)  # unnormalized IFFT
        eigenvectors_real.append(ev_r)

    return eigenvectors_real, freqs, eigenvalues, grid_dims


# ============================================================================
# Overlap computation
# ============================================================================

def compute_overlap_matrix(vecs_a, vecs_b, eps_grid=None, polarization="TE"):
    """
    Compute the overlap matrix between two sets of eigenvectors.
    
    For TE: standard inner product <u_m|u_n> = sum u_m* u_n / N
    For TM: ε-weighted <u_m|ε|u_n> = sum u_m* ε u_n / N
    
    Returns |S_mn|² matrix.
    """
    n_a = len(vecs_a)
    n_b = len(vecs_b)
    S = np.zeros((n_a, n_b), dtype=complex)

    for m in range(n_a):
        for n in range(n_b):
            va = vecs_a[m]
            vb = vecs_b[n]
            # Ensure same shape by interpolating if needed
            if va.shape != vb.shape:
                # Resize vb to match va using Fourier interpolation
                vb = _fourier_resize(vb, va.shape)

            if polarization == "TM" and eps_grid is not None:
                # ε-weighted inner product
                eps_r = eps_grid
                if eps_r.shape != va.shape:
                    eps_r = _fourier_resize_real(eps_r, va.shape)
                S[m, n] = np.sum(np.conj(va) * eps_r * vb) / va.size
            else:
                # Standard inner product
                S[m, n] = np.sum(np.conj(va) * vb) / va.size

    return S


def _fourier_resize(field_2d, target_shape):
    """Resize a complex 2D field to target_shape via Fourier interpolation."""
    ft = np.fft.fft2(field_2d)
    ft_resized = np.zeros(target_shape, dtype=complex)
    
    sx, sy = field_2d.shape
    tx, ty = target_shape
    
    # Copy low-frequency components
    cx = min(sx, tx) // 2
    cy = min(sy, ty) // 2
    
    ft_resized[:cx, :cy] = ft[:cx, :cy]
    ft_resized[:cx, -cy:] = ft[:cx, -cy:]
    ft_resized[-cx:, :cy] = ft[-cx:, :cy]
    ft_resized[-cx:, -cy:] = ft[-cx:, -cy:]
    
    result = np.fft.ifft2(ft_resized) * (tx * ty) / (sx * sy)
    return result


def _fourier_resize_real(field_2d, target_shape):
    """Resize a real 2D field."""
    return _fourier_resize(field_2d.astype(complex), target_shape).real


# ============================================================================
# Plotting
# ============================================================================

def plot_overlap_matrices(all_results):
    """Plot overlap matrices for each configuration."""
    n_configs = len(all_results)
    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 9))
    if n_configs == 1:
        axes = axes[:, np.newaxis]

    config_names = list(all_results.keys())

    for col, cname in enumerate(config_names):
        for row, pol in enumerate(["TE", "TM"]):
            ax = axes[row, col]
            res = all_results[cname][pol]
            S_sq = res["overlap_sq"]
            n = S_sq.shape[0]

            im = ax.imshow(S_sq, vmin=0, vmax=1, cmap='RdYlGn',
                           origin='lower', aspect='equal')
            ax.set_xlabel("Blaze band")
            ax.set_ylabel("MPB band")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(range(1, n + 1))
            ax.set_yticklabels(range(1, n + 1))

            # Annotate with values
            for i in range(n):
                for j in range(n):
                    color = 'white' if S_sq[i, j] < 0.5 else 'black'
                    ax.text(j, i, f"{S_sq[i, j]:.2f}", ha='center', va='center',
                            fontsize=7, color=color)

            if row == 0:
                ax.set_title(f"{CONFIGS[cname]['label']}\n{pol} |<MPB|Blaze>|²")
            else:
                ax.set_title(f"{pol} |<MPB|Blaze>|²")
            plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Eigenfunction Overlap: MPB vs Blaze2D", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eigenfunction_overlap.png", dpi=150)
    plt.close()
    print(f"  Saved {OUTPUT_DIR / 'eigenfunction_overlap.png'}")


def plot_field_profiles(all_results):
    """Plot side-by-side field profiles for the first 3 bands of one config."""
    cname = list(all_results.keys())[0]
    n_show = min(3, N_BANDS)

    for pol in ["TE", "TM"]:
        fig, axes = plt.subplots(2, n_show, figsize=(5 * n_show, 8))
        res = all_results[cname][pol]

        for b in range(n_show):
            # MPB field
            ax = axes[0, b]
            field_mpb = np.abs(res["mpb_vecs"][b])**2
            ax.imshow(field_mpb.T, origin='lower', cmap='inferno')
            ax.set_title(f"MPB band {b+1}")
            ax.set_aspect('equal')

            # Blaze field
            ax = axes[1, b]
            field_blaze = np.abs(res["blaze_vecs"][b])**2
            ax.imshow(field_blaze.T, origin='lower', cmap='inferno')
            ax.set_title(f"Blaze band {b+1}")
            ax.set_aspect('equal')

        fig.suptitle(f"{CONFIGS[cname]['label']} — {pol} |u(r)|²", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"field_profiles_{pol}.png", dpi=150)
        plt.close()
        print(f"  Saved {OUTPUT_DIR / f'field_profiles_{pol}.png'}")


def print_eigenfunction_comparison(all_results):
    """Print summary of eigenfunction comparison."""
    print("\n" + "=" * 80)
    print("EIGENFUNCTION COMPARISON: |<MPB|Blaze>|²")
    print("=" * 80)

    for cname, res_pols in all_results.items():
        print(f"\n--- {CONFIGS[cname]['label']} ---")
        for pol in ["TE", "TM"]:
            res = res_pols[pol]
            S_sq = res["overlap_sq"]
            n = S_sq.shape[0]

            print(f"\n  {pol} at k={CONFIGS[cname]['k0_label']}:")
            print(f"  Frequency comparison (MPB c/a vs Blaze c/a):")
            for b in range(n):
                fmpb = res["freqs_mpb"][b]
                fblz = res["freqs_blaze"][b]
                print(f"    Band {b+1}: MPB={fmpb:.6f}  Blaze={fblz:.6f}  diff={abs(fmpb-fblz):.2e}")

            print(f"\n  Overlap matrix diagonal (should be ~1):")
            diag = np.diag(S_sq)
            for b in range(n):
                print(f"    Band {b+1}: |<MPB|Blaze>|² = {diag[b]:.6f}")

            # Check for band permutations
            max_per_row = np.argmax(S_sq, axis=1)
            if not np.array_equal(max_per_row, np.arange(n)):
                print(f"  WARNING: Band ordering mismatch detected!")
                print(f"    MPB band → best Blaze match: {max_per_row + 1}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Blaze2D vs MPB — Eigenfunction Comparison")
    print(f"Resolution: {RESOLUTION}, Bands: {N_BANDS}")
    print("=" * 80)

    all_results = {}

    # Phase 1: Extract all Blaze eigenvectors BEFORE importing meep
    print("\n--- Phase 1: Extracting Blaze eigenvectors (before meep import) ---")
    blaze_cache = {}
    for cname, config in CONFIGS.items():
        blaze_cache[cname] = {}
        k0_frac = config["k0_frac"]
        for pol in ["TE", "TM"]:
            print(f"  [{cname}/{pol}] Extracting Blaze eigenvectors... ", end="", flush=True)
            t0 = time.time()
            blaze_vecs, blaze_freqs, blaze_evals, grid_dims = get_blaze_eigenvectors(
                config, pol, k0_frac, N_BANDS
            )
            print(f"done ({time.time()-t0:.1f}s)")
            blaze_cache[cname][pol] = {
                "blaze_vecs": blaze_vecs,
                "freqs_blaze": blaze_freqs,
            }

    # Phase 2: Extract all MPB eigenvectors and compute overlaps
    print("\n--- Phase 2: Extracting MPB eigenvectors and computing overlaps ---")
    for cname, config in CONFIGS.items():
        all_results[cname] = {}
        k0_frac = config["k0_frac"]

        for pol in ["TE", "TM"]:
            print(f"  [{cname}/{pol}] Extracting MPB eigenvectors... ", end="", flush=True)
            t0 = time.time()
            mpb_vecs, mpb_freqs, mpb_eps = get_mpb_eigenvectors(
                config, pol, k0_frac, N_BANDS
            )
            print(f"done ({time.time()-t0:.1f}s)")

            bc = blaze_cache[cname][pol]
            print(f"  [{cname}/{pol}] Computing overlaps... ", end="", flush=True)
            S = compute_overlap_matrix(mpb_vecs, bc["blaze_vecs"], mpb_eps, pol)
            S_sq = np.abs(S) ** 2
            print("done")

            all_results[cname][pol] = {
                "overlap_sq": S_sq,
                "freqs_mpb": mpb_freqs[:N_BANDS],
                "freqs_blaze": bc["freqs_blaze"],
                "mpb_vecs": mpb_vecs,
                "blaze_vecs": bc["blaze_vecs"],
            }

    print_eigenfunction_comparison(all_results)

    print("\nGenerating plots...")
    plot_overlap_matrices(all_results)
    plot_field_profiles(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
