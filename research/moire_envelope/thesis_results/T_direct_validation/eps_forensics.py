"""
Phase D1b: Epsilon Grid Forensics — Binary vs Smoothed vs MPB

Builds and compares the dielectric function ε(x, y) from three approaches:
  1. Binary (staircase) — point-in-rod test at grid corners
  2. Smoothed — subpixel smoothing with sub-grid sampling
  3. MPB — MPB's internal subpixel-averaged ε (ground truth)

For BOTH:
  (a) Monolayer honeycomb unit cell
  (b) Moiré supercell (configurable size)

Produces:
  - 3×2 visual comparison plot
  - Numerical statistics (pixelwise differences, means, fill fractions)
  - Difference maps
"""
import numpy as np
import sys
import os
import time
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from T_direct_validation.subpixel_smoothing import build_smoothed_eps_monolayer
from T_direct_validation.supercell_geometry import (
    build_monolayer_basis,
    get_sublattice_positions,
    rotation_matrix_2d,
)
from T_direct_validation.commensurate_utils import (
    commensurate_twist_angle,
)

out_dir = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
EPS_BG = 1.0
EPS_ROD = 11.56
R_OVER_A = 0.2

# Monolayer comparison resolution
MONO_RES = 64

# Supercell comparison — small cell, large angle for fast MPB
SC_M, SC_N = 4, 3       # θ ≈ 9.43°, N_cells = 37
SC_RES = 32              # resolution per unit cell → Nx ≈ sqrt(37)*32 ≈ 195


# ════════════════════════════════════════════════════════════════
# MPB epsilon extraction
# ════════════════════════════════════════════════════════════════
def get_mpb_epsilon_monolayer(resolution):
    """Extract MPB's subpixel-averaged ε for a honeycomb monolayer."""
    import meep as mp
    from meep import mpb

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(1, 0, 0),
        basis2=mp.Vector3(0.5, math.sqrt(3)/2, 0),
    )

    r = R_OVER_A
    geometry = [
        mp.Cylinder(radius=r, center=mp.Vector3(0, 0, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
        mp.Cylinder(radius=r, center=mp.Vector3(1/3, 1/3, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
    ]

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=1,
        resolution=resolution,
    )

    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        ms.init_params(mp.NO_PARITY, False)
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull_fd)
        os.close(old_stdout)
        os.close(old_stderr)

    eps = np.array(ms.get_epsilon(), dtype=np.float64)
    if eps.ndim == 3:
        eps = eps[:, :, 0]
    return eps


def build_coincidence_supercell(m, n, a=1.0):
    """
    Build the COINCIDENCE supercell vectors for a twisted bilayer.

    The standard build_supercell_vectors uses L1 = m*a1 + n*a2, which is
    periodic for the unrotated layer but NOT the rotated layer.

    The correct coincidence vectors are:
        C1 = n*a1 + m*a2
        C2 = -m*a1 + (m+n)*a2
    These are integer combinations of BOTH the unrotated {a1,a2} and
    the rotated {R(θ)a1, R(θ)a2} bases.

    Returns: B_coinc (2,2) array with C1, C2 as columns
    """
    B_mono = build_monolayer_basis('honeycomb', a)
    a1 = B_mono[:, 0]
    a2 = B_mono[:, 1]
    C1 = n * a1 + m * a2
    C2 = -m * a1 + (m + n) * a2
    return np.column_stack([C1, C2])


def get_mpb_epsilon_supercell(m, n, resolution):
    """
    Extract MPB's subpixel-averaged ε for a twisted bilayer honeycomb supercell.

    Uses the COINCIDENCE supercell (periodic for both layers).
    """
    import meep as mp
    from meep import mpb

    a = 1.0
    theta_rad = commensurate_twist_angle('honeycomb', m, n)
    theta_deg = math.degrees(theta_rad)

    B_mono = build_monolayer_basis('honeycomb', a)
    a1 = B_mono[:, 0]
    a2 = B_mono[:, 1]

    B_coinc = build_coincidence_supercell(m, n, a)
    C1 = B_coinc[:, 0]
    C2 = B_coinc[:, 1]

    N_cells = m*m + m*n + n*n

    # MPB convention: size=(1,1) with raw (non-normalized) basis vectors.
    # Radius must be divided by |C1| since MPB scales it by the basis length.
    L1_len = np.linalg.norm(C1)
    L2_len = np.linalg.norm(C2)

    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(C1[0], C1[1], 0),
        basis2=mp.Vector3(C2[0], C2[1], 0),
    )

    # Scale resolution so the physical pixel density matches the input
    # (with size=(1,1), MPB grid = resolution x resolution)
    mpb_resolution = int(round(resolution * L1_len))

    R = rotation_matrix_2d(theta_rad)
    sublattice = get_sublattice_positions('honeycomb')
    B_coinc_inv = np.linalg.inv(B_coinc)

    rod_positions = []

    for layer_idx, B_layer in enumerate([B_mono, np.column_stack([R @ a1, R @ a2])]):
        for sub_frac in sublattice:
            sub_cart = B_layer @ sub_frac

            seen = set()
            N_scan = int(math.sqrt(N_cells)) + 5
            for n1 in range(-N_scan, N_scan + 1):
                for n2 in range(-N_scan, N_scan + 1):
                    pos_cart = n1 * B_layer[:, 0] + n2 * B_layer[:, 1] + sub_cart
                    f = B_coinc_inv @ pos_cart
                    f_wrapped = f - np.floor(f)

                    key = (round(f_wrapped[0], 8), round(f_wrapped[1], 8))
                    if key[0] > 1.0 - 1e-6 or key[1] > 1.0 - 1e-6:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)

                    # Use FDFD fractional coords [0,1); MPB wraps to [-0.5, 0.5) internally.
                    # The output grid will be rolled by (N/2, N/2) in main() to
                    # convert from MPB center-origin to FDFD corner-origin,
                    # matching the same convention as the monolayer function.
                    cx = f_wrapped[0]
                    cy = f_wrapped[1]
                    rod_positions.append((cx, cy))

    geometry = [
        mp.Cylinder(
            radius=R_OVER_A * a / L1_len,
            center=mp.Vector3(cx, cy, 0),
            material=mp.Medium(epsilon=EPS_ROD),
        )
        for cx, cy in rod_positions
    ]

    n_rods = len(geometry)
    expected_rods = 4 * N_cells
    print(f"  MPB supercell ({m},{n}): θ={theta_deg:.2f}°, N_cells={N_cells}, "
          f"{n_rods} rods (expected {expected_rods}), resolution={resolution} "
          f"(mpb_resolution={mpb_resolution})")

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=1,
        resolution=mpb_resolution,
    )

    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        ms.init_params(mp.NO_PARITY, False)
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull_fd)
        os.close(old_stdout)
        os.close(old_stderr)

    eps = np.array(ms.get_epsilon(), dtype=np.float64)
    if eps.ndim == 3:
        eps = eps[:, :, 0]
    return eps


# ════════════════════════════════════════════════════════════════
# FDFD epsilon (binary + smoothed)
# ════════════════════════════════════════════════════════════════
def get_fdfd_eps_monolayer(resolution, n_sub=16):
    """Build binary and smoothed ε for monolayer."""
    eps_binary, eps_smoothed, info = build_smoothed_eps_monolayer(
        resolution=resolution, a=1.0, r_over_a=R_OVER_A,
        eps_rod=EPS_ROD, eps_bg=EPS_BG, n_sub=n_sub,
    )
    return eps_binary, eps_smoothed, info


def get_fdfd_eps_supercell(m, n, resolution, n_sub=16):
    """Build binary and smoothed ε for supercell."""
    from T_direct_validation.subpixel_smoothing import build_smoothed_eps_supercell

    N_cells = m*m + m*n + n*n
    Nx = int(round(math.sqrt(N_cells) * resolution))

    eps_binary, sc_info = build_supercell_eps(
        'honeycomb', m=m, n=n, a=1.0,
        r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
        Nx=Nx, Ny=Nx,
    )

    eps_smoothed, smooth_info = build_smoothed_eps_supercell(
        eps_binary, sc_info, n_sub=n_sub,
        eps_rod=EPS_ROD, eps_bg=EPS_BG,
    )

    return eps_binary, eps_smoothed, sc_info, smooth_info


# ════════════════════════════════════════════════════════════════
# Grid coordinate mapping — where are the pixels?
# ════════════════════════════════════════════════════════════════
def report_grid_conventions():
    """Report where each method places its grid points."""
    print("\n" + "=" * 70)
    print("GRID COORDINATE CONVENTIONS")
    print("=" * 70)
    print("""
  FDFD binary:  s = i/N         (pixel corners, i=0..N-1)
  FDFD smooth:  boundary uses corner coords, sub-grid at center offsets
  MPB:          s = (i+0.5)/N   (pixel centers, i=0..N-1)
  
  For a 4×4 grid (N=4):
    FDFD corners: 0.000, 0.250, 0.500, 0.750
    MPB centers:  0.125, 0.375, 0.625, 0.875
  
  This HALF-PIXEL SHIFT means:
  - A rod boundary at exactly a grid line falls on different pixels
  - The fill fraction differs systematically
  - Direct pixelwise comparison is invalid without resampling
""")


# ════════════════════════════════════════════════════════════════
# Comparison statistics
# ════════════════════════════════════════════════════════════════
def compare_eps(eps_a, eps_b, label_a='A', label_b='B'):
    """Compute comparison statistics between two ε grids."""
    assert eps_a.shape == eps_b.shape, f"Shape mismatch: {eps_a.shape} vs {eps_b.shape}"

    diff = eps_a - eps_b
    abs_diff = np.abs(diff)

    stats = {
        'mean_a': eps_a.mean(),
        'mean_b': eps_b.mean(),
        'harmonic_mean_a': 1.0 / (1.0 / eps_a).mean(),
        'harmonic_mean_b': 1.0 / (1.0 / eps_b).mean(),
        'fill_a': (eps_a > 0.5 * (EPS_BG + EPS_ROD)).mean(),
        'fill_b': (eps_b > 0.5 * (EPS_BG + EPS_ROD)).mean(),
        'mean_diff': diff.mean(),
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'rms_diff': np.sqrt((diff**2).mean()),
        'n_exact_match': (abs_diff < 1e-10).sum(),
        'n_different': (abs_diff > 1e-10).sum(),
        'frac_different': (abs_diff > 1e-10).mean(),
    }

    print(f"\n  {label_a} vs {label_b}:")
    print(f"    Mean ε:           {stats['mean_a']:.6f}  vs  {stats['mean_b']:.6f}  (Δ={stats['mean_diff']:+.6f})")
    print(f"    Harmonic mean ε:  {stats['harmonic_mean_a']:.6f}  vs  {stats['harmonic_mean_b']:.6f}")
    print(f"    Fill fraction:    {stats['fill_a']:.4%}  vs  {stats['fill_b']:.4%}")
    print(f"    Mean |Δε|:        {stats['mean_abs_diff']:.6f}")
    print(f"    Max  |Δε|:        {stats['max_abs_diff']:.6f}")
    print(f"    RMS  Δε:          {stats['rms_diff']:.6f}")
    print(f"    Pixels different: {stats['n_different']} / {eps_a.size} ({stats['frac_different']:.2%})")

    return stats


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════
def plot_eps_grid(ax, eps, B, title='', vmin=None, vmax=None, cmap='RdYlBu_r'):
    """Plot ε on oblique grid using physical coordinates."""
    Nx, Ny = eps.shape
    L1 = B[:, 0]
    L2 = B[:, 1]

    # Create coordinate arrays for pcolormesh
    # Grid in fractional coords: corners of each pixel
    s1 = np.arange(Nx + 1) / Nx
    s2 = np.arange(Ny + 1) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]

    if vmin is None: vmin = EPS_BG
    if vmax is None: vmax = EPS_ROD

    im = ax.pcolormesh(X, Y, eps, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading='flat', rasterized=True)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    return im


def plot_3x2_comparison(mono_data, sc_data, sc_label, fname):
    """
    3×2 subplot: rows = {Binary, Smoothed, MPB}, cols = {Monolayer, Supercell}
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    row_labels = ['Binary ε', 'Smoothed ε', 'MPB ε']
    col_labels = ['Monolayer', sc_label]

    vmin, vmax = EPS_BG, EPS_ROD

    for row_idx, (row_label, (mono_eps, sc_eps)) in enumerate(zip(
        row_labels,
        [(mono_data['binary'], sc_data['binary']),
         (mono_data['smoothed'], sc_data['smoothed']),
         (mono_data['mpb'], sc_data['mpb'])]
    )):
        # Monolayer
        im = plot_eps_grid(axes[row_idx, 0], mono_eps, mono_data['B'],
                           title=f'{row_label} — Monolayer', vmin=vmin, vmax=vmax)

        # Supercell
        im = plot_eps_grid(axes[row_idx, 1], sc_eps, sc_data['B'],
                           title=f'{row_label} — {sc_label}', vmin=vmin, vmax=vmax)

    # Colorbars
    for ax_row in axes:
        for ax in ax_row:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05)
            fig.colorbar(ax.collections[0], cax=cax)

    fig.suptitle('Epsilon Grid Comparison: Binary vs Smoothed vs MPB',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


def plot_difference_maps(mono_data, sc_data, sc_label, fname):
    """
    2×2 difference maps: (Smoothed−Binary, MPB−Binary) × (Mono, SC)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    diffs = [
        ('Smoothed − Binary', mono_data['smoothed'] - mono_data['binary'],
         sc_data['smoothed'] - sc_data['binary']),
        ('MPB − Binary', mono_data['mpb'] - mono_data['binary'],
         sc_data['mpb'] - sc_data['binary']),
    ]

    for row_idx, (label, mono_diff, sc_diff) in enumerate(diffs):
        # Auto-scale to the difference range
        vmax_mono = max(abs(mono_diff.min()), abs(mono_diff.max()))
        vmax_sc = max(abs(sc_diff.min()), abs(sc_diff.max()))

        if vmax_mono < 1e-10: vmax_mono = 1.0
        if vmax_sc < 1e-10: vmax_sc = 1.0

        plot_eps_grid(axes[row_idx, 0], mono_diff, mono_data['B'],
                      title=f'{label} — Monolayer',
                      vmin=-vmax_mono, vmax=vmax_mono, cmap='RdBu_r')
        plot_eps_grid(axes[row_idx, 1], sc_diff, sc_data['B'],
                      title=f'{label} — {sc_label}',
                      vmin=-vmax_sc, vmax=vmax_sc, cmap='RdBu_r')

    for ax_row in axes:
        for ax in ax_row:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.05)
            fig.colorbar(ax.collections[0], cax=cax, label='Δε')

    fig.suptitle('Epsilon Difference Maps', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


def plot_1d_slices(mono_data, sc_data, sc_label, fname):
    """1D slices through the ε grids for detailed comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for col_idx, (data, label) in enumerate([
        (mono_data, 'Monolayer'),
        (sc_data, sc_label),
    ]):
        eps_bin = data['binary']
        eps_smo = data['smoothed']
        eps_mpb = data['mpb']
        Nx = eps_bin.shape[0]

        # Slice through the middle row
        mid = Nx // 2
        x = np.arange(Nx) / Nx

        # Row slice
        ax = axes[0, col_idx]
        ax.plot(x, eps_bin[mid, :], '-', color='#2563EB', lw=1.5, alpha=0.8, label='Binary')
        ax.plot(x, eps_smo[mid, :], '--', color='#059669', lw=1.5, alpha=0.8, label='Smoothed')
        ax.plot(x, eps_mpb[mid, :], ':', color='#DC2626', lw=2.0, alpha=0.8, label='MPB')
        ax.set_ylabel('ε')
        ax.set_xlabel('s₂ (fractional)')
        ax.set_title(f'{label} — row {mid} (s₁={mid/Nx:.3f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

        # Diagonal slice
        ax = axes[1, col_idx]
        diag_idx = np.arange(Nx)
        ax.plot(x, eps_bin[diag_idx, diag_idx], '-', color='#2563EB', lw=1.5, alpha=0.8, label='Binary')
        ax.plot(x, eps_smo[diag_idx, diag_idx], '--', color='#059669', lw=1.5, alpha=0.8, label='Smoothed')
        ax.plot(x, eps_mpb[diag_idx, diag_idx], ':', color='#DC2626', lw=2.0, alpha=0.8, label='MPB')
        ax.set_ylabel('ε')
        ax.set_xlabel('s (fractional, diagonal)')
        ax.set_title(f'{label} — diagonal slice', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('ε Line Profiles: Binary vs Smoothed vs MPB', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


def plot_zoomed_comparison(data, label, fname, zoom_frac=(0.3, 0.5)):
    """Zoomed-in view of a small region for pixel-level comparison."""
    eps_bin = data['binary']
    eps_smo = data['smoothed']
    eps_mpb = data['mpb']
    B = data['B']
    Nx = eps_bin.shape[0]

    # Zoom region in fractional coordinates
    s_lo = int(zoom_frac[0] * Nx)
    s_hi = int(zoom_frac[1] * Nx)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Binary ε', 'Smoothed ε', 'MPB ε']
    grids = [eps_bin, eps_smo, eps_mpb]

    for ax, eps, title in zip(axes, grids, titles):
        region = eps[s_lo:s_hi, s_lo:s_hi]
        L1 = B[:, 0]
        L2 = B[:, 1]
        # Physical coords for zoomed region
        s1 = np.arange(s_lo, s_hi + 1) / Nx
        s2 = np.arange(s_lo, s_hi + 1) / Nx
        S1, S2 = np.meshgrid(s1, s2, indexing='ij')
        X = S1 * L1[0] + S2 * L2[0]
        Y = S1 * L1[1] + S2 * L2[1]

        im = ax.pcolormesh(X, Y, region, cmap='RdYlBu_r',
                           vmin=EPS_BG, vmax=EPS_ROD, shading='flat', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nmean={region.mean():.4f}', fontsize=10, fontweight='bold')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.suptitle(f'{label} — Zoomed [{zoom_frac[0]:.0%}–{zoom_frac[1]:.0%}]',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    print(f"Saved {fname}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Phase D1b: Epsilon Grid Forensics")
    print("Binary vs Smoothed vs MPB")
    print("=" * 70)

    report_grid_conventions()

    # ── Part 1: Monolayer ──
    print("\n" + "=" * 70)
    print(f"MONOLAYER (res={MONO_RES})")
    print("=" * 70)

    # MPB first to establish grid size
    t0 = time.time()
    eps_mpb_mono = get_mpb_epsilon_monolayer(MONO_RES)
    print(f"  MPB epsilon: {time.time()-t0:.1f}s, shape={eps_mpb_mono.shape}")

    # FDFD at same grid size
    # MPB uses lattice coords centered at (0,0) = center of cell,
    # while FDFD uses fractional coords [0,1) with origin at corner.
    # Roll MPB grid by half the size to align origins.
    Nx_mpb, Ny_mpb = eps_mpb_mono.shape
    eps_mpb_mono = np.roll(eps_mpb_mono, (Nx_mpb // 2, Ny_mpb // 2), axis=(0, 1))
    print(f"  MPB grid shifted by ({Nx_mpb//2}, {Ny_mpb//2}) to align with FDFD origin")

    Nx_mono = eps_mpb_mono.shape[0]
    t0 = time.time()
    eps_bin_mono, eps_smo_mono, mono_info = get_fdfd_eps_monolayer(Nx_mono)
    print(f"  FDFD binary + smoothed: {time.time()-t0:.1f}s, shape={eps_bin_mono.shape}")

    # Handle potential shape mismatch
    if eps_mpb_mono.shape != eps_bin_mono.shape:
        print(f"  WARNING: Shape mismatch! FDFD={eps_bin_mono.shape}, MPB={eps_mpb_mono.shape}")
        Nx = min(eps_bin_mono.shape[0], eps_mpb_mono.shape[0])
        Ny = min(eps_bin_mono.shape[1], eps_mpb_mono.shape[1])
        eps_bin_mono = eps_bin_mono[:Nx, :Ny]
        eps_smo_mono = eps_smo_mono[:Nx, :Ny]
        eps_mpb_mono = eps_mpb_mono[:Nx, :Ny]

    print("\n  ── Pixelwise Comparison (different grid conventions!) ──")
    compare_eps(eps_bin_mono, eps_mpb_mono, 'Binary', 'MPB')
    compare_eps(eps_smo_mono, eps_mpb_mono, 'Smoothed', 'MPB')
    compare_eps(eps_bin_mono, eps_smo_mono, 'Binary', 'Smoothed')

    # Bulk statistics that don't depend on pixel alignment
    print("\n  ── Bulk Statistics (grid-convention independent) ──")
    for label, eps in [('Binary', eps_bin_mono), ('Smoothed', eps_smo_mono), ('MPB', eps_mpb_mono)]:
        rod_pixels = (eps > 0.5 * (EPS_BG + EPS_ROD)).sum()
        print(f"    {label:>10}: mean={eps.mean():.6f}, hmean={1/(1/eps).mean():.6f}, "
              f"rod_pixels={rod_pixels}, fill={rod_pixels/eps.size:.4%}")

    mono_data = {
        'binary': eps_bin_mono,
        'smoothed': eps_smo_mono,
        'mpb': eps_mpb_mono,
        'B': mono_info['B_super'],
    }

    # ── Part 2: Small supercell ──
    print("\n" + "=" * 70)
    theta_sc = math.degrees(commensurate_twist_angle('honeycomb', SC_M, SC_N))
    N_cells_sc = SC_M**2 + SC_M*SC_N + SC_N**2
    print(f"SUPERCELL ({SC_M},{SC_N}): θ={theta_sc:.2f}°, N_cells={N_cells_sc}, res={SC_RES}")
    print("=" * 70)

    # MPB first
    t0 = time.time()
    eps_mpb_sc = get_mpb_epsilon_supercell(SC_M, SC_N, SC_RES)
    # Shift MPB origin from cell center to cell corner (same fix as monolayer)
    Nx_mpb, Ny_mpb = eps_mpb_sc.shape
    eps_mpb_sc = np.roll(eps_mpb_sc, (Nx_mpb // 2, Ny_mpb // 2), axis=(0, 1))
    print(f"  MPB epsilon: {time.time()-t0:.1f}s, shape={eps_mpb_sc.shape}")

    # FDFD at MPB's actual grid size — using the SAME coincidence supercell
    Nx_sc = eps_mpb_sc.shape[0]
    t0 = time.time()

    # Build binary ε on coincidence supercell using nearest-rod approach
    B_coinc = build_coincidence_supercell(SC_M, SC_N, a=1.0)
    theta_sc_rad = commensurate_twist_angle('honeycomb', SC_M, SC_N)
    R_sc = rotation_matrix_2d(theta_sc_rad)
    B_mono_sc = build_monolayer_basis('honeycomb', 1.0)
    a1_sc, a2_sc = B_mono_sc[:, 0], B_mono_sc[:, 1]
    sublattice_sc = get_sublattice_positions('honeycomb')

    C1 = B_coinc[:, 0]
    C2 = B_coinc[:, 1]

    s1 = np.arange(Nx_sc) / Nx_sc
    s2 = np.arange(Nx_sc) / Nx_sc
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X_sc = S1 * C1[0] + S2 * C2[0]
    Y_sc = S1 * C1[1] + S2 * C2[1]
    XY_sc = np.stack([X_sc, Y_sc], axis=0)

    eps_bin_sc = np.full((Nx_sc, Nx_sc), EPS_BG, dtype=np.float64)
    r_rod = R_OVER_A * 1.0

    for B_layer in [B_mono_sc, np.column_stack([R_sc @ a1_sc, R_sc @ a2_sc])]:
        B_layer_inv = np.linalg.inv(B_layer)
        for sub_pos in sublattice_sc:
            offset = B_layer @ sub_pos
            shifted = XY_sc - offset[:, None, None]
            frac = np.einsum('ij,jkl->ikl', B_layer_inv, shifted)
            f_near = frac - np.round(frac)
            disp = np.einsum('ij,jkl->ikl', B_layer, f_near)
            dist_sq = disp[0]**2 + disp[1]**2
            eps_bin_sc[dist_sq < r_rod**2] = EPS_ROD

    # Build sc_info dict for smoothing and plotting
    g11 = np.dot(C1, C1)
    g12 = np.dot(C1, C2)
    g22 = np.dot(C2, C2)
    area_unit = abs(np.cross(a1_sc, a2_sc))
    area_super = abs(np.cross(C1, C2))
    sc_info = {
        'lattice_type': 'honeycomb',
        'm': SC_M, 'n': SC_N, 'a': 1.0,
        'theta_deg': math.degrees(theta_sc_rad),
        'theta_rad': theta_sc_rad,
        'r_over_a': R_OVER_A,
        'eps_rod': EPS_ROD, 'eps_bg': EPS_BG,
        'N_cells': N_cells_sc,
        'Nx': Nx_sc, 'Ny': Nx_sc,
        'B_super': B_coinc,
        'B_mono': B_mono_sc,
        'metric': np.array([[g11, g12], [g12, g22]]),
        'det_g': g11 * g22 - g12**2,
        'L1': C1, 'L2': C2,
        'area_super': area_super, 'area_unit': area_unit,
    }

    from T_direct_validation.subpixel_smoothing import build_smoothed_eps_supercell
    eps_smo_sc, smooth_info = build_smoothed_eps_supercell(
        eps_bin_sc, sc_info, n_sub=16,
        eps_rod=EPS_ROD, eps_bg=EPS_BG,
    )
    print(f"  FDFD binary + smoothed: {time.time()-t0:.1f}s, shape={eps_bin_sc.shape}")
    print(f"  Smoothed pixels: {smooth_info['n_smoothed']}")

    if eps_mpb_sc.shape != eps_bin_sc.shape:
        print(f"  WARNING: Shape mismatch! FDFD={eps_bin_sc.shape}, MPB={eps_mpb_sc.shape}")
        Nx = min(eps_bin_sc.shape[0], eps_mpb_sc.shape[0])
        Ny = min(eps_bin_sc.shape[1], eps_mpb_sc.shape[1])
        eps_bin_sc = eps_bin_sc[:Nx, :Ny]
        eps_smo_sc = eps_smo_sc[:Nx, :Ny]
        eps_mpb_sc = eps_mpb_sc[:Nx, :Ny]

    print("\n  ── Pixelwise Comparison ──")
    compare_eps(eps_bin_sc, eps_mpb_sc, 'Binary', 'MPB')
    compare_eps(eps_smo_sc, eps_mpb_sc, 'Smoothed', 'MPB')
    compare_eps(eps_bin_sc, eps_smo_sc, 'Binary', 'Smoothed')

    print("\n  ── Bulk Statistics ──")
    for label, eps in [('Binary', eps_bin_sc), ('Smoothed', eps_smo_sc), ('MPB', eps_mpb_sc)]:
        rod_pixels = (eps > 0.5 * (EPS_BG + EPS_ROD)).sum()
        print(f"    {label:>10}: mean={eps.mean():.6f}, hmean={1/(1/eps).mean():.6f}, "
              f"rod_pixels={rod_pixels}, fill={rod_pixels/eps.size:.4%}")

    sc_data = {
        'binary': eps_bin_sc,
        'smoothed': eps_smo_sc,
        'mpb': eps_mpb_sc,
        'B': sc_info['B_super'],
    }

    sc_label = f'Supercell ({SC_M},{SC_N})'

    # ── Plots ──
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_3x2_comparison(mono_data, sc_data, sc_label, 'fig_d1b_eps_comparison.png')
    plot_difference_maps(mono_data, sc_data, sc_label, 'fig_d1b_eps_diffs.png')
    plot_1d_slices(mono_data, sc_data, sc_label, 'fig_d1b_eps_slices.png')
    plot_zoomed_comparison(mono_data, 'Monolayer', 'fig_d1b_mono_zoom.png', zoom_frac=(0.2, 0.5))
    plot_zoomed_comparison(sc_data, sc_label, 'fig_d1b_sc_zoom.png', zoom_frac=(0.3, 0.5))

    print("\nDone.")


if __name__ == '__main__':
    main()
