#!/usr/bin/env python3
"""
Monolayer + supercell comparison: MPB vs FDFD for the canonical square lattice.

Parameters from canonical_lattice_configs.json:
  Square rods in air:  r/a=0.2, ε_rod=8.9, ε_bg=1.0
  Polarization: TM (E_z modes)

Step 1 (this run):
  - Monolayer band structure along Γ→X→M→Γ, MPB vs FDFD
  - Save band diagram for visual inspection

Step 2 (after choosing target):
  - 10° supercell (m,n)=(11,1), N_cells=122
  - MPB vs FDFD at Γ (where the chosen high-symmetry point folds)

Usage:
    python canonical_mpb_vs_fdfd.py                # monolayer comparison
    python canonical_mpb_vs_fdfd.py --supercell    # supercell comparison (after target chosen)
"""

import sys, os

# Set threading env vars BEFORE any numerical imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(THESIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# ═══════════════════════════════════════════════════
#  Canonical square lattice parameters
# ═══════════════════════════════════════════════════

A        = 1.0
R_OVER_A = 0.2
EPS_ROD  = 8.9       # from canonical_lattice_configs.json
EPS_BG   = 1.0
N_BANDS  = 8
N_K      = 40        # k-points per segment (dense for smooth plot)

# Resolution
RES_MPB  = 128
RES_FDFD = 128

# Supercell indices (for step 2)
M_IDX, N_IDX = 11, 1  # θ ≈ 10.39°, N_cells = 122


# ═══════════════════════════════════════════════════
#  1. MPB monolayer band structure
# ═══════════════════════════════════════════════════

def run_mpb_monolayer(n_k=N_K, resolution=RES_MPB):
    """TM band structure along Γ→X→M→Γ for square monolayer."""
    import meep as mp
    from meep import mpb

    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    geometry = [mp.Cylinder(
        radius=R_OVER_A,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=EPS_ROD),
    )]

    k_points = mp.interpolate(n_k, [
        mp.Vector3(0, 0, 0),       # Γ
        mp.Vector3(0.5, 0, 0),     # X
        mp.Vector3(0.5, 0.5, 0),   # M
        mp.Vector3(0, 0, 0),       # Γ
    ])

    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=N_BANDS,
        resolution=resolution,
        k_points=k_points,
    )

    # Suppress MPB stdout
    mp.verbosity(0)
    fd = os.open(os.devnull, os.O_WRONLY)
    o1, o2 = os.dup(1), os.dup(2)
    os.dup2(fd, 1); os.dup2(fd, 2)
    ms.run_tm()
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(fd); os.close(o1); os.close(o2)

    freqs = np.array(ms.all_freqs)  # (n_k_total, N_BANDS) in c/a

    # Build Cartesian k-path:  b1=2π(1,0), b2=2π(0,1) for square
    k_cart = np.array([[kp.x, kp.y] for kp in ms.k_points]) * 2 * np.pi
    k_dist = np.zeros(len(k_cart))
    for i in range(1, len(k_cart)):
        k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart[i] - k_cart[i-1])

    return freqs, k_dist, k_cart


# ═══════════════════════════════════════════════════
#  2. FDFD monolayer band structure
# ═══════════════════════════════════════════════════

def build_monolayer_eps(Nx):
    """Binary epsilon grid for square monolayer unit cell."""
    s = np.arange(Nx) / Nx
    S1, S2 = np.meshgrid(s, s, indexing='ij')

    X = S1 * A
    Y = S2 * A

    # Rod at origin, periodic images via nearest-lattice-point
    dx = X - np.round(X / A) * A
    dy = Y - np.round(Y / A) * A
    dist_sq = dx**2 + dy**2

    eps = np.where(dist_sq < (R_OVER_A * A)**2, EPS_ROD, EPS_BG)

    info = {
        'B_super': np.array([[A, 0.0], [0.0, A]]),
        'L1': np.array([A, 0.0]),
        'L2': np.array([0.0, A]),
    }
    return eps, info


def run_fdfd_monolayer(k_cart, resolution=RES_FDFD):
    """TM eigenvalues at each k-point via FDFD."""
    from T_direct_validation.fdfd_solver import build_fdfd_operator
    from scipy.sparse.linalg import eigsh

    eps, info = build_monolayer_eps(resolution)
    n_k = len(k_cart)
    freqs = np.zeros((n_k, N_BANDS))

    print(f"  FDFD: {n_k} k-points, res={resolution}, DOF={resolution**2}")

    for iq, q in enumerate(k_cart):
        L_op = build_fdfd_operator(eps, info, q_vec=q, polarization='tm')

        evals, _ = eigsh(L_op, k=N_BANDS, sigma=0.01, which='LM',
                         maxiter=5000, tol=1e-10)
        evals = np.sort(evals)
        # eigenvalue λ = (ω·a/c)² · (2π)² (due to fractional coords)
        # freq [c/a] = sqrt(λ) / (2π)
        omega = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        freqs[iq] = omega

        if iq % 20 == 0:
            print(f"    k {iq+1}/{n_k}: ω₁={omega[0]:.5f}, ω_{N_BANDS}={omega[-1]:.5f}")

    return freqs


# ═══════════════════════════════════════════════════
#  3. Plotting
# ═══════════════════════════════════════════════════

def plot_monolayer(freqs_mpb, freqs_fdfd, k_dist):
    """Band structure + difference plots for monolayer comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # High-symmetry point indices
    # mp.interpolate(n_k, [A,B,C,D]) puts n_k points between each pair
    # Segments: Γ→X (n_k+2 points incl endpoints), X→M, M→Γ
    # Actually mp.interpolate gives n_k points between each consecutive pair
    # Total = 3*(n_k) + 4 = 3*n_k + 4 ... let me just find them
    n_total = len(k_dist)
    # Find points closest to Γ(0,0), X(π,0), M(π,π)
    # k_cart is available from caller; use k_dist breaks
    # Easier: look for jumps in k_dist derivative
    seg_size = N_K + 1  # mp.interpolate(N, [A,B]) gives N+2 points? No.
    # mp.interpolate(n, [A,B,C,D]) -> [A, n between A-B, B, n between B-C, C, n between C-D, D]
    # = 1 + n + 1 + n + 1 + n + 1 = 3n + 4
    hs_indices = [0, N_K + 1, 2*(N_K + 1), n_total - 1]
    hs_labels = ['Γ', 'X', 'M', 'Γ']

    # --- Band structure overlay ---
    ax = axes[0]
    for b in range(N_BANDS):
        ax.plot(k_dist, freqs_mpb[:, b], 'b-', lw=1.5,
                label='MPB' if b == 0 else None)
        ax.plot(k_dist, freqs_fdfd[:, b], 'r--', lw=1.2,
                label='FDFD' if b == 0 else None)

    for idx in hs_indices:
        if idx < len(k_dist):
            ax.axvline(k_dist[idx], color='gray', lw=0.5)
    ax.set_xticks([k_dist[min(i, n_total-1)] for i in hs_indices])
    ax.set_xticklabels(hs_labels)
    ax.set_ylabel(r'$\omega a / 2\pi c$')
    ax.set_title('Band Structure: MPB vs FDFD')
    ax.legend(loc='upper left')
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(bottom=0)

    # --- Absolute difference ---
    ax = axes[1]
    diff = freqs_fdfd - freqs_mpb
    for b in range(N_BANDS):
        ax.plot(k_dist, diff[:, b], lw=1, label=f'band {b}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks([k_dist[min(i, n_total-1)] for i in hs_indices])
    ax.set_xticklabels(hs_labels)
    ax.set_ylabel(r'$\Delta\omega$')
    ax.set_title('Absolute Difference')
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.legend(fontsize=7, ncol=2)

    # --- Relative difference ---
    ax = axes[2]
    for b in range(N_BANDS):
        rel = np.abs(diff[:, b]) / np.maximum(freqs_mpb[:, b], 1e-10) * 100
        ax.plot(k_dist, rel, lw=1, label=f'band {b}')
    ax.set_xticks([k_dist[min(i, n_total-1)] for i in hs_indices])
    ax.set_xticklabels(hs_labels)
    ax.set_ylabel(r'Relative difference (%)')
    ax.set_title('Relative Difference')
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f'Square Monolayer TM — MPB(res={RES_MPB}) vs FDFD(res={RES_FDFD})\n'
        f'r/a={R_OVER_A}, ε_rod={EPS_ROD}, ε_bg={EPS_BG}',
        fontsize=12)
    plt.tight_layout()
    out = SCRIPT_DIR / 'fig_canonical_monolayer_comparison.png'
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supercell', action='store_true',
                        help='Run supercell comparison (requires target freq)')
    args = parser.parse_args()

    if args.supercell:
        print("Supercell comparison not yet configured — "
              "run monolayer first, then choose target.")
        return

    t0 = time.time()
    print("=" * 60)
    print("  Canonical Square Lattice Monolayer: MPB vs FDFD (TM)")
    print(f"  r/a={R_OVER_A}, ε_rod={EPS_ROD}, ε_bg={EPS_BG}")
    print(f"  {N_BANDS} bands, {N_K} k-points/segment")
    print(f"  MPB res={RES_MPB}, FDFD res={RES_FDFD}")
    print("=" * 60)

    # 1. MPB
    print("\n1. Running MPB ...")
    t1 = time.time()
    freqs_mpb, k_dist, k_cart = run_mpb_monolayer()
    print(f"   Done in {time.time()-t1:.1f}s  ({len(freqs_mpb)} k-points)")

    # 2. FDFD
    print("\n2. Running FDFD ...")
    t2 = time.time()
    freqs_fdfd = run_fdfd_monolayer(k_cart)
    print(f"   Done in {time.time()-t2:.1f}s")

    # 3. Summary
    diff = freqs_fdfd - freqs_mpb
    print("\n3. Comparison:")
    print(f"   Max|Δω|  = {np.max(np.abs(diff)):.6f}")
    print(f"   Mean|Δω| = {np.mean(np.abs(diff)):.6f}")

    # Per-band stats
    for b in range(N_BANDS):
        mask = freqs_mpb[:, b] > 0.01
        if mask.any():
            rel = np.abs(diff[mask, b]) / freqs_mpb[mask, b]
            print(f"   Band {b}: max_rel={rel.max():.4%},  mean_rel={rel.mean():.4%}")

    # Print frequencies at high-symmetry points
    targets = {
        'Γ': np.array([0.0, 0.0]),
        'X': np.array([np.pi, 0.0]),
        'M': np.array([np.pi, np.pi]),
    }
    for label, k_target in targets.items():
        idx = np.argmin(np.linalg.norm(k_cart - k_target, axis=1))
        print(f"\n   At {label} (idx={idx}):")
        print(f"   {'Band':>6s}  {'MPB':>10s}  {'FDFD':>10s}  {'Δω':>10s}  {'rel%':>8s}")
        for b in range(N_BANDS):
            d = freqs_fdfd[idx, b] - freqs_mpb[idx, b]
            r = abs(d) / max(freqs_mpb[idx, b], 1e-10) * 100
            print(f"   {b:6d}  {freqs_mpb[idx,b]:10.6f}  {freqs_fdfd[idx,b]:10.6f}  "
                  f"{d:+10.6f}  {r:8.4f}")

    # 4. Plot
    print("\n4. Plotting ...")
    plot_monolayer(freqs_mpb, freqs_fdfd, k_dist)

    # 5. Save data
    out = SCRIPT_DIR / 'canonical_monolayer_data.npz'
    np.savez(out,
             freqs_mpb=freqs_mpb,
             freqs_fdfd=freqs_fdfd,
             k_dist=k_dist,
             k_cart=k_cart,
             eps_rod=EPS_ROD,
             eps_bg=EPS_BG,
             r_over_a=R_OVER_A)
    print(f"  Saved: {out}")
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
