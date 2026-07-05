#!/usr/bin/env python3
"""
Monolayer band comparison: MPB vs FDFD for the square lattice.

Computes TM band structure along Γ→X→M→Γ for a square lattice unit cell
(single rod, r/a=0.2, ε_rod=11.56, ε_bg=1.0) using both MPB (plane-wave
expansion) and FDFD (real-space finite differences), then compares.

Usage:
    python square_monolayer_comparison.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ── Paths ──
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# ── Parameters ──
A        = 1.0
R_OVER_A = 0.2
EPS_ROD  = 11.56
EPS_BG   = 1.0
N_BANDS  = 8
RES_MPB  = 128       # MPB resolution (plane-wave grid)
RES_FDFD = 128       # FDFD grid per unit cell
N_K      = 30        # k-points per segment

# ─────────────────────────────────────────────────
#  1. MPB band structure
# ─────────────────────────────────────────────────

def run_mpb_bands(n_k=N_K, resolution=RES_MPB):
    """Compute TM band structure along Γ→X→M→Γ with MPB."""
    import meep as mp
    from meep import mpb

    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0))
    geometry = [mp.Cylinder(
        radius=R_OVER_A,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=EPS_ROD),
    )]

    # High-symmetry path: Γ(0,0) → X(0.5,0) → M(0.5,0.5) → Γ(0,0)
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

    # Extract frequencies: shape (n_k_total, N_BANDS)
    freqs = np.array(ms.all_freqs)  # in c/a units (monolayer: |a1|=a=1)

    # Build k-distance for plotting
    k_cart = []
    for kp in ms.k_points:
        # Reciprocal lattice for square: b1=2π(1,0), b2=2π(0,1)
        # k_cart = kf1*b1 + kf2*b2 = 2π*(kf1, kf2)
        k_cart.append([kp.x, kp.y])
    k_cart = np.array(k_cart) * 2 * np.pi

    k_dist = np.zeros(len(k_cart))
    for i in range(1, len(k_cart)):
        k_dist[i] = k_dist[i-1] + np.linalg.norm(k_cart[i] - k_cart[i-1])

    return freqs, k_dist, k_cart


# ─────────────────────────────────────────────────
#  2. FDFD band structure
# ─────────────────────────────────────────────────

def build_monolayer_eps(Nx, Ny):
    """Build binary epsilon grid for square monolayer unit cell."""
    s1 = np.arange(Nx) / Nx
    s2 = np.arange(Ny) / Ny
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')

    # Physical coords: r = s1*a1 + s2*a2 = (s1, s2) for square
    X = S1 * A
    Y = S2 * A

    # Rod at origin, periodic
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


def run_fdfd_bands(k_cart, resolution=RES_FDFD, sigma=None):
    """Compute TM eigenvalues at each k-point via FDFD.

    Args:
        k_cart: (n_k, 2) Cartesian k-points
        resolution: FDFD grid per unit cell
        sigma: shift for eigsh. If None, auto-set per k-point.

    Returns:
        freqs_fdfd: (n_k, N_BANDS) in ωa/2πc units
    """
    from fdfd_solver import build_fdfd_operator
    from scipy.sparse.linalg import eigsh

    eps, info = build_monolayer_eps(resolution, resolution)
    n_k = len(k_cart)
    N_dof = resolution * resolution
    freqs_fdfd = np.zeros((n_k, N_BANDS))

    print(f"  FDFD: {n_k} k-points, resolution={resolution}, DOF={N_dof}")

    for iq, q in enumerate(k_cart):
        L_op = build_fdfd_operator(eps, info, q_vec=q, polarization='tm')

        # For lowest bands, use small sigma
        if sigma is None:
            sig = 0.01
        else:
            sig = sigma

        evals, _ = eigsh(L_op, k=N_BANDS, sigma=sig, which='LM',
                         maxiter=5000, tol=1e-10)
        evals = np.sort(evals)
        # ω²= eigenvalue (in (2π/a)² units) → ω = sqrt(eval) / (2π)
        # Actually: eigenvalue = (ω * 2π / (2πc/a))² = (ωa/c)² * (2π)²
        # Wait, let's be careful.
        # The FDFD operator eigenvalue is λ = (ω/c)² in units where
        # the lattice constant a is used in the grid spacing.
        # With a=1 and fractional coords, λ has units of (2π/a)².
        # freq = ω a/(2πc) = sqrt(λ)/(2π)
        omega = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
        freqs_fdfd[iq] = omega

        if iq % 10 == 0:
            print(f"    k-point {iq+1}/{n_k}: "
                  f"q=({q[0]:.3f},{q[1]:.3f}), "
                  f"ω₁={omega[0]:.5f}, ω_{N_BANDS}={omega[-1]:.5f}")

    return freqs_fdfd


# ─────────────────────────────────────────────────
#  3. Plotting
# ─────────────────────────────────────────────────

def plot_comparison(freqs_mpb, freqs_fdfd, k_dist):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Band structure overlay ---
    ax = axes[0]
    for b in range(N_BANDS):
        ax.plot(k_dist, freqs_mpb[:, b], 'b-', lw=1.5,
                label='MPB' if b == 0 else None)
        ax.plot(k_dist, freqs_fdfd[:, b], 'r--', lw=1.2,
                label='FDFD' if b == 0 else None)

    # Mark high-symmetry points
    n_k_total = len(k_dist)
    n_seg = N_K + 1  # points per segment (including endpoints)
    hs_idx = [0, n_seg, 2*n_seg, n_k_total - 1]
    hs_labels = ['Γ', 'X', 'M', 'Γ']
    for idx in hs_idx:
        if idx < len(k_dist):
            ax.axvline(k_dist[idx], color='gray', lw=0.5, ls='-')
    ax.set_xticks([k_dist[i] for i in hs_idx if i < len(k_dist)])
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
    ax.set_xticks([k_dist[i] for i in hs_idx if i < len(k_dist)])
    ax.set_xticklabels(hs_labels)
    ax.set_ylabel(r'$\Delta\omega = \omega_\mathrm{FDFD} - \omega_\mathrm{MPB}$')
    ax.set_title('Absolute Difference')
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.legend(fontsize=7, ncol=2)

    # --- Relative difference ---
    ax = axes[2]
    for b in range(N_BANDS):
        rel = np.abs(diff[:, b]) / np.maximum(freqs_mpb[:, b], 1e-10) * 100
        ax.plot(k_dist, rel, lw=1, label=f'band {b}')
    ax.set_xticks([k_dist[i] for i in hs_idx if i < len(k_dist)])
    ax.set_xticklabels(hs_labels)
    ax.set_ylabel(r'$|\Delta\omega| / \omega_\mathrm{MPB}$ (%)')
    ax.set_title('Relative Difference (%)')
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f'Square Lattice Monolayer TM Bands — MPB(res={RES_MPB}) vs FDFD(res={RES_FDFD})',
                 fontsize=12)
    plt.tight_layout()
    out = SCRIPT_DIR / 'fig_monolayer_mpb_vs_fdfd.png'
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────────
#  4. Main
# ─────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  Square Lattice Monolayer: MPB vs FDFD")
    print(f"  r/a={R_OVER_A}, ε_rod={EPS_ROD}, ε_bg={EPS_BG}")
    print(f"  {N_BANDS} TM bands, {N_K} k-points/segment")
    print(f"  MPB res={RES_MPB}, FDFD res={RES_FDFD}")
    print("=" * 60)

    # 1. MPB
    print("\n1. Running MPB band structure...")
    t1 = time.time()
    freqs_mpb, k_dist, k_cart = run_mpb_bands()
    print(f"   MPB done in {time.time()-t1:.1f}s, {len(freqs_mpb)} k-points")

    # 2. FDFD
    print("\n2. Running FDFD band structure...")
    t2 = time.time()
    freqs_fdfd = run_fdfd_bands(k_cart)
    print(f"   FDFD done in {time.time()-t2:.1f}s")

    # 3. Summary stats
    diff = freqs_fdfd - freqs_mpb
    print(f"\n3. Comparison summary:")
    print(f"   Max |Δω|   = {np.max(np.abs(diff)):.6f}")
    print(f"   Mean |Δω|  = {np.mean(np.abs(diff)):.6f}")
    for b in range(N_BANDS):
        mask = freqs_mpb[:, b] > 0.01  # skip near-zero
        if mask.any():
            rel = np.abs(diff[mask, b]) / freqs_mpb[mask, b]
            print(f"   Band {b}: max_rel={rel.max():.4%}, mean_rel={rel.mean():.4%}")

    # Key frequencies at M point
    # M is at index n_k+1 in the interpolated path
    m_idx = N_K + 1  # end of first segment is X, end of second is M
    # Actually: Γ→X has n_k+1 points (indices 0..n_k),
    #           X→M has n_k+1 points (indices n_k..2*n_k),
    #           M→Γ has n_k+1 points (indices 2*n_k..3*n_k)
    # So M is at index 2*(n_k+1) - 1 = 2*n_k+1
    # But mp.interpolate adds n_k points BETWEEN each pair, so segment has n_k+2...
    # Let me just find the M point by looking at k_cart
    # M = (0.5,0.5) → k_cart = 2π(0.5,0.5) = (π, π)
    k_target = np.array([np.pi, np.pi])
    m_idx = np.argmin(np.linalg.norm(k_cart - k_target, axis=1))
    print(f"\n   At M-point (idx={m_idx}):")
    print(f"   {'Band':>6s}  {'MPB':>10s}  {'FDFD':>10s}  {'Δω':>10s}  {'rel%':>8s}")
    for b in range(N_BANDS):
        d = freqs_fdfd[m_idx, b] - freqs_mpb[m_idx, b]
        r = abs(d) / max(freqs_mpb[m_idx, b], 1e-10) * 100
        print(f"   {b:6d}  {freqs_mpb[m_idx,b]:10.6f}  {freqs_fdfd[m_idx,b]:10.6f}  {d:+10.6f}  {r:8.4f}")

    # 4. Plot
    print("\n4. Generating plots...")
    plot_comparison(freqs_mpb, freqs_fdfd, k_dist)

    # 5. Save data
    out_npz = SCRIPT_DIR / 'monolayer_mpb_vs_fdfd.npz'
    np.savez(out_npz,
             freqs_mpb=freqs_mpb,
             freqs_fdfd=freqs_fdfd,
             k_dist=k_dist,
             k_cart=k_cart,
             resolution_mpb=RES_MPB,
             resolution_fdfd=RES_FDFD,
             n_bands=N_BANDS,
             r_over_a=R_OVER_A,
             eps_rod=EPS_ROD,
             eps_bg=EPS_BG)
    print(f"  Saved data: {out_npz}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
