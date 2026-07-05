#!/usr/bin/env python3
"""
Phase A: MPB Monolayer Resolution Convergence
==============================================

Tests how MPB band frequencies converge with resolution for the monolayer
honeycomb photonic crystal at the K-point. This determines the minimum
mpb_resolution needed for the envelope approximation pipeline.

System: Honeycomb (triangular + 2-atom basis), ε_rod=11.56, ε_bg=1.0, r/a=0.2
Polarization: TM
k-point: K = (2/3, 1/3) in reciprocal lattice coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
import meep.mpb as mpb
import meep as mp
import math
import os
import sys
import json
from datetime import datetime

# ---- System parameters (matching thesis_honeycomb_K_b1.yaml) ----
LATTICE_TYPE = 'honeycomb'
R_OVER_A = 0.2
EPS_ROD = 11.56
EPS_BG = 1.0
POLARIZATION = 'TM'
K_POINT = (2/3, 1/3)  # K-point in reciprocal lattice coords

# Resolutions to sweep
RESOLUTIONS = [16, 32, 48, 64, 96, 128, 192, 256]

# Number of bands to compute (enough to cover subspace + neighbors)
NUM_BANDS = 8

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_mpb_monolayer(resolution, num_bands=NUM_BANDS):
    """
    Run MPB on a single monolayer honeycomb unit cell at the K-point.
    
    Returns array of band frequencies.
    """
    # Hexagonal lattice
    basis1 = mp.Vector3(1, 0, 0)
    basis2 = mp.Vector3(0.5, math.sqrt(3)/2, 0)
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0), basis1=basis1, basis2=basis2)
    
    # Honeycomb: two-atom basis at (0,0) and (1/3, 1/3)
    r = R_OVER_A
    geometry = [
        mp.Cylinder(radius=r, center=mp.Vector3(0, 0, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
        mp.Cylinder(radius=r, center=mp.Vector3(1/3, 1/3, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
    ]
    
    return _run_mpb(geometry, lattice, resolution, num_bands)


def run_mpb_bilayer(resolution, delta_frac, num_bands=NUM_BANDS):
    """
    Run MPB on a bilayer honeycomb unit cell at the K-point with stacking shift delta_frac.
    
    Returns array of band frequencies.
    """
    basis1 = mp.Vector3(1, 0, 0)
    basis2 = mp.Vector3(0.5, math.sqrt(3)/2, 0)
    lattice = mp.Lattice(size=mp.Vector3(1, 1, 0), basis1=basis1, basis2=basis2)
    
    r = R_OVER_A
    geometry = [
        # Layer 1
        mp.Cylinder(radius=r, center=mp.Vector3(0, 0, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
        mp.Cylinder(radius=r, center=mp.Vector3(1/3, 1/3, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
        # Layer 2 (shifted)
        mp.Cylinder(radius=r, center=mp.Vector3(delta_frac[0], delta_frac[1], 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
        mp.Cylinder(radius=r, center=mp.Vector3(delta_frac[0] + 1/3, delta_frac[1] + 1/3, 0),
                    material=mp.Medium(epsilon=EPS_ROD)),
    ]
    
    return _run_mpb(geometry, lattice, resolution, num_bands)


def _run_mpb(geometry, lattice, resolution, num_bands):
    """Run MPB with the given geometry, returning band frequencies at K."""
    ms = mpb.ModeSolver(
        geometry=geometry,
        geometry_lattice=lattice,
        default_material=mp.Medium(epsilon=EPS_BG),
        num_bands=num_bands,
        resolution=resolution,
    )
    
    ms.k_points = [mp.Vector3(K_POINT[0], K_POINT[1], 0)]
    
    # Suppress output
    mp.verbosity(0)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        if POLARIZATION == 'TM':
            ms.run_tm()
        else:
            ms.run_te()
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(devnull_fd)
        os.close(old_stdout)
        os.close(old_stderr)
    
    freqs = np.array(ms.all_freqs[-1])
    return freqs


def main():
    print("=" * 70)
    print("Phase A: MPB Monolayer Resolution Convergence")
    print("=" * 70)
    print(f"System: honeycomb, ε_rod={EPS_ROD}, ε_bg={EPS_BG}, r/a={R_OVER_A}")
    print(f"K-point: ({K_POINT[0]:.4f}, {K_POINT[1]:.4f})")
    print(f"Polarization: {POLARIZATION}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Bands: 1–{NUM_BANDS}")
    print()
    
    # Collect data
    all_freqs = {}
    for res in RESOLUTIONS:
        print(f"  res={res:4d} ... ", end="", flush=True)
        freqs = run_mpb_monolayer(res)
        all_freqs[res] = freqs
        print(f"ω = [{', '.join(f'{f:.8f}' for f in freqs[:6])}]")
    
    print()
    
    # Reference: highest resolution
    ref_res = max(RESOLUTIONS)
    ref_freqs = all_freqs[ref_res]
    
    # Compute relative errors vs reference
    print(f"Relative errors vs res={ref_res}:")
    print(f"{'res':>6s}  {'Band 1':>12s}  {'Band 2':>12s}  {'Band 3':>12s}  {'Band 4':>12s}  {'Band 5':>12s}  {'Band 6':>12s}")
    
    rel_errors = {}
    for res in RESOLUTIONS:
        freqs = all_freqs[res]
        rel_err = np.abs(freqs - ref_freqs) / ref_freqs
        rel_errors[res] = rel_err
        print(f"{res:6d}  " + "  ".join(f"{e:12.2e}" for e in rel_err[:6]))
    
    # Dirac bands (1 & 2) — the ones we care about
    print()
    print("Dirac cone bands (1–2) — the subspace for EA:")
    print(f"{'res':>6s}  {'ω₁':>12s}  {'ω₂':>12s}  {'|Δω₁|/ω₁':>12s}  {'|Δω₂|/ω₂':>12s}  {'|ω₁-ω₂|':>12s}")
    for res in RESOLUTIONS:
        f = all_freqs[res]
        re = rel_errors[res]
        print(f"{res:6d}  {f[0]:12.8f}  {f[1]:12.8f}  {re[0]:12.2e}  {re[1]:12.2e}  {abs(f[0]-f[1]):12.2e}")
    
    # Convergence between consecutive resolutions
    print()
    print("Convergence between consecutive resolutions (Dirac bands, max rel error):")
    sorted_res = sorted(RESOLUTIONS)
    for i in range(1, len(sorted_res)):
        r_prev, r_curr = sorted_res[i-1], sorted_res[i]
        diff = np.abs(all_freqs[r_curr][:2] - all_freqs[r_prev][:2]) / all_freqs[r_curr][:2]
        print(f"  {r_prev:4d} → {r_curr:4d}: max|Δω|/ω = {np.max(diff):.2e}")
    
    # Save data
    results = {
        'resolutions': RESOLUTIONS,
        'num_bands': NUM_BANDS,
        'K_point': list(K_POINT),
        'lattice_type': LATTICE_TYPE,
        'r_over_a': R_OVER_A,
        'eps_rod': EPS_ROD,
        'eps_bg': EPS_BG,
        'polarization': POLARIZATION,
        'frequencies': {str(res): all_freqs[res].tolist() for res in RESOLUTIONS},
        'relative_errors_vs_ref': {str(res): rel_errors[res].tolist() for res in RESOLUTIONS},
        'reference_resolution': ref_res,
        'timestamp': datetime.now().isoformat(),
    }
    json_path = os.path.join(OUTPUT_DIR, 'mpb_resolution_convergence.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to {json_path}")
    
    # ---- Plotting ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_BANDS))
    
    # Panel (a): Band frequencies vs resolution
    ax = axes[0]
    for b in range(min(6, NUM_BANDS)):
        freqs_b = [all_freqs[r][b] for r in sorted_res]
        ax.plot(sorted_res, freqs_b, 'o-', color=colors[b], label=f'Band {b+1}', markersize=5)
    ax.set_xlabel('MPB resolution')
    ax.set_ylabel('ω (c/a)')
    ax.set_title('(a) Band frequencies at K')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Relative error vs resolution (log scale)
    ax = axes[1]
    for b in range(min(6, NUM_BANDS)):
        errs_b = [rel_errors[r][b] for r in sorted_res[:-1]]  # Exclude reference
        ax.semilogy(sorted_res[:-1], errs_b, 'o-', color=colors[b], label=f'Band {b+1}', markersize=5)
    ax.set_xlabel('MPB resolution')
    ax.set_ylabel(f'|Δω|/ω (vs res={ref_res})')
    ax.set_title('(b) Relative error vs reference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(1e-6, color='gray', ls=':', lw=1, label='10⁻⁶')
    
    # Panel (c): Dirac bands zoom — convergence
    ax = axes[2]
    dirac_errs = []
    for r in sorted_res[:-1]:
        max_err = max(rel_errors[r][0], rel_errors[r][1])
        dirac_errs.append(max_err)
    ax.semilogy(sorted_res[:-1], dirac_errs, 'ko-', markersize=7, label='Max(Band 1, 2)')
    ax.set_xlabel('MPB resolution')
    ax.set_ylabel(f'Max relative error (vs res={ref_res})')
    ax.set_title('(c) Dirac band convergence')
    ax.grid(True, alpha=0.3)
    ax.axhline(1e-6, color='red', ls='--', lw=1, alpha=0.7, label='10⁻⁶ target')
    ax.axhline(1e-8, color='blue', ls='--', lw=1, alpha=0.7, label='10⁻⁸')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTPUT_DIR, f'fig_mpb_resolution_convergence.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to {OUTPUT_DIR}/fig_mpb_resolution_convergence.{{png,pdf}}")
    
    plt.close()
    
    # ---- Convergence order analysis ----
    print()
    print("Convergence order (Richardson extrapolation, consecutive pairs):")
    for i in range(1, len(sorted_res) - 1):
        r1, r2, r3 = sorted_res[i-1], sorted_res[i], sorted_res[i+1]
        # Use band 2 (often less symmetric, shows clearer convergence)
        e1 = abs(all_freqs[r1][1] - ref_freqs[1])
        e2 = abs(all_freqs[r2][1] - ref_freqs[1])
        if e1 > 0 and e2 > 0:
            h_ratio = r1 / r2
            order = np.log(e1 / e2) / np.log(r2 / r1)
            print(f"  res={r1:4d}→{r2:4d}: p ≈ {order:.2f}")

    # ---- Key insight: bilayer frequency DIFFERENCES ----
    # The EA uses Λ(s) = ω_bilayer(s) − ω_ref, not absolute ω.
    # Run a bilayer config to test if systematic MPB error cancels.
    print()
    print("=" * 70)
    print("BILAYER FREQUENCY DIFFERENCES (what EA actually uses)")
    print("=" * 70)
    print("Testing: ω_bilayer(δ=0.5,0.5) − ω_monolayer at each resolution")
    print("If systematic error cancels, Δω converges faster than individual ω values")
    print()
    
    delta_shift = [0.5, 0.5]  # Large shift to see effect
    bilayer_freqs = {}
    for res in sorted_res:
        print(f"  Bilayer res={res:4d} ... ", end="", flush=True)
        freqs_bl = run_mpb_bilayer(res, delta_shift)
        bilayer_freqs[res] = freqs_bl
        diff = freqs_bl[:2] - all_freqs[res][:2]
        print(f"Δω = [{diff[0]:.8f}, {diff[1]:.8f}]")
    
    # Convergence of differences
    ref_diff = bilayer_freqs[ref_res][:2] - all_freqs[ref_res][:2]
    print()
    print(f"Convergence of Δω (bilayer − monolayer), Dirac bands, vs res={ref_res}:")
    print(f"{'res':>6s}  {'|δ(Δω₁)|':>12s}  {'|δ(Δω₂)|':>12s}  {'max rel(δΔω)':>14s}  {'vs abs err':>12s}")
    for res in sorted_res[:-1]:
        diff_here = bilayer_freqs[res][:2] - all_freqs[res][:2]
        delta_delta = np.abs(diff_here - ref_diff)
        max_rel = np.max(delta_delta / np.abs(ref_diff)) if np.all(ref_diff != 0) else np.max(delta_delta)
        abs_err = max(rel_errors[res][0], rel_errors[res][1])
        ratio = max_rel / abs_err if abs_err > 0 else float('inf')
        print(f"{res:6d}  {delta_delta[0]:12.2e}  {delta_delta[1]:12.2e}  {max_rel:14.2e}  ratio={ratio:.2f}×")
    
    # ---- Recommendation ----
    print()
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("Absolute frequency convergence (vs res=256):")
    for res in sorted_res[:-1]:
        max_dirac_err = max(rel_errors[res][0], rel_errors[res][1])
        status = "✓ CONVERGED" if max_dirac_err < 1e-6 else "  not converged" if max_dirac_err > 1e-4 else "  marginal"
        print(f"  res={res:4d}: max Dirac band error = {max_dirac_err:.2e}  {status}")
    
    # Find minimum resolution with <1e-6 Dirac error
    for res in sorted_res:
        if res == ref_res:
            continue
        max_dirac_err = max(rel_errors[res][0], rel_errors[res][1])
        if max_dirac_err < 1e-6:
            print(f"\n→ Minimum resolution for <10⁻⁶ Dirac band accuracy: res={res}")
            break
    else:
        print(f"\n→ No resolution below {ref_res} achieves <10⁻⁶ absolute accuracy")
    
    print()
    print("Key observation: The EA uses frequency DIFFERENCES (Λ = ω_bilayer − ω_ref),")
    print("where systematic MPB discretization error cancels. The difference convergence")
    print("rate determines the effective accuracy requirement.")


if __name__ == '__main__':
    main()
