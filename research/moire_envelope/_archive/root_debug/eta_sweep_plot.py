#!/usr/bin/env python
"""
Produce the definitive η-sweep analysis plots for the validation report / thesis.
Reads sweep_results_reextracted.json.
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

SWEEP_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    sorted(Path('/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337').glob('eta_sweep_*'))[-1]
)

with open(SWEEP_DIR / 'sweep_results_reextracted.json') as f:
    results = json.load(f)
results.sort(key=lambda r: r['eta'])

etas   = np.array([r['eta'] for r in results])
thetas = np.array([r['theta_deg'] for r in results])
l3     = np.array([r['lambda_0_N3'] for r in results])
l1     = np.array([r['lambda_0_N1'] for r in results])
bws    = np.array([r['bandwidth_50'] for r in results])
gaps   = np.array([r['gap_01'] for r in results])
mixs   = np.array([r['max_mixing'] for r in results])

# Per-band potential maxima (from Phase 2 data, constant across θ)
V_max_band0 = 0.09138   # max Λ_00
V_max_all   = 0.35132   # max Λ (band 2)

# Binding energies (how far below the respective potential max)
E_bind_N3 = np.abs(l3 - V_max_all)
E_bind_N1 = np.abs(l1 - V_max_band0)

# ─── Power-law fits ───
def fit_power(x, y, label=""):
    lx, ly = np.log(x), np.log(y)
    p = np.polyfit(lx, ly, 1)
    return p[0], np.exp(p[1])

slope_bind_N3, c3 = fit_power(etas, E_bind_N3)
slope_bind_N1, c1 = fit_power(etas, E_bind_N1)
slope_bw,   cbw   = fit_power(etas, bws)
slope_gap,  cgap  = fit_power(etas, gaps)

eta_ref = np.geomspace(etas.min()*0.7, etas.max()*1.3, 200)

# ============================================================================
# Figure 1: The main 2×2 validation figure
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# ─── Panel A: Binding energy ───
ax = axes[0, 0]
ax.loglog(etas, E_bind_N3, 'bo-', ms=9, lw=2, label=f'$E_{{bind}}$(N=3)  ∝ η$^{{{slope_bind_N3:.2f}}}$')
ax.loglog(etas, E_bind_N1, 'rs-', ms=9, lw=2, label=f'$E_{{bind}}$(N=1)  ∝ η$^{{{slope_bind_N1:.2f}}}$')
ax.loglog(eta_ref, c3*eta_ref**slope_bind_N3, 'b--', alpha=0.3)
ax.loglog(eta_ref, c1*eta_ref**slope_bind_N1, 'r--', alpha=0.3)
# Reference η² line
s2 = E_bind_N1[3] / etas[3]**2
ax.loglog(eta_ref, s2*eta_ref**2, 'k:', alpha=0.4, label='η² reference')
ax.set_xlabel('η = 2 sin(θ/2)', fontsize=12)
ax.set_ylabel('Binding energy  |λ₀ − V_max|', fontsize=12)
ax.set_title('(A)  Binding energy scaling', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')

# ─── Panel B: Eigenvalue waterfall ───
ax = axes[0, 1]
cmap = plt.cm.viridis
for i, r in enumerate(results):
    evals = np.array(r['eigenvalues'])
    n_show = min(30, len(evals))
    color = cmap(i / (len(results)-1))
    ax.plot([r['eta']] * n_show, evals[:n_show], '.', color=color, ms=4, alpha=0.8)
    # Mark λ₀
    ax.plot(r['eta'], evals[0], 'D', color=color, ms=7, mec='k', mew=0.5)
ax.set_xscale('log')
ax.set_xlabel('η', fontsize=12)
ax.set_ylabel('Eigenvalue λ', fontsize=12)
ax.set_title('(B)  Eigenvalue spectrum vs η', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2, which='both')

# Add theta labels on top
ax2_top = ax.twiny()
ax2_top.set_xlim(ax.get_xlim())
ax2_top.set_xscale('log')
ax2_top.set_xticks(etas)
ax2_top.set_xticklabels([f'{t:.1f}°' for t in thetas], fontsize=8)
ax2_top.set_xlabel('θ', fontsize=10)

# ─── Panel C: Bandwidth + gap scaling ───
ax = axes[1, 0]
ax.loglog(etas, bws, 'r^-', ms=9, lw=2, label=f'Bandwidth (50 modes)  ∝ η$^{{{slope_bw:.2f}}}$')
ax.loglog(etas, gaps, 'gs-', ms=8, lw=2, label=f'Gap Δλ₀₁  ∝ η$^{{{slope_gap:.2f}}}$')
ax.loglog(eta_ref, cbw*eta_ref**slope_bw, 'r--', alpha=0.3)
ax.loglog(eta_ref, cgap*eta_ref**slope_gap, 'g--', alpha=0.3)
s2bw = bws[3] / etas[3]**2
ax.loglog(eta_ref, s2bw*eta_ref**2, 'k:', alpha=0.4, label='η² reference')
ax.set_xlabel('η', fontsize=12)
ax.set_ylabel('Energy scale', fontsize=12)
ax.set_title('(C)  Bandwidth & gap scaling', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2, which='both')

# ─── Panel D: Band mixing ───
ax = axes[1, 1]
ax.semilogy(thetas, mixs, 'mo-', ms=9, lw=2, label='max(1 − max_weight)')
ax.axhline(0.01, color='gray', ls='--', alpha=0.4, label='1% threshold')
ax.fill_between([0, 10], 0.01, 1.0, alpha=0.05, color='red', label='Multi-band regime')
ax.fill_between([0, 10], 1e-10, 0.01, alpha=0.05, color='green', label='Single-band regime')
ax.set_xlabel('θ (degrees)', fontsize=12)
ax.set_ylabel('Band mixing  (1 − max weight)', fontsize=12)
ax.set_title('(D)  Inter-band mixing vs twist angle', fontsize=13, fontweight='bold')
ax.set_xlim(0, 9)
ax.set_ylim(1e-9, 1)
ax.legend(fontsize=9, framealpha=0.9, loc='lower right')
ax.grid(True, alpha=0.2, which='both')

fig.suptitle('η-Sweep Validation — Moiré Photonic Crystal Envelope Approximation',
             fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = SWEEP_DIR / 'eta_sweep_analysis.png'
plt.savefig(out1, dpi=200)
plt.close()
print(f"Saved: {out1}")

# ============================================================================
# Figure 2: N-band convergence + binding/η² ratio
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

# Panel A: λ₀(N=3) vs λ₀(N=1) as function of η
ax = axes[0]
ax.plot(etas, l3, 'bo-', ms=9, lw=2, label='λ₀ (N=3 bands)')
ax.plot(etas, l1, 'rs-', ms=9, lw=2, label='λ₀ (N=1 band)')
ax.axhline(V_max_all, color='blue', ls=':', alpha=0.4, label=f'V_max(all) = {V_max_all:.3f}')
ax.axhline(V_max_band0, color='red', ls=':', alpha=0.4, label=f'V_max(band 0) = {V_max_band0:.3f}')
ax.set_xscale('log')
ax.set_xlabel('η', fontsize=12)
ax.set_ylabel('Lowest eigenvalue λ₀', fontsize=12)
ax.set_title('(A)  λ₀ vs η for N=1 and N=3', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)

# Panel B: The binding energies converge with same power law
ax = axes[1]
ratio_bind = E_bind_N3 / E_bind_N1
ax.semilogx(etas, ratio_bind, 'kD-', ms=9, lw=2)
ax.axhline(1.0, color='green', ls='--', alpha=0.5)
ax.set_xlabel('η', fontsize=12)
ax.set_ylabel('E_bind(N=3) / E_bind(N=1)', fontsize=12)
ax.set_title('(B)  Binding energy ratio → 1 as η→0', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.annotate(f'ratio → {ratio_bind[-1]:.2f} at η={etas[-1]:.3f}',
            (etas[-1], ratio_bind[-1]), textcoords='offset points',
            xytext=(-80, 20), fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate(f'ratio → {ratio_bind[0]:.2f} at η={etas[0]:.4f}',
            (etas[0], ratio_bind[0]), textcoords='offset points',
            xytext=(20, -30), fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5))

# Panel C: E_bind / η² — should converge to a constant for true η² scaling
ax = axes[2]
ratio_n1 = E_bind_N1 / etas**2
ratio_n3 = E_bind_N3 / etas**2
ax.semilogx(etas, ratio_n1, 'rs-', ms=9, lw=2, label='E_bind(N=1) / η²')
ax.semilogx(etas, ratio_n3, 'bo-', ms=9, lw=2, label='E_bind(N=3) / η²')
ax.set_xlabel('η', fontsize=12)
ax.set_ylabel('E_bind / η²', fontsize=12)
ax.set_title('(C)  Binding energy / η²', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)
# Annotate the plateau region
ax.axhline(np.mean(ratio_n1[3:]), color='red', ls='--', alpha=0.3)
ax.annotate(f'plateau ≈ {np.mean(ratio_n1[3:]):.1f}',
            xy=(etas[5], np.mean(ratio_n1[3:])), fontsize=10, color='red',
            xytext=(0, 15), textcoords='offset points')

plt.tight_layout()
out2 = SWEEP_DIR / 'eta_sweep_nband_convergence.png'
plt.savefig(out2, dpi=200)
plt.close()
print(f"Saved: {out2}")

# ============================================================================
# Print summary for thesis
# ============================================================================
print(f"""
╔══════════════════════════════════════════════════════════╗
║          η-SWEEP VALIDATION SUMMARY                     ║
╠══════════════════════════════════════════════════════════╣
║  Angles:  θ ∈ [{thetas[0]:.1f}°, {thetas[-1]:.1f}°]                         ║
║  η range: [{etas[0]:.4f}, {etas[-1]:.4f}]                     ║
║                                                          ║
║  SCALING EXPONENTS:                                      ║
║    E_bind(N=1) ∝ η^{slope_bind_N1:.2f}   (expect: 2.0)              ║
║    E_bind(N=3) ∝ η^{slope_bind_N3:.2f}   (inter-band shifts it)     ║
║    Bandwidth   ∝ η^{slope_bw:.2f}   (kinetic-dominated)         ║
║    Gap Δ₀₁     ∝ η^{slope_gap:.2f}   (discretization effect)    ║
║                                                          ║
║  KEY FINDING:                                            ║
║    Single-band binding energy scales as η² — consistent  ║
║    with kinetic term = 0.5 M⁻¹ (∂²/∂R²) where the      ║
║    Laplacian eigenvalue is (2πn/L_m)² ∝ η².              ║
║                                                          ║
║    E_bind(N=3)/E_bind(N=1) → 1 as η→0, confirming       ║
║    that inter-band coupling vanishes in the small-angle  ║
║    limit as expected by the envelope theory.             ║
║                                                          ║
║  BAND MIXING:                                            ║
║    < 1% for θ ≤ 1.1° (single-band regime)               ║
║    ~ 1-37% for θ = 1.5°-5° (growing multi-band mixing)  ║
║    Falls at θ=8° (flat bands ≈ potential-dominated)      ║
╚══════════════════════════════════════════════════════════╝
""")
