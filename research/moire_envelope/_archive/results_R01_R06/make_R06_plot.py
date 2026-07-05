#!/usr/bin/env python3
"""
R06 Plot: Disorder Robustness
================================
Visualize how eigenvalues and localization respond to disorder.

Output: R06_disorder_robustness.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent


def main():
    print("="*70)
    print("R06 Plot: Disorder Robustness")
    print("="*70)

    d = np.load(OUTDIR / "R06_disorder.npz")
    with open(OUTDIR / "R06_data.json") as f:
        meta = json.load(f)

    evals_clean  = d['evals_clean']
    ipr_clean    = d['ipr_clean']
    noise_A      = d['noise_levels_onsite']
    evals_A      = d['evals_shift_A']   # (n_noise, n_real, n_modes)
    ipr_A        = d['ipr_disorder_A']
    dom_wt_A     = d['dom_weight_disorder_A']
    noise_B      = d['noise_levels_geom']
    evals_B      = d['evals_shift_B']
    ipr_B        = d['ipr_disorder_B']

    N_MODES = evals_clean.shape[0]
    theta_deg = meta['theta_deg']
    L_moire = meta['L_moire']

    # ── Figure: 2×3 panels ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ── Row 0: On-site noise (Part A) ─────────────────────────────────────

    # Panel (0,0): σ_ω vs noise level
    ax = axes[0, 0]
    # Per-mode sigma_omega
    for m_show in [0, 4, 9, 19]:
        if m_show >= N_MODES:
            continue
        sigma_omega = [np.std(evals_A[ni, :, m_show]) for ni in range(len(noise_A))]
        ax.loglog(noise_A, sigma_omega, 'o-', ms=5,
                  label=f'Mode {m_show}')
    # Quadratic reference
    x_ref = np.array(noise_A)
    y_ref = x_ref**1 * sigma_omega[0] / noise_A[0]
    ax.loglog(x_ref, y_ref, 'k--', lw=0.8, alpha=0.4, label=r'$\sim \sigma_V$')
    ax.set_xlabel(r'$\sigma_V / \Delta V$')
    ax.set_ylabel(r'$\sigma_\omega$ (eigenvalue fluctuation)')
    ax.set_title('(a) On-site: Eigenvalue Robustness')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (0,1): Eigenvalue distribution (violin-like) at selected noise
    ax = axes[0, 1]
    ni_show = min(2, len(noise_A) - 1)  # show σ/ΔV = 0.01
    shifts = evals_A[ni_show, :, :min(10, N_MODES)]  # (n_real, 10)
    parts = ax.violinplot([shifts[:, m] for m in range(shifts.shape[1])],
                           positions=range(shifts.shape[1]),
                           showmeans=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#377eb8')
        pc.set_alpha(0.6)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\delta\lambda$ (eigenvalue shift)')
    ax.set_title(f'(b) On-site: Shifts at $\\sigma_V/\\Delta V={noise_A[ni_show]:.3f}$')
    ax.grid(True, alpha=0.3)

    # Panel (0,2): IPR change vs noise
    ax = axes[0, 2]
    for m_show in [0, 4, 9]:
        if m_show >= N_MODES:
            continue
        ipr_ratio = [np.mean(ipr_A[ni, :, m_show]) / ipr_clean[m_show]
                     if ipr_clean[m_show] > 0 else 1 for ni in range(len(noise_A))]
        ax.semilogx(noise_A, ipr_ratio, 'o-', ms=5, label=f'Mode {m_show}')
    ax.axhline(1.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel(r'$\sigma_V / \Delta V$')
    ax.set_ylabel(r'IPR$_{\rm disorder}$ / IPR$_{\rm clean}$')
    ax.set_title('(c) On-site: Localization Change')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Row 1: Geometric noise (Part B) ───────────────────────────────────

    # Panel (1,0): σ_ω vs noise level
    ax = axes[1, 0]
    for m_show in [0, 4, 9, 19]:
        if m_show >= N_MODES:
            continue
        sigma_omega_B = [np.std(evals_B[ni, :, m_show]) for ni in range(len(noise_B))]
        ax.loglog(noise_B, sigma_omega_B, 's-', ms=5, label=f'Mode {m_show}')
    ax.set_xlabel(r'$\sigma_s / L_m$')
    ax.set_ylabel(r'$\sigma_\omega$')
    ax.set_title('(d) Geometric: Eigenvalue Robustness')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (1,1): Comparison Part A vs Part B (mode 0)
    ax = axes[1, 1]
    sigma_A = [np.std(evals_A[ni, :, 0]) for ni in range(len(noise_A))]
    sigma_B = [np.std(evals_B[ni, :, 0]) for ni in range(len(noise_B))]
    # Normalize x to "disorder strength" relative to characteristic scale
    ax.loglog(noise_A, sigma_A, 'o-', color='#e41a1c', ms=6,
              label='On-site ($\\sigma_V/\\Delta V$)')
    ax.loglog(noise_B, sigma_B, 's-', color='#377eb8', ms=6,
              label='Geometric ($\\sigma_s/L_m$)')
    ax.set_xlabel('Noise amplitude')
    ax.set_ylabel(r'$\sigma_\omega$ (mode 0)')
    ax.set_title('(e) Comparison: On-site vs Geometric')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (1,2): Band mixing under disorder
    ax = axes[1, 2]
    dom_wt_clean_arr = np.array([meta.get('dom_weight_clean', [1.0]*N_MODES)])
    if 'dom_weight_clean' in d.files:
        dom_wt_clean_arr = d['dom_weight_clean']
    for ni in range(len(noise_A)):
        dom_avg = np.mean(dom_wt_A[ni, :, :min(10,N_MODES)], axis=0)  # avg over realizations
        ax.plot(range(min(10,N_MODES)), 1 - dom_avg, 'o-', ms=4,
                alpha=0.7, label=f'$\\sigma/\\Delta V={noise_A[ni]:.3f}$')
    # Clean reference
    if dom_wt_clean_arr.ndim > 0 and len(dom_wt_clean_arr) >= 10:
        ax.plot(range(min(10,N_MODES)), 1 - dom_wt_clean_arr[:min(10,N_MODES)],
                'k^-', ms=5, label='clean', lw=2)
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Band mixing (1 - dom. weight)')
    ax.set_title('(f) Band Mixing vs Disorder')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    fig.suptitle(
        f'Disorder Robustness — $\\theta={theta_deg}°$, $L_m={L_moire:.0f}a$, '
        f'{meta["N_realizations"]} realizations',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        outfile = OUTDIR / f"R06_disorder_robustness.{ext}"
        fig.savefig(outfile, dpi=200, bbox_inches='tight')
        print(f"Saved {outfile}")
    plt.close(fig)

    print("\nR06 plot complete.")


if __name__ == '__main__':
    main()
