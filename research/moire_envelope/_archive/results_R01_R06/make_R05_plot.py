#!/usr/bin/env python3
"""
R05 Plot: Scaling Laws & Tunability
=====================================
Multi-panel figure showing how moiré photonic crystal properties
scale with twist angle θ (or perturbation parameter η).

Output: R05_scaling_laws.png/.pdf, R05_tunability.png/.pdf
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent

BAND_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
BAND_LABELS = ['Band 0 (hole)', 'Band 1 (elec)', 'Band 2 (target, hole)',
               'Band 3 (hole)', 'Band 4 (hole)']


def add_powerlaw_line(ax, x, y, label_prefix="", color='k', ls='--', lw=1.0,
                      fit_mask=None):
    """Overlay a power-law fit line on a log-log axes."""
    xarr = np.asarray(x, float)
    yarr = np.asarray(y, float)
    mask = (xarr > 0) & (yarr > 0)
    if fit_mask is not None:
        mask &= fit_mask
    if mask.sum() < 3:
        return
    lx = np.log(xarr[mask])
    ly = np.log(yarr[mask])
    p = np.polyfit(lx, ly, 1)
    R2 = 1 - np.sum((ly - np.polyval(p, lx))**2) / np.sum((ly - ly.mean())**2)
    x_fit = np.geomspace(xarr[mask].min()*0.8, xarr[mask].max()*1.2, 50)
    y_fit = np.exp(np.polyval(p, np.log(x_fit)))
    ax.plot(x_fit, y_fit, ls=ls, color=color, lw=lw, alpha=0.7)
    ax.text(x_fit[len(x_fit)//2], y_fit[len(y_fit)//2]*1.3,
            f'{label_prefix}$\\sim \\eta^{{{p[0]:.2f}}}$ ($R^2={R2:.2f}$)',
            fontsize=7, color=color, ha='center')


def main():
    print("="*70)
    print("R05 Plot: Scaling Laws & Tunability")
    print("="*70)

    d = np.load(OUTDIR / "R05_scaling.npz")
    with open(OUTDIR / "R05_data.json") as f:
        meta = json.load(f)

    thetas    = d['thetas']
    etas      = d['etas']
    L_moire   = d['L_moire']
    BW_50     = d['BW_50']
    BW_20     = d['BW_20']
    BW_total  = d['BW_total']
    gap_01    = d['gap_01']
    mode_sp   = d['mode_spacing_mean']
    ipr_gnd   = d['ipr_ground']
    ipr_10    = d['ipr_mean_10']
    spread_gnd= d['spread_ground']
    spread_10 = d['spread_mean_10']
    VK_ratio  = d['VK_ratio']
    max_mix   = d['max_mixing']
    band_pur  = d['band_purity']
    N_BANDS   = int(meta['N_BANDS'])
    TARGET    = int(meta['TARGET_BAND'])

    # ── Fig 1: Scaling Laws (3×2) ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel (0,0): Bandwidth vs η
    ax = axes[0, 0]
    ax.loglog(etas, BW_50, 'o-', color='#e41a1c', ms=6, label='$\\Delta\\omega_{50}$')
    ax.loglog(etas, BW_20, 's-', color='#377eb8', ms=5, label='$\\Delta\\omega_{20}$')
    ax.loglog(etas, BW_total, '^-', color='#4daf4a', ms=5, alpha=0.6, label='$\\Delta\\omega_{\\rm total}$')
    add_powerlaw_line(ax, etas, BW_50, 'BW$_{50}$ ', color='#e41a1c')
    add_powerlaw_line(ax, etas, BW_20, 'BW$_{20}$ ', color='#377eb8')
    ax.set_xlabel(r'$\eta = 2\sin(\theta/2)$')
    ax.set_ylabel(r'Bandwidth $(c/a)$')
    ax.set_title('(a) Bandwidth Scaling')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (0,1): Mode spacing vs η
    ax = axes[0, 1]
    ax.loglog(etas, mode_sp, 'o-', color='#984ea3', ms=6, label=r'$\langle\delta\omega\rangle$')
    ax.loglog(etas, gap_01, 's-', color='#ff7f00', ms=5, label=r'$\Delta_{01}$')
    add_powerlaw_line(ax, etas, mode_sp, r'$\delta\omega$ ', color='#984ea3')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'Mode spacing $(c/a)$')
    ax.set_title('(b) Mode Spacing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (0,2): Localization vs η
    ax = axes[0, 2]
    ax.loglog(etas, ipr_gnd, 'o-', color='#e41a1c', ms=6, label='IPR (ground)')
    ax.loglog(etas, ipr_10, 's-', color='#377eb8', ms=5, label=r'$\langle$IPR$\rangle_{10}$')
    add_powerlaw_line(ax, etas, ipr_gnd, 'IPR ', color='#e41a1c')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel('IPR')
    ax.set_title('(c) Localization (IPR)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (1,0): V/KE ratio (target band) vs η
    ax = axes[1, 0]
    for n in range(N_BANDS):
        ax.loglog(etas, VK_ratio[:, n], 'o-', ms=4, color=BAND_COLORS[n],
                  label=BAND_LABELS[n], alpha=0.7 if n != TARGET else 1.0,
                  lw=2 if n == TARGET else 1)
    ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='V = KE')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$V_{\rm depth} / KE_{\rm scale}$')
    ax.set_title('(d) Potential / Kinetic Ratio')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')

    # Panel (1,1): Spread vs η + secondary axis for L_m
    ax = axes[1, 1]
    ax.semilogx(etas, spread_gnd, 'o-', color='#e41a1c', ms=6, label='Ground state')
    ax.semilogx(etas, spread_10, 's-', color='#377eb8', ms=5, label=r'$\langle\sigma\rangle_{10}$')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'Spread $\sigma$ (frac. of $L_m$)')
    ax.set_title(r'(e) Envelope Spread ($\sigma / L_m$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Secondary axis: theta
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale('log')
    tick_etas = etas[::2]
    tick_labels = [f'{t:.1f}°' for t in thetas[::2]]
    ax2.set_xticks(tick_etas)
    ax2.set_xticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel(r'$\theta$', fontsize=9)

    # Panel (1,2): Band mixing vs η
    ax = axes[1, 2]
    ax.semilogx(etas, max_mix, 'ko-', ms=6, label='max mixing (any mode)')
    ax.semilogx(etas, 1 - np.array([e[0]['dominant_band_weight'] for e in
                [json.load(open(
                    Path(OUTDIR.parent / "runsV3" / "phase0_mpb_v3_20260206_152443" /
                         "eta_sweep_20260206_173808" /
                         f"theta_{t:.3f}" / "candidate_0000" / "phase3_mode_stats.json")))
                 for t in thetas]]),
                's-', color='#e41a1c', ms=5, label='1 - dom_weight (ground)')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel('Mixing fraction')
    ax.set_title('(f) Band Mixing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.suptitle('Scaling Laws: Moiré Photonic Crystal vs Twist Angle',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        fig.savefig(OUTDIR / f"R05_scaling_laws.{ext}", dpi=200, bbox_inches='tight')
        print(f"Saved R05_scaling_laws.{ext}")
    plt.close(fig)

    # ── Fig 2: Tunability & Design Space (2×2) ───────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: θ vs L_m with BW color
    ax = axes2[0, 0]
    sc = ax.scatter(thetas, L_moire, c=np.log10(BW_50), s=80, cmap='viridis',
                    edgecolors='k', linewidths=0.5)
    for i, t in enumerate(thetas):
        ax.annotate(f'{t}°', (thetas[i], L_moire[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    plt.colorbar(sc, ax=ax, label=r'$\log_{10}(\Delta\omega_{50})$')
    ax.set_xlabel(r'Twist angle $\theta$ (deg)')
    ax.set_ylabel(r'Moiré length $L_m / a$')
    ax.set_title('(a) Design Space: $L_m$ vs $\\theta$')
    ax.grid(True, alpha=0.3)

    # Panel B: BW vs L_m
    ax = axes2[0, 1]
    ax.loglog(L_moire, BW_50, 'o-', ms=7, color='#e41a1c')
    for i in range(len(thetas)):
        ax.annotate(f'{thetas[i]}°', (L_moire[i], BW_50[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    add_powerlaw_line(ax, L_moire, BW_50, 'BW ', color='#e41a1c')
    ax.set_xlabel(r'$L_m / a$')
    ax.set_ylabel(r'$\Delta\omega_{50}$ (c/a)')
    ax.set_title(r'(b) Bandwidth vs $L_m$')
    ax.grid(True, alpha=0.3, which='both')

    # Panel C: Eigenvalue fan diagram
    ax = axes2[1, 0]
    with open(OUTDIR.parent / "runsV3" / "phase0_mpb_v3_20260206_152443" /
              "eta_sweep_20260206_173808" / "sweep_results.json") as f:
        sweep_raw = json.load(f)
    sweep_raw.sort(key=lambda e: e['theta_deg'])
    for entry in sweep_raw:
        evals = np.array(entry['eigenvalues'][:30])
        ax.plot([entry['theta_deg']]*len(evals), evals, '.', ms=2.5,
                color='#377eb8', alpha=0.5)
    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'$\lambda_n$ (eigenvalue)')
    ax.set_title('(c) Eigenvalue Fan Diagram')
    ax.grid(True, alpha=0.3)

    # Panel D: V/KE phase diagram
    ax = axes2[1, 1]
    for n in range(N_BANDS):
        ax.semilogy(thetas, VK_ratio[:, n], 'o-', ms=5, color=BAND_COLORS[n],
                    label=BAND_LABELS[n])
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.fill_between(thetas, 0.01, 1.0, color='#fee08b', alpha=0.15, label='Dispersive')
    ax.fill_between(thetas, 1.0, 1000, color='#d9ef8b', alpha=0.15, label='Flat-band')
    ax.set_xlabel(r'$\theta$ (deg)')
    ax.set_ylabel(r'$V_{\rm depth} / KE_{\rm scale}$')
    ax.set_title('(d) Operating Regime')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig2.suptitle('Tunability & Design Space', fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ['png', 'pdf']:
        fig2.savefig(OUTDIR / f"R05_tunability.{ext}", dpi=200, bbox_inches='tight')
        print(f"Saved R05_tunability.{ext}")
    plt.close(fig2)

    print("\nR05 plot complete.")


if __name__ == '__main__':
    main()
