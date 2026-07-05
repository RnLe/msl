#!/usr/bin/env python3
"""
Log-log plot of peak RSS (memory) and solve time vs N_cells
for FDFD (CHOLMOD) and MPB at 64 px/cell, Gamma TM, hex rods.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(STUDY_DIR, 'data_gamma_tm_hex')
FIG_DIR   = os.path.join(STUDY_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

FDFD_PURPLE = '#8E7CC3'
MPB_GOLD    = '#D4920B'

def load_results():
    path = os.path.join(DATA_DIR, 'memory_scaling_64px.json')
    with open(path) as f:
        return json.load(f)

def main():
    results = load_results()

    fdfd = sorted([r for r in results if r['solver'] == 'fdfd'],
                  key=lambda r: r['N_cells'])
    mpb  = sorted([r for r in results if r['solver'] == 'mpb'],
                  key=lambda r: r['N_cells'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    # ── Left: Peak RSS ──
    ax = axes[0]
    if fdfd:
        N_f = [r['N_cells'] for r in fdfd]
        rss_f = [r['peak_rss_mb'] / 1024 for r in fdfd]  # GB
        ax.loglog(N_f, rss_f, 'o-', color=FDFD_PURPLE, label='FDFD (CHOLMOD)', ms=6)
    if mpb:
        N_m = [r['N_cells'] for r in mpb]
        rss_m = [r['peak_rss_mb'] / 1024 for r in mpb]  # GB
        ax.loglog(N_m, rss_m, 's-', color=MPB_GOLD, label='MPB', ms=6)

    # Reference slopes
    if fdfd and len(fdfd) >= 2:
        N_ref = np.array([fdfd[0]['N_cells'], fdfd[-1]['N_cells']])
        # N^1 reference line
        c1 = rss_f[0] / N_ref[0]
        ax.loglog(N_ref, c1 * N_ref, '--', color='gray', alpha=0.4, lw=1)
        ax.text(N_ref[-1], c1 * N_ref[-1] * 1.3, r'$\propto N$',
                fontsize=8, color='gray', ha='right')
        # N^1.5 reference
        c15 = rss_f[0] / N_ref[0]**1.5
        ax.loglog(N_ref, c15 * N_ref**1.5, ':', color='gray', alpha=0.4, lw=1)
        ax.text(N_ref[-1], c15 * N_ref[-1]**1.5 * 1.3, r'$\propto N^{1.5}$',
                fontsize=8, color='gray', ha='right')

    ax.set_xlabel(r'$N_{\mathrm{cells}}$ (unit cells in supercell)')
    ax.set_ylabel('Peak RSS (GB)')
    ax.set_title('Memory Consumption (64 px/cell, 50 modes)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.3)

    # ── Right: Solve time ──
    ax = axes[1]
    if fdfd:
        t_f = [r['time_s'] for r in fdfd]
        ax.loglog(N_f, t_f, 'o-', color=FDFD_PURPLE, label='FDFD (CHOLMOD)', ms=6)
    if mpb:
        t_m = [r['time_s'] for r in mpb]
        ax.loglog(N_m, t_m, 's-', color=MPB_GOLD, label='MPB', ms=6)

    ax.set_xlabel(r'$N_{\mathrm{cells}}$ (unit cells in supercell)')
    ax.set_ylabel('Solve time (s)')
    ax.set_title('Wall-clock Time (64 px/cell, 50 modes)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', ls=':', alpha=0.3)

    # Add angle annotations on top axis for both panels
    for ax_i in axes:
        ax2 = ax_i.twiny()
        ax2.set_xscale('log')
        ax2.set_xlim(ax_i.get_xlim())
        # Use FDFD or MPB N_cells as tick positions
        all_data = fdfd if fdfd else mpb
        if all_data:
            ticks = [r['N_cells'] for r in all_data]
            labels = [f"{r['theta_deg']:.1f}°" for r in all_data]
            ax2.set_xticks(ticks)
            ax2.set_xticklabels(labels, fontsize=7)
            ax2.set_xlabel(r'Twist angle $\theta$', fontsize=8)

    for ext in ('svg', 'png'):
        outpath = os.path.join(FIG_DIR, f'memory_scaling_64px.{ext}')
        fig.savefig(outpath, dpi=200)
        print(f'Saved {outpath}')
    plt.close(fig)

    # Print summary
    print(f'\n{"Solver":>6} {"θ°":>7} {"N":>6} {"DOF":>10} {"RSS GB":>8} {"Time s":>8}')
    print('-' * 55)
    for r in sorted(results, key=lambda x: (x['solver'], x['N_cells'])):
        print(f'{r["solver"]:>6} {r["theta_deg"]:>7.2f} {r["N_cells"]:>6} '
              f'{r["DOF"]:>10} {r["peak_rss_mb"]/1024:>8.1f} {r["time_s"]:>8.1f}')


if __name__ == '__main__':
    main()
