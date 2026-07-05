#!/usr/bin/env python3
"""
F05 — 6-panel thesis-grade plot:

Panel (a): Gauge smoothness — overlap magnitude & phase disorder per band
Panel (b): IPR / Participation number vs η for all 3 bands
Panel (c): Energy budget — ||K||/||V||, ||BH||/||V|| vs η (log-log)
Panel (d): Berry connection max|A| vs η
Panel (e): Miniband dispersion Δλ(q) along BZ path
Panel (f): Term convergence — eigenvalue shifts from kinetic correction vs η
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'

with open(f'{FINDINGS}/F05_validation_data.json') as f:
    d = json.load(f)

fig = plt.figure(figsize=(16, 11))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

# Auto-detect bands from gauge_smoothness data
_gauge0 = d['gauge_smoothness'][0]
band_ids = sorted(_gauge0['bands'].keys(), key=lambda x: int(x))
N_bands = len(band_ids)

_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f']
_default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
colors = [_default_colors[i % len(_default_colors)] for i in range(N_bands)]
markers = [_default_markers[i % len(_default_markers)] for i in range(N_bands)]

# Build band labels from data (use type if available, else generic)
band_labels = []
for b in band_ids:
    try:
        btype = d['ipr'][0]['bands'][b].get('type', f'band {b}')
        band_labels.append(f'Band {b} ({btype})')
    except (KeyError, IndexError):
        band_labels.append(f'Band {b}')

# ============================================================================
# (a) Gauge Smoothness
# ============================================================================
ax_a = fig.add_subplot(gs[0, 0])

# The gauge data is identical across angles (expected: Bloch fields are the same
# underlying MPB grid, twist only affects moiré mapping). So we show per-band metrics.
gauge = d['gauge_smoothness'][0]  # Same for all angles
frac_good = []
min_ov = []
phase_std_mean = []
for b in band_ids:
    bd = gauge['bands'][b]
    frac_good.append((bd['frac_good_s1'] + bd['frac_good_s2']) / 2)
    min_ov.append(bd['min_overlap_mag'])
    phase_std_mean.append((bd['phase_std_s1'] + bd['phase_std_s2']) / 2)

x = np.arange(N_bands)
width = 0.35
bars1 = ax_a.bar(x - width/2, frac_good, width, label='Frac |ov|>0.99', color=colors[:N_bands], alpha=0.7)
bars2 = ax_a.bar(x + width/2, [p / np.pi for p in phase_std_mean], width,
                  label='Phase std (×π)', color=colors[:N_bands], alpha=0.4, hatch='//')
ax_a.set_xticks(x)
ax_a.set_xticklabels([f'Band {b}' for b in band_ids])
ax_a.set_ylabel('Metric value')
ax_a.set_title('(a) Gauge Smoothness Diagnostic')
ax_a.legend(fontsize=8, loc='upper right')
ax_a.set_ylim(0, 1.1)
# Add min overlap as text
for i, mo in enumerate(min_ov):
    ax_a.text(i, 0.05, f'min|ov|={mo:.3f}', ha='center', fontsize=7, color='red')

# ============================================================================
# (b) IPR / Participation Number vs η
# ============================================================================
ax_b = fig.add_subplot(gs[0, 1])

for b_idx, b in enumerate(band_ids):
    etas = [entry['eta'] for entry in d['ipr']]
    pns = [entry['bands'][b]['mean_PN_5'] for entry in d['ipr']]
    ax_b.semilogy(etas, pns, f'-{markers[b_idx]}', color=colors[b_idx],
                   label=band_labels[b_idx], markersize=5)

# Auto-detect N_sites from first entry if available
try:
    n_sites = d['ipr'][0]['bands'][band_ids[0]].get('N_sites', 16384)
except (KeyError, IndexError):
    n_sites = 16384
ax_b.axhline(n_sites, color='gray', ls='--', alpha=0.5, label=f'N_sites={n_sites}')
ax_b.set_xlabel('η = a/L_moiré')
ax_b.set_ylabel('Participation Number (5 lowest modes)')
ax_b.set_title('(b) Mode Localization (IPR)')
ax_b.legend(fontsize=7, loc='center right')
ax_b.set_ylim(50, 20000)

# ============================================================================
# (c) Energy Budget: ||K||/||V|| and ||BH||/||V|| vs η
# ============================================================================
ax_c = fig.add_subplot(gs[0, 2])

for b_idx, b in enumerate(band_ids):
    etas = [entry['eta'] for entry in d['energy_budget']]
    k_ratios = [entry['bands'][b]['ratio_K_V'] for entry in d['energy_budget']]
    bh_ratios = [entry['bands'][b]['ratio_BH_V'] for entry in d['energy_budget']]
    
    ax_c.loglog(etas, k_ratios, f'-{markers[b_idx]}', color=colors[b_idx],
                label=f'||K||/||V|| B{b_idx}', markersize=5)
    ax_c.loglog(etas, bh_ratios, f'--{markers[b_idx]}', color=colors[b_idx],
                alpha=0.5, label=f'||BH||/||V|| B{b_idx}', markersize=4)

# Reference lines
eta_ref = np.array([0.008, 0.15])
ax_c.loglog(eta_ref, 5e4 * eta_ref**2, 'k:', alpha=0.3, label='∝η²')
ax_c.axhline(1, color='red', ls=':', alpha=0.3, label='||K||=||V||')

ax_c.set_xlabel('η')
ax_c.set_ylabel('Operator norm ratio')
ax_c.set_title('(c) Energy Budget vs η')
ax_c.legend(fontsize=5.5, ncol=2, loc='upper left')

# ============================================================================
# (d) Berry Connection max|A| vs η
# ============================================================================
ax_d = fig.add_subplot(gs[1, 0])

for b_idx, b in enumerate(band_ids):
    etas = [entry['eta'] for entry in d['energy_budget']]
    max_A = [entry['bands'][b]['max_A_berry'] for entry in d['energy_budget']]
    ax_d.plot(etas, max_A, f'-{markers[b_idx]}', color=colors[b_idx],
              label=band_labels[b_idx], markersize=5)

# Linear reference
eta_arr = np.array(etas)
ax_d.plot(eta_arr, eta_arr * 94, 'k--', alpha=0.3, label='∝η (linear)')
ax_d.set_xlabel('η')
ax_d.set_ylabel('max|A_berry|')
ax_d.set_title('(d) Berry Connection Magnitude')
ax_d.legend(fontsize=7)

# ============================================================================
# (e) Miniband Dispersion Δλ(q)
# ============================================================================
ax_e = fig.add_subplot(gs[1, 1])

# Plot a high-angle case (more dispersive): pick highest available θ ≤ 5°
avail_thetas = sorted(set(e['theta_deg'] for e in d['miniband_dispersion']))
pref = [th for th in avail_thetas if th <= 5.0]
chosen_theta = pref[-1] if pref else avail_thetas[-1]
entry_5 = [e for e in d['miniband_dispersion'] if e['theta_deg'] == chosen_theta][0]
n_q_seg = entry_5['n_qpoints_per_segment']

for b_idx, b in enumerate(band_ids):
    bd = entry_5['bands'][b]
    evals_vs_q = np.array(bd['eigenvalues_vs_q'])
    n_q = evals_vs_q.shape[0]
    q_idx = np.arange(n_q)
    
    # Plot lowest 3 minibands
    for m in range(min(3, evals_vs_q.shape[1])):
        label = f'{band_labels[b_idx]} m={m}' if m == 0 else None
        ax_e.plot(q_idx, evals_vs_q[:, m], '-', color=colors[b_idx],
                  alpha=0.8 - 0.2*m, linewidth=1.5 - 0.3*m, label=label)

# Add BZ path labels
tick_positions = [0, n_q_seg, 2*n_q_seg, 3*n_q_seg]
ax_e.set_xticks(tick_positions)
ax_e.set_xticklabels(['Γ', 'X', 'M', 'Γ'])
for tp in tick_positions:
    ax_e.axvline(tp, color='gray', ls=':', alpha=0.3)
ax_e.set_ylabel('Eigenvalue (c/a units)')
ax_e.set_title(f'(e) Miniband Dispersion (θ={chosen_theta}°)')
ax_e.legend(fontsize=6, loc='best')

# ============================================================================
# (f) Term Convergence: Kinetic shift vs η
# ============================================================================
ax_f = fig.add_subplot(gs[1, 2])

for b_idx, b in enumerate(band_ids):
    etas = [entry['eta'] for entry in d['term_convergence']]
    shifts_K = [abs(entry['bands'][b]['shift_K']) for entry in d['term_convergence']]
    
    ax_f.loglog(etas, shifts_K, f'-{markers[b_idx]}', color=colors[b_idx],
                label=band_labels[b_idx], markersize=5)

# Reference scaling
eta_ref = np.array([0.008, 0.15])
ax_f.loglog(eta_ref, 10 * eta_ref**2, 'k--', alpha=0.3, label='∝η²')
ax_f.set_xlabel('η')
ax_f.set_ylabel('|ΔE₀| from kinetic correction')
ax_f.set_title('(f) Term Convergence: Kinetic Shift')
ax_f.legend(fontsize=7)

# ============================================================================
# Save
# ============================================================================
fig.suptitle('F05 — Additional Thesis Validations: Gauge, IPR, Energy Budget, '
             'Berry, Miniband, Terms', fontsize=13, fontweight='bold', y=0.98)

outpath = f'{FINDINGS}/F05_validation_all.png'
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f'Saved to {outpath}')
