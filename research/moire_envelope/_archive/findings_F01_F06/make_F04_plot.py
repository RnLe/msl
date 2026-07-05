#!/usr/bin/env python3
"""
F04 — Comprehensive Validation Plot: Options A, B, C

Panels:
(a) Option A: |λ(N=3) − λ(N=1)| vs η — N_bands convergence
(b) Option A: Inter-band mixing weight vs η
(c) Option B: BW₂₀ vs η with power-law fits
(d) Option C: FD-corrected Rayleigh ratio vs η — Maxwell residual
"""
import numpy as np
import json
import matplotlib.pyplot as plt

FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'

with open(f'{FINDINGS}/F04_validation_data.json') as f:
    data = json.load(f)

optA = data['option_A']
optB = data['option_B']
optC = data['option_C']

etas = np.array([e['eta'] for e in optA])
thetas = np.array([e['theta_deg'] for e in optA])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ========== Panel (a): Option A — |ΔE₀| vs η ==========
ax = axes[0, 0]

# Auto-detect bands from data
all_band_keys = sorted(optA[0]['bands'].keys(), key=lambda x: int(x))
_plot_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
_plot_labels = {}
for bk in all_band_keys:
    bd = optA[0]['bands'][bk]
    _plot_labels[bk] = f'Band {bk} ({bd["type"]})'

for t_str in all_band_keys:
    diffs = []
    valid_etas = []
    for entry in optA:
        bd = entry['bands'].get(t_str, entry['bands'].get(int(t_str)))
        if bd is None:
            continue
        d = bd.get('delta_N1_N3')
        if d and d.get('max_abs') is not None:
            diffs.append(d['max_abs'])
            valid_etas.append(entry['eta'])
    if diffs:
        diffs = np.array(diffs)
        valid_etas = np.array(valid_etas)
        ci = int(t_str) % len(_plot_colors)
        ax.plot(valid_etas, diffs, 'o-', color=_plot_colors[ci],
                label=_plot_labels[t_str], ms=5)

        # Fit power law
        if len(diffs) >= 3:
            p = np.polyfit(np.log(valid_etas), np.log(diffs), 1)
            alpha = p[0]
            ax.plot(valid_etas, np.exp(p[1]) * valid_etas**p[0], '--',
                    color=_plot_colors[ci], alpha=0.5, label=f'  fit: η^{alpha:.2f}')

# Band 1: annotate if decoupled
for t_str in all_band_keys:
    has_data = False
    for entry in optA:
        bd = entry['bands'].get(t_str, entry['bands'].get(int(t_str)))
        if bd and bd.get('delta_N1_N3') and bd['delta_N1_N3'].get('max_abs') is not None:
            has_data = True
            break
    if not has_data:
        ax.annotate(f'Band {t_str}: decoupled',
                    xy=(0.5, 0.1), xycoords='axes fraction', fontsize=7,
                    color=_plot_colors[int(t_str) % len(_plot_colors)],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel(r'$\eta = a/L_{\mathrm{moiré}}$')
ax.set_ylabel(r'$|\lambda(N_{\rm full}) - \lambda(N{=}1)|_{\mathrm{max}}$ (c/a)')
ax.set_title('(a) Option A: N-band convergence')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ========== Panel (b): Option A — mixing weight vs η ==========
ax = axes[0, 1]
for t_str in all_band_keys:
    mixings = []
    valid_etas = []
    for entry in optA:
        bd = entry['bands'].get(t_str, entry['bands'].get(int(t_str)))
        if bd is None:
            continue
        d = bd.get('delta_N1_N3')
        if d and d.get('mean_mixing') is not None:
            mixings.append(d['mean_mixing'])
            valid_etas.append(entry['eta'])
    if mixings:
        mixings = np.array(mixings)
        valid_etas = np.array(valid_etas)
        ci = int(t_str) % len(_plot_colors)
        ax.plot(valid_etas, mixings * 100, 'o-', color=_plot_colors[ci],
                label=_plot_labels[t_str], ms=5)

ax.set_xlabel(r'$\eta$')
ax.set_ylabel('Mean inter-band mixing (%)')
ax.set_title('(b) Option A: band mixing weight')
ax.set_xscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ========== Panel (c): Option B — BW₂₀ vs η ==========
ax = axes[1, 0]

for t_key in sorted(optB.keys(), key=lambda x: int(x)):
    t = int(t_key)
    b = optB[t_key]
    bw = np.array(b['bw20'])
    e = np.array(b['etas'])
    ci = t % len(_plot_colors)
    ax.plot(e, bw, 'o-', color=_plot_colors[ci], label=b['name'], ms=5)

    # Power law fit line
    alpha = b['power_law_alpha']
    A_val = b['power_law_A']
    if not np.isnan(alpha) and not np.isnan(A_val):
        thetas_list = [d['theta_deg'] for d in optA]
        mask = np.array([th <= 3.0 for th in thetas_list[:len(e)]])
        if mask.sum() > 0:
            e_fit = e[mask]
            ax.plot(e_fit, A_val * e_fit**alpha, '--', color=_plot_colors[ci], alpha=0.5,
                    label=f'  fit: η^{alpha:.2f} (R²={b["power_law_R2"]:.3f})')

ax.set_xlabel(r'$\eta$')
ax.set_ylabel('BW₂₀ (c/a)')
ax.set_title('(c) Option B: miniband bandwidth scaling')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ========== Panel (d): Option C — FD-corrected Rayleigh ratio ==========
ax = axes[1, 1]
for entry in optC:
    for br in entry['per_band_residuals']:
        t = br['band_index']
        if br.get('R_fd_corrected') is not None:
            ci = t % len(_plot_colors)
            ax.plot(entry['eta'], br['R_fd_corrected'], 'o',
                    color=_plot_colors[ci], ms=5)

# Add horizontal lines
ax.axhline(0.01, color='green', ls='--', alpha=0.5, label='1% threshold')
ax.axhline(0.05, color='orange', ls='--', alpha=0.3, label='5% threshold')

ax.set_xlabel(r'$\eta$')
ax.set_ylabel('FD-corrected residual $R_{FD}$')
ax.set_title('(d) Option C: Maxwell residual vs η')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FINDINGS}/F04_validation_all.png', dpi=150)
plt.savefig(f'{FINDINGS}/F04_validation_all.pdf')
print('Saved F04_validation_all.png and .pdf')
plt.close()

# === Additional analysis: print key numbers ===
print("\n=== KEY QUANTITATIVE RESULTS ===")

# Option A: power-law exponent of |ΔE| vs η
print("\nOption A: |ΔE(N_full - N=1)| ~ η^α")
for t_str in all_band_keys:
    diffs = []
    valid_etas = []
    for entry in optA:
        bd = entry['bands'].get(t_str, entry['bands'].get(int(t_str)))
        if bd is None:
            continue
        d = bd.get('delta_N1_N3')
        if d and d.get('max_abs') is not None:
            diffs.append(d['max_abs'])
            valid_etas.append(entry['eta'])
    if len(diffs) >= 3:
        diffs = np.array(diffs)
        valid_etas = np.array(valid_etas)
        p = np.polyfit(np.log(valid_etas), np.log(diffs), 1)
        r2 = 1 - np.var(np.log(diffs) - (p[0]*np.log(valid_etas) + p[1]))/np.var(np.log(diffs))
        print(f"  {_plot_labels[t_str]}: α = {p[0]:.3f}, R² = {r2:.4f}")

# Option C: FD-corrected ratio statistics
print("\nOption C: FD-corrected Rayleigh ratio (should be 1.000)")
for entry in optC:
    parts = [f"  θ={entry['theta_deg']:5.1f}°"]
    for br in entry['per_band_residuals']:
        if br.get('ratio_fd_corrected') is not None:
            parts.append(f"Band {br['band_index']}: {br['ratio_fd_corrected']:.4f} ± {br['R_fd_corrected']:.4f}")
    print(', '.join(parts))
