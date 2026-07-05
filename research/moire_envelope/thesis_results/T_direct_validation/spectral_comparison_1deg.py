"""
Phase 1 + Phase 3: Comprehensive spectral structure comparison.
- Relative differences (|Δω|/ω, |Δω|/spacing)
- Bandwidth ratio
- DOS / KDE overlay
- CDF / Kolmogorov-Smirnov test
- Gap structure comparison

Uses Born-Huang envelope data and best-resolution FDFD (res=20).
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde, ks_2samp

out_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Load data ─────────────────────────────────────────────────
# Envelope (with Born-Huang)
with open('/home/renlephy/msl/research/moire_envelope/runsV3/'
          'thesis_honeycomb_K_b1_20260307_171424/'
          'eta_sweep_20260310_191610/sweep_results.json') as f:
    env_bh = json.load(f)[0]
env_freqs = np.sort(env_bh['omega_ref'] + np.array(env_bh['eigenvalues']))
omega_ref = env_bh['omega_ref']
theta_deg = env_bh['theta_deg']
eta = env_bh['eta']
env_bw = env_freqs.max() - env_freqs.min()
env_spacing = np.mean(np.diff(env_freqs))

# FDFD — use res=40 (best resolution) as primary, others for comparison
results_by_res = {}
for res in [12, 16, 20, 40]:
    fname = os.path.join(out_dir, f'fdfd_dirac_m30_n29_res{res}_v2.npz')
    if os.path.exists(fname):
        results_by_res[res] = np.sort(np.load(fname)['freqs'])

fdfd_all = results_by_res[40]  # Primary comparison against best resolution

# ─── Hungarian matching ───────────────────────────────────────
margin = 0.001
fdfd_mask = (fdfd_all >= env_freqs.min() - margin) & (fdfd_all <= env_freqs.max() + margin)
fdfd_w = fdfd_all[fdfd_mask]

cost = np.abs(env_freqs[:, None] - fdfd_w[None, :])
row_ind, col_ind = linear_sum_assignment(cost)
matched_fdfd = fdfd_w[col_ind]
residuals = env_freqs[row_ind] - matched_fdfd
abs_res = np.abs(residuals)

# Modes in exact envelope range
fdfd_in_env = fdfd_all[(fdfd_all >= env_freqs.min()) & (fdfd_all <= env_freqs.max())]

# ─── Stats ────────────────────────────────────────────────────
print(f"{'='*70}")
print(f"SPECTRAL STRUCTURE COMPARISON")
print(f"θ = {theta_deg:.4f}°, η = {eta:.6f}, (m,n)=(30,29)")
print(f"{'='*70}")
print()

# ────── Relative differences ──────
rel_diff = abs_res / matched_fdfd  # |Δω|/ω_FDFD
spacing_ratio = abs_res / env_spacing  # |Δω|/spacing

print(f"RELATIVE DIFFERENCES")
print(f"  |Δω|/ω_FDFD:  mean={np.mean(rel_diff):.2e}, max={np.max(rel_diff):.2e}")
print(f"  |Δω|/spacing:  mean={np.mean(spacing_ratio):.3f}, max={np.max(spacing_ratio):.3f}")
print(f"  fraction with |Δω| < 1 spacing: {np.sum(spacing_ratio < 1)}/50")
print(f"  fraction with |Δω| < 0.5 spacing: {np.sum(spacing_ratio < 0.5)}/50")
print()

# ────── Bandwidth ──────
# Matched FDFD subset
fdfd_matched_bw = matched_fdfd.max() - matched_fdfd.min()
# Full FDFD in window
fdfd_full_bw = fdfd_in_env.max() - fdfd_in_env.min()

print(f"BANDWIDTH")
print(f"  Envelope:         {env_bw:.6f}")
print(f"  FDFD (matched):   {fdfd_matched_bw:.6f}, ratio = {env_bw/fdfd_matched_bw:.4f}")
print(f"  FDFD (full window): {fdfd_full_bw:.6f}, ratio = {env_bw/fdfd_full_bw:.4f}")
print()

# ────── Spectral moments ──────
print(f"SPECTRAL MOMENTS")
print(f"  Mean:  EA={np.mean(env_freqs):.6f}, FDFD(matched)={np.mean(matched_fdfd):.6f}, "
      f"Δ={abs(np.mean(env_freqs)-np.mean(matched_fdfd)):.6e}")
print(f"  Std:   EA={np.std(env_freqs):.6e}, FDFD(matched)={np.std(matched_fdfd):.6e}, "
      f"ratio={np.std(env_freqs)/np.std(matched_fdfd):.4f}")
print()

# ────── KS test ──────
ks_stat, ks_pval = ks_2samp(env_freqs, matched_fdfd)
ks_stat_full, ks_pval_full = ks_2samp(env_freqs, fdfd_in_env)
print(f"KOLMOGOROV-SMIRNOV TEST")
print(f"  EA vs FDFD (matched): KS={ks_stat:.4f}, p={ks_pval:.4f}")
print(f"  EA vs FDFD (full window): KS={ks_stat_full:.4f}, p={ks_pval_full:.4f}")
print()

# ────── Gap structure ──────
def find_gaps(freqs, min_gap_ratio=2.0):
    sorted_f = np.sort(freqs)
    spacings = np.diff(sorted_f)
    median_sp = np.median(spacings)
    gaps = []
    for i, sp in enumerate(spacings):
        if sp > min_gap_ratio * median_sp:
            gaps.append({'center': 0.5*(sorted_f[i] + sorted_f[i+1]),
                         'size': sp, 'index': i})
    return gaps

env_gaps = find_gaps(env_freqs)
fdfd_matched_gaps = find_gaps(matched_fdfd)
print(f"GAP STRUCTURE")
print(f"  EA gaps (>2× median spacing):  {len(env_gaps)}")
for g in env_gaps:
    print(f"    center={g['center']:.6f}, size={g['size']*1e6:.0f}×10⁻⁶, at index {g['index']}")
print(f"  FDFD matched gaps:  {len(fdfd_matched_gaps)}")
for g in fdfd_matched_gaps:
    print(f"    center={g['center']:.6f}, size={g['size']*1e6:.0f}×10⁻⁶, at index {g['index']}")

# Check gap correspondence
print(f"\n  Gap correspondence:")
for eg in env_gaps:
    best_match = None
    best_dist = float('inf')
    for fg in fdfd_matched_gaps:
        d = abs(eg['center'] - fg['center'])
        if d < best_dist:
            best_dist = d
            best_match = fg
    if best_match and best_dist < 0.001:
        print(f"    EA gap at {eg['center']:.6f} ↔ FDFD gap at {best_match['center']:.6f}, "
              f"Δ_center={best_dist*1e6:.0f}×10⁻⁶, "
              f"size ratio={eg['size']/best_match['size']:.2f}")
    else:
        print(f"    EA gap at {eg['center']:.6f} — NO FDFD match")

# ────── FDFD convergence context ──────
print(f"\nFDFD RESOLUTION CONVERGENCE CONTEXT")
for res in [12, 16, 20]:
    if res in results_by_res:
        w = results_by_res[res]
        mask_r = (w >= env_freqs.min() - margin) & (w <= env_freqs.max() + margin)
        w_r = w[mask_r]
        cost_r = np.abs(env_freqs[:, None] - w_r[None, :])
        r_r, c_r = linear_sum_assignment(cost_r)
        res_r = np.abs(env_freqs[r_r] - w_r[c_r])
        print(f"  res={res}: mean|Δ|={np.mean(res_r)*1e6:.1f}×10⁻⁶, "
              f"max={np.max(res_r)*1e6:.1f}×10⁻⁶")

# FDFD discretization error (16→20 drift)
if 20 in results_by_res and 40 in results_by_res:
    w20 = results_by_res[20][(results_by_res[20] >= env_freqs.min()-margin) &
                              (results_by_res[20] <= env_freqs.max()+margin)]
    w40 = results_by_res[40][(results_by_res[40] >= env_freqs.min()-margin) &
                              (results_by_res[40] <= env_freqs.max()+margin)]
    nm = min(len(w20), len(w40))
    cost_d = np.abs(w20[:nm, None] - w40[:nm][None, :])
    r_d, c_d = linear_sum_assignment(cost_d)
    drift = np.abs(w20[r_d] - w40[c_d])
    print(f"\n  FDFD drift (res20→40): mean={np.mean(drift)*1e6:.1f}×10⁻⁶, "
          f"max={np.max(drift)*1e6:.1f}×10⁻⁶")
    print(f"  → FDFD discretization error ({np.mean(drift)*1e6:.0f}×10⁻⁶) is "
          f"{'LARGER' if np.mean(drift) > np.mean(abs_res) else 'smaller'} "
          f"than EA-FDFD residual ({np.mean(abs_res)*1e6:.0f}×10⁻⁶)")

print()

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Relative differences (Phase 1)
# ═══════════════════════════════════════════════════════════════
C_GREEN = '#16A34A'
C_RED = '#DC2626'
C_BLUE = '#2563EB'
C_ORANGE = '#EA580C'

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) |Δω|/ω vs mode index
ax = axes[0]
ax.scatter(np.arange(50), rel_diff * 1e4, s=25, c=C_GREEN,
           edgecolors='white', linewidths=0.3, zorder=5)
ax.axhline(np.mean(rel_diff)*1e4, color=C_RED, ls='--', lw=1.2,
           label=f'mean = {np.mean(rel_diff):.2e}')
ax.set_xlabel('Envelope mode index', fontsize=10)
ax.set_ylabel('|Δω| / ω_FDFD  (×10⁻⁴)', fontsize=10)
ax.set_title('(a)  Relative frequency error', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# (b) |Δω|/spacing histogram  
ax = axes[1]
bins = np.linspace(0, max(3.5, np.max(spacing_ratio)+0.5), 25)
ax.hist(spacing_ratio, bins=bins, color=C_GREEN, edgecolor='white',
        linewidth=0.5, alpha=0.85)
ax.axvline(1.0, color=C_RED, ls='--', lw=1.5, label='1 mode spacing')
ax.axvline(np.mean(spacing_ratio), color=C_ORANGE, ls=':', lw=1.5,
           label=f'mean = {np.mean(spacing_ratio):.2f}')
ax.set_xlabel('|Δω| / (mean mode spacing)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('(b)  Error in units of mode spacing', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.text(0.97, 0.75,
    f'{np.sum(spacing_ratio<1)}/50 < 1 spacing\n'
    f'{np.sum(spacing_ratio<0.5)}/50 < 0.5 spacing',
    transform=ax.transAxes, ha='right', fontsize=9,
    bbox=dict(facecolor='#F0FDF4', edgecolor=C_GREEN, alpha=0.9))

# (c) Error decomposition: EA error vs FDFD discretization
ax = axes[2]
# Sort residuals
sorted_abs = np.sort(abs_res)[::-1] * 1e6
ax.barh(np.arange(50), sorted_abs, height=0.8, color=C_GREEN, alpha=0.7,
        label=f'EA−FDFD(res=20): mean={np.mean(abs_res)*1e6:.0f}×10⁻⁶')

# FDFD drift line
if 20 in results_by_res and 40 in results_by_res:
    ax.axvline(np.mean(drift)*1e6, color=C_BLUE, ls='--', lw=1.5,
               label=f'FDFD drift (20→40): {np.mean(drift)*1e6:.0f}×10⁻⁶')

ax.set_xlabel('|Δω|  (×10⁻⁶  c/a)', fontsize=10)
ax.set_ylabel('Mode rank (sorted by residual)', fontsize=10)
ax.set_title('(c)  EA error vs FDFD grid error', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.invert_yaxis()

fig.suptitle(f'Relative Error Analysis  |  θ = {theta_deg:.2f}°  |  50 envelope modes vs FDFD(res=40)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_relative_errors.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_relative_errors.pdf'), bbox_inches='tight')
print("Saved fig_relative_errors")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Spectral structure — DOS, CDF, gaps
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) DOS overlay via KDE
ax = axes[0, 0]
kde_bw = env_spacing * 0.7  # KDE bandwidth slightly below mode spacing
x_eval = np.linspace(env_freqs.min() - 3*env_spacing,
                     env_freqs.max() + 3*env_spacing, 500)

kde_env = gaussian_kde(env_freqs, bw_method=kde_bw / np.std(env_freqs))
kde_fdfd_matched = gaussian_kde(matched_fdfd, bw_method=kde_bw / np.std(matched_fdfd))
kde_fdfd_full = gaussian_kde(fdfd_in_env, bw_method=kde_bw / np.std(fdfd_in_env))

ax.plot(x_eval * 1e3, kde_env(x_eval), '-', color=C_RED, lw=2, label=f'Envelope ({len(env_freqs)})')
ax.plot(x_eval * 1e3, kde_fdfd_matched(x_eval), '--', color=C_GREEN, lw=2,
        label=f'FDFD matched ({len(matched_fdfd)})')
ax.plot(x_eval * 1e3, kde_fdfd_full(x_eval), ':', color=C_BLUE, lw=1.5,
        label=f'FDFD all in window ({len(fdfd_in_env)})')
ax.set_xlabel('ω  (×10⁻³ c/a)', fontsize=10)
ax.set_ylabel('KDE density', fontsize=10)
ax.set_title('(a)  Density of States (KDE)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# (b) CDF comparison
ax = axes[0, 1]
env_sorted = np.sort(env_freqs)
fdfd_sorted = np.sort(matched_fdfd)
fdfd_full_sorted = np.sort(fdfd_in_env)

n_e = len(env_sorted)
n_f = len(fdfd_sorted)
n_ff = len(fdfd_full_sorted)

ax.step(env_sorted * 1e3, np.arange(1, n_e+1)/n_e, where='post',
        color=C_RED, lw=2, label='Envelope')
ax.step(fdfd_sorted * 1e3, np.arange(1, n_f+1)/n_f, where='post',
        color=C_GREEN, lw=2, ls='--', label='FDFD matched')
ax.step(fdfd_full_sorted * 1e3, np.arange(1, n_ff+1)/n_ff, where='post',
        color=C_BLUE, lw=1.5, ls=':', label='FDFD all in window')

# Annotate KS statistic
ax.text(0.97, 0.03,
    f'KS (matched): {ks_stat:.4f}, p = {ks_pval:.3f}\n'
    f'KS (full): {ks_stat_full:.4f}, p = {ks_pval_full:.3f}',
    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
    bbox=dict(facecolor='white', edgecolor='#CBD5E1', alpha=0.9))

ax.set_xlabel('ω  (×10⁻³ c/a)', fontsize=10)
ax.set_ylabel('CDF', fontsize=10)
ax.set_title('(b)  Cumulative Distribution Function', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# (c) Gap structure comparison — tick marks
ax = axes[1, 0]
for i, f in enumerate(env_sorted):
    ax.plot([f*1e3, f*1e3], [0.6, 1.0], '-', color=C_RED, lw=0.8, alpha=0.7)
for i, f in enumerate(fdfd_sorted):
    ax.plot([f*1e3, f*1e3], [0.0, 0.4], '-', color=C_GREEN, lw=0.8, alpha=0.7)

# Mark significant gaps
for g in env_gaps:
    ax.axvspan(g['center']*1e3 - g['size']*1e3/2, g['center']*1e3 + g['size']*1e3/2,
               ymin=0.6, ymax=1.0, alpha=0.2, color=C_ORANGE)
for g in fdfd_matched_gaps:
    ax.axvspan(g['center']*1e3 - g['size']*1e3/2, g['center']*1e3 + g['size']*1e3/2,
               ymin=0.0, ymax=0.4, alpha=0.2, color=C_ORANGE)

ax.axhline(0.5, color='gray', lw=0.5, ls='-')
ax.set_yticks([0.2, 0.8])
ax.set_yticklabels(['FDFD\n(matched)', 'Envelope'], fontsize=9, fontweight='bold')
ax.set_xlabel('ω  (×10⁻³ c/a)', fontsize=10)
ax.set_title(f'(c)  Gap structure  ({len(env_gaps)} EA gaps, {len(fdfd_matched_gaps)} FDFD gaps)',
             fontsize=11, fontweight='bold')

# (d) Bandwidth and spectral moments summary
ax = axes[1, 1]
ax.axis('off')

summary = (
    f"━━━━━  SPECTRAL STRUCTURE SUMMARY  ━━━━━\n"
    f"\n"
    f"Bandwidth:\n"
    f"  EA:   {env_bw*1e3:.4f} ×10⁻³ c/a\n"
    f"  FDFD: {fdfd_matched_bw*1e3:.4f} ×10⁻³ c/a\n"
    f"  Ratio: {env_bw/fdfd_matched_bw:.4f}\n"
    f"\n"
    f"Spectral center:\n"
    f"  EA:   {np.mean(env_freqs):.6f}\n"
    f"  FDFD: {np.mean(matched_fdfd):.6f}\n"
    f"  Δ:    {abs(np.mean(env_freqs)-np.mean(matched_fdfd)):.2e}\n"
    f"\n"
    f"KS test (EA vs FDFD matched):\n"
    f"  D = {ks_stat:.4f}, p = {ks_pval:.3f}\n"
    f"  → {'Cannot reject' if ks_pval > 0.05 else 'Reject'} same distribution\n"
    f"\n"
    f"Relative error |Δω|/ω:\n"
    f"  mean = {np.mean(rel_diff):.2e}\n"
    f"  max  = {np.max(rel_diff):.2e}\n"
    f"\n"
    f"FDFD discretization error:\n"
    f"  {np.mean(drift)*1e6:.0f}×10⁻⁶ (res=16→20 drift)\n"
    f"  EA−FDFD residual: {np.mean(abs_res)*1e6:.0f}×10⁻⁶\n"
    f"  → FDFD grid error {'dominates' if np.mean(drift) > np.mean(abs_res) else 'is comparable'}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC',
                  edgecolor='#64748B', alpha=0.95))

fig.suptitle(f'Spectral Structure Comparison  |  θ = {theta_deg:.2f}°  |  '
             f'EA (2-band, 33K DOF) vs FDFD (4.2M DOF, res=40)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, 'fig_spectral_structure.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_spectral_structure.pdf'), bbox_inches='tight')
print("Saved fig_spectral_structure")
plt.close('all')
