"""
Phase 1 & 3: Relative difference analysis + spectral structure metrics.
Computes |Δω|/ω, |Δω|/spacing, bandwidth ratio, DOS, CDF/KS, gap structure.
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, gaussian_kde

out_dir = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
fdfd_all = np.sort(np.load(os.path.join(out_dir, 'fdfd_dirac_m30_n29_res16_v2.npz'))['freqs'])

with open('/home/renlephy/msl/research/moire_envelope/runsV3/'
          'thesis_honeycomb_K_b1_20260307_171424/'
          'eta_sweep_20260310_191610/sweep_results.json') as f:
    env_bh = json.load(f)[0]
env_freqs = np.sort(env_bh['omega_ref'] + np.array(env_bh['eigenvalues']))
omega_ref = env_bh['omega_ref']
theta_deg = env_bh['theta_deg']
eta = env_bh['eta']

env_min, env_max = env_freqs.min(), env_freqs.max()
env_bw = env_max - env_min
env_spacing = np.mean(np.diff(env_freqs))

# ═══════════════════════════════════════════════════════════════
# Hungarian matching
# ═══════════════════════════════════════════════════════════════
margin = 0.001
fdfd_mask = (fdfd_all >= env_min - margin) & (fdfd_all <= env_max + margin)
fdfd_w = fdfd_all[fdfd_mask]
fdfd_w_global_idx = np.where(fdfd_mask)[0]

cost = np.abs(env_freqs[:, None] - fdfd_w[None, :])
row_ind, col_ind = linear_sum_assignment(cost)
matched_fdfd = fdfd_w[col_ind]
residuals = env_freqs[row_ind] - matched_fdfd
abs_res = np.abs(residuals)

# Unmatched FDFD
matched_global = set(fdfd_w_global_idx[col_ind])
fdfd_in_env_idx = np.where((fdfd_all >= env_min) & (fdfd_all <= env_max))[0]
fdfd_in_env = fdfd_all[fdfd_in_env_idx]
unmatched_fdfd = np.array([fdfd_all[i] for i in fdfd_in_env_idx if i not in matched_global])

N_env = len(env_freqs)
N_fdfd_in_env = len(fdfd_in_env)
N_unmatched = len(unmatched_fdfd)

print(f"θ = {theta_deg:.4f}°, η = {eta:.6f}")
print(f"Envelope: {N_env} modes, BW = {env_bw:.6f} c/a")
print(f"FDFD in envelope window: {N_fdfd_in_env} ({N_unmatched} unmatched)")
print(f"Mean spacing: {env_spacing*1e6:.1f}×10⁻⁶")

# ═══════════════════════════════════════════════════════════════
# Phase 1: Relative metrics
# ═══════════════════════════════════════════════════════════════
rel_err = abs_res / matched_fdfd            # |Δω|/ω_FDFD
spacing_ratio = abs_res / env_spacing       # |Δω| / <spacing>
local_spacing = np.diff(env_freqs)
# for each mode, use the smaller of the two adjacent spacings
local_sp = np.minimum(
    np.concatenate([[local_spacing[0]], local_spacing]),
    np.concatenate([local_spacing, [local_spacing[-1]]])
)
local_spacing_ratio = abs_res / local_sp    # |Δω| / local_spacing

print(f"\n--- Relative metrics ---")
print(f"|Δω|/ω:        mean={np.mean(rel_err):.2e}, max={np.max(rel_err):.2e}")
print(f"|Δω|/<spacing>: mean={np.mean(spacing_ratio):.3f}, max={np.max(spacing_ratio):.3f}")
print(f"|Δω|/local_sp:  mean={np.mean(local_spacing_ratio):.3f}, max={np.max(local_spacing_ratio):.3f}")
print(f"Modes within 0.5 spacings: {np.sum(spacing_ratio < 0.5)}/50")
print(f"Modes within 1.0 spacings: {np.sum(spacing_ratio < 1.0)}/50")
print(f"Modes within 2.0 spacings: {np.sum(spacing_ratio < 2.0)}/50")

# ═══════════════════════════════════════════════════════════════
# Phase 3a: Bandwidth comparison
# ═══════════════════════════════════════════════════════════════
bw_env = env_bw
bw_fdfd_matched = matched_fdfd.max() - matched_fdfd.min()
bw_fdfd_full = fdfd_in_env.max() - fdfd_in_env.min()
bw_ratio_matched = bw_env / bw_fdfd_matched
bw_ratio_full = bw_env / bw_fdfd_full

print(f"\n--- Bandwidth ---")
print(f"BW_EA            = {bw_env*1e3:.4f} ×10⁻³")
print(f"BW_FDFD(matched) = {bw_fdfd_matched*1e3:.4f} ×10⁻³")
print(f"BW_FDFD(all {N_fdfd_in_env})  = {bw_fdfd_full*1e3:.4f} ×10⁻³")
print(f"Ratio (matched)  = {bw_ratio_matched:.4f}")
print(f"Ratio (all)      = {bw_env/bw_fdfd_full:.4f}")

# ═══════════════════════════════════════════════════════════════
# Phase 3b: Spectral moments
# ═══════════════════════════════════════════════════════════════
mean_env = np.mean(env_freqs)
mean_fdfd = np.mean(matched_fdfd)
std_env = np.std(env_freqs)
std_fdfd = np.std(matched_fdfd)

print(f"\n--- Spectral moments ---")
print(f"Mean:  EA={mean_env:.6f}, FDFD={mean_fdfd:.6f}, Δ={mean_env-mean_fdfd:.2e}")
print(f"Std:   EA={std_env:.6f}, FDFD={std_fdfd:.6f}, ratio={std_env/std_fdfd:.4f}")

# ═══════════════════════════════════════════════════════════════
# Phase 3c: KS test
# ═══════════════════════════════════════════════════════════════
ks_stat, ks_p = ks_2samp(env_freqs, matched_fdfd)
print(f"\n--- KS test (EA vs matched FDFD) ---")
print(f"KS statistic = {ks_stat:.4f}, p-value = {ks_p:.4f}")
print(f"(p > 0.05 → cannot reject same distribution)")

# ═══════════════════════════════════════════════════════════════
# Phase 3d: Gap structure
# ═══════════════════════════════════════════════════════════════
def find_gaps(freqs, threshold_factor=2.0):
    sp = np.diff(freqs)
    median_sp = np.median(sp)
    gap_mask = sp > threshold_factor * median_sp
    gaps = []
    for i in np.where(gap_mask)[0]:
        gaps.append({
            'position': 0.5 * (freqs[i] + freqs[i+1]),
            'size': sp[i],
            'ratio': sp[i] / median_sp,
        })
    return gaps, median_sp

gaps_env, med_sp_env = find_gaps(env_freqs)
gaps_fdfd, med_sp_fdfd = find_gaps(matched_fdfd)

print(f"\n--- Gap structure (>2× median spacing) ---")
print(f"EA:   {len(gaps_env)} gaps (median spacing = {med_sp_env*1e6:.1f}×10⁻⁶)")
for g in gaps_env:
    print(f"  ω={g['position']:.6f}, size={g['size']*1e6:.1f}×10⁻⁶ ({g['ratio']:.1f}× median)")
print(f"FDFD: {len(gaps_fdfd)} gaps (median spacing = {med_sp_fdfd*1e6:.1f}×10⁻⁶)")
for g in gaps_fdfd:
    print(f"  ω={g['position']:.6f}, size={g['size']*1e6:.1f}×10⁻⁶ ({g['ratio']:.1f}× median)")

# Match gaps
if gaps_env and gaps_fdfd:
    gap_pos_env = np.array([g['position'] for g in gaps_env])
    gap_pos_fdfd = np.array([g['position'] for g in gaps_fdfd])
    print(f"\nGap position agreement:")
    for i, ge in enumerate(gaps_env):
        dist = np.min(np.abs(gap_pos_fdfd - ge['position']))
        match_idx = np.argmin(np.abs(gap_pos_fdfd - ge['position']))
        print(f"  EA gap {i} at {ge['position']:.6f} → nearest FDFD gap at {gap_pos_fdfd[match_idx]:.6f} (Δ={dist*1e6:.1f}×10⁻⁶)")

# ═══════════════════════════════════════════════════════════════
# FIGURE: Relative analysis + spectral structure (6-panel)
# ═══════════════════════════════════════════════════════════════
C_ENV = '#DC2626'
C_FDFD = '#2563EB'
C_MATCH = '#16A34A'
C_EXTRA = '#7C3AED'

fig, axes = plt.subplots(2, 3, figsize=(17, 10))

# ── (a) |Δω|/ω vs mode index ──
ax = axes[0, 0]
ax.scatter(np.arange(N_env), rel_err * 1e4, s=30, c=C_MATCH,
           edgecolors='white', linewidths=0.3, zorder=5)
ax.axhline(np.mean(rel_err) * 1e4, color=C_ENV, ls='--', lw=1.2,
           label=f'mean = {np.mean(rel_err)*1e4:.2f}×10⁻⁴')
ax.set_xlabel('Mode index (sorted by ω)', fontsize=10)
ax.set_ylabel('|Δω| / ω_FDFD  (×10⁻⁴)', fontsize=10)
ax.set_title('(a)  Relative frequency error', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)

# ── (b) |Δω|/spacing histogram ──
ax = axes[0, 1]
bins = np.linspace(0, max(4, np.max(spacing_ratio) + 0.5), 25)
colors_hist = [C_MATCH if b < 1.0 else '#EAB308' if b < 2.0 else '#F97316'
               for b in 0.5*(bins[:-1]+bins[1:])]
n_hist, _, patches = ax.hist(spacing_ratio, bins=bins, edgecolor='white', linewidth=0.5)
for patch, c in zip(patches, colors_hist):
    patch.set_facecolor(c)
ax.axvline(1.0, color='black', ls='--', lw=1.5, alpha=0.5, label='1 spacing')
ax.axvline(np.mean(spacing_ratio), color=C_ENV, ls='--', lw=1.2,
           label=f'mean = {np.mean(spacing_ratio):.2f}')
pct_within_1 = np.sum(spacing_ratio < 1.0) / N_env * 100
ax.text(0.97, 0.97, f'{pct_within_1:.0f}% within\n1 spacing',
        transform=ax.transAxes, ha='right', va='top', fontsize=10, fontweight='bold',
        bbox=dict(facecolor='#F0FDF4', edgecolor=C_MATCH, alpha=0.9))
ax.set_xlabel('|Δω| / mean mode spacing', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('(b)  Residual in units of mode spacing', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)

# ── (c) Bandwidth comparison ──
ax = axes[0, 2]
categories = ['Envelope\n(2-band EA)', 'FDFD\n(matched)', f'FDFD\n(all {N_fdfd_in_env})']
bws = [bw_env * 1e3, bw_fdfd_matched * 1e3, bw_fdfd_full * 1e3]
colors_bw = [C_ENV, C_MATCH, C_FDFD]
bars = ax.bar(categories, bws, color=colors_bw, edgecolor='white', width=0.6, alpha=0.85)
for bar, bw in zip(bars, bws):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bw:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Bandwidth  (×10⁻³  c/a)', fontsize=10)
ax.set_title(f'(c)  Bandwidth: ratio = {bw_ratio_matched:.3f}', fontsize=11, fontweight='bold')
ax.set_ylim(0, max(bws) * 1.25)

# ── (d) DOS comparison (KDE) ──
ax = axes[1, 0]
kde_bw = env_spacing * 0.8  # KDE bandwidth ≈ mode spacing
x_grid = np.linspace(env_min - 3*env_spacing, env_max + 3*env_spacing, 500)

kde_env = gaussian_kde(env_freqs, bw_method=kde_bw / np.std(env_freqs))
kde_fdfd_matched = gaussian_kde(matched_fdfd, bw_method=kde_bw / np.std(matched_fdfd))

ax.fill_between(x_grid * 1e3, kde_env(x_grid), alpha=0.3, color=C_ENV, label='Envelope')
ax.plot(x_grid * 1e3, kde_env(x_grid), color=C_ENV, lw=1.5)
ax.fill_between(x_grid * 1e3, kde_fdfd_matched(x_grid), alpha=0.2, color=C_FDFD, label='FDFD (matched)')
ax.plot(x_grid * 1e3, kde_fdfd_matched(x_grid), color=C_FDFD, lw=1.5, ls='--')

# Also show full FDFD
if N_fdfd_in_env > N_env:
    kde_fdfd_full = gaussian_kde(fdfd_in_env, bw_method=kde_bw / np.std(fdfd_in_env))
    ax.plot(x_grid * 1e3, kde_fdfd_full(x_grid), color=C_EXTRA, lw=1.0, ls=':', alpha=0.6,
            label=f'FDFD (all {N_fdfd_in_env})')

ax.set_xlabel('ω  (×10⁻³  c/a)', fontsize=10)
ax.set_ylabel('Density of States (KDE)', fontsize=10)
ax.set_title('(d)  Spectral density', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# ── (e) CDF comparison ──
ax = axes[1, 1]
cdf_env = np.arange(1, N_env+1) / N_env
cdf_fdfd = np.arange(1, N_env+1) / N_env  # matched has same count

ax.step(env_freqs * 1e3, cdf_env, where='post', color=C_ENV, lw=2, label='Envelope')
ax.step(np.sort(matched_fdfd) * 1e3, cdf_fdfd, where='post', color=C_FDFD, lw=2, ls='--',
        label='FDFD (matched)')

# KS band
ks_band_upper = cdf_env + ks_stat
ks_band_lower = cdf_env - ks_stat
ax.fill_between(env_freqs * 1e3, ks_band_lower, ks_band_upper,
                alpha=0.1, color='gray', label=f'KS band (D={ks_stat:.3f})')

ax.set_xlabel('ω  (×10⁻³  c/a)', fontsize=10)
ax.set_ylabel('Cumulative fraction', fontsize=10)
ax.set_title(f'(e)  CDF: KS = {ks_stat:.3f}, p = {ks_p:.3f}', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

# ── (f) Gap structure ──
ax = axes[1, 2]
# Show spacings in both spectra
sp_env = np.diff(env_freqs) * 1e6
sp_fdfd = np.diff(np.sort(matched_fdfd)) * 1e6

ax.plot(np.arange(len(sp_env)), sp_env, 'o-', ms=3, lw=0.8, color=C_ENV, alpha=0.7, label='Envelope')
ax.plot(np.arange(len(sp_fdfd)), sp_fdfd, 's-', ms=3, lw=0.8, color=C_FDFD, alpha=0.7, label='FDFD (matched)')
ax.axhline(np.median(sp_env), color=C_ENV, ls=':', lw=0.8, alpha=0.5)
ax.axhline(2 * np.median(sp_env), color='gray', ls='--', lw=0.8, alpha=0.4, label='2× median')

# Mark significant gaps
for g in gaps_env:
    idx_g = np.argmin(np.abs(0.5*(env_freqs[:-1]+env_freqs[1:]) - g['position']))
    ax.plot(idx_g, sp_env[idx_g], 'v', ms=8, color=C_ENV, zorder=10)
for g in gaps_fdfd:
    idx_g = np.argmin(np.abs(0.5*(np.sort(matched_fdfd)[:-1]+np.sort(matched_fdfd)[1:]) - g['position']))
    ax.plot(idx_g, sp_fdfd[idx_g], '^', ms=8, color=C_FDFD, zorder=10)

ax.set_xlabel('Mode pair index', fontsize=10)
ax.set_ylabel('Spacing  (×10⁻⁶  c/a)', fontsize=10)
ax.set_title(f'(f)  Gap structure: {len(gaps_env)} EA / {len(gaps_fdfd)} FDFD gaps', fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5)

fig.suptitle(f'Spectral Structure Validation  |  θ = {theta_deg:.2f}°  |  '
             f'EA (33K DOF) vs FDFD (669K DOF)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_spectral_validation.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_spectral_validation.pdf'), bbox_inches='tight')
print("\nSaved fig_spectral_validation.{png,pdf}")
plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"SPECTRAL STRUCTURE VALIDATION SUMMARY")
print(f"{'='*60}")
print(f"{'Metric':<35} {'Value':>15}")
print(f"{'-'*50}")
print(f"{'θ':<35} {f'{theta_deg:.4f}°':>15}")
print(f"{'η':<35} {f'{eta:.6f}':>15}")
print(f"{'EA modes':<35} {f'{N_env}':>15}")
print(f"{'FDFD modes (in window)':<35} {f'{N_fdfd_in_env}':>15}")
print(f"{'Unmatched FDFD (other bands)':<35} {f'{N_unmatched}':>15}")
print(f"{'-'*50}")
print(f"{'|Δω|/ω (mean)':<35} {f'{np.mean(rel_err):.2e}':>15}")
print(f"{'|Δω|/ω (max)':<35} {f'{np.max(rel_err):.2e}':>15}")
print(f"{'|Δω|/spacing (mean)':<35} {f'{np.mean(spacing_ratio):.3f}':>15}")
print(f"{'|Δω|/spacing (max)':<35} {f'{np.max(spacing_ratio):.3f}':>15}")
print(f"{'Modes within 1 spacing':<35} {f'{int(np.sum(spacing_ratio<1))}/50':>15}")
print(f"{'-'*50}")
print(f"{'BW ratio (matched)':<35} {f'{bw_ratio_matched:.4f}':>15}")
print(f"{'Std ratio':<35} {f'{std_env/std_fdfd:.4f}':>15}")
print(f"{'KS statistic':<35} {f'{ks_stat:.4f}':>15}")
print(f"{'KS p-value':<35} {f'{ks_p:.4f}':>15}")
print(f"{'EA gaps (>2× median)':<35} {f'{len(gaps_env)}':>15}")
print(f"{'FDFD gaps (>2× median)':<35} {f'{len(gaps_fdfd)}':>15}")
print(f"{'='*60}")
