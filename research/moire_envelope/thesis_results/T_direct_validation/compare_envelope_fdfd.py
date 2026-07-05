"""
Direct comparison of envelope approximation vs FDFD supercell solve
at the Γ-point of the moiré BZ for honeycomb TM, θ=4.408°.

Comparison strategy (from apples_to_apples.md):
- Miniband bandwidth
- Cluster/gap structure
- Spectral density in target window
- NOT individual eigenvalue matching (fragile in dense spectra)
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load envelope data ──
sweep_file = os.path.join(OUT_DIR, '..', '..', 'runsV3',
    'thesis_honeycomb_K_b1_20260307_171424',
    'eta_sweep_20260310_162339', 'sweep_results.json')
with open(sweep_file) as f:
    sweep = json.load(f)
env = sweep[0]
env_evals = np.array(env['eigenvalues'])
env_omega_ref = env['omega_ref']
env_freqs = env_omega_ref + env_evals  # physical frequencies
env_theta = env['theta_deg']

# ── Load FDFD data ──
fdfd_file = os.path.join(OUT_DIR, 'fdfd_dirac_m8_n7_res20.npz')
fdfd = np.load(fdfd_file)
fdfd_freqs_all = fdfd['freqs']
fdfd_theta = float(fdfd['theta_deg'])

# ── Focus on the overlapping spectral window ──
# Envelope range
env_min, env_max = env_freqs.min(), env_freqs.max()
# Widen window slightly for FDFD
window_lo = env_min - 0.005
window_hi = env_max + 0.005

fdfd_mask = (fdfd_freqs_all > window_lo) & (fdfd_freqs_all < window_hi)
fdfd_freqs = fdfd_freqs_all[fdfd_mask]

print(f"{'='*70}")
print(f"ENVELOPE vs FDFD COMPARISON")
print(f"Honeycomb TM, θ = {env_theta:.3f}°, (m,n)=(8,7), N_cells=169")
print(f"{'='*70}")
print()

print(f"Envelope approximation:")
print(f"  {len(env_freqs)} modes")
print(f"  ω_ref = {env_omega_ref:.6f}")
print(f"  Physical freq range: [{env_min:.6f}, {env_max:.6f}]")
print(f"  Total bandwidth: {env_max - env_min:.6f}")
print()

print(f"FDFD supercell solve:")
print(f"  {len(fdfd_freqs)} modes in window (from {len(fdfd_freqs_all)} total)")
print(f"  Physical freq range: [{fdfd_freqs.min():.6f}, {fdfd_freqs.max():.6f}]")
print(f"  Total bandwidth: {fdfd_freqs.max() - fdfd_freqs.min():.6f}")
print()

# ── Comparison 1: Overall bandwidth ──
bw_env = env_max - env_min
bw_fdfd = fdfd_freqs.max() - fdfd_freqs.min()
bw_ratio = bw_env / bw_fdfd
print(f"Bandwidth comparison:")
print(f"  Envelope: {bw_env:.6f}")
print(f"  FDFD:     {bw_fdfd:.6f}")
print(f"  Ratio:    {bw_ratio:.4f}")
print()

# ── Comparison 2: Spectral moments ──
env_mean = np.mean(env_freqs)
env_std = np.std(env_freqs)
fdfd_mean = np.mean(fdfd_freqs)
fdfd_std = np.std(fdfd_freqs)
print(f"Spectral statistics:")
print(f"  Mean:  Env={env_mean:.6f}, FDFD={fdfd_mean:.6f}, Δ={abs(env_mean-fdfd_mean):.6f}")
print(f"  Std:   Env={env_std:.6f}, FDFD={fdfd_std:.6f}, ratio={env_std/fdfd_std:.4f}")
print()

# ── Comparison 3: Gap structure ──
def find_gaps(freqs, min_gap_ratio=2.0):
    """Find significant gaps (larger than min_gap_ratio × median spacing)."""
    sorted_f = np.sort(freqs)
    spacings = np.diff(sorted_f)
    median_sp = np.median(spacings)
    gaps = []
    for i, sp in enumerate(spacings):
        if sp > min_gap_ratio * median_sp:
            gaps.append({
                'below': sorted_f[i],
                'above': sorted_f[i+1],
                'gap': sp,
                'center': 0.5*(sorted_f[i] + sorted_f[i+1]),
                'index': i,
            })
    return gaps

env_gaps = find_gaps(env_freqs, min_gap_ratio=2.0)
fdfd_gaps = find_gaps(fdfd_freqs, min_gap_ratio=2.0)

print(f"Gap structure:")
print(f"  Envelope: {len(env_gaps)} significant gaps")
for g in env_gaps:
    print(f"    Δω={g['gap']:.6f} at ω={g['center']:.6f} (modes {g['index']}-{g['index']+1})")
print(f"  FDFD: {len(fdfd_gaps)} significant gaps")
for g in fdfd_gaps:
    print(f"    Δω={g['gap']:.6f} at ω={g['center']:.6f} (modes {g['index']}-{g['index']+1})")
print()

# ── Comparison 4: Spectral density (integrated DOS) ──
bins = np.linspace(window_lo, window_hi, 60)
env_hist, _ = np.histogram(env_freqs, bins=bins)
fdfd_hist, _ = np.histogram(fdfd_freqs, bins=bins)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Normalize to same total count
env_hist_norm = env_hist / len(env_freqs)
fdfd_hist_norm = fdfd_hist / len(fdfd_freqs)

# Similarity: correlation coefficient
corr = np.corrcoef(env_hist_norm, fdfd_hist_norm)[0, 1]
print(f"Spectral density correlation: {corr:.4f}")

# ── Comparison 5: Cumulative distribution comparison ──
# Both should show similar "staircase" shapes if they agree
env_sorted = np.sort(env_freqs)
fdfd_sorted = np.sort(fdfd_freqs)

# Normalize to [0, 1] cumulative fraction
env_cdf = np.arange(1, len(env_sorted)+1) / len(env_sorted)
fdfd_cdf = np.arange(1, len(fdfd_sorted)+1) / len(fdfd_sorted)

# Interpolate both CDFs onto a common frequency grid
f_grid = np.linspace(window_lo, window_hi, 500)
env_cdf_interp = np.interp(f_grid, env_sorted, env_cdf, left=0, right=1)
fdfd_cdf_interp = np.interp(f_grid, fdfd_sorted, fdfd_cdf, left=0, right=1)

# KS-like statistic
ks_stat = np.max(np.abs(env_cdf_interp - fdfd_cdf_interp))
print(f"KS-like statistic (max CDF difference): {ks_stat:.4f}")
print()

# ═══════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════

# ── Figure 1: Side-by-side eigenvalue comparison ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# (a) Eigenvalue spectrum comparison
ax = axes[0]
ax.plot(env_freqs, np.arange(len(env_freqs)), 'ro-', ms=4, lw=1, label='Envelope', alpha=0.8)
ax.plot(fdfd_freqs, np.arange(len(fdfd_freqs)), 'bs-', ms=3, lw=1, label='FDFD', alpha=0.8)
ax.axvline(env_omega_ref, color='gray', ls='--', lw=0.8, label=f'ω_ref={env_omega_ref:.4f}')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Mode index')
ax.set_title('(a) Ordered eigenvalue spectrum')
ax.legend(fontsize=9)
ax.set_xlim(window_lo, window_hi)

# (b) Spectral density comparison
ax = axes[1]
ax.step(bin_centers, env_hist, where='mid', color='red', lw=1.5, label=f'Envelope ({len(env_freqs)} modes)')
ax.step(bin_centers, fdfd_hist, where='mid', color='blue', lw=1.5, label=f'FDFD ({len(fdfd_freqs)} modes)')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Count per bin')
ax.set_title(f'(b) Spectral density (corr={corr:.3f})')
ax.legend(fontsize=9)

# (c) Cumulative distribution
ax = axes[2]
ax.step(env_sorted, env_cdf, where='post', color='red', lw=1.5, label='Envelope')
ax.step(fdfd_sorted, fdfd_cdf, where='post', color='blue', lw=1.5, label='FDFD')
ax.set_xlabel('ω (c/a)')
ax.set_ylabel('Cumulative fraction')
ax.set_title(f'(c) CDF (KS={ks_stat:.3f})')
ax.legend(fontsize=9)

fig.suptitle(f'Envelope vs FDFD: Honeycomb TM, θ={env_theta:.2f}°, Γ_m point',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'comparison_envelope_fdfd.png'), dpi=150)
print(f"Saved comparison_envelope_fdfd.png")

# ── Figure 2: Detailed gap/cluster comparison ──
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# Plot as horizontal lines
for i, f in enumerate(env_freqs):
    ax2.plot([0.0, 0.4], [f, f], 'r-', lw=1.2, alpha=0.7)
for i, f in enumerate(fdfd_freqs):
    ax2.plot([0.6, 1.0], [f, f], 'b-', lw=1.2, alpha=0.7)

# Highlight gaps
for g in env_gaps:
    ax2.axhspan(g['below'], g['above'], xmin=0, xmax=0.47, color='red', alpha=0.1)
for g in fdfd_gaps:
    ax2.axhspan(g['below'], g['above'], xmin=0.53, xmax=1, color='blue', alpha=0.1)

ax2.axhline(env_omega_ref, color='green', ls='--', lw=1, alpha=0.6,
            label=f'ω_ref={env_omega_ref:.4f}')

ax2.set_xlim(-0.1, 1.1)
ax2.set_xticks([0.2, 0.8])
ax2.set_xticklabels(['Envelope', 'FDFD'], fontsize=12)
ax2.set_ylabel('ω (c/a)')
ax2.set_title(f'Eigenvalue level diagram: Envelope vs FDFD at Γ_m\n'
              f'θ={env_theta:.2f}°, Honeycomb TM, (m,n)=(8,7)')
ax2.legend(loc='upper right')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'comparison_level_diagram.png'), dpi=150)
print(f"Saved comparison_level_diagram.png")

# ── Figure 3: Matched pairs analysis ──
# Try to match each envelope mode to nearest FDFD mode
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))

# Nearest-neighbor matching: for each envelope mode, find closest FDFD
matched_fdfd = []
for ef in env_freqs:
    idx = np.argmin(np.abs(fdfd_freqs - ef))
    matched_fdfd.append(fdfd_freqs[idx])
matched_fdfd = np.array(matched_fdfd)
residuals = env_freqs - matched_fdfd

ax = axes3[0]
ax.scatter(env_freqs, matched_fdfd, s=15, c='purple', alpha=0.7)
lims = [window_lo, window_hi]
ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
ax.set_xlabel('Envelope ω (c/a)')
ax.set_ylabel('Nearest FDFD ω (c/a)')
ax.set_title('(a) Nearest-neighbor matching')
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

ax = axes3[1]
ax.plot(env_freqs, residuals * 1000, 'o', ms=4, alpha=0.7)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('Envelope ω (c/a)')
ax.set_ylabel('Residual (ω_env - ω_fdfd) × 10³')
ax.set_title(f'(b) Residuals: mean={np.mean(np.abs(residuals))*1000:.2f}×10⁻³, '
             f'max={np.max(np.abs(residuals))*1000:.2f}×10⁻³')

fig3.suptitle(f'Nearest-neighbor matching: Envelope → FDFD', fontsize=12)
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'comparison_matching.png'), dpi=150)
print(f"Saved comparison_matching.png")

plt.close('all')

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"Angle: θ = {env_theta:.3f}°")
print(f"Envelope: {len(env_freqs)} modes, freq range [{env_min:.6f}, {env_max:.6f}]")
print(f"FDFD:     {len(fdfd_freqs)} modes in window, freq range [{fdfd_freqs.min():.6f}, {fdfd_freqs.max():.6f}]")
print(f"")
print(f"Bandwidth:  Env={bw_env:.6f}, FDFD={bw_fdfd:.6f}, ratio={bw_ratio:.4f}")
print(f"Mean freq:  Env={env_mean:.6f}, FDFD={fdfd_mean:.6f}, shift={fdfd_mean-env_mean:+.6f}")
print(f"Spectral density corr: {corr:.4f}")
print(f"CDF KS statistic:     {ks_stat:.4f}")
print(f"Nearest-neighbor matching:")
print(f"  Mean |residual|: {np.mean(np.abs(residuals)):.6f}")
print(f"  Max  |residual|: {np.max(np.abs(residuals)):.6f}")
print(f"  RMS  residual:   {np.sqrt(np.mean(residuals**2)):.6f}")
