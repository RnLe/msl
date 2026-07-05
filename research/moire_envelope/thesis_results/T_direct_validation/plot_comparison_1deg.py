"""
Create transparent, expressive plots for the envelope vs FDFD comparison
at θ ≈ 1.1° for the honeycomb TM bilayer.

Key story:
  - The envelope approximation is a 2-band effective model;
    it only sees 2 of the many monolayer bands folded into the mBZ.
  - The FDFD solves the full Maxwell equations on the moiré supercell;
    it sees ALL folded bands.
  - The envelope modes form a SUBSET of the FDFD modes.
  - Nearest-neighbor matching shows each envelope mode sits on top of
    an FDFD mode to within 16×10⁻⁶ c/a (0.5% of envelope BW).
"""
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

out_dir = os.path.dirname(os.path.abspath(__file__))

# ─── Load data ─────────────────────────────────────────────────
fdfd_data = np.load(os.path.join(out_dir, 'fdfd_dirac_m30_n29_res16_v2.npz'))
fdfd_all = np.sort(fdfd_data['freqs'])

with open('/home/renlephy/msl/research/moire_envelope/runsV3/'
          'thesis_honeycomb_K_b1_20260307_171424/'
          'eta_sweep_20260310_181650/sweep_results.json') as f:
    env_data = json.load(f)[0]

env_evals = np.array(env_data['eigenvalues'])
env_omega_ref = env_data['omega_ref']
env_freqs = np.sort(env_omega_ref + env_evals)
theta_deg = env_data['theta_deg']
eta = env_data['eta']

# ─── Matching ──────────────────────────────────────────────────
matched_fdfd_idx = []
matched_fdfd_freq = []
for ef in env_freqs:
    idx = np.argmin(np.abs(fdfd_all - ef))
    matched_fdfd_idx.append(idx)
    matched_fdfd_freq.append(fdfd_all[idx])
matched_fdfd_freq = np.array(matched_fdfd_freq)
residuals = env_freqs - matched_fdfd_freq

# Identify unmatched FDFD modes in envelope window
env_min, env_max = env_freqs.min(), env_freqs.max()
fdfd_in_window = fdfd_all[(fdfd_all >= env_min) & (fdfd_all <= env_max)]
matched_set = set(matched_fdfd_idx)
fdfd_window_mask = (fdfd_all >= env_min) & (fdfd_all <= env_max)
fdfd_window_idx = np.where(fdfd_window_mask)[0]
unmatched_idx = [i for i in fdfd_window_idx if i not in matched_set]
unmatched_freqs = fdfd_all[unmatched_idx]

env_bw = env_max - env_min
mean_res = np.mean(np.abs(residuals))
max_res = np.max(np.abs(residuals))

print(f"θ = {theta_deg:.4f}°, η = {eta:.6f}")
print(f"Env: {len(env_freqs)} modes, [{env_min:.6f}, {env_max:.6f}], BW={env_bw:.6f}")
print(f"FDFD in window: {len(fdfd_in_window)} ({len(unmatched_freqs)} unmatched)")
print(f"NN residual: mean={mean_res:.7f}, max={max_res:.7f}")
print(f"Relative: mean={mean_res/env_bw*100:.2f}%, max={max_res/env_bw*100:.2f}%")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: The money plot — NN matching with connecting lines
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))

# Extend view slightly beyond envelope window
view_margin = 0.4 * env_bw
view_lo = env_min - view_margin
view_hi = env_max + view_margin
fdfd_view = fdfd_all[(fdfd_all >= view_lo) & (fdfd_all <= view_hi)]

# Plot ALL FDFD modes as blue ticks on the left
for f in fdfd_view:
    in_env = env_min <= f <= env_max
    c = '#3B82F6' if in_env else '#93C5FD'
    ax.plot([0.05, 0.35], [f, f], '-', color=c, lw=0.8, alpha=0.7 if in_env else 0.4)

# Plot envelope modes as red ticks on the right
for f in env_freqs:
    ax.plot([0.65, 0.95], [f, f], '-', color='#EF4444', lw=1.2, alpha=0.8)

# Draw connecting lines for matched pairs
for i, (ef, ff) in enumerate(zip(env_freqs, matched_fdfd_freq)):
    delta = abs(ef - ff)
    # Color by residual magnitude
    rel = delta / env_bw
    if rel < 0.005:
        c = '#22C55E'  # green = excellent
    elif rel < 0.01:
        c = '#EAB308'  # yellow = good
    else:
        c = '#F97316'  # orange = ok
    ax.plot([0.35, 0.65], [ff, ef], '-', color=c, lw=0.6, alpha=0.6)

# Mark unmatched FDFD modes  
for f in unmatched_freqs:
    ax.plot(0.2, f, 'x', color='#6366F1', ms=5, mew=1.2, alpha=0.6)

# Shading for envelope bandwidth
ax.axhspan(env_min, env_max, alpha=0.06, color='red', zorder=0)
ax.axhline(env_omega_ref, color='gray', ls=':', lw=1, alpha=0.5)

# Labels
ax.set_xlim(-0.05, 1.15)
ax.set_ylim(view_lo, view_hi)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['FDFD\n(full Maxwell)', 'Envelope\n(2-band EA)'], fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency  ω  (c/a)', fontsize=12)

# Stats annotation
stats_text = (
    f"Nearest-neighbor matching:\n"
    f"  mean |Δω| = {mean_res*1e6:.0f} × 10⁻⁶ c/a\n"
    f"  max  |Δω| = {max_res*1e6:.0f} × 10⁻⁶ c/a\n"
    f"  = {mean_res/env_bw*100:.1f}% of envelope BW\n\n"
    f"Envelope: {len(env_freqs)} modes\n"
    f"FDFD in window: {len(fdfd_in_window)} modes\n"
    f"  ({len(fdfd_in_window)-len(set(matched_fdfd_idx)&set(fdfd_window_idx))} from other folded bands)"
)
ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor='#CBD5E1'))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#EF4444', lw=2, label='Envelope mode'),
    Line2D([0], [0], color='#3B82F6', lw=2, label='FDFD mode (in env window)'),
    Line2D([0], [0], color='#93C5FD', lw=2, label='FDFD mode (outside env window)'),
    Line2D([0], [0], color='#22C55E', lw=1, label='Match (< 0.5% BW)'),
    Line2D([0], [0], marker='x', color='#6366F1', lw=0, ms=6, label='Unmatched FDFD (other bands)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)

ax.set_title(f'Nearest-Neighbor Matching:  Envelope ↔ FDFD\n'
             f'Honeycomb TM,  θ = {theta_deg:.2f}°,  (m,n) = (30,29),  Γ_m point',
             fontsize=13, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_nn_matching.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_nn_matching.pdf'), bbox_inches='tight')
print("Saved fig_nn_matching.png/pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Zoomed 1-to-1 frequency comparison + residuals
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})

# (a) ω_env vs ω_FDFD scatter — should lie on diagonal
ax = axes[0]
ax.scatter(env_freqs * 1e3, matched_fdfd_freq * 1e3, s=30, c='#7C3AED',
           edgecolors='white', linewidths=0.3, zorder=5, alpha=0.9)

# Perfect diagonal
lims = [env_min * 1e3 - 0.1, env_max * 1e3 + 0.1]
ax.plot(lims, lims, 'k-', lw=1.5, alpha=0.3, label='Perfect agreement')

# Shade ±max residual band around diagonal
x_diag = np.linspace(lims[0], lims[1], 100)
ax.fill_between(x_diag, x_diag - max_res*1e3, x_diag + max_res*1e3,
                alpha=0.15, color='#7C3AED', label=f'±{max_res*1e6:.0f}×10⁻⁶')

ax.set_xlabel('ω_envelope  (× 10⁻³  c/a)', fontsize=11)
ax.set_ylabel('ω_FDFD (nearest)  (× 10⁻³  c/a)', fontsize=11)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.set_title('(a)  Frequency comparison', fontsize=12, fontweight='bold')

# (b) Residuals
ax = axes[1]
mode_idx = np.arange(len(env_freqs))
colors = ['#22C55E' if abs(r)/env_bw < 0.005 else '#EAB308' if abs(r)/env_bw < 0.01 else '#F97316'
          for r in residuals]
ax.bar(mode_idx, residuals * 1e6, color=colors, width=0.8, edgecolor='none', alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(+mean_res*1e6, color='gray', ls='--', lw=0.8, alpha=0.6, label=f'±mean = ±{mean_res*1e6:.0f}×10⁻⁶')
ax.axhline(-mean_res*1e6, color='gray', ls='--', lw=0.8, alpha=0.6)

ax.set_xlabel('Envelope mode index', fontsize=11)
ax.set_ylabel('Δω = ω_env − ω_FDFD  (× 10⁻⁶  c/a)', fontsize=11)
ax.set_title('(b)  Residuals per mode', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Annotation
ax.text(0.98, 0.02, f'mean |Δω| = {mean_res*1e6:.0f}×10⁻⁶\n'
        f'= {mean_res/env_bw*100:.1f}% of BW',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(facecolor='white', edgecolor='#CBD5E1', alpha=0.9))

fig.suptitle(f'Envelope vs FDFD:  θ = {theta_deg:.2f}°,  honeycomb TM,  (30,29)',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_residuals.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_residuals.pdf'), bbox_inches='tight')
print("Saved fig_residuals.png/pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: The "why DOS differs" explanation plot
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# (a) Full FDFD spectrum with envelope window highlighted
ax = axes[0]
fdfd_sorted = np.sort(fdfd_all)
ax.plot(np.arange(len(fdfd_sorted)), fdfd_sorted, 'b-', lw=0.8, alpha=0.7)
ax.scatter(np.arange(len(fdfd_sorted)), fdfd_sorted, s=3, c='#3B82F6', zorder=3)

# Highlight the envelope window
in_window = (fdfd_sorted >= env_min) & (fdfd_sorted <= env_max)
idx_in = np.where(in_window)[0]
ax.scatter(idx_in, fdfd_sorted[in_window], s=8, c='#EF4444', zorder=4,
           label=f'{in_window.sum()} FDFD in env window')

ax.axhspan(env_min, env_max, alpha=0.12, color='red', zorder=0,
           label=f'Envelope window\n[{env_min:.4f}, {env_max:.4f}]')

ax.set_xlabel('FDFD mode index (of 300)', fontsize=10)
ax.set_ylabel('ω (c/a)', fontsize=10)
ax.set_title('(a)  FDFD sees ALL folded bands', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')

# (b) Zoomed: env + FDFD modes side by side, classified
ax = axes[1]

# Classify FDFD modes in window as "matched" or "extra"
fdfd_in_w = fdfd_all[(fdfd_all >= env_min) & (fdfd_all <= env_max)]
matched_freqs_in_w = set()
for ef in env_freqs:
    idx = np.argmin(np.abs(fdfd_all - ef))
    if env_min <= fdfd_all[idx] <= env_max:
        matched_freqs_in_w.add(idx)

fdfd_matched_list = []
fdfd_extra_list = []
for f in fdfd_in_w:
    idx = np.argmin(np.abs(fdfd_all - f))
    if idx in matched_freqs_in_w:
        fdfd_matched_list.append(f)
    else:
        fdfd_extra_list.append(f)

# Draw envelope modes
for f in env_freqs:
    ax.plot([0.55, 0.95], [f, f], '-', color='#EF4444', lw=1.0, alpha=0.7)

# Draw matched FDFD modes (green)
for f in fdfd_matched_list:
    ax.plot([0.05, 0.45], [f, f], '-', color='#22C55E', lw=1.0, alpha=0.7)

# Draw extra FDFD modes (purple/indigo) 
for f in fdfd_extra_list:
    ax.plot([0.05, 0.45], [f, f], '-', color='#6366F1', lw=1.0, alpha=0.6)

ax.set_xlim(-0.05, 1.05)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['FDFD', 'Envelope'], fontsize=10, fontweight='bold')
ax.set_ylabel('ω (c/a)', fontsize=10)
ax.set_title(f'(b)  Zoomed: {len(fdfd_matched_list)} matched + '
             f'{len(fdfd_extra_list)} extra', fontsize=11, fontweight='bold')

legend_elements = [
    Line2D([0], [0], color='#EF4444', lw=2, label=f'Envelope ({len(env_freqs)})'),
    Line2D([0], [0], color='#22C55E', lw=2, label=f'FDFD matched ({len(fdfd_matched_list)})'),
    Line2D([0], [0], color='#6366F1', lw=2, label=f'FDFD extra bands ({len(fdfd_extra_list)})'),
]
ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

# (c) DOS comparison — with envelope scaled 
ax = axes[2]
bins = np.linspace(env_min - 0.0005, env_max + 0.0005, 35)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

env_hist, _ = np.histogram(env_freqs, bins=bins)
fdfd_hist, _ = np.histogram(fdfd_in_w, bins=bins)

# Scale FDFD to same total count for shape comparison
scale = len(env_freqs) / max(len(fdfd_in_w), 1)
fdfd_hist_scaled = fdfd_hist * scale

ax.step(bin_centers, env_hist, where='mid', color='#EF4444', lw=2, label='Envelope')
ax.step(bin_centers, fdfd_hist, where='mid', color='#3B82F6', lw=2, label='FDFD (raw count)')
ax.step(bin_centers, fdfd_hist_scaled, where='mid', color='#3B82F6', lw=2, ls='--',
        alpha=0.5, label=f'FDFD (×{scale:.2f} scaled)')

ax.set_xlabel('ω (c/a)', fontsize=10)
ax.set_ylabel('Mode count per bin', fontsize=10)
ax.set_title('(c)  Density of states', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)

# Explanation annotation
ax.text(0.97, 0.97,
    "FDFD has 30% more modes\n"
    "from other folded bands.\n"
    "Shape differs because extra\n"
    "modes fill the window\n"
    "non-uniformly.",
    transform=ax.transAxes, ha='right', va='top', fontsize=8,
    bbox=dict(facecolor='#FEF3C7', edgecolor='#D97706', alpha=0.9))

fig.suptitle(f'Why DOS differs: Envelope sees 2 bands, FDFD sees all  |  '
             f'θ = {theta_deg:.2f}°,  (30,29)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(out_dir, 'fig_why_dos_differs.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_why_dos_differs.pdf'), bbox_inches='tight')
print("Saved fig_why_dos_differs.png/pdf")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Combined "one-glance" summary figure
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35,
                       height_ratios=[1.2, 1])

# ── Top left: Level diagram with connections ──
ax = fig.add_subplot(gs[0, 0])

for f in fdfd_view:
    in_env = env_min <= f <= env_max
    c = '#3B82F6' if in_env else '#93C5FD'
    ax.plot([0.05, 0.35], [f, f], '-', color=c, lw=0.7, alpha=0.6 if in_env else 0.3)

for f in env_freqs:
    ax.plot([0.65, 0.95], [f, f], '-', color='#EF4444', lw=1.0, alpha=0.7)

for ef, ff in zip(env_freqs, matched_fdfd_freq):
    ax.plot([0.35, 0.65], [ff, ef], '-', color='#22C55E', lw=0.4, alpha=0.5)

ax.axhspan(env_min, env_max, alpha=0.06, color='red', zorder=0)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(view_lo, view_hi)
ax.set_xticks([0.2, 0.8])
ax.set_xticklabels(['FDFD', 'Envelope'], fontsize=9, fontweight='bold')
ax.set_ylabel('ω (c/a)', fontsize=10)
ax.set_title('(a) Level diagram + matching', fontsize=11, fontweight='bold')

# ── Top center: 1:1 scatter ──
ax = fig.add_subplot(gs[0, 1])
ax.scatter(env_freqs * 1e3, matched_fdfd_freq * 1e3, s=25, c='#7C3AED',
           edgecolors='white', linewidths=0.3, zorder=5, alpha=0.9)
lims = [env_min * 1e3 - 0.05, env_max * 1e3 + 0.05]
ax.plot(lims, lims, 'k-', lw=1.5, alpha=0.3)
ax.set_xlabel('ω_envelope  (×10³)', fontsize=10)
ax.set_ylabel('ω_FDFD  (×10³)', fontsize=10)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_aspect('equal')
ax.set_title('(b) 1:1 frequency match', fontsize=11, fontweight='bold')

# ── Top right: Residual bars ──
ax = fig.add_subplot(gs[0, 2])
colors = ['#22C55E' if abs(r)/env_bw < 0.005 else '#EAB308' for r in residuals]
ax.bar(np.arange(len(residuals)), residuals * 1e6, color=colors, width=0.8, alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.axhline(+mean_res*1e6, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.axhline(-mean_res*1e6, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Envelope mode index', fontsize=10)
ax.set_ylabel('Δω  (×10⁻⁶  c/a)', fontsize=10)
ax.set_title(f'(c) Residuals: mean = {mean_res*1e6:.0f}×10⁻⁶', fontsize=11, fontweight='bold')

# ── Bottom left: Full FDFD spectrum w/ env window ──
ax = fig.add_subplot(gs[1, 0])
ax.plot(np.arange(len(fdfd_sorted)), fdfd_sorted, '-', color='#3B82F6', lw=0.6, alpha=0.5)
ax.scatter(np.arange(len(fdfd_sorted)), fdfd_sorted, s=2, c='#3B82F6')
ax.axhspan(env_min, env_max, alpha=0.15, color='red')
ax.set_xlabel('FDFD mode index', fontsize=10)
ax.set_ylabel('ω (c/a)', fontsize=10)
ax.set_title('(d) Full FDFD spectrum', fontsize=11, fontweight='bold')

# ── Bottom center: Zoomed level diagram (matched vs extra) ──
ax = fig.add_subplot(gs[1, 1])
for f in env_freqs:
    ax.plot([0.55, 0.95], [f, f], '-', color='#EF4444', lw=0.9, alpha=0.6)
for f in fdfd_matched_list:
    ax.plot([0.05, 0.45], [f, f], '-', color='#22C55E', lw=0.9, alpha=0.6)
for f in fdfd_extra_list:
    ax.plot([0.05, 0.45], [f, f], '-', color='#6366F1', lw=0.9, alpha=0.5)
ax.set_xlim(-0.05, 1.05)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['FDFD', 'Envelope'], fontsize=9, fontweight='bold')
ax.set_ylabel('ω (c/a)', fontsize=10)
ax.set_title(f'(e) {len(fdfd_matched_list)} matched + {len(fdfd_extra_list)} extra FDFD',
             fontsize=11, fontweight='bold')
legend_elements = [
    Line2D([0], [0], color='#EF4444', lw=2, label=f'Envelope ({len(env_freqs)})'),
    Line2D([0], [0], color='#22C55E', lw=2, label=f'Matched FDFD'),
    Line2D([0], [0], color='#6366F1', lw=2, label=f'Extra bands'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper left')

# ── Bottom right: DOS ──
ax = fig.add_subplot(gs[1, 2])
ax.step(bin_centers, env_hist, where='mid', color='#EF4444', lw=2, label='Envelope')
ax.step(bin_centers, fdfd_hist, where='mid', color='#3B82F6', lw=2, label='FDFD (in window)')
ax.set_xlabel('ω (c/a)', fontsize=10)
ax.set_ylabel('Count', fontsize=10)
ax.set_title('(f) DOS (different N_modes)', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.text(0.97, 0.97, f"Env: {len(env_freqs)} modes\nFDFD: {len(fdfd_in_w)} modes\n"
        f"Extra {len(fdfd_extra_list)} from\nother bands",
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(facecolor='#FEF3C7', edgecolor='#D97706', alpha=0.9))

fig.suptitle(f'Envelope Approximation vs FDFD  |  Honeycomb TM  |  '
             f'θ = {theta_deg:.2f}°  |  (m,n) = (30,29)\n'
             f'Each envelope mode matches an FDFD mode to {mean_res/env_bw*100:.1f}% '
             f'of envelope bandwidth',
             fontsize=14, fontweight='bold')
fig.savefig(os.path.join(out_dir, 'fig_summary_1deg.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(out_dir, 'fig_summary_1deg.pdf'), bbox_inches='tight')
print("Saved fig_summary_1deg.png/pdf")
plt.close(fig)

print("\nDone! All figures saved.")
