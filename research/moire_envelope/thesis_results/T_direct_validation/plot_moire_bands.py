"""Plot moiré band structure from saved .npz data."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

# Load data
data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'moire_bands_m8_n7_res20.npz')
data = np.load(data_file, allow_pickle=True)

freqs = data['freqs']       # (n_q, n_modes)
q_dist = data['q_dist']     # (n_q,)
omega_ref = float(data['omega_ref'])
theta_deg = float(data['theta_deg'])
m, n = int(data['m']), int(data['n'])
N_cells = int(data['N_cells'])
res = int(data['res'])
labels_arr = data['labels']  # list of (index, label_name)

# Parse labels
tick_positions = []
tick_labels = []
for item in labels_arr:
    idx, lbl = int(item[0]), str(item[1])
    tick_positions.append(q_dist[idx])
    tick_labels.append(lbl)

n_q, n_modes = freqs.shape

# ── Figure 1: Full band structure ──
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for i in range(n_modes):
    ax.plot(q_dist, freqs[:, i], 'b-', linewidth=0.8, alpha=0.7)

ax.axhline(omega_ref, color='r', linestyle='--', linewidth=1, alpha=0.7,
           label=f'ω_ref = {omega_ref:.4f}')

ax.set_xlabel('k-path')
ax.set_ylabel('ω  (c/a)')
ax.set_title(f'Moiré band structure: honeycomb TM, (m,n)=({m},{n}), '
             f'θ={theta_deg:.2f}°, N={N_cells}')

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Add vertical lines at high-symmetry points
for pos in tick_positions:
    ax.axvline(pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

ax.legend(loc='upper right')
ax.set_xlim(q_dist[0], q_dist[-1])
fig.tight_layout()

out_dir = os.path.dirname(os.path.abspath(__file__))
fig.savefig(os.path.join(out_dir, 'moire_bands_full.png'), dpi=150)
print(f"Saved moire_bands_full.png")

# ── Figure 2: Zoomed near Dirac manifold ──
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

# Color bands by proximity to omega_ref
for i in range(n_modes):
    f_mean = np.mean(freqs[:, i])
    dist_to_dirac = abs(f_mean - omega_ref)
    if dist_to_dirac < 0.003:
        color, alpha, lw = 'red', 0.9, 1.2
    elif dist_to_dirac < 0.006:
        color, alpha, lw = 'blue', 0.8, 1.0
    else:
        color, alpha, lw = 'gray', 0.4, 0.6
    ax2.plot(q_dist, freqs[:, i], color=color, linewidth=lw, alpha=alpha)

ax2.axhline(omega_ref, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
            label=f'ω_ref = {omega_ref:.4f}')

ax2.set_xlabel('k-path')
ax2.set_ylabel('ω  (c/a)')
ax2.set_title(f'Moiré minibands near Dirac point: θ={theta_deg:.2f}°')

ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels)
for pos in tick_positions:
    ax2.axvline(pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# Focus on ±4% around omega_ref
margin = 0.04 * omega_ref
ax2.set_ylim(omega_ref - margin, omega_ref + margin)
ax2.legend(loc='upper right')
ax2.set_xlim(q_dist[0], q_dist[-1])
fig2.tight_layout()
fig2.savefig(os.path.join(out_dir, 'moire_bands_zoomed.png'), dpi=150)
print(f"Saved moire_bands_zoomed.png")

# ── Figure 3: Band density of states near Dirac ──
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))
all_f = freqs.ravel()
bins = np.linspace(omega_ref - 0.015, omega_ref + 0.015, 100)
ax3.hist(all_f, bins=bins, density=True, color='steelblue', alpha=0.7)
ax3.axvline(omega_ref, color='r', linestyle='--', linewidth=1.5,
            label=f'ω_ref = {omega_ref:.4f}')
ax3.set_xlabel('ω  (c/a)')
ax3.set_ylabel('DOS (arb. units)')
ax3.set_title(f'Density of states near Dirac: θ={theta_deg:.2f}°')
ax3.legend()
fig3.tight_layout()
fig3.savefig(os.path.join(out_dir, 'moire_dos_near_dirac.png'), dpi=150)
print(f"Saved moire_dos_near_dirac.png")

# ── Analysis: gaps and bandwidths ──
print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")

# Group bands at Gamma into clusters
gamma_freqs = freqs[0]
sorted_f = np.sort(gamma_freqs)
gaps = np.diff(sorted_f)
print(f"\nAt Γ_m:")
print(f"  Sorted frequencies: ", ", ".join(f"{f:.6f}" for f in sorted_f))
print(f"  Gaps between adjacent modes:")
for i, g in enumerate(gaps):
    if g > 0.001:
        print(f"    Between mode {i} and {i+1}: Δω = {g:.6f}" +
              (" ** SIGNIFICANT GAP **" if g > 0.002 else ""))

# Identify the gap closest to omega_ref
above = sorted_f[sorted_f > omega_ref]
below = sorted_f[sorted_f <= omega_ref]
if len(above) > 0 and len(below) > 0:
    gap_at_dirac = above[0] - below[-1]
    print(f"\n  Gap straddling ω_ref: Δω = {gap_at_dirac:.6f}")
    print(f"  Gap / ω_ref = {gap_at_dirac / omega_ref:.4f}")
    print(f"  Modes below ω_ref at Γ: {len(below)}")
    print(f"  Modes above ω_ref at Γ: {len(above)}")

plt.close('all')
print("\nDone.")
