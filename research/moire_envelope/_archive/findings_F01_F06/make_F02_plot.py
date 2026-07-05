#!/usr/bin/env python3
"""
F02 Finding Plot: Moire Miniband Scaling & Kinetic Operator Dominance
=====================================================================
6-panel figure:
  (a) BW20 vs eta log-log with power-law fits
  (b) delta_shallow vs eta log-log
  (c) Kinetic diagonal max vs potential range
  (d) Eigenvalue spectrum at theta=0.5 (all 3 bands)
  (e) Band 0 eigenvalue fan vs theta
  (f) Ns convergence test
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FINDINGS = '/home/renlephy/msl/research/moire_envelope/findings'
with open(f'{FINDINGS}/sweep_results_F03_corrected.json') as f:
    data = json.load(f)
data_sorted = sorted(data, key=lambda x: x['theta_deg'])

etas = np.array([d['eta'] for d in data_sorted])
thetas = np.array([d['theta_deg'] for d in data_sorted])
log_eta = np.log(etas)

N_bands = len(data_sorted[0]['per_band'])
_all_colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63', '#9C27B0']
_all_markers = ['o', 's', '^', 'D', 'v']
band_labels = []
band_colors = _all_colors[:N_bands]
band_markers = _all_markers[:N_bands]
for b in range(N_bands):
    bt = data_sorted[0]['per_band'][b]['type']
    band_labels.append(f'Band {b} ({bt})')

# Extract per-band observables
bw20 = {b: [] for b in range(N_bands)}
delta_shallow = {b: [] for b in range(N_bands)}
V_ranges = {b: [] for b in range(N_bands)}
kin_max = {b: [] for b in range(N_bands)}

for d in data_sorted:
    eta_val = d['eta']
    for band in range(N_bands):
        pb = d['per_band'][band]
        evals = np.sort(pb['eigenvalues'])
        V_max, V_min = pb['V_max'], pb['V_min']
        M_abs = abs(pb['mean_mass_trace'])
        bw20[band].append(abs(evals[-1] - evals[0]))
        V_ranges[band].append(V_max - V_min)
        # Kinetic diagonal max estimate: prefactor * max(M_inv_trace) * 5/(2*dR^2) * 2
        # max(M_inv) ~ 13 * mean(M_inv) for our data
        L_m = 1.0 / eta_val  # a=1
        dR = L_m / 128
        kin_max[band].append(0.5 / (2*np.pi)**2 * 13 * M_abs * 5 / (2 * dR**2) * 2)
        if pb['type'] == 'hole':
            delta_shallow[band].append(abs(V_max - evals[-1]))
        else:
            delta_shallow[band].append(abs(evals[0] - V_min))

for b in range(N_bands):
    bw20[b] = np.array(bw20[b])
    delta_shallow[b] = np.array(delta_shallow[b])
    V_ranges[b] = np.array(V_ranges[b])
    kin_max[b] = np.array(kin_max[b])

# ---- Create figure ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "F02: Kinetic Operator Dominance & Miniband Scaling\n"
    f"Square lattice, a=1.0, N_bands={N_bands}, BFS gauge + SVQB",
    fontsize=13, fontweight='bold', y=0.98
)

# (a) BW20 vs eta
ax = axes[0, 0]
for band in range(N_bands):
    ax.loglog(etas, bw20[band], band_markers[band] + '-', color=band_colors[band],
              label=band_labels[band], markersize=7, linewidth=1.5)
    coeffs = np.polyfit(log_eta, np.log(bw20[band]), 1)
    eta_fit = np.logspace(np.log10(etas[0] * 0.8), np.log10(etas[-1] * 1.2), 50)
    ax.loglog(eta_fit, np.exp(coeffs[1]) * eta_fit**coeffs[0], '--',
              color=band_colors[band], alpha=0.5, linewidth=1)
    ax.annotate(r'$\alpha$=' + f'{coeffs[0]:.2f}',
                xy=(etas[-3], bw20[band][-3]), fontsize=9, color=band_colors[band],
                xytext=(8, -5 + band * 15), textcoords='offset points')

eta_ref = np.logspace(np.log10(etas[0]), np.log10(etas[-1]), 50)
ax.loglog(eta_ref, 0.5 * eta_ref**2, ':', color='gray', alpha=0.5, linewidth=1)
ax.annotate(r'$\eta^2$', xy=(eta_ref[5], 0.5*eta_ref[5]**2), fontsize=8, color='gray')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'BW$_{20}$')
ax.set_title(r'(a) k=20 bandwidth vs $\eta$')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# (b) delta_shallow vs eta
ax = axes[0, 1]
for band in range(N_bands):
    ax.loglog(etas, delta_shallow[band], band_markers[band] + '-', color=band_colors[band],
              label=band_labels[band], markersize=7, linewidth=1.5)
    coeffs = np.polyfit(log_eta, np.log(delta_shallow[band]), 1)
    ax.annotate(r'$\alpha$=' + f'{coeffs[0]:.2f}',
                xy=(etas[2], delta_shallow[band][2]), fontsize=9, color=band_colors[band],
                xytext=(8, -5 + band * 15), textcoords='offset points')

ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$|\lambda_{\rm edge} - V_{\rm ext}|$')
ax.set_title(r'(b) Shallowest eigenvalue offset')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# (c) Kinetic vs Potential scale
ax = axes[0, 2]
theta_vals = thetas
bar_width = 0.2
x = np.arange(len(theta_vals))
bar_width = 0.8 / N_bands
for band in range(N_bands):
    ax.bar(x + (band - N_bands/2 + 0.5) * bar_width, kin_max[band] / V_ranges[band],
           bar_width, color=band_colors[band], alpha=0.7, label=band_labels[band])
ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='T = V')
ax.set_xticks(x)
ax.set_xticklabels([f'{t:.1f}' for t in theta_vals], fontsize=8)
ax.set_xlabel(r'$\theta$ (deg)')
ax.set_ylabel(r'$T_{\rm Nyquist} / \Delta V$')
ax.set_title('(c) Kinetic / Potential ratio')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

# (d) Eigenvalue spectrum at theta=0.5
ax = axes[1, 0]
d0 = data_sorted[0]
for band in range(N_bands):
    pb = d0['per_band'][band]
    evals = np.sort(pb['eigenvalues'])
    V_max, V_min = pb['V_max'], pb['V_min']
    y = np.arange(len(evals))
    ax.plot(evals, y, band_markers[band], color=band_colors[band],
            markersize=5, label=band_labels[band])
    ax.axvline(x=V_max, color=band_colors[band], linestyle='--', alpha=0.3)
    ax.axvline(x=V_min, color=band_colors[band], linestyle=':', alpha=0.2)

ax.set_xlabel(r'Eigenvalue $\lambda$')
ax.set_ylabel('State index (sorted)')
ax.set_title(r'(d) Spectrum at $\theta=0.5\degree$')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (e) Band 0 eigenvalue fan
ax = axes[1, 1]
band_plot = 0
for d in data_sorted:
    pb = d['per_band'][band_plot]
    evals = np.sort(pb['eigenvalues'])
    V_max = pb['V_max']
    for ev in evals:
        ax.plot(d['theta_deg'], ev - V_max, '.', color=band_colors[band_plot],
                markersize=3, alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label=r'$V_{\max}$')
ax.set_xlabel(r'$\theta$ (deg)')
ax.set_ylabel(r'$\lambda - V_{\max}$')
ax.set_title(r'(e) Band 0 eigenvalue fan')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (f) Ns convergence
ax = axes[1, 2]
Ns_vals = np.array([16, 32, 64, 128])
delta_vals = np.array([0.01668, 0.01445, 0.01306, 0.01177])
ax.plot(Ns_vals, delta_vals * 1000, 'ko-', markersize=8, linewidth=2, label='Eigsh data')

# Fit curve
Ns_fine = np.linspace(16, 512, 200)
A, B, p = 0.008597, 0.0275, 0.442
delta_fit = A + B / Ns_fine**p
ax.plot(Ns_fine, delta_fit * 1000, 'r--', linewidth=1.5,
        label=f'Fit: {A:.4f} + {B:.3f}/$N_s^{{{p:.2f}}}$')
ax.axhline(y=A * 1000, color='green', linestyle=':', alpha=0.7,
           label=r'$\delta_\infty = $' + f'{A:.4f}')
ax.set_xlabel(r'$N_s$ (grid resolution)')
ax.set_ylabel(r'$\delta_{\rm shallow}$ ($\times 10^{-3}$)')
ax.set_title(r'(f) $N_s$ convergence (Band 1, $\theta$=0.5$\degree$)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 300)

plt.tight_layout()
plt.savefig(f'{FINDINGS}/F02_miniband_scaling.png', dpi=200, bbox_inches='tight')
print(f"Saved {FINDINGS}/F02_miniband_scaling.png")

# Save data
results = {
    'thetas_deg': thetas.tolist(),
    'etas': etas.tolist(),
    'Ns_convergence': {
        'Ns': Ns_vals.tolist(),
        'delta_shallow': delta_vals.tolist(),
        'fit_A': A, 'fit_B': B, 'fit_p': p,
    },
    'power_law_fits': {},
}
for band in range(N_bands):
    coeffs_bw = np.polyfit(log_eta, np.log(bw20[band]), 1)
    coeffs_ds = np.polyfit(log_eta, np.log(delta_shallow[band]), 1)
    results['power_law_fits'][f'band_{band}'] = {
        'type': data_sorted[0]['per_band'][band]['type'],
        'M_trace': data_sorted[0]['per_band'][band]['mean_mass_trace'],
        'BW20_exponent': float(coeffs_bw[0]),
        'delta_shallow_exponent': float(coeffs_ds[0]),
        'BW20_values': bw20[band].tolist(),
        'delta_shallow_values': delta_shallow[band].tolist(),
    }

with open(f'{FINDINGS}/F02_data.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {FINDINGS}/F02_data.json")
