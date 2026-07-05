#!/usr/bin/env python3
"""
F03 sweep results plot: Per-band miniband analysis with corrected Hamiltonian.

Panels:
(a) BW20 vs eta for all 3 bands
(b) delta_shallow/V_range vs eta (bound state depth metric)
(c) T1/V_range vs eta (kinetic energy scale — marks validity boundary)
(d) E0 vs eta for all bands (lowest eigenvalue)
"""
import numpy as np
import json
import matplotlib.pyplot as plt

with open('/home/renlephy/msl/research/moire_envelope/findings/sweep_results_F03_corrected.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Extract per-band data
etas = np.array([d['eta'] for d in data])
thetas = np.array([d['theta_deg'] for d in data])

bands = {0: {'name': 'Band 0 (hole)', 'color': 'C0', 'bw20': [], 'delta': [], 'E0': [], 'V_range': []},
         1: {'name': 'Band 1 (electron)', 'color': 'C1', 'bw20': [], 'delta': [], 'E0': [], 'V_range': []},
         2: {'name': 'Band 2 (hole)', 'color': 'C2', 'bw20': [], 'delta': [], 'E0': [], 'V_range': []}}

for entry in data:
    for pb in entry['per_band']:
        t = pb['band_index']
        bands[t]['bw20'].append(pb.get('bandwidth_20', np.nan))
        bands[t]['delta'].append(pb.get('delta_shallow_rel', np.nan))
        bands[t]['E0'].append(pb['eigenvalues'][0] if pb.get('eigenvalues') else np.nan)
        bands[t]['V_range'].append(pb['V_max'] - pb['V_min'])

for t in bands:
    for key in ['bw20', 'delta', 'E0', 'V_range']:
        bands[t][key] = np.array(bands[t][key])

# ========== Panel (a): BW20 vs eta ==========
ax = axes[0, 0]
for t in [0, 1, 2]:
    b = bands[t]
    ax.plot(etas, b['bw20'], 'o-', color=b['color'], label=b['name'], ms=5)

# Fit power law for small-eta region (theta <= 3°)
mask = thetas <= 3.0
for t in [0, 1, 2]:
    bw = bands[t]['bw20'][mask]
    e = etas[mask]
    valid = ~np.isnan(bw) & (bw > 0)
    if valid.sum() >= 3:
        p = np.polyfit(np.log(e[valid]), np.log(bw[valid]), 1)
        exponent = p[0]
        ax.plot(e[valid], np.exp(p[1]) * e[valid]**p[0], '--', color=bands[t]['color'], 
                alpha=0.5, label=f'  fit: η^{exponent:.1f}')

ax.set_xlabel(r'$\eta = a/L_{\mathrm{moiré}}$')
ax.set_ylabel('BW₂₀ (c/a)')
ax.set_title('(a) Miniband bandwidth vs η')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# ========== Panel (b): delta_shallow/V_range vs eta ==========
ax = axes[0, 1]
for t in [0, 1, 2]:
    b = bands[t]
    ax.plot(etas, b['delta'], 'o-', color=b['color'], label=b['name'], ms=5)

ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.axhline(1, color='red', ls='--', alpha=0.3, label='T₁ = V_range')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$\delta_{\mathrm{shallow}} / V_{\mathrm{range}}$')
ax.set_title(r'(b) Bound state depth ($\delta/V_{\mathrm{range}}$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ========== Panel (c): T1/V_range vs eta ==========
ax = axes[1, 0]
# Recompute T1/V_range
import h5py
SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260205_090337/eta_sweep_20260206_092258'

T1_V = {0: [], 1: [], 2: []}
for theta_str in ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']:
    cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
    with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
        M_inv = hf['M_inv'][:]
        Lambda = hf['Lambda'][:]
        B_m = hf.attrs['B_moire']
    L_m = np.linalg.norm(B_m[0])
    for t in range(3):
        M_tr = abs(M_inv[:,:,t,t,0,0] + M_inv[:,:,t,t,1,1]).mean()
        V_r = Lambda[:,:,t,t].max() - Lambda[:,:,t,t].min()
        T1 = M_tr / (2 * L_m**2)
        T1_V[t].append(T1 / V_r)

for t in [0, 1, 2]:
    ax.plot(etas, T1_V[t], 'o-', color=bands[t]['color'], label=bands[t]['name'], ms=5)

ax.axhline(1.0, color='red', ls='--', alpha=0.5, label='T₁ = V_range (breakdown)')
ax.axhline(0.1, color='orange', ls='--', alpha=0.3, label='T₁ = 0.1×V_range')
ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$T_1 / V_{\mathrm{range}}$')
ax.set_title('(c) Kinetic energy scale')
ax.set_yscale('log')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
# Shade invalid region
ax.axhspan(1.0, 10, alpha=0.1, color='red')

# ========== Panel (d): E0 vs eta ==========
ax = axes[1, 1]
for t in [0, 1, 2]:
    b = bands[t]
    ax.plot(etas, b['E0'], 'o-', color=b['color'], label=b['name'], ms=5)

# Plot V_min and V_max for band 1 as reference
V_min_b1 = np.array([d['per_band'][1]['V_min'] for d in data])
V_max_b1 = np.array([d['per_band'][1]['V_max'] for d in data])
ax.fill_between(etas, V_min_b1, V_max_b1, alpha=0.1, color='C1', label='Band 1 V range')

ax.set_xlabel(r'$\eta$')
ax.set_ylabel(r'$E_0$ (c/a)')
ax.set_title('(d) Ground state eigenvalue vs η')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/renlephy/msl/research/moire_envelope/findings/F03_sweep_corrected.png', dpi=150)
plt.savefig('/home/renlephy/msl/research/moire_envelope/findings/F03_sweep_corrected.pdf')
print('Saved F03_sweep_corrected.png and .pdf')
plt.close()
