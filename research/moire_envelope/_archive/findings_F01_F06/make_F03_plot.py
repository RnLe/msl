#!/usr/bin/env python3
"""
F03 diagnostic plot: Kinetic operator analysis.

4 panels:
(a) M_inv trace heatmap for band 1 (showing hot spots)
(b) Band gap correlation: M_inv vs gap_01 scatter
(c) Kinetic energy scale T1/V_range vs eta (all bands)
(d) Non-Hermiticity |H-H†|/|H| vs theta (Ns=32)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

SWEEP = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260206_152443/eta_sweep_20260206_173808'

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ========== Panel (a): M_inv trace heatmap ==========
ax = axes[0, 0]
cdir = f'{SWEEP}/theta_5.000/candidate_0000'
with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
    M_inv_5 = hf['M_inv'][:]
    Lambda_5 = hf['Lambda'][:]
    B_moire = hf.attrs['B_moire']
    N_bands_5 = Lambda_5.shape[2]
    target_idx = int(hf.attrs.get('target_index_in_subspace', min(1, N_bands_5-1)))

# Show target band M_inv trace
M_trace = M_inv_5[:, :, target_idx, target_idx, 0, 0] + M_inv_5[:, :, target_idx, target_idx, 1, 1]
im = ax.imshow(M_trace.T, origin='lower', cmap='hot', vmin=0, vmax=50)
plt.colorbar(im, ax=ax, label=r'$\mathrm{Tr}[M^{-1}]$')
ax.set_title(f'(a) Band {target_idx}: $M^{{-1}}$ trace (\u03b8=5\u00b0)')
ax.set_xlabel('$s_1$ grid index')
ax.set_ylabel('$s_2$ grid index')

# ========== Panel (b): M_inv vs gap scatter ==========
ax = axes[0, 1]
# Gap between target band and nearest neighbor
gap_idx = max(0, target_idx - 1)
gap = Lambda_5[:, :, target_idx, target_idx] - Lambda_5[:, :, gap_idx, gap_idx]
ax.scatter(gap.ravel(), M_trace.ravel(), s=0.5, alpha=0.3, color='C0')
ax.set_xlabel(r'Band gap $\Delta$ (c/a)')
ax.set_ylabel(r'$\mathrm{Tr}[M^{-1}]$')
ax.set_title('(b) Mass divergence at near-degeneracy')
ax.set_xlim([-0.005, 0.15])
ax.set_ylim([0, 140])
ax.axhline(50, color='red', ls='--', alpha=0.5, label='Hot spot threshold')
ax.axvline(0.01, color='orange', ls='--', alpha=0.5, label=r'$\Delta=0.01$')
ax.legend(fontsize=8)

# ========== Panel (c): Kinetic scale T1/V_range vs eta ==========
ax = axes[1, 0]

thetas = ['0.500', '0.800', '1.100', '1.500', '2.000', '3.000', '5.000', '8.000']
etas = []

# Auto-detect N_bands from first file
first_cdir = f'{SWEEP}/theta_{thetas[0]}/candidate_0000'
with h5py.File(f'{first_cdir}/phase2_multiband_data.h5', 'r') as hf:
    N_bands = hf['Lambda'].shape[2]

T1_V_bands = {t: [] for t in range(N_bands)}
_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
_labels = [f'Band {t}' for t in range(N_bands)]

for theta_str in thetas:
    cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
    with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
        M_inv = hf['M_inv'][:]
        Lambda = hf['Lambda'][:]
        eta = float(hf.attrs['eta'])
        B_m = hf.attrs['B_moire']
    
    L_m = np.linalg.norm(B_m[0])
    etas.append(eta)
    
    for t in range(N_bands):
        M_tr = abs(M_inv[:, :, t, t, 0, 0] + M_inv[:, :, t, t, 1, 1]).mean()
        V_r = Lambda[:, :, t, t].max() - Lambda[:, :, t, t].min()
        T1 = M_tr / (2 * L_m**2)
        T1_V_bands[t].append(T1 / V_r)

etas = np.array(etas)
for t in range(N_bands):
    ax.plot(etas, T1_V_bands[t], 'o-', color=_colors[t % len(_colors)], label=_labels[t])

ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='T₁ = V_range')
ax.set_xlabel(r'$\eta = a/L_{\mathrm{moiré}}$')
ax.set_ylabel(r'$T_1 / V_{\mathrm{range}}$')
ax.set_title(r'(c) Kinetic energy scale vs $\eta$')
ax.set_yscale('log')
ax.legend(fontsize=8)
ax.set_ylim([1e-3, 3])

# ========== Panel (d): Non-Hermiticity vs theta ==========
ax = axes[1, 1]

Ns = 32
nh_norms = []
theta_vals = []

for theta_str in thetas:
    cdir = f'{SWEEP}/theta_{theta_str}/candidate_0000'
    with h5py.File(f'{cdir}/phase2_multiband_data.h5', 'r') as hf:
        Lambda_full = hf['Lambda'][:]
        M_inv_full = hf['M_inv'][:]
        A_berry_full = hf['A_berry'][:]
        Phi_BH_full = hf['Phi_BH'][:]
        v_drift_full = hf['v_drift'][:]
        eta = float(hf.attrs['eta'])
        Ns1 = int(hf.attrs['Ns1'])
        B_m = hf.attrs['B_moire']
    
    L_m = np.linalg.norm(B_m[0])
    N_bands_d = Lambda_full.shape[2]
    t = min(1, N_bands_d - 1)  # Use band 1 if available
    stride = Ns1 // Ns
    Lb = Lambda_full[::stride, ::stride, t:t+1, t:t+1]
    Mb = M_inv_full[::stride, ::stride, t:t+1, t:t+1, :, :]
    vb = v_drift_full[::stride, ::stride, t:t+1, t:t+1, :]
    Ab = A_berry_full[::stride, ::stride, t:t+1, t:t+1, :]
    Pb = Phi_BH_full[::stride, ::stride, t:t+1, t:t+1]
    
    dR = L_m / Ns
    H = p3.assemble_multiband_hamiltonian(
        Lb, vb*0, Mb, Ab*0, Pb*0, eta, Ns, Ns, 1, dR, dR, B_m,
        include_drift=False, include_kinetic=True, include_born_huang=False, order=4
    )
    H_arr = H.toarray()
    nh = np.linalg.norm(H_arr - H_arr.T.conj()) / np.linalg.norm(H_arr)
    
    nh_norms.append(nh)
    theta_vals.append(float(theta_str))

ax.plot(theta_vals, np.array(nh_norms) * 100, 'o-', color='C3')
ax.axhline(10, color='orange', ls='--', alpha=0.5, label='10% threshold')
ax.set_xlabel(r'$\theta$ (degrees)')
ax.set_ylabel(r'$\|H - H^\dagger\| / \|H\|$ (%)')
ax.set_title('(d) Non-Hermiticity of H (Band 1, Ns=32)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/renlephy/msl/research/moire_envelope/findings/F03_kinetic_analysis.png', dpi=150)
plt.savefig('/home/renlephy/msl/research/moire_envelope/findings/F03_kinetic_analysis.pdf')
print('Saved F03_kinetic_analysis.png and .pdf')
plt.close()
