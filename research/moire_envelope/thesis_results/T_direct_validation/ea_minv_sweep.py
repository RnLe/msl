#!/usr/bin/env python3
"""Diagnostic: sweep M_inv_max_trace regularization and compare bandwidth."""
import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phasesV3'))
from phase3_mpb_v3 import assemble_multiband_hamiltonian, solve_multiband_envelope
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = 1.0; M_IDX = 11; N_IDX = 1
L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1@L1)
B_SUPER = np.column_stack([L1, L2])
THETA_RAD = 2*np.arctan2(N_IDX, M_IDX)
OMEGA0 = 0.68457
eta = A / L_SUPER
Ns = 128
dR = L_SUPER / Ns
NR = 32
OUTDIR = os.path.join(os.path.dirname(__file__), 'square_3way')

d = np.load(f'{OUTDIR}/ea_multiband_registry.npz')
omega0_reg = d['omega0']
vg_reg = d['vg']
Minv_reg = d['M_inv']

R_mat = np.array([[np.cos(THETA_RAD), -np.sin(THETA_RAD)],
                   [np.sin(THETA_RAD), np.cos(THETA_RAD)]])
s1 = np.arange(Ns)/Ns; s2 = np.arange(Ns)/Ns
S1, S2 = np.meshgrid(s1, s2, indexing='ij')
X = S1*L1[0] + S2*L2[0]; Y = S1*L1[1] + S2*L2[1]
pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
disp = ((R_mat - np.eye(2)) @ pos.T).T
delta_frac = disp - np.floor(disp)
pts = delta_frac

reg_ax = np.linspace(0, 1, NR, endpoint=False)
reg_ax_ext = np.concatenate([reg_ax, [1.0]])

def interp_field(data_2d, pts):
    padded = np.concatenate([data_2d, data_2d[:1,:]], axis=0)
    padded = np.concatenate([padded, padded[:,:1]], axis=1)
    f = RegularGridInterpolator((reg_ax_ext, reg_ax_ext), padded,
                                method='linear', bounds_error=False, fill_value=None)
    return f(pts)

band = 3
omega_m = interp_field(omega0_reg[:,:,band], pts).reshape(Ns, Ns)
vgx_m = interp_field(vg_reg[:,:,band,0], pts).reshape(Ns, Ns)
vgy_m = interp_field(vg_reg[:,:,band,1], pts).reshape(Ns, Ns)
Mxx_m = interp_field(Minv_reg[:,:,band,0,0], pts).reshape(Ns, Ns)
Mxy_m = interp_field(Minv_reg[:,:,band,0,1], pts).reshape(Ns, Ns)
Myy_m = interp_field(Minv_reg[:,:,band,1,1], pts).reshape(Ns, Ns)

V = omega_m - OMEGA0

# FDFD reference
fdfd = np.load(f'{OUTDIR}/fdfd_supercell.npz')
freqs_fdfd = np.sort(fdfd['freqs'])
bw_fdfd = (freqs_fdfd[-1] - freqs_fdfd[0]) * 1000
center_fdfd = np.mean(freqs_fdfd)

Nb = 1
Lambda = V.reshape(Ns, Ns, 1, 1)
v_drift = np.zeros((Ns, Ns, 1, 1, 2))
v_drift[:,:,0,0,0] = vgx_m; v_drift[:,:,0,0,1] = vgy_m
M_inv_raw = np.zeros((Ns, Ns, 1, 1, 2, 2))
M_inv_raw[:,:,0,0,0,0] = Mxx_m; M_inv_raw[:,:,0,0,0,1] = Mxy_m
M_inv_raw[:,:,0,0,1,0] = Mxy_m; M_inv_raw[:,:,0,0,1,1] = Myy_m
A_berry = np.zeros((Ns, Ns, 1, 1, 2))
Phi_BH = np.zeros((Ns, Ns, 1, 1))

# Test different M_inv_max_trace values
traces = [None, 20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1, 0.0]
results = []

for mt in traces:
    M_inv_use = M_inv_raw.copy()
    if mt is not None and mt > 0:
        # Clamp |Tr(M_inv)| at each point
        tr = M_inv_use[:,:,0,0,0,0] + M_inv_use[:,:,0,0,1,1]
        mask = np.abs(tr) > mt
        if np.any(mask):
            scale = mt / np.abs(tr[mask])
            M_inv_use[mask,0,0,:,:] *= scale[:, None, None]

    if mt == 0.0:
        M_inv_use[:] = 0.0  # V-only

    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv_use, A_berry, Phi_BH,
        eta, Ns, Ns, Nb, dR, dR, B_SUPER,
        include_drift=True, include_kinetic=(mt != 0.0),
        include_born_huang=False, order=4)

    evals, _ = solve_multiband_envelope(H, 50, sigma=0.0)
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    freqs = OMEGA0 + evals
    freqs_s = np.sort(freqs)
    bw = (freqs_s[-1] - freqs_s[0]) * 1000
    
    # RMS vs FDFD
    n = min(len(freqs_s), len(freqs_fdfd))
    rms = np.sqrt(np.mean((freqs_s[:n] - freqs_fdfd[:n])**2)) * 1000
    
    label = f"mt={mt}" if mt is not None else "mt=None"
    print(f'{label:>12s}  bw={bw:6.1f}  RMS={rms:6.2f}  range=[{freqs_s[0]:.4f}, {freqs_s[-1]:.4f}]')
    results.append((mt, bw, rms, freqs_s))

# Also test with drift disabled
print('\n--- No drift, no kinetic (V-only) ---')
H_v = assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv_raw, A_berry, Phi_BH,
    eta, Ns, Ns, Nb, dR, dR, B_SUPER,
    include_drift=False, include_kinetic=False, include_born_huang=False, order=4)
evals_v, _ = solve_multiband_envelope(H_v, 50, sigma=0.0)
idx_v = np.argsort(np.abs(evals_v))
freqs_v = np.sort(OMEGA0 + evals_v[idx_v])
rms_v = np.sqrt(np.mean((freqs_v[:50] - freqs_fdfd[:50])**2)) * 1000
print(f'V-only: bw={(freqs_v[-1]-freqs_v[0])*1000:.1f}  RMS={rms_v:.2f}  range=[{freqs_v[0]:.4f}, {freqs_v[-1]:.4f}]')

print('\n--- No drift, kinetic ON ---')
H_vk = assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv_raw, A_berry, Phi_BH,
    eta, Ns, Ns, Nb, dR, dR, B_SUPER,
    include_drift=False, include_kinetic=True, include_born_huang=False, order=4)
evals_vk, _ = solve_multiband_envelope(H_vk, 50, sigma=0.0)
idx_vk = np.argsort(np.abs(evals_vk))
freqs_vk = np.sort(OMEGA0 + evals_vk[idx_vk])
rms_vk = np.sqrt(np.mean((freqs_vk[:50] - freqs_fdfd[:50])**2)) * 1000
print(f'V+K: bw={(freqs_vk[-1]-freqs_vk[0])*1000:.1f}  RMS={rms_vk:.2f}  range=[{freqs_vk[0]:.4f}, {freqs_vk[-1]:.4f}]')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for mt, bw, rms, fs in results:
    label = f"mt={mt}" if mt is not None else "unclamped"
    ax.plot(range(50), fs, 'o-', ms=2, label=f'{label} (RMS={rms:.1f})')
ax.plot(range(50), freqs_fdfd, 'k-', lw=2, label=f'FDFD (bw={bw_fdfd:.1f})')
ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5)
ax.set_xlabel('Mode index'); ax.set_ylabel('ω')
ax.set_title('Sorted eigenvalues vs M_inv clamp')
ax.legend(fontsize=7); ax.set_ylim(0.62, 0.75)

ax = axes[1]
mts = [t for t, _, _, _ in results if t is not None]
rmss = [r for _, _, r, _ in results if True]
bws = [b for _, b, _, _ in results]
ax.plot(range(len(mts)), rmss[:len(mts)], 'ro-', label='RMS error')
ax.plot(range(len(mts)), bws[:len(mts)], 'bo-', label='Bandwidth')
ax.axhline(bw_fdfd, color='b', ls='--', label=f'FDFD bw={bw_fdfd:.1f}')
ax.set_xticks(range(len(mts)))
ax.set_xticklabels([str(m) for m in mts], rotation=45)
ax.set_xlabel('M_inv_max_trace'); ax.set_ylabel('×10⁻³')
ax.legend()
ax.set_title('RMS error and bandwidth vs regularization')

plt.tight_layout()
fig.savefig(f'{OUTDIR}/fig_minv_regularization_sweep.png', dpi=150)
print(f'\nSaved {OUTDIR}/fig_minv_regularization_sweep.png')
