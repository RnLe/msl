#!/usr/bin/env python3
"""Single-band diagnostic: band 3 only EA vs FDFD."""
import sys, os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../phasesV3'))
from phase3_mpb_v3 import assemble_multiband_hamiltonian, solve_multiband_envelope
from scipy.interpolate import RegularGridInterpolator

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
print(f'Single-band V range: [{V.min():.6f}, {V.max():.6f}]')

V_fft = np.fft.fft2(V) / (Ns*Ns)
V_fft_abs = np.abs(V_fft)
flat_idx = np.argsort(V_fft_abs.ravel())[::-1]
print('V Fourier amplitudes (top 10):')
for i in range(10):
    idx = np.unravel_index(flat_idx[i], (Ns, Ns))
    print(f'  G=({idx[0]:3d},{idx[1]:3d}): |V_G| = {V_fft_abs[idx]:.6f}')

Nb = 1
Lambda = V.reshape(Ns, Ns, 1, 1)
v_drift = np.zeros((Ns, Ns, 1, 1, 2))
v_drift[:,:,0,0,0] = vgx_m; v_drift[:,:,0,0,1] = vgy_m
M_inv = np.zeros((Ns, Ns, 1, 1, 2, 2))
M_inv[:,:,0,0,0,0] = Mxx_m; M_inv[:,:,0,0,0,1] = Mxy_m
M_inv[:,:,0,0,1,0] = Mxy_m; M_inv[:,:,0,0,1,1] = Myy_m
A_berry = np.zeros((Ns, Ns, 1, 1, 2))
Phi_BH = np.zeros((Ns, Ns, 1, 1))

H = assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv, A_berry, Phi_BH,
    eta, Ns, Ns, Nb, dR, dR, B_SUPER,
    include_drift=True, include_kinetic=True, include_born_huang=False, order=4)

h_diag = H.diagonal().real
print(f'H diagonal: [{h_diag.min():.4f}, {h_diag.max():.4f}]')
print(f'H Tr(M_inv) × η²/dR² ≈ Tr(M_inv) × {eta**2/dR**2:.4f}')

evals, evecs = solve_multiband_envelope(H, 50, sigma=0.0)
idx = np.argsort(np.abs(evals))
evals = evals[idx]
freqs = OMEGA0 + evals
print(f'\nSingle-band EA: [{freqs.min():.6f}, {freqs.max():.6f}]')
print(f'Bandwidth: {(freqs.max()-freqs.min())*1000:.3f} × 10⁻³')

fdfd = np.load(f'{OUTDIR}/fdfd_supercell.npz')
freqs_fdfd = fdfd['freqs']
print(f'FDFD: [{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}], bw={(freqs_fdfd[-1]-freqs_fdfd[0])*1000:.3f} × 10⁻³')

f_e = np.sort(freqs)
f_f = np.sort(freqs_fdfd)
n = min(len(f_e), len(f_f))
diff = f_e[:n] - f_f[:n]
print(f'\nRMS error: {np.sqrt(np.mean(diff**2))*1000:.3f} × 10⁻³')
print(f'Max error: {np.max(np.abs(diff))*1000:.3f} × 10⁻³')
print(f'\n{"idx":>3s}  {"FDFD":>10s}  {"EA":>10s}  {"Δω×10³":>8s}')
for i in range(min(20, n)):
    print(f'{i:3d}  {f_f[i]:.6f}  {f_e[i]:.6f}  {diff[i]*1000:+.3f}')

# Also try V-only (no kinetic, no drift) to check potential spectral content
print('\n--- V-only (no kinetic, no drift) ---')
H_pot = assemble_multiband_hamiltonian(
    Lambda, v_drift, M_inv, A_berry, Phi_BH,
    eta, Ns, Ns, Nb, dR, dR, B_SUPER,
    include_drift=False, include_kinetic=False, include_born_huang=False, order=4)
evals_pot, _ = solve_multiband_envelope(H_pot, 50, sigma=0.0)
idx_pot = np.argsort(np.abs(evals_pot))
evals_pot = evals_pot[idx_pot]
freqs_pot = OMEGA0 + evals_pot
print(f'V-only: [{freqs_pot.min():.6f}, {freqs_pot.max():.6f}], bw={(freqs_pot.max()-freqs_pot.min())*1000:.3f} × 10⁻³')
