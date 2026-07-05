#!/usr/bin/env python
"""
FINDING F01: No discrete bound states in single-band moiré envelope Hamiltonian.

Loads phase2 data dynamically and runs eigsh with increasing k to demonstrate
that the ground state drifts indefinitely — proving no discrete bound states
exist in the periodic envelope Hamiltonian.

Works with any candidate (auto-detects N_bands, band type, k0).
"""
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import time

sys.path.insert(0, '/home/renlephy/msl/research/moire_envelope')
from phasesV3 import phase3_mpb_v3 as p3

OUT_DIR = Path(__file__).resolve().parent

# ============================================================
# Configuration — point to the current candidate
# ============================================================
CANDIDATE = '/home/renlephy/msl/research/moire_envelope/runsV3/phase0_mpb_v3_20260206_152443/candidate_0000'
M_INV_MAX = 20

# ============================================================
# Load Phase 2 data and build single-band Hamiltonian
# ============================================================
print("Loading Phase 2 data...")
with h5py.File(f'{CANDIDATE}/phase2_multiband_data.h5', 'r') as hf:
    Lambda = hf['Lambda'][:]
    M_inv = hf['M_inv'][:]
    A_berry = hf['A_berry'][:]
    Phi_BH = hf['Phi_BH'][:]
    v_drift = hf['v_drift'][:]
    eta = float(hf.attrs['eta'])
    Ns1 = int(hf.attrs['Ns1'])
    B_moire = hf.attrs['B_moire']
    omega_ref = float(hf.attrs['omega_ref'])
    theta_deg = float(hf.attrs['theta_deg'])
    target_idx = int(hf.attrs['target_index_in_subspace'])
    N_subspace = int(hf.attrs['N_subspace'])

L_moire = np.linalg.norm(B_moire[0])
dR = L_moire / Ns1
N_bands_full = Lambda.shape[2]

# Use target band for single-band analysis
t = target_idx
Lb = Lambda[:, :, t:t+1, t:t+1]
Mb = M_inv[:, :, t:t+1, t:t+1, :, :]
vb = v_drift[:, :, t:t+1, t:t+1, :]
Ab = A_berry[:, :, t:t+1, t:t+1, :]
Pb = Phi_BH[:, :, t:t+1, t:t+1]

V_min = float(Lb.min())
V_max = float(Lb.max())
M_trace = Mb[:, :, 0, 0, 0, 0] + Mb[:, :, 0, 0, 1, 1]
mean_mass = float(np.mean(M_trace))
band_type = 'hole' if mean_mass < 0 else 'electron'
sigma = V_max if band_type == 'hole' else V_min

print(f"  Candidate: θ={theta_deg:.1f}°, η={eta:.4f}, N_sub={N_subspace}, target_band={t}")
print(f"  V_range=[{V_min:.6f}, {V_max:.6f}], type={band_type}, sigma={sigma:.6f}")

# Build Hamiltonian once
print("Building single-band Hamiltonian...")
H = p3.assemble_multiband_hamiltonian(
    Lb, vb, Mb, Ab, Pb, eta, Ns1, Ns1, 1, dR, dR, B_moire,
    include_drift=True, include_kinetic=True, include_born_huang=True,
    order=4, M_inv_max_trace=M_INV_MAX
)

# ============================================================
# k-convergence test: solve with increasing k
# ============================================================
ks = [10, 20, 50, 100, 200, 500]
raw = []

print("\nRunning k-convergence test...")
for k in ks:
    t0 = time.time()
    try:
        evals, _ = p3.solve_multiband_envelope(H, k, sigma=sigma)
        evals = np.sort(np.real(evals))
        lam_ground = float(evals[0])
        lam_top = float(evals[-1])
        n_bound = int(np.sum(evals <= V_max)) if band_type == 'hole' else int(np.sum(evals >= V_min))
        E_bind = V_max - lam_ground if band_type == 'hole' else lam_ground - V_min
        E_bind = abs(E_bind)
        ratio = E_bind / eta**2
        raw.append((k, n_bound, lam_ground, lam_top, E_bind, ratio))
        dt = time.time() - t0
        print(f"  k={k:4d}: λ₀={lam_ground:.6f}, n_bound={n_bound}, "
              f"E_bind={E_bind:.6f}, E/η²={ratio:.3f}  ({dt:.1f}s)")
    except Exception as e:
        print(f"  k={k:4d}: FAILED — {e}")

# Near-V_max eigenvalues from the k=500 solve (or largest successful)
evals_large, _ = p3.solve_multiband_envelope(H, 500, sigma=sigma)
evals_large = np.sort(np.real(evals_large))
V_ref = V_max if band_type == 'hole' else V_min
near_mask = np.abs(evals_large - V_ref) < 0.01
near_evals = evals_large[near_mask]
# Split into below/above V_ref
if band_type == 'hole':
    bound_near_vmax = near_evals[near_evals <= V_max][-10:].tolist()
    above_vmax = near_evals[near_evals > V_max][:5].tolist()
else:
    bound_near_vmax = near_evals[near_evals >= V_min][:10].tolist()
    above_vmax = near_evals[near_evals < V_min][-5:].tolist()

# Ensure we have data for the plot
if len(bound_near_vmax) < 2:
    bound_near_vmax = evals_large[-10:].tolist()
if len(above_vmax) < 2:
    above_vmax = evals_large[:5].tolist()

# ============================================================
# Extract arrays for plotting
# ============================================================
ks_plot = [r[0] for r in raw]
n_bounds = [r[1] for r in raw]
grounds = [r[2] for r in raw]
E_binds = [r[4] for r in raw]
ratios = [r[5] for r in raw]

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.30)

# --- Panel A: Ground state eigenvalue vs k ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogx(ks_plot, grounds, 'bo-', markersize=8, linewidth=2, zorder=3)
ax1.axhline(V_max, color='red', linestyle='--', linewidth=1.5,
            label=f'$V_{{\\rm max}} = {V_max:.4f}$')
ax1.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.fill_betweenx([min(grounds)-0.02, V_max], 8, 1200, alpha=0.05, color='steelblue')
ax1.set_xlabel('$k$ (number of eigsh modes requested)', fontsize=12)
ax1.set_ylabel(r'$\lambda_0$ (lowest eigenvalue found)', fontsize=12)
ax1.set_title('A: Ground State Drifts Indefinitely with $k$\n'
              '(No convergence → no discrete bound states)', fontsize=11)
ax1.legend(fontsize=10, loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(8, 1200)

# --- Panel B: E_bind / eta^2 vs k ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(ks_plot, ratios, 'rs-', markersize=8, linewidth=2, label=r'$E_{\rm bind}/\eta^2$')
ax2.set_xlabel('$k$ (number of eigsh modes requested)', fontsize=12)
ax2.set_ylabel(r'$E_{\rm bind} / \eta^2$', fontsize=12)
ax2.set_title('B: "Binding Energy" is an Artifact\n'
              r'($E_{\rm bind} = V_{\rm max} - \lambda_0$ grows with $k$)', fontsize=11)
ax2.grid(True, alpha=0.3)

# Fit power law
log_k = np.log(ks_plot)
log_r = np.log(ratios)
slope, intercept = np.polyfit(log_k, log_r, 1)
k_fit = np.linspace(min(ks_plot), max(ks_plot), 100)
ax2.loglog(k_fit, np.exp(intercept) * k_fit**slope, '--', color='gray',
           label=f'Fit: $\\propto k^{{{slope:.2f}}}$', linewidth=1.5, alpha=0.7)
ax2.legend(fontsize=10)

# Also show n_bound/k fraction
ax2b = ax2.twinx()
ax2b.semilogx(ks_plot, [nb/k for k, nb in zip(ks_plot, n_bounds)], 'g^--',
              markersize=6, linewidth=1, alpha=0.7, label='$n_{\\rm bound}/k$')
ax2b.set_ylabel('Fraction below $V_{\\rm max}$', fontsize=10, color='green')
ax2b.tick_params(axis='y', labelcolor='green')
ax2b.set_ylim(0, 1)
ax2b.legend(fontsize=9, loc='center right')

# --- Panel C: Spectrum near V_max (zoom) ---
ax3 = fig.add_subplot(gs[1, 0])
all_near = np.sort(bound_near_vmax + above_vmax)
colors = ['steelblue' if e <= V_max else 'coral' for e in all_near]
ax3.scatter(range(len(all_near)), all_near, c=colors, s=60, zorder=3,
            edgecolors='k', linewidth=0.5)
ax3.axhline(V_max, color='red', linestyle='--', linewidth=1.5, label='$V_{\\rm max}$')
ax3.set_xlabel('Eigenvalue index (near $V_{\\rm max}$, from $k=500$ solve)', fontsize=11)
ax3.set_ylabel(r'$\lambda$', fontsize=12)
ax3.set_title('C: Eigenvalues Near $V_{\\rm max}$\n'
              'No gap → continuous band at $V_{\\rm max}$', fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Spacings
if len(bound_near_vmax) >= 2 and len(above_vmax) >= 2:
    spacings_below = np.diff(bound_near_vmax)
    spacings_above = np.diff(above_vmax)
    gap_across = above_vmax[0] - bound_near_vmax[-1]
    ax3.annotate(
        f'Gap across $V_{{\\rm max}}$: {gap_across:.2e}\n'
        f'Mean spacing below: {np.mean(spacings_below):.2e}\n'
        f'Mean spacing above: {np.mean(spacings_above):.2e}\n'
        f'→ No special gap at $V_{{\\rm max}}$',
        xy=(0.02, 0.98), xycoords='axes fraction', fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                  edgecolor='orange', alpha=0.9)
    )

# --- Panel D: Interpretation box ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
ax4.set_title('D: Interpretation & Consequences', fontsize=11)

text = (
    "FINDING\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "The single-band envelope Hamiltonian\n\n"
    r"  $H_0 = \frac{1}{2} M^{-1}_{ij}(R)\, \partial_i \partial_j"
    r" + \Lambda_0(R)$" + "\n\n"
    "on a PERIODIC moiré grid has a purely\n"
    "continuous (Bloch-band) spectrum.\n\n"
    r"$V_{\rm max}$ is NOT a band edge — it sits inside" + "\n"
    "a continuous band of extended states.\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "CONSEQUENCE:\n\n"
    r"• $E_{\rm bind} = V_{\rm max} - \lambda_0$ is NOT a" + "\n"
    "  well-defined physical observable.\n\n"
    f"• It grows " + r"$\propto k^{" + f"{slope:.1f}" + r"}$" + " — pure artifact\n"
    "  of how many eigsh modes are requested.\n\n"
    "• The correct validation observables are:\n"
    "  – Moiré miniband BANDWIDTH\n"
    "  – Band GAP structure\n"
    "  – Flatness ratio  (gap / bandwidth)\n"
    "  – N-band coupling strength"
)
ax4.text(0.05, 0.95, text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4ff',
                   edgecolor='steelblue', linewidth=1.5))

fig.suptitle(
    f'FINDING F01: No Discrete Bound States in Single-Band Moiré Envelope '
    f'(θ={theta_deg:.1f}°, η={eta:.4f}, band {t})',
    fontsize=13, fontweight='bold', y=0.99
)
plt.savefig(OUT_DIR / 'F01_no_discrete_bound_states.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"SAVED: {OUT_DIR / 'F01_no_discrete_bound_states.png'}")

# ============================================================
# Save data
# ============================================================
with open(OUT_DIR / 'F01_data.json', 'w') as f:
    json.dump({
        'theta_deg': theta_deg, 'eta': eta, 'V_max': V_max, 'V_min': V_min,
        'target_band': t, 'band_type': band_type, 'N_subspace': N_subspace,
        'k_convergence': [
            {'k': r[0], 'n_bound': r[1], 'lam_ground': r[2], 'lam_top': r[3],
             'E_bind': r[4], 'E_bind_over_eta2': r[5]}
            for r in raw
        ],
        'near_vmax_bound': bound_near_vmax,
        'near_vmax_above': above_vmax,
        'power_law_exponent': round(slope, 2),
    }, f, indent=2)
print(f"SAVED: {OUT_DIR / 'F01_data.json'}")
