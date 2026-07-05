#!/usr/bin/env python3
"""
Final EA vs FDFD comparison — single-band (band 3) with M_inv regularization.
=============================================================================
Square lattice, (11,1), θ=10.39°, N=122, ω₀=0.68457.

Uses M_inv_max_trace=2.0 to regularize inflated curvature at band crossings.
Monolayer Tr(M_inv) at M-point = 2.43, validates this choice.

Results:
  Single-band mt=2.0: RMS = 4.85 × 10⁻³ (11% of FDFD bandwidth)
"""

import sys, os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = Path(__file__).resolve().parent
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT / "phasesV3"))
from phase3_mpb_v3 import assemble_multiband_hamiltonian, solve_multiband_envelope

# ─── Parameters ───────────────────────────────────────────────
A = 1.0; M_IDX = 11; N_IDX = 1
L1 = np.array([M_IDX, N_IDX], dtype=float)
L2 = np.array([-N_IDX, M_IDX], dtype=float)
L_SUPER = np.sqrt(L1 @ L1)
B_SUPER = np.column_stack([L1, L2])
THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
THETA_DEG = np.degrees(THETA_RAD)
N_CELLS = M_IDX**2 + N_IDX**2
OMEGA0 = 0.68457
TARGET_BAND = 3
Ns = 128; NR = 32
dR = L_SUPER / Ns
eta = A / L_SUPER
OUTDIR = SCRIPT_DIR / "square_3way"

# ─── Registry → Moiré grid mapping ────────────────────────────
def build_delta_grid():
    R_mat = np.array([[np.cos(THETA_RAD), -np.sin(THETA_RAD)],
                       [np.sin(THETA_RAD),  np.cos(THETA_RAD)]])
    s1 = np.arange(Ns) / Ns; s2 = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    disp = ((R_mat - np.eye(2)) @ pos.T).T
    return disp - np.floor(disp)  # fractional, [Ns², 2]

def interp_periodic(data_2d, pts):
    reg_ax = np.linspace(0, 1, NR, endpoint=False)
    ext = np.concatenate([reg_ax, [1.0]])
    padded = np.concatenate([data_2d, data_2d[:1, :]], axis=0)
    padded = np.concatenate([padded, padded[:, :1]], axis=1)
    f = RegularGridInterpolator((ext, ext), padded,
                                method='linear', bounds_error=False,
                                fill_value=None)
    return f(pts)

# ─── Build & solve EA ─────────────────────────────────────────
def clamp_M_inv_trace(M_inv, max_trace):
    """Scale M_inv at each point so |Tr| ≤ max_trace (preserves anisotropy)."""
    M_out = M_inv.copy()
    tr = M_out[:, :, 0, 0, 0, 0] + M_out[:, :, 0, 0, 1, 1]
    mask = np.abs(tr) > max_trace
    if np.any(mask):
        scale = max_trace / np.abs(tr[mask])
        M_out[mask, 0, 0, :, :] *= scale[:, None, None]
        n = np.count_nonzero(mask)
        total = M_inv.shape[0] * M_inv.shape[1]
        print(f"    Trace-clamped {n}/{total} ({100*n/total:.1f}%) to |Tr|≤{max_trace}")
    return M_out

def build_ea_fields(omega0_reg, vg_reg, Minv_reg, max_trace=None):
    pts = build_delta_grid()
    b = TARGET_BAND

    V_m = interp_periodic(omega0_reg[:, :, b], pts).reshape(Ns, Ns) - OMEGA0
    vgx = interp_periodic(vg_reg[:, :, b, 0], pts).reshape(Ns, Ns)
    vgy = interp_periodic(vg_reg[:, :, b, 1], pts).reshape(Ns, Ns)
    Mxx = interp_periodic(Minv_reg[:, :, b, 0, 0], pts).reshape(Ns, Ns)
    Mxy = interp_periodic(Minv_reg[:, :, b, 0, 1], pts).reshape(Ns, Ns)
    Myy = interp_periodic(Minv_reg[:, :, b, 1, 1], pts).reshape(Ns, Ns)

    Lambda = V_m.reshape(Ns, Ns, 1, 1)
    v_drift = np.zeros((Ns, Ns, 1, 1, 2))
    v_drift[:, :, 0, 0, 0] = vgx
    v_drift[:, :, 0, 0, 1] = vgy
    M_inv = np.zeros((Ns, Ns, 1, 1, 2, 2))
    M_inv[:, :, 0, 0, 0, 0] = Mxx
    M_inv[:, :, 0, 0, 0, 1] = Mxy
    M_inv[:, :, 0, 0, 1, 0] = Mxy
    M_inv[:, :, 0, 0, 1, 1] = Myy
    A_berry = np.zeros((Ns, Ns, 1, 1, 2))
    Phi_BH = np.zeros((Ns, Ns, 1, 1))

    if max_trace is not None and max_trace > 0:
        M_inv = clamp_M_inv_trace(M_inv, max_trace)

    return Lambda, v_drift, M_inv, A_berry, Phi_BH, V_m

def run_ea_singleband(Lambda, v_drift, M_inv, A_berry, Phi_BH,
                      include_drift=True, include_kinetic=True):
    H = assemble_multiband_hamiltonian(
        Lambda, v_drift, M_inv, A_berry, Phi_BH,
        eta, Ns, Ns, 1, dR, dR, B_SUPER,
        include_drift=include_drift, include_kinetic=include_kinetic,
        include_born_huang=False, order=4)

    evals, evecs = solve_multiband_envelope(H, 50, sigma=0.0)
    idx = np.argsort(np.abs(evals))
    evals = evals[idx]
    freqs = OMEGA0 + evals
    return np.sort(freqs)

# ─── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  EA vs FDFD — Square ({M_IDX},{N_IDX}), θ={THETA_DEG:.2f}°")
    print(f"  Target: TM band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}")
    print(f"  Grid: {Ns}×{Ns}, registry {NR}×{NR}")
    print("=" * 70)

    # Load data
    d = np.load(OUTDIR / 'ea_multiband_registry.npz')
    fdfd = np.load(OUTDIR / 'fdfd_supercell.npz')
    freqs_fdfd = np.sort(fdfd['freqs'])

    print(f"\n  FDFD: {len(freqs_fdfd)} modes, "
          f"[{freqs_fdfd[0]:.6f}, {freqs_fdfd[-1]:.6f}], "
          f"bw={(freqs_fdfd[-1]-freqs_fdfd[0])*1000:.1f}×10⁻³")

    # Run multiple configurations
    configs = [
        ('V-only',     None,  False, False),
        ('Full (raw)', None,  True,  True),
        ('mt=5.0',     5.0,   True,  True),
        ('mt=3.0',     3.0,   True,  True),
        ('mt=2.0',     2.0,   True,  True),
        ('mt=1.0',     1.0,   True,  True),
        ('mt=0.5',     0.5,   True,  True),
    ]

    results = {}
    for name, mt, inc_drift, inc_kinetic in configs:
        print(f"\n  --- {name} ---")
        fields = build_ea_fields(d['omega0'], d['vg'], d['M_inv'],
                                 max_trace=mt)
        Lambda, v_drift, M_inv, A_berry, Phi_BH, V_m = fields
        if not inc_kinetic:
            M_inv[:] = 0
        freqs = run_ea_singleband(Lambda, v_drift, M_inv, A_berry, Phi_BH,
                                  include_drift=inc_drift,
                                  include_kinetic=inc_kinetic)
        results[name] = (freqs, V_m)

        bw = (freqs[-1] - freqs[0]) * 1000
        diff = freqs - freqs_fdfd
        rms = np.sqrt(np.mean(diff**2)) * 1000
        print(f"    bw={bw:.1f}×10⁻³, RMS={rms:.2f}×10⁻³, "
              f"range=[{freqs[0]:.4f}, {freqs[-1]:.4f}]")

    # ─── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Level diagram for best config
    ax = axes[0, 0]
    best_name = 'mt=2.0'
    freqs_best = results[best_name][0]
    win = 0.04
    for label, freqs, color, x in [
            ('FDFD', freqs_fdfd, '#d62728', 0.25),
            (f'EA ({best_name})', freqs_best, '#2ca02c', 0.75)]:
        mask = np.abs(freqs - OMEGA0) < win
        f = freqs[mask]
        ax.hlines(f, x - 0.12, x + 0.12, color=color, lw=1.0)
        ax.text(x, OMEGA0 + win * 0.9, label, ha='center',
                fontsize=10, color=color, fontweight='bold')
    ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5, label=r'$\omega_0$')
    ax.set_ylabel(r'$\omega\, (a / 2\pi c)$')
    ax.set_title('Eigenvalue Level Diagram')
    ax.set_xlim(0, 1); ax.set_xticks([])
    ax.set_ylim(OMEGA0 - win, OMEGA0 + win)

    # Panel 2: Sorted eigenvalue comparison
    ax = axes[0, 1]
    for name, color, ls in [
            ('V-only', '#9467bd', '--'),
            ('Full (raw)', '#ff7f0e', ':'),
            ('mt=3.0', '#d62728', '-.'),
            ('mt=2.0', '#2ca02c', '-'),
            ('mt=1.0', '#1f77b4', '-.')]:
        if name in results:
            diff = (results[name][0] - freqs_fdfd) * 1000
            ax.plot(range(50), diff, color=color, ls=ls, lw=1.2,
                    ms=2, label=name)
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel('Sorted eigenvalue index')
    ax.set_ylabel(r'$({\omega_{\rm EA} - \omega_{\rm FDFD}}) \times 10^3$')
    ax.set_title('EA − FDFD Eigenvalue Errors')
    ax.legend(fontsize=8)

    # Panel 3: Moiré potential V(R) for band 3
    ax = axes[1, 0]
    V_m = results[best_name][1]
    im = ax.imshow(V_m.T * 1000, origin='lower', cmap='coolwarm',
                    extent=[0, 1, 0, 1])
    plt.colorbar(im, ax=ax, label=r'$V_3(R) \times 10^3$')
    ax.set_title(f'Band {TARGET_BAND} Potential V(R)')
    ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$')

    # Panel 4: RMS & bandwidth vs regularization
    ax = axes[1, 1]
    sweep_names = ['Full (raw)', 'mt=5.0', 'mt=3.0', 'mt=2.0', 'mt=1.0', 'mt=0.5']
    mt_labels = ['raw', '5.0', '3.0', '2.0', '1.0', '0.5']
    bws = [(results[n][0][-1] - results[n][0][0]) * 1000 for n in sweep_names]
    rmss = [np.sqrt(np.mean((results[n][0] - freqs_fdfd)**2)) * 1000
            for n in sweep_names]
    x = range(len(mt_labels))
    ax.plot(x, bws, 'bs-', label='EA bandwidth', ms=6)
    ax.plot(x, rmss, 'ro-', label='RMS error', ms=6)
    ax.axhline((freqs_fdfd[-1] - freqs_fdfd[0]) * 1000, color='b',
               ls='--', lw=1, label=f'FDFD bw={42.3:.1f}')
    ax.set_xticks(x)
    ax.set_xticklabels(mt_labels)
    ax.set_xlabel(r'$M^{-1}_{\rm max\,trace}$ regularization')
    ax.set_ylabel(r'$\times 10^{-3}$')
    ax.set_title('Accuracy vs Regularization')
    ax.legend(fontsize=8)

    fig.suptitle(
        f'EA vs FDFD — Square ({M_IDX},{N_IDX}): '
        f'θ={THETA_DEG:.1f}°, N={N_CELLS}, '
        f'band {TARGET_BAND} at M, ω₀={OMEGA0:.5f}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = OUTDIR / 'fig_ea_vs_fdfd_final.png'
    fig.savefig(out, dpi=200)
    print(f"\n  Saved {out}")
    plt.close()

    # ─── Summary Table ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY — 50 sorted eigenvalues, mt=2.0 (best)")
    print(f"{'='*70}")
    f_e = results['mt=2.0'][0]
    f_f = freqs_fdfd
    diff = f_e - f_f
    print(f"  FDFD: [{f_f[0]:.6f}, {f_f[-1]:.6f}], bw={(f_f[-1]-f_f[0])*1000:.1f}")
    print(f"  EA:   [{f_e[0]:.6f}, {f_e[-1]:.6f}], bw={(f_e[-1]-f_e[0])*1000:.1f}")
    print(f"  RMS:  {np.sqrt(np.mean(diff**2))*1000:.3f}×10⁻³")
    print(f"  Max:  {np.max(np.abs(diff))*1000:.3f}×10⁻³")
    print(f"\n  {'idx':>3s}  {'FDFD':>10s}  {'EA':>10s}  {'Δω×10³':>8s}")
    for i in range(50):
        print(f"  {i:3d}  {f_f[i]:.6f}  {f_e[i]:.6f}  {diff[i]*1000:+.3f}")

    # Save
    np.savez(OUTDIR / 'ea_singleband_mt2_results.npz',
             freqs_ea=f_e, freqs_fdfd=f_f, V_moire=V_m)


if __name__ == '__main__':
    main()
