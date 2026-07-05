#!/usr/bin/env python3
"""
Improved EA vs FDFD plots — addresses scaling, labeling, and per-eigenvalue issues.
===================================================================================
Loads saved data only — no solver re-runs.

Fixes:
  - Auto-scaled level diagrams (not fixed ±0.04)
  - Consistent axis labels: Δω [10⁻³ c/a]
  - Per-eigenvalue matching (index vs error)
  - Relative errors (normalised to mean spacing)
  - mt=0.5 highlighted as best at 2°
"""

import sys, os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OMEGA0 = 0.68457

# Regenerate 10° sweep from registry (fast — just numpy eigsolve, < 5s)
MOIRE_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MOIRE_ROOT / "phasesV3"))


# ════════════════════════════════════════════════════════════════
#  10° DATA: regenerate all configs from saved registry + FDFD
# ════════════════════════════════════════════════════════════════
def load_10deg():
    """Regenerate 10° sweep results from saved registry + FDFD."""
    from phase3_mpb_v3 import assemble_multiband_hamiltonian, solve_multiband_envelope
    from scipy.interpolate import RegularGridInterpolator

    DIR = SCRIPT_DIR / "square_3way"
    M_IDX, N_IDX = 11, 1
    L1 = np.array([M_IDX, N_IDX], dtype=float)
    L2 = np.array([-N_IDX, M_IDX], dtype=float)
    L_SUPER = np.sqrt(L1 @ L1)
    B_SUPER = np.column_stack([L1, L2])
    THETA_RAD = 2 * np.arctan2(N_IDX, M_IDX)
    Ns = 128; NR = 32; TARGET_BAND = 3
    dR = L_SUPER / Ns
    eta = 1.0 / L_SUPER

    reg = np.load(DIR / 'ea_multiband_registry.npz')
    fdfd = np.load(DIR / 'fdfd_supercell.npz')
    freqs_fdfd = np.sort(fdfd['freqs'])

    omega0_reg = reg['omega0']  # (NR, NR, N_bands)
    vg_reg = reg['vg']
    Minv_reg = reg['M_inv']

    # Build delta grid
    R_mat = np.array([[np.cos(THETA_RAD), -np.sin(THETA_RAD)],
                       [np.sin(THETA_RAD),  np.cos(THETA_RAD)]])
    s1 = np.arange(Ns) / Ns; s2 = np.arange(Ns) / Ns
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')
    X = S1 * L1[0] + S2 * L2[0]
    Y = S1 * L1[1] + S2 * L2[1]
    pos = np.stack([X.ravel(), Y.ravel()], axis=-1)
    disp = ((R_mat - np.eye(2)) @ pos.T).T
    pts = disp - np.floor(disp)

    def interp_periodic(data_2d, pts):
        reg_ax = np.linspace(0, 1, NR, endpoint=False)
        ext = np.concatenate([reg_ax, [1.0]])
        padded = np.concatenate([data_2d, data_2d[:1, :]], axis=0)
        padded = np.concatenate([padded, padded[:, :1]], axis=1)
        f = RegularGridInterpolator((ext, ext), padded,
                                    method='linear', bounds_error=False,
                                    fill_value=None)
        return f(pts)

    b = TARGET_BAND
    V_m = interp_periodic(omega0_reg[:, :, b], pts).reshape(Ns, Ns) - OMEGA0
    vgx = interp_periodic(vg_reg[:, :, b, 0], pts).reshape(Ns, Ns)
    vgy = interp_periodic(vg_reg[:, :, b, 1], pts).reshape(Ns, Ns)
    Mxx = interp_periodic(Minv_reg[:, :, b, 0, 0], pts).reshape(Ns, Ns)
    Mxy = interp_periodic(Minv_reg[:, :, b, 0, 1], pts).reshape(Ns, Ns)
    Myy = interp_periodic(Minv_reg[:, :, b, 1, 1], pts).reshape(Ns, Ns)

    def build_fields(max_trace=None, include_kinetic=True):
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

        if not include_kinetic:
            M_inv[:] = 0

        if max_trace is not None and max_trace > 0:
            tr = M_inv[:, :, 0, 0, 0, 0] + M_inv[:, :, 0, 0, 1, 1]
            mask = np.abs(tr) > max_trace
            if np.any(mask):
                scale = max_trace / np.abs(tr[mask])
                M_inv[mask, 0, 0, :, :] *= scale[:, None, None]

        return Lambda, v_drift, M_inv, A_berry, Phi_BH

    configs = [
        # (name, mt, include_drift, include_kinetic)
        ('V-only',      None,  False, False),
        ('Full (raw)',   None,  True,  True),
        ('mt=5.0',       5.0,  True,  True),
        ('mt=3.0',       3.0,  True,  True),
        ('mt=2.0',       2.0,  True,  True),
        ('mt=1.0',       1.0,  True,  True),
        ('mt=0.5',       0.5,  True,  True),
    ]

    results = {}
    for name, mt, inc_drift, inc_kin in configs:
        Lambda, v_drift, M_inv, A_berry, Phi_BH = build_fields(mt, inc_kin)
        H = assemble_multiband_hamiltonian(
            Lambda, v_drift, M_inv, A_berry, Phi_BH,
            eta, Ns, Ns, 1, dR, dR, B_SUPER,
            include_drift=inc_drift,
            include_kinetic=inc_kin,
            include_born_huang=False, order=4)
        evals, _ = solve_multiband_envelope(H, 50, sigma=0.0)
        idx = np.argsort(np.abs(evals))
        results[name] = np.sort(OMEGA0 + evals[idx])
        print(f"  10° {name}: done")

    return freqs_fdfd, results


# ════════════════════════════════════════════════════════════════
#  2° DATA: load from saved npz
# ════════════════════════════════════════════════════════════════
def load_2deg():
    DIR = SCRIPT_DIR / "square_2deg"
    d = np.load(DIR / 'ea_2deg_results.npz')
    freqs_fdfd = d['freqs_fdfd']

    results = {}
    key_map = {
        'freqs_ea_V-only': 'V-only',
        'freqs_ea_Full_raw': 'Full (raw)',
        'freqs_ea_mt=5.0': 'mt=5.0',
        'freqs_ea_mt=3.0': 'mt=3.0',
        'freqs_ea_mt=2.0': 'mt=2.0',
        'freqs_ea_mt=1.0': 'mt=1.0',
        'freqs_ea_mt=0.5': 'mt=0.5',
    }
    for npz_key, name in key_map.items():
        if npz_key in d:
            results[name] = d[npz_key]

    return freqs_fdfd, results


# ════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════
def make_combined_figure(data_10, data_2):
    freqs_fdfd_10, res_10 = data_10
    freqs_fdfd_2, res_2 = data_2

    bw_10 = freqs_fdfd_10[-1] - freqs_fdfd_10[0]
    bw_2 = freqs_fdfd_2[-1] - freqs_fdfd_2[0]
    N = 50
    spacing_10 = bw_10 / (N - 1)
    spacing_2 = bw_2 / (N - 1)

    # Best configs per angle
    best_10 = 'mt=2.0'
    best_2 = 'mt=0.5'

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # ──── Row 0: Level diagrams (auto-scaled) with matching lines ──
    for col, (ff, res, best, angle_label, bw_val) in enumerate([
        (freqs_fdfd_10, res_10, best_10, r'$\theta=10.4°$', bw_10),
        (freqs_fdfd_2,  res_2,  best_2,  r'$\theta=2.0°$',  bw_2),
    ]):
        ax = fig.add_subplot(gs[0, col])
        freqs_best = res[best]

        # Auto-scale: window covers both FDFD and EA ranges with margin
        all_freqs = np.concatenate([ff, freqs_best])
        center = 0.5 * (all_freqs.min() + all_freqs.max())
        half_range = 0.6 * (all_freqs.max() - all_freqs.min())
        half_range = max(half_range, 0.6 * bw_val)

        x_fdfd, x_ea = 0.25, 0.75
        for label, freqs, color, x in [
                ('FDFD', ff, '#d62728', x_fdfd),
                (f'EA ({best})', freqs_best, '#2ca02c', x_ea)]:
            ax.hlines(freqs, x - 0.12, x + 0.12, color=color, lw=0.8, alpha=0.8)
            ax.text(x, center + half_range * 0.88, label, ha='center',
                    fontsize=9, color=color, fontweight='bold')

        # Connecting lines: thin gray lines from FDFD[i] to EA[i]
        for i in range(len(ff)):
            color_line = '#aaaaaa' if abs(ff[i] - freqs_best[i]) < 0.3 * bw_val / N else '#ff000033'
            ax.plot([x_fdfd + 0.12, x_ea - 0.12], [ff[i], freqs_best[i]],
                    color='#cccccc', lw=0.3, alpha=0.5)

        ax.axhline(OMEGA0, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.set_ylabel(r'$\omega\;[c/a]$')
        ax.set_title(f'Level Diagram — {angle_label}\nFDFD bw = {bw_val*1e3:.2f}' + r'$\times 10^{-3}$')
        ax.set_xlim(0, 1); ax.set_xticks([])
        ax.set_ylim(center - half_range, center + half_range)

    # ──── Row 1: Per-eigenvalue absolute errors ──────────────
    plot_configs = [
        ('V-only', '#9467bd', '--', 1.0),
        ('Full (raw)', '#ff7f0e', ':', 1.0),
        ('mt=3.0', '#d62728', '-.', 1.0),
        ('mt=2.0', '#2ca02c', '-', 1.5),
        ('mt=1.0', '#1f77b4', '-.', 1.0),
        ('mt=0.5', '#e377c2', '-', 1.5),
    ]

    for col, (ff, res, best, angle_label, bw_val) in enumerate([
        (freqs_fdfd_10, res_10, best_10, r'$10.4°$', bw_10),
        (freqs_fdfd_2,  res_2,  best_2,  r'$2.0°$',  bw_2),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for name, color, ls, lw in plot_configs:
            if name in res:
                diff = (res[name] - ff) * 1e3
                marker = 'o' if name == best else ''
                ms = 3 if name == best else 0
                ax.plot(range(N), diff, color=color, ls=ls, lw=lw,
                        marker=marker, ms=ms, label=name)
        ax.axhline(0, color='k', ls='-', lw=0.3)
        ax.set_xlabel('Sorted eigenvalue index $n$')
        ax.set_ylabel(r'$\Delta\omega_n\;[10^{-3}\,c/a]$')
        ax.set_title(f'Per-eigenvalue error — {angle_label}')
        ax.legend(fontsize=7, ncol=2, loc='best')

    # ──── Row 2: Relative error & regularization sweep ───────
    for col, (ff, res, best, angle_label, bw_val, sp) in enumerate([
        (freqs_fdfd_10, res_10, best_10, r'$10.4°$', bw_10, spacing_10),
        (freqs_fdfd_2,  res_2,  best_2,  r'$2.0°$',  bw_2,  spacing_2),
    ]):
        ax = fig.add_subplot(gs[2, col])

        # Exclude V-only from regularization sweep (not a regularization config)
        sweep_names = ['Full (raw)', 'mt=5.0', 'mt=3.0',
                        'mt=2.0', 'mt=1.0', 'mt=0.5']
        sweep_names = [n for n in sweep_names if n in res]
        labels = [n.replace('Full (raw)', 'raw') for n in sweep_names]

        rmss = [np.sqrt(np.mean((res[n] - ff)**2)) / bw_val * 100
                for n in sweep_names]
        maxs = [np.max(np.abs(res[n] - ff)) / bw_val * 100
                for n in sweep_names]
        bws_ratio = [(res[n][-1] - res[n][0]) / bw_val * 100
                     for n in sweep_names]

        x = np.arange(len(labels))
        ax.bar(x - 0.2, rmss, 0.35, color='#d62728', alpha=0.8, label='RMS / bw [%]')
        ax.bar(x + 0.2, maxs, 0.35, color='#ff7f0e', alpha=0.8, label='max / bw [%]')
        ax.axhline(100, color='gray', ls='--', lw=0.8, label='FDFD bw')

        # Mark best
        best_idx = sweep_names.index(best) if best in sweep_names else -1
        if best_idx >= 0:
            ax.annotate('best', (best_idx - 0.2, rmss[best_idx]),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8, fontweight='bold', color='#d62728')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel(r'Regularization ($M^{-1}$ trace clamp)')
        ax.set_ylabel('Relative error [% of FDFD bandwidth]')
        ax.set_title(f'Regularization sweep — {angle_label}')
        ax.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        r'EA vs FDFD Eigenvalue Comparison — Square lattice, TM band 3 at $M$, $\omega_0=$'
        f'{OMEGA0:.5f}',
        fontsize=14, fontweight='bold')

    out = SCRIPT_DIR / 'fig_improved_comparison.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"\nSaved {out}")
    plt.close()
    return out


def print_detailed_table(freqs_fdfd, res, angle_label, best_name):
    """Print per-eigenvalue comparison for the best config."""
    N = len(freqs_fdfd)
    bw = freqs_fdfd[-1] - freqs_fdfd[0]
    spacing = bw / (N - 1)
    f_ea = res[best_name]
    diff = f_ea - freqs_fdfd

    print(f"\n{'='*78}")
    print(f"  {angle_label} — Best config: {best_name}")
    print(f"  FDFD bandwidth = {bw*1e3:.2f} × 10⁻³,  mean spacing = {spacing*1e3:.3f} × 10⁻³")
    print(f"{'='*78}")
    print(f"  {'n':>3s}  {'ω_FDFD':>10s}  {'ω_EA':>10s}  "
          f"{'Δω [10⁻³]':>10s}  {'Δω/spacing':>10s}  {'Δω/bw [%]':>10s}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")

    for i in range(N):
        rel_spacing = diff[i] / spacing if spacing > 0 else 0.0
        rel_bw = diff[i] / bw * 100 if bw > 0 else 0.0
        print(f"  {i:3d}  {freqs_fdfd[i]:.6f}  {f_ea[i]:.6f}  "
              f"{diff[i]*1e3:+10.4f}  {rel_spacing:+10.3f}  {rel_bw:+10.2f}")

    rms = np.sqrt(np.mean(diff**2))
    mx = np.max(np.abs(diff))
    print(f"\n  RMS  = {rms*1e3:.4f} × 10⁻³  ({rms/bw*100:.1f}% of bw)")
    print(f"  max  = {mx*1e3:.4f} × 10⁻³  ({mx/bw*100:.1f}% of bw)")
    print(f"  RMS/spacing = {rms/spacing:.3f}")

    # All configs summary
    print(f"\n  Config comparison:")
    print(f"  {'Config':>12s}  {'RMS [10⁻³]':>10s}  {'max [10⁻³]':>10s}  "
          f"{'RMS/bw [%]':>10s}  {'RMS/Δ':>8s}  {'bw_EA/bw_F':>10s}")
    for name in ['V-only', 'Full (raw)', 'mt=5.0', 'mt=3.0',
                  'mt=2.0', 'mt=1.0', 'mt=0.5']:
        if name not in res:
            continue
        d = res[name] - freqs_fdfd
        r = np.sqrt(np.mean(d**2))
        m = np.max(np.abs(d))
        bw_ea = res[name][-1] - res[name][0]
        star = ' ◀' if name == best_name else ''
        print(f"  {name:>12s}  {r*1e3:10.4f}  {m*1e3:10.4f}  "
              f"{r/bw*100:10.1f}  {r/spacing:8.3f}  {bw_ea/bw:10.2f}{star}")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════
def main():
    print("Loading and regenerating 10° data...")
    data_10 = load_10deg()

    print("\nLoading 2° data from saved npz...")
    data_2 = load_2deg()

    make_combined_figure(data_10, data_2)

    print_detailed_table(data_10[0], data_10[1], 'θ = 10.4°', 'mt=2.0')
    print_detailed_table(data_2[0], data_2[1], 'θ = 2.0°', 'mt=0.5')


if __name__ == '__main__':
    main()
