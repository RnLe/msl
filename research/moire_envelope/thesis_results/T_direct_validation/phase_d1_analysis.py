"""
Phase D1 Analysis: Smoothed vs Binary ε — full EA-FDFD comparison.

Builds both ε grids, solves FDFD, saves eigenvalues, then produces:
  Figure 1: Level diagram + Hungarian matching (binary vs smoothed side by side)
  Figure 2: Residual comparison (raw and mean-shift corrected)
  Figure 3: Mode-by-mode ω scatter + spectral structure

Saves FDFD eigenvalues to .npz for future use.
"""
import numpy as np
import sys
import os
import time
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from T_direct_validation.subpixel_smoothing import build_smoothed_eps_supercell
from T_direct_validation.fdfd_solver import build_fdfd_operator
from T_direct_validation.supercell_geometry import build_supercell_eps
from T_direct_validation.commensurate_utils import commensurate_twist_angle

out_dir = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════
EPS_BG = 1.0
EPS_ROD = 11.56
R_OVER_A = 0.2
SUPERCELL_RES = 40
N_SUB = 16
M, N_MN = 30, 29
N_FDFD_MODES = 100

# File paths
NPZ_BINARY = os.path.join(out_dir, 'fdfd_dirac_m30_n29_res40_v2.npz')
NPZ_SMOOTHED = os.path.join(out_dir, 'fdfd_smoothed_m30_n29_res40.npz')

SWEEP_PATH = ('/home/renlephy/msl/research/moire_envelope/runsV3/'
              'thesis_honeycomb_K_b1_20260307_171424/'
              'eta_sweep_20260310_191610/sweep_results.json')


# ════════════════════════════════════════════════════════════════
# Data loading / solving
# ════════════════════════════════════════════════════════════════
def load_envelope_reference():
    with open(SWEEP_PATH) as f:
        data = json.load(f)
    entry = data[0]
    omega_ref = entry['omega_ref']
    evals = np.array(entry['eigenvalues'])
    freqs = np.sort(omega_ref + evals)
    return freqs, omega_ref, entry['theta_deg'], entry.get('eta', 0)


def solve_fdfd(eps_grid, sc_info, sigma_target, label=''):
    """Solve FDFD eigenproblem with CHOLMOD shift-invert."""
    import scipy.sparse as sp
    from sksparse.cholmod import cholesky
    from scipy.sparse.linalg import LinearOperator

    t0 = time.time()
    L = build_fdfd_operator(eps_grid, sc_info, q_vec=np.zeros(2), polarization='tm')
    t_assemble = time.time() - t0
    N_dof = L.shape[0]
    print(f"  [{label}] Operator: {N_dof:,} DOF, nnz={L.nnz:,}, assembly={t_assemble:.1f}s")

    L_shifted = (L - sigma_target * sp.eye(N_dof, format='csc')).tocsc()

    t0 = time.time()
    factor = cholesky(L_shifted, beta=0, mode='simplicial')
    t_factor = time.time() - t0
    print(f"  [{label}] CHOLMOD factorization: {t_factor:.1f}s")
    del L_shifted

    OPinv = LinearOperator((N_dof, N_dof), matvec=lambda b: factor(b), dtype=L.dtype)

    t0 = time.time()
    evals, _ = eigsh(L, k=N_FDFD_MODES, sigma=sigma_target, which='LM',
                     OPinv=OPinv, maxiter=10000, tol=1e-8)
    t_solve = time.time() - t0
    print(f"  [{label}] Eigsh: {t_solve:.1f}s")

    idx = np.argsort(evals)
    evals = evals[idx]
    freqs = np.sqrt(np.maximum(evals, 0)) / (2 * np.pi)
    return np.sort(freqs), evals, t_factor, t_solve


def get_fdfd_freqs():
    """Load or compute FDFD eigenvalues for both binary and smoothed ε."""
    env_freqs, omega_ref, theta_deg, eta = load_envelope_reference()
    env_center = 0.5 * (env_freqs.min() + env_freqs.max())
    sigma_target = (2 * np.pi * env_center) ** 2

    N_cells = M * M + M * N_MN + N_MN * N_MN
    Nx = int(round(np.sqrt(N_cells) * SUPERCELL_RES))

    # ── Binary: load existing ──
    if os.path.exists(NPZ_BINARY):
        print(f"Loading binary FDFD from {os.path.basename(NPZ_BINARY)}")
        d = np.load(NPZ_BINARY)
        freqs_binary = np.sort(d['freqs'])
    else:
        print("Binary .npz not found — need to solve")
        eps_binary, sc_info = build_supercell_eps(
            'honeycomb', m=M, n=N_MN, a=1.0,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=Nx, Ny=Nx,
        )
        freqs_binary, evals_b, t_f, t_s = solve_fdfd(eps_binary, sc_info, sigma_target, 'binary')
        np.savez(NPZ_BINARY,
                 freqs=freqs_binary, evals=evals_b,
                 m=M, n=N_MN, N_cells=N_cells, res=SUPERCELL_RES, Nx=Nx,
                 n_modes=N_FDFD_MODES, omega_target=env_center,
                 theta_deg=theta_deg, t_factor=t_f, t_solve=t_s)

    # ── Smoothed: load or compute ──
    if os.path.exists(NPZ_SMOOTHED):
        print(f"Loading smoothed FDFD from {os.path.basename(NPZ_SMOOTHED)}")
        d = np.load(NPZ_SMOOTHED)
        freqs_smoothed = np.sort(d['freqs'])
    else:
        print("Smoothed .npz not found — need to solve")
        # Build binary eps first (needed for smoothing)
        eps_binary_grid, sc_info = build_supercell_eps(
            'honeycomb', m=M, n=N_MN, a=1.0,
            r_over_a=R_OVER_A, eps_rod=EPS_ROD, eps_bg=EPS_BG,
            Nx=Nx, Ny=Nx,
        )
        print("Building smoothed epsilon...")
        t0 = time.time()
        eps_smoothed_grid, smooth_info = build_smoothed_eps_supercell(
            eps_binary_grid, sc_info, n_sub=N_SUB,
            eps_rod=EPS_ROD, eps_bg=EPS_BG,
        )
        print(f"  Smoothing: {time.time()-t0:.1f}s, {smooth_info['n_smoothed']} pixels smoothed")

        freqs_smoothed, evals_s, t_f, t_s = solve_fdfd(eps_smoothed_grid, sc_info, sigma_target, 'smoothed')
        np.savez(NPZ_SMOOTHED,
                 freqs=freqs_smoothed, evals=evals_s,
                 m=M, n=N_MN, N_cells=N_cells, res=SUPERCELL_RES, Nx=Nx,
                 n_modes=N_FDFD_MODES, omega_target=env_center,
                 theta_deg=theta_deg, t_factor=t_f, t_solve=t_s,
                 n_smoothed=smooth_info['n_smoothed'])

    return freqs_binary, freqs_smoothed, env_freqs, omega_ref, theta_deg, eta


# ════════════════════════════════════════════════════════════════
# Hungarian matching with mean-shift correction
# ════════════════════════════════════════════════════════════════
def hungarian_match(env_freqs, fdfd_all, margin=0.002):
    """Hungarian 1-to-1 matching of EA to FDFD modes."""
    fdfd_mask = (fdfd_all >= env_freqs.min() - margin) & (fdfd_all <= env_freqs.max() + margin)
    fdfd_w = fdfd_all[fdfd_mask]
    fdfd_w_global_idx = np.where(fdfd_mask)[0]

    cost = np.abs(env_freqs[:, None] - fdfd_w[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_fdfd = fdfd_w[col_ind]
    residuals = env_freqs[row_ind] - matched_fdfd  # signed: positive = EA higher

    # Identify unmatched FDFD in envelope range
    matched_global = set(fdfd_w_global_idx[col_ind])
    fdfd_in_env = fdfd_all[(fdfd_all >= env_freqs.min()) & (fdfd_all <= env_freqs.max())]
    fdfd_in_env_idx = np.where((fdfd_all >= env_freqs.min()) & (fdfd_all <= env_freqs.max()))[0]
    unmatched = [fdfd_all[i] for i in fdfd_in_env_idx if i not in matched_global]

    return {
        'row_ind': row_ind,
        'col_ind': col_ind,
        'matched_fdfd': matched_fdfd,
        'residuals': residuals,
        'abs_res': np.abs(residuals),
        'fdfd_in_env': fdfd_in_env,
        'unmatched_fdfd': unmatched,
        'fdfd_window': fdfd_w,
    }


def compute_stats(match, env_freqs, label=''):
    """Compute and print matching statistics."""
    r = match['abs_res']
    env_bw = env_freqs.max() - env_freqs.min()
    env_spacing = np.mean(np.diff(env_freqs))
    bw_fdfd = match['matched_fdfd'].max() - match['matched_fdfd'].min()

    stats = {
        'n_matched': len(match['row_ind']),
        'mean_abs': r.mean(),
        'max_abs': r.max(),
        'mean_pct_bw': r.mean() / env_bw * 100,
        'bw_ratio': bw_fdfd / env_bw,
        'mean_signed': match['residuals'].mean(),
        'within_1_spacing': (r < env_spacing).sum(),
        'within_2_spacing': (r < 2 * env_spacing).sum(),
        'n_in_env': len(match['fdfd_in_env']),
        'n_unmatched': len(match['unmatched_fdfd']),
    }

    if label:
        print(f"\n  {label}:")
        print(f"    Matched: {stats['n_matched']}/{len(env_freqs)}")
        print(f"    Mean |Δω|: {stats['mean_abs']:.6e}  ({stats['mean_pct_bw']:.2f}% BW)")
        print(f"    Max  |Δω|: {stats['max_abs']:.6e}")
        print(f"    Mean  Δω:  {stats['mean_signed']:+.6e}  (signed → shift direction)")
        print(f"    BW ratio (FDFD/EA): {stats['bw_ratio']:.4f}")
        print(f"    Within 1 spacing: {stats['within_1_spacing']}/{stats['n_matched']}")
        print(f"    FDFD in env window: {stats['n_in_env']} ({stats['n_unmatched']} other bands)")

    return stats


# ════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════
C_ENV = '#DC2626'      # red
C_BIN = '#2563EB'      # blue
C_SMO = '#059669'      # teal/green
C_MATCH = '#16A34A'    # green
C_EXTRA = '#7C3AED'    # purple
C_FADE = '#93C5FD'     # faded blue
C_SHIFTED = '#F59E0B'  # amber


def plot_fig1_level_diagrams(fdfd_bin, fdfd_smo, env_freqs, match_bin, match_smo, theta_deg, eta):
    """Figure 1: Side-by-side level diagrams with Hungarian matching."""
    env_bw = env_freqs.max() - env_freqs.min()
    view_pad = 0.3 * env_bw
    view_lo = env_freqs.min() - view_pad
    view_hi = env_freqs.max() + view_pad

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    for ax, fdfd_all, match, title, c_fdfd in [
        (axes[0], fdfd_bin, match_bin, '(a)  Binary ε', C_BIN),
        (axes[1], fdfd_smo, match_smo, '(b)  Smoothed ε', C_SMO),
    ]:
        fdfd_view = fdfd_all[(fdfd_all >= view_lo) & (fdfd_all <= view_hi)]

        # Background: envelope bandwidth region
        ax.axhspan(env_freqs.min(), env_freqs.max(), alpha=0.07, color=C_ENV, zorder=0)

        # FDFD modes on the left
        unmatched_set = set(np.round(match['unmatched_fdfd'], 10)) if match['unmatched_fdfd'] else set()
        for f in fdfd_view:
            in_env = env_freqs.min() <= f <= env_freqs.max()
            is_unmatched = any(abs(f - u) < 1e-10 for u in match['unmatched_fdfd']) if in_env else False
            if is_unmatched:
                ax.plot([0.03, 0.37], [f, f], '-', color=C_EXTRA, lw=1.0, alpha=0.6)
            elif in_env:
                ax.plot([0.03, 0.37], [f, f], '-', color=c_fdfd, lw=1.0, alpha=0.7)
            else:
                ax.plot([0.03, 0.37], [f, f], '-', color=C_FADE, lw=0.6, alpha=0.3)

        # Envelope modes on the right
        for f in env_freqs:
            ax.plot([0.63, 0.97], [f, f], '-', color=C_ENV, lw=1.2, alpha=0.8)

        # Hungarian connections
        for i in range(len(match['row_ind'])):
            ef = env_freqs[match['row_ind'][i]]
            ff = match['matched_fdfd'][i]
            ax.plot([0.37, 0.63], [ff, ef], '-', color=C_MATCH, lw=0.7, alpha=0.4)

        r = match['abs_res']
        stats_txt = (
            f"Mean |Δω| = {r.mean()*1e6:.0f}×10⁻⁶\n"
            f"= {r.mean()/env_bw*100:.1f}% of BW\n"
            f"BW ratio = {(match['matched_fdfd'].max()-match['matched_fdfd'].min())/env_bw:.3f}\n"
            f"Mean shift = {match['residuals'].mean()*1e6:+.0f}×10⁻⁶"
        )
        ax.text(0.98, 0.02, stats_txt, transform=ax.transAxes, fontsize=8.5,
                ha='right', va='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='#CBD5E1', alpha=0.95))

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(view_lo, view_hi)
        ax.set_xticks([0.2, 0.8])
        ax.set_xticklabels(['FDFD', 'Envelope'], fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')

    axes[0].set_ylabel('Frequency  ω  (c/a)', fontsize=12)

    legend_elts = [
        Line2D([0], [0], color=C_ENV, lw=2.5, label='Envelope (EA+BH)'),
        Line2D([0], [0], color=C_BIN, lw=2.5, label='FDFD binary ε'),
        Line2D([0], [0], color=C_SMO, lw=2.5, label='FDFD smoothed ε'),
        Line2D([0], [0], color=C_EXTRA, lw=2.5, label='Other folded bands'),
        Line2D([0], [0], color=C_MATCH, lw=1.5, label='Hungarian match'),
    ]
    fig.legend(handles=legend_elts, loc='upper center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle(f'EA vs FDFD: Binary vs Smoothed ε  |  θ = {theta_deg:.2f}°  |  (30,29)  res={SUPERCELL_RES}',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'fig_d1_level_diagrams.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved {os.path.basename(fname)}")
    plt.close(fig)


def plot_fig2_residuals(env_freqs, match_bin, match_smo, stats_bin, stats_smo,
                        match_smo_shifted, stats_smo_shifted, theta_deg):
    """Figure 2: Residual comparison — raw and mean-shift corrected."""
    env_bw = env_freqs.max() - env_freqs.min()
    env_spacing = np.mean(np.diff(env_freqs))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ── Row 1: Raw residuals ──
    # (a) Bar chart: binary vs smoothed
    ax = axes[0, 0]
    idx = np.arange(len(match_bin['residuals']))
    w = 0.35
    ax.barh(idx - w/2, match_bin['residuals'] * 1e6, height=w, color=C_BIN, alpha=0.8, label='Binary ε')
    ax.barh(idx + w/2, match_smo['residuals'] * 1e6, height=w, color=C_SMO, alpha=0.8, label='Smoothed ε')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_ylabel('Mode index (sorted by EA ω)')
    ax.set_xlabel('Δω = ω_EA − ω_FDFD  (×10⁻⁶ c/a)')
    ax.set_title('(a)  Signed residuals (raw)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # (b) |Δω| comparison scatter
    ax = axes[0, 1]
    ax.scatter(match_bin['abs_res'] * 1e6, match_smo['abs_res'] * 1e6,
               s=30, c=C_MATCH, edgecolors='white', linewidths=0.3, zorder=5)
    lim = max(match_bin['abs_res'].max(), match_smo['abs_res'].max()) * 1e6 * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.3, label='Equal')
    ax.set_xlabel('|Δω| binary  (×10⁻⁶ c/a)')
    ax.set_ylabel('|Δω| smoothed  (×10⁻⁶ c/a)')
    ax.set_title('(b)  Per-mode |Δω|: smoothed vs binary', fontsize=11, fontweight='bold')
    # Count improvement
    better = (match_smo['abs_res'] < match_bin['abs_res']).sum()
    worse = (match_smo['abs_res'] > match_bin['abs_res']).sum()
    ax.text(0.95, 0.05, f'Smoothed better: {better}\nSmoothed worse: {worse}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='#CBD5E1', alpha=0.9))
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # ── Row 2: Mean-shift corrected ──
    # (c) Bar chart after mean-shift
    ax = axes[1, 0]
    shift = match_smo['residuals'].mean()
    corrected_res = match_smo['residuals'] - shift
    ax.barh(idx - w/2, match_bin['residuals'] * 1e6, height=w, color=C_BIN, alpha=0.8, label='Binary ε')
    ax.barh(idx + w/2, corrected_res * 1e6, height=w, color=C_SHIFTED, alpha=0.8,
            label=f'Smoothed ε (shift={shift*1e6:+.0f}×10⁻⁶)')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_ylabel('Mode index')
    ax.set_xlabel('Δω  (×10⁻⁶ c/a)')
    ax.set_title('(c)  Residuals after mean-shift correction', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # (d) Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    mean_shift = match_smo['residuals'].mean()
    corrected_abs = np.abs(match_smo['residuals'] - mean_shift)

    table_data = [
        ['Metric', 'Binary ε', 'Smoothed ε', 'Smo. (shifted)'],
        ['Mean |Δω| (×10⁻⁶)',
         f"{stats_bin['mean_abs']*1e6:.1f}",
         f"{stats_smo['mean_abs']*1e6:.1f}",
         f"{corrected_abs.mean()*1e6:.1f}"],
        ['Max |Δω| (×10⁻⁶)',
         f"{stats_bin['max_abs']*1e6:.1f}",
         f"{stats_smo['max_abs']*1e6:.1f}",
         f"{corrected_abs.max()*1e6:.1f}"],
        ['Mean |Δω|/BW (%)',
         f"{stats_bin['mean_pct_bw']:.2f}",
         f"{stats_smo['mean_pct_bw']:.2f}",
         f"{corrected_abs.mean()/env_bw*100:.2f}"],
        ['BW ratio (FDFD/EA)',
         f"{stats_bin['bw_ratio']:.4f}",
         f"{stats_smo['bw_ratio']:.4f}",
         f"{stats_smo['bw_ratio']:.4f}"],
        ['Mean signed Δω (×10⁻⁶)',
         f"{match_bin['residuals'].mean()*1e6:+.1f}",
         f"{match_smo['residuals'].mean()*1e6:+.1f}",
         f"{(match_smo['residuals'] - mean_shift).mean()*1e6:+.1f}"],
        ['Within 1 spacing',
         f"{stats_bin['within_1_spacing']}/50",
         f"{stats_smo['within_1_spacing']}/50",
         f"{(corrected_abs < env_spacing).sum()}/50"],
        ['Other bands in window',
         f"{stats_bin['n_unmatched']}",
         f"{stats_smo['n_unmatched']}",
         '—'],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0, 0.05, 1, 0.9])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    # Header row styling
    for j in range(4):
        table[0, j].set_facecolor('#E2E8F0')
        table[0, j].set_text_props(fontweight='bold')
    # Highlight best value in each row
    for i in range(1, len(table_data)):
        for j in range(1, 4):
            table[i, j].set_facecolor('#F8FAFC')

    ax.set_title('(d)  Summary statistics', fontsize=11, fontweight='bold', pad=15)

    fig.suptitle(f'Residual Analysis: Binary vs Smoothed ε  |  θ = {theta_deg:.2f}°  |  res={SUPERCELL_RES}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(out_dir, 'fig_d1_residuals.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved {os.path.basename(fname)}")
    plt.close(fig)


def plot_fig3_spectral_structure(fdfd_bin, fdfd_smo, env_freqs, match_bin, match_smo, theta_deg):
    """Figure 3: ω scatter, spectral structure, DOS overlay."""
    env_bw = env_freqs.max() - env_freqs.min()
    env_spacing = np.mean(np.diff(env_freqs))
    mean_shift = match_smo['residuals'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── (a) ω_EA vs ω_FDFD scatter ──
    ax = axes[0, 0]
    ax.scatter(env_freqs[match_bin['row_ind']] * 1e3,
               match_bin['matched_fdfd'] * 1e3,
               s=30, c=C_BIN, edgecolors='white', linewidths=0.3,
               zorder=5, alpha=0.8, label='Binary ε')
    ax.scatter(env_freqs[match_smo['row_ind']] * 1e3,
               match_smo['matched_fdfd'] * 1e3,
               s=30, c=C_SMO, edgecolors='white', linewidths=0.3,
               zorder=5, alpha=0.8, label='Smoothed ε')
    lims = [(env_freqs.min() - 0.0003) * 1e3, (env_freqs.max() + 0.0003) * 1e3]
    ax.plot(lims, lims, 'k-', lw=1.5, alpha=0.25)
    ax.set_xlabel('ω (envelope)  ×10³ c/a')
    ax.set_ylabel('ω (FDFD)  ×10³ c/a')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title('(a)  Frequency agreement (ω-ω)', fontsize=11, fontweight='bold')

    # ── (b) Mode-by-mode residual vs mode index ──
    ax = axes[0, 1]
    ax.scatter(np.arange(len(match_bin['residuals'])),
               match_bin['residuals'] * 1e6,
               s=20, c=C_BIN, alpha=0.7, label='Binary ε', zorder=3)
    ax.scatter(np.arange(len(match_smo['residuals'])),
               match_smo['residuals'] * 1e6,
               s=20, c=C_SMO, alpha=0.7, label='Smoothed ε', zorder=4)
    # Mean-shift corrected
    ax.scatter(np.arange(len(match_smo['residuals'])),
               (match_smo['residuals'] - mean_shift) * 1e6,
               s=20, c=C_SHIFTED, alpha=0.7, marker='x', label='Smoothed (shifted)', zorder=5)

    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(match_bin['residuals'].mean() * 1e6, color=C_BIN, ls='--', lw=0.8, alpha=0.5)
    ax.axhline(match_smo['residuals'].mean() * 1e6, color=C_SMO, ls='--', lw=0.8, alpha=0.5)

    ax.set_xlabel('Mode index (sorted by ω_EA)')
    ax.set_ylabel('Δω = ω_EA − ω_FDFD  (×10⁻⁶ c/a)')
    ax.legend(fontsize=8)
    ax.set_title('(b)  Mode-by-mode residuals', fontsize=11, fontweight='bold')

    # ── (c) Mode spacing comparison ──
    ax = axes[1, 0]
    env_gaps = np.diff(env_freqs)
    bin_gaps = np.diff(match_bin['matched_fdfd'])
    smo_gaps = np.diff(match_smo['matched_fdfd'])

    ax.plot(env_gaps * 1e6, 'o-', ms=3, lw=0.8, color=C_ENV, label='Envelope', alpha=0.8)
    ax.plot(bin_gaps * 1e6, 's-', ms=3, lw=0.8, color=C_BIN, label='Binary ε', alpha=0.7)
    ax.plot(smo_gaps * 1e6, '^-', ms=3, lw=0.8, color=C_SMO, label='Smoothed ε', alpha=0.7)

    ax.set_xlabel('Gap index (between mode i and i+1)')
    ax.set_ylabel('Δω_gap  (×10⁻⁶ c/a)')
    ax.legend(fontsize=8)
    ax.set_title('(c)  Mode spacing (gap structure)', fontsize=11, fontweight='bold')

    # ── (d) Cumulative distribution (CDF) ──
    ax = axes[1, 1]
    # Normalize to [0, 1] relative to envelope center
    env_center = env_freqs.mean()
    def cdf_y(vals):
        sorted_v = np.sort(vals)
        return sorted_v, np.arange(1, len(sorted_v) + 1) / len(sorted_v)

    x_env, y_env = cdf_y(env_freqs)
    x_bin, y_bin = cdf_y(match_bin['matched_fdfd'])
    x_smo, y_smo = cdf_y(match_smo['matched_fdfd'])

    ax.step(x_env * 1e3, y_env, where='post', color=C_ENV, lw=2, label='Envelope', alpha=0.8)
    ax.step(x_bin * 1e3, y_bin, where='post', color=C_BIN, lw=1.5, label='Binary ε', alpha=0.7)
    ax.step(x_smo * 1e3, y_smo, where='post', color=C_SMO, lw=1.5, label='Smoothed ε', alpha=0.7)

    ax.set_xlabel('ω  ×10³ c/a')
    ax.set_ylabel('CDF')
    ax.legend(fontsize=9)
    ax.set_title('(d)  Cumulative spectral distribution', fontsize=11, fontweight='bold')

    fig.suptitle(f'Spectral Structure: Binary vs Smoothed ε  |  θ = {theta_deg:.2f}°  |  res={SUPERCELL_RES}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fname = os.path.join(out_dir, 'fig_d1_spectral.png')
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved {os.path.basename(fname)}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("Phase D1 Analysis: Smoothed vs Binary ε")
    print("=" * 70)

    # ── Get data ──
    freqs_bin, freqs_smo, env_freqs, omega_ref, theta_deg, eta = get_fdfd_freqs()
    env_bw = env_freqs.max() - env_freqs.min()
    env_spacing = np.mean(np.diff(env_freqs))

    print(f"\nEnvelope: {len(env_freqs)} modes, BW={env_bw:.6f}, spacing={env_spacing*1e6:.0f}×10⁻⁶")
    print(f"Binary FDFD: {len(freqs_bin)} modes, range [{freqs_bin.min():.6f}, {freqs_bin.max():.6f}]")
    print(f"Smoothed FDFD: {len(freqs_smo)} modes, range [{freqs_smo.min():.6f}, {freqs_smo.max():.6f}]")

    # ── Hungarian matching ──
    print(f"\n{'─' * 70}")
    print("HUNGARIAN MATCHING")
    print(f"{'─' * 70}")

    match_bin = hungarian_match(env_freqs, freqs_bin)
    stats_bin = compute_stats(match_bin, env_freqs, 'Binary ε')

    match_smo = hungarian_match(env_freqs, freqs_smo)
    stats_smo = compute_stats(match_smo, env_freqs, 'Smoothed ε')

    # ── Mean-shift correction ──
    print(f"\n{'─' * 70}")
    print("MEAN-SHIFT CORRECTION")
    print(f"{'─' * 70}")

    shift_bin = match_bin['residuals'].mean()
    shift_smo = match_smo['residuals'].mean()
    print(f"  Binary  mean signed Δω: {shift_bin*1e6:+.1f}×10⁻⁶")
    print(f"  Smoothed mean signed Δω: {shift_smo*1e6:+.1f}×10⁻⁶")

    # Correct smoothed: subtract mean shift from FDFD freqs and re-match
    freqs_smo_shifted = freqs_smo - shift_smo  # shift FDFD to align centers
    # Wait — the shift is ω_EA - ω_FDFD, so to correct FDFD: add shift
    # Actually: residual = ω_EA - ω_FDFD > 0 means FDFD is lower → shift FDFD up
    # But we just subtract the mean residual from the residuals:
    corrected_res = match_smo['residuals'] - shift_smo
    corrected_abs = np.abs(corrected_res)

    # Also do the same for binary
    corrected_bin = match_bin['residuals'] - shift_bin
    corrected_bin_abs = np.abs(corrected_bin)

    print(f"\n  After removing mean shift:")
    print(f"    Binary:   mean |Δω| = {corrected_bin_abs.mean()*1e6:.1f}×10⁻⁶  ({corrected_bin_abs.mean()/env_bw*100:.2f}% BW)")
    print(f"    Smoothed: mean |Δω| = {corrected_abs.mean()*1e6:.1f}×10⁻⁶  ({corrected_abs.mean()/env_bw*100:.2f}% BW)")

    improvement = corrected_bin_abs.mean() / corrected_abs.mean() if corrected_abs.mean() > 0 else float('inf')
    print(f"    Improvement (smoothed/binary): {improvement:.2f}×")

    # Create a fake match dict for the shifted version (for plotting)
    match_smo_shifted = {
        'row_ind': match_smo['row_ind'],
        'col_ind': match_smo['col_ind'],
        'matched_fdfd': match_smo['matched_fdfd'],
        'residuals': corrected_res,
        'abs_res': corrected_abs,
        'fdfd_in_env': match_smo['fdfd_in_env'],
        'unmatched_fdfd': match_smo['unmatched_fdfd'],
        'fdfd_window': match_smo['fdfd_window'],
    }
    stats_smo_shifted = compute_stats(match_smo_shifted, env_freqs, 'Smoothed ε (mean-shift corrected)')

    # ── Plots ──
    print(f"\n{'─' * 70}")
    print("GENERATING PLOTS")
    print(f"{'─' * 70}")

    plot_fig1_level_diagrams(freqs_bin, freqs_smo, env_freqs, match_bin, match_smo, theta_deg, eta)
    plot_fig2_residuals(env_freqs, match_bin, match_smo, stats_bin, stats_smo, match_smo_shifted, stats_smo_shifted, theta_deg)
    plot_fig3_spectral_structure(freqs_bin, freqs_smo, env_freqs, match_bin, match_smo, theta_deg)

    # ── Final summary ──
    print(f"\n{'═' * 70}")
    print("FINAL SUMMARY")
    print(f"{'═' * 70}")
    print(f"θ = {theta_deg:.4f}°, (m,n)=({M},{N_MN}), res={SUPERCELL_RES}")
    print(f"")
    print(f"{'Metric':<30} {'Binary':>12} {'Smoothed':>12} {'Smo.(shift)':>12}")
    print(f"{'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    print(f"{'Mean |Δω| (×10⁻⁶)':<30} {stats_bin['mean_abs']*1e6:>12.1f} {stats_smo['mean_abs']*1e6:>12.1f} {corrected_abs.mean()*1e6:>12.1f}")
    print(f"{'Max  |Δω| (×10⁻⁶)':<30} {stats_bin['max_abs']*1e6:>12.1f} {stats_smo['max_abs']*1e6:>12.1f} {corrected_abs.max()*1e6:>12.1f}")
    print(f"{'Mean |Δω|/BW (%)':<30} {stats_bin['mean_pct_bw']:>12.2f} {stats_smo['mean_pct_bw']:>12.2f} {corrected_abs.mean()/env_bw*100:>12.2f}")
    print(f"{'BW ratio (FDFD/EA)':<30} {stats_bin['bw_ratio']:>12.4f} {stats_smo['bw_ratio']:>12.4f} {stats_smo['bw_ratio']:>12.4f}")
    print(f"{'Mean signed Δω (×10⁻⁶)':<30} {shift_bin*1e6:>+12.1f} {shift_smo*1e6:>+12.1f} {'0.0':>12}")
    print(f"{'Within 1 spacing':<30} {stats_bin['within_1_spacing']:>12} {stats_smo['within_1_spacing']:>12} {(corrected_abs < env_spacing).sum():>12}")
    print(f"{'Other bands in window':<30} {stats_bin['n_unmatched']:>12} {stats_smo['n_unmatched']:>12} {'—':>12}")


if __name__ == '__main__':
    main()
