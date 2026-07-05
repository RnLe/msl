#!/usr/bin/env python3
"""
T11: Dense Miniband Validation Suite
======================================

Comprehensive diagnostics to validate the "dense miniband landscape" claim.
Consumes η-sweep data (Phase 3 at multiple θ) and miniband dispersion data
to produce five independent lines of evidence:

  1. Level-spacing statistics   (Poisson vs GOE)
  2. DOS evolution              (histogram of eigenvalues vs θ)
  3. Scaling laws               (BW, gap, IPR, mode count vs θ)
  4. Subspace validity          (BW/ω₀, Born-Huang magnitude)
  5. Single-band vs multi-band  (N=1 vs N=5 eigenvalue comparison)

Input:
  - η-sweep results JSON  (from run_eta_sweep.py → sweep_results.json)
  - Phase 3 mode stats    (per-angle phase3_mode_stats.json)
  - Phase 2 data          (for Born-Huang / subspace validity)

Output:
  - T11_level_statistics.{png,pdf}
  - T11_dos_evolution.{png,pdf}
  - T11_scaling_laws.{png,pdf}
  - T11_subspace_validity.{png,pdf}
  - T11_single_vs_multi.{png,pdf}
  - T11_summary.json

Usage:
  python compute.py square_M_b3
  python compute.py --all
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))
sys.path.insert(0, str(THESIS_DIR))

from thesis_utils import (
    find_thesis_run_dir, find_candidate_dir,
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names,
)

# ── Thesis style colors ───────────────────────────────────────────────
SKY_BLUE    = '#4E9AE1'
STARK_ORANGE = '#EBA538'
STEEL_BLUE  = '#4D7B9E'
LIGHT_STEEL = '#A5C6DF'
DARK_GRAY   = '#3A3A3A'


# =====================================================================
#  Data loading helpers
# =====================================================================

def find_eta_sweep_dir(candidate_name):
    """Find the latest eta_sweep_* directory for a candidate."""
    run_dir = find_thesis_run_dir(candidate_name)
    sweep_dirs = sorted(run_dir.glob("eta_sweep_*"))
    if not sweep_dirs:
        raise FileNotFoundError(
            f"No eta_sweep_* directory in {run_dir}. Run eta-sweep first.")
    return sweep_dirs[-1]


def load_sweep_results(sweep_dir):
    """Load sweep_results.json from an eta_sweep directory."""
    json_path = sweep_dir / "sweep_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    with open(json_path) as f:
        results = json.load(f)
    # Filter out failed runs
    return [r for r in results if 'error' not in r]


def load_per_angle_modes(sweep_dir, results):
    """Load Phase 3 mode_stats.json for each θ in the sweep.
    
    The eta_sweep puts each angle in sweep_dir/theta_X.XXX/candidate_0000/
    """
    angle_modes = {}
    for r in results:
        theta_label = f"theta_{r['theta_deg']:.3f}"
        mode_json = sweep_dir / theta_label / "candidate_0000" / "phase3_mode_stats.json"
        if mode_json.exists():
            with open(mode_json) as f:
                angle_modes[r['theta_deg']] = json.load(f)
    return angle_modes


def load_single_run_modes(candidate_name):
    """Load mode stats from the main (non-sweep) Phase 3 run."""
    cand_dir = find_candidate_dir(candidate_name)
    mode_json = cand_dir / "phase3_mode_stats.json"
    if not mode_json.exists():
        return None
    with open(mode_json) as f:
        return json.load(f)


# =====================================================================
#  1. Level-Spacing Statistics
# =====================================================================

def compute_level_spacings(eigenvalues):
    """Compute normalized nearest-neighbor level spacings.
    
    Returns s_n = (λ_{n+1} - λ_n) / <δ>, normalized to mean 1.
    """
    evals = np.sort(eigenvalues)
    spacings = np.diff(evals)
    if len(spacings) == 0 or spacings.mean() == 0:
        return np.array([])
    return spacings / spacings.mean()


def poisson_pdf(s):
    """P(s) = exp(-s) — uncorrelated levels."""
    return np.exp(-s)


def goe_pdf(s):
    """P(s) = (π/2) s exp(-πs²/4) — GOE level repulsion."""
    return (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4)


def plot_level_statistics(results, angle_modes, sweep_dir, candidate_name):
    """Plot level-spacing histograms at multiple θ values."""
    apply_thesis_style()
    
    # Get angles that have mode data
    angles_with_data = sorted(angle_modes.keys())
    n_angles = min(len(angles_with_data), 6)  # max 6 panels
    if n_angles == 0:
        print("  No per-angle mode data found for level statistics.")
        return None
    
    # Select representative angles (spread across range)
    if n_angles <= 4:
        selected = angles_with_data[:n_angles]
    else:
        indices = np.linspace(0, len(angles_with_data)-1, min(6, len(angles_with_data)),
                              dtype=int)
        selected = [angles_with_data[i] for i in indices]

    ncols = min(3, len(selected))
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    
    s_ref = np.linspace(0, 4, 200)
    
    summary_stats = {}
    for idx, theta_deg in enumerate(selected):
        ax = axes[idx // ncols, idx % ncols]
        modes = angle_modes[theta_deg]
        
        # Get eigenvalues — handle both dict-of-lists and list-of-dicts
        if isinstance(modes, list):
            evals = np.array([m.get('eigenvalue', m.get('omega', 0)) for m in modes])
        else:
            evals = np.array(modes.get('eigenvalues', []))
        
        spacings = compute_level_spacings(evals)
        if len(spacings) < 5:
            ax.text(0.5, 0.5, 'Too few modes', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        
        # Histogram
        ax.hist(spacings, bins=np.linspace(0, 4, 25), density=True,
                alpha=0.5, color=SKY_BLUE, edgecolor=STEEL_BLUE, linewidth=0.5,
                label='Data')
        
        # Reference distributions
        ax.plot(s_ref, poisson_pdf(s_ref), '-', color=STARK_ORANGE, linewidth=2,
                label='Poisson')
        ax.plot(s_ref, goe_pdf(s_ref), '--', color='#E05050', linewidth=2,
                label='GOE')
        
        # Compute Brody parameter as a quality metric
        # <s²> = 2 for Poisson, = 4/π ≈ 1.27 for GOE
        s2_mean = np.mean(spacings**2) if len(spacings) > 0 else 0
        
        eta_val = 2 * np.sin(np.radians(theta_deg) / 2)
        ax.set_title(f'θ = {theta_deg:.1f}° (η = {eta_val:.4f})\n⟨s²⟩ = {s2_mean:.2f}',
                     fontsize=10)
        ax.set_xlabel('s (normalized spacing)')
        ax.set_ylabel('P(s)')
        ax.set_xlim(0, 4)
        ax.legend(fontsize=8)
        
        summary_stats[theta_deg] = {
            'n_spacings': len(spacings),
            's2_mean': float(s2_mean),
            'character': 'Poisson-like' if s2_mean > 1.6 else 'GOE-like' if s2_mean < 1.4 else 'intermediate'
        }
    
    # Hide unused axes
    for idx in range(len(selected), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    
    fig.suptitle(f'Level-Spacing Statistics — {candidate_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'T11_miniband_validation', 'T11_level_statistics')
    plt.close()
    
    return summary_stats


# =====================================================================
#  2. DOS Evolution
# =====================================================================

def plot_dos_evolution(results, angle_modes, sweep_dir, candidate_name):
    """Plot DOS histograms at multiple θ values showing densification."""
    apply_thesis_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel (a): Stacked DOS histograms
    ax = axes[0]
    angles_sorted = sorted(angle_modes.keys())
    n_angles = len(angles_sorted)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_angles))
    
    for i, theta_deg in enumerate(angles_sorted):
        modes = angle_modes[theta_deg]
        if isinstance(modes, list):
            omegas = np.array([m.get('omega', 0) for m in modes])
        else:
            omegas = np.array([])
        
        if len(omegas) == 0:
            continue
        
        # KDE-style smoothed DOS
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(omegas, bw_method=0.05)
            omega_range = np.linspace(omegas.min() - 0.005, omegas.max() + 0.005, 300)
            dos = kde(omega_range)
            ax.plot(omega_range, dos + i * 0.3, color=colors[i], linewidth=1.5,
                    label=f'θ = {theta_deg:.1f}°')
            ax.fill_between(omega_range, i * 0.3, dos + i * 0.3,
                           alpha=0.15, color=colors[i])
        except Exception:
            pass
    
    ax.set_xlabel('Frequency ω (a/λ)')
    ax.set_ylabel('DOS (offset for clarity)')
    ax.set_title('(a) Spectral density evolution')
    ax.legend(fontsize=7, loc='upper right')
    
    # Panel (b): Mode count vs θ in a fixed frequency window
    ax = axes[1]
    thetas = []
    mode_counts = []
    bandwidths = []
    
    for r in sorted(results, key=lambda x: x['theta_deg']):
        thetas.append(r['theta_deg'])
        evals = np.array(r.get('eigenvalues', []))
        mode_counts.append(len(evals))
        bandwidths.append(float(r.get('bandwidth_50', evals[-1] - evals[0]) if len(evals) > 1 else 0))
    
    ax2 = ax.twinx()
    ax.semilogy(thetas, bandwidths, 's-', color=SKY_BLUE, markersize=7,
                linewidth=2, label='Bandwidth (50 modes)')
    ax2.plot(thetas, mode_counts, 'o-', color=STARK_ORANGE, markersize=7,
             linewidth=2, label='Mode count')
    
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('Bandwidth', color=SKY_BLUE)
    ax2.set_ylabel('Mode count', color=STARK_ORANGE)
    ax.set_title('(b) Bandwidth & mode density')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    
    fig.suptitle(f'DOS Evolution — {candidate_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'T11_miniband_validation', 'T11_dos_evolution')
    plt.close()
    
    return {'thetas': thetas, 'bandwidths': bandwidths, 'mode_counts': mode_counts}


# =====================================================================
#  3. Scaling Laws
# =====================================================================

def plot_scaling_laws(results, angle_modes, sweep_dir, candidate_name):
    """Plot BW, gap, IPR, spread vs η with power-law fits."""
    apply_thesis_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract data
    valid = [r for r in sorted(results, key=lambda x: x['eta']) if 'error' not in r]
    etas = np.array([r['eta'] for r in valid])
    thetas = np.array([r['theta_deg'] for r in valid])
    
    # (a) Bandwidth vs η
    ax = axes[0, 0]
    bws = np.array([r.get('bandwidth_50', 0) for r in valid])
    mask = bws > 0
    if mask.sum() >= 2:
        ax.loglog(etas[mask], bws[mask], 'o-', color=SKY_BLUE, markersize=7, linewidth=2)
        # Power-law fit
        log_fit = np.polyfit(np.log(etas[mask]), np.log(bws[mask]), 1)
        eta_fit = np.linspace(etas[mask].min(), etas[mask].max(), 100)
        ax.loglog(eta_fit, np.exp(log_fit[1]) * eta_fit**log_fit[0],
                 '--', color='gray', alpha=0.6,
                 label=f'η^{log_fit[0]:.2f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('η = 2 sin(θ/2)')
    ax.set_ylabel('Bandwidth (50 modes)')
    ax.set_title('(a) Bandwidth scaling')
    ax.grid(True, alpha=0.2)
    
    # (b) Gap vs η  
    ax = axes[0, 1]
    gaps = np.array([r.get('gap_01', 0) for r in valid])
    mask = gaps > 0
    if mask.sum() >= 2:
        ax.loglog(etas[mask], gaps[mask], 's-', color=STARK_ORANGE, markersize=7, linewidth=2)
        log_fit = np.polyfit(np.log(etas[mask]), np.log(gaps[mask]), 1)
        eta_fit = np.linspace(etas[mask].min(), etas[mask].max(), 100)
        ax.loglog(eta_fit, np.exp(log_fit[1]) * eta_fit**log_fit[0],
                 '--', color='gray', alpha=0.6,
                 label=f'η^{log_fit[0]:.2f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('η = 2 sin(θ/2)')
    ax.set_ylabel('Gap Δλ₀₁')
    ax.set_title('(b) Ground-state gap scaling')
    ax.grid(True, alpha=0.2)
    
    # (c) Mean IPR vs θ
    ax = axes[1, 0]
    mean_iprs = []
    for theta_deg in thetas:
        modes = angle_modes.get(theta_deg, [])
        if isinstance(modes, list) and len(modes) > 0:
            iprs = [m.get('ipr', 0) for m in modes]
            mean_iprs.append(np.mean(iprs) if iprs else 0)
        else:
            mean_iprs.append(0)
    mean_iprs = np.array(mean_iprs)
    mask = mean_iprs > 0
    if mask.sum() >= 2:
        ax.semilogy(thetas[mask], mean_iprs[mask], 'D-', color=STEEL_BLUE,
                    markersize=7, linewidth=2, label='Mean IPR')
        # Reference: uniform distribution
        # IPR_uniform ~ 1/N where N = Ns1*Ns2*N_bands
        ax.axhline(1/81920, color='gray', linestyle=':', alpha=0.5, label='Uniform (1/N)')
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('Mean IPR')
    ax.set_title('(c) Localization (IPR) vs θ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    # (d) Mean spread vs θ
    ax = axes[1, 1]
    mean_spreads = []
    for theta_deg in thetas:
        modes = angle_modes.get(theta_deg, [])
        if isinstance(modes, list) and len(modes) > 0:
            spreads = [m.get('spread', 0) for m in modes]
            mean_spreads.append(np.mean(spreads) if spreads else 0)
        else:
            mean_spreads.append(0)
    mean_spreads = np.array(mean_spreads)
    mask = mean_spreads > 0
    if mask.sum() >= 1:
        ax.plot(thetas[mask], mean_spreads[mask], '^-', color='#7B68EE',
                markersize=7, linewidth=2, label='Mean spread')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5,
                   label='Half moiré cell')
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('Mean spread (L_m units)')
    ax.set_title('(d) Spatial extent vs θ')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(f'Scaling Laws — {candidate_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'T11_miniband_validation', 'T11_scaling_laws')
    plt.close()
    
    # Fit results
    fit_results = {}
    for name, y_arr, x_arr in [('bandwidth', bws, etas), ('gap', gaps, etas)]:
        mask = y_arr > 0
        if mask.sum() >= 2:
            p = np.polyfit(np.log(x_arr[mask]), np.log(y_arr[mask]), 1)
            fit_results[name] = {'exponent': float(p[0]), 'prefactor': float(np.exp(p[1]))}
    
    return fit_results


# =====================================================================
#  4. Subspace Validity
# =====================================================================

def plot_subspace_validity(results, sweep_dir, candidate_name):
    """Plot BW/ω₀ and Born-Huang magnitude vs θ to show EA validity."""
    apply_thesis_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid = sorted([r for r in results if 'error' not in r], key=lambda x: x['eta'])
    thetas = [r['theta_deg'] for r in valid]
    etas = [r['eta'] for r in valid]
    
    # (a) BW/ω₀ — must be ≪ 1 for EA to be valid
    ax = axes[0]
    bw_over_omega = []
    for r in valid:
        omega_ref = r.get('omega_ref', 1.0)
        bw = r.get('bandwidth_50', 0)
        bw_over_omega.append(bw / omega_ref if omega_ref > 0 else 0)
    
    ax.semilogy(thetas, bw_over_omega, 'o-', color=SKY_BLUE, markersize=8, linewidth=2)
    ax.axhline(0.1, color='green', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(0.01, color='orange', linestyle='--', alpha=0.5, label='1% threshold')
    ax.fill_between([min(thetas)-0.5, max(thetas)+0.5], 0, 0.01,
                    color='green', alpha=0.08, label='EA valid (BW/ω₀ < 1%)')
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('BW / ω₀')
    ax.set_title('(a) Envelope approximation validity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    # (b) V/E_kin ratio — determines regime
    ax = axes[1]
    # V/E_kin ~ (V_range) / (η² * M_inv_trace)
    # We approximate this from the eigenvalue data
    v_over_ekin = []
    for r in valid:
        evals = np.array(r.get('eigenvalues', [0, 0]))
        eta = r['eta']
        if len(evals) >= 2 and eta > 0:
            # Rough estimate: V_range ~ spread of diagonal potential,
            # E_kin ~ η² * bandwidth
            v_range = evals[-1] - evals[0]
            e_kin_est = eta**2 * 50  # rough scale
            v_over_ekin.append(v_range / (eta**2) if eta > 0 else 0)
        else:
            v_over_ekin.append(0)
    
    ax.semilogy(thetas, v_over_ekin, 's-', color=STARK_ORANGE, markersize=8, linewidth=2)
    ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='Deep well (V/E_kin > 10)')
    ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='Intermediate (V/E_kin ~ 1)')
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('V / E_kin (approx)')
    ax.set_title('(b) Confinement regime')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(f'Subspace Validity — {candidate_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'T11_miniband_validation', 'T11_subspace_validity')
    plt.close()
    
    return {'bw_over_omega': bw_over_omega, 'thetas': thetas}


# =====================================================================
#  5. Single-Band vs Multi-Band
# =====================================================================

def plot_single_vs_multi(results, angle_modes, sweep_dir, candidate_name):
    """Plot single-band vs multi-band comparison.
    
    Panel (a): Per-band eigenvalue decomposition showing independent minibands.
    Panel (b): Band mixing fraction vs θ.
    
    Note: The sweep's delta_lambda_N can be misleading when the target band
    has no modes in the lowest eigenvalues. Instead, we show that eigenvalues
    decompose cleanly into per-band subsets with zero mixing.
    """
    apply_thesis_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid = sorted([r for r in results if 'error' not in r], key=lambda x: x['eta'])
    
    # (a) Per-band eigenvalue decomposition
    ax = axes[0]
    band_colors = [SKY_BLUE, STARK_ORANGE, '#E05050', '#009E73', STEEL_BLUE]
    
    # For each angle, show eigenvalues colored by dominant band
    all_mixing_zero = all(r.get('max_mixing', 0) < 1e-6 for r in valid)
    
    if all_mixing_zero and angle_modes:
        # Show per-band mode count vs θ — demonstrates independent bands
        angles_sorted = sorted(angle_modes.keys())
        band_counts = {}  # band_idx -> list of counts
        for theta in angles_sorted:
            modes = angle_modes[theta]
            if isinstance(modes, list):
                for m in modes:
                    b = m.get('dominant_band', 0)
                    band_counts.setdefault(b, []).append(theta)
        
        for b_idx in sorted(band_counts.keys()):
            from collections import Counter
            theta_counts = Counter(band_counts[b_idx])
            thetas_b = sorted(theta_counts.keys())
            counts_b = [theta_counts[t] for t in thetas_b]
            c = band_colors[b_idx % len(band_colors)]
            ax.plot(thetas_b, counts_b, 'o-', color=c, markersize=7,
                    linewidth=2, label=f'Band {b_idx}')
        
        ax.set_xlabel('Twist angle θ (°)')
        ax.set_ylabel('Mode count (per band, lowest 50)')
        ax.set_title('(a) Per-band decomposition\n(zero inter-band mixing)')
        ax.legend(fontsize=9)
    else:
        # Show delta_lambda_N as originally intended
        etas = np.array([r['eta'] for r in valid])
        delta_lambdas = np.array([abs(r.get('delta_lambda_N', 0)) for r in valid])
        mask = delta_lambdas > 0
        if mask.sum() >= 2:
            ax.loglog(etas[mask], delta_lambdas[mask], 'o-', color=SKY_BLUE,
                      markersize=8, linewidth=2, label='|λ₀(N=1) − λ₀(N=5)|')
            log_fit = np.polyfit(np.log(etas[mask]), np.log(delta_lambdas[mask]), 1)
            eta_fit = np.linspace(etas[mask].min(), etas[mask].max(), 100)
            ax.loglog(eta_fit, np.exp(log_fit[1]) * eta_fit**log_fit[0],
                     '--', color='gray', alpha=0.6, label=f'η^{log_fit[0]:.2f}')
            ax.legend(fontsize=9)
        ax.set_xlabel('η = 2 sin(θ/2)')
        ax.set_ylabel('|Δλ₀|')
        ax.set_title('(a) Single-band vs multi-band error')
    ax.grid(True, alpha=0.2)
    
    # (b) Band mixing vs θ
    ax = axes[1]
    thetas = np.array([r['theta_deg'] for r in valid])
    mixings = np.array([r.get('max_mixing', 0) for r in valid])
    
    ax.semilogy(thetas, np.maximum(mixings, 1e-16), 's-', color=STARK_ORANGE,
                markersize=8, linewidth=2, label='max(1 − max_weight)')
    ax.axhline(0.01, color='green', linestyle='--', alpha=0.4, label='1% mixing')
    ax.axhline(0.10, color='red', linestyle='--', alpha=0.4, label='10% mixing')
    
    ax.set_xlabel('Twist angle θ (°)')
    ax.set_ylabel('Band mixing fraction')
    ax.set_title('(b) Interband coupling strength')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle(f'Single-Band vs Multi-Band — {candidate_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'T11_miniband_validation', 'T11_single_vs_multi')
    plt.close()
    
    # Summary
    delta_lambdas_list = [abs(r.get('delta_lambda_N', 0)) for r in valid]
    return {
        'all_single_band': all_mixing_zero,
        'mean_delta_lambda': float(np.mean(delta_lambdas_list)) if delta_lambdas_list else 0,
        'max_mixing': float(mixings.max()) if len(mixings) > 0 else 0,
        'coupling_onset_theta': float(thetas[mixings > 0.01][0]) if (mixings > 0.01).any() else None,
    }


# =====================================================================
#  Main driver
# =====================================================================

def run_validation(candidate_name):
    """Run the full T11 validation suite for one candidate."""
    print(f"\n{'='*72}")
    print(f"  T11 DENSE MINIBAND VALIDATION: {candidate_name}")
    print(f"{'='*72}")
    
    # Load data
    sweep_dir = find_eta_sweep_dir(candidate_name)
    print(f"  Sweep dir: {sweep_dir}")
    
    results = load_sweep_results(sweep_dir)
    print(f"  Loaded {len(results)} angle points")
    
    angle_modes = load_per_angle_modes(sweep_dir, results)
    print(f"  Per-angle mode data: {len(angle_modes)} angles")
    
    # Also try to include the main (non-sweep) Phase 3 run
    main_modes = load_single_run_modes(candidate_name)
    if main_modes is not None:
        # Check if its theta is already in the sweep
        cand_dir = find_candidate_dir(candidate_name)
        try:
            import h5py
            with h5py.File(cand_dir / 'phase1_multiband_data.h5', 'r') as hf:
                main_theta = float(hf.attrs.get('theta_deg', 0))
            if main_theta > 0 and main_theta not in angle_modes:
                angle_modes[main_theta] = main_modes
                print(f"  Added main run at θ={main_theta:.2f}°")
        except Exception:
            pass
    
    ensure_output_dir('T11_miniband_validation')
    summary = {'candidate': candidate_name, 'n_angles': len(results)}
    
    # Run all five diagnostics
    print(f"\n  [1/5] Level-spacing statistics...")
    level_stats = plot_level_statistics(results, angle_modes, sweep_dir, candidate_name)
    if level_stats:
        summary['level_statistics'] = level_stats
    
    print(f"  [2/5] DOS evolution...")
    dos_stats = plot_dos_evolution(results, angle_modes, sweep_dir, candidate_name)
    if dos_stats:
        summary['dos'] = dos_stats
    
    print(f"  [3/5] Scaling laws...")
    scaling = plot_scaling_laws(results, angle_modes, sweep_dir, candidate_name)
    if scaling:
        summary['scaling_laws'] = scaling
    
    print(f"  [4/5] Subspace validity...")
    validity = plot_subspace_validity(results, sweep_dir, candidate_name)
    if validity:
        summary['subspace_validity'] = validity
    
    print(f"  [5/5] Single-band vs multi-band...")
    sb_mb = plot_single_vs_multi(results, angle_modes, sweep_dir, candidate_name)
    if sb_mb:
        summary['single_vs_multi'] = sb_mb
    
    # Save summary
    out_dir = ensure_output_dir('T11_miniband_validation')
    summary_path = out_dir / f"T11_summary_{candidate_name}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="T11: Dense Miniband Validation Suite")
    parser.add_argument("candidate", nargs='?', default=None,
                        help="Candidate name (e.g. square_M_b3)")
    parser.add_argument("--all", action="store_true",
                        help="Run for all candidates with eta-sweep data")
    args = parser.parse_args()
    
    if args.all:
        names = get_candidate_names()
    elif args.candidate:
        names = [args.candidate]
    else:
        parser.error("Provide a candidate name or --all")
    
    for name in names:
        try:
            run_validation(name)
        except FileNotFoundError as e:
            print(f"  SKIPPED {name}: {e}")
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
