"""
T06: Scaling Laws — Thesis Figure

Analyzes how key quantities scale with the twist angle η = θ:
  - E_gap (band gap) vs η
  - Bandwidth (E_1 - E_0) vs η  
  - IPR (inverse participation ratio) vs η
  - Power-law fits: BW ~ η^α, gap ~ η^β

Expected universal scaling from EA theory:
  - V/E_kin ~ 1/η² (potential strengthens as θ decreases)
  - For V/E_kin >> 1: BW ~ e^(-c/η) (exponentially flat)
  - For V/E_kin ~ 1: BW ~ η² (parabolic, perturbative)

Usage:
    python thesis_results/T06_scaling_laws/compute.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    get_candidate_names, find_candidate_dir, load_eta_sweep_data,
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T06_scaling_laws"


def power_law(x, a, alpha):
    return a * x**alpha


def exponential_decay(x, a, c):
    return a * np.exp(-c / x)


def compute_ipr(modes, Ns, Nb, mode_idx):
    """Inverse Participation Ratio: IPR = Σ|F|⁴ / (Σ|F|²)²."""
    if modes.ndim > 1:
        vec = modes[:, mode_idx]
    else:
        vec = modes
    F = vec.reshape(Ns, Ns, Nb)
    rho = np.sum(np.abs(F)**2, axis=-1)  # (Ns, Ns)
    ipr = np.sum(rho**2) / (np.sum(rho)**2 + 1e-30)
    return ipr


def plot_scaling_laws(sweeps):
    """4-panel scaling law figure."""
    apply_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for name, data in sweeps.items():
        thetas = np.array([d['theta_deg'] for d in data])
        etas = np.radians(thetas)

        # Extract quantities
        bws = []
        gaps = []
        e0s = []
        valid_thetas = []

        for d in data:
            evals = d.get('eigenvalues')
            if evals is not None and len(evals) >= 3:
                bws.append(evals[1] - evals[0])
                gaps.append(evals[2] - evals[1])
                e0s.append(evals[0])
                valid_thetas.append(d['theta_deg'])

        if not bws:
            continue

        bws = np.array(bws)
        gaps = np.array(gaps)
        e0s = np.array(e0s)
        vt = np.array(valid_thetas)

        # (a) Ground state E_0 vs θ
        ax = axes[0, 0]
        ax.plot(vt, e0s, '-o', color=CANDIDATE_COLORS[name],
                marker=CANDIDATE_MARKERS[name],
                label=CANDIDATE_LABELS[name], markersize=5)

        # (b) Bandwidth vs θ (log-log)
        ax = axes[0, 1]
        ax.semilogy(vt, bws, '-o', color=CANDIDATE_COLORS[name],
                     marker=CANDIDATE_MARKERS[name],
                     label=CANDIDATE_LABELS[name], markersize=5)

        # Try power-law fit
        try:
            mask = bws > 0
            popt, _ = curve_fit(power_law, vt[mask], bws[mask], p0=[1, 2],
                                maxfev=5000)
            t_fit = np.linspace(vt.min(), vt.max(), 50)
            ax.plot(t_fit, power_law(t_fit, *popt), '--',
                    color=CANDIDATE_COLORS[name], alpha=0.5,
                    label=f'  fit: α={popt[1]:.1f}')
        except (RuntimeError, ValueError):
            pass

        # (c) Gap vs θ
        ax = axes[1, 0]
        ax.semilogy(vt, gaps, '-o', color=CANDIDATE_COLORS[name],
                     marker=CANDIDATE_MARKERS[name],
                     label=CANDIDATE_LABELS[name], markersize=5)

        # (d) Flat-band ratio BW/gap
        ax = axes[1, 1]
        mask = gaps > 0
        if mask.sum() > 0:
            ax.semilogy(vt[mask], bws[mask] / gaps[mask], '-o',
                        color=CANDIDATE_COLORS[name],
                        marker=CANDIDATE_MARKERS[name],
                        label=CANDIDATE_LABELS[name], markersize=5)

    # Labels
    axes[0, 0].set_xlabel(r'$\theta$ [deg]')
    axes[0, 0].set_ylabel(r'$E_0$')
    axes[0, 0].set_title(r'(a) Ground state energy $E_0(\theta)$')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel(r'$\theta$ [deg]')
    axes[0, 1].set_ylabel(r'$\Delta E = E_1 - E_0$')
    axes[0, 1].set_title(r'(b) Bandwidth scaling')
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_xlabel(r'$\theta$ [deg]')
    axes[1, 0].set_ylabel(r'Gap = $E_2 - E_1$')
    axes[1, 0].set_title('(c) Inter-miniband gap')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_xlabel(r'$\theta$ [deg]')
    axes[1, 1].set_ylabel(r'$\Delta E / \mathrm{gap}$')
    axes[1, 1].set_title('(d) Flat-band ratio')
    axes[1, 1].axhline(1, color='red', ls='--', alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Scaling Laws: Miniband Structure vs Twist Angle',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"T06: Scaling Laws → {out_dir}")

    sweeps = {}
    for name in get_candidate_names():
        try:
            cand_dir = find_candidate_dir(name)
            data = load_eta_sweep_data(cand_dir)
            if data:
                sweeps[name] = data
                print(f"  {name}: {len(data)} angles")
        except FileNotFoundError:
            pass

    if not sweeps:
        print("  No η-sweep data. Run pipeline first.")
        return

    fig = plot_scaling_laws(sweeps)
    save_figure(fig, TASK, "T06_scaling_laws")

    print("  T06 complete.")


if __name__ == "__main__":
    main()
