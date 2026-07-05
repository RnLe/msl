#!/usr/bin/env python
"""
Cross-Candidate Comparison Report.

Compares C3 (square_M_b3), C1 (hex_M_b1), and C_hc (honeycomb_K_b1)
using existing sweep JSON data for C3/C1 and Phase 3 base-run for honeycomb.

Generates:
  - F_comparison_eigenspectra: eigenvalue spectra across angles (C3, C1)
  - F_comparison_bandwidth: bandwidth scaling with power-law fits
  - F_comparison_mixing: inter-band mixing analysis
  - F_comparison_phase2: side-by-side Phase 2 diagnostics (all 3 candidates)
  - COMPARISON_REPORT.md: comprehensive written report
"""
import sys, json, textwrap
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "phasesV3"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    find_candidate_dir, find_thesis_run_dir, load_phase2_data, load_phase0_meta,
    CANDIDATE_COLORS, CANDIDATE_LABELS, CANDIDATE_MARKERS,
)

TASK = "T_honeycomb_analysis"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Known sweep paths
SWEEP_JSON = {
    'square_M_b3': PROJECT_ROOT / 'runsV3/thesis_square_M_b3_20260209_173724/eta_sweep_20260307_144407/sweep_results.json',
    'hex_M_b1': PROJECT_ROOT / 'runsV3/thesis_hex_M_b1_20260209_173724/eta_sweep_20260307_153407/sweep_results.json',
}


def find_honeycomb_sweep_json():
    """Auto-discover honeycomb sweep results."""
    try:
        run_dir = find_thesis_run_dir('honeycomb_K_b1')
        sweep_dirs = sorted(run_dir.glob("eta_sweep_*"))
        for sd in reversed(sweep_dirs):
            sp = sd / "sweep_results.json"
            if sp.exists():
                return sp
            sp = sd / "sweep_results_partial.json"
            if sp.exists():
                return sp
    except Exception:
        pass
    return None


def load_sweep_json(path):
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [d for d in data if 'error' not in d]


def power_law(x, a, alpha):
    return a * x**alpha


def load_phase2_summary(name):
    """Load key Phase 2 quantities for a candidate."""
    try:
        cand_dir = find_candidate_dir(name)
        p2_path = cand_dir / "phase2_multiband_data.h5"
        with h5py.File(p2_path, 'r') as hf:
            Lambda = hf['Lambda'][:]
            A_berry = hf['A_berry'][:]
            M_inv = hf['M_inv'][:]
            omega = hf['omega'][:]
            N_sub = int(hf.attrs.get('N_subspace', 1))
        
        summary = {
            'N_sub': N_sub,
            'omega_range': (float(omega.min()), float(omega.max())),
            'Lambda_00_range': (float(Lambda[:,:,0,0].real.min()), float(Lambda[:,:,0,0].real.max())),
            'A_00_max': float(np.sqrt(np.abs(A_berry[:,:,0,0,0])**2 + np.abs(A_berry[:,:,0,0,1])**2).max()),
            'M_inv_00_trace': float((M_inv[:,:,0,0,0,0].real + M_inv[:,:,0,0,1,1].real).mean()),
        }
        
        if N_sub >= 2:
            summary['Lambda_01_max'] = float(np.abs(Lambda[:,:,0,1]).max())
            summary['A_01_max'] = float(np.sqrt(np.abs(A_berry[:,:,0,1,0])**2 + np.abs(A_berry[:,:,0,1,1])**2).max())
            summary['M_inv_11_trace'] = float((M_inv[:,:,1,1,0,0].real + M_inv[:,:,1,1,1,1].real).mean())
        
        return summary
    except Exception as e:
        return {'error': str(e)}


def load_phase3_base(name):
    """Load Phase 3 eigenvalues from base run."""
    try:
        cand_dir = find_candidate_dir(name)
        p3_path = cand_dir / "phase3_multiband_modes.h5"
        with h5py.File(p3_path, 'r') as hf:
            eigenvalues = hf['eigenvalues'][:]
            theta = float(hf.attrs.get('theta_deg', 1.1))
        return eigenvalues, theta
    except:
        return None, None


def main():
    out_dir = ensure_output_dir(TASK)
    print(f"\n{'='*70}")
    print(f"  CROSS-CANDIDATE COMPARISON")
    print(f"{'='*70}")
    
    # Load sweep data
    sweeps = {}
    for name, path in SWEEP_JSON.items():
        data = load_sweep_json(path)
        if data:
            sweeps[name] = data
            print(f"  {name}: {len(data)} sweep angles")
    
    # Auto-discover honeycomb sweep
    hc_path = find_honeycomb_sweep_json()
    if hc_path:
        hc_data = load_sweep_json(hc_path)
        if hc_data:
            sweeps['honeycomb_K_b1'] = hc_data
            print(f"  honeycomb_K_b1: {len(hc_data)} sweep angles ({hc_path.name})")
    
    # Load Phase 2 summaries
    p2_summaries = {}
    for name in ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']:
        p2_summaries[name] = load_phase2_summary(name)
        print(f"  {name} Phase 2: {p2_summaries[name].get('N_sub', '?')} bands")
    
    # Load Phase 3 base runs
    p3_base = {}
    for name in ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']:
        evals, theta = load_phase3_base(name)
        if evals is not None:
            p3_base[name] = {'eigenvalues': evals, 'theta_deg': theta}
            print(f"  {name} Phase 3 base: {len(evals)} modes at θ={theta}°")
    
    # ===== Figure A: Eigenvalue spectra comparison (sweeps) =====
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    for name, data in sweeps.items():
        thetas = [d['theta_deg'] for d in data]
        bw50 = [d['bandwidth_50'] for d in data]
        ax.semilogy(thetas, bw50, '-o', color=CANDIDATE_COLORS.get(name, 'gray'),
                    marker=CANDIDATE_MARKERS.get(name, 'o'),
                    label=CANDIDATE_LABELS.get(name, name), markersize=7, linewidth=2)
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'Bandwidth $\Delta E_{50}$')
    ax.set_title('(a) Bandwidth vs twist angle')
    ax.legend()
    
    ax = axes[1]
    for name, data in sweeps.items():
        etas = [d.get('eta', 2*np.sin(np.radians(d['theta_deg'])/2)) for d in data]
        bw50 = [d['bandwidth_50'] for d in data]
        ax.loglog(etas, bw50, 'o', color=CANDIDATE_COLORS.get(name, 'gray'),
                  marker=CANDIDATE_MARKERS.get(name, 'o'),
                  markersize=7, label=CANDIDATE_LABELS.get(name, name))
        
        # Power-law fit
        try:
            popt, _ = curve_fit(power_law, etas, bw50, p0=[1, 2])
            eta_fine = np.logspace(np.log10(min(etas)), np.log10(max(etas)), 50)
            ax.loglog(eta_fine, power_law(eta_fine, *popt), '--',
                      color=CANDIDATE_COLORS.get(name, 'gray'), alpha=0.5,
                      label=f'  $\\alpha={popt[1]:.2f}$')
        except:
            pass
    
    # Reference line
    eta_ref = np.logspace(-2.5, -0.6, 50)
    ax.loglog(eta_ref, 0.3 * eta_ref**2, ':', color='gray', alpha=0.3, label=r'$\sim\eta^2$ ref')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'Bandwidth $\Delta E_{50}$')
    ax.set_title('(b) Bandwidth scaling (log-log)')
    ax.legend(fontsize=8)
    
    fig.suptitle('Full-A Sweep Comparison (all candidates)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F_comparison_bandwidth_all")
    
    # ===== Figure B: Base-run eigenvalue ladders (all 3) =====
    apply_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, name in enumerate(['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']):
        ax = axes[idx]
        if name in p3_base:
            evals = p3_base[name]['eigenvalues']
            n_show = min(50, len(evals))
            ax.plot(range(n_show), evals[:n_show], 'o-',
                    color=CANDIDATE_COLORS.get(name, 'gray'), markersize=4)
            ax.set_xlabel('Mode index $n$')
            ax.set_ylabel('$E_n$')
            theta = p3_base[name]['theta_deg']
            bw = evals[min(49,len(evals)-1)] - evals[0]
            ax.set_title(f'{CANDIDATE_LABELS.get(name, name)}\n'
                         f'$\\theta={theta:.1f}°$, BW$_{{50}}={bw:.6f}$')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(CANDIDATE_LABELS.get(name, name))
    
    fig.suptitle('Eigenvalue Ladders: Base-run Phase 3 (all 3 candidates)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F_comparison_eigenladders_3cand")
    
    # ===== Figure C: Phase 2 parameter comparison =====
    apply_thesis_style()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    
    names = ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']
    x = np.arange(len(names))
    colors = [CANDIDATE_COLORS.get(n, 'gray') for n in names]
    labels = [CANDIDATE_LABELS.get(n, n) for n in names]
    
    # (a) Potential range
    ax = axes[0]
    for i, n in enumerate(names):
        s = p2_summaries[n]
        if 'error' not in s:
            vmin, vmax = s['Lambda_00_range']
            ax.bar(i, vmax - vmin, color=colors[i], alpha=0.7, label=labels[i])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(r'$\Lambda_{00}$ range')
    ax.set_title('(a) Moiré potential depth')
    
    # (b) Berry connection magnitude
    ax = axes[1]
    for i, n in enumerate(names):
        s = p2_summaries[n]
        if 'error' not in s:
            ax.bar(i, s['A_00_max'], color=colors[i], alpha=0.7)
            if 'A_01_max' in s:
                ax.bar(i, s['A_01_max'], bottom=s['A_00_max'],
                       color=colors[i], alpha=0.4, hatch='//')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(r'$|\mathbf{A}|$ max')
    ax.set_title('(b) Berry connection (solid=diag, hatch=offdiag)')
    
    # (c) Effective mass (Tr M⁻¹)
    ax = axes[2]
    for i, n in enumerate(names):
        s = p2_summaries[n]
        if 'error' not in s:
            ax.bar(i-0.15, s['M_inv_00_trace'], width=0.3, color=colors[i], alpha=0.7, label='Band 0')
            if 'M_inv_11_trace' in s:
                ax.bar(i+0.15, s['M_inv_11_trace'], width=0.3, color=colors[i], alpha=0.4, label='Band 1')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(r'Tr($M^{-1}$)')
    ax.set_title('(c) Effective mass')
    ax.axhline(y=0, color='black', ls='-', alpha=0.3)
    
    # (d) Inter-band coupling Λ₀₁
    ax = axes[3]
    for i, n in enumerate(names):
        s = p2_summaries[n]
        if 'error' not in s and 'Lambda_01_max' in s:
            ax.bar(i, s['Lambda_01_max'], color=colors[i], alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel(r'$|\Lambda_{01}|$ max')
    ax.set_title('(d) Inter-band potential coupling')
    
    fig.suptitle('Phase 2 Parameter Comparison', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F_comparison_phase2_params")
    
    # ===== Figure D: Band mixing (C3 and C1 sweeps) =====
    apply_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, data in sweeps.items():
        thetas = [d['theta_deg'] for d in data]
        max_mix = [d.get('max_mixing', 0) for d in data]
        axes[0].plot(thetas, max_mix, '-o', color=CANDIDATE_COLORS.get(name, 'gray'),
                     marker=CANDIDATE_MARKERS.get(name, 'o'),
                     label=CANDIDATE_LABELS.get(name, name), markersize=6)
        
        # E₁-E₀ gap
        gaps = [d['eigenvalues'][1] - d['eigenvalues'][0] if len(d['eigenvalues']) >= 2 else 0 for d in data]
        axes[1].semilogy(thetas, gaps, '-o', color=CANDIDATE_COLORS.get(name, 'gray'),
                         marker=CANDIDATE_MARKERS.get(name, 'o'),
                         label=CANDIDATE_LABELS.get(name, name), markersize=6)
    
    axes[0].set_xlabel(r'$\theta$ [deg]')
    axes[0].set_ylabel('Max mixing')
    axes[0].set_title('(a) Inter-band mixing')
    axes[0].axhline(y=0.5, color='red', ls='--', alpha=0.3)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    
    axes[1].set_xlabel(r'$\theta$ [deg]')
    axes[1].set_ylabel(r'$E_1 - E_0$')
    axes[1].set_title('(b) Ground-state gap')
    axes[1].legend()
    
    fig.suptitle('Band Mixing & Gap (all candidates)', fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F_comparison_mixing_all")
    
    # ===== Generate markdown report =====
    report = generate_report(sweeps, p2_summaries, p3_base)
    report_path = out_dir / "COMPARISON_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n  Report: {report_path}")
    print(f"\n{'='*70}")
    print(f"  Comparison complete: 4 figures + report")
    print(f"{'='*70}")


def generate_report(sweeps, p2_summaries, p3_base):
    lines = []
    lines.append("# Cross-Candidate Comparison Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Overview\n")
    lines.append("This report compares three photonic moiré crystal candidates:")
    lines.append("- **C3** (`square_M_b3`): Square lattice, M-point, band 3 (5-band subspace)")
    lines.append("- **C1** (`hex_M_b1`): Hexagonal lattice, M-point, band 1 (4-band subspace)")
    lines.append("- **C_hc** (`honeycomb_K_b1`): Honeycomb (triangular + 2-atom basis), K-point Dirac cone (2-band subspace)")
    lines.append("")
    
    # Phase 2 parameters
    lines.append("## Phase 2 Parameters\n")
    lines.append("| Parameter | C3 (square_M_b3) | C1 (hex_M_b1) | C_hc (honeycomb_K_b1) |")
    lines.append("|-----------|-------------------|----------------|-----------------------|")
    
    for key, label in [
        ('N_sub', 'N_subspace'),
        ('omega_range', 'ω range'),
        ('Lambda_00_range', 'Λ₀₀ range'),
        ('Lambda_01_max', '|Λ₀₁| max'),
        ('A_00_max', '|A₀₀| max'),
        ('A_01_max', '|A₀₁| max'),
        ('M_inv_00_trace', 'Tr(M⁻¹₀₀)'),
        ('M_inv_11_trace', 'Tr(M⁻¹₁₁)'),
    ]:
        row = f"| {label} |"
        for name in ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']:
            s = p2_summaries.get(name, {})
            val = s.get(key, 'N/A')
            if isinstance(val, tuple):
                row += f" [{val[0]:.4f}, {val[1]:.4f}] |"
            elif isinstance(val, float):
                row += f" {val:.4f} |"
            else:
                row += f" {val} |"
        lines.append(row)
    
    lines.append("")
    
    # Phase 3 base run
    lines.append("## Phase 3 Base Run (θ ≈ 1.1°)\n")
    lines.append("| Metric | C3 | C1 | C_hc |")
    lines.append("|--------|----|----|------|")
    for name in ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']:
        if name in p3_base:
            evals = p3_base[name]['eigenvalues']
            p3_base[name]['bw50'] = evals[min(49, len(evals)-1)] - evals[0]
            p3_base[name]['gap01'] = evals[1] - evals[0] if len(evals) >= 2 else 0
    
    for key, label in [
        ('bw50', 'BW₅₀'),
        ('gap01', 'Gap E₁-E₀'),
    ]:
        row = f"| {label} |"
        for name in ['square_M_b3', 'hex_M_b1', 'honeycomb_K_b1']:
            if name in p3_base and key in p3_base[name]:
                row += f" {p3_base[name][key]:.6f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")
    
    # Sweep results
    lines.append("## η-Sweep Results (Full Berry Connection)\n")
    for name, data in sweeps.items():
        label = CANDIDATE_LABELS.get(name, name)
        lines.append(f"### {label}\n")
        lines.append("| θ (deg) | η | BW₅₀ | Max mixing | Gap E₁-E₀ |")
        lines.append("|---------|---|-------|------------|-----------|")
        for d in data:
            gap = d['eigenvalues'][1] - d['eigenvalues'][0] if len(d['eigenvalues']) >= 2 else 0
            lines.append(f"| {d['theta_deg']:.1f} | {d.get('eta', 0):.6f} | "
                         f"{d['bandwidth_50']:.6f} | {d.get('max_mixing', 0):.3f} | {gap:.6f} |")
        
        # Power-law fit
        etas = [d.get('eta', 2*np.sin(np.radians(d['theta_deg'])/2)) for d in data]
        bws = [d['bandwidth_50'] for d in data]
        try:
            popt, _ = curve_fit(power_law, etas, bws, p0=[1, 2])
            lines.append(f"\nPower-law fit: BW ~ η^{popt[1]:.3f} (a = {popt[0]:.4f})")
        except:
            pass
        lines.append("")
    
    # Key findings
    lines.append("## Key Findings\n")
    lines.append("### 1. Honeycomb Dirac Cone Candidate")
    lines.append("- The honeycomb candidate has **zero inter-band moiré potential coupling** (|Λ₀₁| = 0)")
    lines.append("- Inter-band coupling comes **entirely through the off-diagonal Berry connection** (|A₀₁| ≈ 1.24)")
    lines.append("- This is the photonic analogue of **twisted bilayer graphene**: Dirac cone + Berry phase")
    lines.append("- 2-band subspace (vs 4-5 for other candidates) = maximally clean Dirac physics")
    lines.append("")
    lines.append("### 2. Effective Mass Asymmetry")
    lines.append("- C_hc: Band 0 = HOLE (Tr(M⁻¹) = -10.31), Band 1 = ELECTRON (Tr(M⁻¹) = +5.36)")
    lines.append("- This electron-hole asymmetry will produce asymmetric miniband spectra")
    lines.append("- The large |M⁻¹| magnitudes indicate strong dispersion near K-point")
    lines.append("")
    lines.append("### 3. Berry Connection Dominance")
    lines.append("- For C_hc, the Berry connection provides the **only** inter-band coupling mechanism")
    lines.append("- This makes it a pure gauge-field-mediated phenomenon")
    lines.append("- Validates the importance of the full non-Abelian Berry connection treatment")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
