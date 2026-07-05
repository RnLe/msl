#!/usr/bin/env python3
"""
Compare diagonal-A (old) vs full-A (new) Phase 3 results.
Generates thesis-quality figures showing the impact of inter-band Berry coupling.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

def load(path):
    with open(path) as f:
        return json.load(f)

def plot_candidate(old, new, label, n_bands, outdir, prefix):
    """Generate all comparison plots for one candidate."""
    os.makedirs(outdir, exist_ok=True)
    
    N = len(new)
    old_eigs = np.array([m['eigenvalue'] for m in old])
    new_eigs = np.array([m['eigenvalue'] for m in new])
    old_fracs = np.array([m['dominant_band_weight'] for m in old])
    new_fracs = np.array([m['dominant_band_weight'] for m in new])
    old_iprs = np.array([m['ipr'] for m in old])
    new_iprs = np.array([m['ipr'] for m in new])
    old_spreads = np.array([m['spread'] for m in old])
    new_spreads = np.array([m['spread'] for m in new])
    old_bands = np.array([m['dominant_band'] for m in old])
    new_bands = np.array([m['dominant_band'] for m in new])
    
    # Band weight arrays
    new_ppb = np.array([m['prob_per_band'] for m in new])  # (N, n_bands)
    old_ppb = np.array([m['prob_per_band'] for m in old])
    
    # Band participation ratio
    def neff(ppb): return 1.0 / np.sum(ppb**2, axis=1)
    old_neff = neff(old_ppb)
    new_neff = neff(new_ppb)
    
    # Band entropy
    def entropy(ppb):
        S = np.zeros(ppb.shape[0])
        for i in range(ppb.shape[0]):
            p = ppb[i]
            p = p[p > 1e-10]
            S[i] = -np.sum(p * np.log(p))
        return S
    old_S = entropy(old_ppb)
    new_S = entropy(new_ppb)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_bands))
    
    # =========================================================================
    # FIGURE 1: 4-panel overview
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{label}: Diagonal-A vs Full-A Comparison', fontsize=15, fontweight='bold')
    
    # Panel A: Eigenvalue spectrum
    ax = axes[0, 0]
    ax.plot(range(N), old_eigs, 'o', ms=3, color='gray', alpha=0.5, label='diag-A')
    ax.plot(range(N), new_eigs, 'o', ms=3, color='C0', label='full-A')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    ax.set_title('A. Eigenvalue spectrum')
    
    # Panel B: Dominant band fraction
    ax = axes[0, 1]
    ax.plot(range(N), old_fracs, 'o', ms=3, color='gray', alpha=0.5, label='diag-A')
    ax.plot(range(N), new_fracs, 'o', ms=3, color='C1', label='full-A')
    ax.axhline(1.0/n_bands, color='red', ls='--', lw=0.8, label=f'uniform = 1/{n_bands}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Dominant band weight')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title('B. Band hybridization')
    
    # Panel C: Band participation ratio
    ax = axes[1, 0]
    ax.plot(range(N), old_neff, 'o', ms=3, color='gray', alpha=0.5, label='diag-A')
    ax.plot(range(N), new_neff, 'o', ms=3, color='C2', label='full-A')
    ax.axhline(n_bands, color='red', ls='--', lw=0.8, label=f'N_bands = {n_bands}')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('$N_{\\mathrm{eff}} = 1/\\Sigma p_i^2$')
    ax.set_ylim(0, n_bands + 0.5)
    ax.legend()
    ax.set_title('C. Band participation ratio')
    
    # Panel D: IPR (localization)
    ax = axes[1, 1]
    ax.semilogy(range(N), old_iprs, 'o', ms=3, color='gray', alpha=0.5, label='diag-A')
    ax.semilogy(range(N), new_iprs, 'o', ms=3, color='C3', label='full-A')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('IPR')
    ax.legend()
    ax.set_title('D. Spatial localization (IPR)')
    
    plt.tight_layout()
    fig.savefig(f'{outdir}/{prefix}_overview.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {prefix}_overview.png')
    
    # =========================================================================
    # FIGURE 2: Band weight stacked bars (full-A only)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(N)
    for b in range(n_bands):
        ax.bar(range(N), new_ppb[:, b], bottom=bottom, color=colors[b],
               width=1.0, label=f'Band {b}', edgecolor='none')
        bottom += new_ppb[:, b]
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Band weight')
    ax.set_title(f'{label}: Band decomposition (full off-diagonal A)')
    ax.legend(loc='upper right', ncol=n_bands)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(f'{outdir}/{prefix}_band_stacked.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {prefix}_band_stacked.png')
    
    # =========================================================================
    # FIGURE 3: Level spacing statistics
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (eigs, title, color) in enumerate([
        (old_eigs, 'diag-A', 'gray'),
        (new_eigs, 'full-A', 'C0'),
    ]):
        ax = axes[idx]
        dE = np.diff(np.sort(eigs))
        s = dE / dE.mean()
        
        # Histogram
        bins = np.linspace(0, 4, 40)
        ax.hist(s, bins=bins, density=True, alpha=0.7, color=color, edgecolor='black', lw=0.5)
        
        # Reference distributions
        ss = np.linspace(0, 4, 200)
        ax.plot(ss, np.exp(-ss), 'k--', lw=1.5, label='Poisson')
        ax.plot(ss, (np.pi/2) * ss * np.exp(-np.pi * ss**2 / 4), 'r-', lw=1.5, label='GOE')
        
        ax.set_xlabel('$s = \\Delta E / \\langle\\Delta E\\rangle$')
        ax.set_ylabel('$P(s)$')
        ax.set_title(f'{title}')
        ax.legend()
        ax.set_xlim(0, 4)
    
    fig.suptitle(f'{label}: Level spacing distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{outdir}/{prefix}_level_spacing.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {prefix}_level_spacing.png')
    
    # =========================================================================
    # FIGURE 4: Band entropy histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    max_S = np.log(n_bands)
    bins = np.linspace(0, max_S * 1.05, 30)
    ax.hist(old_S, bins=bins, alpha=0.5, color='gray', label='diag-A', edgecolor='black', lw=0.5)
    ax.hist(new_S, bins=bins, alpha=0.7, color='C4', label='full-A', edgecolor='black', lw=0.5)
    ax.axvline(max_S, color='red', ls='--', lw=1.5, label=f'$\\ln({n_bands})$ = {max_S:.3f} (uniform)')
    ax.set_xlabel('Band entropy $S = -\\Sigma p_i \\ln p_i$')
    ax.set_ylabel('Count')
    ax.set_title(f'{label}: Band mixing entropy')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{outdir}/{prefix}_entropy.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {prefix}_entropy.png')


if __name__ == '__main__':
    base = Path('/home/renlephy/msl/research/moire_envelope')
    outdir = base / 'thesis_results' / 'T_fullA_comparison'
    
    # C3 square
    old_sq = load(base / 'runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/phase3_mode_stats_diagA.json')
    new_sq = load(base / 'runsV3/thesis_square_M_b3_20260209_173724/candidate_0000/phase3_mode_stats.json')
    print('Plotting C3 SQUARE...')
    plot_candidate(old_sq, new_sq, 'C3: Square M-point b3 (TM)', 5, str(outdir), 'C3_square')
    
    # C1 hex
    old_hex = load(base / 'runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/phase3_mode_stats_diagA.json')
    new_hex = load(base / 'runsV3/thesis_hex_M_b1_20260209_173724/candidate_0000/phase3_mode_stats.json')
    print('Plotting C1 HEX...')
    plot_candidate(old_hex, new_hex, 'C1: Hex M-point b1 (TE)', 4, str(outdir), 'C1_hex')
    
    print(f'\nAll plots saved to: {outdir}')
