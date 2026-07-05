#!/usr/bin/env python3
"""
Eigenvalue Tube Plot
=====================

Visualizes moiré minibands as filled "tubes" (shaded regions between
consecutive eigenvalue tracks) rather than individual lines.

Each tube represents a continuous energy range occupied by a miniband.
Gaps between tubes are the moiré miniband gaps — frequency windows
where no propagating moiré Bloch states exist.

Two panels:
  (a) Diagonal Berry connection only — shows conventional band structure
  (b) Full off-diagonal Berry connection — shows effect of non-Abelian gauge

Uses: miniband_data.json produced by compute_miniband_structure.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load miniband sweep data from JSON."""
    with open(SCRIPT_DIR / "miniband_data.json") as f:
        data = json.load(f)
    
    evals_diag = np.array(data['evals_diag'])   # (N_q, N_modes)
    evals_full = np.array(data['evals_full'])
    q_dist = np.array(data['q_dist'])
    ticks_pos = data['tick_positions']
    ticks_labels = data['tick_labels']
    meta = {k: data[k] for k in ['theta_deg', 'eta', 'Ns', 'Nb', 'N_modes', 'L_moire']}
    return evals_diag, evals_full, q_dist, ticks_pos, ticks_labels, meta


def identify_tube_groups(evals, gap_threshold_factor=2.0):
    """
    Group consecutive minibands into tubes.
    
    A tube is a group of minibands whose eigenvalue ranges overlap or are
    separated by less than gap_threshold × median_bandwidth.
    
    Returns:
        groups: list of (start_idx, end_idx) tuples (inclusive)
    """
    N_q, N_modes = evals.shape
    
    # Compute per-band energy ranges
    band_mins = np.nanmin(evals, axis=0)  # (N_modes,)
    band_maxs = np.nanmax(evals, axis=0)
    bandwidths = band_maxs - band_mins
    
    # Compute gaps between consecutive eigenvalue tracks
    # gap_n = min_q E_{n+1}(q) - max_q E_n(q)
    gaps = np.array([
        np.nanmin(evals[:, n+1]) - np.nanmax(evals[:, n])
        for n in range(N_modes - 1)
    ])
    
    # Use median bandwidth to set threshold for grouping
    median_bw = np.median(bandwidths[bandwidths > 1e-15]) if np.any(bandwidths > 1e-15) else 1e-6
    threshold = gap_threshold_factor * median_bw
    
    groups = []
    current_start = 0
    for n in range(N_modes - 1):
        if gaps[n] > threshold:
            groups.append((current_start, n))
            current_start = n + 1
    groups.append((current_start, N_modes - 1))
    
    return groups


def plot_tubes(ax, q_dist, evals, ticks_pos, ticks_labels, title,
               gap_threshold_factor=2.0, cmap_name='tab10'):
    """
    Plot eigenvalue tubes on a single axes.
    
    Each tube is a filled region between the min and max eigenvalues
    of the bands in that tube group, at each q-point.
    """
    N_q, N_modes = evals.shape
    groups = identify_tube_groups(evals, gap_threshold_factor)
    cmap = matplotlib.colormaps[cmap_name]
    
    for gi, (start, end) in enumerate(groups):
        color = cmap(gi % 10)
        n_bands = end - start + 1
        
        # The tube envelope: at each q, take min and max across bands in this group
        tube_lo = np.nanmin(evals[:, start:end+1], axis=1)
        tube_hi = np.nanmax(evals[:, start:end+1], axis=1)
        
        # Fill the tube
        ax.fill_between(q_dist, tube_lo, tube_hi,
                        color=color, alpha=0.35, linewidth=0,
                        label=f"Tube {gi} ({n_bands} band{'s' if n_bands > 1 else ''})")
        
        # Draw individual band lines inside the tube
        for n in range(start, end + 1):
            ax.plot(q_dist, evals[:, n], '-', color=color, lw=0.4, alpha=0.6)
        
        # Draw tube edges
        ax.plot(q_dist, tube_lo, '-', color=color, lw=1.0, alpha=0.8)
        ax.plot(q_dist, tube_hi, '-', color=color, lw=1.0, alpha=0.8)
    
    # High-symmetry lines
    for tp in ticks_pos:
        ax.axvline(tp, color='gray', lw=0.5, ls=':')
    
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_labels)
    ax.set_ylabel('E (ω − ω_ref)  [c/a]')
    ax.set_title(title)
    
    # Annotate gaps between tubes
    for i in range(len(groups) - 1):
        _, end_i = groups[i]
        start_j, _ = groups[i + 1]
        gap_top = np.nanmin(evals[:, start_j])
        gap_bot = np.nanmax(evals[:, end_i])
        gap_size = gap_top - gap_bot
        if gap_size > 0:
            mid_q = q_dist[len(q_dist) // 2]
            mid_E = (gap_top + gap_bot) / 2
            ax.annotate(f'Δ={gap_size:.1e}',
                        xy=(mid_q, mid_E), fontsize=6,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))
    
    return groups


def plot_individual_bands(ax, q_dist, evals, ticks_pos, ticks_labels, title,
                          cmap_name='tab20'):
    """
    Plot each miniband as its own thin tube (shaded between its min and max
    across q), with individual coloring.
    """
    N_q, N_modes = evals.shape
    cmap = matplotlib.colormaps[cmap_name]
    
    for n in range(N_modes):
        color = cmap(n / max(N_modes - 1, 1))
        band = evals[:, n]
        valid = ~np.isnan(band)
        if np.sum(valid) < 2:
            continue
        
        # Each individual band is its own "tube"
        ax.fill_between(q_dist[valid], band[valid], band[valid],
                        alpha=0)  # invisible fill, just for structure
        
        # Draw the band as a line
        ax.plot(q_dist[valid], band[valid], '-', color=color, lw=1.0, alpha=0.8)
    
    # Also shade between consecutive bands to show the tube structure
    for n in range(N_modes - 1):
        color = cmap(n / max(N_modes - 1, 1))
        b_lo = evals[:, n]
        b_hi = evals[:, n + 1]
        valid = ~np.isnan(b_lo) & ~np.isnan(b_hi)
        if np.sum(valid) < 2:
            continue
        
        # Only shade if there's a gap (b_hi > b_lo at all q)
        gap = b_hi[valid] - b_lo[valid]
        if np.all(gap >= 0):
            # This is a gap — shade it with a light gray to highlight
            ax.fill_between(q_dist[valid], b_lo[valid], b_hi[valid],
                            color='lightgray', alpha=0.3, linewidth=0)
    
    # High-symmetry lines
    for tp in ticks_pos:
        ax.axvline(tp, color='gray', lw=0.5, ls=':')
    
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_labels)
    ax.set_ylabel('E (ω − ω_ref)  [c/a]')
    ax.set_title(title)


def main():
    print("Loading miniband data...")
    evals_diag, evals_full, q_dist, ticks_pos, ticks_labels, meta = load_data()
    
    theta = meta['theta_deg']
    eta = meta['eta']
    Ns = meta['Ns']
    Nb = meta['Nb']
    N_modes = meta['N_modes']
    
    print(f"  θ = {theta:.2f}°, η = {eta:.5f}")
    print(f"  Grid: {Ns}×{Ns}×{Nb}, {N_modes} modes, {len(q_dist)} q-points")
    
    # ── Create figure with 2 rows × 2 cols ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        f"Moiré Miniband Tubes — θ={theta:.1f}°, η={eta:.5f}, "
        f"Ns={Ns}, {Nb}-band subspace, {N_modes} modes",
        fontsize=13, fontweight='bold'
    )
    
    # Row 1: Grouped tubes (merged bands that overlap)
    groups_diag = plot_tubes(
        axes[0, 0], q_dist, evals_diag, ticks_pos, ticks_labels,
        '(a) Diag Berry — grouped tubes',
        gap_threshold_factor=2.0
    )
    axes[0, 0].legend(fontsize=6, loc='upper right', ncol=2)
    
    groups_full = plot_tubes(
        axes[0, 1], q_dist, evals_full, ticks_pos, ticks_labels,
        '(b) Full Berry — grouped tubes',
        gap_threshold_factor=2.0
    )
    axes[0, 1].legend(fontsize=6, loc='upper right', ncol=2)
    
    # Row 2: Individual band lines (each band its own color)
    plot_individual_bands(
        axes[1, 0], q_dist, evals_diag, ticks_pos, ticks_labels,
        '(c) Diag Berry — individual bands'
    )
    
    plot_individual_bands(
        axes[1, 1], q_dist, evals_full, ticks_pos, ticks_labels,
        '(d) Full Berry — individual bands'
    )
    
    # Share y-axes within each row
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = PLOT_DIR / f"eigenvalue_tubes.{ext}"
        plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved eigenvalue_tubes.png/pdf to {PLOT_DIR}/")
    plt.close()
    
    # ── Print tube summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TUBE SUMMARY")
    print(f"{'='*60}")
    
    for label, evals, groups in [
        ("Diagonal Berry", evals_diag, groups_diag),
        ("Full Berry", evals_full, groups_full)
    ]:
        print(f"\n  --- {label} ---")
        print(f"  {'Tube':>5s}  {'Bands':>10s}  {'E_min':>12s}  {'E_max':>12s}  {'Width':>10s}  {'Gap above':>10s}")
        print(f"  {'-----':>5s}  {'----------':>10s}  {'------------':>12s}  {'------------':>12s}  {'----------':>10s}  {'----------':>10s}")
        
        for gi, (start, end) in enumerate(groups):
            tube_min = np.nanmin(evals[:, start:end+1])
            tube_max = np.nanmax(evals[:, start:end+1])
            tube_width = tube_max - tube_min
            
            gap = None
            if gi < len(groups) - 1:
                next_start = groups[gi + 1][0]
                gap = np.nanmin(evals[:, next_start]) - tube_max
            
            gap_str = f"{gap:.2e}" if gap is not None else "—"
            print(f"  {gi:>5d}  {start:>4d}–{end:<4d}  {tube_min:>+12.6f}  {tube_max:>+12.6f}  "
                  f"{tube_width:>10.2e}  {gap_str:>10s}")


if __name__ == '__main__':
    main()
