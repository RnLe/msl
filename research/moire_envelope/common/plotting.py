"""
Plotting utilities for visualization
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math


def plot_phase1_fields(cdir, R_grid, V, vg, M_inv, candidate_params=None):
    """
    Create visualization of Phase 1 fields
    
    Args:
        cdir: Candidate directory path
        R_grid: Spatial grid [Nx, Ny, 2]
        V: Potential field [Nx, Ny]
        vg: Group velocity [Nx, Ny, 2]
        M_inv: Inverse mass tensor [Nx, Ny, 2, 2]
        candidate_params: Optional candidate parameters for title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Build title
    if candidate_params:
        title_str = (f"Candidate {candidate_params.get('candidate_id', '?')}: "
                    f"{candidate_params.get('lattice_type', '?')}, "
                    f"r/a={candidate_params.get('r_over_a', 0):.3f}, "
                    f"ε={candidate_params.get('eps_bg', 0):.1f}")
        fig.suptitle(title_str, fontsize=12, fontweight='bold')
    
    # Plot V(R)
    ax = axes[0, 0]
    im = ax.imshow(V.T, origin='lower', cmap='RdBu_r', aspect='auto')
    ax.set_title(f'Potential V(R) [range: {V.min():.4f}, {V.max():.4f}]')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    plt.colorbar(im, ax=ax, label='Frequency shift')
    
    # Plot |vg(R)|
    ax = axes[0, 1]
    vg_norm = np.linalg.norm(vg, axis=-1)
    im = ax.imshow(vg_norm.T, origin='lower', cmap='viridis', aspect='auto')
    ax.set_title(f'|v_g(R)| [max: {vg_norm.max():.4f}]')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    plt.colorbar(im, ax=ax, label='Group velocity')
    
    # Plot eigenvalue 1 of M_inv
    ax = axes[1, 0]
    eigvals = np.linalg.eigvalsh(M_inv)
    im = ax.imshow(eigvals[..., 0].T, origin='lower', cmap='plasma', aspect='auto')
    ax.set_title(f'M⁻¹ eigenvalue 1 (smaller)')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    plt.colorbar(im, ax=ax, label='Curvature')
    
    # Plot eigenvalue 2 of M_inv
    ax = axes[1, 1]
    im = ax.imshow(eigvals[..., 1].T, origin='lower', cmap='plasma', aspect='auto')
    ax.set_title(f'M⁻¹ eigenvalue 2 (larger)')
    ax.set_xlabel('x index')
    ax.set_ylabel('y index')
    plt.colorbar(im, ax=ax, label='Curvature')
    
    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase1_fields_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_envelope_modes(cdir, R_grid, F, eigenvalues, n_modes=4):
    """
    Plot envelope mode profiles
    
    Args:
        cdir: Candidate directory path
        R_grid: Spatial grid [Nx, Ny, 2]
        F: Envelope fields [n_modes, Nx, Ny]
        eigenvalues: Array of eigenvalues
        n_modes: Number of modes to plot
    """
    n_plot = min(n_modes, len(eigenvalues))
    n_cols = 2
    n_rows = (n_plot + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4*n_rows))
    axes = axes.flatten() if n_plot > 1 else [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        field_abs2 = np.abs(F[i])**2
        im = ax.imshow(field_abs2.T, origin='lower', cmap='hot', aspect='auto')
        ax.set_title(f'Mode {i}: Δω = {eigenvalues[i]:.6f}')
        ax.set_xlabel('x index')
        ax.set_ylabel('y index')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(n_plot, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase3_cavity_modes.png', dpi=150)
    plt.close()
    
    # Also plot spectrum
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(len(eigenvalues)), eigenvalues, 'o-')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Eigenvalue Δω')
    ax.set_title('Envelope Spectrum')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(Path(cdir) / 'phase3_spectrum.png', dpi=150)
    plt.close()


def make_phase1_plots(cdir, R_grid, V, vg, M_inv):
    """Wrapper for Phase 1 plotting"""
    plot_phase1_fields(cdir, R_grid, V, vg, M_inv)


def plot_phase4_comparison(cdir, ea_eigs, moire_bands, comparison):
    """
    Plot Phase 4 validation comparison
    
    Args:
        cdir: Candidate directory path
        ea_eigs: EA eigenvalues DataFrame
        moire_bands: Full moiré band structure
        comparison: Comparison DataFrame
    """
    # Placeholder - will be implemented in Phase 4
    pass


def plot_band_structure(bands, candidate_row, save_path):
    """
    Plot band structure with highlighted candidate k-point and band
    
    Args:
        bands: Band structure data from compute_bandstructure
        candidate_row: Candidate parameters (dict or DataFrame row)
        save_path: Path to save the figure
    """
    freqs = bands['frequencies']  # Shape: (n_k, n_bands)
    k_labels = bands.get('k_labels', []) or []
    k_path = bands.get('k_path')
    label_positions = bands.get('k_label_positions')
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot all bands
    n_k, n_bands = freqs.shape
    if k_path is not None:
        x = np.asarray(k_path)
    else:
        x = np.arange(n_k)
    
    for band_idx in range(n_bands):
        ax.plot(x, freqs[:, band_idx], 'b-', alpha=0.6, linewidth=1)
    
    # Highlight the candidate band and k-point
    target_band = int(candidate_row.get('band_index', 0))
    k_label = candidate_row.get('k_label', 'Γ')
    
    if target_band < n_bands:
        ax.plot(x, freqs[:, target_band], 'r-', linewidth=2, label=f'Band {target_band}')
    
    # Mark high symmetry points with vertical lines
    if k_labels:
        if label_positions is not None and len(label_positions) == len(k_labels):
            x_positions = label_positions
        else:
            num_segments = max(1, len(k_labels) - 1)
            x_positions = np.linspace(x[0], x[-1], len(k_labels))

        y_min, y_max = freqs.min(), freqs.max()
        y_range = y_max - y_min
        y_label = y_max + 0.04 * (y_range if y_range > 0 else 1.0)

        for pos, label in zip(x_positions, k_labels):
            ax.axvline(pos, color='k', linestyle='-', alpha=0.4, linewidth=1.0)
            ax.text(pos, y_label, label, ha='center', va='bottom', fontsize=10, weight='bold')

        # Highlight the target k-point marker
        if target_band < n_bands and k_label in k_labels:
            label_idx = k_labels.index(k_label)
            x_pos = x_positions[label_idx]
            ax.plot(x_pos, np.interp(x_pos, x, freqs[:, target_band]), 'ro',
                    markersize=8, zorder=10,
                    label=f'{k_label}, ω={np.interp(x_pos, x, freqs[:, target_band]):.4f}')
    
    ax.set_xlabel('k-path', fontsize=10)
    ax.set_ylabel('Frequency (c/a)', fontsize=10)
    ax.set_title(f"{candidate_row.get('lattice_type', 'lattice')}, r/a={candidate_row.get('r_over_a', 0):.2f}, ε={candidate_row.get('eps_bg', 0):.1f}", 
                fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x[0], x[-1])

    # Add margins to y-limits for readability
    y_min, y_max = freqs.min(), freqs.max()
    y_range = y_max - y_min
    y_margin = 0.05 * (y_range if y_range > 0 else 1.0)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_top_candidates_grid(top_candidates, bands_list, save_path, n_cols=4):
    """
    Create a grid of band diagrams for top candidates
    
    Args:
        top_candidates: DataFrame of top candidates
        bands_list: List of band structure data for each candidate
        save_path: Path to save the figure
        n_cols: Number of columns in the grid
    """
    n_candidates = len(top_candidates)
    n_rows = int(np.ceil(n_candidates / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = axes.flatten() if n_candidates > 1 else [axes]
    
    for idx, (_, row) in enumerate(top_candidates.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        bands = bands_list[idx]
        freqs = bands['frequencies']
        k_labels = bands.get('k_labels', []) or []
        k_path = bands.get('k_path')
        label_positions = bands.get('k_label_positions')
        
        # Plot bands
        n_k, n_bands = freqs.shape
        if k_path is not None:
            x = np.asarray(k_path)
        else:
            x = np.arange(n_k)
        
        for band_idx in range(n_bands):
            ax.plot(x, freqs[:, band_idx], 'b-', alpha=0.5, linewidth=0.8)
        
        # Highlight target band
        target_band = int(row['band_index'])
        k_label = row['k_label']
        
        if target_band < n_bands:
            ax.plot(x, freqs[:, target_band], 'r-', linewidth=1.5)
        
        # Mark k-point
        if k_labels and k_label in k_labels:
            if label_positions is not None and len(label_positions) == len(k_labels):
                x_pos = label_positions[k_labels.index(k_label)]
            else:
                num_segments = max(1, len(k_labels) - 1)
                x_pos = np.linspace(x[0], x[-1], len(k_labels))[k_labels.index(k_label)]
            if target_band < n_bands:
                y_val = np.interp(x_pos, x, freqs[:, target_band])
                ax.plot(x_pos, y_val, 'ro', markersize=6)
        
        # Mark high symmetry points with vertical lines
        if k_labels:
            if label_positions is not None and len(label_positions) == len(k_labels):
                x_positions = label_positions
            else:
                x_positions = np.linspace(x[0], x[-1], len(k_labels))

            y_min, y_max = freqs.min(), freqs.max()
            y_range = y_max - y_min
            y_label = y_max + 0.03 * (y_range if y_range > 0 else 1.0)

            for pos, label in zip(x_positions, k_labels):
                ax.axvline(pos, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
                # Only label first/last to avoid crowding in grid
                if label == k_labels[0] or label == k_labels[-1]:
                    ax.text(pos, y_label, label, ha='center', va='bottom', fontsize=7)
        
        # Title with key info
        title = (f"#{row['candidate_id']}: {row['lattice_type']}\n"
            f"r/a={row['r_over_a']:.2f}, ε={row['eps_bg']:.1f}, {k_label}-band{target_band}\n"
                f"Score={row['S_total']:.3f}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('k-path', fontsize=8)
        ax.set_ylabel('ω (c/a)', fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax.set_xlim(x[0], x[-1])
        
        # Set y-limits to show all bands clearly
        y_margin = 0.05 * (freqs.max() - freqs.min())
        ax.set_ylim(freqs.min() - y_margin, freqs.max() + y_margin)
    
    # Hide unused subplots
    for idx in range(n_candidates, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved candidate grid plot to: {save_path}")
