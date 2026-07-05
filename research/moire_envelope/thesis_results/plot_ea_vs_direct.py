import numpy as np
import matplotlib.pyplot as plt

def load_data():
    ea_8 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_8deg.npz")["eigenvalues"]
    ea_3 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_3deg.npz")["eigenvalues"]
    ea_1 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_1deg.npz")["eigenvalues"]

    d_8 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/compare_3solver_8deg.npz")
    fd_8 = d_8["freqs_hybrid"]
    mpb_8 = d_8["freqs_mpb"]
    
    d_3 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/supercell_3deg_50modes_comparison.npz")
    fd_3 = d_3["freqs_fdfd"]
    mpb_3 = d_3["freqs_mpb"]

    d_1 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/hybrid_1deg_res1.npz")
    fd_1 = d_1["freqs"]
    
    # Calculate implied lambda_ref by matching the first mode of EA to FDFD
    ref_8 = fd_8[0]**2 * (2*np.pi)**2 - ea_8[0]
    ref_3 = fd_3[0]**2 * (2*np.pi)**2 - ea_3[0]
    ref_1 = fd_1[0]**2 * (2*np.pi)**2 - ea_1[0]
    
    # Convert EA delta_lambda to frequencies
    omega_ea_8 = np.sqrt(np.maximum(ea_8 + ref_8, 0)) / (2*np.pi)
    omega_ea_3 = np.sqrt(np.maximum(ea_3 + ref_3, 0)) / (2*np.pi)
    omega_ea_1 = np.sqrt(np.maximum(ea_1 + ref_1, 0)) / (2*np.pi)
    
    return {
        8: {'ea': omega_ea_8, 'fdfd': fd_8, 'mpb': mpb_8},
        3: {'ea': omega_ea_3, 'fdfd': fd_3, 'mpb': mpb_3},
        1: {'ea': omega_ea_1, 'fdfd': fd_1, 'mpb': None}
    }

def main():
    data = load_data()
    
    # Configuration
    degrees = [8, 3, 1]
    modes_to_plot = 50
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=150, gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle('Direct Computation vs Envelope Approximation\nEigenfrequencies at $\Gamma$', fontsize=16, y=0.95)
    
    colors = {
        'mpb': '#1f77b4',     # blue
        'fdfd': '#2ca02c',    # green (1 px/cell if available)
        'ea': '#d62728',      # red
    }
    
    markers = {
        'mpb': 'x',
        'fdfd': 's',
        'ea': 'o',
    }
    
    for i, deg in enumerate(degrees):
        d = data[deg]
        n_modes = min(len(d['ea']), len(d['fdfd']))
        if d['mpb'] is not None:
            n_modes = min(n_modes, len(d['mpb']))
        n_modes = min(n_modes, modes_to_plot)
        
        mode_idx = np.arange(1, n_modes + 1)
        
        # --- Axis 1: Line plot of frequencies ---
        ax1 = axes[i, 0]
        if d['mpb'] is not None:
            ax1.plot(mode_idx, d['mpb'][:n_modes], color=colors['mpb'], marker=markers['mpb'], 
                     linewidth=1, markersize=4, label='MPB (High Res)')
                     
        ax1.plot(mode_idx, d['fdfd'][:n_modes], color=colors['fdfd'], marker=markers['fdfd'], 
                 linewidth=1, markersize=4, label='FDFD 1 px/cell')
                 
        ax1.plot(mode_idx, d['ea'][:n_modes], color=colors['ea'], marker=markers['ea'], 
                 linewidth=1, markersize=4, alpha=0.8, label='Envelope Approx')
                 
        ax1.set_title(f'{deg}° Moiré Cell Eigenfrequencies')
        ax1.set_xlabel('Mode Index $n$')
        ax1.set_ylabel('Frequency $\omega [a/2\pi c]$')
        ax1.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax1.legend()
            
        # --- Axis 2: Horizontal Ladder Plot ---
        ax2 = axes[i, 1]
        
        y_positions = [2, 1, 0] # MPB, FDFD, EA for top/mid/bottom
        labels = ['MPB', 'FDFD', 'EA']
        if d['mpb'] is None:
            y_positions = [1, 0]
            labels = ['FDFD', 'EA']
            
        for name, y_pos in zip(['mpb', 'fdfd', 'ea'], [2, 1, 0] if d['mpb'] is not None else [None, 1, 0]):
            if name == 'mpb' and d['mpb'] is None:
                continue
                
            freqs = d[name][:n_modes]
            
            # Draw baseline
            min_f = min([np.min(d[k][:n_modes]) for k in d if d[k] is not None])
            max_f = max([np.max(d[k][:n_modes]) for k in d if d[k] is not None])
            margin = (max_f - min_f) * 0.1
            ax2.hlines(y_pos, min_f - margin, max_f + margin, color='k', linestyle='-', alpha=0.2)
            
            # Draw vertical bars for each mode
            for j, f in enumerate(freqs):
                ax2.vlines(f, y_pos - 0.25, y_pos + 0.25, color=colors[name], linewidth=2, alpha=0.7)
                
            # Add text label
            ax2.text(min_f - margin, y_pos, f" {name.upper()}", va='center', ha='right', fontsize=12)
            
        ax2.set_yticks([]) # Hide y ticks
        ax2.set_title(f'{deg}° Gap Structure Comparison')
        ax2.set_xlabel('Frequency $\omega [a/2\pi c]$')
        
        # Calculate max error and sub-title
        err_ea_fdfd = np.max(np.abs(d['ea'][:n_modes] - d['fdfd'][:n_modes]) / d['fdfd'][:n_modes].clip(1e-10)) * 100
        
        if d['mpb'] is not None:
            err_mpb_fdfd = np.max(np.abs(d['mpb'][:n_modes] - d['fdfd'][:n_modes]) / d['mpb'][:n_modes].clip(1e-10)) * 100
            ax2.text(1.02, 0.5, f"Max Diff:\nMPB vs FDFD: {err_mpb_fdfd:.2f}%\nEA vs FDFD: {err_ea_fdfd:.2f}%", 
                     transform=ax2.transAxes, va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax2.text(1.02, 0.5, f"Max Diff:\nEA vs FDFD: {err_ea_fdfd:.2f}%", 
                     transform=ax2.transAxes, va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                     
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.88)
    plt.savefig("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_vs_direct_comparison.png", bbox_inches='tight')
    print("Saved ea_vs_direct_comparison.png")

if __name__ == "__main__":
    main()
