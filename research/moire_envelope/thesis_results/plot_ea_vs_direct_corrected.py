import numpy as np
import matplotlib.pyplot as plt

def load_data():
    ea_8 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_8deg.npz")["eigenvalues"]
    ea_3 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_3deg.npz")["eigenvalues"]
    ea_1 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_gamma_modes_1deg.npz")["eigenvalues"]

    # 8deg: compare_3solver has MPB + FDFD (hybrid)
    d_8 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/compare_3solver_8deg.npz")
    fd_8 = d_8["freqs_hybrid"]
    mpb_8 = d_8["freqs_mpb"]

    # 3deg: use 4 px/cell FDFD data
    fd_3 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/hybrid_3deg_res4.npz")["freqs"]
    # MPB from 128-res dataset
    d_3mpb = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/supercell_3deg_mpb_128.npz")
    mpb_3 = d_3mpb["freqs_mpb"]

    # 1deg: use 1 px/cell FDFD data, no MPB available
    fd_1 = np.load("/home/renlephy/msl/research/moire_envelope/thesis_results/hybrid_1deg_res1.npz")["freqs"]

    # Calculate implied lambda_ref by matching mode 0 of EA to FDFD
    ref_8 = fd_8[0]**2 * (2*np.pi)**2 - ea_8[0]
    ref_3 = fd_3[0]**2 * (2*np.pi)**2 - ea_3[0]
    ref_1 = fd_1[0]**2 * (2*np.pi)**2 - ea_1[0]

    omega_ea_8 = np.sqrt(np.maximum(ea_8 + ref_8, 0)) / (2*np.pi)
    omega_ea_3 = np.sqrt(np.maximum(ea_3 + ref_3, 0)) / (2*np.pi)
    omega_ea_1 = np.sqrt(np.maximum(ea_1 + ref_1, 0)) / (2*np.pi)

    return {
        8: {'ea': omega_ea_8, 'fdfd': fd_8, 'mpb': mpb_8},
        3: {'ea': omega_ea_3, 'fdfd': fd_3, 'mpb': mpb_3},
        1: {'ea': omega_ea_1, 'fdfd': fd_1, 'mpb': None},
    }

def main():
    data = load_data()
    degrees = [8, 3, 1]
    modes_to_plot = 50

    fig, axes = plt.subplots(3, 2, figsize=(20, 16), dpi=150,
                             gridspec_kw={'width_ratios': [1, 2]})

    colors = {'mpb': '#1f77b4', 'fdfd': '#2ca02c', 'ea': '#d62728'}
    markers = {'mpb': 'x', 'fdfd': 's', 'ea': 'o'}

    for i, deg in enumerate(degrees):
        d = data[deg]
        n_modes = min(len(d['ea']), len(d['fdfd']))
        if d['mpb'] is not None:
            n_modes = min(n_modes, len(d['mpb']))
        n_modes = min(n_modes, modes_to_plot)
        mode_idx = np.arange(1, n_modes + 1)

        # ---- Left: Line plot ----
        ax1 = axes[i, 0]
        if d['mpb'] is not None:
            ax1.plot(mode_idx, d['mpb'][:n_modes], color=colors['mpb'],
                     marker=markers['mpb'], linewidth=1.5, markersize=5,
                     label='MPB (high res)')
        ax1.plot(mode_idx, d['fdfd'][:n_modes], color=colors['fdfd'],
                 marker=markers['fdfd'], linewidth=1.5, markersize=5,
                 label='FDFD 1 px/cell')
        ax1.plot(mode_idx, d['ea'][:n_modes], color=colors['ea'],
                 marker=markers['ea'], linewidth=1.5, markersize=5,
                 alpha=0.8, label='Envelope Approx')
        ax1.set_title(f'{deg}° Moiré — Eigenfrequencies')
        ax1.set_xlabel('Mode index $n$')
        ax1.set_ylabel(r'$\omega\;[a/2\pi c]$')
        ax1.grid(True, ls='--', alpha=0.4)
        ax1.legend(fontsize=9)

        # ---- Right: Ladder plot ----
        ax2 = axes[i, 1]
        solver_list = []
        if d['mpb'] is not None:
            solver_list.append(('mpb', 2))
        solver_list += [('fdfd', 1), ('ea', 0)]

        all_freqs = np.concatenate([d[k][:n_modes] for k, _ in solver_list])
        min_f, max_f = all_freqs.min(), all_freqs.max()
        span = max_f - min_f
        pad = span * 0.06

        for name, y_pos in solver_list:
            freqs = d[name][:n_modes]
            ax2.hlines(y_pos, min_f - pad, max_f + pad, color='k', lw=0.5, alpha=0.15)
            for f in freqs:
                ax2.vlines(f, y_pos - 0.25, y_pos + 0.25,
                           color=colors[name], linewidth=2.5, alpha=0.85)
            ax2.text(min_f - pad * 1.2, y_pos, name.upper(),
                     va='center', ha='right', fontsize=11,
                     fontweight='bold', color=colors[name])

        ax2.set_yticks([])
        ax2.set_xlim(min_f - pad * 3, max_f + pad)
        ylo = -0.5
        yhi = (2.5 if d['mpb'] is not None else 1.5)
        ax2.set_ylim(ylo, yhi)
        ax2.set_title(f'{deg}° Moiré — Gap Structure')
        ax2.set_xlabel(r'$\omega\;[a/2\pi c]$')

    fig.suptitle(
        'Direct Computation vs Envelope Approximation — '
        r'Eigenfrequencies at $\Gamma$',
        fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("/home/renlephy/msl/research/moire_envelope/thesis_results/ea_vs_direct_comparison.png",
                bbox_inches='tight')
    print("Saved ea_vs_direct_comparison.png")

if __name__ == "__main__":
    main()
