#!/usr/bin/env python
"""
Quick honeycomb Phase 1/2/3 diagnostics from the base run (θ=1.1°).
Generates diagnostic plots without needing complete η-sweep data.
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from thesis_utils import (
    apply_thesis_style, save_figure, ensure_output_dir,
    find_candidate_dir, load_phase1_data, load_phase2_data, load_phase0_meta,
    CANDIDATE_COLORS,
)

import h5py

TASK = "T_honeycomb_analysis"
HC_COLOR = CANDIDATE_COLORS.get('honeycomb_K_b1', '#CC79A7')


def main():
    out_dir = ensure_output_dir(TASK)
    cand_dir = find_candidate_dir('honeycomb_K_b1')
    meta = load_phase0_meta(cand_dir)
    print(f"Candidate dir: {cand_dir}")
    print(f"Meta: lattice={meta.get('lattice_type')}, K-point={meta.get('high_symmetry_point')}")
    print(f"  eps_hole={meta.get('eps_hole')}, eps_bg={meta.get('eps_bg')}")

    # === Load Phase 2 data ===
    p2_path = cand_dir / "phase2_multiband_data.h5"
    with h5py.File(p2_path, 'r') as hf:
        Lambda = hf['Lambda'][:]
        A_berry = hf['A_berry'][:]
        M_inv = hf['M_inv'][:]
        v_drift = hf['v_drift'][:]
        Phi_BH = hf['Phi_BH'][:]
        omega = hf['omega'][:]
        Ns = int(hf.attrs.get('Ns1', 128))
        N_sub = int(hf.attrs.get('N_subspace', 2))
        eta_p2 = float(hf.attrs.get('eta', 0))
        print(f"  Phase 2: Ns={Ns}, N_sub={N_sub}, η={eta_p2:.6f}")
        print(f"  Λ shape: {Lambda.shape}, A shape: {A_berry.shape}")
        print(f"  M⁻¹ shape: {M_inv.shape}")
        print(f"  ω shape: {omega.shape}, range [{omega.min():.6f}, {omega.max():.6f}]")

    # === Load Phase 3 data ===
    p3_path = cand_dir / "phase3_multiband_modes.h5"
    with h5py.File(p3_path, 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        envelope_modes = hf['envelope_modes'][:] if 'envelope_modes' in hf else None
        n_modes = len(eigenvalues)
        print(f"  Phase 3: {n_modes} modes, E range [{eigenvalues[0]:.8f}, {eigenvalues[-1]:.8f}]")

    # ========================
    # Figure: Phase 2 diagnostics (6 panels)
    # ========================
    apply_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) ω band 0
    ax = axes[0, 0]
    im = ax.imshow(omega[:, :, 0].T, origin='lower', extent=[0, 1, 0, 1],
                   aspect='equal', cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label=r'$\omega_0(\mathbf{R})$')
    ax.set_title(f'(a) Band 0 frequency')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    # (b) Band splitting ω₁ - ω₀
    ax = axes[0, 1]
    if omega.shape[2] >= 2:
        splitting = omega[:, :, 1] - omega[:, :, 0]
        im = ax.imshow(splitting.T, origin='lower', extent=[0, 1, 0, 1],
                       aspect='equal', cmap='hot')
        plt.colorbar(im, ax=ax, label=r'$\Delta\omega$')
        ax.set_title(f'(b) Splitting $\omega_1 - \omega_0$ (min={splitting.min():.5f})')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    # (c) Moiré potential Λ₀₀
    ax = axes[0, 2]
    V00 = Lambda[:, :, 0, 0].real if Lambda.ndim == 4 else Lambda[:, :, 0].real
    im = ax.imshow(V00.T, origin='lower', extent=[0, 1, 0, 1],
                   aspect='equal', cmap='coolwarm')
    plt.colorbar(im, ax=ax, label=r'$\Lambda_{00}(\mathbf{R})$')
    ax.set_title(f'(c) Potential $\Lambda_{{00}}$ — range [{V00.min():.4f}, {V00.max():.4f}]')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    # (d) Off-diagonal |Λ₀₁|
    ax = axes[1, 0]
    if Lambda.ndim == 4 and Lambda.shape[2] >= 2:
        V01 = np.abs(Lambda[:, :, 0, 1])
        im = ax.imshow(V01.T, origin='lower', extent=[0, 1, 0, 1],
                       aspect='equal', cmap='magma')
        plt.colorbar(im, ax=ax, label=r'$|\Lambda_{01}|$')
        ax.set_title(f'(d) Inter-band $|\\Lambda_{{01}}|$ max={V01.max():.5f}')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    # (e) Diagonal Berry connection |A₀₀|
    ax = axes[1, 1]
    if A_berry.ndim >= 4:
        A_diag = np.sqrt(np.abs(A_berry[:, :, 0, 0, 0])**2 + np.abs(A_berry[:, :, 0, 0, 1])**2)
        im = ax.imshow(A_diag.T, origin='lower', extent=[0, 1, 0, 1],
                       aspect='equal', cmap='viridis')
        plt.colorbar(im, ax=ax, label=r'$|\mathbf{A}_{00}|$')
        ax.set_title(f'(e) Berry $|\\mathbf{{A}}_{{00}}|$ max={A_diag.max():.3f}')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    # (f) Off-diagonal |A₀₁|
    ax = axes[1, 2]
    if A_berry.ndim >= 4 and A_berry.shape[2] >= 2:
        A_off = np.sqrt(np.abs(A_berry[:, :, 0, 1, 0])**2 + np.abs(A_berry[:, :, 0, 1, 1])**2)
        im = ax.imshow(A_off.T, origin='lower', extent=[0, 1, 0, 1],
                       aspect='equal', cmap='inferno')
        plt.colorbar(im, ax=ax, label=r'$|\mathbf{A}_{01}|$')
        ax.set_title(f'(f) Off-diag $|\\mathbf{{A}}_{{01}}|$ max={A_off.max():.3f}')
    ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    fig.suptitle('Honeycomb K-point Dirac Candidate — Phase 2 Diagnostics\n'
                 f'(ε_rod={meta.get("eps_hole",11.56)}, r/a={meta.get("r_over_a",0.2)}, '
                 f'K=(2/3, 1/3), TM, 2-band Dirac subspace, C6 symmetrized)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F1_honeycomb_phase2_diagnostics")

    # ========================
    # Figure: Phase 3 eigenvalue spectrum
    # ========================
    apply_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Eigenvalue ladder
    ax = axes[0]
    ax.plot(range(n_modes), eigenvalues, 'o-', color=HC_COLOR, markersize=4)
    ax.set_xlabel('Mode index $n$')
    ax.set_ylabel('$E_n$')
    ax.set_title(f'(a) Eigenvalue spectrum (N={n_modes})')
    ax.axhline(y=eigenvalues[0], color='gray', ls='--', alpha=0.3)

    # (b) Level spacings Δ_n = E_{n+1} - E_n
    ax = axes[1]
    spacings = np.diff(eigenvalues)
    ax.semilogy(range(len(spacings)), spacings, 'o-', color=HC_COLOR, markersize=3)
    ax.set_xlabel('Index $n$')
    ax.set_ylabel('$E_{n+1} - E_n$')
    ax.set_title(f'(b) Level spacings (median={np.median(spacings):.6f})')

    # (c) Spacing ratio r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    ax = axes[2]
    r_vals = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i+1]
        r = min(s1, s2) / max(s1, s2) if max(s1, s2) > 0 else 0
        r_vals.append(r)
    ax.plot(range(len(r_vals)), r_vals, 'o-', color=HC_COLOR, markersize=3)
    ax.axhline(y=0.386, color='blue', ls='--', alpha=0.4, label='Poisson (0.386)')
    ax.axhline(y=0.536, color='red', ls='--', alpha=0.4, label='GOE (0.536)')
    ax.set_xlabel('Index $n$')
    ax.set_ylabel(r'$r_n$')
    ax.set_title(f'(c) Level spacing ratio (⟨r⟩={np.mean(r_vals):.3f})')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    fig.suptitle(f'Honeycomb Phase 3 Eigenspectrum @ θ={meta.get("theta_deg", 1.1):.1f}°',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F2_honeycomb_phase3_spectrum")

    # ========================
    # Figure: Effective mass tensor & inter-band coupling 
    # ========================
    apply_thesis_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # M⁻¹ tensor components
    for b in range(min(N_sub, 2)):
        for idx_ij, (i, j) in enumerate([(0, 0), (0, 1), (1, 1)]):
            ax = axes[b, idx_ij]
            Mij = M_inv[:, :, b, b, i, j].real
            im = ax.imshow(Mij.T, origin='lower', extent=[0, 1, 0, 1],
                           aspect='equal', cmap='coolwarm')
            plt.colorbar(im, ax=ax, label=f'$M^{{-1}}_{{{b}{b},{i}{j}}}$')
            ax.set_title(f'Band {b}: $M^{{-1}}_{{{i}{j}}}$ [{Mij.min():.2f}, {Mij.max():.2f}]')
            ax.set_xlabel('$s_1$'); ax.set_ylabel('$s_2$')

    fig.suptitle('Effective Mass Tensor Components (diagonal blocks)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, TASK, "F3_honeycomb_mass_tensor")

    # ========================
    # Print summary
    # ========================
    print("\n" + "=" * 70)
    print("HONEYCOMB PHASE 3 SUMMARY (base run, θ=1.1°)")
    print("=" * 70)
    print(f"  Lattice type: {meta.get('lattice_type')}")
    print(f"  K-point: {meta.get('high_symmetry_point')}")
    print(f"  ε_rod/ε_bg: {meta.get('eps_hole')}/{meta.get('eps_bg')}")
    print(f"  N_subspace: {N_sub}")
    print(f"  Grid: {Ns}×{Ns}")
    print(f"  ω range: [{omega.min():.6f}, {omega.max():.6f}]")
    print(f"  Λ₀₀ range: [{Lambda[:,:,0,0].real.min():.6f}, {Lambda[:,:,0,0].real.max():.6f}]")
    if Lambda.ndim == 4 and Lambda.shape[2] >= 2:
        print(f"  |Λ₀₁| max: {np.abs(Lambda[:,:,0,1]).max():.6f}")
    print(f"  |A₀₀| max: {np.sqrt(np.abs(A_berry[:,:,0,0,0])**2 + np.abs(A_berry[:,:,0,0,1])**2).max():.4f}")
    if A_berry.shape[2] >= 2:
        print(f"  |A₀₁| max: {np.sqrt(np.abs(A_berry[:,:,0,1,0])**2 + np.abs(A_berry[:,:,0,1,1])**2).max():.4f}")
    print(f"  M⁻¹₀₀ Tr: {(M_inv[:,:,0,0,0,0] + M_inv[:,:,0,0,1,1]).mean():.4f}")
    if M_inv.shape[2] >= 2:
        print(f"  M⁻¹₁₁ Tr: {(M_inv[:,:,1,1,0,0] + M_inv[:,:,1,1,1,1]).mean():.4f}")
    print(f"  Phase 3 modes: {n_modes}")
    print(f"  E₀ = {eigenvalues[0]:.8f}")
    print(f"  E₁ = {eigenvalues[1]:.8f}")
    print(f"  Gap E₁-E₀ = {eigenvalues[1]-eigenvalues[0]:.8f}")
    print(f"  BW₅₀ (E₄₉-E₀) = {eigenvalues[min(49,n_modes-1)]-eigenvalues[0]:.8f}")
    print(f"  BW₁₀₀ (E₉₉-E₀) = {eigenvalues[-1]-eigenvalues[0]:.8f}")
    print(f"  Median spacing: {np.median(spacings):.8f}")
    print(f"  ⟨r⟩ spacing ratio: {np.mean(r_vals):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
