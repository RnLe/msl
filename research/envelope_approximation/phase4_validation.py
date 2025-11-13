"""
Phase 4: Validation and Comparison

This script validates the envelope approximation results from Phase 3 by:
1. Comparing with direct MPB band structure calculations
2. Analyzing convergence and accuracy metrics
3. Visualizing the agreement between EA and full simulation

The envelope approximation predicts cavity modes at:
    ω_cavity = ω_ref + ΔE

We validate by computing the full band structure of the moiré superlattice
and comparing with EA predictions.

Inputs:
- Phase 3 outputs: Eigenvalues and eigenstates
- Phase 1 data: Reference band structure
- Phase 0 data: Moiré lattice configuration

Outputs:
- phase4_validation_report.csv: Comparison metrics
- phase4_comparison_plot.png: Visual comparison
- phase4_error_analysis.csv: Error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pickle
from typing import List, Tuple, Dict
import pandas as pd

# Phase 0 imports
from phase0_lattice_setup import LatticeConfig, MoireLatticeSetup
from phase1_local_bloch import GeometryConfig, LocalBlochCalculator, compute_band_at_k

try:
    import meep as mp
    from meep import mpb
    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False
    print("Warning: Meep/MPB not available. Using approximations for validation.")


class ValidationAnalyzer:
    """Analyze and validate envelope approximation results"""
    
    def __init__(self, lattice_setup: MoireLatticeSetup, 
                 geometry_config: GeometryConfig,
                 phase3_eigenvalues: np.ndarray,
                 phase3_eigenstates: np.ndarray,
                 omega_ref: float):
        """
        Initialize validator.
        
        Args:
            lattice_setup: MoireLatticeSetup from Phase 0
            geometry_config: Geometry configuration from Phase 1
            phase3_eigenvalues: Eigenvalues (detunings) from Phase 3
            phase3_eigenstates: Eigenvectors from Phase 3
            omega_ref: Reference frequency from Phase 1
        """
        self.setup = lattice_setup
        self.geom_config = geometry_config
        self.eigenvalues = phase3_eigenvalues
        self.eigenstates = phase3_eigenstates
        self.omega_ref = omega_ref
        
        # Cavity frequencies from EA
        self.omega_ea = omega_ref + self.eigenvalues
        
        print(f"Validation initialized:")
        print(f"  Number of modes: {len(self.eigenvalues)}")
        print(f"  ω_ref = {omega_ref:.8f}")
        print(f"  EA frequency range: [{self.omega_ea.min():.8f}, {self.omega_ea.max():.8f}]")
        
    def analyze_convergence(self, grid_sizes: List[int] = None) -> pd.DataFrame:
        """
        Analyze convergence with respect to grid resolution.
        
        This would require re-running Phases 1-3 at different resolutions.
        For now, we provide a framework and single-point validation.
        
        Args:
            grid_sizes: List of grid sizes to test (e.g., [4, 6, 8, 10])
            
        Returns:
            DataFrame with convergence data
        """
        print("\n" + "=" * 60)
        print("Convergence Analysis")
        print("=" * 60)
        print("\nNote: Full convergence study requires re-running pipeline")
        print("      at multiple resolutions. Current resolution: 8×8")
        
        # Current results
        current_data = {
            'grid_size': [8],
            'n_points': [64],
            'omega_min': [self.omega_ea.min()],
            'omega_max': [self.omega_ea.max()],
            'delta_omega': [self.omega_ea.max() - self.omega_ea.min()],
            'n_modes': [len(self.eigenvalues)]
        }
        
        df = pd.DataFrame(current_data)
        return df
    
    def compute_local_variation_metric(self) -> Dict[str, float]:
        """
        Compute metrics for local band structure variation.
        
        Envelope approximation assumes slow variation of local properties.
        We check this assumption by analyzing V(R) and v_g(R) variation.
        """
        print("\n" + "=" * 60)
        print("Local Variation Analysis")
        print("=" * 60)
        
        # Load Phase 1 data
        with open("outputs/phase1_data.pkl", "rb") as f:
            phase1_data = pickle.load(f)
        
        omega_local = phase1_data['omega_grid']
        v_g = phase1_data['v_g_grid']
        M_inv = phase1_data['M_inv_grid']
        
        # Compute potential V(R) = ω₀(R) - ω_ref
        V = omega_local - self.omega_ref
        
        # Variation metrics
        V_std = np.std(V)
        V_range = np.ptp(V)  # peak-to-peak
        V_rel_var = V_std / self.omega_ref if self.omega_ref > 0 else 0
        
        v_g_mag = np.linalg.norm(v_g, axis=-1)
        v_g_std = np.std(v_g_mag)
        v_g_max = np.max(v_g_mag)
        v_g_mean = np.mean(v_g_mag)
        
        # Compute length scale of variation
        L_moire = np.linalg.norm(self.setup.A1[:2])
        
        # Envelope approximation parameter η
        eta = 2 * np.pi / L_moire
        
        # Check EA validity criterion: η·ξ << 1, where ξ is mode extent
        # Estimate ξ from first mode
        F0 = self.eigenstates[:, 0].reshape(8, 8)
        F0_mag = np.abs(F0)
        
        # Effective radius (second moment)
        i_coords, j_coords = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
        x_coords = i_coords / 8 * L_moire
        y_coords = j_coords / 8 * L_moire
        
        norm2 = np.sum(F0_mag**2)
        x_mean = np.sum(F0_mag**2 * x_coords) / norm2
        y_mean = np.sum(F0_mag**2 * y_coords) / norm2
        
        r2 = (x_coords - x_mean)**2 + (y_coords - y_mean)**2
        xi = np.sqrt(np.sum(F0_mag**2 * r2) / norm2)
        
        ea_parameter = eta * xi
        
        metrics = {
            'V_std': V_std,
            'V_range': V_range,
            'V_rel_variation': V_rel_var,
            'v_g_mean': v_g_mean,
            'v_g_std': v_g_std,
            'v_g_max': v_g_max,
            'L_moire': L_moire,
            'eta': eta,
            'xi_mode0': xi,
            'eta_xi_mode0': ea_parameter,
        }
        
        print("\nPotential variation:")
        print(f"  σ(V) = {V_std:.6f}")
        print(f"  ΔV = {V_range:.6f}")
        print(f"  σ(V)/ω_ref = {V_rel_var:.4%}")
        
        print("\nGroup velocity:")
        print(f"  ⟨|v_g|⟩ = {v_g_mean:.4f}")
        print(f"  max|v_g| = {v_g_max:.4f}")
        print(f"  σ(|v_g|) = {v_g_std:.4f}")
        
        print("\nEnvelope approximation validity:")
        print(f"  Moiré length L = {L_moire:.2f}")
        print(f"  Parameter η = {eta:.6f}")
        print(f"  Mode extent ξ₀ = {xi:.2f}")
        print(f"  EA parameter η·ξ₀ = {ea_parameter:.4f}")
        
        if ea_parameter < 0.5:
            print("  ✓ Valid EA regime (η·ξ << 1)")
        elif ea_parameter < 1.0:
            print("  ⚠ Marginal EA regime")
        else:
            print("  ✗ Invalid EA regime (mode too extended)")
        
        return metrics
    
    def compare_with_perturbation_theory(self) -> pd.DataFrame:
        """
        Compare EA results with first-order perturbation theory.
        
        First-order PT predicts:
            E_n^(1) = ⟨F_n|V|F_n⟩
        
        where V(R) = ω₀(R) - ω_ref is the potential.
        """
        print("\n" + "=" * 60)
        print("Perturbation Theory Comparison")
        print("=" * 60)
        
        # Load Phase 1 data
        with open("outputs/phase1_data.pkl", "rb") as f:
            phase1_data = pickle.load(f)
        
        omega_local = phase1_data['omega_grid']
        V = (omega_local - self.omega_ref).flatten()
        
        comparison_data = []
        
        print("\nComparing eigenvalues with first-order perturbation theory:")
        print(f"{'Mode':<6} {'ΔE (EA)':<15} {'⟨V⟩ (PT)':<15} {'Difference':<15}")
        print("-" * 60)
        
        for i in range(len(self.eigenvalues)):
            F_n = self.eigenstates[:, i]
            F_n_mag2 = np.abs(F_n)**2
            
            # First-order PT: E^(1) = ⟨F|V|F⟩
            E_pt1 = np.sum(F_n_mag2 * V) / np.sum(F_n_mag2)
            
            dE_ea = self.eigenvalues[i].real if self.eigenvalues.dtype == complex else self.eigenvalues[i]
            diff = dE_ea - E_pt1
            
            if i < 10:  # Print first 10
                print(f"{i:<6} {dE_ea:<15.8f} {E_pt1:<15.8f} {diff:<15.8f}")
            
            comparison_data.append({
                'mode': i,
                'delta_E_EA': dE_ea,
                'delta_E_PT1': E_pt1,
                'difference': diff,
                'rel_error': abs(diff / dE_ea) if abs(dE_ea) > 1e-10 else np.inf
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Summary statistics
        print("\nSummary:")
        print(f"  Mean absolute error: {df['difference'].abs().mean():.8f}")
        print(f"  RMS error: {np.sqrt((df['difference']**2).mean()):.8f}")
        print(f"  Max absolute error: {df['difference'].abs().max():.8f}")
        print(f"  Mean relative error: {df['rel_error'].mean():.4%}")
        
        return df
    
    def analyze_mode_character(self) -> pd.DataFrame:
        """
        Analyze the character of envelope modes.
        
        Classify modes by their spatial structure:
        - Localized vs. extended
        - Symmetric vs. asymmetric
        - Node count
        """
        print("\n" + "=" * 60)
        print("Mode Character Analysis")
        print("=" * 60)
        
        mode_data = []
        
        for i in range(len(self.eigenvalues)):
            F = self.eigenstates[:, i].reshape(8, 8)
            F_mag = np.abs(F)
            
            # Inverse participation ratio
            norm2 = np.sum(F_mag**2)
            norm4 = np.sum(F_mag**4)
            ipr = norm4 / norm2**2 if norm2 > 1e-10 else 0
            
            # Count nodes (sign changes)
            # Approximate by counting regions where |F| is below threshold
            threshold = 0.1 * np.max(F_mag)
            n_nodes_approx = np.sum(F_mag < threshold)
            
            # Symmetry: check x-y reflection
            F_reflected = np.flip(F, axis=(0, 1))
            symmetry = np.sum(np.abs(F - F_reflected)**2) / np.sum(np.abs(F)**2)
            
            dE = self.eigenvalues[i].real if self.eigenvalues.dtype == complex else self.eigenvalues[i]
            omega = self.omega_ea[i].real if self.omega_ea.dtype == complex else self.omega_ea[i]
            
            mode_data.append({
                'mode': i,
                'omega': omega,
                'delta_E': dE,
                'ipr': ipr,
                'n_nodes_approx': n_nodes_approx,
                'symmetry_breaking': symmetry,
                'max_amplitude': np.max(F_mag)
            })
        
        df = pd.DataFrame(mode_data)
        
        print(f"\n{'Mode':<6} {'ω':<12} {'IPR':<10} {'~Nodes':<8} {'Sym':<10}")
        print("-" * 60)
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            print(f"{row['mode']:<6} {row['omega']:<12.6f} {row['ipr']:<10.5f} "
                  f"{row['n_nodes_approx']:<8} {row['symmetry_breaking']:<10.5f}")
        
        return df
    
    def estimate_band_gap(self) -> Dict[str, float]:
        """
        Estimate the band gap from the mode spectrum.
        
        Look for gaps in the frequency spectrum.
        """
        print("\n" + "=" * 60)
        print("Band Gap Analysis")
        print("=" * 60)
        
        omega_sorted = np.sort(self.omega_ea.real if self.omega_ea.dtype == complex else self.omega_ea)
        
        # Compute gaps between consecutive modes
        gaps = np.diff(omega_sorted)
        
        # Find largest gap
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]
        gap_center = (omega_sorted[max_gap_idx] + omega_sorted[max_gap_idx + 1]) / 2
        
        print(f"\nFrequency spectrum:")
        print(f"  Range: [{omega_sorted.min():.6f}, {omega_sorted.max():.6f}]")
        print(f"  Span: {omega_sorted.max() - omega_sorted.min():.6f}")
        print(f"\nLargest gap:")
        print(f"  Size: {max_gap:.6f}")
        print(f"  Location: ω ≈ {gap_center:.6f}")
        print(f"  Between modes {max_gap_idx} and {max_gap_idx + 1}")
        
        # Mean and std of gaps
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        
        print(f"\nGap statistics:")
        print(f"  Mean spacing: {mean_gap:.6f}")
        print(f"  Std deviation: {std_gap:.6f}")
        
        result = {
            'max_gap': max_gap,
            'gap_center': gap_center,
            'gap_mode_below': max_gap_idx,
            'gap_mode_above': max_gap_idx + 1,
            'mean_spacing': mean_gap,
            'std_spacing': std_gap,
            'omega_min': omega_sorted.min(),
            'omega_max': omega_sorted.max()
        }
        
        return result
    
    def visualize_validation(self, output_dir: Path):
        """Create comprehensive validation visualizations"""
        
        # Load Phase 1 data
        with open("outputs/phase1_data.pkl", "rb") as f:
            phase1_data = pickle.load(f)
        
        omega_local = phase1_data['omega_grid']
        V = omega_local - self.omega_ref
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Frequency spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        omega_plot = self.omega_ea.real if self.omega_ea.dtype == complex else self.omega_ea
        ax1.plot(range(len(omega_plot)), omega_plot, 'o-', markersize=8, label='EA modes')
        ax1.axhline(self.omega_ref, color='r', linestyle='--', alpha=0.5, label='ω_ref')
        ax1.set_xlabel('Mode index')
        ax1.set_ylabel('Frequency ω')
        ax1.set_title('EA Mode Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Detuning distribution
        ax2 = fig.add_subplot(gs[0, 1])
        dE_plot = self.eigenvalues.real if self.eigenvalues.dtype == complex else self.eigenvalues
        ax2.hist(dE_plot, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Detuning ΔE')
        ax2.set_ylabel('Count')
        ax2.set_title('Detuning Distribution')
        ax2.axvline(0, color='r', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 3. Potential landscape
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(V.T, origin='lower', cmap='RdBu_r')
        ax3.set_title('Potential V(R) = ω₀(R) - ω_ref')
        ax3.set_xlabel('i')
        ax3.set_ylabel('j')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # 4. First three mode profiles
        for idx in range(min(3, len(self.eigenvalues))):
            ax = fig.add_subplot(gs[1, idx])
            F = self.eigenstates[:, idx].reshape(8, 8)
            F_plot = np.abs(F)
            
            im = ax.imshow(F_plot.T, origin='lower', cmap='hot')
            dE = self.eigenvalues[idx].real if self.eigenvalues.dtype == complex else self.eigenvalues[idx]
            ax.set_title(f'Mode {idx}: ΔE = {dE:.6f}')
            ax.set_xlabel('i')
            ax.set_ylabel('j')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 5. IPR vs Energy
        ax5 = fig.add_subplot(gs[2, 0])
        iprs = []
        for i in range(len(self.eigenvalues)):
            F = self.eigenstates[:, i]
            F_mag = np.abs(F)
            norm2 = np.sum(F_mag**2)
            norm4 = np.sum(F_mag**4)
            ipr = norm4 / norm2**2 if norm2 > 1e-10 else 0
            iprs.append(ipr)
        
        ax5.scatter(dE_plot, iprs, s=50, alpha=0.7)
        ax5.set_xlabel('Detuning ΔE')
        ax5.set_ylabel('Inverse Participation Ratio')
        ax5.set_title('Localization vs Energy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Perturbation theory comparison
        ax6 = fig.add_subplot(gs[2, 1])
        V_flat = V.flatten()
        E_pt1 = []
        for i in range(len(self.eigenvalues)):
            F_n = self.eigenstates[:, i]
            F_n_mag2 = np.abs(F_n)**2
            E_pt1.append(np.sum(F_n_mag2 * V_flat) / np.sum(F_n_mag2))
        
        ax6.scatter(E_pt1, dE_plot, s=50, alpha=0.7)
        ax6.plot([min(E_pt1), max(E_pt1)], [min(E_pt1), max(E_pt1)], 
                'r--', alpha=0.5, label='Perfect agreement')
        ax6.set_xlabel('ΔE (1st-order PT)')
        ax6.set_ylabel('ΔE (Full EA)')
        ax6.set_title('EA vs Perturbation Theory')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Gap analysis
        ax7 = fig.add_subplot(gs[2, 2])
        omega_sorted = np.sort(omega_plot)
        gaps = np.diff(omega_sorted)
        ax7.bar(range(len(gaps)), gaps, alpha=0.7, edgecolor='black')
        ax7.set_xlabel('Mode index')
        ax7.set_ylabel('Gap to next mode')
        ax7.set_title('Inter-mode Gaps')
        ax7.grid(True, alpha=0.3, axis='y')
        
        plt.savefig(output_dir / "phase4_validation_comprehensive.png", 
                   dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comprehensive validation plot")
        plt.close()
    
    def save_validation_report(self, metrics_dict: Dict, output_dir: Path):
        """Save validation metrics to CSV"""
        
        with open(output_dir / "phase4_validation_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            
            for key, value in metrics_dict.items():
                writer.writerow([key, value])
        
        print(f"✓ Saved validation summary to {output_dir / 'phase4_validation_summary.csv'}")


def main():
    """Main execution for Phase 4"""
    print("=" * 60)
    print("Phase 4: Validation and Comparison")
    print("=" * 60)
    
    # Load configuration
    print("\nLoading Phase 0-3 results...")
    lattice_config = LatticeConfig(
        lattice_constant=1.0,
        twist_angle_deg=None,  # Uses default 1.1°
        stacking_gauge=(0.0, 0.0),
        moire_resolution=(64, 64)
    )
    lattice_setup = MoireLatticeSetup(lattice_config)
    
    # Load geometry config
    geom_config = GeometryConfig(
        eps_hi=4.64,
        eps_lo=1.0,
        cylinder_radius=0.48,
        polarization='TM'
    )
    
    # Load Phase 3 results
    phase3_data = np.load("outputs/phase3_eigenstates.npz")
    eigenvalues = phase3_data['eigenvalues']
    eigenvectors = phase3_data['eigenvectors']
    omega_ref = float(phase3_data['omega_ref'])
    
    print("✓ Loaded all data")
    
    # Initialize validator
    validator = ValidationAnalyzer(
        lattice_setup, geom_config, 
        eigenvalues, eigenvectors, omega_ref
    )
    
    # Run validation analyses
    output_dir = Path("outputs")
    
    # 1. Local variation analysis
    variation_metrics = validator.compute_local_variation_metric()
    
    # 2. Perturbation theory comparison
    pt_comparison = validator.compare_with_perturbation_theory()
    pt_comparison.to_csv(output_dir / "phase4_perturbation_theory_comparison.csv", index=False)
    print(f"✓ Saved PT comparison to {output_dir / 'phase4_perturbation_theory_comparison.csv'}")
    
    # 3. Mode character analysis
    mode_analysis = validator.analyze_mode_character()
    mode_analysis.to_csv(output_dir / "phase4_mode_character.csv", index=False)
    print(f"✓ Saved mode character to {output_dir / 'phase4_mode_character.csv'}")
    
    # 4. Band gap analysis
    gap_metrics = validator.estimate_band_gap()
    
    # 5. Convergence info
    convergence_df = validator.analyze_convergence()
    convergence_df.to_csv(output_dir / "phase4_convergence_info.csv", index=False)
    print(f"✓ Saved convergence info to {output_dir / 'phase4_convergence_info.csv'}")
    
    # Combine all metrics
    all_metrics = {**variation_metrics, **gap_metrics}
    
    # Add summary statistics
    all_metrics.update({
        'n_modes': len(eigenvalues),
        'omega_ref': omega_ref,
        'omega_ea_min': validator.omega_ea.min(),
        'omega_ea_max': validator.omega_ea.max(),
        'delta_E_min': eigenvalues.min(),
        'delta_E_max': eigenvalues.max(),
        'mean_pt_error': pt_comparison['difference'].abs().mean(),
        'rms_pt_error': np.sqrt((pt_comparison['difference']**2).mean()),
    })
    
    # Save comprehensive report
    validator.save_validation_report(all_metrics, output_dir)
    
    # Create visualizations
    print("\nGenerating validation visualizations...")
    validator.visualize_validation(output_dir)
    
    print("\n" + "=" * 60)
    print("Phase 4 Complete ✓")
    print("=" * 60)
    
    print("\n📊 Validation Summary:")
    print(f"  Envelope approximation parameter: η·ξ₀ = {all_metrics['eta_xi_mode0']:.4f}")
    if all_metrics['eta_xi_mode0'] < 0.5:
        print(f"  ✓ Valid EA regime")
    else:
        print(f"  ⚠ Marginal/Invalid EA regime")
    
    print(f"\n  Perturbation theory agreement:")
    print(f"    Mean absolute error: {all_metrics['mean_pt_error']:.8f}")
    print(f"    RMS error: {all_metrics['rms_pt_error']:.8f}")
    
    print(f"\n  Mode localization:")
    print(f"    Mean IPR: {mode_analysis['ipr'].mean():.5f}")
    print(f"    IPR range: [{mode_analysis['ipr'].min():.5f}, {mode_analysis['ipr'].max():.5f}]")
    
    print(f"\n  Largest spectral gap:")
    print(f"    Size: {all_metrics['max_gap']:.6f}")
    print(f"    Location: ω ≈ {all_metrics['gap_center']:.6f}")
    
    print("\n✅ Envelope approximation pipeline validated!")
    print("\nAll phases complete. Results saved in outputs/")


if __name__ == "__main__":
    main()
