"""
Phase 3: Envelope Eigenvalue Solver

This script solves the envelope approximation eigenvalue problem:
    H_EA · F(R) = ΔE · F(R)

where H_EA is the operator from Phase 2, F(R) are the envelope wavefunctions,
and ΔE are the energy detunings from the reference frequency ω_ref.

The physical cavity frequencies are:
    ω_cavity = ω_ref + ΔE

Inputs:
- Phase 2 outputs: Hamiltonian operator (phase2_operator.npz)
- Phase 0/1 data: Grid information and reference frequency

Outputs:
- phase3_eigenvalues.csv: Eigenvalues (detunings ΔE)
- phase3_eigenstates.npz: Envelope wavefunctions F_n(R)
- phase3_cavity_modes.png: Visualization of lowest modes
- phase3_spectrum.png: Energy spectrum
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from scipy import sparse
from scipy.sparse.linalg import eigsh, eigs
import pickle

# Phase 0 imports
from phase0_lattice_setup import LatticeConfig, MoireLatticeSetup
from phase1_local_bloch import GeometryConfig


class EnvelopeSolver:
    """Solve for envelope eigenstates"""
    
    def __init__(self, operator_file: Path, lattice_setup: MoireLatticeSetup, 
                 omega_ref: float):
        """
        Initialize solver.
        
        Args:
            operator_file: Path to phase2_operator.npz
            lattice_setup: MoireLatticeSetup from Phase 0
            omega_ref: Reference frequency from Phase 1
        """
        # Load operator
        self.H = sparse.load_npz(operator_file)
        self.Nx = int(np.sqrt(self.H.shape[0]))
        self.Ny = self.Nx
        self.N_total = self.H.shape[0]
        
        self.setup = lattice_setup
        self.omega_ref = omega_ref
        
        # Check if operator is Hermitian
        self.is_hermitian = self._check_hermiticity()
        
        print(f"Solver initialized:")
        print(f"  Grid: {self.Nx} × {self.Ny} = {self.N_total} points")
        print(f"  Operator size: {self.H.shape[0]} × {self.H.shape[1]}")
        print(f"  Non-zeros: {self.H.nnz}")
        print(f"  Hermitian: {self.is_hermitian}")
        print(f"  ω_ref = {self.omega_ref:.8f}")
        
    def _check_hermiticity(self, tol: float = 1e-10) -> bool:
        """Check if operator is Hermitian"""
        # Sample a few elements
        diff = self.H - self.H.conj().T
        return np.max(np.abs(diff.data)) < tol if diff.nnz > 0 else True
    
    def solve_lowest_modes(self, n_modes: int = 10, sigma: float = None,
                          use_shift_invert: bool = True) -> tuple:
        """
        Solve for the lowest energy envelope modes.
        
        Args:
            n_modes: Number of modes to compute
            sigma: Shift for shift-invert (if None, use middle of spectrum)
            use_shift_invert: Use shift-invert for interior eigenvalues
            
        Returns:
            (eigenvalues, eigenvectors): ΔE and F(R) for each mode
        """
        print("\n" + "=" * 60)
        print("Solving Eigenvalue Problem")
        print("=" * 60)
        
        if n_modes >= self.N_total - 2:
            n_modes = self.N_total - 2
            print(f"Warning: Reduced n_modes to {n_modes} (max for this system)")
        
        print(f"\nComputing {n_modes} lowest energy modes...")
        
        if self.is_hermitian:
            # Use eigsh for Hermitian matrices (faster, more accurate)
            if use_shift_invert:
                if sigma is None:
                    # Estimate spectrum center from diagonal
                    diag = self.H.diagonal()
                    sigma = np.median(diag)
                    print(f"  Using shift σ = {sigma:.6e} (median of diagonal)")
                else:
                    print(f"  Using shift σ = {sigma:.6e}")
                
                try:
                    eigenvalues, eigenvectors = eigsh(
                        self.H, k=n_modes, sigma=sigma,
                        which='LM',  # Largest magnitude near sigma
                        tol=1e-6,
                        maxiter=1000
                    )
                except Exception as e:
                    print(f"  Shift-invert failed: {e}")
                    print(f"  Falling back to standard solver...")
                    eigenvalues, eigenvectors = eigsh(
                        self.H, k=n_modes, which='SA'  # Smallest algebraic
                    )
            else:
                eigenvalues, eigenvectors = eigsh(
                    self.H, k=n_modes, which='SA'  # Smallest algebraic
                )
        else:
            # Use eigs for non-Hermitian matrices (has drift term)
            print("  Using non-Hermitian solver (operator has drift term)")
            if sigma is None:
                diag = self.H.diagonal()
                sigma = np.median(diag.real)
                print(f"  Using shift σ = {sigma:.6e}")
            
            eigenvalues, eigenvectors = eigs(
                self.H, k=n_modes, sigma=sigma,
                which='LM',
                tol=1e-6,
                maxiter=1000
            )
            
            # eigs returns complex eigenvalues, take real part if imaginary is small
            if np.max(np.abs(eigenvalues.imag)) < 1e-8:
                eigenvalues = eigenvalues.real
            else:
                print(f"  Warning: Eigenvalues have significant imaginary part")
                print(f"    Max |Im(E)|: {np.max(np.abs(eigenvalues.imag)):.6e}")
        
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues.real if eigenvalues.dtype == complex else eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"\n✓ Converged!")
        print(f"\nLowest {min(5, n_modes)} eigenvalues (ΔE):")
        for i in range(min(5, n_modes)):
            if eigenvalues.dtype == complex:
                print(f"  Mode {i}: ΔE = {eigenvalues[i].real:.8f} + {eigenvalues[i].imag:.2e}j")
            else:
                print(f"  Mode {i}: ΔE = {eigenvalues[i]:.8f}")
        
        if n_modes > 5:
            print(f"  ...")
            if eigenvalues.dtype == complex:
                print(f"  Mode {n_modes-1}: ΔE = {eigenvalues[-1].real:.8f} + {eigenvalues[-1].imag:.2e}j")
            else:
                print(f"  Mode {n_modes-1}: ΔE = {eigenvalues[-1]:.8f}")
        
        print("\n" + "=" * 60)
        
        return eigenvalues, eigenvectors
    
    def compute_cavity_frequencies(self, eigenvalues: np.ndarray) -> np.ndarray:
        """
        Convert eigenvalues (detunings) to cavity frequencies.
        
        Args:
            eigenvalues: Detunings ΔE from eigenvalue problem
            
        Returns:
            Cavity frequencies ω_cavity = ω_ref + ΔE
        """
        omega_cavity = self.omega_ref + eigenvalues
        return omega_cavity
    
    def analyze_mode(self, eigenvector: np.ndarray) -> dict:
        """
        Analyze properties of a single mode.
        
        Args:
            eigenvector: Envelope wavefunction F(R) (flattened)
            
        Returns:
            Dictionary with mode properties
        """
        F = eigenvector.reshape(self.Nx, self.Ny)
        
        # Take absolute value (or magnitude if complex)
        if F.dtype == complex:
            F_mag = np.abs(F)
        else:
            F_mag = np.abs(F)
        
        # Localization: inverse participation ratio
        norm2 = np.sum(F_mag**2)
        norm4 = np.sum(F_mag**4)
        ipr = norm4 / norm2**2 if norm2 > 1e-10 else 0
        
        # Peak location
        peak_idx = np.unravel_index(np.argmax(F_mag), F.shape)
        peak_i, peak_j = peak_idx
        
        # Convert to real-space position
        Lx = np.linalg.norm(self.setup.A1[:2])
        Ly = np.linalg.norm(self.setup.A2[:2])
        peak_x = peak_i / self.Nx * Lx
        peak_y = peak_j / self.Ny * Ly
        
        # Effective radius (second moment)
        i_coords, j_coords = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny), indexing='ij')
        x_coords = i_coords / self.Nx * Lx
        y_coords = j_coords / self.Ny * Ly
        
        x_mean = np.sum(F_mag**2 * x_coords) / norm2 if norm2 > 1e-10 else 0
        y_mean = np.sum(F_mag**2 * y_coords) / norm2 if norm2 > 1e-10 else 0
        
        r2 = (x_coords - x_mean)**2 + (y_coords - y_mean)**2
        r_eff = np.sqrt(np.sum(F_mag**2 * r2) / norm2) if norm2 > 1e-10 else 0
        
        return {
            'ipr': ipr,
            'peak_i': peak_i,
            'peak_j': peak_j,
            'peak_x': peak_x,
            'peak_y': peak_y,
            'center_x': x_mean,
            'center_y': y_mean,
            'r_eff': r_eff,
            'max_amplitude': np.max(F_mag)
        }
    
    def save_results(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                    output_dir: Path):
        """Save eigenvalues and eigenstates"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute cavity frequencies
        omega_cavity = self.compute_cavity_frequencies(eigenvalues)
        
        # Save eigenvalues
        with open(output_dir / "phase3_eigenvalues.csv", "w", newline="") as f:
            writer = csv.writer(f)
            if eigenvalues.dtype == complex:
                writer.writerow(["mode", "delta_E_real", "delta_E_imag", "omega_cavity"])
                for i, (dE, omega) in enumerate(zip(eigenvalues, omega_cavity)):
                    writer.writerow([i, dE.real, dE.imag, omega.real])
            else:
                writer.writerow(["mode", "delta_E", "omega_cavity"])
                for i, (dE, omega) in enumerate(zip(eigenvalues, omega_cavity)):
                    writer.writerow([i, dE, omega])
        
        print(f"\n✓ Saved eigenvalues to {output_dir / 'phase3_eigenvalues.csv'}")
        
        # Save eigenvectors
        np.savez_compressed(
            output_dir / "phase3_eigenstates.npz",
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            omega_ref=self.omega_ref,
            grid_shape=(self.Nx, self.Ny)
        )
        print(f"✓ Saved eigenstates to {output_dir / 'phase3_eigenstates.npz'}")
        
        # Analyze and save mode properties
        with open(output_dir / "phase3_mode_analysis.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "mode", "delta_E", "omega_cavity", "ipr", 
                "peak_i", "peak_j", "peak_x", "peak_y",
                "center_x", "center_y", "r_eff", "max_amplitude"
            ])
            
            for i in range(len(eigenvalues)):
                props = self.analyze_mode(eigenvectors[:, i])
                dE = eigenvalues[i].real if eigenvalues.dtype == complex else eigenvalues[i]
                omega = omega_cavity[i].real if omega_cavity.dtype == complex else omega_cavity[i]
                
                writer.writerow([
                    i, dE, omega, props['ipr'],
                    props['peak_i'], props['peak_j'],
                    props['peak_x'], props['peak_y'],
                    props['center_x'], props['center_y'],
                    props['r_eff'], props['max_amplitude']
                ])
        
        print(f"✓ Saved mode analysis to {output_dir / 'phase3_mode_analysis.csv'}")
    
    def visualize_modes(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                       n_display: int = 6, output_dir: Path = None):
        """Visualize the lowest energy envelope modes"""
        n_modes = min(n_display, len(eigenvalues))
        
        # Determine layout
        n_cols = min(3, n_modes)
        n_rows = (n_modes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_modes == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        omega_cavity = self.compute_cavity_frequencies(eigenvalues)
        
        for i in range(n_modes):
            F = eigenvectors[:, i].reshape(self.Nx, self.Ny)
            
            # Take magnitude if complex
            if F.dtype == complex:
                F_plot = np.abs(F)
                title_suffix = " |F|"
            else:
                F_plot = F
                title_suffix = ""
            
            im = axes[i].imshow(F_plot.T, origin='lower', cmap='hot')
            
            dE = eigenvalues[i].real if eigenvalues.dtype == complex else eigenvalues[i]
            omega = omega_cavity[i].real if omega_cavity.dtype == complex else omega_cavity[i]
            
            axes[i].set_title(
                f"Mode {i}{title_suffix}\n"
                f"ΔE = {dE:.6f}, ω = {omega:.6f}",
                fontsize=10
            )
            axes[i].set_xlabel('i')
            axes[i].set_ylabel('j')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Mark peak
            props = self.analyze_mode(eigenvectors[:, i])
            axes[i].plot(props['peak_i'], props['peak_j'], 'g*', 
                        markersize=15, markeredgecolor='white', markeredgewidth=1)
        
        # Hide unused subplots
        for i in range(n_modes, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / "phase3_cavity_modes.png", dpi=150, bbox_inches='tight')
            print(f"✓ Saved mode visualization to {output_dir / 'phase3_cavity_modes.png'}")
        
        plt.close()
    
    def visualize_spectrum(self, eigenvalues: np.ndarray, output_dir: Path = None):
        """Visualize the energy spectrum"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        omega_cavity = self.compute_cavity_frequencies(eigenvalues)
        
        # Eigenvalue spectrum (detunings)
        ax = axes[0]
        dE_plot = eigenvalues.real if eigenvalues.dtype == complex else eigenvalues
        ax.plot(range(len(eigenvalues)), dE_plot, 'o-', markersize=8)
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Detuning ΔE')
        ax.set_title('Eigenvalue Spectrum (Detunings from ω_ref)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5, label='ΔE = 0')
        ax.legend()
        
        # Cavity frequency spectrum
        ax = axes[1]
        omega_plot = omega_cavity.real if omega_cavity.dtype == complex else omega_cavity
        ax.plot(range(len(omega_cavity)), omega_plot, 's-', markersize=8, color='orange')
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Frequency ω')
        ax.set_title('Cavity Mode Frequencies')
        ax.grid(True, alpha=0.3)
        ax.axhline(self.omega_ref, color='r', linestyle='--', alpha=0.5, 
                  label=f'ω_ref = {self.omega_ref:.6f}')
        ax.legend()
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / "phase3_spectrum.png", dpi=150, bbox_inches='tight')
            print(f"✓ Saved spectrum visualization to {output_dir / 'phase3_spectrum.png'}")
        
        plt.close()


def main():
    """Main execution for Phase 3"""
    print("=" * 60)
    print("Phase 3: Envelope Eigenvalue Solver")
    print("=" * 60)
    
    # Load Phase 0 configuration
    print("\nLoading configuration...")
    lattice_config = LatticeConfig(
        lattice_constant=1.0,
        twist_angle_deg=None,  # Uses default 1.1°
        stacking_gauge=(0.0, 0.0),
        moire_resolution=(64, 64)
    )
    
    lattice_setup = MoireLatticeSetup(lattice_config)
    
    # Load operator info to get omega_ref
    import pandas as pd
    op_info = pd.read_csv("outputs/phase2_operator_info.csv")
    omega_ref = float(op_info[op_info['parameter'] == 'omega_ref']['value'].values[0])
    
    print("✓ Loaded configuration")
    
    # Initialize solver
    operator_file = Path("outputs/phase2_operator.npz")
    solver = EnvelopeSolver(operator_file, lattice_setup, omega_ref)
    
    # Solve for modes
    n_modes = 10  # Number of modes to compute
    eigenvalues, eigenvectors = solver.solve_lowest_modes(
        n_modes=n_modes,
        sigma=None,  # Auto-detect
        use_shift_invert=True
    )
    
    # Print summary
    print("\nCavity Mode Summary:")
    print("-" * 60)
    omega_cavity = solver.compute_cavity_frequencies(eigenvalues)
    for i in range(min(5, len(eigenvalues))):
        dE = eigenvalues[i].real if eigenvalues.dtype == complex else eigenvalues[i]
        omega = omega_cavity[i].real if omega_cavity.dtype == complex else omega_cavity[i]
        print(f"  Mode {i}: ω = {omega:.8f} (ΔE = {dE:+.8f})")
    
    # Save results
    output_dir = Path("outputs")
    solver.save_results(eigenvalues, eigenvectors, output_dir)
    
    # Visualize
    print("\nGenerating visualizations...")
    solver.visualize_modes(eigenvalues, eigenvectors, n_display=6, output_dir=output_dir)
    solver.visualize_spectrum(eigenvalues, output_dir=output_dir)
    
    print("\n" + "=" * 60)
    print("Phase 3 Complete ✓")
    print("=" * 60)
    print("\nEnvelope approximation pipeline complete!")
    print("\nKey results:")
    print(f"  - Found {len(eigenvalues)} cavity modes")
    print(f"  - Frequency range: [{omega_cavity.min():.6f}, {omega_cavity.max():.6f}]")
    print(f"  - Detuning range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}]")
    print("\nNext steps:")
    print("  - Review mode profiles in phase3_cavity_modes.png")
    print("  - Check localization in phase3_mode_analysis.csv")
    print("  - Validate with full Meep simulation (Phase 4)")


if __name__ == "__main__":
    main()
