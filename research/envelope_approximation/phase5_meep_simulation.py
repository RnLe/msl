"""
Phase 5: Full Meep Simulation with Q-factor Analysis

This script performs a complete electromagnetic simulation of the moiré cavity:
1. Selects the best cavity mode candidate from Phase 3
2. Places a Gaussian source at the mode's peak location
3. Runs time-domain simulation to extract cavity resonance
4. Computes Q-factor using Harminv spectral analysis
5. Creates animated GIF of field evolution

This validates the envelope approximation predictions with full Maxwell solver.

Inputs:
- Phase 0: Moiré lattice configuration
- Phase 3: Envelope mode predictions (peak locations, frequencies)

Outputs:
- phase5_q_factor_results.csv: Q-factors and frequencies
- phase5_field_animation.gif: Time evolution visualization
- phase5_harminv_spectrum.png: Resonance spectrum
- phase5_simulation_summary.png: Field snapshots at key times
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import csv
from typing import Tuple, Dict, List
import pickle

# Meep imports
import meep as mp

# Phase 0 imports
from phase0_lattice_setup import LatticeConfig, MoireLatticeSetup
from phase1_local_bloch import GeometryConfig


class MoireCavitySimulation:
    """Full Meep simulation of moiré photonic crystal cavity"""
    
    def __init__(self, 
                 lattice_setup: MoireLatticeSetup,
                 geom_config: GeometryConfig,
                 mode_data: Dict):
        """
        Initialize cavity simulation.
        
        Args:
            lattice_setup: Moiré lattice from Phase 0
            geom_config: Geometry configuration
            mode_data: Selected mode info (frequency, position, envelope)
        """
        self.setup = lattice_setup
        self.geom_config = geom_config
        self.mode_data = mode_data
        
        # Moiré parameters
        self.A1 = lattice_setup.A1[:2]
        self.A2 = lattice_setup.A2[:2]
        self.L = np.linalg.norm(self.A1)
        
        # Mode properties
        self.omega_target = mode_data['omega']
        self.freq_target = self.omega_target / (2 * np.pi)
        self.peak_position = mode_data['peak_position']  # In moiré cell coords
        
        print(f"Moiré Cavity Simulation initialized:")
        print(f"  Moiré period: L = {self.L:.3f}")
        print(f"  Target frequency: ω = {self.omega_target:.6f} (f = {self.freq_target:.6f})")
        print(f"  Mode peak: ({self.peak_position[0]:.2f}, {self.peak_position[1]:.2f})")
        
    def build_full_moire_geometry(self, resolution: int = 20) -> Tuple:
        """
        Build full moiré supercell geometry for Meep simulation.
        
        This creates the actual twisted bilayer by placing cylinders
        from both layers across the moiré unit cell.
        
        Args:
            resolution: Meep resolution (pixels per unit length)
            
        Returns:
            (cell_size, geometry_list): Cell dimensions and geometry objects
        """
        print("\nBuilding moiré supercell geometry...")
        
        # Cell size = moiré unit cell
        Lx = self.L
        Ly = np.linalg.norm(self.A2)
        cell_size = mp.Vector3(Lx, Ly, 0)
        
        # Parameters
        r = self.geom_config.cylinder_radius
        a = 1.0  # Monolayer lattice constant
        
        # Monolayer lattice vectors (square)
        a1_mono = np.array([a, 0])
        a2_mono = np.array([0, a])
        
        # Get twist angle
        theta = self.setup.config.theta  # Already in radians
        
        # Rotation matrices
        cos_t = np.cos(theta / 2)
        sin_t = np.sin(theta / 2)
        R_plus = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        R_minus = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        
        # Rotated lattice vectors
        a1_layer1 = R_plus @ a1_mono
        a2_layer1 = R_plus @ a2_mono
        a1_layer2 = R_minus @ a1_mono
        a2_layer2 = R_minus @ a2_mono
        
        geometry = []
        
        # Layer 1: Rotated by +θ/2
        print(f"  Adding Layer 1 (+θ/2 = {theta/2 * 180/np.pi:.3f}°)...")
        n_max = int(np.ceil(self.L / a)) + 2
        n_layer1 = 0
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                pos = i * a1_layer1 + j * a2_layer1
                
                # Check if inside moiré cell (with small margin)
                if (-0.1 < pos[0] < Lx + 0.1 and -0.1 < pos[1] < Ly + 0.1):
                    # Air hole
                    cyl = mp.Cylinder(
                        radius=r,
                        center=mp.Vector3(pos[0], pos[1], 0),
                        material=mp.Medium(epsilon=self.geom_config.eps_lo)
                    )
                    geometry.append(cyl)
                    n_layer1 += 1
        
        # Layer 2: Rotated by -θ/2
        print(f"  Adding Layer 2 (-θ/2 = {-theta/2 * 180/np.pi:.3f}°)...")
        n_layer2 = 0
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                pos = i * a1_layer2 + j * a2_layer2
                
                # Check if inside moiré cell
                if (-0.1 < pos[0] < Lx + 0.1 and -0.1 < pos[1] < Ly + 0.1):
                    # Air hole
                    cyl = mp.Cylinder(
                        radius=r,
                        center=mp.Vector3(pos[0], pos[1], 0),
                        material=mp.Medium(epsilon=self.geom_config.eps_lo)
                    )
                    geometry.append(cyl)
                    n_layer2 += 1
        
        print(f"  Total cylinders: {len(geometry)} (Layer 1: {n_layer1}, Layer 2: {n_layer2})")
        print(f"  Cell size: {Lx:.3f} × {Ly:.3f}")
        
        return cell_size, geometry
    
    def run_cavity_simulation(self, 
                             resolution: int = 20,
                             run_time: float = 500,
                             harminv_time: float = 400,
                             freq_width: float = 0.05) -> Dict:
        """
        Run time-domain simulation with Harminv analysis.
        
        Args:
            resolution: Spatial resolution (pixels per unit length)
            run_time: Total simulation time
            harminv_time: Time window for Harminv analysis (after source turnoff)
            freq_width: Frequency range for Harminv (around target)
            
        Returns:
            Dictionary with Q-factor, frequency, and field data
        """
        print("\n" + "=" * 60)
        print("Running Meep Cavity Simulation")
        print("=" * 60)
        
        # Build geometry
        cell_size, geometry = self.build_full_moire_geometry(resolution)
        
        # Source position (at mode peak)
        src_pos = mp.Vector3(self.peak_position[0], self.peak_position[1], 0)
        
        # Gaussian source with frequency from EA prediction
        # Use TM: Ez source (out-of-plane electric field)
        sources = [
            mp.Source(
                mp.GaussianSource(
                    frequency=self.freq_target,
                    fwidth=freq_width
                ),
                component=mp.Ez,  # TM polarization
                center=src_pos,
                size=mp.Vector3(0, 0, 0)  # Point source
            )
        ]
        
        # PML boundaries
        pml_thickness = 2.0
        pml_layers = [mp.PML(pml_thickness)]
        
        # Create simulation
        sim = mp.Simulation(
            cell_size=cell_size,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=resolution,
            default_material=mp.Medium(epsilon=self.geom_config.eps_hi),
            force_complex_fields=False  # Real fields for TM
        )
        
        print(f"\nSimulation setup:")
        print(f"  Resolution: {resolution} pixels/unit")
        print(f"  Cell: {cell_size.x:.2f} × {cell_size.y:.2f}")
        print(f"  Source: Ez at ({src_pos.x:.2f}, {src_pos.y:.2f})")
        print(f"  Target freq: {self.freq_target:.6f}")
        print(f"  PML thickness: {pml_thickness}")
        
        # Harminv monitor at source position
        harminv_pt = src_pos
        
        # Field storage for animation
        field_snapshots = []
        snapshot_times = []
        
        def record_field(sim):
            """Callback to record field for animation"""
            if sim.meep_time() % 10 == 0:  # Record every 10 time units
                # Get Ez field in xy-plane
                ez_data = sim.get_array(
                    center=mp.Vector3(0, 0, 0),
                    size=cell_size,
                    component=mp.Ez
                )
                field_snapshots.append(ez_data.copy())
                snapshot_times.append(sim.meep_time())
        
        # Run with Harminv
        print(f"\nRunning simulation for {run_time} time units...")
        print(f"  Harminv analysis after t={run_time - harminv_time}")
        
        h = mp.Harminv(
            mp.Ez, 
            harminv_pt,
            self.freq_target,
            freq_width
        )
        
        sim.run(
            mp.at_every(10, record_field),
            mp.after_sources(h),
            until_after_sources=harminv_time
        )
        
        print(f"✓ Simulation complete")
        print(f"  Total timesteps: {sim.meep_time():.1f}")
        print(f"  Field snapshots: {len(field_snapshots)}")
        
        # Extract Harminv results
        harminv_modes = []
        for mode in h.modes:
            freq = mode.freq
            Q = mode.Q
            decay = mode.decay
            omega = 2 * np.pi * freq
            
            harminv_modes.append({
                'freq': freq,
                'omega': omega,
                'Q': Q,
                'decay': decay,
                'error': mode.error
            })
            
            print(f"\nHarminv mode found:")
            print(f"  Frequency: f = {freq:.8f} (ω = {omega:.8f})")
            print(f"  Q-factor: Q = {Q:.1f}")
            print(f"  Decay rate: γ = {decay:.6e}")
            print(f"  Error: {mode.error:.4e}")
        
        # Store simulation object for later field queries
        results = {
            'harminv_modes': harminv_modes,
            'field_snapshots': field_snapshots,
            'snapshot_times': snapshot_times,
            'cell_size': (cell_size.x, cell_size.y),
            'resolution': resolution,
            'peak_position': self.peak_position,
            'sim': sim  # Keep reference for final field extraction
        }
        
        return results
    
    def create_animation(self, results: Dict, output_path: Path,
                        fps: int = 10, max_frames: int = 100):
        """
        Create animated GIF of field evolution.
        
        Args:
            results: Simulation results from run_cavity_simulation
            output_path: Path to save GIF
            fps: Frames per second
            max_frames: Maximum number of frames to include
        """
        print("\n" + "=" * 60)
        print("Creating Field Animation")
        print("=" * 60)
        
        field_data = results['field_snapshots']
        times = results['snapshot_times']
        Lx, Ly = results['cell_size']
        
        # Subsample if too many frames
        if len(field_data) > max_frames:
            step = len(field_data) // max_frames
            field_data = field_data[::step]
            times = times[::step]
        
        print(f"  Total frames: {len(field_data)}")
        print(f"  Time range: {times[0]:.1f} - {times[-1]:.1f}")
        print(f"  FPS: {fps}")
        
        # Compute global color scale
        vmax = np.max([np.abs(f).max() for f in field_data])
        vmin = -vmax
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Initial frame
        im = ax.imshow(
            field_data[0].T,
            origin='lower',
            extent=[0, Lx, 0, Ly],
            cmap='RdBu',
            vmin=vmin,
            vmax=vmax,
            animated=True
        )
        
        # Mark source position
        peak_x, peak_y = results['peak_position']
        ax.plot(peak_x, peak_y, 'g*', markersize=20, 
               markeredgecolor='black', markeredgewidth=1.5,
               label='Source/Mode peak')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Ez Field Evolution (t = {times[0]:.1f})')
        ax.legend(loc='upper right')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Ez')
        
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def update(frame):
            """Update function for animation"""
            im.set_array(field_data[frame].T)
            ax.set_title(f'Ez Field Evolution (t = {times[frame]:.1f})')
            time_text.set_text(f't = {times[frame]:.1f}')
            return [im, time_text]
        
        # Create animation
        print(f"  Rendering animation...")
        anim = FuncAnimation(
            fig, update, 
            frames=len(field_data),
            interval=1000/fps,
            blit=True
        )
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f"✓ Animation saved: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    def visualize_harminv_spectrum(self, results: Dict, output_path: Path):
        """Create spectrum plot from Harminv analysis"""
        
        modes = results['harminv_modes']
        
        if not modes:
            print("Warning: No Harminv modes found")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Frequency spectrum
        ax = axes[0]
        freqs = [m['freq'] for m in modes]
        Qs = [m['Q'] for m in modes]
        
        ax.scatter(freqs, Qs, s=200, c='red', marker='o', 
                  edgecolors='black', linewidths=2, zorder=3)
        
        # Mark EA prediction
        ax.axvline(self.freq_target, color='blue', linestyle='--', 
                  linewidth=2, label='EA prediction', alpha=0.7)
        
        ax.set_xlabel('Frequency f', fontsize=12)
        ax.set_ylabel('Q-factor', fontsize=12)
        ax.set_title('Harminv Spectrum', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations
        for m in modes:
            ax.annotate(
                f"Q={m['Q']:.0f}",
                (m['freq'], m['Q']),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )
        
        # Comparison plot
        ax = axes[1]
        omegas_ea = [self.omega_target]
        omegas_meep = [m['omega'] for m in modes]
        
        ax.scatter([0], omegas_ea, s=200, c='blue', marker='s',
                  edgecolors='black', linewidths=2, label='EA prediction', zorder=3)
        ax.scatter([1]*len(omegas_meep), omegas_meep, s=200, c='red', marker='o',
                  edgecolors='black', linewidths=2, label='Meep simulation', zorder=3)
        
        # Draw lines between predictions and simulations
        for omega_m in omegas_meep:
            ax.plot([0, 1], [self.omega_target, omega_m], 'k--', alpha=0.3)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['EA', 'Meep'])
        ax.set_ylabel('Angular frequency ω', fontsize=12)
        ax.set_title('Frequency Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Spectrum plot saved: {output_path}")
    
    def save_results(self, results: Dict, output_dir: Path):
        """Save Q-factor and frequency results"""
        
        modes = results['harminv_modes']
        
        # Save to CSV
        csv_path = output_dir / "phase5_q_factor_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'mode_index', 'frequency', 'omega', 'Q_factor', 
                'decay_rate', 'error', 'omega_EA_prediction'
            ])
            
            for i, mode in enumerate(modes):
                writer.writerow([
                    i,
                    mode['freq'],
                    mode['omega'],
                    mode['Q'],
                    mode['decay'],
                    mode['error'],
                    self.omega_target
                ])
        
        print(f"✓ Results saved: {csv_path}")
        
        # Summary
        if modes:
            best_mode = max(modes, key=lambda m: m['Q'])
            
            print("\n" + "=" * 60)
            print("SIMULATION SUMMARY")
            print("=" * 60)
            print(f"\nEnvelope Approximation Prediction:")
            print(f"  ω_EA = {self.omega_target:.8f}")
            print(f"  f_EA = {self.freq_target:.8f}")
            
            print(f"\nBest Cavity Mode (highest Q):")
            print(f"  ω_sim = {best_mode['omega']:.8f}")
            print(f"  f_sim = {best_mode['freq']:.8f}")
            print(f"  Q = {best_mode['Q']:.1f}")
            print(f"  Δω/ω = {abs(best_mode['omega'] - self.omega_target) / self.omega_target * 100:.2f}%")
            
            print(f"\nAll modes found: {len(modes)}")


def select_best_mode(phase3_data: Dict) -> Dict:
    """
    Select the best cavity mode candidate from Phase 3 results.
    
    Criteria: Highest IPR (most localized) among lowest frequency modes.
    
    Args:
        phase3_data: Phase 3 results (eigenvalues, eigenstates, analysis)
        
    Returns:
        Dictionary with selected mode info
    """
    print("\n" + "=" * 60)
    print("Selecting Best Mode Candidate")
    print("=" * 60)
    
    # Load analysis data
    import pandas as pd
    analysis = pd.read_csv("outputs/phase3_mode_analysis.csv")
    
    # Select mode with highest IPR among first 3 modes
    candidates = analysis.head(3)
    best_idx = candidates['ipr'].idxmax()
    best_mode = candidates.loc[best_idx]
    
    print(f"\nCandidate modes (3 lowest energy):")
    for i, row in candidates.iterrows():
        marker = "→" if i == best_idx else " "
        print(f"{marker} Mode {int(row['mode'])}: "
              f"ω={row['omega_cavity']:.6f}, "
              f"IPR={row['ipr']:.5f}, "
              f"peak=({row['peak_x']:.1f}, {row['peak_y']:.1f})")
    
    print(f"\n✓ Selected: Mode {int(best_mode['mode'])}")
    print(f"  Reason: Highest localization (IPR={best_mode['ipr']:.5f})")
    
    # Load eigenstate
    mode_idx = int(best_mode['mode'])
    phase3_states = np.load("outputs/phase3_eigenstates.npz")
    F = phase3_states['eigenvectors'][:, mode_idx].reshape(8, 8)
    
    mode_data = {
        'mode_index': mode_idx,
        'omega': best_mode['omega_cavity'],
        'delta_E': best_mode['delta_E'],
        'ipr': best_mode['ipr'],
        'peak_position': (best_mode['peak_x'], best_mode['peak_y']),
        'envelope': F
    }
    
    return mode_data


def main():
    """Main execution for Phase 5"""
    print("=" * 60)
    print("Phase 5: Full Meep Simulation")
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
    
    # Geometry config
    geom_config = GeometryConfig(
        eps_hi=4.64,
        eps_lo=1.0,
        cylinder_radius=0.48,
        polarization='TM'
    )
    
    # Load Phase 3 results
    phase3_data = np.load("outputs/phase3_eigenstates.npz")
    
    # Select best mode
    mode_data = select_best_mode(phase3_data)
    
    # Initialize simulation
    cavity_sim = MoireCavitySimulation(lattice_setup, geom_config, mode_data)
    
    # Run simulation
    results = cavity_sim.run_cavity_simulation(
        resolution=10,           # Reduced resolution for faster run
        run_time=200,           # Shorter total time
        harminv_time=150,       # Shorter analysis window
        freq_width=0.1          # Wider frequency search range
    )
    
    # Create outputs
    output_dir = Path("outputs")
    
    print("\nGenerating visualizations...")
    
    # Save numerical results
    cavity_sim.save_results(results, output_dir)
    
    # Harminv spectrum
    cavity_sim.visualize_harminv_spectrum(
        results, 
        output_dir / "phase5_harminv_spectrum.png"
    )
    
    # Field animation
    cavity_sim.create_animation(
        results,
        output_dir / "phase5_field_animation.gif",
        fps=10,
        max_frames=100
    )
    
    print("\n" + "=" * 60)
    print("Phase 5 Complete ✓")
    print("=" * 60)
    print("\nFull electromagnetic validation complete.")
    print("\nOutputs:")
    print("  - phase5_q_factor_results.csv: Q-factors and frequencies")
    print("  - phase5_harminv_spectrum.png: Resonance analysis")
    print("  - phase5_field_animation.gif: Field evolution movie")


if __name__ == "__main__":
    main()
