"""
Phase 0: Geometry & Lattice Scaffolding

This script sets up the base lattice and moiré lattice geometry.
It provides the registry map δ(R) and generates visualization outputs.

Inputs:
- Monolayer lattice parameters (a1, a2)
- Twist angle θ
- Stacking gauge τ
- Moiré resolution (Nx, Ny)

Outputs:
- phase0_moire_vectors.csv: Moiré lattice vectors
- phase0_registry_map.csv: Registry map δ(R) on the moiré grid
- phase0_lattice_visualization.png: Visual representation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Import Rust bindings
import moire_lattice_py as ml


class LatticeConfig:
    """Configuration for the lattice setup (Square lattice only)"""
    
    # Optimization bounds for twist angle
    THETA_MIN_DEG = 0.5
    THETA_MAX_DEG = 2.0
    THETA_DEFAULT_DEG = 1.1
    
    def __init__(
        self,
        lattice_constant: float = 1.0,
        twist_angle_deg: float = None,
        stacking_gauge: tuple = (0.0, 0.0),
        moire_resolution: tuple = (64, 64),
    ):
        if twist_angle_deg is None:
            twist_angle_deg = self.THETA_DEFAULT_DEG
        
        # Validate twist angle
        if not (self.THETA_MIN_DEG <= twist_angle_deg <= self.THETA_MAX_DEG):
            raise ValueError(
                f"Twist angle {twist_angle_deg}° outside allowed range "
                f"[{self.THETA_MIN_DEG}°, {self.THETA_MAX_DEG}°]"
            )
        
        self.a = lattice_constant
        self.theta = np.radians(twist_angle_deg)
        self.tau = np.array(stacking_gauge)
        self.Nx, self.Ny = moire_resolution
        
    def get_monolayer_vectors(self):
        """Get monolayer lattice vectors for square lattice"""
        a1 = np.array([self.a, 0.0, 0.0])
        a2 = np.array([0.0, self.a, 0.0])
        return a1, a2


def rotation_matrix_2d(theta):
    """Create 2D rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fractional_coordinates(v, lattice_matrix):
    """
    Return fractional coordinates of vector v in lattice defined by lattice_matrix.
    
    Args:
        v: 2D vector
        lattice_matrix: 2x2 matrix where columns are lattice vectors
    
    Returns:
        Fractional coordinates wrapped to [0,1)^2
    """
    frac = np.linalg.solve(lattice_matrix, v)
    return frac - np.floor(frac)


def compute_registry_map(R_grid, a1, a2, theta, tau, eta):
    """
    Compute the local registry map δ(R) on the moiré grid.
    
    δ(R) = (R_θ - I) * R/η + τ (mod lattice)
    
    In the slow coordinate R = η*r, this gives the local stacking shift.
    
    Args:
        R_grid: Nx x Ny x 2 array of R coordinates on moiré cell
        a1, a2: Monolayer lattice vectors (3D)
        theta: Twist angle
        tau: Stacking gauge (2D)
        eta: Small parameter a/L
    
    Returns:
        delta_grid: Nx x Ny x 2 array of fractional shifts
    """
    Nx, Ny = R_grid.shape[0], R_grid.shape[1]
    delta_grid = np.zeros((Nx, Ny, 2))
    
    R_rot = rotation_matrix_2d(theta)
    I = np.eye(2)
    lattice_mat = np.column_stack([a1[:2], a2[:2]])
    
    for i in range(Nx):
        for j in range(Ny):
            R_vec = R_grid[i, j, :]
            # δ = (R_θ - I) * R/η + τ
            # Since R is already the slow coordinate, we use it directly
            # The physical coordinate r = R/η in the two-scale picture
            delta_physical = (R_rot - I) @ R_vec / eta + tau
            # Convert to fractional coordinates in monolayer cell
            delta_frac = fractional_coordinates(delta_physical, lattice_mat)
            delta_grid[i, j, :] = delta_frac
    
    return delta_grid


def create_moire_grid(A1, A2, Nx, Ny):
    """
    Create a grid of R points on the moiré unit cell.
    
    Args:
        A1, A2: Moiré lattice vectors (3D)
        Nx, Ny: Grid resolution
    
    Returns:
        R_grid: Nx x Ny x 2 array of coordinates
    """
    R_grid = np.zeros((Nx, Ny, 2))
    
    for i in range(Nx):
        for j in range(Ny):
            # Fractional coordinates in moiré cell
            s = i / Nx
            t = j / Ny
            # Real space position
            R = s * A1[:2] + t * A2[:2]
            R_grid[i, j, :] = R
    
    return R_grid


class MoireLatticeSetup:
    """Main class for Phase 0 setup"""
    
    def __init__(self, config: LatticeConfig):
        self.config = config
        self.a1, self.a2 = config.get_monolayer_vectors()
        
        # Create base lattice using Rust bindings
        self.base_lattice = ml.Lattice2D.from_basis_vectors(
            self.a1.tolist(),
            self.a2.tolist()
        )
        
        # Create moiré lattice using Rust
        self.moire_lattice = self._create_moire_lattice()
        
        # Get moiré vectors
        self.A1, self.A2 = self._get_moire_vectors()
        
        # Compute eta (small parameter)
        a_mono = self.config.a
        A_moire = np.linalg.norm(self.A1[:2])
        self.eta = a_mono / A_moire
        
        # Create moiré grid
        self.R_grid = create_moire_grid(self.A1, self.A2, config.Nx, config.Ny)
        
        # Compute registry map
        self.delta_grid = compute_registry_map(
            self.R_grid, self.a1, self.a2, 
            config.theta, config.tau, self.eta
        )
        
    def _create_moire_lattice(self):
        """Create moiré lattice using Rust bindings"""
        # Create transformation
        transformation = ml.MoireTransformation.twist(self.config.theta)
        
        # Create Moire2D - need to use the Rust constructor properly
        # Since from_transformation is an instance method with issues,
        # we'll manually construct the moiré lattice
        
        # Apply transformation to base vectors
        theta = self.config.theta
        R = rotation_matrix_2d(theta)
        
        # Transform the second lattice vectors
        a1_prime = R @ self.a1[:2]
        a2_prime = R @ self.a2[:2]
        
        # Create second lattice
        lattice2 = ml.Lattice2D.from_basis_vectors(
            np.append(a1_prime, 0.0).tolist(),
            np.append(a2_prime, 0.0).tolist()
        )
        
        # Get reciprocal vectors for moiré calculation
        # g_m = g' - g (following Rust implementation)
        g1 = np.array(self.base_lattice.reciprocal_basis().base_vectors()[0][:2])
        g2 = np.array(self.base_lattice.reciprocal_basis().base_vectors()[1][:2])  # Fixed: was [0]
        g1_prime = np.array(lattice2.reciprocal_basis().base_vectors()[0][:2])
        g2_prime = np.array(lattice2.reciprocal_basis().base_vectors()[1][:2])
        
        gm1 = g1_prime - g1
        gm2 = g2_prime - g2
        
        # Moiré direct vectors from reciprocal (following Rust: inverse then transpose times 2π)
        G_moire_reciprocal = np.column_stack([gm1, gm2])
        G_moire_direct = 2 * np.pi * np.linalg.inv(G_moire_reciprocal).T
        
        Am1 = np.append(G_moire_direct[:, 0], 0.0)
        Am2 = np.append(G_moire_direct[:, 1], 0.0)
        
        # Create moiré lattice object
        moire = ml.Lattice2D.from_basis_vectors(Am1.tolist(), Am2.tolist())
        
        return moire
    
    def _get_moire_vectors(self):
        """Extract moiré lattice vectors"""
        basis = self.moire_lattice.direct_basis()
        vecs = basis.base_vectors()
        A1 = np.array(vecs[0])
        A2 = np.array(vecs[1])
        return A1, A2
    
    def save_outputs(self, output_dir: Path):
        """Save all Phase 0 outputs"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save moiré vectors
        with open(output_dir / "phase0_moire_vectors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["vector", "x", "y", "z"])
            writer.writerow(["A1", self.A1[0], self.A1[1], self.A1[2]])
            writer.writerow(["A2", self.A2[0], self.A2[1], self.A2[2]])
            writer.writerow(["a1", self.a1[0], self.a1[1], self.a1[2]])
            writer.writerow(["a2", self.a2[0], self.a2[1], self.a2[2]])
        
        # Save parameters
        with open(output_dir / "phase0_parameters.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["lattice_constant", self.config.a])
            writer.writerow(["lattice_type", "square"])
            writer.writerow(["twist_angle_deg", np.degrees(self.config.theta)])
            writer.writerow(["twist_angle_rad", self.config.theta])
            writer.writerow(["eta", self.eta])
            writer.writerow(["moire_length", np.linalg.norm(self.A1[:2])])
            writer.writerow(["Nx", self.config.Nx])
            writer.writerow(["Ny", self.config.Ny])
        
        # Save registry map (sampled points for verification)
        with open(output_dir / "phase0_registry_map_sample.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "j", "Rx", "Ry", "delta_x", "delta_y"])
            # Save every 4th point to keep file size manageable
            step = max(1, self.config.Nx // 16)
            for i in range(0, self.config.Nx, step):
                for j in range(0, self.config.Ny, step):
                    writer.writerow([
                        i, j,
                        self.R_grid[i, j, 0], self.R_grid[i, j, 1],
                        self.delta_grid[i, j, 0], self.delta_grid[i, j, 1]
                    ])
        
        print(f"✓ Saved Phase 0 outputs to {output_dir}")
        print(f"  - Moiré length: {np.linalg.norm(self.A1[:2]):.4f}")
        print(f"  - η = a/L: {self.eta:.6f}")
        print(f"  - Grid resolution: {self.config.Nx} × {self.config.Ny}")
    
    def visualize(self, output_dir: Path):
        """Create comprehensive visualizations of the lattice setup"""
        
        # Figure 1: Overlapped lattices and moiré cell
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Two overlapped lattices
        ax1 = plt.subplot(131)
        self._plot_overlapped_lattices(ax1)
        
        # Plot 2: Moiré lattice
        ax2 = plt.subplot(132)
        self._plot_moire_lattice(ax2)
        
        # Plot 3: Stacking shift map
        ax3 = plt.subplot(133)
        self._plot_stacking_shift_map(ax3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase0_lattice_visualization.png", dpi=200)
        print(f"✓ Saved lattice visualization to {output_dir / 'phase0_lattice_visualization.png'}")
        plt.close()
        
        # Figure 2: Detailed stacking shift analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Stacking shift components and magnitude
        delta_x = self.delta_grid[:, :, 0]
        delta_y = self.delta_grid[:, :, 1]
        delta_mag = np.sqrt(delta_x**2 + delta_y**2)
        
        im1 = axes[0, 0].imshow(delta_x.T, origin='lower', cmap='RdBu_r', 
                                extent=[0, np.linalg.norm(self.A1[:2]), 
                                       0, np.linalg.norm(self.A2[:2])])
        axes[0, 0].set_title('Stacking Shift δₓ(R)')
        axes[0, 0].set_xlabel('R_x')
        axes[0, 0].set_ylabel('R_y')
        plt.colorbar(im1, ax=axes[0, 0], label='δₓ (fractional)')
        
        im2 = axes[0, 1].imshow(delta_y.T, origin='lower', cmap='RdBu_r',
                                extent=[0, np.linalg.norm(self.A1[:2]), 
                                       0, np.linalg.norm(self.A2[:2])])
        axes[0, 1].set_title('Stacking Shift δᵧ(R)')
        axes[0, 1].set_xlabel('R_x')
        axes[0, 1].set_ylabel('R_y')
        plt.colorbar(im2, ax=axes[0, 1], label='δᵧ (fractional)')
        
        im3 = axes[1, 0].imshow(delta_mag.T, origin='lower', cmap='viridis',
                                extent=[0, np.linalg.norm(self.A1[:2]), 
                                       0, np.linalg.norm(self.A2[:2])])
        axes[1, 0].set_title('Stacking Shift Magnitude |δ(R)|')
        axes[1, 0].set_xlabel('R_x')
        axes[1, 0].set_ylabel('R_y')
        plt.colorbar(im3, ax=axes[1, 0], label='|δ| (fractional)')
        
        # Vector field representation
        step = max(1, self.config.Nx // 16)
        X = self.R_grid[::step, ::step, 0]
        Y = self.R_grid[::step, ::step, 1]
        U = self.delta_grid[::step, ::step, 0]
        V = self.delta_grid[::step, ::step, 1]
        axes[1, 1].quiver(X, Y, U, V, delta_mag[::step, ::step], 
                         cmap='viridis', alpha=0.8)
        axes[1, 1].set_title('Stacking Shift Vector Field')
        axes[1, 1].set_xlabel('R_x')
        axes[1, 1].set_ylabel('R_y')
        axes[1, 1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase0_stacking_shift_detailed.png", dpi=200)
        print(f"✓ Saved stacking shift details to {output_dir / 'phase0_stacking_shift_detailed.png'}")
        plt.close()
    
    def _plot_overlapped_lattices(self, ax):
        """Plot the two overlapped square lattices"""
        # Generate lattice points for layer 1 (unrotated)
        n_cells = 3
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                pos = i * self.a1[:2] + j * self.a2[:2]
                ax.plot(pos[0], pos[1], 'o', color='blue', markersize=8, alpha=0.6)
        
        # Generate lattice points for layer 2 (rotated)
        theta = self.config.theta
        R = rotation_matrix_2d(theta)
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                pos = i * self.a1[:2] + j * self.a2[:2]
                pos_rot = R @ pos
                ax.plot(pos_rot[0], pos_rot[1], 's', color='red', markersize=8, alpha=0.6)
        
        # Draw lattice vectors for both layers
        ax.arrow(0, 0, self.a1[0], self.a1[1], head_width=0.15, head_length=0.15,
                fc='blue', ec='blue', linewidth=2, label='Layer 1', zorder=10)
        ax.arrow(0, 0, self.a2[0], self.a2[1], head_width=0.15, head_length=0.15,
                fc='blue', ec='blue', linewidth=2, zorder=10)
        
        a1_rot = R @ self.a1[:2]
        a2_rot = R @ self.a2[:2]
        ax.arrow(0, 0, a1_rot[0], a1_rot[1], head_width=0.15, head_length=0.15,
                fc='red', ec='red', linewidth=2, label='Layer 2 (rotated)', zorder=10)
        ax.arrow(0, 0, a2_rot[0], a2_rot[1], head_width=0.15, head_length=0.15,
                fc='red', ec='red', linewidth=2, zorder=10)
        
        ax.set_xlabel('x/a')
        ax.set_ylabel('y/a')
        ax.set_title(f'Overlapped Square Lattices (θ = {np.degrees(theta):.2f}°)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    
    def _plot_moire_lattice(self, ax):
        """Plot the moiré superlattice"""
        # Show the moiré unit cell
        moire_corners = np.array([
            [0, 0],
            self.A1[:2],
            self.A1[:2] + self.A2[:2],
            self.A2[:2],
            [0, 0]
        ])
        ax.plot(moire_corners[:, 0], moire_corners[:, 1], 'k-', linewidth=2, label='Moiré cell')
        
        # Draw moiré lattice vectors
        ax.arrow(0, 0, self.A1[0], self.A1[1], head_width=2, head_length=2,
                fc='purple', ec='purple', linewidth=3, label='A₁', zorder=10)
        ax.arrow(0, 0, self.A2[0], self.A2[1], head_width=2, head_length=2,
                fc='orange', ec='orange', linewidth=3, label='A₂', zorder=10)
        
        # Show a few monolayer lattice points for scale
        n_cells = int(np.linalg.norm(self.A1[:2]) / self.config.a) + 2
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                pos = i * self.a1[:2] + j * self.a2[:2]
                if (abs(pos[0]) <= 1.1 * np.linalg.norm(self.A1[:2]) and 
                    abs(pos[1]) <= 1.1 * np.linalg.norm(self.A2[:2])):
                    ax.plot(pos[0], pos[1], '.', color='lightgray', markersize=2, alpha=0.5)
        
        ax.set_xlabel('x/a')
        ax.set_ylabel('y/a')
        ax.set_title(f'Moiré Superlattice (L = {np.linalg.norm(self.A1[:2]):.1f}a)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    def _plot_stacking_shift_map(self, ax):
        """Plot the stacking shift map over the moiré cell"""
        delta_mag = np.sqrt(self.delta_grid[:, :, 0]**2 + self.delta_grid[:, :, 1]**2)
        
        im = ax.imshow(delta_mag.T, origin='lower', cmap='viridis',
                      extent=[0, np.linalg.norm(self.A1[:2]), 
                             0, np.linalg.norm(self.A2[:2])])
        
        # Overlay vector field
        step = max(1, self.config.Nx // 12)
        X = self.R_grid[::step, ::step, 0]
        Y = self.R_grid[::step, ::step, 1]
        U = self.delta_grid[::step, ::step, 0]
        V = self.delta_grid[::step, ::step, 1]
        ax.quiver(X, Y, U, V, color='white', alpha=0.6, scale=5)
        
        ax.set_xlabel('R_x/a')
        ax.set_ylabel('R_y/a')
        ax.set_title('Stacking Shift Map δ(R)')
        plt.colorbar(im, ax=ax, label='|δ| (fractional)')
        ax.set_aspect('equal')


def main():
    """Main execution for Phase 0"""
    print("=" * 60)
    print("Phase 0: Geometry & Lattice Scaffolding")
    print("=" * 60)
    
    # Configuration (using defaults: θ = 1.1°, constrained to [0.5°, 2.0°])
    config = LatticeConfig(
        lattice_constant=1.0,
        twist_angle_deg=None,  # Uses default 1.1°
        stacking_gauge=(0.0, 0.0),
        moire_resolution=(64, 64)
    )
    
    print(f"\nConfiguration:")
    print(f"  Lattice type: Square")
    print(f"  Lattice constant: {config.a}")
    print(f"  Twist angle: {np.degrees(config.theta):.2f}°")
    print(f"  Moiré grid: {config.Nx} × {config.Ny}")
    
    # Setup
    print("\nInitializing lattice setup...")
    setup = MoireLatticeSetup(config)
    
    print(f"\nResults:")
    print(f"  Base lattice type: {setup.base_lattice.direct_bravais().name()}")
    print(f"  Moiré lattice type: {setup.moire_lattice.direct_bravais().name()}")
    print(f"  Moiré vectors:")
    print(f"    A₁ = [{setup.A1[0]:.4f}, {setup.A1[1]:.4f}]")
    print(f"    A₂ = [{setup.A2[0]:.4f}, {setup.A2[1]:.4f}]")
    
    # Save outputs
    output_dir = Path(__file__).parent / "outputs"
    setup.save_outputs(output_dir)
    setup.visualize(output_dir)
    
    print("\n" + "=" * 60)
    print("Phase 0 Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
