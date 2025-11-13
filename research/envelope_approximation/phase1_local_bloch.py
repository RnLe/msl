"""
Phase 1: Local Bloch Problems at Frozen Registry

This script computes band structure data at frozen registry points δ.
For each point on the δ-grid (moiré grid), it runs MPB to compute:
- ω₀(R) - band edge frequency at k₀
- M⁻¹(R) - inverse effective mass tensor
- v_g(R) - group velocity (should vanish at band extremum)

Inputs:
- Phase 0 outputs (moiré vectors, registry map)
- Geometry parameters (cylinder radius, dielectric constants)
- Target band index and high-symmetry point k₀

Outputs:
- phase1_band_data.csv: ω₀, M⁻¹ components, v_g for each grid point
- phase1_band_visualization.png: Spatial maps of band parameters
- phase1_reference_band.csv: Reference band structure at one registry
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from dataclasses import dataclass
from typing import Tuple, List
import pickle

# MPB imports
import meep as mp
from meep import mpb

# Phase 0 imports
from phase0_lattice_setup import LatticeConfig, MoireLatticeSetup


@dataclass
class GeometryConfig:
    """Photonic crystal geometry configuration - TM polarization
    
    Geometry: Air holes in dielectric background
    - Cylinders (holes): ε = eps_lo = 1.0 (air)
    - Background: ε = eps_hi = 4.64 (dielectric)
    """
    # Dielectric constants (specific values for this study)
    eps_hi: float = 4.64  # High dielectric (background)
    eps_lo: float = 1.0   # Low dielectric (air holes)
    
    # Geometry
    cylinder_radius: float = 0.48  # r/a ratio
    
    # Polarization
    polarization: str = "TM"  # TM or TE
    
    # MPB settings
    resolution: int = 32
    num_bands: int = 8
    
    # Band analysis (using M point, first band = index 0, but MPB is 1-indexed)
    target_band: int = 1  # Band index to analyze (1-indexed for MPB) - 1st band
    k_point_name: str = "M"  # High symmetry point: "Gamma", "M", "K"
    
    # Finite difference for derivatives
    dk: float = 0.01  # Small k-step for numerical derivatives


def build_bilayer_geometry(
    delta_frac: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    geom_config: GeometryConfig
) -> Tuple[mp.Lattice, List]:
    """
    Build a bilayer photonic crystal geometry with relative shift δ.
    
    Geometry: Air holes (ε=1.0) in high-dielectric background (ε=4.64).
    The background material is set in the ModeSolver's default_material.
    
    This represents the "frozen registry" approximation where we replace
    the twisted bilayer with an untwisted bilayer at a specific stacking shift.
    
    Args:
        delta_frac: Fractional shift (2D) in monolayer cell coordinates
        a1, a2: Monolayer lattice vectors (3D)
        geom_config: Geometry configuration
    
    Returns:
        (lattice, geometry_list): MPB lattice and geometry objects
    """
    # Build lattice
    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, 0),
        basis1=mp.Vector3(a1[0], a1[1], 0),
        basis2=mp.Vector3(a2[0], a2[1], 0)
    )
    
    # Layer 1: air hole at origin
    r = geom_config.cylinder_radius
    eps_hole = geom_config.eps_lo  # Air holes
    
    cyl1 = mp.Cylinder(
        radius=r,
        center=mp.Vector3(0, 0, 0),
        material=mp.Medium(epsilon=eps_hole)
    )
    
    # Layer 2: air hole shifted by δ
    # Convert fractional to Cartesian
    shift_cart = delta_frac[0] * a1[:2] + delta_frac[1] * a2[:2]
    cyl2 = mp.Cylinder(
        radius=r,
        center=mp.Vector3(shift_cart[0], shift_cart[1], 0),
        material=mp.Medium(epsilon=eps_hole)
    )
    
    geometry = [cyl1, cyl2]
    
    return lattice, geometry


def get_high_symmetry_k_point(k_name: str, reciprocal_vecs: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Get the Cartesian coordinates of a high-symmetry k-point.
    
    Args:
        k_name: Name of the point ("Gamma", "M", "K", "X")
        reciprocal_vecs: Tuple of (b1, b2) reciprocal lattice vectors
    
    Returns:
        k_cart: 3D k-vector in Cartesian coordinates
    """
    b1, b2 = reciprocal_vecs
    
    # Define in fractional coordinates
    if k_name == "Gamma":
        k_frac = np.array([0.0, 0.0])
    elif k_name == "M":
        # For triangular: M = (b1 + b2) / 2
        k_frac = np.array([0.5, 0.5])
    elif k_name == "K":
        # For triangular: K = (2*b1 + b2) / 3
        k_frac = np.array([2.0/3.0, 1.0/3.0])
    elif k_name == "X":
        # For square: X = b1/2
        k_frac = np.array([0.5, 0.0])
    else:
        raise ValueError(f"Unknown k-point: {k_name}")
    
    # Convert to Cartesian (only 2D components)
    k_cart = k_frac[0] * b1[:2] + k_frac[1] * b2[:2]
    
    return np.append(k_cart, 0.0)


def compute_band_at_k(
    ms: mpb.ModeSolver,
    k_point: mp.Vector3,
    band_index: int
) -> float:
    """
    Compute frequency at a specific k-point for a specific band.
    
    Args:
        ms: MPB ModeSolver
        k_point: k-vector
        band_index: Band index (1-indexed)
    
    Returns:
        frequency: ω(k) for the specified band
    """
    ms.k_points = [k_point]
    ms.run()
    # all_freqs is shape (num_k_points, num_bands), we have 1 k-point
    freqs = ms.all_freqs[0]  # Get frequencies for the single k-point
    return freqs[band_index - 1]


def compute_local_band_data(
    delta_frac: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    geom_config: GeometryConfig,
    reciprocal_vecs: Tuple[np.ndarray, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ω₀, M⁻¹, and v_g at a frozen registry via finite differences.
    
    Args:
        delta_frac: Fractional shift in monolayer cell
        a1, a2: Monolayer lattice vectors
        geom_config: Geometry configuration
        reciprocal_vecs: (b1, b2) reciprocal vectors
    
    Returns:
        (omega0, M_inv, v_g):
            omega0: Frequency at k₀
            M_inv: 2×2 inverse mass tensor
            v_g: 2D group velocity
    """
    # Build geometry
    lattice, geometry = build_bilayer_geometry(delta_frac, a1, a2, geom_config)
    
    # Create ModeSolver
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        resolution=geom_config.resolution,
        num_bands=geom_config.num_bands,
        default_material=mp.Medium(epsilon=geom_config.eps_hi)  # High-ε background
    )
    
    # Get k₀
    k0_cart = get_high_symmetry_k_point(geom_config.k_point_name, reciprocal_vecs)
    k0 = mp.Vector3(k0_cart[0], k0_cart[1], k0_cart[2])
    
    # Central frequency
    omega0 = compute_band_at_k(ms, k0, geom_config.target_band)
    
    # Compute derivatives via finite differences
    dk = geom_config.dk
    b1, b2 = reciprocal_vecs
    
    # First derivatives for group velocity: v_g = ∇_k ω
    k_px = mp.Vector3(k0.x + dk*b1[0], k0.y + dk*b1[1], 0)
    k_mx = mp.Vector3(k0.x - dk*b1[0], k0.y - dk*b1[1], 0)
    k_py = mp.Vector3(k0.x + dk*b2[0], k0.y + dk*b2[1], 0)
    k_my = mp.Vector3(k0.x - dk*b2[0], k0.y - dk*b2[1], 0)
    
    omega_px = compute_band_at_k(ms, k_px, geom_config.target_band)
    omega_mx = compute_band_at_k(ms, k_mx, geom_config.target_band)
    omega_py = compute_band_at_k(ms, k_py, geom_config.target_band)
    omega_my = compute_band_at_k(ms, k_my, geom_config.target_band)
    
    # Group velocity (should be ~0 at band extremum)
    dw_dkx = (omega_px - omega_mx) / (2 * dk)
    dw_dky = (omega_py - omega_my) / (2 * dk)
    v_g = np.array([dw_dkx, dw_dky])
    
    # Second derivatives for effective mass: M⁻¹ᵢⱼ = ∂²ω/∂kᵢ∂kⱼ
    # Diagonal terms
    d2w_dkx2 = (omega_px - 2*omega0 + omega_mx) / (dk**2)
    d2w_dky2 = (omega_py - 2*omega0 + omega_my) / (dk**2)
    
    # Off-diagonal (mixed derivative)
    k_pp = mp.Vector3(k0.x + dk*b1[0] + dk*b2[0], k0.y + dk*b1[1] + dk*b2[1], 0)
    k_pm = mp.Vector3(k0.x + dk*b1[0] - dk*b2[0], k0.y + dk*b1[1] - dk*b2[1], 0)
    k_mp = mp.Vector3(k0.x - dk*b1[0] + dk*b2[0], k0.y - dk*b1[1] + dk*b2[1], 0)
    k_mm = mp.Vector3(k0.x - dk*b1[0] - dk*b2[0], k0.y - dk*b1[1] - dk*b2[1], 0)
    
    omega_pp = compute_band_at_k(ms, k_pp, geom_config.target_band)
    omega_pm = compute_band_at_k(ms, k_pm, geom_config.target_band)
    omega_mp = compute_band_at_k(ms, k_mp, geom_config.target_band)
    omega_mm = compute_band_at_k(ms, k_mm, geom_config.target_band)
    
    d2w_dkxdky = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk**2)
    
    # Inverse mass tensor
    M_inv = np.array([
        [d2w_dkx2, d2w_dkxdky],
        [d2w_dkxdky, d2w_dky2]
    ])
    
    return omega0, M_inv, v_g


class LocalBlochCalculator:
    """Main class for Phase 1 calculations"""
    
    def __init__(
        self,
        lattice_setup: MoireLatticeSetup,
        geom_config: GeometryConfig
    ):
        self.setup = lattice_setup
        self.geom_config = geom_config
        
        # Get reciprocal vectors
        b_vecs = self.setup.base_lattice.reciprocal_basis().base_vectors()
        self.b1 = np.array(b_vecs[0])
        self.b2 = np.array(b_vecs[1])
        
        # Storage for results
        self.omega_grid = None
        self.M_inv_grid = None
        self.v_g_grid = None
        self.omega_ref = None
        
    def compute_reference_band(self, delta_ref: np.ndarray = None):
        """
        Compute reference band structure at a specific registry.
        This determines ω₀^ref and confirms the band extremum.
        
        Args:
            delta_ref: Reference registry shift (fractional). If None, use (0,0)
        """
        if delta_ref is None:
            delta_ref = np.array([0.0, 0.0])
        
        print(f"\nComputing reference band at δ_ref = [{delta_ref[0]:.3f}, {delta_ref[1]:.3f}]")
        
        omega0, M_inv, v_g = compute_local_band_data(
            delta_ref,
            self.setup.a1,
            self.setup.a2,
            self.geom_config,
            (self.b1, self.b2)
        )
        
        self.omega_ref = omega0
        
        print(f"  ω₀^ref = {omega0:.6f}")
        print(f"  |v_g| = {np.linalg.norm(v_g):.6e} (should be ≈0 at extremum)")
        print(f"  M⁻¹ eigenvalues: {np.linalg.eigvals(M_inv)}")
        
        return omega0, M_inv, v_g
    
    def compute_grid(self, subsample: int = 4):
        """
        Compute band data on a subsampled moiré grid.
        
        Args:
            subsample: Factor to subsample the grid (e.g., 4 means every 4th point)
        """
        Nx, Ny = self.setup.config.Nx, self.setup.config.Ny
        Nx_sub = Nx // subsample
        Ny_sub = Ny // subsample
        
        print(f"\nComputing on {Nx_sub} × {Ny_sub} grid...")
        print("This may take a while (MPB calculations)...")
        
        self.omega_grid = np.zeros((Nx_sub, Ny_sub))
        self.M_inv_grid = np.zeros((Nx_sub, Ny_sub, 2, 2))
        self.v_g_grid = np.zeros((Nx_sub, Ny_sub, 2))
        
        total = Nx_sub * Ny_sub
        count = 0
        
        for i in range(Nx_sub):
            for j in range(Ny_sub):
                # Get delta at this grid point
                i_full = i * subsample
                j_full = j * subsample
                delta_frac = self.setup.delta_grid[i_full, j_full, :]
                
                # Compute band data
                omega, M_inv, v_g = compute_local_band_data(
                    delta_frac,
                    self.setup.a1,
                    self.setup.a2,
                    self.geom_config,
                    (self.b1, self.b2)
                )
                
                self.omega_grid[i, j] = omega
                self.M_inv_grid[i, j, :, :] = M_inv
                self.v_g_grid[i, j, :] = v_g
                
                count += 1
                if count % 10 == 0:
                    print(f"  Progress: {count}/{total} ({100*count/total:.1f}%)")
        
        print(f"✓ Grid computation complete")
    
    def save_outputs(self, output_dir: Path):
        """Save Phase 1 outputs"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.omega_grid is None:
            print("Warning: No grid data to save. Run compute_grid() first.")
            return
        
        # Save band data
        Nx_sub, Ny_sub = self.omega_grid.shape
        with open(output_dir / "phase1_band_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "i", "j", "omega0", 
                "M_inv_xx", "M_inv_xy", "M_inv_yy",
                "v_g_x", "v_g_y"
            ])
            
            for i in range(Nx_sub):
                for j in range(Ny_sub):
                    M = self.M_inv_grid[i, j, :, :]
                    v = self.v_g_grid[i, j, :]
                    writer.writerow([
                        i, j, self.omega_grid[i, j],
                        M[0, 0], M[0, 1], M[1, 1],
                        v[0], v[1]
                    ])
        
        # Save reference
        if self.omega_ref is not None:
            with open(output_dir / "phase1_reference.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["parameter", "value"])
                writer.writerow(["omega_ref", self.omega_ref])
                writer.writerow(["k_point", self.geom_config.k_point_name])
                writer.writerow(["target_band", self.geom_config.target_band])
        
        # Save full data as pickle for Phase 2
        data = {
            "omega_grid": self.omega_grid,
            "M_inv_grid": self.M_inv_grid,
            "v_g_grid": self.v_g_grid,
            "omega_ref": self.omega_ref,
            "config": self.geom_config
        }
        with open(output_dir / "phase1_data.pkl", "wb") as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved Phase 1 outputs to {output_dir}")
    
    def visualize(self, output_dir: Path):
        """Create visualizations"""
        if self.omega_grid is None:
            print("Warning: No grid data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Compute potential V(R) = ω₀(R) - ω₀^ref
        V = self.omega_grid - self.omega_ref
        
        # Plot 1: ω₀(R)
        ax = axes[0, 0]
        im = ax.imshow(self.omega_grid.T, origin='lower', cmap='viridis')
        ax.set_title('ω₀(R)')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        # Plot 2: V(R) = ω₀(R) - ω₀^ref
        ax = axes[0, 1]
        im = ax.imshow(V.T, origin='lower', cmap='RdBu_r')
        ax.set_title('V(R) = ω₀(R) - ω₀^ref')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        # Plot 3: |v_g(R)|
        ax = axes[0, 2]
        v_g_mag = np.sqrt(self.v_g_grid[:, :, 0]**2 + self.v_g_grid[:, :, 1]**2)
        im = ax.imshow(v_g_mag.T, origin='lower', cmap='plasma')
        ax.set_title('|v_g(R)| (should be ≈0)')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        # Plot 4: M⁻¹_xx
        ax = axes[1, 0]
        im = ax.imshow(self.M_inv_grid[:, :, 0, 0].T, origin='lower', cmap='coolwarm')
        ax.set_title('M⁻¹_xx (curvature)')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        # Plot 5: M⁻¹_yy
        ax = axes[1, 1]
        im = ax.imshow(self.M_inv_grid[:, :, 1, 1].T, origin='lower', cmap='coolwarm')
        ax.set_title('M⁻¹_yy (curvature)')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        # Plot 6: M⁻¹_xy
        ax = axes[1, 2]
        im = ax.imshow(self.M_inv_grid[:, :, 0, 1].T, origin='lower', cmap='coolwarm')
        ax.set_title('M⁻¹_xy (anisotropy)')
        ax.set_xlabel('i')
        ax.set_ylabel('j')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase1_band_visualization.png", dpi=150)
        print(f"✓ Saved visualization to {output_dir / 'phase1_band_visualization.png'}")
        plt.close()


def main():
    """Main execution for Phase 1"""
    print("=" * 60)
    print("Phase 1: Local Bloch Problems")
    print("=" * 60)
    
    # Load Phase 0 results
    print("\nLoading Phase 0 configuration...")
    lattice_config = LatticeConfig(
        lattice_constant=1.0,
        twist_angle_deg=None,  # Uses default 1.1°
        stacking_gauge=(0.0, 0.0),
        moire_resolution=(64, 64)
    )
    
    lattice_setup = MoireLatticeSetup(lattice_config)
    
    # Geometry configuration (TM polarization, r=0.48, ε=4.64, M point, 1st band)
    geom_config = GeometryConfig(
        eps_hi=4.64,
        eps_lo=1.0,
        cylinder_radius=0.48,
        polarization="TM",
        resolution=32,
        num_bands=8,
        target_band=1,  # 1st band (0th index)
        k_point_name="M",
        dk=0.01
    )
    
    print(f"\nGeometry Configuration:")
    print(f"  Polarization: {geom_config.polarization}")
    print(f"  ε_hi = {geom_config.eps_hi}, ε_lo = {geom_config.eps_lo}")
    print(f"  Cylinder radius r/a = {geom_config.cylinder_radius}")
    print(f"  Target band = {geom_config.target_band} ({geom_config.k_point_name} point)")
    print(f"  MPB resolution = {geom_config.resolution}")
    
    # Initialize calculator
    calculator = LocalBlochCalculator(lattice_setup, geom_config)
    
    # Compute reference band
    print("\n" + "-" * 60)
    print("Step 1: Computing reference band structure")
    print("-" * 60)
    calculator.compute_reference_band()
    
    # Compute on grid (use small subsample for testing)
    print("\n" + "-" * 60)
    print("Step 2: Computing band data on moiré grid")
    print("-" * 60)
    print("Note: Using subsample=8 for demonstration (change to 4 or 2 for production)")
    calculator.compute_grid(subsample=8)
    
    # Save and visualize
    output_dir = Path(__file__).parent / "outputs"
    calculator.save_outputs(output_dir)
    calculator.visualize(output_dir)
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete ✓")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Review phase1_band_visualization.png")
    print("  - Check that |v_g| ≈ 0 (confirms band extremum)")
    print("  - Verify V(R) shows expected moiré modulation")
    print("  - Run Phase 2 to build envelope operator")


if __name__ == "__main__":
    main()
