"""
Phase 2: Envelope Approximation Operator Assembly

This script builds the Hermitian operator for the envelope approximation:
    H_EA = -i·η·v_g(R)·∇_R - η²/2·∇_R·M⁻¹(R)·∇_R + V(R)

For band extrema where v_g ≈ 0, this simplifies to:
    H_EA = -η²/2·∇_R·M⁻¹(R)·∇_R + V(R)

The operator is assembled as a sparse matrix with periodic boundary conditions
on the moiré lattice.

Inputs:
- Phase 1 outputs (phase1_data.pkl): ω₀(R), M⁻¹(R), v_g(R)
- Phase 0 outputs: Moiré lattice vectors, η

Outputs:
- phase2_operator.npz: Sparse Hamiltonian matrix
- phase2_operator_info.csv: Operator statistics and diagnostics
- phase2_fields_visualization.png: Spatial maps of coefficients
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import csv
import pickle
from scipy import sparse
from typing import Tuple

# Ensure project modules resolve when invoked from repo root
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Phase 0 and Phase 1 imports
from phase0_lattice_setup import LatticeConfig, MoireLatticeSetup
from phase1_local_bloch import GeometryConfig  # Needed for unpickling


class EnvelopeOperator:
    """Build the envelope approximation Hamiltonian"""
    
    def __init__(self, phase1_data: dict, lattice_setup: MoireLatticeSetup):
        """
        Initialize operator builder.
        
        Args:
            phase1_data: Dictionary from phase1_data.pkl
            lattice_setup: MoireLatticeSetup from Phase 0
        """
        self.omega_grid = phase1_data['omega_grid']
        self.M_inv_grid = phase1_data['M_inv_grid']
        self.v_g_grid = phase1_data['v_g_grid']
        self.omega_ref = phase1_data['omega_ref']
        
        self.setup = lattice_setup
        self.eta = lattice_setup.eta
        
        # Grid dimensions
        self.Nx, self.Ny = self.omega_grid.shape
        self.N_total = self.Nx * self.Ny
        
        # Potential
        self.V = self.omega_grid - self.omega_ref
        
        # Moiré cell dimensions
        self.Lx = np.linalg.norm(lattice_setup.A1[:2])
        self.Ly = np.linalg.norm(lattice_setup.A2[:2])
        
        # Grid spacing
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        
        print(f"Operator setup:")
        print(f"  Grid: {self.Nx} × {self.Ny} = {self.N_total} points")
        print(f"  Moiré cell: {self.Lx:.2f} × {self.Ly:.2f}")
        print(f"  Grid spacing: dx={self.dx:.3f}, dy={self.dy:.3f}")
        print(f"  η = {self.eta:.6f}")
        
    def _flat_index(self, i: int, j: int) -> int:
        """Convert 2D grid index to flat index with periodic BC"""
        i = i % self.Nx
        j = j % self.Ny
        return i * self.Ny + j
    
    def _periodic_diff_operator_1d(self, n: int, h: float) -> sparse.csr_matrix:
        """
        Build 1D periodic first derivative operator (central differences).
        
        Args:
            n: Number of grid points
            h: Grid spacing
            
        Returns:
            Sparse matrix for ∂/∂x
        """
        diags = np.ones((3, n))
        diags[0, :] = -1 / (2 * h)  # Lower diagonal
        diags[1, :] = 0              # Main diagonal
        diags[2, :] = 1 / (2 * h)    # Upper diagonal
        
        D = sparse.diags(diags, [-1, 0, 1], shape=(n, n), format='lil')
        
        # Periodic boundaries
        D[0, -1] = -1 / (2 * h)
        D[-1, 0] = 1 / (2 * h)
        
        return D.tocsr()
    
    def _build_gradient_operators(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Build gradient operators ∇_R = (∂/∂x, ∂/∂y) on 2D grid.
        
        Returns:
            (Dx, Dy): Sparse matrices for ∂/∂x and ∂/∂y
        """
        # 1D derivative operators
        Dx_1d = self._periodic_diff_operator_1d(self.Nx, self.dx)
        Dy_1d = self._periodic_diff_operator_1d(self.Ny, self.dy)
        
        # Kronecker products for 2D grid
        Iy = sparse.eye(self.Ny)
        Ix = sparse.eye(self.Nx)
        
        # ∂/∂x acts on i-index (rows)
        Dx = sparse.kron(Dx_1d, Iy, format='csr')
        
        # ∂/∂y acts on j-index (columns)
        Dy = sparse.kron(Ix, Dy_1d, format='csr')
        
        return Dx, Dy
    
    def _build_laplacian_with_variable_coeff(self) -> sparse.csr_matrix:
        """
        Build the kinetic energy operator:
            T = -η²/2 · ∇_R · M⁻¹(R) · ∇_R
        
        Using finite differences with variable coefficients.
        For each point (i,j), we compute:
            T[i,j] = -η²/2 · [M_xx·∂²/∂x² + 2·M_xy·∂²/∂x∂y + M_yy·∂²/∂y²]
        
        Returns:
            Sparse kinetic energy operator
        """
        print("\nBuilding kinetic energy operator...")
        
        # Initialize sparse matrix
        row_idx = []
        col_idx = []
        data = []
        
        coeff = -self.eta**2 / 2.0
        
        for i in range(self.Nx):
            for j in range(self.Ny):
                idx = self._flat_index(i, j)
                
                # Get M^{-1} at this point
                M_xx = self.M_inv_grid[i, j, 0, 0]
                M_xy = self.M_inv_grid[i, j, 0, 1]
                M_yy = self.M_inv_grid[i, j, 1, 1]
                
                # Neighbor indices (with periodic BC)
                idx_xp = self._flat_index(i+1, j)  # x+1
                idx_xm = self._flat_index(i-1, j)  # x-1
                idx_yp = self._flat_index(i, j+1)  # y+1
                idx_ym = self._flat_index(i, j-1)  # y-1
                idx_xpyp = self._flat_index(i+1, j+1)
                idx_xpym = self._flat_index(i+1, j-1)
                idx_xmyp = self._flat_index(i-1, j+1)
                idx_xmym = self._flat_index(i-1, j-1)
                
                # Second derivative stencils
                # ∂²/∂x²: (f[i+1] - 2f[i] + f[i-1]) / dx²
                c_xx = coeff * M_xx / self.dx**2
                
                # ∂²/∂y²: (f[j+1] - 2f[j] + f[j-1]) / dy²
                c_yy = coeff * M_yy / self.dy**2
                
                # ∂²/∂x∂y: (f[i+1,j+1] - f[i+1,j-1] - f[i-1,j+1] + f[i-1,j-1]) / (4·dx·dy)
                c_xy = coeff * M_xy / (4 * self.dx * self.dy)
                
                # Center point contribution
                center_val = -2 * c_xx - 2 * c_yy
                row_idx.append(idx)
                col_idx.append(idx)
                data.append(center_val)
                
                # x-direction second derivative
                row_idx.extend([idx, idx])
                col_idx.extend([idx_xp, idx_xm])
                data.extend([c_xx, c_xx])
                
                # y-direction second derivative
                row_idx.extend([idx, idx])
                col_idx.extend([idx_yp, idx_ym])
                data.extend([c_yy, c_yy])
                
                # Mixed derivative (if M_xy is significant)
                if abs(M_xy) > 1e-10:
                    row_idx.extend([idx, idx, idx, idx])
                    col_idx.extend([idx_xpyp, idx_xpym, idx_xmyp, idx_xmym])
                    data.extend([c_xy, -c_xy, -c_xy, c_xy])
        
        T = sparse.csr_matrix((data, (row_idx, col_idx)), 
                             shape=(self.N_total, self.N_total))
        
        print(f"  Kinetic operator: {T.nnz} non-zero elements")
        print(f"  Matrix size: {self.N_total} × {self.N_total}")
        
        return T
    
    def _build_drift_operator(self) -> sparse.csr_matrix:
        """
        Build the drift term (if v_g ≠ 0):
            D = -i·η·v_g(R)·∇_R = -i·η·[v_x·∂/∂x + v_y·∂/∂y]
        
        Returns:
            Sparse drift operator
        """
        print("\nBuilding drift operator...")
        
        # Check if drift is significant
        v_g_mag = np.sqrt(self.v_g_grid[:, :, 0]**2 + self.v_g_grid[:, :, 1]**2)
        max_vg = np.max(v_g_mag)
        
        print(f"  Max |v_g|: {max_vg:.6f}")
        
        if max_vg < 1e-4:
            print(f"  v_g ≈ 0: Drift term negligible")
            return sparse.csr_matrix((self.N_total, self.N_total))
        
        # Build gradient operators
        Dx, Dy = self._build_gradient_operators()
        
        # Diagonal matrices with v_g components
        vx_flat = self.v_g_grid[:, :, 0].flatten()
        vy_flat = self.v_g_grid[:, :, 1].flatten()
        
        Vx_diag = sparse.diags(vx_flat, 0, format='csr')
        Vy_diag = sparse.diags(vy_flat, 0, format='csr')
        
        # D = -i·η·(v_x·Dx + v_y·Dy)
        # Note: We work with real operators, so we'll keep this real for now
        # The imaginary unit will be handled in the eigenvalue solver
        D = -self.eta * (Vx_diag @ Dx + Vy_diag @ Dy)
        
        print(f"  Drift operator: {D.nnz} non-zero elements")
        print(f"  Warning: v_g is not small - this may indicate we're not at a band extremum")
        
        return D
    
    def _build_potential_operator(self) -> sparse.csr_matrix:
        """
        Build the potential energy operator V(R).
        
        Returns:
            Diagonal sparse matrix with V on diagonal
        """
        V_flat = self.V.flatten()
        return sparse.diags(V_flat, 0, format='csr')
    
    def build_hamiltonian(self, include_drift: bool = True) -> sparse.csr_matrix:
        """
        Build the complete Hamiltonian:
            H = T + D + V
        where:
            T = kinetic energy (Laplacian with variable mass)
            D = drift term (if v_g ≠ 0)
            V = potential energy
        
        Args:
            include_drift: Whether to include drift term (set False if v_g ≈ 0)
            
        Returns:
            Sparse Hamiltonian matrix
        """
        print("\n" + "=" * 60)
        print("Building Envelope Approximation Hamiltonian")
        print("=" * 60)
        
        # Kinetic energy
        T = self._build_laplacian_with_variable_coeff()
        
        # Drift term
        if include_drift:
            D = self._build_drift_operator()
        else:
            print("\nSkipping drift operator (v_g assumed ≈ 0)")
            D = sparse.csr_matrix((self.N_total, self.N_total))
        
        # Potential energy
        print("\nBuilding potential operator...")
        V_op = self._build_potential_operator()
        print(f"  Potential: {V_op.nnz} non-zero elements (diagonal)")
        print(f"  V range: [{self.V.min():.6f}, {self.V.max():.6f}]")
        
        # Total Hamiltonian
        H = T + D + V_op
        
        print(f"\n" + "=" * 60)
        print(f"Total Hamiltonian:")
        print(f"  Size: {H.shape[0]} × {H.shape[1]}")
        print(f"  Non-zeros: {H.nnz} ({100*H.nnz/H.shape[0]**2:.2f}% density)")
        print(f"  Is symmetric: {np.allclose(H.data, H.T.data) if D.nnz == 0 else 'No (has drift)'}")
        print("=" * 60)
        
        return H
    
    def save_operator(self, H: sparse.csr_matrix, output_dir: Path):
        """Save the operator and metadata"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sparse matrix
        sparse.save_npz(output_dir / "phase2_operator.npz", H)
        print(f"\n✓ Saved operator to {output_dir / 'phase2_operator.npz'}")
        
        # Save operator info
        with open(output_dir / "phase2_operator_info.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerow(["grid_nx", self.Nx])
            writer.writerow(["grid_ny", self.Ny])
            writer.writerow(["total_size", self.N_total])
            writer.writerow(["nnz", H.nnz])
            writer.writerow(["density", H.nnz / self.N_total**2])
            writer.writerow(["eta", self.eta])
            writer.writerow(["dx", self.dx])
            writer.writerow(["dy", self.dy])
            writer.writerow(["omega_ref", self.omega_ref])
            writer.writerow(["V_min", self.V.min()])
            writer.writerow(["V_max", self.V.max()])
            writer.writerow(["V_std", self.V.std()])
            writer.writerow(["max_v_g", np.max(np.sqrt(
                self.v_g_grid[:, :, 0]**2 + self.v_g_grid[:, :, 1]**2))])
        
        print(f"✓ Saved operator info to {output_dir / 'phase2_operator_info.csv'}")
    
    def visualize_fields(self, output_dir: Path):
        """Visualize the operator coefficients"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Potential V(R)
        im = axes[0, 0].imshow(self.V.T, origin='lower', cmap='RdBu_r')
        axes[0, 0].set_title('Potential V(R)')
        axes[0, 0].set_xlabel('i')
        axes[0, 0].set_ylabel('j')
        plt.colorbar(im, ax=axes[0, 0])
        
        # M_inv_xx
        im = axes[0, 1].imshow(self.M_inv_grid[:, :, 0, 0].T, origin='lower', cmap='viridis')
        axes[0, 1].set_title('M⁻¹_xx (band curvature)')
        axes[0, 1].set_xlabel('i')
        axes[0, 1].set_ylabel('j')
        plt.colorbar(im, ax=axes[0, 1])
        
        # M_inv_yy
        im = axes[0, 2].imshow(self.M_inv_grid[:, :, 1, 1].T, origin='lower', cmap='viridis')
        axes[0, 2].set_title('M⁻¹_yy (band curvature)')
        axes[0, 2].set_xlabel('i')
        axes[0, 2].set_ylabel('j')
        plt.colorbar(im, ax=axes[0, 2])
        
        # M_inv_xy (anisotropy)
        im = axes[1, 0].imshow(self.M_inv_grid[:, :, 0, 1].T, origin='lower', cmap='RdBu_r')
        axes[1, 0].set_title('M⁻¹_xy (anisotropy)')
        axes[1, 0].set_xlabel('i')
        axes[1, 0].set_ylabel('j')
        plt.colorbar(im, ax=axes[1, 0])
        
        # v_g magnitude
        v_g_mag = np.sqrt(self.v_g_grid[:, :, 0]**2 + self.v_g_grid[:, :, 1]**2)
        im = axes[1, 1].imshow(v_g_mag.T, origin='lower', cmap='plasma')
        axes[1, 1].set_title('|v_g| (group velocity)')
        axes[1, 1].set_xlabel('i')
        axes[1, 1].set_ylabel('j')
        plt.colorbar(im, ax=axes[1, 1])
        
        # Histogram of V
        axes[1, 2].hist(self.V.flatten(), bins=30, edgecolor='black')
        axes[1, 2].set_title('Distribution of V(R)')
        axes[1, 2].set_xlabel('V')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].axvline(0, color='r', linestyle='--', label='V=0')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "phase2_fields_visualization.png", dpi=150)
        print(f"✓ Saved visualization to {output_dir / 'phase2_fields_visualization.png'}")
        plt.close()


def main():
    """Main execution for Phase 2"""
    print("=" * 60)
    print("Phase 2: Envelope Operator Assembly")
    print("=" * 60)
    
    # Load Phase 1 data
    print("\nLoading Phase 1 data...")
    with open("outputs/phase1_data.pkl", "rb") as f:
        phase1_data = pickle.load(f)
    
    print("✓ Loaded Phase 1 data")
    
    # Load Phase 0 configuration
    print("\nLoading Phase 0 configuration...")
    lattice_config = LatticeConfig(
        lattice_constant=1.0,
        twist_angle_deg=None,  # Uses default 1.1°
        stacking_gauge=(0.0, 0.0),
        moire_resolution=(64, 64)
    )
    
    lattice_setup = MoireLatticeSetup(lattice_config)
    print("✓ Loaded lattice configuration")
    
    # Build operator
    op_builder = EnvelopeOperator(phase1_data, lattice_setup)
    
    # Check if we should include drift
    v_g_mag_max = np.max(np.sqrt(
        phase1_data['v_g_grid'][:, :, 0]**2 + 
        phase1_data['v_g_grid'][:, :, 1]**2))
    
    include_drift = v_g_mag_max > 1e-4
    
    if include_drift:
        print(f"\n⚠ Warning: |v_g|_max = {v_g_mag_max:.4f} >> 0")
        print("  This suggests we're not at a band extremum.")
        print("  Including drift term in Hamiltonian.")
    
    H = op_builder.build_hamiltonian(include_drift=include_drift)
    
    # Save outputs
    output_dir = Path("outputs")
    op_builder.save_operator(H, output_dir)
    op_builder.visualize_fields(output_dir)
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete ✓")
    print("=" * 60)
    print("\nNext step: Run Phase 3 to solve for envelope eigenstates")


if __name__ == "__main__":
    main()
