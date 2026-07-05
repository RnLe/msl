#!/usr/bin/env python3
"""
Inspect Phase 1 Results

This script reads and summarizes Phase 1 output data for a candidate.

Usage:
    mamba run -n msl python inspect_phase1_results.py <run_dir> <candidate_id>

Example:
    mamba run -n msl python inspect_phase1_results.py runs/phase0_real_run_20241113_120000 1
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.io_utils import candidate_dir


def inspect_phase1_candidate(run_dir, candidate_id):
    """Inspect Phase 1 results for a specific candidate"""
    
    cdir = candidate_dir(run_dir, candidate_id)
    
    print("="*70)
    print(f"Phase 1 Results for Candidate {candidate_id}")
    print("="*70)
    print(f"Directory: {cdir}")
    print()
    
    # Check if directory exists
    if not cdir.exists():
        print(f"Error: Candidate directory not found: {cdir}")
        return
    
    # Load metadata
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        print("Candidate Metadata:")
        print(f"  Lattice type: {meta.get('lattice_type')}")
        print(f"  Lattice constant: {meta.get('a', 0):.4f}")
        print(f"  Hole radius ratio: {meta.get('r_over_a', 0):.4f}")
        print(f"  Background ε: {meta.get('eps_bg', 0):.2f}")
        print(f"  Band index: {meta.get('band_index')}")
        print(f"  k-point: {meta.get('k_label')} ({meta.get('k0_x', 0):.4f}, {meta.get('k0_y', 0):.4f})")
        print(f"  Reference ω₀: {meta.get('omega0', 0):.6f}")
        if 'theta_deg' in meta:
            print(f"  Twist angle: {meta.get('theta_deg', 0):.2f}°")
        print()
    
    # Load HDF5 data
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        print(f"Error: Phase 1 data not found: {h5_path}")
        print("Please run Phase 1 first.")
        return
    
    with h5py.File(h5_path, 'r') as hf:
        print("Phase 1 Data Summary:")
        print(f"  HDF5 file: {h5_path.name}")
        print()
        
        # Grid dimensions
        R_grid = hf['R_grid'][:]
        Nx, Ny, _ = R_grid.shape
        print(f"  Grid dimensions: {Nx} × {Ny}")
        print(f"  Total points: {Nx * Ny}")
        print()
        
        # Attributes
        print("  Attributes:")
        for key in hf.attrs.keys():
            print(f"    {key}: {hf.attrs[key]}")
        print()
        
        # Load fields
        omega0 = hf['omega0'][:]
        V = hf['V'][:]
        vg = hf['vg'][:]
        M_inv = hf['M_inv'][:]
        omega_ref = hf.attrs['omega_ref']
        
        # Statistics
        print("  Field Statistics:")
        print(f"    ω₀(R):")
        print(f"      Range: [{omega0.min():.6f}, {omega0.max():.6f}]")
        print(f"      Mean: {omega0.mean():.6f}")
        print(f"      Std: {omega0.std():.6f}")
        print()
        
        print(f"    V(R) = ω₀(R) - ω_ref:")
        print(f"      ω_ref: {omega_ref:.6f}")
        print(f"      Range: [{V.min():.6f}, {V.max():.6f}]")
        print(f"      Mean: {V.mean():.6f}")
        print(f"      Std: {V.std():.6f}")
        print(f"      Depth: {V.max() - V.min():.6f}")
        print()
        
        vg_norm = np.linalg.norm(vg, axis=-1)
        print(f"    |v_g(R)|:")
        print(f"      Range: [{vg_norm.min():.6f}, {vg_norm.max():.6f}]")
        print(f"      Mean: {vg_norm.mean():.6f}")
        print()
        
        # M_inv eigenvalues
        eigvals = np.linalg.eigvalsh(M_inv)
        print(f"    M⁻¹(R) eigenvalues:")
        print(f"      λ₁ range: [{eigvals[..., 0].min():.6f}, {eigvals[..., 0].max():.6f}]")
        print(f"      λ₂ range: [{eigvals[..., 1].min():.6f}, {eigvals[..., 1].max():.6f}]")
        print(f"      λ₁ mean: {eigvals[..., 0].mean():.6f}")
        print(f"      λ₂ mean: {eigvals[..., 1].mean():.6f}")
        print()
        
        # Effective mass estimate
        m_eff_inv = eigvals.mean()
        print(f"    Effective mass (inverse): {m_eff_inv:.6f}")
        if m_eff_inv > 0:
            print(f"    Effective mass: {1.0/m_eff_inv:.6f}")
        print()
    
    # Check for visualization
    viz_path = cdir / "phase1_fields_visualization.png"
    if viz_path.exists():
        print(f"  Visualization: {viz_path.name}")
    else:
        print("  No visualization found")
    
    print()
    print("="*70)
    print("Ready for Phase 2: EA Operator Assembly")
    print("="*70)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nAvailable run directories with Phase 0 results:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for run_dir in sorted(runs_dir.iterdir()):
                if run_dir.is_dir() and (run_dir / "phase0_candidates.csv").exists():
                    print(f"  {run_dir}")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    candidate_id = int(sys.argv[2])
    
    inspect_phase1_candidate(run_dir, candidate_id)


if __name__ == "__main__":
    main()
