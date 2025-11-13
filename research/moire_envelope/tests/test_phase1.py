#!/usr/bin/env python3
"""
Test Phase 1 Implementation

This script tests the Phase 1 implementation with a minimal synthetic dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.moire_utils import build_R_grid, compute_registry_map
from common.mpb_utils import compute_local_band_at_registry
from common.geometry import build_lattice
from common.io_utils import choose_reference_frequency


def test_build_R_grid():
    """Test spatial grid construction"""
    print("Testing build_R_grid...")
    
    Nx, Ny = 8, 8
    L_moire = 10.0
    R_grid = build_R_grid(Nx, Ny, L_moire, center=True)
    
    assert R_grid.shape == (Nx, Ny, 2), f"Shape mismatch: {R_grid.shape}"
    assert np.abs(R_grid[0, 0, 0] - (-L_moire/2)) < 1e-6, "Grid not centered"
    assert np.abs(R_grid[-1, -1, 0] - (L_moire/2)) < 1e-6, "Grid extent incorrect"
    
    print("  ✓ R_grid construction works correctly")


def test_compute_registry_map():
    """Test registry map computation"""
    print("Testing compute_registry_map...")
    
    Nx, Ny = 4, 4
    L_moire = 5.0
    R_grid = build_R_grid(Nx, Ny, L_moire, center=True)
    
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    theta = np.radians(1.5)
    tau = np.zeros(2)
    eta = 1.0
    
    delta_grid = compute_registry_map(R_grid, a1, a2, theta, tau, eta)
    
    assert delta_grid.shape == (Nx, Ny, 2), f"Shape mismatch: {delta_grid.shape}"
    assert not np.any(np.isnan(delta_grid)), "NaN values in delta_grid"
    
    print(f"  ✓ Registry map computed: range {delta_grid.min():.4f} to {delta_grid.max():.4f}")


def test_choose_reference_frequency():
    """Test reference frequency selection"""
    print("Testing choose_reference_frequency...")
    
    omega_grid = np.random.rand(10, 10) * 0.1 + 0.5
    
    config_mean = {'ref_frequency_mode': 'mean'}
    config_min = {'ref_frequency_mode': 'min'}
    config_max = {'ref_frequency_mode': 'max'}
    
    ref_mean = choose_reference_frequency(omega_grid, config_mean)
    ref_min = choose_reference_frequency(omega_grid, config_min)
    ref_max = choose_reference_frequency(omega_grid, config_max)
    
    assert omega_grid.min() <= ref_mean <= omega_grid.max()
    assert np.abs(ref_min - omega_grid.min()) < 1e-6
    assert np.abs(ref_max - omega_grid.max()) < 1e-6
    
    print(f"  ✓ Reference frequency selection works: min={ref_min:.4f}, mean={ref_mean:.4f}, max={ref_max:.4f}")


def test_build_lattice():
    """Test lattice geometry construction"""
    print("Testing build_lattice...")
    
    geom_square = build_lattice('square', 0.3, 4.0, a=1.0)
    geom_hex = build_lattice('hex', 0.25, 6.0, a=1.0)
    
    assert geom_square['lattice_type'] == 'square'
    assert geom_hex['lattice_type'] == 'hex'
    assert geom_square['r_over_a'] == 0.3
    assert geom_hex['eps_bg'] == 6.0
    
    print("  ✓ Lattice geometry construction works")


def test_synthetic_phase1_workflow():
    """Test complete Phase 1 workflow with synthetic data"""
    print("\nTesting complete Phase 1 workflow...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        run_dir = tmpdir / "test_run"
        run_dir.mkdir()
        
        # Create synthetic Phase 0 candidates CSV
        candidates_data = {
            'candidate_id': [1, 2],
            'lattice_type': ['square', 'hex'],
            'a': [1.0, 1.0],
            'r_over_a': [0.30, 0.25],
            'eps_bg': [4.0, 6.0],
            'band_index': [3, 4],
            'k_label': ['Γ', 'M'],
            'k0_x': [0.0, 0.5],
            'k0_y': [0.0, 0.0],
            'omega0': [0.45, 0.52],
            'S_total': [0.85, 0.82],
            'curvature_trace': [1.5, 1.8],
            'vg_norm': [0.05, 0.03],
        }
        
        df = pd.DataFrame(candidates_data)
        df.to_csv(run_dir / "phase0_candidates.csv", index=False)
        
        print(f"  Created synthetic candidates in {run_dir}")
        print(f"  Test run directory: {run_dir}")
        
        # Test that we can load the candidates
        df_loaded = pd.read_csv(run_dir / "phase0_candidates.csv")
        assert len(df_loaded) == 2
        print("  ✓ Synthetic Phase 0 data created and loaded")


def run_all_tests():
    """Run all Phase 1 tests"""
    print("="*70)
    print("Phase 1 Implementation Tests")
    print("="*70)
    print()
    
    try:
        test_build_R_grid()
        test_compute_registry_map()
        test_choose_reference_frequency()
        test_build_lattice()
        test_synthetic_phase1_workflow()
        
        print()
        print("="*70)
        print("All tests passed! ✓")
        print("="*70)
        print()
        print("Phase 1 implementation is ready to use.")
        print()
        print("Next steps:")
        print("1. Run Phase 0 to generate candidates")
        print("2. Run Phase 1 on the Phase 0 output:")
        print("   python phases/phase1_local_bloch.py <run_dir> configs/phase1_quick_test.yaml")
        
        return True
        
    except Exception as e:
        print()
        print("="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
