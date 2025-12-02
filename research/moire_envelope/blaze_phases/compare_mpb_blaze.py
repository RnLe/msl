"""
Compare MPB vs BLAZE band structure calculations.

This script runs both MPB and BLAZE for a small grid of 2-atom registry shifts
and compares the results to validate the BLAZE data extraction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import time

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from blaze import BulkDriver
except ImportError:
    print("ERROR: blaze2d package not installed. Install with: pip install blaze2d")
    sys.exit(1)

# Try to import MPB
try:
    import meep as mp
    from meep import mpb
    HAS_MPB = True
except ImportError:
    print("WARNING: MPB not available, will only show BLAZE results")
    HAS_MPB = False


def log(msg):
    print(msg, flush=True)


def run_mpb_single(lattice_type, r_over_a, eps_bg, k0, band_index, atom2_pos, resolution=32):
    """
    Run MPB for a single 2-atom configuration.
    
    Args:
        lattice_type: 'square' or 'hex'/'triangular'
        r_over_a: hole radius / lattice constant
        eps_bg: background dielectric
        k0: k-point (kx, ky) in reciprocal lattice units
        band_index: which band (0-indexed)
        atom2_pos: position of second atom (x, y) in [0, 1)
        resolution: MPB resolution
        
    Returns:
        dict with omega, vg_x, vg_y, d2_xx, d2_yy, d2_xy
    """
    if not HAS_MPB:
        return None
    
    # Build geometry
    if lattice_type in ['hex', 'triangular']:
        geometry_lattice = mp.Lattice(
            size=mp.Vector3(1, 1),
            basis1=mp.Vector3(1, 0),
            basis2=mp.Vector3(0.5, np.sqrt(3)/2)
        )
    else:  # square
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
    
    # Two atoms: one at center, one at atom2_pos
    geometry = [
        mp.Cylinder(
            radius=r_over_a,
            height=mp.inf,
            center=mp.Vector3(0.5, 0.5),
            material=mp.Medium(epsilon=1.0)  # Air hole
        ),
        mp.Cylinder(
            radius=r_over_a,
            height=mp.inf,
            center=mp.Vector3(atom2_pos[0], atom2_pos[1]),
            material=mp.Medium(epsilon=1.0)  # Air hole
        ),
    ]
    
    # Suppress MPB output
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        default_material=mp.Medium(epsilon=eps_bg),
        resolution=resolution,
        num_bands=band_index + 1,
    )
    
    # Run at k0 and nearby points for derivatives
    dk = 0.005
    k_points = [
        (k0[0], k0[1]),           # center
        (k0[0] + dk, k0[1]),      # +x
        (k0[0] - dk, k0[1]),      # -x
        (k0[0], k0[1] + dk),      # +y
        (k0[0], k0[1] - dk),      # -y
        (k0[0] + dk, k0[1] + dk), # +x+y
        (k0[0] + dk, k0[1] - dk), # +x-y
        (k0[0] - dk, k0[1] + dk), # -x+y
        (k0[0] - dk, k0[1] - dk), # -x-y
    ]
    
    omegas = []
    
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        for kx, ky in k_points:
            ms.k_points = [mp.Vector3(kx, ky)]
            ms.run_tm()
            omega = ms.all_freqs[0][band_index]
            omegas.append(omega)
    
    omega0 = omegas[0]
    omega_xp = omegas[1]
    omega_xm = omegas[2]
    omega_yp = omegas[3]
    omega_ym = omegas[4]
    omega_pp = omegas[5]
    omega_pm = omegas[6]
    omega_mp = omegas[7]
    omega_mm = omegas[8]
    
    # Compute derivatives
    vg_x = (omega_xp - omega_xm) / (2 * dk)
    vg_y = (omega_yp - omega_ym) / (2 * dk)
    d2_xx = (omega_xp - 2*omega0 + omega_xm) / (dk**2)
    d2_yy = (omega_yp - 2*omega0 + omega_ym) / (dk**2)
    d2_xy = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk**2)
    
    return {
        'omega': omega0,
        'vg_x': vg_x,
        'vg_y': vg_y,
        'd2_xx': d2_xx,
        'd2_yy': d2_yy,
        'd2_xy': d2_xy,
    }


def run_blaze_grid(lattice_type, r_over_a, eps_bg, k0, band_index, 
                   pos_grid, resolution=32, threads=16):
    """
    Run BLAZE for a grid of 2-atom configurations.
    
    Args:
        lattice_type: 'square' or 'hex'/'triangular'
        r_over_a: hole radius / lattice constant
        eps_bg: background dielectric
        k0: k-point (kx, ky)
        band_index: which band (0-indexed)
        pos_grid: list of (x, y) positions for atom 2
        resolution: BLAZE resolution
        threads: number of threads
        
    Returns:
        dict mapping (x, y) -> result dict
    """
    # Map lattice type
    blaze_lattice = 'triangular' if lattice_type in ['hex', 'triangular'] else 'square'
    
    # Build k-stencil for derivatives (2nd order for simplicity)
    dk = 0.005
    k_stencil = [
        [k0[0], k0[1]],           # 0: center
        [k0[0] + dk, k0[1]],      # 1: +x
        [k0[0] - dk, k0[1]],      # 2: -x
        [k0[0], k0[1] + dk],      # 3: +y
        [k0[0], k0[1] - dk],      # 4: -y
        [k0[0] + dk, k0[1] + dk], # 5: +x+y
        [k0[0] + dk, k0[1] - dk], # 6: +x-y
        [k0[0] - dk, k0[1] + dk], # 7: -x+y
        [k0[0] - dk, k0[1] - dk], # 8: -x-y
    ]
    
    k_path_str = ", ".join(f"[{kx:.8f}, {ky:.8f}]" for kx, ky in k_stencil)
    
    # Determine sweep range from pos_grid
    xs = sorted(set(p[0] for p in pos_grid))
    ys = sorted(set(p[1] for p in pos_grid))
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Calculate step
    if len(xs) > 1:
        x_step = (x_max - x_min) / (len(xs) - 1)
    else:
        x_step = 0.1
    if len(ys) > 1:
        y_step = (y_max - y_min) / (len(ys) - 1)
    else:
        y_step = 0.1
    
    num_bands = band_index + 1
    target_band = band_index + 1  # 1-indexed for output.selective
    
    config_content = f'''# BLAZE comparison test config
# Custom k-path for FD stencil
k_path = [{k_path_str}]

[bulk]
threads = {threads}
verbose = false

[geometry]
eps_bg = {eps_bg}

[geometry.lattice]
type = "{blaze_lattice}"
a = 1.0

# Atom 1: fixed at center
[[geometry.atoms]]
pos = [0.5, 0.5]
radius = {r_over_a}
eps_inside = 1.0

# Atom 2: swept position
[[geometry.atoms]]
pos = [{x_min:.8f}, {y_min:.8f}]
radius = {r_over_a}
eps_inside = 1.0

[grid]
nx = {resolution}
ny = {resolution}
lx = 1.0
ly = 1.0

polarization = "TM"

[eigensolver]
n_bands = {num_bands}
max_iter = 200
tol = 1e-6

[dielectric.smoothing]
mesh_size = 4

[ranges]
[[ranges.atoms]]
# First atom: fixed

[[ranges.atoms]]
pos_x = {{ min = {x_min:.8f}, max = {x_max:.8f}, step = {x_step:.8f} }}
pos_y = {{ min = {y_min:.8f}, max = {y_max:.8f}, step = {y_step:.8f} }}

[output]
mode = "selective"

[output.selective]
k_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
bands = [{target_band}]
'''
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "blaze_config.toml"
        config_path.write_text(config_content)
        
        log(f"\nBLAZE config written to: {config_path}")
        log(f"Config content:\n{config_content[:500]}...")
        
        driver = BulkDriver(str(config_path))
        log(f"BLAZE job count: {driver.job_count}")
        
        results = {}
        for r in driver.run_streaming():
            atoms = r['params'].get('atoms', [])
            if len(atoms) < 2:
                continue
            
            pos = atoms[1]['pos']
            x, y = pos[0], pos[1]
            
            bands = r['bands']
            num_k = r['num_k_points']
            num_bands = r['num_bands']
            
            # Extract omegas at each k-point
            # BLAZE returns all computed bands (0 to num_bands-1)
            # We want band_index (0-indexed), which is at index band_index
            actual_band_idx = min(band_index, num_bands - 1)
            
            if num_k < 9:
                log(f"  WARNING: Only {num_k} k-points, expected 9")
                continue
            
            omega0 = bands[0][actual_band_idx]
            omega_xp = bands[1][actual_band_idx]
            omega_xm = bands[2][actual_band_idx]
            omega_yp = bands[3][actual_band_idx]
            omega_ym = bands[4][actual_band_idx]
            omega_pp = bands[5][actual_band_idx]
            omega_pm = bands[6][actual_band_idx]
            omega_mp = bands[7][actual_band_idx]
            omega_mm = bands[8][actual_band_idx]
            
            # Compute derivatives
            vg_x = (omega_xp - omega_xm) / (2 * dk)
            vg_y = (omega_yp - omega_ym) / (2 * dk)
            d2_xx = (omega_xp - 2*omega0 + omega_xm) / (dk**2)
            d2_yy = (omega_yp - 2*omega0 + omega_ym) / (dk**2)
            d2_xy = (omega_pp - omega_pm - omega_mp + omega_mm) / (4 * dk**2)
            
            key = (round(x, 6), round(y, 6))
            results[key] = {
                'omega': omega0,
                'vg_x': vg_x,
                'vg_y': vg_y,
                'd2_xx': d2_xx,
                'd2_yy': d2_yy,
                'd2_xy': d2_xy,
                'raw_bands': [bands[i][actual_band_idx] for i in range(min(9, num_k))],
            }
        
        return results


def main():
    log("="*70)
    log("MPB vs BLAZE Comparison Test")
    log("="*70)
    
    # Test configuration
    lattice_type = 'square'
    r_over_a = 0.39
    eps_bg = 4.3
    k0 = (0.5, 0.5)  # M point for square lattice
    band_index = 3   # Band 4 (0-indexed)
    resolution = 32
    
    # 5x5 grid of atom2 positions
    n_grid = 5
    pos_x = np.linspace(0.1, 0.4, n_grid)
    pos_y = np.linspace(0.1, 0.4, n_grid)
    pos_grid = [(x, y) for y in pos_y for x in pos_x]
    
    log(f"\nTest configuration:")
    log(f"  Lattice: {lattice_type}")
    log(f"  r/a: {r_over_a}")
    log(f"  eps_bg: {eps_bg}")
    log(f"  k0: {k0}")
    log(f"  Band index: {band_index}")
    log(f"  Resolution: {resolution}")
    log(f"  Grid: {n_grid}x{n_grid} = {len(pos_grid)} positions")
    
    # Run BLAZE
    log("\n" + "-"*50)
    log("Running BLAZE...")
    t0 = time.time()
    blaze_results = run_blaze_grid(
        lattice_type, r_over_a, eps_bg, k0, band_index, 
        pos_grid, resolution=resolution, threads=16
    )
    blaze_time = time.time() - t0
    log(f"BLAZE completed in {blaze_time:.2f}s")
    log(f"BLAZE returned {len(blaze_results)} results")
    
    # Run MPB if available
    mpb_results = {}
    if HAS_MPB:
        log("\n" + "-"*50)
        log("Running MPB...")
        t0 = time.time()
        for i, (x, y) in enumerate(pos_grid):
            log(f"  MPB {i+1}/{len(pos_grid)}: pos=({x:.2f}, {y:.2f})")
            result = run_mpb_single(
                lattice_type, r_over_a, eps_bg, k0, band_index,
                (x, y), resolution=resolution
            )
            if result:
                key = (round(x, 6), round(y, 6))
                mpb_results[key] = result
        mpb_time = time.time() - t0
        log(f"MPB completed in {mpb_time:.2f}s")
    
    # Compare results
    log("\n" + "="*70)
    log("COMPARISON RESULTS")
    log("="*70)
    
    rows = []
    
    for x, y in pos_grid:
        key = (round(x, 6), round(y, 6))
        
        row = {
            'pos_x': x,
            'pos_y': y,
        }
        
        # BLAZE data
        if key in blaze_results:
            b = blaze_results[key]
            row['blaze_omega'] = b['omega']
            row['blaze_vg_x'] = b['vg_x']
            row['blaze_vg_y'] = b['vg_y']
            row['blaze_d2_xx'] = b['d2_xx']
            row['blaze_d2_yy'] = b['d2_yy']
            row['blaze_d2_xy'] = b['d2_xy']
            # Store raw bands for debugging
            for i, omega in enumerate(b.get('raw_bands', [])):
                row[f'blaze_k{i}_omega'] = omega
        else:
            row['blaze_omega'] = np.nan
            row['blaze_vg_x'] = np.nan
            row['blaze_vg_y'] = np.nan
            row['blaze_d2_xx'] = np.nan
            row['blaze_d2_yy'] = np.nan
            row['blaze_d2_xy'] = np.nan
        
        # MPB data
        if key in mpb_results:
            m = mpb_results[key]
            row['mpb_omega'] = m['omega']
            row['mpb_vg_x'] = m['vg_x']
            row['mpb_vg_y'] = m['vg_y']
            row['mpb_d2_xx'] = m['d2_xx']
            row['mpb_d2_yy'] = m['d2_yy']
            row['mpb_d2_xy'] = m['d2_xy']
            
            # Compute differences
            if key in blaze_results:
                row['diff_omega'] = b['omega'] - m['omega']
                row['diff_vg_x'] = b['vg_x'] - m['vg_x']
                row['diff_vg_y'] = b['vg_y'] - m['vg_y']
                row['diff_d2_xx'] = b['d2_xx'] - m['d2_xx']
                row['diff_d2_yy'] = b['d2_yy'] - m['d2_yy']
                row['diff_d2_xy'] = b['d2_xy'] - m['d2_xy']
                
                # Relative error
                if abs(m['omega']) > 1e-10:
                    row['rel_err_omega'] = abs(row['diff_omega'] / m['omega'])
        else:
            row['mpb_omega'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print summary
    log("\nBLAZE Results:")
    blaze_cols = ['pos_x', 'pos_y', 'blaze_omega', 'blaze_vg_x', 'blaze_vg_y', 
                  'blaze_d2_xx', 'blaze_d2_yy']
    log(df[blaze_cols].to_string(index=False, float_format='{:.6f}'.format))
    
    if HAS_MPB and mpb_results:
        log("\nMPB Results:")
        mpb_cols = ['pos_x', 'pos_y', 'mpb_omega', 'mpb_vg_x', 'mpb_vg_y',
                    'mpb_d2_xx', 'mpb_d2_yy']
        log(df[mpb_cols].to_string(index=False, float_format='{:.6f}'.format))
        
        log("\nDifferences (BLAZE - MPB):")
        diff_cols = ['pos_x', 'pos_y', 'diff_omega', 'diff_vg_x', 'diff_vg_y',
                     'diff_d2_xx', 'diff_d2_yy']
        if 'diff_omega' in df.columns:
            log(df[diff_cols].to_string(index=False, float_format='{:.6f}'.format))
            
            log("\nSummary Statistics:")
            log(f"  Omega: mean diff = {df['diff_omega'].mean():.6f}, "
                f"max abs diff = {df['diff_omega'].abs().max():.6f}")
            if 'rel_err_omega' in df.columns:
                log(f"  Omega: max rel error = {df['rel_err_omega'].max():.4%}")
            log(f"  vg_x: mean diff = {df['diff_vg_x'].mean():.6f}, "
                f"max abs diff = {df['diff_vg_x'].abs().max():.6f}")
            log(f"  d2_xx: mean diff = {df['diff_d2_xx'].mean():.4f}, "
                f"max abs diff = {df['diff_d2_xx'].abs().max():.4f}")
    
    # Save to CSV
    output_dir = Path(__file__).parent.parent / "runs"
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / "blaze_mpb_comparison.csv"
    df.to_csv(csv_path, index=False)
    log(f"\nSaved comparison data to: {csv_path}")
    
    # Also save raw BLAZE bands for debugging
    raw_bands_rows = []
    for key, b in blaze_results.items():
        row = {'pos_x': key[0], 'pos_y': key[1]}
        for i, omega in enumerate(b.get('raw_bands', [])):
            row[f'k{i}_omega'] = omega
        raw_bands_rows.append(row)
    
    raw_df = pd.DataFrame(raw_bands_rows)
    raw_csv_path = output_dir / "blaze_raw_bands.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    log(f"Saved raw BLAZE bands to: {raw_csv_path}")
    
    log("\n" + "="*70)
    log("DONE")
    log("="*70)
    
    return df


if __name__ == "__main__":
    main()
