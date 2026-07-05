#!/usr/bin/env python3
"""
Phase 6 (Meep): Rigorous Validation with Full-Area Excitation — V3 Multi-Band Pipeline

FEATURES:
1. Validates Phase 3 Envelope Modes using exact Moiré Geometry (Full FDTD).
2. Uses "Universal Super-Bloch Excitation":
   - Source: Exact Initialization via F(R) * u(r).
3. Initialization ONLY:
   - Sets fields at t=0.
   - Lets simulation evolve naturally.
   - Monitors stability (snapshotting).
4. Visualization:
   - Geometry
   - Snapshots in PNG and H5.

USAGE:
    python phasesV3/phase6_meep_validation.py [candidate_id] [mode_index] [scaling_factor]
"""

import os
import sys
import argparse
import math
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import h5py

try:
    import meep as mp
except ImportError:
    print("Warning: meep not installed. Using mock if validation not required.")
    mp = None

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import logic from Phase 5 (reuse geometry tools)
from phasesV3.phase5_meep_v3 import (
    build_monolayer_basis,
    compute_moire_basis,
    lattice_points_in_region,
    log,
    SimulationConfig
)
from common.io_utils import candidate_dir, load_yaml, load_json
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import RegularGridInterpolator

def get_bloch_fields(cdir):
    """Load Phase 1 Bloch fields for source reconstruction."""
    h5_path = cdir / "phase1_multiband_data.h5"
    if not h5_path.exists():
        # Fallback to old name if needed?
        return None
        
    with h5py.File(h5_path, 'r') as hf:
        if 'bloch_fields' not in hf:
             # Try fallback: look for 'fields' or 'bloch_modes'
            return None
        
        # Shape: (Ns1, Ns2, N_bands, Nx, Ny, 3) 
        # Note: N_bands likely matches the number of computed bands in Phase 1
        return hf['bloch_fields'][:]

def get_phase3_data(cdir):
    """Load Phase 3 Mode Data."""
    h5_path = cdir / "phase3_multiband_modes.h5"
    json_path = cdir / "phase3_mode_stats.json"
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Phase 3 data missing: {h5_path}")
        
    # Load Stats
    mode_stats = []
    if json_path.exists():
        mode_stats = load_json(json_path)
    
    # Load H5
    with h5py.File(h5_path, 'r') as hf:
        eigenvalues = hf['eigenvalues'][:]
        # Load all spinor envelopes (n_modes, Ns1, Ns2, N_subspace)
        # Check keys
        if 'F_spinor' in hf:
            F_all = hf['F_spinor'][:]
        elif 'F_envelope' in hf:
            F_all = hf['F_envelope'][:] 
        else:
             # Fallback or reconstruct? 
             # Phase 3 v3 generic saver uses 'F_spinor' or 'eigenvectors'
             # If stored as eigenvectors, we might need reshaping, but let's assume F_spinor exists 
             # based on previous tool output reading.
             if 'eigenvectors' in hf:
                 log("Warning: F_spinor not found, using raw eigenvectors (might require reshaping)")
                 F_all = hf['eigenvectors'][:]
             else:
                 raise KeyError("Could not find F_spinor or eigenvectors in H5")
                 
    return mode_stats, eigenvalues, F_all

def analyze_frequency_isolation(target_idx, eigenvalues):
    """
    Analyze spectral isolation of the target mode.
    Returns:
        target_diff: min distance to nearest neighbor
        sigma_f: required spectral width (standard deviation)
        sigma_t: resulting temporal width
        pulse_duration: estimated total pulse length (6*sigma_t)
    """
    omega_target = eigenvalues[target_idx]
    n_modes = len(eigenvalues)
    
    # Calculate distances to all other modes
    diffs = []
    # Increase tolerance: We don't need to resolve degenerate partners (e.g. splitting < 1e-3)
    # Beating from degenerate modes is slow and acceptable for envelope validation.
    degeneracy_tol = 2e-3 
    
    for i in range(n_modes):
        if i == target_idx:
            continue
        d = abs(eigenvalues[i] - omega_target)
        if d > degeneracy_tol:
            diffs.append(d)
    
    if not diffs:
        # Single mode case or all degenerate
        min_diff = 0.05 * omega_target # 5% bandwidth
        log("  Warning: No distinct neighbors found. Using loose isolation.")
    else:
        min_diff = min(diffs)
        
    # RELAXED ISOLATION:
    # We want neighbor at > 2*sigma_f (instead of 3 or 4)
    # This admits some spectral leakage but significantly shortens the pulse.
    sigma_f = min_diff / 2.0
    
    # Gaussian pulse relations:
    # sigma_t = 1 / (2 * pi * sigma_f)
    
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    
    # Total effective pulse duration
    # Cutoff at 5 sigma_t captures >99.9% of energy
    pulse_duration = 6.0 * sigma_t # Reduced from 10.0
    
    return min_diff, sigma_f, sigma_t, pulse_duration


def find_latest_run_dir(base_name="phase0_mpb_v3"):
    """Find the latest run directory in runsV3 matching the base name."""
    runs_dir = PROJECT_ROOT / "runsV3"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
        
    candidates = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(base_name)])
    if not candidates:
        raise FileNotFoundError(f"No runs found matching {base_name} in {runs_dir}")
        
    return candidates[-1]

def run_phase6_validation(candidate_id, mode_idx, time_mult=2.0):
    # Find latest run automatically
    run_dir = find_latest_run_dir()
    cdir = candidate_dir(run_dir, candidate_id)
    
    log(f"=== Phase 6: MEEP Validation (Initialization Only) for Candidate {candidate_id} Mode {mode_idx} ===")
    log(f"Run Directory: {run_dir.name}")

    # 0. Load Phase 3 Data
    stats, evals, F_all = get_phase3_data(cdir)
    
    # 1. Load Data
    p0_meta = load_json(cdir / "phase0_meta.json") # Geometry metadata
    
    # Load Config (New)
    p6_config_path = PROJECT_ROOT / "configsV3" / "phase6_meep.yaml"
    if p6_config_path.exists():
        p6_conf = load_yaml(p6_config_path)
    else:
        p6_conf = {} 
        log("Warning: phase6_meep.yaml not found. Using defaults.")
    
    # Config params
    res = p6_conf.get('resolution', 20)
    save_stride = p6_conf.get('outputs', {}).get('snapshot_stride', 4) 
    
    # Force disable video to prevent hanging, unless explicitly requested in config (default off here)
    enable_video = p6_conf.get('outputs', {}).get('enable_video', False)
    
    use_periodic_bc = p6_conf.get('use_periodic_bc', False)
    
    # 2. Get Targets
    if mode_idx < len(stats):
        omega_target = stats[mode_idx]['omega']
    else:
        raise ValueError("Mode index out of range in stats")
        
    log(f"Target Mode {mode_idx}: Freq = {omega_target:.6f}")
    
    # Simulation Time - Just run for a few periods to prove stability
    period = 1.0 / omega_target
    sim_time = 50.0 * period # Run for 50 periods
    log(f"  Simulation Time = {sim_time:.2f} (50 periods)")
    
    # 3. Build Geometry
    a0 = p0_meta.get('a', 1.0)
    theta = p0_meta['theta_deg']
    theta_rad = math.radians(theta)
    
    r_h = p0_meta.get('r_over_a', 0.28)
    eps_slab = p0_meta.get('eps_bg', 12.0)
    lattice_type = p0_meta.get('lattice_type', 'triangular')
    h_slab = mp.inf
    
    padding_factor = p6_conf.get('padding_factor', 0.2)
    pml_thickness = p6_conf.get('pml_thickness', 1.0)
    
    # Determine basic vectors
    B_mono = build_monolayer_basis(lattice_type, a0)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    
    # Load Bloch Fields
    bloch_data = get_bloch_fields(cdir)
    if bloch_data is None:
        raise RuntimeError("Phase 6 Validation requires Bloch fields (Phase 1 data) for exact initialization.")
    
    # Cell Size Calculation
    L1 = B_moire[:, 0]
    L2 = B_moire[:, 1]
    
    symmetries = []
    R_global = np.eye(2)
    
    if use_periodic_bc:
        log("Geometry: Using Periodic Boundaries (Rectangular Supercell Strategy)")
        theta_L1 = math.atan2(L1[1], L1[0])
        c_rot = math.cos(-theta_L1)
        s_rot = math.sin(-theta_L1)
        R_global = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
        
        L1_rot = R_global @ L1
        L2_rot = R_global @ L2
        
        if lattice_type == 'square':
            sx = abs(L1_rot[0])
            sy = abs(L2_rot[1])
            symmetries = [] 
        else: # Triangular
            sx = abs(L1_rot[0])
            sy = 2.0 * abs(L2_rot[1])
            symmetries = []

        cell_size = mp.Vector3(sx, sy, 0)
        k_point = mp.Vector3(0,0,0)
        pml_layers = []
        
        def to_sim_coords(p_cart):
             p_rot = R_global @ p_cart
             return mp.Vector3(p_rot[0], p_rot[1], 0)
        
        diag = math.hypot(sx, sy)
        sx_fetch = diag * 1.2
        sy_fetch = diag * 1.2
        fetch_bounds = (-sx_fetch/2, sx_fetch/2, -sy_fetch/2, sy_fetch/2)

    else:
        # Standard Rectangular Cell with PML
        xs = [0, L1[0], L2[0], (L1+L2)[0]]
        ys = [0, L1[1], L2[1], (L1+L2)[1]]
        w_x0 = max(xs) - min(xs)
        w_y0 = max(ys) - min(ys)
        
        sx = w_x0 * (1.0 + 2*padding_factor)
        sy = w_y0 * (1.0 + 2*padding_factor)
        
        cell_size = mp.Vector3(sx, sy, 0)
        pml_layers = [mp.PML(pml_thickness)]
        k_point = False 
        
        R_global = np.eye(2)
        def to_sim_coords(p_cart):
            return mp.Vector3(p_cart[0], p_cart[1], 0)
            
        fetch_bounds = (-sx/2, sx/2, -sy/2, sy/2)

    log(f"  Cell Size: {sx:.2f} x {sy:.2f}")

    # Build Holes
    def rot_mat(ang):
        return np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        
    R1 = rot_mat(-theta_rad/2)
    R2 = rot_mat(+theta_rad/2)
    
    a1_bottom = R1 @ B_mono[:,0]
    a2_bottom = R1 @ B_mono[:,1]
    
    a1_top = R2 @ B_mono[:,0]
    a2_top = R2 @ B_mono[:,1]
    
    # Generate points
    pts_bottom = lattice_points_in_region(a1_bottom, a2_bottom, fetch_bounds, padding=1.0)
    pts_top = lattice_points_in_region(a1_top, a2_top, fetch_bounds, padding=1.0)
    
    # Create Geometry
    geometry = [
        mp.Block(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_slab)
        )
    ]
    
    air = mp.Medium(epsilon=1.0)
    all_pts = np.vstack((pts_bottom, pts_top))
    
    for p in all_pts:
        center_vec = to_sim_coords(np.array(p))
        if abs(center_vec.x) < sx/2 + a0 and abs(center_vec.y) < sy/2 + a0:
             geometry.append(mp.Cylinder(radius=r_h*a0, center=center_vec, height=mp.inf, material=air))
        
    # No Sources - We use Initial Fields
    sources = []
    
    # Simulation
    mp.verbosity(0)
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=res,
        default_material=air,
        k_point=k_point,
        force_complex_fields=True,
        symmetries=symmetries,
        split_chunks_evenly=False,
        eps_averaging=False,
        Courant=0.3, 
    )
    sim.verbose = False
    
    # Output Dir
    out_dir = cdir / f"phase6_val_m{mode_idx}"
    out_dir.mkdir(exist_ok=True)
    
    # A) MEEP Init
    sim.init_sim()
    
    # --- EXACT INITIALIZATION ---
    log("  Generating Exact Initial Fields (Vectorized on Rank 0)...")
    
    # Get Grid Coords from Sim
    meta = sim.get_array_metadata(center=mp.Vector3(0,0,0), size=cell_size)
    
    if mp.am_master():
        # Clean up old init file
        init_h5_path = out_dir / "init_fields_manual.h5"
        if init_h5_path.exists(): init_h5_path.unlink()

        x_coords = meta[0] 
        y_coords = meta[1] 
        
        # Grid for Interpolation
        X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij') 
        flat_sh = X_mesh.shape
        X_flat = X_mesh.ravel()
        Y_flat = Y_mesh.ravel()
        N_pts = len(X_flat)
        
        # --- RECONSTRUCTION LOGIC ---
        # 1. Envelope F
        F_env = F_all[mode_idx]
        N_sub = F_env.shape[-1]
        ns1_env, ns2_env = F_env.shape[:2]
        
        # FIX: Center the envelope grid so that (0,0,0) in Sim matches Center of Envelope
        # Assume F_env is defined on unit cell.
        # We map s1, s2 to [-0.5, 0.5] if the simulation is centered at 0 and contains the cell.
        s_env_grid = (np.linspace(-0.5, 0.5, ns1_env), np.linspace(-0.5, 0.5, ns2_env))
        
        interp_F = []
        for n in range(N_sub):
            r_int = RegularGridInterpolator(s_env_grid, F_env[:,:,n].real, bounds_error=False, fill_value=0)
            i_int = RegularGridInterpolator(s_env_grid, F_env[:,:,n].imag, bounds_error=False, fill_value=0)
            interp_F.append((r_int, i_int))
        
        # 2. Bloch u
        # Bloch u(r; R) is likely also periodic in R or slowly varying?
        # If u(r; R) was computed on grid [0,1], we assume it maps to [-0.5, 0.5] as well if standard centering used.
        # Microscopic u(r) is periodic in [0,1].
        nr1, nr2, _, uNx, uNy, _ = bloch_data.shape
        reg_grid = (np.linspace(-0.5, 0.5, nr1), np.linspace(-0.5, 0.5, nr2))
        u_grid_x = np.linspace(0, 1, uNx)
        u_grid_y = np.linspace(0, 1, uNy)
        
        interp_u = []
        target_comps = [0, 1] 
        
        # Subspace mapping logic...
        try: 
                with h5py.File(cdir / "phase1_multiband_data.h5", 'r') as hf:
                    sub_bands = hf.attrs["subspace_bands"]
                    all_bands_p1 = hf.attrs["all_bands"]
                    band_indices_map = [np.where(all_bands_p1 == sb)[0][0] for sb in sub_bands]
        except:
            band_indices_map = list(range(N_sub))
            
        for bn_idx in band_indices_map:
            band_interps = {}
            for comp in target_comps:
                d4 = bloch_data[:, :, bn_idx, :, :, comp]
                r_int = RegularGridInterpolator((reg_grid[0], reg_grid[1], u_grid_x, u_grid_y), d4.real, bounds_error=False, fill_value=0)
                i_int = RegularGridInterpolator((reg_grid[0], reg_grid[1], u_grid_x, u_grid_y), d4.imag, bounds_error=False, fill_value=0)
                band_interps[comp] = (r_int, i_int)
            interp_u.append(band_interps)
        
        # 3. Transform Coordinates
        inv_B_moire = np.linalg.inv(B_moire)
        P_vec = np.vstack((X_flat, Y_flat)) 
        S_moire = inv_B_moire @ P_vec 
        
        # S coordinates - NO MODULO for Envelope (assuming it's centered single cell)
        # If it's a superlattice mode, modulo would be needed, but 'squished' implies mismatch.
        # Centering at 0 means valid range is [-0.5, 0.5].
        # Any S outside this range will be 0 (fill_value).
        s1_flat = S_moire[0, :]
        s2_flat = S_moire[1, :]
        
        # Microscopic coords - periodic in Monolayer Cell
        inv_B_mono = np.linalg.inv(B_mono)
        S_mono = inv_B_mono @ P_vec
        ux_flat = np.mod(S_mono[0, :], 1.0)
        uy_flat = np.mod(S_mono[1, :], 1.0)
        
        pts_F = np.stack((s1_flat, s2_flat), axis=-1)
        pts_u = np.stack((s1_flat, s2_flat, ux_flat, uy_flat), axis=-1)
        
        # 4. Evaluate
        Ex_acc = np.zeros(N_pts, dtype=np.complex128)
        Ey_acc = np.zeros(N_pts, dtype=np.complex128)
        
        log(f"    Evaluating reconstruction on {N_pts} points...")
        
        for n in range(N_sub):
            f_r = interp_F[n][0](pts_F)
            f_i = interp_F[n][1](pts_F)
            F_val = f_r + 1j*f_i
            
            ur_ex = interp_u[n][0][0](pts_u)
            ui_ex = interp_u[n][0][1](pts_u)
            u_ex = ur_ex + 1j*ui_ex
            
            ur_ey = interp_u[n][1][0](pts_u)
            ui_ey = interp_u[n][1][1](pts_u)
            u_ey = ur_ey + 1j*ui_ey
            
            Ex_acc += F_val * u_ex
            Ey_acc += F_val * u_ey
        
        # Write Result
        with h5py.File(init_h5_path, 'w') as hf_init:
            hf_init.create_dataset('ex_r', data=Ex_acc.reshape(flat_sh).real)
            hf_init.create_dataset('ex_i', data=Ex_acc.reshape(flat_sh).imag)
            hf_init.create_dataset('ey_r', data=Ey_acc.reshape(flat_sh).real)
            hf_init.create_dataset('ey_i', data=Ey_acc.reshape(flat_sh).imag)
    
    # Barrier
    mp.all_wait()
    
    # Load and Set Fields
    init_h5_path = out_dir / "init_fields_manual.h5"
    with h5py.File(init_h5_path, 'r') as hf:
            ex_r = hf['ex_r'][:]
            ex_i = hf['ex_i'][:]
            ey_r = hf['ey_r'][:]
            ey_i = hf['ey_i'][:]
            
    Ex_full = ex_r + 1j*ex_i
    Ey_full = ey_r + 1j*ey_i
    
    # Grid metadata for lookup
    x0 = meta[0][0]
    y0 = meta[1][0]
    dx = meta[0][1] - meta[0][0] if len(meta[0]) > 1 else 1.0
    dy = meta[1][1] - meta[1][0] if len(meta[1]) > 1 else 1.0
    Nx = len(meta[0])
    Ny = len(meta[1])
    
    def field_func(arr, v):
            i = int(round((v.x - x0)/dx))
            j = int(round((v.y - y0)/dy))
            if i < 0: i = 0
            elif i >= Nx: i = Nx-1
            if j < 0: j = 0
            elif j >= Ny: j = Ny-1
            return arr[i, j]
            
    log("    Setting Ex...")
    sim.fields.initialize_field(mp.Ex, lambda v: field_func(Ex_full, v))
    log("    Setting Ey...")
    sim.fields.initialize_field(mp.Ey, lambda v: field_func(Ey_full, v))
    
    mp.all_wait()
    
    # Plot Geometry with Final Overlay
    plt.figure(figsize=(10,10))
    sim.plot2D()
    plt.title(f"MEEP Geometry + Init (M{mode_idx})")
    plt.savefig(out_dir / "geometry.png")
    plt.close()
    
    # 6. Run Simulation
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    field_h5_path = out_dir / "field_evolution.h5"

    snapshot_interval = 10.0 * period # Save every 10 periods
    
    def step_function(sim):
        # Simply print progress
        t = sim.meep_time()
        if mp.am_master():
             print(f"t={t:.2f} ...", end='\r')

    # Save initial state
    sim.run(until=0.0)
    hz = sim.get_array(component=mp.Hz, cmplx=True)
    if mp.am_master():
        with h5py.File(field_h5_path, 'w') as hf:
             hf.create_dataset("t_0.00", data=hz[::save_stride, ::save_stride], compression="gzip")
    
    log(f"Starting Run... (saving every {snapshot_interval}) without video.")
    
    # Loop manually to avoid "hanging" in complex callbacks
    t_curr = 0.0
    while t_curr < sim_time:
        sim.run(until=snapshot_interval)
        t_curr = sim.meep_time()
        
        # Save Snapshot_
        hz = sim.get_array(component=mp.Hz, cmplx=True)
        if mp.am_master():
            with h5py.File(field_h5_path, 'a') as hf:
                 grp = f"t_{t_curr:.2f}"
                 hf.create_dataset(grp, data=hz[::save_stride, ::save_stride], compression="gzip")
            
            # Save PNG
            hz_real = hz.real[::save_stride, ::save_stride]
            plt.imsave(snapshot_dir / f"snap_{t_curr:06.1f}.png", hz_real.T, cmap='RdBu', origin='lower')
            
    log(f"Done. Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 MEEP Validation")
    parser.add_argument("candidate_id", type=int, help="Candidate ID")
    parser.add_argument("mode_index", type=int, help="Mode Index to validate")
    parser.add_argument("--scale", type=float, default=2.0, help="Sim time multiplier (x PulseDuration)")
    
    args = parser.parse_args()
    
    if mp is None:
        log("MEEP not found. Exiting.")
        sys.exit(0)
        
    run_phase6_validation(args.candidate_id, args.mode_index, args.scale)
