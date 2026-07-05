#!/usr/bin/env python3
"""
Phase 7: Full Moiré Cavity Launch

Initializes a Meep FDTD simulation with the exact reconstructed Moiré field
from Phase 4 and lets it evolve to verify mode stability.

KEY FEATURES:
1. Reuses `phasesV3/phase4_field_reconstruction.py` logic to evaluate E-field.
2. Initializes simulation with Ex, Ey (TE) or Ez (TM) from reconstruction.
3. Uses PML (Safe Cavity Strategy) to absorb reconstruction noise at boundaries.
4. Monitors the mode evolution.

USAGE:
    python phasesV3/phase7_meep_launch.py [candidate_id] [mode_index]
"""

import os
import sys
import argparse
import math
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import meep as mp
except ImportError:
    print("Warning: meep not installed.")
    mp = None

# Imports from our pipeline
from phasesV3.phase5_meep_v3 import (
    build_monolayer_basis,
    compute_moire_basis,
    lattice_points_in_region,
    log
)
from phasesV3 import phase4_field_reconstruction as p4
from common.io_utils import candidate_dir, load_json, load_yaml

def run_phase7_launch(candidate_id, mode_idx, use_periodic=False):
    # 0. Setup Paths and Config
    run_dir = p4.find_latest_run_dir()
    cdir = p4.candidate_dir(run_dir, candidate_id)
    suffix = "_periodic" if use_periodic else "_pml"
    out_dir = cdir / f"phase7_mode{mode_idx}{suffix}"
    out_dir.mkdir(exist_ok=True)
    
    log(f"=== Phase 7: Launching Mode {mode_idx} in Meep ===")
    log(f"Candidate: {candidate_id}")
    log(f"Output: {out_dir}")
    log(f"Boundary Conditions: {'PERIODIC' if use_periodic else 'PML'}")
    
    # Load Phase 0 Metadata
    p0_meta = load_json(cdir / "phase0_meta.json")
    a0 = p0_meta.get('a', 1.0)
    theta_deg = p0_meta['theta_deg']
    theta_rad = math.radians(theta_deg)
    r_h = p0_meta.get('r_over_a', 0.28)
    eps_bg = p0_meta.get('eps_bg', 12.0)
    lattice_type = p0_meta.get('lattice_type', 'triangular')
    
    # Load Phase 1 & 3 Data via Phase 4 helpers (MASTER ONLY to save RAM)
    bloch_fields = None
    F_spinor = None
    band_indices = None
    target_freq = 0.0
    
    if mp.am_master():
        bloch_fields, subspace_bands, all_bands = p4.load_phase1_bloch_fields(cdir)
        F_spinor, eigenvalues, mode_stats = p4.load_phase3_envelopes(cdir)
        band_indices = p4.get_subspace_band_indices(subspace_bands, all_bands)
        
        target_freq = eigenvalues[mode_idx]
        if mode_stats and mode_idx < len(mode_stats):
            target_freq = mode_stats[mode_idx].get('omega', target_freq)
            
    # Broadcast critical scalar metadata
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        target_freq = comm.bcast(target_freq, root=0)
    except ImportError:
        pass # Serial mode typically doesn't need explicit bcast if not using mpi4py
    
    log(f"Target Frequency: {target_freq:.6f} (Period: {1.0/target_freq:.2f})")
    
    # 1. Geometry Construction
    # Determine Bases
    B_mono = build_monolayer_basis(lattice_type, a0)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    
    # Supercell vectors (Original)
    L1_orig = B_moire[:, 0]
    L2_orig = B_moire[:, 1]
    
    # Global Rotation for Simulation Domain Alignment
    # If Periodic: Rotate so L1 aligns with X-axis (+X) to allow Rectangular Supercell approximation.
    sim_rot_angle = 0.0
    if use_periodic:
        # Calculate angle of L1
        sim_rot_angle = -math.atan2(L1_orig[1], L1_orig[0])
        log(f"Aligning L1 to X-axis (Rot: {math.degrees(sim_rot_angle):.2f} deg)")
    
    c_rot, s_rot = math.cos(sim_rot_angle), math.sin(sim_rot_angle)
    R_align = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    
    # Update L1, L2 to Simulation Frame
    L1 = R_align @ L1_orig
    L2 = R_align @ L2_orig
    
    # IMPORTANT: Keep B_moire and B_mono in the ORIGINAL frame for reconstruction.
    # The Bloch fields and envelopes were computed in that frame.
    # We only use L1, L2 (rotated) for cell sizing and geometry placement.
    B_moire_orig = B_moire.copy()  # Save original for reconstruction
    B_mono_orig = B_mono.copy()
    
    # Determine Cell Size and Geometry Bounds
    if use_periodic:
        # Case A: Rectangular Supercell for Periodic BCs
        # After rotation, L1 is along X and L2 is along Y (for square moiré)
        # or has components in both (for triangular). Use actual components.
        lx = abs(L1[0])  # x-extent of L1 (aligned to x)
        ly = abs(L2[1])  # y-extent of L2
        
        cell_size = mp.Vector3(lx, ly, 0)
        pml_thickness = 0.0
        boundary_layers = []
        k_point = mp.Vector3(0,0,0)
        
        log(f"Using Rectangular Supercell: {lx:.3f} x {ly:.3f}")
        
        # Fill the box with lattice points
        fetch_bounds_x = (-lx/2 - a0, lx/2 + a0)
        fetch_bounds_y = (-ly/2 - a0, ly/2 + a0)
        
    else:
        # Case B: PML Cavity (Original Logic)
        xs = [0, L1_orig[0], L2_orig[0], (L1_orig + L2_orig)[0]]
        ys = [0, L1_orig[1], L2_orig[1], (L1_orig + L2_orig)[1]]
        
        width_x = max(xs) - min(xs)
        width_y = max(ys) - min(ys)
        
        padding = 0.2 * max(width_x, width_y)
        pml_thickness = 1.0
        boundary_layers = [mp.PML(pml_thickness)]
        
        sx = width_x + 2*padding + 2*pml_thickness
        sy = width_y + 2*padding + 2*pml_thickness
        cell_size = mp.Vector3(sx, sy, 0)
        k_point = None
        
        fetch_bounds_x = (-sx/2 - a0, sx/2 + a0)
        fetch_bounds_y = (-sy/2 - a0, sy/2 + a0)
    
    # Build Holes (Geometry)
    # We populate the entire simulation cell with the lattice
    geometry = [
        mp.Block(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_bg)
        )
    ]
    
    # 1. Geometry Generation
    # Rotate Monolayer Basis by (theta/2 + sim_rot_angle)
    def rot_mat(ang):
        return np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    
    # Total rotation for layers: original twist ± theta/2, PLUS alignment rotation
    R1 = rot_mat(-theta_rad/2 + sim_rot_angle)
    R2 = rot_mat(+theta_rad/2 + sim_rot_angle)
    
    # Use ORIGINAL monolayer basis (not rotated) since R1/R2 already include sim_rot_angle
    a1_bottom = R1 @ B_mono_orig[:,0]
    a2_bottom = R1 @ B_mono_orig[:,1]
    a1_top = R2 @ B_mono_orig[:,0]
    a2_top = R2 @ B_mono_orig[:,1]
    
    air = mp.Medium(epsilon=1.0)
    
    # Generate points
    # Need to cover the whole cell
    # Note: Phase 5 `lattice_points_in_region` assumes fetching by basis expansion or valid check
    # Let's assume we just need to fill the box.
    pts_bottom = lattice_points_in_region(a1_bottom, a2_bottom, (fetch_bounds_x[0], fetch_bounds_x[1], fetch_bounds_y[0], fetch_bounds_y[1]), padding=1.0)
    pts_top = lattice_points_in_region(a1_top, a2_top, (fetch_bounds_x[0], fetch_bounds_x[1], fetch_bounds_y[0], fetch_bounds_y[1]), padding=1.0)
    
    all_pts = np.vstack((pts_bottom, pts_top))
    log(f"Generated {len(all_pts)} holes")
    
    for p in all_pts:
        geometry.append(mp.Cylinder(radius=r_h*a0, center=mp.Vector3(p[0], p[1], 0), height=mp.inf, material=air))
        
    # 2. Initialize Simulation
    resolution = 20 # Pixels per `a`
    
    sim_kwargs = {
        'cell_size': cell_size,
        'boundary_layers': boundary_layers,
        'geometry': geometry,
        'resolution': resolution,
        'default_material': air,
        'force_complex_fields': True
    }
    
    if k_point:
        sim_kwargs['k_point'] = k_point
    
    sim = mp.Simulation(**sim_kwargs)
    
    sim.init_sim()
    
    # 3. Field Initialization
    log("Computing Reconstruction on Meep Grid...")
    
    # Get grid axes from Meep
    # Note: get_array_metadata for center/size args describes the global array
    # For Non-Orthogonal Lattice, get_array_metadata returns axes in Lattice Basis?
    # Or Cartesian? Usually Cartesian coordinates of the grid points.
    
    # If using Lattice, we request data for the Unit Cell.
    # mp.Volume(center=..., size=...) defines a box.
    # If we pass `vol=None` to get_array on a periodic simulation, it returns the unit cell data.
    
    # We always use orthogonal cell_size now (Rectangular Supercell or PML)
    grid_meta = sim.get_array_metadata(center=mp.Vector3(0,0,0), size=cell_size)
        
    x_coords = grid_meta[0]
    y_coords = grid_meta[1]
    
    # Grid Coordinates logic
    u_coords = x_coords
    v_coords = y_coords
    
    # Meshgrid (always Cartesian - simulation frame)
    UU, VV = np.meshgrid(u_coords, v_coords, indexing='ij')
    shape = UU.shape
    
    # Simulation frame coordinates
    X_sim_flat = UU.flatten()
    Y_sim_flat = VV.flatten()
    
    # Map simulation-frame coordinates BACK to original (physical) frame
    # for field reconstruction. R_align rotated phys->sim, so R_align^T maps sim->phys.
    R_inv = R_align.T
    X_flat = R_inv[0, 0] * X_sim_flat + R_inv[0, 1] * Y_sim_flat
    Y_flat = R_inv[1, 0] * X_sim_flat + R_inv[1, 1] * Y_sim_flat
    
    plot_extent = [u_coords[0], u_coords[-1], v_coords[0], v_coords[-1]]
    
    # Helper to map position to grid index for initialize_field
    x_min, y_min = u_coords[0], v_coords[0]
    dx_grid = u_coords[1] - u_coords[0]
    dy_grid = v_coords[1] - v_coords[0]
    Nx_grid, Ny_grid = len(u_coords), len(v_coords)
    
    def make_field_func(grid_data):
        def _func(p):
            # p is Cartesian Vector3
            # Map p to grid index (Cartesian)
            # grid_data matches x_coords, y_coords (which are Cartesian)
            ix = int(round((p.x - x_min) / dx_grid))
            iy = int(round((p.y - y_min) / dy_grid))
            
            if use_periodic:
                # Wrap for Periodic BCs
                # Note: With Rectangular Supercell, p should wrap naturally via ix % Nx
                ix = ix % Nx_grid
                iy = iy % Ny_grid
            else:
                # Clamp for PML/Open
                if ix < 0: ix = 0
                if ix >= Nx_grid: ix = Nx_grid - 1
                if iy < 0: iy = 0
                if iy >= Ny_grid: iy = Ny_grid - 1
                
            return grid_data[ix, iy]
        return _func
    
    # 5. Parallel Field Reconstruction
    # Only Master loads the heavy Bloch fields, so only Master can reconstruct.
    # We reconstruct on Master, then broadcast the (small) Grid Field to all nodes.
    
    is_tm = False
    Ez_grid = None
    Ex_grid = None
    Ey_grid = None
    
    if mp.am_master():
        # Heuristic: Check Bloch Ez
        comp_z_max = np.max(np.abs(bloch_fields[:, :, :, :, :, 2]))
        is_tm = comp_z_max > 1e-5
        
        # Coordinates are already in original (physical) frame via R_inv mapping above.
        # Reconstruct using ORIGINAL bases (B_moire_orig, B_mono_orig).
        
        if is_tm:
            log("Mode is TM (Ez dominant)")
            Ez_flat_phys = p4.reconstruct_full_field_for_meep(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                B_moire=B_moire_orig,
                B_mono=B_mono_orig,
                target_coords=(X_flat, Y_flat),
                component=2,
                normalize_bloch=True
            )
            # Ez is a scalar (pseudoscalar) — no vector rotation needed
            Ez_grid = Ez_flat_phys.reshape(shape)
            
        else:
            log("Mode is TE (Ex, Ey dominant)")
            log("  Reconstructing Ex, Ey in physical frame...")
            
            Ex_phys = p4.reconstruct_full_field_for_meep(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                B_moire=B_moire_orig,
                B_mono=B_mono_orig,
                target_coords=(X_flat, Y_flat),
                component=0,
                normalize_bloch=True
            )
            
            Ey_phys = p4.reconstruct_full_field_for_meep(
                mode_idx=mode_idx,
                F_spinor=F_spinor,
                bloch_fields=bloch_fields,
                band_indices=band_indices,
                B_moire=B_moire_orig,
                B_mono=B_mono_orig,
                target_coords=(X_flat, Y_flat),
                component=1,
                normalize_bloch=True
            )
            
            # Rotate VECTOR components from physical frame to simulation frame.
            # R_align maps physical->sim, so: E_sim = R_align @ E_phys
            Ex_flat = R_align[0, 0] * Ex_phys + R_align[0, 1] * Ey_phys
            Ey_flat = R_align[1, 0] * Ex_phys + R_align[1, 1] * Ey_phys
            
            Ex_grid = Ex_flat.reshape(shape)
            Ey_grid = Ey_flat.reshape(shape)
            
            log(f"  Field stats: max|Ex|={np.max(np.abs(Ex_grid)):.4e}, max|Ey|={np.max(np.abs(Ey_grid)):.4e}")
    
    # Broadcast Data to all nodes
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        is_tm = comm.bcast(is_tm, root=0)
        
        if is_tm:
            Ez_grid = comm.bcast(Ez_grid, root=0)
            log("Injecting Ez (Broadcasted)...")
            sim.initialize_field(mp.Ez, make_field_func(Ez_grid))
        else:
            Ex_grid = comm.bcast(Ex_grid, root=0)
            Ey_grid = comm.bcast(Ey_grid, root=0)
            log("Injecting Ex, Ey (Broadcasted)...")
            sim.initialize_field(mp.Ex, make_field_func(Ex_grid))
            sim.initialize_field(mp.Ey, make_field_func(Ey_grid))
            
    except ImportError:
        # Fallback for Serial Mode (No MPI)
        # In serial, mp.am_master() is True, so arrays are already set.
        pass

    # 4. Validation Plots (Before Evolution)
    
    # === PLOT: 1. Meep Geometry Validation ===
    if mp.am_master():
        log("Saving Meep Geometry Plot...")
        plt.figure(figsize=(10,8))
        sim.plot2D()
        plt.title("Meep Internal Geometry View")
        plt.savefig(out_dir / "check_meep_geometry.png")
        plt.close()

    # === PLOT: 2. Custom Overlay Validation ===
    if mp.am_master():
        log("Generating Overlay Validation Plot...")
        plt.figure(figsize=(10,8))
        
        # Plot Field Magnitude
        if is_tm:
            f_mag = np.abs(Ez_grid)
        else:
            f_mag = np.sqrt(np.abs(Ex_grid)**2 + np.abs(Ey_grid)**2)
            
        plt.imshow(f_mag.T, origin='lower', cmap='hot', extent=plot_extent, alpha=0.9)
        plt.colorbar(label="|E|")
        
        # Overlay Holes (Centers)
        # We have `all_pts` from geometry construction
        xh = all_pts[:,0]
        yh = all_pts[:,1]
        
        # Filter holes outside plot extent to keep plot clean (optional, but scatter handles it)
        plt.scatter(xh, yh, s=2, c='cyan', marker='.', alpha=0.5, label='Holes')
        
        plt.xlim(plot_extent[0], plot_extent[1])
        plt.ylim(plot_extent[2], plot_extent[3])
        plt.title("Custom Overlay: Field + Lattice Centers")
        plt.legend()
        plt.savefig(out_dir / "check_overlay.png")
        plt.close()
    
    # Plot initial state standard (Corrected Axis)
    # Gather data (All nodes must participate in get_array to avoid MPI hang)
    # If using Lattice, simple get_array works for the unit cell
    f_init = None
    if is_tm:
        f_init = sim.get_array(component=mp.Ez, cmplx=False)
    else:
        fx = sim.get_array(component=mp.Ex, cmplx=False)
        fy = sim.get_array(component=mp.Ey, cmplx=False)
        if mp.am_master():
             f_init = np.sqrt(fx**2 + fy**2)
             
    # Store 'Exact' initial field for Overlap Diagnostic
    # Note: We need complex field for correct phase overlap
    E_ref_complex = None
    if is_tm:
        E_ref_complex = sim.get_array(component=mp.Ez, cmplx=True)
    else:
        # For TE, we technically need vector overlap: (Ex* . Ex + Ey* . Ey)
        # We'll store both components
        Ex_ref = sim.get_array(component=mp.Ex, cmplx=True)
        Ey_ref = sim.get_array(component=mp.Ey, cmplx=True)
        
    # Pre-calculate norm of reference
    norm_ref = 0.0
    if mp.am_master():
        if is_tm:
            norm_ref = np.sum(np.abs(E_ref_complex)**2)
        else:
            norm_ref = np.sum(np.abs(Ex_ref)**2 + np.abs(Ey_ref)**2)
             
    if mp.am_master():
        plt.figure(figsize=(10,8))
        label = "Ez" if is_tm else "|E|"
        plt.imshow(f_init.T, origin='lower', cmap='RdBu' if is_tm else 'hot', extent=plot_extent)
        plt.colorbar(label=label)
        plt.title(f"Initial State (t=0) - {label}")
        plt.savefig(out_dir / "initial_state.png")
        plt.close()
        
    # Calculate Initial Energy
    # Step a tiny amount to ensure Meep has initialized the fields on grid
    sim.run(until=sim.fields.dt)
    sim.fields.synchronize_magnetic_fields()
    # Explicitly specify volume to avoid ambiguity
    E_init = sim.electric_energy_in_box(center=mp.Vector3(0,0,0), size=cell_size)
    log(f"Initial Electric Energy: {E_init:.6e}")

    # 4. Evolution
    log("Starting Time Evolution...")
    
    # Run
    T_period = 1/target_freq
    run_time = 30 * T_period # Run for 30 periods
    
    # Animation setup
    field_history = []
    
    overlap_history = []
    energy_history = []
    time_points = []
    
    def record_diagnostics(sim):
        current_time = sim.meep_time()
        
        # 1. Visualization Sampling (Every 1/10 period)
        # All nodes participate in gathering (MPI requirement)
        if is_tm:
             dat = sim.get_array(component=mp.Ez, cmplx=False)
        else:
             dat = sim.get_array(component=mp.Ex, cmplx=False)
             
        if mp.am_master() and dat is not None:
             # Downsample for history
             field_history.append(dat[::2, ::2])
             sys.stdout.write(f"t = {current_time:.2f} / {run_time:.2f}\r")
        
        # 2. Overlap Diagnostic (Every step? Or every few steps?)
        # Let's do it every 5% of a period to get smooth curves, or every step is too heavy?
        # User requested "Export these values every 100 time steps".
        # sim.meep_time() is continuous.
        # We can use a counter or just check if we are close to interval.
        # But `record_diagnostics` is called by `mp.at_every(T_period / 10, ...)`.
        # So it runs ~10 times per period. This is frequent enough.
        
        if mp.am_master():
             # Only Master computes overlap
             # Note: We used `get_array` above which is real. We need complex for overlap.
             pass
        
        # Need complex fields for overlap
        ov_val = 0.0
        if is_tm:
             Ec = sim.get_array(component=mp.Ez, cmplx=True)
             if mp.am_master():
                 # Overlap <E_ref | E_curr>
                 inner = np.sum(np.conj(E_ref_complex) * Ec)
                 norm_curr = np.sum(np.abs(Ec)**2)
                 ov_val = np.abs(inner) / np.sqrt(norm_ref * norm_curr)
        else:
             Exc = sim.get_array(component=mp.Ex, cmplx=True)
             Eyc = sim.get_array(component=mp.Ey, cmplx=True)
             if mp.am_master():
                 inner = np.sum(np.conj(Ex_ref) * Exc + np.conj(Ey_ref) * Eyc)
                 norm_curr = np.sum(np.abs(Exc)**2 + np.abs(Eyc)**2)
                 ov_val = np.abs(inner) / np.sqrt(norm_ref * norm_curr)
        
        E_curr_total = sim.electric_energy_in_box(center=mp.Vector3(0,0,0), size=cell_size)
        
        if mp.am_master():
             overlap_history.append(ov_val)
             energy_history.append(E_curr_total)
             time_points.append(current_time)

    # Run
    # Note: at_every calls the function.
    sim.run(mp.at_every(T_period / 10, record_diagnostics), until=run_time)
    
    # Calculate Final Energy
    E_final = sim.electric_energy_in_box(center=mp.Vector3(0,0,0), size=cell_size)
    log(f"\nSimulation Complete.")
    log(f"Final Electric Energy: {E_final:.6e}")
    if E_init > 1e-12:
        log(f"Energy Ratio (Final/Initial): {E_final/E_init:.4f}")
    else:
        log(f"Energy Ratio: Undefined (E_init ~ 0). Check injection.")
    
    # Final State Plot
    f_final = None
    if is_tm:
        f_final = sim.get_array(component=mp.Ez, cmplx=False)
    else:
        fx = sim.get_array(component=mp.Ex, cmplx=False)
        fy = sim.get_array(component=mp.Ey, cmplx=False)
        if mp.am_master():
             f_final = np.sqrt(fx**2 + fy**2)

    if mp.am_master():
        plt.figure(figsize=(10,8))
        label = "Ez" if is_tm else "|E|"
        plt.imshow(f_final.T, origin='lower', cmap='RdBu' if is_tm else 'hot', extent=plot_extent)
        plt.colorbar(label=label)
        plt.title(f"Final State (t={run_time:.1f})")
        plt.savefig(out_dir / "final_state.png")
        plt.close()
        
        # Plot Diagnostics
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Time (Periods)')
        ax1.set_ylabel('Field Overlap |c(t)|', color=color)
        ax1.plot(np.array(time_points)/T_period, overlap_history, color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.1)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Total Energy', color=color)  # we already handled the x-label with ax1
        ax2.plot(np.array(time_points)/T_period, energy_history, color=color, linestyle='--', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Stability Diagnostics")
        fig.tight_layout()
        plt.savefig(out_dir / "diagnostics.png")
        plt.close()
        
        log(f"Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7: Meep Launch")
    parser.add_argument("candidate_id", type=int, help="Candidate ID")
    parser.add_argument("mode_index", type=int, help="Mode Index")
    parser.add_argument("--periodic", action="store_true", help="Use Periodic BCs instead of PML (Ideal Energy Conservation)")
    args = parser.parse_args()
    
    if mp is None:
        print("Error: Meep not installed.")
        sys.exit(1)
        
    run_phase7_launch(args.candidate_id, args.mode_index, args.periodic)
