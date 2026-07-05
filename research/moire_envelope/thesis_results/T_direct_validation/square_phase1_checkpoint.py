#!/usr/bin/env python3
"""
Checkpointed Phase 1: Square Lattice M-point Bloch-Field Extraction
====================================================================

Drop-in replacement for square_phase1.py that:
  - Processes registry points row-by-row (128 points per row = ~0.78%)
  - After each row, writes results to HDF5 on disk and frees memory
  - On restart, skips already-completed rows (fully resumable)
  - Peak RAM: ~2 GB instead of 48 GB

The checkpoint file is:
    <output_dir>/candidate_0000/registry_checkpoint.h5

After all 128 rows are done, it assembles the final
    phase1_multiband_data.h5
using the same extract_multiband_data_from_mpb_v3 as the original pipeline.

Usage:
    python square_phase1_checkpoint.py              # run (or resume)
    nohup python -u square_phase1_checkpoint.py > square_phase1_ckpt.log 2>&1 &
"""

import sys, os

# CRITICAL: Set threading env vars BEFORE importing numpy/scipy/mpb.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

import json, time, gc, math
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

sys.stdout.reconfigure(line_buffering=True)

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent  # moire_envelope
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

# Import the worker function (module-level, pickle-able for multiprocessing)
from phase1_mpb_v3 import (
    _compute_single_registry_point,
    extract_multiband_data_from_mpb_v3,
    ensure_moire_metadata,
    build_fractional_grid,
    fractional_to_cartesian,
    compute_registry_fractional_v3,
    choose_reference_frequency,
    plot_phase1_fields_v2,
    log,
)
from common.io_utils import candidate_dir, save_json

# =============================================================================
# Parameters — identical to square_phase1.py
# =============================================================================

MPB_RESOLUTION = 128
REGISTRY_SAMPLES = 128       # n_registry per direction
NS = 128
FD_ORDER = 4
N_WORKERS = 16

OUTPUT_DIR = SCRIPT_DIR / "square_M_b3_phase1_run"

CANDIDATE_PARAMS = {
    "candidate_id": 0,
    "lattice_type": "square",
    "a": 1.0,
    "r_over_a": 0.2,
    "eps_bg": 1.0,
    "eps_hole": 11.56,
    "band_index": 3,
    "k_label": "M",
    "k0_x": 0.5,
    "k0_y": 0.5,
    "omega0": 0.68457,
    "polarization": "TM",
    "dominant_polarization": "TM",
    "local_polarization": "TM",
    "n_subspace_bands": 1,
    "subspace_bands": [3],
    "all_bands": [0, 1, 2, 3, 4, 5, 6, 7],
    "target_index_in_subspace": 0,
    "theta_deg": 2.01,
    "theta_rad": 0.035089,
    "moire_length": 28.51,
    "eta": 0.035087,
}

CONFIG_P1 = {
    'phase1_Ns1': NS,
    'phase1_Ns2': NS,
    'mpb_resolution': MPB_RESOLUTION,
    'mpb_registry_samples': REGISTRY_SAMPLES,
    'mpb_dk': 0.01,
    'mpb_fd_order': FD_ORDER,
    'mpb_polarization': 'TM',
    'export_bloch_fields': True,
    'mpb_n_workers': N_WORKERS,
    'tau': [0.0, 0.0],
    'default_theta_deg': CANDIDATE_PARAMS['theta_deg'],
}

# =============================================================================
# Derived constants
# =============================================================================

ALL_BANDS = CANDIDATE_PARAMS['all_bands']
SUBSPACE_BANDS = CANDIDATE_PARAMS['subspace_bands']
N_ALL = len(ALL_BANDS)
MAX_BAND = max(ALL_BANDS) + 1
N_REGISTRY = REGISTRY_SAMPLES
N_STENCIL = 5 if FD_ORDER == 4 else 3
NX = NY = MPB_RESOLUTION   # MPB grid size

POLARIZATION = 'TM'

# Worker params dict — same structure as run_mpb_registry_sweep builds
PARAMS_DICT = {
    'lattice_type': CANDIDATE_PARAMS['lattice_type'],
    'r_over_a': CANDIDATE_PARAMS['r_over_a'],
    'eps_bg': CANDIDATE_PARAMS['eps_bg'],
    'eps_hole': CANDIDATE_PARAMS['eps_hole'],
    'k0': [CANDIDATE_PARAMS['k0_x'], CANDIDATE_PARAMS['k0_y']],
    'dk': CONFIG_P1['mpb_dk'],
    'all_bands': ALL_BANDS,
    'polarization': POLARIZATION,
    'fd_order': FD_ORDER,
    'resolution': MPB_RESOLUTION,
    'max_band': MAX_BAND,
    'export_bloch_fields': True,
}


# =============================================================================
# Checkpoint helpers
# =============================================================================

def create_checkpoint_file(path):
    """Create a fresh checkpoint HDF5 with pre-sized datasets."""
    with h5py.File(path, 'w') as hf:
        # Completion mask: one bool per row (ix)
        hf.create_dataset('completed', data=np.zeros(N_REGISTRY, dtype=bool))

        # Scalar/vector registry data — small, write per-row
        hf.create_dataset('registry_omega0',
                          shape=(N_REGISTRY, N_REGISTRY, N_ALL),
                          dtype=np.float64, fillvalue=np.nan)
        hf.create_dataset('registry_vg',
                          shape=(N_REGISTRY, N_REGISTRY, N_ALL, 2),
                          dtype=np.float64, fillvalue=np.nan)
        hf.create_dataset('registry_M_inv',
                          shape=(N_REGISTRY, N_REGISTRY, N_ALL, 2, 2),
                          dtype=np.float64, fillvalue=np.nan)
        hf.create_dataset('stencil_omega',
                          shape=(N_REGISTRY, N_REGISTRY, N_ALL, N_STENCIL, N_STENCIL),
                          dtype=np.float64, fillvalue=np.nan)

        # Bloch fields — large, chunked by row for streaming writes
        hf.create_dataset('bloch_fields',
                          shape=(N_REGISTRY, N_REGISTRY, N_ALL, NX, NY, 3),
                          dtype=np.complex64,
                          chunks=(1, 1, N_ALL, NX, NY, 3),
                          compression='lzf')

        # Epsilon grid
        hf.create_dataset('epsilon',
                          shape=(N_REGISTRY, N_REGISTRY, NX, NY),
                          dtype=np.float64,
                          chunks=(1, 1, NX, NY),
                          compression='lzf')

        # Metadata
        hf.attrs['n_registry'] = N_REGISTRY
        hf.attrs['n_all_bands'] = N_ALL
        hf.attrs['resolution'] = MPB_RESOLUTION
        hf.attrs['fd_order'] = FD_ORDER
        hf.attrs['created'] = datetime.now().isoformat()

    print(f"  Created checkpoint file: {path}")


def get_completed_rows(ckpt_path):
    """Return array of which rows are done."""
    with h5py.File(ckpt_path, 'r') as hf:
        return hf['completed'][:]


def write_row_to_checkpoint(ckpt_path, ix, row_results):
    """
    Write one row of results to the checkpoint file and mark complete.

    row_results: list of (iy, result_dict) for iy in 0..N_REGISTRY-1
    """
    # Build temporary row arrays
    omega0_row = np.full((N_REGISTRY, N_ALL), np.nan)
    vg_row = np.full((N_REGISTRY, N_ALL, 2), np.nan)
    M_inv_row = np.full((N_REGISTRY, N_ALL, 2, 2), np.nan)
    stencil_row = np.full((N_REGISTRY, N_ALL, N_STENCIL, N_STENCIL), np.nan)
    bloch_row = np.zeros((N_REGISTRY, N_ALL, NX, NY, 3), dtype=np.complex64)
    eps_row = np.zeros((N_REGISTRY, NX, NY), dtype=np.float64)

    for iy, result in row_results:
        omega0_row[iy] = result['omega0']
        vg_row[iy] = result['vg']
        M_inv_row[iy] = result['M_inv']
        stencil_row[iy] = result['omega_stencil']
        if 'bloch_fields' in result:
            bloch_row[iy] = result['bloch_fields']
        if 'epsilon' in result:
            eps_row[iy] = result['epsilon']

    # Write to HDF5 in a single open/close cycle
    with h5py.File(ckpt_path, 'a') as hf:
        hf['registry_omega0'][ix] = omega0_row
        hf['registry_vg'][ix] = vg_row
        hf['registry_M_inv'][ix] = M_inv_row
        hf['stencil_omega'][ix] = stencil_row
        hf['bloch_fields'][ix] = bloch_row
        hf['epsilon'][ix] = eps_row
        hf['completed'][ix] = True
        hf.flush()


# =============================================================================
# Row-by-row sweep with checkpointing
# =============================================================================

def run_checkpointed_sweep(ckpt_path):
    """Run the MPB registry sweep row-by-row with disk checkpointing."""
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    completed = get_completed_rows(ckpt_path)
    n_done = int(completed.sum())
    n_total_rows = N_REGISTRY
    total_points = N_REGISTRY * N_REGISTRY

    print(f"\n  Registry sweep: {N_REGISTRY}×{N_REGISTRY} = {total_points} points")
    print(f"  Completed rows: {n_done}/{n_total_rows} "
          f"({n_done * N_REGISTRY}/{total_points} points, "
          f"{100*n_done/n_total_rows:.1f}%)")
    print(f"  Workers: {N_WORKERS} (CPU count: {cpu_count()})")

    if n_done == n_total_rows:
        print(f"  All rows complete — nothing to do.")
        return

    step = 1.0 / N_REGISTRY

    # Process remaining rows
    remaining_rows = [ix for ix in range(n_total_rows) if not completed[ix]]
    pbar = tqdm(total=total_points, initial=n_done * N_REGISTRY,
                desc="    MPB registry sweep", unit="pt")

    for row_num, ix in enumerate(remaining_rows):
        t_row = time.time()

        # Build work items for this row
        work_items = []
        for iy in range(N_REGISTRY):
            delta_frac = np.array([ix * step, iy * step])
            work_items.append((ix, iy, delta_frac, PARAMS_DICT))

        # Process row with worker pool
        row_results = []
        with Pool(processes=N_WORKERS) as pool:
            for result_ix, result_iy, result_data in pool.imap_unordered(
                _compute_single_registry_point, work_items, chunksize=4
            ):
                row_results.append((result_iy, result_data))
                pbar.update(1)

        # Write to disk and free memory
        write_row_to_checkpoint(ckpt_path, ix, row_results)
        del row_results
        gc.collect()

        elapsed_row = time.time() - t_row
        rows_done_now = n_done + row_num + 1
        pct = 100 * rows_done_now / n_total_rows

        if (rows_done_now) % max(1, n_total_rows // 20) == 0 or row_num == 0:
            remaining = n_total_rows - rows_done_now
            eta_s = elapsed_row * remaining
            print(f"\n    Row {ix:3d} done in {elapsed_row:.1f}s | "
                  f"{rows_done_now}/{n_total_rows} ({pct:.1f}%) | "
                  f"ETA: {eta_s/3600:.1f}h")

    pbar.close()
    print(f"\n  Sweep complete.")


# =============================================================================
# Assemble final phase1_multiband_data.h5 from checkpoint
# =============================================================================

def assemble_final_output(ckpt_path, cdir):
    """
    Load checkpoint data and run the standard extract + save pipeline.
    This mirrors the second half of process_candidate_v3.
    """
    print(f"\n  Assembling final output from checkpoint...")

    with h5py.File(ckpt_path, 'r') as hf:
        completed = hf['completed'][:]
        if not np.all(completed):
            missing = np.where(~completed)[0]
            raise RuntimeError(
                f"Cannot assemble: {len(missing)} rows incomplete: {missing[:10]}...")

        # Load the scalar/vector data (fits in memory: ~2 GB)
        registry_omega0 = hf['registry_omega0'][:]
        registry_vg = hf['registry_vg'][:]
        registry_M_inv = hf['registry_M_inv'][:]
        stencil_omega = hf['stencil_omega'][:]

    registry_data = {
        'registry_omega0': registry_omega0,
        'registry_vg': registry_vg,
        'registry_M_inv': registry_M_inv,
        'stencil_omega': stencil_omega,
        'n_registry': N_REGISTRY,
        'dk': CONFIG_P1['mpb_dk'],
        'fd_order': FD_ORDER,
        'all_bands': ALL_BANDS,
        'subspace_bands': SUBSPACE_BANDS,
    }

    # Build moiré grids (same as process_candidate_v3)
    moire_meta = ensure_moire_metadata(CANDIDATE_PARAMS.copy(), CONFIG_P1)
    B_mono = moire_meta['B_mono']
    B_moire = moire_meta['B_moire']
    eta = moire_meta['eta']
    theta_rad = moire_meta['theta_rad']

    Ns1 = CONFIG_P1['phase1_Ns1']
    Ns2 = CONFIG_P1['phase1_Ns2']
    s_grid = build_fractional_grid(Ns1, Ns2)
    R_grid = fractional_to_cartesian(s_grid, B_moire)

    tau_frac = np.array(CONFIG_P1.get('tau', [0.0, 0.0]))
    delta_frac = compute_registry_fractional_v3(s_grid, B_moire, B_mono, theta_rad, tau_frac)

    # Extract multi-band data
    print(f"  Extracting multi-band data...")
    omega_grid, vg_grid, M_inv_grid, stencil_info = extract_multiband_data_from_mpb_v3(
        registry_data, delta_frac, ALL_BANDS, SUBSPACE_BANDS
    )
    N_subspace = omega_grid.shape[2]

    target_idx = CANDIDATE_PARAMS.get('target_index_in_subspace', N_subspace // 2)
    omega_ref = choose_reference_frequency(omega_grid[:, :, target_idx], CONFIG_P1)
    V_grid = omega_grid - omega_ref

    print(f"  ω_ref = {omega_ref:.6f}")
    print(f"  V range: [{V_grid.min():.6f}, {V_grid.max():.6f}]")

    # ── Save final HDF5 (identical format to process_candidate_v3) ──
    h5_path = cdir / "phase1_multiband_data.h5"
    with h5py.File(h5_path, 'w') as hf:
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("delta_frac", data=delta_frac, compression="gzip")
        hf.create_dataset("omega", data=omega_grid, compression="gzip")
        hf.create_dataset("vg", data=vg_grid, compression="gzip")
        hf.create_dataset("M_inv", data=M_inv_grid, compression="gzip")
        hf.create_dataset("V", data=V_grid, compression="gzip")

        # Stencil data
        stencil_grp = hf.create_group("stencil")
        stencil_grp.create_dataset("omega_all",
                                   data=stencil_info['stencil_omega_all'], compression="gzip")
        stencil_grp.create_dataset("registry_omega_all",
                                   data=stencil_info['registry_omega_all'], compression="gzip")
        stencil_grp.create_dataset("offsets", data=stencil_info['offsets'])
        stencil_grp.attrs["dk"] = stencil_info['dk']
        stencil_grp.attrs["fd_order"] = stencil_info['fd_order']
        stencil_grp.attrs["n_registry"] = stencil_info['n_registry']

        # Bloch fields — stream from checkpoint row-by-row (never all in RAM)
        print(f"  Streaming Bloch fields from checkpoint to final HDF5...")
        bf_ds = hf.create_dataset(
            'bloch_fields',
            shape=(N_REGISTRY, N_REGISTRY, N_ALL, NX, NY, 3),
            dtype=np.complex64,
            compression='lzf',
            chunks=(1, 1, N_ALL, NX, NY, 3),
        )
        bf_ds.attrs['resolution'] = MPB_RESOLUTION
        bf_ds.attrs['polarization'] = POLARIZATION
        bf_ds.attrs['description'] = (
            'Periodic Bloch functions u_n(r; R) for Born-Huang computation. '
            'Shape: (Ns1, Ns2, N_bands, Nx, Ny, 3) where 3 is (Ex, Ey, Ez).'
        )

        eps_ds = hf.create_dataset(
            'epsilon',
            shape=(N_REGISTRY, N_REGISTRY, NX, NY),
            dtype=np.float64,
            compression='lzf',
            chunks=(1, 1, NX, NY),
        )

        with h5py.File(ckpt_path, 'r') as ckpt:
            for ix in range(N_REGISTRY):
                bf_ds[ix] = ckpt['bloch_fields'][ix]
                eps_ds[ix] = ckpt['epsilon'][ix]
                if ix % 16 == 0:
                    print(f"    Streamed row {ix}/{N_REGISTRY}")

        # Attributes
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["theta_deg"] = CANDIDATE_PARAMS.get('theta_deg', 0.0)
        hf.attrs["theta_rad"] = theta_rad
        hf.attrs["target_band_index"] = CANDIDATE_PARAMS['band_index']
        hf.attrs["target_index_in_subspace"] = target_idx
        hf.attrs["k0_x"] = CANDIDATE_PARAMS['k0_x']
        hf.attrs["k0_y"] = CANDIDATE_PARAMS['k0_y']
        hf.attrs["lattice_type"] = CANDIDATE_PARAMS['lattice_type']
        hf.attrs["r_over_a"] = CANDIDATE_PARAMS['r_over_a']
        hf.attrs["eps_bg"] = CANDIDATE_PARAMS['eps_bg']
        hf.attrs["a"] = CANDIDATE_PARAMS['a']
        hf.attrs["moire_length"] = moire_meta['moire_length']
        hf.attrs["Ns1"] = Ns1
        hf.attrs["Ns2"] = Ns2
        hf.attrs["N_subspace"] = N_subspace
        hf.attrs["B_moire"] = B_moire
        hf.attrs["B_mono"] = B_mono
        hf.attrs["subspace_bands"] = np.array(SUBSPACE_BANDS)
        hf.attrs["all_bands"] = np.array(ALL_BANDS)
        hf.attrs["solver"] = "mpb"
        hf.attrs["pipeline_version"] = "V3"
        hf.attrs["coordinate_system"] = "fractional"

    print(f"  Saved final output: {h5_path}")
    return h5_path


# =============================================================================
# Main
# =============================================================================

def main():
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  Square Lattice Phase 1 — CHECKPOINTED")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Resolution: MPB={MPB_RESOLUTION}, Registry={REGISTRY_SAMPLES}, Ns={NS}")
    print(f"  Band: {CANDIDATE_PARAMS['band_index']} at {CANDIDATE_PARAMS['k_label']}")
    print(f"  ω₀ ≈ {CANDIDATE_PARAMS['omega0']:.5f} (c/a)")
    print(f"  Workers: {N_WORKERS}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Checkpoint: row-by-row ({N_REGISTRY} rows, ~{N_REGISTRY} pts each)")
    print(f"{'='*70}")

    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cdir = OUTPUT_DIR / "candidate_0000"
    cdir.mkdir(parents=True, exist_ok=True)

    # Save candidate meta
    save_json(CANDIDATE_PARAMS, cdir / "phase0_meta.json")

    # Check if final output already exists
    p1_h5 = cdir / "phase1_multiband_data.h5"
    if p1_h5.exists():
        print(f"\n  Final output already exists: {p1_h5}")
        with h5py.File(p1_h5, 'r') as hf:
            print(f"  omega_ref = {hf.attrs.get('omega_ref', 'N/A')}")
            if 'bloch_fields' in hf:
                print(f"  bloch_fields: {hf['bloch_fields'].shape}")
        return

    # ── Checkpoint file ──
    ckpt_path = cdir / "registry_checkpoint.h5"
    if not ckpt_path.exists():
        create_checkpoint_file(ckpt_path)
    else:
        completed = get_completed_rows(ckpt_path)
        n_done = int(completed.sum())
        print(f"\n  Resuming from checkpoint: {n_done}/{N_REGISTRY} rows done "
              f"({100*n_done/N_REGISTRY:.1f}%)")

    # ── Run sweep ──
    run_checkpointed_sweep(ckpt_path)

    # ── Assemble final output ──
    assemble_final_output(ckpt_path, cdir)

    wall = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Phase 1 complete in {wall:.0f}s ({wall/3600:.2f}h)")
    print(f"  Output: {p1_h5}")
    print(f"{'='*70}")

    # Save wall time
    with open(OUTPUT_DIR / 'wall_times.json', 'w') as f:
        json.dump({'phase1_s': wall, 'phase1_h': wall / 3600,
                   'mode': 'checkpointed'}, f, indent=2)


if __name__ == '__main__':
    main()
