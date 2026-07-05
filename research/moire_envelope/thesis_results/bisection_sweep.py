#!/usr/bin/env python
"""
Bisection η-sweep for the honeycomb K-point system.

Runs indefinitely, refining the angle grid between θ_min and θ_max.
At each step, picks the LARGEST gap between adjacent measured angles
and computes the midpoint. Regenerates plots after each new data point.

Approach:
  Phase 1 is θ-INDEPENDENT → stored once, reused via external HDF5 links.
  Phases 2+3 are re-run per angle in a TEMP directory, then cleaned up.
  This guarantees correctness (no rescaling hacks) with ~0 disk footprint.

Usage:
    python thesis_results/bisection_sweep.py                    # default [0.4, 1.5]
    python thesis_results/bisection_sweep.py --theta_min 0.6 --theta_max 0.8
    
    Ctrl+C to stop gracefully (results are saved after each angle).

Disk usage: ~50 KB total (one JSON file + PNG plots). No per-angle HDF5.
"""

import sys, math, json, time, signal, gc, argparse, shutil, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase2_mpb_v3 as p2
import phase3_mpb_v3 as p3
from common.io_utils import load_json

# Also import symmetrize if available
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False


# ─────────────────────────────────────────────────────────────────────────────
# Geometry (matches eta_sweep.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moire_params(theta_deg, lattice_type='honeycomb', a=1.0):
    """Compute moiré geometric parameters. Convention matches eta_sweep.py."""
    theta_rad = math.radians(theta_deg)
    eta = 2 * math.sin(theta_rad / 2)

    if lattice_type == 'square':
        B_mono = np.array([[a, 0.0], [0.0, a]])
    elif lattice_type in ('hex', 'honeycomb'):
        B_mono = np.array([[a, 0.0], [a / 2.0, a * math.sqrt(3) / 2.0]])
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R_theta = np.array([[c, -s], [s, c]])
    B_moire = np.linalg.inv(R_theta - np.eye(2)) @ B_mono
    moire_length = np.linalg.norm(B_moire[:, 0])

    return {
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'eta': eta,
        'B_moire': B_moire,
        'moire_length': moire_length,
    }


def patch_h5_theta(h5_path, moire_params):
    """Patch θ-dependent attributes in HDF5 file. Matches eta_sweep.py."""
    with h5py.File(h5_path, 'r+') as hf:
        hf.attrs['theta_deg'] = moire_params['theta_deg']
        hf.attrs['theta_rad'] = moire_params['theta_rad']
        hf.attrs['eta'] = moire_params['eta']
        hf.attrs['moire_length'] = moire_params['moire_length']
        hf.attrs['B_moire'] = moire_params['B_moire']

        if 'R_grid' in hf and 's_grid' in hf:
            s_grid = hf['s_grid'][:]
            R_new = np.einsum('ij,...j->...i', moire_params['B_moire'], s_grid)
            hf['R_grid'][...] = R_new


def patch_meta_theta(meta_path, moire_params):
    """Patch θ-dependent fields in phase0_meta.json."""
    with open(meta_path) as f:
        meta = json.load(f)
    meta['theta_deg'] = moire_params['theta_deg']
    meta['theta_rad'] = moire_params['theta_rad']
    meta['eta'] = moire_params['eta']
    meta['moire_length'] = moire_params['moire_length']
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Single-angle solver (runs Phase 2 + Phase 3 in temp directory)
# ─────────────────────────────────────────────────────────────────────────────

def solve_at_angle(theta_deg, source_cdir, n_modes=20, lattice_type='honeycomb', a=1.0):
    """
    Solve the envelope eigenvalue problem at a single twist angle.

    Creates a temporary directory, smart-copies Phase 1 data (external-linking
    the large bloch_fields array), patches θ, runs Phase 2+3, reads results,
    then cleans up the temp directory.

    Returns dict with gap, bandwidth, eigenvalues, etc.
    """
    t0 = time.time()

    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    eta = moire_params['eta']

    phase1_src = source_cdir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"

    # Create temp directory
    tmp_base = tempfile.mkdtemp(prefix=f"bisect_theta{theta_deg:.4f}_")
    work_dir = Path(tmp_base) / "candidate_0000"
    work_dir.mkdir(parents=True)

    try:
        # 1. Smart-copy Phase 1 HDF5 (external-link large datasets)
        phase1_dst = work_dir / "phase1_multiband_data.h5"
        with h5py.File(phase1_src, 'r') as src, h5py.File(phase1_dst, 'w') as dst:
            for key, val in src.attrs.items():
                dst.attrs[key] = val
            n_linked = 0
            for key in src.keys():
                obj = src[key]
                if isinstance(obj, h5py.Dataset) and obj.nbytes > 1e9:
                    dst[key] = h5py.ExternalLink(str(phase1_src), f'/{key}')
                    n_linked += 1
                else:
                    src.copy(key, dst)

        # Copy meta
        shutil.copy2(meta_src, work_dir / "phase0_meta.json")

        # 2. Patch θ-dependent attributes
        patch_h5_theta(phase1_dst, moire_params)
        patch_meta_theta(work_dir / "phase0_meta.json", moire_params)

        # 3. Run Phase 2
        p2_config = {
            'include_born_huang': False,
            'include_drift_term': True,
            'use_parallel_transport_gauge': True,
            'n_extra_bands': 4,
            'mpb_fd_order': 4,
        }
        p2.process_candidate_phase2_v3(str(work_dir), p2_config)

        # 3b. Symmetrize Phase 2 data (C6 for honeycomb K)
        if HAS_SYMMETRIZE:
            try:
                symmetrize_phase2(work_dir, 'C6')
            except Exception as e:
                print(f"  WARNING: Symmetrization failed: {e}. Using unsymmetrized data.")

        gc.collect()

        # 4. Run Phase 3
        p3_config = {
            'n_modes': n_modes,
            'include_drift_term': True,
            'include_kinetic_term': True,
            'include_born_huang': False,
            'include_offdiag_A': True,
            'fd_order': 4,
            'sigma_shift': None,
        }
        p3.process_candidate_phase3_v3(str(work_dir), p3_config)

        # 5. Extract results
        phase3_h5 = work_dir / "phase3_multiband_modes.h5"
        with h5py.File(phase3_h5, 'r') as hf:
            eigenvalues = hf['eigenvalues'][:]
            F_spinor = hf['F_spinor'][:]
            N_sub = int(hf.attrs['N_subspace'])

        wall_time = time.time() - t0

        # Metrics
        gap_01 = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
        bw = float(eigenvalues[-1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

        # Band composition for modes 0 and 1
        n_actual = min(n_modes, len(eigenvalues))
        mode_weights = []
        for m in range(min(2, n_actual)):
            F_m = F_spinor[m]
            w = np.array([np.sum(np.abs(F_m[:, :, n]) ** 2) for n in range(N_sub)])
            w /= w.sum()
            mode_weights.append(w.tolist())

        return {
            "theta_deg": theta_deg,
            "eta": eta,
            "L_moire": moire_params['moire_length'],
            "gap_01": gap_01,
            "bandwidth": bw,
            "eigenvalues": eigenvalues[:min(n_actual, 20)].tolist(),
            "mode0_weights": mode_weights[0] if len(mode_weights) > 0 else [],
            "mode1_weights": mode_weights[1] if len(mode_weights) > 1 else [],
            "wall_time_s": wall_time,
        }

    finally:
        # Clean up temp directory
        shutil.rmtree(tmp_base, ignore_errors=True)
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Bisection scheduler
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_results(result_file):
    """Load existing results from JSON file."""
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def load_previous_sweep_results(run_dir):
    """Load results from previous eta_sweep runs for seed data."""
    results = []
    for sweep_dir in sorted(Path(run_dir).glob("eta_sweep_*")):
        json_path = sweep_dir / "sweep_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for r in data:
                if "error" not in r and "theta_deg" in r:
                    results.append({
                        "theta_deg": r["theta_deg"],
                        "eta": r.get("eta", 2 * math.sin(math.radians(r["theta_deg"]) / 2)),
                        "L_moire": r.get("moire_length", 0),
                        "gap_01": float(r.get("gap_01", 0)),
                        "bandwidth": float(r.get("bandwidth_50", r.get("bandwidth", 0))),
                        "eigenvalues": r.get("eigenvalues", []),
                        "wall_time_s": r.get("wall_time_s", 0),
                        "source": f"previous_sweep:{sweep_dir.name}",
                    })
    return results


def pick_next_angle(computed_angles, theta_min, theta_max):
    """
    Bisection: pick the midpoint of the largest gap between adjacent angles.
    Returns the next angle to compute, or None if resolution limit reached.
    """
    angles = sorted(set(computed_angles) | {theta_min, theta_max})

    if len(angles) < 2:
        return (theta_min + theta_max) / 2.0

    max_gap = 0
    best_mid = None
    for i in range(len(angles) - 1):
        gap = angles[i + 1] - angles[i]
        if gap > max_gap:
            max_gap = gap
            best_mid = (angles[i] + angles[i + 1]) / 2.0

    if max_gap < 0.001:
        return None

    return best_mid


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_bisection_results(results, output_dir, grid_label="128"):
    """Generate gap vs angle plot from accumulated results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(results) < 2:
        return

    results_sorted = sorted(results, key=lambda r: r["theta_deg"])
    thetas = np.array([r["theta_deg"] for r in results_sorted])
    gaps = np.array([r["gap_01"] for r in results_sorted])
    bws = np.array([r["bandwidth"] for r in results_sorted])

    mask = (thetas >= 0.35) & (thetas <= 1.55)
    thetas_m, gaps_m, bws_m = thetas[mask], gaps[mask], bws[mask]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Gap
    ax = axes[0]
    ax.semilogy(thetas_m, gaps_m, "bo-", markersize=4, linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Gap (E₁ - E₀)", fontsize=12)
    ax.set_title(
        f"Honeycomb K-point Moiré — Bisection Sweep ({len(results)} points, {grid_label} grid)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-5, color="red", linestyle="--", alpha=0.3, label="10⁻⁵ reference")

    if len(gaps_m) > 0:
        imin = np.argmin(gaps_m)
        ax.annotate(
            f"θ* = {thetas_m[imin]:.3f}°\ngap = {gaps_m[imin]:.2e}",
            xy=(thetas_m[imin], gaps_m[imin]),
            xytext=(thetas_m[imin] + 0.1, gaps_m[imin] * 10),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red", fontweight="bold",
        )
    ax.legend(fontsize=9)

    # Panel 2: Bandwidth
    ax = axes[1]
    ax.plot(thetas_m, bws_m, "rs-", markersize=4, linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Twist angle θ (°)", fontsize=12)
    ax.set_ylabel("Bandwidth (20 modes)", fontsize=12)
    ax.grid(True, alpha=0.3)

    if len(gaps_m) > 2 and len(bws_m) > 2:
        ax2 = ax.twinx()
        ratio = gaps_m / np.maximum(bws_m, 1e-15)
        ax2.semilogy(thetas_m, ratio, "g^-", markersize=3, linewidth=0.6, alpha=0.6, label="gap/BW")
        ax2.set_ylabel("gap / BW", color="green", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"bisection_sweep_gap.{ext}", dpi=150)
    plt.close()

    # Eigenvalue waterfall
    fig, ax = plt.subplots(figsize=(14, 6))
    for r in results_sorted:
        if r["theta_deg"] < 0.35 or r["theta_deg"] > 1.55:
            continue
        evals = np.array(r.get("eigenvalues", []))
        if len(evals) > 0:
            n_show = min(10, len(evals))
            ax.scatter([r["theta_deg"]] * n_show, evals[:n_show], s=8, c="steelblue", alpha=0.5)

    ax.set_xlabel("Twist angle θ (°)", fontsize=12)
    ax.set_ylabel("Eigenvalue λ = ω - ω_ref", fontsize=12)
    ax.set_title(f"Eigenvalue Spectrum vs Twist Angle ({len(results)} points)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "bisection_sweep_spectrum.png", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_results(result_file, results, args):
    """Save results to JSON."""
    data = {
        "metadata": {
            "script": "bisection_sweep.py",
            "n_modes": args.n_modes,
            "theta_range": [args.theta_min, args.theta_max],
            "last_updated": datetime.now().isoformat(),
            "n_points": len(results),
        },
        "results": results,
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

GRACEFUL_EXIT = False


def signal_handler(sig, frame):
    global GRACEFUL_EXIT
    print("\n\n  >>> Ctrl+C received. Finishing current angle, then exiting... <<<\n")
    GRACEFUL_EXIT = True


def main():
    parser = argparse.ArgumentParser(description="Bisection η-sweep (runs until Ctrl+C)")
    parser.add_argument("--n_modes", type=int, default=20, help="Number of modes (default: 20)")
    parser.add_argument("--theta_min", type=float, default=0.4, help="Min angle (default: 0.4)")
    parser.add_argument("--theta_max", type=float, default=1.5, help="Max angle (default: 1.5)")
    parser.add_argument("--run_dir", type=str, default=None, help="Override run directory")
    parser.add_argument("--no_seed", action="store_true", help="Don't load previous sweep results")
    parser.add_argument("--verify", type=float, default=None,
                        help="Run a single angle for verification, then exit")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Find honeycomb run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        base = PROJECT_ROOT / "runsV3"
        candidates = sorted(base.glob("thesis_honeycomb_K_b1_2026*"))
        candidates = [c for c in candidates if "TE" not in c.name]
        if not candidates:
            print("ERROR: No honeycomb run directory found!")
            sys.exit(1)
        run_dir = candidates[-1]

    source_cdir = run_dir / "candidate_0000"

    # Read lattice info from meta
    meta = load_json(source_cdir / "phase0_meta.json")
    lattice_type = meta.get('lattice_type', 'honeycomb')
    a = meta.get('a', 1.0)

    print(f"{'=' * 70}")
    print(f"  BISECTION η-SWEEP (Phase 2+3 per angle)")
    print(f"  Run dir: {run_dir}")
    print(f"  Lattice: {lattice_type}, a={a}")
    print(f"  Modes: {args.n_modes}")
    print(f"  Range: [{args.theta_min}°, {args.theta_max}°]")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"{'=' * 70}\n")

    # ── Verification mode ──
    if args.verify is not None:
        theta_v = args.verify
        print(f"  VERIFICATION MODE: solving θ = {theta_v}°\n")
        result = solve_at_angle(theta_v, source_cdir, n_modes=args.n_modes,
                                lattice_type=lattice_type, a=a)
        print(f"\n  θ = {theta_v}°")
        print(f"  gap    = {result['gap_01']:.6e}")
        print(f"  BW     = {result['bandwidth']:.6e}")
        print(f"  L_moire= {result['L_moire']:.2f}")
        print(f"  time   = {result['wall_time_s']:.1f}s")
        print(f"  evals  = {result['eigenvalues'][:5]}")
        return

    # ── Normal bisection mode ──
    output_dir = SCRIPT_DIR / "T_bisection_sweep"
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / "bisection_results.json"

    # Seed with previous results
    all_results = load_existing_results(result_file)
    existing_angles = {round(r["theta_deg"], 6) for r in all_results}

    if not args.no_seed and len(all_results) == 0:
        print("Loading previous sweep results as seed data...")
        prev_results = load_previous_sweep_results(run_dir)
        for r in prev_results:
            theta = r["theta_deg"]
            if args.theta_min <= theta <= args.theta_max and round(theta, 6) not in existing_angles:
                all_results.append(r)
                existing_angles.add(round(theta, 6))
        print(f"  Loaded {len(all_results)} seed points from previous sweeps")

    save_results(result_file, all_results, args)

    if len(all_results) >= 2:
        plot_bisection_results(all_results, output_dir)
        print(f"  Initial plot saved: {output_dir / 'bisection_sweep_gap.png'}")

    # Main bisection loop
    iteration = 0
    total_compute_time = 0

    while not GRACEFUL_EXIT:
        iteration += 1

        computed_angles = [r["theta_deg"] for r in all_results]
        next_theta = pick_next_angle(computed_angles, args.theta_min, args.theta_max)

        if next_theta is None:
            print("\n  Resolution limit reached (< 0.001°). Stopping.")
            break

        if any(abs(next_theta - a_) < 1e-5 for a_ in computed_angles):
            print(f"  Skipping θ={next_theta:.4f}° (already computed)")
            continue

        sorted_angles = sorted(computed_angles)
        min_gap_between = min(
            (sorted_angles[i + 1] - sorted_angles[i])
            for i in range(len(sorted_angles) - 1)
        ) if len(sorted_angles) > 1 else float("inf")

        print(f"\n[{iteration}] θ = {next_theta:.4f}° | "
              f"{len(all_results)} points done | "
              f"min Δθ = {min_gap_between:.4f}° | "
              f"total compute: {total_compute_time:.0f}s")

        try:
            result = solve_at_angle(next_theta, source_cdir, n_modes=args.n_modes,
                                    lattice_type=lattice_type, a=a)
            all_results.append(result)
            existing_angles.add(round(next_theta, 6))
            total_compute_time += result["wall_time_s"]

            gap = result["gap_01"]
            bw = result["bandwidth"]
            wt = result["wall_time_s"]
            print(f"  → gap={gap:.2e}, BW={bw:.2e}, time={wt:.1f}s")

        except Exception as e:
            print(f"  ERROR at θ={next_theta:.4f}°: {e}")
            import traceback
            traceback.print_exc()
            continue

        save_results(result_file, all_results, args)

        try:
            plot_bisection_results(all_results, output_dir)
        except Exception as e:
            print(f"  WARNING: Plot generation failed: {e}")

        gc.collect()

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  BISECTION SWEEP COMPLETE")
    print(f"  Total points: {len(all_results)}")
    print(f"  Total compute time: {total_compute_time:.0f}s ({total_compute_time / 60:.1f} min)")
    print(f"  Results: {result_file}")
    print(f"  Plots: {output_dir}")

    if all_results:
        filtered = [r for r in all_results
                    if args.theta_min <= r["theta_deg"] <= args.theta_max and "gap_01" in r]
        if filtered:
            best = min(filtered, key=lambda r: r["gap_01"])
            print(f"\n  MAGIC ANGLE: θ* = {best['theta_deg']:.4f}° (gap = {best['gap_01']:.2e})")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
