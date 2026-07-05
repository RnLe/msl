#!/usr/bin/env python
"""
Multi-lattice bisection sweep — hex_M_b1 and square_M_b3 in alternation.

Runs indefinitely, alternating between hex and square lattice computations.
Each lattice maintains its own JSON result file and plot set.
After each angle, saves results and regenerates plots.

Phase 1 is θ-INDEPENDENT → stored once, reused via external HDF5 links.
Phases 2+3 are re-run per angle in a TEMP directory, then cleaned up.

Usage:
    python thesis_results/bisection_sweep_multi.py
    python thesis_results/bisection_sweep_multi.py --theta_min 0.4 --theta_max 1.5
    python thesis_results/bisection_sweep_multi.py --only hex   # run only hex
    python thesis_results/bisection_sweep_multi.py --only square  # run only square

    Ctrl+C to stop gracefully (results saved after each angle).
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

try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False


# ─────────────────────────────────────────────────────────────────────────────
# Lattice configurations
# ─────────────────────────────────────────────────────────────────────────────

LATTICE_CONFIGS = {
    "hex": {
        "name": "hex_M_b1",
        "label": "Hexagonal M-point (C1)",
        "run_dir_pattern": "thesis_hex_M_b1_2026*",
        "symmetry": "C2",  # C2 for hex M-point
        "output_subdir": "T_bisection_hex",
        "sweep_pattern": "eta_sweep_*",
    },
    "square": {
        "name": "square_M_b3",
        "label": "Square M-point (C3)",
        "run_dir_pattern": "thesis_square_M_b3_2026*",
        "symmetry": "C4",  # C4 for square M-point
        "output_subdir": "T_bisection_square",
        "sweep_pattern": "eta_sweep_*",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry (matches eta_sweep.py / bisection_sweep.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_moire_params(theta_deg, lattice_type, a=1.0):
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
        'theta_deg': theta_deg, 'theta_rad': theta_rad,
        'eta': eta, 'B_moire': B_moire, 'moire_length': moire_length,
    }


def patch_h5_theta(h5_path, moire_params):
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
    with open(meta_path) as f:
        meta = json.load(f)
    meta['theta_deg'] = moire_params['theta_deg']
    meta['theta_rad'] = moire_params['theta_rad']
    meta['eta'] = moire_params['eta']
    meta['moire_length'] = moire_params['moire_length']
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Single-angle solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_at_angle(theta_deg, source_cdir, n_modes, lattice_type, symmetry, a=1.0):
    """
    Solve envelope eigenproblem at one twist angle.
    Creates temp dir, copies Phase 1, patches θ, runs Phase 2+3, extracts results.
    """
    t0 = time.time()

    moire_params = compute_moire_params(theta_deg, lattice_type, a)
    eta = moire_params['eta']

    phase1_src = source_cdir / "phase1_multiband_data.h5"
    meta_src = source_cdir / "phase0_meta.json"

    tmp_base = tempfile.mkdtemp(prefix=f"bisect_{lattice_type}_{theta_deg:.4f}_")
    work_dir = Path(tmp_base) / "candidate_0000"
    work_dir.mkdir(parents=True)

    try:
        # Smart-copy Phase 1 HDF5 (external-link large datasets)
        phase1_dst = work_dir / "phase1_multiband_data.h5"
        with h5py.File(phase1_src, 'r') as src, h5py.File(phase1_dst, 'w') as dst:
            for key, val in src.attrs.items():
                dst.attrs[key] = val
            for key in src.keys():
                obj = src[key]
                if isinstance(obj, h5py.Dataset) and obj.nbytes > 1e9:
                    dst[key] = h5py.ExternalLink(str(phase1_src), f'/{key}')
                else:
                    src.copy(key, dst)

        shutil.copy2(meta_src, work_dir / "phase0_meta.json")

        patch_h5_theta(phase1_dst, moire_params)
        patch_meta_theta(work_dir / "phase0_meta.json", moire_params)

        # Phase 2
        p2_config = {
            'include_born_huang': False,
            'include_drift_term': True,
            'use_parallel_transport_gauge': True,
            'n_extra_bands': 4,
            'mpb_fd_order': 4,
        }
        p2.process_candidate_phase2_v3(str(work_dir), p2_config)

        # Symmetrize
        if HAS_SYMMETRIZE and symmetry:
            try:
                symmetrize_phase2(work_dir, symmetry)
            except Exception as e:
                print(f"  WARNING: Symmetrization ({symmetry}) failed: {e}")

        gc.collect()

        # Phase 3
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

        # Extract results
        phase3_h5 = work_dir / "phase3_multiband_modes.h5"
        with h5py.File(phase3_h5, 'r') as hf:
            eigenvalues = hf['eigenvalues'][:]
            F_spinor = hf['F_spinor'][:]
            N_sub = int(hf.attrs['N_subspace'])

        wall_time = time.time() - t0

        gap_01 = float(eigenvalues[1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0
        bw = float(eigenvalues[-1] - eigenvalues[0]) if len(eigenvalues) > 1 else 0.0

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
            "eigenvalues": eigenvalues[:min(n_actual, n_modes)].tolist(),
            "mode0_weights": mode_weights[0] if mode_weights else [],
            "mode1_weights": mode_weights[1] if len(mode_weights) > 1 else [],
            "wall_time_s": wall_time,
        }

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Bisection scheduler
# ─────────────────────────────────────────────────────────────────────────────

def pick_next_angle(computed_angles, theta_min, theta_max):
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
# Data management
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_results(result_file):
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def load_previous_sweep_results(run_dir, theta_min, theta_max):
    """Load results from previous eta_sweep runs for seed data."""
    results = []
    for sweep_dir in sorted(Path(run_dir).glob("eta_sweep_*")):
        # Prefer the non-diagA sweep (consistent with fullA approach)
        if "diagA" in sweep_dir.name:
            continue
        json_path = sweep_dir / "sweep_results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("results", [])
            for r in data:
                if "error" in r or "theta_deg" not in r:
                    continue
                th = r["theta_deg"]
                if th < theta_min or th > theta_max:
                    continue
                results.append({
                    "theta_deg": th,
                    "eta": r.get("eta", 2 * math.sin(math.radians(th) / 2)),
                    "L_moire": r.get("moire_length", 0),
                    "gap_01": float(r.get("gap_01", 0)),
                    "bandwidth": float(r.get("bandwidth_50", r.get("bandwidth", 0))),
                    "eigenvalues": r.get("eigenvalues", []),
                    "wall_time_s": r.get("wall_time_s", 0),
                    "source": f"previous_sweep:{sweep_dir.name}",
                })
    return results


def save_results(result_file, results, lattice_key, n_modes, theta_min, theta_max):
    data = {
        "metadata": {
            "script": "bisection_sweep_multi.py",
            "lattice": lattice_key,
            "n_modes": n_modes,
            "theta_range": [theta_min, theta_max],
            "last_updated": datetime.now().isoformat(),
            "n_points": len(results),
        },
        "results": results,
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_bisection_results(results, output_dir, lattice_label, grid_label="128"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(results) < 2:
        return

    results_sorted = sorted(results, key=lambda r: r["theta_deg"])
    thetas = np.array([r["theta_deg"] for r in results_sorted])
    gaps = np.array([r["gap_01"] for r in results_sorted])
    bws = np.array([r["bandwidth"] for r in results_sorted])

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Gap
    ax = axes[0]
    ax.semilogy(thetas, gaps, "bo-", markersize=4, linewidth=0.8, alpha=0.8)
    ax.set_ylabel("Gap (E₁ - E₀)", fontsize=12)
    ax.set_title(
        f"{lattice_label} — Bisection Sweep ({len(results)} points, {grid_label} grid)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1e-5, color="red", linestyle="--", alpha=0.3, label="10⁻⁵ ref")
    if len(gaps) > 0:
        imin = np.argmin(gaps)
        ax.annotate(
            f"θ* = {thetas[imin]:.3f}°\ngap = {gaps[imin]:.2e}",
            xy=(thetas[imin], gaps[imin]),
            xytext=(thetas[imin] + 0.1, gaps[imin] * 10),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red", fontweight="bold",
        )
    ax.legend(fontsize=9)

    # Panel 2: Eigenvalue spectrum
    ax = axes[1]
    for r in results_sorted:
        evals = np.array(r.get("eigenvalues", []))
        if len(evals) > 0:
            n_show = min(10, len(evals))
            ax.scatter([r["theta_deg"]] * n_show, evals[:n_show],
                       s=8, c="steelblue", alpha=0.5)
    ax.set_xlabel("Twist angle θ (°)", fontsize=12)
    ax.set_ylabel("Eigenvalue λ = ω - ω_ref", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"bisection_sweep.{ext}", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Lattice state manager
# ─────────────────────────────────────────────────────────────────────────────

class LatticeState:
    """Manages the bisection state for one lattice configuration."""

    def __init__(self, key, config, n_modes, theta_min, theta_max, no_seed=False):
        self.key = key
        self.config = config
        self.n_modes = n_modes
        self.theta_min = theta_min
        self.theta_max = theta_max

        # Find run directory
        base = PROJECT_ROOT / "runsV3"
        candidates = sorted(base.glob(config["run_dir_pattern"]))
        if not candidates:
            raise FileNotFoundError(f"No run directory for {key}: {config['run_dir_pattern']}")
        self.run_dir = candidates[-1]
        self.source_cdir = self.run_dir / "candidate_0000"

        # Read lattice info from meta
        meta = load_json(self.source_cdir / "phase0_meta.json")
        self.lattice_type = meta.get('lattice_type', key)
        self.a = meta.get('a', 1.0)
        self.symmetry = config["symmetry"]

        # Output
        self.output_dir = SCRIPT_DIR / config["output_subdir"]
        self.output_dir.mkdir(exist_ok=True)
        self.result_file = self.output_dir / "bisection_results.json"

        # Load existing results
        self.results = load_existing_results(self.result_file)
        existing_angles = {round(r["theta_deg"], 6) for r in self.results}

        if not no_seed and len(self.results) == 0:
            prev = load_previous_sweep_results(self.run_dir, theta_min, theta_max)
            for r in prev:
                if round(r["theta_deg"], 6) not in existing_angles:
                    self.results.append(r)
                    existing_angles.add(round(r["theta_deg"], 6))

        self.save()
        self.total_compute_time = 0

    def save(self):
        save_results(self.result_file, self.results, self.key,
                     self.n_modes, self.theta_min, self.theta_max)

    def plot(self):
        try:
            plot_bisection_results(self.results, self.output_dir, self.config["label"])
        except Exception as e:
            print(f"  WARNING: Plot failed for {self.key}: {e}")

    def get_next_angle(self):
        computed = [r["theta_deg"] for r in self.results]
        return pick_next_angle(computed, self.theta_min, self.theta_max)

    def has_angle(self, theta):
        return any(abs(theta - r["theta_deg"]) < 1e-5 for r in self.results)

    def compute_one(self):
        """Compute one bisection point. Returns True if successful, False if done."""
        next_theta = self.get_next_angle()
        if next_theta is None:
            return False
        if self.has_angle(next_theta):
            return True  # skip but not done

        n_pts = len(self.results)
        sorted_angles = sorted(r["theta_deg"] for r in self.results)
        min_gap = min(
            (sorted_angles[i+1] - sorted_angles[i]) for i in range(len(sorted_angles)-1)
        ) if len(sorted_angles) > 1 else float("inf")

        print(f"\n  [{self.key}] θ = {next_theta:.4f}° | "
              f"{n_pts} pts | min Δθ = {min_gap:.4f}° | "
              f"compute: {self.total_compute_time:.0f}s")

        result = solve_at_angle(
            next_theta, self.source_cdir, self.n_modes,
            self.lattice_type, self.symmetry, self.a
        )
        self.results.append(result)
        self.total_compute_time += result["wall_time_s"]

        gap = result["gap_01"]
        bw = result["bandwidth"]
        wt = result["wall_time_s"]
        print(f"    → gap={gap:.2e}, BW={bw:.2e}, time={wt:.1f}s")

        self.save()
        self.plot()
        gc.collect()
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

GRACEFUL_EXIT = False


def signal_handler(sig, frame):
    global GRACEFUL_EXIT
    print("\n\n  >>> Ctrl+C received. Finishing current angle, then exiting... <<<\n")
    GRACEFUL_EXIT = True


def main():
    parser = argparse.ArgumentParser(description="Multi-lattice bisection sweep")
    parser.add_argument("--n_modes", type=int, default=20)
    parser.add_argument("--theta_min", type=float, default=0.4)
    parser.add_argument("--theta_max", type=float, default=1.5)
    parser.add_argument("--no_seed", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        choices=["hex", "square"],
                        help="Run only one lattice type")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Determine which lattices to run
    if args.only:
        lattice_keys = [args.only]
    else:
        lattice_keys = ["hex", "square"]

    print(f"{'=' * 70}")
    print(f"  MULTI-LATTICE BISECTION SWEEP")
    print(f"  Lattices: {', '.join(lattice_keys)}")
    print(f"  Modes: {args.n_modes}")
    print(f"  Range: [{args.theta_min}°, {args.theta_max}°]")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"{'=' * 70}")

    # Initialize lattice states
    states = {}
    for key in lattice_keys:
        try:
            states[key] = LatticeState(
                key, LATTICE_CONFIGS[key], args.n_modes,
                args.theta_min, args.theta_max, args.no_seed
            )
            cfg = states[key]
            print(f"\n  {key}: {cfg.config['label']}")
            print(f"    Run dir:  {cfg.run_dir.name}")
            print(f"    Lattice:  {cfg.lattice_type}, a={cfg.a}")
            print(f"    Symmetry: {cfg.symmetry}")
            print(f"    Existing: {len(cfg.results)} points")
        except FileNotFoundError as e:
            print(f"\n  WARNING: Skipping {key}: {e}")

    if not states:
        print("ERROR: No lattice configurations available!")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  Starting alternating bisection loop...")
    print(f"{'=' * 70}")

    # Generate initial plots
    for s in states.values():
        s.plot()

    # Main alternation loop
    iteration = 0
    active_keys = list(states.keys())

    while not GRACEFUL_EXIT and active_keys:
        iteration += 1

        # Round-robin through active lattices
        key = active_keys[iteration % len(active_keys)]
        state = states[key]

        print(f"\n{'─' * 50}")
        print(f"  Iteration {iteration} — {state.config['label']}")
        print(f"{'─' * 50}")

        try:
            ok = state.compute_one()
            if not ok:
                print(f"  {key}: Resolution limit reached. Removing from rotation.")
                active_keys.remove(key)
        except Exception as e:
            print(f"  ERROR in {key} at current angle: {e}")
            import traceback
            traceback.print_exc()
            # Continue with other lattice rather than stopping

    # Final summary
    print(f"\n{'=' * 70}")
    print(f"  SWEEP COMPLETE")
    for key, state in states.items():
        n = len(state.results)
        ct = state.total_compute_time
        print(f"    {key}: {n} points, {ct:.0f}s compute time")
        print(f"      Results: {state.result_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
