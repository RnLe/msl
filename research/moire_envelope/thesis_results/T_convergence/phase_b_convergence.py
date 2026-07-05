#!/usr/bin/env python3
"""
Phase B: EA Multi-Axis Resolution Convergence
==============================================

Three convergence tests at θ≈1.1° (honeycomb K-point, TM):

  B1: Registry sampling convergence
      - Fix Ns=128, mpb_resolution=64
      - Vary mpb_registry_samples ∈ {32, 64, 128}
      - Requires new Phase 1+2+3 runs for registry={32, 64}
      - registry=128 reuses existing thesis data

  B2: Hamiltonian grid (Ns) convergence
      - Fix registry=128, mpb_resolution=64
      - Vary Ns ∈ {32, 48, 64, 96, 128, 192, 256}
      - Reuses existing Phase 2 data (only Phase 3 reruns)

  B3: "Honest" combined convergence (registry = Ns)
      - Fix mpb_resolution=64
      - Vary registry=Ns ∈ {64, 128, 192}
      - registry=64: resamples B1 Phase 2 data to Ns=64
      - registry=128: reuses existing thesis data
      - registry=192: full Phase 1+2+3 (slow, ~8h)

Outputs:
  - convergence_results_B.json: all eigenvalue data
  - fig_phaseB_convergence.{png,pdf}: multi-panel convergence figure

Usage:
    python phase_b_convergence.py              # run all
    python phase_b_convergence.py --skip_b3    # skip expensive B3
    python phase_b_convergence.py --only b1    # only registry sweep
"""

import sys, os

# CRITICAL: Set threading env vars BEFORE importing numpy/scipy/mpb.
# MPB uses internal OMP/BLAS multithreading that thrashes when combined
# with Python multiprocessing workers — must be pinned to 1 thread.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MEEP_NUM_THREADS'] = '1'

import math, json, time, gc, argparse, shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

# Flush stdout after every print so nohup logs appear in real time
sys.stdout.reconfigure(line_buffering=True)

# ── paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
THESIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = THESIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

import phase1_mpb_v3 as p1
import phase2_mpb_v3 as p2
import phase3_mpb_v3 as p3

try:
    sys.path.insert(0, str(THESIS_DIR))
    from symmetrize import symmetrize_phase2
    HAS_SYMMETRIZE = True
except ImportError:
    HAS_SYMMETRIZE = False

OUTPUT_DIR = SCRIPT_DIR

# ── parameters ──
THETA_DEG = 1.1
MPB_RESOLUTION = 64
N_MODES = 50
FD_ORDER = 4

RUNS_BASE = PROJECT_ROOT / "runsV3"


# =============================================================================
# Locate existing data
# =============================================================================

def find_thesis_run():
    """Find the existing thesis honeycomb K-point run directory."""
    candidates = sorted(RUNS_BASE.glob("thesis_honeycomb_K_b1_2026*"))
    candidates = [c for c in candidates if "TE" not in c.name]
    if not candidates:
        raise FileNotFoundError("No thesis honeycomb K-point run found in runsV3/")
    return candidates[-1]


def find_theta_1121_cdir():
    """Find the eta_sweep theta=1.121 candidate dir with BH+C6sym Phase 2 data."""
    run_dir = find_thesis_run()
    sweep_dirs = sorted(run_dir.glob("eta_sweep_*"))
    for sd in reversed(sweep_dirs):
        cdir = sd / "theta_1.121" / "candidate_0000"
        c6sym = cdir / "phase2_multiband_data_c6sym.h5"
        if c6sym.exists():
            with h5py.File(c6sym, 'r') as hf:
                if hf.attrs.get('include_born_huang', False):
                    return cdir
    # Fallback: any theta_1.121 with c6sym
    for sd in reversed(sweep_dirs):
        cdir = sd / "theta_1.121" / "candidate_0000"
        if (cdir / "phase2_multiband_data_c6sym.h5").exists():
            return cdir
    raise FileNotFoundError("No theta=1.121 C6-symmetrized Phase 2 data found")


def find_fdfd_eigenvalues():
    """Load FDFD eigenvalues at res=40."""
    fdfd_path = THESIS_DIR / "T_direct_validation" / "fdfd_dirac_m30_n29_res40_v2.npz"
    if fdfd_path.exists():
        return np.load(fdfd_path)['freqs']
    return None


# =============================================================================
# Phase 2 data loading and resampling
# =============================================================================

def load_phase2_data(cdir):
    """Load Phase 2 data from HDF5 (prefers c6sym, then raw)."""
    for fname in ['phase2_multiband_data_c6sym.h5', 'phase2_multiband_data.h5']:
        p2_h5 = cdir / fname
        if p2_h5.exists():
            break
    else:
        raise FileNotFoundError(f"No Phase 2 data in {cdir}")

    print(f"    Loading: {p2_h5.name}")
    with h5py.File(p2_h5, 'r') as hf:
        data = {
            's_grid': hf['s_grid'][:],
            'R_grid': hf['R_grid'][:],
            'Lambda': hf['Lambda'][:],
            'A_berry': hf['A_berry'][:],
            'Phi_BH': hf['Phi_BH'][:],
            'v_drift': hf['v_drift'][:],
            'M_inv': hf['M_inv'][:],
            'omega_grid': hf['omega'][:],
            'omega_ref': float(hf.attrs['omega_ref']),
            'eta': float(hf.attrs['eta']),
            'theta_rad': float(hf.attrs['theta_rad']),
            'Ns1': int(hf.attrs['Ns1']),
            'Ns2': int(hf.attrs['Ns2']),
            'N_subspace': int(hf.attrs['N_subspace']),
            'target_idx': int(hf.attrs['target_index_in_subspace']),
            'B_moire': hf.attrs['B_moire'][:],
            'B_mono': hf.attrs['B_mono'][:],
        }
    return data


def resample_phase2(data, Ns_target):
    """Resample Phase 2 data to (Ns_target, Ns_target) via periodic interpolation."""
    Ns_orig = data['Ns1']
    if Ns_target == Ns_orig:
        return data

    s1_orig = np.linspace(0, 1, Ns_orig, endpoint=False)
    s2_orig = np.linspace(0, 1, Ns_orig, endpoint=False)
    s1_new = np.linspace(0, 1, Ns_target, endpoint=False)
    s2_new = np.linspace(0, 1, Ns_target, endpoint=False)

    out = dict(data)
    out['Ns1'] = Ns_target
    out['Ns2'] = Ns_target

    for name in ['Lambda', 'A_berry', 'Phi_BH', 'v_drift', 'M_inv']:
        arr = data[name]
        trailing = arr.shape[2:]
        flat = arr.reshape(Ns_orig, Ns_orig, -1)
        n_comp = flat.shape[2]
        new_flat = np.empty((Ns_target, Ns_target, n_comp), dtype=arr.dtype)
        for c in range(n_comp):
            interp = RegularGridInterpolator(
                (s1_orig, s2_orig), flat[:, :, c],
                method='linear', bounds_error=False, fill_value=None,
            )
            s1g, s2g = np.meshgrid(s1_new, s2_new, indexing='ij')
            pts = np.stack([s1g.ravel(), s2g.ravel()], axis=-1)
            new_flat[:, :, c] = interp(pts).reshape(Ns_target, Ns_target)
        out[name] = new_flat.reshape((Ns_target, Ns_target) + trailing)

    s1g, s2g = np.meshgrid(s1_new, s2_new, indexing='ij')
    out['s_grid'] = np.stack([s1g, s2g], axis=-1)
    out['R_grid'] = np.einsum('ij,...j->...i', data['B_moire'], out['s_grid'])
    return out


# =============================================================================
# Hamiltonian assembly + solve (Phase 3 only)
# =============================================================================

def solve_at_ns(data, n_modes=N_MODES):
    """Assemble Hamiltonian from Phase 2 data and solve for eigenvalues."""
    Ns1, Ns2 = data['Ns1'], data['Ns2']
    Nb = data['N_subspace']
    eta = data['eta']
    B_moire = data['B_moire']
    target_idx = data['target_idx']

    L_moire = np.linalg.norm(B_moire[0])
    dR1 = L_moire / Ns1
    dR2 = L_moire / Ns2

    H = p3.assemble_multiband_hamiltonian(
        data['Lambda'], data['v_drift'], data['M_inv'],
        data['A_berry'], data['Phi_BH'],
        eta, Ns1, Ns2, Nb, dR1, dR2, B_moire,
        include_drift=True, include_kinetic=True, include_born_huang=True,
        order=FD_ORDER, include_offdiag_A=True,
    )

    sigma, _ = p3.compute_sigma(data['Lambda'], data['M_inv'], target_idx)
    eigenvalues, _ = p3.solve_multiband_envelope(H, n_modes, sigma=sigma)
    return eigenvalues


# =============================================================================
# Full Phase 1+2+3 pipeline for new registry values
# =============================================================================

def run_full_pipeline(registry_samples, Ns, work_dir, source_cdir):
    """
    Run Phase 1+2+3 from scratch at given registry_samples and Ns.
    Skips any phase whose output already exists (for resumability).

    Returns (eigenvalues, omega_ref, wall_times_dict).
    """
    cdir = work_dir / "candidate_0000"

    # ── Check for existing Phase 3 results ──
    p3_h5 = cdir / "phase3_multiband_modes.h5"
    if p3_h5.exists():
        print(f"  Found existing Phase 3 results, loading...")
        with h5py.File(p3_h5, 'r') as hf:
            return hf['eigenvalues'][:], float(hf.attrs['omega_ref']), \
                   {'phase1': 0, 'phase2': 0, 'phase3': 0}

    # Load candidate parameters from the original thesis run
    with open(source_cdir / "phase0_meta.json") as f:
        candidate_params = json.load(f)

    wall_times = {}

    # ── Phase 1 ──
    p1_h5 = cdir / "phase1_multiband_data.h5"
    if not p1_h5.exists():
        print(f"  Running Phase 1 (registry={registry_samples}, res={MPB_RESOLUTION}, Ns={Ns})...")
        config_p1 = {
            'phase1_Ns1': Ns,
            'phase1_Ns2': Ns,
            'mpb_resolution': MPB_RESOLUTION,
            'mpb_registry_samples': registry_samples,
            'mpb_dk': 0.01,
            'mpb_fd_order': FD_ORDER,
            'mpb_polarization': candidate_params.get('dominant_polarization',
                                candidate_params.get('polarization', 'TM')),
            'export_bloch_fields': True,
            'mpb_n_workers': 16,
            'tau': [0.0, 0.0],
            'default_theta_deg': candidate_params.get('theta_deg', THETA_DEG),
        }
        t0 = time.time()
        p1.process_candidate_v3(candidate_params, config_p1, work_dir)
        wall_times['phase1'] = time.time() - t0
        print(f"  Phase 1 done in {wall_times['phase1']:.1f}s")
        gc.collect()
    else:
        print(f"  Phase 1 already exists, skipping.")
        wall_times['phase1'] = 0

    # ── Phase 2 ──
    p2_h5 = cdir / "phase2_multiband_data.h5"
    if not p2_h5.exists():
        print(f"  Running Phase 2 (include_born_huang=True)...")
        p2_config = {
            'include_born_huang': True,
            'include_drift_term': True,
            'use_parallel_transport_gauge': True,
            'n_extra_bands': 4,
            'mpb_fd_order': FD_ORDER,
        }
        t0 = time.time()
        p2.process_candidate_phase2_v3(str(cdir), p2_config)
        wall_times['phase2'] = time.time() - t0
        print(f"  Phase 2 done in {wall_times['phase2']:.1f}s")
        gc.collect()

        # Symmetrize (C6 for honeycomb)
        if HAS_SYMMETRIZE:
            print(f"  Symmetrizing (C6)...")
            try:
                symmetrize_phase2(cdir, 'C6')
            except Exception as e:
                print(f"  WARNING: Symmetrization failed: {e}")
    else:
        print(f"  Phase 2 already exists, skipping.")
        wall_times['phase2'] = 0

    # ── Phase 3 ──
    print(f"  Running Phase 3 ({N_MODES} modes, include_born_huang=True)...")
    p3_config = {
        'n_modes': N_MODES,
        'include_drift_term': True,
        'include_kinetic_term': True,
        'include_born_huang': True,
        'include_offdiag_A': True,
        'fd_order': FD_ORDER,
        'sigma_shift': None,
    }
    t0 = time.time()
    p3.process_candidate_phase3_v3(str(cdir), p3_config)
    wall_times['phase3'] = time.time() - t0
    print(f"  Phase 3 done in {wall_times['phase3']:.1f}s")
    gc.collect()

    with h5py.File(p3_h5, 'r') as hf:
        evals = hf['eigenvalues'][:]
        omega_ref = float(hf.attrs['omega_ref'])
    return evals, omega_ref, wall_times


# =============================================================================
# Results I/O
# =============================================================================

def _save_results(results):
    """Save results to JSON (numpy-safe)."""
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    json_path = OUTPUT_DIR / "convergence_results_B.json"
    with open(json_path, 'w') as f:
        json.dump(sanitize(results), f, indent=2)
    print(f"  Results saved: {json_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase B: EA multi-axis resolution convergence")
    parser.add_argument("--only", type=str, default=None, choices=['b1', 'b2', 'b3'])
    parser.add_argument("--skip_b3", action='store_true',
                        help="Skip B3 (honest combined — expensive)")
    parser.add_argument("--no_b3_192", action='store_true',
                        help="Skip registry=192 in B3")
    args = parser.parse_args()

    run_b1 = args.only is None or args.only == 'b1'
    run_b2 = args.only is None or args.only == 'b2'
    run_b3 = (args.only is None and not args.skip_b3) or args.only == 'b3'
    b3_192 = not args.no_b3_192

    print(f"\n{'#'*72}")
    print(f"  PHASE B: EA Multi-Axis Resolution Convergence")
    print(f"  {'='*68}")
    print(f"  θ ≈ 1.1° (honeycomb K, TM)")
    print(f"  mpb_resolution = {MPB_RESOLUTION}")
    print(f"  n_modes = {N_MODES}")
    print(f"  Run B1: {run_b1}  |  Run B2: {run_b2}  |  Run B3: {run_b3}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'#'*72}\n")

    t_total_start = time.time()

    # ── Find existing data ──
    thesis_run = find_thesis_run()
    source_cdir = thesis_run / "candidate_0000"
    theta_cdir = find_theta_1121_cdir()
    print(f"  Thesis run: {thesis_run.name}")
    print(f"  Source candidate: {source_cdir}")
    print(f"  θ=1.121 BH+C6sym: {theta_cdir}")

    fdfd_freqs = find_fdfd_eigenvalues()
    if fdfd_freqs is not None:
        print(f"  FDFD reference: {len(fdfd_freqs)} frequencies (res=40)")

    # Load previous partial results if they exist (for resumability)
    json_path = OUTPUT_DIR / "convergence_results_B.json"
    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)
        print(f"  Loaded previous partial results from {json_path.name}")
        results['resumed_at'] = datetime.now().isoformat()
    else:
        results = {
            'theta_deg': THETA_DEG,
            'mpb_resolution': MPB_RESOLUTION,
            'n_modes': N_MODES,
            'timestamp': datetime.now().isoformat(),
        }

    phase_b_runs = OUTPUT_DIR / "runs"
    phase_b_runs.mkdir(exist_ok=True)

    # =====================================================================
    # B1: Registry Sampling Convergence
    # =====================================================================
    if run_b1:
        print(f"\n{'='*72}")
        print(f"  B1: REGISTRY SAMPLING CONVERGENCE")
        print(f"  Fix Ns=128, mpb_resolution={MPB_RESOLUTION}")
        print(f"  Registry samples: [32, 64, 128]")
        print(f"{'='*72}")

        b1_results = []

        for reg in [32, 64, 128]:
            print(f"\n  ── registry = {reg} ──")

            if reg == 128:
                # Reuse existing BH+C6sym Phase 2 data from eta_sweep
                print(f"  Reusing existing thesis Phase 2 data (registry=128)")
                p2_data = load_phase2_data(theta_cdir)
                t0 = time.time()
                evals = solve_at_ns(p2_data, n_modes=N_MODES)
                wt = time.time() - t0
                omega_ref = p2_data['omega_ref']
                wall_times = {'phase1': 0, 'phase2': 0, 'phase3': wt}
                print(f"  Phase 3 solve: {wt:.1f}s, {len(evals)} eigenvalues")
            else:
                work_dir = phase_b_runs / f"b1_registry{reg}_Ns128"
                evals, omega_ref, wall_times = run_full_pipeline(
                    registry_samples=reg, Ns=128,
                    work_dir=work_dir, source_cdir=source_cdir
                )

            b1_results.append({
                'registry': reg, 'Ns': 128,
                'eigenvalues': evals.tolist(),
                'omega_ref': float(omega_ref),
                'wall_times': wall_times,
                'n_evals': len(evals),
            })

            bw = np.max(evals[:N_MODES]) - np.min(evals[:N_MODES])
            print(f"  BW = {bw:.6f}")
            gc.collect()

            # Save continuously after each data point
            results['b1'] = b1_results
            _save_results(results)

        _print_convergence("B1", b1_results, ref_idx=-1, param_key='registry')

    # =====================================================================
    # B2: Hamiltonian Grid (Ns) Convergence
    # =====================================================================
    if run_b2:
        print(f"\n{'='*72}")
        print(f"  B2: HAMILTONIAN GRID (Ns) CONVERGENCE")
        print(f"  Fix registry=128, mpb_resolution={MPB_RESOLUTION}")
        print(f"  Ns: [32, 48, 64, 96, 128, 192, 256]")
        print(f"{'='*72}")

        print(f"\n  Loading Phase 2 data (registry=128, BH+C6sym)...")
        p2_data = load_phase2_data(theta_cdir)
        print(f"  Loaded: Ns={p2_data['Ns1']}×{p2_data['Ns2']}, Nb={p2_data['N_subspace']}")

        b2_results = []

        for ns in [32, 48, 64, 96, 128, 192, 256]:
            print(f"\n  ── Ns = {ns} ──")
            data_rs = resample_phase2(p2_data, ns)
            N_total = ns * ns * data_rs['N_subspace']
            print(f"  Matrix: {N_total}×{N_total}")

            t0 = time.time()
            evals = solve_at_ns(data_rs, n_modes=N_MODES)
            wt = time.time() - t0

            b2_results.append({
                'Ns': ns, 'registry': 128,
                'eigenvalues': evals.tolist(),
                'omega_ref': p2_data['omega_ref'],
                'wall_time_s': wt,
                'n_evals': len(evals),
                'matrix_size': N_total,
            })

            bw = np.max(evals[:N_MODES]) - np.min(evals[:N_MODES])
            print(f"  {len(evals)} eigenvalues, BW={bw:.6f}, t={wt:.1f}s")
            del data_rs
            gc.collect()

            # Save continuously after each data point
            results['b2'] = b2_results
            _save_results(results)

        _print_convergence("B2", b2_results, ref_idx=-1, param_key='Ns')
        del p2_data
        gc.collect()

    # =====================================================================
    # B3: "Honest" Combined Convergence (registry = Ns)
    # =====================================================================
    if run_b3:
        b3_values = [64, 128]
        if b3_192:
            b3_values.append(192)
        print(f"\n{'='*72}")
        print(f"  B3: HONEST COMBINED CONVERGENCE (registry = Ns)")
        print(f"  mpb_resolution={MPB_RESOLUTION}")
        print(f"  registry=Ns: {b3_values}")
        print(f"{'='*72}")

        b3_results = []

        for val in b3_values:
            print(f"\n  ── registry = Ns = {val} ──")

            if val == 128:
                # Reuse existing data (same Phase 2 as B1/B2)
                print(f"  Reusing existing thesis data")
                p2_data = load_phase2_data(theta_cdir)
                t0 = time.time()
                evals = solve_at_ns(p2_data, n_modes=N_MODES)
                wt = time.time() - t0
                omega_ref = p2_data['omega_ref']
                wall_times = {'phase1': 0, 'phase2': 0, 'phase3': wt}
                del p2_data

            elif val == 64:
                # Reuse B1 Phase 2 data (registry=64) resampled to Ns=64
                b1_64_cdir = phase_b_runs / "b1_registry64_Ns128" / "candidate_0000"
                if b1_64_cdir.exists():
                    print(f"  Reusing B1 Phase 2 (registry=64), resampling to Ns=64")
                    p2_data = load_phase2_data(b1_64_cdir)
                    data_rs = resample_phase2(p2_data, 64)
                    t0 = time.time()
                    evals = solve_at_ns(data_rs, n_modes=N_MODES)
                    wt = time.time() - t0
                    omega_ref = data_rs['omega_ref']
                    wall_times = {'phase1': 0, 'phase2': 0, 'phase3': wt}
                    del p2_data, data_rs
                else:
                    # No B1 data — run full pipeline
                    work_dir = phase_b_runs / f"b3_reg{val}_Ns{val}"
                    evals, omega_ref, wall_times = run_full_pipeline(
                        registry_samples=val, Ns=val,
                        work_dir=work_dir, source_cdir=source_cdir
                    )

            else:
                # Full pipeline (e.g., registry=192)
                work_dir = phase_b_runs / f"b3_reg{val}_Ns{val}"
                evals, omega_ref, wall_times = run_full_pipeline(
                    registry_samples=val, Ns=val,
                    work_dir=work_dir, source_cdir=source_cdir
                )

            b3_results.append({
                'registry': val, 'Ns': val,
                'eigenvalues': evals.tolist(),
                'omega_ref': float(omega_ref),
                'wall_times': wall_times,
                'n_evals': len(evals),
            })

            bw = np.max(evals[:N_MODES]) - np.min(evals[:N_MODES])
            print(f"  BW = {bw:.6f}")
            gc.collect()

            # Save continuously after each data point
            results['b3'] = b3_results
            _save_results(results)

        _print_convergence("B3", b3_results, ref_idx=-1, param_key='registry')

    # =====================================================================
    # FDFD comparison
    # =====================================================================
    if fdfd_freqs is not None:
        print(f"\n{'='*72}")
        print(f"  COMPARISON WITH FDFD (res=40)")
        print(f"{'='*72}")
        _print_fdfd_comparison(results, fdfd_freqs)

    # =====================================================================
    # Plots
    # =====================================================================
    print(f"\n  Generating plots...")
    try:
        generate_plots(results, fdfd_freqs)
    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")
        import traceback; traceback.print_exc()

    t_total = time.time() - t_total_start
    print(f"\n{'='*72}")
    print(f"  PHASE B COMPLETE — total time: {t_total/3600:.2f} hours ({t_total:.0f}s)")
    print(f"{'='*72}")

    results['total_time_s'] = t_total
    _save_results(results)


# =============================================================================
# Analysis helpers
# =============================================================================

def _print_convergence(label, results_list, ref_idx, param_key):
    """Print convergence table against the reference entry."""
    ref = results_list[ref_idx]
    ref_ev = np.array(ref['eigenvalues'])
    ref_param = ref[param_key]
    print(f"\n  ── {label} Analysis (reference: {param_key}={ref_param}) ──")
    for r in results_list:
        ev = np.array(r['eigenvalues'])
        n = min(len(ev), len(ref_ev))
        diff = np.abs(ev[:n] - ref_ev[:n])
        bw = np.max(ev[:N_MODES]) - np.min(ev[:N_MODES])
        print(f"  {param_key}={r[param_key]:4d}: max|Δλ|={np.max(diff):.4e}, "
              f"mean|Δλ|={np.mean(diff):.4e}, BW={bw:.6f}")


def _print_fdfd_comparison(results, fdfd_freqs):
    """Print comparison against FDFD eigenvalues using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment

    for section_name in ['b1', 'b2', 'b3']:
        if section_name not in results:
            continue
        print(f"\n  ── {section_name.upper()} vs FDFD ──")
        for r in results[section_name]:
            ev = np.array(r['eigenvalues'])
            omega_ref = r['omega_ref']
            ea_freqs = omega_ref + ev[:N_MODES]
            fdfd_window = fdfd_freqs[
                (fdfd_freqs >= np.min(ea_freqs) - 0.002) &
                (fdfd_freqs <= np.max(ea_freqs) + 0.002)
            ]
            n = min(len(ea_freqs), len(fdfd_window))
            if n > 0:
                cost = np.abs(ea_freqs[:n, None] - fdfd_window[None, :n])
                ri, ci = linear_sum_assignment(cost)
                resids = np.abs(ea_freqs[ri] - fdfd_window[ci])
                label = f"reg={r.get('registry','?')},Ns={r.get('Ns','?')}"
                print(f"  {label:20s}: mean|Δ|={np.mean(resids)*1e6:.1f}×10⁻⁶, "
                      f"max|Δ|={np.max(resids)*1e6:.1f}×10⁻⁶, "
                      f"matched={len(ri)}/{N_MODES}")


# =============================================================================
# Plots
# =============================================================================

def generate_plots(results, fdfd_freqs=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_panels = sum(1 for k in ['b1', 'b2', 'b3'] if k in results)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    pi = 0

    # ── B1 ──
    if 'b1' in results:
        ax = axes[pi]; pi += 1
        b1 = results['b1']
        ref_ev = np.array(b1[-1]['eigenvalues'])
        regs, means, maxes = [], [], []
        for r in b1[:-1]:
            ev = np.array(r['eigenvalues'])
            n = min(len(ev), len(ref_ev))
            d = np.abs(ev[:n] - ref_ev[:n])
            regs.append(r['registry']); means.append(np.mean(d)); maxes.append(np.max(d))
        ax.semilogy(regs, means, 'o-', color='C0', label='mean |Δλ|')
        ax.semilogy(regs, maxes, 's--', color='C1', label='max |Δλ|')
        ax.set_xlabel('Registry samples')
        ax.set_ylabel('|Δλ| (vs registry=128)')
        ax.set_title('(a) B1: Registry convergence')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── B2 ──
    if 'b2' in results:
        ax = axes[pi]; pi += 1
        b2 = results['b2']
        ref_ev = np.array(b2[-1]['eigenvalues'])
        nss, means, maxes = [], [], []
        for r in b2[:-1]:
            ev = np.array(r['eigenvalues'])
            n = min(len(ev), len(ref_ev))
            d = np.abs(ev[:n] - ref_ev[:n])
            nss.append(r['Ns']); means.append(np.mean(d)); maxes.append(np.max(d))
        ax.semilogy(nss, means, 'o-', color='C0', label='mean |Δλ|')
        ax.semilogy(nss, maxes, 's--', color='C1', label='max |Δλ|')
        ax.set_xlabel('Ns (Hamiltonian grid)')
        ax.set_ylabel('|Δλ| (vs Ns=256)')
        ax.set_title('(b) B2: Ns convergence')
        ax.legend(); ax.grid(True, alpha=0.3)

    # ── B3 ──
    if 'b3' in results:
        ax = axes[pi]; pi += 1
        b3 = results['b3']
        if fdfd_freqs is not None:
            from scipy.optimize import linear_sum_assignment
            regs, mean_r, max_r = [], [], []
            for r in b3:
                ev = np.array(r['eigenvalues'])
                ea_f = r['omega_ref'] + ev[:N_MODES]
                fw = fdfd_freqs[(fdfd_freqs >= np.min(ea_f) - 0.002) &
                                (fdfd_freqs <= np.max(ea_f) + 0.002)]
                n = min(len(ea_f), len(fw))
                if n > 0:
                    cost = np.abs(ea_f[:n, None] - fw[None, :n])
                    ri, ci = linear_sum_assignment(cost)
                    res = np.abs(ea_f[ri] - fw[ci])
                    regs.append(r['registry'])
                    mean_r.append(np.mean(res))
                    max_r.append(np.max(res))
            if regs:
                ax.semilogy(regs, [m*1e6 for m in mean_r], 'o-', color='C2',
                            label='mean |Δ| vs FDFD')
                ax.semilogy(regs, [m*1e6 for m in max_r], 's--', color='C3',
                            label='max |Δ| vs FDFD')
                ax.set_ylabel('|Δω| ×10⁻⁶ (vs FDFD res=40)')
        else:
            ref_ev = np.array(b3[-1]['eigenvalues'])
            regs, means = [], []
            for r in b3[:-1]:
                ev = np.array(r['eigenvalues'])
                n = min(len(ev), len(ref_ev))
                d = np.abs(ev[:n] - ref_ev[:n])
                regs.append(r['registry']); means.append(np.mean(d))
            ax.semilogy(regs, means, 'o-', color='C0')
            ax.set_ylabel(f'|Δλ| (vs reg=Ns={b3[-1]["registry"]})')
        ax.set_xlabel('Registry = Ns')
        ax.set_title('(c) B3: Honest combined')
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'fig_phaseB_convergence.{ext}', dpi=150,
                    bbox_inches='tight')
    plt.close()
    print(f"  Plots saved: fig_phaseB_convergence.{{png,pdf}}")


if __name__ == '__main__':
    main()
