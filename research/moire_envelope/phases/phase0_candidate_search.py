"""
Phase 0: Monolayer Candidate Search & Scoring

IMPORTANT: This phase analyzes MONOLAYER photonic crystals only.
No moiré physics or twist angles are involved at this stage.

This phase explores the design space of monolayer photonic crystal geometries
and outputs K best candidates based on their band structure properties at high
symmetry points. These candidates will be used in subsequent phases to construct
moiré superlattices.

For each combination of (lattice_type, r, eps_bg, k_point, band_index),
we compute monolayer band structure, extract local dispersion metrics
(frequency, group velocity, effective mass), and score candidates based on
their suitability for envelope approximation in future moiré systems.

Key metrics evaluated:
- Band extrema (high curvature, strong isolation)
- Spectral isolation (band gaps above/below the extremum)
- Parabolic fidelity (reject saddle points and highly anisotropic extrema)
- Parabola span (reward extrema that stay quadratic over a wide momentum range)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

try:
    from scipy import optimize
except ImportError:  # pragma: no cover - optional dependency
    optimize = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.geometry import build_lattice, high_symmetry_points
from common.mpb_utils import compute_bandstructure, fit_local_dispersion
from common.scoring import score_candidate
from common.io_utils import ensure_run_dir, load_yaml


def _env_flag(name, default=None):
    """Interpret environment variables like 1/true/on as booleans."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    return default
def _find_latest_run_dir(config):
    base_dir = Path(config.get('output_dir', 'runs'))
    run_name = config.get('run_name', 'run')
    candidates = sorted(base_dir.glob(f"{run_name}_*"))
    if not candidates:
        raise FileNotFoundError(
            f"No existing Phase 0 runs found under '{base_dir}' with prefix '{run_name}_'."
        )
    return candidates[-1]
from common.plotting import (
    plot_top_candidates_grid,
    plot_optimizer_candidate_summary,
)


def compute_geometry_bands(geom_params, config, num_bands):
    """
    Worker function to compute band structure for a single geometry.
    
    Args:
        geom_params: Tuple of (lattice_type, r_over_a, eps_bg)
        config: Configuration dictionary
        num_bands: Number of bands to calculate
        
    Returns:
        Tuple of (geom_params, bands) where bands is the computed band structure
    """
    lattice_type, r_over_a, eps_bg = geom_params
    geom = build_lattice(lattice_type, r_over_a, eps_bg, a=1.0)
    
    # Ensure MPB output is suppressed in workers
    config_copy = config.copy()
    config_copy['quiet_mpb'] = True

    # Limit per-worker threading to avoid CPU oversubscription
    per_worker_threads = config_copy.get('mpb_threads_per_worker')
    if per_worker_threads is None:
        per_worker_threads = config_copy.get('per_worker_threads')
    if per_worker_threads is None:
        n_workers = config_copy.get('_n_workers', 1)
        if n_workers > 1 and config_copy.get('lock_worker_threads', True):
            # Default to 1 thread per worker when parallelizing
            per_worker_threads = 1
    if per_worker_threads:
        thread_str = str(max(1, int(per_worker_threads)))
        os.environ['OMP_NUM_THREADS'] = thread_str
        os.environ['OMP_THREAD_LIMIT'] = thread_str
        os.environ['OPENBLAS_NUM_THREADS'] = thread_str
        os.environ['MKL_NUM_THREADS'] = thread_str
    
    bands = compute_bandstructure(geom, config_copy, num_bands=num_bands)
    bands['k_interp'] = config.get('k_interp', 19)
    return (geom_params, bands)


def assemble_candidate_row(candidate_id, lattice_type, r_over_a, eps_bg,
                           band_index, k_label, k_vec, metrics):
    """
    Assemble a candidate row with all metadata and computed metrics
    
    Args:
        candidate_id: Unique candidate ID
        lattice_type: Lattice type string
        r_over_a: Normalized hole radius
        eps_bg: Background dielectric constant
        band_index: Band index
        k_label: High symmetry point label
        k_vec: k-vector
        metrics: Dispersion metrics dictionary
        
    Returns:
        dict: Complete candidate row
    """
    row = {
        'candidate_id': candidate_id,
        'lattice_type': lattice_type,
        'a': 1.0,  # Default lattice constant
        'r_over_a': r_over_a,
        'eps_bg': eps_bg,
        'band_index': band_index,
        'k_label': k_label,
        'k0_x': k_vec[0],
        'k0_y': k_vec[1],
        'omega0': metrics['omega0'],
        'curvature_xx': metrics['curvature_xx'],
        'curvature_xy': metrics['curvature_xy'],
        'curvature_yy': metrics['curvature_yy'],
        'curvature_trace': metrics['curvature_trace'],
        'curvature_det': metrics['curvature_det'],
        'vg_x': metrics['vg_x'],
        'vg_y': metrics['vg_y'],
        'vg_norm': metrics['vg_norm'],
        'k_parab': metrics['k_parab'],
        'k_parab_far': metrics.get('k_parab_far', metrics['k_parab']),
        'parab_error_near': metrics.get('parab_error_near'),
        'parab_error_far': metrics.get('parab_error_far'),
        'gap_above': metrics['gap_above'],
        'gap_below': metrics['gap_below'],
        'gap_min': min(metrics['gap_above'], metrics['gap_below']),
    }
    
    return row


def _bands_cache_key(lattice_type: str, r_over_a: float, eps_bg: float) -> tuple:
    return (lattice_type, float(f"{r_over_a:.6f}"), float(f"{eps_bg:.6f}"))


def _ensure_bandstructure(lattice_type, r_over_a, eps_bg, config, bands_cache, num_bands):
    key = _bands_cache_key(lattice_type, r_over_a, eps_bg)
    if key in bands_cache:
        return bands_cache[key]

    geom = build_lattice(lattice_type, r_over_a, eps_bg, a=1.0)
    bands = compute_bandstructure(geom, config, num_bands=num_bands)
    bands['k_interp'] = config.get('k_interp', 19)
    _store_bandstructure(bands_cache, lattice_type, r_over_a, eps_bg, bands)
    return bands


def _store_bandstructure(bands_cache, lattice_type, r_over_a, eps_bg, bands):
    direct_key = (lattice_type, r_over_a, eps_bg)
    bands_cache[direct_key] = bands
    bands_cache[_bands_cache_key(lattice_type, r_over_a, eps_bg)] = bands


def _evaluate_continuous_candidate(lattice_type, r_over_a, eps_bg, band_index, hs_point,
                                   config, bands_cache, num_bands):
    k_label, k_vec = hs_point
    bands = _ensure_bandstructure(lattice_type, r_over_a, eps_bg, config, bands_cache, num_bands)
    metrics = fit_local_dispersion(bands, k_label, band_index)
    row = assemble_candidate_row(
        candidate_id=-1,
        lattice_type=lattice_type,
        r_over_a=r_over_a,
        eps_bg=eps_bg,
        band_index=band_index,
        k_label=k_label,
        k_vec=k_vec,
        metrics=metrics,
    )
    scores = score_candidate(row, config)
    row.update(scores)
    return row


def _optimize_lattice_type(lattice_type, config, bands_cache, num_bands):
    if optimize is None:
        raise RuntimeError("SciPy is not available; install scipy to enable optimization.")

    hs_points = high_symmetry_points(lattice_type)
    if not hs_points:
        return None

    r_list = config.get('r_over_a_list', [0.2, 0.3, 0.4])
    eps_list = config.get('eps_bg_list', [4.0, 6.0, 9.0])
    band_list = config.get('target_bands', list(range(8)))

    r_bounds = tuple(config.get('phase0_opt_r_bounds', [min(r_list), max(r_list)]))
    eps_bounds = tuple(config.get('phase0_opt_eps_bounds', [min(eps_list), max(eps_list)]))
    band_bounds = (min(band_list), max(band_list))
    k_bounds = (0, len(hs_points) - 1)

    maxiter = max(1, int(config.get('phase0_opt_maxiter', 12)))
    popsize = int(config.get('phase0_opt_popsize', 6))
    polish = bool(config.get('phase0_opt_polish', False))
    tol = float(config.get('phase0_opt_tol', 1e-3))

    best = {'score': -np.inf, 'row': None, 'x': None}
    eval_cache = {}

    def _objective(x):
        r = float(np.clip(x[0], *r_bounds))
        eps = float(np.clip(x[1], *eps_bounds))
        band_idx = int(round(np.clip(x[2], band_bounds[0], band_bounds[1])))
        k_idx = int(round(np.clip(x[3], k_bounds[0], k_bounds[1])))

        cache_key = (round(r, 5), round(eps, 5), band_idx, k_idx)
        cached = eval_cache.get(cache_key)
        if cached is None:
            try:
                row = _evaluate_continuous_candidate(
                    lattice_type,
                    r,
                    eps,
                    band_idx,
                    hs_points[k_idx],
                    config,
                    bands_cache,
                    num_bands,
                )
                s_total = float(row['S_total'])
                eval_cache[cache_key] = {'score': s_total, 'row': row}
            except Exception:
                eval_cache[cache_key] = {'score': -np.inf, 'row': None}
                s_total = -np.inf
        else:
            s_total = cached['score']

        if s_total > best['score'] and np.isfinite(s_total):
            best['score'] = s_total
            best['row'] = eval_cache[cache_key]['row']
            best['x'] = {
                'r_over_a': r,
                'eps_bg': eps,
                'band_index': band_idx,
                'k_index': k_idx,
            }

        return -s_total if np.isfinite(s_total) else 1e6

    progress = tqdm(
        total=maxiter,
        desc=f"Opt {lattice_type}",
        unit="iter",
        leave=False,
    )

    def _callback(xk, convergence=None):  # pragma: no cover - SciPy callback
        if progress.n < progress.total:
            progress.update(1)
        if np.isfinite(best['score']):
            progress.set_postfix(best=f"{best['score']:.3f}")
        return False

    try:
        bounds = [r_bounds, eps_bounds, band_bounds, k_bounds]
        optimize.differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            updating='deferred',
            disp=False,
            callback=_callback,
        )
    finally:
        if progress.n < progress.total:
            progress.update(progress.total - progress.n)
        progress.close()

    if best['row'] is None:
        return None

    k_idx = best['x']['k_index']
    best['row']['optimization_k_label'] = hs_points[k_idx][0]
    best['row']['optimization_strategy'] = 'differential_evolution'
    return best['row']


def _run_continuous_optimizer(
    lattice_types,
    config,
    bands_cache,
    num_bands,
    candidate_id_start=0,
    run_dir=None,
):
    if not config.get('phase0_enable_optimizer', True):
        return [], candidate_id_start
    if optimize is None:
        print("SciPy is unavailable; skipping continuous optimization.")
        return [], candidate_id_start

    results = []
    next_candidate_id = int(candidate_id_start)
    run_dir_path = Path(run_dir) if run_dir is not None else None

    for lattice_type in lattice_types:
        print(f"  Optimizing lattice type '{lattice_type}'...")

        try:
            row = _optimize_lattice_type(lattice_type, config, bands_cache, num_bands)
        except RuntimeError as exc:
            print(f"    Skipping optimizer for {lattice_type}: {exc}")
            row = None

        if row is not None:
            row['optimization_lattice'] = lattice_type
            row['candidate_id'] = next_candidate_id
            row.setdefault('candidate_source', 'optimizer')
            results.append(row)
            print(
                f"    Best score {row['S_total']:.4f} at r/a={row['r_over_a']:.4f}, "
                f"eps_bg={row['eps_bg']:.3f}, band={row['band_index']}, "
                f"k={row.get('optimization_k_label', row['k_label'])}"
            )

            if run_dir_path is not None:
                geom = build_lattice(
                    lattice_type,
                    row['r_over_a'],
                    row['eps_bg'],
                    a=float(row.get('a', 1.0)),
                )
                bands = _ensure_bandstructure(
                    lattice_type,
                    row['r_over_a'],
                    row['eps_bg'],
                    config,
                    bands_cache,
                    num_bands,
                )
                plot_path = run_dir_path / f"phase0_optimizer_{lattice_type}_{next_candidate_id:04d}.png"
                plot_optimizer_candidate_summary(geom, bands, row, plot_path)
                print(f"    Saved optimizer plot: {plot_path.name}")

            next_candidate_id += 1
        else:
            print(f"    No feasible solution found for {lattice_type}.")

    return results, next_candidate_id


def run_phase0(config_path: str):
    """
    Run Phase 0: Candidate search and scoring
    
    This explores the design space and outputs K best candidates.
    
    Args:
        config_path: Path to configuration YAML file
    """
    print("=" * 70)
    print("Phase 0: Candidate Search & Scoring")
    print("=" * 70)
    
    # Load configuration
    config = load_yaml(config_path)
    env_disable_opt = _env_flag('MSL_PHASE0_DISABLE_OPT')
    if env_disable_opt is not None:
        config['phase0_enable_optimizer'] = not env_disable_opt
    env_enable_opt = _env_flag('MSL_PHASE0_ENABLE_OPT')
    if env_enable_opt is not None:
        config['phase0_enable_optimizer'] = env_enable_opt
    print(f"\nLoaded configuration from: {config_path}")
    print(f"Run name: {config.get('run_name', 'run')}")
    
    # Create run directory
    run_dir = ensure_run_dir(config)
    print(f"Output directory: {run_dir}")
    
    # Extract search parameters (monolayer only)
    lattice_types = config.get('lattice_types', ['square', 'hex'])
    r_over_a_list = config.get('r_over_a_list', [0.2, 0.3, 0.4])
    eps_bg_list = config.get('eps_bg_list', [4.0, 6.0, 9.0])
    target_bands = config.get('target_bands', list(range(8)))  # 0-7 for 8 bands
    
    # MPB settings
    use_simplified = config.get('use_simplified_model', False)
    num_bands = config.get('num_bands', 8)
    
    print(f"\nMonolayer search space:")
    print(f"  Lattice types: {lattice_types}")
    print(f"  Hole radii (r/a): {r_over_a_list}")
    print(f"  Background ε: {eps_bg_list}")
    print(f"  Target bands: {target_bands}")
    print(f"  Number of bands to calculate: {num_bands}")
    print(f"  Use real MPB: {not use_simplified}")
    
    # Setup multiprocessing
    n_cpus = mp.cpu_count()
    n_workers = max(1, n_cpus - 2)  # Use all but 2 cores by default

    # Allow configuration override for worker count
    max_workers_cfg = config.get('max_workers')
    if isinstance(max_workers_cfg, int) and max_workers_cfg > 0:
        n_workers = min(n_workers, max_workers_cfg)
    
    # Build list of unique geometries to compute
    unique_geometries = []
    for lattice_type in lattice_types:
        for r_over_a in r_over_a_list:
            for eps_bg in eps_bg_list:
                geom_key = (lattice_type, r_over_a, eps_bg)
                unique_geometries.append(geom_key)
    
    total_geometries = len(unique_geometries)
    
    # Adjust workers if there are fewer geometries than workers
    if not use_simplified and total_geometries < n_workers:
        n_workers = max(1, total_geometries)
    
    print(f"\nParallel processing:")
    print(f"  Total CPUs: {n_cpus}")
    print(f"  Workers: {n_workers}")
    
    # Generate candidate list
    rows = []
    bands_cache = {}  # Cache band structures by geometry parameters
    candidate_id = 0
    
    print(f"\nGenerating candidates...")
    print(f"Total unique geometries to calculate: {total_geometries}")
    
    # Compute band structures (serial baseline + optional parallel)
    if not use_simplified and total_geometries > 0:
        # Always compute the first geometry serially to warm caches and measure runtime
        baseline_geom = unique_geometries.pop(0)
        geom = build_lattice(baseline_geom[0], baseline_geom[1], baseline_geom[2], a=1.0)
        t_start = time.perf_counter()
        baseline_bands = compute_bandstructure(geom, config, num_bands=num_bands)
        baseline_bands['k_interp'] = config.get('k_interp', 19)
        _store_bandstructure(bands_cache, baseline_geom[0], baseline_geom[1], baseline_geom[2], baseline_bands)
        baseline_time = time.perf_counter() - t_start
        
        remaining_geoms = unique_geometries
        remaining_count = len(remaining_geoms)
        
        # Decide whether to parallelize remaining geometries
        threshold = config.get('parallel_geom_time_threshold', 2.0)
        force_parallel = config.get('force_parallel', False)
        should_parallel = (
            force_parallel
            and remaining_count > 0
            and n_workers > 1
        ) or (
            not force_parallel
            and remaining_count > 0
            and n_workers > 1
            and baseline_time >= threshold
        )
        
        with tqdm(total=total_geometries, desc="Band structures", unit="geom") as pbar:
            pbar.update(1)  # baseline geometry already computed
            
            if should_parallel:
                print(f"Computing remaining {remaining_count} band structures in parallel with {n_workers} workers...")
                worker_config = config.copy()
                worker_config['_n_workers'] = n_workers
                threads_per_worker = config.get('mpb_threads_per_worker')
                if threads_per_worker is None and n_workers > 1:
                    threads_per_worker = max(1, n_cpus // n_workers)
                    worker_config['mpb_threads_per_worker'] = threads_per_worker
                compute_func = partial(compute_geometry_bands, config=worker_config, num_bands=num_bands)
                with mp.Pool(processes=n_workers) as pool:
                    for geom_params, bands in pool.imap_unordered(compute_func, remaining_geoms, chunksize=1):
                        _store_bandstructure(bands_cache, geom_params[0], geom_params[1], geom_params[2], bands)
                        pbar.update(1)
            else:
                if remaining_count > 0:
                    print("Computing remaining band structures serially (parallelism not beneficial for this workload)...")
                for geom_params in remaining_geoms:
                    geom = build_lattice(geom_params[0], geom_params[1], geom_params[2], a=1.0)
                    bands = compute_bandstructure(geom, config, num_bands=num_bands)
                    bands['k_interp'] = config.get('k_interp', 19)
                    _store_bandstructure(bands_cache, geom_params[0], geom_params[1], geom_params[2], bands)
                    pbar.update(1)
        
        print(f"✓ Completed {len(bands_cache)} band structure calculations (baseline {baseline_time:.3f}s per geometry)")
    
    # Now generate candidates from cached band structures
    print(f"\nGenerating candidates from band structures...")
    
    pbar_cand = tqdm(
        total=len(lattice_types) * len(r_over_a_list) * len(eps_bg_list),
        desc="Creating candidates",
        unit="geom"
    )
    
    for lattice_type in lattice_types:
        # Get high symmetry points for this lattice
        hs_points = high_symmetry_points(lattice_type)
        
        for r_over_a in r_over_a_list:
            for eps_bg in eps_bg_list:
                # Get band structure from cache (already computed in parallel)
                geom_key = (lattice_type, r_over_a, eps_bg)
                bands = None
                if not use_simplified:
                    bands = bands_cache[geom_key]
                
                # Create candidates for each k-point and band
                for k_label, k_vec in hs_points:
                    for band_index in target_bands:
                        if use_simplified:
                            # Generate simplified metrics
                            metrics = generate_simplified_metrics(
                                lattice_type, k_label, band_index, 
                                r_over_a, eps_bg, config
                            )
                        else:
                            # Extract real metrics from MPB
                            metrics = fit_local_dispersion(
                                bands, k_label, band_index
                            )
                        
                        # Assemble row
                        row = assemble_candidate_row(
                            candidate_id, lattice_type,
                            r_over_a, eps_bg, band_index, k_label, 
                            k_vec, metrics
                        )
                        
                        # Score candidate
                        scores = score_candidate(row, config)
                        row.update(scores)
                        
                        rows.append(row)
                        candidate_id += 1
                
                pbar_cand.update(1)
    
    pbar_cand.close()
    
    print(f"\nGenerated {len(rows)} candidates")
    
    # Create DataFrame and sort by score
    df = pd.DataFrame(rows)
    df['candidate_source'] = 'grid'
    df.sort_values('S_total', ascending=False, inplace=True)

    optimizer_rows = []
    if not use_simplified:
        print("\nRunning continuous optimization per lattice type...")
        optimizer_rows, candidate_id = _run_continuous_optimizer(
            lattice_types,
            config,
            bands_cache,
            num_bands,
            candidate_id_start=candidate_id,
            run_dir=run_dir,
        )
        if optimizer_rows:
            opt_df = pd.DataFrame(optimizer_rows)
            if 'candidate_source' not in opt_df.columns:
                opt_df['candidate_source'] = 'optimizer'
            opt_path = run_dir / 'phase0_optimizer_results.csv'
            opt_df.to_csv(opt_path, index=False)
            print(f"Saved optimizer summary to: {opt_path}")

            df = pd.concat([df, opt_df], ignore_index=True)
            df.sort_values('S_total', ascending=False, inplace=True)
            print("Merged optimizer picks into phase0_candidates.csv")
    else:
        print("\nSkipping continuous optimization because simplified model is enabled.")

    # Save full results (grid + optimizer)
    output_file = run_dir / 'phase0_candidates.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved candidates to: {output_file}")

    # Get top K candidates using merged list
    K_display = config.get('K_candidates', 16)
    top_candidates = df.head(K_display)

    print(f"\nTop {K_display} candidates:")
    print(top_candidates[['candidate_id', 'lattice_type',
                          'r_over_a', 'eps_bg', 'k_label', 'band_index', 
                          'S_total', 'valid_ea_flag', 'candidate_source']].to_string(index=False))

    # Plot band diagrams for top candidates if using real MPB
    if not use_simplified and bands_cache:
        print(f"\nGenerating band diagram plots for top {K_display} candidates...")

        bands_list = []
        for _, row in tqdm(top_candidates.iterrows(), total=len(top_candidates), 
                          desc="Gathering band data", unit="candidate"):
            geom_key = (row['lattice_type'], row['r_over_a'], row['eps_bg'])
            if geom_key in bands_cache:
                bands_list.append(bands_cache[geom_key])
            else:
                geom = build_lattice(row['lattice_type'], row['r_over_a'], 
                                    row['eps_bg'], a=1.0)
                bands = compute_bandstructure(geom, config, num_bands=num_bands)
                bands['k_interp'] = config.get('k_interp', 19)
                bands_list.append(bands)

        plot_path = run_dir / 'phase0_top_candidates_bands.png'
        plot_top_candidates_grid(top_candidates, bands_list, plot_path, n_cols=4)
        print(f"  Saved to: {plot_path}")

    # Statistics after merging
    print(f"\nStatistics:")
    print(f"  Valid EA candidates: {df['valid_ea_flag'].sum()} / {len(df)}")
    print(f"  Mean score: {df['S_total'].mean():.4f}")
    print(f"  Max score: {df['S_total'].max():.4f}")
    print(f"  Min score: {df['S_total'].min():.4f}")
    if 'candidate_source' in df.columns:
        source_counts = df['candidate_source'].value_counts().to_dict()
        print(f"  Source breakdown: {source_counts}")

    print("\n" + "=" * 70)
    print("Phase 0 complete!")
    print("=" * 70)
    
    return run_dir, df


def run_phase0_optimizer_only(config_path: str):
    config = load_yaml(config_path)
    print("=" * 70)
    print("Phase 0 Optimizer-Only Mode")
    print("=" * 70)

    run_dir = _find_latest_run_dir(config)
    print(f"Using latest Phase 0 run directory: {run_dir}")

    candidates_csv = run_dir / 'phase0_candidates.csv'
    if not candidates_csv.exists():
        raise FileNotFoundError(
            f"Cannot locate {candidates_csv}. Run full Phase 0 before optimizer-only mode."
        )

    df = pd.read_csv(candidates_csv)
    if 'candidate_source' not in df.columns:
        df['candidate_source'] = 'grid'
    print(f"Loaded {len(df)} cached candidates from previous grid search.")

    if config.get('use_simplified_model', False):
        print("Simplified model is enabled; skipping SciPy optimizer.")
        return run_dir

    lattice_types = config.get('lattice_types', ['square', 'hex'])
    num_bands = config.get('num_bands', 8)
    bands_cache = {}
    next_candidate_id = int(df['candidate_id'].max()) + 1 if len(df) else 0

    optimizer_rows, _ = _run_continuous_optimizer(
        lattice_types,
        config,
        bands_cache,
        num_bands,
        candidate_id_start=next_candidate_id,
        run_dir=run_dir,
    )
    if optimizer_rows:
        opt_df = pd.DataFrame(optimizer_rows)
        if 'candidate_source' not in opt_df.columns:
            opt_df['candidate_source'] = 'optimizer'
        opt_path = run_dir / 'phase0_optimizer_results.csv'
        opt_df.to_csv(opt_path, index=False)
        print(f"Saved optimizer summary to: {opt_path}")

        df = pd.concat([df, opt_df], ignore_index=True)
        df.sort_values('S_total', ascending=False, inplace=True)
        df.to_csv(candidates_csv, index=False)
        print(f"Updated {candidates_csv.name} with optimizer picks (total {len(df)} rows).")
    else:
        print("No optimizer results produced.")

    print("Phase 0 optimizer-only run complete.\n")
    return run_dir


def generate_simplified_metrics(lattice_type, k_label, band_index, 
                                r_over_a, eps_bg, config):
    """
    Generate simplified analytical metrics for rapid design space exploration
    
    This uses approximate formulas and scaling laws instead of full MPB.
    
    Args:
        lattice_type: Lattice type
        k_label: High symmetry point label
        band_index: Band index
        r_over_a: Normalized hole radius
        eps_bg: Background dielectric constant
        config: Configuration dict
        
    Returns:
        dict: Metrics dictionary
    """
    # Base frequency scales with sqrt(eps_bg) and band index
    omega_base = 0.3 + 0.1 * band_index / np.sqrt(eps_bg)
    
    # Frequency depends on k-point
    if k_label == 'Γ':
        omega0 = omega_base
    elif k_label == 'M':
        omega0 = omega_base * 1.2
    elif k_label == 'K':
        omega0 = omega_base * 1.15
    elif k_label == 'X':
        omega0 = omega_base * 1.1
    elif k_label == 'Y':
        omega0 = omega_base * 1.05
    else:
        omega0 = omega_base
    
    # Add some variation based on r/a
    omega0 *= (1 + 0.1 * (r_over_a - 0.3))
    
    # Group velocity (small at band extrema)
    if k_label == 'Γ':
        vg_x, vg_y = 0.0, 0.0
    elif k_label == 'M':
        vg_x, vg_y = 0.05, 0.05
    else:
        vg_x, vg_y = 0.02, 0.02
    vg_norm = np.sqrt(vg_x**2 + vg_y**2)
    
    # Curvature (effective mass)
    # Flatter bands (lower curvature) at higher ε and band index
    curvature_scale = 1.0 / (eps_bg * 0.2) * (1 + 0.1 * band_index)
    curvature_xx = curvature_scale * (1 + 0.2 * np.random.rand())
    curvature_yy = curvature_scale * (1 + 0.2 * np.random.rand())
    curvature_xy = 0.0  # Assume aligned axes
    curvature_trace = curvature_xx + curvature_yy
    curvature_det = curvature_xx * curvature_yy - curvature_xy**2
    
    # Parabolic validity radius (larger for flatter bands)
    k_parab = 0.2 + 0.1 / curvature_trace
    
    # Spectral gaps (larger at band center, smaller at edges)
    gap_base = 0.05 * np.sqrt(eps_bg) * (r_over_a / 0.3)
    gap_above = gap_base * (1 + 0.3 * np.random.rand())
    gap_below = gap_base * (1 + 0.3 * np.random.rand())
    
    metrics = {
        'omega0': omega0,
        'vg_x': vg_x,
        'vg_y': vg_y,
        'vg_norm': vg_norm,
        'curvature_xx': curvature_xx,
        'curvature_xy': curvature_xy,
        'curvature_yy': curvature_yy,
        'curvature_trace': curvature_trace,
        'curvature_det': curvature_det,
        'k_parab': k_parab,
        'k_parab_far': k_parab * 1.5,
        'parab_error_near': 0.4 + 0.2 * np.random.rand(),
        'parab_error_far': 0.8 + 0.3 * np.random.rand(),
        'gap_above': gap_above,
        'gap_below': gap_below,
    }
    
    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase0_candidate_search.py [optimize] <config_path>")
        sys.exit(1)

    if len(sys.argv) >= 3 and sys.argv[1].lower() in {"opt", "optimize"}:
        run_phase0_optimizer_only(sys.argv[2])
    else:
        config_path = sys.argv[-1]
        run_phase0(config_path)
