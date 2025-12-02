"""
Phase 0: Monolayer Candidate Search & Scoring (Library-Based)

This is a drop-in replacement for phase0_candidate_search.py that uses
pre-computed band structures from an HDF5 library instead of running MPB.

ADVANTAGES:
- Much faster execution (no MPB calls needed)
- Access to BOTH polarizations (TE and TM)
- Larger parameter space coverage from pre-computed data

The HDF5 library structure:
- scans/<scan_id>/axes/lattice_type: ['square', 'hex']
- scans/<scan_id>/axes/polarization: ['TE', 'TM']
- scans/<scan_id>/axes/eps_bg: array of epsilon values
- scans/<scan_id>/axes/r_over_a: array of r/a values
- scans/<scan_id>/axes/hole_eps: array (typically [1.0] for air)
- scans/<scan_id>/axes/k_path/<lattice>: k-point coordinates
- scans/<scan_id>/data/freq: (lattice, pol, hole, eps, r, band, k) array

High symmetry point indices in k-path (for 40 points/segment, 3 segments):
- Hex: Γ(0) → K(39) → M(78) → Γ(117)
- Square: Γ(0) → X(39) → M(78) → Γ(117)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import h5py
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.geometry import high_symmetry_points
from common.scoring import score_candidate
from common.io_utils import ensure_run_dir, load_yaml
from common.plotting import plot_top_candidates_grid


@dataclass
class BandLibrary:
    """Wrapper for the HDF5 band library."""
    file_path: Path
    scan_id: str
    lattice_types: Tuple[str, ...]
    polarizations: Tuple[str, ...]
    eps_bg: np.ndarray
    r_over_a: np.ndarray
    hole_eps: np.ndarray
    k_paths: Dict[str, np.ndarray]
    freq_data: np.ndarray  # Shape: (lattice, pol, hole, eps, r, band, k)
    num_bands: int
    num_kpoints: int
    hs_indices: Dict[str, Dict[str, int]]  # High symmetry point indices for each lattice type


def load_band_library(library_path: Path, scan_id: str = "square_hex_eps_r_v1") -> BandLibrary:
    """
    Load the pre-computed band library from HDF5.
    
    Args:
        library_path: Path to the HDF5 file
        scan_id: Scan identifier within the file
        
    Returns:
        BandLibrary object with all data loaded
    """
    with h5py.File(library_path, 'r') as f:
        scan = f[f'scans/{scan_id}']
        
        # Load axes
        lattice_types = tuple(scan['axes/lattice_type'].asstr()[...])
        polarizations = tuple(scan['axes/polarization'].asstr()[...])
        eps_bg = scan['axes/eps_bg'][...]
        r_over_a = scan['axes/r_over_a'][...]
        hole_eps = scan['axes/hole_eps'][...]
        
        # Load k-paths for each lattice type
        k_paths = {}
        for lattice in lattice_types:
            k_paths[lattice] = scan[f'axes/k_path/{lattice}'][...]
        
        # Load frequency data
        freq_data = scan['data/freq'][...]
        
        num_bands = freq_data.shape[5]
        num_kpoints = freq_data.shape[6]
    
    # Determine high symmetry point indices
    # For hex: Γ(0) → K(39) → M(78) → Γ(117)
    # For square: Γ(0) → X(39) → M(78) → Γ(117)
    hs_indices = {
        'hex': {'Γ': 0, 'K': 39, 'M': 78},
        'square': {'Γ': 0, 'X': 39, 'M': 78},
    }
    
    return BandLibrary(
        file_path=library_path,
        scan_id=scan_id,
        lattice_types=lattice_types,
        polarizations=polarizations,
        eps_bg=eps_bg,
        r_over_a=r_over_a,
        hole_eps=hole_eps,
        k_paths=k_paths,
        freq_data=freq_data,
        num_bands=num_bands,
        num_kpoints=num_kpoints,
        hs_indices=hs_indices,
    )


def find_closest_index(array: np.ndarray, value: float) -> int:
    """Find the index of the closest value in a sorted array."""
    idx = np.searchsorted(array, value)
    if idx == 0:
        return 0
    if idx == len(array):
        return len(array) - 1
    if abs(array[idx] - value) < abs(array[idx - 1] - value):
        return idx
    return idx - 1


def get_band_frequencies(
    library: BandLibrary,
    lattice_type: str,
    polarization: str,
    eps_bg: float,
    r_over_a: float,
    hole_eps: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Get band frequencies for a specific geometry from the library.
    
    Args:
        library: BandLibrary object
        lattice_type: 'square' or 'hex'
        polarization: 'TE' or 'TM'
        eps_bg: Background epsilon
        r_over_a: Hole radius / lattice constant
        hole_eps: Hole epsilon (default 1.0 for air)
        
    Returns:
        Array of shape (num_bands, num_kpoints) or None if not found
    """
    if lattice_type not in library.lattice_types:
        return None
    if polarization not in library.polarizations:
        return None
    
    lat_idx = library.lattice_types.index(lattice_type)
    pol_idx = library.polarizations.index(polarization)
    
    # Find closest epsilon and r/a values
    eps_idx = find_closest_index(library.eps_bg, eps_bg)
    r_idx = find_closest_index(library.r_over_a, r_over_a)
    hole_idx = find_closest_index(library.hole_eps, hole_eps)
    
    # Check if the values are within acceptable tolerance
    eps_tol = 0.15  # Tolerance for epsilon matching
    r_tol = 0.015   # Tolerance for r/a matching
    
    if abs(library.eps_bg[eps_idx] - eps_bg) > eps_tol:
        return None
    if abs(library.r_over_a[r_idx] - r_over_a) > r_tol:
        return None
    
    freqs = library.freq_data[lat_idx, pol_idx, hole_idx, eps_idx, r_idx, :, :]
    
    # Check for NaN (incomplete data)
    if np.any(np.isnan(freqs)):
        return None
    
    return freqs


def fit_local_dispersion_from_library(
    freqs: np.ndarray,
    k_path: np.ndarray,
    k_label: str,
    band_index: int,
    hs_indices: Dict[str, int],
) -> Dict[str, float]:
    """
    Extract local dispersion metrics for a band extremum from library data.
    
    This is adapted from mpb_utils.fit_local_dispersion but works directly
    with the library's frequency array format.
    
    Args:
        freqs: Frequency array of shape (num_bands, num_kpoints)
        k_path: K-point coordinates of shape (num_kpoints, 2)
        k_label: High symmetry point label ('Γ', 'K', 'M', 'X')
        band_index: Band index (0-based)
        hs_indices: Dict mapping k_label to k-point index
        
    Returns:
        Dict with dispersion metrics
    """
    n_bands, n_k = freqs.shape
    
    if band_index >= n_bands:
        band_index = n_bands - 1
    
    # Get k-point index for the high symmetry point
    if k_label not in hs_indices:
        k_idx = n_k // 2  # Default to middle
    else:
        k_idx = hs_indices[k_label]
    
    # Handle path wrapping (Γ point appears at both ends)
    wrap_path = np.allclose(k_path[0], k_path[-1])
    if wrap_path and k_idx == n_k - 1:
        k_idx = 0
    
    omega0 = float(freqs[band_index, k_idx])
    
    # Compute group velocity using finite differences
    prev_idx = k_idx - 1
    next_idx = k_idx + 1
    
    if prev_idx < 0:
        prev_idx = n_k - 2 if wrap_path else 0
    if next_idx >= n_k:
        next_idx = 1 if wrap_path else n_k - 1
    if prev_idx == next_idx:
        prev_idx = max(prev_idx - 1, 0)
        next_idx = min(next_idx + 1, n_k - 1)
    
    omega_prev = float(freqs[band_index, prev_idx])
    omega_next = float(freqs[band_index, next_idx])
    
    # K-space offsets
    k_prev = k_path[prev_idx]
    k_curr = k_path[k_idx]
    k_next = k_path[next_idx]
    
    dk_prev = float(np.linalg.norm(k_curr - k_prev))
    dk_next = float(np.linalg.norm(k_next - k_curr))
    chord_vec = k_next - k_prev
    chord_len = float(np.linalg.norm(chord_vec))
    
    if chord_len < 1e-9:
        vg = np.zeros(2)
        domega_dk = 0.0
    else:
        domega_dk = (omega_next - omega_prev) / chord_len
        tangent = chord_vec / chord_len
        vg = domega_dk * tangent
    
    vg_norm = float(np.linalg.norm(vg))
    vg_x = float(vg[0])
    vg_y = float(vg[1])
    
    # Compute curvature (second derivative)
    if dk_prev > 1e-9 and dk_next > 1e-9:
        term_next = (omega_next - omega0) / dk_next
        term_prev = (omega0 - omega_prev) / dk_prev
        d2omega_dk2 = 2.0 * (term_next - term_prev) / (dk_prev + dk_next)
    else:
        d2omega_dk2 = 0.0
    
    # For 2D, assume isotropic curvature at high symmetry points
    curvature_xx = abs(d2omega_dk2)
    curvature_yy = abs(d2omega_dk2)
    curvature_xy = 0.0
    curvature_trace = curvature_xx + curvature_yy
    curvature_det = curvature_xx * curvature_yy
    
    # Parabolic validity radius (rough estimate)
    if curvature_trace > 1e-6:
        k_parab = 0.2 / np.sqrt(curvature_trace)
    else:
        k_parab = 0.5
    
    # Extended parabola analysis
    parab_error_floor = 1e-4
    parab_error_threshold = 1.25
    
    def _neighbor_sample(offset):
        idx = k_idx + offset
        if wrap_path:
            idx = idx % n_k
        if idx < 0 or idx >= n_k or idx == k_idx:
            return None
        omega = float(freqs[band_index, idx])
        distance = float(np.linalg.norm(k_path[idx] - k_curr))
        if distance < 1e-12:
            return None
        return {'idx': idx, 'omega': omega, 'distance': distance}
    
    near_samples = [s for s in (_neighbor_sample(-1), _neighbor_sample(1)) if s]
    far_samples = [s for s in (_neighbor_sample(-2), _neighbor_sample(2)) if s]
    
    def _parabola_error(samples):
        if not samples:
            return None
        errors = []
        for sample in samples:
            s = sample['distance']
            expected_delta = 0.5 * d2omega_dk2 * (s ** 2)
            actual_delta = sample['omega'] - omega0
            scale = max(abs(expected_delta), parab_error_floor)
            errors.append(abs(actual_delta - expected_delta) / scale)
        return float(np.mean(errors)) if errors else None
    
    def _mean_distance(samples):
        if not samples:
            return None
        return float(np.mean([s['distance'] for s in samples]))
    
    parab_error_near = _parabola_error(near_samples)
    parab_error_far = _parabola_error(far_samples)
    near_span = _mean_distance(near_samples)
    far_span = _mean_distance(far_samples)
    
    k_parab_far = k_parab
    span_candidates = []
    if near_span is not None and (parab_error_near is None or parab_error_near <= parab_error_threshold):
        span_candidates.append(near_span)
    if far_span is not None and (parab_error_far is None or parab_error_far <= parab_error_threshold):
        span_candidates.append(far_span)
    if span_candidates:
        k_parab_far = max(k_parab_far, max(span_candidates))
    
    # Spectral gaps
    gap_above = 0.1
    gap_below = 0.1
    if band_index < n_bands - 1:
        gap_above = float(freqs[band_index + 1, k_idx]) - omega0
    if band_index > 0:
        gap_below = omega0 - float(freqs[band_index - 1, k_idx])
    
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
        'k_parab_far': k_parab_far,
        'parab_error_near': parab_error_near,
        'parab_error_far': parab_error_far,
        'gap_above': gap_above,
        'gap_below': gap_below,
    }
    
    return metrics


def assemble_candidate_row(
    candidate_id: int,
    lattice_type: str,
    polarization: str,
    r_over_a: float,
    eps_bg: float,
    band_index: int,
    k_label: str,
    k_vec: np.ndarray,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Assemble a candidate row with all metadata and computed metrics.
    """
    row = {
        'candidate_id': candidate_id,
        'lattice_type': lattice_type,
        'polarization': polarization,
        'a': 1.0,
        'r_over_a': r_over_a,
        'eps_bg': eps_bg,
        'band_index': band_index,
        'k_label': k_label,
        'k0_x': float(k_vec[0]),
        'k0_y': float(k_vec[1]),
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


def create_bands_dict_from_library(
    library: BandLibrary,
    lattice_type: str,
    polarization: str,
    eps_bg: float,
    r_over_a: float,
) -> Optional[Dict[str, Any]]:
    """
    Create a bands dictionary compatible with plot_band_structure.
    
    Returns the same structure as compute_bandstructure from mpb_utils.
    """
    freqs = get_band_frequencies(library, lattice_type, polarization, eps_bg, r_over_a)
    if freqs is None:
        return None
    
    k_path_coords = library.k_paths[lattice_type]
    
    # Build k_path distances for plotting
    n_k = k_path_coords.shape[0]
    k_path = np.zeros(n_k)
    for i in range(1, n_k):
        k_path[i] = k_path[i-1] + np.linalg.norm(k_path_coords[i] - k_path_coords[i-1])
    
    # High symmetry labels
    if lattice_type == 'hex':
        k_labels = ['Γ', 'K', 'M', 'Γ']
    else:  # square
        k_labels = ['Γ', 'X', 'M', 'Γ']
    
    # Label positions (segment boundaries)
    # With 40 points per segment and 3 segments: indices 0, 39, 78, 117
    k_break_indices = [0, 39, 78, n_k - 1]
    k_label_positions = np.array([k_path[idx] for idx in k_break_indices])
    
    return {
        'frequencies': freqs.T,  # Transpose to (n_k, n_bands) like MPB output
        'k_labels': k_labels,
        'k_path': k_path,
        'k_label_positions': k_label_positions,
        'k_break_indices': k_break_indices,
        'lattice_type': lattice_type,
        'polarization': polarization,
        'num_bands': freqs.shape[0],
        'k_interp': 39,  # Points per segment - 1
    }


def get_available_parameters(library: BandLibrary) -> Dict[str, Any]:
    """Get the available parameter ranges from the library."""
    return {
        'lattice_types': list(library.lattice_types),
        'polarizations': list(library.polarizations),
        'eps_bg_range': (float(library.eps_bg.min()), float(library.eps_bg.max())),
        'eps_bg_values': library.eps_bg.tolist(),
        'r_over_a_range': (float(library.r_over_a.min()), float(library.r_over_a.max())),
        'r_over_a_values': library.r_over_a.tolist(),
        'num_bands': library.num_bands,
        'num_kpoints': library.num_kpoints,
    }


def run_phase0_library(config_path: str):
    """
    Run Phase 0 using the pre-computed band library.
    
    This is a drop-in replacement for run_phase0 that uses library data
    instead of running MPB calculations.
    """
    print("=" * 70)
    print("Phase 0: Candidate Search & Scoring (Library-Based)")
    print("=" * 70)
    
    # Load configuration
    config = load_yaml(config_path)
    print(f"\nLoaded configuration from: {config_path}")
    print(f"Run name: {config.get('run_name', 'run')}")
    
    # Load band library
    library_path = Path(config.get('band_library_path', 
        '/home/renlephy/msl/research/band_diagram_scan/data/band_library.h5'))
    scan_id = config.get('band_library_scan_id', 'square_hex_eps_r_v1')
    
    print(f"\nLoading band library from: {library_path}")
    print(f"  Scan ID: {scan_id}")
    
    library = load_band_library(library_path, scan_id)
    available = get_available_parameters(library)
    
    print(f"\nLibrary contains:")
    print(f"  Lattice types: {available['lattice_types']}")
    print(f"  Polarizations: {available['polarizations']}")
    print(f"  ε_bg range: {available['eps_bg_range'][0]:.2f} - {available['eps_bg_range'][1]:.2f}")
    print(f"  r/a range: {available['r_over_a_range'][0]:.2f} - {available['r_over_a_range'][1]:.2f}")
    print(f"  Bands: {available['num_bands']}")
    print(f"  K-points: {available['num_kpoints']}")
    
    # Create run directory
    run_dir = ensure_run_dir(config)
    print(f"\nOutput directory: {run_dir}")
    
    # Extract search parameters
    # Support both explicit lists and 'use_library' mode
    lattice_types = config.get('lattice_types', list(library.lattice_types))
    polarizations = config.get('polarizations', list(library.polarizations))
    
    # For r/a and eps values, can use config lists or sample from library
    use_library_params = config.get('use_library_parameters', True)
    
    if use_library_params:
        # Use all available values from library (with optional subsampling)
        eps_step = config.get('eps_bg_step', 1)  # Sample every Nth value
        r_step = config.get('r_over_a_step', 1)
        
        eps_bg_list = library.eps_bg[::eps_step].tolist()
        r_over_a_list = library.r_over_a[::r_step].tolist()
        
        # Apply range filters if specified
        eps_min = config.get('eps_bg_min', 0)
        eps_max = config.get('eps_bg_max', 100)
        r_min = config.get('r_over_a_min', 0)
        r_max = config.get('r_over_a_max', 1)
        
        eps_bg_list = [e for e in eps_bg_list if eps_min <= e <= eps_max]
        r_over_a_list = [r for r in r_over_a_list if r_min <= r <= r_max]
    else:
        # Use explicit lists from config
        r_over_a_list = config.get('r_over_a_list', [0.2, 0.3, 0.4])
        eps_bg_list = config.get('eps_bg_list', [4.0, 6.0, 9.0])
    
    target_bands = config.get('target_bands', list(range(library.num_bands)))
    
    print(f"\nSearch space:")
    print(f"  Lattice types: {lattice_types}")
    print(f"  Polarizations: {polarizations}")
    print(f"  r/a values: {len(r_over_a_list)} points from {min(r_over_a_list):.2f} to {max(r_over_a_list):.2f}")
    print(f"  ε_bg values: {len(eps_bg_list)} points from {min(eps_bg_list):.1f} to {max(eps_bg_list):.1f}")
    print(f"  Target bands: {target_bands}")
    
    # Count total candidates
    total_geometries = len(lattice_types) * len(polarizations) * len(r_over_a_list) * len(eps_bg_list)
    print(f"\nTotal geometries to evaluate: {total_geometries}")
    
    # Generate candidates
    rows = []
    candidate_id = 0
    bands_cache = {}  # Cache for plotting
    skipped_count = 0
    
    pbar = tqdm(
        total=total_geometries,
        desc="Processing geometries",
        unit="geom"
    )
    
    for lattice_type in lattice_types:
        if lattice_type not in library.lattice_types:
            print(f"  Warning: {lattice_type} not in library, skipping")
            continue
        
        # Get high symmetry points for this lattice
        hs_points = high_symmetry_points(lattice_type)
        hs_indices = library.hs_indices.get(lattice_type, {})
        k_path = library.k_paths[lattice_type]
        
        for polarization in polarizations:
            if polarization not in library.polarizations:
                print(f"  Warning: {polarization} not in library, skipping")
                continue
            
            for r_over_a in r_over_a_list:
                for eps_bg in eps_bg_list:
                    pbar.update(1)
                    
                    # Get frequencies from library
                    freqs = get_band_frequencies(
                        library, lattice_type, polarization, eps_bg, r_over_a
                    )
                    
                    if freqs is None:
                        skipped_count += 1
                        continue
                    
                    # Cache for plotting
                    cache_key = (lattice_type, polarization, r_over_a, eps_bg)
                    bands_cache[cache_key] = create_bands_dict_from_library(
                        library, lattice_type, polarization, eps_bg, r_over_a
                    )
                    
                    # Create candidates for each k-point and band
                    for k_label, k_vec in hs_points:
                        # Skip if this k-point isn't in our indices
                        if k_label not in hs_indices and k_label != 'Γ':
                            # Try to handle Γ at end of path
                            if k_label == 'Γ':
                                continue
                        
                        for band_index in target_bands:
                            if band_index >= freqs.shape[0]:
                                continue
                            
                            # Compute dispersion metrics
                            metrics = fit_local_dispersion_from_library(
                                freqs, k_path, k_label, band_index, hs_indices
                            )
                            
                            # Assemble row
                            row = assemble_candidate_row(
                                candidate_id,
                                lattice_type,
                                polarization,
                                r_over_a,
                                eps_bg,
                                band_index,
                                k_label,
                                k_vec[:2],  # 2D k-vector
                                metrics,
                            )
                            
                            # Score candidate
                            scores = score_candidate(row, config)
                            row.update(scores)
                            
                            rows.append(row)
                            candidate_id += 1
    
    pbar.close()
    
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} geometries (not in library or incomplete data)")
    
    print(f"\nGenerated {len(rows)} candidates")
    
    # Create DataFrame and sort by score
    df = pd.DataFrame(rows)
    df['candidate_source'] = 'library'
    df.sort_values('S_total', ascending=False, inplace=True)
    
    # Reset candidate IDs based on ranking
    df['candidate_id'] = range(len(df))
    
    # Save full results
    output_file = run_dir / 'phase0_candidates.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved candidates to: {output_file}")
    
    # Get top K candidates
    K_display = config.get('K_candidates', 16)
    top_candidates = df.head(K_display)
    
    print(f"\nTop {K_display} candidates:")
    display_cols = ['candidate_id', 'lattice_type', 'polarization',
                    'r_over_a', 'eps_bg', 'k_label', 'band_index', 
                    'S_total', 'valid_ea_flag']
    print(top_candidates[display_cols].to_string(index=False))
    
    # Plot band diagrams for top candidates
    if bands_cache:
        print(f"\nGenerating band diagram plots for top {K_display} candidates...")
        
        bands_list = []
        for _, row in tqdm(top_candidates.iterrows(), total=len(top_candidates),
                          desc="Gathering band data", unit="candidate"):
            cache_key = (row['lattice_type'], row['polarization'],
                        row['r_over_a'], row['eps_bg'])
            if cache_key in bands_cache:
                bands_list.append(bands_cache[cache_key])
            else:
                # Try to create from library
                bands = create_bands_dict_from_library(
                    library, row['lattice_type'], row['polarization'],
                    row['eps_bg'], row['r_over_a']
                )
                if bands is not None:
                    bands_list.append(bands)
                else:
                    # Create placeholder
                    bands_list.append({
                        'frequencies': np.zeros((1, 1)),
                        'k_labels': [],
                        'k_path': np.array([0]),
                        'k_label_positions': np.array([0]),
                        'k_break_indices': [0],
                    })
        
        plot_path = run_dir / 'phase0_top_candidates_bands.png'
        plot_top_candidates_grid(top_candidates, bands_list, plot_path, n_cols=4)
        print(f"  Saved to: {plot_path}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Valid EA candidates: {df['valid_ea_flag'].sum()} / {len(df)}")
    print(f"  Mean score: {df['S_total'].mean():.4f}")
    print(f"  Max score: {df['S_total'].max():.4f}")
    print(f"  Min score: {df['S_total'].min():.4f}")
    
    # Polarization breakdown
    if 'polarization' in df.columns:
        pol_counts = df['polarization'].value_counts().to_dict()
        print(f"  Polarization breakdown: {pol_counts}")
        
        # Top candidates by polarization
        for pol in polarizations:
            pol_top = df[df['polarization'] == pol].head(3)
            if len(pol_top) > 0:
                print(f"\n  Top 3 {pol} candidates:")
                print(pol_top[display_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Phase 0 (Library-Based) complete!")
    print("=" * 70)
    
    return run_dir, df


if __name__ == "__main__":
    # Default config path
    default_config = Path(__file__).parent.parent / "configs" / "phase0_library.yaml"
    
    if len(sys.argv) < 2:
        if default_config.exists():
            print(f"Using default config: {default_config}")
            config_path = str(default_config)
        else:
            print("Usage: python phase0_candidate_search_library.py [config_path]")
            print(f"\nDefault config not found: {default_config}")
            print("\nThis script uses pre-computed band structures from an HDF5 library")
            print("instead of running MPB calculations. Supports both TE and TM polarizations.")
            sys.exit(1)
    else:
        config_path = sys.argv[1]
    
    run_phase0_library(config_path)
