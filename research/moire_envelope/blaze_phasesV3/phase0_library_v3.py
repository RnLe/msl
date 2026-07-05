"""
Phase 0 (BLAZE): Monolayer Candidate Search & Scoring — V3 Multi-Band Pipeline

This is Phase 0 for the V3 multi-band envelope approximation pipeline.
It uses pre-computed band structures from an HDF5 library and selects
candidates with their surrounding band neighborhood for multi-band treatment.

V3 KEY FEATURES:
- Single candidate selection (same as V2)
- Records band neighborhood information for multi-band subspace
- Stores n_neighbor_bands and n_extra_bands in output for subsequent phases
- Candidate CSV includes band subspace indices

THEORY REFERENCE: docs/envelopeApproximationDerivation/5_FinalMultiBandTwoScaleEA.md

MULTI-BAND SUBSPACE:
For n_neighbor_bands=2 with target band n:
  - Tracked bands: [n-2, n-1, n, n+1, n+2] (total: 5 bands)
  - Extra bands for Born-Huang: +4 above and below tracked set

Output is compatible with blaze_phasesV3/phase1_blaze_v3.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import math
import os
import h5py
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.geometry import high_symmetry_points
from common.scoring import score_candidate
from common.io_utils import load_yaml, save_json
from common.plotting import plot_top_candidates_grid


def log(message):
    """Print message with flush."""
    print(message, flush=True)


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
    hs_indices: Dict[str, Dict[str, int]]


def load_band_library(library_path: Path, scan_id: str = "square_hex_eps_r_v1") -> BandLibrary:
    """Load the pre-computed band library from HDF5."""
    with h5py.File(library_path, 'r') as f:
        scan = f[f'scans/{scan_id}']
        
        lattice_types = tuple(scan['axes/lattice_type'].asstr()[...])
        polarizations = tuple(scan['axes/polarization'].asstr()[...])
        eps_bg = scan['axes/eps_bg'][...]
        r_over_a = scan['axes/r_over_a'][...]
        hole_eps = scan['axes/hole_eps'][...]
        
        k_paths = {}
        for lattice in lattice_types:
            k_paths[lattice] = scan[f'axes/k_path/{lattice}'][...]
        
        freq_data = scan['data/freq'][...]
        
        num_bands = freq_data.shape[5]
        num_kpoints = freq_data.shape[6]
    
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
    """Get band frequencies for a specific geometry from the library."""
    if lattice_type not in library.lattice_types:
        return None
    if polarization not in library.polarizations:
        return None
    
    lat_idx = library.lattice_types.index(lattice_type)
    pol_idx = library.polarizations.index(polarization)
    
    eps_idx = find_closest_index(library.eps_bg, eps_bg)
    r_idx = find_closest_index(library.r_over_a, r_over_a)
    hole_idx = find_closest_index(library.hole_eps, hole_eps)
    
    eps_tol = 0.15
    r_tol = 0.015
    
    if abs(library.eps_bg[eps_idx] - eps_bg) > eps_tol:
        return None
    if abs(library.r_over_a[r_idx] - r_over_a) > r_tol:
        return None
    
    freqs = library.freq_data[lat_idx, pol_idx, hole_idx, eps_idx, r_idx, :, :]
    
    if np.any(np.isnan(freqs)):
        return None
    
    return freqs


@dataclass
class MergedBands:
    """Result of merging TE and TM bands at each k-point."""
    frequencies: np.ndarray      # Shape: (n_merged_bands, n_k)
    polarizations: np.ndarray    # Shape: (n_merged_bands, n_k), 0=TE, 1=TM
    original_indices: np.ndarray # Shape: (n_merged_bands, n_k)
    
    @property
    def num_bands(self) -> int:
        return self.frequencies.shape[0]
    
    @property
    def num_kpoints(self) -> int:
        return self.frequencies.shape[1]


def get_merged_band_frequencies(
    library: BandLibrary,
    lattice_type: str,
    eps_bg: float,
    r_over_a: float,
    hole_eps: float = 1.0,
) -> Optional[MergedBands]:
    """Get MERGED TE+TM band frequencies for a specific geometry."""
    freqs_te = get_band_frequencies(library, lattice_type, 'TE', eps_bg, r_over_a, hole_eps)
    freqs_tm = get_band_frequencies(library, lattice_type, 'TM', eps_bg, r_over_a, hole_eps)
    
    if freqs_te is None and freqs_tm is None:
        return None
    
    if freqs_te is None:
        n_bands, n_k = freqs_tm.shape
        return MergedBands(
            frequencies=freqs_tm.copy(),
            polarizations=np.ones((n_bands, n_k), dtype=int),
            original_indices=np.tile(np.arange(n_bands)[:, None], (1, n_k)),
        )
    if freqs_tm is None:
        n_bands, n_k = freqs_te.shape
        return MergedBands(
            frequencies=freqs_te.copy(),
            polarizations=np.zeros((n_bands, n_k), dtype=int),
            original_indices=np.tile(np.arange(n_bands)[:, None], (1, n_k)),
        )
    
    n_te, n_k = freqs_te.shape
    n_tm = freqs_tm.shape[0]
    n_merged = n_te + n_tm
    
    merged_freqs = np.zeros((n_merged, n_k))
    merged_pols = np.zeros((n_merged, n_k), dtype=int)
    merged_orig_idx = np.zeros((n_merged, n_k), dtype=int)
    
    for k_idx in range(n_k):
        all_freqs = np.concatenate([freqs_te[:, k_idx], freqs_tm[:, k_idx]])
        all_pols = np.concatenate([np.zeros(n_te, dtype=int), np.ones(n_tm, dtype=int)])
        all_orig_idx = np.concatenate([np.arange(n_te), np.arange(n_tm)])
        
        sort_idx = np.argsort(all_freqs)
        merged_freqs[:, k_idx] = all_freqs[sort_idx]
        merged_pols[:, k_idx] = all_pols[sort_idx]
        merged_orig_idx[:, k_idx] = all_orig_idx[sort_idx]
    
    return MergedBands(
        frequencies=merged_freqs,
        polarizations=merged_pols,
        original_indices=merged_orig_idx,
    )


def get_band_polarization_label(merged: MergedBands, band_idx: int, k_idx: int) -> str:
    """Get polarization label ('TE' or 'TM') for a band at a specific k-point."""
    pol_code = merged.polarizations[band_idx, k_idx]
    return 'TE' if pol_code == 0 else 'TM'


def get_dominant_polarization(merged: MergedBands, band_idx: int) -> Tuple[str, float]:
    """Get the dominant polarization for a band across all k-points."""
    pol_codes = merged.polarizations[band_idx, :]
    te_fraction = np.mean(pol_codes == 0)
    if te_fraction >= 0.5:
        return ('TE', te_fraction)
    else:
        return ('TM', 1.0 - te_fraction)


def compute_band_subspace_indices(
    target_band: int,
    n_neighbor_bands: int,
    n_extra_bands: int,
    total_bands: int,
) -> Dict[str, Any]:
    """
    Compute the band indices for multi-band subspace.
    
    V3 multi-band structure:
    - Tracked bands: [target-n_neighbor, ..., target, ..., target+n_neighbor]
    - Extra bands: for Born-Huang potential calculation
    
    Args:
        target_band: Central band index (0-based)
        n_neighbor_bands: Number of bands above/below target to include in subspace
        n_extra_bands: Additional bands for Born-Huang (outside subspace)
        total_bands: Total number of bands available
    
    Returns:
        Dict with:
        - subspace_bands: list of band indices in tracked subspace
        - extra_bands_below: list of band indices below subspace
        - extra_bands_above: list of band indices above subspace
        - all_bands: complete list of bands needed for Phase 1
        - n_subspace: size of tracked subspace
    """
    # Subspace bands: [target-n, ..., target, ..., target+n]
    subspace_min = max(0, target_band - n_neighbor_bands)
    subspace_max = min(total_bands - 1, target_band + n_neighbor_bands)
    subspace_bands = list(range(subspace_min, subspace_max + 1))
    
    # Extra bands below subspace (for Born-Huang)
    extra_below_min = max(0, subspace_min - n_extra_bands)
    extra_bands_below = list(range(extra_below_min, subspace_min))
    
    # Extra bands above subspace (for Born-Huang)
    extra_above_max = min(total_bands - 1, subspace_max + n_extra_bands)
    extra_bands_above = list(range(subspace_max + 1, extra_above_max + 1))
    
    # All bands needed (for BLAZE)
    all_bands = extra_bands_below + subspace_bands + extra_bands_above
    
    return {
        'subspace_bands': subspace_bands,
        'extra_bands_below': extra_bands_below,
        'extra_bands_above': extra_bands_above,
        'all_bands': all_bands,
        'n_subspace': len(subspace_bands),
        'n_extra_below': len(extra_bands_below),
        'n_extra_above': len(extra_bands_above),
        'target_band': target_band,
        'target_index_in_subspace': subspace_bands.index(target_band) if target_band in subspace_bands else -1,
    }


def fit_local_dispersion_from_library(
    freqs: np.ndarray,
    k_path: np.ndarray,
    k_label: str,
    band_index: int,
    hs_indices: Dict[str, int],
) -> Dict[str, float]:
    """Extract local dispersion metrics for a band extremum from library data."""
    n_bands, n_k = freqs.shape
    
    if band_index >= n_bands:
        band_index = n_bands - 1
    
    if k_label not in hs_indices:
        k_idx = n_k // 2
    else:
        k_idx = hs_indices[k_label]
    
    wrap_path = np.allclose(k_path[0], k_path[-1])
    if wrap_path and k_idx == n_k - 1:
        k_idx = 0
    
    omega0 = float(freqs[band_index, k_idx])
    
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
    
    if dk_prev > 1e-9 and dk_next > 1e-9:
        term_next = (omega_next - omega0) / dk_next
        term_prev = (omega0 - omega_prev) / dk_prev
        d2omega_dk2 = 2.0 * (term_next - term_prev) / (dk_prev + dk_next)
    else:
        d2omega_dk2 = 0.0
    
    curvature_xx = abs(d2omega_dk2)
    curvature_yy = abs(d2omega_dk2)
    curvature_xy = 0.0
    curvature_trace = curvature_xx + curvature_yy
    curvature_det = curvature_xx * curvature_yy
    
    if curvature_trace > 1e-6:
        k_parab = 0.2 / np.sqrt(curvature_trace)
    else:
        k_parab = 0.5
    
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
        'k_parab_far': k_parab,
        'gap_above': gap_above,
        'gap_below': gap_below,
    }
    
    return metrics


def assemble_candidate_row_v3(
    candidate_id: int,
    lattice_type: str,
    polarization: str,
    r_over_a: float,
    eps_bg: float,
    band_index: int,
    k_label: str,
    k_vec: np.ndarray,
    metrics: Dict[str, float],
    band_subspace: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a V3 candidate row with multi-band subspace information."""
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
        'gap_above': metrics['gap_above'],
        'gap_below': metrics['gap_below'],
        'gap_min': min(metrics['gap_above'], metrics['gap_below']),
        # V3 multi-band information
        'n_subspace_bands': band_subspace['n_subspace'],
        'subspace_bands': str(band_subspace['subspace_bands']),
        'all_bands': str(band_subspace['all_bands']),
        'target_index_in_subspace': band_subspace['target_index_in_subspace'],
    }
    return row


def create_merged_bands_dict(
    merged: MergedBands,
    k_path_coords: np.ndarray,
    lattice_type: str,
) -> Dict[str, Any]:
    """Create a bands dictionary for merged TE+TM bands, compatible with plotting."""
    n_k = k_path_coords.shape[0]
    
    k_path = np.zeros(n_k)
    for i in range(1, n_k):
        k_path[i] = k_path[i-1] + np.linalg.norm(k_path_coords[i] - k_path_coords[i-1])
    
    if lattice_type == 'hex':
        k_labels = ['Γ', 'K', 'M', 'Γ']
    else:
        k_labels = ['Γ', 'X', 'M', 'Γ']
    
    k_break_indices = [0, 39, 78, n_k - 1]
    k_label_positions = np.array([k_path[idx] for idx in k_break_indices])
    
    return {
        'frequencies': merged.frequencies.T,
        'polarizations': merged.polarizations.T,
        'original_indices': merged.original_indices.T,
        'k_labels': k_labels,
        'k_path': k_path,
        'k_label_positions': k_label_positions,
        'k_break_indices': k_break_indices,
        'lattice_type': lattice_type,
        'polarization': 'merged',
        'num_bands': merged.num_bands,
        'k_interp': 39,
    }


def ensure_run_dir(config: Dict) -> Path:
    """Create a timestamped run directory for Phase 0 output."""
    output_base = Path(config.get('output_dir', 'runsV3'))
    run_name = config.get('run_name', 'blaze_v3')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_base / f"phase0_{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


def run_phase0_library_v3(config_path: str, max_bands: Optional[int] = None):
    """
    Run Phase 0 V3 using the pre-computed band library.
    
    Output is compatible with blaze_phasesV3/phase1_blaze_v3.py
    """
    log("=" * 70)
    log("Phase 0 (BLAZE): Candidate Search & Scoring — V3 Multi-Band Pipeline")
    log("=" * 70)
    
    config = load_yaml(config_path)
    log(f"\nLoaded configuration from: {config_path}")
    log(f"Run name: {config.get('run_name', 'blaze_v3')}")
    
    # V3 multi-band parameters
    n_neighbor_bands = config.get('n_neighbor_bands', 2)
    n_extra_bands = config.get('n_extra_bands', 4)
    log(f"\nV3 Multi-Band Settings:")
    log(f"  n_neighbor_bands: {n_neighbor_bands} (subspace size: {2*n_neighbor_bands + 1})")
    log(f"  n_extra_bands: {n_extra_bands} (for Born-Huang)")
    
    # Load band library
    library_path = Path(config.get('band_library_path', 
        '/home/renlephy/msl/research/band_diagram_scan/data/band_library.h5'))
    scan_id = config.get('band_library_scan_id', 'square_hex_eps_r_v1')
    
    log(f"\nLoading band library from: {library_path}")
    library = load_band_library(library_path, scan_id)
    
    log(f"\nLibrary contains:")
    log(f"  Lattice types: {list(library.lattice_types)}")
    log(f"  Polarizations: {list(library.polarizations)}")
    log(f"  ε_bg range: {library.eps_bg.min():.2f} - {library.eps_bg.max():.2f}")
    log(f"  r/a range: {library.r_over_a.min():.2f} - {library.r_over_a.max():.2f}")
    log(f"  Bands: {library.num_bands}")
    
    run_dir = ensure_run_dir(config)
    log(f"\nOutput directory: {run_dir}")
    
    # Save config for reproducibility
    save_json(config, run_dir / "phase0_config.json")
    
    # Extract search parameters
    lattice_types = config.get('lattice_types', list(library.lattice_types))
    
    use_library_params = config.get('use_library_parameters', True)
    
    if use_library_params:
        eps_step = config.get('eps_bg_step', 1)
        r_step = config.get('r_over_a_step', 1)
        
        eps_bg_list = library.eps_bg[::eps_step].tolist()
        r_over_a_list = library.r_over_a[::r_step].tolist()
        
        eps_min = config.get('eps_bg_min', 0)
        eps_max = config.get('eps_bg_max', 100)
        r_min = config.get('r_over_a_min', 0)
        r_max = config.get('r_over_a_max', 1)
        
        eps_bg_list = [e for e in eps_bg_list if eps_min <= e <= eps_max]
        r_over_a_list = [r for r in r_over_a_list if r_min <= r <= r_max]
    else:
        r_over_a_list = config.get('r_over_a_list', [0.2, 0.3, 0.4])
        eps_bg_list = config.get('eps_bg_list', [4.0, 6.0, 9.0])
    
    target_bands = config.get('target_bands', list(range(library.num_bands)))
    
    if max_bands is not None:
        target_bands = list(range(min(max_bands, library.num_bands)))
    
    # For merged bands
    n_merged = 2 * library.num_bands
    if max(target_bands) < library.num_bands:
        target_bands = list(range(min(n_merged, max(target_bands) + library.num_bands + 1)))
    
    log(f"\nSearch space:")
    log(f"  Lattice types: {lattice_types}")
    log(f"  Polarizations: MERGED (TE+TM)")
    log(f"  r/a values: {len(r_over_a_list)} points")
    log(f"  ε_bg values: {len(eps_bg_list)} points")
    log(f"  Target bands: {target_bands}")
    
    total_geometries = len(lattice_types) * len(r_over_a_list) * len(eps_bg_list)
    log(f"\nTotal geometries to evaluate: {total_geometries}")
    
    rows = []
    candidate_id = 0
    bands_cache = {}
    merged_cache = {}
    skipped_count = 0
    
    pbar = tqdm(total=total_geometries, desc="Processing geometries", unit="geom")
    
    for lattice_type in lattice_types:
        if lattice_type not in library.lattice_types:
            log(f"  Warning: {lattice_type} not in library, skipping")
            continue
        
        hs_points = high_symmetry_points(lattice_type)
        hs_indices = library.hs_indices.get(lattice_type, {})
        k_path = library.k_paths[lattice_type]
        
        for r_over_a in r_over_a_list:
            for eps_bg in eps_bg_list:
                pbar.update(1)
                
                merged = get_merged_band_frequencies(library, lattice_type, eps_bg, r_over_a)
                
                if merged is None:
                    skipped_count += 1
                    continue
                
                freqs = merged.frequencies
                
                cache_key = (lattice_type, 'merged', r_over_a, eps_bg)
                merged_cache[cache_key] = merged
                bands_cache[cache_key] = create_merged_bands_dict(
                    merged, library.k_paths[lattice_type], lattice_type
                )
                
                for k_label, k_vec in hs_points:
                    if k_label not in hs_indices and k_label != 'Γ':
                        continue
                    
                    k_idx = hs_indices.get(k_label, 0)
                    
                    for band_index in target_bands:
                        if band_index >= freqs.shape[0]:
                            continue
                        
                        # Compute band subspace indices for V3
                        band_subspace = compute_band_subspace_indices(
                            band_index, n_neighbor_bands, n_extra_bands, freqs.shape[0]
                        )
                        
                        metrics = fit_local_dispersion_from_library(
                            freqs, k_path, k_label, band_index, hs_indices
                        )
                        
                        dominant_pol, pol_fraction = get_dominant_polarization(merged, band_index)
                        local_pol = get_band_polarization_label(merged, band_index, k_idx)
                        
                        row = assemble_candidate_row_v3(
                            candidate_id,
                            lattice_type,
                            'merged',
                            r_over_a,
                            eps_bg,
                            band_index,
                            k_label,
                            k_vec[:2],
                            metrics,
                            band_subspace,
                        )
                        
                        row['dominant_polarization'] = dominant_pol
                        row['polarization_fraction'] = pol_fraction
                        row['local_polarization'] = local_pol
                        row['original_band_idx'] = int(merged.original_indices[band_index, k_idx])
                        
                        scores = score_candidate(row, config)
                        row.update(scores)
                        
                        rows.append(row)
                        candidate_id += 1
    
    pbar.close()
    
    if skipped_count > 0:
        log(f"\nSkipped {skipped_count} geometries")
    
    log(f"\nGenerated {len(rows)} candidates")
    
    df = pd.DataFrame(rows)
    df['candidate_source'] = 'library'
    df['pipeline_version'] = 'V3'
    df['merge_mode'] = 'TE+TM'
    df['n_neighbor_bands'] = n_neighbor_bands
    df['n_extra_bands'] = n_extra_bands
    df.sort_values('S_total', ascending=False, inplace=True)
    
    df['candidate_id'] = range(len(df))
    
    output_file = run_dir / 'phase0_candidates.csv'
    df.to_csv(output_file, index=False)
    log(f"\nSaved candidates to: {output_file}")
    
    K_display = config.get('K_candidates', 16)
    top_candidates = df.head(K_display)
    
    log(f"\nTop {K_display} candidates:")
    display_cols = ['candidate_id', 'lattice_type', 'dominant_polarization',
                    'r_over_a', 'eps_bg', 'k_label', 'band_index', 
                    'n_subspace_bands', 'S_total', 'valid_ea_flag']
    log(top_candidates[display_cols].to_string(index=False))
    
    # Plot band diagrams
    if bands_cache:
        log(f"\nGenerating band diagram plots...")
        
        bands_list = []
        for _, row in tqdm(top_candidates.iterrows(), total=len(top_candidates),
                          desc="Gathering band data", unit="candidate"):
            cache_key = (row['lattice_type'], 'merged', row['r_over_a'], row['eps_bg'])
            
            if cache_key in bands_cache:
                bands_list.append(bands_cache[cache_key])
            else:
                merged = get_merged_band_frequencies(
                    library, row['lattice_type'], row['eps_bg'], row['r_over_a']
                )
                if merged is not None:
                    bands = create_merged_bands_dict(
                        merged, library.k_paths[row['lattice_type']], row['lattice_type']
                    )
                    bands_list.append(bands)
                else:
                    bands_list.append({'frequencies': np.zeros((1, 1)), 'k_labels': [], 
                                      'k_path': np.array([0]), 'k_label_positions': np.array([0]),
                                      'k_break_indices': [0]})
        
        plot_path = run_dir / 'phase0_top_candidates_bands.png'
        plot_top_candidates_grid(top_candidates, bands_list, plot_path, n_cols=4)
        log(f"  Saved to: {plot_path}")
    
    log(f"\nStatistics:")
    log(f"  Valid EA candidates: {df['valid_ea_flag'].sum()} / {len(df)}")
    log(f"  Mean score: {df['S_total'].mean():.4f}")
    log(f"  Max score: {df['S_total'].max():.4f}")
    
    log("\n" + "=" * 70)
    log("Phase 0 (BLAZE) — V3 Multi-Band Pipeline complete!")
    log("=" * 70)
    log(f"\nNext step: Run Phase 1 with:")
    log(f"  python blaze_phasesV3/phase1_blaze_v3.py auto")
    
    return run_dir, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 0 (BLAZE V3): Multi-Band Candidate Search using band library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase0_library_v3.py                          # Use default config
  python phase0_library_v3.py config.yaml              # Use custom config
  python phase0_library_v3.py --max-bands 5            # Limit bands

This is part of the V3 multi-band pipeline.
See 5_FinalMultiBandTwoScaleEA.md for theory.
"""
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Path to configuration YAML (default: configsV3/phase0_blaze.yaml)"
    )
    parser.add_argument(
        "--max-bands", type=int, default=None, metavar="N",
        help="Only consider the first N bands per polarization"
    )
    
    args = parser.parse_args()
    
    default_config = Path(__file__).parent.parent / "configsV3" / "phase0_blaze.yaml"
    
    if args.config is None:
        if default_config.exists():
            log(f"Using default config: {default_config}")
            config_path = str(default_config)
        else:
            raise SystemExit(f"Default config not found: {default_config}")
    else:
        config_path = args.config
        if not Path(config_path).exists():
            raise SystemExit(f"Config not found: {config_path}")
    
    run_phase0_library_v3(config_path, max_bands=args.max_bands)
