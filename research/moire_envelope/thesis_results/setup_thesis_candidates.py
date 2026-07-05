"""
Setup Thesis Candidates — Create Phase 0 run directories for thesis pipeline.

Creates a Phase 0-compatible run directory for each thesis candidate with:
  - phase0_candidates.csv (pipeline-compatible format)
  - candidate_XXXX/phase0_meta.json (per-candidate metadata)

Usage:
    python thesis_results/setup_thesis_candidates.py [--output-dir runsV3]
"""

import json
import math
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_candidates(yaml_path: Path) -> dict:
    """Load candidate definitions from candidates.yaml."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


def create_phase0_csv_row(name: str, cand: dict, candidate_id: int) -> dict:
    """Create a row for phase0_candidates.csv from candidate spec."""
    row = {
        'candidate_id': candidate_id,
        'lattice_type': cand['lattice_type'],
        'polarization': cand.get('polarization', 'merged'),
        'a': 1.0,
        'r_over_a': cand['r_over_a'],
        'eps_bg': cand['eps_bg'],
        'band_index': cand['band_index'],
        'k_label': cand['k_label'],
        'k0_x': cand['k0_x'],
        'k0_y': cand['k0_y'],
        'omega0': cand['omega0'],
        # Curvature from Phase 0 (1D) — not used, Tier 2 is authoritative
        'curvature_xx': 0.0,
        'curvature_xy': 0.0,
        'curvature_yy': 0.0,
        'curvature_trace': 0.0,
        'curvature_det': 0.0,
        'vg_x': 0.0, 'vg_y': 0.0, 'vg_norm': 0.0,
        'k_parab': 0.0, 'k_parab_far': 0.0,
        'gap_above': cand.get('gap_min', 0.01),
        'gap_below': cand.get('gap_min', 0.01),
        'gap_min': cand.get('gap_min', 0.01),
        'n_subspace_bands': cand['n_subspace_bands'],
        'subspace_bands': str(cand['subspace_bands']),
        'all_bands': str(cand['all_bands']),
        'target_index_in_subspace': cand['target_index_in_subspace'],
        'dominant_polarization': cand.get('dominant_polarization', 'TE'),
        'polarization_fraction': 1.0,
        'local_polarization': cand.get('local_polarization', 'TE'),
        'original_band_idx': cand['band_index'],
        # Scoring fields (not used for thesis; keep for compatibility)
        'S_flat': 1.0, 'S_gap': 1.0, 'S_parab': 1.0,
        'S_vg': 1.0, 'S_linear': 1.0, 'S_sym': 1.0, 'S_total': 6.0,
        'valid_ea_flag': True,
        'candidate_source': 'thesis_selection',
        'pipeline_version': 'V3',
        'merge_mode': 'TE+TM',
        'n_neighbor_bands': cand.get('n_neighbor_bands', 2),
        'n_extra_bands': cand.get('n_extra_bands', 4),
    }
    # Propagate eps_hole for rod-based lattices (e.g. honeycomb)
    if 'eps_hole' in cand:
        row['eps_hole'] = cand['eps_hole']
    return row


def create_phase0_meta(name: str, cand: dict, candidate_id: int,
                       theta_deg: float = 1.1) -> dict:
    """Create phase0_meta.json content for a candidate."""
    theta_rad = math.radians(theta_deg)
    a = 1.0

    # Moiré length ≈ a / (2 sin(θ/2)) ≈ a / θ for small θ
    moire_length = a / (2 * math.sin(theta_rad / 2))
    eta = theta_rad  # Small angle: η ≈ θ

    meta = {
        'candidate_id': candidate_id,
        'lattice_type': cand['lattice_type'],
        'a': a,
        'r_over_a': cand['r_over_a'],
        'eps_bg': cand['eps_bg'],
        'band_index': cand['band_index'],
        'merged_band_index': cand['band_index'],
        'k_label': cand['k_label'],
        'k0_x': cand['k0_x'],
        'k0_y': cand['k0_y'],
        'omega0': cand['omega0'],
        'polarization': cand.get('polarization', 'merged'),
        'dominant_polarization': cand.get('dominant_polarization', 'TE'),
        'local_polarization': cand.get('local_polarization', 'TE'),
        'n_subspace_bands': cand['n_subspace_bands'],
        'subspace_bands': cand['subspace_bands'],
        'all_bands': cand['all_bands'],
        'target_index_in_subspace': cand['target_index_in_subspace'],
        'theta_deg': theta_deg,
        'theta_rad': theta_rad,
        'moire_length': moire_length,
        'eta': eta,
    }
    # Propagate eps_hole for rod-based lattices (e.g. honeycomb)
    if 'eps_hole' in cand:
        meta['eps_hole'] = cand['eps_hole']
    return meta


def setup_candidate_run(name: str, cand: dict, candidate_id: int,
                        output_base: Path, timestamp: str) -> Path:
    """Create a Phase 0-style run directory for one thesis candidate."""

    run_name = f"thesis_{name}_{timestamp}"
    run_dir = output_base / run_name
    cand_dir = run_dir / f"candidate_{candidate_id:04d}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # Write phase0_candidates.csv
    row = create_phase0_csv_row(name, cand, candidate_id)
    df = pd.DataFrame([row])
    csv_path = run_dir / "phase0_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Wrote {csv_path}")

    # Write phase0_meta.json
    meta = create_phase0_meta(name, cand, candidate_id)
    meta_path = cand_dir / "phase0_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {meta_path}")

    # Write phase0_config.json (minimal)
    config = {
        'run_name': run_name,
        'candidate_source': 'thesis_selection',
        'pipeline_version': 'V3',
        'thesis_candidate': name,
        'thesis_label': cand.get('label', name),
        'created_at': timestamp,
    }
    config_path = run_dir / "phase0_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {config_path}")

    return run_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup thesis candidate run directories")
    parser.add_argument('--output-dir', default='runsV3',
                        help='Base output directory (default: runsV3)')
    parser.add_argument('--candidates-yaml', default=None,
                        help='Path to candidates.yaml')
    args = parser.parse_args()

    # Find candidates.yaml
    if args.candidates_yaml:
        yaml_path = Path(args.candidates_yaml)
    else:
        yaml_path = Path(__file__).parent / "candidates.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"candidates.yaml not found: {yaml_path}")

    data = load_candidates(yaml_path)
    candidates = data['candidates']

    output_base = PROJECT_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Setting up {len(candidates)} thesis candidates")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}")
    print()

    run_dirs = {}
    for i, (name, cand) in enumerate(candidates.items()):
        candidate_id = 0  # Always 0: each thesis run has exactly one candidate
        print(f"[{i+1}/{len(candidates)}] {name} ({cand.get('label', '?')})")
        print(f"  {cand['lattice_type']}/{cand['local_polarization']}, "
              f"r/a={cand['r_over_a']}, ε={cand['eps_bg']}, "
              f"band={cand['band_index']}, k={cand['k_label']}")

        run_dir = setup_candidate_run(name, cand, candidate_id,
                                      output_base, timestamp)
        run_dirs[name] = run_dir
        print()

    # Write summary
    print("=" * 60)
    print("THESIS CANDIDATE SETUP COMPLETE")
    print("=" * 60)
    for name, rd in run_dirs.items():
        print(f"  {name}: {rd}")
    print()
    print("Next steps:")
    print("  1. Run Phase 1 for each candidate (see run_pipeline.sh)")
    print("  2. Or use the individual config files in configsV3/")

    # Save run directory mapping
    mapping_path = output_base / f"thesis_runs_{timestamp}.json"
    with open(mapping_path, 'w') as f:
        json.dump({k: str(v) for k, v in run_dirs.items()}, f, indent=2)
    print(f"\nRun directory mapping: {mapping_path}")


if __name__ == "__main__":
    main()
