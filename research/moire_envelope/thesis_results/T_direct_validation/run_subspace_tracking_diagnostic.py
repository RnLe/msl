#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MOIRE_ROOT = SCRIPT_DIR.parent.parent
if str(MOIRE_ROOT) not in sys.path:
    sys.path.insert(0, str(MOIRE_ROOT))
if str(MOIRE_ROOT / 'phasesV3') not in sys.path:
    sys.path.insert(0, str(MOIRE_ROOT / 'phasesV3'))

from phase2_mpb_v3 import apply_abelian_gauge_2d, apply_svqb_to_bloch_fields
from subspace_tracking import analyze_registry_subspace_tracking


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _parse_int_list(text: str | None, fallback: list[int]) -> list[int]:
    if not text:
        return fallback
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def _write_markdown(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Subspace Tracking Diagnostic",
        "",
        f"- Phase 1: `{report['phase1_path']}`",
        f"- Subspace bands: `{report['subspace_bands']}`",
        f"- Seed: `{report['diagnostic']['seed']}`",
        f"- Periodic BFS: `{report['diagnostic']['periodic']}`",
        "",
        "## Transport Edge Quality",
        f"- Min singular value: `{report['diagnostic']['transport_edge_min_singular_value']}`",
        f"- Mean singular value: `{report['diagnostic']['transport_edge_mean_singular_value']}`",
        "",
        "## Path Consistency",
        f"- Alternate-parent min singular value: `{report['diagnostic']['path_consistency_min_singular_value']}`",
        f"- Alternate-parent projector Frobenius distance: `{report['diagnostic']['path_consistency_projector_frobenius_distance']}`",
        "",
        "## Raw Indexed Subspace Fidelity",
        f"- Min singular value: `{report['diagnostic']['raw_subspace_fidelity_min_singular_value']}`",
        f"- Projector Frobenius distance: `{report['diagnostic']['raw_subspace_fidelity_projector_frobenius_distance']}`",
    ]
    (output_dir / 'subspace_tracking_diagnostic.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description='Run projector-aware subspace tracking diagnostics on a Phase 1 file')
    parser.add_argument('--phase1', required=True, help='Path to phase1_multiband_data.h5')
    parser.add_argument('--subspace-bands', default=None, help='Comma-separated band list; defaults to stored subspace_bands attr')
    parser.add_argument('--output-dir', default=None, help='Directory for JSON and markdown output')
    parser.add_argument('--seed-i', type=int, default=None, help='Optional seed i-index for BFS')
    parser.add_argument('--seed-j', type=int, default=None, help='Optional seed j-index for BFS')
    parser.add_argument('--non-periodic', action='store_true', help='Disable periodic wraparound in the BFS walk')
    args = parser.parse_args()

    phase1_path = Path(args.phase1)
    output_dir = Path(args.output_dir) if args.output_dir else phase1_path.parent / 'subspace_tracking_diagnostic'
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(phase1_path, 'r') as hf:
        bloch_fields = hf['bloch_fields'][:]
        epsilon = hf['epsilon'][:]
        stored_subspace = [int(x) for x in hf.attrs['subspace_bands'][:]]
        all_bands = [int(x) for x in hf.attrs['all_bands'][:]]

    bloch_fields = np.array(bloch_fields, copy=True)
    bloch_fields, gauge_diag = apply_abelian_gauge_2d(bloch_fields)
    bloch_fields, svqb_stats = apply_svqb_to_bloch_fields(bloch_fields, epsilon)

    subspace_bands = _parse_int_list(args.subspace_bands, stored_subspace)
    subspace_indices = [all_bands.index(band) for band in subspace_bands]
    seed = None
    if args.seed_i is not None and args.seed_j is not None:
        seed = (args.seed_i, args.seed_j)

    diagnostic = analyze_registry_subspace_tracking(
        bloch_fields,
        epsilon,
        subspace_indices,
        seed=seed,
        periodic=not args.non_periodic,
    )
    report = {
        'phase1_path': str(phase1_path),
        'subspace_bands': subspace_bands,
        'gauge_diagnostics': _to_jsonable(gauge_diag),
        'svqb_stats': _to_jsonable(svqb_stats),
        'diagnostic': _to_jsonable(diagnostic),
    }
    (output_dir / 'subspace_tracking_diagnostic.json').write_text(json.dumps(report, indent=2))
    _write_markdown(output_dir, report)
    print(json.dumps({
        'phase1_path': str(phase1_path),
        'subspace_bands': subspace_bands,
        'output_dir': str(output_dir),
    }, indent=2))


if __name__ == '__main__':
    main()