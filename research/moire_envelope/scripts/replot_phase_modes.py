#!/usr/bin/env python3
"""Regenerate Phase 3/4 plots from saved HDF5 datasets."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, cast

import h5py
import numpy as np

from common.io_utils import candidate_dir, load_json, load_yaml
from common.plotting import (
    plot_envelope_modes,
    plot_phase4_mode_differences,
    plot_phase4_mode_profiles,
)
from phases.phase2_ea_operator import resolve_run_dir as resolve_phase_run_dir


def _discover_candidates(run_dir: Path) -> List[int]:
    candidates: List[int] = []
    for path in sorted(run_dir.glob("candidate_*")):
        try:
            candidates.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return candidates


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: Iterable[Dict] | None,
) -> Dict:
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception:
            pass
    if candidate_frame is not None:
        for row in candidate_frame:
            try:
                if int(row.get("candidate_id", -1)) == candidate_id:
                    row = dict(row)
                    row["candidate_id"] = candidate_id
                    return row
            except (TypeError, ValueError):
                continue
    return {"candidate_id": candidate_id}


def _load_candidate_frame(run_dir: Path) -> List[Dict] | None:
    csv_path = run_dir / "phase0_candidates.csv"
    if not csv_path.exists():
        return None
    rows: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _resolve_run_dir(run_dir: str | Path, config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    return resolve_phase_run_dir(run_dir, config)


def _probability(field: np.ndarray) -> np.ndarray:
    prob = np.abs(field) ** 2
    total = float(prob.sum())
    if total > 0:
        prob /= total
    return prob


def regenerate_phase3(run_dir: Path, candidate_ids: Sequence[int], n_modes: int):
    frame = _load_candidate_frame(run_dir)
    for cid in candidate_ids:
        cdir = candidate_dir(run_dir, cid)
        h5_path = cdir / "phase3_eigenstates.h5"
        if not h5_path.exists():
            print(f"[phase3] Skipping candidate {cid}: {h5_path.name} missing")
            continue
        with h5py.File(h5_path, "r") as hf:
            fields = np.asarray(cast(h5py.Dataset, hf["F"])[:])
            R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
            eigenvalues = np.asarray(cast(h5py.Dataset, hf["eigenvalues"])[:])
        row = _load_candidate_metadata(run_dir, cid, frame)
        plot_envelope_modes(cdir, R_grid, fields, eigenvalues, n_modes=n_modes, candidate_params=row)
        print(f"[phase3] Replotted candidate {cid:04d}")


def regenerate_phase4(run_dir: Path, candidate_ids: Sequence[int], n_modes: int):
    frame = _load_candidate_frame(run_dir)
    for cid in candidate_ids:
        cdir = candidate_dir(run_dir, cid)
        h5_path = cdir / "phase4_gamma_modes.h5"
        if not h5_path.exists():
            print(f"[phase4] Skipping candidate {cid}: {h5_path.name} missing")
            continue
        with h5py.File(h5_path, "r") as hf:
            fields = np.asarray(cast(h5py.Dataset, hf["F_gamma"])[:])
            R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
            eigenvalues = np.asarray(cast(h5py.Dataset, hf["eigenvalues"])[:])
        row = _load_candidate_metadata(run_dir, cid, frame)
        plot_phase4_mode_profiles(cdir, R_grid, fields, eigenvalues, n_modes=n_modes, candidate_params=row)

        # Optional difference plot if Phase 3 data exists.
        phase3_h5 = cdir / "phase3_eigenstates.h5"
        if phase3_h5.exists():
            with h5py.File(phase3_h5, "r") as hf3:
                phase3_fields = np.asarray(cast(h5py.Dataset, hf3["F"])[:])
                R_grid_p3 = np.asarray(cast(h5py.Dataset, hf3["R_grid"])[:])
            if np.allclose(R_grid, R_grid_p3):
                n_diff = min(n_modes, phase3_fields.shape[0], fields.shape[0])
                diff_fields = []
                for mode_idx in range(n_diff):
                    diff = _probability(fields[mode_idx]) - _probability(phase3_fields[mode_idx])
                    diff_fields.append(diff)
                plot_phase4_mode_differences(
                    cdir,
                    R_grid,
                    np.asarray(diff_fields),
                    eigenvalues,
                    n_modes=n_diff,
                    candidate_params=row,
                )
            else:
                print(f"[phase4] Candidate {cid:04d}: grid mismatch, skipping difference plot")
        print(f"[phase4] Replotted candidate {cid:04d}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate phase plots from saved HDF5 data")
    parser.add_argument("phase", choices=["phase3", "phase4", "both"], help="Which plots to regenerate")
    parser.add_argument("run_dir", help="Phase 0 run directory or 'auto'")
    parser.add_argument("config", help="Config file used to resolve run_dir")
    parser.add_argument("--n-modes", type=int, default=8, dest="n_modes")
    parser.add_argument(
        "--candidates",
        nargs="*",
        type=int,
        help="Optional list of candidate IDs to restrict",
    )
    args = parser.parse_args()

    resolved = _resolve_run_dir(args.run_dir, args.config)
    all_candidates = _discover_candidates(resolved)
    if not all_candidates:
        raise SystemExit(f"No candidate_* directories found in {resolved}")

    if args.candidates:
        candidate_ids = [cid for cid in args.candidates if cid in all_candidates]
        missing = set(args.candidates) - set(candidate_ids)
        if missing:
            print(f"WARNING: Unknown candidates skipped: {sorted(missing)}")
    else:
        candidate_ids = all_candidates

    if args.phase in {"phase3", "both"}:
        regenerate_phase3(resolved, candidate_ids, args.n_modes)
    if args.phase in {"phase4", "both"}:
        regenerate_phase4(resolved, candidate_ids, args.n_modes)


if __name__ == "__main__":
    main()
