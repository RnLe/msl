#!/usr/bin/env python3
"""One-off helper to retrofit Phase 0 optimizer outputs into legacy runs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

# Ensure we can import the phase modules without installation
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry import build_lattice  # type: ignore  # noqa: E402
from common.io_utils import load_yaml  # type: ignore  # noqa: E402
from common.mpb_utils import compute_bandstructure  # type: ignore  # noqa: E402
from common.plotting import plot_optimizer_candidate_summary  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge optimizer results into an existing Phase 0 run without re-running the sweep."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the existing phase0_real_run_* directory.",
    )
    return parser.parse_args()


def stream_max_candidate_id(csv_path: Path) -> int:
    max_id = -1
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "candidate_id" not in reader.fieldnames:
            raise ValueError("candidate_id column missing in phase0_candidates.csv")
        for row in reader:
            try:
                cid = int(row["candidate_id"])
            except (TypeError, ValueError):
                continue
            if cid > max_id:
                max_id = cid
    return max_id


OptimizerEntry = Dict[str, Any]


def load_optimizer_rows(opt_csv: Path, start_id: int) -> Tuple[List[OptimizerEntry], int]:
    if not opt_csv.exists():
        return [], start_id

    df = pd.read_csv(opt_csv)
    if df.empty:
        return [], start_id

    current_id = start_id
    rows: List[OptimizerEntry] = []
    for _, series in df.iterrows():
        row = series.to_dict()
        row["candidate_id"] = current_id
        if not row.get("optimization_k_label"):
            row["optimization_k_label"] = row.get("k_label", "")
        row.setdefault("optimization_strategy", "differential_evolution")
        row.setdefault("optimization_lattice", row.get("lattice_type", ""))
        row["candidate_source"] = "optimizer"
        rows.append({"data": row, "score": float(row["S_total"])})
        current_id += 1

    rows.sort(key=lambda entry: entry["score"], reverse=True)
    return rows, current_id


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        return
    shutil.copy2(path, backup)


def merge_candidates(
    grid_csv: Path,
    optimizer_rows: Sequence[OptimizerEntry],
    output_csv: Path,
) -> None:
    if not grid_csv.exists():
        raise FileNotFoundError(grid_csv)

    with grid_csv.open(newline="") as grid_handle:
        reader = csv.DictReader(grid_handle)
        grid_fields = list(reader.fieldnames or [])

        opt_fields: List[str] = []
        if optimizer_rows:
            opt_fields = list(optimizer_rows[0]["data"].keys())

        fieldnames: List[str] = []
        seen = set()
        for name in grid_fields + opt_fields:
            if name and name not in seen:
                fieldnames.append(name)
                seen.add(name)
        if "candidate_source" not in seen:
            fieldnames.append("candidate_source")
            seen.add("candidate_source")

        with output_csv.open("w", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()

            opt_idx = 0
            current_opt = optimizer_rows[opt_idx]["data"] if optimizer_rows else None
            current_opt_score = (
                float(current_opt["S_total"]) if current_opt is not None else None
            )

            for row in reader:
                grid_score = float(row["S_total"])
                while current_opt is not None and current_opt_score is not None and current_opt_score >= grid_score:
                    writer.writerow({name: current_opt.get(name, "") for name in fieldnames})
                    opt_idx += 1
                    if opt_idx >= len(optimizer_rows):
                        current_opt = None
                        current_opt_score = None
                    else:
                        current_opt = optimizer_rows[opt_idx]["data"]
                        current_opt_score = float(current_opt["S_total"])

                row["candidate_source"] = row.get("candidate_source", "grid")
                writer.writerow({name: row.get(name, "") for name in fieldnames})

            while current_opt is not None:
                writer.writerow({name: current_opt.get(name, "") for name in fieldnames})
                opt_idx += 1
                if opt_idx >= len(optimizer_rows):
                    current_opt = None
                    current_opt_score = None
                else:
                    current_opt = optimizer_rows[opt_idx]["data"]
                    current_opt_score = float(current_opt["S_total"])


def regenerate_optimizer_csv(opt_csv: Path, optimizer_rows: Sequence[OptimizerEntry]) -> None:
    if not optimizer_rows:
        return
    df = pd.DataFrame([entry["data"] for entry in optimizer_rows])
    df.to_csv(opt_csv, index=False)


def generate_optimizer_plots(
    optimizer_rows: Sequence[OptimizerEntry],
    config: dict,
    run_dir: Path,
) -> None:
    if not optimizer_rows:
        return

    num_bands = int(config.get("num_bands", 8))
    bands_cache: Dict[Tuple[str, float, float], dict] = {}

    for entry in optimizer_rows:
        row = entry["data"]
        lattice = row["lattice_type"]
        r_over_a = float(row["r_over_a"])
        eps_bg = float(row["eps_bg"])
        a_value = float(row.get("a", 1.0))
        key = (lattice, r_over_a, eps_bg)

        geom = build_lattice(lattice, r_over_a, eps_bg, a=a_value)
        if key in bands_cache:
            bands = bands_cache[key]
        else:
            bands = compute_bandstructure(geom, config, num_bands=num_bands)
            bands["k_interp"] = config.get("k_interp", 19)
            bands_cache[key] = bands

        plot_path = run_dir / f"phase0_optimizer_{lattice}_{int(row['candidate_id']):04d}.png"
        plot_optimizer_candidate_summary(geom, bands, row, plot_path)
        print(f"  Wrote {plot_path.name}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    candidate_csv = run_dir / "phase0_candidates.csv"
    optimizer_csv = run_dir / "phase0_optimizer_results.csv"
    config_path = run_dir / "config.yaml"

    print(f"Retrofit target: {run_dir}")
    max_id = stream_max_candidate_id(candidate_csv)
    print(f"  Max existing candidate_id: {max_id}")

    optimizer_entries, next_id = load_optimizer_rows(optimizer_csv, max_id + 1)
    if not optimizer_entries:
        print("No optimizer rows found; nothing to merge.")
        return

    print(f"  Assigning candidate_ids {max_id + 1}..{next_id - 1} to optimizer picks")

    tmp_csv = candidate_csv.with_suffix(".merged.tmp")
    merge_candidates(candidate_csv, optimizer_entries, tmp_csv)
    backup_file(candidate_csv)
    tmp_csv.replace(candidate_csv)
    print(f"  Updated {candidate_csv.name} with merged candidates + source column")

    regenerate_optimizer_csv(optimizer_csv, optimizer_entries)
    print(f"  Rewrote {optimizer_csv.name} with new candidate_ids")

    config = load_yaml(config_path)
    generate_optimizer_plots(optimizer_entries, config, run_dir)

    print("Done. Phase 0 run now mirrors the latest pipeline expectations.")


if __name__ == "__main__":
    main()
