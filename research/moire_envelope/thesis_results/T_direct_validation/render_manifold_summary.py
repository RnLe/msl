#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def _current_row(case: dict) -> dict:
    return next(item for item in case["results"] if item.get("is_current_subspace"))


def _extract_rows(case: dict, include_size1: bool) -> list[dict]:
    rows: list[dict] = []
    current = _current_row(case)
    rows.append({
        "k_label": case["k_label"],
        "kind": "current",
        "size": len(current["subset_bands"]),
        "subset_bands": current["subset_bands"],
        "health": current["health"],
        "min_gap": current["spectral"]["min_complement_gap"],
        "isolation_ratio": current["spectral"]["isolation_ratio"],
        "bh_max": None if current.get("born_huang") is None else current["born_huang"]["total_max"],
    })
    for size_str, item in sorted(case["best_by_size"].items(), key=lambda kv: int(kv[0])):
        size = int(size_str)
        if size == 1 and not include_size1:
            continue
        rows.append({
            "k_label": case["k_label"],
            "kind": f"best_s{size}",
            "size": size,
            "subset_bands": item["subset_bands"],
            "health": item["health"],
            "min_gap": item["spectral"]["min_complement_gap"],
            "isolation_ratio": item["spectral"]["isolation_ratio"],
            "bh_max": None if item.get("born_huang") is None else item["born_huang"]["total_max"],
        })
    return rows


def _write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["k_label", "kind", "size", "subset_bands", "health", "min_gap", "isolation_ratio", "bh_max"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "subset_bands": ",".join(str(x) for x in row["subset_bands"]),
            })


def _plot(rows: list[dict], output_path: Path, title: str) -> None:
    labels = [f"{row['k_label']}\n{row['kind']}\n{row['subset_bands']}" for row in rows]
    min_gap = np.array([max(row["min_gap"], 1e-12) for row in rows], dtype=float)
    isolation = np.array([max(row["isolation_ratio"] if row["isolation_ratio"] is not None else 1e-12, 1e-12) for row in rows], dtype=float)
    bh = np.array([max(row["bh_max"] if row["bh_max"] is not None else 1e-12, 1e-12) for row in rows], dtype=float)

    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True)

    axes[0].bar(x, min_gap, color="#295F98")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("min complement gap")
    axes[0].set_title(title)

    axes[1].bar(x, isolation, color="#4C956C")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("isolation ratio")

    axes[2].bar(x, bh, color="#D17B0F")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("BH max")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a compact CSV/PNG summary from manifold diagnostics JSON")
    parser.add_argument("--diagnostics-json", required=True, help="Path to manifold_diagnostics.json")
    parser.add_argument("--output-dir", default=None, help="Output directory for summary CSV/PNG")
    parser.add_argument("--include-size1", action="store_true", help="Include best size-1 manifolds in the summary")
    args = parser.parse_args()

    report = _load_report(Path(args.diagnostics_json))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.diagnostics_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for case in report["cases"]:
        rows.extend(_extract_rows(case, include_size1=args.include_size1))

    csv_path = output_dir / "manifold_summary_table.csv"
    png_path = output_dir / "manifold_summary_plot.png"
    _write_csv(rows, csv_path)
    _plot(rows, png_path, title=f"Manifold Summary: {report['scan_name']}")

    print(json.dumps({
        "csv": str(csv_path),
        "png": str(png_path),
        "rows": len(rows),
    }, indent=2))


if __name__ == "__main__":
    main()