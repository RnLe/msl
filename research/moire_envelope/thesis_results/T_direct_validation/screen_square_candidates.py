#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def classify_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hessian_class"] = "other"
    out.loc[out["eigval_min"] > 0, "hessian_class"] = "minimum"
    out.loc[out["eigval_max"] < 0, "hessian_class"] = "maximum"
    out.loc[out["eigval_min"] * out["eigval_max"] < 0, "hessian_class"] = "saddle"
    return out


def shortlist_square_candidates(df: pd.DataFrame) -> pd.DataFrame:
    square = df[df["lattice_type"] == "square"].copy()
    minima = square[square["hessian_class"] == "minimum"].copy()
    minima = minima[minima["theta_star_deg"].between(0.5, 10.0)]
    minima = minima[minima["cond_number"] < 20.0]

    minima["score"] = (
        2.0 * minima["cond_number"]
        + 1.5 * (minima["theta_star_deg"] - 2.5).abs()
        + 3.0 * (minima["VEkin_2deg"] - 3.0).abs()
        + 4.0 * minima["omega0_phase0"]
    )
    return minima.sort_values(["score", "theta_star_deg", "omega0_phase0"]).reset_index(drop=True)


def run_screen(csv_path: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    df = classify_candidates(df)
    shortlist = shortlist_square_candidates(df)

    report = {
        "csv_path": str(csv_path),
        "n_total": int(len(df)),
        "n_square": int((df["lattice_type"] == "square").sum()),
        "k_labels_present": sorted(df.loc[df["lattice_type"] == "square", "k_label"].unique().tolist()),
        "non_high_symmetry_present": bool(
            any(label not in {"Γ", "X", "M", "Gamma"} for label in df.loc[df["lattice_type"] == "square", "k_label"].unique())
        ),
        "top_candidates": shortlist[
            [
                "family",
                "k_label",
                "band_index",
                "omega0_phase0",
                "eigval_min",
                "eigval_max",
                "cond_number",
                "theta_star_deg",
                "V_depth",
                "VEkin_2deg",
                "score",
            ]
        ].head(10).to_dict(orient="records"),
        "all_square_candidates": df[
            [
                "family",
                "k_label",
                "band_index",
                "omega0_phase0",
                "hessian_class",
                "cond_number",
                "theta_star_deg",
                "V_depth",
                "VEkin_2deg",
            ]
        ].sort_values(["hessian_class", "theta_star_deg"]).to_dict(orient="records"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "square_candidate_screen.json"
    md_path = output_dir / "square_candidate_screen.md"
    json_path.write_text(json.dumps(_to_serializable(report), indent=2))

    lines = [
        "# Square Candidate Screen",
        "",
        f"Source CSV: `{csv_path}`",
        f"Square k-labels present: `{report['k_labels_present']}`",
        f"Non-high-symmetry points present: `{report['non_high_symmetry_present']}`",
        "",
        "## Top Candidates",
    ]
    for item in report["top_candidates"]:
        lines.append(
            f"- `{item['family']}`: k=`{item['k_label']}`, band={item['band_index']}, "
            f"omega0={item['omega0_phase0']:.6f}, cond={item['cond_number']:.3f}, "
            f"theta*={item['theta_star_deg']:.3f}, V/E@2°={item['VEkin_2deg']:.3f}, score={item['score']:.3f}"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen square candidates from Tier 2 ranked results")
    parser.add_argument("--csv", required=True, help="Path to a tier2_ranked.csv file")
    parser.add_argument("--output-dir", required=True, help="Directory for square_candidate_screen.{json,md}")
    args = parser.parse_args()

    report = run_screen(Path(args.csv), Path(args.output_dir))
    print(json.dumps({
        "top_candidates": report["top_candidates"][:3],
        "k_labels_present": report["k_labels_present"],
        "non_high_symmetry_present": report["non_high_symmetry_present"],
    }, indent=2))


if __name__ == "__main__":
    main()