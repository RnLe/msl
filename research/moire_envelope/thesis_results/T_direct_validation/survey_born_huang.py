#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasesV3.bloch_fields import compute_born_huang_from_fields, diagnose_born_huang_values
from phasesV3.phase2_mpb_v3 import apply_abelian_gauge_2d, apply_svqb_to_bloch_fields


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _physical_kinetic_proxy(
    m_inv_registry: np.ndarray,
    subset_bands: list[int],
    available_m_inv_bands: list[int],
    length_scale: float,
) -> float | None:
    traces = []
    for band in subset_bands:
        if band not in available_m_inv_bands:
            return None
        idx = available_m_inv_bands.index(band)
        diag = m_inv_registry[:, :, idx, :, :]
        traces.append(np.abs(np.trace(diag, axis1=2, axis2=3)))
    if not traces:
        return None
    mean_trace = float(np.mean(np.stack(traces, axis=0)))
    return 0.5 * mean_trace / max(length_scale**2, 1e-15)


def _lambda_proxy(omega_registry: np.ndarray, omega_ref: float, subset_indices: list[int]) -> float:
    widths = []
    for idx in subset_indices:
        band = omega_registry[:, :, idx] - omega_ref
        widths.append(float(np.max(band) - np.min(band)))
    return float(np.mean(widths)) if widths else 0.0


def _bh_metrics(phi_bh: np.ndarray, subset_bands: list[int], kinetic_proxy: float | None, lambda_proxy: float) -> dict[str, Any]:
    basic = diagnose_born_huang_values(phi_bh)
    n_sub = phi_bh.shape[2]
    offdiag = phi_bh.copy()
    for idx in range(n_sub):
        offdiag[:, :, idx, idx] = 0.0
    offdiag_fro = np.linalg.norm(offdiag.reshape(-1, n_sub, n_sub), axis=(1, 2))
    herm = phi_bh - np.swapaxes(phi_bh, 2, 3)
    diag_means = [float(np.mean(phi_bh[:, :, idx, idx])) for idx in range(n_sub)]
    diag_max = [float(np.max(phi_bh[:, :, idx, idx])) for idx in range(n_sub)]
    diag_min = [float(np.min(phi_bh[:, :, idx, idx])) for idx in range(n_sub)]
    total_max = float(np.max(np.abs(phi_bh)))
    offdiag_max = float(np.max(np.abs(offdiag)))
    offdiag_mean = float(np.mean(offdiag_fro))
    trace_mean = float(np.mean(np.trace(phi_bh, axis1=2, axis2=3)))
    bh_score = total_max + offdiag_mean
    return {
        "subset_bands": subset_bands,
        "diag_means": diag_means,
        "diag_min": diag_min,
        "diag_max": diag_max,
        "offdiag_max": offdiag_max,
        "offdiag_fro_mean": offdiag_mean,
        "trace_mean": trace_mean,
        "total_max": total_max,
        "hermiticity_max_abs": float(np.max(np.abs(herm))) if herm.size else 0.0,
        "kinetic_proxy": kinetic_proxy,
        "lambda_proxy": lambda_proxy,
        "bh_to_kinetic_ratio": None if kinetic_proxy is None else total_max / max(kinetic_proxy, 1e-15),
        "bh_to_lambda_ratio": total_max / max(lambda_proxy, 1e-15),
        "bh_score": bh_score,
        "basic": _to_serializable(basic),
    }


def _subset_band_lists(all_bands: list[int], mode: str, sizes: list[int]) -> list[list[int]]:
    subsets: list[list[int]] = []
    if mode == "contiguous":
        for size in sizes:
            if size <= 0 or size > len(all_bands):
                continue
            for start in range(0, len(all_bands) - size + 1):
                subsets.append(all_bands[start:start + size])
    elif mode == "combinations":
        for size in sizes:
            if size <= 0 or size > len(all_bands):
                continue
            subsets.extend([list(combo) for combo in combinations(all_bands, size)])
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for subset in subsets:
        key = tuple(subset)
        if key not in seen:
            seen.add(key)
            unique.append(subset)
    return unique


def run_survey(phase1_path: Path, output_dir: Path, mode: str, sizes: list[int], include_current_subspace: bool = True) -> dict[str, Any]:
    with h5py.File(phase1_path, "r") as hf:
        bloch_fields = hf["bloch_fields"][:]
        epsilon = hf["epsilon"][:]
        omega_registry = hf["stencil"]["registry_omega_all"][:]
        m_inv_grid = hf["M_inv"][:]
        omega_ref = float(hf.attrs["omega_ref"])
        theta_rad = float(hf.attrs["theta_rad"])
        moire_length = float(hf.attrs["moire_length"])
        all_bands = [int(band) for band in hf.attrs["all_bands"][:]]
        current_subspace = [int(band) for band in hf.attrs["subspace_bands"][:]]
        available_m_inv_bands = current_subspace[:]
        n_registry = int(hf["stencil"].attrs["n_registry"])

    bloch_fields, gauge_diag = apply_abelian_gauge_2d(bloch_fields)
    bloch_fields, svqb_stats = apply_svqb_to_bloch_fields(bloch_fields, epsilon)
    dR = (moire_length / n_registry, moire_length / n_registry)
    subsets = _subset_band_lists(all_bands, mode, sizes)
    if include_current_subspace and current_subspace not in subsets:
        subsets.append(current_subspace)

    results = []
    for subset_bands in subsets:
        subset_indices = [all_bands.index(band) for band in subset_bands]
        extra_indices = [idx for idx, band in enumerate(all_bands) if band not in subset_bands]
        phi_bh = compute_born_huang_from_fields(
            bloch_fields=bloch_fields,
            dR=dR,
            subspace_indices=subset_indices,
            extra_indices=extra_indices,
            epsilon=epsilon,
        )
        kinetic_proxy = _physical_kinetic_proxy(m_inv_grid, subset_bands, available_m_inv_bands, moire_length)
        lambda_proxy = _lambda_proxy(omega_registry, omega_ref, subset_indices)
        metrics = _bh_metrics(phi_bh, subset_bands, kinetic_proxy, lambda_proxy)
        metrics["subset_size"] = len(subset_bands)
        metrics["contains_current_target_band"] = 3 in subset_bands
        metrics["is_current_subspace"] = subset_bands == current_subspace
        results.append(metrics)

    results.sort(key=lambda item: (item["subset_size"], item["bh_score"]))
    current_result = next((item for item in results if item["subset_bands"] == current_subspace), None)
    best_by_size: dict[int, dict[str, Any]] = {}
    for item in results:
        size = int(item["subset_size"])
        best_by_size.setdefault(size, item)

    report = {
        "phase1_path": str(phase1_path),
        "mode": mode,
        "sizes": sizes,
        "theta_deg": math.degrees(theta_rad),
        "theta_rad": theta_rad,
        "moire_length": moire_length,
        "n_registry": n_registry,
        "all_bands": all_bands,
        "current_subspace": current_subspace,
        "dR_registry": dR[0],
        "gauge_diagnostics": _to_serializable(gauge_diag),
        "svqb_stats": _to_serializable(svqb_stats),
        "current_subspace_metrics": _to_serializable(current_result),
        "best_by_size": _to_serializable(best_by_size),
        "results": _to_serializable(results),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "born_huang_survey.json"
    md_path = output_dir / "born_huang_survey.md"
    json_path.write_text(json.dumps(report, indent=2))

    lines = [
        f"# Born-Huang Survey: {phase1_path.parent.parent.parent.name}",
        "",
        f"Mode: `{mode}`",
        f"Sizes: `{sizes}`",
        f"Current subspace: `{current_subspace}`",
        "",
    ]
    if current_result is not None:
        lines.extend([
            "## Current Subspace",
            f"- Bands: `{current_result['subset_bands']}`",
            f"- Total BH max: `{current_result['total_max']:.6e}`",
            f"- Off-diagonal BH max: `{current_result['offdiag_max']:.6e}`",
            f"- BH / kinetic proxy: `{current_result['bh_to_kinetic_ratio']:.6e}`",
            f"- BH / lambda proxy: `{current_result['bh_to_lambda_ratio']:.6e}`",
            "",
        ])
    lines.append("## Best By Size")
    for size, item in sorted(best_by_size.items()):
        lines.append(
            f"- Size {size}: bands `{item['subset_bands']}`, total BH max `{item['total_max']:.6e}`, offdiag max `{item['offdiag_max']:.6e}`, BH score `{item['bh_score']:.6e}`"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Survey Born-Huang magnitude across band subsets")
    parser.add_argument("--phase1", required=True, help="Path to phase1_multiband_data.h5")
    parser.add_argument("--output-dir", required=True, help="Directory for born_huang_survey.{json,md}")
    parser.add_argument("--mode", choices=["contiguous", "combinations"], default="contiguous")
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 4, 5])
    args = parser.parse_args()

    report = run_survey(Path(args.phase1), Path(args.output_dir), args.mode, args.sizes)
    print(json.dumps({
        "current_subspace": report["current_subspace"],
        "current_subspace_metrics": report["current_subspace_metrics"],
        "best_by_size": report["best_by_size"],
        "results_count": len(report["results"]),
    }, indent=2))


if __name__ == "__main__":
    main()