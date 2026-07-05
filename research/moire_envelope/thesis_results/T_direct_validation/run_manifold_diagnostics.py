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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phasesV3.bloch_fields import compute_born_huang_from_fields, diagnose_born_huang_values
from phasesV3.phase2_mpb_v3 import (
    apply_abelian_gauge_2d,
    apply_svqb_to_bloch_fields,
    compute_berry_connection_from_eigenvectors,
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _parse_sizes(text: str | None, max_size: int) -> list[int]:
    if text is None:
        return list(range(1, min(max_size, 6) + 1))
    sizes = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    return [size for size in sizes if 1 <= size <= max_size]


def _fd_coefficients(fd_order: int) -> tuple[np.ndarray, np.ndarray]:
    if fd_order == 6:
        coeff_first = np.array([-1, 9, -45, 0, 45, -9, 1], dtype=float) / 60.0
        coeff_second = np.array([2, -27, 270, -490, 270, -27, 2], dtype=float) / 180.0
    elif fd_order == 4:
        coeff_first = np.array([1, -8, 0, 8, -1], dtype=float) / 12.0
        coeff_second = np.array([-1, 16, -30, 16, -1], dtype=float) / 12.0
    else:
        coeff_first = np.array([-0.5, 0.0, 0.5], dtype=float)
        coeff_second = np.array([1.0, -2.0, 1.0], dtype=float)
    return coeff_first, coeff_second


def _reconstruct_band_kinematics(stencil_omega: np.ndarray, dk: float, fd_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeff_first, coeff_second = _fd_coefficients(fd_order)
    center = len(coeff_first) // 2
    omega0 = stencil_omega[:, :, :, center, center]
    vg = np.zeros(stencil_omega.shape[:3] + (2,), dtype=float)
    m_inv = np.zeros(stencil_omega.shape[:3] + (2, 2), dtype=float)

    vg[..., 0] = np.tensordot(stencil_omega[:, :, :, :, center], coeff_first, axes=([3], [0])) / dk
    vg[..., 1] = np.tensordot(stencil_omega[:, :, :, center, :], coeff_first, axes=([3], [0])) / dk

    m_inv[..., 0, 0] = np.tensordot(stencil_omega[:, :, :, :, center], coeff_second, axes=([3], [0])) / (dk ** 2)
    m_inv[..., 1, 1] = np.tensordot(stencil_omega[:, :, :, center, :], coeff_second, axes=([3], [0])) / (dk ** 2)

    mixed = np.zeros(stencil_omega.shape[:3], dtype=float)
    for ix, cx in enumerate(coeff_first):
        for iy, cy in enumerate(coeff_first):
            mixed += cx * cy * stencil_omega[:, :, :, ix, iy]
    mixed /= dk ** 2
    m_inv[..., 0, 1] = mixed
    m_inv[..., 1, 0] = mixed
    return omega0, vg, m_inv


def _subset_lists(all_bands: list[int], mode: str, sizes: list[int]) -> list[list[int]]:
    subsets: list[list[int]] = []
    if mode == "contiguous":
        for size in sizes:
            for start in range(0, len(all_bands) - size + 1):
                subsets.append(all_bands[start:start + size])
    elif mode == "combinations":
        for size in sizes:
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


def _classify_extremum(vg_mag: np.ndarray, m_inv_band: np.ndarray, vg_tol: float) -> dict[str, Any]:
    if not np.isfinite(vg_mag).all():
        finite = vg_mag[np.isfinite(vg_mag)]
        vg_max = float(finite.max()) if finite.size else float("inf")
    else:
        vg_max = float(vg_mag.max())
    trace = m_inv_band[..., 0, 0] + m_inv_band[..., 1, 1]
    det = m_inv_band[..., 0, 0] * m_inv_band[..., 1, 1] - m_inv_band[..., 0, 1] * m_inv_band[..., 1, 0]
    evals = np.linalg.eigvalsh(m_inv_band.reshape(-1, 2, 2))
    min_mask = np.logical_and(evals[:, 0] > 0, evals[:, 1] > 0)
    max_mask = np.logical_and(evals[:, 0] < 0, evals[:, 1] < 0)
    saddle_mask = np.logical_and(evals[:, 0] * evals[:, 1] < 0, np.ones_like(evals[:, 0], dtype=bool))
    near_extremum_fraction = float(np.mean(vg_mag <= vg_tol))
    dominant = "mixed"
    fractions = {
        "minimum": float(np.mean(min_mask)),
        "maximum": float(np.mean(max_mask)),
        "saddle": float(np.mean(saddle_mask)),
    }
    if near_extremum_fraction >= 0.8:
        dominant = max(fractions, key=fractions.get)
        if fractions[dominant] < 0.5:
            dominant = "mixed"
    else:
        dominant = "non_extremal"
    return {
        "vg_max": vg_max,
        "vg_mean": float(np.mean(vg_mag)),
        "near_extremum_fraction": near_extremum_fraction,
        "trace_mean": float(np.mean(trace)),
        "trace_abs_max": float(np.max(np.abs(trace))),
        "det_mean": float(np.mean(det)),
        "fractions": fractions,
        "dominant_type": dominant,
    }


def _spectral_metrics(omega_registry: np.ndarray, subset_indices: list[int]) -> dict[str, Any]:
    n_bands = omega_registry.shape[2]
    subset_set = set(subset_indices)
    outside_indices = [idx for idx in range(n_bands) if idx not in subset_set]
    subset = omega_registry[:, :, subset_indices]
    modulation = np.max(subset, axis=(0, 1)) - np.min(subset, axis=(0, 1))
    manifold_span = float(np.max(subset) - np.min(subset))

    if outside_indices:
        outside = omega_registry[:, :, outside_indices]
        diff = np.abs(subset[:, :, :, np.newaxis] - outside[:, :, np.newaxis, :])
        complement_gap = np.min(diff, axis=(2, 3))
        min_complement_gap = float(np.min(complement_gap))
        mean_complement_gap = float(np.mean(complement_gap))
    else:
        complement_gap = np.full(omega_registry.shape[:2], np.inf)
        min_complement_gap = float("inf")
        mean_complement_gap = float("inf")

    max_modulation = float(np.max(modulation)) if modulation.size else 0.0
    isolation_ratio = None if not np.isfinite(min_complement_gap) else min_complement_gap / max(max_modulation, 1e-15)
    return {
        "modulation_ranges": [float(val) for val in modulation],
        "max_modulation_range": max_modulation,
        "manifold_span": manifold_span,
        "min_complement_gap": min_complement_gap,
        "mean_complement_gap": mean_complement_gap,
        "isolation_ratio": isolation_ratio,
    }


def _expand_epsilon(epsilon: np.ndarray, n_components: int) -> np.ndarray:
    return np.repeat(epsilon[:, :, :, :, np.newaxis], n_components, axis=4).reshape(epsilon.shape[0], epsilon.shape[1], -1)


def _projector_smoothness(subset_fields: np.ndarray, epsilon: np.ndarray) -> dict[str, Any]:
    ns1, ns2, n_sub = subset_fields.shape[:3]
    n_components = subset_fields.shape[-1]
    eps_flat = _expand_epsilon(epsilon, n_components)
    flat = subset_fields.reshape(ns1, ns2, n_sub, -1)
    min_singular_values: list[float] = []
    unitary_residuals: list[float] = []
    for axis in (0, 1):
        shifted = np.roll(flat, -1, axis=axis)
        for i in range(ns1):
            for j in range(ns2):
                u_curr = flat[i, j]
                u_next = shifted[i, j]
                eps_ij = eps_flat[i, j]
                overlap = (u_curr.conj() * eps_ij[np.newaxis, :]) @ u_next.T
                singular_values = np.linalg.svd(overlap, compute_uv=False)
                min_singular_values.append(float(np.min(singular_values)))
                unitary = overlap.conj().T @ overlap
                unitary_residuals.append(float(np.linalg.norm(unitary - np.eye(n_sub), ord="fro")))
    return {
        "min_singular_value": float(np.min(min_singular_values)) if min_singular_values else 0.0,
        "mean_min_singular_value": float(np.mean(min_singular_values)) if min_singular_values else 0.0,
        "max_unitarity_residual": float(np.max(unitary_residuals)) if unitary_residuals else 0.0,
        "mean_unitarity_residual": float(np.mean(unitary_residuals)) if unitary_residuals else 0.0,
    }


def _berry_metrics(subset_fields: np.ndarray, epsilon: np.ndarray, moire_length: float) -> dict[str, Any]:
    ns1, ns2 = subset_fields.shape[:2]
    dR = moire_length / ns1
    berry, diagnostics = compute_berry_connection_from_eigenvectors(
        subset_fields,
        dR,
        dR,
        fd_order=4,
        epsilon=epsilon,
        return_diagnostics=True,
    )
    n_sub = subset_fields.shape[2]
    offdiag = berry.copy()
    for idx in range(n_sub):
        offdiag[:, :, idx, idx, :] = 0.0
    offdiag_flat = offdiag.reshape(-1, n_sub * n_sub * 2)
    offdiag_norm = np.sqrt(np.sum(np.abs(offdiag_flat) ** 2, axis=1))
    diag_x = np.diagonal(berry[..., 0], axis1=2, axis2=3)
    diag_y = np.diagonal(berry[..., 1], axis1=2, axis2=3)
    diag_norm = float(np.sqrt(np.sum(np.abs(diag_x) ** 2) + np.sum(np.abs(diag_y) ** 2)))
    return {
        "offdiag_abs_max": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_fro_mean": float(np.mean(offdiag_norm)) if offdiag_norm.size else 0.0,
        "diag_total_norm": diag_norm,
        "raw_hermiticity_max_abs": diagnostics["raw_hermiticity_max_abs"],
    }


def _bh_metrics(subset_fields: np.ndarray, subset_indices: list[int], epsilon: np.ndarray, moire_length: float, total_bands: int) -> dict[str, Any]:
    extra_indices = [idx for idx in range(total_bands) if idx not in subset_indices]
    dR = (moire_length / subset_fields.shape[0], moire_length / subset_fields.shape[1])
    phi_bh = compute_born_huang_from_fields(
        bloch_fields=subset_fields if len(subset_indices) == total_bands else None,
        dR=dR,
        subspace_indices=list(range(len(subset_indices))) if len(subset_indices) == total_bands else [],
    )
    raise RuntimeError("Internal misuse")


def _bh_metrics_from_all_fields(all_fields: np.ndarray, subset_indices: list[int], epsilon: np.ndarray, moire_length: float) -> dict[str, Any]:
    extra_indices = [idx for idx in range(all_fields.shape[2]) if idx not in subset_indices]
    dR = (moire_length / all_fields.shape[0], moire_length / all_fields.shape[1])
    phi_bh = compute_born_huang_from_fields(
        bloch_fields=all_fields,
        dR=dR,
        subspace_indices=subset_indices,
        extra_indices=extra_indices,
        epsilon=epsilon,
    )
    basic = diagnose_born_huang_values(phi_bh)
    offdiag = phi_bh.copy()
    for idx in range(phi_bh.shape[2]):
        offdiag[:, :, idx, idx] = 0.0
    return {
        "total_max": float(np.max(np.abs(phi_bh))) if phi_bh.size else 0.0,
        "trace_mean": float(np.mean(np.trace(phi_bh, axis1=2, axis2=3))) if phi_bh.size else 0.0,
        "offdiag_max": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "basic": _to_jsonable(basic),
    }


def _health_label(isolation_ratio: float | None, smoothness: dict[str, Any], bh_metrics: dict[str, Any] | None) -> str:
    if isolation_ratio is None:
        return "unknown"
    min_sv = smoothness.get("min_singular_value", 0.0)
    bh_penalty = bh_metrics["total_max"] / max(isolation_ratio, 1e-15) if bh_metrics is not None else 0.0
    if isolation_ratio >= 0.75 and min_sv >= 0.85 and bh_penalty < 0.25:
        return "green"
    if isolation_ratio >= 0.25 and min_sv >= 0.50 and bh_penalty < 1.0:
        return "yellow"
    return "red"


def _score_candidate(spectral: dict[str, Any], smoothness: dict[str, Any], bh_metrics: dict[str, Any] | None, band_summaries: list[dict[str, Any]]) -> float:
    isolation_raw = spectral["isolation_ratio"] if spectral["isolation_ratio"] is not None else 0.0
    isolation = float(np.tanh(max(isolation_raw, 0.0)))
    smooth = float(np.tanh(max(smoothness.get("mean_min_singular_value", 0.0), 0.0)))
    extremum_bonus = float(np.mean([summary["near_extremum_fraction"] for summary in band_summaries]))
    bh_penalty = 0.0
    if bh_metrics is not None:
        bh_ratio = bh_metrics["total_max"] / max(spectral["min_complement_gap"], 1e-12)
        bh_penalty = float(np.tanh(max(bh_ratio, 0.0)))
    return float(3.0 * isolation + 1.5 * smooth + 0.5 * extremum_bonus - 1.5 * bh_penalty)


def _load_phase1_for_diagnostics(phase1_path: Path) -> dict[str, Any]:
    with h5py.File(phase1_path, "r") as hf:
        stencil_omega = hf["stencil"]["omega_all"][:]
        registry_omega = hf["stencil"]["registry_omega_all"][:]
        all_bands = [int(band) for band in hf.attrs["all_bands"][:]]
        current_subspace = [int(band) for band in hf.attrs["subspace_bands"][:]]
        dk = float(hf["stencil"].attrs["dk"])
        fd_order = int(hf["stencil"].attrs["fd_order"])
        moire_length = float(hf.attrs["moire_length"])
        theta_deg = float(hf.attrs["theta_deg"])
        has_fields = "bloch_fields" in hf and "epsilon" in hf
        bloch_fields = hf["bloch_fields"][:] if has_fields else None
        epsilon = hf["epsilon"][:] if has_fields else None
    omega0, vg_all, m_inv_all = _reconstruct_band_kinematics(stencil_omega, dk, fd_order)
    return {
        "phase1_path": str(phase1_path),
        "all_bands": all_bands,
        "current_subspace": current_subspace,
        "registry_omega": registry_omega,
        "omega0": omega0,
        "vg_all": vg_all,
        "m_inv_all": m_inv_all,
        "moire_length": moire_length,
        "theta_deg": theta_deg,
        "bloch_fields": bloch_fields,
        "epsilon": epsilon,
    }


def diagnose_phase1(phase1_path: Path, mode: str, sizes: list[int], vg_tol: float) -> dict[str, Any]:
    data = _load_phase1_for_diagnostics(phase1_path)
    all_bands = data["all_bands"]
    registry_omega = data["registry_omega"]
    vg_all = data["vg_all"]
    m_inv_all = data["m_inv_all"]
    subsets = _subset_lists(all_bands, mode, sizes)

    fields = data["bloch_fields"]
    epsilon = data["epsilon"]
    gauge_diag = None
    svqb_stats = None
    if fields is not None and epsilon is not None:
        fields = np.array(fields, copy=True)
        fields, gauge_diag = apply_abelian_gauge_2d(fields)
        fields, svqb_stats = apply_svqb_to_bloch_fields(fields, epsilon)

    per_subset: list[dict[str, Any]] = []
    for subset_bands in subsets:
        subset_indices = [all_bands.index(band) for band in subset_bands]
        spectral = _spectral_metrics(registry_omega, subset_indices)
        band_summaries = []
        for subset_idx, band in zip(subset_indices, subset_bands):
            vg_mag = np.linalg.norm(vg_all[:, :, subset_idx, :], axis=2)
            band_summary = _classify_extremum(vg_mag, m_inv_all[:, :, subset_idx, :, :], vg_tol)
            band_summary["band"] = band
            band_summary["omega_min"] = float(np.min(registry_omega[:, :, subset_idx]))
            band_summary["omega_max"] = float(np.max(registry_omega[:, :, subset_idx]))
            band_summaries.append(band_summary)

        smoothness = None
        berry_metrics = None
        bh_metrics = None
        if fields is not None and epsilon is not None:
            subset_fields = fields[:, :, subset_indices, ...]
            smoothness = _projector_smoothness(subset_fields, epsilon)
            berry_metrics = _berry_metrics(subset_fields, epsilon, data["moire_length"])
            bh_metrics = _bh_metrics_from_all_fields(fields, subset_indices, epsilon, data["moire_length"])

        health = _health_label(spectral["isolation_ratio"], smoothness or {}, bh_metrics)
        score = _score_candidate(spectral, smoothness or {}, bh_metrics, band_summaries)
        per_subset.append({
            "subset_bands": subset_bands,
            "subset_size": len(subset_bands),
            "spectral": spectral,
            "bands": band_summaries,
            "smoothness": smoothness,
            "berry": berry_metrics,
            "born_huang": bh_metrics,
            "health": health,
            "score": score,
            "is_current_subspace": subset_bands == data["current_subspace"],
        })

    per_subset.sort(key=lambda item: item["score"], reverse=True)
    best_by_size: dict[int, dict[str, Any]] = {}
    for item in per_subset:
        best_by_size.setdefault(item["subset_size"], item)

    return {
        "phase1_path": data["phase1_path"],
        "theta_deg": data["theta_deg"],
        "all_bands": all_bands,
        "current_subspace": data["current_subspace"],
        "mode": mode,
        "sizes": sizes,
        "gauge_diagnostics": _to_jsonable(gauge_diag),
        "svqb_stats": _to_jsonable(svqb_stats),
        "best_by_size": _to_jsonable(best_by_size),
        "results": _to_jsonable(per_subset),
    }


def _write_markdown(output_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Manifold Diagnostics: {report['scan_name']}",
        "",
        f"Mode: `{report['mode']}`",
        f"Sizes: `{report['sizes']}`",
        "",
    ]
    for case in report["cases"]:
        k_label = case["k_label"]
        lines.extend([
            f"## {k_label}",
            f"- Phase 1: `{case['phase1_path']}`",
            f"- Current subspace: `{case['current_subspace']}`",
            "",
            "### Best by size",
        ])
        for size, item in sorted(case["best_by_size"].items(), key=lambda pair: int(pair[0])):
            spectral = item["spectral"]
            lines.append(
                f"- Size {size}: bands `{item['subset_bands']}`, score `{item['score']:.3f}`, health `{item['health']}`, isolation `{spectral['isolation_ratio']}`, min-gap `{spectral['min_complement_gap']:.6e}`"
            )
        lines.append("")
        lines.append("### Top candidates")
        for item in case["results"][:10]:
            spectral = item["spectral"]
            bh = item.get("born_huang")
            bh_str = f", BH max `{bh['total_max']:.6e}`" if bh is not None else ""
            lines.append(
                f"- `{item['subset_bands']}`: score `{item['score']:.3f}`, health `{item['health']}`, isolation `{spectral['isolation_ratio']}`, min-gap `{spectral['min_complement_gap']:.6e}`{bh_str}"
            )
        lines.append("")
    (output_dir / "manifold_diagnostics.md").write_text("\n".join(lines))


def run_diagnostics(scan_manifest: Path, output_dir: Path | None, mode: str, sizes: list[int], vg_tol: float) -> dict[str, Any]:
    manifest = json.loads(scan_manifest.read_text())
    target_dir = output_dir or (scan_manifest.parent / "diagnostics")
    target_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for item in manifest["phase1_outputs"]:
        case_report = diagnose_phase1(Path(item["phase1_path"]), mode, sizes, vg_tol)
        case_report["k_label"] = item["k_label"]
        cases.append(case_report)

    report = {
        "scan_name": manifest["run_name"],
        "scan_dir": manifest["scan_dir"],
        "mode": mode,
        "sizes": sizes,
        "cases": cases,
    }
    (target_dir / "manifold_diagnostics.json").write_text(json.dumps(_to_jsonable(report), indent=2))
    _write_markdown(target_dir, _to_jsonable(report))
    return _to_jsonable(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 2 manifold diagnostics on a Phase 1 scan manifest")
    parser.add_argument("--scan-manifest", required=True, help="Path to scan_manifest.json from run_manifold_phase1_scan.py")
    parser.add_argument("--output-dir", default=None, help="Optional diagnostics output directory")
    parser.add_argument("--mode", choices=["contiguous", "combinations"], default="contiguous")
    parser.add_argument("--sizes", default=None, help="Comma-separated manifold sizes, defaults to 1..min(6, N_bands)")
    parser.add_argument("--vg-tol", type=float, default=1e-3, help="Threshold for identifying near-extremal points from |vg|")
    args = parser.parse_args()

    manifest_path = Path(args.scan_manifest)
    manifest = json.loads(manifest_path.read_text())
    max_size = len(manifest["bands"]["all_bands"])
    sizes = _parse_sizes(args.sizes, max_size)
    report = run_diagnostics(
        manifest_path,
        Path(args.output_dir) if args.output_dir else None,
        args.mode,
        sizes,
        args.vg_tol,
    )
    print(json.dumps({
        "scan_name": report["scan_name"],
        "cases": [case["k_label"] for case in report["cases"]],
        "sizes": report["sizes"],
    }, indent=2))


if __name__ == "__main__":
    main()