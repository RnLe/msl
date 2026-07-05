#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict[str, Any]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
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


def _status_rank(status: str) -> int:
    return {"green": 0, "yellow": 1, "red": 2}[status]


def _combine_status(*statuses: str) -> str:
    return max(statuses, key=_status_rank)


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _read_attrs(hf: h5py.File) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in hf.attrs.keys():
        value = hf.attrs[key]
        attrs[key] = _to_serializable(value)
    return attrs


def _read_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as hf:
        data: dict[str, Any] = {
            "path": str(path),
            "attrs": _read_attrs(hf),
            "datasets": {},
        }
        for key in hf.keys():
            obj = hf[key]
            if isinstance(obj, h5py.Dataset):
                data["datasets"][key] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            elif isinstance(obj, h5py.Group):
                data["datasets"][key] = {
                    "type": "group",
                    "keys": sorted(list(obj.keys())),
                    "attrs": {name: _to_serializable(obj.attrs[name]) for name in obj.attrs.keys()},
                }
        return data


def _as_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value)


def _compare_float(name: str, values: dict[str, float | None], tol: float = 1e-10) -> CheckResult:
    present = {key: val for key, val in values.items() if val is not None}
    missing = sorted(set(values) - set(present))
    if len(present) <= 1:
        status = "yellow" if missing else "green"
        return CheckResult(name, status, "Insufficient data to compare across all phases", {"values": values})

    vals = list(present.values())
    spread = float(max(vals) - min(vals))
    status = "green" if spread <= tol and not missing else ("yellow" if spread <= tol else "red")
    summary = f"spread={spread:.3e}" if not missing else f"spread={spread:.3e}, missing={missing}"
    return CheckResult(name, status, summary, {"values": values, "spread": spread, "tolerance": tol})


def _compare_array(name: str, values: dict[str, Any], tol: float = 1e-10) -> CheckResult:
    arrays = {key: _as_array(val) for key, val in values.items() if val is not None}
    missing = sorted(set(values) - set(arrays))
    if len(arrays) <= 1:
        status = "yellow" if missing else "green"
        return CheckResult(name, status, "Insufficient data to compare across all phases", {"values": values})

    ref_key = next(iter(arrays))
    ref = arrays[ref_key]
    max_abs_diff = 0.0
    shape_mismatch: list[str] = []
    for key, arr in arrays.items():
        if arr.shape != ref.shape:
            shape_mismatch.append(key)
            continue
        diff = np.max(np.abs(arr - ref)) if arr.size else 0.0
        max_abs_diff = max(max_abs_diff, float(diff))

    if shape_mismatch:
        status = "red"
        summary = f"shape mismatch in {shape_mismatch}"
    else:
        status = "green" if max_abs_diff <= tol and not missing else ("yellow" if max_abs_diff <= tol else "red")
        summary = f"max_abs_diff={max_abs_diff:.3e}" if not missing else f"max_abs_diff={max_abs_diff:.3e}, missing={missing}"

    return CheckResult(
        name,
        status,
        summary,
        {
            "values": values,
            "max_abs_diff": max_abs_diff,
            "shape_mismatch": shape_mismatch,
            "tolerance": tol,
        },
    )


def _hermiticity_metrics(matrix: np.ndarray, axis_a: int, axis_b: int) -> dict[str, float]:
    diff = matrix - np.swapaxes(np.conj(matrix), axis_a, axis_b)
    denom = float(np.max(np.abs(matrix))) if matrix.size else 0.0
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    rel = max_abs / max(denom, 1e-15)
    return {
        "max_abs": max_abs,
        "relative_max_abs": rel,
        "max_value": denom,
    }


def _check_phase2_tensor(
    path: Path,
    dataset_name: str,
    axis_a: int,
    axis_b: int,
    tol_abs: float = 1e-8,
    tol_rel: float = 1e-6,
) -> CheckResult:
    with h5py.File(path, "r") as hf:
        tensor = hf[dataset_name][:]

    metrics = _hermiticity_metrics(tensor, axis_a, axis_b)
    finite = bool(np.all(np.isfinite(tensor)))
    status = "green"
    if not finite:
        status = "red"
    elif metrics["max_abs"] > tol_abs and metrics["relative_max_abs"] > tol_rel:
        status = "red"

    summary = (
        f"finite={finite}, herm_max={metrics['max_abs']:.3e}, rel={metrics['relative_max_abs']:.3e}, "
        f"value_max={metrics['max_value']:.3e}"
    )
    details = {
        "finite": finite,
        **metrics,
        "min_real": float(np.min(np.real(tensor))),
        "max_real": float(np.max(np.real(tensor))),
        "mean_abs": float(np.mean(np.abs(tensor))),
        "shape": list(tensor.shape),
        "tolerance_abs": tol_abs,
        "tolerance_rel": tol_rel,
    }
    return CheckResult(f"phase2_{dataset_name}", status, summary, details)


def _check_sparse_hamiltonian(path: Path, tol: float = 1e-8) -> CheckResult:
    with h5py.File(path, "r") as hf:
        h_shape = tuple(int(x) for x in hf.attrs["H_shape"])
        h = sparse.csr_matrix((hf["H_data"][:], hf["H_indices"][:], hf["H_indptr"][:]), shape=h_shape)
        diag = h.diagonal()

    diff = h - h.getH()
    if diff.nnz:
        max_abs = float(np.max(np.abs(diff.data)))
    else:
        max_abs = 0.0
    max_val = float(np.max(np.abs(h.data))) if h.nnz else 0.0
    rel = max_abs / max(max_val, 1e-15)
    finite = bool(np.all(np.isfinite(h.data)))
    status = "green"
    if not finite:
        status = "red"
    elif max_abs > tol:
        status = "red"

    summary = f"finite={finite}, herm_max={max_abs:.3e}, rel={rel:.3e}, nnz={h.nnz}"
    details = {
        "finite": finite,
        "hermiticity_max_abs": max_abs,
        "relative_max_abs": rel,
        "nnz": int(h.nnz),
        "shape": list(h.shape),
        "diag_min": float(np.min(np.real(diag))),
        "diag_max": float(np.max(np.real(diag))),
        "tolerance": tol,
    }
    return CheckResult("phase3_hamiltonian", status, summary, details)


def _check_scale_consistency(phase1_attrs: dict[str, Any], phase2_meta: dict[str, Any], phase3_attrs: dict[str, Any]) -> CheckResult:
    theta_rad = float(phase1_attrs["theta_rad"])
    eta = float(phase1_attrs["eta"])
    n_registry = int(phase2_meta["n_registry"])
    ns1 = int(phase1_attrs["Ns1"])
    moire_length = float(phase1_attrs["moire_length"])
    b_moire = np.asarray(phase1_attrs["B_moire"], dtype=float)
    basis_length = float(np.linalg.norm(b_moire[:, 0]))
    single_moire_length = float(1.0 / (2.0 * math.sin(theta_rad / 2.0)))
    ratio_to_single = moire_length / single_moire_length
    dR_registry = moire_length / n_registry
    dR_phase3 = basis_length / ns1
    implied_eta = 2.0 * math.sin(theta_rad / 2.0)

    statuses = [
        "green" if abs(implied_eta - eta) <= 1e-12 else "red",
        "green" if abs(moire_length - basis_length) <= 1e-10 else "red",
        "green" if abs(dR_phase3 - moire_length / ns1) <= 1e-12 else "red",
    ]
    if abs(ratio_to_single - 1.0) > 1e-6:
        statuses.append("red")
        cell_domain = "commensurate_coincidence_cell"
    else:
        cell_domain = "single_moire_cell"

    status = _combine_status(*statuses)
    summary = (
        f"stored_L={moire_length:.6f}, basis_L={basis_length:.6f}, single_moire_L={single_moire_length:.6f}, "
        f"ratio_to_single={ratio_to_single:.6f}, cell_domain={cell_domain}"
    )
    details = {
        "stored_moire_length": moire_length,
        "basis_length": basis_length,
        "single_moire_length_from_theta": single_moire_length,
        "ratio_to_single": ratio_to_single,
        "cell_domain": cell_domain,
        "dR_registry": dR_registry,
        "dR_phase3": dR_phase3,
        "eta_from_attr": eta,
        "eta_from_theta": implied_eta,
        "phase3_has_moire_length_attr": "moire_length" in phase3_attrs,
    }
    return CheckResult("scale_and_domain", status, summary, details)


def _phase_paths(run_dir: Path, case_name: str, config_name: str) -> dict[str, Path]:
    case_dir = run_dir / case_name
    config_dir = case_dir / config_name
    return {
        "case_dir": case_dir,
        "phase1": case_dir / "shared_phase1" / "candidate_0000" / "phase1_multiband_data.h5",
        "phase2": config_dir / "candidate_0000" / "phase2_multiband_data.h5",
        "phase3": config_dir / "candidate_0000" / "phase3_multiband_modes.h5",
        "shared_config": case_dir / "shared_phase1" / "config.json",
        "config": config_dir / "config.json",
        "tracking": case_dir / "shared_phase1" / "candidate_0000" / "phase1_tracking_diagnostics.json",
        "subspace": case_dir / "subspace_diagnostic.json",
    }


def run_audit(run_dir: Path, case_name: str, config_name: str) -> dict[str, Any]:
    paths = _phase_paths(run_dir, case_name, config_name)
    missing = [str(path) for key, path in paths.items() if key in {"phase1", "phase2", "phase3", "config"} and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    phase1 = _read_h5(paths["phase1"])
    phase2 = _read_h5(paths["phase2"])
    phase3 = _read_h5(paths["phase3"])
    config = _load_json(paths["config"])
    shared_config = _load_json(paths["shared_config"]) if paths["shared_config"].exists() else None
    tracking = _load_json(paths["tracking"]) if paths["tracking"].exists() else None
    subspace = _load_json(paths["subspace"]) if paths["subspace"].exists() else None

    checks: list[CheckResult] = []
    checks.append(_compare_float("theta_deg", {
        "phase1": phase1["attrs"].get("theta_deg"),
        "phase2": phase2["attrs"].get("theta_deg"),
        "phase3": phase3["attrs"].get("theta_deg"),
    }, tol=1e-10))
    checks.append(_compare_float("theta_rad", {
        "phase1": phase1["attrs"].get("theta_rad"),
        "phase2": phase2["attrs"].get("theta_rad"),
        "phase3": phase3["attrs"].get("theta_rad"),
    }, tol=1e-12))
    checks.append(_compare_float("eta", {
        "phase1": phase1["attrs"].get("eta"),
        "phase2": phase2["attrs"].get("eta"),
        "phase3": phase3["attrs"].get("eta"),
    }, tol=1e-12))
    checks.append(_compare_float("omega_ref", {
        "phase1": phase1["attrs"].get("omega_ref"),
        "phase2": phase2["attrs"].get("omega_ref"),
        "phase3": phase3["attrs"].get("omega_ref"),
    }, tol=1e-12))
    checks.append(_compare_float("Ns1", {
        "phase1": phase1["attrs"].get("Ns1"),
        "phase2": phase2["attrs"].get("Ns1"),
        "phase3": phase3["attrs"].get("Ns1"),
    }, tol=0.0))
    checks.append(_compare_float("Ns2", {
        "phase1": phase1["attrs"].get("Ns2"),
        "phase2": phase2["attrs"].get("Ns2"),
        "phase3": phase3["attrs"].get("Ns2"),
    }, tol=0.0))
    checks.append(_compare_float("N_subspace", {
        "phase1": phase1["attrs"].get("N_subspace"),
        "phase2": phase2["attrs"].get("N_subspace"),
        "phase3": phase3["attrs"].get("N_subspace"),
    }, tol=0.0))
    checks.append(_compare_array("subspace_bands", {
        "phase1": phase1["attrs"].get("subspace_bands"),
        "phase2": phase2["attrs"].get("subspace_bands"),
        "phase3": phase3["attrs"].get("subspace_bands"),
    }, tol=0.0))
    checks.append(_compare_array("all_bands", {
        "phase1": phase1["attrs"].get("all_bands"),
        "phase2": phase2["attrs"].get("all_bands"),
        "phase3": None,
    }, tol=0.0))
    checks.append(_compare_array("B_moire", {
        "phase1": phase1["attrs"].get("B_moire"),
        "phase2": phase2["attrs"].get("B_moire"),
        "phase3": phase3["attrs"].get("B_moire"),
    }, tol=1e-10))
    checks.append(_compare_array("B_mono", {
        "phase1": phase1["attrs"].get("B_mono"),
        "phase2": phase2["attrs"].get("B_mono"),
        "phase3": phase3["attrs"].get("B_mono"),
    }, tol=1e-10))
    checks.append(_compare_float("moire_length", {
        "phase1": phase1["attrs"].get("moire_length"),
        "phase2": phase2["attrs"].get("moire_length"),
        "phase3": phase3["attrs"].get("moire_length"),
    }, tol=1e-12))

    phase2_n_registry = phase1["datasets"]["stencil"]["attrs"]["n_registry"]
    checks.append(_check_scale_consistency(phase1["attrs"], {"n_registry": phase2_n_registry}, phase3["attrs"]))
    checks.append(_check_phase2_tensor(paths["phase2"], "Lambda", -2, -1, tol_abs=1e-10, tol_rel=1e-10))
    checks.append(_check_phase2_tensor(paths["phase2"], "A_berry", -3, -2, tol_abs=1e-8, tol_rel=1e-6))
    checks.append(_check_phase2_tensor(paths["phase2"], "Phi_BH", -2, -1, tol_abs=1e-8, tol_rel=1e-6))
    checks.append(_check_phase2_tensor(paths["phase2"], "v_drift", -3, -2, tol_abs=1e-6, tol_rel=1e-6))
    checks.append(_check_phase2_tensor(paths["phase2"], "M_inv", -4, -3, tol_abs=1e-8, tol_rel=1e-6))
    checks.append(_check_sparse_hamiltonian(paths["phase3"], tol=1e-8))

    phase3_expected_size = int(phase3["attrs"]["Ns1"]) * int(phase3["attrs"]["Ns2"]) * int(phase3["attrs"]["N_subspace"])
    h_shape = tuple(int(x) for x in phase3["attrs"]["H_shape"])
    h_status = "green" if h_shape == (phase3_expected_size, phase3_expected_size) else "red"
    checks.append(CheckResult(
        "phase3_dimension_consistency",
        h_status,
        f"stored_H_shape={h_shape}, expected={(phase3_expected_size, phase3_expected_size)}",
        {
            "stored_H_shape": list(h_shape),
            "expected_H_shape": [phase3_expected_size, phase3_expected_size],
        },
    ))

    overall = "green"
    for check in checks:
        overall = _combine_status(overall, check.status)

    report = {
        "run_dir": str(run_dir),
        "case": case_name,
        "config_name": config_name,
        "overall_status": overall,
        "paths": {key: str(val) for key, val in paths.items()},
        "config": config,
        "shared_config": shared_config,
        "tracking_diagnostics": tracking,
        "subspace_diagnostic": subspace,
        "phase_summaries": {
            "phase1": phase1,
            "phase2": phase2,
            "phase3": phase3,
        },
        "checks": [_to_serializable(check.__dict__) for check in checks],
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Phase 1-3 Audit: {report['case']} / {report['config_name']}",
        "",
        f"Overall status: **{report['overall_status'].upper()}**",
        "",
        "## Checks",
    ]
    for check in report["checks"]:
        lines.append(f"- `{check['name']}`: **{check['status'].upper()}** — {check['summary']}")
    lines.extend([
        "",
        "## Key Findings",
    ])
    for check in report["checks"]:
        if check["name"] == "scale_and_domain":
            details = check["details"]
            lines.append(
                f"- Domain classification: `{details['cell_domain']}` with stored/basis length {details['stored_moire_length']:.6f}/{details['basis_length']:.6f} and single-cell length {details['single_moire_length_from_theta']:.6f}."
            )
        if check["name"] == "phase3_hamiltonian":
            details = check["details"]
            lines.append(
                f"- Phase 3 Hamiltonian Hermiticity: max abs error {details['hermiticity_max_abs']:.3e}, relative {details['relative_max_abs']:.3e}, nnz={details['nnz']}."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 1-3 handoff consistency for direct validation runs")
    parser.add_argument("--run-dir", required=True, help="Path to run_*/ directory")
    parser.add_argument("--case", required=True, help="Case name, e.g. 10deg")
    parser.add_argument("--config", default="mt_raw", help="Config folder, e.g. mt_raw or single/mt_raw")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report = run_audit(run_dir, args.case, args.config)
    case_dir = run_dir / args.case / args.config
    out_json = case_dir / "phase123_audit.json"
    out_md = case_dir / "phase123_audit.md"
    out_json.write_text(json.dumps(_to_serializable(report), indent=2))
    out_md.write_text(_render_markdown(report))
    print(json.dumps({
        "overall_status": report["overall_status"],
        "json": str(out_json),
        "markdown": str(out_md),
    }, indent=2))


if __name__ == "__main__":
    main()