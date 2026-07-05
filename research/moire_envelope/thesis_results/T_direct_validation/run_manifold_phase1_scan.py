#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "phasesV3") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "phasesV3"))

from common.geometry import high_symmetry_points
from common.io_utils import save_json
import phase1_mpb_v3 as p1


def _parse_band_list(text: str) -> list[int]:
    bands = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not bands:
        raise ValueError("Band list must not be empty")
    unique = sorted(set(bands))
    return unique


def _select_hs_points(lattice_type: str, requested: str) -> list[tuple[str, tuple[float, float]]]:
    points = high_symmetry_points(lattice_type)
    seen: set[str] = set()
    unique_points: list[tuple[str, tuple[float, float]]] = []
    for label, k_vec in points:
        if label in seen:
            continue
        seen.add(label)
        unique_points.append((label, (float(k_vec[0]), float(k_vec[1]))))

    if requested.lower() == "all":
        return unique_points

    labels = [item.strip() for item in requested.split(",") if item.strip()]
    wanted = {"Γ" if label.lower() == "gamma" else label for label in labels}
    selected = [item for item in unique_points if item[0] in wanted]
    if not selected:
        raise ValueError(f"No matching high-symmetry points for '{requested}' and lattice '{lattice_type}'")
    return selected


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _build_candidate(
    candidate_id: int,
    hs_label: str,
    k0: tuple[float, float],
    args: argparse.Namespace,
    subspace_bands: list[int],
    all_bands: list[int],
) -> dict[str, Any]:
    target_band = args.target_band if args.target_band is not None else subspace_bands[len(subspace_bands) // 2]
    if target_band not in subspace_bands:
        raise ValueError(f"Target band {target_band} must lie inside subspace bands {subspace_bands}")
    return {
        "candidate_id": candidate_id,
        "lattice_type": args.lattice_type,
        "a": args.a,
        "r_over_a": args.r_over_a,
        "eps_bg": args.eps_bg,
        "eps_hole": args.eps_hole,
        "band_index": target_band,
        "merged_band_index": target_band,
        "k_label": hs_label,
        "k0_x": k0[0],
        "k0_y": k0[1],
        "omega0": 0.0,
        "polarization": args.polarization,
        "dominant_polarization": args.polarization,
        "local_polarization": args.polarization,
        "n_subspace_bands": len(subspace_bands),
        "subspace_bands": subspace_bands,
        "all_bands": all_bands,
        "target_index_in_subspace": subspace_bands.index(target_band),
        "theta_deg": args.theta_deg,
        "theta_rad": math.radians(args.theta_deg),
    }


def _build_phase1_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "phase1_Ns1": args.phase1_ns,
        "phase1_Ns2": args.phase1_ns,
        "mpb_resolution": args.mpb_resolution,
        "mpb_registry_samples": args.registry_samples,
        "mpb_dk": args.mpb_dk,
        "mpb_fd_order": args.mpb_fd_order,
        "mpb_polarization": args.polarization,
        "export_bloch_fields": args.export_bloch_fields,
        "mpb_n_workers": args.n_workers,
        "tau": [args.tau_x, args.tau_y],
        "default_theta_deg": args.theta_deg,
        "ref_frequency_mode": args.ref_frequency_mode,
    }


def _summarize_phase1(phase1_path: Path) -> dict[str, Any]:
    with h5py.File(phase1_path, "r") as hf:
        omega = hf["omega"][:]
        vg = hf["vg"][:]
        m_inv = hf["M_inv"][:]
        summary = {
            "path": str(phase1_path),
            "omega_shape": list(omega.shape),
            "vg_shape": list(vg.shape),
            "m_inv_shape": list(m_inv.shape),
            "omega_ref": float(hf.attrs["omega_ref"]),
            "subspace_bands": [int(b) for b in hf.attrs["subspace_bands"][:]],
            "all_bands": [int(b) for b in hf.attrs["all_bands"][:]],
            "omega_min": float(omega.min()),
            "omega_max": float(omega.max()),
            "vg_abs_max": float(abs(vg).max()),
            "m_inv_abs_max": float(abs(m_inv).max()),
            "has_bloch_fields": "bloch_fields" in hf,
            "has_epsilon": "epsilon" in hf,
        }
    return summary


def _write_markdown_summary(scan_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# Manifold Phase 1 Scan: {manifest['run_name']}",
        "",
        "## Geometry",
        f"- Lattice: `{manifest['geometry']['lattice_type']}`",
        f"- Polarization: `{manifest['geometry']['polarization']}`",
        f"- a: `{manifest['geometry']['a']}`",
        f"- r/a: `{manifest['geometry']['r_over_a']}`",
        f"- eps_bg: `{manifest['geometry']['eps_bg']}`",
        f"- eps_hole: `{manifest['geometry']['eps_hole']}`",
        f"- theta_deg: `{manifest['geometry']['theta_deg']}`",
        "",
        "## Bands",
        f"- Subspace bands: `{manifest['bands']['subspace_bands']}`",
        f"- All exported bands: `{manifest['bands']['all_bands']}`",
        f"- Target band for Phase 1 metadata: `{manifest['bands']['target_band']}`",
        "",
        "## High-Symmetry Points",
    ]
    for item in manifest["phase1_outputs"]:
        lines.extend([
            f"### {item['k_label']}",
            f"- Candidate dir: `{item['candidate_dir']}`",
            f"- Phase 1 file: `{item['phase1_path']}`",
            f"- omega range: `{item['summary']['omega_min']:.6f}` → `{item['summary']['omega_max']:.6f}`",
            f"- max |vg|: `{item['summary']['vg_abs_max']:.6e}`",
            f"- max |M_inv|: `{item['summary']['m_inv_abs_max']:.6e}`",
            f"- Bloch fields exported: `{item['summary']['has_bloch_fields']}`",
            "",
        ])
    (scan_dir / "scan_summary.md").write_text("\n".join(lines))


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    scan_root = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.lattice_type}_{args.polarization.lower()}_manifold_phase1"
    scan_dir = scan_root / f"{run_name}_{timestamp}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    subspace_bands = _parse_band_list(args.subspace_bands)
    all_bands = _parse_band_list(args.all_bands) if args.all_bands else subspace_bands
    if not set(subspace_bands).issubset(set(all_bands)):
        raise ValueError("All subspace bands must be contained in all_bands")

    hs_points = _select_hs_points(args.lattice_type, args.high_symmetry_points)
    config = _build_phase1_config(args)
    save_json(config, scan_dir / "phase1_config.json")

    manifest: dict[str, Any] = {
        "run_name": run_name,
        "scan_dir": str(scan_dir),
        "created_at": timestamp,
        "geometry": {
            "lattice_type": args.lattice_type,
            "polarization": args.polarization,
            "a": args.a,
            "r_over_a": args.r_over_a,
            "eps_bg": args.eps_bg,
            "eps_hole": args.eps_hole,
            "theta_deg": args.theta_deg,
            "tau": [args.tau_x, args.tau_y],
        },
        "bands": {
            "subspace_bands": subspace_bands,
            "all_bands": all_bands,
            "target_band": args.target_band if args.target_band is not None else subspace_bands[len(subspace_bands) // 2],
        },
        "phase1_outputs": [],
    }

    for candidate_id, (hs_label, k0) in enumerate(hs_points):
        candidate = _build_candidate(candidate_id, hs_label, k0, args, subspace_bands, all_bands)
        p1.process_candidate_v3(candidate, config, scan_dir)
        candidate_dir = scan_dir / f"candidate_{candidate_id:04d}"
        phase1_path = candidate_dir / "phase1_multiband_data.h5"
        summary = _summarize_phase1(phase1_path)
        manifest["phase1_outputs"].append({
            "candidate_id": candidate_id,
            "k_label": hs_label,
            "k0": list(k0),
            "candidate_dir": str(candidate_dir),
            "phase1_path": str(phase1_path),
            "summary": summary,
        })

    save_json(manifest, scan_dir / "scan_manifest.json")
    _write_markdown_summary(scan_dir, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 MPB sweeps at high-symmetry points for manifold diagnostics")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "manifold_phase1_scans"), help="Directory that will contain the scan run directory")
    parser.add_argument("--run-name", default=None, help="Optional run-name prefix")
    parser.add_argument("--lattice-type", required=True, choices=["square", "hex", "honeycomb", "rect"], help="Monolayer lattice type")
    parser.add_argument("--polarization", default="TM", choices=["TE", "TM"], help="MPB polarization")
    parser.add_argument("--a", type=float, default=1.0, help="Lattice constant")
    parser.add_argument("--r-over-a", type=float, required=True, help="Cylinder/rod radius over lattice constant")
    parser.add_argument("--eps-bg", type=float, required=True, help="Background dielectric constant")
    parser.add_argument("--eps-hole", type=float, default=1.0, help="Hole/rod dielectric constant")
    parser.add_argument("--theta-deg", type=float, required=True, help="Twist angle used to define the moire sampling path")
    parser.add_argument("--tau-x", type=float, default=0.0, help="Fractional registry offset x")
    parser.add_argument("--tau-y", type=float, default=0.0, help="Fractional registry offset y")
    parser.add_argument("--high-symmetry-points", default="all", help="Comma-separated list like 'Gamma,M' or 'all'")
    parser.add_argument("--subspace-bands", required=True, help="Comma-separated retained band list, e.g. '1,2,3,4'")
    parser.add_argument("--all-bands", default=None, help="Comma-separated full exported band list; defaults to subspace-bands")
    parser.add_argument("--target-band", type=int, default=None, help="Band used as Phase 1 target metadata; defaults to subspace midpoint")
    parser.add_argument("--phase1-ns", type=int, default=128, help="Phase 1 moire grid size per direction")
    parser.add_argument("--registry-samples", type=int, default=48, help="MPB registry sample count per direction")
    parser.add_argument("--mpb-resolution", type=int, default=64, help="MPB spatial resolution")
    parser.add_argument("--mpb-dk", type=float, default=0.06, help="k-space finite-difference step")
    parser.add_argument("--mpb-fd-order", type=int, default=6, choices=[2, 4, 6], help="Finite-difference stencil order")
    parser.add_argument("--n-workers", type=int, default=16, help="Worker count for the MPB registry sweep")
    parser.add_argument("--ref-frequency-mode", default="mean", choices=["mean", "min", "max", "median"], help="Reference-frequency selector used by Phase 1")
    parser.add_argument("--export-bloch-fields", action="store_true", help="Export Bloch fields and epsilon grids for later Berry/BH diagnostics")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest = run_scan(args)
    print(json.dumps({
        "scan_dir": manifest["scan_dir"],
        "high_symmetry_points": [item["k_label"] for item in manifest["phase1_outputs"]],
        "subspace_bands": manifest["bands"]["subspace_bands"],
        "all_bands": manifest["bands"]["all_bands"],
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()