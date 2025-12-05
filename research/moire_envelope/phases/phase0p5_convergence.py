"""Phase 0.5: MPB convergence checks before running the full moiré pipeline."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, save_json
from common.moire_utils import compute_registry_map
from common.mpb_utils import compute_local_band_at_registry
from phases.phase1_local_bloch import (
    extract_candidate_parameters,
    ensure_moire_metadata,
    build_bilayer_geometry_at_delta,
)


def resolve_run_dir(run_dir: str | Path, config: Dict) -> Path:
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        runs_base = Path(config.get("output_dir", "runs"))
        phase0_runs = sorted(runs_base.glob("phase0_real_run_*"))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_real_run_* directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest Phase 0 run: {run_dir}")
    return run_dir


def _select_candidate_ids(frame: pd.DataFrame, config: Dict, env_override: str | None) -> List[int]:
    if frame.empty:
        return []
    subset = frame.copy()
    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        subset = subset.head(K)
    ids = subset["candidate_id"].astype(int).tolist()
    if env_override:
        try:
            value = int(env_override)
            if value in ids:
                return [value]
            if value in frame["candidate_id"].values:
                return [value]
            print(f"  WARNING: MSL_PHASE0P5_CANDIDATE_ID={value} not in current selection; skipping phase 0.5.")
            return []
        except ValueError:
            print(f"  WARNING: Invalid MSL_PHASE0P5_CANDIDATE_ID='{env_override}'; ignoring override.")
    return ids


def _resolve_registry_eta(config: Dict, eta_physical: float) -> float:
    registry_cfg = config.get("phase0p5_registry_eta", "physical")
    if registry_cfg is None:
        return 1.0
    if isinstance(registry_cfg, str):
        key = registry_cfg.lower()
        if key in {"auto", "physical"}:
            return eta_physical
        try:
            return float(registry_cfg)
        except ValueError:
            return eta_physical
    try:
        return float(registry_cfg)
    except (TypeError, ValueError):
        return eta_physical


def _prepare_sample_points(L_moire: float, samples_cfg: Iterable[Dict] | None) -> List[Dict[str, np.ndarray]]:
    samples = []
    default_cfg = [
        {"label": "center", "frac": [0.0, 0.0]},
        {"label": "edge_x", "frac": [0.5, 0.0]},
        {"label": "edge_y", "frac": [0.0, 0.5]},
    ]
    entries = list(samples_cfg) if samples_cfg else default_cfg
    for idx, entry in enumerate(entries):
        frac = np.asarray(entry.get("frac", [0.0, 0.0]), dtype=float)
        label = entry.get("label") or f"P{idx}"
        # frac coordinates interpreted as multiples of L_moire relative to the cell center
        R_vec = frac * float(L_moire)
        samples.append({"label": label, "R": R_vec})
    return samples


def _attach_registry(samples: List[Dict[str, np.ndarray]], a1, a2, theta: float, tau: np.ndarray, eta_registry: float):
    if not samples:
        return samples
    Nx = len(samples)
    R_grid = np.zeros((Nx, 1, 2))
    for idx, sample in enumerate(samples):
        R_grid[idx, 0, :] = sample["R"]
    delta_grid = compute_registry_map(R_grid, a1, a2, theta, tau, eta_registry)
    for idx, sample in enumerate(samples):
        sample["delta_frac"] = delta_grid[idx, 0, :].copy()
    return samples


def _summaries(df: pd.DataFrame, ref_resolution: float, ref_dk: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    refs: Dict[str, Dict[str, float]] = {}
    ok_rows = df[df["status"] == "ok"].copy()
    for label in ok_rows["sample_label"].unique():
        subset = ok_rows[ok_rows["sample_label"] == label]
        if subset.empty:
            continue
        ref = subset[
            (np.isclose(subset["resolution"], ref_resolution)) &
            (np.isclose(subset["dk"], ref_dk))
        ]
        if ref.empty:
            ref = subset.sort_values(["resolution", "dk"], ascending=[False, True]).head(1)
        if ref.empty:
            continue
        row = ref.iloc[0]
        refs[label] = {
            "omega_ref": float(row["omega0"]),
            "vg_ref": float(row["vg_norm"]),
            "m_ref_min": float(row["mass_min"]),
            "m_ref_max": float(row["mass_max"]),
        }
    if not refs:
        return df, {}

    def _map(label: str, key: str):
        ref = refs.get(label)
        return float(ref[key]) if ref else np.nan

    df["omega_ref"] = df["sample_label"].map(lambda lbl: _map(lbl, "omega_ref"))
    df["vg_ref"] = df["sample_label"].map(lambda lbl: _map(lbl, "vg_ref"))
    df["mass_ref_min"] = df["sample_label"].map(lambda lbl: _map(lbl, "m_ref_min"))
    df["mass_ref_max"] = df["sample_label"].map(lambda lbl: _map(lbl, "m_ref_max"))

    df["omega_delta"] = df["omega0"] - df["omega_ref"]
    df["vg_delta"] = df["vg_norm"] - df["vg_ref"]
    df["mass_delta_min"] = df["mass_min"] - df["mass_ref_min"]
    df["mass_delta_max"] = df["mass_max"] - df["mass_ref_max"]

    summary = {}
    ok_mask = df["status"] == "ok"
    if ok_mask.any():
        summary = {
            "omega_max_abs": float(np.nanmax(np.abs(df.loc[ok_mask, "omega_delta"]))),
            "vg_max_abs": float(np.nanmax(np.abs(df.loc[ok_mask, "vg_delta"]))),
            "mass_min_max_abs": float(np.nanmax(np.abs(df.loc[ok_mask, "mass_delta_min"]))),
            "mass_max_max_abs": float(np.nanmax(np.abs(df.loc[ok_mask, "mass_delta_max"]))),
        }
    return df, summary


def _write_outputs(
    cdir: Path,
    cid: int,
    df: pd.DataFrame,
    ref_resolution: float,
    ref_dk: float,
    summary: Dict[str, float],
):
    csv_path = cdir / "phase0p5_convergence.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved convergence table to {csv_path}")

    report_lines = [
        "# Phase 0.5 MPB Convergence Report",
        "",
        f"**Candidate**: {cid:04d}",
        "",
        "## Reference settings",
        f"- Resolution: {ref_resolution}",
        f"- Δk: {ref_dk}",
        "",
        "## Max deviations relative to reference",
    ]
    if summary:
        report_lines.extend(
            [
                f"- |Δω|_max = {summary['omega_max_abs']:.4e}",
                f"- |Δ|v_g||_max = {summary['vg_max_abs']:.4e}",
                f"- |Δ m_min|_max = {summary['mass_min_max_abs']:.4e}",
                f"- |Δ m_max|_max = {summary['mass_max_max_abs']:.4e}",
            ]
        )
    else:
        report_lines.append("- No valid reference combination was computed (check MPB logs).")

    ok_df = df[df["status"] == "ok"]
    if not ok_df.empty:
        grouped = (
            ok_df.groupby(["resolution", "dk"])
            [["omega_delta", "vg_delta", "mass_delta_min", "mass_delta_max"]]
            .agg(lambda vals: np.nanmax(np.abs(vals)))
            .reset_index()
        )
        report_lines.append("")
        report_lines.append("## Combination summary")
        report_lines.append("Resolution | Δk | max |Δω| | max |Δ|v_g|| | max |Δ m_min| | max |Δ m_max|")
        report_lines.append("---|---|---|---|---|---")
        for _, row in grouped.iterrows():
            report_lines.append(
                f"{row['resolution']:.0f} | {row['dk']:.4f} | {row['omega_delta']:.3e} | "
                f"{row['vg_delta']:.3e} | {row['mass_delta_min']:.3e} | {row['mass_delta_max']:.3e}"
            )
    else:
        report_lines.append("")
        report_lines.append("No successful MPB runs were recorded.")

    report_path = cdir / "phase0p5_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"    Wrote summary to {report_path}")


def _combination_metrics(df: pd.DataFrame) -> List[Dict[str, float]]:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return []

    combos: List[Dict[str, float]] = []
    grouped = ok.groupby(["resolution", "dk"])
    for (resolution, dk), subset in grouped:
        omega_abs = float(np.nanmax(np.abs(subset["omega_delta"])))
        vg_abs = float(np.nanmax(np.abs(subset["vg_delta"])))
        mass_abs = float(
            np.nanmax(
                np.abs(
                    np.vstack([subset["mass_delta_min"].to_numpy(), subset["mass_delta_max"].to_numpy()])
                )
            )
        )
        omega_ref = float(np.nanmax(np.abs(subset["omega_ref"])))
        vg_ref = float(np.nanmax(np.abs(subset["vg_ref"])))
        mass_ref = float(
            np.nanmax(
                np.abs(
                    np.vstack([subset["mass_ref_min"].to_numpy(), subset["mass_ref_max"].to_numpy()])
                )
            )
        )
        eps = 1e-12
        entry = {
            "resolution": float(resolution),
            "dk": float(dk),
            "omega_max_abs": omega_abs,
            "vg_max_abs": vg_abs,
            "mass_max_abs": mass_abs,
            "omega_rel_max": float(omega_abs / max(omega_ref, eps)),
            "vg_rel_max": float(vg_abs / max(vg_ref, eps)),
            "mass_rel_max": float(mass_abs / max(mass_ref, eps)),
        }
        entry["score"] = float(
            max(entry["omega_rel_max"], entry["vg_rel_max"], entry["mass_rel_max"])
        )
        combos.append(entry)
    combos.sort(key=lambda row: (row["score"], row["resolution"], row["dk"]))
    return combos


def _select_recommended_combination(
    combos: List[Dict[str, float]],
    ref_resolution: float,
    ref_dk: float,
    config: Dict,
) -> Dict[str, float] | None:
    if not combos:
        return None

    prefer_reference = bool(config.get("phase0p5_prefer_reference", True))
    if prefer_reference:
        for entry in combos:
            if np.isclose(entry["resolution"], ref_resolution) and np.isclose(entry["dk"], ref_dk):
                return entry
    return combos[0]


def _write_recommendation(
    cdir: Path,
    cid: int,
    combos: List[Dict[str, float]],
    selected: Dict[str, float],
    ref_resolution: float,
    ref_dk: float,
):
    payload = {
        "candidate_id": cid,
        "reference": {"resolution": ref_resolution, "dk": ref_dk},
        "selected": selected,
        "combination_metrics": combos,
        "selection_method": "relative-score",
    }
    out_path = cdir / "phase0p5_recommended.json"
    save_json(payload, out_path)
    print(
        "    Recommended MPB settings: resolution=%.0f, Δk=%.4f (score %.3e)"
        % (selected["resolution"], selected["dk"], selected["score"])
    )
    print(f"    Saved recommendation to {out_path}")


def process_candidate(row, config: Dict, run_dir: Path, overwrite: bool):
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    cdir.mkdir(parents=True, exist_ok=True)
    if not overwrite and (cdir / "phase0p5_convergence.csv").exists():
        print(f"  Candidate {cid}: convergence data already exists; skipping (set phase0p5_overwrite: true to recompute).")
        return

    params = extract_candidate_parameters(row)
    moire_meta = ensure_moire_metadata(params, config)
    L_moire = moire_meta["moire_length"]
    eta_physical = float(params["a"]) / float(L_moire)
    params["eta"] = eta_physical

    tau = np.asarray(config.get("tau", [0.0, 0.0]), dtype=float)
    registry_eta = _resolve_registry_eta(config, eta_physical)
    samples = _prepare_sample_points(L_moire, config.get("phase0p5_sample_points"))
    if not samples:
        print(f"  Candidate {cid}: no sampling points configured; skipping.")
        return
    samples = _attach_registry(samples, moire_meta["a1_vec"], moire_meta["a2_vec"], moire_meta["theta_rad"], tau, registry_eta)

    resolutions = sorted(set(float(r) for r in config.get("phase0p5_resolutions", [config.get("phase1_resolution", 24)])))
    dk_values = sorted(set(float(d) for d in config.get("phase0p5_dk_values", [config.get("phase1_dk", 0.005)])))
    if not resolutions or not dk_values:
        print(f"  Candidate {cid}: missing resolution or Δk list; skipping.")
        return

    mpb_template = {
        "num_bands": int(config.get("phase0p5_num_bands", config.get("num_bands", 8))),
        "quiet_mpb": bool(config.get("phase0p5_quiet_mpb", True)),
    }

    base_params = {
        "lattice_type": params["lattice_type"],
        "a": params["a"],
        "r_over_a": params["r_over_a"],
        "eps_bg": params["eps_bg"],
        "k0_x": params["k0_x"],
        "k0_y": params["k0_y"],
        "band_index": params["band_index"],
        "omega0": params["omega0"],
        "a1_vec": moire_meta["a1_vec"],
        "a2_vec": moire_meta["a2_vec"],
    }

    k0 = np.array([params["k0_x"], params["k0_y"], 0.0])
    band_idx = params["band_index"]

    rows = []
    for sample in samples:
        for resolution in resolutions:
            for dk in dk_values:
                cfg = dict(mpb_template)
                cfg["phase1_resolution"] = resolution
                cfg["phase1_dk"] = dk
                geom = build_bilayer_geometry_at_delta(base_params, sample["delta_frac"])
                try:
                    omega0, vg, M_inv, _stencil = compute_local_band_at_registry(geom, k0, band_idx, cfg)
                    eigvals = np.linalg.eigvalsh(M_inv)
                    status = "ok"
                    error_msg = ""
                except Exception as exc:  # pragma: no cover - MPB failures
                    omega0 = np.nan
                    vg = np.array([np.nan, np.nan])
                    M_inv = np.full((2, 2), np.nan)
                    eigvals = np.array([np.nan, np.nan])
                    status = "error"
                    error_msg = str(exc)
                rows.append(
                    {
                        "candidate_id": cid,
                        "sample_label": sample["label"],
                        "R_x": float(sample["R"][0]),
                        "R_y": float(sample["R"][1]),
                        "delta_u": float(sample["delta_frac"][0]),
                        "delta_v": float(sample["delta_frac"][1]),
                        "resolution": float(resolution),
                        "dk": float(dk),
                        "omega0": float(omega0),
                        "vg_x": float(vg[0]),
                        "vg_y": float(vg[1]),
                        "vg_norm": float(np.linalg.norm(vg)),
                        "mass_xx": float(M_inv[0, 0]),
                        "mass_xy": float(M_inv[0, 1]),
                        "mass_yy": float(M_inv[1, 1]),
                        "mass_min": float(eigvals.min()),
                        "mass_max": float(eigvals.max()),
                        "status": status,
                        "error_message": error_msg,
                    }
                )
    if not rows:
        print(f"  Candidate {cid}: no MPB rows computed.")
        return

    df = pd.DataFrame(rows)
    ref_resolution = float(config.get("phase0p5_reference_resolution", max(resolutions)))
    ref_dk = float(config.get("phase0p5_reference_dk", min(dk_values)))
    df, summary = _summaries(df, ref_resolution, ref_dk)
    _write_outputs(cdir, cid, df, ref_resolution, ref_dk, summary)

    combos = _combination_metrics(df)
    selected = _select_recommended_combination(combos, ref_resolution, ref_dk, config)
    if selected is not None:
        _write_recommendation(cdir, cid, combos, selected, ref_resolution, ref_dk)


def run_phase0p5(run_dir: str | Path, config_path: str | Path):
    print("\n" + "=" * 70)
    print("PHASE 0.5: MPB Convergence Scan")
    print("=" * 70)

    config = load_yaml(config_path)
    run_dir = resolve_run_dir(run_dir, config)
    print(f"Using run directory: {run_dir}")

    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"{candidates_path} not found; run Phase 0 before Phase 0.5.")
    frame = pd.read_csv(candidates_path)
    env_override = os.environ.get("MSL_PHASE0P5_CANDIDATE_ID")
    candidate_ids = _select_candidate_ids(frame, config, env_override)
    if not candidate_ids:
        print("No candidates selected for Phase 0.5.")
        return

    overwrite = bool(config.get("phase0p5_overwrite", False))
    for cid in candidate_ids:
        row = frame[frame["candidate_id"] == cid]
        if row.empty:
            print(f"  Candidate {cid} not present in CSV; skipping.")
            continue
        print(f"  Candidate {cid}: running convergence scan")
        process_candidate(row.iloc[0], config, run_dir, overwrite)

    print("Phase 0.5 completed.\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase0p5_convergence.py <run_dir|auto> <phase0p5_config.yaml>")
    run_phase0p5(sys.argv[1], sys.argv[2])
