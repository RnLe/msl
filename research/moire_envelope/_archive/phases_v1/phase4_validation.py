"""Phase 4: Validate EA minibands by sampling Bloch boundary conditions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, cast

import h5py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable

PROJ_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json
from common.plotting import (
    plot_phase4_bandstructure,
    plot_phase4_mode_profiles,
    plot_phase4_mode_differences,
)
from phases.phase2_ea_operator import (
    assemble_ea_operator,
    _regularize_mass_tensor,
    resolve_run_dir,
)


def _discover_phase1_candidates(run_dir: Path) -> List[Tuple[int, Path]]:
    """Return candidate directories that already contain Phase 1 data."""
    cands: List[Tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase1_band_data.h5").exists():
            cands.append((cid, cdir))
    return cands


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception as exc:
            print(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
    return {"candidate_id": candidate_id}


def _load_phase1_data(cdir: Path):
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        vg = np.asarray(cast(h5py.Dataset, hf["vg"])[:]) if "vg" in hf else None
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))
        eta = float(hf.attrs.get("eta", np.nan))
        lattice_type = hf.attrs.get("lattice_type", "square")
    return R_grid, V, M_inv, vg, omega_ref, eta, lattice_type


def _slow_periods(R_grid: np.ndarray, eta: float) -> Tuple[float, float]:
    Nx, Ny, _ = R_grid.shape
    dx_phys = float(R_grid[1, 0, 0] - R_grid[0, 0, 0]) if Nx > 1 else 0.0
    dy_phys = float(R_grid[0, 1, 1] - R_grid[0, 0, 1]) if Ny > 1 else 0.0
    Lx = dx_phys * (Nx - 1) * eta
    Ly = dy_phys * (Ny - 1) * eta
    return Lx, Ly


def _default_path(lattice: str, Lx: float, Ly: float) -> List[Tuple[str, np.ndarray]]:
    eps = 1e-12
    kx = math.pi / max(Lx, eps)
    ky = math.pi / max(Ly, eps)
    if lattice.lower().startswith("hex"):
        # Approximate hex BZ using rectangular surrogate; users can override in config.
        return [
            ("Γ", np.array([0.0, 0.0])),
            ("K", np.array([2.0 * kx / 3.0, 2.0 * ky / 3.0])),
            ("M", np.array([kx, 0.0])),
            ("Γ", np.array([0.0, 0.0])),
        ]
    return [
        ("Γ", np.array([0.0, 0.0])),
        ("X", np.array([kx, 0.0])),
        ("M", np.array([kx, ky])),
        ("Γ", np.array([0.0, 0.0])),
    ]


def _parse_custom_points(cfg_points: Sequence[Dict[str, Iterable[float]]]) -> List[Tuple[str, np.ndarray]]:
    points = []
    for entry in cfg_points:
        label = entry.get("label", "?")
        coords = np.array(entry.get("coords", [0.0, 0.0]), dtype=float)
        points.append((label, coords))
    return points


def _sample_path(points: Sequence[Tuple[str, np.ndarray]], samples_per_segment: int):
    if len(points) < 2:
        raise ValueError("Need at least two high-symmetry points for a path")
    k_list = []
    d_list = []
    seg_labels = []
    ticks = []
    total = 0.0
    for seg_idx in range(len(points) - 1):
        (label_a, k_a) = points[seg_idx]
        (_, k_b) = points[seg_idx + 1]
        delta = k_b - k_a
        ticks.append((label_a, total))
        for s in range(samples_per_segment):
            frac = s / samples_per_segment
            k = k_a + frac * delta
            k_list.append(k)
            d_list.append(total + frac * np.linalg.norm(delta))
            seg_labels.append(label_a)
        total += np.linalg.norm(delta)
    # append final point
    label_last, k_last = points[-1]
    k_list.append(k_last)
    d_list.append(total)
    seg_labels.append(label_last)
    ticks.append((label_last, total))
    return np.stack(k_list), np.array(d_list), seg_labels, ticks


def _solve_bloch_modes(
    operator,
    n_modes: int,
    tol: float,
    maxiter: int,
    *,
    return_eigenvectors: bool = False,
):
    dim = operator.shape[0]
    k = max(1, min(n_modes, dim - 2))
    if return_eigenvectors:
        eigvals, eigvecs = eigsh(
            operator,
            k=k,
            which="SA",
            tol=tol,  # type: ignore[arg-type]
            maxiter=maxiter,
            return_eigenvectors=True,
        )
        order = np.argsort(eigvals)
        eigvals = np.real(eigvals[order])
        eigvecs = eigvecs[:, order]
        return eigvals, eigvecs

    eigvals = eigsh(
        operator,
        k=k,
        which="SA",
        tol=tol,  # type: ignore[arg-type]
        maxiter=maxiter,
        return_eigenvectors=False,
    )
    return np.real(np.sort(eigvals))


def process_candidate(row, config: Dict, run_dir: Path):
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    R_grid, V, M_inv, vg_field, omega_ref, eta, lattice_type = _load_phase1_data(cdir)
    if not np.isfinite(eta) or eta <= 0:
        a = float(row.get("a", np.nan))
        L_m = float(row.get("moire_length", np.nan))
        if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
            eta = a / L_m
        else:
            raise ValueError(f"Candidate {cid} missing eta in Phase 1 attrs and Phase 0 row")

    mass_floor = config.get("phase4_min_mass_eig", config.get("phase2_min_mass_eig"))
    mass_tensor = _regularize_mass_tensor(M_inv, mass_floor)

    Lx, Ly = _slow_periods(R_grid, eta)
    custom_points = config.get("phase4_points")
    if custom_points:
        high_symm = _parse_custom_points(custom_points)
    else:
        high_symm = _default_path(str(lattice_type), Lx, Ly)
    samples = int(config.get("phase4_samples_per_segment", 24))
    k_path, distances, segments, ticks = _sample_path(high_symm, samples)

    modes = int(config.get("phase4_n_modes", 6))
    tol = float(config.get("phase4_solver_tol", 1e-8))
    maxiter = int(config.get("phase4_solver_maxiter", 2000))
    include_vg_term = bool(config.get("phase4_include_vg_term", True))
    vg_scale = float(config.get("phase4_vg_scale", 1.0))
    fd_order = int(config.get("phase4_fd_order", config.get("phase2_fd_order", 4)))
    use_vg_term = include_vg_term and vg_field is not None
    if include_vg_term and vg_field is None:
        print("    WARNING: Phase 1 dataset missing v_g; Bloch operator will omit drift term.")

    plot_modes = int(config.get("phase4_mode_plot_count", 8))
    Nx, Ny, _ = R_grid.shape
    gamma_fields = None
    gamma_eigvals = None

    band_values = np.zeros((k_path.shape[0], modes))
    records = []
    k_iterator = enumerate(k_path)
    k_iterator = tqdm(
        list(k_iterator),
        desc=f"    Bloch samples (candidate {cid})",
        leave=False,
    )
    for idx, k_vec in k_iterator:
        operator = assemble_ea_operator(
            R_grid,
            mass_tensor,
            V,
            eta,
            include_cross_terms=False,
            bloch_k=(float(k_vec[0]), float(k_vec[1])),
            vg_field=vg_field,
            include_vg_term=use_vg_term,
            vg_scale=vg_scale,
            fd_order=fd_order,
        )
        capture_vectors = plot_modes > 0 and idx == 0
        if capture_vectors:
            eigvals, eigvecs = _solve_bloch_modes(
                operator,
                modes,
                tol,
                maxiter,
                return_eigenvectors=True,
            )
            gamma_eigvals = eigvals
            gamma_fields = eigvecs.T.reshape(eigvecs.shape[1], Nx, Ny)
        else:
            eigvals = _solve_bloch_modes(operator, modes, tol, maxiter)
        band_values[idx, : len(eigvals)] = eigvals[: len(eigvals)]
        for mode_idx, d_omega in enumerate(eigvals):
            records.append(
                {
                    "candidate_id": cid,
                    "k_index": idx,
                    "segment": segments[idx],
                    "path_coordinate": distances[idx],
                    "kx": float(k_vec[0]),
                    "ky": float(k_vec[1]),
                    "mode_index": mode_idx,
                    "delta_omega": float(d_omega),
                    "omega_cavity": float(omega_ref + d_omega),
                }
            )

    bands_df = pd.DataFrame(records)
    bands_df.to_csv(cdir / "phase4_bandstructure.csv", index=False)

    tick_labels = [(label, pos) for label, pos in ticks]
    plot_phase4_bandstructure(cdir, distances, band_values, tick_labels)

    if plot_modes > 0 and gamma_fields is not None and gamma_eigvals is not None:
        plot_phase4_mode_profiles(
            cdir,
            R_grid,
            gamma_fields,
            gamma_eigvals,
            n_modes=plot_modes,
            candidate_params=row,
        )

        gamma_h5 = cdir / "phase4_gamma_modes.h5"
        with h5py.File(gamma_h5, "w") as hf:
            hf.create_dataset("F_gamma", data=gamma_fields, compression="gzip")
            hf.create_dataset("R_grid", data=R_grid, compression="gzip")
            hf.create_dataset("eigenvalues", data=gamma_eigvals)
            hf.attrs["omega_ref"] = omega_ref
            hf.attrs["eta"] = eta
            hf.attrs["k_point"] = "Gamma"
        print(f"    Saved Gamma-point modes to {gamma_h5}")

        phase3_h5 = cdir / "phase3_eigenstates.h5"
        if phase3_h5.exists():
            with h5py.File(phase3_h5, "r") as hf3:
                phase3_fields = np.asarray(hf3["F"][:])
                R_grid_p3 = np.asarray(hf3["R_grid"][:])
            if phase3_fields.shape[1:] == gamma_fields.shape[1:] and np.allclose(R_grid_p3, R_grid):
                n_diff = min(plot_modes, phase3_fields.shape[0], gamma_fields.shape[0])
                diff_fields = []
                for mode_idx in range(n_diff):
                    p4 = np.abs(gamma_fields[mode_idx]) ** 2
                    p3 = np.abs(phase3_fields[mode_idx]) ** 2
                    s4 = float(p4.sum()) or 1.0
                    s3 = float(p3.sum()) or 1.0
                    diff_fields.append(p4 / s4 - p3 / s3)
                if diff_fields:
                    plot_phase4_mode_differences(
                        cdir,
                        R_grid,
                        np.asarray(diff_fields),
                        gamma_eigvals,
                        n_modes=n_diff,
                        candidate_params=row,
                    )
            else:
                print("    WARNING: Phase 3/4 grids mismatch; skipping difference plot.")

    summary = _build_summary(cdir, cid, band_values, distances)
    pd.DataFrame([summary]).to_csv(cdir / "phase4_validation_summary.csv", index=False)

    print(f"    Phase 4 bands written for candidate {cid}")


def _build_summary(cdir: Path, cid: int, band_values: np.ndarray, distances: np.ndarray) -> Dict[str, float]:
    phase3_csv = cdir / "phase3_eigenvalues.csv"
    gamma_vals = band_values[0]
    if phase3_csv.exists():
        phase3 = pd.read_csv(phase3_csv)
        compare = phase3.sort_values("mode_index")
        n = min(len(compare), len(gamma_vals))
        diff = gamma_vals[:n] - (compare["delta_omega"].values[:n])
        max_abs = float(np.max(np.abs(diff))) if diff.size else float("nan")
        rms = float(np.sqrt(np.mean(diff ** 2))) if diff.size else float("nan")
    else:
        max_abs = float("nan")
        rms = float("nan")
    bandwidth = float(band_values[:, 0].max() - band_values[:, 0].min())
    return {
        "candidate_id": cid,
        "n_k_points": band_values.shape[0],
        "n_modes": band_values.shape[1],
        "gamma_max_abs_diff": max_abs,
        "gamma_rms_diff": rms,
        "mode0_bandwidth": bandwidth,
        "path_length": float(distances[-1] if len(distances) else 0.0),
    }


def run_phase4(run_dir: str | Path, config_path: str | Path):
    print("\n" + "=" * 70)
    print("PHASE 4: Bloch Validation of EA Modes")
    print("=" * 70)

    config = load_yaml(config_path)
    run_dir = resolve_run_dir(run_dir, config)
    print(f"Using run directory: {run_dir}")

    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    else:
        print(f"WARNING: {candidates_path} not found; relying on per-candidate metadata only.")

    discovered = _discover_phase1_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 1 data found in {run_dir}. "
            "Run Phase 1 before Phase 4."
        )

    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        discovered = discovered[:K]
        print(f"Processing {len(discovered)} candidate directories (limited to K={K}).")
    else:
        print(f"Processing {len(discovered)} candidate directories (all Phase 1 outputs found).")

    for cid, _ in discovered:
        row = _load_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            print(f"  Candidate {cid}: sampling Bloch path")
            process_candidate(row, config, run_dir)
        except FileNotFoundError as exc:
            print(f"    Skipping candidate {cid}: {exc}")

    print("Phase 4 completed.\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase4_validation.py <run_dir|auto> <phase4_config.yaml>")
    run_phase4(sys.argv[1], sys.argv[2])
