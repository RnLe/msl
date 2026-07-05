"""Phase 3: Solve envelope-approximation eigenmodes for each candidate."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Tuple, cast

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal envs
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh, eigs

from common.io_utils import candidate_dir, load_yaml, load_json
from common.plotting import plot_envelope_modes
from common.scoring import participation_ratio, field_entropy, localization_length


def resolve_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        runs_base = Path(config.get("output_dir", "runs"))
        phase0_runs = sorted(runs_base.glob("phase0_real_run_*"))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_real_run_* directories found in {runs_base}")
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest Phase 0 run: {run_dir}")
    return run_dir


def _discover_phase2_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Collect candidates that already have Phase 2 operators."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase2_operator.npz").exists():
            discovered.append((cid, cdir))
    return discovered


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    """Load per-candidate metadata from JSON or fallback to CSV row if available."""
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
        matches = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not matches.empty:
            return matches.iloc[0].to_dict()
    return {"candidate_id": candidate_id}


def _grid_metrics(R_grid: np.ndarray) -> Dict[str, float]:
    Nx, Ny, _ = R_grid.shape
    x_vals = R_grid[..., 0]
    y_vals = R_grid[..., 1]
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    x_line = np.unique(R_grid[:, 0, 0])
    y_line = np.unique(R_grid[0, :, 1])
    dx = float(np.mean(np.diff(x_line))) if x_line.size > 1 else float("nan")
    dy = float(np.mean(np.diff(y_line))) if y_line.size > 1 else float("nan")
    return {
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "cell_area": float((x_max - x_min) * (y_max - y_min)),
    }


def _load_phase1_metadata(cdir: Path) -> Tuple[np.ndarray, float, float]:
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    override_grid = cdir / "phase2_R_grid.npy"
    R_grid_override = None
    if override_grid.exists():
        try:
            R_grid_override = np.load(override_grid)
            print(f"    Using supercell grid from {override_grid.name}")
        except Exception as exc:
            print(f"    WARNING: Failed to load {override_grid}: {exc}; falling back to Phase 1 grid")
    with h5py.File(h5_path, "r") as hf:
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))
        eta = float(hf.attrs.get("eta", np.nan))
    if R_grid_override is not None:
        R_grid = R_grid_override
    return R_grid, omega_ref, eta


def _is_hermitian(op, abs_tol: float = 1e-10, rel_tol: float = 1e-9) -> bool:
    diff = op - op.getH()
    if diff.nnz == 0:
        return True
    max_abs = float(np.max(np.abs(diff.data)))
    if not np.isfinite(max_abs):
        return False
    norm = float(np.max(np.abs(op.data))) if op.nnz else 0.0
    threshold = max(abs_tol, rel_tol * max(norm, 1e-16))
    return max_abs <= threshold


def _solve_eigenpairs(
    operator,
    n_modes: int,
    hermitian: bool,
    solver_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    total_dim = operator.shape[0]
    if total_dim < 2:
        raise ValueError("Operator must be at least 2×2 for Phase 3")
    k = max(1, min(n_modes, total_dim - 1))

    tol = float(solver_cfg.get("phase3_solver_tol", 1e-8))
    maxiter = int(solver_cfg.get("phase3_solver_maxiter", 1000))
    use_shift = bool(solver_cfg.get("phase3_use_shift_invert", False))
    sigma_cfg = solver_cfg.get("phase3_solver_sigma")
    solver_name = "eigsh" if hermitian else "eigs"
    shift_value = None

    if hermitian:
        which = solver_cfg.get("phase3_solver_which", "SA")
        try:
            if use_shift:
                if sigma_cfg is not None:
                    shift_value = float(sigma_cfg)
                else:
                    diag = operator.diagonal()
                    shift_value = float(np.median(diag))
                eigvals, eigvecs = eigsh(
                    operator,
                    k=k,
                    sigma=shift_value,
                    which="LM",
                    tol=tol,  # type: ignore[arg-type]
                    maxiter=maxiter,
                )
            else:
                eigvals, eigvecs = eigsh(
                    operator,
                    k=k,
                    which=which,
                    tol=tol,  # type: ignore[arg-type]
                    maxiter=maxiter,
                )
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"    eigsh failed ({exc}); retrying with default settings.")
            eigvals, eigvecs = eigsh(operator, k=k, which="SA")
    else:
        which = solver_cfg.get("phase3_solver_which", "SR")
        try:
            eigvals_raw = eigs(
                operator,
                k=k,
                sigma=float(sigma_cfg) if sigma_cfg is not None else None,
                which=which,
                tol=tol,  # type: ignore[arg-type]
                maxiter=maxiter,
            )
            eigvals, eigvecs = cast(Tuple[np.ndarray, np.ndarray], eigvals_raw)
        except Exception as exc:
            print(f"    eigs failed ({exc}); retrying without shift.")
            eigvals_raw = eigs(operator, k=k, which="SR")
            eigvals, eigvecs = cast(Tuple[np.ndarray, np.ndarray], eigvals_raw)

    if np.iscomplexobj(eigvals):
        order = np.argsort(eigvals.real)
    else:
        order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    solver_details = {
        "requested_modes": float(n_modes),
        "solved_modes": float(k),
        "solver_name": solver_name,
        "shift_value": float(shift_value) if shift_value is not None else np.nan,
        "used_shift": bool(use_shift),
        "which": which,
        "tol": tol,
        "maxiter": float(maxiter),
    }
    return eigvals, eigvecs, solver_details


def _analyze_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    R_grid: np.ndarray,
    omega_ref: float,
) -> Tuple[np.ndarray, Iterable[Dict[str, float]]]:
    Nx, Ny, _ = R_grid.shape
    n_modes = eigenvectors.shape[1]
    fields = eigenvectors.T.reshape((n_modes, Nx, Ny))
    rows = []
    for idx in range(n_modes):
        field = fields[idx]
        abs_field = np.abs(field)
        abs2 = abs_field ** 2
        norm_l2 = float(np.linalg.norm(field.ravel()))
        delta_omega = eigenvalues[idx]
        delta_real = float(delta_omega.real if np.iscomplexobj(delta_omega) else delta_omega)
        omega_cav = float(omega_ref + delta_real)
        pr = float(participation_ratio(field))
        entropy = float(field_entropy(field))
        loc_len = float(localization_length(field, R_grid))
        weight = abs2 / max(abs2.sum(), 1e-20)
        center = np.tensordot(weight, R_grid, axes=([0, 1], [0, 1]))
        peak_idx = np.unravel_index(np.argmax(abs2), abs2.shape)
        peak_pos = R_grid[peak_idx]
        rows.append(
            {
                "mode_index": idx,
                "delta_omega": delta_real,
                "omega_cavity": omega_cav,
                "norm_l2": norm_l2,
                "participation_ratio": pr,
                "entropy": entropy,
                "localization_length": loc_len,
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "peak_x": float(peak_pos[0]),
                "peak_y": float(peak_pos[1]),
                "max_amplitude": float(abs_field.max()),
            }
        )
    return fields, rows


def _write_phase3_report(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    eigenvalues: np.ndarray,
    omega_ref: float,
    eta: float,
    hermitian: bool,
    solver_details: Dict[str, float],
    mode_rows: Iterable[Dict[str, float]],
):
    report_path = cdir / "phase3_report.md"
    rows = list(mode_rows)
    lines = [
        "# Phase 3 Envelope Solver Report",
        "",
        f"**Candidate**: {candidate_id:04d}",
        "",
        "## Discretization",
        f"- Grid: {grid_info['Nx']} × {grid_info['Ny']} points",
        f"- Δx = {grid_info['dx']:.5f}, Δy = {grid_info['dy']:.5f}",
        f"- Window: x ∈ [{grid_info['x_min']:.4f}, {grid_info['x_max']:.4f}], y ∈ [{grid_info['y_min']:.4f}, {grid_info['y_max']:.4f}]",
        f"- η (small twist parameter) = {eta:.6f}",
        "",
        "## Solver",
        f"- Operator Hermitian: {hermitian}",
        f"- Solver: {solver_details['solver_name']} (which={solver_details['which']})",
        f"- Modes requested: {int(solver_details['requested_modes'])}, solved: {int(solver_details['solved_modes'])}",
        f"- Shift-invert: {'yes' if solver_details['used_shift'] else 'no'}",
    ]
    if solver_details['used_shift']:
        lines.append(f"  - σ = {solver_details['shift_value']:.6e}")
    lines.extend(
        [
            f"- Tolerance: {solver_details['tol']:.2e}",
            f"- Max iterations: {int(solver_details['maxiter'])}",
            f"- ω_ref = {omega_ref:.6f}",
            "",
            "## Lowest Eigenvalues",
        ]
    )
    for row in rows[: min(5, len(rows))]:
        lines.append(
            f"- Mode {int(row['mode_index'])}: Δω = {row['delta_omega']:.6e}, "
            f"ω = {row['omega_cavity']:.6e}, PR = {row['participation_ratio']:.3f}, ξ = {row['localization_length']:.3f}"
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"    Wrote {report_path}")


def process_candidate(row, run_dir: Path, config: Dict):
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    operator_path = cdir / "phase2_operator.npz"
    if not operator_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 operator for candidate {cid}: {operator_path}")

    H = load_npz(operator_path).tocsr()
    R_grid, omega_ref, eta = _load_phase1_metadata(cdir)
    if not np.isfinite(omega_ref):
        omega_ref = float(row.get("omega0", 0.0))
    if not np.isfinite(eta):
        a = float(row.get("a", np.nan))
        L_m = float(row.get("moire_length", np.nan))
        if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
            eta = a / L_m

    grid_info = _grid_metrics(R_grid)

    hermitian = _is_hermitian(H)
    n_modes = int(config.get("ea_n_modes", 8))
    eigenvalues, eigenvectors, solver_details = _solve_eigenpairs(H, n_modes, hermitian, config)
    if np.iscomplexobj(eigenvalues):
        max_imag = float(np.max(np.abs(eigenvalues.imag)))
        if max_imag > 1e-8:
            print(f"    WARNING: eigenvalues have imaginary component up to {max_imag:.2e}")
    fields, mode_rows = _analyze_modes(eigenvalues, eigenvectors, R_grid, omega_ref)

    # Persist eigenstates
    eig_h5 = cdir / "phase3_eigenstates.h5"
    with h5py.File(eig_h5, "w") as hf:
        hf.create_dataset("F", data=fields, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")
        hf.create_dataset("eigenvalues", data=eigenvalues)
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
    print(f"    Saved eigenstates to {eig_h5}")

    # Tabulate metrics
    df = pd.DataFrame(mode_rows)
    df.insert(0, "candidate_id", cid)
    df_path = cdir / "phase3_eigenvalues.csv"
    df.to_csv(df_path, index=False)
    print(f"    Wrote eigenvalue table to {df_path}")

    eigenvalues_for_plots = eigenvalues.real if np.iscomplexobj(eigenvalues) else eigenvalues
    plot_envelope_modes(
        cdir,
        R_grid,
        fields,
        eigenvalues_for_plots,
        n_modes=n_modes,
        candidate_params=row,
    )
    _write_phase3_report(cdir, cid, grid_info, eigenvalues_for_plots, omega_ref, eta, hermitian, solver_details, mode_rows)


def run_phase3(run_dir: str | Path, config_path: str | Path):
    print("\n" + "=" * 70)
    print("PHASE 3: Envelope Eigenvalue Solver")
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

    discovered = _discover_phase2_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 2 operators found in {run_dir}. "
            "Run Phase 2 before Phase 3."
        )

    K = config.get("K_candidates")
    if isinstance(K, int) and K > 0:
        discovered = discovered[:K]
        print(f"Processing {len(discovered)} candidate directories (limited to K={K}).")
    else:
        print(f"Processing {len(discovered)} candidate directories (all Phase 2 outputs found).")

    candidate_iter = list(discovered)
    iterator = tqdm(candidate_iter, desc="Phase 3 Progress", unit="candidate", ncols=80)
    for cid, _ in iterator:
        iterator.set_postfix_str(f"CID {cid}")
        row = _load_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            print(f"  Candidate {cid}: solving EA modes")
            process_candidate(row, run_dir, config)
        except FileNotFoundError as exc:
            print(f"    Skipping candidate {cid}: {exc}")

    print("Phase 3 completed.\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase3_ea_solver.py <run_dir|auto> <phase3_config.yaml>")
    run_phase3(sys.argv[1], sys.argv[2])
