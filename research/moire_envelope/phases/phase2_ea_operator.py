"""Phase 2: Assemble envelope-approximation operator on the moiré grid."""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Tuple, cast
import textwrap
from scipy.sparse import lil_matrix, save_npz, csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_yaml, load_json
from common.plotting import plot_phase2_fields


def _regularize_mass_tensor(M_inv: np.ndarray, min_eig: float | None) -> np.ndarray:
    """Floor |eigenvalues| of each 2×2 tensor to keep the EA operator elliptic."""
    if min_eig is None or min_eig <= 0:
        return M_inv
    tensors = M_inv.reshape(-1, 2, 2)
    adjusted = False
    for idx, tensor in enumerate(tensors):
        eigvals, eigvecs = np.linalg.eigh(tensor)
        mask = np.abs(eigvals) < min_eig
        if not np.any(mask):
            continue
        adjusted = True
        eigvals[mask] = np.sign(eigvals[mask]) * min_eig
        tensors[idx] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    if adjusted:
        return tensors.reshape(M_inv.shape)
    return M_inv


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


def _discover_phase1_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (candidate_id, path) pairs that already have Phase 1 data."""
    candidates: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase1_band_data.h5").exists():
            candidates.append((cid, cdir))
    return candidates


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    """Load per-candidate metadata from JSON or fallback to CSV row if needed."""
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
    """Extract grid spacing and spatial span from R-grid."""
    Nx, Ny, _ = R_grid.shape
    x_line = np.unique(R_grid[:, 0, 0])
    y_line = np.unique(R_grid[0, :, 1])
    dx = float(np.mean(np.diff(x_line))) if x_line.size > 1 else float("nan")
    dy = float(np.mean(np.diff(y_line))) if y_line.size > 1 else float("nan")
    x_min = float(R_grid[..., 0].min())
    x_max = float(R_grid[..., 0].max())
    y_min = float(R_grid[..., 1].min())
    y_max = float(R_grid[..., 1].max())
    area = (x_max - x_min) * (y_max - y_min)
    return {
        "Nx": Nx,
        "Ny": Ny,
        "dx": dx,
        "dy": dy,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "cell_area": float(area),
    }


def _operator_diagnostics(operator: csr_matrix) -> Dict[str, float]:
    """Compute quick diagnostics for the assembled sparse operator."""
    diag = operator.diagonal()
    diag_real = np.real(diag)
    abs_diag = np.abs(diag_real)
    abs_entries = operator.copy()
    abs_entries.data = np.abs(abs_entries.data)
    row_sums = np.asarray(abs_entries.sum(axis=1)).ravel()
    symmetry_defect = int((operator - operator.transpose()).nnz)
    n_rows, n_cols = cast(Tuple[int, int], operator.shape)
    return {
        "diag_min": float(diag_real.min()),
        "diag_max": float(diag_real.max()),
        "diag_mean": float(diag_real.mean()),
        "diag_std": float(diag_real.std()),
        "diag_abs_max": float(abs_diag.max()),
        "max_row_sum": float(row_sums.max()),
        "density": float(operator.nnz) / float(n_rows * n_cols),
        "nnz": int(operator.nnz),
        "symmetry_defect_nnz": symmetry_defect,
    }


def _write_phase2_report(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    field_stats: Dict,
    operator_stats: Dict[str, float],
    eta: float,
    include_plot: bool,
):
    """Emit a human-readable summary of Phase 2 results."""
    lines = [
        "# Phase 2 Envelope Operator Report",
        "",
        f"**Candidate**: {candidate_id:04d}",
        "",
        "## Discretization",
        f"- Grid: {grid_info['Nx']} × {grid_info['Ny']} points",
        f"- Δx = {grid_info['dx']:.5f}, Δy = {grid_info['dy']:.5f}",
        f"- Moiré window span: x ∈ [{grid_info['x_min']:.4f}, {grid_info['x_max']:.4f}], "
        f"y ∈ [{grid_info['y_min']:.4f}, {grid_info['y_max']:.4f}]",
        f"- η (small twist parameter) = {eta:.6f}",
        "",
        "## Input Field Diagnostics",
        f"- Potential V(R): min {field_stats['V_min']:.6f}, max {field_stats['V_max']:.6f}, "
        f"mean {field_stats['V_mean']:.6f}",
        f"- M⁻¹ eigenvalues: min {field_stats['M_inv_min_eig']:.4f}, max {field_stats['M_inv_max_eig']:.4f}",
        "",
        "## Operator Diagnostics",
        f"- Matrix size: {grid_info['Nx'] * grid_info['Ny']} × {grid_info['Nx'] * grid_info['Ny']}",
        f"- Non-zeros: {operator_stats['nnz']} ({operator_stats['density']*100:.3f}% density)",
        f"- Diagonal range: [{operator_stats['diag_min']:.6f}, {operator_stats['diag_max']:.6f}]",
        f"- Diagonal std dev: {operator_stats['diag_std']:.6f}",
        f"- Max absolute row sum: {operator_stats['max_row_sum']:.6f}",
        f"- Symmetry defect nnz: {operator_stats['symmetry_defect_nnz']} (0 means perfectly symmetric)",
    ]

    lines.append("")
    if include_plot:
        lines.append("- Field visualization was requested via config and saved alongside this report.")
    else:
        lines.append("- Field visualization skipped (config default). Report captures key diagnostics instead.")

    report_path = Path(cdir) / "phase2_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"    Wrote {report_path}")
def assemble_ea_operator(
    R_grid: np.ndarray,
    mass_tensor: np.ndarray,
    V: np.ndarray,

    eta: float,
    include_cross_terms: bool = False,
    bloch_k: Tuple[float, float] | None = None,
) -> csr_matrix:
    """Assemble the EA Hamiltonian using finite differences with (Bloch) periodic BCs."""
    Nx, Ny, _ = R_grid.shape
    total_points = Nx * Ny
    dx_physical = float(R_grid[1, 0, 0] - R_grid[0, 0, 0]) if Nx > 1 else None
    dy_physical = float(R_grid[0, 1, 1] - R_grid[0, 0, 1]) if Ny > 1 else None

    if dx_physical is None or abs(dx_physical) < 1e-12 or dy_physical is None or abs(dy_physical) < 1e-12:
        raise ValueError("Phase 2 requires Nx, Ny > 1 with non-zero grid spacing")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Phase 2 requires a positive finite eta value")

    # R_grid is sampled in physical moiré coordinates (|R| ~ L_moiré), but the
    # envelope PDE is derived in the slow coordinate R_slow = eta * R_phys.
    # Convert spacings so the discrete Laplacian acts on the slow variable.
    dx = dx_physical * eta
    dy = dy_physical * eta

    H = lil_matrix((total_points, total_points), dtype=np.complex128)
    eta_sq = eta ** 2

    # Lattice periods in the slow coordinate (length traversed when wrapping edges).
    period_x = dx * (Nx - 1)
    period_y = dy * (Ny - 1)

    bloch_k = bloch_k or (0.0, 0.0)
    phase_x = np.exp(1j * bloch_k[0] * period_x)
    phase_y = np.exp(1j * bloch_k[1] * period_y)
    phase_x_conj = np.conjugate(phase_x)
    phase_y_conj = np.conjugate(phase_y)

    def idx(i: int, j: int) -> int:
        return (i % Nx) * Ny + (j % Ny)

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            H[p, p] += V[i, j]

            mx_plus = 0.5 * (mass_tensor[i, j, 0, 0] + mass_tensor[(i + 1) % Nx, j, 0, 0])
            coeff = -0.5 * eta_sq * mx_plus / (dx ** 2)
            wrap_forward_x = phase_x if i == Nx - 1 else 1.0
            H[p, idx(i + 1, j)] += coeff * wrap_forward_x
            H[p, p] -= coeff

            mx_minus = 0.5 * (mass_tensor[i, j, 0, 0] + mass_tensor[(i - 1) % Nx, j, 0, 0])
            coeff = -0.5 * eta_sq * mx_minus / (dx ** 2)
            wrap_backward_x = phase_x_conj if i == 0 else 1.0
            H[p, idx(i - 1, j)] += coeff * wrap_backward_x
            H[p, p] -= coeff

            my_plus = 0.5 * (mass_tensor[i, j, 1, 1] + mass_tensor[i, (j + 1) % Ny, 1, 1])
            coeff = -0.5 * eta_sq * my_plus / (dy ** 2)
            wrap_forward_y = phase_y if j == Ny - 1 else 1.0
            H[p, idx(i, j + 1)] += coeff * wrap_forward_y
            H[p, p] -= coeff

            my_minus = 0.5 * (mass_tensor[i, j, 1, 1] + mass_tensor[i, (j - 1) % Ny, 1, 1])
            coeff = -0.5 * eta_sq * my_minus / (dy ** 2)
            wrap_backward_y = phase_y_conj if j == 0 else 1.0
            H[p, idx(i, j - 1)] += coeff * wrap_backward_y
            H[p, p] -= coeff

            if include_cross_terms:
                raise NotImplementedError("Cross-derivative coupling not yet implemented")

    return csr_matrix(H)


def summarize_phase2_fields(
    V: np.ndarray,
    mass_tensor: np.ndarray,
    operator: csr_matrix,
    candidate_id: int,
    grid_info: Dict[str, float],
    eta: float,
) -> Dict:
    eigvals = np.linalg.eigvalsh(mass_tensor.reshape(-1, 2, 2))
    stats = {
        "candidate_id": candidate_id,
        "Nx": grid_info["Nx"],
        "Ny": grid_info["Ny"],
        "dx": grid_info["dx"],
        "dy": grid_info["dy"],
        "eta": eta,
        "nnz": int(operator.nnz),
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "V_mean": float(V.mean()),
        "M_inv_min_eig": float(eigvals.min()),
        "M_inv_max_eig": float(eigvals.max()),
        "M_inv_mean_eig": float(eigvals.mean()),
    }
    return stats


def process_candidate(
    row,
    config: Dict,
    run_dir: Path,
    include_cross_terms: bool,
    plot_fields: bool,
):
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data for candidate {cid}: {h5_path}")

    print(f"  Candidate {cid}: loading Phase 1 data from {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        R_grid = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        eta_attr = hf.attrs.get("eta")

    if eta_attr is not None:
        eta = float(eta_attr)
    else:
        eta_cfg = config.get("eta")
        if isinstance(eta_cfg, str) and eta_cfg.lower() == "auto":
            eta_cfg = None
        if eta_cfg is not None:
            eta = float(eta_cfg)
        else:
            a = float(row.get("a", np.nan))
            L_m = float(row.get("moire_length", np.nan))
            if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
                eta = a / L_m
            else:
                print("    WARNING: Missing eta in Phase 1 attrs and config; defaulting to 1.0")
                eta = 1.0

    grid_info = _grid_metrics(R_grid)
    min_mass_eig = config.get("phase2_min_mass_eig")
    mass_tensor = _regularize_mass_tensor(M_inv, min_mass_eig)
    operator = assemble_ea_operator(R_grid, mass_tensor, V, eta, include_cross_terms)

    op_path = cdir / "phase2_operator.npz"
    save_npz(op_path, operator)
    print(f"    Saved operator to {op_path}")

    stats = summarize_phase2_fields(V, mass_tensor, operator, cid, grid_info, eta)
    info_path = cdir / "phase2_operator_info.csv"
    pd.DataFrame([stats]).to_csv(info_path, index=False)
    print(f"    Wrote stats to {info_path}")

    operator_stats = _operator_diagnostics(operator)
    _write_phase2_report(cdir, cid, grid_info, stats, operator_stats, eta, plot_fields)

    if plot_fields:
        plot_phase2_fields(cdir, R_grid, V, M_inv)
        print("    Rendered phase2_fields_visualization.png")
    else:
        print("    Skipped redundant field visualization (set phase2_plot_fields: true to enable).")


def run_phase2(run_dir: str | Path, config_path: str | Path):
    print("\n" + "=" * 70)
    print("PHASE 2: Envelope Operator Assembly")
    print("=" * 70)

    config = load_yaml(config_path)
    include_cross_terms = bool(config.get("phase2_include_cross_terms", False))
    plot_fields = bool(config.get("phase2_plot_fields", False))
    if include_cross_terms:
        print("  WARNING: phase2_include_cross_terms=True is not yet implemented; falling back to axis-aligned terms.")
        include_cross_terms = False

    run_dir = resolve_run_dir(run_dir, config)
    print(f"Using run directory: {run_dir}")

    candidates_path = Path(run_dir) / "phase0_candidates.csv"
    candidate_frame = None
    if candidates_path.exists():
        candidate_frame = pd.read_csv(candidates_path)
    else:
        print(f"WARNING: {candidates_path} not found; relying solely on per-candidate metadata.")

    discovered = _discover_phase1_candidates(run_dir)
    if not discovered:
        raise FileNotFoundError(
            f"No candidate_* directories with Phase 1 data found in {run_dir}. "
            "Run Phase 1 before Phase 2."
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
            process_candidate(row, config, run_dir, include_cross_terms, plot_fields)
        except FileNotFoundError as exc:
            print(f"  Skipping candidate {cid}: {exc}")

    print("Phase 2 completed.\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase2_ea_operator.py <run_dir|auto> <phase2_config.yaml>")
    run_phase2(sys.argv[1], sys.argv[2])
