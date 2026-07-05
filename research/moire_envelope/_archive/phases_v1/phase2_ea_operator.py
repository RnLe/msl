"""Phase 2: Assemble envelope-approximation operator on the moiré grid."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, cast

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io_utils import candidate_dir, load_json, load_yaml, save_json
from common.plotting import plot_phase2_fields


def _regularize_mass_tensor(M_inv: np.ndarray, min_eig: float | None) -> np.ndarray:
    """Clamp tiny curvature eigenvalues to keep the operator elliptic."""
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
    return tensors.reshape(M_inv.shape) if adjusted else M_inv


def resolve_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        base = Path(config.get("output_dir", "runs"))
        phase0_runs = sorted(base.glob("phase0_real_run_*"))
        if not phase0_runs:
            raise FileNotFoundError(f"No phase0_real_run_* directories found in {base}")
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runs")) / run_dir
        if candidate.exists():
            run_dir = candidate
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist")
    return run_dir


def _discover_phase1_candidates(run_dir: Path) -> list[tuple[int, Path]]:
    """Return candidate directories that already contain Phase 1 data."""
    discovered: list[tuple[int, Path]] = []
    for cdir in sorted(run_dir.glob("candidate_*")):
        try:
            cid = int(cdir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if (cdir / "phase1_band_data.h5").exists():
            discovered.append((cid, cdir))
    return discovered


def _load_candidate_metadata(
    run_dir: Path,
    candidate_id: int,
    candidate_frame: pd.DataFrame | None,
) -> Dict:
    """Load per-candidate metadata JSON, falling back to CSV if necessary."""
    cdir = candidate_dir(run_dir, candidate_id)
    meta_path = cdir / "phase0_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
            meta.setdefault("candidate_id", candidate_id)
            return meta
        except Exception as exc:  # pragma: no cover - defensive parsing
            print(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
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
    abs_entries = operator.copy()
    abs_entries.data = np.abs(abs_entries.data)
    row_sums = np.asarray(abs_entries.sum(axis=1)).ravel()
    hermitian_gap = operator - operator.getH()
    symmetry_defect = int(hermitian_gap.nnz)
    if symmetry_defect == 0:
        hermitian_max = hermitian_mean = hermitian_median = hermitian_rms = 0.0
    else:
        gap_abs = np.abs(hermitian_gap.data)
        hermitian_max = float(gap_abs.max())
        hermitian_mean = float(gap_abs.mean())
        hermitian_median = float(np.median(gap_abs))
        hermitian_rms = float(np.sqrt(np.mean(gap_abs ** 2)))
    operator_norm = float(np.max(np.abs(operator.data))) if operator.nnz else 0.0
    rel_max = hermitian_max / max(operator_norm, 1e-16)
    rel_mean = hermitian_mean / max(operator_norm, 1e-16) if operator_norm else 0.0
    hermitian_close = hermitian_max <= max(1e-10, 1e-9 * max(operator_norm, 1e-16))
    n_rows, n_cols = cast(Tuple[int, int], operator.shape)
    return {
        "diag_min": float(diag_real.min()),
        "diag_max": float(diag_real.max()),
        "diag_mean": float(diag_real.mean()),
        "diag_std": float(diag_real.std()),
        "diag_abs_max": float(np.abs(diag_real).max()),
        "max_row_sum": float(row_sums.max()),
        "density": float(operator.nnz) / float(n_rows * n_cols),
        "nnz": int(operator.nnz),
        "symmetry_defect_nnz": symmetry_defect,
        "hermitian_max_diff": hermitian_max,
        "hermitian_mean_diff": hermitian_mean,
        "hermitian_median_diff": hermitian_median,
        "hermitian_rms_diff": hermitian_rms,
        "hermitian_rel_max": rel_max,
        "hermitian_rel_mean": rel_mean,
        "hermitian_is_close": hermitian_close,
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
        f"- Symmetry defect nnz: {operator_stats['symmetry_defect_nnz']} (0 → perfectly symmetric)",
        f"- Hermitian check (H ≈ H†?): {'pass' if operator_stats['hermitian_is_close'] else 'fail'}",
        f"  - max |H - H†| = {operator_stats['hermitian_max_diff']:.3e}",
        f"  - mean |H - H†| = {operator_stats['hermitian_mean_diff']:.3e}",
        f"  - median |H - H†| = {operator_stats['hermitian_median_diff']:.3e}",
        f"  - RMS |H - H†| = {operator_stats['hermitian_rms_diff']:.3e}",
        f"  - relative max = {operator_stats['hermitian_rel_max']:.3e}",
        f"  - relative mean = {operator_stats['hermitian_rel_mean']:.3e}",
    ]
    if "vg_norm_min" in field_stats:
        lines.append(
            f"- |v_g(R)|: min {field_stats['vg_norm_min']:.4e}, "
            f"max {field_stats['vg_norm_max']:.4e}, mean {field_stats['vg_norm_mean']:.4e}"
        )

    lines.append("")
    if include_plot:
        lines.append("- Field visualization was requested via config and saved alongside this report.")
    else:
        lines.append("- Field visualization skipped (config default). Report captures key diagnostics instead.")

    report_path = Path(cdir) / "phase2_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"    Wrote {report_path}")


def _wrap_index_with_phase(
    index: int,
    offset: int,
    size: int,
    phase_forward: complex,
    phase_backward: complex,
) -> tuple[int, complex]:
    """Return wrapped index and Bloch phase for a periodic shift."""
    if size <= 0:
        raise ValueError("Grid dimension must be positive")
    new_index = index + offset
    wraps = 0
    while new_index >= size:
        new_index -= size
        wraps += 1
    while new_index < 0:
        new_index += size
        wraps -= 1
    if wraps > 0:
        phase = phase_forward ** wraps
    elif wraps < 0:
        phase = phase_backward ** (-wraps)
    else:
        phase = 1.0
    return new_index, phase


def _apply_fd2_mass_terms(
    H: lil_matrix,
    R_grid: np.ndarray,
    mass_tensor: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    eta_sq: float,
    idx_fn,
    phase_x: complex,
    phase_x_conj: complex,
    phase_y: complex,
    phase_y_conj: complex,
):
    for i in range(Nx):
        for j in range(Ny):
            p = idx_fn(i, j)

            mx_plus = 0.5 * (mass_tensor[i, j, 0, 0] + mass_tensor[(i + 1) % Nx, j, 0, 0])
            coeff = -0.5 * eta_sq * mx_plus / (dx ** 2)
            wrap_forward_x = phase_x if i == Nx - 1 else 1.0
            H[p, idx_fn(i + 1, j)] += coeff * wrap_forward_x
            H[p, p] -= coeff

            mx_minus = 0.5 * (mass_tensor[i, j, 0, 0] + mass_tensor[(i - 1) % Nx, j, 0, 0])
            coeff = -0.5 * eta_sq * mx_minus / (dx ** 2)
            wrap_backward_x = phase_x_conj if i == 0 else 1.0
            H[p, idx_fn(i - 1, j)] += coeff * wrap_backward_x
            H[p, p] -= coeff

            my_plus = 0.5 * (mass_tensor[i, j, 1, 1] + mass_tensor[i, (j + 1) % Ny, 1, 1])
            coeff = -0.5 * eta_sq * my_plus / (dy ** 2)
            wrap_forward_y = phase_y if j == Ny - 1 else 1.0
            H[p, idx_fn(i, j + 1)] += coeff * wrap_forward_y
            H[p, p] -= coeff

            my_minus = 0.5 * (mass_tensor[i, j, 1, 1] + mass_tensor[i, (j - 1) % Ny, 1, 1])
            coeff = -0.5 * eta_sq * my_minus / (dy ** 2)
            wrap_backward_y = phase_y_conj if j == 0 else 1.0
            H[p, idx_fn(i, j - 1)] += coeff * wrap_backward_y
            H[p, p] -= coeff


def _apply_fd4_mass_terms(
    H: lil_matrix,
    mass_tensor: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    eta_sq: float,
    idx_fn,
    phase_x: complex,
    phase_x_conj: complex,
    phase_y: complex,
    phase_y_conj: complex,
):
    if Nx < 5 or Ny < 5:
        raise ValueError("phase2_fd_order=4 requires Nx, Ny ≥ 5 to build the stencil")

    # Interface-averaged tensors (a_{i+1/2}) for each direction.
    mx = mass_tensor[..., 0, 0]
    my = mass_tensor[..., 1, 1]
    mx_faces = 0.5 * (mx + np.roll(mx, -1, axis=0))
    my_faces = 0.5 * (my + np.roll(my, -1, axis=1))

    grad_offsets = (-1, 0, 1, 2)
    grad_weights_x = {off: weight / (12.0 * dx) for off, weight in zip(grad_offsets, (1.0, -8.0, 8.0, -1.0))}
    grad_weights_y = {off: weight / (12.0 * dy) for off, weight in zip(grad_offsets, (1.0, -8.0, 8.0, -1.0))}

    prefactor = 0.5 * eta_sq

    # X-direction contributions: sum_j (eta^2/2) * a_{j+1/2} * g_j^H g_j.
    for i in range(Nx):
        node_cache = {}
        for offset in grad_offsets:
            node_cache[offset] = _wrap_index_with_phase(i, offset, Nx, phase_x, phase_x_conj)
        for j in range(Ny):
            a_face = mx_faces[i, j]
            if a_face == 0.0:
                continue
            for s_off in grad_offsets:
                w_s = grad_weights_x[s_off]
                if w_s == 0.0:
                    continue
                i_s, phase_s = node_cache[s_off]
                p = idx_fn(i_s, j)
                for t_off in grad_offsets:
                    w_t = grad_weights_x[t_off]
                    if w_t == 0.0:
                        continue
                    i_t, phase_t = node_cache[t_off]
                    q = idx_fn(i_t, j)
                    phase_factor = np.conjugate(phase_s) * phase_t
                    coeff = prefactor * a_face * w_s * w_t * phase_factor
                    H[p, q] += coeff

    # Y-direction contributions (analogue of the loop above).
    for j in range(Ny):
        node_cache = {}
        for offset in grad_offsets:
            node_cache[offset] = _wrap_index_with_phase(j, offset, Ny, phase_y, phase_y_conj)
        for i in range(Nx):
            b_face = my_faces[i, j]
            if b_face == 0.0:
                continue
            for s_off in grad_offsets:
                w_s = grad_weights_y[s_off]
                if w_s == 0.0:
                    continue
                j_s, phase_s = node_cache[s_off]
                p = idx_fn(i, j_s)
                for t_off in grad_offsets:
                    w_t = grad_weights_y[t_off]
                    if w_t == 0.0:
                        continue
                    j_t, phase_t = node_cache[t_off]
                    q = idx_fn(i, j_t)
                    phase_factor = np.conjugate(phase_s) * phase_t
                    coeff = prefactor * b_face * w_s * w_t * phase_factor
                    H[p, q] += coeff


def _apply_vg_term(
    H: lil_matrix,
    vx: np.ndarray,
    vy: np.ndarray,
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    vg_prefactor: float,
    idx_fn,
    phase_x: complex,
    phase_x_conj: complex,
    phase_y: complex,
    phase_y_conj: complex,
    fd_order: int,
):
    if fd_order == 4:
        weights_x = {1: 8.0 / (12.0 * dx), -1: -8.0 / (12.0 * dx), 2: -1.0 / (12.0 * dx), -2: 1.0 / (12.0 * dx)}
        weights_y = {1: 8.0 / (12.0 * dy), -1: -8.0 / (12.0 * dy), 2: -1.0 / (12.0 * dy), -2: 1.0 / (12.0 * dy)}
        for i in range(Nx):
            for j in range(Ny):
                p = idx_fn(i, j)
                for offset, weight in weights_x.items():
                    q_i, phase = _wrap_index_with_phase(i, offset, Nx, phase_x, phase_x_conj)
                    q = idx_fn(q_i, j)
                    coef = -1j * vg_prefactor * vx[i, j] * weight * phase
                    H[p, q] += coef
                    H[q, p] += np.conjugate(coef)
                for offset, weight in weights_y.items():
                    q_j, phase = _wrap_index_with_phase(j, offset, Ny, phase_y, phase_y_conj)
                    q = idx_fn(i, q_j)
                    coef = -1j * vg_prefactor * vy[i, j] * weight * phase
                    H[p, q] += coef
                    H[q, p] += np.conjugate(coef)
        return

    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    for i in range(Nx):
        for j in range(Ny):
            p = idx_fn(i, j)
            # +x interface
            qx = idx_fn(i + 1, j)
            vx_half = 0.5 * (vx[i, j] + vx[(i + 1) % Nx, j])
            wrap = phase_x if i == Nx - 1 else 1.0
            coef = -1j * vg_prefactor * vx_half * inv_2dx * wrap
            H[p, qx] += coef
            H[qx, p] += np.conjugate(coef)

            # +y interface
            qy = idx_fn(i, j + 1)
            vy_half = 0.5 * (vy[i, j] + vy[i, (j + 1) % Ny])
            wrap = phase_y if j == Ny - 1 else 1.0
            coef = -1j * vg_prefactor * vy_half * inv_2dy * wrap
            H[p, qy] += coef
            H[qy, p] += np.conjugate(coef)


def _tile_fields(
    R_grid: np.ndarray,
    V: np.ndarray,
    mass_tensor: np.ndarray,
    vg_field: np.ndarray | None,
    tiles: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    tx, ty = tiles
    if tx == 1 and ty == 1:
        return R_grid, V, mass_tensor, vg_field

    Nx, Ny, _ = R_grid.shape
    span_x = float(R_grid[-1, 0, 0] - R_grid[0, 0, 0])
    span_y = float(R_grid[0, -1, 1] - R_grid[0, 0, 1])
    offset_x0 = -0.5 * span_x * (tx - 1)
    offset_y0 = -0.5 * span_y * (ty - 1)

    new_R = np.zeros((Nx * tx, Ny * ty, 2), dtype=R_grid.dtype)
    for ix in range(tx):
        for iy in range(ty):
            xs = ix * Nx
            xe = xs + Nx
            ys = iy * Ny
            ye = ys + Ny
            shift_x = offset_x0 + ix * span_x
            shift_y = offset_y0 + iy * span_y
            new_R[xs:xe, ys:ye, 0] = R_grid[..., 0] + shift_x
            new_R[xs:xe, ys:ye, 1] = R_grid[..., 1] + shift_y

    tiled_V = np.tile(V, (tx, ty))
    tiled_mass = np.tile(mass_tensor, (tx, ty, 1, 1))
    if vg_field is not None:
        tiled_vg = np.tile(vg_field, (tx, ty, 1))
    else:
        tiled_vg = None
    return new_R, tiled_V, tiled_mass, tiled_vg


def assemble_ea_operator(
    R_grid: np.ndarray,
    mass_tensor: np.ndarray,
    V: np.ndarray,
    eta: float,
    include_cross_terms: bool = False,
    bloch_k: Tuple[float, float] | None = None,
    vg_field: np.ndarray | None = None,
    include_vg_term: bool = True,
    vg_scale: float = 1.0,
    fd_order: int = 4,
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

    fd_order = int(fd_order)
    if fd_order not in {2, 4}:
        raise ValueError(f"phase2_fd_order={fd_order} is not supported (choose 2 or 4)")

    vg_enabled = include_vg_term and vg_field is not None
    if vg_enabled:
        if vg_field.shape[:2] != (Nx, Ny) or vg_field.shape[-1] < 2:
            raise ValueError("vg_field must have shape (Nx, Ny, 2) to include v_g term")
        vg_field = np.asarray(vg_field, dtype=float)
        vx = vg_field[..., 0]
        vy = vg_field[..., 1]
        vg_prefactor = float(vg_scale) * eta
    else:
        vx = vy = None  # type: ignore[assignment]
        vg_prefactor = 0.0

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

    H.setdiag(V.reshape(-1))

    if fd_order == 4:
        _apply_fd4_mass_terms(
            H,
            mass_tensor,
            Nx,
            Ny,
            dx,
            dy,
            eta_sq,
            idx,
            phase_x,
            phase_x_conj,
            phase_y,
            phase_y_conj,
        )
    else:
        _apply_fd2_mass_terms(
            H,
            R_grid,
            mass_tensor,
            Nx,
            Ny,
            dx,
            dy,
            eta_sq,
            idx,
            phase_x,
            phase_x_conj,
            phase_y,
            phase_y_conj,
        )

    if include_cross_terms:
        raise NotImplementedError("Cross-derivative coupling not yet implemented")

    if vg_enabled:
        vx = cast(np.ndarray, vx)
        vy = cast(np.ndarray, vy)
        _apply_vg_term(
            H,
            vx,
            vy,
            Nx,
            Ny,
            dx,
            dy,
            vg_prefactor,
            idx,
            phase_x,
            phase_x_conj,
            phase_y,
            phase_y_conj,
            fd_order,
        )

    return csr_matrix(H)


def summarize_phase2_fields(
    V: np.ndarray,
    mass_tensor: np.ndarray,
    operator: csr_matrix,
    candidate_id: int,
    grid_info: Dict[str, float],
    eta: float,
    vg_field: np.ndarray | None = None,
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
    if vg_field is not None:
        vg_norm = np.linalg.norm(vg_field[..., :2], axis=-1)
        stats.update(
            {
                "vg_norm_min": float(vg_norm.min()),
                "vg_norm_max": float(vg_norm.max()),
                "vg_norm_mean": float(vg_norm.mean()),
            }
        )
    return stats


def process_candidate(
    row,
    config: Dict,
    run_dir: Path,
    include_cross_terms: bool,
    plot_fields: bool,
    include_vg_term: bool,
    vg_scale: float,
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
        vg_field = np.asarray(cast(h5py.Dataset, hf["vg"])[:]) if "vg" in hf else None
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

    min_mass_eig = config.get("phase2_min_mass_eig")
    mass_tensor = _regularize_mass_tensor(M_inv, min_mass_eig)

    tiles_cfg = config.get("phase2_supercell_tiles", [1, 1])
    if isinstance(tiles_cfg, int):
        tiles = (int(tiles_cfg), int(tiles_cfg))
    elif isinstance(tiles_cfg, (list, tuple)) and len(tiles_cfg) == 2:
        tiles = (int(tiles_cfg[0]), int(tiles_cfg[1]))
    else:
        raise ValueError("phase2_supercell_tiles must be an int or [Tx, Ty] list")
    if tiles[0] < 1 or tiles[1] < 1:
        raise ValueError("phase2_supercell_tiles entries must be ≥ 1")

    if tiles != (1, 1):
        print(f"    Tiling Phase 1 fields into a {tiles[0]}×{tiles[1]} placement")
    R_grid, V, mass_tensor, vg_field = _tile_fields(R_grid, V, mass_tensor, vg_field, tiles)
    np.save(cdir / "phase2_R_grid.npy", R_grid)

    grid_info = _grid_metrics(R_grid)
    use_vg_term = include_vg_term and vg_field is not None
    if include_vg_term and vg_field is None:
        print("    WARNING: Phase 1 file missing 'vg' dataset; skipping group-velocity term for this candidate.")

    fd_order = int(config.get("phase2_fd_order", 4))
    operator = assemble_ea_operator(
        R_grid,
        mass_tensor,
        V,
        eta,
        include_cross_terms,
        vg_field=vg_field,
        include_vg_term=use_vg_term,
        vg_scale=vg_scale,
        fd_order=fd_order,
    )

    op_path = cdir / "phase2_operator.npz"
    save_npz(op_path, operator)
    print(f"    Saved operator to {op_path}")

    stats = summarize_phase2_fields(V, mass_tensor, operator, cid, grid_info, eta, vg_field)
    info_path = cdir / "phase2_operator_info.csv"
    pd.DataFrame([stats]).to_csv(info_path, index=False)
    print(f"    Wrote stats to {info_path}")

    operator_stats = _operator_diagnostics(operator)
    meta = {
        "fd_order": fd_order,
        "tiles": {"x": tiles[0], "y": tiles[1]},
        "vg_term_included": bool(use_vg_term),
        "vg_scale": float(vg_scale),
    }
    save_json(meta, cdir / "phase2_operator_meta.json")

    _write_phase2_report(cdir, cid, grid_info, stats, operator_stats, eta, plot_fields)

    if plot_fields:
        plot_phase2_fields(cdir, R_grid, V, mass_tensor, vg_field if use_vg_term else None)
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
    include_vg_term = bool(config.get("phase2_include_vg_term", True))
    vg_scale = float(config.get("phase2_vg_scale", 1.0))
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
            process_candidate(
                row,
                config,
                run_dir,
                include_cross_terms,
                plot_fields,
                include_vg_term,
                vg_scale,
            )
        except FileNotFoundError as exc:
            print(f"  Skipping candidate {cid}: {exc}")

    print("Phase 2 completed.\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python phases/phase2_ea_operator.py <run_dir|auto> <phase2_config.yaml>")
    run_phase2(sys.argv[1], sys.argv[2])
