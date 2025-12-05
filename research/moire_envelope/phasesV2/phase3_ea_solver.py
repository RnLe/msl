"""
Phase 3 V2: Envelope Eigenvalue Solver in Fractional Coordinates

Key V2 changes from legacy:
- Operates on unit square [0,1)² with fractional coordinates (s1, s2)
- Reads s_grid as primary grid (not R_grid)
- Stores eigenstates F(s1, s2) on fractional grid
- Includes B_moire for Cartesian visualization transforms
- All localization metrics computed in fractional space first, then scaled

The eigenvalue problem:
    H_EA F_n(s) = Δω_n F_n(s)

with periodic BCs on [0,1)² (automatic from Phase 2 V2 operator).
"""
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

from common.io_utils import candidate_dir, load_yaml, load_json, save_json
from common.plotting import plot_envelope_modes
from common.scoring import participation_ratio, field_entropy, localization_length


# =============================================================================
# Run Directory Resolution
# =============================================================================

def resolve_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete MPB Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        # V2: look in runsV2 directory for MPB runs (exclude BLAZE)
        runs_base = Path(config.get("output_dir", "runsV2"))
        all_runs = sorted(runs_base.glob("phase0_*"))
        phase0_runs = [r for r in all_runs if 'blaze' not in r.name.lower()]
        if not phase0_runs:
            raise FileNotFoundError(
                f"No MPB phase0 run directories found in {runs_base}\n"
                f"  (Looking for phase0_v2_* or phase0_*, excluding phase0_blaze_*)\n"
                f"  Found BLAZE runs? Use explicit path instead."
            )
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest MPB Phase 0 run: {run_dir}")
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


# =============================================================================
# Grid Metrics (V2: Fractional Coordinates)
# =============================================================================

def _grid_metrics_fractional(s_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics for fractional coordinate system (unit square)."""
    Ns1, Ns2, _ = s_grid.shape
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    return {
        "Ns1": Ns1,
        "Ns2": Ns2,
        "ds1": ds1,
        "ds2": ds2,
        "s1_min": 0.0,
        "s1_max": 1.0 - ds1,
        "s2_min": 0.0,
        "s2_max": 1.0 - ds2,
        "cell_area": 1.0,  # Unit square in fractional coords
    }


def _grid_metrics_cartesian(R_grid: np.ndarray) -> Dict[str, float]:
    """Compute grid metrics in Cartesian coordinates (for legacy compatibility)."""
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


# =============================================================================
# Phase 1/2 Metadata Loading (V2)
# =============================================================================

def _load_phase1_metadata_v2(cdir: Path) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """
    Load Phase 1 V2 metadata including fractional grid and basis matrices.
    
    Returns:
        s_grid: [Ns1, Ns2, 2] fractional coordinates
        R_grid: [Ns1, Ns2, 2] Cartesian coordinates (computed from s_grid and B_moire)
        omega_ref: reference frequency
        eta: moiré scale ratio
        B_moire: [2, 2] moiré basis matrix
    """
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data: {h5_path}")
    
    # Check for supercell grid override (V2 uses s_grid)
    override_grid = cdir / "phase2_s_grid.npy"
    s_grid_override = None
    if override_grid.exists():
        try:
            s_grid_override = np.load(override_grid)
            print(f"    Using supercell grid from {override_grid.name}")
        except Exception as exc:
            print(f"    WARNING: Failed to load {override_grid}: {exc}; falling back to Phase 1 grid")
    
    with h5py.File(h5_path, "r") as hf:
        # V2: Primary grid is fractional
        if "s_grid" in hf:
            s_grid = np.asarray(cast(h5py.Dataset, hf["s_grid"])[:])
        else:
            # Fallback for V1 data
            R_grid_raw = np.asarray(cast(h5py.Dataset, hf["R_grid"])[:])
            print("    WARNING: Phase 1 data is V1 format (R_grid only)")
            s_grid = None
            
        omega_ref = float(hf.attrs.get("omega_ref", np.nan))
        eta = float(hf.attrs.get("eta", np.nan))
        
        # V2: Must have B_moire
        if "B_moire" in hf.attrs:
            B_moire = np.asarray(hf.attrs["B_moire"])
        else:
            # Fallback: try to load from phase2 meta
            phase2_meta_path = cdir / "phase2_operator_meta.json"
            if phase2_meta_path.exists():
                phase2_meta = load_json(phase2_meta_path)
                B_moire = np.array(phase2_meta.get("B_moire", np.eye(2)))
            else:
                print("    WARNING: Missing B_moire, using identity")
                B_moire = np.eye(2)
    
    if s_grid_override is not None:
        s_grid = s_grid_override
    
    # Handle V1 fallback
    if s_grid is None:
        # V1 data: R_grid is primary, compute s_grid
        B_inv = np.linalg.inv(B_moire)
        s_grid = np.einsum('ij,...j->...i', B_inv, R_grid_raw)
        R_grid = R_grid_raw
    else:
        # V2: compute R_grid from s_grid for visualization
        R_grid = np.einsum('ij,...j->...i', B_moire, s_grid)
    
    return s_grid, R_grid, omega_ref, eta, B_moire


# =============================================================================
# Eigensolver
# =============================================================================

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
    """Solve the eigenvalue problem for the envelope operator."""
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
                    tol=tol,
                    maxiter=maxiter,
                )
            else:
                eigvals, eigvecs = eigsh(
                    operator,
                    k=k,
                    which=which,
                    tol=tol,
                    maxiter=maxiter,
                )
        except Exception as exc:
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
                tol=tol,
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


# =============================================================================
# Mode Analysis (V2: Fractional + Cartesian)
# =============================================================================

def _analyze_modes_v2(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    s_grid: np.ndarray,
    R_grid: np.ndarray,
    omega_ref: float,
    B_moire: np.ndarray,
) -> Tuple[np.ndarray, Iterable[Dict[str, float]]]:
    """
    Analyze envelope modes in both fractional and Cartesian coordinates.
    
    The modes F(s1, s2) are defined on the unit square.
    For physical interpretation, we also compute metrics in Cartesian coords.
    """
    Ns1, Ns2, _ = s_grid.shape
    n_modes = eigenvectors.shape[1]
    fields = eigenvectors.T.reshape((n_modes, Ns1, Ns2))
    
    # Moiré cell area in Cartesian (for proper normalization)
    A_moire = abs(np.linalg.det(B_moire))
    
    rows = []
    for idx in range(n_modes):
        field = fields[idx]
        abs_field = np.abs(field)
        abs2 = abs_field ** 2
        norm_l2 = float(np.linalg.norm(field.ravel()))
        
        delta_omega = eigenvalues[idx]
        delta_real = float(delta_omega.real if np.iscomplexobj(delta_omega) else delta_omega)
        omega_cav = float(omega_ref + delta_real)
        
        # Mode metrics
        pr = float(participation_ratio(field))
        entropy = float(field_entropy(field))
        
        # Localization length: compute in Cartesian for physical meaning
        loc_len = float(localization_length(field, R_grid))
        
        # Center of mass in fractional coords
        weight = abs2 / max(abs2.sum(), 1e-20)
        center_s = np.tensordot(weight, s_grid, axes=([0, 1], [0, 1]))
        
        # Convert to Cartesian for reporting
        center_R = B_moire @ center_s
        
        # Peak position
        peak_idx = np.unravel_index(np.argmax(abs2), abs2.shape)
        peak_s = s_grid[peak_idx]
        peak_R = R_grid[peak_idx]
        
        rows.append({
            "mode_index": idx,
            "delta_omega": delta_real,
            "omega_cavity": omega_cav,
            "norm_l2": norm_l2,
            "participation_ratio": pr,
            "entropy": entropy,
            "localization_length": loc_len,
            # Fractional coordinates
            "center_s1": float(center_s[0]),
            "center_s2": float(center_s[1]),
            "peak_s1": float(peak_s[0]),
            "peak_s2": float(peak_s[1]),
            # Cartesian coordinates (for compatibility)
            "center_x": float(center_R[0]),
            "center_y": float(center_R[1]),
            "peak_x": float(peak_R[0]),
            "peak_y": float(peak_R[1]),
            "max_amplitude": float(abs_field.max()),
        })
    
    return fields, rows


# =============================================================================
# Phase 3 Report (V2 format)
# =============================================================================

def _write_phase3_report_v2(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    eigenvalues: np.ndarray,
    omega_ref: float,
    eta: float,
    B_moire: np.ndarray,
    hermitian: bool,
    solver_details: Dict[str, float],
    mode_rows: Iterable[Dict[str, float]],
):
    """Write Phase 3 V2 report in markdown format."""
    report_path = cdir / "phase3_report.md"
    rows = list(mode_rows)
    
    lines = [
        "# Phase 3 V2 Envelope Solver Report",
        "",
        f"**Candidate**: {candidate_id:04d}",
        f"**Pipeline Version**: V2 (Fractional Coordinates)",
        "",
        "## Coordinate System",
        f"- **Type**: Fractional (unit square [0,1)²)",
        f"- Grid: {int(grid_info['Ns1'])} × {int(grid_info['Ns2'])} points",
        f"- Δs₁ = {grid_info['ds1']:.6f}, Δs₂ = {grid_info['ds2']:.6f}",
        f"- Moiré basis B_moire:",
        f"  - A₁ = [{B_moire[0, 0]:.4f}, {B_moire[1, 0]:.4f}]",
        f"  - A₂ = [{B_moire[0, 1]:.4f}, {B_moire[1, 1]:.4f}]",
        f"- Moiré cell area: {abs(np.linalg.det(B_moire)):.4f}",
        "",
        "## Parameters",
        f"- η (moiré scale ratio) = {eta:.6f}",
        f"- ω_ref = {omega_ref:.6f}",
        "",
        "## Solver",
        f"- Operator Hermitian: {hermitian}",
        f"- Solver: {solver_details['solver_name']} (which={solver_details['which']})",
        f"- Modes requested: {int(solver_details['requested_modes'])}, solved: {int(solver_details['solved_modes'])}",
        f"- Shift-invert: {'yes' if solver_details['used_shift'] else 'no'}",
    ]
    
    if solver_details['used_shift']:
        lines.append(f"  - σ = {solver_details['shift_value']:.6e}")
    
    lines.extend([
        f"- Tolerance: {solver_details['tol']:.2e}",
        f"- Max iterations: {int(solver_details['maxiter'])}",
        "",
        "## Lowest Eigenvalues",
    ])
    
    for row in rows[: min(5, len(rows))]:
        lines.append(
            f"- Mode {int(row['mode_index'])}: Δω = {row['delta_omega']:.6e}, "
            f"ω = {row['omega_cavity']:.6e}, PR = {row['participation_ratio']:.3f}, "
            f"ξ = {row['localization_length']:.3f}"
        )
    
    lines.extend([
        "",
        "## Mode Centers (Fractional → Cartesian)",
    ])
    
    for row in rows[: min(5, len(rows))]:
        lines.append(
            f"- Mode {int(row['mode_index'])}: "
            f"s = ({row['center_s1']:.3f}, {row['center_s2']:.3f}) → "
            f"R = ({row['center_x']:.3f}, {row['center_y']:.3f})"
        )
    
    report_path.write_text("\n".join(lines) + "\n")
    print(f"    Wrote {report_path}")


# =============================================================================
# Candidate Processing (V2)
# =============================================================================

def process_candidate_v2(row, run_dir: Path, config: Dict):
    """Process a single candidate for Phase 3 V2."""
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    operator_path = cdir / "phase2_operator.npz"
    if not operator_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 operator for candidate {cid}: {operator_path}")

    H = load_npz(operator_path).tocsr()
    
    # V2: Load fractional grid and basis
    s_grid, R_grid, omega_ref, eta, B_moire = _load_phase1_metadata_v2(cdir)
    
    if not np.isfinite(omega_ref):
        omega_ref = float(row.get("omega0", 0.0))
    if not np.isfinite(eta):
        a = float(row.get("a", np.nan))
        L_m = float(row.get("moire_length", np.nan))
        if np.isfinite(a) and np.isfinite(L_m) and L_m > 0:
            eta = a / L_m

    grid_info = _grid_metrics_fractional(s_grid)

    hermitian = _is_hermitian(H)
    n_modes = int(config.get("ea_n_modes", 8))
    eigenvalues, eigenvectors, solver_details = _solve_eigenpairs(H, n_modes, hermitian, config)
    
    if np.iscomplexobj(eigenvalues):
        max_imag = float(np.max(np.abs(eigenvalues.imag)))
        if max_imag > 1e-8:
            print(f"    WARNING: eigenvalues have imaginary component up to {max_imag:.2e}")
    
    fields, mode_rows = _analyze_modes_v2(eigenvalues, eigenvectors, s_grid, R_grid, omega_ref, B_moire)

    # Persist eigenstates (V2 format: s_grid as primary)
    eig_h5 = cdir / "phase3_eigenstates.h5"
    with h5py.File(eig_h5, "w") as hf:
        hf.create_dataset("F", data=fields, compression="gzip")
        hf.create_dataset("s_grid", data=s_grid, compression="gzip")
        hf.create_dataset("R_grid", data=R_grid, compression="gzip")  # For visualization
        hf.create_dataset("eigenvalues", data=eigenvalues)
        hf.attrs["omega_ref"] = omega_ref
        hf.attrs["eta"] = eta
        hf.attrs["B_moire"] = B_moire
        hf.attrs["pipeline_version"] = "V2"
        hf.attrs["coordinate_system"] = "fractional"
    print(f"    Saved eigenstates to {eig_h5}")

    # Tabulate metrics
    df = pd.DataFrame(list(mode_rows))
    df.insert(0, "candidate_id", cid)
    df_path = cdir / "phase3_eigenvalues.csv"
    df.to_csv(df_path, index=False)
    print(f"    Wrote eigenvalue table to {df_path}")

    # Save metadata JSON
    meta = {
        "n_modes": n_modes,
        "hermitian": hermitian,
        "omega_ref": float(omega_ref),
        "eta": float(eta),
        "coordinate_system": "fractional",
        "pipeline_version": "V2",
        "Ns1": int(grid_info["Ns1"]),
        "Ns2": int(grid_info["Ns2"]),
        "B_moire": B_moire.tolist(),
    }
    save_json(meta, cdir / "phase3_solver_meta.json")

    # Plot modes (using Cartesian R_grid for visualization)
    eigenvalues_for_plots = eigenvalues.real if np.iscomplexobj(eigenvalues) else eigenvalues
    plot_envelope_modes(
        cdir,
        R_grid,
        fields,
        eigenvalues_for_plots,
        n_modes=n_modes,
        candidate_params=row,
    )
    
    _write_phase3_report_v2(
        cdir, cid, grid_info, eigenvalues_for_plots, omega_ref, eta, 
        B_moire, hermitian, solver_details, mode_rows
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def run_phase3(run_dir: str | Path, config_path: str | Path):
    """
    Run Phase 3 V2: Envelope Eigenvalue Solver in Fractional Coordinates.
    """
    print("\n" + "=" * 70)
    print("PHASE 3 V2: Envelope Eigenvalue Solver (Fractional Coordinates)")
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
    iterator = tqdm(candidate_iter, desc="Phase 3 V2 Progress", unit="candidate", ncols=80)
    for cid, _ in iterator:
        iterator.set_postfix_str(f"CID {cid}")
        row = _load_candidate_metadata(run_dir, cid, candidate_frame)
        try:
            print(f"  Candidate {cid}: solving EA modes (V2)")
            process_candidate_v2(row, run_dir, config)
        except FileNotFoundError as exc:
            print(f"    Skipping candidate {cid}: {exc}")

    print("Phase 3 V2 completed.\n")


def get_default_config_path() -> Path:
    """Return the path to the default MPB Phase 3 V2 config."""
    return Path(__file__).parent.parent / "configsV2" / "phase3.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase3("auto", str(default_config))
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase3(sys.argv[1], str(default_config))
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase3(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python phasesV2/phase3_ea_solver.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
