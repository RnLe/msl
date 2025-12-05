"""
Phase 2 V2: Envelope Approximation Operator in Fractional Coordinates

Key V2 changes from legacy:
- Operates on unit square [0,1)² with fractional coordinates (s1, s2)
- Transformed mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ for correct Laplacian
- Grid spacing: ds1 = 1/Ns1, ds2 = 1/Ns2 (trivially uniform on unit square)
- Periodic BCs are exact: F(s1+1, s2) = F(s1, s2)
- No coordinate-warping issues for non-rectangular lattices (triangular, etc.)

The envelope Hamiltonian in fractional coordinates:
    H_EA = -η²/2 ∇_s · M̃⁻¹(s) ∇_s + V(s)

where M̃⁻¹(s) = B⁻¹ M⁻¹(s) B⁻ᵀ transforms the Cartesian mass tensor.
"""
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


# =============================================================================
# Mass Tensor Transformation
# =============================================================================

def transform_mass_tensor_to_fractional(
    M_inv: np.ndarray, 
    B_moire: np.ndarray
) -> np.ndarray:
    """
    Transform mass tensor from Cartesian to fractional coordinates.
    
    M̃⁻¹(s) = B⁻¹ M⁻¹(s) B⁻ᵀ
    
    This is the key V2 change: the Laplacian ∇_R · M⁻¹ ∇_R becomes
    ∇_s · M̃⁻¹ ∇_s when R = B·s.
    
    Args:
        M_inv: [Ns1, Ns2, 2, 2] inverse mass tensor in Cartesian coords
        B_moire: [2, 2] moiré basis matrix (columns = A1, A2)
    
    Returns:
        M_inv_tilde: [Ns1, Ns2, 2, 2] transformed mass tensor for fractional Laplacian
    """
    B_inv = np.linalg.inv(B_moire)
    # M̃⁻¹[i,j] = B_inv @ M⁻¹[i,j] @ B_inv.T
    # Using einsum: 'ia,...ab,jb->...ij' contracts properly
    return np.einsum('ia,...ab,jb->...ij', B_inv, M_inv, B_inv)


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


# =============================================================================
# Run Directory Resolution
# =============================================================================

def resolve_run_dir(run_dir: str | Path, config: Dict) -> Path:
    """Resolve 'auto'/'latest' shortcuts to concrete MPB Phase 0 run directories."""
    run_dir = Path(run_dir)
    if str(run_dir) in {"auto", "latest"}:
        # V2: look in runsV2 directory for MPB runs (exclude BLAZE)
        base = Path(config.get("output_dir", "runsV2"))
        all_runs = sorted(base.glob("phase0_*"))
        phase0_runs = [r for r in all_runs if 'blaze' not in r.name.lower()]
        if not phase0_runs:
            raise FileNotFoundError(
                f"No MPB phase0 run directories found in {base}\n"
                f"  (Looking for phase0_v2_* or phase0_*, excluding phase0_blaze_*)\n"
                f"  Found BLAZE runs? Use explicit path instead."
            )
        run_dir = phase0_runs[-1]
        print(f"Auto-selected latest MPB Phase 0 run: {run_dir}")
    if not run_dir.exists() and not run_dir.is_absolute():
        candidate = Path(config.get("output_dir", "runsV2")) / run_dir
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
        except Exception as exc:
            print(f"    WARNING: Failed to parse {meta_path}: {exc}")
    if candidate_frame is not None:
        match = candidate_frame[candidate_frame["candidate_id"] == candidate_id]
        if not match.empty:
            return match.iloc[0].to_dict()
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


# =============================================================================
# Operator Diagnostics
# =============================================================================

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


# =============================================================================
# Phase 2 Report (V2 format)
# =============================================================================

def _write_phase2_report(
    cdir: Path,
    candidate_id: int,
    grid_info: Dict[str, float],
    field_stats: Dict,
    operator_stats: Dict[str, float],
    eta: float,
    B_moire: np.ndarray,
    include_plot: bool,
):
    """Emit a human-readable summary of Phase 2 V2 results."""
    lines = [
        "# Phase 2 V2 Envelope Operator Report",
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
        "",
        "## V2 Key Changes",
        "- Laplacian formulated on unit square: ∇_s · M̃⁻¹(s) ∇_s",
        "- Transformed mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ",
        "- True periodic BCs (no Cartesian wrapping issues)",
        "",
        "## Discretization",
        f"- η (moiré scale ratio) = {eta:.6f}",
        "",
        "## Input Field Diagnostics",
        f"- Potential V(s): min {field_stats['V_min']:.6f}, max {field_stats['V_max']:.6f}, "
        f"mean {field_stats['V_mean']:.6f}",
        f"- M̃⁻¹ eigenvalues (transformed): min {field_stats['M_inv_tilde_min_eig']:.4f}, "
        f"max {field_stats['M_inv_tilde_max_eig']:.4f}",
        "",
        "## Operator Diagnostics",
        f"- Matrix size: {int(grid_info['Ns1'] * grid_info['Ns2'])} × {int(grid_info['Ns1'] * grid_info['Ns2'])}",
        f"- Non-zeros: {operator_stats['nnz']} ({operator_stats['density']*100:.3f}% density)",
        f"- Diagonal range: [{operator_stats['diag_min']:.6f}, {operator_stats['diag_max']:.6f}]",
        f"- Diagonal std dev: {operator_stats['diag_std']:.6f}",
        f"- Max absolute row sum: {operator_stats['max_row_sum']:.6f}",
        f"- Symmetry defect nnz: {operator_stats['symmetry_defect_nnz']} (0 → perfectly symmetric)",
        f"- Hermitian check (H ≈ H†?): {'pass' if operator_stats['hermitian_is_close'] else 'fail'}",
        f"  - max |H - H†| = {operator_stats['hermitian_max_diff']:.3e}",
        f"  - mean |H - H†| = {operator_stats['hermitian_mean_diff']:.3e}",
        f"  - RMS |H - H†| = {operator_stats['hermitian_rms_diff']:.3e}",
    ]
    
    if "vg_norm_min" in field_stats:
        lines.append(
            f"- |v_g(s)|: min {field_stats['vg_norm_min']:.4e}, "
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


# =============================================================================
# Bloch Phase Utilities
# =============================================================================

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


# =============================================================================
# Finite Difference Operators (V2: Fractional Coordinates)
# =============================================================================

def _apply_fd2_mass_terms_fractional(
    H: lil_matrix,
    M_inv_tilde: np.ndarray,
    Ns1: int,
    Ns2: int,
    ds1: float,
    ds2: float,
    eta_sq: float,
    idx_fn,
    phase_s1: complex,
    phase_s1_conj: complex,
    phase_s2: complex,
    phase_s2_conj: complex,
):
    """
    Apply 2nd-order FD mass terms in fractional coordinates.
    
    The kinetic term is: -η²/2 ∇_s · M̃⁻¹(s) ∇_s
    
    Using flux-conservative form at half-points.
    """
    for i in range(Ns1):
        for j in range(Ns2):
            p = idx_fn(i, j)

            # +s1 direction: interface at (i+1/2, j)
            m11_plus = 0.5 * (M_inv_tilde[i, j, 0, 0] + M_inv_tilde[(i + 1) % Ns1, j, 0, 0])
            coeff = -0.5 * eta_sq * m11_plus / (ds1 ** 2)
            wrap_forward = phase_s1 if i == Ns1 - 1 else 1.0
            H[p, idx_fn((i + 1) % Ns1, j)] += coeff * wrap_forward
            H[p, p] -= coeff

            # -s1 direction: interface at (i-1/2, j)
            m11_minus = 0.5 * (M_inv_tilde[i, j, 0, 0] + M_inv_tilde[(i - 1) % Ns1, j, 0, 0])
            coeff = -0.5 * eta_sq * m11_minus / (ds1 ** 2)
            wrap_backward = phase_s1_conj if i == 0 else 1.0
            H[p, idx_fn((i - 1) % Ns1, j)] += coeff * wrap_backward
            H[p, p] -= coeff

            # +s2 direction: interface at (i, j+1/2)
            m22_plus = 0.5 * (M_inv_tilde[i, j, 1, 1] + M_inv_tilde[i, (j + 1) % Ns2, 1, 1])
            coeff = -0.5 * eta_sq * m22_plus / (ds2 ** 2)
            wrap_forward = phase_s2 if j == Ns2 - 1 else 1.0
            H[p, idx_fn(i, (j + 1) % Ns2)] += coeff * wrap_forward
            H[p, p] -= coeff

            # -s2 direction: interface at (i, j-1/2)
            m22_minus = 0.5 * (M_inv_tilde[i, j, 1, 1] + M_inv_tilde[i, (j - 1) % Ns2, 1, 1])
            coeff = -0.5 * eta_sq * m22_minus / (ds2 ** 2)
            wrap_backward = phase_s2_conj if j == 0 else 1.0
            H[p, idx_fn(i, (j - 1) % Ns2)] += coeff * wrap_backward
            H[p, p] -= coeff


def _apply_fd4_mass_terms_fractional(
    H: lil_matrix,
    M_inv_tilde: np.ndarray,
    Ns1: int,
    Ns2: int,
    ds1: float,
    ds2: float,
    eta_sq: float,
    idx_fn,
    phase_s1: complex,
    phase_s1_conj: complex,
    phase_s2: complex,
    phase_s2_conj: complex,
):
    """
    Apply 4th-order FD mass terms in fractional coordinates.
    
    Uses interface-averaged tensors and 4-point gradient stencil.
    """
    if Ns1 < 5 or Ns2 < 5:
        raise ValueError("phase2_fd_order=4 requires Ns1, Ns2 ≥ 5 to build the stencil")

    # Interface-averaged tensors for each direction
    m11 = M_inv_tilde[..., 0, 0]
    m22 = M_inv_tilde[..., 1, 1]
    m11_faces = 0.5 * (m11 + np.roll(m11, -1, axis=0))
    m22_faces = 0.5 * (m22 + np.roll(m22, -1, axis=1))

    grad_offsets = (-1, 0, 1, 2)
    grad_weights_s1 = {off: weight / (12.0 * ds1) for off, weight in zip(grad_offsets, (1.0, -8.0, 8.0, -1.0))}
    grad_weights_s2 = {off: weight / (12.0 * ds2) for off, weight in zip(grad_offsets, (1.0, -8.0, 8.0, -1.0))}

    prefactor = 0.5 * eta_sq

    # s1-direction contributions
    for i in range(Ns1):
        node_cache = {}
        for offset in grad_offsets:
            node_cache[offset] = _wrap_index_with_phase(i, offset, Ns1, phase_s1, phase_s1_conj)
        for j in range(Ns2):
            a_face = m11_faces[i, j]
            if a_face == 0.0:
                continue
            for s_off in grad_offsets:
                w_s = grad_weights_s1[s_off]
                if w_s == 0.0:
                    continue
                i_s, phase_s = node_cache[s_off]
                p = idx_fn(i_s, j)
                for t_off in grad_offsets:
                    w_t = grad_weights_s1[t_off]
                    if w_t == 0.0:
                        continue
                    i_t, phase_t = node_cache[t_off]
                    q = idx_fn(i_t, j)
                    phase_factor = np.conjugate(phase_s) * phase_t
                    coeff = prefactor * a_face * w_s * w_t * phase_factor
                    H[p, q] += coeff

    # s2-direction contributions
    for j in range(Ns2):
        node_cache = {}
        for offset in grad_offsets:
            node_cache[offset] = _wrap_index_with_phase(j, offset, Ns2, phase_s2, phase_s2_conj)
        for i in range(Ns1):
            b_face = m22_faces[i, j]
            if b_face == 0.0:
                continue
            for s_off in grad_offsets:
                w_s = grad_weights_s2[s_off]
                if w_s == 0.0:
                    continue
                j_s, phase_s = node_cache[s_off]
                p = idx_fn(i, j_s)
                for t_off in grad_offsets:
                    w_t = grad_weights_s2[t_off]
                    if w_t == 0.0:
                        continue
                    j_t, phase_t = node_cache[t_off]
                    q = idx_fn(i, j_t)
                    phase_factor = np.conjugate(phase_s) * phase_t
                    coeff = prefactor * b_face * w_s * w_t * phase_factor
                    H[p, q] += coeff


def _apply_vg_term_fractional(
    H: lil_matrix,
    vg_tilde: np.ndarray,
    Ns1: int,
    Ns2: int,
    ds1: float,
    ds2: float,
    vg_prefactor: float,
    idx_fn,
    phase_s1: complex,
    phase_s1_conj: complex,
    phase_s2: complex,
    phase_s2_conj: complex,
    fd_order: int,
):
    """
    Apply group velocity term in fractional coordinates.
    
    The v_g term also needs to be transformed: v_g^tilde = B^{-1} v_g
    """
    vs1 = vg_tilde[..., 0]
    vs2 = vg_tilde[..., 1]
    
    if fd_order == 4:
        weights_s1 = {1: 8.0 / (12.0 * ds1), -1: -8.0 / (12.0 * ds1), 
                      2: -1.0 / (12.0 * ds1), -2: 1.0 / (12.0 * ds1)}
        weights_s2 = {1: 8.0 / (12.0 * ds2), -1: -8.0 / (12.0 * ds2), 
                      2: -1.0 / (12.0 * ds2), -2: 1.0 / (12.0 * ds2)}
        for i in range(Ns1):
            for j in range(Ns2):
                p = idx_fn(i, j)
                for offset, weight in weights_s1.items():
                    q_i, phase = _wrap_index_with_phase(i, offset, Ns1, phase_s1, phase_s1_conj)
                    q = idx_fn(q_i, j)
                    coef = -1j * vg_prefactor * vs1[i, j] * weight * phase
                    H[p, q] += coef
                    H[q, p] += np.conjugate(coef)
                for offset, weight in weights_s2.items():
                    q_j, phase = _wrap_index_with_phase(j, offset, Ns2, phase_s2, phase_s2_conj)
                    q = idx_fn(i, q_j)
                    coef = -1j * vg_prefactor * vs2[i, j] * weight * phase
                    H[p, q] += coef
                    H[q, p] += np.conjugate(coef)
        return

    # 2nd order FD
    inv_2ds1 = 1.0 / (2.0 * ds1)
    inv_2ds2 = 1.0 / (2.0 * ds2)
    for i in range(Ns1):
        for j in range(Ns2):
            p = idx_fn(i, j)
            # +s1 interface
            qs1 = idx_fn((i + 1) % Ns1, j)
            vs1_half = 0.5 * (vs1[i, j] + vs1[(i + 1) % Ns1, j])
            wrap = phase_s1 if i == Ns1 - 1 else 1.0
            coef = -1j * vg_prefactor * vs1_half * inv_2ds1 * wrap
            H[p, qs1] += coef
            H[qs1, p] += np.conjugate(coef)

            # +s2 interface
            qs2 = idx_fn(i, (j + 1) % Ns2)
            vs2_half = 0.5 * (vs2[i, j] + vs2[i, (j + 1) % Ns2])
            wrap = phase_s2 if j == Ns2 - 1 else 1.0
            coef = -1j * vg_prefactor * vs2_half * inv_2ds2 * wrap
            H[p, qs2] += coef
            H[qs2, p] += np.conjugate(coef)


# =============================================================================
# Supercell Tiling (V2: Fractional)
# =============================================================================

def _tile_fields_fractional(
    s_grid: np.ndarray,
    V: np.ndarray,
    M_inv_tilde: np.ndarray,
    vg_tilde: np.ndarray | None,
    tiles: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Tile fields for supercell calculation (rare use case in V2).
    
    In fractional coords, tiling means extending the unit square to [0, Tx) x [0, Ty).
    """
    tx, ty = tiles
    if tx == 1 and ty == 1:
        return s_grid, V, M_inv_tilde, vg_tilde

    Ns1, Ns2, _ = s_grid.shape
    
    # Create tiled fractional grid
    new_s = np.zeros((Ns1 * tx, Ns2 * ty, 2), dtype=s_grid.dtype)
    for ix in range(tx):
        for iy in range(ty):
            xs = ix * Ns1
            xe = xs + Ns1
            ys = iy * Ns2
            ye = ys + Ns2
            new_s[xs:xe, ys:ye, 0] = s_grid[..., 0] + ix
            new_s[xs:xe, ys:ye, 1] = s_grid[..., 1] + iy

    tiled_V = np.tile(V, (tx, ty))
    tiled_M = np.tile(M_inv_tilde, (tx, ty, 1, 1))
    if vg_tilde is not None:
        tiled_vg = np.tile(vg_tilde, (tx, ty, 1))
    else:
        tiled_vg = None
    
    return new_s, tiled_V, tiled_M, tiled_vg


# =============================================================================
# Main Operator Assembly (V2: Fractional Coordinates)
# =============================================================================

def assemble_ea_operator_fractional(
    s_grid: np.ndarray,
    M_inv_tilde: np.ndarray,
    V: np.ndarray,
    eta: float,
    B_moire: np.ndarray,
    include_cross_terms: bool = False,
    bloch_k: Tuple[float, float] | None = None,
    vg_tilde: np.ndarray | None = None,
    include_vg_term: bool = True,
    vg_scale: float = 1.0,
    fd_order: int = 4,
) -> csr_matrix:
    """
    Assemble the EA Hamiltonian in fractional coordinates (V2).
    
    H_EA = -η²/2 ∇_s · M̃⁻¹(s) ∇_s + V(s)
    
    on the unit square [0,1)² with (Bloch) periodic BCs.
    
    Key V2 advantage: True periodic BCs on unit square.
    Grid spacing is uniform: ds1 = 1/Ns1, ds2 = 1/Ns2.
    
    Args:
        s_grid: [Ns1, Ns2, 2] fractional coordinates
        M_inv_tilde: [Ns1, Ns2, 2, 2] TRANSFORMED mass tensor (already B⁻¹ M⁻¹ B⁻ᵀ)
        V: [Ns1, Ns2] potential
        eta: moiré/monolayer scale ratio
        B_moire: [2, 2] moiré basis for Bloch phase calculation
        include_cross_terms: if True, include off-diagonal mass tensor terms
        bloch_k: (k1, k2) Bloch wavevector in Cartesian reciprocal space
        vg_tilde: [Ns1, Ns2, 2] TRANSFORMED group velocity (already B⁻¹ @ vg)
        include_vg_term: whether to include group velocity term
        vg_scale: scaling factor for vg term
        fd_order: finite difference order (2 or 4)
    
    Returns:
        H: sparse [N, N] matrix where N = Ns1 * Ns2
    """
    Ns1, Ns2, _ = s_grid.shape
    total_points = Ns1 * Ns2
    
    # V2: Grid spacing on unit square is trivially uniform
    ds1 = 1.0 / Ns1
    ds2 = 1.0 / Ns2
    
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("Phase 2 requires a positive finite eta value")

    H = lil_matrix((total_points, total_points), dtype=np.complex128)
    eta_sq = eta ** 2

    fd_order = int(fd_order)
    if fd_order not in {2, 4}:
        raise ValueError(f"phase2_fd_order={fd_order} is not supported (choose 2 or 4)")

    vg_enabled = include_vg_term and vg_tilde is not None
    if vg_enabled:
        if vg_tilde.shape[:2] != (Ns1, Ns2) or vg_tilde.shape[-1] < 2:
            raise ValueError("vg_tilde must have shape (Ns1, Ns2, 2) to include v_g term")
        vg_prefactor = float(vg_scale) * eta
    else:
        vg_prefactor = 0.0

    # Bloch phases: for fractional coords, the phase when crossing boundary is
    # e^{i k · A_j} where A_j is the j-th moiré lattice vector (column of B_moire)
    bloch_k = bloch_k or (0.0, 0.0)
    A1 = B_moire[:, 0]  # First moiré lattice vector
    A2 = B_moire[:, 1]  # Second moiré lattice vector
    phase_s1 = np.exp(1j * np.dot(bloch_k, A1))
    phase_s2 = np.exp(1j * np.dot(bloch_k, A2))
    phase_s1_conj = np.conjugate(phase_s1)
    phase_s2_conj = np.conjugate(phase_s2)

    def idx(i: int, j: int) -> int:
        return (i % Ns1) * Ns2 + (j % Ns2)

    # On-site potential
    H.setdiag(V.reshape(-1))

    # Kinetic term
    if fd_order == 4:
        _apply_fd4_mass_terms_fractional(
            H, M_inv_tilde, Ns1, Ns2, ds1, ds2, eta_sq, idx,
            phase_s1, phase_s1_conj, phase_s2, phase_s2_conj,
        )
    else:
        _apply_fd2_mass_terms_fractional(
            H, M_inv_tilde, Ns1, Ns2, ds1, ds2, eta_sq, idx,
            phase_s1, phase_s1_conj, phase_s2, phase_s2_conj,
        )

    if include_cross_terms:
        raise NotImplementedError("Cross-derivative coupling not yet implemented in V2")

    if vg_enabled:
        _apply_vg_term_fractional(
            H, vg_tilde, Ns1, Ns2, ds1, ds2, vg_prefactor, idx,
            phase_s1, phase_s1_conj, phase_s2, phase_s2_conj, fd_order,
        )

    return csr_matrix(H)


# =============================================================================
# Field Statistics (V2)
# =============================================================================

def summarize_phase2_fields_v2(
    V: np.ndarray,
    M_inv_tilde: np.ndarray,
    operator: csr_matrix,
    candidate_id: int,
    grid_info: Dict[str, float],
    eta: float,
    vg_tilde: np.ndarray | None = None,
) -> Dict:
    """Compute statistics for Phase 2 V2 fields."""
    eigvals = np.linalg.eigvalsh(M_inv_tilde.reshape(-1, 2, 2))
    stats = {
        "candidate_id": candidate_id,
        "Ns1": int(grid_info["Ns1"]),
        "Ns2": int(grid_info["Ns2"]),
        "ds1": grid_info["ds1"],
        "ds2": grid_info["ds2"],
        "eta": eta,
        "nnz": int(operator.nnz),
        "V_min": float(V.min()),
        "V_max": float(V.max()),
        "V_mean": float(V.mean()),
        "M_inv_tilde_min_eig": float(eigvals.min()),
        "M_inv_tilde_max_eig": float(eigvals.max()),
        "M_inv_tilde_mean_eig": float(eigvals.mean()),
        "coordinate_system": "fractional",
        "pipeline_version": "V2",
    }
    if vg_tilde is not None:
        vg_norm = np.linalg.norm(vg_tilde[..., :2], axis=-1)
        stats.update({
            "vg_norm_min": float(vg_norm.min()),
            "vg_norm_max": float(vg_norm.max()),
            "vg_norm_mean": float(vg_norm.mean()),
        })
    return stats


# =============================================================================
# Candidate Processing (V2)
# =============================================================================

def process_candidate_v2(
    row,
    config: Dict,
    run_dir: Path,
    include_cross_terms: bool,
    plot_fields: bool,
    include_vg_term: bool,
    vg_scale: float,
):
    """Process a single candidate for Phase 2 V2."""
    cid = int(row["candidate_id"])
    cdir = candidate_dir(run_dir, cid)
    h5_path = cdir / "phase1_band_data.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 data for candidate {cid}: {h5_path}")

    print(f"  Candidate {cid}: loading Phase 1 V2 data from {h5_path}")
    with h5py.File(h5_path, "r") as hf:
        # V2: Primary grid is fractional
        s_grid = np.asarray(cast(h5py.Dataset, hf["s_grid"])[:])
        V = np.asarray(cast(h5py.Dataset, hf["V"])[:])
        M_inv = np.asarray(cast(h5py.Dataset, hf["M_inv"])[:])
        vg_field = np.asarray(cast(h5py.Dataset, hf["vg"])[:]) if "vg" in hf else None
        
        # V2: Must have basis matrices
        if "B_moire" not in hf.attrs:
            raise ValueError(f"Phase 1 V2 data missing B_moire attribute for candidate {cid}")
        B_moire = np.asarray(hf.attrs["B_moire"])
        B_mono = np.asarray(hf.attrs.get("B_mono", np.eye(2)))
        
        eta_attr = hf.attrs.get("eta")
        pipeline_version = hf.attrs.get("pipeline_version", "V1")
        
        if pipeline_version != "V2":
            print(f"    WARNING: Phase 1 data is from {pipeline_version}, expected V2")

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

    # V2: Transform mass tensor to fractional coordinates
    min_mass_eig = config.get("phase2_min_mass_eig")
    M_inv_reg = _regularize_mass_tensor(M_inv, min_mass_eig)
    M_inv_tilde = transform_mass_tensor_to_fractional(M_inv_reg, B_moire)
    print(f"    Transformed mass tensor: M̃⁻¹ = B⁻¹ M⁻¹ B⁻ᵀ")

    # V2: Transform group velocity to fractional coordinates
    if vg_field is not None:
        B_inv = np.linalg.inv(B_moire)
        vg_tilde = np.einsum('ij,...j->...i', B_inv, vg_field)
    else:
        vg_tilde = None

    # Handle supercell tiling (rare in V2)
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
        print(f"    Tiling Phase 1 fields into a {tiles[0]}×{tiles[1]} supercell")
        s_grid, V, M_inv_tilde, vg_tilde = _tile_fields_fractional(
            s_grid, V, M_inv_tilde, vg_tilde, tiles
        )

    # Save the fractional grid (V2)
    np.save(cdir / "phase2_s_grid.npy", s_grid)

    grid_info = _grid_metrics_fractional(s_grid)
    use_vg_term = include_vg_term and vg_tilde is not None
    if include_vg_term and vg_tilde is None:
        print("    WARNING: Phase 1 file missing 'vg' dataset; skipping group-velocity term for this candidate.")

    fd_order = int(config.get("phase2_fd_order", 4))
    operator = assemble_ea_operator_fractional(
        s_grid,
        M_inv_tilde,
        V,
        eta,
        B_moire,
        include_cross_terms,
        vg_tilde=vg_tilde,
        include_vg_term=use_vg_term,
        vg_scale=vg_scale,
        fd_order=fd_order,
    )

    op_path = cdir / "phase2_operator.npz"
    save_npz(op_path, operator)
    print(f"    Saved operator to {op_path}")

    stats = summarize_phase2_fields_v2(V, M_inv_tilde, operator, cid, grid_info, eta, vg_tilde)
    info_path = cdir / "phase2_operator_info.csv"
    pd.DataFrame([stats]).to_csv(info_path, index=False)
    print(f"    Wrote stats to {info_path}")

    operator_stats = _operator_diagnostics(operator)
    
    # V2: Enhanced metadata
    meta = {
        "fd_order": fd_order,
        "tiles": {"x": tiles[0], "y": tiles[1]},
        "vg_term_included": bool(use_vg_term),
        "vg_scale": float(vg_scale),
        "coordinate_system": "fractional",
        "pipeline_version": "V2",
        "Ns1": int(grid_info["Ns1"]),
        "Ns2": int(grid_info["Ns2"]),
        "eta": float(eta),
        "B_moire": B_moire.tolist(),
        "B_mono": B_mono.tolist() if B_mono is not None else None,
    }
    save_json(meta, cdir / "phase2_operator_meta.json")

    _write_phase2_report(
        cdir, cid, grid_info, stats, operator_stats, eta, B_moire, plot_fields
    )

    if plot_fields:
        # For plotting, convert to Cartesian
        R_grid = np.einsum('ij,...j->...i', B_moire, s_grid)
        plot_phase2_fields(cdir, R_grid, V, M_inv_tilde, vg_tilde if use_vg_term else None)
        print("    Rendered phase2_fields_visualization.png")
    else:
        print("    Skipped field visualization (set phase2_plot_fields: true to enable).")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_phase2(run_dir: str | Path, config_path: str | Path):
    """
    Run Phase 2 V2: Envelope Operator in Fractional Coordinates.
    """
    print("\n" + "=" * 70)
    print("PHASE 2 V2: Envelope Operator Assembly (Fractional Coordinates)")
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
            process_candidate_v2(
                row, config, run_dir, include_cross_terms,
                plot_fields, include_vg_term, vg_scale,
            )
        except FileNotFoundError as exc:
            print(f"  Skipping candidate {cid}: {exc}")

    print("Phase 2 V2 completed.\n")


def get_default_config_path() -> Path:
    """Return the path to the default MPB Phase 2 V2 config."""
    return Path(__file__).parent.parent / "configsV2" / "phase2.yaml"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments: use latest run and default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2("auto", str(default_config))
    elif len(sys.argv) == 2:
        # One argument: interpret as run_dir, use default config
        default_config = get_default_config_path()
        if not default_config.exists():
            raise SystemExit(f"Default config not found: {default_config}")
        print(f"Using default config: {default_config}")
        run_phase2(sys.argv[1], str(default_config))
    elif len(sys.argv) == 3:
        # Two arguments: run_dir and config_path
        run_phase2(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage: python phasesV2/phase2_ea_operator.py [run_dir|auto] [config.yaml]\n"
            "       No arguments: uses latest run with default config\n"
            "       One argument: uses specified run_dir with default config\n"
            "       Two arguments: uses specified run_dir and config"
        )
