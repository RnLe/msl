from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse


def _to_python(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def classify_status(max_abs: float, rel: float, tol_abs: float, tol_rel: float) -> str:
    if not np.isfinite(max_abs) or not np.isfinite(rel):
        return 'WARN'
    return 'WARN' if max_abs > tol_abs and rel > tol_rel else 'OK'


def hermiticity_metrics(array: np.ndarray, axis_a: int, axis_b: int, tol_abs: float, tol_rel: float) -> dict[str, Any]:
    diff = array - np.swapaxes(np.conj(array), axis_a, axis_b)
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    max_val = float(np.max(np.abs(array))) if array.size else 0.0
    rel = max_abs / max(max_val, 1e-15)
    return {
        'max_abs': max_abs,
        'relative_max_abs': rel,
        'max_value': max_val,
        'tol_abs': tol_abs,
        'tol_rel': tol_rel,
        'status': classify_status(max_abs, rel, tol_abs, tol_rel),
    }


def diagonal_imag_metrics(array: np.ndarray, axis_a: int, axis_b: int, tol_abs: float, tol_rel: float) -> dict[str, Any]:
    diag = np.diagonal(array, axis1=axis_a, axis2=axis_b)
    max_abs = float(np.max(np.abs(np.imag(diag)))) if diag.size else 0.0
    max_val = float(np.max(np.abs(diag))) if diag.size else 0.0
    rel = max_abs / max(max_val, 1e-15)
    return {
        'max_abs_imag': max_abs,
        'relative_max_abs_imag': rel,
        'max_diag_value': max_val,
        'tol_abs': tol_abs,
        'tol_rel': tol_rel,
        'status': classify_status(max_abs, rel, tol_abs, tol_rel),
    }


def real_symmetric_2x2_metrics(array: np.ndarray, tol_abs: float, tol_rel: float) -> dict[str, Any]:
    antisym = array[..., 0, 1] - array[..., 1, 0]
    imag = np.imag(array)
    max_abs = max(
        float(np.max(np.abs(antisym))) if antisym.size else 0.0,
        float(np.max(np.abs(imag))) if imag.size else 0.0,
    )
    max_val = float(np.max(np.abs(array))) if array.size else 0.0
    rel = max_abs / max(max_val, 1e-15)
    return {
        'max_abs': max_abs,
        'relative_max_abs': rel,
        'max_value': max_val,
        'tol_abs': tol_abs,
        'tol_rel': tol_rel,
        'status': classify_status(max_abs, rel, tol_abs, tol_rel),
    }


def psd_metrics(array: np.ndarray, axis_a: int, axis_b: int, tol_negative: float = 1e-8) -> dict[str, Any]:
    moved = np.moveaxis(array, (axis_a, axis_b), (-2, -1))
    mats = moved.reshape(-1, moved.shape[-2], moved.shape[-1])
    eig_min = float('inf')
    eig_max = float('-inf')
    for mat in mats:
        vals = np.linalg.eigvalsh(0.5 * (mat + mat.conj().T))
        eig_min = min(eig_min, float(vals.min()))
        eig_max = max(eig_max, float(vals.max()))
    status = 'WARN' if eig_min < -tol_negative else 'OK'
    return {
        'min_eigenvalue': eig_min,
        'max_eigenvalue': eig_max,
        'tol_negative': tol_negative,
        'status': status,
    }


def band_matrix_range_metrics(array: np.ndarray) -> dict[str, Any]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            'min_real': float('nan'),
            'max_real': float('nan'),
            'mean_abs': float('nan'),
            'max_abs': float('nan'),
        }
    return {
        'min_real': float(np.min(np.real(finite))),
        'max_real': float(np.max(np.real(finite))),
        'mean_abs': float(np.mean(np.abs(finite))),
        'max_abs': float(np.max(np.abs(finite))),
    }


def sparse_hermiticity_metrics(matrix: sparse.spmatrix, tol_abs: float, tol_rel: float) -> dict[str, Any]:
    diff = matrix - matrix.getH()
    max_abs = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    max_val = float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
    rel = max_abs / max(max_val, 1e-15)
    return {
        'max_abs': max_abs,
        'relative_max_abs': rel,
        'max_value': max_val,
        'nnz': int(matrix.nnz),
        'tol_abs': tol_abs,
        'tol_rel': tol_rel,
        'status': classify_status(max_abs, rel, tol_abs, tol_rel),
    }


def phase2_sanity_report(A_berry: np.ndarray, Phi_BH: np.ndarray, v_drift: np.ndarray,
                         M_inv: np.ndarray, Lambda: np.ndarray,
                         berry_raw_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {
        'Lambda': {
            'hermiticity': hermiticity_metrics(Lambda, -2, -1, 1e-10, 1e-10),
            'diagonal_imag': diagonal_imag_metrics(Lambda, -2, -1, 1e-12, 1e-12),
            'range': band_matrix_range_metrics(Lambda),
        },
        'A_berry': {
            'hermiticity': hermiticity_metrics(A_berry, -3, -2, 1e-8, 1e-6),
            'diagonal_imag': diagonal_imag_metrics(A_berry, -3, -2, 1e-8, 1e-6),
            'range': band_matrix_range_metrics(A_berry),
            'raw_diagnostics': _to_python(berry_raw_diagnostics or {}),
        },
        'Phi_BH': {
            'hermiticity': hermiticity_metrics(Phi_BH, -2, -1, 1e-8, 1e-6),
            'diagonal_imag': diagonal_imag_metrics(Phi_BH, -2, -1, 1e-12, 1e-12),
            'range': band_matrix_range_metrics(Phi_BH),
            'psd': psd_metrics(Phi_BH, -2, -1, 1e-8),
        },
        'v_drift': {
            'hermiticity': hermiticity_metrics(v_drift, -3, -2, 1e-6, 1e-6),
            'diagonal_imag': diagonal_imag_metrics(v_drift, -3, -2, 1e-6, 1e-6),
            'range': band_matrix_range_metrics(v_drift),
        },
        'M_inv_band': {
            'hermiticity': hermiticity_metrics(M_inv, -4, -3, 1e-8, 1e-6),
            'range': band_matrix_range_metrics(M_inv),
        },
        'M_inv_tensor': {
            'real_symmetry': real_symmetric_2x2_metrics(M_inv, 1e-8, 1e-6),
        },
    }
    return _to_python(report)


def phase3_sanity_report(H: sparse.spmatrix) -> dict[str, Any]:
    diag = H.diagonal()
    return _to_python({
        'Hamiltonian': {
            'hermiticity': sparse_hermiticity_metrics(H, 1e-8, 1e-6),
            'diag_min': float(np.min(np.real(diag))),
            'diag_max': float(np.max(np.real(diag))),
        }
    })


def log_sanity_block(log_fn, title: str, report: dict[str, Any]) -> None:
    log_fn(f'  {title}:')
    for name, entry in report.items():
        if 'hermiticity' in entry:
            herm = entry['hermiticity']
            log_fn(
                f"    {name}: herm={herm['status']} max={herm['max_abs']:.3e} rel={herm['relative_max_abs']:.3e}"
            )
        if 'diagonal_imag' in entry:
            diag = entry['diagonal_imag']
            log_fn(
                f"      diag-imag={diag['status']} max={diag['max_abs_imag']:.3e} rel={diag['relative_max_abs_imag']:.3e}"
            )
        if 'psd' in entry:
            psd = entry['psd']
            log_fn(
                f"      psd={psd['status']} min_eig={psd['min_eigenvalue']:.3e}"
            )
        if 'real_symmetry' in entry:
            rs = entry['real_symmetry']
            log_fn(
                f"      real-sym={rs['status']} max={rs['max_abs']:.3e} rel={rs['relative_max_abs']:.3e}"
            )
