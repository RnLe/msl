from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def _flatten_band_set(fields: np.ndarray) -> np.ndarray:
    return np.asarray(fields).reshape(fields.shape[0], -1)


def _epsilon_weights(epsilon: np.ndarray, state_shape: tuple[int, ...]) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=np.float64)
    n_flat = int(np.prod(state_shape))
    if eps.size == n_flat:
        return eps.reshape(-1)
    if eps.ndim + 1 == len(state_shape) and state_shape[-1] > 1:
        return np.repeat(eps[..., None], state_shape[-1], axis=-1).reshape(-1)
    if eps.size == int(np.prod(state_shape[:-1])) and state_shape[-1] > 1:
        return np.repeat(eps.reshape(*state_shape[:-1])[..., None], state_shape[-1], axis=-1).reshape(-1)
    raise ValueError(
        f"Cannot broadcast epsilon shape {eps.shape} onto state shape {state_shape}"
    )


def epsilon_weighted_overlap_matrix(
    fields_ref: np.ndarray,
    fields_cur: np.ndarray,
    epsilon: np.ndarray,
) -> np.ndarray:
    """Return normalized ε-weighted overlap matrix between two band sets."""
    ref_flat = _flatten_band_set(fields_ref)
    cur_flat = _flatten_band_set(fields_cur)
    weights = _epsilon_weights(epsilon, tuple(fields_ref.shape[1:]))
    overlaps = (ref_flat.conj() * weights[None, :]) @ cur_flat.T
    norms_ref = np.sqrt(np.maximum(np.real(np.diag((ref_flat.conj() * weights[None, :]) @ ref_flat.T)), 1e-30))
    norms_cur = np.sqrt(np.maximum(np.real(np.diag((cur_flat.conj() * weights[None, :]) @ cur_flat.T)), 1e-30))
    return overlaps / (norms_ref[:, None] * norms_cur[None, :] + 1e-30)


def compare_subspaces(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    epsilon: np.ndarray,
) -> dict[str, Any]:
    """Compare two equal-rank subspaces via canonical overlap singular values."""
    overlap = epsilon_weighted_overlap_matrix(basis_a, basis_b, epsilon)
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    rank = min(basis_a.shape[0], basis_b.shape[0])
    projector_fro = float(np.sqrt(max(0.0, 2.0 * rank - 2.0 * np.sum(np.abs(singular_values) ** 2))))
    return {
        'singular_values': singular_values,
        'min_singular_value': float(np.min(singular_values)) if singular_values.size else 0.0,
        'mean_singular_value': float(np.mean(singular_values)) if singular_values.size else 0.0,
        'projector_frobenius_distance': projector_fro,
    }


def transport_child_subspace(
    parent_basis: np.ndarray,
    child_all_bands: np.ndarray,
    epsilon: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a transported subspace into the child's full band space and align it."""
    n_sub = parent_basis.shape[0]
    overlap = epsilon_weighted_overlap_matrix(parent_basis, child_all_bands, epsilon)
    u_mat, singular_values, vh_mat = np.linalg.svd(overlap, full_matrices=False)
    coeff = vh_mat.conj().T[:, :n_sub] @ u_mat.conj().T
    child_flat = _flatten_band_set(child_all_bands)
    transported_flat = coeff.T @ child_flat
    transported = transported_flat.reshape((n_sub,) + child_all_bands.shape[1:])
    self_metrics = compare_subspaces(transported, transported, epsilon)
    return transported, {
        'overlap_matrix': overlap,
        'coefficients': coeff,
        'singular_values': singular_values,
        'min_singular_value': float(np.min(singular_values)) if singular_values.size else 0.0,
        'mean_singular_value': float(np.mean(singular_values)) if singular_values.size else 0.0,
        'self_orthogonality_error': self_metrics['projector_frobenius_distance'],
    }


def _neighbor_indices(ix: int, iy: int, n1: int, n2: int, periodic: bool) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jx = ix + dx
        jy = iy + dy
        if periodic:
            out.append((jx % n1, jy % n2))
        elif 0 <= jx < n1 and 0 <= jy < n2:
            out.append((jx, jy))
    return out


def analyze_registry_subspace_tracking(
    bloch_fields: np.ndarray,
    epsilon: np.ndarray,
    subspace_indices: list[int],
    *,
    seed: tuple[int, int] | None = None,
    periodic: bool = True,
) -> dict[str, Any]:
    """Run BFS non-Abelian subspace transport and summarize projector diagnostics."""
    n_reg1, n_reg2, n_all = bloch_fields.shape[:3]
    n_sub = len(subspace_indices)
    if n_sub == 0:
        raise ValueError('subspace_indices must be non-empty')
    if any(idx < 0 or idx >= n_all for idx in subspace_indices):
        raise ValueError('subspace_indices out of range for bloch_fields')

    seed_ix, seed_iy = seed if seed is not None else (n_reg1 // 2, n_reg2 // 2)
    transported = np.zeros((n_reg1, n_reg2, n_sub) + bloch_fields.shape[3:], dtype=np.complex64)
    visited = np.zeros((n_reg1, n_reg2), dtype=bool)

    edge_min_svs: list[float] = []
    edge_mean_svs: list[float] = []
    path_min_svs: list[float] = []
    path_fro_dists: list[float] = []
    raw_min_svs: list[float] = []
    raw_mean_svs: list[float] = []
    raw_fro_dists: list[float] = []
    node_metrics: list[dict[str, Any]] = []

    transported[seed_ix, seed_iy] = bloch_fields[seed_ix, seed_iy, subspace_indices]
    visited[seed_ix, seed_iy] = True

    queue: deque[tuple[int, int, int, int]] = deque()
    for nx, ny in _neighbor_indices(seed_ix, seed_iy, n_reg1, n_reg2, periodic):
        queue.append((nx, ny, seed_ix, seed_iy))

    while queue:
        ix, iy, pix, piy = queue.popleft()
        if visited[ix, iy]:
            continue

        parent_basis = transported[pix, piy]
        child_all = bloch_fields[ix, iy]
        child_eps = epsilon[ix, iy]
        transported_basis, step = transport_child_subspace(parent_basis, child_all, child_eps)
        transported[ix, iy] = transported_basis.astype(np.complex64)
        visited[ix, iy] = True

        edge_min_svs.append(step['min_singular_value'])
        edge_mean_svs.append(step['mean_singular_value'])

        raw_basis = child_all[subspace_indices]
        raw_cmp = compare_subspaces(transported_basis, raw_basis, child_eps)
        raw_min_svs.append(raw_cmp['min_singular_value'])
        raw_mean_svs.append(raw_cmp['mean_singular_value'])
        raw_fro_dists.append(raw_cmp['projector_frobenius_distance'])

        alt_parent_metrics: list[dict[str, Any]] = []
        for qx, qy in _neighbor_indices(ix, iy, n_reg1, n_reg2, periodic):
            if not visited[qx, qy] or (qx == pix and qy == piy):
                continue
            alt_basis, _ = transport_child_subspace(transported[qx, qy], child_all, child_eps)
            cmp = compare_subspaces(transported_basis, alt_basis, child_eps)
            path_min_svs.append(cmp['min_singular_value'])
            path_fro_dists.append(cmp['projector_frobenius_distance'])
            alt_parent_metrics.append({
                'parent': [int(qx), int(qy)],
                'min_singular_value': cmp['min_singular_value'],
                'projector_frobenius_distance': cmp['projector_frobenius_distance'],
            })

        node_metrics.append({
            'ix': int(ix),
            'iy': int(iy),
            'transport_min_singular_value': step['min_singular_value'],
            'transport_mean_singular_value': step['mean_singular_value'],
            'raw_subspace_min_singular_value': raw_cmp['min_singular_value'],
            'raw_subspace_frobenius_distance': raw_cmp['projector_frobenius_distance'],
            'alternate_parent_consistency': alt_parent_metrics,
        })

        for nx, ny in _neighbor_indices(ix, iy, n_reg1, n_reg2, periodic):
            if not visited[nx, ny]:
                queue.append((nx, ny, ix, iy))

    if not np.all(visited):
        raise RuntimeError('BFS subspace tracking did not visit the full registry grid')

    def _summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {'min': None, 'mean': None, 'p05': None, 'median': None, 'max': None}
        arr = np.asarray(values, dtype=float)
        return {
            'min': float(np.min(arr)),
            'mean': float(np.mean(arr)),
            'p05': float(np.quantile(arr, 0.05)),
            'median': float(np.median(arr)),
            'max': float(np.max(arr)),
        }

    return {
        'method': 'bfs_subspace_transport',
        'seed': [int(seed_ix), int(seed_iy)],
        'periodic': bool(periodic),
        'subspace_indices': [int(idx) for idx in subspace_indices],
        'grid_shape': [int(n_reg1), int(n_reg2)],
        'n_all_bands': int(n_all),
        'n_subspace_bands': int(n_sub),
        'transport_edge_min_singular_value': _summary(edge_min_svs),
        'transport_edge_mean_singular_value': _summary(edge_mean_svs),
        'path_consistency_min_singular_value': _summary(path_min_svs),
        'path_consistency_projector_frobenius_distance': _summary(path_fro_dists),
        'raw_subspace_fidelity_min_singular_value': _summary(raw_min_svs),
        'raw_subspace_fidelity_mean_singular_value': _summary(raw_mean_svs),
        'raw_subspace_fidelity_projector_frobenius_distance': _summary(raw_fro_dists),
        'node_metrics': node_metrics,
    }