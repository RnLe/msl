#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from subspace_tracking import analyze_registry_subspace_tracking


def _rotation(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=np.complex128)


def _make_synthetic_fields(case: str, n_grid: int = 8) -> tuple[np.ndarray, np.ndarray]:
    n_all = 4
    dim = 4
    fields = np.zeros((n_grid, n_grid, n_all, dim), dtype=np.complex128)
    epsilon = np.ones((n_grid, n_grid, dim), dtype=np.float64)
    eye = np.eye(dim, dtype=np.complex128)

    for ix in range(n_grid):
        for iy in range(n_grid):
            theta = 0.35 * np.sin(2.0 * np.pi * ix / n_grid) + 0.21 * np.cos(2.0 * np.pi * iy / n_grid)
            unitary = np.eye(dim, dtype=np.complex128)
            unitary[:2, :2] = _rotation(theta)
            if case == 'leaky':
                radius2 = (ix - n_grid / 2) ** 2 + (iy - n_grid / 2) ** 2
                leak = 1.10 * np.exp(-radius2 / 6.0)
                block_02 = np.array([
                    [np.cos(leak), -np.sin(leak)],
                    [np.sin(leak), np.cos(leak)],
                ], dtype=np.complex128)
                block_13 = np.array([
                    [np.cos(0.85 * leak), -np.sin(0.85 * leak)],
                    [np.sin(0.85 * leak), np.cos(0.85 * leak)],
                ], dtype=np.complex128)
                unitary[np.ix_([0, 2], [0, 2])] = block_02
                unitary[np.ix_([1, 3], [1, 3])] = block_13
            point_fields = unitary.T.copy()
            if (ix + iy) % 2 == 1:
                point_fields[[0, 1]] = point_fields[[1, 0]]
            fields[ix, iy] = point_fields
    return fields, epsilon


def _run_case(case: str) -> dict:
    fields, epsilon = _make_synthetic_fields(case)
    diag = analyze_registry_subspace_tracking(fields, epsilon, [0, 1], seed=(0, 0), periodic=True)
    return diag


def main() -> None:
    good = _run_case('smooth_swap')
    leaky = _run_case('leaky')

    good_edge = good['transport_edge_min_singular_value']['min']
    good_raw = good['raw_subspace_fidelity_min_singular_value']['min']
    leaky_raw = leaky['raw_subspace_fidelity_min_singular_value']['min']
    good_path = good['path_consistency_min_singular_value']['min']
    leaky_path = leaky['path_consistency_min_singular_value']['min']

    assert good_edge is not None and good_edge > 0.99, good_edge
    assert good_raw is not None and good_raw > 0.99, good_raw
    assert good_path is not None and good_path > 0.99, good_path
    assert leaky_raw is not None and leaky_raw < 0.90, leaky_raw
    assert leaky_path is not None and leaky_path < 0.95, leaky_path

    summary = {
        'smooth_swap': {
            'transport_edge_min_singular_value': good['transport_edge_min_singular_value'],
            'raw_subspace_fidelity_min_singular_value': good['raw_subspace_fidelity_min_singular_value'],
        },
        'leaky': {
            'transport_edge_min_singular_value': leaky['transport_edge_min_singular_value'],
            'raw_subspace_fidelity_min_singular_value': leaky['raw_subspace_fidelity_min_singular_value'],
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()