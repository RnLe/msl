"""Regression tests for the fd4 stencil used in Phase 2."""
from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phases.phase2_ea_operator import assemble_ea_operator  # noqa: E402


def _build_test_fields(Nx: int, Ny: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, Nx)
    y = np.linspace(-1.0, 1.0, Ny)
    R_grid = np.zeros((Nx, Ny, 2), dtype=float)
    R_grid[..., 0] = x[:, None]
    R_grid[..., 1] = y[None, :]

    V = 0.05 * rng.random((Nx, Ny))

    mass_tensor = np.zeros((Nx, Ny, 2, 2), dtype=float)
    # Positive-definite diagonal tensor with mild spatial variation.
    mass_tensor[..., 0, 0] = 3.0 + 0.2 * rng.random((Nx, Ny))
    mass_tensor[..., 1, 1] = 1.5 + 0.1 * rng.random((Nx, Ny))

    vg_field = np.zeros((Nx, Ny, 2), dtype=float)
    vg_field[..., 0] = 1e-3 * rng.standard_normal((Nx, Ny))
    vg_field[..., 1] = 2e-3 * rng.standard_normal((Nx, Ny))

    return R_grid, V, mass_tensor, vg_field


def _max_hermitian_gap(operator) -> float:
    diff = operator - operator.getH()
    if diff.nnz == 0:
        return 0.0
    return float(np.abs(diff.data).max())


def test_fd4_mass_term_is_hermitian_without_vg():
    R_grid, V, mass_tensor, _ = _build_test_fields(8, 8, seed=1)
    op = assemble_ea_operator(
        R_grid,
        mass_tensor,
        V,
        eta=0.02,
        vg_field=None,
        include_vg_term=False,
        fd_order=4,
    )
    assert _max_hermitian_gap(op) < 1e-12


def test_fd4_mass_term_is_hermitian_with_vg():
    R_grid, V, mass_tensor, vg_field = _build_test_fields(8, 8, seed=7)
    op = assemble_ea_operator(
        R_grid,
        mass_tensor,
        V,
        eta=0.02,
        vg_field=vg_field,
        include_vg_term=True,
        fd_order=4,
    )
    assert _max_hermitian_gap(op) < 1e-12
