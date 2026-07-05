#!/usr/bin/env python3
"""
Phase 1 (Blaze): Local Bloch problems at frozen registry — V4 Multi-Band Pipeline

Replaces the V3 MPB-based Phase 1 with Blaze's EAExtractor.
Blaze computes all EA Hamiltonian ingredients natively in Rust:
eigenvalues, velocity matrices, mass tensors, W matrices, Berry connection,
Born-Huang potential, registry derivatives, and the TE slow coefficient.

THE "UNIVERSAL MASTER MAP"
--------------------------
The output of this phase is the Universal Master Map of local band structure
over the configuration space δ (registry shift).
- This map is INDEPENDENT of the moiré twist angle or lattice constant.
- It encodes the fundamental response of the monolayer bands to interlayer stacking.
- Once computed (expensive), it can be used to simulate ANY moiré geometry (cheap)
  by simply sampling the map along the path δ(r) ≈ θ ẑ × r.

DATA STRUCTURES (V4):
- eigenvalues: (N_reg, N_retained) — local band energies λ_n = (ω/c)²
- velocity_x/y: (N_reg, N_retained, N_total) — velocity matrix elements
- mass_xx/xy/yx/yy: (N_reg, N_retained, N_retained) — inverse mass tensors
- w_xx/xy/yx/yy: (N_reg, N_retained, N_retained) — raw 2nd-derivative matrices
- berry_x/y: (N_reg, N_retained, N_retained) — Berry connection matrices
- born_huang: (N_reg, N_retained, N_retained) — Born-Huang potential
- born_huang_tensor_xx/xy/yx/yy: (N_reg, N_retained, N_retained) — directional BH tensor
- r_derivative_x/y: (N_reg, N_retained, N_total) — registry derivatives
- slow_coefficient: (N_reg, N_retained, N_retained) — TE slow coefficient (TE only)
- slow_coefficient_tensor_xx/xy/yx/yy: (N_reg, N_retained, N_retained) — directional TE slow-coefficient tensor
- registry: (N_reg, 2) — fractional registry coordinates

Convention reference: python/phasesV4/ea_pipeline_guide.md
"""

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml

try:
    from blaze import EAExtractor
except ImportError:  # renamed in blaze2d 99a4442 (May 2026)
    from blaze._native import OperatorDataExtractor as EAExtractor


# ==============================================================================
# Logging
# ==============================================================================

_log_fn = None

EXACT_TM_ARCHIVE_KEYS = (
    "tm_exact_epsilon",
    "tm_exact_inv_epsilon",
    "tm_exact_sqrt_epsilon",
    "tm_exact_inv_sqrt_epsilon",
    "tm_exact_fast_gradient_epsilon_x",
    "tm_exact_fast_gradient_epsilon_y",
    "tm_exact_fast_laplacian_epsilon",
    "tm_exact_slow_gradient_epsilon_x",
    "tm_exact_slow_gradient_epsilon_y",
    "tm_exact_slow_hessian_epsilon_xx",
    "tm_exact_slow_hessian_epsilon_xy",
    "tm_exact_slow_hessian_epsilon_yx",
    "tm_exact_slow_hessian_epsilon_yy",
    "tm_exact_mixed_hessian_epsilon_xx",
    "tm_exact_mixed_hessian_epsilon_xy",
    "tm_exact_mixed_hessian_epsilon_yx",
    "tm_exact_mixed_hessian_epsilon_yy",
    "tm_exact_hermitized_eigenvectors",
    "tm_exact_velocity_x",
    "tm_exact_velocity_y",
    "tm_exact_local_r_x",
    "tm_exact_local_r_y",
    "tm_exact_local_rr_x",
    "tm_exact_local_rr_y",
    "tm_exact_first_order_remainder",
    "tm_exact_direct_metric",
    "tm_exact_direct_b_x",
    "tm_exact_direct_b_y",
    "tm_exact_direct_gamma2",
    "tm_exact_mass_xx",
    "tm_exact_mass_xy",
    "tm_exact_mass_yx",
    "tm_exact_mass_yy",
    "tm_exact_epsilon_r_derivatives_x",
    "tm_exact_epsilon_r_derivatives_y",
    "tm_exact_epsilon_r_second_derivatives_x",
    "tm_exact_epsilon_r_second_derivatives_y",
    "tm_exact_rho_r_derivatives_x",
    "tm_exact_rho_r_derivatives_y",
    "tm_exact_rho_r_second_derivatives_x",
    "tm_exact_rho_r_second_derivatives_y",
    "tm_projected_fast_derivative_matrix_x",
    "tm_projected_fast_derivative_matrix_y",
    "tm_projected_fast_derivative_remote_matrix_x",
    "tm_projected_fast_derivative_remote_matrix_y",
    "tm_first_order_scalar_matrix",
    "tm_first_order_scalar_remote_matrix",
)


def log(message: str) -> None:
    if _log_fn is not None:
        _log_fn(message)
    else:
        print(message, flush=True)


# ==============================================================================
# Simple IO utilities (self-contained, no dependency on common.io_utils)
# ==============================================================================

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)


# ==============================================================================
# Lattice geometry
# ==============================================================================

def build_lattice_vectors(lattice_type: str, a: float = 1.0) -> list[list[float]]:
    """Build 2D lattice vectors in Cartesian coordinates."""
    if lattice_type == "square":
        return [[a, 0.0], [0.0, a]]
    elif lattice_type in ("hex", "triangular", "honeycomb"):
        return [[a, 0.0], [a * 0.5, a * math.sqrt(3.0) / 2.0]]
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")


def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build monolayer lattice basis matrix B = (a1 | a2), shape (2, 2).
    Column i is lattice vector a_i."""
    vecs = build_lattice_vectors(lattice_type, a)
    return np.array(vecs).T  # columns = lattice vectors


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """Compute moiré lattice basis: B_moire = (R(θ) - I)^{-1} @ B_mono"""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    Delta_R_inv = np.linalg.inv(R - np.eye(2))
    return Delta_R_inv @ B_mono


def frac_to_cartesian_k(
    k_frac: list[float], lattice_vectors: list[list[float]]
) -> np.ndarray:
    """Convert fractional BZ coordinates to Cartesian reciprocal-space."""
    A = np.asarray(lattice_vectors, dtype=float)
    return 2.0 * np.pi * np.linalg.inv(A) @ np.asarray(k_frac, dtype=float)


def compute_eta_physics(theta_rad: float) -> float:
    """Small parameter η = a / L_m ≈ 2 sin(θ/2)."""
    return 2.0 * math.sin(theta_rad / 2.0)


def compute_eta_geometric(theta_rad: float) -> float:
    """Geometric moiré scale factor η_geom = L_m / a."""
    return 1.0 / (2.0 * math.sin(theta_rad / 2.0))


# ==============================================================================
# Registry grid
# ==============================================================================

def build_registry_grid(n: int, offset: bool = False) -> np.ndarray:
    """Build a flat array of (n*n, 2) fractional registry points on [0, 1)².

    Args:
        n: grid points per direction.
        offset: if True, shift sample points to cell centres (Monkhorst-Pack style).

    Returns:
        (n*n, 2) array of fractional registry coordinates (s1, s2).
    """
    if offset:
        vals = (np.arange(n, dtype=float) + 0.5) / float(n)
    else:
        vals = np.arange(n, dtype=float) / float(n)
    grid = np.array([[s1, s2] for s2 in vals for s1 in vals], dtype=float)
    return grid


def build_registry_grid_2d(n1: int, n2: int) -> np.ndarray:
    """Build a 2D (n1, n2, 2) fractional registry grid on [0, 1)².

    This grid shape is useful for downstream moiré assembly where one
    needs a 2D index (i, j) → δ(i,j).
    """
    s1 = np.arange(n1, dtype=float) / n1
    s2 = np.arange(n2, dtype=float) / n2
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    return np.stack([S1, S2], axis=-1)


# ==============================================================================
# Blaze raw-result reshaping
# ==============================================================================

def tuples_to_matrix(
    values: list[tuple[float, float]], rows: int, cols: int
) -> np.ndarray:
    """Convert Blaze's flat (re, im) tuple list to a complex matrix."""
    flat = np.array([complex(re, im) for re, im in values], dtype=np.complex128)
    return flat.reshape(rows, cols)


def tuples_to_vector(values: list[tuple[float, float]]) -> np.ndarray:
    """Convert Blaze's flat (re, im) tuple list to a complex 1-D array."""
    return np.array([complex(re, im) for re, im in values], dtype=np.complex128)


def tuples_to_band_grid(values: list[list[tuple[float, float]]]) -> np.ndarray:
    """Convert Blaze's nested band -> grid tuple structure to a complex array."""
    return np.array(
        [[complex(re, im) for re, im in band] for band in values],
        dtype=np.complex128,
    )


def _get_exact_tm_entry(raw: dict, key: str):
    tm_exact = raw.get("tm_exact")
    if isinstance(tm_exact, dict) and key in tm_exact:
        return tm_exact.get(key)
    return raw.get(key)


def raw_to_blocks(raw: dict, n_retained: int, n_remote: int) -> dict:
    """Reshape a single Blaze extract result into usable numpy arrays.

    This mirrors the structure in ea_hamiltonian_registry_sweep.py but is
    decoupled from a specific case dict.
    """
    # Derive n_total from eigenvalue count (robust: works even if band_lo
    # is not present in checkpoint JSON).
    eigs = raw["eigenvalues"]
    n_total = len(eigs) if hasattr(eigs, '__len__') else n_retained + n_remote

    blocks: dict = {
        # Scalar / 1-D
        "eigenvalues": np.asarray(raw["eigenvalues"], dtype=float),
        "converged": bool(raw["converged"]),
        "n_iterations": int(raw["n_iterations"]),
        "solve_time_seconds": float(raw["solve_time_seconds"]),
        "extract_time_seconds": float(raw["extract_time_seconds"]),
        "registry": np.asarray(raw["registry"], dtype=float),
        # Velocity matrices  (N_ret × N_total)
        "velocity_x": tuples_to_matrix(raw["velocity_matrices_x"], n_retained, n_total),
        "velocity_y": tuples_to_matrix(raw["velocity_matrices_y"], n_retained, n_total),
        # W matrices  (N_ret × N_ret)
        "w_xx": tuples_to_matrix(raw["w_matrices_xx"], n_retained, n_retained),
        "w_xy": tuples_to_matrix(raw["w_matrices_xy"], n_retained, n_retained),
        "w_yx": tuples_to_matrix(raw["w_matrices_yx"], n_retained, n_retained),
        "w_yy": tuples_to_matrix(raw["w_matrices_yy"], n_retained, n_retained),
        # Mass tensor (N_ret × N_ret)  — Löwdin-corrected inverse mass
        "mass_xx": tuples_to_matrix(raw["mass_tensor_inv_xx"], n_retained, n_retained),
        "mass_xy": tuples_to_matrix(raw["mass_tensor_inv_xy"], n_retained, n_retained),
        "mass_yx": tuples_to_matrix(raw["mass_tensor_inv_yx"], n_retained, n_retained),
        "mass_yy": tuples_to_matrix(raw["mass_tensor_inv_yy"], n_retained, n_retained),
    }

    # Registry derivatives (N_ret × N_total) — optional
    if raw.get("r_derivative_matrices_x") is not None:
        blocks["r_derivative_x"] = tuples_to_matrix(
            raw["r_derivative_matrices_x"], n_retained, n_total
        )
        blocks["r_derivative_y"] = tuples_to_matrix(
            raw["r_derivative_matrices_y"], n_retained, n_total
        )

    # Berry connection (N_ret × N_ret) — optional
    if raw.get("has_berry_connection", False):
        blocks["berry_x"] = tuples_to_matrix(
            raw["berry_connection_x"], n_retained, n_retained
        )
        blocks["berry_y"] = tuples_to_matrix(
            raw["berry_connection_y"], n_retained, n_retained
        )

    # Born-Huang potential (N_ret × N_ret) — optional
    if raw.get("has_born_huang", False):
        blocks["born_huang"] = tuples_to_matrix(
            raw["born_huang"], n_retained, n_retained
        )
        for suffix in ("xx", "xy", "yx", "yy"):
            key = f"born_huang_tensor_{suffix}"
            if raw.get(key) is not None:
                blocks[key] = tuples_to_matrix(raw[key], n_retained, n_retained)

    # TE slow coefficient (N_ret × N_ret) — optional, TE only
    if raw.get("has_slow_coefficient", False):
        blocks["slow_coefficient"] = tuples_to_matrix(
            raw["slow_coefficient_potential"], n_retained, n_retained
        )
        for suffix in ("xx", "xy", "yx", "yy"):
            key = f"slow_coefficient_tensor_{suffix}"
            if raw.get(key) is not None:
                blocks[key] = tuples_to_matrix(raw[key], n_retained, n_retained)

    # Metric derivatives (N_ret × N_total) — optional, TM
    if raw.get("has_metric_derivatives", False):
        blocks["metric_derivative_x"] = tuples_to_matrix(
            raw["metric_derivative_matrices_x"], n_retained, n_total
        )
        blocks["metric_derivative_y"] = tuples_to_matrix(
            raw["metric_derivative_matrices_y"], n_retained, n_total
        )

    # Overlap matrix (N_ret × N_ret) — optional
    if raw.get("overlap_matrix") is not None:
        blocks["overlap"] = tuples_to_matrix(
            raw["overlap_matrix"], n_retained, n_retained
        )

    # Exact TM field-level export — optional, TM only
    if raw.get("has_exact_tm_fields", False) or isinstance(raw.get("tm_exact"), dict):
        for key in (
            "tm_exact_epsilon",
            "tm_exact_inv_epsilon",
            "tm_exact_sqrt_epsilon",
            "tm_exact_inv_sqrt_epsilon",
            "tm_exact_fast_gradient_epsilon_x",
            "tm_exact_fast_gradient_epsilon_y",
            "tm_exact_fast_laplacian_epsilon",
            "tm_exact_slow_gradient_epsilon_x",
            "tm_exact_slow_gradient_epsilon_y",
            "tm_exact_slow_hessian_epsilon_xx",
            "tm_exact_slow_hessian_epsilon_xy",
            "tm_exact_slow_hessian_epsilon_yx",
            "tm_exact_slow_hessian_epsilon_yy",
            "tm_exact_mixed_hessian_epsilon_xx",
            "tm_exact_mixed_hessian_epsilon_xy",
            "tm_exact_mixed_hessian_epsilon_yx",
            "tm_exact_mixed_hessian_epsilon_yy",
        ):
            value = _get_exact_tm_entry(raw, key)
            if value is not None:
                blocks[key] = np.asarray(value, dtype=np.float64)

        hermitized = _get_exact_tm_entry(raw, "hermitized_eigenvectors")
        if hermitized is None:
            hermitized = raw.get("tm_exact_hermitized_eigenvectors")
        if hermitized is not None:
            blocks["tm_exact_hermitized_eigenvectors"] = tuples_to_band_grid(hermitized)

        for block_key, exact_key, rows, cols in (
            ("tm_exact_velocity_x", "velocity_matrices_x", n_total, n_total),
            ("tm_exact_velocity_y", "velocity_matrices_y", n_total, n_total),
            ("tm_exact_local_r_x", "local_r_derivative_matrices_x", n_total, n_total),
            ("tm_exact_local_r_y", "local_r_derivative_matrices_y", n_total, n_total),
            ("tm_exact_local_rr_x", "local_r_second_derivative_matrices_x", n_total, n_total),
            ("tm_exact_local_rr_y", "local_r_second_derivative_matrices_y", n_total, n_total),
            ("tm_exact_first_order_remainder", "first_order_remainder", n_total, n_total),
            ("tm_exact_direct_metric", "direct_metric", n_retained, n_retained),
            ("tm_exact_direct_b_x", "direct_b_matrix_x", n_retained, n_retained),
            ("tm_exact_direct_b_y", "direct_b_matrix_y", n_retained, n_retained),
            ("tm_exact_direct_gamma2", "direct_gamma2", n_retained, n_retained),
            ("tm_exact_mass_xx", "mass_tensor_inv_xx", n_retained, n_retained),
            ("tm_exact_mass_xy", "mass_tensor_inv_xy", n_retained, n_retained),
            ("tm_exact_mass_yx", "mass_tensor_inv_yx", n_retained, n_retained),
            ("tm_exact_mass_yy", "mass_tensor_inv_yy", n_retained, n_retained),
        ):
            value = _get_exact_tm_entry(raw, exact_key)
            if value is not None:
                blocks[block_key] = tuples_to_matrix(value, rows, cols)

        for block_key, exact_key in (
            ("tm_exact_epsilon_r_derivatives_x", "epsilon_r_derivatives_x"),
            ("tm_exact_epsilon_r_derivatives_y", "epsilon_r_derivatives_y"),
            ("tm_exact_epsilon_r_second_derivatives_x", "epsilon_r_second_derivatives_x"),
            ("tm_exact_epsilon_r_second_derivatives_y", "epsilon_r_second_derivatives_y"),
            ("tm_exact_rho_r_derivatives_x", "rho_r_derivatives_x"),
            ("tm_exact_rho_r_derivatives_y", "rho_r_derivatives_y"),
            ("tm_exact_rho_r_second_derivatives_x", "rho_r_second_derivatives_x"),
            ("tm_exact_rho_r_second_derivatives_y", "rho_r_second_derivatives_y"),
        ):
            value = _get_exact_tm_entry(raw, exact_key)
            if value is not None:
                blocks[block_key] = np.asarray(value, dtype=np.float64)

        if raw.get("tm_projected_fast_derivative_matrix_x") is not None:
            blocks["tm_projected_fast_derivative_matrix_x"] = tuples_to_matrix(
                raw["tm_projected_fast_derivative_matrix_x"], n_retained, n_retained
            )
            blocks["tm_projected_fast_derivative_matrix_y"] = tuples_to_matrix(
                raw["tm_projected_fast_derivative_matrix_y"], n_retained, n_retained
            )

        if raw.get("tm_projected_fast_derivative_remote_matrix_x") is not None:
            blocks["tm_projected_fast_derivative_remote_matrix_x"] = tuples_to_matrix(
                raw["tm_projected_fast_derivative_remote_matrix_x"], n_remote, n_retained
            )
            blocks["tm_projected_fast_derivative_remote_matrix_y"] = tuples_to_matrix(
                raw["tm_projected_fast_derivative_remote_matrix_y"], n_remote, n_retained
            )

        if raw.get("tm_first_order_scalar_matrix") is not None:
            blocks["tm_first_order_scalar_matrix"] = tuples_to_matrix(
                raw["tm_first_order_scalar_matrix"], n_retained, n_retained
            )

        if raw.get("tm_first_order_scalar_remote_matrix") is not None:
            blocks["tm_first_order_scalar_remote_matrix"] = tuples_to_matrix(
                raw["tm_first_order_scalar_remote_matrix"], n_remote, n_retained
            )

    # Exact TE downfolding remainder blocks — optional, TE only
    if raw.get("has_exact_te_downfolding", False):
        if raw.get("xi_scalar_first_order") is not None:
            blocks["xi_scalar_first_order"] = tuples_to_matrix(
                raw["xi_scalar_first_order"], n_retained, n_retained
            )
        if raw.get("kappa_matrix_x") is not None:
            blocks["kappa_matrix_x"] = tuples_to_matrix(
                raw["kappa_matrix_x"], n_retained, n_retained
            )
            blocks["kappa_matrix_y"] = tuples_to_matrix(
                raw["kappa_matrix_y"], n_retained, n_retained
            )
        if raw.get("weighted_leakage_scalar") is not None:
            blocks["weighted_leakage_scalar"] = tuples_to_matrix(
                raw["weighted_leakage_scalar"], n_retained, n_retained
            )
        if raw.get("lowdin_t_matrix_x") is not None:
            blocks["lowdin_t_matrix_x"] = tuples_to_matrix(
                raw["lowdin_t_matrix_x"], n_remote, n_retained
            )
            blocks["lowdin_t_matrix_y"] = tuples_to_matrix(
                raw["lowdin_t_matrix_y"], n_remote, n_retained
            )
        if raw.get("lowdin_r_matrix") is not None:
            blocks["lowdin_r_matrix"] = tuples_to_matrix(
                raw["lowdin_r_matrix"], n_remote, n_retained
            )

    # Eigenvectors — store as flat complex for HDF5 archival
    if raw.get("eigenvectors") is not None:
        blocks["eigenvectors_raw"] = raw["eigenvectors"]
        blocks["grid_dims"] = raw.get("grid_dims")

    return blocks


# ==============================================================================
# Sanity checks
# ==============================================================================

def hermiticity_error(M: np.ndarray) -> float:
    return float(np.max(np.abs(M - M.conj().T)))


def dagger_consistency_error(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.max(np.abs(A - B.conj().T)))


def positive_min(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.min(values))


def validate_exact_tm_stacked(stacked: dict) -> dict:
    """Compute TM-exact field validation metrics over the stacked Phase 1 output.

    The gate key is ``tm_exact_velocity_x`` — the first operator-level matrix
    that the Rust serializer always emits when exact-TM extraction is active.
    The epsilon-family primitives (epsilon, inv_epsilon, sqrt_epsilon, …) are
    optional validation diagnostics and do NOT gate ``has_exact_tm_fields``.
    """
    metrics: dict = {
        "has_exact_tm_fields": False,
        # -- Operator-level structural metrics --
        "tm_exact_velocity_x_hermiticity": None,
        "tm_exact_velocity_y_hermiticity": None,
        "tm_exact_direct_metric_hermiticity": None,
        "tm_exact_direct_metric_min_eigenvalue": None,
        "tm_exact_direct_b_x_hermiticity": None,
        "tm_exact_direct_b_y_hermiticity": None,
        "tm_exact_direct_gamma2_hermiticity": None,
        "tm_exact_mass_xx_hermiticity": None,
        "tm_exact_mass_yy_hermiticity": None,
        "tm_exact_mass_xy_yx_dagger": None,
        "tm_exact_first_order_remainder_abs_max": None,
        # -- Eigenvector metrics --
        "tm_exact_hermitized_diag_norm_error": None,
        "tm_exact_hermitized_offdiag_norm": None,
        # -- Optional epsilon-family diagnostics --
        "tm_exact_epsilon_min": None,
        "tm_exact_inv_epsilon_min": None,
        "tm_exact_identity_error": None,
        "tm_exact_sqrt_error": None,
        "tm_exact_inv_sqrt_error": None,
        "tm_exact_fast_grad_mean_abs": None,
        "tm_exact_fast_laplacian_mean_abs": None,
        "tm_exact_slow_hessian_xy_yx_error": None,
        "tm_exact_mixed_hessian_xy_yx_error": None,
        "tm_first_order_scalar_hermiticity": None,
    }

    # Gate on an operator key that IS produced by the current Rust serializer
    if "tm_exact_velocity_x" not in stacked:
        return metrics

    metrics["has_exact_tm_fields"] = True

    # ---- Operator-level structural checks ----
    def _herm_err(arr):
        return float(np.max(np.abs(arr - np.swapaxes(arr.conj(), -2, -1))))

    for key_suffix, metric_key in [
        ("velocity_x", "tm_exact_velocity_x_hermiticity"),
        ("velocity_y", "tm_exact_velocity_y_hermiticity"),
        ("direct_metric", "tm_exact_direct_metric_hermiticity"),
        ("direct_b_x", "tm_exact_direct_b_x_hermiticity"),
        ("direct_b_y", "tm_exact_direct_b_y_hermiticity"),
        ("direct_gamma2", "tm_exact_direct_gamma2_hermiticity"),
        ("mass_xx", "tm_exact_mass_xx_hermiticity"),
        ("mass_yy", "tm_exact_mass_yy_hermiticity"),
    ]:
        full_key = f"tm_exact_{key_suffix}"
        if full_key in stacked:
            arr = np.asarray(stacked[full_key], dtype=np.complex128)
            metrics[metric_key] = _herm_err(arr)

    if "tm_exact_direct_metric" in stacked:
        g = np.asarray(stacked["tm_exact_direct_metric"], dtype=np.complex128)
        eigvals = np.linalg.eigvalsh(g)
        metrics["tm_exact_direct_metric_min_eigenvalue"] = float(eigvals.min())

    if "tm_exact_mass_xy" in stacked and "tm_exact_mass_yx" in stacked:
        mxy = np.asarray(stacked["tm_exact_mass_xy"], dtype=np.complex128)
        myx = np.asarray(stacked["tm_exact_mass_yx"], dtype=np.complex128)
        metrics["tm_exact_mass_xy_yx_dagger"] = float(
            np.max(np.abs(mxy - np.swapaxes(myx.conj(), -2, -1)))
        )

    if "tm_exact_first_order_remainder" in stacked:
        metrics["tm_exact_first_order_remainder_abs_max"] = float(
            np.max(np.abs(stacked["tm_exact_first_order_remainder"]))
        )

    # ---- Eigenvector orthonormality ----
    # Stored with per-band Euclidean norm ~1 (no 1/N_grid factor needed).
    if "tm_exact_hermitized_eigenvectors" in stacked:
        eigvecs = np.asarray(stacked["tm_exact_hermitized_eigenvectors"], dtype=np.complex128)
        if eigvecs.ndim == 3 and eigvecs.shape[-1] > 0:
            n_pts, n_bands = eigvecs.shape[0], eigvecs.shape[1]
            max_diag_err = 0.0
            max_offdiag = 0.0
            chunk = min(256, n_pts)
            for s in range(0, n_pts, chunk):
                e = min(s + chunk, n_pts)
                blk = eigvecs[s:e]
                overlap = np.einsum("pbg,pcg->pbc", blk.conj(), blk)
                eye = np.eye(n_bands, dtype=np.complex128)[None, :, :]
                delta = overlap - eye
                diag = np.diagonal(delta, axis1=-2, axis2=-1)
                offdiag = delta.copy()
                diag_idx = np.arange(n_bands)
                offdiag[:, diag_idx, diag_idx] = 0.0
                max_diag_err = max(max_diag_err, float(np.max(np.abs(diag))))
                max_offdiag = max(max_offdiag, float(np.max(np.abs(offdiag))))
            metrics["tm_exact_hermitized_diag_norm_error"] = max_diag_err
            metrics["tm_exact_hermitized_offdiag_norm"] = max_offdiag

    # ---- Optional epsilon-family diagnostics ----
    if "tm_exact_epsilon" in stacked:
        epsilon = np.asarray(stacked["tm_exact_epsilon"], dtype=np.float64)
        inv_epsilon = np.asarray(stacked["tm_exact_inv_epsilon"], dtype=np.float64)
        sqrt_epsilon = np.asarray(stacked["tm_exact_sqrt_epsilon"], dtype=np.float64)
        inv_sqrt_epsilon = np.asarray(stacked["tm_exact_inv_sqrt_epsilon"], dtype=np.float64)

        metrics["tm_exact_epsilon_min"] = positive_min(epsilon)
        metrics["tm_exact_inv_epsilon_min"] = positive_min(inv_epsilon)
        metrics["tm_exact_identity_error"] = float(
            np.max(np.abs(epsilon * inv_epsilon - 1.0))
        )
        metrics["tm_exact_sqrt_error"] = float(
            np.max(np.abs(sqrt_epsilon * sqrt_epsilon - epsilon))
        )
        metrics["tm_exact_inv_sqrt_error"] = float(
            np.max(np.abs(sqrt_epsilon * inv_sqrt_epsilon - 1.0))
        )

    grad_errors = []
    for key in (
        "tm_exact_fast_gradient_epsilon_x",
        "tm_exact_fast_gradient_epsilon_y",
    ):
        if key in stacked:
            grad_errors.append(np.mean(np.abs(stacked[key]), axis=-1))
    if grad_errors:
        metrics["tm_exact_fast_grad_mean_abs"] = float(
            np.max(np.stack(grad_errors, axis=0))
        )

    if "tm_exact_fast_laplacian_epsilon" in stacked:
        metrics["tm_exact_fast_laplacian_mean_abs"] = float(
            np.max(np.mean(np.abs(stacked["tm_exact_fast_laplacian_epsilon"]), axis=-1))
        )

    if (
        "tm_exact_slow_hessian_epsilon_xy" in stacked
        and "tm_exact_slow_hessian_epsilon_yx" in stacked
    ):
        metrics["tm_exact_slow_hessian_xy_yx_error"] = float(
            np.max(
                np.abs(
                    stacked["tm_exact_slow_hessian_epsilon_xy"]
                    - stacked["tm_exact_slow_hessian_epsilon_yx"]
                )
            )
        )

    if (
        "tm_exact_mixed_hessian_epsilon_xy" in stacked
        and "tm_exact_mixed_hessian_epsilon_yx" in stacked
    ):
        metrics["tm_exact_mixed_hessian_xy_yx_error"] = float(
            np.max(
                np.abs(
                    stacked["tm_exact_mixed_hessian_epsilon_xy"]
                    - stacked["tm_exact_mixed_hessian_epsilon_yx"]
                )
            )
        )

    if "tm_first_order_scalar_matrix" in stacked:
        scalar = np.asarray(stacked["tm_first_order_scalar_matrix"], dtype=np.complex128)
        if scalar.ndim >= 3:
            metrics["tm_first_order_scalar_hermiticity"] = float(
                np.max([hermiticity_error(block) for block in scalar.reshape(-1, *scalar.shape[-2:])])
            )

    return metrics


def compute_sanity_summary(stacked: dict, n_retained: int) -> dict:
    """Compute summary diagnostics over a stacked sweep."""
    summary: dict = {
        "all_finite": True,
        "all_converged": bool(np.all(stacked["converged"])),
        "n_points": int(stacked["eigenvalues"].shape[0]),
        "n_retained": n_retained,
        "mean_iterations": float(np.mean(stacked["n_iterations"])),
        "max_iterations": int(np.max(stacked["n_iterations"])),
        "mean_solve_time_s": float(np.mean(stacked["solve_time_seconds"])),
        "mean_extract_time_s": float(np.mean(stacked["extract_time_seconds"])),
        "eigenvalue_range": [
            float(np.min(stacked["eigenvalues"])),
            float(np.max(stacked["eigenvalues"])),
        ],
    }

    # Check finiteness across all numeric arrays
    for key, val in stacked.items():
        if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.number):
            if not np.all(np.isfinite(val)):
                summary["all_finite"] = False
                break

    # Hermiticity checks on matrix blocks
    def _max_herm(key):
        if key not in stacked:
            return None
        return float(max(hermiticity_error(m) for m in stacked[key]))

    def _max_dag(key_a, key_b):
        if key_a not in stacked or key_b not in stacked:
            return None
        return float(
            max(
                dagger_consistency_error(a, b)
                for a, b in zip(stacked[key_a], stacked[key_b])
            )
        )

    summary["max_w_xx_hermiticity"] = _max_herm("w_xx")
    summary["max_w_yy_hermiticity"] = _max_herm("w_yy")
    summary["max_w_xy_yx_dagger"] = _max_dag("w_xy", "w_yx")
    summary["max_mass_xx_hermiticity"] = _max_herm("mass_xx")
    summary["max_mass_yy_hermiticity"] = _max_herm("mass_yy")
    summary["max_mass_xy_yx_dagger"] = _max_dag("mass_xy", "mass_yx")
    summary["max_born_huang_hermiticity"] = _max_herm("born_huang")
    summary["max_born_huang_tensor_xx_hermiticity"] = _max_herm("born_huang_tensor_xx")
    summary["max_born_huang_tensor_yy_hermiticity"] = _max_herm("born_huang_tensor_yy")
    summary["max_born_huang_tensor_xy_yx_dagger"] = _max_dag(
        "born_huang_tensor_xy", "born_huang_tensor_yx"
    )
    summary["max_berry_x_hermiticity"] = _max_herm("berry_x")
    summary["max_berry_y_hermiticity"] = _max_herm("berry_y")
    summary["max_slow_coeff_hermiticity"] = _max_herm("slow_coefficient")
    summary["max_slow_coeff_tensor_xx_hermiticity"] = _max_herm("slow_coefficient_tensor_xx")
    summary["max_slow_coeff_tensor_yy_hermiticity"] = _max_herm("slow_coefficient_tensor_yy")
    summary["max_slow_coeff_tensor_xy_yx_dagger"] = _max_dag(
        "slow_coefficient_tensor_xy", "slow_coefficient_tensor_yx"
    )

    summary.update(validate_exact_tm_stacked(stacked))

    return summary


# ==============================================================================
# Stacking helpers
# ==============================================================================

def stack_results(results: list[dict]) -> dict:
    """Stack a list of per-point block dicts into arrays with leading index."""
    first = results[0]
    stacked: dict = {}
    for key, val in first.items():
        if key in ("eigenvectors_raw", "grid_dims"):
            # Keep as list — these are variable-format
            stacked[key] = [r[key] for r in results]
            continue
        if isinstance(val, np.ndarray):
            stacked[key] = np.stack([r[key] for r in results], axis=0)
        else:
            stacked[key] = np.asarray([r[key] for r in results])
    if stacked is None:
        raise RuntimeError("Blaze sweep returned no Phase 1 data")

    return stacked


# ==============================================================================
# Case definitions
# ==============================================================================

# These mirror the canonical cases in ea_hamiltonian_registry_sweep.py but are
# organized as a registry that can be extended or overridden via YAML config.

CANONICAL_CASES: dict[str, dict] = {
    "square": {
        "label": "Square rods in air",
        "lattice_type": "square",
        "lattice_vectors": [[1.0, 0.0], [0.0, 1.0]],
        "base_atoms": [
            {"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 8.9},
            {"pos": [0.0, 0.0], "radius": 0.2, "eps_inside": 8.9},
        ],
        "eps_bg": 1.0,
        "k0_frac": [0.5, 0.5],
        "k0_label": "M",
    },
    "hex": {
        "label": "Hex holes in dielectric",
        "lattice_type": "hex",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        "base_atoms": [
            {"pos": [0.0, 0.0], "radius": 0.48, "eps_inside": 1.0},
            {"pos": [0.0, 0.0], "radius": 0.48, "eps_inside": 1.0},
        ],
        "eps_bg": 13.0,
        "k0_frac": [1.0 / 3.0, 1.0 / 3.0],
        "k0_label": "K",
    },
    "honeycomb": {
        "label": "Honeycomb holes in dielectric",
        "lattice_type": "honeycomb",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        "base_atoms": [
            {"pos": [0.0, 0.0], "radius": 0.25, "eps_inside": 1.0},
            {"pos": [1.0 / 3.0, 1.0 / 3.0], "radius": 0.25, "eps_inside": 1.0},
        ],
        "eps_bg": 12.0,
        "k0_frac": [1.0 / 3.0, 1.0 / 3.0],
        "k0_label": "K",
    },
    "honeycomb_dirac": {
        "label": "Honeycomb air rods — Dirac cone at K (TM bands 3-4)",
        "lattice_type": "honeycomb",
        "lattice_vectors": [[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0]],
        "base_atoms": [
            {"pos": [0.0, 0.0], "radius": 0.408425, "eps_inside": 1.0},
            {"pos": [1.0 / 3.0, 1.0 / 3.0], "radius": 0.408425, "eps_inside": 1.0},
        ],
        "eps_bg": 3.77658,
        "k0_frac": [1.0 / 3.0, 1.0 / 3.0],
        "k0_label": "K",
    },
}


# ==============================================================================
# Core sweep logic
# ==============================================================================

def run_blaze_registry_sweep(
    case: dict,
    polarization: str,
    registry_points: np.ndarray,
    *,
    resolution: int = 32,
    band_lo: int = 0,
    n_retained: int = 4,
    n_remote: int = 12,
    fd_step: float = 0.005,
    tolerance: float = 1e-8,
    max_iterations: int = 700,
    threads: int = 8,
    compute_born_huang: bool = True,
    compute_r_derivatives: bool = True,
    compute_exact_tm_fields: bool = False,
    smoothing: bool = True,
) -> dict:
    """Run Blaze EA extraction over a registry grid.

    Uses EAExtractor.extract_registry_sweep which parallelises in Rust.
    Progress is reported natively from Rust via stderr (no GIL issues).
    Max-iterations warnings are also printed from Rust.

    Returns:
        Stacked dict of numpy arrays (already stacked, memory-efficient).
    """
    k0 = frac_to_cartesian_k(case["k0_frac"], case["lattice_vectors"])
    compute_slow = polarization == "TE"
    total = len(registry_points)

    log(f"  Blaze sweep: {total} points, pol={polarization}, "
        f"res={resolution}, N_ret={n_retained}, N_rem={n_remote}, threads={threads}")

    t0 = time.time()
    raw_results = EAExtractor.extract_registry_sweep(
        lattice_vectors=case["lattice_vectors"],
        atoms=case["base_atoms"],
        eps_bg=case["eps_bg"],
        registries=registry_points.tolist(),
        k0=k0.tolist(),
        polarization=polarization,
        resolution=resolution,
        band_lo=band_lo,
        n_retained=n_retained,
        n_remote=n_remote,
        compute_born_huang=compute_born_huang,
        compute_slow_coefficient=compute_slow,
        atom_index=1,
        fd_step=fd_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
        compute_r_derivatives=compute_r_derivatives,
        smoothing=smoothing,
        threads=threads,
    )
    elapsed = time.time() - t0
    log(f"  Blaze sweep completed in {elapsed:.2f}s "
        f"({elapsed / total * 1000:.1f} ms/point)")

    # Memory-efficient: convert and stack incrementally, dropping raw data
    # as we go.  At res=128, eigenvectors alone are ~9 MB/point and would
    # OOM for large grids (e.g. 9216 points ≈ 87 GB of eigenvectors).
    stacked: dict | None = None
    for i in range(len(raw_results)):
        # Drop eigenvectors — not saved to HDF5 and extremely large
        raw_results[i].pop("eigenvectors", None)
        raw_results[i].pop("grid_dims", None)

        block = raw_to_blocks(raw_results[i], n_retained, n_remote)
        raw_results[i] = None  # release Rust dict

        # Drop eigenvectors from block too
        block.pop("eigenvectors_raw", None)
        block.pop("grid_dims", None)

        if stacked is None:
            # Pre-allocate arrays on first point
            stacked = {}
            for key, val in block.items():
                if isinstance(val, np.ndarray):
                    stacked[key] = np.empty((total, *val.shape), dtype=val.dtype)
                else:
                    stacked[key] = np.empty(total, dtype=type(val))

        for key, val in block.items():
            stacked[key][i] = val

    if stacked is None:
        raise RuntimeError("No checkpoint rows were loaded for the Phase 1 sweep")

    return stacked


# ==============================================================================
# Checkpointed sweep (OOM-safe, resumable)
# ==============================================================================

def run_blaze_registry_sweep_checkpointed(
    case: dict,
    polarization: str,
    registry_points: np.ndarray,
    *,
    checkpoint_dir: str | Path,
    resolution: int = 32,
    band_lo: int = 0,
    n_retained: int = 4,
    n_remote: int = 12,
    fd_step: float = 0.005,
    tolerance: float = 1e-8,
    max_iterations: int = 700,
    threads: int = 8,
    compute_born_huang: bool = True,
    compute_r_derivatives: bool = True,
    compute_exact_tm_fields: bool = False,
    smoothing: bool = True,
    archive_exact_tm_hermitized_eigenvectors: bool = False,
) -> dict:
    """Run Blaze EA extraction with Rust-side checkpoint streaming.

    Processes one row at a time, writes each row to a checkpoint JSON file,
    and frees the memory.  If checkpoint files already exist they are
    skipped (resume semantics).  Never holds more than one row in memory.

    Returns:
        Summary dict from Rust with n_rows, n_points, n_skipped, etc.
    """
    k0 = frac_to_cartesian_k(case["k0_frac"], case["lattice_vectors"])
    compute_slow = polarization == "TE"
    total = len(registry_points)
    n_per_row = int(round(total ** 0.5))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log(f"  Blaze checkpointed sweep: {total} points, pol={polarization}, "
        f"res={resolution}, N_ret={n_retained}, N_rem={n_remote}, "
        f"threads={threads}, row_size={n_per_row}")
    log(f"  Checkpoint dir: {checkpoint_dir}")

    t0 = time.time()
    summary = EAExtractor.extract_registry_sweep_checkpointed(
        lattice_vectors=case["lattice_vectors"],
        atoms=case["base_atoms"],
        eps_bg=case["eps_bg"],
        registries=registry_points.tolist(),
        k0=k0.tolist(),
        polarization=polarization,
        resolution=resolution,
        checkpoint_dir=str(checkpoint_dir),
        band_lo=band_lo,
        n_retained=n_retained,
        n_remote=n_remote,
        compute_born_huang=compute_born_huang,
        compute_slow_coefficient=compute_slow,
        atom_index=1,
        fd_step=fd_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
        compute_r_derivatives=compute_r_derivatives,
        smoothing=smoothing,
        threads=threads,
        n_per_row=n_per_row,
        skip_eigenvectors=not archive_exact_tm_hermitized_eigenvectors,
    )
    elapsed = time.time() - t0
    log(f"  Blaze checkpointed sweep completed in {elapsed:.2f}s "
        f"({summary['n_skipped']} rows resumed, "
        f"{summary['n_computed']} rows computed, "
        f"{summary['n_unconverged']} unconverged)")

    return summary


def load_and_stack_checkpoints(
    checkpoint_dir: str | Path,
    n_retained: int,
    n_remote: int,
) -> dict:
    """Load checkpoint files one row at a time and build stacked arrays.

    This never holds more than one row of Python dicts in memory — each
    row is converted to numpy blocks and then discarded immediately.

    Returns:
        Stacked dict of numpy arrays (same schema as ``run_blaze_registry_sweep``).
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Read sweep metadata
    meta_path = checkpoint_dir / "sweep_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    total = meta["total_points"]

    # Discover row files
    row_files = sorted(checkpoint_dir.glob("row_*.json"))
    row_files = [f for f in row_files if not f.name.endswith(".tmp")]

    log(f"  Loading {len(row_files)} checkpoint rows ({total} points total)")

    stacked: dict | None = None
    global_idx = 0

    for row_file in row_files:
        # Load one row via Rust (JSON → Python dicts)
        row_dicts = EAExtractor.load_checkpoint_row(str(row_file))

        for raw in row_dicts:
            # Drop eigenvectors (not needed and not saved by default)
            if hasattr(raw, 'pop'):
                raw.pop("eigenvectors", None)
                raw.pop("grid_dims", None)

            block = raw_to_blocks(raw, n_retained, n_remote)
            block.pop("eigenvectors_raw", None)
            block.pop("grid_dims", None)

            if stacked is None:
                stacked = {}
                for key, val in block.items():
                    if isinstance(val, np.ndarray):
                        stacked[key] = np.empty(
                            (total, *val.shape), dtype=val.dtype
                        )
                    else:
                        stacked[key] = np.empty(total, dtype=type(val))

            for key, val in block.items():
                stacked[key][global_idx] = val
            global_idx += 1

        # row_dicts is dropped here — memory freed

    if global_idx != total:
        log(f"  WARNING: loaded {global_idx} points but expected {total}")

    if stacked is None:
        raise RuntimeError(f"No checkpoint rows were loaded from {checkpoint_dir}")

    return stacked


# ==============================================================================
# Moiré metadata
# ==============================================================================

def compute_moire_metadata(
    lattice_type: str, a: float, theta_deg: float
) -> dict:
    """Compute moiré-related quantities from twist angle and lattice info."""
    theta_rad = math.radians(theta_deg)
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    eta = compute_eta_physics(theta_rad)
    eta_geom = compute_eta_geometric(theta_rad)
    moire_length = eta_geom * a

    return {
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "B_mono": B_mono,
        "B_moire": B_moire,
        "eta": eta,
        "eta_geom": eta_geom,
        "moire_length": moire_length,
    }


# ==============================================================================
# HDF5 output
# ==============================================================================

def save_sweep_h5(
    path: Path,
    stacked: dict,
    case: dict,
    polarization: str,
    config: dict,
    moire_meta: dict | None = None,
    sanity: dict | None = None,
) -> None:
    """Save the full sweep data to a compressed HDF5 file.

    This is the primary archival format for the Universal Master Map.
    The structure is designed to be consumed by Phase 2 / Phase 3.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keys to store as datasets (numpy arrays)
    dataset_keys = [
        "eigenvalues", "registry",
        "velocity_x", "velocity_y",
        "w_xx", "w_xy", "w_yx", "w_yy",
        "mass_xx", "mass_xy", "mass_yx", "mass_yy",
        "r_derivative_x", "r_derivative_y",
        "berry_x", "berry_y",
        "born_huang",
        "born_huang_tensor_xx", "born_huang_tensor_xy",
        "born_huang_tensor_yx", "born_huang_tensor_yy",
        "slow_coefficient",
        "slow_coefficient_tensor_xx", "slow_coefficient_tensor_xy",
        "slow_coefficient_tensor_yx", "slow_coefficient_tensor_yy",
        "metric_derivative_x", "metric_derivative_y",
        "overlap",
        "xi_scalar_first_order",
        "kappa_matrix_x", "kappa_matrix_y",
        "weighted_leakage_scalar",
        "lowdin_t_matrix_x", "lowdin_t_matrix_y",
        "lowdin_r_matrix",
        *EXACT_TM_ARCHIVE_KEYS,
    ]

    with h5py.File(path, "w") as hf:
        # Store numeric arrays
        for key in dataset_keys:
            if key in stacked and isinstance(stacked[key], np.ndarray):
                hf.create_dataset(key, data=stacked[key], compression="gzip")

        # Scalar metadata arrays
        for key in ("converged", "n_iterations", "solve_time_seconds", "extract_time_seconds"):
            if key in stacked:
                hf.create_dataset(key, data=stacked[key])

        # Attributes: case definition
        hf.attrs["case_label"] = case.get("label", "unknown")
        hf.attrs["lattice_type"] = case.get("lattice_type", "unknown")
        hf.attrs["eps_bg"] = case["eps_bg"]
        hf.attrs["k0_frac"] = np.array(case["k0_frac"])
        hf.attrs["k0_label"] = case.get("k0_label", "")
        k0_cart = frac_to_cartesian_k(case["k0_frac"], case["lattice_vectors"])
        hf.attrs["k0_cartesian"] = k0_cart
        hf.attrs["lattice_vectors"] = np.array(case["lattice_vectors"])
        hf.attrs["polarization"] = polarization

        # Solver config
        hf.attrs["resolution"] = config.get("resolution", 32)
        hf.attrs["band_lo"] = config.get("band_lo", 0)
        hf.attrs["n_retained"] = config.get("n_retained", 4)
        hf.attrs["n_remote"] = config.get("n_remote", 12)
        hf.attrs["fd_step"] = config.get("fd_step", 0.005)
        hf.attrs["n_registry"] = int(config.get("registry_grid_size", 8))

        # Atom info (serialise as JSON string for complex nested data)
        hf.attrs["base_atoms_json"] = json.dumps(case["base_atoms"])

        # Pipeline version
        hf.attrs["solver"] = "blaze"
        hf.attrs["pipeline_version"] = "V4"

        # Moiré metadata (optional — only present when a twist angle is specified)
        if moire_meta is not None:
            moire_grp = hf.create_group("moire")
            moire_grp.attrs["theta_deg"] = moire_meta["theta_deg"]
            moire_grp.attrs["theta_rad"] = moire_meta["theta_rad"]
            moire_grp.attrs["eta"] = moire_meta["eta"]
            moire_grp.attrs["eta_geom"] = moire_meta["eta_geom"]
            moire_grp.attrs["moire_length"] = moire_meta["moire_length"]
            moire_grp.create_dataset("B_mono", data=moire_meta["B_mono"])
            moire_grp.create_dataset("B_moire", data=moire_meta["B_moire"])

        # Sanity summary (optional)
        if sanity is not None:
            san_grp = hf.create_group("sanity")
            for k, v in sanity.items():
                if v is not None:
                    if isinstance(v, list):
                        san_grp.attrs[k] = np.array(v)
                    else:
                        san_grp.attrs[k] = v


def save_sweep_npz(
    path: Path,
    stacked: dict,
) -> None:
    """Save stacked numeric arrays as compressed NPZ for quick loading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in stacked.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)


# ==============================================================================
# Frequency conversion helpers
# ==============================================================================

def eigenvalues_to_omega(eigenvalues: np.ndarray) -> np.ndarray:
    """Convert Blaze eigenvalues λ = (ω/c)² to MPB-style frequencies ω/(2πc/a)."""
    return np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)


# ==============================================================================
# Process a single case
# ==============================================================================

def process_case(
    case_name: str,
    case: dict,
    polarization: str,
    output_dir: Path,
    config: dict,
) -> tuple[dict, dict]:
    """Run the full Phase 1 extraction for one case and polarization.

    Returns:
        (stacked, sanity) — the stacked result arrays and sanity summary.
    """
    n_registry = config.get("registry_grid_size", 8)
    n_retained = config.get("n_retained", 4)
    n_remote = config.get("n_remote", 12)
    compute_exact_tm_fields = bool(
        config.get("compute_exact_tm_fields", polarization == "TM")
    )
    archive_exact_tm_hermitized_eigenvectors = bool(
        config.get(
            "archive_exact_tm_hermitized_eigenvectors",
            polarization == "TM" and compute_exact_tm_fields,
        )
    )

    log(f"\n{'='*72}")
    log(f"Case: {case_name} | {case.get('label', '')} | {polarization}")
    log(f"{'='*72}")

    # Build registry grid
    registry_points = build_registry_grid(n_registry, offset=config.get("registry_offset", True))
    log(f"  Registry grid: {n_registry}×{n_registry} = {len(registry_points)} points")

    # Run checkpointed Blaze sweep (row-by-row, OOM-safe, resumable)
    stem = f"{case_name}_{polarization.lower()}"
    ckpt_dir = output_dir / f"{stem}_checkpoints"

    run_blaze_registry_sweep_checkpointed(
        case,
        polarization,
        registry_points,
        checkpoint_dir=ckpt_dir,
        resolution=config.get("resolution", 32),
        band_lo=config.get("band_lo", 0),
        n_retained=n_retained,
        n_remote=n_remote,
        fd_step=config.get("fd_step", 0.005),
        tolerance=config.get("tolerance", 1e-8),
        max_iterations=config.get("max_iterations", 700),
        threads=config.get("threads", 8),
        compute_born_huang=config.get("compute_born_huang", True),
        compute_r_derivatives=config.get("compute_r_derivatives", True),
        compute_exact_tm_fields=compute_exact_tm_fields,
        smoothing=config.get("smoothing", True),
        archive_exact_tm_hermitized_eigenvectors=archive_exact_tm_hermitized_eigenvectors,
    )

    # Load checkpoints and stack (one row at a time, memory-efficient)
    stacked = load_and_stack_checkpoints(ckpt_dir, n_retained, n_remote)

    # Log convergence for sampled points
    n_pts = stacked["registry"].shape[0]
    for idx in range(n_pts):
        if idx == 0 or (idx + 1) % 16 == 0 or idx == n_pts - 1:
            reg = stacked["registry"][idx]
            log(
                f"  [{idx + 1:3d}/{n_pts}] "
                f"δ=({reg[0]:.4f}, {reg[1]:.4f})  "
                f"iters={stacked['n_iterations'][idx]}  "
                f"conv={stacked['converged'][idx]}"
            )

    # Sanity-check
    sanity = compute_sanity_summary(stacked, n_retained)

    log(f"\n  Sanity:")
    log(f"    all_finite={sanity['all_finite']}  all_converged={sanity['all_converged']}")
    eig_range = sanity["eigenvalue_range"]
    log(f"    eigenvalue range: [{eig_range[0]:.6f}, {eig_range[1]:.6f}]")
    omega_range = eigenvalues_to_omega(np.array(eig_range))
    log(f"    frequency range:  [{omega_range[0]:.6f}, {omega_range[1]:.6f}] (c/a)")
    log(f"    mean_iters={sanity['mean_iterations']:.1f}  "
        f"max_iters={sanity['max_iterations']}")
    if sanity["max_mass_xx_hermiticity"] is not None:
        log(f"    max mass_xx hermiticity: {sanity['max_mass_xx_hermiticity']:.2e}")
    if sanity["max_born_huang_hermiticity"] is not None:
        log(f"    max Born-Huang hermiticity: {sanity['max_born_huang_hermiticity']:.2e}")

    # Compute moiré metadata if a twist angle is specified
    theta_deg = config.get("theta_deg")
    moire_meta = None
    if theta_deg is not None:
        lattice_type = case.get("lattice_type", case_name)
        a = 1.0  # normalised
        moire_meta = compute_moire_metadata(lattice_type, a, theta_deg)
        log(f"  Moiré: θ={theta_deg}°  η={moire_meta['eta']:.4f}  "
            f"L_m={moire_meta['moire_length']:.2f} a")

    # Save outputs
    stem = f"{case_name}_{polarization.lower()}"
    npz_path = output_dir / f"{stem}_phase1.npz"
    json_path = output_dir / f"{stem}_phase1_sanity.json"

    save_sweep_npz(npz_path, stacked)
    save_json(sanity, json_path)

    log(f"  Saved: {npz_path.name}, {json_path.name}")

    # Clean up checkpoint JSON files now that NPZ is written successfully
    if ckpt_dir.is_dir():
        import shutil
        n_removed = sum(1 for f in ckpt_dir.iterdir() if f.suffix == ".json")
        shutil.rmtree(ckpt_dir)
        log(f"  Cleaned up {n_removed} checkpoint files from {ckpt_dir.name}/")

    return stacked, sanity


# ==============================================================================
# Main entry point
# ==============================================================================

DEFAULT_CONFIG: dict = {
    "resolution": 32,
    "n_retained": 4,
    "n_remote": 12,
    "registry_grid_size": 8,
    "registry_offset": True,
    "fd_step": 0.005,
    "tolerance": 1e-8,
    "max_iterations": 700,
    "threads": 8,
    "compute_born_huang": True,
    "compute_r_derivatives": True,
    "compute_exact_tm_fields": True,
    "archive_exact_tm_hermitized_eigenvectors": True,
    "smoothing": True,
    "polarizations": ["TE", "TM"],
    "cases": ["square", "hex", "honeycomb"],
    # Optional: set theta_deg to attach moiré metadata
    # "theta_deg": 3.0,
}


def run_phase1_v4(
    config: dict | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Main Phase 1 V4 driver.

    Args:
        config: configuration dict (overrides config_path).
        config_path: path to YAML config file.
        output_dir: override output directory.

    Returns:
        dict mapping "case_pol" → sanity summary.
    """
    # Load config
    if config is None:
        if config_path is not None:
            config = load_yaml(Path(config_path))
        else:
            config = dict(DEFAULT_CONFIG)
    else:
        # Merge with defaults for any missing keys
        merged = dict(DEFAULT_CONFIG)
        merged.update(config)
        config = merged

    # Output directory
    if output_dir is not None:
        out = Path(output_dir)
    else:
        out = Path(config.get(
            "output_dir",
            str(Path(__file__).resolve().parent / "output" / "phase1_blaze_v4"),
        ))
    out.mkdir(parents=True, exist_ok=True)

    polarizations = config.get("polarizations", ["TE", "TM"])
    case_names = config.get("cases", list(CANONICAL_CASES.keys()))

    # Allow config to define custom cases alongside canonical ones
    custom_cases = config.get("custom_cases", {})
    all_cases = dict(CANONICAL_CASES)
    all_cases.update(custom_cases)

    log("=" * 72)
    log("PHASE 1 V4 (Blaze): Multi-Band Local EA Extraction")
    log("=" * 72)
    log(f"Output directory: {out}")
    log(f"Cases: {case_names}")
    log(f"Polarizations: {polarizations}")
    log(f"Registry grid: {config['registry_grid_size']}×{config['registry_grid_size']}")
    log(f"Resolution: {config['resolution']}, N_ret={config['n_retained']}, "
        f"N_rem={config['n_remote']}, threads={config['threads']}")

    all_summaries: dict = {}
    for case_name in case_names:
        if case_name not in all_cases:
            log(f"\n  WARNING: case '{case_name}' not found, skipping")
            continue
        case = all_cases[case_name]
        for pol in polarizations:
            _, sanity = process_case(case_name, case, pol, out, config)
            key = f"{case_name}_{pol.lower()}"
            all_summaries[key] = sanity

    # Write global summary
    summary_path = out / "phase1_summary.json"
    save_json(all_summaries, summary_path)

    log(f"\n{'='*72}")
    log("PHASE 1 V4 (Blaze) COMPLETE")
    log(f"{'='*72}")
    log(f"Summary: {summary_path}")

    return all_summaries


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 1 V4: Blaze EA extraction over registry grid"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional, uses defaults otherwise)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--cases", nargs="+", default=None,
        help="Case names to run (default: all canonical)",
    )
    parser.add_argument(
        "--polarizations", nargs="+", default=None,
        help="Polarizations to run (default: TE TM)",
    )
    parser.add_argument(
        "--registry-grid-size", type=int, default=None,
        help="Registry grid points per direction",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Blaze solver resolution",
    )
    parser.add_argument(
        "--n-retained", type=int, default=None,
        help="Number of retained bands",
    )
    parser.add_argument(
        "--n-remote", type=int, default=None,
        help="Number of remote bands for Löwdin correction",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Number of Rust-side sweep threads",
    )
    parser.add_argument(
        "--theta-deg", type=float, default=None,
        help="Twist angle in degrees (attaches moiré metadata)",
    )

    args = parser.parse_args()

    # Build config from defaults + CLI overrides
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        cfg.update(load_yaml(Path(args.config)))
    if args.cases:
        cfg["cases"] = args.cases
    if args.polarizations:
        cfg["polarizations"] = args.polarizations
    if args.registry_grid_size is not None:
        cfg["registry_grid_size"] = args.registry_grid_size
    if args.resolution is not None:
        cfg["resolution"] = args.resolution
    if args.n_retained is not None:
        cfg["n_retained"] = args.n_retained
    if args.n_remote is not None:
        cfg["n_remote"] = args.n_remote
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.theta_deg is not None:
        cfg["theta_deg"] = args.theta_deg

    run_phase1_v4(config=cfg, output_dir=args.output_dir)
