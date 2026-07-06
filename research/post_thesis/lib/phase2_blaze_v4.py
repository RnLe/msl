#!/usr/bin/env python3
"""
Phase 2 (Blaze V4): Moiré Hamiltonian Assembly + Eigensolve

Replaces BOTH V3 Phase 2 and Phase 3.

Since Blaze computes all EA ingredients natively in Rust (Berry connection,
Born-Huang potential, Löwdin-corrected mass tensors, velocity matrices, etc.),
the old Phase 2's heavy computation — gauge fixing, SVQB, finite-difference
Berry/BH from Bloch fields — is completely eliminated.

What remains is a lightweight data-prep + eigensolve pipeline:

  1. LOAD: Phase 1 master map (θ-independent registry sweep)
  2. INTERPOLATE: map registry grid → moiré grid for a given twist angle θ
  3. TRANSFORM: convert Cartesian k-space quantities to the moiré
     fractional-coordinate basis for the FD discretization
  4. ASSEMBLE: build sparse multi-band envelope Hamiltonian
  5. SOLVE: shift-invert Lanczos (scipy.sparse.linalg.eigsh)
  6. ANALYZE: mode statistics (IPR, band composition, localization)
  7. SAVE: HDF5 + JSON

THE ENVELOPE HAMILTONIAN (in fractional moiré coordinates s ∈ [0,1)²):
--------------------------------------------------------------------
    Ĥ_mn = Λ_mn(s)
             + (1/2) Σ_α [ṽ_α Π_α + Π_α ṽ_α]_{mn}
             + (1/2) Σ_αβ [Π_α M̃_αβ Π_β]_{mn}
             + Φ̃_BH,mn + S̃_mn

  - Λ_mn = (λ_n - λ_ref) δ_mn          (diagonal potential from local eigenvalues)
  - ṽ = B⁻¹ v_cart                      (velocity transformed to s-basis)
  - M̃ = B⁻¹ M⁻¹_cart B⁻ᵀ               (mass tensor transformed to s-basis)
    - Π_α = -i ∂/∂s_α + A_α + 2πk_{s,α}   (full retained-space covariant derivative)
    - Φ̃_BH = exact metric contraction of the directional Born-Huang tensor
    - S̃ = exact metric contraction of the directional slow-coefficient tensor

  where B = B_moire (columns = moiré lattice vectors).

COORDINATE CONVENTION:
  - s ∈ [0,1)²: fractional moiré coordinates, s = B⁻¹ R
  - δ ∈ [0,1)²: fractional registry coordinates, δ = s (identity mapping)
  - k ∈ Cartesian reciprocal space (Blaze convention, includes 2π)

UNIT SYSTEM:
  - Eigenvalues λ = (ωa/c)² = (2π f_MPB)²  (dimensionless)
  - No extra 1/(2π) factors needed (Blaze uses Cartesian k, not MPB k)
  - FD operators in fractional coords naturally produce the correct 2π factors
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh
from scipy.interpolate import RegularGridInterpolator


# ==============================================================================
# Logging
# ==============================================================================

_log_fn = None

EXACT_TM_DATASET_KEYS = (
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
# Phase 1 Data Loading
# ==============================================================================

def load_phase1_h5(path: Path) -> dict:
    """Load Phase 1 master map from HDF5 **or** NPZ.

    If *path* points to a ``.h5`` file that does not exist, the loader
    automatically tries the sibling ``.npz`` (and vice-versa).  A ``.npz``
    path is also accepted directly.

    Returns a dict with all arrays reshaped to 2D registry grid
    (n_reg, n_reg, ...) plus metadata.
    """
    path = Path(path)

    # Resolve .h5 ↔ .npz fallback
    if path.suffix == ".h5" and not path.exists():
        npz_alt = path.with_suffix(".npz")
        if npz_alt.exists():
            path = npz_alt
    elif path.suffix == ".npz" and not path.exists():
        h5_alt = path.with_suffix(".h5")
        if h5_alt.exists():
            path = h5_alt

    if path.suffix == ".npz":
        return _load_phase1_npz(path)
    return _load_phase1_h5_impl(path)


def _load_phase1_npz(path: Path) -> dict:
    """Load Phase 1 master map from a compressed NPZ archive.

    Metadata that the HDF5 format stored as attributes is inferred from
    array shapes and, when available, from the companion ``_sanity.json``.
    """
    raw = dict(np.load(path, allow_pickle=False))

    # Infer grid dimensions from the eigenvalue array
    n_pts = raw["eigenvalues"].shape[0]
    n_reg = int(round(n_pts ** 0.5))
    n_total_eig = raw["eigenvalues"].shape[1] if raw["eigenvalues"].ndim > 1 else 1
    n_retained = raw["mass_xx"].shape[1]  # mass is (N, n_ret, n_ret)
    n_remote = n_total_eig - n_retained   # approximate; band_lo unknown

    # Try loading companion sanity JSON for authoritative metadata
    sanity_path = path.with_name(path.stem.replace("_phase1", "_phase1_sanity") + ".json")
    sanity: dict = {}
    if sanity_path.exists():
        import json as _json
        with open(sanity_path) as _f:
            sanity = _json.load(_f)
    if "n_retained" in sanity:
        n_retained = int(sanity["n_retained"])

    data: dict = {
        "n_reg": n_reg,
        "band_lo": 0,
        "n_retained": n_retained,
        "n_remote": n_remote,
        # NPZ files carry no HDF5 attrs; allow explicit overrides (vendored patch,
        # post-thesis strict campaign) since the original .h5 archives were pruned.
        "polarization": os.environ.get("MSL_POLARIZATION", "unknown"),
        "lattice_type": os.environ.get("MSL_LATTICE_TYPE", "unknown"),
        "eps_bg": 1.0,
        "k0_frac": np.zeros(2),
        "k0_cartesian": np.zeros(2),
        "lattice_vectors": np.eye(2),
        "resolution": 0,
    }

    def _load_2d(key):
        arr = raw[key]
        return arr.reshape(n_reg, n_reg, *arr.shape[1:])

    # Core arrays
    for key in ("eigenvalues", "registry", "converged",
                "velocity_x", "velocity_y",
                "mass_xx", "mass_xy", "mass_yx", "mass_yy",
                "berry_x", "berry_y",
                "metric_derivative_x", "metric_derivative_y",
                "overlap", "born_huang",
                "slow_coefficient"):
        if key in raw:
            data[key] = _load_2d(key)

    for suffix in ("xx", "xy", "yx", "yy"):
        for prefix in ("born_huang_tensor", "slow_coefficient_tensor"):
            key = f"{prefix}_{suffix}"
            if key in raw:
                data[key] = _load_2d(key)

    for key in EXACT_TM_DATASET_KEYS:
        if key in raw:
            data[key] = _load_2d(key)

    return data


def _load_phase1_h5_impl(path: Path) -> dict:
    """Original HDF5-based Phase 1 loader (internal)."""
    data: dict = {}

    with h5py.File(path, "r") as hf:
        n_reg = int(hf.attrs["n_registry"])
        band_lo = int(hf.attrs.get("band_lo", 0))
        n_retained = int(hf.attrs["n_retained"])
        n_remote = int(hf.attrs["n_remote"])
        n_total = band_lo + n_retained + n_remote

        # Metadata
        data["n_reg"] = n_reg
        data["band_lo"] = band_lo
        data["n_retained"] = n_retained
        data["n_remote"] = n_remote
        data["polarization"] = str(hf.attrs["polarization"])
        data["lattice_type"] = str(hf.attrs["lattice_type"])
        data["eps_bg"] = float(hf.attrs["eps_bg"])
        data["k0_frac"] = np.array(hf.attrs["k0_frac"])
        data["k0_cartesian"] = np.array(hf.attrs["k0_cartesian"])
        data["lattice_vectors"] = np.array(hf.attrs["lattice_vectors"])
        data["resolution"] = int(hf.attrs["resolution"])

        # Moiré metadata (optional)
        if "moire" in hf:
            mg = hf["moire"]
            data["theta_deg"] = float(mg.attrs["theta_deg"])
            data["theta_rad"] = float(mg.attrs["theta_rad"])
            data["eta"] = float(mg.attrs["eta"])
            data["moire_length"] = float(mg.attrs["moire_length"])
            data["B_mono"] = mg["B_mono"][:]
            data["B_moire"] = mg["B_moire"][:]

        # Reshape helper: (N_reg_total, ...) → (n_reg, n_reg, ...)
        def _load_2d(key):
            arr = hf[key][:]
            return arr.reshape(n_reg, n_reg, *arr.shape[1:])

        # Core arrays
        data["eigenvalues"] = _load_2d("eigenvalues")       # (nr, nr, N_ret)
        data["registry"] = _load_2d("registry")             # (nr, nr, 2)
        data["converged"] = _load_2d("converged")            # (nr, nr)

        # Velocity (N_ret × N_total) — take retained block for drift
        data["velocity_x"] = _load_2d("velocity_x")         # (nr, nr, N_ret, N_total)
        data["velocity_y"] = _load_2d("velocity_y")

        # Mass tensor (N_ret × N_ret), already Löwdin-corrected
        data["mass_xx"] = _load_2d("mass_xx")
        data["mass_xy"] = _load_2d("mass_xy")
        data["mass_yx"] = _load_2d("mass_yx")
        data["mass_yy"] = _load_2d("mass_yy")

        # Berry connection (N_ret × N_ret) — optional
        if "berry_x" in hf:
            data["berry_x"] = _load_2d("berry_x")
            data["berry_y"] = _load_2d("berry_y")

        # TM generalized-operator auxiliaries — optional
        if "metric_derivative_x" in hf:
            data["metric_derivative_x"] = _load_2d("metric_derivative_x")
            data["metric_derivative_y"] = _load_2d("metric_derivative_y")
        if "overlap" in hf:
            data["overlap"] = _load_2d("overlap")

        # Born-Huang (N_ret × N_ret) — optional
        if "born_huang" in hf:
            data["born_huang"] = _load_2d("born_huang")
        for suffix in ("xx", "xy", "yx", "yy"):
            key = f"born_huang_tensor_{suffix}"
            if key in hf:
                data[key] = _load_2d(key)

        # Slow coefficient (N_ret × N_ret), TE only — optional
        if "slow_coefficient" in hf:
            data["slow_coefficient"] = _load_2d("slow_coefficient")
        for suffix in ("xx", "xy", "yx", "yy"):
            key = f"slow_coefficient_tensor_{suffix}"
            if key in hf:
                data[key] = _load_2d(key)

        # Exact TM field-level payload — optional
        for key in EXACT_TM_DATASET_KEYS:
            if key in hf:
                data[key] = _load_2d(key)

    return data


# ==============================================================================
# Moiré Geometry
# ==============================================================================

def build_monolayer_basis(lattice_type: str, a: float = 1.0) -> np.ndarray:
    """Build monolayer lattice basis B = (a1 | a2), columns = lattice vectors."""
    if lattice_type == "square":
        return a * np.eye(2)
    elif lattice_type in ("hex", "honeycomb"):
        return a * np.array([[1.0, 0.5], [0.0, math.sqrt(3) / 2]])
    raise ValueError(f"Unknown lattice type: {lattice_type}")


def compute_moire_basis(B_mono: np.ndarray, theta_rad: float) -> np.ndarray:
    """Compute moiré basis: B_moire = (R(θ) - I)⁻¹ B_mono."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    return np.linalg.inv(R - np.eye(2)) @ B_mono


def commensurate_angle(m: int, n: int) -> float:
    """Exact commensurate twist angle for a square lattice with indices (m, n).

    The coincidence superlattice has vectors L1 = (m, n), L2 = (-n, m)
    with area = m² + n² primitive cells per layer.  The twist angle is
    θ = 2·arctan(n / m).

    Returns:
        Angle in degrees.
    """
    return math.degrees(2.0 * math.atan2(n, m))


def commensurate_moire_metadata(
    lattice_type: str,
    a: float,
    m: int,
    n: int,
) -> dict:
    """Compute moiré geometry from commensurate indices (m, n).

    For a square lattice the commensurate supercell has:
      - Superlattice vectors  L1 = m·a1 + n·a2,  L2 = -n·a1 + m·a2
      - Area = (m² + n²)·a²
      - θ = 2·arctan(n/m)
      - The supercell is exactly 2×2 moiré unit cells (area ratio = 4).
    """
    theta_deg = commensurate_angle(m, n)
    meta = compute_moire_metadata(lattice_type, a, theta_deg)
    N_cells = m * m + n * n
    L_super = a * math.sqrt(N_cells)
    meta["commensurate"] = True
    meta["m"] = m
    meta["n"] = n
    meta["N_cells"] = N_cells
    meta["L_super"] = L_super
    meta["B_super"] = np.array([[m, n], [-n, m]], dtype=float) * a
    return meta


def compute_moire_metadata(lattice_type: str, a: float, theta_deg: float) -> dict:
    """Compute moiré geometry from lattice type and twist angle."""
    theta_rad = math.radians(theta_deg)
    B_mono = build_monolayer_basis(lattice_type, a)
    B_moire = compute_moire_basis(B_mono, theta_rad)
    eta = 2.0 * math.sin(theta_rad / 2.0)
    L_moire = a / eta
    return {
        "theta_deg": theta_deg,
        "theta_rad": theta_rad,
        "eta": eta,
        "moire_length": L_moire,
        "B_mono": B_mono,
        "B_moire": B_moire,
        "commensurate": False,
    }


# ==============================================================================
# Interpolation: Registry Grid → Moiré Grid
# ==============================================================================

def interpolate_to_moire_grid(
    field_2d: np.ndarray,
    n_reg: int,
    Ns: int,
) -> np.ndarray:
    """Interpolate a 2D registry field to a moiré grid via periodic interpolation.

    Since registry fractional coords = moiré fractional coords (identity mapping
    δ = s), this is just grid-to-grid resampling on [0,1)².

    Args:
        field_2d: (n_reg, n_reg, ...) array on registry grid
        n_reg: registry grid resolution
        Ns: moiré grid resolution

    Returns:
        (Ns, Ns, ...) interpolated array
    """
    if n_reg == Ns:
        return field_2d.copy()

    trailing_shape = field_2d.shape[2:]
    ds_reg = 1.0 / n_reg
    s_reg = np.arange(n_reg) * ds_reg

    # Extend for periodicity
    s_ext = np.append(s_reg, 1.0)

    # Target grid
    ds_moire = 1.0 / Ns
    s_moire = np.arange(Ns) * ds_moire
    S1, S2 = np.meshgrid(s_moire, s_moire, indexing="ij")
    query = np.stack([S1.ravel(), S2.ravel()], axis=-1)

    if np.iscomplexobj(field_2d):
        return _interp_complex(field_2d, s_ext, n_reg, query, Ns, trailing_shape)
    else:
        return _interp_real(field_2d, s_ext, n_reg, query, Ns, trailing_shape)


def _extend_periodic(arr_2d, n_reg):
    """Extend a 2D array with one extra row/col for periodic wrapping."""
    ext = np.zeros((n_reg + 1, n_reg + 1), dtype=arr_2d.dtype)
    ext[:n_reg, :n_reg] = arr_2d
    ext[n_reg, :n_reg] = arr_2d[0, :]
    ext[:n_reg, n_reg] = arr_2d[:, 0]
    ext[n_reg, n_reg] = arr_2d[0, 0]
    return ext


def _interp_real(field, s_ext, n_reg, query, Ns, trailing):
    out = np.empty((Ns, Ns, *trailing), dtype=field.dtype)
    for idx in np.ndindex(*trailing):
        slab = field[(slice(None), slice(None)) + idx]
        ext = _extend_periodic(slab, n_reg)
        interp = RegularGridInterpolator(
            (s_ext, s_ext), ext, method="linear",
            bounds_error=False, fill_value=None,
        )
        out[(slice(None), slice(None)) + idx] = interp(query).reshape(Ns, Ns)
    return out


def _interp_complex(field, s_ext, n_reg, query, Ns, trailing):
    out_re = _interp_real(field.real, s_ext, n_reg, query, Ns, trailing)
    out_im = _interp_real(field.imag, s_ext, n_reg, query, Ns, trailing)
    return out_re + 1j * out_im


def tile_fields_2d(field: np.ndarray, tiling: int) -> np.ndarray:
    """Tile a (Ns, Ns, ...) field to (tiling*Ns, tiling*Ns, ...).

    The first two axes are tiled; all trailing axes are preserved.
    This implements the supercell tiling for commensurate-angle band folding:
    an (Ns, Ns) primitive moiré field becomes (tiling*Ns, tiling*Ns) with
    periodic copies, so the PW solver sees a larger real-space domain and
    automatically folds the Brillouin zone.
    """
    if tiling <= 1:
        return field
    n_trailing = len(field.shape) - 2
    reps = (tiling, tiling) + (1,) * n_trailing
    return np.tile(field, reps)


# ==============================================================================
# Coordinate Transforms (Cartesian → fractional moiré)
# ==============================================================================

def transform_mass_tensor(
    mass_xx: np.ndarray, mass_xy: np.ndarray,
    mass_yx: np.ndarray, mass_yy: np.ndarray,
    B_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform mass tensor from Cartesian to fractional moiré basis.

    M̃ = B⁻¹ M_cart B⁻ᵀ

    M_cart = [[Mxx, Mxy], [Myx, Myy]]
    B_inv  = B_moire⁻¹

    Returns:
        (M̃_11, M̃_12, M̃_21, M̃_22) in fractional basis
    """
    a, b = B_inv[0, 0], B_inv[0, 1]
    c, d = B_inv[1, 0], B_inv[1, 1]

    # M̃ = B⁻¹ M B⁻ᵀ
    # M̃_αβ = Σ_{ij} B⁻¹_{αi} M_{ij} B⁻¹_{βj}
    M11 = a * a * mass_xx + a * b * mass_xy + b * a * mass_yx + b * b * mass_yy
    M12 = a * c * mass_xx + a * d * mass_xy + b * c * mass_yx + b * d * mass_yy
    M21 = c * a * mass_xx + c * b * mass_xy + d * a * mass_yx + d * b * mass_yy
    M22 = c * c * mass_xx + c * d * mass_xy + d * c * mass_yx + d * d * mass_yy

    return M11, M12, M21, M22


def transform_velocity(
    v_x: np.ndarray, v_y: np.ndarray, B_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform velocity from Cartesian to fractional moiré basis.

    ṽ = B⁻¹ v_cart

    Returns:
        (ṽ_1, ṽ_2) in fractional basis
    """
    a, b = B_inv[0, 0], B_inv[0, 1]
    c, d = B_inv[1, 0], B_inv[1, 1]
    return a * v_x + b * v_y, c * v_x + d * v_y


def born_huang_metric_factor(B_moire: np.ndarray) -> float:
    """Compute metric correction factor for Born-Huang / slow coefficient.

    Blaze computes Φ = Σ_j Φ_{jj} in flat fractional metric.
    Physical BH = Σ_{kl} G^{kl} Φ_{kl} where G = B⁻¹ B⁻ᵀ.
    Approximation (assuming Φ_{00} ≈ Φ_{11}, Φ_{01} ≈ 0):
        Φ_phys ≈ Tr(G)/2 × Φ_Blaze

    Returns the factor Tr(G)/2.
    """
    B_inv = np.linalg.inv(B_moire)
    G = B_inv @ B_inv.T
    return float(np.trace(G) / 2.0)


def exact_metric_tensor(B_moire: np.ndarray) -> np.ndarray:
    """Return the exact inverse metric G = B⁻¹B⁻ᵀ for moiré fractional coordinates."""
    B_inv = np.linalg.inv(B_moire)
    return B_inv @ B_inv.T


def contract_directional_potential_tensor(
    xx: np.ndarray,
    xy: np.ndarray,
    yx: np.ndarray,
    yy: np.ndarray,
    metric: np.ndarray,
) -> np.ndarray:
    """Contract a directional scalar-potential tensor with the inverse metric."""
    contracted = (
        metric[0, 0] * xx
        + metric[0, 1] * xy
        + metric[1, 0] * yx
        + metric[1, 1] * yy
    )
    return 0.5 * (contracted + np.swapaxes(contracted.conj(), -1, -2))


def build_lambda_potential(
    eigenvalues: np.ndarray,
    Ns: int,
    Nb: int,
    lambda_ref: float,
    band_lo: int = 0,
) -> np.ndarray:
    """Build the diagonal local potential matrix from retained eigenvalues."""
    Lambda = np.zeros((Ns, Ns, Nb, Nb), dtype=complex)
    for n in range(Nb):
        Lambda[..., n, n] = eigenvalues[..., band_lo + n] - lambda_ref
    return Lambda


def contract_scalar_tensor_with_metric(
    xx: np.ndarray | None,
    xy: np.ndarray | None,
    yx: np.ndarray | None,
    yy: np.ndarray | None,
    scalar_legacy: np.ndarray | None,
    B_moire: np.ndarray,
) -> np.ndarray | None:
    """Contract a directional scalar tensor when present, else rescale legacy data."""
    metric = exact_metric_tensor(B_moire)
    if xx is not None and xy is not None and yx is not None and yy is not None:
        return contract_directional_potential_tensor(xx, xy, yx, yy, metric)
    if scalar_legacy is None:
        return None
    return scalar_legacy * born_huang_metric_factor(B_moire)


def _require_exact_tm_fields(exact_tm_fields: dict, required_keys: tuple[str, ...]) -> None:
    missing = [key for key in required_keys if exact_tm_fields.get(key) is None]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            "Exact TM Phase 2 assembly requires the full archived tm_exact payload. "
            f"Missing datasets: {missing_str}. Regenerate Phase 1 with the updated exact TM archive contract."
        )


def build_tm_exact_operator_inputs(
    *,
    eigenvalues: np.ndarray,
    exact_tm_fields: dict,
    Ns: int,
    Nb: int,
    lambda_ref: float,
    B_inv: np.ndarray,
    B_moire: np.ndarray,
    eta: float,
    band_lo: int = 0,
) -> dict:
    """Prepare the exact TM primitive operator blocks on the moire grid."""
    required = (
        "tm_exact_velocity_x",
        "tm_exact_velocity_y",
        "tm_exact_first_order_remainder",
        "tm_exact_direct_metric",
        "tm_exact_direct_b_x",
        "tm_exact_direct_b_y",
        "tm_exact_direct_gamma2",
        "tm_exact_mass_xx",
        "tm_exact_mass_xy",
        "tm_exact_mass_yx",
        "tm_exact_mass_yy",
    )
    _require_exact_tm_fields(exact_tm_fields, required)

    n_total = exact_tm_fields["tm_exact_velocity_x"].shape[-1]
    n_remote = n_total - Nb
    if n_remote < 0:
        raise ValueError("Exact TM archive has fewer total bands than retained bands")
    if eigenvalues.shape[-1] < n_total:
        raise ValueError(
            "Phase 1 eigenvalue archive does not include enough bands for exact TM remote resolvents"
        )

    Lambda = build_lambda_potential(eigenvalues, Ns, Nb, lambda_ref, band_lo=band_lo)

    v1_full, v2_full = transform_velocity(
        exact_tm_fields["tm_exact_velocity_x"],
        exact_tm_fields["tm_exact_velocity_y"],
        B_inv,
    )
    # Retained-retained block: absolute indices [band_lo:band_lo+Nb]
    ret = slice(band_lo, band_lo + Nb)
    v1_pp = v1_full[..., ret, ret]
    v2_pp = v2_full[..., ret, ret]

    gamma1_full = exact_tm_fields["tm_exact_first_order_remainder"]
    gamma1_pp = gamma1_full[..., ret, ret]

    direct_b1, direct_b2 = transform_velocity(
        exact_tm_fields["tm_exact_direct_b_x"],
        exact_tm_fields["tm_exact_direct_b_y"],
        B_inv,
    )
    metric = exact_metric_tensor(B_moire)

    M11, M12, M21, M22 = transform_mass_tensor(
        exact_tm_fields["tm_exact_mass_xx"],
        exact_tm_fields["tm_exact_mass_xy"],
        exact_tm_fields["tm_exact_mass_yx"],
        exact_tm_fields["tm_exact_mass_yy"],
        B_inv,
    )

    # Remote eigenvalues: lower remote [0:band_lo) + upper remote [band_lo+Nb:n_total)
    resolvent_q = np.zeros((Ns, Ns, n_remote, n_remote), dtype=np.complex128)
    if n_remote > 0:
        remote_eigenvalues = np.concatenate(
            [eigenvalues[..., :band_lo], eigenvalues[..., band_lo + Nb:n_total]],
            axis=-1,
        )
        denom = remote_eigenvalues - lambda_ref
        inv_denom = np.zeros_like(denom, dtype=np.complex128)
        mask = np.abs(denom) > 1e-12
        inv_denom[mask] = 1.0 / denom[mask]
        for remote_idx in range(n_remote):
            resolvent_q[..., remote_idx, remote_idx] = inv_denom[..., remote_idx]

    # Retained-remote and remote-retained blocks (remote = lower ∪ upper)
    def _pq(full):
        """Extract retained-rows × remote-cols from [n_total × n_total] matrix."""
        return np.concatenate(
            [full[..., ret, :band_lo], full[..., ret, band_lo + Nb:n_total]],
            axis=-1,
        )

    def _qp(full):
        """Extract remote-rows × retained-cols from [n_total × n_total] matrix."""
        return np.concatenate(
            [full[..., :band_lo, ret], full[..., band_lo + Nb:n_total, ret]],
            axis=-2,
        )

    return {
        "operator_model": "tm_exact",
        "Lambda": Lambda,
        "v1": v1_pp,
        "v2": v2_pp,
        "M11": M11,
        "M12": M12,
        "M21": M21,
        "M22": M22,
        "A1": None,
        "A2": None,
        "Phi_BH": None,
        "Phi_slow": None,
        "has_exact_tm_payload": True,
        "tm_exact": {
            "eta": float(eta),
            "metric": metric,
            "v1_pp": v1_pp,
            "v2_pp": v2_pp,
            "gamma1_pp": gamma1_pp,
            "direct_metric": exact_tm_fields["tm_exact_direct_metric"],
            "direct_b1": direct_b1,
            "direct_b2": direct_b2,
            "direct_gamma2": exact_tm_fields["tm_exact_direct_gamma2"],
            "v1_pq": _pq(v1_full),
            "v2_pq": _pq(v2_full),
            "v1_qp": _qp(v1_full),
            "v2_qp": _qp(v2_full),
            "gamma1_pq": _pq(gamma1_full),
            "gamma1_qp": _qp(gamma1_full),
            "resolvent_q": resolvent_q,
            "n_remote": n_remote,
        },
    }


def regularize_transformed_mass_tensor(
    M11: np.ndarray,
    M12: np.ndarray,
    M21: np.ndarray,
    M22: np.ndarray,
    max_trace: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optionally clamp the diagonal-band mass blocks at near-degeneracies."""
    if max_trace is None:
        return M11, M12, M21, M22

    Ns = M11.shape[0]
    Nb = M11.shape[2]
    log(f"  Regularizing M̃ (max trace = {max_trace})...")
    M_diag = np.zeros((Ns, Ns, Nb, 2, 2), dtype=complex)
    for n in range(Nb):
        M_diag[..., n, 0, 0] = M11[..., n, n]
        M_diag[..., n, 0, 1] = M12[..., n, n]
        M_diag[..., n, 1, 0] = M21[..., n, n]
        M_diag[..., n, 1, 1] = M22[..., n, n]
    M_diag = regularize_mass_tensor(M_diag, max_trace)
    for n in range(Nb):
        M11[..., n, n] = M_diag[..., n, 0, 0]
        M12[..., n, n] = M_diag[..., n, 0, 1]
        M21[..., n, n] = M_diag[..., n, 1, 0]
        M22[..., n, n] = M_diag[..., n, 1, 1]
    return M11, M12, M21, M22


def build_te_operator_inputs(
    *,
    eigenvalues: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    mass_xx: np.ndarray,
    mass_xy: np.ndarray,
    mass_yx: np.ndarray,
    mass_yy: np.ndarray,
    berry_x: np.ndarray | None,
    berry_y: np.ndarray | None,
    born_huang: np.ndarray | None,
    born_huang_tensor_xx: np.ndarray | None,
    born_huang_tensor_xy: np.ndarray | None,
    born_huang_tensor_yx: np.ndarray | None,
    born_huang_tensor_yy: np.ndarray | None,
    slow_coeff: np.ndarray | None,
    slow_coeff_tensor_xx: np.ndarray | None,
    slow_coeff_tensor_xy: np.ndarray | None,
    slow_coeff_tensor_yx: np.ndarray | None,
    slow_coeff_tensor_yy: np.ndarray | None,
    Ns: int,
    Nb: int,
    lambda_ref: float,
    B_inv: np.ndarray,
    B_moire: np.ndarray,
    M_inv_max_trace: float | None,
    band_lo: int = 0,
) -> dict:
    """Prepare the TE operator ingredients on the moire grid."""
    Lambda = build_lambda_potential(eigenvalues, Ns, Nb, lambda_ref, band_lo=band_lo)

    v_x_ret = velocity_x[..., band_lo:band_lo + Nb]
    v_y_ret = velocity_y[..., band_lo:band_lo + Nb]
    v1, v2 = transform_velocity(v_x_ret, v_y_ret, B_inv)

    M11, M12, M21, M22 = transform_mass_tensor(
        mass_xx, mass_xy, mass_yx, mass_yy, B_inv,
    )
    M11, M12, M21, M22 = regularize_transformed_mass_tensor(
        M11, M12, M21, M22, M_inv_max_trace,
    )

    Phi_BH = contract_scalar_tensor_with_metric(
        born_huang_tensor_xx,
        born_huang_tensor_xy,
        born_huang_tensor_yx,
        born_huang_tensor_yy,
        born_huang,
        B_moire,
    )
    Phi_slow = contract_scalar_tensor_with_metric(
        slow_coeff_tensor_xx,
        slow_coeff_tensor_xy,
        slow_coeff_tensor_yx,
        slow_coeff_tensor_yy,
        slow_coeff,
        B_moire,
    )

    return {
        "operator_model": "te_compact",
        "Lambda": Lambda,
        "v1": v1,
        "v2": v2,
        "M11": M11,
        "M12": M12,
        "M21": M21,
        "M22": M22,
        "A1": berry_x,
        "A2": berry_y,
        "Phi_BH": Phi_BH,
        "Phi_slow": Phi_slow,
        "has_exact_tm_payload": False,
    }


def build_tm_operator_inputs(
    *,
    eigenvalues: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    mass_xx: np.ndarray,
    mass_xy: np.ndarray,
    mass_yx: np.ndarray,
    mass_yy: np.ndarray,
    berry_x: np.ndarray | None,
    berry_y: np.ndarray | None,
    born_huang: np.ndarray | None,
    born_huang_tensor_xx: np.ndarray | None,
    born_huang_tensor_xy: np.ndarray | None,
    born_huang_tensor_yx: np.ndarray | None,
    born_huang_tensor_yy: np.ndarray | None,
    metric_derivative_x: np.ndarray | None,
    metric_derivative_y: np.ndarray | None,
    overlap: np.ndarray | None,
    exact_tm_fields: dict,
    Ns: int,
    Nb: int,
    lambda_ref: float,
    B_inv: np.ndarray,
    B_moire: np.ndarray,
    eta: float,
    M_inv_max_trace: float | None,
    tm_operator_model: str,
    band_lo: int = 0,
) -> dict:
    """Prepare the TM operator ingredients on the moire grid.

    The current assembled TM branch is the compact projected operator. The exact
    field-level TM archive is carried forward explicitly so a future exact branch
    can be implemented without overloading the compact path.
    """
    if tm_operator_model == "exact":
        return build_tm_exact_operator_inputs(
            eigenvalues=eigenvalues,
            exact_tm_fields=exact_tm_fields,
            Ns=Ns,
            Nb=Nb,
            lambda_ref=lambda_ref,
            B_inv=B_inv,
            B_moire=B_moire,
            eta=eta,
            band_lo=band_lo,
        )

    Lambda = build_lambda_potential(eigenvalues, Ns, Nb, lambda_ref, band_lo=band_lo)

    v_x_ret = velocity_x[..., band_lo:band_lo + Nb]
    v_y_ret = velocity_y[..., band_lo:band_lo + Nb]
    v1, v2 = transform_velocity(v_x_ret, v_y_ret, B_inv)

    M11, M12, M21, M22 = transform_mass_tensor(
        mass_xx, mass_xy, mass_yx, mass_yy, B_inv,
    )
    M11, M12, M21, M22 = regularize_transformed_mass_tensor(
        M11, M12, M21, M22, M_inv_max_trace,
    )

    Phi_BH = contract_scalar_tensor_with_metric(
        born_huang_tensor_xx,
        born_huang_tensor_xy,
        born_huang_tensor_yx,
        born_huang_tensor_yy,
        born_huang,
        B_moire,
    )

    has_exact_tm_payload = bool(exact_tm_fields)
    if has_exact_tm_payload:
        log(f"  TM exact payload detected: {len(exact_tm_fields)} archived field blocks")
    if metric_derivative_x is not None and metric_derivative_y is not None:
        log("  TM generalized payload detected: metric-derivative matrices present")
    if overlap is not None:
        log("  TM generalized payload detected: overlap matrix present")

    return {
        "operator_model": "tm_compact",
        "Lambda": Lambda,
        "v1": v1,
        "v2": v2,
        "M11": M11,
        "M12": M12,
        "M21": M21,
        "M22": M22,
        "A1": berry_x,
        "A2": berry_y,
        "Phi_BH": Phi_BH,
        "Phi_slow": None,
        "has_exact_tm_payload": has_exact_tm_payload,
    }


# ==============================================================================
# Finite-Difference Operators (periodic, fractional coords)
# ==============================================================================

def build_fd_derivative(N: int, ds: float, order: int = 4) -> csr_matrix:
    """Periodic finite-difference first derivative matrix.

    Eigenvalue for mode e^{2πi n s}: ≈ 2πi n.
    """
    if order == 4:
        coeffs = np.array([1, -8, 0, 8, -1]) / (12 * ds)
        offsets = [-2, -1, 0, 1, 2]
    else:
        coeffs = np.array([-1, 0, 1]) / (2 * ds)
        offsets = [-1, 0, 1]

    D = sparse.lil_matrix((N, N), dtype=float)
    for coeff, off in zip(coeffs, offsets):
        for i in range(N):
            j = (i + off) % N
            D[i, j] += coeff
    return D.tocsr()


def build_fd_laplacian(N: int, ds: float, order: int = 4) -> csr_matrix:
    """Periodic finite-difference second derivative (Laplacian) matrix.

    Eigenvalue for mode e^{2πi n s}: ≈ -(2πn)².
    """
    if order == 4:
        coeffs = np.array([-1, 16, -30, 16, -1]) / (12 * ds**2)
        offsets = [-2, -1, 0, 1, 2]
    else:
        coeffs = np.array([1, -2, 1]) / ds**2
        offsets = [-1, 0, 1]

    L = sparse.lil_matrix((N, N), dtype=float)
    for coeff, off in zip(coeffs, offsets):
        for i in range(N):
            j = (i + off) % N
            L[i, j] += coeff
    return L.tocsr()


# ==============================================================================
# Sparse Hamiltonian Operators
# ==============================================================================

def _build_block_diagonal(vals: np.ndarray, N_s: int, Nb: int) -> csr_matrix:
    """Build sparse block-diagonal-in-space from per-point band matrices.

    Args:
        vals: (N_s, Nb, Nb) complex — one Nb×Nb block per spatial point
    """
    return _build_block_operator(vals, N_s, Nb, Nb)


def _build_block_operator(
    vals: np.ndarray,
    N_s: int,
    n_rows: int,
    n_cols: int,
) -> csr_matrix:
    """Build a sparse block-diagonal-in-space operator from local band blocks."""
    total_rows = N_s * n_rows
    total_cols = N_s * n_cols
    row_off = np.arange(N_s) * n_rows
    col_off = np.arange(N_s) * n_cols
    row_idx, col_idx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows = (row_off[:, None, None] + row_idx[None]).ravel()
    cols = (col_off[:, None, None] + col_idx[None]).ravel()
    data = vals.reshape(N_s, n_rows, n_cols).ravel()
    return csr_matrix((data, (rows, cols)), shape=(total_rows, total_cols))


def build_potential_operator(Lambda: np.ndarray) -> csr_matrix:
    """Diagonal potential Λ_mn(s) as sparse matrix.

    Args:
        Lambda: (Ns, Ns, Nb, Nb) potential matrix
    """
    Ns1, Ns2, Nb, _ = Lambda.shape
    N_s = Ns1 * Ns2
    return _build_block_diagonal(Lambda.reshape(N_s, Nb, Nb), N_s, Nb)


def build_drift_operator(
    v1: np.ndarray,
    v2: np.ndarray,
    Pi1_or_Ns,
    Pi2_or_Nb,
    Ns_or_ds,
    Nb_or_order=None,
    order: int = 4,
) -> csr_matrix:
    """Drift term: (1/2) Σ_α [ṽ_α Π_α + Π_α ṽ_α].

    Supports both the current signature ``(v1, v2, Pi1, Pi2, Ns, Nb)`` and the
    legacy helper signature ``(v1, v2, Ns, Nb, ds, order)``.
    """
    if sparse.issparse(Pi1_or_Ns):
        Pi1 = Pi1_or_Ns
        Pi2 = Pi2_or_Nb
        Ns = int(Ns_or_ds)
        if Nb_or_order is None:
            raise ValueError("Nb must be provided when passing prebuilt Pi operators")
        Nb = int(Nb_or_order)
    else:
        Ns = int(Pi1_or_Ns)
        Nb = int(Pi2_or_Nb)
        ds = float(Ns_or_ds)
        if Nb_or_order is not None:
            order = int(Nb_or_order)
        Pi1, Pi2 = build_covariant_derivative_operators(None, None, Ns, Nb, ds, order)

    N_s = Ns * Ns

    V1_op = _build_block_diagonal(v1.reshape(N_s, Nb, Nb), N_s, Nb)
    V2_op = _build_block_diagonal(v2.reshape(N_s, Nb, Nb), N_s, Nb)

    T = 0.5 * (V1_op @ Pi1 + Pi1 @ V1_op + V2_op @ Pi2 + Pi2 @ V2_op)
    T = (T + T.conj().T) / 2
    return T


def build_covariant_derivative_operators(
    A1: np.ndarray | None,
    A2: np.ndarray | None,
    Ns: int,
    Nb: int,
    ds: float,
    order: int = 4,
    k_s: tuple[float, float] = (0.0, 0.0),
) -> tuple[csr_matrix, csr_matrix]:
    """Build Π_α = -i∂_α + A_α + 2πk_{s,α} as sparse operators."""
    N_s = Ns * Ns
    N_total = N_s * Nb

    D1 = build_fd_derivative(Ns, ds, order)
    D2 = build_fd_derivative(Ns, ds, order)
    D1_full = kron(D1, eye(Ns * Nb), format="csr")
    D2_full = kron(eye(Ns), kron(D2, eye(Nb)), format="csr")

    if A1 is None:
        A1_op = csr_matrix((N_total, N_total), dtype=complex)
    else:
        A1_op = _build_block_diagonal(A1.reshape(N_s, Nb, Nb), N_s, Nb)

    if A2 is None:
        A2_op = csr_matrix((N_total, N_total), dtype=complex)
    else:
        A2_op = _build_block_diagonal(A2.reshape(N_s, Nb, Nb), N_s, Nb)

    k1, k2 = k_s
    twopi = 2.0 * math.pi
    identity = eye(N_total, format="csr", dtype=complex)
    Pi1 = -1j * D1_full + A1_op + (twopi * k1) * identity
    Pi2 = -1j * D2_full + A2_op + (twopi * k2) * identity
    return Pi1.tocsr(), Pi2.tocsr()


def build_kinetic_operator(
    M11: np.ndarray,
    M12: np.ndarray,
    M21_or_M22,
    M22_or_Pi1,
    Pi1_or_Pi2,
    Pi2_or_Ns,
    Ns_or_Nb,
    Nb_or_ds=None,
    order: int = 4,
) -> csr_matrix:
    """Kinetic term: (1/2) Σ_αβ Π_α M̃_αβ Π_β with full retained-space blocks.

    Supports both the current signature ``(M11, M12, M21, M22, Pi1, Pi2, Ns, Nb)``
    and the legacy helper signature ``(M11, M12, M22, A1, A2, Ns, Nb, ds, order)``.
    """
    if sparse.issparse(Pi1_or_Pi2):
        M21 = M21_or_M22
        M22 = M22_or_Pi1
        Pi1 = Pi1_or_Pi2
        Pi2 = Pi2_or_Ns
        Ns = int(Ns_or_Nb)
        if Nb_or_ds is None:
            raise ValueError("Nb must be provided when passing prebuilt Pi operators")
        Nb = int(Nb_or_ds)
    else:
        M21 = M12
        M22 = M21_or_M22
        A1 = M22_or_Pi1
        A2 = Pi1_or_Pi2
        Ns = int(Pi2_or_Ns)
        Nb = int(Ns_or_Nb)
        if Nb_or_ds is None:
            raise ValueError("ds must be provided when building Pi operators internally")
        ds = float(Nb_or_ds)
        Pi1, Pi2 = build_covariant_derivative_operators(A1, A2, Ns, Nb, ds, order)

    N_s = Ns * Ns
    M11_op = _build_block_diagonal(M11.reshape(N_s, Nb, Nb), N_s, Nb)
    M12_op = _build_block_diagonal(M12.reshape(N_s, Nb, Nb), N_s, Nb)
    M21_op = _build_block_diagonal(M21.reshape(N_s, Nb, Nb), N_s, Nb)
    M22_op = _build_block_diagonal(M22.reshape(N_s, Nb, Nb), N_s, Nb)

    K = 0.5 * (
        Pi1 @ M11_op @ Pi1
        + Pi1 @ M12_op @ Pi2
        + Pi2 @ M21_op @ Pi1
        + Pi2 @ M22_op @ Pi2
    )
    K = (K + K.conj().T) / 2
    return K


def build_scalar_potential_operator(
    Phi: np.ndarray, Ns: int, Nb: int,
) -> csr_matrix:
    """Generic N×N scalar potential (Born-Huang or slow coefficient)."""
    N_s = Ns * Ns
    return _build_block_diagonal(Phi.reshape(N_s, Nb, Nb), N_s, Nb)


def assemble_exact_tm_hamiltonian(
    Lambda: np.ndarray,
    tm_exact: dict,
    Ns: int,
    Nb: int,
    *,
    fd_order: int = 4,
    k_s: tuple[float, float] = (0.0, 0.0),
    ds_override: float | None = None,
) -> csr_matrix:
    """Assemble the raw exact TM operator from the primitive Phase 1 blocks.

    Args:
        ds_override: if set, use this FD grid spacing instead of 1/Ns.
            Required when the grid has been tiled (supercell_tiling > 1)
            so that the derivative operator retains the physical moiré spacing.
    """
    ds = ds_override if ds_override is not None else 1.0 / Ns
    N_s = Ns * Ns
    n_remote = int(tm_exact["n_remote"])
    eta = float(tm_exact.get("eta", 1.0))

    Pi1_p, Pi2_p = build_covariant_derivative_operators(
        None, None, Ns, Nb, ds, fd_order, k_s=k_s,
    )

    H = build_potential_operator(Lambda)

    v1_pp = _build_block_diagonal(tm_exact["v1_pp"].reshape(N_s, Nb, Nb), N_s, Nb)
    v2_pp = _build_block_diagonal(tm_exact["v2_pp"].reshape(N_s, Nb, Nb), N_s, Nb)
    gamma1_pp = eta * _build_block_diagonal(tm_exact["gamma1_pp"].reshape(N_s, Nb, Nb), N_s, Nb)
    H = H + v1_pp @ Pi1_p + v2_pp @ Pi2_p + gamma1_pp

    direct_metric = _build_block_diagonal(
        tm_exact["direct_metric"].reshape(N_s, Nb, Nb), N_s, Nb
    )
    direct_b1 = eta * _build_block_diagonal(tm_exact["direct_b1"].reshape(N_s, Nb, Nb), N_s, Nb)
    direct_b2 = eta * _build_block_diagonal(tm_exact["direct_b2"].reshape(N_s, Nb, Nb), N_s, Nb)
    direct_gamma2 = (eta ** 2) * _build_block_diagonal(
        tm_exact["direct_gamma2"].reshape(N_s, Nb, Nb), N_s, Nb
    )
    metric = tm_exact["metric"]
    H = H + direct_metric @ (
        metric[0, 0] * (Pi1_p @ Pi1_p)
        + metric[0, 1] * (Pi1_p @ Pi2_p)
        + metric[1, 0] * (Pi2_p @ Pi1_p)
        + metric[1, 1] * (Pi2_p @ Pi2_p)
    )
    H = H - 1j * (direct_b1 @ Pi1_p + direct_b2 @ Pi2_p) + direct_gamma2

    if n_remote > 0:
        Pi1_q, Pi2_q = build_covariant_derivative_operators(
            None, None, Ns, n_remote, ds, fd_order, k_s=k_s,
        )
        v1_pq = _build_block_operator(
            tm_exact["v1_pq"].reshape(N_s, Nb, n_remote), N_s, Nb, n_remote
        )
        v2_pq = _build_block_operator(
            tm_exact["v2_pq"].reshape(N_s, Nb, n_remote), N_s, Nb, n_remote
        )
        v1_qp = _build_block_operator(
            tm_exact["v1_qp"].reshape(N_s, n_remote, Nb), N_s, n_remote, Nb
        )
        v2_qp = _build_block_operator(
            tm_exact["v2_qp"].reshape(N_s, n_remote, Nb), N_s, n_remote, Nb
        )
        gamma1_pq = eta * _build_block_operator(
            tm_exact["gamma1_pq"].reshape(N_s, Nb, n_remote), N_s, Nb, n_remote
        )
        gamma1_qp = eta * _build_block_operator(
            tm_exact["gamma1_qp"].reshape(N_s, n_remote, Nb), N_s, n_remote, Nb
        )
        resolvent_q = _build_block_diagonal(
            tm_exact["resolvent_q"].reshape(N_s, n_remote, n_remote), N_s, n_remote
        )

        left = v1_pq @ Pi1_q + v2_pq @ Pi2_q + gamma1_pq
        right = v1_qp @ Pi1_p + v2_qp @ Pi2_p + gamma1_qp
        H = H - left @ resolvent_q @ right

    return H.tocsr()


# ==============================================================================
# M_inv Regularization
# ==============================================================================

def regularize_mass_tensor(M_diag, max_trace: float) -> np.ndarray:
    """Clamp 2×2 mass tensor eigenvalues at each (spatial, band) point.

    At near-degeneracy hotspots, M_inv ∝ 1/Δgap can diverge; this clamps it.

    Args:
        M_diag: (Ns, Ns, Nb, 2, 2) diagonal-band mass tensor
        max_trace: maximum allowed eigenvalue magnitude
    """
    M = M_diag.copy()
    shape = M.shape[:3]  # (Ns1, Ns2, Nb)
    n_clamped = 0
    for idx in np.ndindex(*shape):
        mat = M[idx]
        eigs, vecs = np.linalg.eigh(mat)
        if np.max(np.abs(eigs)) > max_trace:
            eigs = np.clip(eigs, -max_trace, max_trace)
            M[idx] = vecs @ np.diag(eigs) @ vecs.T
            n_clamped += 1
    if n_clamped > 0:
        total = int(np.prod(shape))
        log(f"      M_inv regularized: clamped {n_clamped}/{total} points")
    return M


# ==============================================================================
# Hamiltonian Assembly
# ==============================================================================

def assemble_hamiltonian(
    Lambda: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    M11: np.ndarray,
    M12: np.ndarray,
    *operator_args,
    include_drift: bool = True,
    include_kinetic: bool = True,
    include_born_huang: bool = True,
    include_slow_coeff: bool = True,
    fd_order: int = 4,
    M_inv_max_trace: float | None = None,
    k_s: tuple[float, float] = (0.0, 0.0),
    ds_override: float | None = None,
) -> csr_matrix:
    """Assemble the full multi-band envelope Hamiltonian.

    All inputs should be on the moiré grid (Ns, Ns, Nb, Nb) and already
    in fractional-coordinate units.

    Args:
        k_s: moiré Bloch momentum in fractional reciprocal coords, included through
            the full covariant derivatives Π_α = -i∂_α + A_α + 2πk_{s,α}.
        ds_override: if set, use this FD grid spacing instead of 1/Ns.
            Required when the grid has been tiled (supercell_tiling > 1)
            so that the derivative operator retains the physical moiré spacing.
    """
    if len(operator_args) < 7:
        raise TypeError("assemble_hamiltonian requires at least 7 operator arguments after M12")

    if np.isscalar(operator_args[5]) and np.isscalar(operator_args[6]):
        M21 = M12
        M22, A1, A2, Phi_BH, Phi_slow, Ns, Nb, *extra = operator_args
    else:
        M21, M22, A1, A2, Phi_BH, Phi_slow, Ns, Nb, *extra = operator_args

    if extra:
        raise TypeError("assemble_hamiltonian received too many positional arguments")

    Ns = int(Ns)
    Nb = int(Nb)
    N_total = Ns * Ns * Nb
    ds = ds_override if ds_override is not None else 1.0 / Ns

    log(f"    Building Hamiltonian: {N_total}×{N_total} ({Ns}×{Ns}×{Nb})")

    Pi1, Pi2 = build_covariant_derivative_operators(
        A1, A2, Ns, Nb, ds, fd_order, k_s=k_s,
    )

    # 1. Potential
    log("    - Potential Λ...")
    H = build_potential_operator(Lambda)
    log(f"      nnz = {H.nnz}")

    # 2. Drift
    if include_drift:
        log("    - Drift (1/2)(ṽ·Π + Π·ṽ)...")
        T = build_drift_operator(v1, v2, Pi1, Pi2, Ns, Nb)
        H = H + T
        log(f"      nnz = {T.nnz}")

    # 3. Kinetic
    if include_kinetic:
        log("    - Kinetic (1/2) Π M̃ Π...")
        K = build_kinetic_operator(M11, M12, M21, M22, Pi1, Pi2, Ns, Nb)
        H = H + K
        log(f"      nnz = {K.nnz}")

    # 4. Born-Huang
    if include_born_huang and Phi_BH is not None:
        if np.max(np.abs(Phi_BH)) > 1e-15:
            log("    - Born-Huang Φ_BH...")
            U_BH = build_scalar_potential_operator(Phi_BH, Ns, Nb)
            H = H + U_BH
            log(f"      nnz = {U_BH.nnz}")

    # 5. Slow coefficient (TE only)
    if include_slow_coeff and Phi_slow is not None:
        if np.max(np.abs(Phi_slow)) > 1e-15:
            log("    - Slow coefficient S...")
            U_S = build_scalar_potential_operator(Phi_slow, Ns, Nb)
            H = H + U_S
            log(f"      nnz = {U_S.nnz}")

    H = H.tocsr()
    log(f"    Total nnz = {H.nnz}")
    return H


# ==============================================================================
# Sigma Selection
# ==============================================================================

def compute_sigma(
    Lambda: np.ndarray, M11: np.ndarray, M22: np.ndarray,
    target_idx: int = 0,
) -> tuple[float, dict]:
    """Auto-select shift-invert target σ for eigsh.

    For each band, determines if it's HOLE (negative mass trace) or
    ELECTRON (positive mass trace), then targets the appropriate
    potential edge.

    Args:
        Lambda: (Ns, Ns, Nb, Nb) potential (diagonal)
        M11, M22: (Ns, Ns, Nb, Nb) mass tensor diagonal components
        target_idx: which band to target
    """
    Nb = Lambda.shape[2]
    per_band = []
    for n in range(Nb):
        tr = (M11[..., n, n] + M22[..., n, n]).real
        mean_tr = float(np.mean(tr))
        V_n = Lambda[..., n, n].real
        V_min, V_max = float(np.min(V_n)), float(np.max(V_n))
        band_type = "HOLE" if mean_tr < 0 else "ELECTRON"
        per_band.append({
            "mean_trace": mean_tr, "type": band_type,
            "V_min": V_min, "V_max": V_max,
        })

    info = {"Nb": Nb, "target_idx": target_idx, "per_band": per_band}

    if Nb <= 2:
        # Dirac-like: target manifold center
        center = np.mean([Lambda[..., n, n].real for n in range(Nb)], axis=0)
        sigma = float(np.mean(center))
        info["method"] = "manifold_center"
    else:
        t = per_band[target_idx]
        sigma = t["V_max"] if t["type"] == "HOLE" else t["V_min"]
        info["method"] = f"V_{'max' if t['type'] == 'HOLE' else 'min'} band {target_idx} [{t['type']}]"

    info["sigma"] = sigma
    return sigma, info


# ==============================================================================
# Eigensolve
# ==============================================================================

def solve_envelope(
    H: csr_matrix, n_modes: int, sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the multi-band envelope eigenvalue problem.

    Uses shift-invert Lanczos (eigsh) for sparse Hermitian H.

    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue
    """
    N = H.shape[0]
    H_herm = 0.5 * (H + H.conj().T)

    # Downcast to complex64 (float32 per component) for ~1.3× speedup
    # and 50% memory reduction.  Eigenvalue accuracy is ~6e-5 relative,
    # well below FD discretisation error.
    H_herm = H_herm.astype(np.complex64)

    h_diag = H_herm.diagonal().real
    log(f"    H diagonal range: [{np.min(h_diag):.4f}, {np.max(h_diag):.4f}]")

    if sigma is None:
        sigma = float(np.min(h_diag))
    log(f"    σ = {sigma:.6f}")

    log(f"    Solving ({N}×{N}, {n_modes} modes, complex64)...")
    t0 = time.time()

    try:
        eigenvalues, eigenvectors = eigsh(
            H_herm, k=n_modes, sigma=np.float32(sigma), which="LM",
            maxiter=10000, tol=1e-7,
        )
    except Exception as e:
        log(f"    eigsh failed ({e}), falling back to dense solve...")
        H_dense = H_herm.toarray()
        all_eigs, all_vecs = np.linalg.eigh(H_dense)
        order = np.argsort(np.abs(all_eigs - sigma))
        eigenvalues = all_eigs[order[:n_modes]]
        eigenvectors = all_vecs[:, order[:n_modes]]

    elapsed = time.time() - t0
    log(f"    Eigensolve: {elapsed:.2f}s")

    order = np.argsort(eigenvalues)
    return eigenvalues[order], eigenvectors[:, order]


# ==============================================================================
# Moiré BZ k-path
# ==============================================================================

def moire_bz_path(lattice_type: str, n_per_segment: int = 30) -> tuple[np.ndarray, np.ndarray, list[tuple[float, str]]]:
    """Generate high-symmetry k-path in fractional reciprocal coords of the moiré lattice.

    Returns:
        k_points: (N_k, 2) array of fractional reciprocal coordinates
        k_dist:   (N_k,) cumulative distance along path (for plotting x-axis)
        ticks:    list of (distance, label) for high-symmetry points
    """
    if lattice_type == "square":
        # Γ → X → M → Γ
        vertices = np.array([
            [0.0, 0.0],    # Γ
            [0.5, 0.0],    # X
            [0.5, 0.5],    # M
            [0.0, 0.0],    # Γ
        ])
        labels = ["Γ", "X", "M", "Γ"]
    elif lattice_type in ("hex", "honeycomb"):
        # Γ → M → K → Γ
        vertices = np.array([
            [0.0,       0.0],        # Γ
            [0.5,       0.0],        # M
            [1.0 / 3.0, 1.0 / 3.0], # K
            [0.0,       0.0],        # Γ
        ])
        labels = ["Γ", "M", "K", "Γ"]
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    segments = []
    for i in range(len(vertices) - 1):
        t = np.linspace(0, 1, n_per_segment, endpoint=(i == len(vertices) - 2))
        seg = vertices[i][None, :] * (1 - t[:, None]) + vertices[i + 1][None, :] * t[:, None]
        segments.append(seg)

    k_points = np.vstack(segments)

    # Compute cumulative distance
    dk = np.diff(k_points, axis=0)
    seg_len = np.sqrt(np.sum(dk**2, axis=1))
    k_dist = np.zeros(len(k_points))
    k_dist[1:] = np.cumsum(seg_len)

    # Tick positions
    ticks = [(k_dist[0], labels[0])]
    idx = 0
    for i in range(len(vertices) - 2):
        idx += n_per_segment
        ticks.append((k_dist[idx], labels[i + 1]))
    ticks.append((k_dist[-1], labels[-1]))

    return k_points, k_dist, ticks


def solve_k_path(
    Lambda: np.ndarray,
    v1: np.ndarray, v2: np.ndarray,
    M11: np.ndarray, M12: np.ndarray, M21: np.ndarray, M22: np.ndarray,
    A1: np.ndarray | None, A2: np.ndarray | None,
    Phi_BH: np.ndarray | None,
    Phi_slow: np.ndarray | None,
    Ns: int, Nb: int,
    k_points: np.ndarray,
    n_modes: int = 20,
    sigma: float | None = None,
    *,
    target_frequency: float | None = None,
    lambda_ref: float | None = None,
    include_drift: bool = True,
    include_kinetic: bool = True,
    include_born_huang: bool = True,
    include_slow_coeff: bool = True,
    fd_order: int = 4,
    tm_exact: dict | None = None,
) -> np.ndarray:
    """Solve the envelope Hamiltonian along a moiré BZ k-path.

    For each k_s in k_points, builds H(k_s) and eigensolves.

    Args:
        Lambda..Phi_slow: EA ingredients on moiré grid (Ns, Ns, Nb, Nb)
        k_points: (N_k, 2) fractional reciprocal coords
        n_modes: number of eigenvalues per k-point
        sigma: shift-invert target (auto if None)
        target_frequency: preferred alternative — a physical frequency in c/a;
            converted to sigma internally (requires lambda_ref).
        lambda_ref: reference eigenvalue, needed when target_frequency is used.
        tm_exact: if provided, use assemble_exact_tm_hamiltonian instead of
            the compact operator.

    Returns:
        bands: (N_k, n_modes) eigenvalue array
    """
    N_k = len(k_points)
    bands = np.zeros((N_k, n_modes))

    # Resolve sigma: target_frequency > explicit sigma > auto
    if target_frequency is not None:
        if lambda_ref is None:
            raise ValueError("lambda_ref is required when target_frequency is given")
        sigma = frequency_to_sigma(target_frequency, lambda_ref)
    elif sigma is None:
        sigma, _ = compute_sigma(Lambda, M11, M22, target_idx=0)

    use_exact_tm = tm_exact is not None
    label = "exact-TM" if use_exact_tm else "compact"
    log(f"  k-path solve ({label}): {N_k} k-points, {n_modes} modes, σ={sigma:.6f}")
    t0_total = time.time()

    for ik, ks in enumerate(k_points):
        # Suppress per-point logging for speed
        old_log = globals().get("_log_fn")
        globals()["_log_fn"] = lambda _: None

        if use_exact_tm:
            H = assemble_exact_tm_hamiltonian(
                Lambda, tm_exact, Ns, Nb,
                fd_order=fd_order,
                k_s=(float(ks[0]), float(ks[1])),
            )
        else:
            H = assemble_hamiltonian(
                Lambda, v1, v2, M11, M12, M21, M22, A1, A2,
                Phi_BH, Phi_slow, Ns, Nb,
                include_drift=include_drift,
                include_kinetic=include_kinetic,
                include_born_huang=include_born_huang,
                include_slow_coeff=include_slow_coeff,
                fd_order=fd_order,
                k_s=(float(ks[0]), float(ks[1])),
            )

        evals, _ = solve_envelope(H, n_modes, sigma)
        bands[ik, :] = evals

        globals()["_log_fn"] = old_log

        if (ik + 1) % 10 == 0 or ik == N_k - 1:
            log(f"    k-point {ik + 1}/{N_k} done")

    elapsed = time.time() - t0_total
    log(f"  k-path solve complete: {elapsed:.1f}s total")
    return bands


# ==============================================================================
# Mode Analysis
# ==============================================================================

def eigenvalue_to_frequency(eigenvalue: float, lambda_ref: float) -> float:
    """Convert Hamiltonian eigenvalue Δλ to MPB-style frequency f = ω/(2πc/a)."""
    lam = lambda_ref + eigenvalue
    if lam < 0:
        return float("nan")
    return math.sqrt(lam) / (2.0 * math.pi)


def frequency_to_sigma(target_frequency: float, lambda_ref: float) -> float:
    """Convert a target frequency f (in c/a units) to a shift-invert sigma.

    Inverse of eigenvalue_to_frequency:
        sigma = (2π f)² − λ_ref
    """
    return (2.0 * math.pi * target_frequency) ** 2 - lambda_ref


def resolve_sigma(
    config: dict,
    lambda_ref: float,
    Lambda: np.ndarray | None = None,
    M11: np.ndarray | None = None,
    M22: np.ndarray | None = None,
    target_band: int = 0,
) -> tuple[float, str]:
    """Determine the eigsh shift-invert sigma from the config.

    Priority:
        1. config["target_frequency"] — converted via frequency_to_sigma.
        2. config["sigma"]             — raw eigenvalue-space override.
        3. Auto-select via compute_sigma (needs Lambda, M11, M22).

    Returns (sigma_value, description_string).
    """
    tf = config.get("target_frequency")
    if tf is not None:
        sigma = frequency_to_sigma(float(tf), lambda_ref)
        f_check = eigenvalue_to_frequency(sigma, lambda_ref)
        desc = (f"target_frequency={tf:.6f} c/a -> sigma={sigma:.6f} "
                f"(back-check f={f_check:.6f})")
        return sigma, desc

    raw = config.get("sigma")
    if raw is not None:
        return float(raw), f"sigma={raw:.6f} (explicit override)"

    if Lambda is not None and M11 is not None and M22 is not None:
        sigma, info = compute_sigma(Lambda, M11, M22, target_band)
        return sigma, f"auto: {info['method']}, sigma={sigma:.6f}"

    raise ValueError(
        "Cannot determine sigma: provide 'target_frequency', 'sigma', "
        "or Lambda/M fields for auto-selection."
    )


def compute_mode_stats(
    F: np.ndarray, eigenvalue: float, lambda_ref: float,
) -> dict:
    """Compute statistics for a single envelope mode.

    Args:
        F: (Ns1, Ns2, Nb) spinor envelope
        eigenvalue: Δλ from eigensolver
        lambda_ref: reference eigenvalue
    """
    Ns1, Ns2, Nb = F.shape
    prob_band = np.sum(np.abs(F) ** 2, axis=(0, 1))
    prob_band /= np.sum(prob_band) + 1e-30
    prob_total = np.sum(np.abs(F) ** 2, axis=2)
    prob_total /= np.sum(prob_total) + 1e-30

    s1 = np.arange(Ns1, dtype=float) / Ns1
    s2 = np.arange(Ns2, dtype=float) / Ns2
    S1, S2 = np.meshgrid(s1, s2, indexing="ij")
    cx = float(np.sum(S1 * prob_total))
    cy = float(np.sum(S2 * prob_total))
    spread = float(np.sqrt(np.sum(((S1 - cx) ** 2 + (S2 - cy) ** 2) * prob_total)))
    ipr = float(np.sum(prob_total ** 2))

    # Confinement: fraction in central 25% area
    a1, b1 = Ns1 // 4, 3 * Ns1 // 4
    a2, b2 = Ns2 // 4, 3 * Ns2 // 4
    conf = float(np.sum(prob_total[a1:b1, a2:b2]))

    freq = eigenvalue_to_frequency(eigenvalue, lambda_ref)

    return {
        "frequency": freq,
        "eigenvalue": float(eigenvalue),
        "lambda_ref": float(lambda_ref),
        "lambda_total": float(lambda_ref + eigenvalue),
        "prob_per_band": prob_band.tolist(),
        "dominant_band": int(np.argmax(prob_band)),
        "dominant_band_weight": float(np.max(prob_band)),
        "centroid": [cx, cy],
        "spread": spread,
        "ipr": ipr,
        "confinement": conf,
    }


# ==============================================================================
# I/O
# ==============================================================================

def save_json(data, path: Path) -> None:
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


def save_phase2_h5(
    path: Path,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    H: csr_matrix,
    Lambda: np.ndarray,
    moire_meta: dict,
    phase1_meta: dict,
    config: dict,
    mode_stats: list[dict],
    Ns: int,
    Nb: int,
) -> None:
    """Save Phase 2 results to HDF5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_modes = len(eigenvalues)

    with h5py.File(path, "w") as hf:
        # Eigenvalues / eigenvectors
        hf.create_dataset("eigenvalues", data=eigenvalues)
        hf.create_dataset("eigenvectors", data=eigenvectors, compression="gzip")

        # Spinor envelopes: (n_modes, Ns, Ns, Nb)
        F_all = eigenvectors.T.reshape(n_modes, Ns, Ns, Nb)
        hf.create_dataset("F_spinor", data=F_all, compression="gzip")

        # Hamiltonian (sparse CSR)
        hf.create_dataset("H_data", data=H.data)
        hf.create_dataset("H_indices", data=H.indices)
        hf.create_dataset("H_indptr", data=H.indptr)
        hf.attrs["H_shape_0"] = H.shape[0]
        hf.attrs["H_shape_1"] = H.shape[1]

        # Potential (for visualization)
        hf.create_dataset("Lambda", data=Lambda, compression="gzip")

        # Moiré metadata
        hf.attrs["theta_deg"] = moire_meta["theta_deg"]
        hf.attrs["eta"] = moire_meta["eta"]
        hf.attrs["moire_length"] = moire_meta["moire_length"]
        hf.create_dataset("B_moire", data=moire_meta["B_moire"])
        hf.create_dataset("B_mono", data=moire_meta["B_mono"])

        # Phase 1 metadata
        hf.attrs["polarization"] = phase1_meta["polarization"]
        hf.attrs["lattice_type"] = phase1_meta["lattice_type"]
        hf.attrs["n_retained"] = phase1_meta["n_retained"]
        hf.attrs["n_remote"] = phase1_meta["n_remote"]
        hf.attrs["n_registry"] = phase1_meta["n_reg"]
        hf.attrs["k0_cartesian"] = phase1_meta["k0_cartesian"]

        # Solve metadata
        hf.attrs["Ns"] = Ns
        hf.attrs["Nb"] = Nb
        hf.attrs["n_modes"] = n_modes
        hf.attrs["pipeline_version"] = "V4"
        hf.attrs["solver"] = "blaze"

        # Config flags
        for k in ("include_drift", "include_kinetic", "include_born_huang",
                   "include_slow_coeff", "fd_order"):
            if k in config:
                hf.attrs[k] = config[k]

    log(f"  Saved {path}")


# ==============================================================================
# Process One Case
# ==============================================================================

def process_case(
    phase1_path: Path,
    config: dict,
    output_dir: Path,
) -> dict:
    """Run Phase 2 (data prep + eigensolve) for one Phase 1 HDF5 file.

    Args:
        phase1_path: path to Phase 1 HDF5 (e.g. square_tm_phase1.h5)
        config: dict with keys:
            - Ns: moiré grid resolution (default: same as n_registry)
            - n_modes: number of eigenmodes (default: 20)
            - target_band: target band index for sigma (default: 0)
            - sigma: manual sigma override (default: auto)
            - include_drift, include_kinetic, include_born_huang,
              include_slow_coeff: bool flags (default: True)
            - fd_order: FD stencil order (default: 4)
            - M_inv_max_trace: mass tensor clamp (default: None)
        output_dir: where to save results

    Returns:
        Summary dict with eigenvalue info and mode stats.
    """
    phase1_path = Path(phase1_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_label = phase1_path.stem.replace("_phase1", "")

    log(f"\n{'='*72}")
    log(f"Phase 2 V4: {case_label}")
    log(f"{'='*72}")

    # ── 1. Load Phase 1 ──────────────────────────────────────────────────
    log("  Loading Phase 1 master map...")
    p1 = load_phase1_h5(phase1_path)

    n_reg = p1["n_reg"]
    Nb = p1["n_retained"]
    band_lo = p1.get("band_lo", 0)
    pol = p1["polarization"]
    lattice_type = p1["lattice_type"]

    log(f"  Registry: {n_reg}×{n_reg}, N_ret={Nb}, band_lo={band_lo}, pol={pol}, lattice={lattice_type}")

    # ── 2. Moiré geometry ────────────────────────────────────────────────
    commensurate_mn = config.get("commensurate_mn")
    if commensurate_mn is not None:
        cm, cn = int(commensurate_mn[0]), int(commensurate_mn[1])
        theta_deg = commensurate_angle(cm, cn)
        moire = commensurate_moire_metadata(lattice_type, 1.0, cm, cn)
        log(f"  Commensurate mode: (m,n) = ({cm},{cn})")
        log(f"  N_cells = {moire['N_cells']}, L_super = {moire['L_super']:.4f} a")
    else:
        theta_deg = config.get("theta_deg", p1.get("theta_deg"))
        if theta_deg is None:
            raise ValueError("theta_deg must be specified in config or Phase 1 HDF5")
        moire = compute_moire_metadata(lattice_type, 1.0, theta_deg)

    B_moire = moire["B_moire"]
    B_inv = np.linalg.inv(B_moire)
    eta = moire["eta"]
    L_m = moire["moire_length"]

    log(f"  θ = {theta_deg}°, η = {eta:.4f}, L_m = {L_m:.2f} a")

    # k-point: always a single Bloch momentum (default Γ).
    # For commensurate angles, use supercell_tiling instead of k-folding.
    k_s_cfg = config.get("k_s", (0.0, 0.0))
    k_points = [((float(k_s_cfg[0]), float(k_s_cfg[1])), "k_s")]
    log(f"  k-points: {[(lbl, ks) for ks, lbl in k_points]}")

    # Supercell tiling: replicate fields N×N to fold the BZ.
    supercell_tiling = config.get("supercell_tiling", 1)
    if supercell_tiling > 1:
        log(f"  Supercell tiling: {supercell_tiling}×{supercell_tiling} "
            f"({supercell_tiling**2} moiré cells)")

    # ── 3. Interpolate to moiré grid ─────────────────────────────────────
    Ns = config.get("Ns", n_reg)
    log(f"  Moiré grid: {Ns}×{Ns} (registry: {n_reg}×{n_reg})")

    def _interp(key):
        if key in p1:
            return interpolate_to_moire_grid(p1[key], n_reg, Ns)
        return None

    eigenvalues = _interp("eigenvalues")       # (Ns, Ns, Nb)
    velocity_x = _interp("velocity_x")         # (Ns, Ns, Nb, Nt)
    velocity_y = _interp("velocity_y")
    mass_xx = _interp("mass_xx")               # (Ns, Ns, Nb, Nb)
    mass_xy = _interp("mass_xy")
    mass_yx = _interp("mass_yx")
    mass_yy = _interp("mass_yy")
    berry_x = _interp("berry_x")               # (Ns, Ns, Nb, Nb)
    berry_y = _interp("berry_y")
    born_huang = _interp("born_huang")          # (Ns, Ns, Nb, Nb)
    slow_coeff = _interp("slow_coefficient")    # (Ns, Ns, Nb, Nb)
    born_huang_tensor_xx = _interp("born_huang_tensor_xx")
    born_huang_tensor_xy = _interp("born_huang_tensor_xy")
    born_huang_tensor_yx = _interp("born_huang_tensor_yx")
    born_huang_tensor_yy = _interp("born_huang_tensor_yy")
    slow_coeff_tensor_xx = _interp("slow_coefficient_tensor_xx")
    slow_coeff_tensor_xy = _interp("slow_coefficient_tensor_xy")
    slow_coeff_tensor_yx = _interp("slow_coefficient_tensor_yx")
    slow_coeff_tensor_yy = _interp("slow_coefficient_tensor_yy")
    metric_derivative_x = _interp("metric_derivative_x")
    metric_derivative_y = _interp("metric_derivative_y")
    overlap = _interp("overlap")
    exact_tm_fields = {
        key: _interp(key) for key in EXACT_TM_DATASET_KEYS if key in p1
    }

    log("  Interpolation done.")

    if eigenvalues is None:
        raise ValueError("Phase 1 archive is missing eigenvalues")
    if velocity_x is None or velocity_y is None:
        raise ValueError("Phase 1 archive is missing velocity matrices")
    if any(field is None for field in (mass_xx, mass_xy, mass_yx, mass_yy)):
        raise ValueError("Phase 1 archive is missing mass tensor blocks")

    # ── 3b. Supercell tiling ─────────────────────────────────────────────
    # When supercell_tiling > 1, replicate all fields N×N to represent a
    # larger real-space domain.  This folds the Brillouin zone so that the
    # Γ-point eigensolve captures all folded k-points automatically.
    # CRITICAL: ds must stay at 1/Ns_original so the FD stencil keeps the
    # correct physical moiré spacing.
    ds_override = None
    if supercell_tiling > 1:
        ds_override = 1.0 / Ns  # physical spacing of the primitive moiré grid

        def _tile(field):
            return tile_fields_2d(field, supercell_tiling) if field is not None else None

        eigenvalues = _tile(eigenvalues)
        velocity_x = _tile(velocity_x)
        velocity_y = _tile(velocity_y)
        mass_xx = _tile(mass_xx)
        mass_xy = _tile(mass_xy)
        mass_yx = _tile(mass_yx)
        mass_yy = _tile(mass_yy)
        berry_x = _tile(berry_x)
        berry_y = _tile(berry_y)
        born_huang = _tile(born_huang)
        slow_coeff = _tile(slow_coeff)
        born_huang_tensor_xx = _tile(born_huang_tensor_xx)
        born_huang_tensor_xy = _tile(born_huang_tensor_xy)
        born_huang_tensor_yx = _tile(born_huang_tensor_yx)
        born_huang_tensor_yy = _tile(born_huang_tensor_yy)
        slow_coeff_tensor_xx = _tile(slow_coeff_tensor_xx)
        slow_coeff_tensor_xy = _tile(slow_coeff_tensor_xy)
        slow_coeff_tensor_yx = _tile(slow_coeff_tensor_yx)
        slow_coeff_tensor_yy = _tile(slow_coeff_tensor_yy)
        metric_derivative_x = _tile(metric_derivative_x)
        metric_derivative_y = _tile(metric_derivative_y)
        overlap = _tile(overlap)
        exact_tm_fields = {k: _tile(v) for k, v in exact_tm_fields.items()}

        Ns = supercell_tiling * Ns
        log(f"  Tiled grid: {Ns}×{Ns} (ds = {ds_override:.6f})")

    # ── 4. Reference eigenvalue ──────────────────────────────────────────
    target_band = config.get("target_band", 0)
    lambda_ref = float(np.mean(eigenvalues[..., band_lo + target_band]))
    log(f"  λ_ref = {lambda_ref:.6f} (mean of retained band {target_band}, absolute band {band_lo + target_band})")
    log(f"  f_ref = {math.sqrt(max(lambda_ref, 0)) / (2 * math.pi):.6f} (c/a)")

    # ── 5. Construct polarization-specific operator inputs ───────────────
    tm_operator_model = config.get("tm_operator_model", "compact")
    if pol == "TE":
        operator_inputs = build_te_operator_inputs(
            eigenvalues=eigenvalues,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            mass_xx=mass_xx,
            mass_xy=mass_xy,
            mass_yx=mass_yx,
            mass_yy=mass_yy,
            berry_x=berry_x,
            berry_y=berry_y,
            born_huang=born_huang,
            born_huang_tensor_xx=born_huang_tensor_xx,
            born_huang_tensor_xy=born_huang_tensor_xy,
            born_huang_tensor_yx=born_huang_tensor_yx,
            born_huang_tensor_yy=born_huang_tensor_yy,
            slow_coeff=slow_coeff,
            slow_coeff_tensor_xx=slow_coeff_tensor_xx,
            slow_coeff_tensor_xy=slow_coeff_tensor_xy,
            slow_coeff_tensor_yx=slow_coeff_tensor_yx,
            slow_coeff_tensor_yy=slow_coeff_tensor_yy,
            Ns=Ns,
            Nb=Nb,
            lambda_ref=lambda_ref,
            B_inv=B_inv,
            B_moire=B_moire,
            M_inv_max_trace=config.get("M_inv_max_trace"),
            band_lo=band_lo,
        )
    elif pol == "TM":
        operator_inputs = build_tm_operator_inputs(
            eigenvalues=eigenvalues,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            mass_xx=mass_xx,
            mass_xy=mass_xy,
            mass_yx=mass_yx,
            mass_yy=mass_yy,
            berry_x=berry_x,
            berry_y=berry_y,
            born_huang=born_huang,
            born_huang_tensor_xx=born_huang_tensor_xx,
            born_huang_tensor_xy=born_huang_tensor_xy,
            born_huang_tensor_yx=born_huang_tensor_yx,
            born_huang_tensor_yy=born_huang_tensor_yy,
            metric_derivative_x=metric_derivative_x,
            metric_derivative_y=metric_derivative_y,
            overlap=overlap,
            exact_tm_fields={key: value for key, value in exact_tm_fields.items() if value is not None},
            Ns=Ns,
            Nb=Nb,
            lambda_ref=lambda_ref,
            B_inv=B_inv,
            B_moire=B_moire,
            eta=eta,
            M_inv_max_trace=config.get("M_inv_max_trace"),
            tm_operator_model=tm_operator_model,
            band_lo=band_lo,
        )
    else:
        raise ValueError(f"Unsupported polarization: {pol}")

    Lambda = operator_inputs["Lambda"]
    log(f"  Λ range: [{Lambda.real.min():.6e}, {Lambda.real.max():.6e}]")

    # ── 6. Report transformed operator scale ─────────────────────────────
    log(f"  Operator model: {operator_inputs['operator_model']}")
    log(
        f"  Drift |ṽ| max: {max(np.max(np.abs(operator_inputs['v1'])), np.max(np.abs(operator_inputs['v2']))):.4e}"
    )
    log(
        f"  M̃ range: M̃₁₁∈[{operator_inputs['M11'].real.min():.2e}, {operator_inputs['M11'].real.max():.2e}], "
        f"M̃₂₂∈[{operator_inputs['M22'].real.min():.2e}, {operator_inputs['M22'].real.max():.2e}]"
    )
    if operator_inputs["Phi_BH"] is not None:
        log(
            f"  Φ_BH range: [{operator_inputs['Phi_BH'].real.min():.4e}, {operator_inputs['Phi_BH'].real.max():.4e}]"
        )
    if operator_inputs["Phi_slow"] is not None:
        log(
            f"  Slow coeff range: [{operator_inputs['Phi_slow'].real.min():.4e}, {operator_inputs['Phi_slow'].real.max():.4e}]"
        )

    # ── 7. Assemble Hamiltonian & solve ─────────────────────────────────
    fd_order = config.get("fd_order", 4)
    n_modes = config.get("n_modes", 20)

    # Sigma selection (k-independent — based on operator scale)
    sigma, sigma_desc = resolve_sigma(
        config, lambda_ref, operator_inputs["Lambda"], operator_inputs["M11"], operator_inputs["M22"], target_band,
    )
    log(f"  σ selection: {sigma_desc}")
    target_freq_used = config.get("target_frequency")
    if target_freq_used is not None:
        log(f"    target frequency = {target_freq_used:.6f} c/a")
    log(f"    σ = {sigma:.6f}")

    all_eigenvals = []
    all_eigenvecs = []
    all_mode_stats = []
    all_k_labels = []

    for k_s, k_label in k_points:
        log(f"\n  ── k-point: {k_label} = ({k_s[0]:.4f}, {k_s[1]:.4f}) ──")
        log("  Assembling Hamiltonian...")
        if operator_inputs["operator_model"] == "tm_exact":
            H = assemble_exact_tm_hamiltonian(
                operator_inputs["Lambda"],
                operator_inputs["tm_exact"],
                Ns,
                Nb,
                fd_order=fd_order,
                k_s=k_s,
                ds_override=ds_override,
            )
        else:
            H = assemble_hamiltonian(
                operator_inputs["Lambda"],
                operator_inputs["v1"],
                operator_inputs["v2"],
                operator_inputs["M11"],
                operator_inputs["M12"],
                operator_inputs["M21"],
                operator_inputs["M22"],
                operator_inputs["A1"],
                operator_inputs["A2"],
                operator_inputs["Phi_BH"],
                operator_inputs["Phi_slow"],
                Ns,
                Nb,
                include_drift=config.get("include_drift", True),
                include_kinetic=config.get("include_kinetic", True),
                include_born_huang=config.get("include_born_huang", True),
                include_slow_coeff=config.get("include_slow_coeff", pol == "TE"),
                fd_order=fd_order,
                k_s=k_s,
                ds_override=ds_override,
            )

        # ── Eigensolve ───────────────────────────────────────────────────
        eigenvals, eigenvecs = solve_envelope(H, n_modes, sigma)
        log(f"  Eigenvalue range: [{eigenvals.min():.6e}, {eigenvals.max():.6e}]")

        # ── Mode statistics ──────────────────────────────────────────────
        log("  Mode analysis:")
        mode_stats = []
        for i in range(n_modes):
            F = eigenvecs[:, i].reshape(Ns, Ns, Nb)
            stats = compute_mode_stats(F, eigenvals[i], lambda_ref)
            stats["mode_index"] = i
            stats["k_label"] = k_label
            stats["k_s"] = list(k_s)
            mode_stats.append(stats)

            if i < 10 or i == n_modes - 1:
                log(f"    Mode {i:3d}: f={stats['frequency']:.6f} c/a  "
                    f"dom_band={stats['dominant_band']} "
                    f"({stats['dominant_band_weight']:.0%})  "
                    f"IPR={stats['ipr']:.2e}  spread={stats['spread']:.3f}")

        all_eigenvals.append(eigenvals)
        all_eigenvecs.append(eigenvecs)
        all_mode_stats.extend(mode_stats)
        all_k_labels.append(k_label)

    # Concatenate eigenvalues across k-points
    eigenvals = np.concatenate(all_eigenvals)
    eigenvecs = all_eigenvecs[0]

    # ── 11. Save ─────────────────────────────────────────────────────────
    h5_path = output_dir / f"{case_label}_phase2.h5"
    save_phase2_h5(
        h5_path, all_eigenvals[0], eigenvecs, H, Lambda.real,
        moire, p1, config, all_mode_stats, Ns, Nb,
    )

    json_path = output_dir / f"{case_label}_modes.json"
    save_json(all_mode_stats, json_path)
    log(f"  Saved {json_path}")

    # Summary
    summary = {
        "case": case_label,
        "polarization": pol,
        "lattice_type": lattice_type,
        "theta_deg": theta_deg,
        "eta": eta,
        "moire_length": L_m,
        "Ns": Ns,
        "Nb": Nb,
        "n_modes": n_modes,
        "n_k_points": len(k_points),
        "k_points": [(list(ks), lbl) for ks, lbl in k_points],
        "operator_model": operator_inputs["operator_model"],
        "has_exact_tm_payload": bool(operator_inputs["has_exact_tm_payload"]),
        "lambda_ref": lambda_ref,
        "f_ref": math.sqrt(max(lambda_ref, 0)) / (2 * math.pi),
        "eigenvalue_range": [float(eigenvals.min()), float(eigenvals.max())],
        "frequency_range": [
            eigenvalue_to_frequency(eigenvals.min(), lambda_ref),
            eigenvalue_to_frequency(eigenvals.max(), lambda_ref),
        ],
        "sigma": sigma,
    }
    if supercell_tiling > 1:
        summary["supercell_tiling"] = supercell_tiling
    if commensurate_mn is not None:
        summary["commensurate_mn"] = [cm, cn]
        summary["N_cells"] = moire["N_cells"]
        summary["L_super"] = moire["L_super"]
    if config.get("target_frequency") is not None:
        summary["target_frequency"] = float(config["target_frequency"])
    return summary


# ==============================================================================
# Main Driver
# ==============================================================================

def run_phase2_v4(
    phase1_dir: str | Path,
    config: dict | None = None,
    output_dir: str | Path | None = None,
) -> None:
    """Run Phase 2 V4 for all Phase 1 HDF5 files in a directory.

    Args:
        phase1_dir: directory containing *_phase1.h5 files
        config: solver configuration dict
        output_dir: output directory (default: same as phase1_dir)
    """
    phase1_dir = Path(phase1_dir)
    if output_dir is None:
        output_dir = phase1_dir
    output_dir = Path(output_dir)
    config = config or {}

    log("=" * 72)
    log("PHASE 2 V4 (Blaze): Moiré Envelope Eigensolver")
    log("=" * 72)
    log(f"Phase 1 dir: {phase1_dir}")
    log(f"Output dir:  {output_dir}")

    # Find Phase 1 files
    h5_files = sorted(phase1_dir.glob("*_phase1.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No *_phase1.h5 files in {phase1_dir}")

    log(f"Found {len(h5_files)} Phase 1 file(s)")

    summaries = {}
    for h5_path in h5_files:
        try:
            summary = process_case(h5_path, config, output_dir)
            summaries[summary["case"]] = summary
        except Exception as e:
            log(f"\n  ERROR processing {h5_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Save master summary
    summary_path = output_dir / "phase2_summary.json"
    save_json(summaries, summary_path)

    log(f"\n{'='*72}")
    log(f"PHASE 2 V4 COMPLETE — {len(summaries)} case(s)")
    log(f"{'='*72}")
    log(f"Summary: {summary_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 V4: Moiré Envelope Eigensolver (Blaze)"
    )
    parser.add_argument("phase1_dir", help="Directory with *_phase1.h5 files")
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--theta", type=float, default=None, help="Twist angle (°)")
    parser.add_argument("--Ns", type=int, default=None, help="Moiré grid resolution")
    parser.add_argument("--n-modes", type=int, default=20, help="Number of modes")
    parser.add_argument("--target-band", type=int, default=0)
    parser.add_argument("--fd-order", type=int, default=4, choices=[2, 4])
    parser.add_argument("--no-drift", action="store_true")
    parser.add_argument("--no-kinetic", action="store_true")
    parser.add_argument("--no-born-huang", action="store_true")
    parser.add_argument("--M-clamp", type=float, default=None, help="M_inv trace clamp")
    parser.add_argument(
        "--commensurate", type=int, nargs=2, metavar=("M", "N"),
        default=None, help="Commensurate indices (m, n). Overrides --theta.",
    )
    parser.add_argument(
        "--tm-operator-model",
        choices=["compact", "exact"],
        default="compact",
        help="TM assembly branch to use. 'exact' currently requires extra projected TM data not yet archived.",
    )

    args = parser.parse_args()

    cfg = {
        "n_modes": args.n_modes,
        "target_band": args.target_band,
        "fd_order": args.fd_order,
        "include_drift": not args.no_drift,
        "include_kinetic": not args.no_kinetic,
        "include_born_huang": not args.no_born_huang,
        "tm_operator_model": args.tm_operator_model,
    }
    if args.commensurate is not None:
        cfg["commensurate_mn"] = tuple(args.commensurate)
    elif args.theta is not None:
        cfg["theta_deg"] = args.theta
    if args.Ns is not None:
        cfg["Ns"] = args.Ns
    if args.M_clamp is not None:
        cfg["M_inv_max_trace"] = args.M_clamp

    run_phase2_v4(args.phase1_dir, cfg, args.output_dir)
