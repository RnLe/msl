from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from .axes import GeometryParams
from .config import MPBConfig

try:  # pragma: no cover - heavy dependency
    import meep as mp
    from meep import mpb

    HAS_MPB = True
except Exception:  # pragma: no cover - when MPB is absent
    mp = None  # type: ignore[assignment]
    mpb = None  # type: ignore[assignment]
    HAS_MPB = False


class SolverName(str, Enum):
    AUTO = "auto"
    MPB = "mpb"
    SYNTHETIC = "synthetic"


class MPBUnavailableError(RuntimeError):
    """Raised when an MPB-backed solver is requested but dependencies are missing."""


BandSolver = Callable[[GeometryParams, MPBConfig, NDArray[np.float64]], NDArray[np.float32]]


def get_band_solver(choice: SolverName = SolverName.AUTO) -> Tuple[BandSolver, SolverName]:
    """Return the actual solver callable for the requested backend."""

    if choice == SolverName.SYNTHETIC:
        return synthetic_band_solver, SolverName.SYNTHETIC
    if choice == SolverName.MPB:
        if not HAS_MPB:
            raise MPBUnavailableError(
                "MPB/Meep Python bindings are not available in this environment."
            )
        return mpb_band_solver, SolverName.MPB
    # AUTO fallback
    if HAS_MPB:
        return mpb_band_solver, SolverName.MPB
    return synthetic_band_solver, SolverName.SYNTHETIC


def synthetic_band_solver(
    params: GeometryParams,
    mpb_cfg: MPBConfig,
    k_path: NDArray[np.float64],
    noise_fn: Callable[[int, int], float] | None = None,
) -> NDArray[np.float32]:
    """Generate deterministic stand-in band data.

    Useful for development environments that lack MPB or when running very fast
    smoke tests where real simulations would be too expensive.
    """

    num_k = k_path.shape[0]
    bands = np.zeros((mpb_cfg.num_bands, num_k), dtype=np.float32)
    k_norm = np.linalg.norm(k_path, axis=1)

    pol_factor = 1.0 if params.polarization.lower() == "te" else 1.15
    lattice_factor = 1.0 + 0.05 * (hash(params.lattice_type) % 5)
    eps_factor = 1.0 / math.sqrt(params.epsilon_bg)
    radius_factor = 1.0 - params.r_over_a

    for band_idx in range(mpb_cfg.num_bands):
        harmonic = band_idx + 1
        dispersion = np.sin(k_norm * math.pi * harmonic) + harmonic * 0.05
        bands[band_idx, :] = pol_factor * lattice_factor * eps_factor * dispersion + radius_factor * 0.1
        if noise_fn is not None:
            bands[band_idx, :] += noise_fn(band_idx, num_k)
    return bands


def mpb_band_solver(
    params: GeometryParams,
    mpb_cfg: MPBConfig,
    k_path: NDArray[np.float64],
) -> NDArray[np.float32]:  # pragma: no cover - requires heavy dependencies
    """Compute band diagrams using MPB/Meep."""

    if not HAS_MPB:
        raise MPBUnavailableError(
            "MPB solver requested but Meep/MPB bindings are unavailable."
        )

    geometry_lattice = _build_lattice(params.lattice_type)
    k_points = [_kvec(coords) for coords in k_path]
    geometry = _build_geometry(params)
    default_material = mp.Medium(epsilon=params.epsilon_bg)

    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        resolution=mpb_cfg.resolution,
        num_bands=mpb_cfg.num_bands,
        default_material=default_material,
    )

    polarization = params.polarization.lower()
    if polarization == "te":
        ms.run_te()
    elif polarization == "tm":
        ms.run_tm()
    else:
        raise ValueError(f"Unsupported polarization: {params.polarization}")

    freqs = np.array(ms.all_freqs, dtype=np.float64)
    if freqs.shape != (len(k_points), mpb_cfg.num_bands):
        freqs = np.reshape(freqs, (len(k_points), mpb_cfg.num_bands))
    return freqs.T.astype(np.float32)


def _build_lattice(lattice_type: str):  # pragma: no cover - thin wrapper
    if not HAS_MPB:
        raise MPBUnavailableError("MPB is required to build lattices")
    lattice_type = lattice_type.lower()
    if lattice_type == "square":
        return mp.Lattice(
            size=mp.Vector3(1, 1),
            basis1=mp.Vector3(1, 0),
            basis2=mp.Vector3(0, 1),
        )
    if lattice_type == "hex":
        return mp.Lattice(
            size=mp.Vector3(1, 1),
            basis1=mp.Vector3(0.5, np.sqrt(3) / 2.0),
            basis2=mp.Vector3(-0.5, np.sqrt(3) / 2.0),
        )
    raise ValueError(f"Unsupported lattice type: {lattice_type}")


def _build_geometry(params: GeometryParams):  # pragma: no cover - thin wrapper
    if not HAS_MPB:
        raise MPBUnavailableError("MPB is required to build geometry")
    hole_material = mp.Medium(epsilon=params.hole_eps)
    hole = mp.Cylinder(
        radius=params.r_over_a,
        material=hole_material,
        center=mp.Vector3(),
        height=mp.inf,
        axis=mp.Vector3(0, 0, 1),
    )
    return [hole]


def _kvec(coords: NDArray[np.float64]):  # pragma: no cover - thin wrapper
    if not HAS_MPB:
        raise MPBUnavailableError("MPB is required to build k-points")
    kx, ky = float(coords[0]), float(coords[1])
    return mp.Vector3(kx, ky)


__all__ = [
    "HAS_MPB",
    "SolverName",
    "MPBUnavailableError",
    "get_band_solver",
    "mpb_band_solver",
    "synthetic_band_solver",
]
