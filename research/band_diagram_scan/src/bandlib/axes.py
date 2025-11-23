from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import LatticePathConfig, RangeSpec, ScanConfig


@dataclass(frozen=True)
class GeometryIndices:
    lattice: int
    polarization: int
    hole: int
    epsilon: int
    radius: int


@dataclass(frozen=True)
class GeometryParams:
    lattice_type: str
    polarization: str
    epsilon_bg: float
    r_over_a: float
    hole_eps: float


@dataclass(frozen=True)
class Axes:
    lattice_types: Tuple[str, ...]
    polarizations: Tuple[str, ...]
    epsilon_background: NDArray[np.float64]
    r_over_a: NDArray[np.float64]
    hole_eps: NDArray[np.float64]
    k_paths: Dict[str, NDArray[np.float64]]
    num_kpoints: int

    def indices_to_params(self, idx: GeometryIndices) -> GeometryParams:
        return GeometryParams(
            lattice_type=self.lattice_types[idx.lattice],
            polarization=self.polarizations[idx.polarization],
            epsilon_bg=float(self.epsilon_background[idx.epsilon]),
            r_over_a=float(self.r_over_a[idx.radius]),
            hole_eps=float(self.hole_eps[idx.hole]),
        )

    def params_to_indices(self, params: GeometryParams) -> GeometryIndices:
        return GeometryIndices(
            lattice=self.lattice_types.index(params.lattice_type),
            polarization=self.polarizations.index(params.polarization),
            hole=int(np.where(self.hole_eps == params.hole_eps)[0][0]),
            epsilon=int(np.where(self.epsilon_background == params.epsilon_bg)[0][0]),
            radius=int(np.where(self.r_over_a == params.r_over_a)[0][0]),
        )

    def k_path_for(self, lattice_type: str) -> NDArray[np.float64]:
        return self.k_paths[lattice_type]


def build_axes(config: ScanConfig) -> Axes:
    eps_axis = _range_to_array(config.epsilon_background)
    r_axis = _range_to_array(config.r_over_a)
    hole_axis = np.array(config.hole_eps_values, dtype=np.float64)

    k_paths = {
        lattice: _build_k_path(cfg)
        for lattice, cfg in config.mpb.k_path.items()
    }
    lengths = {lat: arr.shape[0] for lat, arr in k_paths.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            "All lattices must share the same number of k-points to share one dataset."
        )

    return Axes(
        lattice_types=config.lattice_types,
        polarizations=config.polarizations,
        epsilon_background=eps_axis,
        r_over_a=r_axis,
        hole_eps=hole_axis,
        k_paths=k_paths,
        num_kpoints=unique_lengths.pop(),
    )


def iter_geometry_indices(axes: Axes) -> Iterable[GeometryIndices]:
    for i_lat in range(len(axes.lattice_types)):
        for i_pol in range(len(axes.polarizations)):
            for i_hole in range(len(axes.hole_eps)):
                for i_eps in range(len(axes.epsilon_background)):
                    for i_r in range(len(axes.r_over_a)):
                        yield GeometryIndices(i_lat, i_pol, i_hole, i_eps, i_r)


def _range_to_array(range_spec: RangeSpec) -> NDArray[np.float64]:
    range_spec.validate()
    count = int(round((range_spec.max - range_spec.min) / range_spec.step)) + 1
    axis = range_spec.min + range_spec.step * np.arange(count, dtype=np.float64)
    axis = np.clip(axis, range_spec.min, range_spec.max)
    axis[-1] = range_spec.max
    return axis


def _build_k_path(path_cfg: LatticePathConfig) -> NDArray[np.float64]:
    segments: list[NDArray[np.float64]] = []
    total_points = 0
    for idx, segment in enumerate(path_cfg.segments):
        start = np.asarray(segment[0], dtype=np.float64)
        end = np.asarray(segment[1], dtype=np.float64)
        t = np.linspace(0.0, 1.0, path_cfg.points_per_segment, dtype=np.float64)
        coords = start + (end - start) * t[:, None]
        if idx > 0:
            coords = coords[1:]
        segments.append(coords)
        total_points += coords.shape[0]
    result = np.concatenate(segments, axis=0)
    assert result.shape == (total_points, 2)
    return result
