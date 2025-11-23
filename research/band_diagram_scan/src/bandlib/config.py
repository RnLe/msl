from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import hashlib

import yaml


@dataclass(frozen=True)
class RangeSpec:
    """Closed range specification with constant spacing."""

    min: float
    max: float
    step: float

    def validate(self) -> None:
        if self.step <= 0:
            raise ValueError("Range step must be positive")
        if self.max < self.min:
            raise ValueError("Range max must be >= min")


@dataclass(frozen=True)
class LatticePathConfig:
    segments: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    points_per_segment: int

    def validate(self) -> None:
        if self.points_per_segment < 2:
            raise ValueError("Each k-path segment needs at least 2 points")
        if not self.segments:
            raise ValueError("At least one k-path segment is required")


@dataclass(frozen=True)
class MPBConfig:
    resolution: int
    num_bands: int
    dimensions: int
    k_path: Dict[str, LatticePathConfig]

    def validate(self, lattice_types: Sequence[str]) -> None:
        if self.resolution <= 0:
            raise ValueError("MPB resolution must be positive")
        if self.num_bands <= 0:
            raise ValueError("Number of bands must be positive")
        if self.dimensions != 2:
            raise ValueError("This project currently supports only 2D simulations")
        missing = [lat for lat in lattice_types if lat not in self.k_path]
        if missing:
            raise ValueError(f"Missing k-path definition for lattices: {missing}")
        for cfg in self.k_path.values():
            cfg.validate()


@dataclass(frozen=True)
class RuntimeOptions:
    max_workers: int
    checkpoint_interval: int

    def validate(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be >= 1")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be >= 1")


@dataclass(frozen=True)
class ScanConfig:
    scan_id: str
    lattice_types: Tuple[str, ...]
    polarizations: Tuple[str, ...]
    epsilon_background: RangeSpec
    r_over_a: RangeSpec
    hole_eps_values: Tuple[float, ...]
    mpb: MPBConfig
    runtime: RuntimeOptions

    def validate(self) -> None:
        if not self.scan_id:
            raise ValueError("scan_id must be non-empty")
        if not self.lattice_types:
            raise ValueError("At least one lattice type is required")
        if not self.polarizations:
            raise ValueError("At least one polarization is required")
        self.epsilon_background.validate()
        self.r_over_a.validate()
        if not self.hole_eps_values:
            raise ValueError("Provide at least one hole permittivity value")
        self.mpb.validate(self.lattice_types)
        self.runtime.validate()


def _as_tuple_str(items: Sequence[str]) -> Tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(item.strip() for item in items))
    if any(not item for item in cleaned):
        raise ValueError("Empty strings are not allowed")
    return cleaned


def _parse_segments(raw_segments: Iterable[Sequence[Sequence[float]]]) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]:
    parsed: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for segment in raw_segments:
        if len(segment) != 2:
            raise ValueError("Each segment must define [start, end]")
        start = tuple(float(x) for x in segment[0])  # type: ignore[arg-type]
        end = tuple(float(x) for x in segment[1])  # type: ignore[arg-type]
        if len(start) != 2 or len(end) != 2:
            raise ValueError("Segment endpoints must be 2D coordinates")
        parsed.append(((start[0], start[1]), (end[0], end[1])))
    return tuple(parsed)


def load_scan_config(path: Path | str) -> Tuple[ScanConfig, str, str]:
    """Load a YAML config file.

    Returns (config, raw_yaml, config_hash).
    """

    path = Path(path)
    raw_yaml = path.read_text()
    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError("Scan config must be a mapping")

    epsilon = RangeSpec(**data["epsilon_background"])
    r_over_a = RangeSpec(**data["r_over_a"])
    hole_eps = tuple(float(v) for v in data["hole_material"]["eps_values"])

    lattice_types = _as_tuple_str(data["lattice_types"])
    polarizations = _as_tuple_str(data["polarizations"])

    k_path_cfg: Dict[str, LatticePathConfig] = {}
    for lattice, cfg in data["mpb"]["k_path"].items():
        segments = _parse_segments(cfg["segments"])
        k_path_cfg[lattice] = LatticePathConfig(
            segments=segments,
            points_per_segment=int(cfg["points_per_segment"]),
        )

    mpb = MPBConfig(
        resolution=int(data["mpb"]["resolution"]),
        num_bands=int(data["mpb"]["num_bands"]),
        dimensions=int(data["mpb"]["dimensions"]),
        k_path=k_path_cfg,
    )

    runtime = RuntimeOptions(
        max_workers=int(data["runtime"]["max_workers"]),
        checkpoint_interval=int(data["runtime"]["checkpoint_interval"]),
    )

    config = ScanConfig(
        scan_id=data["scan_id"],
        lattice_types=lattice_types,
        polarizations=polarizations,
        epsilon_background=epsilon,
        r_over_a=r_over_a,
        hole_eps_values=hole_eps,
        mpb=mpb,
        runtime=runtime,
    )
    config.validate()
    config_hash = hashlib.sha1(raw_yaml.encode("utf-8")).hexdigest()
    return config, raw_yaml, config_hash
