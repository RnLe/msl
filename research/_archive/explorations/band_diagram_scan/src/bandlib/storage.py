from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, Tuple, cast

import h5py
import numpy as np

from .axes import Axes, GeometryIndices
from .config import ScanConfig


class Status(IntEnum):
    PENDING = 0
    COMPLETE = 1
    IN_PROGRESS = 2
    FAILED = 3


@dataclass(frozen=True)
class ScanHandle:
    file_path: Path
    scan_id: str
    axes: Axes
    num_bands: int


def _require_group(parent: h5py.Group | h5py.File, name: str) -> h5py.Group:
    node = parent[name]
    if not isinstance(node, h5py.Group):  # pragma: no cover - defensive
        raise TypeError(f"Expected '{name}' to be an HDF5 group")
    return node


def _require_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    node = parent[name]
    if not isinstance(node, h5py.Dataset):  # pragma: no cover - defensive
        raise TypeError(f"Expected '{name}' to be an HDF5 dataset")
    return cast(h5py.Dataset, node)


def ensure_scan(
    file_path: Path | str,
    config: ScanConfig,
    axes: Axes,
    raw_yaml: str,
    config_hash: str,
) -> ScanHandle:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "a") as h5:
        _ensure_meta(h5)
        scans = h5.require_group("scans")
        if config.scan_id in scans:
            scan_obj = scans[config.scan_id]
            if not isinstance(scan_obj, h5py.Group):  # pragma: no cover - defensive
                raise TypeError("Existing scan entry must be an HDF5 group")
            scan_grp = scan_obj
            _validate_existing_scan(scan_grp, axes, config)
        else:
            scan_grp = scans.create_group(config.scan_id)
            _initialize_scan(scan_grp, axes, config, raw_yaml, config_hash)

    return ScanHandle(file_path=file_path, scan_id=config.scan_id, axes=axes, num_bands=config.mpb.num_bands)


def get_scan_group(handle: ScanHandle) -> h5py.Group:
    h5 = h5py.File(handle.file_path, "r+")
    scans = h5["scans"]
    if not isinstance(scans, h5py.Group):  # pragma: no cover - defensive
        raise TypeError("/scans must be an HDF5 group")
    scan = scans[handle.scan_id]
    if not isinstance(scan, h5py.Group):  # pragma: no cover - defensive
        raise TypeError("scan entry must be an HDF5 group")
    return scan


def list_pending_indices(
    scan_grp: h5py.Group, resume_incomplete: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    status_grp = _require_group(scan_grp, "status")
    status_ds = _require_dataset(status_grp, "geom_status")
    status = status_ds[...]
    if resume_incomplete and np.any(status == Status.IN_PROGRESS):
        status[status == Status.IN_PROGRESS] = Status.PENDING
        status_ds[...] = status
    return np.argwhere(status == Status.PENDING), status


def mark_status(status_ds: h5py.Dataset, idx: GeometryIndices, value: Status) -> None:
    status_ds[idx.lattice, idx.polarization, idx.hole, idx.epsilon, idx.radius] = int(value)


def write_frequency(freq_ds: h5py.Dataset, idx: GeometryIndices, bands: np.ndarray) -> None:
    freq_ds[
        idx.lattice,
        idx.polarization,
        idx.hole,
        idx.epsilon,
        idx.radius,
        :, :,
    ] = bands.astype(np.float32, copy=False)


def summarize_status(scan_grp: h5py.Group) -> Dict[Status, int]:
    status_grp = _require_group(scan_grp, "status")
    status_ds = _require_dataset(status_grp, "geom_status")
    status = status_ds[...]
    return {status_enum: int(np.sum(status == status_enum)) for status_enum in Status}


def _ensure_meta(h5: h5py.File) -> None:
    meta = h5.require_group("meta")
    meta.attrs.setdefault("version", "1.0")
    meta.attrs.setdefault("created", datetime.now(timezone.utc).isoformat())
    meta.attrs.setdefault("description", "2D photonic band library")


def _initialize_scan(scan_grp: h5py.Group, axes: Axes, config: ScanConfig, raw_yaml: str, config_hash: str) -> None:
    axes_grp = scan_grp.create_group("axes")
    str_dtype = h5py.string_dtype(encoding="utf-8")
    axes_grp.create_dataset("lattice_type", data=np.array(axes.lattice_types, dtype=str_dtype))
    axes_grp.create_dataset("polarization", data=np.array(axes.polarizations, dtype=str_dtype))
    axes_grp.create_dataset("eps_bg", data=axes.epsilon_background)
    axes_grp.create_dataset("r_over_a", data=axes.r_over_a)
    axes_grp.create_dataset("hole_eps", data=axes.hole_eps)

    k_group = axes_grp.create_group("k_path")
    for name, arr in axes.k_paths.items():
        k_group.create_dataset(name, data=arr)

    data_grp = scan_grp.create_group("data")
    freq_shape = (
        len(axes.lattice_types),
        len(axes.polarizations),
        len(axes.hole_eps),
        len(axes.epsilon_background),
        len(axes.r_over_a),
        config.mpb.num_bands,
        axes.num_kpoints,
    )
    data_grp.create_dataset(
        "freq",
        shape=freq_shape,
        dtype="f4",
        chunks=(1, 1, 1, 1, 1, config.mpb.num_bands, axes.num_kpoints),
        compression="gzip",
        compression_opts=4,
        fillvalue=np.nan,
    )

    status_grp = scan_grp.create_group("status")
    status_grp.create_dataset(
        "geom_status",
        shape=freq_shape[:-2],
        dtype="u1",
        data=np.zeros(freq_shape[:-2], dtype=np.uint8),
    )

    cfg_grp = scan_grp.create_group("config")
    cfg_grp.create_dataset("raw_yaml", data=np.array(raw_yaml, dtype=str_dtype))
    cfg_grp.attrs["scan_id"] = config.scan_id
    cfg_grp.attrs["config_hash"] = config_hash

    scan_grp.attrs["num_bands"] = config.mpb.num_bands
    scan_grp.attrs["num_kpoints"] = axes.num_kpoints


def _validate_existing_scan(scan_grp: h5py.Group, axes: Axes, config: ScanConfig) -> None:
    axes_grp = _require_group(scan_grp, "axes")
    lattice_ds = _require_dataset(axes_grp, "lattice_type")
    pol_ds = _require_dataset(axes_grp, "polarization")
    existing_lattices = tuple(lattice_ds.asstr()[...])
    existing_pols = tuple(pol_ds.asstr()[...])
    if existing_lattices != axes.lattice_types or existing_pols != axes.polarizations:
        raise ValueError("Existing scan axes do not match the provided config")
    eps_ds = _require_dataset(axes_grp, "eps_bg")
    r_ds = _require_dataset(axes_grp, "r_over_a")
    hole_ds = _require_dataset(axes_grp, "hole_eps")
    if not np.allclose(eps_ds[...], axes.epsilon_background):
        raise ValueError("epsilon axis mismatch")
    if not np.allclose(r_ds[...], axes.r_over_a):
        raise ValueError("r/a axis mismatch")
    if not np.allclose(hole_ds[...], axes.hole_eps):
        raise ValueError("hole_eps axis mismatch")
    if scan_grp.attrs.get("num_bands") != config.mpb.num_bands:
        raise ValueError("num_bands mismatch")
    if scan_grp.attrs.get("num_kpoints") != axes.num_kpoints:
        raise ValueError("k_path length mismatch")
